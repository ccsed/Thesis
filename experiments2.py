import os
import json
import yaml
import random
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import sys
import multiprocessing
import io
from collections import defaultdict

import RoothPath
from Utils.pool_with_subprocess import PoolWithSubprocess
from Simulation.simulation_new_recovery import SimulationNewRecovery
from Simulation.TP_with_recovery import TokenPassingRecovery
from Simulation.tasks_and_delays_maker import gen_tasks_and_delays

NUM_RUNS = 25
MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000

# I 4 scenari di Terraform da testare sulla scalabilità
SCENARIOS = {
    # "Scenario_1_The_Sliding_Puzzle": {"alpha": 1.0, "radius": 3, "laziness": 1},
    # "Scenario_2_Proactive_Clearance": {"alpha": 1.0, "radius": 5, "laziness": 4},
    # "Scenario_3_Heavy_Duty_Bottleneck": {"alpha": 20.0, "radius": 4, "laziness": 3},
    "Scenario_4_Cheap_Deep_Relocation": {"alpha": 0.0001, "radius": 3, "laziness": 2}
}

# Parametri base per la generazione del traffico
BASE_SIM_PARAMS = {
    "default_n_tasks": 20,  # Usato come default quando varia il numero di agenti
    "task_freq": 0.2,
    "n_delays_per_agent": 0
}

class LogCatcher:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        
    def get_critical_warnings(self):
        output = self.captured_output.getvalue()
        warnings = set()
        keywords = ['fatal exception', 'fatal error']
        
        for line in output.split('\n'):
            line = line.strip()
            if any(kw in line.lower() for kw in keywords):
                warnings.add(line)
        return list(warnings)
        
    def has_deadlock_recovery(self):
        output = self.captured_output.getvalue().lower()
        return 'causing deadlock' in output or 'deadlock recovery' in output

def verify_no_collisions(simulation, tp):
    states_dict = defaultdict(int)
    full_paths = {}
    
    for agent_name, path in simulation.actual_paths.items():
        full_paths[agent_name] = path
        
    obstacles_paths = tp.get_token()['obstacles_paths']
    max_time = simulation.get_time()
    
    for obs_id, path in obstacles_paths.items():
        continuous_path = []
        path_dict = {step['t']: step for step in path}
        if path:
            last_pos = path[0]
        else:
            last_pos = {'x': 0, 'y': 0}
            
        for t in range(max_time + 1):
            if t in path_dict:
                last_pos = path_dict[t]
            continuous_path.append({'t': t, 'x': last_pos['x'], 'y': last_pos['y']})
        full_paths[obs_id] = continuous_path

    for entity, path in full_paths.items():
        for step in path:
            state_key = f"{step['t']}_{step['x']}_{step['y']}"
            states_dict[state_key] += 1
            if states_dict[state_key] > 1:
                return True
    return False

def generate_map_preview(yaml_path, output_filename):
    with open(yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)

    dims = map_data['map']['dimensions']
    static_obs = map_data['map'].get('obstacles', [])
    movable_obs = map_data['map'].get('movable_obstacles', [])
    endpoints = map_data['map'].get('non_task_endpoints', [])
    
    start_locs = map_data['map'].get('start_locations', [])
    goal_locs = map_data['map'].get('goal_locations', [])
    task_endpoints = set(tuple(ep) for ep in start_locs + goal_locs)

    fig, ax = plt.subplots(figsize=(dims[0], dims[1]))
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_xticks(range(dims[0] + 1))
    ax.set_yticks(range(dims[1] + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_aspect('equal')

    def draw_square(coords, color, edgecolor='black', hatch=None, zorder=1, linestyle='solid', linewidth=1):
        x, y = coords[0], coords[1]
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=edgecolor, 
                                       hatch=hatch, zorder=zorder, linestyle=linestyle, linewidth=linewidth))

    for ep in task_endpoints:
        draw_square(ep, '#e5f5e0', edgecolor='#31a354', zorder=0.5, linestyle='--', linewidth=1.5)
    for ep in endpoints:
        draw_square(ep, '#cce5ff', edgecolor='#0066cc', zorder=1)
        ax.text(ep[0] + 0.5, ep[1] + 0.5, 'EP', color='#0066cc', ha='center', va='center', fontsize=10, fontweight='bold')

    for obs in static_obs:
        draw_square(obs, '#333333', zorder=2)
    for mobs in movable_obs:
        draw_square(mobs, '#ff9900', edgecolor='#cc0000', hatch='///', zorder=3)

    legend_elements = [
        patches.Patch(facecolor='#333333', edgecolor='black', label='Static Obstacle'),
        patches.Patch(facecolor='#ff9900', edgecolor='#cc0000', hatch='///', label='Movable Obstacle'),
        patches.Patch(facecolor='#cce5ff', edgecolor='#0066cc', label='Parking Endpoint'),
        patches.Patch(facecolor='#e5f5e0', edgecolor='#31a354', linestyle='--', linewidth=1.5, label='Task Endpoint')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    plt.title(f"Scalability Map Preview", fontsize=16, pad=15)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close('all')

def worker_task(payload):
    yaml_path = payload['yaml_path']
    seed = payload['seed']
    mode = payload['mode']
    
    n_agents = payload['n_agents']
    n_tasks = payload['n_tasks']
    alpha = payload['alpha']
    radius = payload['radius']
    laziness = payload['laziness']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    base_movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']

    # --- Generazione Dinamica Agenti ---
    random.seed(seed)
    if n_agents > len(non_task_endpoints):
        n_agents = len(non_task_endpoints)
        
    selected_endpoints = random.sample(non_task_endpoints, n_agents)
    agents = [{'name': f'agent{i}', 'start': ep, 'goal': ep} for i, ep in enumerate(selected_endpoints)]

    # --- Generazione Tasks (stesso seed per consistenza tra LB, UB e TF) ---
    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, BASE_SIM_PARAMS['task_freq'], BASE_SIM_PARAMS['n_delays_per_agent']
    )

    # Impostazione logica per LB, UB o Terraforming Standard
    movable_obs_run = base_movable_obstacles
    obs_run = obstacles
    rad_run = radius
    laz_run = laziness
    
    if mode == 'lb':
        movable_obs_run = []
        obs_run = obstacles  # Nessun ostacolo mobile
        rad_run = 1
        laz_run = 1
    elif mode == 'ub':
        movable_obs_run = []
        obs_run = obstacles + base_movable_obstacles  # Gli ostacoli mobili si comportano da ostacoli statici
        rad_run = 1
        laz_run = 1

    with LogCatcher() as catcher:
        try:
            simulation = SimulationNewRecovery(tasks, agents, delays=delays)
            tp = TokenPassingRecovery(
                agents, dimensions, obs_run, non_task_endpoints, simulation,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=rad_run, new_recovery=True,
                movable_obstacles=movable_obs_run, alpha=alpha, laziness=laz_run
            )

            while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < MAX_SIM_TIME:
                simulation.time_forward(tp)

            warnings = catcher.get_critical_warnings()
            used_deadlock = catcher.has_deadlock_recovery()
        except SystemExit:
            warnings = catcher.get_critical_warnings()
            return {'failed': True, 'mode': mode, 'seed': seed, 'warnings': warnings}

        # Segnalazione Deadlock (NON viene scartata la run)
        if used_deadlock:
            warnings.append("DEADLOCK RECOVERY TRIGGERED (Included in stats)")

        if simulation.get_time() >= MAX_SIM_TIME:
            warnings.append(f"TIMEOUT: MAX_SIM_TIME reached.")
            return {'failed': True, 'mode': mode, 'seed': seed, 'warnings': warnings}
            
        has_collision = verify_no_collisions(simulation, tp)
        if has_collision:
            warnings.append("PHYSICAL COLLISION DETECTED DURING VALIDATION!")
            return {'failed': True, 'mode': mode, 'seed': seed, 'warnings': warnings}

        completed_times = tp.get_completed_tasks_times()
        task_pickup = tp.get_pickup_tasks_times() 
        task_assigned = tp.get_assigned_tasks_times() # USA IL NUOVO GETTER

        makespan = max(completed_times.values()) if completed_times else 0

        # exec_times = []
        # for t_name, c_time in completed_times.items():
        #     exec_times.append(c_time - task_assigned[t_name]) # CALCOLA L'EXECUTION TIME
            
        # avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        pure_delivery_times = []
        for t_name, c_time in completed_times.items():
            # Tempo puro: (Arrivo a destinazione) - (Momento esatto del Pickup)
            pure_delivery_times.append(c_time - task_pickup[t_name]) 
            
        avg_exec_time = sum(pure_delivery_times) / len(pure_delivery_times) if pure_delivery_times else 0

        # Terraforming Cost Engine
        p_total = 0
        for obs_id, path in tp.get_token()['obstacles_paths'].items():
            for i in range(1, len(path)):
                if path[i]['x'] != path[i-1]['x'] or path[i]['y'] != path[i-1]['y']:
                    p_total += 1
                    
        terraforming_cost = alpha * p_total
        sim_time = simulation.get_algo_time()

    return {
        'mode': mode,
        'makespan': makespan,
        'execution_time': avg_exec_time,
        'sim_time': sim_time,
        'terraforming_cost': terraforming_cost,
        'tf_steps': p_total if alpha == 0.0001 else None,
        'seed': seed,
        'failed': False,
        'warnings': warnings
    }

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" STARTING SCALABILITY ANALYSIS (SCENARIOS)")
    print("="*50)

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as f:
        config = json.load(f)
        
    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)

    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots_exp2', map_name_clean)
    os.makedirs(base_plots_dir, exist_ok=True)
    
    map_img_path = os.path.join(base_plots_dir, 'map_preview.png')
    generate_map_preview(map_path, map_img_path)
    
    with open(map_path, 'r') as f:
        param = yaml.safe_load(f)
    
    non_task_endpoints = param['map']['non_task_endpoints']
    goal_locations = param['map']['goal_locations']
    
    # Calcolo dinamico del numero massimo di agenti
    max_possible_agents = min(len(non_task_endpoints), len(goal_locations), 6)
    default_agents = max(1, max_possible_agents // 2)

    SWEEP_VALUES = {
        "N_Agents": list(range(1, max_possible_agents + 1)),
        "N_Tasks": [10, 20, 30, 40, 50, 60, 70, 80, 100]
    }

    print(f"Test Map: {map_yaml_name}")
    print(f"Max Agents (min between endpoints and goals): {max_possible_agents}")
    print(f"Base Directory: {base_plots_dir}\n")

    cores = max(1, multiprocessing.cpu_count() - 2)
    colors_st = {'N_Agents': '#1f77b4', 'N_Tasks': '#d62728'}
    colors_ms = {'N_Agents': '#9467bd', 'N_Tasks': '#ff7f0e'}
    colors_sim = {'N_Agents': '#2ca02c', 'N_Tasks': '#8c564b'}
    colors_tf = {'N_Agents': '#e377c2', 'N_Tasks': '#17becf'}

    for scen_name, scen_params in SCENARIOS.items():
        print(f"\n>>> Executing Scenario: {scen_name} <<<")
        
        scen_dir = os.path.join(base_plots_dir, scen_name)
        os.makedirs(scen_dir, exist_ok=True)
        master_results = {}

        for param_name, sweep_vals in SWEEP_VALUES.items():
            print(f"--- Sweeping Parameter: {param_name} ---")
            
            master_results[param_name] = {
                'values': sweep_vals,
                'standard': {'st': [], 'st_err': [], 'ms': [], 'ms_err': [], 'sim': [], 'sim_err': [], 'tf': [], 'tf_err': [], 'tf_steps_raw': []},
                'lb': {'st': [], 'ms': [], 'sim': []},
                'ub': {'st': [], 'ms': [], 'sim': []},
            }

            for val in sweep_vals:
                n_agents = val if param_name == "N_Agents" else default_agents
                n_tasks = val if param_name == "N_Tasks" else BASE_SIM_PARAMS["default_n_tasks"]
                
                print(f"  Testing {param_name} = {val} [Agents: {n_agents}, Tasks: {n_tasks}] | Runs: {NUM_RUNS}...")
                
                payloads = []
                for seed in range(1, NUM_RUNS + 1):
                    base_payload = {
                        'yaml_path': map_path,
                        'seed': seed,
                        'n_agents': n_agents,
                        'n_tasks': n_tasks,
                        'alpha': scen_params['alpha'],
                        'radius': scen_params['radius'],
                        'laziness': scen_params['laziness']
                    }
                    payloads.append({**base_payload, 'mode': 'lb'})
                    payloads.append({**base_payload, 'mode': 'ub'})
                    payloads.append({**base_payload, 'mode': 'standard'})

                pool = PoolWithSubprocess(processes=cores)
                try:
                    results = pool.map_async(worker_task, payloads).get(timeout=86400)
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    print("\n[!] Ctrl+C premuto! Terminazione forzata dei processi in corso...")
                    pool.terminate() 
                    pool.join()
                    sys.exit(1)   

                successful_results = [r for r in results if not r.get('failed', False)]
                failed_count = len(results) - len(successful_results)
                
                all_warnings = []
                for r in results:
                    if r.get('warnings'):
                        for w in r['warnings']:
                            all_warnings.append(f"[{r['mode'].upper()}] Seed {r['seed']}: {w}")
                
                if failed_count > 0:
                    print(f"      [!] WARNING: {failed_count} runs ended in collision or timeout (excluded).")
                    
                if all_warnings:
                    print(f"      [!] LOGS DETECTED:")
                    for w in sorted(list(set(all_warnings))): 
                        print(f"          - {w}")

                std_res = [r for r in successful_results if r['mode'] == 'standard']
                lb_res = [r for r in successful_results if r['mode'] == 'lb']
                ub_res = [r for r in successful_results if r['mode'] == 'ub']

                # Standard
                if std_res:
                    master_results[param_name]['standard']['st'].append(np.mean([r['execution_time'] for r in std_res]))
                    master_results[param_name]['standard']['st_err'].append(np.std([r['execution_time'] for r in std_res]))
                    master_results[param_name]['standard']['ms'].append(np.mean([r['makespan'] for r in std_res]))
                    master_results[param_name]['standard']['ms_err'].append(np.std([r['makespan'] for r in std_res]))
                    master_results[param_name]['standard']['sim'].append(np.mean([r['sim_time'] for r in std_res]))
                    master_results[param_name]['standard']['sim_err'].append(np.std([r['sim_time'] for r in std_res]))
                    master_results[param_name]['standard']['tf'].append(np.mean([r['terraforming_cost'] for r in std_res]))
                    master_results[param_name]['standard']['tf_err'].append(np.std([r['terraforming_cost'] for r in std_res]))
                    if scen_params['alpha'] == 0.0001:
                        master_results[param_name]['standard']['tf_steps_raw'].append(np.mean([r['tf_steps'] for r in std_res]))
                    else:
                        master_results[param_name]['standard']['tf_steps_raw'].append(None)
                else:
                    for k in master_results[param_name]['standard']: master_results[param_name]['standard'][k].append(np.nan)

                # Lower Bound
                if lb_res:
                    master_results[param_name]['lb']['st'].append(np.mean([r['execution_time'] for r in lb_res]))
                    master_results[param_name]['lb']['ms'].append(np.mean([r['makespan'] for r in lb_res]))
                    master_results[param_name]['lb']['sim'].append(np.mean([r['sim_time'] for r in lb_res]))
                else:
                    for k in master_results[param_name]['lb']: master_results[param_name]['lb'][k].append(np.nan)

                # Upper Bound
                if ub_res:
                    master_results[param_name]['ub']['st'].append(np.mean([r['execution_time'] for r in ub_res]))
                    master_results[param_name]['ub']['ms'].append(np.mean([r['makespan'] for r in ub_res]))
                    master_results[param_name]['ub']['sim'].append(np.mean([r['sim_time'] for r in ub_res]))
                else:
                    for k in master_results[param_name]['ub']: master_results[param_name]['ub'][k].append(np.nan)

        json_path = os.path.join(scen_dir, 'scalability_results.json')
        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)

        # Generazione Grafici per lo Scenario Corrente
        for param_name, data in master_results.items():
            x = data['values']
            fixed_text = (f"Alpha: {scen_params['alpha']}\nRadius: {scen_params['radius']}\nLaziness: {scen_params['laziness']}")
            if param_name == "N_Agents": fixed_text += f"\nTasks: {BASE_SIM_PARAMS['default_n_tasks']}"
            else: fixed_text += f"\nAgents: {default_agents}"

            # # 1. Execution Time
            # plt.figure(figsize=(8, 6))
            # plt.errorbar(x, data['standard']['st'], yerr=data['standard']['st_err'], fmt='-o', color=colors_st[param_name], 
            #              ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Terraforming (Avg)')
            # plt.plot(x, data['lb']['st'], '--', color='green', linewidth=2, label='Lower Bound (No Movable Obs)')
            # plt.plot(x, data['ub']['st'], '--', color='red', linewidth=2, label='Upper Bound (Static M.O.)')
            
            # plt.title(f'{scen_name}\nScalability: Execution Time vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            # plt.xlabel(f'{param_name}', fontsize=12)
            # plt.ylabel('Average Execution Time (Timesteps)', fontsize=12)
            # 1. Pure Delivery Time
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['standard']['st'], yerr=data['standard']['st_err'], fmt='-o', color=colors_st[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Terraforming (Avg)')
            plt.plot(x, data['lb']['st'], '--', color='green', linewidth=2, label='Lower Bound (No Movable Obs)')
            plt.plot(x, data['ub']['st'], '--', color='red', linewidth=2, label='Upper Bound (Static M.O.)')
            
            plt.title(f'{scen_name}\nPure Delivery Time (Pickup to Goal) vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name}', fontsize=12)
            plt.ylabel('Avg Delivery Time (Timesteps)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_ExecutionTime_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            # 2. Makespan
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['standard']['ms'], yerr=data['standard']['ms_err'], fmt='-s', color=colors_ms[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Terraforming (Avg)')
            plt.plot(x, data['lb']['ms'], '--', color='green', linewidth=2, label='Lower Bound (No Movable Obs)')
            plt.plot(x, data['ub']['ms'], '--', color='red', linewidth=2, label='Upper Bound (Static M.O.)')
            
            plt.title(f'{scen_name}\nScalability: Makespan vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name}', fontsize=12)
            plt.ylabel('Average Makespan (Timesteps)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            # 3. Sim Time
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['standard']['sim'], yerr=data['standard']['sim_err'], fmt='-^', color=colors_sim[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Terraforming')
            plt.plot(x, data['lb']['sim'], '--', color='green', linewidth=2, label='Lower Bound')
            plt.plot(x, data['ub']['sim'], '--', color='red', linewidth=2, label='Upper Bound')
            
            plt.title(f'{scen_name}\nComputational Effort: Sim Time vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name}', fontsize=12)
            plt.ylabel('Processing Time (Seconds)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_SimTime_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            # 4. Terraforming Cost (Solo Standard)
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['standard']['tf'], yerr=data['standard']['tf_err'], fmt='-d', color=colors_tf[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='True TF Cost')
            plt.title(f'{scen_name}\nScalability: Terraforming Cost vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name}', fontsize=12)
            plt.ylabel('Energy/Penalty Score', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_TerraformingCost_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

    print("\n[+] Scalability Tests Completed Successfully. All plots saved to", base_plots_dir)