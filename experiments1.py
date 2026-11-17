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

NUM_RUNS = 50
MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000

# SCENARIOS = {
#     "Scenario_1_Heavy_World": {"alpha": 20.0, "radius": 2, "laziness": 3},
#     "Scenario_2_Highly_Malleable": {"alpha": 0.5, "radius": 5, "laziness": 3},
#     "Scenario_3_Strictly_Reactive": {"alpha": 1.0, "radius": 3, "laziness": 1},
#     "Scenario_4_Topologically_Constrained": {"alpha": 1.0, "radius": 1, "laziness": 5}
# }
SCENARIOS = {
    "Scenario_1_The_Sliding_Puzzle": {"alpha": 1.0, "radius": 3, "laziness": 1},
    "Scenario_2_Proactive_Clearance": {"alpha": 1.0, "radius": 5, "laziness": 4},
    "Scenario_3_Heavy_Duty_Bottleneck": {"alpha": 20.0, "radius": 5, "laziness": 3},
    "Scenario_4_Cheap_Deep_Relocation": {"alpha": 0.1, "radius": 3, "laziness": 5}
}

SWEEP_VALUES = {
    "Alpha": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10, 20, 100],
    "Radius": [1, 2, 3, 4, 5, 6],
    "Laziness": [1, 2, 3, 4, 5]
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
    
    # Identify Task Endpoints
    start_locs = map_data['map'].get('start_locations', [])
    goal_locs = map_data['map'].get('goal_locations', [])
    task_endpoints = set(tuple(ep) for ep in start_locs + goal_locs)
    
    agents = map_data.get('agents', [])

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

    # Plot Task Endpoints (Pale Green) -> Sent to Background
    for ep in task_endpoints:
        draw_square(ep, '#e5f5e0', edgecolor='#31a354', zorder=0.5, linestyle='--', linewidth=1.5)

    # Plot Non-Task Parking Endpoints (Light Blue)
    for ep in endpoints:
        draw_square(ep, '#cce5ff', edgecolor='#0066cc', zorder=1)
        ax.text(ep[0] + 0.5, ep[1] + 0.5, 'EP', color='#0066cc', ha='center', va='center', fontsize=10, fontweight='bold')

    for obs in static_obs:
        draw_square(obs, '#333333', zorder=2)

    for mobs in movable_obs:
        draw_square(mobs, '#ff9900', edgecolor='#cc0000', hatch='///', zorder=3)

    for agent in agents:
        start = agent['start']
        cx, cy = start[0] + 0.5, start[1] + 0.5
        ax.add_patch(patches.Circle((cx, cy), 0.35, facecolor='#0033cc', edgecolor='black', zorder=4))
        label = agent['name'].replace('agent', 'A')
        ax.text(cx, cy, label, color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)

    legend_elements = [
        patches.Patch(facecolor='#333333', edgecolor='black', label='Static Obstacle'),
        patches.Patch(facecolor='#ff9900', edgecolor='#cc0000', hatch='///', label='Movable Obstacle'),
        patches.Patch(facecolor='#cce5ff', edgecolor='#0066cc', label='Parking Endpoint'),
        patches.Patch(facecolor='#e5f5e0', edgecolor='#31a354', linestyle='--', linewidth=1.5, label='Task Endpoint'),
        Line2D([0], [0], marker='o', color='w', label='Agent', markerfacecolor='#0033cc', markersize=12, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    plt.title(f"Map Preview", fontsize=16, pad=15)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close('all')

def baseline_worker_task(payload):
    yaml_path = payload['yaml_path']
    seed = payload['seed']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    agents = param['agents']
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']

    n_tasks = param.get('n_tasks', 20)
    task_freq = param.get('task_freq', 1.0)
    n_delays_per_agent = param.get('n_delays_per_agent', 0)

    random.seed(seed)
    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, task_freq, n_delays_per_agent
    )

    # LOWER BOUND CONFIGURATION (Movable Obstacles Do Not Exist)
    tasks_lb = copy.deepcopy(tasks)
    delays_lb = copy.deepcopy(delays)
    lb_failed = False
    lb_makespan, lb_service_time = 0, 0

    with LogCatcher() as catcher:
        try:
            sim_lb = SimulationNewRecovery(tasks_lb, agents, delays=delays_lb)
            tp_lb = TokenPassingRecovery(
                agents, dimensions, obstacles, non_task_endpoints, sim_lb,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=1, new_recovery=True, movable_obstacles=[], alpha=1.0, laziness=1
            )
            while tp_lb.get_completed_tasks() != len(tasks_lb) and sim_lb.get_time() < MAX_SIM_TIME:
                sim_lb.time_forward(tp_lb)

            # if sim_lb.get_time() >= MAX_SIM_TIME or verify_no_collisions(sim_lb, tp_lb):
            #     lb_failed = True
            if sim_lb.get_time() >= MAX_SIM_TIME or verify_no_collisions(sim_lb, tp_lb) or catcher.has_deadlock_recovery():
                lb_failed = True
        except SystemExit:
            lb_failed = True
        else:
            completed_times = tp_lb.get_completed_tasks_times()
            task_starts = {t['task_name']: t['start_time'] for t in tasks_lb}
            lb_makespan = max(completed_times.values()) if completed_times else 0
            serv_times = [c_time - task_starts[t_name] for t_name, c_time in completed_times.items()]
            lb_service_time = sum(serv_times) / len(serv_times) if serv_times else 0

    # UPPER BOUND CONFIGURATION (Movable Obstacles act as Static Obstacles)
    tasks_ub = copy.deepcopy(tasks)
    delays_ub = copy.deepcopy(delays)
    ub_failed = False
    ub_makespan, ub_service_time = 0, 0
    static_obs_ub = obstacles + movable_obstacles

    with LogCatcher() as catcher:
        try:
            sim_ub = SimulationNewRecovery(tasks_ub, agents, delays=delays_ub)
            tp_ub = TokenPassingRecovery(
                agents, dimensions, static_obs_ub, non_task_endpoints, sim_ub,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=1, new_recovery=True, movable_obstacles=[], alpha=1.0, laziness=1
            )
            while tp_ub.get_completed_tasks() != len(tasks_ub) and sim_ub.get_time() < MAX_SIM_TIME:
                sim_ub.time_forward(tp_ub)

            # if sim_ub.get_time() >= MAX_SIM_TIME or verify_no_collisions(sim_ub, tp_ub):
            #     ub_failed = True
            if sim_ub.get_time() >= MAX_SIM_TIME or verify_no_collisions(sim_ub, tp_ub) or catcher.has_deadlock_recovery():
                ub_failed = True
        except SystemExit:
            ub_failed = True
        else:
            completed_times = tp_ub.get_completed_tasks_times()
            task_starts = {t['task_name']: t['start_time'] for t in tasks_ub}
            ub_makespan = max(completed_times.values()) if completed_times else 0
            serv_times = [c_time - task_starts[t_name] for t_name, c_time in completed_times.items()]
            ub_service_time = sum(serv_times) / len(serv_times) if serv_times else 0

    return {
        'seed': seed,
        'lb_failed': lb_failed, 'lb_makespan': lb_makespan, 'lb_service_time': lb_service_time,
        'ub_failed': ub_failed, 'ub_makespan': ub_makespan, 'ub_service_time': ub_service_time
    }

def worker_task(payload):
    yaml_path = payload['yaml_path']
    laziness = payload['laziness']
    radius = payload['radius']
    alpha = payload['alpha']
    seed = payload['seed']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    agents = param['agents']
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']

    n_tasks = param.get('n_tasks', 20)
    task_freq = param.get('task_freq', 1.0)
    n_delays_per_agent = param.get('n_delays_per_agent', 0)

    random.seed(seed)
    
    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, task_freq, n_delays_per_agent
    )

    with LogCatcher() as catcher:
        try:
            simulation = SimulationNewRecovery(tasks, agents, delays=delays)
            tp = TokenPassingRecovery(
                agents, dimensions, obstacles, non_task_endpoints, simulation,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=radius, new_recovery=True,
                movable_obstacles=movable_obstacles, alpha=alpha, laziness=laziness
            )

        #     while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < MAX_SIM_TIME:
        #         simulation.time_forward(tp)

        #     warnings = catcher.get_critical_warnings()
        # except SystemExit:
        #     warnings = catcher.get_critical_warnings()
        #     return {'failed': True, 'seed': seed, 'warnings': warnings}

        # if simulation.get_time() >= MAX_SIM_TIME:
        #     warnings.append(f"TIMEOUT: MAX_SIM_TIME reached.")
        #     return {'failed': True, 'seed': seed, 'warnings': warnings}
            while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < MAX_SIM_TIME:
                simulation.time_forward(tp)

            warnings = catcher.get_critical_warnings()
            used_deadlock = catcher.has_deadlock_recovery() # <-- Legge il log
        except SystemExit:
            warnings = catcher.get_critical_warnings()
            return {'failed': True, 'seed': seed, 'warnings': warnings}

        # SEGNALA FALLIMENTO SE HA USATO IL DEADLOCK
        if used_deadlock:
            warnings.append("DEADLOCK RECOVERY TRIGGERED (Excluded from stats)")
            return {'failed': True, 'seed': seed, 'warnings': warnings}

        if simulation.get_time() >= MAX_SIM_TIME:
            warnings.append(f"TIMEOUT: MAX_SIM_TIME reached.")
            return {'failed': True, 'seed': seed, 'warnings': warnings}
            
        has_collision = verify_no_collisions(simulation, tp)
        if has_collision:
            warnings.append("PHYSICAL COLLISION DETECTED DURING VALIDATION!")
            return {'failed': True, 'seed': seed, 'warnings': warnings}

        completed_times = tp.get_completed_tasks_times()
        task_starts = {t['task_name']: t['start_time'] for t in tasks}

        makespan = max(completed_times.values()) if completed_times else 0

        serv_times = []
        for t_name, c_time in completed_times.items():
            serv_times.append(c_time - task_starts[t_name])
            
        avg_service_time = sum(serv_times) / len(serv_times) if serv_times else 0

        # Terraforming Cost Engine
        p_total = 0
        for obs_id, path in tp.get_token()['obstacles_paths'].items():
            for i in range(1, len(path)):
                if path[i]['x'] != path[i-1]['x'] or path[i]['y'] != path[i-1]['y']:
                    p_total += 1
                    
        terraforming_cost = alpha * p_total
        sim_time = simulation.get_algo_time()

    return {
        'makespan': makespan,
        'service_time': avg_service_time,
        'sim_time': sim_time,
        'terraforming_cost': terraforming_cost,
        'seed': seed,
        'failed': False,
        'warnings': warnings
    }


if __name__ == '__main__':
    print("\n" + "="*50)
    print(" STARTING MULTI-SCENARIO SENSITIVITY ANALYSIS")
    print("="*50)

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as f:
        config = json.load(f)
        
    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)

    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots7', map_name_clean)
    os.makedirs(base_plots_dir, exist_ok=True)
    
    map_img_path = os.path.join(base_plots_dir, 'map_preview.png')
    generate_map_preview(map_path, map_img_path)
    
    print(f"Test Map: {map_yaml_name}")
    print(f"Base Directory: {base_plots_dir}")
    print(f"Map Preview Generated: {map_img_path}\n")

    cores = max(1, multiprocessing.cpu_count() - 2)
    
    # Standard Plot Colors Dictionary Setup
    colors_st = {'Alpha': '#d62728', 'Radius': '#1f77b4', 'Laziness': '#2ca02c'}
    colors_ms = {'Alpha': '#ff7f0e', 'Radius': '#9467bd', 'Laziness': '#8c564b'}
    colors_sim = {'Alpha': '#2ca02c', 'Radius': '#d62728', 'Laziness': '#1f77b4'}
    colors_tf = {'Alpha': '#9467bd', 'Radius': '#8c564b', 'Laziness': '#e377c2'}

    print(">>> Calculating Standard Baselines (Lower and Upper Bounds) <<<")
    baseline_payloads = [{'yaml_path': map_path, 'seed': seed} for seed in range(1, NUM_RUNS + 1)]
    pool = PoolWithSubprocess(processes=cores)
    try:
        baseline_results = pool.map_async(baseline_worker_task, baseline_payloads).get(timeout=86400)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n[!] Process Forcefully Terminated by User...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    lb_ms = [r['lb_makespan'] for r in baseline_results if not r['lb_failed']]
    lb_st = [r['lb_service_time'] for r in baseline_results if not r['lb_failed']]
    ub_ms = [r['ub_makespan'] for r in baseline_results if not r['ub_failed']]
    ub_st = [r['ub_service_time'] for r in baseline_results if not r['ub_failed']]

    BASELINES = {
        'LB': {
            'makespan': np.mean(lb_ms) if lb_ms else None,
            'service_time': np.mean(lb_st) if lb_st else None
        },
        'UB': {
            'makespan': np.mean(ub_ms) if ub_ms else None,
            'service_time': np.mean(ub_st) if ub_st else None
        }
    }
    lb_ms_str = f"{BASELINES['LB']['makespan']:.2f}" if BASELINES['LB']['makespan'] is not None else "N/A (Failed)"
    lb_st_str = f"{BASELINES['LB']['service_time']:.2f}" if BASELINES['LB']['service_time'] is not None else "N/A"
    
    ub_ms_str = f"{BASELINES['UB']['makespan']:.2f}" if BASELINES['UB']['makespan'] is not None else "N/A (Disconnected Graph)"
    ub_st_str = f"{BASELINES['UB']['service_time']:.2f}" if BASELINES['UB']['service_time'] is not None else "N/A"

    print(f"    - LB (No Movable Obstacles) -> Makespan: {lb_ms_str}, Service Time: {lb_st_str}")
    print(f"    - UB (Static Blockades)     -> Makespan: {ub_ms_str}, Service Time: {ub_st_str}\n")


    for scen_name, base_params in SCENARIOS.items():
        print(f">>> Executing Scenario: {scen_name} <<<")
        
        scen_dir = os.path.join(base_plots_dir, scen_name)
        os.makedirs(scen_dir, exist_ok=True)
        
        master_results = {'baselines': BASELINES}

        for param_name, sweep_vals in SWEEP_VALUES.items():
            print(f"--- Sweeping Parameter: {param_name} ---")
            
            master_results[param_name] = {
                'values': sweep_vals,
                'base_settings': base_params.copy(),
                
                'avg_service_time': [], 'std_service_time': [],
                'avg_makespan': [], 'std_makespan': [],
                'avg_sim_time': [], 'std_sim_time': [],
                'avg_terraforming_cost': [], 'std_terraforming_cost': [],
                
                'raw_data': {}
            }

            for val in sweep_vals:
                print(f"  Testing {param_name} = {val} | Parallel Runs: {NUM_RUNS}...")
                
                payloads = []
                for seed in range(1, NUM_RUNS + 1):
                    payload = {
                        'yaml_path': map_path,
                        'seed': seed,
                        'alpha': val if param_name == 'Alpha' else base_params['alpha'],
                        'radius': val if param_name == 'Radius' else base_params['radius'],
                        'laziness': val if param_name == 'Laziness' else base_params['laziness']
                    }
                    payloads.append(payload)

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
                            all_warnings.append(f"Seed {r['seed']}: {w}")
                
                if failed_count > 0:
                    print(f"      [!] WARNING: {failed_count} runs ended in deadlock or collision (excluded).")
                    
                if all_warnings:
                    print(f"      [!] CRITICAL LOGS DETECTED:")
                    for w in sorted(list(set(all_warnings))): 
                        print(f"          - {w}")

                if successful_results:
                    service_times = [r['service_time'] for r in successful_results]
                    makespans = [r['makespan'] for r in successful_results]
                    sim_times = [r['sim_time'] for r in successful_results]
                    tf_costs = [r['terraforming_cost'] for r in successful_results]

                    mean_st = np.mean(service_times); std_st = np.std(service_times)
                    mean_ms = np.mean(makespans); std_ms = np.std(makespans)
                    mean_sim = np.mean(sim_times); std_sim = np.std(sim_times)
                    mean_tf = np.mean(tf_costs); std_tf = np.std(tf_costs)
                else:
                    # Invece di 0, usiamo np.nan. Matplotlib ignorerà questi punti nei grafici!
                    mean_st = std_st = mean_ms = std_ms = mean_sim = std_sim = mean_tf = std_tf = np.nan
                    service_times = makespans = sim_times = tf_costs = []

                master_results[param_name]['avg_service_time'].append(mean_st)
                master_results[param_name]['std_service_time'].append(std_st)
                master_results[param_name]['avg_makespan'].append(mean_ms)
                master_results[param_name]['std_makespan'].append(std_ms)
                master_results[param_name]['avg_sim_time'].append(mean_sim)
                master_results[param_name]['std_sim_time'].append(std_sim)
                master_results[param_name]['avg_terraforming_cost'].append(mean_tf)
                master_results[param_name]['std_terraforming_cost'].append(std_tf)
                
                master_results[param_name]['raw_data'][str(val)] = {
                    'service_times': service_times,
                    'makespans': makespans,
                    'sim_times': sim_times,
                    'terraforming_costs': tf_costs
                }

                print(f"    -> Avg ST: {mean_st:.2f} | Avg MS: {mean_ms:.2f} | Avg Sim Time: {mean_sim:.2f}s | Avg TF Cost: {mean_tf:.2f}")

        json_path = os.path.join(scen_dir, 'scenario_results.json')
        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)

        # Generate the 4 core plots
        for param_name, data in master_results.items():
            if param_name == 'baselines':
                continue
                
            x = data['values']
            fixed_text = "Base Params:\n"
            for k, v in data['base_settings'].items():
                if k.capitalize() != param_name:
                    fixed_text += f"{k.capitalize()} = {v}\n"

            # 1. Plot Service Time
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['avg_service_time'], yerr=data['std_service_time'], fmt='-o', color=colors_st[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Avg Service Time')
            
            if BASELINES['LB']['service_time'] is not None:
                plt.axhline(BASELINES['LB']['service_time'], color='green', linestyle='--', linewidth=2, label='Lower Bound (No Movable Obs)')
            if BASELINES['UB']['service_time'] is not None:
                plt.axhline(BASELINES['UB']['service_time'], color='red', linestyle='--', linewidth=2, label='Upper Bound (Static M.O.)')
            
            plt.title(f'{scen_name}: Service Time vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name} Value', fontsize=12)
            plt.ylabel('Average Service Time (Timesteps)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_ServiceTime_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            # 2. Plot Makespan
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['avg_makespan'], yerr=data['std_makespan'], fmt='-s', color=colors_ms[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Avg Makespan')
            
            if BASELINES['LB']['makespan'] is not None:
                plt.axhline(BASELINES['LB']['makespan'], color='green', linestyle='--', linewidth=2, label='Lower Bound (No Movable Obs)')
            if BASELINES['UB']['makespan'] is not None:
                plt.axhline(BASELINES['UB']['makespan'], color='red', linestyle='--', linewidth=2, label='Upper Bound (Static M.O.)')
            
            plt.title(f'{scen_name}: Makespan vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name} Value', fontsize=12)
            plt.ylabel('Average Makespan (Timesteps)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            # 3. Plot Simulation Time (Algorithm Computation Time)
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['avg_sim_time'], yerr=data['std_sim_time'], fmt='-^', color=colors_sim[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='Avg Sim Time')
            plt.title(f'{scen_name}: Algorithmic Sim Time vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name} Value', fontsize=12)
            plt.ylabel('Simulation Processing Time (Seconds)', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_SimTime_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

            # 4. Plot True Terraforming Cost
            plt.figure(figsize=(8, 6))
            plt.errorbar(x, data['avg_terraforming_cost'], yerr=data['std_terraforming_cost'], fmt='-d', color=colors_tf[param_name], 
                         ecolor='gray', elinewidth=1.5, capsize=5, capthick=1.5, markersize=8, label='True TF Cost (α * P_total)')
            plt.title(f'{scen_name}: True Terraforming Cost vs {param_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f'{param_name} Value', fontsize=12)
            plt.ylabel('Energy/Penalty Score', fontsize=12)
            plt.text(0.05, 0.95, fixed_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(scen_dir, f'Plot_TerraformingCost_{param_name}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')

        print(f"[+] Scenario {scen_name} completed. Data saved in {scen_dir}\n")

    print("==================================================")
    print(" ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print("==================================================\n")