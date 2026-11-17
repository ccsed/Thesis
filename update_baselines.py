import os
import json
import yaml
import random
import copy
import numpy as np
import sys
import multiprocessing
import io
from collections import defaultdict

import RoothPath
from Utils.pool_with_subprocess import PoolWithSubprocess
from Simulation.simulation_new_recovery import SimulationNewRecovery
from Simulation.TP_with_recovery import TokenPassingRecovery
from Simulation.tasks_and_delays_maker import gen_tasks_and_delays

# =====================================================================================
# CONFIGURATION
# =====================================================================================
NUM_RUNS_PARAMS = 25   
NUM_RUNS_SCALE = 25    
MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000

MAX_AGENTS_CAP = 10  # <-- REINSERITO IL CAP A 10

SCENARIOS = ["Scenario_1", "Scenario_2", "Scenario_3"]

N_TASKS_SWEEP_VALUES = [10, 20, 30, 50, 80, 100]
DEFAULT_N_TASKS_FOR_PARAM_SWEEP = 20
DEFAULT_TASK_FREQ = 1.0
DEFAULT_N_DELAYS_PER_AGENT = 0

SCALE_TASK_FREQ = 0.2
SCALE_N_DELAYS_PER_AGENT = 0
SCALE_DEFAULT_N_TASKS = 20

# =====================================================================================
# LOGGING / VALIDATION HELPERS
# =====================================================================================

class LogCatcher:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

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
        last_pos = path[0] if path else {'x': 0, 'y': 0}
        for t in range(max_time + 1):
            if t in path_dict: last_pos = path_dict[t]
            continuous_path.append({'t': t, 'x': last_pos['x'], 'y': last_pos['y']})
        full_paths[obs_id] = continuous_path
    for entity, path in full_paths.items():
        for step in path:
            state_key = f"{step['t']}_{step['x']}_{step['y']}"
            states_dict[state_key] += 1
            if states_dict[state_key] > 1: return True
    return False

# =====================================================================================
# BASELINE WORKERS
# =====================================================================================

def _run_baseline_once(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints, tasks, delays, mode, all_task_endpoints):
    obs_run = obstacles if mode == 'lb' else obstacles + movable_obstacles
    failed = False
    sim_time = 0
    with LogCatcher() as catcher:
        try:
            sim = SimulationNewRecovery(copy.deepcopy(tasks), agents, delays=copy.deepcopy(delays))
            tp = TokenPassingRecovery(
                agents, dimensions, obs_run, non_task_endpoints, sim,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=1, new_recovery=True, movable_obstacles=[], alpha=1.0, laziness=1, all_task_endpoints=all_task_endpoints
            )
            while tp.get_completed_tasks() != len(tasks) and sim.get_time() < MAX_SIM_TIME:
                sim.time_forward(tp)
            if sim.get_time() >= MAX_SIM_TIME or verify_no_collisions(sim, tp):
                failed = True
            else:
                sim_time = sim.get_algo_time()  
        except SystemExit:
            failed = True
    return failed, sim_time


def baseline_worker_task_params(payload):
    yaml_path = payload['yaml_path']
    seed = payload['seed']
    with open(yaml_path, 'r') as f: param = yaml.safe_load(f)
    agents = param['agents']
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']
    all_task_endpoints = param['map']['start_locations'] + param['map']['goal_locations']

    random.seed(seed)
    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'], 
        DEFAULT_N_TASKS_FOR_PARAM_SWEEP, DEFAULT_TASK_FREQ, DEFAULT_N_DELAYS_PER_AGENT
    )

    lb_failed, lb_sim = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints, tasks, delays, 'lb', all_task_endpoints)
    ub_failed, ub_sim = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints, tasks, delays, 'ub', all_task_endpoints)
    return {'lb_failed': lb_failed, 'lb_sim': lb_sim, 'ub_failed': ub_failed, 'ub_sim': ub_sim}


def baseline_worker_task_scale(payload):
    yaml_path = payload['yaml_path']
    seed = payload['seed']
    n_agents, n_tasks = payload['n_agents'], payload['n_tasks']
    with open(yaml_path, 'r') as f: param = yaml.safe_load(f)
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']
    all_task_endpoints = param['map']['start_locations'] + param['map']['goal_locations']

    random.seed(seed)
    n_agents = min(n_agents, len(non_task_endpoints))
    agents = [{'name': f'agent{i}', 'start': ep, 'goal': ep} for i, ep in enumerate(random.sample(non_task_endpoints, n_agents))]
    tasks, delays = gen_tasks_and_delays(agents, param['map']['start_locations'], param['map']['goal_locations'], n_tasks, SCALE_TASK_FREQ, SCALE_N_DELAYS_PER_AGENT)

    lb_failed, lb_sim = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints, tasks, delays, 'lb', all_task_endpoints)
    ub_failed, ub_sim = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints, tasks, delays, 'ub', all_task_endpoints)
    return {'lb_failed': lb_failed, 'lb_sim': lb_sim, 'ub_failed': ub_failed, 'ub_sim': ub_sim}

# =====================================================================================
# MAIN
# =====================================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" BASELINE EXECUTION TIME CALCULATOR & JSON UPDATER ")
    print("=" * 60)

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as f:
        config = json.load(f)

    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)
    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots_final', map_name_clean)

    with open(map_path, 'r') as f:
        map_param = yaml.safe_load(f)
    
    # <-- CORREZIONE QUI: IL MASSIMO E' CAPPATO A 10 (o a quello che specifichi in MAX_AGENTS_CAP)
    max_possible_agents = min(len(map_param['map']['non_task_endpoints']), MAX_AGENTS_CAP)
    default_agents = max(1, max_possible_agents // 2)
    n_agents_sweep_values = list(range(1, max_possible_agents + 1))
    
    print(f"Max agents capped at: {max_possible_agents}")
    cores = max(1, multiprocessing.cpu_count() - 2)

    # 1. Calc Params Baseline (Global)
    print(">>> Calculating Execution Time for Params Sweep (Alpha/Radius/Laziness)...")
    payloads = [{'yaml_path': map_path, 'seed': seed} for seed in range(1, NUM_RUNS_PARAMS + 1)]
    pool = PoolWithSubprocess(processes=cores)
    res_params = pool.map_async(baseline_worker_task_params, payloads).get(timeout=86400)
    pool.close(); pool.join()
    
    lb_ok_params = [r['lb_sim'] for r in res_params if not r['lb_failed']]
    ub_ok_params = [r['ub_sim'] for r in res_params if not r['ub_failed']]
    
    lb_sim_val = np.mean(lb_ok_params) if lb_ok_params else None
    ub_sim_val = np.mean(ub_ok_params) if ub_ok_params else None
    
    print(f"    LB Execution Time: {lb_sim_val}")
    print(f"    UB Execution Time: {ub_sim_val}")

    # 2. Calc Scale Baseline (N_Agents, N_Tasks)
    print("\n>>> Calculating Execution Time for Scale Sweep (N_Agents, N_Tasks)...")
    SCALE_SIM_RESULTS = {"N_Agents": {"lb_sim": [], "ub_sim": []}, "N_Tasks": {"lb_sim": [], "ub_sim": []}}
    
    for param_name, sweep_vals in [("N_Agents", n_agents_sweep_values), ("N_Tasks", N_TASKS_SWEEP_VALUES)]:
        for val in sweep_vals:
            n_ag = val if param_name == "N_Agents" else default_agents
            n_ts = val if param_name == "N_Tasks" else SCALE_DEFAULT_N_TASKS
            payloads = [{'yaml_path': map_path, 'seed': s, 'n_agents': n_ag, 'n_tasks': n_ts} for s in range(1, NUM_RUNS_SCALE + 1)]
            pool = PoolWithSubprocess(processes=cores)
            res_scale = pool.map_async(baseline_worker_task_scale, payloads).get(timeout=86400)
            pool.close(); pool.join()
            
            lb_ok = [r['lb_sim'] for r in res_scale if not r['lb_failed']]
            ub_ok = [r['ub_sim'] for r in res_scale if not r['ub_failed']]
            SCALE_SIM_RESULTS[param_name]["lb_sim"].append(np.mean(lb_ok) if lb_ok else np.nan)
            SCALE_SIM_RESULTS[param_name]["ub_sim"].append(np.mean(ub_ok) if ub_ok else np.nan)

    # 3. Update JSONs
    print("\n>>> Updating scenario_results.json files...")
    for scen in SCENARIOS:
        json_path = os.path.join(base_plots_dir, scen, 'scenario_results.json')
        if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f: data = json.load(f)

        # Update Params Sweep
        if 'baselines_params_sweep' in data:
            if 'LB' in data['baselines_params_sweep']: data['baselines_params_sweep']['LB']['sim_time'] = lb_sim_val
            if 'UB' in data['baselines_params_sweep']: data['baselines_params_sweep']['UB']['sim_time'] = ub_sim_val

        # Update Scale Sweep
        if 'baselines_scale_sweep' in data:
            for p in ["N_Agents", "N_Tasks"]:
                if p in data['baselines_scale_sweep']:
                    data['baselines_scale_sweep'][p]['lb_sim'] = SCALE_SIM_RESULTS[p]['lb_sim']
                    data['baselines_scale_sweep'][p]['ub_sim'] = SCALE_SIM_RESULTS[p]['ub_sim']

        with open(json_path, 'w') as f: json.dump(data, f, indent=4)
        print(f"  [+] Added Execution Time bounds to {scen}/scenario_results.json")

    print("\n[✓] All JSONs updated successfully!\n")