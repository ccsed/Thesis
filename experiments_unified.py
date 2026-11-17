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

# =====================================================================================
# CONFIGURATION
# =====================================================================================

# Number of seeds per configuration. Matches experiments1.py (param sweeps) and
# experiments2.py (scalability sweeps) respectively -- these were different in the
# original scripts (50 vs 25), kept as-is here. Change if you want them equal.
NUM_RUNS_PARAMS = 25   # Alpha / Radius / Laziness sweeps
NUM_RUNS_SCALE = 25    # N_Agents / N_Tasks sweeps

MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000

# Hard cap on how many agents the N_Agents sweep will ever request, on top of the
# map-dependent cap (min(#non_task_endpoints, #goal_locations)). This existed in the
# original experiments2.py (hardcoded 6) purely to bound runtime on large maps.
# Raise/lower this per map if needed.
MAX_AGENTS_CAP = 10

# Three presets spanning low / mid / high alpha, reusing three of the four original
# (alpha, radius, laziness) presets from experiments1.py. No scenario "story names"
# are used anymore -- see thesis Ch.5 framing (behavior is explained by the parameter
# values themselves, not by a named preset).
SCENARIOS = {
    "Scenario_1": {"alpha": 0.1,  "radius": 3, "laziness": 5},
    "Scenario_2": {"alpha": 1.0,  "radius": 3, "laziness": 1},
    "Scenario_3": {"alpha": 20.0, "radius": 5, "laziness": 3},
}

# Alpha / Radius / Laziness sweep grids -- shared across all maps.
PARAM_SWEEP_VALUES = {
    "Alpha": [0.1, 0.5, 1.0, 5.0, 10, 20, 100],
    "Radius": [1, 2, 3, 4, 5, 6],
    "Laziness": [1, 2, 3, 4, 5],
}

# N_Tasks sweep grid -- shared across all maps. N_Agents grid is computed per-map
# (see main loop), since it depends on how many endpoints the map actually has.
N_TASKS_SWEEP_VALUES = [10, 20, 30, 50, 80, 100]

# Defaults used while sweeping Alpha/Radius/Laziness (team size and task load are
# held fixed at these values -- "gli altri parametri fissi").
DEFAULT_N_TASKS_FOR_PARAM_SWEEP = 20
DEFAULT_TASK_FREQ = 1.0
DEFAULT_N_DELAYS_PER_AGENT = 0

# Defaults used while sweeping N_Agents/N_Tasks (this mirrors experiments2.py).
SCALE_TASK_FREQ = 0.2
SCALE_N_DELAYS_PER_AGENT = 0
SCALE_DEFAULT_N_TASKS = 20


# =====================================================================================
# LOGGING / VALIDATION HELPERS  (unchanged from experiments1.py / experiments2.py)
# =====================================================================================

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


# =====================================================================================
# MAP PREVIEW  -- redesigned to distinguish parking / start / goal endpoints even
# when they occupy the same cell (drawn as pie-wedge slices instead of overlapping
# rectangles, which previously just hid whichever role was drawn last).
# =====================================================================================

ROLE_STYLE = {
    'parking':  {'color': '#cce5ff', 'edge': '#0066cc', 'label': 'Parking Endpoint'},
    'start':    {'color': '#ffe5cc', 'edge': '#cc7a00', 'label': 'Start Endpoint'},
    'goal':     {'color': '#e5f5e0', 'edge': '#31a354', 'label': 'Goal Endpoint'},
}


def _draw_role_wedges(ax, x, y, roles):
    """Draw a unit cell at (x, y) [bottom-left corner] split into equal pie-wedge
    slices, one per role present in `roles` (a list of role keys from ROLE_STYLE).
    If only one role is present, the cell is simply filled with that role's color."""
    from matplotlib.patches import Wedge
    cx, cy = x + 0.5, y + 0.5
    n = len(roles)
    if n == 0:
        return
    if n == 1:
        style = ROLE_STYLE[roles[0]]
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=style['color'],
                                        edgecolor=style['edge'], linewidth=1.2, zorder=1))
        return
    angle_step = 360.0 / n
    r = 0.62  # slightly larger than half-diagonal so wedges fully cover the cell
    for i, role in enumerate(roles):
        style = ROLE_STYLE[role]
        theta1 = i * angle_step
        theta2 = (i + 1) * angle_step
        ax.add_patch(Wedge((cx, cy), r, theta1, theta2, facecolor=style['color'],
                            edgecolor=style['edge'], linewidth=1.0, zorder=1))
    # crisp cell border on top so wedges don't bleed into neighboring cells visually
    ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='none', edgecolor='#888888',
                                    linewidth=0.5, zorder=1.5))


def generate_map_preview(yaml_path, output_filename):
    with open(yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)

    dims = map_data['map']['dimensions']
    static_obs = map_data['map'].get('obstacles', [])
    movable_obs = map_data['map'].get('movable_obstacles', [])
    parking_endpoints = map_data['map'].get('non_task_endpoints', [])
    start_locs = map_data['map'].get('start_locations', [])
    goal_locs = map_data['map'].get('goal_locations', [])
    agents = map_data.get('agents', [])

    # Build role membership per cell.
    cell_roles = defaultdict(list)
    for ep in parking_endpoints:
        cell_roles[tuple(ep)].append('parking')
    for ep in start_locs:
        cell_roles[tuple(ep)].append('start')
    for ep in goal_locs:
        cell_roles[tuple(ep)].append('goal')

    fig, ax = plt.subplots(figsize=(max(6, dims[0] * 0.9), max(6, dims[1] * 0.9)))
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_xticks(range(dims[0] + 1))
    ax.set_yticks(range(dims[1] + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=9)

    def draw_square(coords, color, edgecolor='black', hatch=None, zorder=1, linestyle='solid', linewidth=1):
        x, y = coords[0], coords[1]
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=edgecolor,
                                        hatch=hatch, zorder=zorder, linestyle=linestyle, linewidth=linewidth))

    # Endpoint cells (parking / start / goal, with overlap handled as wedges)
    for (x, y), roles in cell_roles.items():
        _draw_role_wedges(ax, x, y, roles)

    # Obstacles are drawn on top of endpoint roles (a movable obstacle can rest on an
    # otherwise-unused endpoint cell at scenario-generation time).
    for obs in static_obs:
        draw_square(obs, '#333333', zorder=2)

    for mobs in movable_obs:
        draw_square(mobs, '#ff9900', edgecolor='#cc0000', hatch='///', zorder=3)

    # Agents drawn last, on top of everything.
    for agent in agents:
        start = agent['start']
        cx, cy = start[0] + 0.5, start[1] + 0.5
        ax.add_patch(patches.Circle((cx, cy), 0.35, facecolor='#0033cc', edgecolor='black', zorder=4))
        label = agent['name'].replace('agent', 'A')
        ax.text(cx, cy, label, color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)

    legend_elements = [
        patches.Patch(facecolor='#333333', edgecolor='black', label='Static Obstacle'),
        patches.Patch(facecolor='#ff9900', edgecolor='#cc0000', hatch='///', label='Movable Obstacle'),
        patches.Patch(facecolor=ROLE_STYLE['parking']['color'], edgecolor=ROLE_STYLE['parking']['edge'], label=ROLE_STYLE['parking']['label']),
        patches.Patch(facecolor=ROLE_STYLE['start']['color'], edgecolor=ROLE_STYLE['start']['edge'], label=ROLE_STYLE['start']['label']),
        patches.Patch(facecolor=ROLE_STYLE['goal']['color'], edgecolor=ROLE_STYLE['goal']['edge'], label=ROLE_STYLE['goal']['label']),
        Line2D([0], [0], marker='o', color='w', label='Agent', markerfacecolor='#0033cc', markersize=12, markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=9)
    plt.title("Map Preview", fontsize=16, pad=15)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close('all')


# =====================================================================================
# BASELINE (LB / UB) WORKERS
# =====================================================================================

def _run_baseline_once(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints,
                        tasks, delays, mode, all_task_endpoints=None):
    """mode in {'lb', 'ub'}. Returns (failed, makespan, service_time)."""
    if mode == 'lb':
        obs_run = obstacles
    elif mode == 'ub':
        obs_run = obstacles + movable_obstacles
    else:
        raise ValueError(mode)

    failed = False
    makespan, service_time = 0, 0
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
        except SystemExit:
            failed = True
        else:
            completed_times = tp.get_completed_tasks_times()
            task_pickups = tp.get_pickup_tasks_times()
            makespan = max(completed_times.values()) if completed_times else 0
            delivery_times = [c_time - task_pickups[t_name] for t_name, c_time in completed_times.items()]
            service_time = sum(delivery_times) / len(delivery_times) if delivery_times else 0

    return failed, makespan, service_time


def baseline_worker_task_params(payload):
    """LB/UB for the Alpha/Radius/Laziness sweep -- team & task load fixed at the
    map's own agent set and DEFAULT_N_TASKS_FOR_PARAM_SWEEP. Mirrors
    experiments1.py::baseline_worker_task."""
    yaml_path = payload['yaml_path']
    seed = payload['seed']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    agents = param['agents']
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']
    all_task_endpoints = param['map']['start_locations'] + param['map']['goal_locations']

    n_tasks = param.get('n_tasks', DEFAULT_N_TASKS_FOR_PARAM_SWEEP)
    task_freq = param.get('task_freq', DEFAULT_TASK_FREQ)
    n_delays_per_agent = param.get('n_delays_per_agent', DEFAULT_N_DELAYS_PER_AGENT)

    random.seed(seed)
    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, task_freq, n_delays_per_agent
    )

    lb_failed, lb_ms, lb_st = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles,
                                                  non_task_endpoints, tasks, delays, 'lb', all_task_endpoints=all_task_endpoints)
    ub_failed, ub_ms, ub_st = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles,
                                                  non_task_endpoints, tasks, delays, 'ub', all_task_endpoints=all_task_endpoints)

    return {
        'seed': seed,
        'lb_failed': lb_failed, 'lb_makespan': lb_ms, 'lb_service_time': lb_st,
        'ub_failed': ub_failed, 'ub_makespan': ub_ms, 'ub_service_time': ub_st,
    }


def baseline_worker_task_scale(payload):
    """LB/UB for one (param_name, value) point of the N_Agents/N_Tasks sweep.
    Does NOT depend on scenario (alpha/radius/laziness), only on team/task size --
    computed once per map and reused across all three scenarios (see main loop)."""
    yaml_path = payload['yaml_path']
    seed = payload['seed']
    n_agents = payload['n_agents']
    n_tasks = payload['n_tasks']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']
    all_task_endpoints = param['map']['start_locations'] + param['map']['goal_locations']

    random.seed(seed)
    n_agents = min(n_agents, len(non_task_endpoints))
    selected_endpoints = random.sample(non_task_endpoints, n_agents)
    agents = [{'name': f'agent{i}', 'start': ep, 'goal': ep} for i, ep in enumerate(selected_endpoints)]

    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, SCALE_TASK_FREQ, SCALE_N_DELAYS_PER_AGENT
    )

    lb_failed, lb_ms, lb_st = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles,
                                                  non_task_endpoints, tasks, delays, 'lb', all_task_endpoints=all_task_endpoints)
    ub_failed, ub_ms, ub_st = _run_baseline_once(agents, dimensions, obstacles, movable_obstacles,
                                                  non_task_endpoints, tasks, delays, 'ub', all_task_endpoints=all_task_endpoints)

    return {
        'seed': seed,
        'lb_failed': lb_failed, 'lb_makespan': lb_ms, 'lb_service_time': lb_st,
        'ub_failed': ub_failed, 'ub_makespan': ub_ms, 'ub_service_time': ub_st,
    }


# =====================================================================================
# T-A* (TERRAFORMING) WORKERS
# =====================================================================================

def worker_task_params(payload):
    """One T-A* run for a given Alpha/Radius/Laziness sweep point. Team/task load
    fixed at the map's own agent set and DEFAULT_N_TASKS_FOR_PARAM_SWEEP. Mirrors
    experiments1.py::worker_task."""
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
    all_task_endpoints = param['map']['start_locations'] + param['map']['goal_locations']

    n_tasks = param.get('n_tasks', DEFAULT_N_TASKS_FOR_PARAM_SWEEP)
    task_freq = param.get('task_freq', DEFAULT_TASK_FREQ)
    n_delays_per_agent = param.get('n_delays_per_agent', DEFAULT_N_DELAYS_PER_AGENT)

    random.seed(seed)
    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, task_freq, n_delays_per_agent
    )

    return _run_terraforming(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints,
                              tasks, delays, alpha, radius, laziness, seed, all_task_endpoints=all_task_endpoints)


def worker_task_scale(payload):
    """One T-A* run for a given N_Agents/N_Tasks sweep point, at a fixed scenario
    (alpha/radius/laziness). Mirrors experiments2.py::worker_task (mode='standard' path)."""
    yaml_path = payload['yaml_path']
    seed = payload['seed']
    n_agents = payload['n_agents']
    n_tasks = payload['n_tasks']
    alpha = payload['alpha']
    radius = payload['radius']
    laziness = payload['laziness']

    with open(yaml_path, 'r') as f:
        param = yaml.safe_load(f)

    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    movable_obstacles = param['map'].get('movable_obstacles', [])
    non_task_endpoints = param['map']['non_task_endpoints']
    all_task_endpoints = param['map']['start_locations'] + param['map']['goal_locations']

    random.seed(seed)
    n_agents = min(n_agents, len(non_task_endpoints))
    selected_endpoints = random.sample(non_task_endpoints, n_agents)
    agents = [{'name': f'agent{i}', 'start': ep, 'goal': ep} for i, ep in enumerate(selected_endpoints)]

    tasks, delays = gen_tasks_and_delays(
        agents, param['map']['start_locations'], param['map']['goal_locations'],
        n_tasks, SCALE_TASK_FREQ, SCALE_N_DELAYS_PER_AGENT
    )

    return _run_terraforming(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints,
                              tasks, delays, alpha, radius, laziness, seed, all_task_endpoints=all_task_endpoints)


def _run_terraforming(agents, dimensions, obstacles, movable_obstacles, non_task_endpoints,
                       tasks, delays, alpha, radius, laziness, seed, all_task_endpoints=None):
    with LogCatcher() as catcher:
        try:
            simulation = SimulationNewRecovery(tasks, agents, delays=delays)
            tp = TokenPassingRecovery(
                agents, dimensions, obstacles, non_task_endpoints, simulation,
                a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                terraforming_radius=radius, new_recovery=True,
                movable_obstacles=movable_obstacles, alpha=alpha, laziness=laziness, all_task_endpoints=all_task_endpoints
            )
            while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < MAX_SIM_TIME:
                simulation.time_forward(tp)

            warnings = catcher.get_critical_warnings()
            used_deadlock = catcher.has_deadlock_recovery()
        except SystemExit:
            warnings = catcher.get_critical_warnings()
            return {'failed': True, 'seed': seed, 'warnings': warnings}

        # Deadlock-recovered runs excluded from stats, exactly as in experiments1.py.
        # if used_deadlock:
        #     warnings.append("DEADLOCK RECOVERY TRIGGERED (kept in stats)")
            # return {'failed': True, 'seed': seed, 'warnings': warnings}

        if simulation.get_time() >= MAX_SIM_TIME:
            warnings.append("TIMEOUT: MAX_SIM_TIME reached.")
            return {'failed': True, 'seed': seed, 'warnings': warnings}

        if verify_no_collisions(simulation, tp):
            warnings.append("PHYSICAL COLLISION DETECTED DURING VALIDATION!")
            return {'failed': True, 'seed': seed, 'warnings': warnings}

        completed_times = tp.get_completed_tasks_times()
        task_pickups = tp.get_pickup_tasks_times()
        makespan = max(completed_times.values()) if completed_times else 0

        delivery_times = [c_time - task_pickups[t_name] for t_name, c_time in completed_times.items()]
        avg_service_time = sum(delivery_times) / len(delivery_times) if delivery_times else 0

        p_total = 0
        for obs_id, path in tp.get_token()['obstacles_paths'].items():
            for i in range(1, len(path)):
                if path[i]['x'] != path[i - 1]['x'] or path[i]['y'] != path[i - 1]['y']:
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
        'warnings': warnings,
    }


# =====================================================================================
# PLOTTING HELPERS
# =====================================================================================

# Fixed color-per-metric (not per swept-parameter) so the same quantity always looks
# the same across every figure in the thesis, regardless of which parameter is swept.
COLOR_ST = '#1f77b4'    # Service Time -- blue
COLOR_MS = '#ff7f0e'    # Makespan -- orange
COLOR_SIM = '#2ca02c'   # Sim Time -- green
COLOR_TF = '#9467bd'    # Terraforming Cost -- purple
COLOR_LB = '#2ca02c'    # Lower Bound -- green dashed
COLOR_UB = '#d62728'    # Upper Bound -- red dashed

PARAM_AXIS_LABEL = {
    'Alpha': r'$\alpha$',
    'Radius': r'$k$',
    'Laziness': r'$\lambda$',
    'N_Agents': r'$N_{\mathrm{agents}}$',
    'N_Tasks': r'$N_{\mathrm{tasks}}$',
}


def clip_series(values, lb=None, ub=None):
    """Clip a series for PLOTTING ONLY -- the raw values saved to JSON are never
    touched. This exists because a per-seed A* tie-breaking difference can push the
    *mean* of a handful of seeds a hair outside the theoretical [LB, UB] band; that
    is aggregation noise, not a violation of the bound, and plotting it as such is
    misleading. lb/ub can be a scalar (constant baseline) or an array of the same
    length as `values` (per-sweep-point baseline, as in the N_Agents/N_Tasks sweep)."""
    v = np.array(values, dtype=float)
    if lb is not None:
        lb_arr = np.full_like(v, float(lb)) if np.isscalar(lb) else np.array(lb, dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(lb_arr)
        v[mask] = np.maximum(v[mask], lb_arr[mask])
    if ub is not None:
        ub_arr = np.full_like(v, float(ub)) if np.isscalar(ub) else np.array(ub, dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(ub_arr)
        v[mask] = np.minimum(v[mask], ub_arr[mask])
    return v


def _style_axes(ax):
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def make_plot(x, y, yerr, out_path, *, color, marker, ylabel, param_name,
              metric_title, base_text, lb=None, ub=None, lb_is_series=False,
              ub_is_series=False, clip_bounds=True, force_ylim_zero=False):
    """One standardized figure. lb/ub are either a scalar (drawn as a flat dashed
    line) or, when lb_is_series/ub_is_series, an array plotted point-by-point
    (used for the N_Agents/N_Tasks sweep, where the baseline itself moves)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    y_plot = clip_series(y, lb=lb if clip_bounds else None, ub=ub if clip_bounds else None)

    ax.errorbar(x, y_plot, yerr=yerr, fmt=f'-{marker}', color=color, ecolor='gray',
                elinewidth=1.3, capsize=4, capthick=1.3, markersize=7, linewidth=1.8,
                label='T-A* (Avg)', zorder=3)

    if lb is not None:
        if lb_is_series:
            ax.plot(x, lb, '--', color=COLOR_LB, linewidth=1.8, label='Lower Bound (No Movable Obs.)', zorder=2)
        else:
            ax.axhline(lb, color=COLOR_LB, linestyle='--', linewidth=1.8, label='Lower Bound (No Movable Obs.)', zorder=2)
    if ub is not None:
        if ub_is_series:
            ax.plot(x, ub, '--', color=COLOR_UB, linewidth=1.8, label='Upper Bound (Static Obs.)', zorder=2)
        else:
            ax.axhline(ub, color=COLOR_UB, linestyle='--', linewidth=1.8, label='Upper Bound (Static Obs.)', zorder=2)

    ax.set_xlabel(PARAM_AXIS_LABEL.get(param_name, param_name), fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(metric_title, fontsize=14, fontweight='bold', pad=12)
    if base_text:
        ax.text(0.02, 0.97, base_text, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    if force_ylim_zero:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top)
    _style_axes(ax)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def base_params_text(fixed_params, exclude_key):
    lines = []
    for k, v in fixed_params.items():
        if k != exclude_key:
            lines.append(f"{PARAM_AXIS_LABEL.get(k, k)} = {v}")
    return "Fixed:\n" + "\n".join(lines) if lines else ""


# =====================================================================================
# MAIN
# =====================================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" UNIFIED SENSITIVITY + SCALABILITY ANALYSIS")
    print("=" * 60)

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as f:
        config = json.load(f)

    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)

    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots_final', map_name_clean)
    os.makedirs(base_plots_dir, exist_ok=True)

    map_img_path = os.path.join(base_plots_dir, 'map_preview.png')
    generate_map_preview(map_path, map_img_path)

    print(f"Test Map: {map_yaml_name}")
    print(f"Base Directory: {base_plots_dir}")
    print(f"Map Preview Generated: {map_img_path}\n")

    with open(map_path, 'r') as f:
        map_param = yaml.safe_load(f)
    non_task_endpoints = map_param['map']['non_task_endpoints']
    goal_locations = map_param['map']['goal_locations']

    # N_Agents sweep range: map-dependent, capped by MAX_AGENTS_CAP -- the only
    # sweep grid that is not identical across maps, since it must not exceed the
    # number of endpoints actually available on that map.
    max_possible_agents = len(non_task_endpoints) #min(len(non_task_endpoints), len(goal_locations), MAX_AGENTS_CAP)
    default_agents = max(1, max_possible_agents // 2)
    n_agents_sweep_values = list(range(1, max_possible_agents + 1))

    print(f"Max Agents on this map (capped at {MAX_AGENTS_CAP}): {max_possible_agents}")
    print(f"Default team size while sweeping N_Tasks: {default_agents}\n")

    cores = max(1, multiprocessing.cpu_count() - 2)

    # -----------------------------------------------------------------------------
    # STEP 1: Global LB/UB baseline for the Alpha/Radius/Laziness sweep (team and
    # task load fixed at the map's own defaults). Shared across all 3 scenarios.
    # -----------------------------------------------------------------------------
    print(">>> [1/3] Baselines for Alpha / Radius / Laziness sweep <<<")
    baseline_payloads = [{'yaml_path': map_path, 'seed': seed} for seed in range(1, NUM_RUNS_PARAMS + 1)]
    pool = PoolWithSubprocess(processes=cores)
    try:
        baseline_results = pool.map_async(baseline_worker_task_params, baseline_payloads).get(timeout=86400)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C -- terminating.")
        pool.terminate()
        pool.join()
        sys.exit(1)

    lb_ms = [r['lb_makespan'] for r in baseline_results if not r['lb_failed']]
    lb_st = [r['lb_service_time'] for r in baseline_results if not r['lb_failed']]
    ub_ms = [r['ub_makespan'] for r in baseline_results if not r['ub_failed']]
    ub_st = [r['ub_service_time'] for r in baseline_results if not r['ub_failed']]

    PARAM_BASELINES = {
        'LB': {'makespan': np.mean(lb_ms) if lb_ms else None, 'service_time': np.mean(lb_st) if lb_st else None},
        'UB': {'makespan': np.mean(ub_ms) if ub_ms else None, 'service_time': np.mean(ub_st) if ub_st else None},
    }
    print(f"    LB -> Makespan: {PARAM_BASELINES['LB']['makespan']}, Service Time: {PARAM_BASELINES['LB']['service_time']}")
    print(f"    UB -> Makespan: {PARAM_BASELINES['UB']['makespan']}, Service Time: {PARAM_BASELINES['UB']['service_time']}")
    print("    (UB may be None: on some well-formed maps no valid plan exists when")
    print("     movable obstacles are frozen in place -- see Ch.3 well-formedness.)\n")

    # -----------------------------------------------------------------------------
    # STEP 2: LB/UB baseline for the N_Agents/N_Tasks sweep -- computed ONCE per
    # map (it does not depend on alpha/radius/laziness), then reused by all 3
    # scenarios below. This is an efficiency change vs. the original experiments2.py,
    # which recomputed it redundantly inside each scenario's loop.
    # -----------------------------------------------------------------------------
    print(">>> [2/3] Baselines for N_Agents / N_Tasks sweep (shared across scenarios) <<<")
    SCALE_SWEEP_VALUES = {"N_Agents": n_agents_sweep_values, "N_Tasks": N_TASKS_SWEEP_VALUES}
    SCALE_BASELINES = {}  # {param_name: {'lb_ms':[...], 'lb_st':[...], 'ub_ms':[...], 'ub_st':[...]}}

    for param_name, sweep_vals in SCALE_SWEEP_VALUES.items():
        print(f"--- {param_name} ---")
        lb_ms_list, lb_st_list, ub_ms_list, ub_st_list = [], [], [], []
        for val in sweep_vals:
            n_agents = val if param_name == "N_Agents" else default_agents
            n_tasks = val if param_name == "N_Tasks" else SCALE_DEFAULT_N_TASKS
            payloads = [{'yaml_path': map_path, 'seed': seed, 'n_agents': n_agents, 'n_tasks': n_tasks}
                        for seed in range(1, NUM_RUNS_SCALE + 1)]
            pool = PoolWithSubprocess(processes=cores)
            try:
                results = pool.map_async(baseline_worker_task_scale, payloads).get(timeout=86400)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("\n[!] Ctrl+C -- terminating.")
                pool.terminate()
                pool.join()
                sys.exit(1)

            lb_ok = [r for r in results if not r['lb_failed']]
            ub_ok = [r for r in results if not r['ub_failed']]
            lb_ms_list.append(np.mean([r['lb_makespan'] for r in lb_ok]) if lb_ok else np.nan)
            lb_st_list.append(np.mean([r['lb_service_time'] for r in lb_ok]) if lb_ok else np.nan)
            ub_ms_list.append(np.mean([r['ub_makespan'] for r in ub_ok]) if ub_ok else np.nan)
            ub_st_list.append(np.mean([r['ub_service_time'] for r in ub_ok]) if ub_ok else np.nan)
            print(f"    {param_name}={val}: LB_MS={lb_ms_list[-1]:.2f} LB_ST={lb_st_list[-1]:.2f} "
                  f"UB_MS={ub_ms_list[-1]:.2f} UB_ST={ub_st_list[-1]:.2f}")

        SCALE_BASELINES[param_name] = {
            'lb_ms': lb_ms_list, 'lb_st': lb_st_list, 'ub_ms': ub_ms_list, 'ub_st': ub_st_list,
        }
    print()

    # -----------------------------------------------------------------------------
    # STEP 3: For each of the 3 scenarios, run all 5 sweeps (Alpha, Radius,
    # Laziness, N_Agents, N_Tasks), producing 4 plots each (Service Time, Makespan,
    # Sim Time, Terraforming Cost) = 20 plots per scenario, plus one JSON.
    # -----------------------------------------------------------------------------
    print(">>> [3/3] Running scenarios <<<")
    for scen_name, base_params in SCENARIOS.items():
        print(f"\n>>> Executing {scen_name}  (alpha={base_params['alpha']}, "
              f"radius={base_params['radius']}, laziness={base_params['laziness']}) <<<")

        scen_dir = os.path.join(base_plots_dir, scen_name)
        os.makedirs(scen_dir, exist_ok=True)

        master_results = {
            'baselines_params_sweep': PARAM_BASELINES,
            'baselines_scale_sweep': SCALE_BASELINES,
            'scenario_params': base_params,
        }

        # --- 3a. Alpha / Radius / Laziness sweeps -----------------------------------
        for param_name, sweep_vals in PARAM_SWEEP_VALUES.items():
            print(f"--- Sweeping {param_name} ---")
            entry = {
                'values': sweep_vals, 'base_settings': base_params.copy(),
                'avg_service_time': [], 'std_service_time': [],
                'avg_makespan': [], 'std_makespan': [],
                'avg_sim_time': [], 'std_sim_time': [],
                'avg_terraforming_cost': [], 'std_terraforming_cost': [],
                'raw_data': {},
            }

            for val in sweep_vals:
                print(f"  {param_name} = {val} | seeds: {NUM_RUNS_PARAMS}...")
                payloads = [{
                    'yaml_path': map_path, 'seed': seed,
                    'alpha': val if param_name == 'Alpha' else base_params['alpha'],
                    'radius': val if param_name == 'Radius' else base_params['radius'],
                    'laziness': val if param_name == 'Laziness' else base_params['laziness'],
                } for seed in range(1, NUM_RUNS_PARAMS + 1)]

                pool = PoolWithSubprocess(processes=cores)
                try:
                    results = pool.map_async(worker_task_params, payloads).get(timeout=86400)
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    print("\n[!] Ctrl+C -- terminating.")
                    pool.terminate()
                    pool.join()
                    sys.exit(1)

                ok = [r for r in results if not r.get('failed', False)]
                failed_count = len(results) - len(ok)
                if failed_count > 0:
                    print(f"      [!] {failed_count} runs excluded (deadlock/collision/timeout).")
                all_warnings = [f"Seed {r['seed']}: {w}" for r in results if r.get('warnings') for w in r['warnings']]
                if all_warnings:
                    for w in sorted(set(all_warnings)):
                        print(f"          - {w}")

                if ok:
                    st = [r['service_time'] for r in ok]; ms = [r['makespan'] for r in ok]
                    sim = [r['sim_time'] for r in ok]; tf = [r['terraforming_cost'] for r in ok]
                    mean_st, std_st = np.mean(st), np.std(st)
                    mean_ms, std_ms = np.mean(ms), np.std(ms)
                    mean_sim, std_sim = np.mean(sim), np.std(sim)
                    mean_tf, std_tf = np.mean(tf), np.std(tf)
                else:
                    mean_st = std_st = mean_ms = std_ms = mean_sim = std_sim = mean_tf = std_tf = np.nan
                    st = ms = sim = tf = []

                entry['avg_service_time'].append(mean_st); entry['std_service_time'].append(std_st)
                entry['avg_makespan'].append(mean_ms); entry['std_makespan'].append(std_ms)
                entry['avg_sim_time'].append(mean_sim); entry['std_sim_time'].append(std_sim)
                entry['avg_terraforming_cost'].append(mean_tf); entry['std_terraforming_cost'].append(std_tf)
                entry['raw_data'][str(val)] = {'service_times': st, 'makespans': ms, 'sim_times': sim, 'terraforming_costs': tf}

                print(f"    -> ST={mean_st:.2f} MS={mean_ms:.2f} Sim={mean_sim:.2f}s TF={mean_tf:.2f}")

            master_results[param_name] = entry

            fixed_display_params = {
                **base_params,
                'N_Tasks': DEFAULT_N_TASKS_FOR_PARAM_SWEEP,
                'N_Agents': len(map_param['agents']),
            }
            base_text = base_params_text(fixed_display_params, param_name)
            lb_ms_v, ub_ms_v = PARAM_BASELINES['LB']['makespan'], PARAM_BASELINES['UB']['makespan']
            lb_st_v, ub_st_v = PARAM_BASELINES['LB']['service_time'], PARAM_BASELINES['UB']['service_time']

            make_plot(sweep_vals, entry['avg_service_time'], entry['std_service_time'],
                      os.path.join(scen_dir, f'Plot_ServiceTime_{param_name}.png'),
                      color=COLOR_ST, marker='o', ylabel='Avg. Delivery Time (timesteps)',
                      param_name=param_name, metric_title='Delivery Time (Pickup -> Delivery)', base_text=base_text,
                      lb=lb_st_v, ub=ub_st_v)

            make_plot(sweep_vals, entry['avg_makespan'], entry['std_makespan'],
                      os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'),
                      color=COLOR_MS, marker='s', ylabel='Avg. Makespan (timesteps)',
                      param_name=param_name, metric_title='Makespan', base_text=base_text,
                      lb=lb_ms_v, ub=ub_ms_v)

            make_plot(sweep_vals, entry['avg_sim_time'], entry['std_sim_time'],
                      os.path.join(scen_dir, f'Plot_SimTime_{param_name}.png'),
                      color=COLOR_SIM, marker='^', ylabel='Avg. Sim Time (seconds)',
                      param_name=param_name, metric_title='Computational Cost', base_text=base_text,
                      force_ylim_zero=True)

            make_plot(sweep_vals, entry['avg_terraforming_cost'], entry['std_terraforming_cost'],
                      os.path.join(scen_dir, f'Plot_TerraformingCost_{param_name}.png'),
                      color=COLOR_TF, marker='d', ylabel=r'Terraforming Cost ($\alpha \cdot P_{total}$)',
                      param_name=param_name, metric_title='Terraforming Cost', base_text=base_text,
                      force_ylim_zero=True)

        # --- 3b. N_Agents / N_Tasks sweeps ------------------------------------------
        for param_name, sweep_vals in SCALE_SWEEP_VALUES.items():
            print(f"--- Sweeping {param_name} ---")
            entry = {
                'values': sweep_vals, 'base_settings': base_params.copy(),
                'avg_service_time': [], 'std_service_time': [],
                'avg_makespan': [], 'std_makespan': [],
                'avg_sim_time': [], 'std_sim_time': [],
                'avg_terraforming_cost': [], 'std_terraforming_cost': [],
                'raw_data': {},
            }

            for val in sweep_vals:
                n_agents = val if param_name == "N_Agents" else default_agents
                n_tasks = val if param_name == "N_Tasks" else SCALE_DEFAULT_N_TASKS
                print(f"  {param_name} = {val} [agents={n_agents}, tasks={n_tasks}] | seeds: {NUM_RUNS_SCALE}...")

                payloads = [{
                    'yaml_path': map_path, 'seed': seed, 'n_agents': n_agents, 'n_tasks': n_tasks,
                    'alpha': base_params['alpha'], 'radius': base_params['radius'], 'laziness': base_params['laziness'],
                } for seed in range(1, NUM_RUNS_SCALE + 1)]

                pool = PoolWithSubprocess(processes=cores)
                try:
                    results = pool.map_async(worker_task_scale, payloads).get(timeout=86400)
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    print("\n[!] Ctrl+C -- terminating.")
                    pool.terminate()
                    pool.join()
                    sys.exit(1)

                ok = [r for r in results if not r.get('failed', False)]
                failed_count = len(results) - len(ok)
                if failed_count > 0:
                    print(f"      [!] {failed_count} runs excluded (deadlock/collision/timeout).")
                all_warnings = [f"Seed {r['seed']}: {w}" for r in results if r.get('warnings') for w in r['warnings']]
                if all_warnings:
                    for w in sorted(set(all_warnings)):
                        print(f"          - {w}")

                if ok:
                    st = [r['service_time'] for r in ok]; ms = [r['makespan'] for r in ok]
                    sim = [r['sim_time'] for r in ok]; tf = [r['terraforming_cost'] for r in ok]
                    mean_st, std_st = np.mean(st), np.std(st)
                    mean_ms, std_ms = np.mean(ms), np.std(ms)
                    mean_sim, std_sim = np.mean(sim), np.std(sim)
                    mean_tf, std_tf = np.mean(tf), np.std(tf)
                else:
                    mean_st = std_st = mean_ms = std_ms = mean_sim = std_sim = mean_tf = std_tf = np.nan
                    st = ms = sim = tf = []

                entry['avg_service_time'].append(mean_st); entry['std_service_time'].append(std_st)
                entry['avg_makespan'].append(mean_ms); entry['std_makespan'].append(std_ms)
                entry['avg_sim_time'].append(mean_sim); entry['std_sim_time'].append(std_sim)
                entry['avg_terraforming_cost'].append(mean_tf); entry['std_terraforming_cost'].append(std_tf)
                entry['raw_data'][str(val)] = {'service_times': st, 'makespans': ms, 'sim_times': sim, 'terraforming_costs': tf}

                print(f"    -> ST={mean_st:.2f} MS={mean_ms:.2f} Sim={mean_sim:.2f}s TF={mean_tf:.2f}")

            master_results[param_name] = entry

            base_text = base_params_text(
                {**base_params, ('N_Tasks' if param_name == 'N_Agents' else 'N_Agents'):
                    (SCALE_DEFAULT_N_TASKS if param_name == 'N_Agents' else default_agents)},
                param_name)
            lb_ms_arr = SCALE_BASELINES[param_name]['lb_ms']; ub_ms_arr = SCALE_BASELINES[param_name]['ub_ms']
            lb_st_arr = SCALE_BASELINES[param_name]['lb_st']; ub_st_arr = SCALE_BASELINES[param_name]['ub_st']

            make_plot(sweep_vals, entry['avg_service_time'], entry['std_service_time'],
                      os.path.join(scen_dir, f'Plot_ServiceTime_{param_name}.png'),
                      color=COLOR_ST, marker='o', ylabel='Avg. Delivery Time (timesteps)',
                      param_name=param_name, metric_title='Delivery Time (Pickup -> Delivery)', base_text=base_text,
                      lb=lb_st_arr, ub=ub_st_arr, lb_is_series=True, ub_is_series=True)

            make_plot(sweep_vals, entry['avg_makespan'], entry['std_makespan'],
                      os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'),
                      color=COLOR_MS, marker='s', ylabel='Avg. Makespan (timesteps)',
                      param_name=param_name, metric_title='Makespan', base_text=base_text,
                      lb=lb_ms_arr, ub=ub_ms_arr, lb_is_series=True, ub_is_series=True)

            make_plot(sweep_vals, entry['avg_sim_time'], entry['std_sim_time'],
                      os.path.join(scen_dir, f'Plot_SimTime_{param_name}.png'),
                      color=COLOR_SIM, marker='^', ylabel='Avg. Sim Time (seconds)',
                      param_name=param_name, metric_title='Computational Cost', base_text=base_text,
                      force_ylim_zero=True)

            make_plot(sweep_vals, entry['avg_terraforming_cost'], entry['std_terraforming_cost'],
                      os.path.join(scen_dir, f'Plot_TerraformingCost_{param_name}.png'),
                      color=COLOR_TF, marker='d', ylabel=r'Terraforming Cost ($\alpha \cdot P_{total}$)',
                      param_name=param_name, metric_title='Terraforming Cost', base_text=base_text,
                      force_ylim_zero=True)

        json_path = os.path.join(scen_dir, 'scenario_results.json')
        with open(json_path, 'w') as f:
            json.dump(master_results, f, indent=4)

        print(f"[+] {scen_name} completed. 20 plots + scenario_results.json saved in {scen_dir}\n")

    print("=" * 60)
    print(" ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print(f" Output root: {base_plots_dir}")
    print("=" * 60 + "\n")