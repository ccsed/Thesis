"""
Zero-delay consistency sweep over the thesis experiments (experiments_unified.py).

Every run reproduces worker_task_params / worker_task_scale of experiments_unified.py exactly
(same yaml loading, task generation, all_task_endpoints, MAX_SIM_TIME, ASTAR_MAX_ITER), with
delays forced to zero, and adds the checks of batch_find_conflicts.py:
[BLOCKED], [PLAN CONFLICT], [PLAN DEVIATION], [OBS DEVIATION], [OBS PAST REWRITE], [COLLISION].
Each run is a subprocess with a timeout and its own log; summary.jsonl collects the outcomes.

    python Utils/sweep_unified.py --logdir DIR [--maps M1 ...] [--seeds 1 25] [--jobs 10] [--timeout 300]
                                  [--modes params scale] [--stop-on-fail]
Single runs (used by the sweep, handy to reproduce):
    python Utils/sweep_unified.py --single-params MAP SEED ALPHA RADIUS LAZINESS
    python Utils/sweep_unified.py --single-scale  MAP SEED ALPHA RADIUS LAZINESS N_AGENTS N_TASKS
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import batch_find_conflicts as bfc  # noqa: E402

# Thesis map suite (Table 2), fastest first
THESIS_MAPS = ['cross_with_pocket.yaml', 'input_tunnel.yaml', 'input_cross_scaled.yaml', 'four_rooms.yaml',
               'shared_junction.yaml', 'spine_warehouse.yaml', 'traditional_warehouse_cross.yaml']
MAP_LABEL = {'cross_with_pocket.yaml': 'M1', 'shared_junction.yaml': 'M2', 'input_tunnel.yaml': 'M3',
             'four_rooms.yaml': 'M4', 'input_cross_scaled.yaml': 'M5', 'spine_warehouse.yaml': 'M6',
             'traditional_warehouse_cross.yaml': 'M7'}


def eu():
    import experiments_unified
    return experiments_unified


def run_checked(agents, m, tasks, alpha, radius, laziness, all_task_endpoints, header):
    """Simulation loop of experiments_unified._run_terraforming plus the consistency checks."""
    from Simulation.simulation_new_recovery import SimulationNewRecovery
    from Simulation.TP_with_recovery import TokenPassingRecovery
    E = eu()
    delays = {a['name']: [] for a in agents}
    tee = bfc.Tee(sys.stdout)
    sys.stdout = tee
    print(header)
    simulation = SimulationNewRecovery(tasks, agents, delays=delays)
    tp = TokenPassingRecovery(agents, m['dimensions'], m['obstacles'], m['non_task_endpoints'], simulation,
                              a_star_max_iter=E.ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                              terraforming_radius=radius, new_recovery=True,
                              movable_obstacles=m.get('movable_obstacles', []), alpha=alpha, laziness=laziness,
                              all_task_endpoints=all_task_endpoints)
    monitor = bfc.PlanMonitor(tp, simulation)
    while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < E.MAX_SIM_TIME:
        simulation.time_forward(tp)
        monitor.check_after_move()
    bfc.check_physical_collisions(simulation, tp)
    status = 'FAIL' if tee.hit else ('TIMEOUT_SIM' if simulation.get_time() >= E.MAX_SIM_TIME else 'OK')
    print(f"[END] status={status} time={simulation.get_time()} completed={tp.get_completed_tasks()}/{len(tasks)}")
    sys.stdout = tee.real


def single_params(map_path, seed, alpha, radius, laziness):
    """= experiments_unified.worker_task_params, delays forced to 0."""
    import yaml
    from Simulation.tasks_and_delays_maker import gen_tasks_and_delays
    E = eu()
    with open(map_path) as f:
        param = yaml.safe_load(f)
    m = param['map']
    agents = param['agents']
    assert param.get('n_delays_per_agent', 0) == 0
    n_tasks = param.get('n_tasks', E.DEFAULT_N_TASKS_FOR_PARAM_SWEEP)
    task_freq = param.get('task_freq', E.DEFAULT_TASK_FREQ)
    random.seed(seed)
    tasks, _ = gen_tasks_and_delays(agents, m['start_locations'], m['goal_locations'], n_tasks, task_freq, 0)
    run_checked(agents, m, tasks, alpha, radius, laziness, m['start_locations'] + m['goal_locations'],
                f"[RUN] params map={os.path.basename(map_path)} seed={seed} alpha={alpha} radius={radius} laziness={laziness}")


def single_scale(map_path, seed, alpha, radius, laziness, n_agents, n_tasks):
    """= experiments_unified.worker_task_scale, delays forced to 0."""
    import yaml
    from Simulation.tasks_and_delays_maker import gen_tasks_and_delays
    E = eu()
    with open(map_path) as f:
        param = yaml.safe_load(f)
    m = param['map']
    nte = m['non_task_endpoints']
    random.seed(seed)
    n_agents = min(n_agents, len(nte))
    selected = random.sample(nte, n_agents)
    agents = [{'name': f'agent{i}', 'start': ep, 'goal': ep} for i, ep in enumerate(selected)]
    tasks, _ = gen_tasks_and_delays(agents, m['start_locations'], m['goal_locations'], n_tasks, E.SCALE_TASK_FREQ, 0)
    run_checked(agents, m, tasks, alpha, radius, laziness, m['start_locations'] + m['goal_locations'],
                f"[RUN] scale map={os.path.basename(map_path)} seed={seed} alpha={alpha} radius={radius} "
                f"laziness={laziness} agents={n_agents} tasks={n_tasks}")


def jobs_for(map_name, seeds, modes):
    """All (alpha, radius, laziness[, n_agents, n_tasks]) points experiments_unified runs on this map."""
    import yaml
    E = eu()
    with open(os.path.join(ROOT, 'Environments', map_name)) as f:
        nte = yaml.safe_load(f)['map']['non_task_endpoints']
    param_pts, scale_pts = [], []
    for base in E.SCENARIOS.values():
        for name, vals in E.PARAM_SWEEP_VALUES.items():
            for v in vals:
                c = {'alpha': base['alpha'], 'radius': base['radius'], 'laziness': base['laziness']}
                c[{'Alpha': 'alpha', 'Radius': 'radius', 'Laziness': 'laziness'}[name]] = v
                param_pts.append((float(c['alpha']), int(c['radius']), int(c['laziness'])))
        default_agents = max(1, len(nte) // 2)
        for na in range(1, len(nte) + 1):
            scale_pts.append((float(base['alpha']), int(base['radius']), int(base['laziness']), na, E.SCALE_DEFAULT_N_TASKS))
        for nt in E.N_TASKS_SWEEP_VALUES:
            scale_pts.append((float(base['alpha']), int(base['radius']), int(base['laziness']), default_agents, nt))
    out = []
    for s in seeds:
        if 'params' in modes:
            out += [('params', map_name, s, p) for p in dict.fromkeys(param_pts)]
        if 'scale' in modes:
            out += [('scale', map_name, s, p) for p in dict.fromkeys(scale_pts)]
    return out


def job_name(job):
    mode, mp, s, p = job
    tag = '_'.join(str(x) for x in p)
    return f"{mp[:-5]}__{mode}__s{s}__{tag}.log"


def run_job(job, logdir, timeout, stop):
    mode, mp, s, p = job
    log_path = os.path.join(logdir, job_name(job))
    if stop.is_set():
        return job, 'SKIPPED', log_path, None, 0.0
    cmd = [sys.executable, os.path.abspath(__file__), f'--single-{mode}', os.path.join(ROOT, 'Environments', mp),
           str(s)] + [str(x) for x in p]
    t0 = time.time()
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        timed_out = False
        while proc.poll() is None:
            if stop.is_set():
                proc.kill(); proc.wait()
                return job, 'KILLED', log_path, None, time.time() - t0
            if time.time() - t0 > timeout:
                proc.kill(); proc.wait(); timed_out = True
                break
            time.sleep(0.05)
    dt = time.time() - t0
    with open(log_path, errors='replace') as f:
        text = f.read()
    hit = next((l for l in text.splitlines() if any(k in l for k in bfc.MARKERS)), None)
    if hit:
        status = 'FAIL'
    elif timed_out:
        status = 'TIMEOUT'
    elif '[END] status=OK' in text:
        status = 'OK'
    elif '[END] status=TIMEOUT_SIM' in text:
        status = 'TIMEOUT_SIM'
    elif 'FATAL ERROR' in text:
        status = 'NOT_WELL_FORMED'
    else:
        status = f'CRASH(rc={proc.returncode})'
    return job, status, log_path, hit, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-params', nargs=5)
    ap.add_argument('--single-scale', nargs=7)
    ap.add_argument('--logdir')
    ap.add_argument('--maps', nargs='*', default=None)
    ap.add_argument('--seeds', nargs=2, type=int, default=[1, 25])
    ap.add_argument('--modes', nargs='*', default=['params', 'scale'])
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--stop-on-fail', action='store_true')
    a = ap.parse_args()
    if a.single_params:
        mp, s, al, r, l = a.single_params
        return single_params(mp, int(s), float(al), int(r), int(l))
    if a.single_scale:
        mp, s, al, r, l, na, nt = a.single_scale
        return single_scale(mp, int(s), float(al), int(r), int(l), int(na), int(nt))

    os.makedirs(a.logdir, exist_ok=True)
    maps = a.maps or THESIS_MAPS
    seeds = range(a.seeds[0], a.seeds[1] + 1)
    jobs = []
    for s in seeds:  # seed-major so that every map/config gets coverage early
        for mp in maps:
            jobs += jobs_for(mp, [s], a.modes)
    print(f"maps={maps} seeds={a.seeds} modes={a.modes} runs={len(jobs)} jobs={a.jobs} timeout={a.timeout}s", flush=True)
    stop = threading.Event()
    counts = {}
    done = 0
    t_start = time.time()
    with open(os.path.join(a.logdir, 'summary.jsonl'), 'a') as summary, ThreadPoolExecutor(a.jobs) as ex:
        it = iter(jobs)
        pending = {ex.submit(run_job, j, a.logdir, a.timeout, stop) for j, _ in zip(it, range(a.jobs * 3))}
        while pending:
            fut = next(as_completed(pending))
            pending.discard(fut)
            job, status, log_path, hit, dt = fut.result()
            done += 1
            counts[status] = counts.get(status, 0) + 1
            if status not in ('SKIPPED', 'KILLED'):
                mode, mp, s, p = job
                summary.write(json.dumps({'mode': mode, 'map': mp, 'label': MAP_LABEL.get(mp), 'seed': s, 'point': p,
                                          'status': status, 'marker': hit, 'secs': round(dt, 1),
                                          'log': log_path}) + '\n')
                summary.flush()
            if status == 'FAIL':
                print(f"*** FAIL {job}: {hit}", flush=True)
                if a.stop_on_fail:
                    stop.set()
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} in {time.time() - t_start:.0f}s: {counts}", flush=True)
            if not stop.is_set():
                nxt = next(it, None)
                if nxt is not None:
                    pending.add(ex.submit(run_job, nxt, a.logdir, a.timeout, stop))
    print(f"finished in {time.time() - t_start:.0f}s: {counts}", flush=True)


if __name__ == '__main__':
    main()
