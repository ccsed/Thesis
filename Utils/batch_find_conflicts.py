"""
Batch search for planning inconsistencies with zero delays.

Runs the TP + Terraforming A* simulation (same setup as experiments.worker_task) over
every map in Environments/, every seed in [seed_min, seed_max] and every
(alpha, radius, laziness) combination used in experiments.py, with n_delays forced to 0.
Each run is a separate subprocess with a timeout; its stdout is saved as a log.
The batch stops at the first run whose log contains "[BLOCKED]" or "[PLAN CONFLICT]".

Usage:
    python Utils/batch_find_conflicts.py --logdir <dir> [--jobs 10] [--timeout 120]
           [--seeds 0 200] [--maps a.yaml b.yaml] [--combos all|scenarios]

Single run (used by the batch, also handy for reproducing):
    python Utils/batch_find_conflicts.py --single <map.yaml> <seed> <alpha> <radius> <laziness> [--max-time T]
"""
import argparse
import glob
import io
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

MARKERS = ("[COLLISION]", "[BLOCKED]", "[PLAN CONFLICT]", "[PLAN DEVIATION]", "[OBS DEVIATION]", "[OBS PAST REWRITE]")
MAX_SIM_TIME = 3000
ASTAR_MAX_ITER = 100000


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def param_combos(which):
    import experiments  # SCENARIOS / SWEEP_VALUES, no side effects outside __main__
    combos = []
    for base in experiments.SCENARIOS.values():
        combos.append((base['alpha'], base['radius'], base['laziness']))
        if which == 'scenarios':
            continue
        for name, vals in experiments.SWEEP_VALUES.items():
            for v in vals:
                c = dict(base)
                c[name.lower()] = v
                combos.append((c['alpha'], c['radius'], c['laziness']))
    seen, out = set(), []
    for c in combos:
        key = (float(c[0]), int(c[1]), int(c[2]))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


class Tee(io.TextIOBase):
    """stdout replacement that forwards to the real stdout and remembers if a marker was printed."""

    def __init__(self, real):
        self.real = real
        self.hit = None

    def write(self, s):
        if self.hit is None:
            for m in MARKERS:
                if m in s:
                    self.hit = m
        return self.real.write(s)

    def flush(self):
        self.real.flush()


class PlanMonitor:
    """
    With zero delays, execution must follow the plan exactly. This wraps the planner's time_forward
    (called by the simulator at the start of each step) and checks, after the simulator has moved:
      [PLAN DEVIATION]    an agent's executed cell at time T differs from the plan it was committed to
      [OBS DEVIATION]     the simulator changed planned obstacle trajectories during the move phase
      [OBS PAST REWRITE]  the planner rewrote obstacle steps that are already in the past (t < T-1)
    A plan is "committed" when the token path of an agent stops being the tail of its previous plan.
    Token path index 0 is absolute time T-1 when planning at simulation time T.
    """

    def __init__(self, tp, simulation):
        self.tp, self.sim = tp, simulation
        self.plan = {}  # agent -> (abs_t0, [cells], planned_at)
        self.inner = tp.time_forward
        tp.time_forward = self.time_forward

    @staticmethod
    def obs_snapshot(token, upto=None):
        return {o: [(s['t'], s['x'], s['y']) for s in p if upto is None or s['t'] < upto]
                for o, p in token['obstacles_paths'].items()}

    def expected(self, name, t):
        t0, cells, _ = self.plan[name]
        return cells[min(max(t - t0, 0), len(cells) - 1)]

    def time_forward(self):
        T = self.sim.get_time()
        token = self.tp.get_token()
        past_before = self.obs_snapshot(token, upto=T - 1)
        self.inner()
        past_after = self.obs_snapshot(token, upto=T - 1)
        for o in past_before:
            if past_before[o] != past_after.get(o):
                print(f"[OBS PAST REWRITE] time {T}: {o} steps before t={T - 1} changed "
                      f"{past_before[o][-3:]} -> {past_after.get(o, [])[-3:]}")
        for name, path in token['agents'].items():
            cells = [tuple(c) for c in path]
            if name in self.plan:
                tail = [self.expected(name, T - 1 + i) for i in range(len(cells))]
                if tail == cells:
                    continue  # still following the committed plan
            self.plan[name] = (T - 1, cells, T)
        self.obs_after_plan = self.obs_snapshot(token)
        self.pending_check = T

    def check_after_move(self):
        T = self.pending_check
        token = self.tp.get_token()
        for name, ap in self.sim.actual_paths.items():
            act = (ap[-1]['x'], ap[-1]['y'])
            exp = self.expected(name, T)
            if act != exp:
                t0, cells, at = self.plan[name]
                print(f"[PLAN DEVIATION] time {T}: {name} executed {act}, planned {exp} "
                      f"(plan made at t={at}: {cells})")
        now = self.obs_snapshot(token)
        for o, steps in self.obs_after_plan.items():
            if now.get(o) != steps:
                print(f"[OBS DEVIATION] time {T}: simulator changed {o}: planned {[s for s in steps if s[0] >= T - 1]} "
                      f"-> {[s for s in now.get(o, []) if s[0] >= T - 1]}")


def executed_trajectories(simulation, tp):
    """Executed cell per time step for every agent and movable obstacle, t = 0..sim time."""
    T = simulation.get_time()
    traj = {}
    for name, path in simulation.actual_paths.items():
        by_t = {s['t']: (s['x'], s['y']) for s in path}
        last, cells = by_t[0], []
        for t in range(T + 1):
            last = by_t.get(t, last)
            cells.append(last)
        traj[name] = cells
    for obs_id, path in tp.get_token()['obstacles_paths'].items():
        # same semantics as the simulator: last step with step['t'] <= t wins
        last, cells = ((path[0]['x'], path[0]['y']) if path else None), []
        for t in range(T + 1):
            for s in path:
                if s['t'] <= t:
                    last = (s['x'], s['y'])
            cells.append(last)
        traj[obs_id] = cells
    return traj


def check_physical_collisions(simulation, tp):
    """Vertex and swap collisions between any two entities (agents and movable obstacles) on executed paths,
    plus the exact check used by experiments.py (verify_no_collisions, vertex only)."""
    traj = executed_trajectories(simulation, tp)
    names = list(traj)
    T = simulation.get_time()
    reported = 0
    for t in range(T + 1):
        seen = {}
        for n in names:
            c = traj[n][t]
            if c in seen:
                print(f"[COLLISION] vertex time {t}: {seen[c]} and {n} both at {c}")
                reported += 1
            seen[c] = n
        if t > 0:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    if traj[a][t - 1] != traj[a][t] and traj[a][t - 1] == traj[b][t] and traj[a][t] == traj[b][t - 1]:
                        print(f"[COLLISION] swap time {t - 1}->{t}: {a} {traj[a][t - 1]}->{traj[a][t]}, "
                              f"{b} {traj[b][t - 1]}->{traj[b][t]}")
                        reported += 1
        if reported > 20:
            print("[COLLISION] ... (truncated)")
            break
    try:
        import experiments
        if experiments.verify_no_collisions(simulation, tp):
            print("[COLLISION] experiments.verify_no_collisions -> True")
    except Exception as e:  # never let the check hide the run result
        print(f"[CHECK ERROR] verify_no_collisions raised {e!r}")


def single_run(map_path, seed, alpha, radius, laziness, max_time=MAX_SIM_TIME, stop_on_marker=True):
    """Same setup as experiments.worker_task, delays forced to zero. Prints the full log to stdout."""
    from Simulation.simulation_new_recovery import SimulationNewRecovery
    from Simulation.TP_with_recovery import TokenPassingRecovery
    from Simulation.tasks_and_delays_maker import gen_tasks_and_delays

    param = load_yaml(map_path)
    agents = param['agents']
    m = param['map']
    n_tasks = param.get('n_tasks') or 20
    task_freq = param.get('task_freq') or 1.0

    random.seed(seed)
    tasks, delays = gen_tasks_and_delays(agents, m['start_locations'], m['goal_locations'], n_tasks, task_freq, 0)
    assert all(len(d) == 0 for d in delays.values())

    tee = Tee(sys.stdout)
    sys.stdout = tee
    print(f"[RUN] map={os.path.basename(map_path)} seed={seed} alpha={alpha} radius={radius} laziness={laziness}")
    simulation = SimulationNewRecovery(tasks, agents, delays=delays)
    assert simulation.n_delays == len(delays)  # dict of empty lists -> no delay ever fires
    tp = TokenPassingRecovery(agents, m['dimensions'], m['obstacles'], m['non_task_endpoints'], simulation,
                              a_star_max_iter=ASTAR_MAX_ITER, k=0, pd=0.0, p_max=1, p_iter=1,
                              terraforming_radius=radius, new_recovery=True,
                              movable_obstacles=m.get('movable_obstacles', []),
                              alpha=alpha, laziness=laziness)
    monitor = PlanMonitor(tp, simulation)
    while tp.get_completed_tasks() != len(tasks) and simulation.get_time() < max_time:
        simulation.time_forward(tp)
        monitor.check_after_move()
        if stop_on_marker and tee.hit:
            print(f"[STOP] first marker {tee.hit} at time {simulation.get_time()}")
            break
    check_physical_collisions(simulation, tp)
    status = 'FAIL' if tee.hit else ('TIMEOUT_SIM' if simulation.get_time() >= max_time else 'OK')
    print(f"[END] status={status} time={simulation.get_time()} completed={tp.get_completed_tasks()}/{len(tasks)}")
    sys.stdout = tee.real
    return status


def map_is_runnable(path):
    try:
        p = load_yaml(path)
        m = p['map']
        return bool(p.get('agents')) and bool(m.get('start_locations')) and bool(m.get('goal_locations'))
    except Exception:
        return False


def map_size(path):
    p = load_yaml(path)
    d = p['map']['dimensions']
    return d[0] * d[1], len(p['agents'])


def run_job(job, logdir, timeout, stop_event, no_stop=False):
    map_path, seed, (alpha, radius, laziness) = job
    name = f"{os.path.basename(map_path)[:-5]}__s{seed}__a{alpha}_r{radius}_l{laziness}.log"
    log_path = os.path.join(logdir, name)
    if stop_event.is_set():
        return job, 'SKIPPED', log_path, None
    cmd = [sys.executable, os.path.abspath(__file__), '--single', map_path, str(seed), str(alpha), str(radius),
           str(laziness)] + (['--no-stop'] if no_stop else [])
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        t0 = time.time()
        while proc.poll() is None:  # poll so that a global stop kills running jobs immediately
            if stop_event.is_set():
                proc.kill()
                proc.wait()
                return job, 'KILLED', log_path, None
            if time.time() - t0 > timeout:
                proc.kill()
                proc.wait()
                break
            time.sleep(0.05)
    with open(log_path, errors='replace') as f:
        text = f.read()
    hit = next((l for l in text.splitlines() if any(m in l for m in MARKERS)), None)
    if hit:
        status = 'FAIL'
    elif proc.returncode is None or (time.time() - t0 > timeout and '[END]' not in text):
        status = 'TIMEOUT'
    elif '[END] status=OK' in text:
        status = 'OK'
    elif '[END] status=TIMEOUT_SIM' in text:
        status = 'TIMEOUT_SIM'
    elif 'FATAL ERROR' in text:  # initial Dynamic Well-Formedness check failed
        status = 'NOT_WELL_FORMED'
    else:
        status = f'CRASH(rc={proc.returncode})'
    return job, status, log_path, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', nargs=5, metavar=('MAP', 'SEED', 'ALPHA', 'RADIUS', 'LAZINESS'))
    ap.add_argument('--max-time', type=int, default=MAX_SIM_TIME)
    ap.add_argument('--no-stop', action='store_true', help='keep each run going after its first marker (full log, collisions checked on the whole run)')
    ap.add_argument('--logdir', default=os.path.join(ROOT, 'batch_logs'))
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--timeout', type=float, default=120.0, help='seconds per run')
    ap.add_argument('--seeds', nargs=2, type=int, default=[0, 200])
    ap.add_argument('--maps', nargs='*', default=None)
    ap.add_argument('--combos', choices=['all', 'scenarios'], default='all')
    ap.add_argument('--max-fails', type=int, default=1, help='stop after this many failing runs (default 1)')
    args = ap.parse_args()

    if args.single:
        mp, seed, a, r, l = args.single
        single_run(mp, int(seed), float(a), int(r), int(l), args.max_time, stop_on_marker=not args.no_stop)
        return

    os.makedirs(args.logdir, exist_ok=True)
    maps = args.maps or sorted(glob.glob(os.path.join(ROOT, 'Environments', '*.yaml')))
    maps = [os.path.abspath(m) for m in maps]
    skipped = [m for m in maps if not map_is_runnable(m)]
    maps = sorted([m for m in maps if map_is_runnable(m)], key=map_size)  # smallest first
    combos = param_combos(args.combos)
    seeds = range(args.seeds[0], args.seeds[1] + 1)
    # seed-major order: every map/combo gets coverage early, smallest maps first inside each seed
    jobs = [(m, s, c) for s in seeds for m in maps for c in combos]
    print(f"maps={len(maps)} (skipped, no start/goal locations: {[os.path.basename(m) for m in skipped]})")
    print(f"combos={len(combos)} seeds={len(seeds)} total runs={len(jobs)} jobs={args.jobs} timeout={args.timeout}s")
    print(f"logs -> {args.logdir}")

    stop = threading.Event()
    counts, first_fail = {}, None
    summary_path = os.path.join(args.logdir, 'summary.jsonl')
    done = 0
    with open(summary_path, 'w') as summary, ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = []
        it = iter(jobs)
        # bounded submission so that stopping is immediate
        for job in it:
            futures.append(ex.submit(run_job, job, args.logdir, args.timeout, stop, args.no_stop))
            if len(futures) >= args.jobs * 4:
                break
        pending = set(futures)
        while pending:
            fut = next(as_completed(pending))
            pending.discard(fut)
            job, status, log_path, hit = fut.result()
            done += 1
            counts[status] = counts.get(status, 0) + 1
            if status not in ('SKIPPED', 'KILLED'):
                summary.write(json.dumps({'map': os.path.basename(job[0]), 'seed': job[1], 'alpha': job[2][0],
                                          'radius': job[2][1], 'laziness': job[2][2], 'status': status,
                                          'log': log_path, 'marker': hit}) + '\n')
                summary.flush()
            if status == 'FAIL':
                if first_fail is None:
                    first_fail = (job, log_path, hit)
                if counts['FAIL'] >= args.max_fails:
                    stop.set()
                print(f"\n*** FAILURE #{counts['FAIL']}: {os.path.basename(job[0])} seed={job[1]} combo={job[2]}\n    {hit}\n    log: {log_path}")
            if done % 50 == 0:
                print(f"  {done} runs finished: {counts}", flush=True)
            if not stop.is_set():
                nxt = next(it, None)
                if nxt is not None:
                    pending.add(ex.submit(run_job, nxt, args.logdir, args.timeout, stop, args.no_stop))
    print(f"finished: {counts}")
    if first_fail is None:
        print("no [BLOCKED] / [PLAN CONFLICT] found")
    sys.exit(1 if first_fail else 0)


if __name__ == '__main__':
    main()
