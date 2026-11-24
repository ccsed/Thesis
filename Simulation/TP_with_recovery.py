"""
Python implementation of Token Passing algorithms to solve MAPD problems with delays
author: Anonymous
"""
from math import fabs
import random
from Simulation.CBS.cbs import CBS, Environment
from Simulation.markov_chains import MarkovChainsMaker
from collections import defaultdict


class TokenPassingRecovery(object):
    def __init__(self, agents, dimesions, obstacles, non_task_endpoints, simulation, a_star_max_iter=4000, k=0,
                 replan_every_k_delays=False, pd=None, p_max=1, p_iter=1, new_recovery=False, movable_obstacles=None, alpha=0.5, terraforming_radius=3, laziness=5, all_task_endpoints=None):
        self.agents = agents
        self.laziness = laziness
        self.dimensions = dimesions
        self.obstacles = set(tuple(o) for o in obstacles)
        self.movable_obstacles = set(tuple(o) for o in (movable_obstacles or []))
        self.non_task_endpoints = non_task_endpoints
        if len(agents) > len(non_task_endpoints):
            print('There are more agents than non task endpoints, instance is not well-formed.')
            exit(1)
        self.v_ep = set(tuple(ep) for ep in self.non_task_endpoints)
        if all_task_endpoints is not None:
            self.v_ep |= set(tuple(ep) for ep in all_task_endpoints)
        else:
            for t in simulation.tasks:
                self.v_ep.add(tuple(t['start']))
                self.v_ep.add(tuple(t['goal']))
        # TODO: Check all properties for well-formedness
        self.token = {}
        self.simulation = simulation
        self.a_star_max_iter = a_star_max_iter
        self.k = k
        self.p_max = p_max
        self.alpha = alpha
        self.movable_obstacles_list = movable_obstacles or []
        self.terraforming_radius = terraforming_radius
        if self.k != 0 and self.p_max != 1:
            print('Use of k and p robustness at same time not allowed.')
            exit(-1)
        if self.k < 0:
            print('k should be >= 0.')
            exit(1)
        if self.k == 0 and not new_recovery:
            print('k = 0 not supported for this recovery type.')
            exit(1)
        self.replan_every_k_delays = replan_every_k_delays
        if k == 0 and replan_every_k_delays:
            print('If k = 0 replan_every_k_delays must be False. Setting to False...')
            self.replan_every_k_delays = False
        if self.k == 0:
            if (self.p_max < 0 or self.p_max > 1):
                print('Max conflict probability must be between 0 and 1.')
                exit(1)
            self.pd = pd
            if self.p_max != 1 and not self.pd:
                print('To use p robustness need to set pd.')
                exit(1)
            if self.pd and (pd < 0 or pd > 1):
                print('Probability of delay must be between 0 and 1.')
                exit(1)
            self.p_iter = p_iter
            if self.p_iter <= 0:
                print('p_iter should be > 0.')
                exit(1)
        else:
            self.p_max = 1
            self.pd = None
            self.p_iter = 1
        self.new_recovery = new_recovery
        self.init_token()
        self.check_initial_well_formedness()

    def init_token(self):
        self.token['agents'] = {}
        self.token['tasks'] = {}
        self.token['obstacles_paths'] = {}
        self.token['original_obs_pos'] = {}
        for i, pos in enumerate(self.movable_obstacles_list):
            obs_name = f"obs_{i}"
            self.token['obstacles_paths'][obs_name] = [{'t': 0, 'x': pos[0], 'y': pos[1], 'agent': None}]
            self.token['original_obs_pos'][obs_name] = tuple(pos)
        self.token['start_tasks_times'] = {}
        self.token['completed_tasks_times'] = {}
        self.token['assigned_tasks_times'] = {}
        self.token['pickup_tasks_times'] = {}
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        self.token['agents_to_tasks'] = {}
        self.token['completed_tasks'] = 0
        self.token['n_replans'] = 0
        self.token['path_ends'] = set()
        self.token['occupied_non_task_endpoints'] = set()
        self.token['delayed_agents'] = []
        self.token['delayed_agents_to_reach_task_start'] = []
        self.token['agent_at_end_path'] = []
        self.token['agent_at_end_path_pos'] = []
        self.token['agents_in_recovery_trial'] = []
        for a in self.agents:
            self.token['agents'][a['name']] = [a['start']]
            self.token['path_ends'].add(tuple(a['start']))
        self.token['prob_exceeded'] = False
        self.token['deadlock_count_per_agent'] = defaultdict(lambda: 0)

    def check_initial_well_formedness(self):
        import sys
        import collections
        import heapq
        from itertools import combinations

        ep = set(tuple(e) for e in self.v_ep)
        movable = [tuple(o) for o in self.movable_obstacles_list]
        R = self.terraforming_radius
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def fail(msg):
            print("\nFATAL ERROR: instance is not Dynamically Well-Formed. " + msg + "\n")
            sys.exit(1)

        def connected(a, b, blocked):
            # path from a to b that does not traverse other endpoints
            visited = {a}
            queue = collections.deque([a])
            while queue:
                curr = queue.popleft()
                for dx, dy in moves:
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if not (0 <= nxt[0] < self.dimensions[0] and 0 <= nxt[1] < self.dimensions[1]):
                        continue
                    if nxt in blocked or nxt in visited:
                        continue
                    if nxt == b:
                        return True
                    visited.add(nxt)
                    if nxt not in ep:
                        queue.append(nxt)
            return False

        # Condition 1: static well-formedness (movable obstacles removed)
        if len(self.non_task_endpoints) < len(self.agents):
            fail("Fewer non-task endpoints than agents.")
        for o in movable:
            if o in ep:
                fail(f"Movable obstacle {o} lies on an endpoint.")
        pairs = list(combinations(sorted(ep), 2))
        for a, b in pairs:
            if not connected(a, b, self.obstacles):
                fail(f"Endpoints {a} and {b} are not connected in the static grid.")

        # Condition 2: topological reachability
        # admissible cells of each obstacle: within radius R of its initial cell, not static, not an endpoint
        candidates = []
        for (ox, oy) in movable:
            cells = []
            for dx in range(-R, R + 1):
                for dy in range(-(R - abs(dx)), R - abs(dx) + 1):
                    c = (ox + dx, oy + dy)
                    if 0 <= c[0] < self.dimensions[0] and 0 <= c[1] < self.dimensions[1] \
                            and c not in self.obstacles and c not in ep:
                        cells.append((abs(dx) + abs(dy), c))
            cells.sort()
            candidates.append(cells)

        n = len(movable)
        cand_cells = [[c for _, c in cells] for cells in candidates]

        def constructive_config(a, b, tries=30):
            # Pre-check that builds a witness configuration directly: a path from a to b that avoids the cells the
            # obstacles depend on, then each obstacle matched to a distinct admissible cell off that path.
            # It returns a configuration only after connected() has verified it, so a success is a valid witness
            # for Definition 4.1; on failure the exhaustive search below decides, so the verdict is unchanged.
            penalty = collections.Counter()
            for cells in cand_cells:
                for c in cells:
                    penalty[c] += 1.0 / len(cells)  # cells an obstacle has few alternatives to are expensive
            for _ in range(tries):
                dist = {a: 0.0}
                prev = {}
                pq = [(0.0, a)]
                reached = False
                while pq:
                    d, c = heapq.heappop(pq)
                    if d > dist[c]:
                        continue
                    if c == b:
                        reached = True
                        break
                    if c != a and c in ep:
                        continue  # the path may not traverse other endpoints
                    for dx, dy in moves:
                        nb = (c[0] + dx, c[1] + dy)
                        if not (0 <= nb[0] < self.dimensions[0] and 0 <= nb[1] < self.dimensions[1]) \
                                or nb in self.obstacles:
                            continue
                        nd = d + 1.0 + 10.0 * penalty[nb]
                        if nd < dist.get(nb, float('inf')):
                            dist[nb] = nd
                            prev[nb] = c
                            heapq.heappush(pq, (nd, nb))
                if not reached:
                    return None
                path = {b}
                c = b
                while c != a:
                    c = prev[c]
                    path.add(c)
                allowed = [[c for c in cells if c not in path] for cells in cand_cells]
                owner = {}

                def augment(j, visited):
                    for c in allowed[j]:
                        if c in visited:
                            continue
                        visited.add(c)
                        if c not in owner or augment(owner[c], visited):
                            owner[c] = j
                            return True
                    return False

                unmatched = [j for j in range(n) if not augment(j, set())]
                if not unmatched:
                    config = [None] * n
                    for c, j in owner.items():
                        config[j] = c
                    return config if connected(a, b, self.obstacles | set(config)) else None
                for j in unmatched:  # steer the next path away from the cells these obstacles need
                    for c in cand_cells[j]:
                        if c in path:
                            penalty[c] += 1.0
            return None

        for a, b in pairs:
            # first configuration of the search below (each obstacle on its nearest admissible cell), then the pre-check
            config0 = [cells[0] for cells in cand_cells]
            if (len(set(config0)) == n and connected(a, b, self.obstacles | set(config0))) \
                    or constructive_config(a, b) is not None:
                continue
            # joint configurations in increasing order of total displacement, stop at the first that connects a and b
            start = tuple([0] * n)
            heap = [(0, start)]
            seen = {start}
            found = False
            while heap:
                cost, idx = heapq.heappop(heap)
                config = [candidates[j][idx[j]][1] for j in range(n)]
                if len(set(config)) == n and connected(a, b, self.obstacles | set(config)):
                    found = True
                    break
                for j in range(n):
                    if idx[j] + 1 < len(candidates[j]):
                        nxt = idx[:j] + (idx[j] + 1,) + idx[j + 1:]
                        if nxt not in seen:
                            seen.add(nxt)
                            nxt_cost = cost - candidates[j][idx[j]][0] + candidates[j][idx[j] + 1][0]
                            heapq.heappush(heap, (nxt_cost, nxt))
            if not found:
                fail(f"No admissible configuration of the movable obstacles connects endpoints {a} and {b} (radius {R}).")

    def get_idle_agents(self):
        agents = {}
        for name, path in self.token['agents'].items():
            if len(path) == 1:
                agents[name] = path
        return agents
    
    def get_current_movable_obstacles(self):
        timelines = {}
        abs_time_start = self.simulation.get_time() - 1
        for i in range(len(self.movable_obstacles_list)):
            obs_id = f"obs_{i}"
            path = self.token['obstacles_paths'][obs_id]
            
            valid_steps = [step for step in path if step['t'] <= abs_time_start]
            base_pos = max(valid_steps, key=lambda s: s['t']) if valid_steps else ({'x': 0, 'y': 0} if not path else path[0])
                    
            rel_path = [{'t': 0, 'x': base_pos['x'], 'y': base_pos['y']}]
            
            for step in path:
                if step['t'] > abs_time_start:
                    rel_path.append({'t': step['t'] - abs_time_start, 'x': step['x'], 'y': step['y']})
                    
            timelines[obs_id] = rel_path
        return timelines

    def admissible_heuristic(self, task_pos, agent_pos):
        return fabs(task_pos[0] - agent_pos[0]) + fabs(task_pos[1] - agent_pos[1])

    def get_closest_task_name(self, available_tasks, agent_pos):
        closest = random.choice(list(available_tasks.keys()))
        dist = self.admissible_heuristic(available_tasks[closest][0], agent_pos)
        for task_name, task in available_tasks.items():
            if self.admissible_heuristic(task[0], agent_pos) < dist:
                closest = task_name
        return closest

    def get_moving_obstacles_agents(self, agents, time_start):
        obstacles = {}
        current_sim_time = self.simulation.get_time()
        abs_time_start = (current_sim_time - 1) + time_start
        base_time = current_sim_time - 1
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    obstacles[(path[i][0], path[i][1], k)] = name
                    for j in range(1, self.k + 1):
                        if i - j >= time_start:
                            obstacles[(path[i][0], path[i][1], k - j)] = name
                        obstacles[(path[i][0], path[i][1], k + j)] = name
                    # Mark last element with negative time to later turn it into idle obstacle
                    if i == len(path) - 1:
                        obstacles[(path[i][0], path[i][1], -k)] = name
        
        def get_obs_pos_at(path, target_t):
            valid_steps = [s for s in path if s['t'] <= target_t]
            if not valid_steps:
                return (path[0]['x'], path[0]['y']) if path else (0,0)
            best = max(valid_steps, key=lambda s: s['t'])
            return (best['x'], best['y'])

        for obs_id, path in self.token['obstacles_paths'].items():
            if not path:
                continue
            max_t = max(step['t'] for step in path)
            for t in range(abs_time_start, max_t + 2):
                curr_pos = get_obs_pos_at(path, t)
                prev_pos = get_obs_pos_at(path, t - 1)
                next_pos = get_obs_pos_at(path, t + 1)
                
                if curr_pos != prev_pos or curr_pos != next_pos:
                    relative_t = t - abs_time_start
                    if relative_t >= 0:
                        obstacles[(curr_pos[0], curr_pos[1], relative_t)] = obs_id
                        for j in range(1, self.k + 1):
                            if relative_t - j >= 0:
                                obstacles[(curr_pos[0], curr_pos[1], relative_t - j)] = obs_id
                            obstacles[(curr_pos[0], curr_pos[1], relative_t + j)] = obs_id
        return obstacles

    def get_idle_obstacles_agents(self, agents_paths, delayed_agents, time_start):
        obstacles = set()
        for path in agents_paths:
            if len(path) == 1:
                obstacles.add((path[0][0], path[0][1]))
            if 1 < len(path) <= time_start + 1:
                obstacles.add((path[-1][0], path[-1][1]))
        for agent_name in delayed_agents:
            obstacles.add(tuple(self.token['agents'][agent_name][0]))
        return obstacles

    def check_safe_idle(self, agent_pos):
        # Free agents may only wait on non-task endpoints: task endpoints are not safe resting cells
        if tuple(agent_pos) not in {tuple(ep) for ep in self.non_task_endpoints}:
            return False
        for task_name, task in self.token['tasks'].items():
            if tuple(task[0]) == tuple(agent_pos) or tuple(task[1]) == tuple(agent_pos):
                return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
                
        # --- NEW: Controllo traiettorie future degli ostacoli ---
        current_time = self.simulation.get_time() - 1
        cell = tuple(agent_pos)
        for obs_path in self.token['obstacles_paths'].values():
            for step in obs_path:
                if step['t'] >= current_time and (step['x'], step['y']) == cell:
                    return False
                    
        return True


    def get_closest_non_task_endpoint(self, agent_pos):
        dist = -1
        res = -1
        
        active_task_nodes = set()
        for task in self.token['tasks'].values():
            active_task_nodes.add(tuple(task[0]))
            active_task_nodes.add(tuple(task[1]))
        for task in self.token['agents_to_tasks'].values():
            active_task_nodes.add(tuple(task['start']))
            active_task_nodes.add(tuple(task['goal']))

        # A non-task endpoint is free only if no other agent is resting there or ends its path there
        taken = self.token['occupied_non_task_endpoints'] | (self.token['path_ends'] - {tuple(agent_pos)})

        for endpoint in self.non_task_endpoints:
            ep_tuple = tuple(endpoint)
            if ep_tuple not in taken and ep_tuple not in active_task_nodes:
                if dist == -1:
                    dist = self.admissible_heuristic(endpoint, agent_pos)
                    res = endpoint
                else:
                    tmp = self.admissible_heuristic(endpoint, agent_pos)
                    if tmp < dist:
                        dist = tmp
                        res = endpoint
                        
        
        if res == -1:
            for endpoint in self.non_task_endpoints:
                if tuple(endpoint) not in taken:
                    return endpoint
            return agent_pos # Estremo fallback
            
        return res

    def update_ends(self, agent_pos):
        if tuple(agent_pos) in self.token['path_ends']:
            self.token['path_ends'].remove(tuple(agent_pos))
        elif tuple(agent_pos) in self.token['occupied_non_task_endpoints']:
            self.token['occupied_non_task_endpoints'].remove(tuple(agent_pos))

    def get_agents_to_tasks_goals(self):
        goals = set()
        for el in self.token['agents_to_tasks'].values():
            goals.add(tuple(el['goal']))
        return goals

    def get_agents_to_tasks_starts_goals(self):
        starts_goals = set()
        for el in self.token['agents_to_tasks'].values():
            starts_goals.add(tuple(el['goal']))
            starts_goals.add(tuple(el['start']))
        return starts_goals

    def get_completed_tasks(self):
        return self.token['completed_tasks']

    def get_completed_tasks_times(self):
        return self.token['completed_tasks_times']

    def get_n_replans(self):
        return self.token['n_replans']

    def get_assigned_tasks_times(self):
        return self.token['assigned_tasks_times']
    
    def get_pickup_tasks_times(self):
        return self.token['pickup_tasks_times']

    def get_token(self):
        return self.token

    def get_k(self):
        return self.k

    def get_replan_every_k_delays(self):
        return self.replan_every_k_delays

    def search(self, cbs, agent_name, moving_obstacles_agents):
        path = None
        if self.p_max == 1:
            path = cbs.search()
        else:
            self.token['prob_exceeded'] = False
            mk = MarkovChainsMaker(self.token['agents'], self.pd)
            for iter in range(self.p_iter):
                path = cbs.search()
                tmp = []
                if path and len(path[agent_name]) > 1:
                    for el in path[agent_name]:
                        tmp.append([el['x'], el['y']])
                    dic = mk.get_conflict_prob_given_path(tmp)
                    if dic['prob'] > self.p_max:
                        self.token['prob_exceeded'] = True
                        print('Conflict probablility to high (', dic['prob'], ') replanning...')
                        path = None
                        moving_obstacles_agents[(dic['pos_max_conf'][0], dic['pos_max_conf'][1], -1)] = agent_name
                    else:
                        break
        return path

    def go_to_closest_non_task_endpoint(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents):
        closest_non_task_endpoint = self.get_closest_non_task_endpoint(agent_pos)
        moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
        idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, 0)
        agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_non_task_endpoint}
        env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents, moving_obstacles_agents, movable_obstacles=self.get_current_movable_obstacles(), original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep,
                  a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, 
                  terraforming_radius=0,
                  static_obstacles=self.obstacles, laziness=self.laziness)
        cbs = CBS(env)
        result = self.search(cbs, agent_name, moving_obstacles_agents)
        if not result:
            print("Solution to non-task endpoint not found for agent", agent_name, " instance is not well-formed.")
            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, 4)
        else:
            path_to_non_task_endpoint = result['agents']
            obs_trajectories = result['obstacles']
            base_time = self.simulation.get_time() - 1
            for obs_id, traj in obs_trajectories.items():
                if len(traj) > 0:
                    first_abs_t = base_time + traj[0]['t']
                    self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                for step in traj:
                    abs_t = base_time + step['t']
                    self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
            print('No available task for agent', agent_name, ' moving to safe idling position...')
            self.update_ends(agent_pos)
            self.token['occupied_non_task_endpoints'].add(tuple(closest_non_task_endpoint))
            self.token['agents_to_tasks'][agent_name] = {'task_name': 'safe_idle', 'start': agent_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['agents'][agent_name] = []
            for el in path_to_non_task_endpoint[agent_name]:
                self.token['agents'][agent_name].append([el['x'], el['y']])

    def get_random_close_cell(self, agent_pos, r):
        current_time = self.simulation.get_time() - 1
        while True:
            cell = (agent_pos[0] + random.choice(range(-r - 1, r + 1)), agent_pos[1] + random.choice(range(-r - 1, r + 1)))
            if cell not in self.obstacles and cell not in self.token['path_ends'] and \
                    cell not in self.token['occupied_non_task_endpoints'] \
                    and cell not in self.get_agents_to_tasks_goals() \
                    and 0 <= cell[0] < self.dimensions[0] and 0 <= cell[1] < self.dimensions[1]:
                
                # --- NEW: Controllo traiettorie future degli ostacoli ---
                safe_from_obstacles = True
                for obs_path in self.token['obstacles_paths'].values():
                    for step in obs_path:
                        # Discard the cell if the obstacle occupies it from now on
                        if step['t'] >= current_time and (step['x'], step['y']) == cell:
                            safe_from_obstacles = False
                            break
                    if not safe_from_obstacles:
                        break
                
                if safe_from_obstacles:
                    return cell

    def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, all_delayed_agents, r):
        self.token['deadlock_count_per_agent'][agent_name] += 1
        if self.token['deadlock_count_per_agent'][agent_name] >= 5:
            self.token['deadlock_count_per_agent'][agent_name] = 0
            random_close_cell = self.get_random_close_cell(agent_pos, r)
            moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
            idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, 0)
            agent = {'name': agent_name, 'start': agent_pos, 'goal': random_close_cell}
            env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents, moving_obstacles_agents, movable_obstacles=self.get_current_movable_obstacles(), original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep,
                  a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, 
                  terraforming_radius=0, 
                  static_obstacles=self.obstacles, laziness=self.laziness)
            cbs = CBS(env)
            result = self.search(cbs, agent_name, moving_obstacles_agents)
            if not result:
                print("No solution to deadlock recovery for agent", agent_name, " retrying later.")
            else:
                path_to_non_task_endpoint = result['agents']
                obs_trajectories = result['obstacles']
                base_time = self.simulation.get_time() - 1
                for obs_id, traj in obs_trajectories.items():
                    if len(traj) > 0:
                        first_abs_t = base_time + traj[0]['t']
                        self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                    for step in traj:
                        abs_t = base_time + step['t']
                        self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
                # Don't consider this a task, so don't add to agents_to_tasks
                print('Agent', agent_name, 'causing deadlock, moving to safer position...')
                self.update_ends(agent_pos)
                self.token['agents'][agent_name] = []
                for el in path_to_non_task_endpoint[agent_name]:
                    self.token['agents'][agent_name].append([el['x'], el['y']])

    def time_forward(self):
        # Update completed tasks
        for agent_name in self.token['agents']:
            pos = self.simulation.actual_paths[agent_name][-1]
            if agent_name in self.token['agents_to_tasks'] and (pos['x'], pos['y']) == tuple(
                    self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 and self.token['agents_to_tasks'][agent_name][
                'task_name'] != 'safe_idle':
                self.token['completed_tasks'] = self.token['completed_tasks'] + 1
                self.token['completed_tasks_times'][
                    self.token['agents_to_tasks'][agent_name]['task_name']] = self.simulation.get_time()
                self.token['agents_to_tasks'].pop(agent_name)
            if agent_name in self.token['agents_to_tasks'] and (pos['x'], pos['y']) == tuple(
                    self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 and self.token['agents_to_tasks'][agent_name][
                'task_name'] == 'safe_idle':
                self.token['agents_to_tasks'].pop(agent_name)

        # Check delayed agents and agents affected by delays
        if self.new_recovery:
            self.token['delayed_agents'] = self.simulation.get_delayed_agents()
            for name in self.token['delayed_agents']:
                print('Agent', name, 'delayed or affected by delay!')
                path = self.token['agents'][name]
                self.token['n_replans'] = self.token['n_replans'] + 1
                self.update_ends(path[-1])
                if path[0] in self.non_task_endpoints:
                    self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                else:
                    self.token['path_ends'].add(tuple(path[0]))
                if name in self.token['agents_to_tasks']:
                    if self.token['agents_to_tasks'][name]['start'] not in path:
                        self.token['delayed_agents_to_reach_task_start'].append(name)
                self.token['agents'][name] = [path[0]]
                self.scrub_future_obstacle_trajectories(name, self.simulation.get_time() - 1)
        else:
            self.token['delayed_agents'] = []
            delayed_agents_pos = []
            for name, path in self.token['agents'].items():
                actual_state = self.simulation.get_actual_paths()[name][-1]
                if path[0] != [actual_state['x'], actual_state['y']]:
                    print('Agent', name, 'delayed!')
                    self.token['delayed_agents'].append(name)
                    delayed_agents_pos.append([actual_state['x'], actual_state['y']])
                    self.update_ends(path[-1])
                    pos = tuple([actual_state['x'], actual_state['y']])
                    if pos in self.non_task_endpoints:
                        self.token['occupied_non_task_endpoints'].add(pos)
                    else:
                        self.token['path_ends'].add(pos)
                    if self.token['agents_to_tasks'][name]['start'] not in path:
                        self.token['delayed_agents_to_reach_task_start'].append(name)
                    self.token['agents'][name] = [[actual_state['x'], actual_state['y']]]
            for name, path in self.token['agents'].items():
                if name not in self.token['delayed_agents']:
                    for i in range(len(path)):
                        if path[i] in delayed_agents_pos:
                            print('Agent', name, 'affected by delay!')
                            self.token['n_replans'] = self.token['n_replans'] + 1
                            self.update_ends(path[-1])
                            if path[0] in self.non_task_endpoints:
                                self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                            else:
                                self.token['path_ends'].add(tuple(path[0]))
                            if self.token['agents_to_tasks'][name]['start'] not in path:
                                self.token['delayed_agents_to_reach_task_start'].append(name)
                            self.token['agents'][name] = [path[0]]
                            break

        # TODO do this maybe only for old recovery
        # Check if somehow an agent path collides with an idle agent (may happen because of delays)
        if not self.new_recovery:
            for name, path in self.get_idle_agents().items():
                self.token['agent_at_end_path'].append(name)
                self.token['agent_at_end_path_pos'].append(path[0])
            for name, path in self.token['agents'].items():
                if name not in self.token['agent_at_end_path']:
                    for i in range(len(path)):
                        if path[i] in self.token['agent_at_end_path_pos']:
                            print('Agent', name, 'will impact end task agent, replanning...')
                            if path[0] in self.non_task_endpoints:
                                self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                            else:
                                self.token['path_ends'].add(tuple(path[0]))
                            # TODO check this rare keyerror
                            if self.token['agents_to_tasks'][name]['start'] not in path:
                                self.token['delayed_agents_to_reach_task_start'].append(name)
                            self.token['agents'][name] = [path[0]]
                            break
            self.token['agent_at_end_path'] = []
            self.token['agent_at_end_path_pos'] = []

        # Collect new tasks and assign them, if possible
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        idle_agents = self.get_idle_agents()
        while len(idle_agents) > 0:
            agent_name = random.choice(list(idle_agents.keys()))
            all_idle_agents = self.token['agents'].copy()
            all_idle_agents.pop(agent_name)
            all_delayed_agents = self.token['delayed_agents'].copy()
            if agent_name in all_delayed_agents:
                all_delayed_agents.remove(agent_name)
            agent_pos = idle_agents.pop(agent_name)[0]
            available_tasks = {}
            for task_name, task in self.token['tasks'].items():
                if tuple(task[0]) not in self.token['path_ends'].difference({tuple(agent_pos)}) and tuple(
                        task[1]) not in self.token['path_ends'].difference({tuple(agent_pos)}) \
                        and tuple(task[0]) not in self.get_agents_to_tasks_goals() and tuple(
                    task[1]) not in self.get_agents_to_tasks_goals():
                    available_tasks[task_name] = task
            if len(available_tasks) > 0 or agent_name in self.token['agents_to_tasks']:
                if agent_name in self.token['agents_to_tasks']:
                    closest_task_name = self.token['agents_to_tasks'][agent_name]['task_name']
                    if agent_name in self.token['delayed_agents_to_reach_task_start']:
                        closest_task = [self.token['agents'][agent_name][0],
                                        self.token['agents_to_tasks'][agent_name]['goal']]
                    else:
                        closest_task = [self.token['agents_to_tasks'][agent_name]['start'],
                                        self.token['agents_to_tasks'][agent_name]['goal']]
                else:
                    closest_task_name = self.get_closest_task_name(available_tasks, agent_pos)
                    closest_task = available_tasks[closest_task_name]
                moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
                idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents, 0)
                agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_task[0]}
                env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents,
                                  moving_obstacles_agents, movable_obstacles=self.get_current_movable_obstacles(), original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep, a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, terraforming_radius=self.terraforming_radius, static_obstacles=self.obstacles, laziness=self.laziness)
                cbs = CBS(env)
                result_start = self.search(cbs, agent_name, moving_obstacles_agents)
                if not result_start:
                    print("Solution not found to task start for agent", agent_name, " idling at current position...")
                    if len(self.token['delayed_agents']) == 0 and not self.token['prob_exceeded']:
                        print('Instance is not well-formed or a_star_max_iter is too low for this environment.')
                        self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, 4)
                else:
                    path_to_task_start = result_start['agents']
                    start_obs_trajectories = result_start['obstacles']
                    print("Solution found to task start for agent", agent_name, " searching solution to task goal...")
                    
                    cost1 = len(path_to_task_start[agent_name])
                    
                    # Use cost - 1 because idle cost is 1
                    moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], cost1 - 1)
                    idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), all_delayed_agents,
                                                                           cost1 - 1)
                    updated_obs_pos = {}
                    abs_start_time = self.simulation.get_time() - 1 + (cost1 - 1)
                    for i in range(len(self.movable_obstacles_list)):
                        obs_id = f"obs_{i}"
                        if obs_id in start_obs_trajectories and len(start_obs_trajectories[obs_id]) > 0:
                            last_pos = start_obs_trajectories[obs_id][-1]
                            updated_obs_pos[obs_id] = [{'t': 0, 'x': last_pos['x'], 'y': last_pos['y']}]
                        else:
                            path = self.token['obstacles_paths'][obs_id]
                            valid_steps = [step for step in path if step['t'] <= abs_start_time]
                            base_pos = max(valid_steps, key=lambda s: s['t']) if valid_steps else ({'x': 0, 'y': 0} if not path else path[0])
                            rel_path = [{'t': 0, 'x': base_pos['x'], 'y': base_pos['y']}]
                            for step in path:
                                if step['t'] >= abs_start_time:
                                    rel_path.append({'t': step['t'] - abs_start_time, 'x': step['x'], 'y': step['y']})
                            updated_obs_pos[obs_id] = rel_path
                    agent = {'name': agent_name, 'start': closest_task[0], 'goal': closest_task[1]}
                    env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents,
                                      moving_obstacles_agents, movable_obstacles=updated_obs_pos, original_obs_pos=self.token['original_obs_pos'], v_ep=self.v_ep, a_star_max_iter=self.a_star_max_iter, alpha=self.alpha, terraforming_radius=self.terraforming_radius, static_obstacles=self.obstacles, laziness=self.laziness)
                    cbs = CBS(env)
                    result_goal = self.search(cbs, agent_name, moving_obstacles_agents)
                    if not result_goal:
                        print("Solution not found to task goal for agent", agent_name, " idling at current position...")
                        if len(self.token['delayed_agents']) == 0 and not self.token['prob_exceeded']:
                            print('Instance is not well-formed  or a_star_max_iter is too low for this environment.')
                            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, all_delayed_agents, 4)
                    else:
                        print("Solution found to task goal for agent", agent_name, " doing task...")
                        path_to_task_goal = result_goal['agents']
                        goal_obs_trajectories = result_goal['obstacles']
                        
                        cost2 = len(path_to_task_goal[agent_name])
                        
                        current_sim_time = self.simulation.get_time()
                        base_time = current_sim_time - 1
                        for obs_id, traj in start_obs_trajectories.items():
                            if len(traj) > 0:
                                first_abs_t = base_time + traj[0]['t']
                                self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                            for step in traj:
                                abs_t = base_time + step['t']
                                self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
                        for obs_id, traj in goal_obs_trajectories.items():
                            if len(traj) > 0:
                                first_abs_t = base_time + (cost1 - 1) + traj[0]['t']
                                self.token['obstacles_paths'][obs_id] = [step for step in self.token['obstacles_paths'][obs_id] if step['t'] < first_abs_t]
                            for step in traj:
                                abs_t = base_time + (cost1 - 1) + step['t']
                                self.token['obstacles_paths'][obs_id].append({'t': abs_t, 'x': step['x'], 'y': step['y'], 'agent': agent_name})
                        if agent_name not in self.token['agents_to_tasks']:
                            self.token['tasks'].pop(closest_task_name)
                            task = available_tasks.pop(closest_task_name)
                        else:
                            task = closest_task
                        if agent_name in self.token['delayed_agents_to_reach_task_start']:
                            self.token['delayed_agents_to_reach_task_start'].remove(agent_name)
                        last_step = path_to_task_goal[agent_name][-1]
                        self.update_ends(agent_pos)
                        self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                        self.token['agents_to_tasks'][agent_name] = {'task_name': closest_task_name, 'start': task[0],
                                                                     'goal': task[1], 'predicted_cost': cost1 + cost2}
                        self.token['assigned_tasks_times'][closest_task_name] = self.simulation.get_time()
                        self.token['pickup_tasks_times'][closest_task_name] = self.simulation.get_time() + (cost1 - 1)
                        self.token['agents'][agent_name] = []
                        for el in path_to_task_start[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
                        # Don't repeat twice same step
                        self.token['agents'][agent_name] = self.token['agents'][agent_name][:-1]
                        for el in path_to_task_goal[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
            elif self.check_safe_idle(agent_pos):
                print('No available tasks for agent', agent_name, ' idling at current position...')
            else:
                self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents, all_delayed_agents)


        self.check_plan_conflicts()

        # Advance along paths in the token
        if not self.new_recovery:
            for name, path in self.token['agents'].items():
                if len(path) > 1:
                    self.token['agents'][name] = path[1:]
    
    def check_plan_conflicts(self):
        now = self.simulation.get_time()
        paths = {n: [tuple(c) for c in p] for n, p in self.token['agents'].items()}
        names = list(paths)
        at = lambda p, t: p[min(t, len(p) - 1)]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = paths[a], paths[b]
                for t in range(max(len(pa), len(pb))):
                    if at(pa, t) == at(pb, t):
                        print(f"[PLAN CONFLICT] time {now}: vertex {a}-{b} at +{t} in {at(pa, t)}")
                        break
                    if at(pa, t) == at(pb, t + 1) and at(pa, t + 1) == at(pb, t):
                        print(f"[PLAN CONFLICT] time {now}: swap {a}-{b} at +{t} between {at(pa, t)} and {at(pa, t + 1)}")
                        break

    def shift_obstacle_trajectories(self, agent_name, current_time):
        for obs_id, path in self.token['obstacles_paths'].items():
            shifted = False
            for step in path:
                # Target future steps belonging to this specific delayed agent
                if step.get('agent') == agent_name and step['t'] >= current_time:
                    step['t'] += 1
                    shifted = True
            
            # Re-sort to maintain strict chronological order if multiple agents interact
            if shifted:
                self.token['obstacles_paths'][obs_id] = sorted(path, key=lambda s: s['t'])
    
    def scrub_future_obstacle_trajectories(self, agent_name, current_time):
       
        for obs_id, path in self.token['obstacles_paths'].items():
            valid_path = []
            for step in path:
                if step.get('agent') == agent_name and step['t'] >= current_time:
                    continue
                valid_path.append(step)
            self.token['obstacles_paths'][obs_id] = valid_path

    
