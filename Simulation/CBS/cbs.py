"""
Python implementation of Conflict-based search
author: Ashwin Bose (@atb033)
author: Giacomo Lodigiani (@Lodz97)
"""
import sys
sys.path.insert(0, '../')
import argparse
import yaml
from math import fabs
from itertools import combinations
from copy import deepcopy

from Simulation.CBS.a_star import AStar

class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))
    def __str__(self):
        return str((self.x, self.y))
    def to_tuple(self):
        return (self.x, self.y)

class State(object):
    def __init__(self, time, location, delta_o=None, p=0.0, to_move=None):
        self.time = time
        self.location = location
        self.delta_o = delta_o if delta_o is not None else frozenset()
        self.p = p
        self.to_move = to_move if to_move is not None else ()
    def __eq__(self, other):
        return (self.time == other.time and 
                self.location == other.location and 
                self.delta_o == other.delta_o and self.to_move == other.to_move)
    def __hash__(self):
        return hash((self.time, self.location.x, self.location.y, self.delta_o, self.to_move))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def is_equal_except_time_and_p(self, state):
        return self.location == state.location and len(self.to_move) == 0
    def is_intermidiate(self):
        return len(self.to_move) > 0
    def __str__(self):
        status = f"Next:{self.to_move}" if self.is_intermidiate() else "Next:Agent"
        return f"T:{self.time} L:{self.location} {status} P:{self.p}"

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class Environment(object):
    def __init__(self, dimension, agents, obstacles, moving_obstacles=None, movable_obstacles=None, v_ep=None, a_star_max_iter=-1, alpha=0.5, terraforming_radius=3):
        if moving_obstacles is None:
            moving_obstacles = []
        self.dimension = dimension
        self.obstacles = obstacles
        self.moving_obstacles = moving_obstacles
        self.movable_obstacles_map = {}
        if movable_obstacles:
            for i, pos in enumerate(movable_obstacles):
                self.movable_obstacles_map[tuple(pos)] = f"obs_{i}"
        self.v_ep = set(tuple(e) for e in v_ep) if v_ep else set()
        self.a_star_max_iter = a_star_max_iter
        self.alpha = alpha
        self.k = terraforming_radius
        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self)
        self.obs_id_to_init_loc = {v: k for k, v in self.movable_obstacles_map.items()}

    def get_obstacle_at(self, location, delta_o):
        loc_tuple = location.to_tuple()
        curr_delta_dict = dict(delta_o)
        for obs_id, obs_pos in curr_delta_dict.items():
            if obs_pos == loc_tuple:
                return obs_id
        if loc_tuple in self.movable_obstacles_map:
            obs_id = self.movable_obstacles_map[loc_tuple]
            if obs_id not in curr_delta_dict:
                return obs_id
        return None

    def get_neighbors(self, state):
        neighbors = []
        if not state.is_intermidiate():
            candidates = [Location(state.location.x, state.location.y), 
                        Location(state.location.x+1, state.location.y), 
                        Location(state.location.x-1, state.location.y), 
                        Location(state.location.x, state.location.y+1), 
                        Location(state.location.x, state.location.y-1)]
            for l_prime in candidates:
                if not ( 0 <= l_prime.x < self.dimension[0] and 0 <= l_prime.y < self.dimension[1]):
                    continue
                if l_prime.to_tuple() in self.obstacles:
                    continue
                if (l_prime.x, l_prime.y, state.time + 1) in self.moving_obstacles:
                    continue
                collision_idle = False
                for (ox, oy, ot), name in self.moving_obstacles.items():
                    if ot < 0: 
                        if (l_prime.x, l_prime.y) == (ox, oy) and (state.time + 1) >= abs(ot):
                            collision_idle = True
                            break
                if collision_idle: 
                    continue
                obs_id = self.get_obstacle_at(l_prime, state.delta_o)
                if obs_id is None:
                    new_state = State(time=state.time+1, location=l_prime, delta_o=state.delta_o, p=state.p)
                    neighbors.append(new_state)
                else:
                    new_state = State(time=state.time, location=state.location, delta_o=state.delta_o, p=state.p, to_move=((obs_id, l_prime.to_tuple()),))
                    neighbors.append(new_state)
        else:
            (obs_id, target_agent), *rest_to_move = state.to_move
            curr_delta = dict(state.delta_o)
            l_init = self.obs_id_to_init_loc[obs_id]
            curr_obs_loc = curr_delta.get(obs_id, l_init)
            
            moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in moves:
                    l_obs_prime = (curr_obs_loc[0] + dx, curr_obs_loc[1] + dy)
                    if abs(l_obs_prime[0] -l_init[0]) + abs(l_obs_prime[1] - l_init[1]) > self.k:
                        continue
                    if not (0 <= l_obs_prime[0] < self.dimension[0] and 0 <= l_obs_prime[1] < self.dimension[1]):
                        continue
                    if l_obs_prime in self.obstacles or l_obs_prime in self.v_ep:
                        continue
                    if l_obs_prime == state.location.to_tuple():
                        continue
                    if (l_obs_prime[0], l_obs_prime[1], state.time) in self.moving_obstacles:
                        continue
                    collision_idle = False
                    for (ox, oy, ot), name in self.moving_obstacles.items():
                        if ot < 0: 
                            if l_obs_prime == (ox, oy) and state.time >= abs(ot):
                                collision_idle = True
                                break
                    if collision_idle: 
                        continue
                    blocking_obs_id = self.get_obstacle_at(Location(l_obs_prime[0], l_obs_prime[1]), state.delta_o)
                    if blocking_obs_id == obs_id:
                        blocking_obs_id = None
                    if blocking_obs_id is None:
                        new_delta = dict(state.delta_o)
                        new_delta[obs_id] = l_obs_prime
                        new_p = state.p + (1.0 if (dx, dy) != (0, 0) else 0.0)
                        if not rest_to_move:
                            if l_obs_prime != target_agent and curr_obs_loc == target_agent:
                                new_state = State(time=state.time + 1, location=Location(target_agent[0], target_agent[1]), delta_o=frozenset(new_delta.items()), p=new_p, to_move=())
                            else:
                                new_state = State(time=state.time + 1, location=state.location, delta_o=frozenset(new_delta.items()), p=new_p, to_move=())
                        else:
                            new_state = State(time=state.time, location=state.location, delta_o=frozenset(new_delta.items()), p=new_p, to_move=tuple(rest_to_move))
                        neighbors.append(new_state)
                    else:
                        if any(blocking_obs_id == item[0] for item in state.to_move) or blocking_obs_id == obs_id:
                            continue
                        new_to_move = ((blocking_obs_id, l_obs_prime), (obs_id, target_agent)) + tuple(rest_to_move)
                        new_state = State(time=state.time, location=state.location, delta_o=state.delta_o, p=state.p, to_move=new_to_move)
                        neighbors.append(new_state)
        return neighbors
    
    def get_valid_parking_spots(self, obs_id, delta_o):
        l_init = [k for k, v in self.movable_obstacles_map.items() if v == obs_id][0]
        valid_spots = []
        for dx in range(-self.k, self.k + 1):
            for dy in range(-(self.k - abs(dx)), (self.k - abs(dx)) + 1):
                target = (l_init[0] + dx, l_init[1] + dy)
                if (0 <= target[0] < self.dimension[0] and 0 <= target[1] < self.dimension[1] and target not in self.obstacles and target not in self.v_ep):
                    valid_spots.append(target)
        return valid_spots


    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def get_all_obstacles(self, time):
        all_obs = set()
        for o in self.moving_obstacles:
            if o[2] < 0 and time >= -o[2]:
                all_obs.add((o[0], o[1]))
        return self.obstacles | all_obs

    def state_valid(self, state):
        if not (0 <= state.location.x < self.dimension[0] and 0 <= state.location.y < self.dimension[1]):
            return False
        if state.location.to_tuple() in self.obstacles:
            return False
        curr_delta = dict(state.delta_o)
        for obs_id, trajectory in curr_delta.items():
            if state.time < len(trajectory):
                if state.location.to_tuple() == tuple(trajectory[state.time]):
                    return False
            else:
                if state.location.to_tuple() == tuple(trajectory[-1]):
                    return False

        # 4. Controllo collisioni con ostacoli mobili NON ancora spostati
        # Se un ostacolo non è in delta_o, significa che è ancora fermo alla sua posizione originale
        for loc, obs_id in self.movable_obstacles_map.items():
            if obs_id not in curr_delta:
                if state.location.to_tuple() == loc:
                    return False

        # 5. Controllo vincoli del CBS (altri agenti)
        if VertexConstraint(state.time, state.location) in self.constraints.vertex_constraints:
            return False
            
        return True

    def transition_valid(self, state_1, state_2):
        tup_1 = (state_1.location.x, state_1.location.y, state_2.time)
        tup_2 = (state_2.location.x, state_2.location.y, state_1.time)
        if tup_1 in self.moving_obstacles and tup_2 in self.moving_obstacles and \
                self.moving_obstacles[tup_1] == self.moving_obstacles[tup_2]:
            return False
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)


    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = State(0, Location(agent['goal'][0], agent['goal'][1]))

            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})
    
    def calculate_g(self, state):
        return state.time + (self.alpha * state.p)

    def compute_solution(self):
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution = self.a_star.search(agent)
            if not local_solution:
                return False
            solution.update({agent:local_solution})
        return solution

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost

class CBS(object):
    def __init__(self, environment):
        self.env = environment
        self.open_set = set()
        self.closed_set = set()

    def search(self):
        start = HighLevelNode()
        # TODO: Initialize it in a better way
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)

        self.open_set |= {start}

        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                print("Low level CBS - Solution found")

                return self.generate_plan(P.solution)

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # TODO: ending condition
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return {}

    def generate_plan(self, solution):
        plan = {'agents': {}, 'obstacles': {}}
        init_locations = self.env.obs_id_to_init_loc
        for agent, path in solution.items():
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
            plan['agents'][agent] = path_dict_list
            for state in path:
                curr_delta = dict(state.delta_o)
                for obs_id, init_pos in init_locations.items():
                    if obs_id not in plan['obstacles']:
                        plan['obstacles'][obs_id] = []
                    pos = curr_delta.get(obs_id, init_pos)
                    plan['obstacles'][obs_id].append({'t': state.time, 'x': pos[0], 'y': pos[1]})
            for entity_id in plan['obstacles']:
                original_traj = plan['obstacles'][entity_id]
                new_traj = []
                if original_traj:
                    new_traj.append(original_traj[0])
                    for i in range(1, len(original_traj)):
                        if original_traj[i] != original_traj[i-1]:
                            new_traj.append(original_traj[i])
                plan['obstacles'][entity_id] = new_traj
        return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-param", help="input file containing map and obstacles")
    parser.add_argument("-output", help="output file with the schedule")
    args = parser.parse_args()

    if args.param is None:
        args.param = 'input.yaml'
        args.output = 'output.yaml'


    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']

    env = Environment(dimension, agents, obstacles, a_star_max_iter=1000)

    # Searching
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print("Solution not found")
        exit(0)

    # Write to output file
    with open(args.output, 'r') as output_yaml:
        try:
            output = yaml.load(output_yaml, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)
