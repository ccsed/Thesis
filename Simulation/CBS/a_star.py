"""
AStar search
author: Ashwin Bose (@atb033)
author: Anonymous
"""
import heapq
from itertools import count
from collections import defaultdict


class AStar:
    def __init__(self, env):
        self.env = env
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.max_iter = env.a_star_max_iter
        self.iter = 0

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        detour_path = self._run_search(agent_name, allow_terraforming=False, upper_bound=float('inf'))
        
        if detour_path:
            upper_bound = self.env.calculate_g(detour_path[-1])
        else:
            upper_bound = float('inf')

        self.iter = 0 
        return self._run_search(agent_name, allow_terraforming=True, upper_bound=upper_bound)

    def _run_search(self, agent_name, allow_terraforming=True, upper_bound=float('inf')):
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1
        closed_set = defaultdict(list)
        came_from = {}
        
        g_score = {}
        bonus_score = {}
        g_score[initial_state] = self.env.calculate_g(initial_state)    
        bonus_score[initial_state] = 0.0
        h_score = self.admissible_heuristic(initial_state, agent_name)

        heap = []
        index = count(0)
        heapq.heappush(heap, (g_score[initial_state] + h_score, 0, h_score, next(index), initial_state))
        normal_expanded = 0
        intermediate_expanded = 0
        max_closed_set_bucket = 0
        while heap and (self.max_iter == -1 or self.iter < self.max_iter):
            self.iter = self.iter + 1
            
            current_f, _, _, _, current = heapq.heappop(heap)
            if current.is_intermidiate():
                intermediate_expanded += 1
            else:
                normal_expanded += 1
            
            if current_f > upper_bound:
                continue
                        
            current_g = self.env.calculate_g(current)

            effective_time = current.time if current.time < self.env.max_time else self.env.max_time
            base_key = (current.location.to_tuple(), effective_time, current.to_move)
            
            is_dominated = False
            for (c_delta_o, c_p, c_g, c_bonus) in closed_set[base_key]:
               
                if c_delta_o == current.delta_o and c_p <= current.p:
                    if c_g < current_g:
                        
                        is_dominated = True
                        break
                    elif c_g == current_g and c_bonus >= current.lookahead_bonus:
                        
                        is_dominated = True
                        break
                        
            if is_dominated:
                continue
                
            closed_set[base_key].append((current.delta_o, current.p, current_g, current.lookahead_bonus))
            bucket_size = len(closed_set[base_key])
            if bucket_size > max_closed_set_bucket:
                max_closed_set_bucket = bucket_size
            
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)
            
            for neighbor in self.get_neighbors(current, allow_terraforming=allow_terraforming, agent_name=agent_name):
                tentative_g_score = self.env.calculate_g(neighbor)
                prev_g = g_score.get(neighbor, float('inf'))
                prev_bonus = bonus_score.get(neighbor, -1.0)
                
                # Tie-breaking order:
                # 1. lower g cost
                # 2. equal g, more proactive delegations
                if (tentative_g_score < prev_g) or \
                   (tentative_g_score == prev_g and neighbor.lookahead_bonus > prev_bonus):
                    
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    bonus_score[neighbor] = neighbor.lookahead_bonus
                    base_h = self.admissible_heuristic(neighbor, agent_name)
                    f_score = tentative_g_score + base_h 
                    
                    heapq.heappush(heap, (f_score, -neighbor.lookahead_bonus, base_h, next(index), neighbor))
            
        print('Low level A* - Maximum iteration reached')
        return False