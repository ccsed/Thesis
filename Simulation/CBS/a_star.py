"""
AStar search
author: Ashwin Bose (@atb033)
author: Giacomo Lodigiani (@Lodz97)
"""
import heapq
from itertools import count


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
        """
        low level search
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        closed_set = {}

        came_from = {}

        g_score = {}
        g_score[initial_state] = self.env.calculate_g(initial_state)    

        h_score = self.admissible_heuristic(initial_state, agent_name)

        heap = []
        index = count(0)
        heapq.heappush(heap, (g_score[initial_state] + h_score, h_score, next(index), initial_state))

        while heap and (self.max_iter == -1 or self.iter < self.max_iter):
            self.iter = self.iter + 1
            current = heapq.heappop(heap)[3]
            state_key = (current.location.to_tuple(), current.time, current.to_move, current.delta_o)
            current_g = self.env.calculate_g(current)
            if state_key in closed_set and current_g >= closed_set[state_key]:
                continue
            closed_set[state_key] = current_g
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = self.env.calculate_g(neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.admissible_heuristic(neighbor, agent_name)
                    heapq.heappush(heap, (f_score, self.admissible_heuristic(neighbor, agent_name), next(index), neighbor))
            if self.iter == self.max_iter:
                print('Low level A* - Maximum iteration reached')
        return False