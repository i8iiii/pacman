"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import heapq
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """
        A* Search Algorithm Seeker.

        Strategy:
        - When enemy is visible: A* searches the shortest path to the target and
            follows the next move on that path.
        - When the target disappears: the agent remembers the last known enemy
            position and keeps pursuing it.
        - If no target is known: the agent falls back to the first valid move it can
            find on the current map.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "A-Star Seeker"
        self.last_known_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            return (self._random_valid_move(my_position, map_state), 1)

        # A* Search
        path = self._a_star(my_position, target, map_state)
        if not path or len(path) < 2:
            return (Move.STAY, 1)

        next_pos = path[1]
        move = self._get_move_dir(my_position, next_pos)
        
        if len(path) > 2:
            second_pos = path[2]
            second_move = self._get_move_dir(next_pos, second_pos)
            if second_move == move:
                return (move, 2)
        
        return (move, 1)

    def _a_star(self, start, goal, map_state):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal: break

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_node = (current[0] + dr, current[1] + dc)
                
                if self._is_valid(next_node, map_state):
                    new_cost = cost_so_far[current] + 1
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + abs(next_node[0]-goal[0]) + abs(next_node[1]-goal[1])
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current
        
        # Tái tạo đường đi
        path = []
        curr = goal if goal in came_from else None
        while curr:
            path.append(curr)
            curr = came_from[curr]
        return path[::-1]

    def _get_move_dir(self, p1, p2):
        dr, dc = p2[0]-p1[0], p2[1]-p1[1]
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if m.value == (dr, dc): return m
        return Move.STAY

    def _is_valid(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _random_valid_move(self, pos, map_state):
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid((pos[0]+m.value[0], pos[1]+m.value[1]), map_state): return m
        return Move.STAY

class GhostAgent(BaseGhostAgent):
    """
    BFS-based evasive ghost.

    Strategy:
    - When Pacman is visible: build a BFS danger map from Pacman's position
        and choose the move that maximizes distance from the threat.
    - When Pacman is hidden: reuse the last known position and keep evading.
    - If no threat is known: choose the first valid random move.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if step_number == 1:
            return Move.STAY
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return self._random_move(my_position, map_state)

        # BFS từ vị trí Pacman để tìm "bản đồ nguy hiểm"
        danger_map = self._get_danger_map(threat, map_state)
        
        best_move = Move.STAY
        max_dist = danger_map.get(my_position, 0)

        # Ghost chọn hướng đi có khoảng cách đến Pacman xa nhất 
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            dr, dc = move.value
            next_pos = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid(next_pos, map_state):
                dist = danger_map.get(next_pos, 0)
                if dist > max_dist:
                    max_dist = dist
                    best_move = move
        
        return best_move

    def _get_danger_map(self, threat_pos, map_state):
        dists = {threat_pos: 0}
        queue = deque([threat_pos])
        while queue:
            curr = queue.popleft()
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0]+m.value[0], curr[1]+m.value[1])
                if self._is_valid(nxt, map_state) and nxt not in dists:
                    dists[nxt] = dists[curr] + 1
                    queue.append(nxt)
        return dists

    def _is_valid(self, pos, map_state):
        r, c = pos
        return 0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] == 0

    def _random_move(self, pos, map_state):
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid((pos[0]+m.value[0], pos[1]+m.value[1]), map_state): return m
        return Move.STAY