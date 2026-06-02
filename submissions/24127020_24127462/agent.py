import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
import heapq
from collections import deque

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HomeLander"
        self.pacman_speed = max(1, int(kwargs.get('pacman_speed', 1)))
        self.last_known_enemy_pos = None
        self.visit_count = {}
        self.recent_positions = deque(maxlen=8)

    def manhattan_distance(self, pos1: tuple, pos2: tuple) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def astar(self, start: tuple, goal: tuple, map_state: np.ndarray):
        open_set = [(0, start)]
        g_score = {}       
        f_score = {}
        g_score[start] = 0
        f_score[start] = self.manhattan_distance(start, goal)
        came_from = {}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current, map_state):
                neighbor_cost = 1 + self._visit_penalty(neighbor)
                tentative_g_score = g_score[current] + neighbor_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if self._is_valid_position(next_pos, map_state):
                neighbors.append(next_pos)

        return neighbors
    
    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def move_from_positions(self, current_pos, next_pos):
        row_diff = next_pos[0] - current_pos[0]
        col_diff = next_pos[1] - current_pos[1]

        if row_diff == -1:
            return Move.UP
        elif row_diff == 1:
            return Move.DOWN
        elif col_diff == -1:
            return Move.LEFT
        elif col_diff == 1:
            return Move.RIGHT
        else:
            return Move.STAY
        
    def _straight_steps_from_path(self, path: list, move: Move) -> int:
        if path is None or len(path) < 2:
            return 0

        steps = 0
        max_steps = min(self.pacman_speed, len(path) - 1)

        for i in range(max_steps):
            current = path[i]
            next_pos = path[i + 1]

            expected_next = (
                current[0] + move.value[0],
                current[1] + move.value[1]
            )

            if next_pos == expected_next:
                steps += 1
            else:
                break

        return steps
    
    def _visit_penalty(self, pos: tuple):
        penalty = self.visit_count.get(pos, 0) * 0.5

        if pos in self.recent_positions:
            penalty += 3

        return penalty

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1
        self.recent_positions.append(my_position)

        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            return self._explore(my_position, map_state)

        path = self.astar(my_position, target, map_state)
        if path is None or len(path) < 2:
            return self._explore(my_position, map_state)

        next_pos = path[1]
        next_move = self.move_from_positions(my_position, next_pos)
        steps = self._straight_steps_from_path(path, next_move)
        steps = min(self.pacman_speed, steps)

        return (next_move, steps)
    
    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    # def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
    #     if move in (Move.UP, Move.DOWN):
    #         return abs(row_diff)
    #     if move in (Move.LEFT, Move.RIGHT):
    #         return abs(col_diff)
    #     return 1


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.name = "A-Train"
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            enemy_position = self.predictEnemyNextPosition(map_state, my_position, enemy_position)
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position if enemy_position is not None else self.last_known_enemy_pos

        if threat is None:
            return Move.STAY

        cur_distance = self._calc_path_distance(map_state, my_position, threat)

        candidates = []
        for neighbor_cell in self._get_neighbors(my_position, map_state):
            if not self._is_valid_position(neighbor_cell, map_state):
                continue
            if neighbor_cell == my_position:
                continue

            for neighbor_neighbor_cell in self._get_neighbors(neighbor_cell, map_state):
                if not self._is_valid_position(neighbor_neighbor_cell, map_state):
                    continue
                if neighbor_neighbor_cell == neighbor_cell:
                    continue

                if self._calc_path_distance(map_state, threat, neighbor_neighbor_cell) < cur_distance:
                    continue

                candidates.append(neighbor_cell)
                break

        move = Move.STAY
        max_distance = 0
        for candidate in candidates:
            distance = self._calc_path_distance(map_state, threat, candidate)
            if distance > max_distance:
                max_distance = distance
                move = self._translate_move(my_position, candidate)

        return move    
    
    # Helper methods (you can add more)
    def bfs(self,map_state: np.ndarray,start_pos: tuple,des_pos: tuple) -> list[tuple[int, int]]:
        from collections import deque

        if start_pos == des_pos:
            return [start_pos]

        queue = deque([start_pos])
        visited = {start_pos}
        parent = {}

        moves = [Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN]

        while queue:
            cur_pos = queue.popleft()

            if cur_pos == des_pos:
                break

            for move in moves:
                if self._is_valid_move(cur_pos, move, map_state):
                    new_pos = self._apply_move(cur_pos, move)

                    if new_pos not in visited:
                        visited.add(new_pos)
                        parent[new_pos] = cur_pos
                        queue.append(new_pos)

        if des_pos not in visited:
            return []

        path = []
        cur_pos = des_pos

        while cur_pos != start_pos:
            path.append(cur_pos)
            cur_pos = parent[cur_pos]

        path.append(start_pos)
        path.reverse()

        return path
    
    def predictEnemyNextPosition(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple) -> tuple:
        path = self.bfs(map_state, enemy_pos, my_pos)

        if len(path) <= 1:
            return enemy_pos
        if len(path) == 2:
            return path[1]
        
        first_move = self._translate_move(path[0], path[1])
        second_move = self._translate_move(path[1], path[2])

        if first_move == second_move: # Same move - Straight = Move two cells
            return path[2]

        return path[1]
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        row, col = pos
        if move == Move.UP:
            return (row - 1, col)
        elif move == Move.DOWN:
            return (row + 1, col)
        elif move == Move.LEFT:
            return (row, col - 1)
        elif move == Move.RIGHT:
            return (row, col + 1)
        return pos  # Move.STAY

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list[tuple[int, int]]:
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            neighbor = self._apply_move(pos, move)
            row, col = neighbor
            if 0 <= row < map_state.shape[0] and 0 <= col < map_state.shape[1]:
                neighbors.append(neighbor)
        return neighbors

    def _calc_path_distance(self, map_state: np.ndarray, start: tuple, des: tuple) -> int:
        path = self.bfs(map_state, start, des)
        if not path:
            return 99999999  # Unreachable
        return len(path) - 1    # Number of steps, not cells

    def _translate_move(self, start: tuple, des: tuple) -> Move:
        sr, sc = start
        dr, dc = des
        if dr == sr - 1 and dc == sc:
            return Move.UP
        elif dr == sr + 1 and dc == sc:
            return Move.DOWN
        elif dr == sr and dc == sc - 1:
            return Move.LEFT
        elif dr == sr and dc == sc + 1:
            return Move.RIGHT
        return Move.STAY