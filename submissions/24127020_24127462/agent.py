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
        self._minimax_cache = {}
        self._eval_cache = {}
        self._dist_cache = {}

    def _normalize_pos(self, pos: tuple) -> tuple:
        return (int(pos[0]), int(pos[1]))
    
    def _is_catch_position(self, p_pos: tuple, g_pos: tuple) -> bool:
        return self.manhattan_distance(p_pos, g_pos) < 2

    def _capture_goals(self, enemy_pos: tuple, map_state: np.ndarray) -> list:
        goals = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            goals.append(enemy_pos)
            next_pos = (enemy_pos[0] + move.value[0], enemy_pos[1] + move.value[1])
            if self._is_valid_position(next_pos, map_state):
                goals.append(next_pos)
        return goals

    def manhattan_distance(self, pos1: tuple, pos2: tuple) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def astar_to_any_goal(self, start: tuple, goal: tuple, map_state: np.ndarray):
        goals = self._capture_goals(goal, map_state)
        all_goals = [goal] + goals

        def heuristic(pos):
            return min(self.manhattan_distance(pos, g) for g in all_goals)

        open_set = [(0.0, start)]
        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current in closed_set:
                continue
            closed_set.add(current)

            if current in all_goals:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current, map_state):
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (f, neighbor))

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
    
    def seek_turn_distance(self, start: tuple, enemy_pos: tuple, map_state: np.ndarray) -> int:
        goals = self._capture_goals(enemy_pos, map_state)

        visited = set()
        queue = deque([(start, 0)])  # (current_pos, distance)

        while queue:
            current, distance = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if current in goals:
                return distance

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = (current[0] + move.value[0], current[1] + move.value[1])
                if next_pos not in visited and self._is_valid_position(next_pos, map_state):
                    queue.append((next_pos, distance + 1))

        return 99999999  # Unreachable

    def _get_seek_action(self, seek_pos: tuple, map_state: np.ndarray):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            for step in range(1, self._max_valid_steps(seek_pos, move, map_state, self.pacman_speed) + 1):
                next_pos = (
                    seek_pos[0] + move.value[0] * step,
                    seek_pos[1] + move.value[1] * step
                )

                if self._is_valid_position(next_pos, map_state) and self._max_valid_steps(seek_pos, move, map_state, step) >= step:
                    actions.append((move, step))

        actions.append((Move.STAY, 1))

        return actions

    def _get_hide_action(self, hide_pos: tuple, map_state: np.ndarray):
        actions = []
        for move in Move:
            next_pos = (
                hide_pos[0] + move.value[0],
                hide_pos[1] + move.value[1]
            )

            if self._is_valid_position(next_pos, map_state):
                actions.append((move, 1))

        return actions

    def _apply_seek_action(self, seek_pos: tuple, action: tuple, map_state: np.ndarray):
        move, steps = action

        if steps == 0:
            return seek_pos

        new_pos = (seek_pos[0] + move.value[0] * steps, seek_pos[1] + move.value[1] * steps)

        return new_pos


    def _apply_hide_action(self, hide_pos: tuple, action: tuple, map_state: np.ndarray):
        move, steps = action

        if steps == 0:
            return hide_pos

        new_pos = (hide_pos[0] + move.value[0] * steps, hide_pos[1] + move.value[1] * steps)
        if not self._is_valid_position(new_pos, map_state):
            return hide_pos

        return new_pos

    def _evaluate_state(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray):
        if self._is_catch_position(my_pos, enemy_pos):
            return 1000
        
        distance = self.seek_turn_distance(my_pos, enemy_pos, map_state)
        
        if distance is None:
            return -10000
        
        hide_mobility = len(self._get_hide_action(enemy_pos, map_state))
        score = -distance * 100 - hide_mobility * 20

        if hide_mobility <=1:
            score += 50

        return score

    def _minimax_value(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray, depth: int):
        if self._is_catch_position(my_pos, enemy_pos) or depth == 0:
            return self._evaluate_state(my_pos, enemy_pos, map_state)

        best_score = -float('inf')  # Seek muốn score lớn nhất

        for seek_action in self._get_seek_action(my_pos, map_state):
            seek_next = self._apply_seek_action(my_pos, seek_action, map_state)

            worst_score = float('inf')  # Hide muốn score nhỏ nhất

            for hide_action in self._get_hide_action(enemy_pos, map_state):
                hide_next = self._apply_hide_action(enemy_pos, hide_action, map_state)

                score = self._minimax_value(
                    seek_next,
                    hide_next,
                    map_state,
                    depth - 1
                )

                if score < worst_score:
                    worst_score = score

            if worst_score > best_score:
                best_score = worst_score

        return best_score

    def _minimax_decision(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray):
        best_score = float('-inf')
        best_action = (Move.STAY, 1)

        for action in self._get_seek_action(my_pos, map_state):
            new_my_pos = self._apply_seek_action(my_pos, action, map_state)
            score = self._minimax_value(new_my_pos, enemy_pos, map_state, depth=2)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self._minimax_cache = {}
        self._eval_cache = {}
        self._dist_cache = {}

        if enemy_position is not None:
            enemy_position = self._normalize_pos(enemy_position)
            self.last_known_enemy_pos = enemy_position

        my_position = self._normalize_pos(my_position)
        
        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            return self._explore(my_position, map_state)

        path = self.astar_to_any_goal(my_position, target, map_state)

        if self._is_catch_position(my_position, target):
            return Move.STAY, 1
        
        if path is None or len(path) < 2:
            return Move.STAY, 1
        
        if len(path) <= 7:
            best_action = self._minimax_decision(my_position, target, map_state)
            next_move = best_action[0]
            straight_steps = self._straight_steps_from_path(path, next_move)

            if straight_steps == len(path) - 1 and straight_steps < self.pacman_speed:
                desired_steps = self.pacman_speed
            else:
                desired_steps = straight_steps

            steps = self._max_valid_steps(my_position, next_move, map_state, desired_steps)
            best_action = (next_move, steps)

            return best_action
        else:
            next_pos = path[1]
            next_move = self.move_from_positions(my_position, next_pos)
            straight_steps = self._straight_steps_from_path(path, next_move)

            if straight_steps == len(path) - 1 and straight_steps < self.pacman_speed:
                desired_steps = self.pacman_speed
            else:
                desired_steps = straight_steps

            steps = self._max_valid_steps(my_position, next_move, map_state, desired_steps)
            steps = max(1, steps)

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

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1

from hide_agent import (
    Move,
    np,
    _apply_move,
    _is_valid_position,
    _topology_score,
    _local_space_score,
    _bfs_distances,
    _bfs_path,
    _get_neighbors,
    _is_straight_corridor,
    _translate_move,
    THREAT_TRIGGER,
    W_DIST,
    W_RATIO,
)

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A-Train"

        self._map_built = False
        self.last_known_enemy_pos = None
        self.safety_map: dict[tuple, float] = {}

    def step(self, map_state, my_position, enemy_position, step_number) -> Move:
        if not self._map_built:
            self._build_safety_map(map_state)
            # debugging
            self._dump_safety_map(map_state, Path(__file__).parent / "hide_agent" / "safety_map.txt")  
            self._map_built = True

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = (
            enemy_position if enemy_position is not None else self.last_known_enemy_pos
        )

        # LAB1: 
        return self._best_move(map_state, my_position, threat)

    def _best_move(
        self, map_state: np.ndarray, my_pos: tuple, threat_pos: tuple
    ) -> Move:

        # TODO: Blind Version
        if my_pos is None or threat_pos is None:
            return Move.STAY

        my_dist = _bfs_distances(my_pos, map_state)
        seeker_dist = _bfs_distances(threat_pos, map_state)

        seeker_steps_to_me = seeker_dist.get(my_pos, 3636) / 2
        seeker_is_close = seeker_steps_to_me <=  THREAT_TRIGGER

        best_score = float("-inf")
        best_dest = None

        for tile in self.safety_map:
            if tile == my_pos:
                continue

            my_d = my_dist.get(tile)
            seek_d = seeker_dist.get(tile)

            if my_d is None or seek_d is None:
                continue

            # Seeker moves 2 cells
            seek_d = (seek_d + 1) // 2

            # Seeker can reach this tile first
            if not seeker_is_close and my_d >= seek_d:
                continue

            neighbors = _get_neighbors(tile, map_state)

            is_corner = len(neighbors) == 2 and not _is_straight_corridor(
                tile, neighbors
            )

            ratio = my_d / max(seek_d, 1)

            score = self.safety_map[tile] + W_DIST * seek_d - W_RATIO * ratio

            if seeker_is_close and is_corner:
                score += 30

            dx = tile[0] - my_pos[0]
            dy = tile[1] - my_pos[1]
            sx = threat_pos[0] - my_pos[0]
            sy = threat_pos[1] - my_pos[1]
            if dx * sx + dy * sy > 0:
                score -= 10    

            if score > best_score:
                best_score = score
                best_dest = tile

        # Fallback
        if best_dest is None:
            neighbors = _get_neighbors(my_pos, map_state)

            if not neighbors:
                return Move.STAY

            best_dest = max(neighbors, key=lambda pos: seeker_dist.get(pos, 0))

        path = _bfs_path(my_pos, best_dest, map_state)

        if len(path) < 2:
            return Move.STAY

        return _translate_move(my_pos, path[1])

    def _build_safety_map(self, map_state: np.ndarray) -> None:
        self.safety_map = {}

        for row in range(map_state.shape[0]):
            for col in range(map_state.shape[1]):
                pos = (row, col)

                if not _is_valid_position(pos, map_state):
                    continue

                topo = _topology_score(pos, map_state)
                space = _local_space_score(pos, map_state)

                self.safety_map[pos] = topo + space

    def _dump_safety_map(
        self, map_state: np.ndarray, filename=Path(__file__).parent / "safety_map.txt"
    ) -> None:
        with open(filename, "w") as f:
            for row in range(map_state.shape[0]):
                line = []
                for col in range(map_state.shape[1]):
                    pos = (row, col)
                    if not _is_valid_position(pos, map_state):
                        line.append("####")
                    else:
                        line.append(f"{ self.safety_map[pos]:4.0f}")
                f.write(" ".join(line) + "\n")
