import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent

from environment import Move
import numpy as np
import random
import heapq
from collections import deque

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HomeLander"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
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

        def heuristic(pos):
            return min(self.manhattan_distance(pos, g) for g in goals)

        open_set = [(0.0, start)]
        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current in closed_set:
                continue
            closed_set.add(current)

            if current in goals:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current, map_state):
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
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

            expected_next = (current[0] + move.value[0], current[1] + move.value[1])

            if next_pos == expected_next:
                steps += 1
            else:
                break

        return steps

    def seek_turn_distance(
        self, start: tuple, enemy_pos: tuple, map_state: np.ndarray
    ) -> int:
        key = (start, enemy_pos)
        if key in self._dist_cache:
            return self._dist_cache[key]

        goals = self._capture_goals(enemy_pos, map_state)

        visited = set()
        queue = deque([(start, 0)])  # (current_pos, distance)

        while queue:
            current, distance = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if current in goals:
                self._dist_cache[key] = distance
                return distance

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = (current[0] + move.value[0], current[1] + move.value[1])
                if next_pos not in visited and self._is_valid_position(
                    next_pos, map_state
                ):
                    queue.append((next_pos, distance + 1))

        self._dist_cache[key] = 99999999  # Unreachable
        return 99999999

    def _get_seek_action(self, seek_pos: tuple, map_state: np.ndarray):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            for step in range(
                1,
                self._max_valid_steps(seek_pos, move, map_state, self.pacman_speed) + 1,
            ):
                next_pos = (
                    seek_pos[0] + move.value[0] * step,
                    seek_pos[1] + move.value[1] * step,
                )

                if (
                    self._is_valid_position(next_pos, map_state)
                    and self._max_valid_steps(seek_pos, move, map_state, step) >= step
                ):
                    actions.append((move, step))

        actions.append((Move.STAY, 1))

        return actions

    def _get_hide_action(self, hide_pos: tuple, map_state: np.ndarray):
        actions = []
        for move in Move:
            next_pos = (hide_pos[0] + move.value[0], hide_pos[1] + move.value[1])

            if self._is_valid_position(next_pos, map_state):
                actions.append((move, 1))

        return actions

    def _apply_seek_action(self, seek_pos: tuple, action: tuple, map_state: np.ndarray):
        move, steps = action

        if steps == 0:
            return seek_pos

        new_pos = (
            seek_pos[0] + move.value[0] * steps,
            seek_pos[1] + move.value[1] * steps,
        )

        return new_pos

    def _apply_hide_action(self, hide_pos: tuple, action: tuple, map_state: np.ndarray):
        move, steps = action

        if steps == 0:
            return hide_pos

        new_pos = (
            hide_pos[0] + move.value[0] * steps,
            hide_pos[1] + move.value[1] * steps,
        )
        if not self._is_valid_position(new_pos, map_state):
            return hide_pos

        return new_pos

    def _evaluate_state(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray):
        key = (my_pos, enemy_pos)
        if key in self._eval_cache:
            return self._eval_cache[key]

        if self._is_catch_position(my_pos, enemy_pos):
            return 1000

        distance = -self.manhattan_distance(
            my_pos, enemy_pos
        ) + 4 * self.seek_turn_distance(my_pos, enemy_pos, map_state)

        if distance is None:
            return -10000

        hide_mobility = len(self._get_hide_action(enemy_pos, map_state))
        score = -distance * 15 - hide_mobility * 7

        if hide_mobility <= 1:
            score += 100

        self._eval_cache[key] = score
        return score

    def _minimax_value(
        self,
        my_pos: tuple,
        enemy_pos: tuple,
        map_state: np.ndarray,
        depth: int,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:
        key = (my_pos, enemy_pos, depth)
        if key in self._minimax_cache:
            value = self._evaluate_state(my_pos, enemy_pos, map_state)
            self._minimax_cache[key] = value
            return value

        if self._is_catch_position(my_pos, enemy_pos) or depth == 0:
            return self._evaluate_state(my_pos, enemy_pos, map_state)

        best_score = -float("inf")  # Seek muốn score lớn nhất
        is_pruned = False

        for seek_action in self._get_seek_action(my_pos, map_state):
            seek_next = self._apply_seek_action(my_pos, seek_action, map_state)

            worst_score = float("inf")  # Hide muốn score nhỏ nhất
            local_beta = beta

            for hide_action in self._get_hide_action(enemy_pos, map_state):
                hide_next = self._apply_hide_action(enemy_pos, hide_action, map_state)

                score = self._minimax_value(
                    seek_next, hide_next, map_state, depth - 1, alpha, local_beta
                )

                if score < worst_score:
                    worst_score = score

                local_beta = min(local_beta, worst_score)

                if local_beta <= alpha:
                    is_pruned = True
                    break  # Alpha-beta pruning

            if worst_score > best_score:
                best_score = worst_score

            alpha = max(alpha, best_score)
            if beta <= alpha:
                is_pruned = True
                break  # Alpha-beta pruning
        if not is_pruned:
            self._minimax_cache[key] = best_score
        return best_score

    def _minimax_decision(
        self,
        my_pos: tuple,
        enemy_pos: tuple,
        map_state: np.ndarray,
        depth: int,
        alpha_d: float = float("-inf"),
        beta_d: float = float("inf"),
    ) -> tuple:
        best_score = float("-inf")
        best_action = (Move.STAY, 1)

        for action in self._get_seek_action(my_pos, map_state):
            new_my_pos = self._apply_seek_action(my_pos, action, map_state)

            worst_score = float("inf")
            local_beta = beta_d

            for hide_action in self._get_hide_action(enemy_pos, map_state):
                new_enemy_pos = self._apply_hide_action(
                    enemy_pos, hide_action, map_state
                )
                score = self._minimax_value(
                    new_my_pos,
                    new_enemy_pos,
                    map_state,
                    depth=depth - 1,
                    alpha=alpha_d,
                    beta=local_beta,
                )

                if score < worst_score:
                    worst_score = score

                local_beta = min(local_beta, worst_score)
                if local_beta <= alpha_d:
                    break  # Alpha-beta pruning

            if worst_score > best_score:
                best_score = worst_score
                best_action = action

            alpha_d = max(alpha_d, best_score)
            if beta_d <= alpha_d:
                break  # Alpha-beta pruning

        return best_action

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ):
        self._minimax_cache.clear()
        self._eval_cache.clear()
        self._dist_cache.clear()

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

        if len(path) <= 12:
            best_action = self._minimax_decision(
                my_position, target, map_state, depth=3
            )
            next_move = best_action[0]
            straight_steps = self._straight_steps_from_path(path, next_move)

            if straight_steps == len(path) - 1 and straight_steps < self.pacman_speed:
                desired_steps = self.pacman_speed
            else:
                desired_steps = straight_steps

            steps = self._max_valid_steps(
                my_position, next_move, map_state, desired_steps
            )
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

            steps = self._max_valid_steps(
                my_position, next_move, map_state, desired_steps
            )
            steps = max(1, steps)

        return (next_move, steps)

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)

        for move in all_moves:
            steps = self._max_valid_steps(
                my_position, move, map_state, self.pacman_speed
            )
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

    def _max_valid_steps(
        self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int
    ) -> int:
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


from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from hide_agent import panic
from debug import debug
from hide_agent import topology
from hide_agent import control
from time import perf_counter

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "SurvivabilityGhost"
        self.last_known_enemy_pos = None

        self.pacman_speed = 2
        self.topology_map = None
        self.previous_position = None
        debug.clear_log()
        debug.log("- - AGENT INITIALIZED - -")

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        start = perf_counter()
        map_state = np.asarray(map_state)

        if self.topology_map is None:
            self.topology_map = topology.build_topology_score_map(map_state)
            topology.write_topology_score_map(self.topology_map, map_state)

        my_position = (int(my_position[0]), int(my_position[1]))
        enemy_position = (int(enemy_position[0]), int(enemy_position[1]))

        move = self._choose_move(map_state, my_position, enemy_position, self.pacman_speed)
        self.previous_position = my_position

        if not isinstance(move, Move):
            return Move.STAY

        debug.log(f"[STEP] move={move}, time={(perf_counter() - start) * 1000:.3f} ms")
        return move

    def _choose_move(self, map_state: np.ndarray, my_position: tuple[int, int], enemy_position: tuple[int, int], pacman_speed=2):
        if enemy_position is None:
            return Move.STAY
        
        if panic.should_panic(enemy_position, my_position, map_state, pacman_speed):
            debug.log(f"[PANIC] Pacman={enemy_position}, Ghost{my_position}")
            return panic.choose_move(map_state, my_position, enemy_position)
        
        return control.choose_move(
            my_position,
            enemy_position,
            map_state,
            self.topology_map,
            pacman_speed,
            self.previous_position
        )