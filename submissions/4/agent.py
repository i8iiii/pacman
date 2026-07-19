import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np
import random
import heapq

from seek_agent.topology import (
    build_trap_map,
    precompute_valid_positions,
    is_valid_fast,
    manhattan_distance,
)


class PacmanAgent(BasePacmanAgent):
    """Pacman seeker: A* shortest-path with max-speed straight-line moves."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HomeLander"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self._valid_positions = None

    # --------------------------------------------------------------
    # public
    # --------------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            enemy_position = (int(enemy_position[0]), int(enemy_position[1]))
        my_position = (int(my_position[0]), int(my_position[1]))

        if self._valid_positions is None:
            self._valid_positions = precompute_valid_positions(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            return self._explore(my_position)

        if manhattan_distance(my_position, target) < 2:
            return Move.STAY, 1

        path = self._astar(my_position, target)
        if path and len(path) >= 2:
            return self._follow_path(my_position, path)
        return self._greedy_move(my_position, target)

    # --------------------------------------------------------------
    # A* -- pure shortest path
    # --------------------------------------------------------------
    def _astar(self, start, goal):
        if start == goal or not is_valid_fast(start, self._valid_positions):
            return None
        if not is_valid_fast(goal, self._valid_positions):
            return None

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = mv.value
                nxt = (current[0] + dr, current[1] + dc)
                if not is_valid_fast(nxt, self._valid_positions):
                    continue
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + manhattan_distance(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return None

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    # --------------------------------------------------------------
    # path following -- max-speed in straight A* direction
    # --------------------------------------------------------------
    def _follow_path(self, my_pos, path):
        nxt = path[1]
        dr = nxt[0] - my_pos[0]
        dc = nxt[1] - my_pos[1]
        if dr != 0:
            direction = Move.DOWN if dr > 0 else Move.UP
        else:
            direction = Move.RIGHT if dc > 0 else Move.LEFT

        consecutive = 0
        cur = my_pos
        for node in path[1:]:
            ndr = node[0] - cur[0]
            ndc = node[1] - cur[1]
            if ndr != 0:
                node_dir = Move.DOWN if ndr > 0 else Move.UP
            else:
                node_dir = Move.RIGHT if ndc > 0 else Move.LEFT
            if node_dir != direction:
                break
            cur = node
            consecutive += 1

        steps = min(consecutive, self.pacman_speed)
        if steps == 0:
            steps = 1
        return direction, steps

    # --------------------------------------------------------------
    # fallback
    # --------------------------------------------------------------
    def _greedy_move(self, my_pos, target):
        best_move, best_steps, best_dist = Move.STAY, 1, manhattan_distance(my_pos, target)
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            pos = my_pos
            valid_steps = 0
            for _ in range(self.pacman_speed):
                candidate = (pos[0] + move.value[0], pos[1] + move.value[1])
                if not is_valid_fast(candidate, self._valid_positions):
                    break
                pos = candidate
                valid_steps += 1
            for s in range(1, valid_steps + 1):
                dr = move.value[0] * s
                dc = move.value[1] * s
                dist = manhattan_distance((my_pos[0] + dr, my_pos[1] + dc), target)
                if dist < best_dist:
                    best_dist = dist
                    best_move = move
                    best_steps = s
        return best_move, best_steps

    def _explore(self, my_pos):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for mv in moves:
            dr, dc = mv.value
            pos = my_pos
            valid_steps = 0
            for _ in range(self.pacman_speed):
                candidate = (pos[0] + dr, pos[1] + dc)
                if not is_valid_fast(candidate, self._valid_positions):
                    break
                pos = candidate
                valid_steps += 1
            if valid_steps > 0:
                return mv, min(valid_steps, self.pacman_speed)
        return Move.STAY, 1


from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from hide_agent import panic
from debug import debug
from hide_agent import topology
from hide_agent import control

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "Skidadal"
        self.last_known_enemy_pos = None

        self.pacman_speed = 2
        self.topology_map = None
        self.previous_position = None

        debug.clear_log()
        debug.log("- - AGENT INITIALIZED - -")

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        control.reset_timer()
        map_state = np.asarray(map_state)

        if self.topology_map is None:
            self.topology_map = topology.build_topology_score_map(map_state)
            if debug.DEBUG_ENABLED:
                topology.write_topology_score_map(self.topology_map, map_state)

        my_position = (int(my_position[0]), int(my_position[1]))
        enemy_position = (int(enemy_position[0]), int(enemy_position[1]))

        move = self._choose_move(map_state, my_position, enemy_position, self.pacman_speed)
        self.previous_position = my_position

        if not isinstance(move, Move):
            return Move.STAY

        debug.log(f"[STEP] move={move}, time={control.get_run_time() / 1000:.3f} s\n\n")
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
