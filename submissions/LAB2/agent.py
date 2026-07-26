#!/usr/bin/env python3
"""PacmanAgent: Speed-2 A* chaser + upper-half staged search."""
import sys, random, numpy as np
from collections import deque
from pathlib import Path
from heapq import heappush, heappop

src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from hide_agent.controller import HideController

DIRS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

# ---------- GhostAgent (unchanged) --------------------------------------

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hide-Agent"
        self._controller = HideController(
            log_path=kwargs.get("log_path"),
            map_text_path=kwargs.get("map_text_path"),
            map_jsonl_path=kwargs.get("map_jsonl_path"),
            diagnostics_enabled=kwargs.get("diagnostics_enabled"),
            pacman_speed=kwargs.get("pacman_speed", 2),
            capture_distance=kwargs.get("capture_distance", 2),
            observation_radius=kwargs.get("observation_radius", 5),
        )
    def step(self, map_state, my_position, enemy_position, step_number):
        return self._controller.step(map_state, my_position, enemy_position, step_number)

# ---------- Pathfinding helpers -----------------------------------------

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _is_valid(pos, map_state):
    r, c = pos
    if map_state is None: return False
    h, w = map_state.shape
    return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1

def _astar(start, goal, map_state):
    if start == goal: return []
    open_set = [(0, 0, start, [])]
    g_score = {start: 0}
    closed = set()
    counter = 0
    while open_set:
        _, _, current, path = heappop(open_set)
        if current in closed: continue
        closed.add(current)
        if current == goal: return path
        for mv in DIRS:
            nxt = (current[0] + mv.value[0], current[1] + mv.value[1])
            if nxt in closed or not _is_valid(nxt, map_state): continue
            tg = g_score[current] + 1
            if tg < g_score.get(nxt, float('inf')):
                g_score[nxt] = tg
                h = _manhattan(nxt, goal)
                counter += 1
                heappush(open_set, (tg + h, counter, nxt, path + [mv]))
    return []

def _greedy_toward(my_pos, target, map_state):
    best_move, best_dist = Move.STAY, _manhattan(my_pos, target)
    for mv in DIRS:
        nxt = (my_pos[0] + mv.value[0], my_pos[1] + mv.value[1])
        if _is_valid(nxt, map_state):
            d = _manhattan(nxt, target)
            if d < best_dist: best_dist, best_move = d, mv
    return best_move

# ---------- Staged search targets ---------------------------------------

_SIDE_CORNERS = {(1,1),(1,2),(1,3),(1,17),(1,18),(1,19)}
_MIDDLE_POCKETS = {(5,5),(5,6),(5,14),(5,15),(9,8),(9,9),(9,10),(9,11),(9,12)}

def _find_stage_target(my_pos, internal_map, stage):
    if internal_map is None: return None
    h, w = internal_map.shape
    mid_row = h // 2
    best, best_score = None, -1.0
    for r in range(h):
        for c in range(w):
            if internal_map[r,c] != 0: continue
            has_unknown = any(0 <= r+mv.value[0] < h and 0 <= c+mv.value[1] < w
                            and internal_map[r+mv.value[0], c+mv.value[1]] == -1 for mv in DIRS)
            if not has_unknown: continue
            if stage == 1 and (r,c) not in _SIDE_CORNERS: continue
            if stage == 2 and (r,c) not in _MIDDLE_POCKETS: continue
            if stage == 3 and (r >= mid_row or (r,c) in _SIDE_CORNERS or (r,c) in _MIDDLE_POCKETS): continue
            if stage == 4 and r < mid_row: continue
            dist = _manhattan(my_pos, (r,c))
            if dist == 0: continue
            score = 1.0/dist
            if score > best_score: best_score, best = score, (r,c)
    return best

def _staged_search(my_pos, internal_map, current_stage):
    for stage in range(current_stage, 5):
        t = _find_stage_target(my_pos, internal_map, stage)
        if t is not None: return t, stage
    return None, current_stage

# ---------- Pacman Agent ------------------------------------------------

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "A*-Chaser"
        self.internal_map = None
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.last_move = None
        self._search_stage = 1

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_map_memory(map_state)
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        my_pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos = (int(enemy_position[0]), int(enemy_position[1])) if enemy_position is not None else None

        chosen_move = Move.STAY
        path = None

        if enemy_pos is not None:
            # Ghost visible: A* chase (speed-2 closes gap quickly)
            if _manhattan(my_pos, enemy_pos) <= 1:
                chosen_move = Move.STAY  # already adjacent
            else:
                # Predict: if ghost is near a pocket, cut it off
                target = enemy_pos
                # Check if ghost is heading toward a pocket
                for pocket in _SIDE_CORNERS | _MIDDLE_POCKETS:
                    if _is_valid(pocket, self.internal_map):
                        gd = _manhattan(enemy_pos, pocket)
                        if gd <= 3:
                            # Ghost may be heading to pocket, intercept
                            target = pocket
                            break
                path = _astar(my_pos, target, self.internal_map)
                if path: chosen_move = path[0]
                else: chosen_move = _greedy_toward(my_pos, target, self.internal_map)
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 10:
            # Recently lost sight: go to last known position
            path = _astar(my_pos, self.last_known_enemy_pos, self.internal_map)
            if path: chosen_move = path[0]
            else: chosen_move = _greedy_toward(my_pos, self.last_known_enemy_pos, self.internal_map)
        else:
            # Ghost hidden: staged search of upper half
            target, stage = _staged_search(my_pos, self.internal_map, self._search_stage)
            if target is not None:
                path = _astar(my_pos, target, self.internal_map)
                if path: chosen_move = path[0]
            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_pos)

        # Speed-2 whenever possible
        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            dr, dc = chosen_move.value
            nr2, nc2 = my_pos[0] + dr * 2, my_pos[1] + dc * 2
            if _is_valid((nr2, nc2), self.internal_map):
                if not (path and len(path) >= 2 and path[0] == chosen_move and path[1] != chosen_move):
                    steps = 2

        self.last_move = chosen_move
        return (chosen_move, steps)

    # ---------- Map memory -----------------------------------------------

    def _update_map_memory(self, map_state):
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.internal_map[map_state == 1] = 1
            self.map_initialized = True
        visible = map_state != -1
        self.internal_map[visible] = map_state[visible]
        self.internal_map[(self.internal_map == 2) | (self.internal_map == 3)] = 0

    def _random_valid_move(self, my_pos):
        moves = [mv for mv in DIRS if _is_valid((my_pos[0] + mv.value[0], my_pos[1] + mv.value[1]), self.internal_map)]
        return random.choice(moves) if moves else Move.STAY
