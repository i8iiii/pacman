"""
PacmanAgent: Reactive speed-2 chaser with interception.
No ML — pure A* + greedy interception + dead-end cornering.
"""
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

# ============================================================
# GhostAgent (unchanged)
# ============================================================
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

# ============================================================
# Utility helpers
# ============================================================
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

def _bfs_dist(start, goal, map_state):
    """BFS distance between two points."""
    if start == goal: return 0
    q = deque([(start, 0)])
    visited = {start}
    while q:
        cur, d = q.popleft()
        for mv in DIRS:
            nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
            if nxt == goal: return d + 1
            if nxt not in visited and _is_valid(nxt, map_state):
                visited.add(nxt)
                q.append((nxt, d + 1))
    return 999

def _count_exits(pos, map_state):
    return sum(1 for mv in DIRS
               if _is_valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state))

# ============================================================
# Map analysis: detect dead-ends and corners
# ============================================================
def _analyze_map(map_state):
    h, w = map_state.shape
    dead_ends = set()
    corners = set()
    exits_cache = {}
    
    for r in range(h):
        for c in range(w):
            if map_state[r, c] != 0: continue
            e = _count_exits((r, c), map_state)
            exits_cache[(r, c)] = e
            if e == 1:
                dead_ends.add((r, c))
            elif e == 2:
                neighbors = [(r + mv.value[0], c + mv.value[1]) for mv in DIRS
                            if _is_valid((r + mv.value[0], c + mv.value[1]), map_state)]
                if len(neighbors) == 2:
                    r1, c1 = neighbors[0]; r2, c2 = neighbors[1]
                    if r1 != r2 and c1 != c2:  # perpendicular = corner
                        corners.add((r, c))
    
    return dead_ends, corners, exits_cache

# ============================================================
# Interception target: predict where ghost will be
# ============================================================
def _intercept_target(my_pos, ghost_pos, map_state, dead_ends, corners, exits):
    """Predict where the ghost will go and move to cut it off."""
    mid_row = map_state.shape[0] // 2
    
    candidates = []
    
    # Check: is the ghost in or near a dead-end? If so, block the exit.
    for de in dead_ends:
        # If ghost is close to a dead-end, move to block the entrance
        if _bfs_dist(ghost_pos, de, map_state) <= 4:
            # Find cells that lead INTO this dead-end area
            for mv in DIRS:
                entrance = (de[0] + mv.value[0], de[1] + mv.value[1])
                if _is_valid(entrance, map_state) and exits.get(entrance, 0) >= 2:
                    candidates.append(entrance)
    
    # Check ghost's neighbors: which one looks like an escape route?
    for mv in DIRS:
        nxt = (ghost_pos[0] + mv.value[0], ghost_pos[1] + mv.value[1])
        if not _is_valid(nxt, map_state): continue
        
        # Ghost likely moves toward exits-rich or upper-half cells
        score = exits.get(nxt, 0) * 2
        if nxt[0] < mid_row: score += 3
        if nxt in dead_ends: score += 5
        if nxt in corners: score += 3
        candidates.append((nxt, score))
    
    if candidates:
        # If candidates with scores exist, pick the best
        if isinstance(candidates[0], tuple) and len(candidates[0]) == 2:
            best = max(candidates, key=lambda x: x[1])
            return best[0]
        else:
            # Plain list of positions
            best = min(candidates, key=lambda p: _manhattan(my_pos, p))
            return best
    return ghost_pos

# ============================================================
# Search target: find the best cell to explore
# ============================================================
def _find_search_target(my_pos, internal_map, dead_ends, corners):
    if internal_map is None: return None
    h, w = internal_map.shape
    mid_row = h // 2
    best, best_score = None, -1.0
    
    for r in range(h):
        for c in range(w):
            if internal_map[r, c] != 0: continue
            
            # Must border fog
            has_unknown = any(
                0 <= r + mv.value[0] < h and 0 <= c + mv.value[1] < w
                and internal_map[r + mv.value[0], c + mv.value[1]] == -1
                for mv in DIRS
            )
            if not has_unknown: continue
            
            dist = _manhattan(my_pos, (r, c))
            if dist == 0: continue
            score = 1.0 / dist
            
            # Upper half priority (ghost spawns there)
            if r < mid_row: score *= 4.0
            # Dead-end priority (ghost hides there)
            if (r, c) in dead_ends: score *= 8.0
            elif (r, c) in corners: score *= 5.0
            
            if score > best_score:
                best_score = score
                best = (r, c)
    
    return best

# ============================================================
# PacmanAgent: Reactive chaser
# ============================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "ReactiveChaser"
        
        self.internal_map = None
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.last_move = None
        
        # Map analysis
        self._dead_ends = set()
        self._corners = set()
        self._exits = {}
        self._map_analyzed = False
        
        # Ghost position tracking
        self._ghost_history = []  # last N positions

    # ---------- step ----------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_map_memory(map_state)
        
        if not self._map_analyzed and self.internal_map is not None:
            try:
                self._dead_ends, self._corners, self._exits = _analyze_map(self.internal_map)
                self._map_analyzed = True
            except: pass
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
            self._ghost_history.append(enemy_position)
            if len(self._ghost_history) > 5:
                self._ghost_history.pop(0)
        else:
            self.steps_since_seen += 1

        my_pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos = (int(enemy_position[0]), int(enemy_position[1])) if enemy_position is not None else None

        chosen_move = Move.STAY
        path = None

        if enemy_pos is not None:
            # Ghost visible: chase with interception
            dist = _manhattan(my_pos, enemy_pos)
            
            if dist <= 3:
                # Close range: rush directly toward ghost with A* or greedy
                path = _astar(my_pos, enemy_pos, self.internal_map)
                if path: chosen_move = path[0]
                else: chosen_move = _greedy_toward(my_pos, enemy_pos, self.internal_map)
            else:
                # Predict where ghost will go and intercept
                target = _intercept_target(my_pos, enemy_pos, self.internal_map,
                                           self._dead_ends, self._corners, self._exits)
                path = _astar(my_pos, target, self.internal_map)
                if path: chosen_move = path[0]
                else: chosen_move = _greedy_toward(my_pos, target, self.internal_map)
        
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 15:
            # Recently lost sight: go to last known position
            target = self.last_known_enemy_pos
            # If ghost was near a dead-end, check that dead-end
            for de in self._dead_ends:
                if _bfs_dist(target, de, self.internal_map) <= 3:
                    target = de
                    break
            path = _astar(my_pos, target, self.internal_map)
            if path: chosen_move = path[0]
            else: chosen_move = _greedy_toward(my_pos, target, self.internal_map)
        
        else:
            # Ghost hidden: systematic search
            target = _find_search_target(my_pos, self.internal_map, self._dead_ends, self._corners)
            if target is not None:
                path = _astar(my_pos, target, self.internal_map)
                if path: chosen_move = path[0]
            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_pos)

        # Speed-2 whenever possible
        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            dr, dc = chosen_move.value
            # Check if we can move 2 steps
            nr1, nc1 = my_pos[0] + dr, my_pos[1] + dc
            nr2, nc2 = my_pos[0] + dr * 2, my_pos[1] + dc * 2
            if _is_valid((nr1, nc1), self.internal_map) and _is_valid((nr2, nc2), self.internal_map):
                # Don't overshoot if the path turns at step 2
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
