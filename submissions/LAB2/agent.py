"""
PacmanAgent: Pure algorithmic seeker (No ML).
- Map analysis: dead-ends, corners, pockets.
- Probabilistic upper-half search when ghost hidden.
- 6-ply Minimax + Alpha-Beta chase when ghost visible.
- A* fallback for close-range / pathfinding.
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
INF = 10 ** 9

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

# ============================================================
# Map analysis: dead-ends, corners, pockets
# ============================================================
def _analyze_map(map_state):
    h, w = map_state.shape
    exits = {}
    dead_ends = set()
    corners = set()

    for r in range(h):
        for c in range(w):
            if map_state[r, c] != 0: continue
            e = 0
            for mv in DIRS:
                nr, nc = r + mv.value[0], c + mv.value[1]
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0:
                    e += 1
            exits[(r, c)] = e
            if e == 1:
                dead_ends.add((r, c))
            elif e == 2:
                neighbors = []
                for mv in DIRS:
                    nr, nc = r + mv.value[0], c + mv.value[1]
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0:
                        neighbors.append((nr, nc))
                if len(neighbors) == 2:
                    r1, c1 = neighbors[0]
                    r2, c2 = neighbors[1]
                    if r1 != r2 and c1 != c2:  # perpendicular = corner
                        corners.add((r, c))

    # Flood-fill pocket regions from dead-ends
    pocket_regions = {}
    visited = set()
    for start in dead_ends:
        if start in visited: continue
        region = set()
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in visited: continue
            visited.add(cur)
            region.add(cur)
            if exits.get(cur, 0) >= 3: continue  # stop at junctions
            for mv in DIRS:
                nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                if nxt not in visited and _is_valid(nxt, map_state) and exits.get(nxt, 0) <= 2:
                    q.append(nxt)
        pocket_regions[frozenset(region)] = region

    return exits, dead_ends, corners, pocket_regions

# ============================================================
# Probabilistic search: upper half + dead-ends + corners
# ============================================================
def _find_best_search_target(my_pos, internal_map, dead_ends, corners, pocket_regions):
    if internal_map is None: return None
    h, w = internal_map.shape
    mid_row = h // 2
    best, best_score = None, -1.0

    for r in range(h):
        for c in range(w):
            if internal_map[r, c] != 0: continue
            # Must border fog (information gain)
            has_unknown = False
            for mv in DIRS:
                nr, nc = r + mv.value[0], c + mv.value[1]
                if 0 <= nr < h and 0 <= nc < w and internal_map[nr, nc] == -1:
                    has_unknown = True
                    break
            if not has_unknown: continue

            dist = _manhattan(my_pos, (r, c))
            if dist == 0: continue
            score = 1.0 / dist

            # Upper half bonus (3x)
            if r < mid_row: score *= 3.0
            # Dead-end bonus (5x) — ghost hides here
            if (r, c) in dead_ends: score *= 5.0
            # Corner bonus (4x) — likely ambush
            elif (r, c) in corners: score *= 4.0
            # Pocket region bonus (1.5x)
            for region in pocket_regions.values():
                if (r, c) in region: score *= 1.5; break

            if score > best_score: best_score, best = score, (r, c)

    return best

# ============================================================
# PacmanAgent
# ============================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "AlgoSeeker-6ply"

        self.internal_map = None
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.last_move = None

        # Minimax caches
        self._valid = None
        self._bfs_cache = {}
        self._pair_dist = {}
        self._exit_cache = {}

        # Map analysis cache
        self._dead_ends = set()
        self._corners = set()
        self._pocket_regions = {}
        self._map_analyzed = False

    # ---------- step ----------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_map_memory(map_state)

        if not self._map_analyzed and self.internal_map is not None:
            try:
                _, self._dead_ends, self._corners, self._pocket_regions = _analyze_map(self.internal_map)
                self._map_analyzed = True
            except:
                pass

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        my_pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos = (int(enemy_position[0]), int(enemy_position[1])) if enemy_position is not None else None

        if self._valid is None and self.internal_map is not None:
            self._precompute(self.internal_map)

        chosen_move = Move.STAY
        path = None

        if enemy_pos is not None:
            # Ghost visible: 6-ply Minimax + Alpha-Beta
            if _manhattan(my_pos, enemy_pos) < 2:
                chosen_move = Move.STAY
            else:
                self._bfs_cache.clear(); self._pair_dist.clear()
                action = self._minimax_root(my_pos, enemy_pos)
                chosen_move = action[0] if action else Move.STAY
                if chosen_move == Move.STAY:
                    path = _astar(my_pos, enemy_pos, self.internal_map)
                    if path: chosen_move = path[0]
                    else: chosen_move = _greedy_toward(my_pos, enemy_pos, self.internal_map)
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 10:
            path = _astar(my_pos, self.last_known_enemy_pos, self.internal_map)
            if path: chosen_move = path[0]
            else: chosen_move = _greedy_toward(my_pos, self.last_known_enemy_pos, self.internal_map)
        else:
            # Ghost hidden: probabilistic search
            target = _find_best_search_target(my_pos, self.internal_map,
                                               self._dead_ends, self._corners, self._pocket_regions)
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

    # ---------- Minimax (6-ply + Alpha-Beta) ----------------------------
    def _precompute(self, map_state):
        h, w = map_state.shape
        self._valid = {(r, c) for r in range(h) for c in range(w) if map_state[r, c] == 0}
        for pos in self._valid:
            self._exit_cache[pos] = sum(1 for mv in DIRS if (pos[0] + mv.value[0], pos[1] + mv.value[1]) in self._valid)

    def _minimax_root(self, pac_pos, ghost_pos):
        actions = self._pacman_actions(pac_pos)
        actions.sort(key=lambda a: self._bfs_dist(self._apply_action(pac_pos, a), ghost_pos))
        best_score, best_action = -INF, (Move.STAY, 1)
        alpha, beta = -INF, INF
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            score = self._min_node(new_pac, ghost_pos, 6, alpha, beta)
            if score > best_score: best_score, best_action = score, action
            alpha = max(alpha, score)
        return best_action

    def _min_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        if _manhattan(pac_pos, ghost_pos) < 2: return 100000 + depth
        if depth == 0: return self._evaluate(pac_pos, ghost_pos)
        ghost_moves = self._scored_ghost_moves(pac_pos, ghost_pos)
        best = INF
        for new_ghost, _ in ghost_moves:
            val = self._max_node(pac_pos, new_ghost, depth - 1, alpha, beta)
            if val < best: best = val
            if best <= alpha: return best
            beta = min(beta, best)
        return best if best != INF else self._evaluate(pac_pos, ghost_pos)

    def _max_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        if _manhattan(pac_pos, ghost_pos) < 2: return 100000 + depth
        if depth == 0: return self._evaluate(pac_pos, ghost_pos)
        actions = self._pacman_actions(pac_pos)
        if not actions: return self._evaluate(pac_pos, ghost_pos)
        actions.sort(key=lambda a: self._bfs_dist(self._apply_action(pac_pos, a), ghost_pos))
        best = -INF
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            val = self._min_node(new_pac, ghost_pos, depth - 1, alpha, beta)
            if val > best: best = val
            if best >= beta: return best
            alpha = max(alpha, best)
        return best

    def _scored_ghost_moves(self, pac_pos, ghost_pos):
        moves = []
        aligned = self._aligned_with_pacman(ghost_pos, pac_pos)
        perpendicular = self._perpendicular_to(ghost_pos, pac_pos) if aligned else set()
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
            new_pos = ghost_pos if mv == Move.STAY else (ghost_pos[0] + mv.value[0], ghost_pos[1] + mv.value[1])
            if mv != Move.STAY and new_pos not in self._valid: continue
            dist = self._bfs_dist(pac_pos, new_pos)
            exits = self._exit_cache.get(new_pos, 0)
            score = dist * 10 + exits * 3
            if new_pos in self._dead_ends: score += 20
            if new_pos in self._corners: score += 10
            if mv in perpendicular: score += 30
            moves.append((new_pos, score))
        moves.sort(key=lambda x: x[1], reverse=True)
        return moves

    def _aligned_with_pacman(self, ghost_pos, pac_pos):
        if ghost_pos[0] == pac_pos[0]:
            left, right = sorted((ghost_pos[1], pac_pos[1]))
            return all((ghost_pos[0], c) in self._valid for c in range(left + 1, right))
        if ghost_pos[1] == pac_pos[1]:
            top, bottom = sorted((ghost_pos[0], pac_pos[0]))
            return all((r, ghost_pos[1]) in self._valid for r in range(top + 1, bottom))
        return False

    def _perpendicular_to(self, ghost_pos, pac_pos):
        if ghost_pos[0] == pac_pos[0]: return {Move.UP, Move.DOWN}
        return {Move.LEFT, Move.RIGHT}

    def _evaluate(self, pac_pos, ghost_pos):
        dist = self._bfs_dist(pac_pos, ghost_pos)
        exits = self._exit_cache.get(ghost_pos, 0)
        return -(dist * 10 + exits * 3)

    # ---------- BFS -----------------------------------------------------
    def _bfs_dist(self, a, b):
        if a == b: return 0
        if a not in self._valid or b not in self._valid: return INF
        key = (a, b)
        if key not in self._pair_dist:
            self._pair_dist[key] = self._bfs_compute(a).get(b, INF)
        return self._pair_dist[key]

    def _bfs_compute(self, start):
        if start not in self._bfs_cache:
            dist = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for mv in DIRS:
                    nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                    if nxt in self._valid and nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            self._bfs_cache[start] = dist
        return self._bfs_cache[start]

    # ---------- Pacman actions ------------------------------------------
    def _pacman_actions(self, pos):
        actions = []
        for mv in DIRS:
            r, c = pos
            valid_steps = 0
            for _ in range(self.pacman_speed):
                r += mv.value[0]; c += mv.value[1]
                if (r, c) not in self._valid: break
                valid_steps += 1
            for s in range(1, valid_steps + 1):
                actions.append((mv, s))
        return actions if actions else [(Move.STAY, 1)]

    def _apply_action(self, pos, action):
        move, steps = action
        if move == Move.STAY: return pos
        r, c = pos
        for _ in range(steps):
            nr, nc = r + move.value[0], c + move.value[1]
            if (nr, nc) not in self._valid: break
            r, c = nr, nc
        return (r, c)
