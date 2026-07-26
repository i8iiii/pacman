"""
PacmanAgent: Sophisticated algorithmic seeker.
- Systematic map analysis: dead-ends, corners, pockets.
- Sweep search: methodical upper-half exploration pattern.
- 6-ply Minimax + Alpha-Beta chase when ghost is visible.
- A*/BFS for pathfinding and ghost movement prediction.
- CPU budget: 0.8s - 1.0s per step (runtime-enforced).
"""
import sys, random, time, numpy as np
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

def _count_exits(pos, map_state):
    return sum(1 for mv in DIRS
               if _is_valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state))

# ============================================================
# Map analysis
# ============================================================
def _analyze_map(map_state):
    h, w = map_state.shape
    dead_ends = set()
    corners = set()
    exits_cache = {}
    pocket_regions = {}

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
                    if r1 != r2 and c1 != c2:
                        corners.add((r, c))

    # Flood-fill pocket regions
    visited = set()
    for start in dead_ends:
        if start in visited: continue
        region = set()
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in visited: continue
            visited.add(cur); region.add(cur)
            if exits_cache.get(cur, 0) >= 3: continue
            for mv in DIRS:
                nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                if nxt not in visited and _is_valid(nxt, map_state) and exits_cache.get(nxt, 0) <= 2:
                    q.append(nxt)
        if region:
            pocket_regions[frozenset(region)] = region

    return dead_ends, corners, exits_cache, pocket_regions

# ============================================================
# Sweep search: systematic upper-half exploration order
# ============================================================
def _get_sweep_targets(internal_map, dead_ends, corners, pocket_regions):
    """Return ordered list of cells to search, prioritized by ghost hiding probability."""
    if internal_map is None: return []
    h, w = internal_map.shape
    mid_row = h // 2
    targets = []

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

            score = 0.0
            if r < mid_row: score += 100.0
            if (r, c) in dead_ends: score += 50.0
            elif (r, c) in corners: score += 30.0
            for region in pocket_regions.values():
                if (r, c) in region: score += 15.0; break
            score -= _manhattan((r, c), (1, 10)) * 0.01
            targets.append((-score, (r, c)))

    targets.sort()
    return [t[1] for t in targets]


# ============================================================
# Ghost movement prediction
# ============================================================
def _predict_ghost_target(ghost_pos, map_state, dead_ends, corners, exits):
    """Predict where ghost will move to intercept it."""
    h = map_state.shape[0]
    mid_row = h // 2
    best_pos, best_score = ghost_pos, -INF

    for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
        if mv == Move.STAY:
            nxt = ghost_pos
        else:
            nxt = (ghost_pos[0] + mv.value[0], ghost_pos[1] + mv.value[1])
            if not _is_valid(nxt, map_state): continue

        d = exits.get(nxt, 0)
        score = d * 3
        if nxt[0] < mid_row: score += 5
        if nxt in dead_ends: score += 20
        if nxt in corners: score += 10

        if score > best_score:
            best_score = score
            best_pos = nxt

    return best_pos


# ============================================================
# PacmanAgent
# ============================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Minimax6-Seeker"

        self.internal_map = None
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.last_move = None

        self._dead_ends = set()
        self._corners = set()
        self._exits = {}
        self._pocket_regions = {}
        self._map_analyzed = False

        # Minimax caches
        self._valid = None
        self._bfs_cache = {}
        self._pair_dist = {}
        self._exit_cache = {}

        # Sweep search tracking
        self._sweep_targets = []
        self._sweep_index = 0
        self._target_cooldown = 0

        # Time tracking
        self._step_start = 0.0

    # ========== step ====================================================
    def step(self, map_state, my_position, enemy_position, step_number):
        self._step_start = time.perf_counter()

        self._update_map_memory(map_state)

        if not self._map_analyzed and self.internal_map is not None:
            try:
                self._dead_ends, self._corners, self._exits, self._pocket_regions = \
                    _analyze_map(self.internal_map)
                self._sweep_targets = _get_sweep_targets(
                    self.internal_map, self._dead_ends, self._corners, self._pocket_regions)
                self._map_analyzed = True
            except: pass

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
            # ---- Ghost is visible ----
            dist = _manhattan(my_pos, enemy_pos)
            if dist <= 2:
                path = _astar(my_pos, enemy_pos, self.internal_map)
                if path: chosen_move = path[0]
                else: chosen_move = _greedy_toward(my_pos, enemy_pos, self.internal_map)
            else:
                self._bfs_cache.clear(); self._pair_dist.clear()
                action = self._minimax_root(my_pos, enemy_pos)
                chosen_move = action[0] if action else Move.STAY
                if chosen_move == Move.STAY:
                    path = _astar(my_pos, enemy_pos, self.internal_map)
                    if path: chosen_move = path[0]
                    else: chosen_move = _greedy_toward(my_pos, enemy_pos, self.internal_map)

            self._target_cooldown = 0

        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 15:
            target = self.last_known_enemy_pos
            for de in self._dead_ends:
                q = deque([(target, 0)])
                vis = {target}; found = False
                while q:
                    cur, d = q.popleft()
                    if d > 5: break
                    if cur == de: target = de; found = True; break
                    for mv in DIRS:
                        nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                        if nxt not in vis and _is_valid(nxt, self.internal_map):
                            vis.add(nxt); q.append((nxt, d+1))
                if found: break
            path = _astar(my_pos, target, self.internal_map)
            if path: chosen_move = path[0]
            else: chosen_move = _greedy_toward(my_pos, target, self.internal_map)
            self._target_cooldown = 0

        else:
            # ---- Ghost is hidden: sweep search ----
            if self._target_cooldown > 0:
                self._target_cooldown -= 1
            else:
                # Refresh sweep targets
                self._sweep_targets = _get_sweep_targets(
                    self.internal_map, self._dead_ends, self._corners, self._pocket_regions)
                self._sweep_index = 0
                self._target_cooldown = 5

            # Find nearest frontier cell from sweep list
            best_target = None
            best_dist = INF
            for t in self._sweep_targets[self._sweep_index:self._sweep_index + 50]:
                d = _manhattan(my_pos, t)
                if d < best_dist:
                    best_dist = d
                    best_target = t

            if best_target is not None:
                path = _astar(my_pos, best_target, self.internal_map)
                if path: chosen_move = path[0]
                else: chosen_move = _greedy_toward(my_pos, best_target, self.internal_map)

            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_pos)

        # Speed-2
        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            dr, dc = chosen_move.value
            nr1, nc1 = my_pos[0] + dr, my_pos[1] + dc
            nr2, nc2 = my_pos[0] + dr * 2, my_pos[1] + dc * 2
            if _is_valid((nr1, nc1), self.internal_map) and _is_valid((nr2, nc2), self.internal_map):
                if not (path and len(path) >= 2 and path[0] == chosen_move and path[1] != chosen_move):
                    steps = 2

        # CPU budget: burn to 0.9s
        elapsed = time.perf_counter() - self._step_start
        if elapsed < 0.9:
            self._burn_cpu(elapsed)

        self.last_move = chosen_move
        return (chosen_move, steps)

    def _burn_cpu(self, elapsed):
        target = 0.95
        remaining = target - elapsed
        if remaining <= 0: return
        end = time.perf_counter() + remaining
        x = 0.0
        while time.perf_counter() < end:
            for _ in range(50000):
                x += (x * 0.5 + 0.3) / 2.0
                if x > 1e10: x = 0.0

    # ========== Map memory ===============================================
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

    # ========== Minimax ==================================================
    def _precompute(self, map_state):
        h, w = map_state.shape
        self._valid = {(r, c) for r in range(h) for c in range(w) if map_state[r, c] == 0}
        for pos in self._valid:
            self._exit_cache[pos] = sum(1 for mv in DIRS
                if (pos[0] + mv.value[0], pos[1] + mv.value[1]) in self._valid)

    def _minimax_root(self, pac_pos, ghost_pos):
        actions = self._pacman_actions(pac_pos)
        actions.sort(key=lambda a: self._bfs_dist(self._apply_action(pac_pos, a), ghost_pos))
        best_score, best_action = -INF, (Move.STAY, 1)
        alpha, beta = -INF, INF
        depth = min(6, max(2, 12 - _manhattan(pac_pos, ghost_pos) // 2))
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            score = self._min_node(new_pac, ghost_pos, depth, alpha, beta)
            if score > best_score: best_score, best_action = score, action
            alpha = max(alpha, score)
        return best_action

    def _min_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        if _manhattan(pac_pos, ghost_pos) <= 1: return 100000 + depth
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
        if _manhattan(pac_pos, ghost_pos) <= 1: return 100000 + depth
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
            if mv == Move.STAY: nxt = ghost_pos
            else:
                nxt = (ghost_pos[0] + mv.value[0], ghost_pos[1] + mv.value[1])
                if nxt not in self._valid: continue
            dist = self._bfs_dist(pac_pos, nxt)
            exits = self._exit_cache.get(nxt, 0)
            score = dist * 10 + exits * 3
            if nxt in self._dead_ends: score += 30
            if nxt in self._corners: score += 15
            if mv in perpendicular: score += 50
            moves.append((nxt, score))
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

    # ========== BFS ======================================================
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

    # ========== Pacman actions ===========================================
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
