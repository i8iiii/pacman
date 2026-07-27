
"""
PacmanAgent: Clean modular rewrite.
- MapMemory: persistent internal map with cross-match caching
- MapAnalyzer: dead-ends, corners, pockets, exit counts
- PathFinder: A* pathfinding + BFS distance caching
- GhostProbability: weighted probability over ghost hiding spots
- MinimaxEngine: 6-ply alpha-beta adversarial search
- SweepPlanner: systematic upper-half exploration
"""
import sys
import random
import numpy as np
from collections import deque
from pathlib import Path
from heapq import heappush, heappop

src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from hide_agent.controller import HideController

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
DIRS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
INF = 10 ** 9

# ---------------------------------------------------------------------------
# Persistent map cache (survives agent re-instantiation across matches)
# ---------------------------------------------------------------------------
_MAP_CACHE: dict[int, "np.ndarray"] = {}


def _fingerprint(map_state: "np.ndarray") -> int:
    """Hash the wall pattern. Walls are always visible (value=1) even
    with fog of war, so this fingerprint is stable across observations."""
    flat = map_state.ravel()
    wall_positions = tuple(i for i, v in enumerate(flat) if v == 1)
    return hash(wall_positions)


# ---------------------------------------------------------------------------
# Pure helpers (no state, no class)
# ---------------------------------------------------------------------------
def _manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_valid(pos: tuple, map_state: "np.ndarray") -> bool:
    """Check if position is within bounds and not a wall (value != 1)."""
    r, c = pos
    if map_state is None:
        return False
    h, w = map_state.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return map_state[r, c] != 1


def _count_exits(pos: tuple, map_state: "np.ndarray") -> int:
    return sum(
        1 for mv in DIRS
        if _is_valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state)
    )


# ============================================================
# MapMemory -- persistent internal map, cached across matches
# ============================================================
class MapMemory:
    """Maintains the growing internal map.

    On first update: fingerprints the wall pattern and checks _MAP_CACHE.
    If this map was seen in a previous match, restores the full discovered
    layout. On subsequent matches, Pacman starts with full map knowledge.
    """

    def __init__(self):
        self._internal_map: "np.ndarray | None" = None
        self._fingerprint: "int | None" = None
        self._started = False

    def update(self, map_state: "np.ndarray") -> bool:
        """Merge the current observation into the internal map.

        Returns True if any previously-unknown cells were discovered this step.
        """
        if not self._started:
            self._started = True
            fp = _fingerprint(map_state)
            self._fingerprint = fp
            if fp in _MAP_CACHE:
                self._internal_map = _MAP_CACHE[fp].copy()
            else:
                self._internal_map = np.full_like(map_state, -1, dtype=np.int8)
                self._internal_map[map_state == 1] = 1

        known_before = int((self._internal_map == 0).sum())
        visible_mask = map_state != -1
        self._internal_map[visible_mask] = map_state[visible_mask].astype(np.int8)
        # Normalize agent markers (2,3) back to empty (0)
        self._internal_map[(self._internal_map == 2) | (self._internal_map == 3)] = 0
        known_after = int((self._internal_map == 0).sum())

        return known_after > known_before

    def save_to_cache(self):
        """Persist the current internal map to the module-level cache."""
        if self._fingerprint is not None and self._internal_map is not None:
            _MAP_CACHE[self._fingerprint] = self._internal_map.copy()

    def get_map(self) -> "np.ndarray":
        return self._internal_map

    def is_initialized(self) -> bool:
        return self._started


# ============================================================
# MapAnalyzer -- structural map analysis
# ============================================================
class MapAnalyzer:
    """Analyzes the known map to find dead ends, corners, and pocket regions.

    Results are cached. Re-analyze when new cells are discovered.
    """

    def __init__(self):
        self._cached: "dict | None" = None

    def analyze(self, internal_map: "np.ndarray") -> dict:
        """Run full structural analysis. Returns dict with keys:
        dead_ends, corners, pockets, exit_counts, mid_row.
        """
        h, w = internal_map.shape
        dead_ends: set = set()
        corners: set = set()
        exit_counts: dict = {}
        pocket_regions: dict = {}

        # Pass 1: count exits and classify
        for r in range(h):
            for c in range(w):
                if internal_map[r, c] != 0:
                    continue
                e = _count_exits((r, c), internal_map)
                exit_counts[(r, c)] = e
                if e == 1:
                    dead_ends.add((r, c))
                elif e == 2:
                    neighbors = [
                        (r + mv.value[0], c + mv.value[1]) for mv in DIRS
                        if _is_valid((r + mv.value[0], c + mv.value[1]), internal_map)
                    ]
                    if len(neighbors) == 2:
                        r1, c1 = neighbors[0]
                        r2, c2 = neighbors[1]
                        # Perpendicular (L-shaped), not opposite (straight through)
                        if r1 != r2 and c1 != c2:
                            corners.add((r, c))

        # Pass 2: flood-fill pocket regions from dead ends
        visited: set = set()
        for start in dead_ends:
            if start in visited:
                continue
            region: set = set()
            q = deque([start])
            while q:
                cur = q.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                region.add(cur)
                if exit_counts.get(cur, 0) >= 3:
                    continue
                for mv in DIRS:
                    nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                    if nxt not in visited and _is_valid(nxt, internal_map) and exit_counts.get(nxt, 0) <= 2:
                        q.append(nxt)
            if region:
                pocket_regions[len(pocket_regions)] = region

        self._cached = {
            "dead_ends": dead_ends,
            "corners": corners,
            "pockets": pocket_regions,
            "exit_counts": exit_counts,
            "mid_row": h // 2,
        }
        return self._cached

    def get_analysis(self) -> "dict | None":
        return self._cached


# ============================================================
# PathFinder -- A* pathfinding + BFS distance caching
# ============================================================
class PathFinder:
    """A* for pathfinding, BFS for exact distance computations.

    BFS distances are cached per source within a single step (the cache
    is cleared between steps by the orchestrator).
    """

    def __init__(self, map_getter):
        """map_getter: callable that returns the current known np.ndarray map."""
        self._map_getter = map_getter
        self._bfs_source_cache: dict = {}
        self._pair_cache: dict = {}

    @property
    def _map(self) -> "np.ndarray":
        return self._map_getter()

    # ---- A* ----------------------------------------------------------------
    def astar(self, start: tuple, goal: tuple) -> "list | None":
        """Return list of Move from start to goal, or None if unreachable."""
        if start == goal:
            return []
        m = self._map
        open_set = [(0, 0, start, [])]
        g_score = {start: 0}
        closed: set = set()
        counter = 0
        while open_set:
            _, _, current, path = heappop(open_set)
            if current in closed:
                continue
            closed.add(current)
            if current == goal:
                return path
            for mv in DIRS:
                nxt = (current[0] + mv.value[0], current[1] + mv.value[1])
                if nxt in closed or not _is_valid(nxt, m):
                    continue
                tg = g_score[current] + 1
                if tg < g_score.get(nxt, float("inf")):
                    g_score[nxt] = tg
                    h = _manhattan(nxt, goal)
                    counter += 1
                    heappush(open_set, (tg + h, counter, nxt, path + [mv]))
        return None

    # ---- BFS pair distance ------------------------------------------------
    def bfs_dist(self, a: tuple, b: tuple) -> int:
        """Shortest-path distance from a to b on the known map.
        Returns INF if unreachable."""
        if a == b:
            return 0
        m = self._map
        if not _is_valid(a, m) or not _is_valid(b, m):
            return INF
        key = (a, b)
        if key not in self._pair_cache:
            all_dists = self.bfs_all_dists(a)
            self._pair_cache[key] = all_dists.get(b, INF)
        return self._pair_cache[key]

    def bfs_all_dists(self, source: tuple) -> dict:
        """BFS from source to all reachable known cells. Cached per source."""
        if source not in self._bfs_source_cache:
            m = self._map
            dist = {source: 0}
            q = deque([source])
            while q:
                cur = q.popleft()
                for mv in DIRS:
                    nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                    if _is_valid(nxt, m) and nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            self._bfs_source_cache[source] = dist
        return self._bfs_source_cache[source]

    def clear_caches(self):
        """Clear per-step BFS caches. Call at the start of each step()."""
        self._bfs_source_cache.clear()
        self._pair_cache.clear()

    @staticmethod
    def manhattan(a: tuple, b: tuple) -> int:
        return _manhattan(a, b)


# ============================================================
# GhostProbability -- weighted distribution over hiding spots
# ============================================================
class GhostProbability:
    """Estimates where the ghost is likely hiding when out of sight.

    Weights: dead ends > corners > upper-half > pockets > distance.
    """

    def __init__(self, analyzer: MapAnalyzer, pathfinder: PathFinder):
        self._analyzer = analyzer
        self._pathfinder = pathfinder

    def compute(self, pacman_pos: tuple) -> list:
        """Return list of (row, col) sorted by descending ghost probability."""
        analysis = self._analyzer.get_analysis()
        if analysis is None:
            return []

        dead_ends = analysis["dead_ends"]
        corners = analysis["corners"]
        pockets = analysis["pockets"]
        mid_row = analysis["mid_row"]
        exit_counts = analysis["exit_counts"]

        # Build pocket membership lookup
        pocket_of: dict = {}
        for pid, region in pockets.items():
            for cell in region:
                pocket_of[cell] = pid

        # Score every known empty cell
        scored: list = []
        for pos, _ in exit_counts.items():
            score = 1.0

            if pos[0] < mid_row:
                score *= 3.0
            if pos in dead_ends:
                score *= 5.0
            if pos in corners:
                score *= 3.0
            if pos in pocket_of:
                score *= 2.0

            # Slight inverse distance: ghost unlikely to be right next to Pacman
            dist = _manhattan(pacman_pos, pos)
            score *= 1.0 + 1.0 / max(1, dist)

            scored.append((-score, pos))

        scored.sort()
        return [pos for _, pos in scored]


# ============================================================
# MinimaxEngine -- 6-ply alpha-beta adversarial search
# ============================================================
class MinimaxEngine:
    """6-ply minimax with alpha-beta pruning.

    Depth = 6 plies (3 Pacman moves + 3 Ghost moves).
    Pacman is the maximizer (wants capture), Ghost is the minimizer (wants to evade).
    """

    def __init__(self, analyzer: MapAnalyzer, pathfinder: PathFinder, pacman_speed: int):
        self._analyzer = analyzer
        self._pf = pathfinder
        self._pacman_speed = max(1, int(pacman_speed))
        self._depth = 6

    def search(self, pac_pos: tuple, ghost_pos: tuple) -> tuple:
        """Return best Pacman action as (Move, steps)."""
        actions = self._pacman_actions(pac_pos)
        if not actions:
            return (Move.STAY, 1)

        # Sort by distance to ghost for better pruning
        actions.sort(key=lambda a: self._pf.bfs_dist(
            self._apply_action(pac_pos, a), ghost_pos))

        best_score = -INF
        best_action = (Move.STAY, 1)
        alpha, beta = -INF, INF

        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            score = self._min_node(new_pac, ghost_pos, self._depth, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)

        return best_action

    # ---- Max node (Pacman's turn) ----------------------------------------
    def _max_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        if _manhattan(pac_pos, ghost_pos) <= 1:
            return 100000 + depth
        if depth == 0:
            return self._evaluate(pac_pos, ghost_pos)

        actions = self._pacman_actions(pac_pos)
        if not actions:
            return self._evaluate(pac_pos, ghost_pos)

        actions.sort(key=lambda a: self._pf.bfs_dist(
            self._apply_action(pac_pos, a), ghost_pos))

        best = -INF
        for action in actions:
            new_pac = self._apply_action(pac_pos, action)
            val = self._min_node(new_pac, ghost_pos, depth - 1, alpha, beta)
            if val > best:
                best = val
            if best >= beta:
                return best
            alpha = max(alpha, best)
        return best

    # ---- Min node (Ghost's turn) -----------------------------------------
    def _min_node(self, pac_pos, ghost_pos, depth, alpha, beta):
        if _manhattan(pac_pos, ghost_pos) <= 1:
            return 100000 + depth
        if depth == 0:
            return self._evaluate(pac_pos, ghost_pos)

        ghost_moves = self._scored_ghost_moves(pac_pos, ghost_pos)
        if not ghost_moves:
            return self._evaluate(pac_pos, ghost_pos)

        best = INF
        for new_ghost, _ in ghost_moves:
            val = self._max_node(pac_pos, new_ghost, depth - 1, alpha, beta)
            if val < best:
                best = val
            if best <= alpha:
                return best
            beta = min(beta, best)
        return best

    # ---- Evaluation ------------------------------------------------------
    def _evaluate(self, pac_pos, ghost_pos):
        analysis = self._analyzer.get_analysis()
        exit_counts = analysis["exit_counts"] if analysis else {}
        dist = self._pf.bfs_dist(pac_pos, ghost_pos)
        ghost_exits = exit_counts.get(ghost_pos, 0)

        # Turn-distance with speed: ceil(bfs / speed)
        effective = (dist + self._pacman_speed - 1) // self._pacman_speed
        return -(effective * 10 + ghost_exits * 3)

    # ---- Ghost move scoring ----------------------------------------------
    def _scored_ghost_moves(self, pac_pos, ghost_pos):
        """Simulate ghost's best evasive move.
        Ghost maximizes: distance from Pacman + good exits + avoiding traps."""
        analysis = self._analyzer.get_analysis()
        dead_ends = analysis["dead_ends"] if analysis else set()
        corners = analysis["corners"] if analysis else set()
        exit_counts = analysis["exit_counts"] if analysis else {}
        pockets = analysis["pockets"] if analysis else {}

        # Build pocket membership
        pocket_of = {}
        for pid, region in pockets.items():
            for cell in region:
                pocket_of[cell] = pid

        # Calculate Pacman's best reach after 1 step (for look-ahead evasion)
        pac_actions = self._pacman_actions(pac_pos)
        pac_reach = {pac_pos}
        for action in pac_actions:
            pac_reach.add(self._apply_action(pac_pos, action))

        aligned = self._aligned_with_pacman(ghost_pos, pac_pos)
        perpendicular = self._perpendicular_to(ghost_pos, pac_pos) if aligned else set()

        moves = []
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
            if mv == Move.STAY:
                nxt = ghost_pos
            else:
                nxt = (ghost_pos[0] + mv.value[0], ghost_pos[1] + mv.value[1])
                if not _is_valid(nxt, self._pf._map):
                    continue

            # Base: maximize BFS distance from Pacman
            dist = self._pf.bfs_dist(pac_pos, nxt)
            exits = exit_counts.get(nxt, 0)

            # Look-ahead: worst-case distance if Pacman moves optimally
            min_dist = INF
            for r in pac_reach:
                d = self._pf.bfs_dist(r, nxt)
                if d < min_dist:
                    min_dist = d

            score = dist * 8 + min_dist * 4 + exits * 3

            # Bonuses for hiding spots
            if nxt in dead_ends:
                # Dead end is good ONLY if Pacman can't reach it quickly
                if min_dist > 3:
                    score += 40
                else:
                    score -= 20  # trap! avoid backing into a dead end
            if nxt in corners:
                score += 15
            if nxt in pocket_of:
                score += 10
            if mv in perpendicular:
                score += 50

            moves.append((nxt, score))

        moves.sort(key=lambda x: x[1], reverse=True)
        return moves

    def _aligned_with_pacman(self, ghost_pos, pac_pos):
        m = self._pf._map
        if ghost_pos[0] == pac_pos[0]:
            left, right = sorted((ghost_pos[1], pac_pos[1]))
            return all(_is_valid((ghost_pos[0], c), m) for c in range(left + 1, right))
        if ghost_pos[1] == pac_pos[1]:
            top, bottom = sorted((ghost_pos[0], pac_pos[0]))
            return all(_is_valid((r, ghost_pos[1]), m) for r in range(top + 1, bottom))
        return False

    def _perpendicular_to(self, ghost_pos, pac_pos):
        if ghost_pos[0] == pac_pos[0]:
            return {Move.UP, Move.DOWN}
        return {Move.LEFT, Move.RIGHT}

    # ---- Pacman action generation ----------------------------------------
    def _pacman_actions(self, pos):
        m = self._pf._map
        actions = []
        for mv in DIRS:
            r, c = pos
            valid_steps = 0
            for _ in range(self._pacman_speed):
                r += mv.value[0]
                c += mv.value[1]
                if not _is_valid((r, c), m):
                    break
                valid_steps += 1
            for s in range(1, valid_steps + 1):
                actions.append((mv, s))
        return actions if actions else [(Move.STAY, 1)]

    def _apply_action(self, pos, action):
        move, steps = action
        if move == Move.STAY:
            return pos
        m = self._pf._map
        r, c = pos
        for _ in range(steps):
            nr, nc = r + move.value[0], c + move.value[1]
            if not _is_valid((nr, nc), m):
                break
            r, c = nr, nc
        return (r, c)


# ============================================================
# SweepPlanner -- systematic exploration when ghost is hidden
# ============================================================
class SweepPlanner:
    """Plans systematic exploration using the ghost probability distribution.

    Picks the highest-probability unknown-adjacent cell, navigates to it
    via A*, and maintains a cooldown on recently visited cells.
    When the map is fully explored, cycles through hiding spots
    instead of random wandering.
    """

    def __init__(self, ghost_prob: GhostProbability, pathfinder: PathFinder, map_memory: MapMemory):
        self._ghost_prob = ghost_prob
        self._pf = pathfinder
        self._map_memory = map_memory
        self._recent_cooldown: list = []
        self._cooldown_size = 20
        self._current_path: "list | None" = None
        self._current_target: "tuple | None" = None
        self._patrol_candidates: list = []
        self._patrol_index = 0
        self._last_pacman_pos: tuple | None = None
        self._upper_only_steps = 0  # steps spent in upper-half-only mode

    def next_move(self, pacman_pos: tuple) -> Move:
        """Return the next move to systematically explore the map.

        Priority queue for upper-half hiding spots:
          1. Corners & pockets in the zone nearest Pacman (left/middle/right)
          2. Other corners & pockets in remaining upper-half zones
          3. Dead ends anywhere in upper half
          4. Lower-half cells (only after upper is exhausted)
        """
        self._last_pacman_pos = pacman_pos

        # If we have a current path, continue following it
        if self._current_path:
            move = self._current_path.pop(0)
            next_pos = (pacman_pos[0] + move.value[0], pacman_pos[1] + move.value[1])
            m = self._pf._map
            if _is_valid(next_pos, m):
                return move
            self._current_path = None
            self._current_target = None

        m = self._pf._map
        has_unknown = (m == -1).any()
        h, w = m.shape
        analysis = self._ghost_prob._analyzer.get_analysis()
        mid_row = analysis["mid_row"] if analysis else h // 2

        # Zone boundaries for upper half
        left_max = w // 3
        right_min = 2 * w // 3

        # Collect upper-half hiding spots by zone
        dead_ends = analysis["dead_ends"] if analysis else set()
        corners = analysis["corners"] if analysis else set()
        pockets = analysis["pockets"] if analysis else {}

        # Build pocket membership
        pocket_of = {}
        for pid, region in pockets.items():
            for cell in region:
                pocket_of[cell] = pid

        # Zone classification helper
        def _zone(cell):
            if cell[1] < left_max:
                return "left"
            elif cell[1] >= right_min:
                return "right"
            return "middle"

        # Build priority lists per zone (upper-half only, unvisited first)
        zones = {"left": [], "middle": [], "right": []}
        for cell in corners:
            if cell[0] < mid_row and cell not in self._recent_cooldown:
                zones[_zone(cell)].append(("corner", cell))
        for cell in pocket_of:
            if cell[0] < mid_row and cell not in self._recent_cooldown:
                if cell not in corners:
                    zones[_zone(cell)].append(("pocket", cell))
        for cell in dead_ends:
            if cell[0] < mid_row and cell not in self._recent_cooldown:
                if cell not in corners and cell not in pocket_of:
                    zones[_zone(cell)].append(("dead_end", cell))

        # ---- Hardcoded priority hiding spots (ghost's favorite corners) ----
        # These are the complex corner regions the ghost most frequently uses
        _priority_regions = [
            # Upper-left pocket: rows 1-5, cols 1-4
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 4), (3, 1), (3, 4),
            (4, 1), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4),
            # Middle pocket: rows 7-9, cols 4-8
            (7, 4), (7, 5), (7, 6),
            (8, 4), (8, 6),
            (9, 4), (9, 5), (9, 6), (9, 7), (9, 8),
            # Lower-left pocket: rows 11-19, cols 1-4
            (11, 1), (11, 2), (11, 3), (11, 4),
            (13, 1), (13, 4), (15, 1), (15, 4),
            (17, 1), (17, 4), (19, 1), (19, 2), (19, 3), (19, 4),
            # Lower corridor: rows 14-15, cols 4-14
            (14, 4), (14, 7), (14, 10), (14, 13), (14, 14),
            (15, 4), (15, 7), (15, 10), (15, 13), (15, 14),
            # Center pocket near row 5, col 10
            (5, 10),
        ]
        _priority_set = set(_priority_regions)

        # Try hardcoded priority spots first (nearest first)
        priority_sorted = sorted(
            [c for c in _priority_set if c not in self._recent_cooldown and c != pacman_pos],
            key=lambda c: _manhattan(pacman_pos, c)
        )
        for cell in priority_sorted:
            if has_unknown:
                r, c = cell
                if not any(0 <= r + mv.value[0] < h and 0 <= c + mv.value[1] < w
                           and m[r + mv.value[0], c + mv.value[1]] == -1
                           for mv in DIRS):
                    continue
            path = self._pf.astar(pacman_pos, cell)
            if path:
                self._current_target = cell
                self._current_path = list(path)
                self._add_cooldown(cell)
                return self._current_path.pop(0)

        # Determine which zone Pacman is in (or nearest to)
        pac_zone = _zone(pacman_pos)
        zone_order = [pac_zone] + [z for z in ["left", "middle", "right"] if z != pac_zone]

        # Helper: get freshness score
        def _freshness(cell):
            r, c = cell
            fresh = 0
            for mv in DIRS:
                nr, nc = r + mv.value[0], c + mv.value[1]
                if _is_valid((nr, nc), m) and (nr, nc) not in self._recent_cooldown:
                    fresh += 1
            return fresh

        # Helper: try to A* to the best cell from a list of typed cells
        def _try_typed(typed_cells):
            # Sort: type priority (corner > pocket > dead_end), then freshness
            type_order = {"corner": 0, "pocket": 1, "dead_end": 2}
            scored = [(type_order[t], -_freshness(c), -_manhattan(pacman_pos, c), c)
                      for t, c in typed_cells if c != pacman_pos]
            scored.sort()
            for _, _, _, cell in scored:
                if has_unknown:
                    r, c = cell
                    if not any(0 <= r + mv.value[0] < h and 0 <= c + mv.value[1] < w
                               and m[r + mv.value[0], c + mv.value[1]] == -1
                               for mv in DIRS):
                        continue
                path = self._pf.astar(pacman_pos, cell)
                if path:
                    self._current_target = cell
                    self._current_path = list(path)
                    self._add_cooldown(cell)
                    return self._current_path.pop(0)
            return None

        # ---- Try each zone in order ----
        for zone in zone_order:
            result = _try_typed(zones[zone])
            if result is not None:
                return result

        # ---- Fallback: all upper-half cells (non-zone-classified) ----
        candidates = self._ghost_prob.compute(pacman_pos)
        for cell in candidates:
            if cell[0] >= mid_row:
                continue
            if cell in self._recent_cooldown or cell == pacman_pos:
                continue
            if has_unknown:
                r, c = cell
                if not any(0 <= r + mv.value[0] < h and 0 <= c + mv.value[1] < w
                           and m[r + mv.value[0], c + mv.value[1]] == -1
                           for mv in DIRS):
                    continue
            path = self._pf.astar(pacman_pos, cell)
            if path:
                self._current_target = cell
                self._current_path = list(path)
                self._add_cooldown(cell)
                return self._current_path.pop(0)

        # ---- Lower half (only after upper exhausted or timeout) ----
        upper_exhausted = all(c in self._recent_cooldown or c == pacman_pos
                              for c in candidates if c[0] < mid_row)
        if upper_exhausted:
            self._upper_only_steps = 0
        else:
            self._upper_only_steps += 1

        if upper_exhausted or self._upper_only_steps >= 30:
            for cell in candidates:
                if cell[0] < mid_row:
                    continue
                if cell in self._recent_cooldown or cell == pacman_pos:
                    continue
                if has_unknown:
                    r, c = cell
                    if not any(0 <= r + mv.value[0] < h and 0 <= c + mv.value[1] < w
                               and m[r + mv.value[0], c + mv.value[1]] == -1
                               for mv in DIRS):
                        continue
                path = self._pf.astar(pacman_pos, cell)
                if path:
                    self._current_target = cell
                    self._current_path = list(path)
                    self._add_cooldown(cell)
                    return self._current_path.pop(0)

        # Absolute fallback
        if has_unknown:
            return self._fallback_frontier(pacman_pos)
        else:
            return self._fallback_random(pacman_pos)

    def _fallback_frontier(self, pacman_pos: tuple) -> Move:
        """Find the nearest known cell adjacent to unknown, move toward it."""
        m = self._pf._map
        h, w = m.shape
        best_move = Move.STAY
        best_dist = INF

        for r in range(h):
            for c in range(w):
                if m[r, c] != 0:
                    continue
                borders = any(
                    0 <= r + mv.value[0] < h and 0 <= c + mv.value[1] < w
                    and m[r + mv.value[0], c + mv.value[1]] == -1
                    for mv in DIRS
                )
                if not borders:
                    continue
                if (r, c) in self._recent_cooldown:
                    continue

                d = _manhattan(pacman_pos, (r, c))
                if d < best_dist:
                    best_dist = d
                    path = self._pf.astar(pacman_pos, (r, c))
                    if path:
                        best_move = path[0]

        if best_move == Move.STAY:
            valid = [mv for mv in DIRS
                     if _is_valid((pacman_pos[0] + mv.value[0], pacman_pos[1] + mv.value[1]), m)]
            best_move = random.choice(valid) if valid else Move.STAY

        return best_move

    def _fallback_random(self, pacman_pos: tuple) -> Move:
        """When map is fully known, pick a hiding spot to patrol toward."""
        m = self._pf._map
        candidates = self._ghost_prob.compute(pacman_pos)

        for cell in candidates:
            if cell == pacman_pos:
                continue
            path = self._pf.astar(pacman_pos, cell)
            if path:
                self._current_path = list(path)
                self._add_cooldown(cell)
                return self._current_path.pop(0)

        valid = [mv for mv in DIRS
                 if _is_valid((pacman_pos[0] + mv.value[0], pacman_pos[1] + mv.value[1]), m)]
        return random.choice(valid) if valid else Move.STAY

    def _add_cooldown(self, cell: tuple):
        self._recent_cooldown.append(cell)
        if len(self._recent_cooldown) > self._cooldown_size:
            self._recent_cooldown.pop(0)

    def invalidate_path(self):
        """Force replan on next move."""
        self._current_path = None
        self._current_target = None


# ============================================================
# PacmanAgent -- orchestrator
# ============================================================
class PacmanAgent(BasePacmanAgent):
    """Orchestrates MapMemory, MapAnalyzer, PathFinder, GhostProbability,
    MinimaxEngine, and SweepPlanner to catch the ghost."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Components (wired in first step when map is available)
        self._map_memory = MapMemory()
        self._analyzer = MapAnalyzer()
        self._pathfinder: "PathFinder | None" = None
        self._ghost_prob: "GhostProbability | None" = None
        self._minimax: "MinimaxEngine | None" = None
        self._sweep: "SweepPlanner | None" = None

        # State tracking
        self._last_enemy_pos: "tuple | None" = None
        self._steps_since_seen = 0
        self._last_step_number = 0
        self._wired = False
        self._prev_mode = None  # 'chase', 'recent', or 'sweep'

    def _ensure_wired(self):
        if self._wired:
            return
        self._pathfinder = PathFinder(self._map_memory.get_map)
        self._analyzer.analyze(self._map_memory.get_map())
        self._ghost_prob = GhostProbability(self._analyzer, self._pathfinder)
        self._minimax = MinimaxEngine(self._analyzer, self._pathfinder, self._pacman_speed)
        self._sweep = SweepPlanner(self._ghost_prob, self._pathfinder, self._map_memory)
        self._wired = True

    def step(self, map_state, my_position, enemy_position, step_number):
        # ---- Detect new match (step number reset) ----
        if step_number < self._last_step_number:
            self._wired = False
            self._last_enemy_pos = None
            self._steps_since_seen = 0
        self._last_step_number = step_number

        # ---- Update map memory first (needed before wiring) ----
        new_cells = self._map_memory.update(map_state)
        # Persist map to module cache every step (cheap: dict assignment)
        self._map_memory.save_to_cache()

        # ---- Wire components on first step ----
        self._ensure_wired()

        # Refresh analysis if new cells were discovered
        if new_cells:
            self._analyzer.analyze(self._map_memory.get_map())

        # Clear per-step BFS caches (fresh for each decision)
        self._pathfinder.clear_caches()

        # ---- Track ghost visibility ----
        if enemy_position is not None:
            self._last_enemy_pos = (int(enemy_position[0]), int(enemy_position[1]))
            self._steps_since_seen = 0
        else:
            self._steps_since_seen += 1

        my_pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos = self._last_enemy_pos if self._steps_since_seen == 0 else None
        internal_map = self._map_memory.get_map()

        # ---- Decision tree ----
        chosen_move = Move.STAY

        if enemy_pos is not None:
            # Ghost is visible
            if self._prev_mode != 'chase':
                self._sweep.invalidate_path()
                self._prev_mode = 'chase'
            dist = _manhattan(my_pos, enemy_pos)
            if dist <= 2:
                # Close: A* direct chase
                path = self._pathfinder.astar(my_pos, enemy_pos)
                if path:
                    chosen_move = path[0]
                else:
                    chosen_move = self._greedy_toward(my_pos, enemy_pos, internal_map)
            else:
                # Far: 6-ply minimax
                try:
                    action = self._minimax.search(my_pos, enemy_pos)
                    chosen_move = action[0]
                    if chosen_move == Move.STAY:
                        path = self._pathfinder.astar(my_pos, enemy_pos)
                        if path:
                            chosen_move = path[0]
                except Exception:
                    path = self._pathfinder.astar(my_pos, enemy_pos)
                    if path:
                        chosen_move = path[0]
                    else:
                        chosen_move = self._greedy_toward(my_pos, enemy_pos, internal_map)

        elif self._last_enemy_pos is not None and self._steps_since_seen <= 5:
            # Recently lost sight: expand-search from last known position
            if self._prev_mode != 'recent':
                self._sweep.invalidate_path()
                self._prev_mode = 'recent'

            # Get all hiding spots, sorted by distance from last-known position
            analysis = self._analyzer.get_analysis()
            if analysis is not None:
                # Build priority: dead ends near last-known, then corners, then any cell
                search = []
                for de in analysis["dead_ends"]:
                    d = _manhattan(self._last_enemy_pos, de)
                    if d <= 15:
                        search.append((d, de))
                search.sort()
                for _, target in search:
                    path = self._pathfinder.astar(my_pos, target)
                    if path:
                        chosen_move = path[0]
                        break

                if chosen_move == Move.STAY:
                    for co in analysis["corners"]:
                        d = _manhattan(self._last_enemy_pos, co)
                        if d <= 10:
                            path = self._pathfinder.astar(my_pos, co)
                            if path:
                                chosen_move = path[0]
                                break

            if chosen_move == Move.STAY:
                target = self._bias_toward_dead_end(self._last_enemy_pos)
                path = self._pathfinder.astar(my_pos, target)
                if path:
                    chosen_move = path[0]
                else:
                    chosen_move = self._greedy_toward(my_pos, target, internal_map)

        else:
            # Ghost hidden: sweep search (only invalidate on mode entry)
            if self._prev_mode != 'sweep':
                self._sweep.invalidate_path()
                self._prev_mode = 'sweep'
            try:
                chosen_move = self._sweep.next_move(my_pos)
            except Exception:
                chosen_move = self._random_valid(my_pos, internal_map)

        # ---- Speed multiplier ----
        steps = self._compute_speed_steps(chosen_move, my_pos, internal_map)

        return (chosen_move, steps)

    # ---- Helpers ---------------------------------------------------------
    def _greedy_toward(self, my_pos, target, map_state):
        best_move, best_dist = Move.STAY, _manhattan(my_pos, target)
        for mv in DIRS:
            nxt = (my_pos[0] + mv.value[0], my_pos[1] + mv.value[1])
            if _is_valid(nxt, map_state):
                d = _manhattan(nxt, target)
                if d < best_dist:
                    best_dist, best_move = d, mv
        return best_move

    def _bias_toward_dead_end(self, pos):
        """BFS-expand from pos to find the nearest dead end within 8 steps.
        If none found, return pos unchanged."""
        analysis = self._analyzer.get_analysis()
        if analysis is None:
            return pos
        dead_ends = analysis["dead_ends"]
        internal_map = self._map_memory.get_map()

        # BFS from last known position, looking for nearest dead end
        q = deque([(pos, 0)])
        visited = {pos}
        while q:
            cur, d = q.popleft()
            if d > 15:
                break
            if cur in dead_ends:
                return cur
            for mv in DIRS:
                nxt = (cur[0] + mv.value[0], cur[1] + mv.value[1])
                if nxt not in visited and _is_valid(nxt, internal_map):
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        return pos

    def _random_valid(self, pos, map_state):
        moves = [mv for mv in DIRS
                 if _is_valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state)]
        return random.choice(moves) if moves else Move.STAY

    def _compute_speed_steps(self, move, my_pos, map_state):
        if move == Move.STAY or self._pacman_speed < 2:
            return 1
        dr, dc = move.value
        steps = 1
        for s in range(2, self._pacman_speed + 1):
            nr = my_pos[0] + dr * s
            nc = my_pos[1] + dc * s
            if _is_valid((nr, nc), map_state):
                steps = s
            else:
                break
        return steps

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
