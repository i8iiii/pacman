import sys
import time
import random
import heapq
import collections
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Deque
from collections import deque
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from utils import MemoryMap, BeliefTracker

class TimeoutException(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════ #
# PACMAN AGENT — Seeker under limited observability                         #
# ═══════════════════════════════════════════════════════════════════════════ #

_EXACT, _LOWERBOUND, _UPPERBOUND = 0, 1, 2
_DIRS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
_DELTA = {Move.UP: (-1, 0), Move.DOWN: (1, 0), Move.LEFT: (0, -1), Move.RIGHT: (0, 1)}


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent — catches the Ghost under partial observability.

    Design:
      1. **Persistent memory**: every revealed cell (0 or 1) is stored in
         `self.memory` (a MemoryMap) and never forgotten.  The current step's
         map_state may show -1 for cells outside the current cross-shaped FOV
         (radius 5, wall-blocked), but memory preserves them.
      2. **Belief tracking**: a BeliefTracker maintains a probability
         distribution over the Ghost's position.  When the Ghost is directly
         observed, belief collapses to a delta.  When hidden, belief spreads
         across reachable cells from the last sighting, decayed over time.
      3. **Three operating modes**:
           - VISIBLE (enemy_position is given): A* pursuit to the Ghost's
             current location, with interception prediction when close.
           - HIDDEN-RECENT (lost sight within ~10 steps): pathfind toward
             the belief-weighted target (highest probability position).
           - HIDDEN-LONG (lost sight longer): frontier-based exploration
             to uncover unseen territory and re-acquire the Ghost.
      4. **Speed optimisation**: on straight-line segments with no turns
         and no obstacle, uses the full pacman_speed for burst movement.
      5. **Anti-loop detection**: tracks recent positions; if stuck in a
         cycle of 3 or fewer positions, inserts random perturbation.
    """

    # ---- tunable constants ----
    BELIEF_PURSUIT_STEPS = 10      # steps to keep pursuing belief target
    STAY_PENALTY         = 50
    CORNER_TURN_PENALTY  = 1.02    # A* tie-breaker favouring straight lines

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.name = "Seeker v1 — Fog Hunter"

        # Persistent map memory & belief tracker
        self.memory = MemoryMap(shape=(21, 21))
        self.belief = BeliefTracker(shape=(21, 21), enemy_speed=1, decay=0.85)

        # Stateful tracking
        self.last_known_enemy_pos: Optional[Tuple[int, int]] = None
        self.last_seen_step: int = -100
        self.visit_count: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=12)
        self.last_move: Optional[Move] = None

    # ────────────────────────────────────────────────────────────────────── #
    # Main entry point
    # ────────────────────────────────────────────────────────────────────── #

    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Update persistent memory and belief
        self.memory.update(map_state)
        known = self.memory.get_map()
        self.belief.update(enemy_position, my_position, known)

        # 2. Track visit / history for anti-loop
        self.visit_count[my_position] += 1
        self.position_history.append(my_position)

        # 3. Update last known position
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number

        # 4. Determine target and build path
        target = self._select_target(my_position, enemy_position, step_number, known)

        # 5. Compute next move
        move, steps = self._compute_move(my_position, target, known)

        # 6. Anti-loop: if stuck, perturb
        if self._is_looping(my_position):
            move, steps = self._escape_loop(my_position, known)

        self.last_move = move
        return (move, steps)

    # ────────────────────────────────────────────────────────────────────── #
    # Target selection
    # ────────────────────────────────────────────────────────────────────── #

    def _select_target(
        self,
        my_pos: Tuple[int, int],
        enemy_pos: Optional[Tuple[int, int]],
        step_number: int,
        known: np.ndarray,
    ) -> Tuple[int, int]:
        """Return the (row, col) target for the current step."""
        # Mode 1: Ghost is visible — pursue directly
        if enemy_pos is not None:
            return self._intercept_target(my_pos, enemy_pos, known)

        # Mode 2: Ghost was recently seen — follow belief
        steps_since_seen = step_number - self.last_seen_step
        if (
            self.last_known_enemy_pos is not None
            and steps_since_seen <= self.BELIEF_PURSUIT_STEPS
        ):
            return self._belief_target(my_pos, known)

        # Mode 3: Ghost long lost — frontier exploration
        return self._frontier_target(my_pos, known)

    def _intercept_target(
        self,
        my_pos: Tuple[int, int],
        ghost_pos: Tuple[int, int],
        known: np.ndarray,
    ) -> Tuple[int, int]:
        """
        When Ghost is visible, either pursue directly or compute an
        interception point if the Ghost is running away in a predictable
        direction.
        """
        dist = abs(my_pos[0] - ghost_pos[0]) + abs(my_pos[1] - ghost_pos[1])

        # Very close: just go straight to the ghost
        if dist <= 4:
            return ghost_pos

        # Try to predict where the ghost is heading and cut it off
        # The ghost tends to maximise distance, so it runs away along
        # the axis that gives the most separation.  We pick a cell
        # along that axis that shortens Pacman's approach.
        dr = ghost_pos[0] - my_pos[0]
        dc = ghost_pos[1] - my_pos[1]

        # Ghost-escape direction (opposite to the difference)
        escape_r = 1 if dr < 0 else (-1 if dr > 0 else 0)
        escape_c = 1 if dc < 0 else (-1 if dc > 0 else 0)

        # Look ahead 2–4 cells in the escape direction
        candidates = [ghost_pos]
        for steps_ahead in range(2, 5):
            for sr in (0, escape_r):
                for sc in (0, escape_c):
                    if sr == 0 and sc == 0:
                        continue
                    ahead = (
                        ghost_pos[0] + sr * steps_ahead,
                        ghost_pos[1] + sc * steps_ahead,
                    )
                    if self.memory.is_walkable(ahead):
                        candidates.append(ahead)

        # Pick the candidate closest to Pacman (easiest to reach)
        best = min(candidates, key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]))
        return best

    def _belief_target(
        self,
        my_pos: Tuple[int, int],
        known: np.ndarray,
    ) -> Tuple[int, int]:
        """Weight belief probability by distance to pick the best target."""
        prob = self.belief.get_belief()

        # Zero out walls and cells we can currently see (ghost is not there)
        prob[known == 1] = 0.0
        # Also zero out my own position
        prob[my_pos] = 0.0

        total = prob.sum()
        if total < 1e-12:
            # Fallback to last known position
            return self.last_known_enemy_pos or my_pos

        # Weight: probability / distance^1.5  (closer high-prob cells win)
        h, w = known.shape
        best_score = -1.0
        best_cell = self.last_known_enemy_pos or my_pos
        for r in range(h):
            for c in range(w):
                if prob[r, c] <= 1e-9:
                    continue
                d = abs(r - my_pos[0]) + abs(c - my_pos[1])
                if d < 1:
                    d = 1
                score = prob[r, c] / (d ** 1.5)
                if score > best_score:
                    best_score = score
                    best_cell = (r, c)
        return best_cell

    def _frontier_target(
        self,
        my_pos: Tuple[int, int],
        known: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Find the best frontier cell — a known-empty cell adjacent to
        unknown (-1) territory that is reachable and maximises a score
        combining information gain and distance.
        """
        h, w = known.shape
        best_score = -1.0
        best_cell = my_pos

        for r in range(h):
            for c in range(w):
                if known[r, c] != 0:
                    continue
                # Check adjacency to unknown
                has_unknown = False
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and known[nr, nc] == -1:
                        has_unknown = True
                        break
                if not has_unknown:
                    continue

                # Score: number of adjacent unknown cells / distance
                unknown_count = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and known[nr, nc] == -1:
                        unknown_count += 1

                d = abs(r - my_pos[0]) + abs(c - my_pos[1])
                if d < 1:
                    d = 1
                score = unknown_count / d
                if score > best_score:
                    best_score = score
                    best_cell = (r, c)

        return best_cell

    # ────────────────────────────────────────────────────────────────────── #
    # Movement computation (A* + speed optimisation)
    # ────────────────────────────────────────────────────────────────────── #

    def _compute_move(
        self,
        my_pos: Tuple[int, int],
        target: Tuple[int, int],
        known: np.ndarray,
    ) -> Tuple[Move, int]:
        """Return the (move, steps) to execute this turn."""
        if my_pos == target:
            return Move.STAY, 1

        # Try a straight-line rush first (same row or column, no wall in between)
        straight_move, straight_steps = self._straight_rush(my_pos, target, known)
        if straight_move is not None:
            return straight_move, straight_steps

        # Otherwise run A* for the first move
        path = self._astar_first_move(my_pos, target, known)
        if path is None:
            # Fallback: pick any valid move
            for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                nr, nc = my_pos[0] + mv.value[0], my_pos[1] + mv.value[1]
                if self.memory.is_walkable((nr, nc)) and known[nr, nc] != 1 and known[nr, nc] != -1:
                    return mv, 1
            return Move.STAY, 1

        move = path[0]
        steps = 1

        # Speed optimisation: count consecutive identical moves in the path
        if self.pacman_speed > 1:
            steps = 0
            for m in path:
                if m == move and steps < self.pacman_speed:
                    steps += 1
                else:
                    break
            # But never overshoot into a wall
            actual_steps = 0
            cur = my_pos
            for _ in range(steps):
                nxt = (cur[0] + move.value[0], cur[1] + move.value[1])
                if not self.memory.is_walkable(nxt):
                    break
                actual_steps += 1
                cur = nxt
            steps = max(1, actual_steps)

        return move, steps

    def _straight_rush(
        self,
        my_pos: Tuple[int, int],
        target: Tuple[int, int],
        known: np.ndarray,
    ) -> Tuple[Optional[Move], int]:
        """
        If target is on the same row or column and the path is clear,
        return the move and max steps.  Otherwise return (None, 0).
        """
        dr = target[0] - my_pos[0]
        dc = target[1] - my_pos[1]

        if dr == 0 and dc == 0:
            return None, 0

        if dr == 0:
            move = Move.RIGHT if dc > 0 else Move.LEFT
            dist = abs(dc)
        elif dc == 0:
            move = Move.DOWN if dr > 0 else Move.UP
            dist = abs(dr)
        else:
            return None, 0  # not axis-aligned

        # Verify the path is clear
        r, c = my_pos
        for i in range(1, min(dist, self.pacman_speed) + 1):
            nr, nc = r + move.value[0] * i, c + move.value[1] * i
            if not self.memory.is_walkable((nr, nc)) or known[nr, nc] == -1:
                # Can only go up to i-1 steps
                if i == 1:
                    return None, 0
                return move, i - 1

        return move, min(dist, self.pacman_speed)

    def _astar_first_move(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        known: np.ndarray,
    ) -> Optional[List[Move]]:
        """
        A* on the known map.  Walks only over cells that are known empty (0)
        or unknown (-1) — we optimistically treat unknown as traversable but
        penalise it slightly to favour revealed paths.
        Returns the full path as a list of Moves, or None if unreachable.
        """
        if start == goal:
            return []

        h_map = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        # (f, tie_breaker, g, pos, path)
        tie = 0
        heap = [(0 + h_map(start), tie, 0, start, [])]
        visited: Dict[Tuple[int, int], int] = {}

        while heap:
            f, _, g, pos, path = heapq.heappop(heap)

            if pos == goal:
                return path

            # Prune: if we've visited this state with a lower g, skip
            if pos in visited and visited[pos] <= g:
                continue
            visited[pos] = g

            r, c = pos
            for mv in _DIRS:
                nr, nc = r + mv.value[0], c + mv.value[1]
                nxt = (nr, nc)
                if not self.memory.is_walkable(nxt):
                    continue
                # Treat unknown (-1) as passable but add a small penalty
                cell_type = known[nr, nc] if (0 <= nr < known.shape[0] and 0 <= nc < known.shape[1]) else -1
                unknown_penalty = 2.0 if cell_type == -1 else 0.0

                turn_penalty = 0.0
                if path and path[-1] != mv:
                    turn_penalty = self.CORNER_TURN_PENALTY

                ng = g + 1 + unknown_penalty + turn_penalty
                nh = h_map(nxt)

                if nxt in visited and visited[nxt] <= ng:
                    continue

                tie += 1
                heapq.heappush(heap, (ng + nh, tie, ng, nxt, path + [mv]))

        return None

    # ────────────────────────────────────────────────────────────────────── #
    # Anti-loop
    # ────────────────────────────────────────────────────────────────────── #

    def _is_looping(self, my_pos: Tuple[int, int]) -> bool:
        """Detect if stuck in a small cycle."""
        if len(self.position_history) < 8:
            return False
        unique = len(set(self.position_history))
        return unique <= 3

    def _escape_loop(
        self,
        my_pos: Tuple[int, int],
        known: np.ndarray,
    ) -> Tuple[Move, int]:
        """Break out of the cycle with a random valid move."""
        self.position_history.clear()
        candidates = []
        for mv in _DIRS:
            nr, nc = my_pos[0] + mv.value[0], my_pos[1] + mv.value[1]
            if self.memory.is_walkable((nr, nc)):
                candidates.append(mv)
        if candidates:
            m = random.choice(candidates)
            return m, 1
        return Move.STAY, 1


# ═══════════════════════════════════════════════════════════════════════════ #
# GHOST AGENT — Hider under limited observability                           #
# ═══════════════════════════════════════════════════════════════════════════ #


class GhostAgent(BaseGhostAgent):
    """
    Hide agent under partial observability (cross-shaped FOV, radius `vision`).

    Core ideas:
      1. Persistent memory: walls/paths never change mid-match, so every cell
         ever revealed is remembered forever in `self.known_map`, even on
         steps where it falls back out of the current fog mask (-1).
      2. Two operating modes each step:
           - VISIBLE  (enemy_position given): the seeker is by definition
             within `vision` cells on a clear straight line (that's *how*
             it became visible), so a bounded alpha-beta minimax over the
             locally-known map is cheap and gives strong short-horizon play.
           - HIDDEN (enemy_position is None): we do not know where the
             seeker is, so we fall back to a belief anchored at the last
             confirmed sighting, decayed over elapsed steps, blended with
             general "stay safe / stay hidden" heuristics (mobility, escape
             room, low exposure to open sight-lines, mild exploration).
      3. Line-of-sight is symmetric here (both agents use the same
         straight-ray-blocked-by-walls rule), so "I can currently see the
         seeker" <=> "the seeker can currently see me". Breaking the sight
         line (ducking behind a corner) is therefore one of the single most
         valuable defensive moves, so it is rewarded explicitly, not just
         left as a side effect of maximizing raw distance.

    All root-level move choices are validated against the *current, exactly
    known* immediate neighbours (the four adjacent cells are always fully
    revealed every step, since the vision ray always covers at least 1 cell
    in each direction) — so the agent can never accidentally "choose" an
    illegal or unsafe-by-ignorance move at the top level.
    """

    # ---- tunable weights (safe to retune offline via self-play) ----
    W_DIST = 40
    W_MOBILITY = 20
    W_SAFE_AREA = 3
    W_LOS_BREAK = 250
    W_OPENNESS = 4
    W_UNKNOWN = 15
    STAY_PENALTY = 30
    REVISIT_PENALTY = 15

    CAPTURE_DIST = 2
    SEEKER_SPEED = 2
    VISION = 5
    BELIEF_DECAY = 18       # steps until a stale sighting stops driving movement

    MAX_TIME = 0.85
    TIME_CHECK_MASK = 127
    MAX_TT_ENTRIES = 120_000
    MAX_DEPTH = 14

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Fog Hider"
        self.pacman_speed = int(kwargs.get("pacman_speed", self.SEEKER_SPEED))
        self.capture_distance = int(kwargs.get("capture_distance", self.CAPTURE_DIST))
        self.vision = int(kwargs.get("vision_range", self.VISION))
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.known_map = None
        self.last_seen_pos = None
        self.last_seen_step = None
        self.recent = collections.deque(maxlen=6)
        self.last_move = None
        self.visit_count = collections.defaultdict(int)
        self.tt = {}
        self.killers = {}
        self.node_count = 0
        self._nbr_cache = {}
        self._nbr_cache_version = -1
        self._map_version = 0
        self._bfs_cache = {}

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def step(self, map_state, my_position, enemy_position, step_number):
        if step_number == 0:
            self._reset_episode_state()

        self._update_known_map(map_state)
        self.visit_count[my_position] += 1

        legal = self._legal_moves(map_state, my_position)
        if not legal:
            return Move.STAY

        if enemy_position is not None:
            self.last_seen_pos = enemy_position
            self.last_seen_step = step_number
            move = self._visible_tactics(my_position, enemy_position, legal)
            if move not in legal:
                move = self._safe_fallback(my_position, enemy_position, legal)
        else:
            move = self._hidden_heuristic(my_position, step_number, legal)

        self.last_move = move
        self.recent.append(my_position)
        return move

    # ------------------------------------------------------------------ #
    # persistent map memory
    # ------------------------------------------------------------------ #
    def _update_known_map(self, map_state):
        if self.known_map is None:
            self.known_map = np.full(map_state.shape, -1, dtype=np.int8)
        mask = map_state != -1
        if np.any(mask):
            changed = np.any(self.known_map[mask] != map_state[mask])
            self.known_map[mask] = map_state[mask]
            if changed:
                self._map_version += 1

    def _legal_moves(self, map_state, pos):
        """Moves whose destination is *this step's* exactly-known FOV.
        The 4 immediate neighbours are always inside the vision cross, so
        this never depends on stale memory."""
        r, c = pos
        rows, cols = map_state.shape
        moves = [Move.STAY]
        for mv in _DIRS:
            dr, dc = _DELTA[mv]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and map_state[nr, nc] == 0:
                moves.append(mv)
        return moves

    def _step_pos(self, pos, mv):
        if mv is Move.STAY:
            return pos
        dr, dc = _DELTA[mv]
        return (pos[0] + dr, pos[1] + dc)

    def _is_known_empty(self, pos):
        r, c = pos
        rows, cols = self.known_map.shape
        return 0 <= r < rows and 0 <= c < cols and self.known_map[r, c] == 0

    def _known_neighbors(self, pos):
        if self._nbr_cache_version != self._map_version:
            self._nbr_cache = {}
            self._nbr_cache_version = self._map_version
        cached = self._nbr_cache.get(pos)
        if cached is not None:
            return cached
        r, c = pos
        rows, cols = self.known_map.shape
        result = []
        for mv in _DIRS:
            dr, dc = _DELTA[mv]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and self.known_map[nr, nc] == 0:
                result.append((nr, nc))
        result = tuple(result)
        self._nbr_cache[pos] = result
        return result

    # ------------------------------------------------------------------ #
    # distance / mobility / hiding-quality helpers (over known_map)
    # ------------------------------------------------------------------ #
    def _bfs_from(self, source):
        dist = {source: 0}
        dq = collections.deque([source])
        while dq:
            cur = dq.popleft()
            d = dist[cur]
            for nb in self._known_neighbors(cur):
                if nb not in dist:
                    dist[nb] = d + 1
                    dq.append(nb)
        return dist

    @staticmethod
    def _dist(dist_map, target, origin):
        d = dist_map.get(target)
        if d is not None:
            return d
        # Target lies outside our currently-known territory: fall back to
        # a Manhattan lower-bound rather than pretending it is unreachable.
        return abs(target[0] - origin[0]) + abs(target[1] - origin[1])

    def _mobility(self, pos):
        return len(self._known_neighbors(pos))

    def _safe_area(self, pos, limit=4):
        seen = {pos}
        dq = collections.deque([(pos, 0)])
        count = 0
        while dq:
            cur, d = dq.popleft()
            if d >= limit:
                continue
            for nb in self._known_neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    count += 1
                    dq.append((nb, d + 1))
        return count

    def _openness(self, pos):
        """How many cells of open sight-line radiate from `pos` (capped at
        vision range). Unknown cells are conservatively assumed to keep the
        ray going (worst case), only a *confirmed* wall stops it early."""
        total = 0
        rows, cols = self.known_map.shape
        for mv in _DIRS:
            dr, dc = _DELTA[mv]
            r, c = pos
            for _ in range(self.vision):
                r, c = r + dr, c + dc
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                v = self.known_map[r, c]
                if v == 1:
                    break
                total += 1
        return total

    def _line_clear(self, a, b):
        """Tri-state visibility between two aligned cells over known_map:
        True=confirmed clear, False=confirmed blocked, None=runs through
        still-unknown territory."""
        if a[0] == b[0]:
            axis, fixed = 1, a[0]
            lo, hi = min(a[1], b[1]), max(a[1], b[1])
        elif a[1] == b[1]:
            axis, fixed = 0, a[1]
            lo, hi = min(a[0], b[0]), max(a[0], b[0])
        else:
            return False
        if hi - lo > self.vision:
            return False
        for k in range(lo + 1, hi):
            cell = (fixed, k) if axis == 1 else (k, fixed)
            v = self.known_map[cell]
            if v == 1:
                return False
            if v == -1:
                return None
        return True

    def _los_bonus(self, ghost_pos, pacman_pos):
        state = self._line_clear(ghost_pos, pacman_pos)
        if state is True:
            return -1.0
        if state is False:
            return 1.0
        return 0.4

    # ------------------------------------------------------------------ #
    # VISIBLE mode: bounded alpha-beta minimax (mutual visibility implies
    # the seeker is already within `vision` cells, so the tree is small)
    # ------------------------------------------------------------------ #
    def _ghost_sim_moves(self, pos):
        opts = [(Move.STAY, pos)]
        r, c = pos
        for mv in _DIRS:
            dr, dc = _DELTA[mv]
            nb = (r + dr, c + dc)
            if self._is_known_empty(nb):
                opts.append((mv, nb))
        return opts

    def _pacman_sim_moves(self, p_pos, g_pos):
        options = {p_pos}
        rows, cols = self.known_map.shape
        for mv in _DIRS:
            dr, dc = _DELTA[mv]
            cur = p_pos
            for _ in range(self.pacman_speed):
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or self.known_map[nr, nc] != 0:
                    break
                cur = (nr, nc)
                options.add(cur)
        ordered = sorted(options, key=lambda x: abs(x[0] - g_pos[0]) + abs(x[1] - g_pos[1]))
        return ordered[:7]

    def _evaluate(self, g_pos, p_pos):
        dmap = self._bfs_cache.get(p_pos)
        if dmap is None:
            dmap = self._bfs_from(p_pos)
            self._bfs_cache[p_pos] = dmap
        dist = self._dist(dmap, g_pos, p_pos)
        mobility = self._mobility(g_pos)
        los = self._los_bonus(g_pos, p_pos)
        return dist * self.W_DIST + mobility * self.W_MOBILITY + los * self.W_LOS_BREAK

    def _minimax(self, g_pos, p_pos, depth, is_ghost, alpha, beta, start):
        self.node_count += 1
        if (self.node_count & self.TIME_CHECK_MASK) == 0 and time.time() - start > self.MAX_TIME:
            raise TimeoutException()

        if abs(g_pos[0] - p_pos[0]) + abs(g_pos[1] - p_pos[1]) < self.capture_distance:
            return -100000 + (20 - depth), None
        if depth == 0:
            return self._evaluate(g_pos, p_pos), None

        key = (g_pos, p_pos, is_ghost)
        a0, b0 = alpha, beta
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            e_depth, e_val, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_flag == _EXACT:
                    return e_val, e_move
                elif e_flag == _LOWERBOUND:
                    alpha = max(alpha, e_val)
                elif e_flag == _UPPERBOUND:
                    beta = min(beta, e_val)
                if alpha >= beta:
                    return e_val, e_move

        if is_ghost:
            val, best_m = float('-inf'), Move.STAY
            killer_list = self.killers.get(depth, ())
            options = self._ghost_sim_moves(g_pos)
            options.sort(key=lambda item: (item[0] == tt_move, item[0] in killer_list), reverse=True)
            for mv, nxt in options:
                res, _ = self._minimax(nxt, p_pos, depth - 1, False, alpha, beta, start)
                if mv is Move.STAY:
                    res -= self.STAY_PENALTY
                if res > val:
                    val, best_m = res, mv
                alpha = max(alpha, val)
                if beta <= alpha:
                    if mv is not Move.STAY:
                        kl = self.killers.setdefault(depth, [])
                        if mv not in kl:
                            kl.insert(0, mv)
                            del kl[2:]
                    break
        else:
            val, best_m = float('inf'), None
            for nxt in self._pacman_sim_moves(p_pos, g_pos):
                res, _ = self._minimax(g_pos, nxt, depth - 1, True, alpha, beta, start)
                if res < val:
                    val = res
                beta = min(beta, val)
                if beta <= alpha:
                    break

        flag = _UPPERBOUND if val <= a0 else (_LOWERBOUND if val >= b0 else _EXACT)
        self.tt[key] = (depth, val, flag, best_m)
        return val, best_m

    def _safe_fallback(self, my_pos, enemy_pos, legal):
        best, best_key = Move.STAY, None
        for mv in legal:
            nxt = self._step_pos(my_pos, mv)
            d = abs(nxt[0] - enemy_pos[0]) + abs(nxt[1] - enemy_pos[1])
            key = (d, self._mobility(nxt))
            if best_key is None or key > best_key:
                best_key, best = key, mv
        return best

    def _visible_tactics(self, my_pos, enemy_pos, legal):
        start = time.time()
        self.node_count = 0
        self._bfs_cache = {}
        if len(self.tt) > self.MAX_TT_ENTRIES:
            self.tt = {}
        self.killers = {}

        best_move = self._safe_fallback(my_pos, enemy_pos, legal)
        try:
            for depth in range(2, self.MAX_DEPTH):
                _, mv = self._minimax(my_pos, enemy_pos, depth, True,
                                       float('-inf'), float('inf'), start)
                if mv is not None:
                    best_move = mv
        except TimeoutException:
            pass
        return best_move

    # ------------------------------------------------------------------ #
    # HIDDEN mode: belief-decayed heuristic scoring
    # ------------------------------------------------------------------ #
    def _explore_safely(self, my_pos, legal):
        best_move, best_score = Move.STAY, float('-inf')
        for mv in legal:
            nxt = self._step_pos(my_pos, mv)
            score = (self._mobility(nxt) * self.W_MOBILITY
                     + self._safe_area(nxt) * self.W_SAFE_AREA
                     + self.W_UNKNOWN / (1 + self.visit_count.get(nxt, 0)))
            if mv is Move.STAY:
                score -= self.STAY_PENALTY
            if score > best_score:
                best_score, best_move = score, mv
        return best_move

    def _hidden_heuristic(self, my_pos, step_number, legal):
        if self.last_seen_pos is None:
            return self._explore_safely(my_pos, legal)

        elapsed = max(0, step_number - self.last_seen_step)
        confidence = max(0.0, 1.0 - elapsed / self.BELIEF_DECAY)
        dmap = self._bfs_from(self.last_seen_pos)

        candidates = []
        for mv in legal:
            nxt = self._step_pos(my_pos, mv)
            dist = self._dist(dmap, nxt, self.last_seen_pos)
            mobility = self._mobility(nxt)
            safe = self._safe_area(nxt)
            openness = self._openness(nxt)
            novelty = self.W_UNKNOWN / (1 + self.visit_count.get(nxt, 0))
            revisit = self.REVISIT_PENALTY if nxt in self.recent else 0
            stay_pen = self.STAY_PENALTY if mv is Move.STAY else 0

            score = (confidence * dist * self.W_DIST
                     + mobility * self.W_MOBILITY
                     + safe * self.W_SAFE_AREA
                     - openness * self.W_OPENNESS
                     + novelty - revisit - stay_pen)
            candidates.append((score, mv))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[0][0]
        ties = [mv for s, mv in candidates if s >= top - 1e-6]
        if len(ties) > 1 and self.last_move in ties:
            return self.last_move
        return random.choice(ties)