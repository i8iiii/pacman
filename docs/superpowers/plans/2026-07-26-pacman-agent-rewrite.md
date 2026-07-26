# PacmanAgent Clean Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `submissions/LAB2/agent.py` with a clean modular class-based PacmanAgent using 6-ply minimax, A*/BFS, ghost probability estimation, and persistent map cache.

**Architecture:** Single file `agent.py` containing 6 internal classes (MapMemory, MapAnalyzer, PathFinder, GhostProbability, MinimaxEngine, SweepPlanner) orchestrated by PacmanAgent. GhostAgent is kept unchanged as-is.

**Tech Stack:** Python 3.7+, numpy, standard library (deque, heapq, time, random, pathlib)

## Global Constraints

- Single file output: `submissions/LAB2/agent.py`
- GhostAgent class must remain identical to current (imports `HideController`)
- PacmanAgent must inherit from `agent_interface.PacmanAgent`
- Must handle `enemy_position is None` (fog of war)
- Must support Pacman speed multiplier (return `(Move, steps)` tuple when speed > 1)
- Map cache persists across arena matches via module-level `_MAP_CACHE` dict
- BFS for evaluation distances, A* for pathfinding
- 6-ply minimax with alpha-beta pruning (fixed depth, not adaptive)

---

### Task 1: File skeleton, helpers, _MAP_CACHE, and GhostAgent

**Files:**
- Create: `submissions/LAB2/agent.py` (full rewrite)

**Interfaces:**
- Produces: `_MAP_CACHE: dict[int, np.ndarray]`, `_fingerprint(map_state) -> int`, `_manhattan(a, b) -> int`, `_is_valid(pos, map_state) -> bool`, `_count_exits(pos, map_state) -> int`, `DIRS`, `INF`, `GhostAgent` class

**Purpose:** Set up module-level infrastructure: imports, constants, helper functions, cache dict, and the unchanged GhostAgent. This is the foundation that all subsequent tasks build on.

- [ ] **Step 1: Write the skeleton file with imports, helpers, cache, and GhostAgent**

```python
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
import time
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
_MAP_CACHE: dict[int, np.ndarray] = {}


def _fingerprint(map_state: np.ndarray) -> int:
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


def _is_valid(pos: tuple, map_state: np.ndarray) -> bool:
    """Check if position is within bounds and not a wall (value != 1)."""
    r, c = pos
    if map_state is None:
        return False
    h, w = map_state.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return map_state[r, c] != 1


def _count_exits(pos: tuple, map_state: np.ndarray) -> int:
    return sum(
        1 for mv in DIRS
        if _is_valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state)
    )


# ============================================================
# GhostAgent (unchanged from existing — kept verbatim)
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
```

- [ ] **Step 2: Verify the file imports correctly**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
from agent import _manhattan, _is_valid, _fingerprint, _MAP_CACHE, GhostAgent, DIRS, INF
print('Imports OK')
print('manhattan((0,0),(3,4)):', _manhattan((0,0),(3,4)))
"
```

Expected: `Imports OK` + `manhattan((0,0),(3,4)): 7`

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add agent skeleton with helpers, cache, and GhostAgent"
```

---

### Task 2: MapMemory component

**Files:**
- Modify: `submissions/LAB2/agent.py` — add MapMemory class before GhostAgent section

**Interfaces:**
- Consumes: `_MAP_CACHE`, `_fingerprint`, `_is_valid`
- Produces: `MapMemory` class with:
  - `__init__()`: creates empty state (lazy init on first update)
  - `update(map_state: np.ndarray) -> bool`: merge observation, return True if new cells discovered
  - `get_map() -> np.ndarray`: return current internal_map
  - `is_initialized() -> bool`: whether we've received first observation

**Purpose:** Manages the growing internal map with cross-match persistence. On first update, checks module cache for previously-discovered map. Merges visible cells each step.

- [ ] **Step 1: Add MapMemory class to agent.py**

Insert before the `# ============================================================` line that precedes GhostAgent:

```python
# ============================================================
# MapMemory — persistent internal map, cached across matches
# ============================================================
class MapMemory:
    """Maintains the growing internal map.

    On first update: fingerprints the wall pattern and checks _MAP_CACHE.
    If this map was seen in a previous match, restores the full discovered
    layout. On subsequent matches, Pacman starts with full map knowledge.
    """

    def __init__(self):
        self._internal_map: np.ndarray | None = None
        self._fingerprint: int | None = None
        self._previous_cell_count = 0
        self._started = False

    def update(self, map_state: np.ndarray) -> bool:
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

    def get_map(self) -> np.ndarray:
        return self._internal_map

    def is_initialized(self) -> bool:
        return self._started
```

- [ ] **Step 2: Verify MapMemory standalone**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
import numpy as np
from agent import MapMemory, _MAP_CACHE
_MAP_CACHE.clear()

mm = MapMemory()
obs = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int8)
mm.update(obs)
print('Known cells:', (mm.get_map() == 0).sum())
print('Expected: 1')
mm.save_to_cache()
print('Cache size:', len(_MAP_CACHE))
"
```

Expected: `Known cells: 1` + `Cache size: 1`

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add MapMemory with cross-match persistence"
```

---

### Task 3: MapAnalyzer component

**Files:**
- Modify: `submissions/LAB2/agent.py` — add MapAnalyzer class after MapMemory

**Interfaces:**
- Consumes: `_is_valid`, `_count_exits`, `DIRS`, `deque`
- Produces: `MapAnalyzer` class with:
  - `analyze(internal_map: np.ndarray) -> dict`: returns analysis dict
  - `get_analysis() -> dict`: returns cached analysis or None
  - Analysis dict keys: `dead_ends`, `corners`, `pockets`, `exit_counts`, `mid_row`

**Purpose:** One-shot structural map analysis. Detects dead ends (1 exit), corners (2 perpendicular exits), pocket regions (clusters of low-exit cells), and exit counts for every cell.

- [ ] **Step 1: Add MapAnalyzer class to agent.py**

Insert after MapMemory class:

```python
# ============================================================
# MapAnalyzer — structural map analysis
# ============================================================
class MapAnalyzer:
    """Analyzes the known map to find dead ends, corners, and pocket regions.

    Results are cached. Re-analyze when new cells are discovered.
    """

    def __init__(self):
        self._cached: dict | None = None

    def analyze(self, internal_map: np.ndarray) -> dict:
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

    def get_analysis(self) -> dict | None:
        return self._cached
```

- [ ] **Step 2: Verify MapAnalyzer on the default map**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
import numpy as np
from agent import MapAnalyzer
from environment import Environment

env = Environment()
full_map = env.map

ma = MapAnalyzer()
result = ma.analyze(full_map)
print('Dead ends:', len(result['dead_ends']))
print('Corners:', len(result['corners']))
print('Pockets:', len(result['pockets']))
print('Mid row:', result['mid_row'])
print('Exit counts sample:', list(result['exit_counts'].items())[:3])
"
```

Expected: Reasonable counts (e.g., 10-30 dead ends, several corners and pockets on the default 21x19 map).

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add MapAnalyzer with dead-end/corner/pocket detection"
```

---

### Task 4: PathFinder component (A* + BFS)

**Files:**
- Modify: `submissions/LAB2/agent.py` — add PathFinder class after MapAnalyzer

**Interfaces:**
- Consumes: `DIRS`, `_manhattan`, `_is_valid`, `INF`, `deque`, `heappush`, `heappop`
- Produces: `PathFinder` class with:
  - `__init__(internal_map_getter: callable)`: stores a function that returns current known map
  - `astar(start, goal) -> list[Move] | None`: A* path, returns None if unreachable
  - `bfs_dist(a, b) -> int`: cached pair distance (INF if unreachable)
  - `bfs_all_dists(source) -> dict`: all distances from source (cached per source)
  - `clear_caches()`: reset per-step BFS caches
  - `manhattan(a, b) -> int`: delegate to module helper

**Purpose:** Provides pathfinding and distance computation for all other components. A* for path planning, BFS for exact distances in evaluation. BFS results are cached within a step to avoid redundant work during minimax search.

- [ ] **Step 1: Add PathFinder class to agent.py**

Insert after MapAnalyzer class:

```python
# ============================================================
# PathFinder — A* pathfinding + BFS distance caching
# ============================================================
class PathFinder:
    """A* for pathfinding, BFS for exact distance computations.

    BFS distances are cached per source within a single step (the cache
    is cleared between steps by the orchestrator).
    """

    def __init__(self, map_getter):
        """map_getter: callable that returns the current known np.ndarray map."""
        self._map_getter = map_getter
        self._bfs_source_cache: dict[tuple, dict] = {}
        self._pair_cache: dict[tuple, int] = {}

    @property
    def _map(self) -> np.ndarray:
        return self._map_getter()

    # ---- A* ----------------------------------------------------------------
    def astar(self, start: tuple, goal: tuple) -> list | None:
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

    def bfs_all_dists(self, source: tuple) -> dict[tuple, int]:
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
```

- [ ] **Step 2: Verify PathFinder on the default map**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
import numpy as np
from agent import PathFinder
from environment import Environment

env = Environment()
full_map = env.map

pf = PathFinder(lambda: full_map)
path = pf.astar((1,1), (19,19))
print('A* path length:', len(path) if path else 'None')
print('BFS dist (1,1)->(19,19):', pf.bfs_dist((1,1), (19,19)))
dists = pf.bfs_all_dists((10,10))
print('Cells reachable from (10,10):', len(dists))
"
```

Expected: A* path exists, BFS distance matches Manhattan bound, 100+ reachable cells.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add PathFinder with A* and cached BFS"
```

---

### Task 5: GhostProbability component

**Files:**
- Modify: `submissions/LAB2/agent.py` — add GhostProbability class after PathFinder

**Interfaces:**
- Consumes: `MapAnalyzer.get_analysis()`, `PathFinder.bfs_dist()`, `_manhattan`
- Produces: `GhostProbability` class with:
  - `__init__(analyzer: MapAnalyzer, pathfinder: PathFinder)`
  - `compute(pacman_pos: tuple) -> list[tuple]`: returns list of `(row, col)` sorted by descending probability

**Purpose:** When the ghost is hidden, estimates a probability distribution over all known reachable cells. Weights dead ends (×5), corners (×3), upper half (×3), pockets (×2), and applies inverse distance from Pacman.

- [ ] **Step 1: Add GhostProbability class to agent.py**

Insert after PathFinder class:

```python
# ============================================================
# GhostProbability — weighted distribution over hiding spots
# ============================================================
class GhostProbability:
    """Estimates where the ghost is likely hiding when out of sight.

    Weights: dead ends > corners > upper-half > pockets > distance.
    """

    def __init__(self, analyzer: MapAnalyzer, pathfinder: PathFinder):
        self._analyzer = analyzer
        self._pathfinder = pathfinder

    def compute(self, pacman_pos: tuple) -> list[tuple]:
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
        pocket_of: dict[tuple, int] = {}
        for pid, region in pockets.items():
            for cell in region:
                pocket_of[cell] = pid

        # Score every known empty cell
        scored: list[tuple[float, tuple]] = []
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
```

- [ ] **Step 2: Verify GhostProbability on default map**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
import numpy as np
from agent import MapAnalyzer, PathFinder, GhostProbability
from environment import Environment

env = Environment()
full_map = env.map

ma = MapAnalyzer()
ma.analyze(full_map)

pf = PathFinder(lambda: full_map)

gp = GhostProbability(ma, pf)
top = gp.compute((1, 1))
print('Top 5 ghost hiding spots:', top[:5])
print('Total scored cells:', len(top))
"
```

Expected: Top spots should include upper-half dead ends and corners.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add GhostProbability with weighted hiding-spot estimation"
```

---

### Task 6: MinimaxEngine component

**Files:**
- Modify: `submissions/LAB2/agent.py` — add MinimaxEngine class after GhostProbability

**Interfaces:**
- Consumes: `MapAnalyzer.get_analysis()`, `PathFinder`, `DIRS`, `INF`, `Move`
- Produces: `MinimaxEngine` class with:
  - `__init__(analyzer, pathfinder, pacman_speed)`
  - `search(pac_pos, ghost_pos) -> tuple[Move, int]`: returns best (move, steps) action
  - Internal: `_max_node`, `_min_node`, `_evaluate`, `_pacman_actions`, `_apply_action`, `_scored_ghost_moves`, `_aligned_with_pacman`, `_perpendicular_to`

**Purpose:** 6-ply minimax with alpha-beta pruning. Models Pacman (max) trying to minimize distance, Ghost (min) trying to maximize distance. Uses BFS distances for evaluation. Ghost move scoring includes dead-end, corner, and perpendicular bonuses.

- [ ] **Step 1: Add MinimaxEngine class to agent.py**

Insert after GhostProbability class:

```python
# ============================================================
# MinimaxEngine — 6-ply alpha-beta adversarial search
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
        self._depth = 6  # fixed 6 plies

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
                return best  # prune
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
                return best  # prune
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
        analysis = self._analyzer.get_analysis()
        dead_ends = analysis["dead_ends"] if analysis else set()
        corners = analysis["corners"] if analysis else set()
        exit_counts = analysis["exit_counts"] if analysis else {}

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

            dist = self._pf.bfs_dist(pac_pos, nxt)
            exits = exit_counts.get(nxt, 0)
            score = dist * 10 + exits * 3
            if nxt in dead_ends:
                score += 30
            if nxt in corners:
                score += 15
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
```

- [ ] **Step 2: Verify MinimaxEngine produces valid moves**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
import numpy as np
from agent import MapAnalyzer, PathFinder, MinimaxEngine, _is_valid
from environment import Environment

env = Environment()
full_map = env.map

ma = MapAnalyzer()
ma.analyze(full_map)
pf = PathFinder(lambda: full_map)
engine = MinimaxEngine(ma, pf, pacman_speed=1)

# Test with pacman at default start, ghost nearby
pac_pos = env.default_pacman_start
ghost_pos = (pac_pos[0] + 2, pac_pos[1])

action = engine.search(pac_pos, ghost_pos)
print('Minimax action:', action)
print('Valid move type:', isinstance(action, tuple))
"
```

Expected: Returns a valid (Move, steps) tuple.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add MinimaxEngine with 6-ply alpha-beta search"
```

---

### Task 7: SweepPlanner component

**Files:**
- Modify: `submissions/LAB2/agent.py` — add SweepPlanner class after MinimaxEngine

**Interfaces:**
- Consumes: `GhostProbability`, `PathFinder`, `MapMemory`
- Produces: `SweepPlanner` class with:
  - `__init__(ghost_prob: GhostProbability, pathfinder: PathFinder, map_memory: MapMemory)`
  - `next_move(pacman_pos: tuple) -> Move`: returns best move toward highest-probability unexplored cell

**Purpose:** Systematic exploration when ghost is hidden. Uses GhostProbability to pick the best cell to explore, then A* to navigate there. Maintains a cooldown list of recently visited cells to prevent oscillation.

- [ ] **Step 1: Add SweepPlanner class to agent.py**

Insert after MinimaxEngine class:

```python
# ============================================================
# SweepPlanner — systematic exploration when ghost is hidden
# ============================================================
class SweepPlanner:
    """Plans systematic exploration using the ghost probability distribution.

    Picks the highest-probability unknown-adjacent cell, navigates to it
    via A*, and maintains a cooldown on recently visited cells.
    """

    def __init__(self, ghost_prob: GhostProbability, pathfinder: PathFinder, map_memory: MapMemory):
        self._ghost_prob = ghost_prob
        self._pf = pathfinder
        self._map_memory = map_memory
        self._recent_cooldown: list[tuple] = []  # last 20 visited frontier cells
        self._cooldown_size = 20
        self._current_path: list | None = None
        self._current_target: tuple | None = None

    def next_move(self, pacman_pos: tuple) -> Move:
        """Return the next move to systematically explore the map."""
        # If we have a current path, continue following it
        if self._current_path:
            move = self._current_path.pop(0)
            # Revalidate: is the path still walkable?
            next_pos = (pacman_pos[0] + move.value[0], pacman_pos[1] + move.value[1])
            m = self._pf._map
            if _is_valid(next_pos, m):
                return move
            # Path invalidated, replan
            self._current_path = None
            self._current_target = None

        # Get probability-ranked hiding spots
        candidates = self._ghost_prob.compute(pacman_pos)
        m = self._pf._map

        for cell in candidates:
            if cell in self._recent_cooldown:
                continue

            # Check if cell is adjacent to unknown (-1) region
            h, w = m.shape
            r, c = cell
            borders_unknown = False
            for mv in DIRS:
                nr, nc = r + mv.value[0], c + mv.value[1]
                if 0 <= nr < h and 0 <= nc < w and m[nr, nc] == -1:
                    borders_unknown = True
                    break

            if not borders_unknown:
                continue

            # Try A* to this cell
            path = self._pf.astar(pacman_pos, cell)
            if path:
                self._current_target = cell
                self._current_path = list(path)
                self._add_cooldown(cell)
                move = self._current_path.pop(0)
                return move

        # Fallback: nearest frontier (known cell adjacent to unknown)
        return self._fallback_frontier(pacman_pos)

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
                # Check if borders unknown
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
            # Truly stuck: pick any valid random move
            valid = [mv for mv in DIRS
                     if _is_valid((pacman_pos[0] + mv.value[0], pacman_pos[1] + mv.value[1]), m)]
            best_move = random.choice(valid) if valid else Move.STAY

        return best_move

    def _add_cooldown(self, cell: tuple):
        self._recent_cooldown.append(cell)
        if len(self._recent_cooldown) > self._cooldown_size:
            self._recent_cooldown.pop(0)

    def invalidate_path(self):
        """Force replan on next move."""
        self._current_path = None
        self._current_target = None
```

- [ ] **Step 2: Verify SweepPlanner picks reasonable targets**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && python -c "
import sys; sys.path.insert(0, '../../src')
import numpy as np
from agent import MapMemory, MapAnalyzer, PathFinder, GhostProbability, SweepPlanner, _MAP_CACHE
from environment import Environment

_MAP_CACHE.clear()

env = Environment()
# Simulate partial map with fog
partial = env.map.copy()
partial[10:, :] = -1  # bottom half unknown

mm = MapMemory()
mm.update(partial)

ma = MapAnalyzer()
ma.analyze(mm.get_map())

pf = PathFinder(mm.get_map)
gp = GhostProbability(ma, pf)
sp = SweepPlanner(gp, pf, mm)

move = sp.next_move((3, 3))
print('Sweep move:', move)
"
```

Expected: Returns a valid Move enum (not STAY) pointing toward upper-half hiding spots.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add SweepPlanner with probability-guided exploration"
```

---

### Task 8: PacmanAgent orchestrator

**Files:**
- Modify: `submissions/LAB2/agent.py` — add PacmanAgent class as the final class in the file

**Interfaces:**
- Consumes: All previous components (MapMemory, MapAnalyzer, PathFinder, GhostProbability, MinimaxEngine, SweepPlanner)
- Produces: `PacmanAgent(BasePacmanAgent)` with:
  - `__init__(**kwargs)`: wires up all components, reads pacman_speed
  - `step(map_state, my_position, enemy_position, step_number) -> tuple[Move, int]`: main decision logic

**Purpose:** Orchestrates all components. Decision tree: direct chase (A*) when ghost is close, minimax when visible and far, recent-position search when ghost was recently seen, sweep planner when ghost is hidden.

- [ ] **Step 1: Add PacmanAgent class to agent.py**

Insert after SweepPlanner class and replace any existing PacmanAgent:

```python
# ============================================================
# PacmanAgent — orchestrator
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
        self._pathfinder: PathFinder | None = None
        self._ghost_prob: GhostProbability | None = None
        self._minimax: MinimaxEngine | None = None
        self._sweep: SweepPlanner | None = None

        # State tracking
        self._last_enemy_pos: tuple | None = None
        self._steps_since_seen = 0
        self._last_step_number = 0
        self._wired = False

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
        elif self._last_enemy_pos is not None and self._steps_since_seen <= 10:
            # Recently lost sight: search around last known position
            target = self._bias_toward_dead_end(self._last_enemy_pos)
            path = self._pathfinder.astar(my_pos, target)
            if path:
                chosen_move = path[0]
            else:
                chosen_move = self._greedy_toward(my_pos, target, internal_map)
        else:
            # Ghost hidden: sweep search
            self._sweep.invalidate_path()
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
        """If a dead end is within 5 BFS steps of pos, return that dead end.
        Otherwise return pos unchanged."""
        analysis = self._analyzer.get_analysis()
        if analysis is None:
            return pos
        dead_ends = analysis["dead_ends"]
        internal_map = self._map_memory.get_map()

        best_de = pos
        best_d = INF
        for de in dead_ends:
            d = _manhattan(pos, de)
            if d <= 5 and d < best_d:
                best_d = d
                best_de = de
        return best_de

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
```

- [ ] **Step 2: Run a full game to verify basic integration**

```bash
cd /home/ntdat/Documents/pacman/src && python arena.py --seek LAB2 --hide reference/LAB1/0 --no-viz --max-steps 50 2>&1 | head -20
```

Expected: Game runs to completion without crashes. Should show either Pacman wins or Ghost wins.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: add PacmanAgent orchestrator wiring all components"
```

---

### Task 9: Integration testing, edge cases, and cleanup

**Files:**
- Modify: `submissions/LAB2/agent.py` — any bug fixes found during testing
- No new files

**Purpose:** Run comprehensive tests across multiple scenarios and fix any issues. Verify all requirements from the spec are met.

- [ ] **Step 1: Test basic chase against reference/LAB1/0 ghost**

```bash
cd /home/ntdat/Documents/pacman/src && python arena.py --seek LAB2 --hide reference/LAB1/0 --no-viz --max-steps 200
```

Expected: Game completes. If ghost starts visible, Pacman should catch it within reasonable steps.

- [ ] **Step 2: Test fog-of-war sweep search**

```bash
cd /home/ntdat/Documents/pacman/src && python arena.py --seek LAB2 --hide reference/LAB1/0 --no-viz --max-steps 300 --pacman-obs-radius 5 --ghost-obs-radius 3
```

Expected: Game completes. Pacman sweeps the map systematically even when ghost is hidden.

- [ ] **Step 3: Test speed multiplier**

```bash
cd /home/ntdat/Documents/pacman/src && python arena.py --seek LAB2 --hide reference/LAB1/0 --no-viz --max-steps 100 --pacman-speed 2
```

Expected: No "exceeds maximum speed" errors. Pacman uses speed-2 moves effectively.

- [ ] **Step 4: Test map cache persistence (two consecutive matches)**

```bash
cd /home/ntdat/Documents/pacman/src && python -c "
from arena import Arena
# Match 1
a = Arena('LAB2', 'reference/LAB1/0', visualize=False, max_steps=10)
a.run()
# Match 2 — should load cached map
a2 = Arena('LAB2', 'reference/LAB1/0', visualize=False, max_steps=10)
a2.run()
print('Map cache test passed')
"
```

Expected: Both matches complete. Second match starts with full map knowledge from cache.

- [ ] **Step 5: Test against stronger reference ghosts**

```bash
cd /home/ntdat/Documents/pacman/src && for gid in 0 2 3 5; do
  echo "--- Testing against reference/LAB1/$gid ---"
  python arena.py --seek LAB2 --hide "reference/LAB1/$gid" --no-viz --max-steps 200 2>&1 | tail -1
done
```

Expected: Pacman wins or at least doesn't crash against each reference ghost.

- [ ] **Step 6: Fix any issues found and commit**

```bash
cd /home/ntdat/Documents/pacman && git add -A && git commit -m "fix: integration fixes from comprehensive testing"
```

- [ ] **Step 7: Final verification — run all test scenarios again**

Rerun steps 1-5 to confirm all pass after fixes.

- [ ] **Step 8: Final commit**

```bash
cd /home/ntdat/Documents/pacman && git add -A && git commit -m "test: final integration verification complete"
```

---

## Verification Checklist

After all tasks are complete, verify against every spec requirement:

| Requirement | How to verify |
|-------------|---------------|
| Pure algorithm, no ML | Code review: no ML imports or training |
| 6-ply minimax + alpha-beta | Code review: MinimaxEngine._depth = 6, alpha/beta parameters |
| A* for pathfinding | Code review: PathFinder.astar uses heapq + Manhattan heuristic |
| BFS for evaluation distances | Code review: PathFinder.bfs_dist / bfs_all_dists use deque BFS |
| Ghost probability estimation | Run with fog of war, verify exploration order includes dead ends/corners |
| Upper-half priority | Code review: GhostProbability multiplies upper-half cells by ×3 |
| Map cache persistence | Step 4 of Task 9: two consecutive matches, second uses cache |
| Turn-distance for speed | Code review: MinimaxEngine._evaluate uses ceil(bfs/pacman_speed) |
| Handle enemy_position is None | Run with fog of war |
| Speed multiplier support | Step 3 of Task 9: --pacman-speed 2 |
| GhostAgent unchanged | Diff against original GhostAgent code |
