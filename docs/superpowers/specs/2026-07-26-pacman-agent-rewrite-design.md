# PacmanAgent Clean Rewrite — Design Spec

**Date:** 2026-07-26
**Scope:** Rewrite `PacmanAgent` only; keep existing `GhostAgent` unchanged.
**Approach:** Modular class-based design with persistent map cache across matches.

---

## Requirements

1. Pure algorithmic agent — no machine learning
2. 6-ply minimax with alpha-beta pruning (ghost visible + far away)
3. A* search for pathfinding to specific targets
4. BFS for exact shortest-path distances in minimax evaluation
5. Ghost probability estimation when hidden: detect dead ends, corners, pockets
6. Upper-half map priority during sweep search
7. Map memory persists across matches (module-level cache by map fingerprint)
8. Turn-distance adjustment for Pacman speed multiplier in evaluation

---

## Architecture

```
PacmanAgent (orchestrator)
├── MapMemory          — persistent, module-level cached
├── MapAnalyzer        — one-shot analysis per map
├── PathFinder         — A* pathfinding + BFS distance cache
├── GhostProbability   — probability distribution over hiding spots
├── MinimaxEngine      — 6-ply alpha-beta adversarial search
└── SweepPlanner       — systematic exploration order
```

### Component responsibilities

#### MapMemory

Module-level cache `_MAP_CACHE: dict[int, np.ndarray]` persists across arena matches.

- On **first step()** of every match: extract wall pattern from `map_state[map_state == 1]`, compute fingerprint (hash of wall positions tuple).
  - If `_MAP_CACHE[fingerprint]` exists: restore cached `internal_map` — we played this map before
  - Else: initialize fresh `internal_map` with walls = 1, everything else = -1 (unknown)
- Each `step()` merges visible cells (`map_state != -1`) into `internal_map`
- At **match end** (detected when `step_number` resets to 1 in a new match): save `internal_map` to `_MAP_CACHE[fingerprint]`
- Walls are always visible (even with fog of war), so fingerprint is always computable from step 1

#### MapAnalyzer

Takes known `internal_map` and produces structural analysis:

- `dead_ends`: set of `(r, c)` with exactly 1 valid neighbor
- `corners`: set of `(r, c)` with exactly 2 valid neighbors that are perpendicular (L-shaped)
- `pockets`: dict region_id → set of cells. Flood-fill from dead ends outward through cells with ≤ 2 exits
- `exit_counts`: dict `(r, c) → int` — number of valid neighbors
- `mid_row`: `height // 2` for upper-half queries
- `all_reachable`: set of all known empty cells

Recomputed incrementally as new cells are discovered. On matches where map is cached from previous runs, full analysis is available from step 1.

#### PathFinder

- `astar(start, goal) → list[Move]`: A* with Manhattan heuristic. Treats unknown (`-1`) as impassable.
- `bfs_all_dists(source) → dict[pos → distance]`: BFS distance from source to all reachable cells. Cached per source within a step.
- `manhattan(a, b) → int`: O(1) heuristic.

#### GhostProbability

When ghost is hidden and map is known enough:

1. Start uniform probability = 1.0 for all reachable known cells
2. Apply multiplicative weights:
   - Upper half (`row < mid_row`): ×3.0
   - Dead end: ×5.0
   - Corner: ×3.0
   - Pocket: ×2.0
   - Distance from Pacman: `1.0 + 1.0 / max(1, dist)` (ghost unlikely to be adjacent)
3. Normalize to sum = 1.0
4. Sort cells by descending probability → sweep order

#### MinimaxEngine

Invoked when ghost is visible and BFS distance > 2.

**Parameters:** max depth = 6 plies (3 Pacman + 3 Ghost moves), Pacman speed from config.

**Max node (Pacman's turn, depth d):**
- Generate valid actions: 4 single-step dirs + speed-2 combos (4 dirs × 2 steps)
- Sort by BFS distance (pac → ghost) ascending for alpha-beta pruning
- Alpha-beta cutoff: score ≥ beta → prune
- Return max score across children

**Min node (Ghost's turn, depth d):**
- Generate valid moves: UP, DOWN, LEFT, RIGHT, STAY
- Score each: `bfs_dist(pac, ghost) × 10 + exits × 3 + dead_end_bonus(30) + corner_bonus(15) + perpendicular_bonus(50)`
- Sort descending (ghost prefers high-score moves)
- Alpha-beta cutoff: score ≤ alpha → prune
- Return min score across children

**Terminal evaluation:**
- Manhattan distance ≤ 1 → `+100000 + depth` (Pacman captures)
- depth == 0 → `-(bfs_dist × 10 + ghost_exits × 3)` — negative so shorter distance = better for Pacman (maximizer)

**Turn-distance for speed:** `effective_turns = ceil(bfs_dist / pacman_speed)`. Used in evaluation to model speed advantage.

#### SweepPlanner

When ghost is hidden (unseen for ≥ 10 steps):

1. Update GhostProbability distribution over known cells
2. Pick highest-probability cell that is: known, reachable, not in recent-cooldown list (last 20 visited)
3. A* to that cell, return first move
4. Fallback: nearest frontier cell (known cell adjacent to unknown)

#### PacmanAgent (orchestrator)

```
step(map_state, my_position, enemy_position, step_number):
  1. Update MapMemory (first step: check cache, init map)
  2. If new cells discovered, re-run MapAnalyzer incrementally
  3. If ghost visible + manhattan ≤ 2:          A* direct chase
  4. If ghost visible + manhattan > 2:          6-ply MinimaxEngine
  5. If ghost recently seen (≤ 10 steps ago):   A* to last known position,
                                                 biased toward nearest dead end nearby
  6. If ghost hidden long-term:                 SweepPlanner
  7. Compute speed multiplier steps (speed-2 if path allows)
  8. Return (move, steps)
```

---

## Map Cache Persistence

```python
_MAP_CACHE: dict[int, np.ndarray] = {}

def _fingerprint(map_state: np.ndarray) -> int:
    """Hash wall pattern. Walls are always 1, so fingerprint is stable."""
    walls = tuple(int(x) for x in map_state.flat if x == 1)
    return hash(walls)
```

Flow:
1. Match 1, step 1: compute fingerprint → no cache hit → init fresh map → discover map during gameplay → save to cache at end
2. Match 2+, step 1: compute fingerprint → cache hit → restore full internal map → MapAnalyzer has full analysis immediately

---

## File Structure

```
submissions/LAB2/agent.py   (single file with all components)
├── Module-level helpers (manhattan, is_valid, etc.) + _MAP_CACHE
├── MapMemory class
├── MapAnalyzer class
├── PathFinder class
├── GhostProbability class
├── MinimaxEngine class
├── SweepPlanner class
├── PacmanAgent class (orchestrator)
└── GhostAgent class (unchanged, copied as-is)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| BFS for evaluation, A* for pathfinding | BFS exact for eval; A* faster for single-target pathfinding |
| Module-level map cache | Arena creates new agent per match; module cache survives |
| Ghost probability multiplicative weights | Dead ends > corners > pockets > upper half |
| Fixed 6-ply minimax depth | Consistent lookahead; alpha-beta makes it fast enough |
| Perpendicular bonus in ghost move scoring | Ghost moving perpendicular to Pacman is harder to corner |
| Sweep cooldown of 20 cells | Prevents oscillation between nearby targets |
| turn-dist = ceil(bfs/pacman_speed) | Correctly models speed advantage in evaluation |

---

## Testing Strategy

- Run against `example_student` ghost for basic chasing correctness
- Run against stronger reference ghosts for minimax quality
- Run with `--pacman-obs-radius 5 --ghost-obs-radius 3` for fog-of-war sweep
- Run with `--pacman-speed 2` for speed multiplier correctness
- Verify map cache: run two matches with same map, confirm second match analysis is instant
