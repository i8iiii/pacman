"""
Pacman topology analysis -- identifies map features from the seeker's perspective.

Concepts:
- trap_score: negative = dead-end (good for Pacman), positive = junction (good for Ghost)
- interception_value: how well a cell blocks ghost escape toward safe junctions
- cut_point: a corridor cell where Pacman can block the ghost's retreat
"""

from collections import deque
from environment import Move

PACMAN_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
INF = 10 ** 9


# ── precomputation utilities ─────────────────────────────────────

def precompute_valid_positions(map_state):
    """Return set of (r, c) positions that are walkable (value == 0)."""
    import numpy as np
    h, w = map_state.shape
    return {(r, c) for r in range(h) for c in range(w) if map_state[r, c] == 0}


def is_valid_fast(pos, valid_positions):
    """O(1) check if pos is walkable using precomputed set."""
    return pos in valid_positions


def manhattan_distance(a, b):
    """Manhattan distance between two (r, c) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── existing topology analysis ───────────────────────────────────

def _valid(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

def _neighbors(pos, map_state):
    return [(pos[0] + m.value[0], pos[1] + m.value[1])
            for m in PACMAN_MOVES
            if _valid((pos[0] + m.value[0], pos[1] + m.value[1]), map_state)]

def _exits(pos, map_state):
    return len(_neighbors(pos, map_state))

def classify_cell(exits):
    if exits <= 1:
        return "DEAD_END"
    if exits == 2:
        return "CORRIDOR"
    return "JUNCTION"

def junction_distance(pos, map_state, max_depth=10):
    """Distance to nearest junction (>=3 exits)."""
    if not _valid(pos, map_state):
        return max_depth + 1
    if _exits(pos, map_state) >= 3:
        return 0
    q = deque([(pos, 0)])
    seen = {pos}
    while q:
        cur, d = q.popleft()
        if _exits(cur, map_state) >= 3:
            return d
        if d >= max_depth:
            continue
        for n in _neighbors(cur, map_state):
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return max_depth + 1

def local_area(pos, map_state, limit=6):
    """Number of reachable cells within *limit* BFS steps."""
    if not _valid(pos, map_state):
        return 0
    q = deque([(pos, 0)])
    seen = {pos}
    while q:
        cur, d = q.popleft()
        if d >= limit:
            continue
        for n in _neighbors(cur, map_state):
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return len(seen)

# Weights: negative = advantageous for Pacman (ghost in trappable area)
TRAP_WEIGHTS = {
    "dead_end":       -900,
    "corridor":        100,
    "junction_3":      600,
    "junction_4":      800,
    "local_area":       15,
    "at_junction":     400,
    "junction_dist":  -100,
}

def trap_score(pos, map_state):
    """Score a cell from Pacman's perspective.
    Lower (more negative) = easier to trap the ghost here.
    """
    if not _valid(pos, map_state):
        return 0
    exits = _exits(pos, map_state)
    ctype = classify_cell(exits)
    jd = junction_distance(pos, map_state)
    area = local_area(pos, map_state)

    score = 0
    if ctype == "DEAD_END":
        score += TRAP_WEIGHTS["dead_end"]
    elif ctype == "CORRIDOR":
        score += TRAP_WEIGHTS["corridor"]
    else:
        score += (TRAP_WEIGHTS["junction_4"] if exits >= 4
                  else TRAP_WEIGHTS["junction_3"])

    score += min(area, 40) * TRAP_WEIGHTS["local_area"]

    if jd == 0:
        score += TRAP_WEIGHTS["at_junction"]
    else:
        score += min(jd, 8) * TRAP_WEIGHTS["junction_dist"]

    return score

def build_trap_map(map_state):
    """Return {pos: int} trap scores for all walkable cells."""
    h, w = map_state.shape
    tmap = {}
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if _valid(pos, map_state):
                tmap[pos] = trap_score(pos, map_state)
    return tmap
