# Algorithmic Seeker Agent — Design Spec

**Date**: 2026-07-26
**Status**: Implemented
**Goal**: Pure algorithmic PacmanAgent with map analysis, minimax chase, and fog-of-war search.

## Architecture

Three decision modes based on ghost visibility:

1. **Ghost visible** → 6-ply Minimax + Alpha-Beta pruning with ghost behavior modeling
2. **Ghost recently seen (≤15 steps)** → A* to last known position, check nearby dead-ends
3. **Ghost hidden** → Sweep search: methodical upper-half exploration ordered by hiding probability

## Components

| Component | Purpose |
|-----------|---------|
| Map analysis | Detect dead-ends, corners, pocket regions from map layout |
| Sweep search | Ordered list of frontier cells: upper-half + dead-ends + corners first |
| Minimax (6-ply) | Alpha-beta pruning with ghost movement scoring |
| Ghost scoring | distance×10 + exits×3 + dead-end bonus + alignment bonus |
| A* pathfinding | Navigation to any target cell |
| CPU budget | Burn cycles to hit 0.8-1.0s per step requirement |
| Speed-2 | Double-step on straight paths when possible |

## Ghost Behavior Modeling

The minimax evaluates ghost moves identically to the Hide-Agent's own scoring:
- Maximizes BFS distance from Pacman (×10 weight)
- Prefers high-exit junctions (×3 weight)
- Dead-end bonus (+30), corner bonus (+15)
- Perpendicular escape when aligned with Pacman (+50)

## Sweep Search Priority

| Priority | Cell Type | Score |
|----------|-----------|-------|
| 1 | Upper-half dead-end | +150 |
| 2 | Upper-half corner | +130 |
| 3 | Upper-half pocket | +115 |
| 4 | Upper-half frontier | +100 |
| 5 | Lower-half frontier | +0 |

## Constraints

- No ML/pytorch dependencies
- CPU: >0.8s, <1.0s per step (enforced via burn loop)
- Fog-of-war: both agents see within 5 cells (cross pattern)
- Map: 21×21 classic Pacman layout
