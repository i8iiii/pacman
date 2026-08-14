# Fog-Aware Hideout Ranking

## Goal

Rank only structurally valid hideouts so the Hide agent prefers locations that
are far from the ghost, far from known or believed Pacman positions, close to
map corners, and difficult for Pacman to see.

## Eligibility

`scan_hideouts()` remains the sole structural safety gate. The ranker only
receives its candidates and excludes unreachable ones. It never selects an
arbitrary traversable corridor cell.

For routing, targeting, and line-of-sight checks, map values `0` and `-1` are
traversable. Only `1` is a wall and blocks both movement and vision.

## Ranking

Add a pure `rank_hideouts()` helper in `hide_agent.concealment`. It accepts the
map, ghost position, hideout candidates, optional visible Pacman position,
optional possible Pacman positions, and excluded positions.

The helper computes structural BFS distances from the ghost and, when enemy
information is available, from Pacman or all possible Pacman positions. It
scores each reachable, non-excluded candidate with:

`self_distance + enemy_distance + corner_affinity - 2 * visibility_count`

The enemy-distance and visibility terms are omitted when there is no enemy
information. Corner affinity is larger nearer a grid corner. Visibility uses
cardinal line of sight at most five cells away, with walls as the only blocker.

Candidates are returned in descending score order, with coordinates as a
stable final tie-breaker.

## Controller Integration

Pass `enemy_position` through the Hide controller target-selection path.
Select the first ranked candidate for a new anchor. During rotation, exclude
the current anchor and fall back to it only when there is no other reachable
candidate. Retain the current detour eligibility window, but rank eligible
detours instead of selecting solely by distance.

## Tests

Add unit coverage for fog traversability and visibility, wall blocking,
unreachable exclusions, known-enemy safety ranking, deterministic no-enemy
ranking, and controller anchor/detour integration.
