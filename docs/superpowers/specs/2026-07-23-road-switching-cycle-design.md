# Road-Switching Cycle Design

## Purpose

Extend the current per-road visibility filter with a repeating prediction of
which main road Pacman is likely to traverse next. Only the current stage's
road visibility excludes hideouts. Every stage lasts five completed game
turns.

## Spawn-mirrored cycle

The first stage always assumes Pacman crosses vertically from the side opposite
Ghost toward Ghost's spawn side.

For a Ghost that spawned in the top half:

1. vertical roads, BOTTOM to TOP;
2. the connected TOP horizontal road;
3. connected vertical roads, TOP to BOTTOM;
4. the connected BOTTOM horizontal road;
5. repeat from stage 1.

For a Ghost that spawned in the bottom half:

1. vertical roads, TOP to BOTTOM;
2. the connected BOTTOM horizontal road;
3. connected vertical roads, BOTTOM to TOP;
4. the connected TOP horizontal road;
5. repeat from stage 1.

Turns 1–5 use stage 1. Stage 2 begins on turn 6, stage 3 on turn 11,
stage 4 on turn 16, and the next cycle begins on turn 21.

## Road connectivity and deterministic selection

- Two roads are connected when their road-cell sets intersect.
- Initial stage 1 contains every vertical major road.
- Stage 2 is one horizontal road that intersects a stage-1 vertical road. Pick
  the road closest to Ghost's spawn side: smallest row for a top Ghost, largest
  row for a bottom Ghost. Break a complete tie by road ID.
- Stage 3 contains every vertical road intersecting the selected stage-2
  horizontal road.
- Stage 4 is one horizontal road intersecting a stage-3 vertical road. Pick the
  road closest to the side opposite Ghost: largest row for a top Ghost,
  smallest row for a bottom Ghost. Break a complete tie by road ID.
- The four static stage definitions are built once per match from cached R08-B
  road visibility.
- If the next stage has no connected road, keep the current stage active rather
  than switching to an unconnected prediction.

## Timing and active exclusion

- Store the first Ghost step number as the match start.
- Calculate `elapsed_turns = current_step - match_start_step`.
- The requested stage is `(elapsed_turns // 5) % 4`.
- At a valid stage change, release every previously active road and exclude
  hideouts only from the newly active road or road set.
- Recompute the active eligible and excluded candidate collections from the
  cached per-road visibility; do not recompute line of sight.
- Road switching affects normal unseen strategic selection only. Visible and
  `HOT_UNSEEN` behavior retain priority.

## Hideout behavior

- During stage 1 at match start, preserve the existing R08-B behavior: apply
  the active vertical-road filter and run the normal R07 hideout selector.
- At a later stage change, preserve the current selected hideout if it remains
  outside the new active visibility set. Ghost stays there if already arrived
  or continues its existing route if still travelling.
- If the selected hideout becomes exposed, discard that target and choose the
  closest reachable safe, uncompromised hideout from Ghost's current position.
- For equal route distance, use existing static R07 hideout quality, then
  deterministic coordinates.
- Use structural `0` and `-1` cells equally for reachability.
- If no safe reachable hideout exists, preserve the existing no-target
  behavior.
- Route cells remain unrestricted. This phase filters destinations, not paths.

## Components

- `roads.py` owns the four-stage road cycle, connectivity, stage timing, active
  visibility selection, candidate filtering, and closest-safe relocation
  selection.
- `controller.py` owns match elapsed time, applies stage changes only during
  normal strategic selection, preserves safe targets, and routes to a newly
  selected target when required.
- `diagnostics.py` records the current cycle and active exclusion without
  changing matrix symbols.

## Diagnostics

- Retain `main_road_scan` and its complete cached per-road visibility.
- Add one `road_cycle_built` event per match with Ghost spawn side, fixed
  five-turn stage duration, and the four ordered stage road-ID sets.
- Add `road_stage_changed` on initial activation and every valid switch with:
  previous/requested/active stage, elapsed turns, released road IDs, active
  road IDs, active excluded-cell count, excluded-hideout count, selected
  hideout, and whether the selected hideout remained safe.
- Both agent-map formats retain complete per-road visibility and add current
  stage index, active road IDs, active excluded cells, and currently excluded
  hideouts below the matrix.
- Do not add timing-based pursuit, fallback, or route-exposure events.

## Verification

- Top and bottom Ghost spawns produce mirrored stage order.
- Turns 1, 5, 6, 10, 11, 15, 16, 20, and 21 resolve to the correct stage.
- Previous roads are no longer active after a successful switch.
- Each horizontal stage selects only a directly intersecting horizontal road on
  the correct side.
- Each vertical return stage contains only vertical roads connected to the
  preceding horizontal road.
- A missing connected next road preserves the current stage.
- A safe selected hideout remains selected across a switch.
- An exposed selected hideout is replaced by the closest safe reachable
  uncompromised hideout; static R07 quality and coordinates break distance
  ties.
- Horizontal-stage exclusion uses only the selected horizontal road's cached
  visibility.
- Visible and `HOT_UNSEEN` decisions remain unchanged.
- Routes may cross active visibility because route exclusion remains out of
  scope.
- Diagnostics and decisions reset on a new match and remain aligned per step.
