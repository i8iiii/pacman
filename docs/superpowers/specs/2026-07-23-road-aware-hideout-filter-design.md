# Road-Aware Hideout Filter Design

## Purpose

Add the first road-aware hiding behavior on top of R08-A without restoring the
discarded R08-B–F strategy.

When Ghost spawns on one vertical side of the map, Pacman is assumed likely to
spawn on the opposite side. Pacman must gain vertical distance to cross from
DOWN to TOP or TOP to DOWN, so this increment treats vertical major roads as
the relevant approach roads. Horizontal roads do not exclude hideouts.

## Road visibility

- Preserve R08-A's definition of a major road: a maximal straight traversable
  run at least `ceil(2/3)` of the corresponding map dimension.
- Calculate one exact visibility set for each detected major road.
- A road's visibility set is the union of the cells Pacman can see while
  occupying every cell on that road.
- Visibility uses the Arena's existing cardinal, radius-limited,
  wall-blocked line-of-sight rules and the configured observation radius.
- Cache the set per road for the current match. Direction does not affect it:
  DOWN-to-TOP and TOP-to-DOWN traversal of the same road expose the same cells.

## Hideout exclusion and selection

- Mark every vertical major road as an opposite-side approach road. A vertical
  road meeting the two-thirds threshold necessarily reaches both the top and
  bottom regions of the map.
- Form the excluded-cell set by combining the visibility sets of the marked
  vertical roads.
- Before the existing R07 `select_hideout` call, remove every hideout candidate
  whose position is in the excluded-cell set.
- Run the unchanged R07 selector on the remaining candidates.
- Apply the filter only to normal unseen strategic selection. Existing visible
  escape and `HOT_UNSEEN` survival behavior retain their current priority.
- If no candidate remains, preserve the existing no-target behavior.

## Explicitly out of scope

This increment does not:

- exclude route cells or change path planning;
- predict Pacman's progress or arrival time;
- switch active roads as time passes;
- relocate a selected hideout because of a road transition;
- add a least-exposed fallback;
- carry road information between games.

## Diagnostics

- Retain the existing once-per-match `main_road_scan`.
- For each road, record its exact visible-cell list and whether it is marked as
  a vertical approach road.
- Record the combined excluded-cell set and the hideout positions rejected by
  the road filter.
- Keep complete coordinate lists in diagnostics so a road-to-visible-cell and
  excluded-hideout decision can be audited directly.

## Verification

- Horizontal roads never exclude a hideout.
- A hideout seen from any position on a vertical major road is excluded.
- Walls stop visibility and diagonal cells are not visible.
- Visibility at the configured radius boundary is included; cells beyond it
  are excluded.
- Reversing the assumed traversal direction does not change a road's
  visibility set.
- Changing structural cells between `0` and `-1` does not change roads or
  road visibility.
- With no qualifying vertical road, selection is identical to R07.
- With qualifying roads, the chosen hideout is exactly the R07 result over the
  filtered candidate collection.
- Visible and `HOT_UNSEEN` decisions remain unchanged.
