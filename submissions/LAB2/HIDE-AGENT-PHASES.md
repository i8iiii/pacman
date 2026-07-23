# Hide Agent: Implementation Phases

Status: **R08-B complete — vertical-road visibility now filters strategic
hideouts; no implementation phase is active**.

## Control rules

- Only the user can start a phase by explicitly naming its ID, for example: `START P00`.
- Only that phase may be implemented. Finishing it does not start the next phase.
- Every phase must be checked through the Arena and must produce its listed diagnostics before it can be called complete.
- After each phase, report the changed files, Arena result, and a short log excerpt, then stop and wait.
- Do not commit unless the user explicitly asks for a commit.
- If a chosen phase depends on unfinished work, explain the dependency and wait; do not implement it automatically.
- Terrain and pursuit state exist only for the current game. The agent never reads previous logs or carries knowledge into another game.
- Terrain values never encode Pacman's occupancy. Only the current `enemy_position` and later pursuit-state phases may represent where Pacman is or could be.

## Phase board

| ID | Status | Purpose | Completion check | Required diagnostics |
|---|---|---|---|---|
| `P00` | Complete | Add a legal baseline Hide action and safe logging. | Every call returns a legal move; logging failure cannot crash the agent. | `match_start`, `decision` |
| `P01` | Complete; revised by P04 | Record match-local map diagnostics without carrying observations across games. | Arbitrary map dimensions work; both agent-map files retain every step. P04 replaces accumulated observations with the raw current observation so an old `0` is never mistaken for current evidence about Pacman. | `match_state_reset`; per-step snapshots in `hide-agent-map.txt` and `hide-agent-map.jsonl` |
| `P02` | Complete | Implement exact movement, line-of-sight, and capture geometry for Hide and Pacman. | Arena checks cover walls, occlusion, diagonal blindness, Pacman's straight one/two-cell moves, and capture distance. | `geometry_summary` |
| `P03` | Complete | Detect and rank campsite candidates from known topology. | The topology scan records structural exits, occlusion, capture approaches, loops, and warning distance for junctions, and exposes the universal four-way safe campsites consumed by P04. | `campsite_scan`, `campsite_selected`; campsite lists in each map snapshot |
| `P04` | Complete | Finalize the campsite model and add the complete no-sight `SCOUT`/`CAMP` controller. | `safe_campsites` contains only structural four-way junctions, treating `0` and `-1` as traversable for identification and planning. Hide selects and locks the best safe campsite, plans directly through all non-wall cells, recalculates each turn, and executes only the currently visible adjacent first move. It does not accumulate old `0` observations or use terrain values as evidence about Pacman's current location. Frontier scouting is removed. Hide enters `CAMP` on arrival and stays there while Pacman is unseen. Both agent-map formats append the raw current observation, safe list, and selected campsite every step. | `scout_target`, `route_planned`, `route_replanned`, `state_changed`, `scout_move`, `camp_hold`; `safe_campsites` and `selected_campsite` in each map snapshot |
| `P05` | Merged into P04 | No separate implementation. | The former no-sight state-controller work is included in P04 so scouting, selection, arrival, and camping can be verified end to end. | None |
| `P06` | Complete | Escape when Pacman is visible and Hide is already at a campsite or useful junction. | A four-way campsite is always eligible. A T-junction is eligible only when its missing branch is opposite Pacman's approach. Hide evaluates only the two perpendicular branches. It enumerates Pacman's legal `STAY` and straight one/two-cell endpoints, prefers branches safe from every endpoint, then ranks continuation by worst-case distance, freedom from a trapped dead end, loop/reconnection, another junction, and escape depth. It randomizes only among branches tied on the complete ranking. If every branch is capturable, Hide takes the branch with the best forced-escape ranking rather than staying. P06 ends after this visible move; post-escape pursuit remains P08. | `visible_at_camp`, `escape_branch_chosen` |
| `P07` | Complete | Escape when Pacman is visible and Hide is not at a campsite. | Hide evaluates `0` and `-1` as equally traversable and ranks each legal action in this strict order: guaranteed survival against every Pacman endpoint, worst-case separation, broken lines of sight, freedom from a bad dead end, movement over an equally ranked `STAY`, destination quality, and continuation depth. Destination quality prefers a reachable safe four-way campsite, then a T-junction containing both perpendicular escapes for Pacman's current approach. The destination is recalculated every visible step and never overrides an earlier survival criterion. Exact ties alone are randomized. | `visible_while_mobile`, `escape_target_chosen` |
| `P08` | Complete | Add the first `HOT_UNSEEN` pursuit model: assume Pacman follows Hide's ordered trail. | After a P06/P07 escape breaks sight, Hide tracks only Pacman positions that advance along its departed cells. The first unseen update permits zero/one/two-cell progress; later updates require one/two-cell progress, and no two-cell action may cross a corner. Contradicted followers are removed. While any remain, Hide ranks legal moves by guaranteed survival, worst-case separation, occlusion, dead-end safety, non-reversal, movement, safe-campsite progress, and continuation depth. An empty follower set returns control to P04. | `hot_unseen_entered`, `follower_updated`, `hot_move`, `follower_invalidated` |
| `P09` | Complete | Strengthen `HOT_UNSEEN` against intercepting or searching Pacmen. | Hide considers junction/campsite interception; if simple predictions fail, it keeps a reachable set of possible Pacman cells. `HOT_UNSEEN` ends only on a new sighting or arrival at a campsite safe from every plausible next-turn capture. | `interceptor_updated`, `belief_rebuilt`, `hot_unseen_exited` |
| `R00` | Complete | Compute exact hideout visibility footprints. | Every traversable cell has the complete deterministic set of radius-limited observer cells; current movement remains unchanged. | `hideout_scan` visibility summary |
| `R01` | Complete | Detect terminal pockets and nested articulation gates. | Branch-gate analysis distinguishes bent terminal pockets, nested gates, and maps without a wider backbone; current movement remains unchanged. | `hideout_scan` terminal-pocket summary |
| `R02` | Complete | Detect reconnecting side branches and general fallbacks. | Loop branches cannot be mislabeled terminal; every reachable map has a deterministic concealment fallback. | `hideout_scan` class and rejection summary |
| `R03` | Complete | Add exact inspection depth and soft spawn-band discovery distance. | Pacman speed-two timing, entrance visibility, inspection depth, and opposite-band distance are correct without changing decisions. | `hideout_scan` metric summary |
| `R04` | Complete | Select and lock the deterministic best hideout. | The approved class, four-step admission, lexicographic ranking, compromise exclusion input, and target lock choose exactly one reachable hideout. | `hideout_selected`; hideout lists and metrics in both agent-map formats |
| `R05` | Complete | Replace pre-detection campsite routing and waiting with concealed hideouts. | With Pacman unseen and no pursuit seed, Hide routes to the locked hideout, arrives, and stays; four-way junctions are no longer waiting targets. | `route_planned`, `scout_move`, `hideout_arrived`, `decision` |
| `R06` | Complete | Add match-local hideout compromise and reselection. | A sighting at or en route to a selected hideout compromises it, selects a different target after sight loss, never reuses it that game, and resets on a new match. | `hideout_compromised`, `hideout_selected` |
| `R07` | Complete | Retarget P08/P09 and finish the strategic campsite replacement. | Broad survival remains first; interception and hot guidance use uncompromised hideouts; P06/P07 retain tactical junctions; hot exit and diagnostics use hideout semantics end to end. | `interceptor_updated`, `hot_move`, `hot_unseen_exited`, final hideout agent-map state |
| `R08` | In progress; stopped after R08-B | Add road-aware strategic hiding in user-approved increments. | R08-A detects roads. R08-B excludes hideouts visible from vertical DOWN↔TOP approach roads. No timing, route exclusion, relocation, or fallback is implemented. | `main_road_scan`; road visibility and excluded-hideout map fields |
| `R08-A` | Complete | Detect deterministic major straight roads. | Exact two-thirds thresholds, maximal-run splitting, dimensions, intersections, arbitrary sizes, and `0`/`-1` equivalence pass without changing movement. | `main_road_scan` |
| `R08-B` | Complete | Cache the cells visible while traversing each road and exclude hideouts visible from vertical DOWN↔TOP roads. | Horizontal roads never exclude hideouts; exact LOS, walls, radius, `0`/`-1`, filtered R07 selection, diagnostics, and unchanged route/pursuit scope pass. | Expanded `main_road_scan`; `road_visibility` and `road_excluded_hideouts` in both agent-map formats |
| `R08-C` | Not designed | Reserved for the next user-approved road-aware behavior. | The user defines and starts this phase before implementation. | Not defined |
| `R08-D` | Not designed | Reserved for a later user-approved road-aware behavior. | The user defines and starts this phase before implementation. | Not defined |
| `R08-E` | Not designed | Reserved for a later user-approved road-aware behavior. | The user defines and starts this phase before implementation. | Not defined |
| `R08-F` | Not designed | Reserved for a later user-approved road-aware behavior. | The user defines and starts this phase before implementation. | Not defined |
| `P10` | Not started | Add emergency behavior and enforce the per-move time limit. | No-hideout, no-route, all-dangerous, exception, and deadline cases still return a legal move within budget. | `emergency`, `deadline_fallback`, `decision_error` |
| `P11` | Not started | Validate and tune the completed strategy without adding unapproved features. | Seeded games on varied unknown maps against follower, interceptor, sweeper, belief-based, random, and A/B/C reference seekers report survival, wins, invalid moves, and timing. | `match_end`, `benchmark_summary` |

## Dependency order

`P00 → P01 → P02 → P03 → P04 → P06/P07 → P08 → P09 → R00 → R01 → R02 → R03 → R04 → R05 → R06 → R07 → R08-A → R08-B → R08-C → R08-D → R08-E → R08-F → P10 → P11`

`P06` and `P07` are separate so each visible-escape situation can be tested and accepted independently.

`P05` is retained only as a historical ID; all of its work is part of P04.

## P07 implementation record

P07 was implemented and verified through these checkpoints in order.

### P07-A — Mobile survival ranking

- Add a phase-owned mobile-escape module rather than expanding the controller
  with pathfinding and tactical calculations.
- Enumerate every structurally legal one-cell action plus `STAY`, treating `0`
  and `-1` identically.
- Evaluate each endpoint against the same Pacman `STAY` and straight
  one/two-cell endpoints used by P06.
- Rank guaranteed survival, worst-case distance, broken Pacman lines of sight,
  and freedom from a trapped continuation before considering any destination.
- Let movement beat an otherwise equal `STAY`; randomize only actions tied on
  the complete ranking.
- Verify with small deterministic maps that a guaranteed move beats a
  capturable one, greater worst-case separation wins, an occluding turn wins
  after equal separation, a non-trapped branch wins after equal visibility,
  and `STAY` is retained only when strictly safer or no move exists.

### P07-B — Dynamic destination guidance

- From each still-tied safe continuation, search structurally without using
  remembered observations and without treating `-1` as blocked.
- Prefer a reachable safe four-way campsite. If none is reachable through that
  continuation, consider T-junctions whose two perpendicular branches exist
  for Pacman's current approach direction.
- Use destination kind and remaining route length only after all immediate
  survival criteria. Prefer greater continuation depth after destination
  quality.
- Recalculate both the useful T-junction set and the selected destination on
  every visible step; do not lock a P07 route.
- Verify that a farther safe campsite can guide an otherwise tied action over
  a useful T-junction, a changed Pacman approach invalidates an old T target,
  and destination progress never defeats a safer immediate action.

### P07-C — Controller and diagnostics integration

- Preserve P06 as the first visible-Pacman handler. Call P07 only when P06
  returns no junction escape.
- Clear the no-sight route when P07 takes control and return P07's selected
  legal action instead of `visible_behavior_not_implemented`.
- Write `visible_while_mobile` with Hide position, Pacman position, and current
  approach direction.
- Write `escape_target_chosen` with the chosen move, its complete rank,
  exact-tie alternatives, target kind/position/distance when present, and a
  reason when no target is usable. Do not duplicate the matrix or campsite
  lists already present in the agent-map diagnostics.
- Run an Arena game that reaches the mobile-visible branch. Confirm both P07
  events occur, P06 events still occur at eligible junctions, every decision
  is legal, the log resets per game, and per-step runtime remains recorded.

### P07 completion evidence

- Six fresh deterministic verification scenarios covered survival ranking,
  line-of-sight escape, safe-campsite preference, dynamic T orientation,
  controller integration, and preservation of P06 priority.
- A stochastic Arena game against reference seeker B entered P07 on steps 2,
  3, and 4. It selected legal guaranteed-safe moves `UP`, `UP`, and `LEFT`;
  both required P07 events were written each time.
- P07 runtimes in that game were 6.813 ms, 9.994 ms, and 7.237 ms. The complete
  Hide step remained below 21 ms.
- Pacman caught Hide on step 5 after sight was broken and the current no-sight
  controller routed back toward the old campsite. Retaining pursuit after
  sight loss is intentionally deferred to P08.

## Approved P08 design

This section records the design boundary used while P08 was active. P09's
interceptor and full reachable-position belief model were then out of scope.

### Pursuit state and data flow

- P06 and P07 remain the only visible-Pacman decision makers. After selecting a
  visible escape, the controller records Pacman's observed position, Hide's
  position, the chosen move, and the expected Hide endpoint as a pursuit seed.
- The initial ordered chase trail contains every cardinal cell from Pacman's
  observed position through Hide's departure cell and the endpoint selected by
  the visible escape. It does not store terrain or previous map observations.
- If the next step is unseen and Hide reached the expected endpoint, P08 enters
  `HOT_UNSEEN`. Its initial follower set advances zero, one, or two indices
  from Pacman's seed position.
- On later hot updates, every follower must advance one or two trail indices.
  A two-index advance is legal only when both trail edges have the same
  cardinal direction; Pacman cannot cross a corner with an L-shaped action.
- Hide's current position is always the last committed trail cell. For each
  proposed moving action, append the proposed endpoint to that candidate trail;
  `STAY` leaves the trail unchanged. Then predict the simultaneous follower
  advances. The chosen trail, follower set, and expected Hide endpoint remain
  pending until the following call confirms that endpoint.
- A follower that reaches Hide's candidate endpoint is evaluated as a capture.
  Therefore `STAY` cannot make pursuit disappear merely by withholding a new
  trail cell.
- Remove a follower when its trail advance is illegal, its cell would be in
  Hide's current line of sight despite `enemy_position` being `None`, or it
  would already have captured Hide even though the match continued.
- If every follower is removed, write `follower_invalidated`, leave
  `HOT_UNSEEN`, and let P04 make the current no-sight decision.
- A new sighting invalidates the old follower model before P06/P07 acts. The
  resulting visible escape records a fresh seed. A new match clears both
  active and pending pursuit data.

### HOT_UNSEEN movement

- Evaluate every structurally legal one-cell action plus `STAY`, treating `0`
  and `-1` identically.
- Rank each action in this strict order:
  1. safe from capture by every predicted next follower;
  2. greatest worst-case Manhattan separation;
  3. hidden from the greatest number of predicted followers;
  4. freedom from a trapped dead-end continuation;
  5. not reversing onto Hide's previous trail cell;
  6. movement over an otherwise equal `STAY`;
  7. progress toward a reachable safe four-way campsite;
  8. greater continuation depth.
- Randomize only actions tied on the complete rank. Destination quality never
  overrides an earlier survival criterion.
- Recalculate the decision every hot step. A campsite guides tied safe moves
  but does not end `HOT_UNSEEN` while a follower remains.
- Do not explicitly target T-junctions in P08 because different unseen follower
  positions can imply different approach directions. That uncertainty belongs
  to P09.

### Module boundary and diagnostics

- A phase-owned `pursuit.py` module owns the visible seed, ordered trail,
  follower indices, pending transition, contradiction filtering, and hot move
  ranking. The controller owns only state dispatch and diagnostic calls.
- `hot_unseen_entered` records the visible seed, escape move, initial trail,
  and initial follower positions.
- `follower_updated` records the previous and resulting follower positions,
  with removals grouped by visibility contradiction, capture contradiction,
  illegal advance, or stationary-after-entry.
- `hot_move` records the selected action, complete rank, exact-tie actions,
  predicted next followers, and safe-campsite guidance.
- `follower_invalidated` records whether the cause was a new sighting, an empty
  follower set, or an expected-endpoint mismatch.
- These logs do not duplicate terrain or campsite lists already present in the
  agent-map diagnostics.

### P08 acceptance checks

- Entry works after both P06 and P07 visible escapes.
- The first unseen update includes zero/one/two-cell followers; subsequent
  updates exclude stationary followers.
- A speed-two follower cannot cross a trail corner.
- `STAY` cannot empty the follower model merely because the trail was not
  extended.
- Visible-but-unseen, already-capturing, structurally illegal, and
  stationary-after-entry predictions are removed for the correct reason.
- Immediate survival always outranks safe-campsite progress, and an equally
  ranked move beats `STAY`.
- P04 cannot take control while at least one valid follower remains.
- New sightings and new matches clear the old pursuit model.
- A reproduced version of P07's step-5 Arena failure emits `hot_move` and does
  not route back along the P04 path while the follower model is valid.
- All four P08 diagnostics appear when applicable, every returned move is
  legal, and `decision.runtime_ms` remains present.

## P08 implementation checkpoints

Implement and verify these checkpoints in order. Do not begin P09.

### P08-A — Visible seed and ordered trail

**Files:** create `submissions/LAB2/hide_agent/pursuit.py`.

- Add immutable pursuit records plus a match-local `PursuitTracker`.
- `record_visible_escape(map_state, ghost_position, pacman_position,
  escape_move, step_number)` replaces any old seed and records the expected
  Hide endpoint.
- Build a cardinal ordered trail from Pacman's visible position through Hide's
  position and the selected endpoint. Reject a seed if the visible positions
  are not cardinally aligned or the selected endpoint is not a legal adjacent
  structural cell.
- `reset()` clears the visible seed, active follower state, and pending hot
  transition.
- Verify cardinal horizontal/vertical seeds, a perpendicular escape corner,
  rejected malformed seeds, endpoint inclusion, and complete reset.

### P08-B — Follower transition and contradiction filtering

**Files:** modify `submissions/LAB2/hide_agent/pursuit.py`.

- `observe_unseen(map_state, ghost_position, observation_radius,
  capture_distance)` verifies the expected Hide endpoint before committing a
  seed or pending hot transition.
- On entry, expand the seed follower by zero, one, and two trail indices. On
  later commits, accept only the one/two-index predictions produced by the
  chosen hot move.
- Reject a two-index transition when its two trail edges have different
  directions.
- Filter follower indices whose cells are structurally illegal, visible from
  Hide despite absent `enemy_position`, or already within capture distance.
- Return an update record containing status, previous/resulting follower
  positions, grouped removal reasons, and invalidation reason. An empty result
  clears active pursuit.
- Verify first-update `STAY`, later no-`STAY`, no L-shaped speed-two advance,
  each contradiction reason, endpoint mismatch, empty-set invalidation, and
  distinct indices when a trail revisits a coordinate.

### P08-C — Survival-first hot movement

**Files:** modify `submissions/LAB2/hide_agent/pursuit.py`.

- `choose_hot_move(map_state, ghost_position, safe_campsites,
  observation_radius, capture_distance)` requires an active follower state and
  evaluates every structural cardinal action plus `STAY`.
- A moving candidate appends its endpoint to its candidate trail; `STAY` keeps
  the current trail, whose last cell is already Hide's current position.
- Advance each current follower one/two indices along the candidate trail,
  respecting the no-L rule, and evaluate capture, distance, visibility,
  dead-end continuation, reversal, movement, nearest reachable safe campsite,
  and continuation depth in the approved order.
- Store only the selected candidate's expected Hide endpoint, trail, and next
  followers as pending. Randomize only complete-rank ties.
- Verify every ranking boundary independently, confirm a campsite never
  defeats immediate survival, confirm `STAY` cannot erase pursuit, and confirm
  a pending state is not committed before endpoint observation.

### P08-D — Controller state and diagnostics

**Files:** modify `submissions/LAB2/hide_agent/controller.py`; reuse
`submissions/LAB2/hide_agent/diagnostics.py`.

- Add `HOT_UNSEEN` dispatch and one `PursuitTracker` owned by the controller.
- On every new match, reset the tracker. On a new sighting, invalidate old hot
  state, let P06/P07 choose first, and record that selected visible escape as
  the new seed.
- When Pacman is unseen, call `observe_unseen` before P04. If pursuit remains
  active, clear the P04 route, choose and return `hot_move`, and do not enter
  `SCOUT` or `CAMP`. If pursuit invalidates, continue through the existing P04
  branch in the same step.
- Add only the approved `hot_unseen_entered`, `follower_updated`, `hot_move`,
  and `follower_invalidated` records; keep diagnostic failure non-fatal and do
  not duplicate agent-map terrain/campsite data.
- Verify entry from both P06 and P07, new-sighting replacement, match reset,
  endpoint mismatch, P04 suppression while hot, P04 resumption after
  invalidation, required event fields, and unchanged P06/P07 priority.

### P08-E — Arena gate and phase closure

**Files:** update `submissions/LAB2/HIDE-AGENT-PHASES.md` only after the gate
passes. Arena continues writing the existing files under
`submissions/LAB2/debug/`.

- Compile the Hide package and run all deterministic P08 contract scenarios.
- Run stochastic Arena games until at least one P06/P07 escape transitions
  into `HOT_UNSEEN`.
- Confirm the first unseen action is `hot_move`, not a P04 `scout_move` or
  `camp_hold`; the chosen move is legal; follower updates match the trail; the
  log resets once per game; map snapshots remain one per step; and all decision
  runtimes are recorded.
- Reproduce or closely match the P07 step-5 failure and confirm Hide does not
  immediately reverse along the old campsite route while a follower remains.
- Record the actual Arena outcome and any remaining P09-scoped limitation,
  mark P08 complete, then stop.

### P08 completion evidence

- Eighteen deterministic contract scenarios covered seed construction, reset,
  first/later follower advancement, no-L speed-two movement, every
  contradiction filter, endpoint mismatch, distinct loop indices, hot ranking,
  pending-state timing, `STAY`, P06/P07 entry, P04 suppression/resumption, new
  sightings, match reset, and corrected pending-follower diagnostics.
- A stochastic Arena game against reference seeker B entered P08 after a
  guaranteed P06 `DOWN` escape on step 22. On step 23, Hide remained
  `HOT_UNSEEN` and selected another guaranteed `DOWN` move; no `scout_move` or
  `camp_hold` occurred on that step.
- Step 24 reacquired Pacman. `follower_invalidated` correctly reported the
  pending post-move follower indices 1 through 4 before P06 handled the new
  sighting.
- The Arena produced 24 legal decisions and 24 map snapshots with one
  `match_start`. The maximum complete Hide-step runtime was 19.512 ms.
- Reference seeker B caught Hide on step 24 after reacquiring it. Modeling
  interception and search behavior beyond a trail-following Pacman remains
  explicitly scoped to P09.

## Current gate

`P00` through the merged `P04`, `P06` through `P09`, and `R00` through `R07`
are complete. R08-A through R08-F are complete. No implementation phase is
active. Wait for the user to explicitly start `P10`, `P11`, or another named
phase.

## Approved P09 design

P09 remains the completed broad-belief foundation. P10 emergency/deadline
behavior and P11 tuning remain out of scope.

### Broad belief lifecycle

- P09 keeps a match-local set of every plausible Pacman position. This broad
  set is the safety authority; P08's ordered follower set remains only a
  likely-case tie-breaker.
- A visible Pacman observation replaces the old belief with one exact position.
  On the first unseen turn, expand that position through every legal `STAY` and
  straight one/two-cell Pacman action. Repeat the same expansion on every later
  unseen turn. Pacman may turn between turns but never midway through one
  speed-two action.
- Treat `0` and `-1` identically as structurally traversable and use only the
  current map frame. Do not accumulate observable terrain.
- After each unseen expansion, remove positions that would be visible from
  Hide's actual position and positions that would already have captured Hide.
  The continued unseen, uncaptured match contradicts those positions.
- Keep exact coordinate sets without probabilities or weighted scores. Sort
  coordinates only for deterministic records and diagnostics.
- If filtering empties the belief, rebuild from the last visible Pacman
  position by replaying every elapsed unseen turn against the recorded Hide
  positions. If that remains empty, rebuild a conservative structural
  reachable set while ignoring negative visibility/capture evidence. Never
  treat an empty belief as safety or return to P04 because of it.
- A new sighting replaces both broad and follower models. A new match clears
  their complete active, pending, and history state.

### Survival and interception rules

- Before choosing a Hide action, expand the broad belief once more to obtain
  every plausible next-turn Pacman endpoint.
- A Hide action is guaranteed safe only when no such endpoint can capture its
  resulting position. If every action is threatened, first minimize the number
  of capturing endpoints, then maximize the minimum Manhattan separation.
- Ghost travel time is exact structural shortest-path time at one cell per
  turn. Pacman threat time is an exact multi-source action-graph search using
  `STAY` and straight one/two-cell actions. Interception is reaching capture
  range, not merely occupying the same cell; equal arrival times are contested.
- For every candidate first move, consider routes to safe four-way campsites.
  A campsite is usable only when Pacman's earliest capture-range arrival is
  later than both Hide's arrival and the campsite's following turn. Any
  junction on the route is contested when Pacman can reach its capture range
  on or before Hide reaches that junction. Reject routes with a contested
  campsite or junction and recalculate every step.
- Rank candidate moves lexicographically in this strict order:
  1. guaranteed immediate survival;
  2. fewest capturing broad-belief endpoints;
  3. availability of an uncontested safe-campsite route;
  4. greatest worst-case distance from the broad next-turn belief;
  5. hidden from the greatest number of broad-belief endpoints;
  6. P08 follower survival, distance, and concealment as likely-case
     tie-breakers;
  7. freedom from a trapped continuation;
  8. not reversing onto Hide's previous trail cell;
  9. movement over an otherwise equal `STAY`;
  10. shorter uncontested-campsite distance;
  11. greater continuation depth.
- Randomize only moves tied on the complete tuple. If movement is structurally
  impossible, `STAY` is the explicit fallback.
- P08 invalidation never returns control to P04 while the broad belief remains
  active. `HOT_UNSEEN` ends only on a new sighting or when Hide is currently at
  a structurally safe campsite and every plausible next-turn Pacman endpoint
  is outside capture range.

### Module boundary and diagnostics

- Add `hide_agent/belief.py` for broad belief expansion, evidence filtering,
  conservative rebuilding, Pacman threat-time search, and route interception
  assessment.
- Keep `pursuit.py` responsible for the P08 ordered trail and follower
  projections. It exposes likely-case projections for P09 and stages only the
  move selected by the broad model.
- Keep `controller.py` responsible for state dispatch, tracker coordination,
  campsite-safe exit, and concise event calls.
- Keep all file writing and formatting in `diagnostics.py`. Every human
  agent-map snapshot lists the current broad Pacman coordinates below the
  matrix; every machine snapshot stores the same coordinates as
  `pacman_belief`.
- `belief_rebuilt` records the reason, elapsed unseen turns, and resulting
  belief size. `interceptor_updated` records only the selected target or route
  junction, Hide arrival time, earliest Pacman threat time, and contested
  state. `hot_unseen_exited` records the exact exit reason. Full belief
  coordinates are not duplicated in the normal log.
- Diagnostics remain best-effort, reset every match, and can be disabled.
  `decision.runtime_ms` remains present on every completed Hide step.

### P09 acceptance checks

- First and later unseen updates expand every legal Pacman action, including
  `STAY`, without allowing a two-cell turn.
- Visibility and continued-survival evidence remove the correct belief cells.
  Empty filtering triggers replay and then conservative fallback rather than
  P04.
- Broad safety outranks every P08 likely-follower preference.
- Equal arrival is contested; capture range is used; a safe camp requires an
  additional safe turn; and contested junctions invalidate their routes.
- P08 remains usable as a tie-breaker but may invalidate without ending
  `HOT_UNSEEN`.
- `HOT_UNSEEN` exits only for a new sighting or a broad-belief-safe campsite.
- Both map formats contain the complete sorted broad belief every step without
  duplicating it in the normal log.
- Required P09 events appear when applicable, returned moves are legal, logs
  reset once per match, diagnostics-off remains non-fatal, and decision runtime
  remains recorded.

## P09 implementation checkpoints

Implement and verify these checkpoints in order. Do not begin P10.

### P09-A — Broad match-local belief

**Files:** create `submissions/LAB2/hide_agent/belief.py`.

- Add immutable belief update records plus `PacmanBeliefTracker`.
- `record_visible(pacman_position, step_number)` replaces the previous seed,
  reachable set, unseen counter, and absence history.
- `observe_unseen(map_state, ghost_position, observation_radius,
  capture_distance, pacman_speed)` expands one turn, applies current absence
  evidence, and retains the last visible origin plus actual Hide positions for
  deterministic replay.
- `predict_next(map_state, pacman_speed)` returns the unfiltered structural
  next-turn threat set used for action safety.
- Implement filtered replay and the unfiltered structural fallback when an
  update would empty the belief.
- Verify first/later expansion, between-turn direction changes, no within-turn
  turn, visibility/capture filtering, rebuild fallback, deterministic ordering,
  replacement on sighting, and reset.

### P09-B — Exact interception assessment

**Files:** modify `submissions/LAB2/hide_agent/belief.py`.

- Add exact multi-source Pacman threat-time search over legal action endpoints.
- Assess each safe-campsite route after a proposed first move using Ghost
  one-cell travel time, capture-range Pacman arrival time, the additional
  campsite-safe turn, and every structural junction on the route.
- Return immutable assessment records containing target, route, Ghost arrival,
  earliest Pacman threat arrival, first contested junction, and contested
  reason.
- Verify walls, Pacman speed two, turns only between actions, capture range,
  equal-time rejection, extra campsite turn, contested junctions, and selection
  of an available uncontested alternative.

### P09-C — Broad-first HOT_UNSEEN movement

**Files:** modify `submissions/LAB2/hide_agent/pursuit.py`; modify
`submissions/LAB2/hide_agent/belief.py`.

- Split P08 candidate projection from final choice so the ordered trail can
  supply optional likely-follower results without controlling broad safety.
- Evaluate every legal Hide action plus `STAY` with the approved complete
  lexicographic rank. Stage the selected move in P08 only when a follower model
  still exists.
- Continue choosing from the broad model when P08 followers are empty. Never
  invoke P04 while the broad belief is active.
- Verify each ranking boundary independently, broad-over-follower priority,
  forced-move exposure ordering, exact-tie randomization, follower staging,
  follower loss without hot exit, and structurally trapped `STAY`.

### P09-D — Controller and diagnostic integration

**Files:** modify `submissions/LAB2/hide_agent/controller.py`; modify
`submissions/LAB2/hide_agent/diagnostics.py`.

- Own and reset one `PacmanBeliefTracker`; replace it from every visible
  observation and update it before every unseen decision.
- Coordinate P08 updates without allowing their invalidation to suppress P09.
  Clear both models and write `hot_unseen_exited` on a new sighting. At a
  broad-safe campsite, write the same event, clear both models, and enter
  `CAMP`.
- Write the three approved concise P09 events and pass the sorted current belief
  into both agent-map formats on every step.
- Verify P06/P07 visible priority, first unseen entry, follower-independent hot
  continuation, both valid hot exits, match reset, diagnostics disabled,
  required fields, map snapshot alignment, and unchanged P04 behavior before
  any visible Pacman seed exists.

### P09-E — Arena gate and phase closure

**Files:** update `submissions/LAB2/HIDE-AGENT-PHASES.md` only after the gate
passes. Arena continues writing the existing files under
`submissions/LAB2/debug/`.

- Compile the Hide package and run the deterministic P09 contract scenarios.
- Run Arena games that transition from a P06/P07 visible escape into
  `HOT_UNSEEN`; retain a game that exercises the broad belief and interception
  diagnostics.
- Confirm no P04 action occurs while belief is active, all selected actions are
  legal, belief snapshots align one-for-one with map steps, logs reset once,
  and every decision retains runtime.
- Record the actual Arena outcome and any P10/P11-scoped limitation, mark P09
  complete, delete temporary verification helpers, and stop.

### P09 completion evidence

- Sixteen deterministic contract scenarios covered first/later broad expansion,
  speed-two straightness, between-turn direction changes, absence and survival
  filtering, conservative fallback, exact threat timing, equal-time and
  next-turn campsite interception, contested junctions, uncontested target
  selection, broad-over-follower priority, forced exposure ordering, P08
  projection staging, follower-independent hot continuation, safe-campsite
  exit, event lifecycle, and synchronized belief snapshots.
- The Hide package compiled successfully after integration. The temporary P09
  contract helper was removed after its final 16/16 passing run.
- The retained stochastic Arena game against reference seeker A produced 37
  legal Hide decisions and 37 map snapshots with one `match_start`.
- P09 entered on step 12 and produced 25 consecutive `hot_move` and
  `interceptor_updated` records through step 36. The broad next-turn set grew
  from 19 to 198 cells while every selected P09 action remained marked
  guaranteed-safe for its immediate turn.
- P08's ordered follower set emptied on step 23. P09 logged
  `follower_invalidated` but remained `HOT_UNSEEN`, selected another
  broad-guaranteed move, and did not emit `scout_move` or `camp_hold`.
- A new sighting on step 37 emitted `hot_unseen_exited` with
  `reason=new_sighting` before P07 handled the visible forced escape. Reference
  seeker A then caught Hide on that step. This does not contradict P09's
  immediate broad-safety or exit contract; broader strategy tuning remains
  P11.
- The retained game's maximum complete Hide-step runtime was 79.577 ms.
  Explicit deadline enforcement remains scoped to P10.

## Approved concealed-hideout strategy revision

This design supersedes the strategic campsite objective used by P03/P04/P09
once its implementation phases are explicitly started and completed. Until
then, the current four-way campsite code remains the active implementation.
No P10 or P11 work is authorized by this design.

### Strategic diagnosis and scope

- A four-way junction is useful after detection because it offers tactical
  escape choices, but it is a poor waiting location. It exposes Hide along up
  to four sight rays, so Pacman normally discovers Hide at four or five cells
  and can close the remaining distance within a few turns.
- The revised pre-detection objective is therefore to delay first discovery,
  not to maximize the number of exits at the waiting position.
- Preserve P06/P07 visible escape and P08/P09 pursuit safety ordering. Replace
  only their strategic destination concept: four-way junctions remain tactical
  escape junctions, while concealed hideouts become the places where Hide
  routes and waits.
- Do not assume a particular reference seeker. The selector uses current map
  structure and the local Arena's spawn tendency, never data carried from a
  previous match.

### Structural hideout model

- Treat every non-wall `0` or `-1` cell as structurally traversable. Any such
  cell may be a hideout; a hideout does not require multiple exits.
- Build the current map's traversable graph and identify:
  - articulation gates and terminal components attached through one entrance;
  - nested one-entrance pockets;
  - side branches that eventually reconnect;
  - general occluded fallback cells when no branch structure qualifies.
- Derive terminal pockets from the graph's block-cut tree. A leaf-side chain is
  a terminal pocket only when it connects through articulation nodes to a
  wider block that has multiple outward continuations. Count the articulation
  nodes on that chain as `gate_depth`; the articulation adjoining the
  candidate's innermost leaf-side component is its branch entrance. If a
  connected component is only one unbranched corridor and has no wider block,
  it has no terminal-pocket class and uses the fallback rules.
- Detect reconnecting branches separately at junction edges: follow the branch
  after excluding its entrance edge and record whether it reaches the wider
  graph through another edge. This prevents a loop corridor from being
  mislabeled as a terminal pocket.
- A branch entrance is the gate where Pacman leaves a wider connecting route
  to inspect the candidate's branch. A terminal pocket requires Pacman to
  enter and later backtrack; a reconnecting branch may be crossed naturally.
- For every candidate, calculate exact structural facts:
  - `pocket_type`: terminal one-entrance, reconnecting occluded branch, or
    general fallback;
  - `entrance_hidden`: whether the candidate is outside radius-five cardinal
    line of sight from its branch entrance;
  - `gate_depth`: how many narrow branch gates separate the candidate from the
    wider map;
  - `inspection_depth`: the minimum number of legal Pacman turns after entering
    the innermost pocket before Pacman can occupy any cell that sees the
    candidate;
  - `visibility_footprint`: the complete set and count of traversable cells
    that see the candidate within the configured observation radius;
  - `must_backtrack`: whether inspecting the candidate's branch requires
    returning through the same entrance;
  - `spawn_discovery_distance`: the minimum legal Pacman turns from the likely
    opposite vertical spawn band to the candidate's visibility footprint;
  - `ghost_route_distance`: Hide's structural shortest-path distance to the
    candidate.
- Pacman timing uses its exact action graph: `STAY`, one cell straight, or two
  cells straight, with turns allowed only between actions.
- The local stochastic Arena deliberately places Ghost in the top 40% and
  Pacman in the bottom 40%, while choosing their columns independently.
  Generalize this as an opposite vertical spawn band inferred from Hide's start.
  Because evaluator spawning may differ, spawn distance is a late soft
  criterion and never overrides structural concealment.

### Deterministic best-cell selection

- Selection is deterministic. Tournament opponents are not expected to play
  repeated adaptive matches against this Hide instance, so do not randomize
  among near-equal strategic targets.
- Exclude unreachable candidates and every hideout compromised during the
  current match.
- Choose the highest non-empty concealment class:
  1. terminal one-entrance pockets whose candidates are hidden from their
     entrances;
  2. reconnecting side branches whose candidates are hidden from their
     entrances;
  3. general occluded cells with the smallest visibility footprint.
- Within that class, find the nearest qualifying candidate. Only candidates no
  more than four Ghost steps farther than this nearest baseline may compete.
  This prevents a small concealment improvement from requiring a long exposed
  crossing.
- Rank the admitted candidates lexicographically, without weighted scores:
  1. greater `gate_depth`;
  2. greater `inspection_depth`;
  3. smaller `visibility_footprint`;
  4. `must_backtrack` over pass-through;
  5. greater `spawn_discovery_distance`;
  6. shorter `ghost_route_distance`;
  7. smallest `(row, column)` for a deterministic final tie.
- A distant straight dead end that remains visible from its entrance cannot
  beat a closer candidate concealed behind a turn merely because it is farther
  from Pacman's likely spawn.
- Lock the selected hideout while it remains reachable and uncompromised.
  Recalculate only on a new match, target compromise, or structural route
  failure; do not switch targets because another candidate becomes one step
  closer.

### Controller behavior

- Before any Pacman sighting, route directly to the selected concealed hideout.
  On arrival, remain still while Pacman is unseen. Do not relocate merely
  because time passes.
- A new sighting preserves the current P06/P07 visible escape decision.
  Immediately mark the selected hideout compromised when Hide is at it or
  travelling toward it; observed movement may reveal the intended destination.
- Compromise state is match-local, survives later sight loss, and resets only
  when a new game starts. A compromised hideout is never reused during that
  match.
- During `HOT_UNSEEN`, preserve broad-belief immediate survival as the highest
  move criterion. Replace four-way campsite guidance and interception targets
  with the best reachable uncompromised concealed hideout.
- `HOT_UNSEEN` ends at a hideout only when no plausible next-turn Pacman
  endpoint can capture Hide there. A new sighting remains the other valid exit.
- If all qualifying hideouts are compromised, fall through the concealment
  classes to any uncompromised occluded cell. If no uncompromised reachable
  target remains, continue the active P08/P09 survival behavior without a
  strategic target; when no pursuit state exists, remain still rather than
  deliberately return to a known compromised location.
- Four-way and useful T-junction detection remains available to P06/P07 as
  tactical escape geometry. Such junctions are no longer strategic camps.

### Module boundaries

- Add `hide_agent/hideout.py` for pocket extraction, visibility footprints,
  inspection timing, candidate records, concealment classes, deterministic
  selection, and current-match compromise tracking.
- Retain `topology.py` for tactical junction facts consumed by P06/P07. Rename
  four-way campsite concepts to escape-junction concepts where the code is
  touched by this revision.
- Update `navigation.py` to route to selected concealed hideouts.
- Update `belief.py` so interception assessment and P09 destination guidance
  consume hideouts while leaving broad immediate-safety ordering unchanged.
- Update `pursuit.py` only where its likely-case destination tie-breaker must
  refer to hideouts instead of strategic four-way campsites.
- Update `controller.py` to own separate tactical escape-junction and strategic
  hideout collections, target locking, compromise transitions, and arrival
  state.
- Keep all file output and diagnostic formatting inside `diagnostics.py`.

### Diagnostics

- Replace obsolete strategic campsite events with:
  - `hideout_scan`: class counts and rejection counts;
  - `hideout_selected`: selected coordinate and complete structural rank;
  - `hideout_compromised`: coordinate and whether compromise occurred at the
    hideout or while travelling;
  - `hideout_arrived`: selected coordinate and arrival step.
- Preserve existing visible-escape, follower, broad-belief, interception,
  hot-move, exit, and runtime events.
- Every human agent-map snapshot lists qualifying hideouts with their
  concealment classes, the selected hideout, compromised hideouts, and the
  current Pacman belief below the matrix.
- Every machine snapshot records the same positions plus the complete
  structural metrics for each qualifying hideout.
- Do not duplicate full hideout lists or Pacman belief coordinates in the
  normal event log. Diagnostics remain optional, best-effort, and reset each
  game.

### Acceptance checks

- A terminal pocket hidden behind a bend beats a four-way junction.
- A hidden bent pocket beats a longer straight dead end visible from its
  entrance.
- Nested gate depth and exact Pacman inspection depth are ordered correctly.
- A reconnecting branch is used only when no qualifying terminal pocket exists,
  and the lowest-footprint fallback works when neither branch class exists.
- Candidates more than four Ghost steps beyond the nearest candidate in the
  same class are excluded.
- Spawn-band distance breaks only a later concealment tie and never overrides
  pocket class, gate depth, inspection depth, or footprint.
- Selection is deterministic, remains locked, and excludes a compromised
  destination for the rest of the match.
- P06/P07 continue using tactical junctions; P08/P09 continue prioritizing
  immediate survival; P09 guides toward a new hideout and cannot return to a
  compromised one.
- `HOT_UNSEEN` exits only on a new sighting or broad-belief-safe hideout
  arrival.
- Both agent-map formats retain aligned per-step hideout, compromise, and
  belief state. Normal logs contain the four required concise hideout events.
- Arena comparison records time to first Pacman sighting as the primary
  strategic result, along with final outcome, legal moves, snapshot alignment,
  and per-step runtime.

## Concealed-hideout revision implementation plan

The revision is implemented only through the following user-started gates.
Completing one gate never starts the next. All verification helpers are
temporary and removed before a gate closes; retained evidence is written by
the normal diagnostics and Arena files.

### Shared interfaces and constraints

- `hideout.py` owns immutable records for visibility, branch structure,
  candidate metrics, and final selection. No other production module
  reimplements these calculations.
- `HideoutCandidate.position` is the stable coordinate consumed by navigation,
  belief, pursuit, controller, and diagnostics.
- Static candidate facts are calculated from the current structural map.
  Dynamic route distance, compromise exclusion, and preferred-target locking
  are applied by selection from Hide's current position.
- `0` and `-1` are structurally equal. No observed terrain or opponent state is
  carried across matches.
- P06/P07 tactical survival and P08/P09 immediate broad safety cannot be
  reordered by any hideout preference.
- Diagnostics can be disabled and cannot crash or change a decision.
- Work is performed directly on main without commits or worktrees, and only
  after the user explicitly starts the named revision phase.

### R00 — Visibility footprint foundation

**Purpose:** establish the most important primitive without changing behavior.

**Files:** create `hide_agent/hideout.py`; modify `controller.py` and
`diagnostics.py` only for read-only analysis output.

**Produces:**

- `visibility_footprints(map_state, observation_radius)` returning a
  deterministic mapping from every structural cell to the sorted cells from
  which Pacman has exact cardinal line of sight.
- A minimal immutable visibility record consumed by later hideout phases.

**Checks:**

- Walls stop rays, diagonals never see, radius is inclusive, arbitrary map
  dimensions work, and `0`/`-1` observer cells are equal.
- Footprints are symmetric with `has_line_of_sight` and deterministically
  ordered.
- `hideout_scan` reports structural-cell count plus minimum and maximum
  footprint sizes while the existing campsite controller returns the same
  decisions as before R00.

### R01 — Terminal pockets and nested gates

**Purpose:** identify true places Pacman must deliberately enter and backtrack.

**Files:** modify `hideout.py`, `controller.py`, and diagnostic formatting.

**Consumes:** R00 visibility mapping.

**Produces:**

- Block-cut-tree records for articulation gates, wider blocks, terminal
  leaf-side chains, entrances, `must_backtrack`, and `gate_depth`.
- Terminal-pocket candidates only; no selector or movement change.

**Checks:**

- A bent cul-de-sac is terminal, a nested branch receives greater gate depth,
  and a simple unbranched corridor has no invented wider backbone.
- Removing the recorded entrance separates every terminal candidate from the
  wider graph.
- `hideout_scan` adds terminal-pocket, entrance, and nested-gate counts.

### R02 — Reconnecting branches and fallback classes

**Purpose:** cover looped maps without weakening the definition of a terminal
pocket.

**Files:** modify `hideout.py`, `controller.py`, and diagnostic formatting.

**Consumes:** R00 footprints and R01 block-cut records.

**Produces:**

- Junction-edge branch exploration that labels a branch reconnecting when it
  reaches the wider graph through another edge.
- Three exact classes: `terminal`, `reconnecting`, and `fallback`.
- A fallback candidate set based on occlusion and minimum footprint when no
  qualifying branch exists.

**Checks:**

- A loop side route is reconnecting and never terminal.
- Terminal candidates remain the higher class regardless of distance.
- Open or corridor-only maps still produce deterministic fallback candidates.
- `hideout_scan` reports class counts and rejection reasons such as
  `entrance_visible`, `pass_through`, and `unreachable`.

### R03 — Pacman inspection and spawn timing

**Purpose:** finish all quality facts before selection is allowed to affect a
target.

**Files:** modify `hideout.py`; reuse Pacman's exact action-graph primitives
from `belief.py`; extend read-only diagnostics.

**Consumes:** R00–R02 candidates and entrances.

**Produces:**

- Exact `entrance_hidden`, `inspection_depth`, `visibility_footprint`,
  `must_backtrack`, and `spawn_discovery_distance` on every candidate.
- Opposite vertical spawn-band inference from Hide's match-start position; no
  mirrored-column assumption.

**Checks:**

- Pacman may move zero/one/two straight cells and turn only between turns.
- A candidate visible at the entrance has inspection depth zero and cannot
  enter an entrance-hidden class.
- Multiple bends, walls, speed-two movement, top/bottom spawn inference, and
  unreachable observer regions produce exact deterministic values.
- `hideout_scan` adds inspection-depth and spawn-distance ranges without
  changing Hide's selected move.

### R04 — Deterministic selection and agent-map evidence

**Purpose:** prove which hideout would be chosen before trusting it with
movement.

**Files:** modify `hideout.py`, `controller.py`, and `diagnostics.py`.

**Consumes:** complete R03 candidates.

**Produces:**

- `select_hideout(map_state, ghost_position, candidates, compromised,
  preferred_position)` returning the chosen candidate, structural route,
  route distance, complete lexicographic rank, and rejection/admission facts.
- Target locking through `preferred_position` while the target remains
  reachable and uncompromised.

**Checks:**

- Class order, four-step admission, every ranking boundary, deterministic
  coordinate tie, unreachable filtering, compromised filtering, and preferred
  lock are independently exercised.
- A hidden bent pocket beats both a four-way junction and a longer
  entrance-visible dead end.
- Controller movement remains the existing campsite behavior, but
  `hideout_selected` reports the shadow selection.
- Human snapshots list qualifying positions by class, selected position, and
  compromised positions below the matrix. Machine snapshots contain complete
  candidate metrics and the same positions.

### R05 — Pre-detection routing and hiding

**Purpose:** activate the new strategy only for unseen play before pursuit.

**Files:** modify `navigation.py` and `controller.py`; update diagnostic field
names in `diagnostics.py`.

**Consumes:** R04 selected candidate and route.

**Produces:**

- No-sight routing to the locked concealed hideout through all structural
  cells.
- Arrival detection and stationary hiding at the target.
- Separate controller collections for strategic hideouts and tactical
  four-way/T-junction geometry.

**Checks:**

- Before any sighting, the first route move heads toward the selected hideout,
  route recalculation cannot switch a valid locked target, arrival emits
  `hideout_arrived`, and later unseen steps return `STAY`.
- A closer four-way junction cannot become the waiting target.
- Existing P06/P07 visible handling and P08/P09 behavior remain unchanged in
  this gate.
- Arena reaches a concealed hideout and retains one legal decision, map
  snapshot, and runtime per step.

### R06 — Compromise lifecycle

**Purpose:** prevent Hide from returning to a destination Pacman has learned.

**Files:** modify `hideout.py`, `controller.py`, and `diagnostics.py`.

**Consumes:** R04 selection and R05 strategic target state.

**Produces:**

- Match-local compromised-position state.
- One transition that compromises the selected destination on a sighting
  whether Hide is at the destination or travelling toward it.
- Reselection after sight loss with permanent exclusion until match reset.

**Checks:**

- At-target and en-route sightings emit `hideout_compromised` with distinct
  reasons.
- Repeated visible calls cannot duplicate the same transition.
- The next selection differs, the old target cannot return later that match,
  and a new match clears the exclusion.
- P06/P07 still choose the visible action before any new strategic route acts.

### R07 — HOT_UNSEEN retargeting and revision gate

**Purpose:** make concealed hideouts the only strategic destinations throughout
the completed agent.

**Files:** modify `belief.py`, `pursuit.py`, `controller.py`, `topology.py`,
`navigation.py`, and `diagnostics.py`; update this phase document only after
verification.

**Consumes:** final R04 selector, R06 compromise state, P08 followers, and P09
broad belief.

**Produces:**

- P09 interception and route-junction assessment against the selected
  uncompromised hideout.
- P08 likely-case destination guidance using hideouts without overriding broad
  safety.
- Hideout-safe `HOT_UNSEEN` exit and complete removal of strategic campsite
  routing, holding, and diagnostic semantics.
- Tactical escape-junction naming retained for P06/P07.

**Checks:**

- Broad immediate survival defeats hideout progress, contested hideouts and
  junctions are rejected, follower loss cannot end hot behavior, and a
  compromised hideout cannot guide or terminate pursuit.
- `HOT_UNSEEN` ends only for a new sighting or arrival at an uncompromised
  hideout safe from every plausible next-turn capture.
- No strategic `campsite_scan`, `campsite_selected`, or `camp_hold` event
  remains; the approved four hideout events and existing pursuit events contain
  their required concise fields.
- Fresh deterministic checks cover all R00–R07 contracts. Arena games confirm
  legal moves, aligned snapshots, reset behavior, runtime recording, and the
  time to first sighting compared with the retained four-way-campsite baseline.
- Record the actual comparison and remaining P10/P11 limitation, remove
  temporary verification helpers, mark R07 complete, and stop.

## R00–R07 implementation record

The concealed-hideout revision was implemented and verified through these
checkpoints in dependency order.

### Structural analysis and selection

- `hide_agent/hideout.py` now computes deterministic wall-blocked cardinal
  visibility footprints for every non-wall cell. Observed `0` and unseen `-1`
  cells are structurally identical.
- Junction removal separates single-entry terminal branches from branches
  whose neighboring edges reconnect. Maps without a qualifying branch retain
  deterministic fallback candidates.
- Every candidate records its branch entrance, terminal/reconnecting/fallback
  class, gate depth, backtracking requirement, entrance concealment, exact
  speed-two inspection depth, visibility footprint, inferred opposite-band
  discovery time, and soft vertical-band tie-break.
- Selection first takes the strongest available class. Within that class it
  admits candidates no more than four Ghost steps beyond the nearest one, then
  applies the approved deterministic lexicographic order. A reachable,
  uncompromised preferred target remains locked.

### Controller lifecycle

- Before detection, Hide routes through either `0` or `-1` toward the selected
  strategic hideout. It emits `hideout_arrived` once and stays there.
- Four-way campsite records remain only as P06/P07 tactical escape geometry;
  they are not strategic waiting locations.
- Seeing Pacman compromises the active hideout once, with distinct
  `visible_at_hideout` and `visible_en_route` reasons. The position remains
  excluded until the next match, while repeated visible steps cannot duplicate
  the event.
- After sight loss, P08/P09 receive only the newly selected uncompromised
  hideout as strategic guidance. P09 broad-belief immediate survival remains
  ahead of destination progress, and a contested target is rejected.
- Human map snapshots list all candidate metrics, the selected hideout,
  compromised hideouts, and Pacman belief below the matrix. Machine snapshots
  contain the matching structured records. No strategic `campsite_scan`,
  `campsite_selected`, or `camp_hold` event remains.

### Verification evidence

- Fresh deterministic contracts covered R00 visibility, bent terminal
  pockets, reconnecting loops, corridor fallback, exact timing metrics,
  `0`/`-1` equivalence, four-step admission, target lock, compromise
  exclusion, duplicate suppression, match reset, arrival/holding, and P09
  retargeting.
- Profiling found that the initial implementation repeated Pacman's exact
  action-graph search twice per candidate. Caching the shared spawn search and
  each unique entrance search reduced the same 21×21 hideout scan from
  0.451 seconds to 0.054 seconds while retaining all contracts.
- A fresh stochastic game against reference seeker A reached its hideout on
  step 5 and lasted 26 steps without Hide observing Pacman. Its maximum Hide
  runtime was 52.489 ms.
- A fresh stochastic game against reference seeker B reached its hideout on
  step 4 and lasted 19 steps without Hide observing Pacman. Its maximum Hide
  runtime was 52.829 ms.
- A fresh stochastic game against reference seeker C saw Pacman on step 3
  while en route, compromised the target once, preserved P07's guaranteed
  visible escape, selected a different hideout after sight loss, and fed that
  target into P09 interception on step 4. P09 rejected the route as contested;
  the maximum Hide runtime was 55.481 ms.
- The earlier documented stochastic B campsite run first entered visible
  escape on step 2 and ended on step 5. The fresh B hideout run delayed capture
  to step 19 without a Hide-side sighting. Because stochastic starts differed,
  this is directional evidence rather than a controlled benchmark; P11 retains
  responsibility for seeded comparative tuning.
- Every Arena decision was legal, each log reset at match start, runtime
  remained present on every decision, and no legacy strategic campsite event
  appeared. P10 emergency/deadline behavior and P11 benchmarking remain
  intentionally unimplemented.
- The final source-only verification compiled every Hide module and won a
  30-step stochastic game against reference seeker B. Hide reached its target
  on step 4, remained uncaught at a final distance of 10, produced 30 decisions
  and 30 synchronized map snapshots, and had a maximum complete step runtime of
  52.513 ms with no legacy strategic campsite event.

## Approved R08 design

R08 corrects the remaining false-positive hideouts on long roads. A cell is not
concealed merely because it is hidden from a distant branch entrance: it must
also be hidden from the major road Pacman is predicted to search during the
current road phase.

### Major-road model

- A major road is a maximal uninterrupted horizontal or vertical run of
  structurally traversable cells.
- A horizontal run qualifies when its length is at least
  `ceil(2/3 × map width)`. A vertical run qualifies when its length is at least
  `ceil(2/3 × map height)`.
- Runs are identified from the current structural map, treating `0` and `-1`
  identically. Their orientation, endpoints, ordered cells, and length are
  deterministic.
- Pacman's inferred start region remains the map-side band opposite Hide's
  match-start band. No exact row, column, or mirrored spawn point is assumed.
- Every qualifying road reachable from that inferred region is a possible
  search road. Its predicted arrival is Pacman's exact earliest action-graph
  distance from the inferred region to any road cell, using legal zero/one/two
  straight-cell actions.
- Roads are ordered by predicted arrival, then orientation and endpoints for
  deterministic ties. Roads with the same predicted arrival form one road
  phase and share one visibility mask.

### Road phases and exposure

- Before the first predicted arrival, the earliest road phase is active.
- At elapsed match turn `t`, the active phase is the latest phase whose
  predicted arrival is at most `t`; if none has arrived, it remains the earliest
  phase. After the final predicted arrival, the final phase remains active.
- A phase's exposure mask is the union of every cell Pacman can see while
  standing on any road cell in that phase. It uses the Arena's exact cardinal,
  radius-limited, wall-blocked line of sight.
- When the active phase advances, the previous road mask is removed and its
  cells become eligible again. Only the newly predicted road phase constrains
  strategic hiding and routing.
- A real Pacman sighting and all P06–P09 visible/HOT behavior override the road
  schedule. Road timing never weakens broad-belief immediate survival.

### Initial selection and phase reevaluation

- Initial selection assumes Pacman is closing in from the earliest road phase.
- A normal strategic destination is eligible only when its cell is outside the
  active road mask and a route exists whose every position is also outside that
  mask.
- Among eligible destinations, choose the shortest concealed route first.
  Existing hideout quality and deterministic coordinates break equal route
  distances.
- Recalculate road eligibility and the route every unseen, non-HOT step.
- A road-phase change triggers reevaluation but never forces movement. If the
  current hideout remains outside the new mask, Hide stays there.
- If the current hideout becomes exposed, select the closest eligible hideout
  with a fully concealed route and move along that route.
- Match-local compromised hideouts remain excluded exactly as in R06.

### Bounded fallback

- The fallback is used only when no hideout has a completely concealed route
  under the active road mask.
- For each reachable destination, compute a lexicographically minimal route
  whose first cost is the number of occupied route positions in the active road
  mask and whose second cost is route length. The occupied positions include
  the origin and destination; `STAY` therefore evaluates the current cell once
  instead of receiving an artificial zero-exposure route.
- Rank fallback destinations by:
  1. fewer exposed route cells;
  2. greater structural distance from the nearest four-way junction;
  3. smaller destination visibility footprint;
  4. shorter route;
  5. deterministic coordinate order.
- Evaluate the current position by the same fallback rank. If no destination
  improves on it, return `STAY`; otherwise route to the improving fallback and
  stay there on arrival.

### Module boundary and diagnostics

- Major-road discovery, phase timing, masks, concealed routing, and fallback
  ranking belong in a dedicated road-analysis module rather than topology,
  belief, or the controller.
- The controller owns only match elapsed time, the active road phase, the
  selected strategic destination, and transitions between normal, fallback,
  visible, and HOT behavior.
- `main_road_scan` records road count, endpoints, lengths, predicted arrivals,
  and phase grouping once per match.
- `road_phase_changed` records previous/new phase, elapsed turns, active roads,
  mask size, and whether the current hideout remains safe.
- Human agent-map snapshots list major roads, active phase, active road
  positions, and exposed-cell coordinates below the matrix without overlaying
  it. Machine snapshots store the same data structurally.
- Existing `hideout_selected`, route, decision, compromise, visible, and HOT
  events remain authoritative; selection records add `road_phase`,
  `road_concealed`, and `fallback` facts without duplicating the map.

### R08 verification gate

- Straight-run checks cover exact two-thirds boundaries, just-short runs,
  horizontal/vertical dimensions, walls splitting runs, intersections,
  arbitrary map sizes, and `0`/`-1` equivalence.
- The reported example road from `(1, 4)` through `(19, 4)` qualifies on the
  21×21 map. Cells `(4, 4)` and `(9, 4)` in display `(x, y)` coordinates are
  rejected whenever visible from its active road phase.
- Phase checks cover opposite top/bottom starts, tied road arrivals, previous
  mask release, final-phase retention, and match reset.
- Routing checks prove that normal targets and every route cell avoid the
  active mask, a safe current hideout does not move at phase change, an exposed
  hideout chooses the closest concealed replacement, and compromised targets
  remain excluded.
- Fallback checks independently exercise exposed-step minimization,
  four-way-junction distance, footprint, route length, deterministic ties, and
  `STAY` when movement cannot improve the current rank.
- Controller checks preserve P06/P07 priority and P08/P09 broad safety. Arena
  games must exercise at least one road-phase transition, produce one legal
  decision and synchronized snapshot per step, retain runtime logging, and
  contain no legacy strategic campsite event.
- Mark R08 complete only after recording deterministic and Arena evidence.
  P10 and P11 remain inactive.

# R08 Dynamic Main-Road Avoidance Implementation Plan

> **For agentic workers:** use the `executing-plans` workflow and implement
> only the exact R08 checkpoint explicitly started by the user. Stop for review
> after every checkpoint.

**Goal:** prevent Hide from waiting in or routing through cells Pacman can see
from the major road it is predicted to search during the current match phase.

**Architecture:** add one focused `roads.py` module for road discovery, Pacman
arrival phases, exposure masks, concealed routes, and fallback ranking.
`hideout.py` continues to own static hideout quality. `controller.py` owns only
match lifecycle and behavioral priority, while `diagnostics.py` serializes the
road state it receives.

**Tech stack:** Python, NumPy map arrays, existing `Move` geometry, existing
Pacman action-graph distances, immutable dataclasses, deterministic BFS/Dijkstra.

## Global R08 constraints

- No cross-game terrain, road, timing, or Pacman memory.
- Treat structural `0` and `-1` cells identically.
- Horizontal threshold is `ceil(2 × width / 3)`; vertical threshold is
  `ceil(2 × height / 3)`.
- Use exact Arena cardinal LOS, configured observation radius, and walls.
- Use exact Pacman zero/one/two-cell action timing.
- Only the active predicted road phase constrains normal strategic movement.
- P06, P07, P08, and P09 always take priority over road prediction.
- Never reuse a match-local compromised hideout.
- Diagnostics reset every match and remain optional.
- Do not add permanent tests, commit, create a worktree, or start the next
  checkpoint automatically.

---

## R08-A — Major-road detection

**Purpose:** establish and verify road geometry without affecting selection or
movement.

**Files:**

- Create `submissions/LAB2/hide_agent/roads.py`.
- Modify `submissions/LAB2/hide_agent/controller.py` only to run the scan once
  per match and emit its summary.
- Use temporary `revision_r08_check.py` for deterministic verification; do not
  retain it after R08-F.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class MajorRoad:
    road_id: int
    orientation: str          # "horizontal" or "vertical"
    start: tuple
    end: tuple
    cells: tuple
    length: int

    def to_log_record(self) -> dict: ...


def detect_major_roads(map_state) -> tuple[MajorRoad, ...]: ...
```

Road IDs follow the final deterministic sort
`(orientation, start_row, start_col, end_row, end_col)`.

**Implementation steps:**

1. Add failing temporary checks for:
   - a horizontal run exactly `ceil(2w/3)`;
   - a run one cell shorter;
   - the equivalent vertical boundary;
   - a wall splitting one apparent road into two maximal runs;
   - horizontal and vertical roads sharing an intersection;
   - rectangular maps;
   - identical results after changing all structural `-1` cells to `0`;
   - display road `(1,4)` through `(19,4)` on a 21×21 map.
2. Run `rtk python -X utf8 revision_r08_check.py`; verify failure because
   `hide_agent.roads` does not exist.
3. Implement row and column scans that close a run at a wall or dimension
   boundary, retain only qualifying maximal runs, then assign IDs after sorting.
4. Add `to_log_record()` with ID, orientation, start, end, length, and complete
   ordered coordinates.
5. Add match fields `_major_roads` and `_road_schedule` to the controller.
   Reset both in `_start_match`; on the first step call
   `detect_major_roads(current_map)`.
6. Emit `main_road_scan` once with map dimensions, horizontal/vertical
   thresholds, road count, and road records. Do not alter selection, target,
   route, state, or returned action.
7. Run the temporary checks and compile `roads.py` plus `controller.py`.
8. Run one Arena game and compare every returned decision with the R07
   decision path for that same call sequence. Confirm `main_road_scan` appears
   once and runtime remains recorded.
9. Mark only R08-A complete, record evidence, and stop.

### R08-A completion evidence

- The temporary R08 verifier first failed because `hide_agent.roads` did not
  exist, then passed after the focused module was added.
- Deterministic cases covered exact horizontal and vertical two-thirds
  thresholds, a just-short run, wall splitting, crossing horizontal/vertical
  roads, rectangular dimensions, stable IDs and records, `0`/`-1` equivalence,
  and the 21×21 display road from `(1,4)` through `(19,4)`.
- A three-step controller sequence returned the retained R07 actions `RIGHT`,
  `RIGHT`, `RIGHT` before and after read-only road integration.
- `main_road_scan` was emitted exactly once for that match with the correct
  dimensions, thresholds, count, endpoints, lengths, and complete coordinates.
- Fresh compilation succeeded for `roads.py`, `controller.py`, and the Arena
  entry point.
- A stochastic Arena game against reference seeker B detected four major roads
  with horizontal/vertical thresholds of 14, produced 19 legal Hide decisions,
  and emitted one `main_road_scan`. The maximum complete Hide-step runtime was
  53.985 ms. Seeker B caught Hide on step 19; R08-A intentionally did not alter
  selection or movement.
- No strategic `campsite_scan`, `campsite_selected`, or `camp_hold` event
  appeared. The temporary R08 verifier remains for R08-B through R08-F as
  planned.

---

## R08-B — Vertical approach-road hideout filter

**Purpose:** exclude strategic hiding positions Pacman can see while traversing
the long vertical roads connecting the bottom and top regions.

**Implemented scope:**

- Each R08-A road receives one match-local cached visibility set: the union of
  exact cardinal, radius-limited, wall-blocked visibility from every road cell.
- Traversal direction does not alter visibility. All vertical major roads are
  marked as DOWN↔TOP approach roads; horizontal roads remain diagnostic and
  never exclude a hideout.
- The normal unseen R07 selector receives only candidates outside the combined
  vertical-road visibility set. Route cells are not filtered.
- Existing visible and `HOT_UNSEEN` behavior retains priority. No road timing,
  phase switching, relocation, or fallback is present.
- `main_road_scan` records the per-road mapping and exclusion counts. Both
  agent-map formats retain the complete per-road visible cells and the excluded
  hideout positions below the matrix.

**Verification evidence:**

- Focused checks passed for per-road union, direction independence, horizontal
  non-exclusion, candidate partitioning, shared footprint reuse, controller
  selector input, match-local state, and aligned diagnostics.
- Exact geometry checks passed for the radius boundary, beyond-radius
  rejection, wall occlusion, and structural `0`/`-1` equivalence.
- A stochastic 80-step Arena game against reference seeker B detected four
  major roads and marked vertical roads 2 and 3 as approach roads. Their union
  exposed 124 structural cells and excluded 124 hideout candidates.
- R07 selected `(2,19)`, which was outside every marked-road visibility set,
  and returned `RIGHT` on its first move. The game produced 80 legal decisions
  and 80 synchronized snapshots, Hide won at final distance 6, and maximum
  complete-step runtime was 58.106 ms.
- No `road_phase_changed`, `road_fallback`, shadow-selection, timing,
  relocation, or route-exclusion behavior appeared.

## Superseded R08-B–F design — do not implement

The following earlier phase descriptions are retained only as historical
context. They were rolled back, are not approved, and do not describe current
runtime behavior.

### Former R08-B — Predicted phases and exposure masks

**Purpose:** calculate which road matters at each match turn while keeping R07
movement unchanged.

**Files:**

- Modify `submissions/LAB2/hide_agent/roads.py`.
- Modify `submissions/LAB2/hide_agent/controller.py`.
- Modify `submissions/LAB2/hide_agent/diagnostics.py`.
- Extend temporary `revision_r08_check.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class TimedRoad:
    road: MajorRoad
    predicted_arrival: int


@dataclass(frozen=True)
class RoadPhase:
    index: int
    predicted_arrival: int
    roads: tuple[MajorRoad, ...]
    road_cells: tuple
    exposed_cells: tuple

    def to_log_record(self) -> dict: ...


@dataclass(frozen=True)
class RoadSchedule:
    timed_roads: tuple[TimedRoad, ...]
    phases: tuple[RoadPhase, ...]

    def phase_at(self, elapsed_match_turns: int) -> RoadPhase | None: ...


def inferred_pacman_start_region(map_state, ghost_spawn) -> tuple: ...


def build_road_schedule(
    map_state,
    roads,
    ghost_spawn,
    observation_radius,
    pacman_speed,
) -> RoadSchedule: ...
```

**Implementation steps:**

1. Add failing checks for opposite-band inference with top and bottom Ghost
   starts; no mirrored-column assumption; exact speed-two arrival; tied road
   grouping; turns before the first arrival; transitions between phases;
   release of the old mask; final-phase retention; walls, radius, and diagonal
   blindness in exposure masks; and match reset.
2. Run only the R08-B check group and verify the schedule interfaces are
   missing.
3. Reuse `pacman_turn_distances()` for one multi-source search from the inferred
   opposite 40-percent band. A road's arrival is the minimum distance of its
   cells. Exclude structurally unreachable roads.
4. Group roads by equal arrival. For each group, union its road coordinates and
   the existing exact `visibility_footprints()` observers/targets into sorted
   `road_cells` and `exposed_cells`.
5. Implement `phase_at(t)` as:
   - `None` when there are no phases;
   - the earliest phase before its arrival;
   - otherwise the latest phase with arrival `<= t`;
   - the final phase after all arrivals.
6. Build the schedule once per match after R08-A scanning. Calculate elapsed
   match turns from the current match's first step rather than accumulated
   history.
7. Track the last reported phase index. Emit `road_phase_changed` on initial
   activation and every index change, including old/new phase, elapsed turn,
   active road IDs, active road coordinates, mask size, and a read-only
   `current_hideout_safe` value.
8. Extend both map formats with complete major-road records, timed arrivals,
   active phase, active road cells, and active exposed cells below the matrix.
   Movement must still be R07.
9. Run R08-A and R08-B checks, compile touched modules, and run an Arena game
   long enough to observe a phase transition in diagnostics.
10. Mark only R08-B complete, record evidence, and stop.

---

### Former R08-C — Concealed routing and shadow selection

**Purpose:** prove road-aware target decisions before allowing them to move
Hide.

**Files:**

- Modify `submissions/LAB2/hide_agent/roads.py`.
- Modify `submissions/LAB2/hide_agent/hideout.py` to expose the existing quality
  tuple without changing its ordering.
- Modify `submissions/LAB2/hide_agent/controller.py` only for shadow output.
- Extend temporary `revision_r08_check.py`.

**Interfaces produced:**

```python
def hideout_quality_rank(candidate) -> tuple:
    return (
        candidate.gate_depth,
        candidate.inspection_depth,
        -candidate.visibility_footprint,
        int(candidate.must_backtrack),
        candidate.spawn_discovery_distance,
        int(candidate.opposite_vertical_band),
    )


@dataclass(frozen=True)
class RoadHideoutSelection:
    candidate: object | None
    path: tuple
    route_distance: int | None
    road_concealed: bool
    fallback: bool
    exposed_steps: int
    junction_distance: int | None
    rank: tuple
    rejections: Mapping[str, int]


def select_concealed_hideout(
    map_state,
    ghost_position,
    candidates,
    compromised,
    active_phase,
    preferred_position=None,
) -> RoadHideoutSelection: ...
```

**Implementation steps:**

1. Add failing checks proving:
   - a destination inside the active mask is rejected;
   - a hidden destination whose only routes cross the mask is rejected;
   - an alternate longer but fully concealed route is accepted;
   - shortest concealed route wins before hideout quality;
   - existing quality and coordinates break equal-distance ties;
   - compromised positions remain excluded;
   - a preferred current hideout stays only while outside the mask;
   - display cells `(4,4)` and `(9,4)` are rejected under the example road.
2. Run the R08-C group and verify the selector is absent.
3. Extract the first six static fields of the private R07 rank into
   `hideout_quality_rank(candidate)` exactly as shown above. Rebuild R07's
   complete selector rank as
   `hideout_quality_rank(candidate) + (-route_distance, -row, -column)` so its
   behavior remains unchanged.
4. Implement BFS over structural cells excluding `active_phase.exposed_cells`.
   A concealed search is invalid when its origin is exposed. Reconstruct paths
   with the existing navigation helper.
5. Filter compromised and unreachable candidates. Preserve the preferred
   position only if reachable through the concealed graph.
6. Select with the exact key:
   `(shorter route, greater hideout quality, smaller coordinate)`.
   Return rejection counts for `compromised`, `destination_exposed`,
   `origin_exposed`, and `no_concealed_route`.
7. Run the selector each unseen, non-HOT step in shadow mode. Emit
   `road_selection_shadow` only when phase or shadow target changes, containing
   phase index, selected candidate, path, route distance, rank, and rejections.
   Do not store it as the active target or alter the returned move.
8. Run R08-A–C checks, compile, and compare Arena decisions with R07 behavior.
9. Mark only R08-C complete, record evidence, and stop.

---

### Former R08-D — Activate normal road-aware hiding

**Purpose:** replace R07's normal unseen selector with the verified road-aware
selector while preserving all visible and HOT priorities.

**Files:**

- Modify `submissions/LAB2/hide_agent/controller.py`.
- Modify `submissions/LAB2/hide_agent/diagnostics.py` only for selection fields.
- Extend temporary `revision_r08_check.py`.

**Implementation steps:**

1. Add failing controller checks for:
   - initial selection against the earliest phase;
   - every executed route position outside the active mask;
   - staying at a current hideout that remains safe after phase change;
   - replacing a current hideout that becomes exposed;
   - choosing the closest concealed replacement;
   - retaining compromised exclusion;
   - P06/P07 visible priority;
   - P08/P09 HOT priority and broad immediate safety.
2. Run the R08-D group against the still-shadow controller and verify that the
   expected road-aware move differs.
3. On unseen steps with inactive belief/pursuit, use
   `select_concealed_hideout()` as the active strategic selection. Convert its
   candidate/path to the existing `RouteTarget(kind="strategic_hideout", ...)`.
4. At a phase change, reevaluate the selected hideout:
   - retain it and return `STAY` when it is the current position and remains
     concealed;
   - retain/replan it when its route remains concealed;
   - otherwise select the closest concealed replacement.
5. Do not call road selection in visible or HOT branches. Those branches retain
   the exact P06–P09 order and actions.
6. Remove `road_selection_shadow`. Extend `hideout_selected` with phase index,
   active road IDs, `road_concealed=True`, and `fallback=False`.
7. Extend route events with the same phase index without duplicating road or
   mask coordinate lists already present in the map snapshot.
8. Run R08-A–D checks, compile, and run Arena cases that both retain and replace
   a hideout across phase changes.
9. Mark only R08-D complete, record evidence, and stop.

---

### Former R08-E — Bounded fallback

**Purpose:** return a deterministic useful action when the active road leaves no
fully concealed hideout route.

**Files:**

- Modify `submissions/LAB2/hide_agent/roads.py`.
- Modify `submissions/LAB2/hide_agent/controller.py`.
- Extend temporary `revision_r08_check.py`.

**Interfaces produced:**

```python
def minimum_exposure_paths(
    map_state,
    start,
    exposed_cells,
) -> tuple[dict, dict]: ...


def four_way_junction_distances(map_state) -> dict: ...


def select_road_fallback(
    map_state,
    ghost_position,
    candidates,
    compromised,
    active_phase,
) -> RoadHideoutSelection: ...
```

`minimum_exposure_paths()` returns lexicographic costs
`position -> (exposed_occupied_positions, route_length)` plus parents.
The origin contributes one exposure when it lies in the mask.

**Implementation steps:**

1. Add failing checks isolating each fallback boundary:
   - fewer exposed occupied positions wins;
   - greater distance from the nearest degree-four junction wins after equal
     exposure;
   - smaller footprint wins next;
   - shorter route wins next;
   - coordinates resolve a complete tie;
   - compromised candidates are absent;
   - maps without four-way junctions give every destination the same junction
     distance;
   - current-position rank returns `STAY` when no move improves it;
   - an improving destination produces its complete path.
2. Run the R08-E group and verify missing fallback interfaces.
3. Implement lexicographic Dijkstra with initial cost
   `(int(start in mask), 0)` and neighbor increment
   `(int(neighbor in mask), 1)`.
4. Compute distance from the nearest structural degree-four cell with
   multi-source BFS. When none exist, assign the common value
   `len(structural_cells) + 1` to every structural cell.
5. Rank each reachable, uncompromised candidate by:
   `(-exposed_count, junction_distance, -footprint, -route_length, -row, -col)`
   using maximum comparison.
6. Evaluate the current position with the same exposure, junction, footprint,
   and zero route-length rules. Select a destination only if its complete rank
   is strictly greater.
7. In the normal unseen branch, call fallback only when concealed selection has
   no candidate. Emit `road_fallback` with phase, selected position or `None`,
   path, exposed count, junction distance, footprint, route distance, rank, and
   `stay_reason`.
8. Mark fallback selections in `hideout_selected` with
   `road_concealed=False` and `fallback=True`. Route using the returned path;
   return `STAY` when no destination improves the current rank.
9. Run R08-A–E checks, compile, and exercise both moving and staying fallbacks
   through controller calls.
10. Mark only R08-E complete, record evidence, and stop.

---

### Former R08-F — Integration and Arena gate

**Purpose:** verify the complete strategy, remove phase-only artifacts, and
finish documentation without starting P10 or P11.

**Files:**

- Modify `submissions/LAB2/hide_agent/roads.py`,
  `submissions/LAB2/hide_agent/controller.py`, and
  `submissions/LAB2/hide_agent/diagnostics.py` only for defects demonstrated by
  R08-F verification.
- Modify `submissions/LAB2/HIDE-AGENT-PHASES.md` with actual evidence.
- Delete temporary `revision_r08_check.py`.

**Implementation steps:**

1. Run the complete temporary R08 contract suite. It must cover every R08-A–E
   completion check and every example from the approved R08 design.
2. Compile `submissions/LAB2/agent.py` and every module in
   `submissions/LAB2/hide_agent`.
3. Run stochastic Arena games against reference seekers A, B, and C with
   partial observation, Pacman speed two, and sufficient steps to exercise road
   transitions.
4. For each game, verify:
   - one `match_start`;
   - one legal `decision` and one map snapshot per Ghost step;
   - runtime on every decision;
   - one `main_road_scan`;
   - correct `road_phase_changed` ordering;
   - active-road data aligned between human and machine snapshots;
   - no normal route crossing its active mask;
   - no duplicate compromise transition;
   - P06–P09 priority when sight/HOT occurs;
   - no `campsite_scan`, `campsite_selected`, or `camp_hold`.
5. Compare time to first sight and survival length with the documented R07
   runs, clearly labeling stochastic comparisons as directional rather than
   controlled.
6. Diagnose any failure before changing code; rerun the smallest failing
   contract and then the full gate after its root-cause fix.
7. Delete `revision_r08_check.py`. Confirm no phase-only helper remains.
8. Record exact contract counts, Arena outcomes, phase transitions, maximum
   runtime, comparison limitations, and remaining P10/P11 scope in this file.
9. Mark R08-A through R08-F and parent R08 complete. Leave P10 and P11 not
   started, then stop.
