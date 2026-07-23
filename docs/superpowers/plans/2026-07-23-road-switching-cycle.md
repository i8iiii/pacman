# Road-Switching Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or `executing-plans` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cycle the active road-aware hideout exclusion every five turns
through vertical travel, Ghost-side horizontal travel, reverse vertical
travel, and opposite-side horizontal travel.

**Architecture:** Build one deterministic four-stage cycle from the cached
R08-B road records. The controller advances the active stage only during normal
unseen play, recomputes candidate exclusion from cached visibility, preserves a
safe target, and uses a distance-first selector only when the old target becomes
exposed.

**Tech Stack:** Python 3, NumPy map arrays, existing Hide navigation,
visibility, selection, and diagnostics modules.

## Global Constraints

- Every stage lasts exactly five completed turns.
- Top and bottom Ghost spawns use mirrored stage order.
- Release the previous road set whenever a valid next stage activates.
- Keep a selected hideout when it remains safe.
- Replace an exposed target with the closest reachable safe uncompromised
  hideout; R07 quality and coordinates break distance ties.
- Filter destinations only; routes remain unrestricted.
- Visible and `HOT_UNSEEN` behavior retain priority.
- Reuse cached road visibility; do not recalculate line of sight per stage.
- Do not add pursuit timing, route exposure, or fallback behavior.
- Reset every cycle field on a new match.
- Do not commit without explicit user authorization.
- Prefix every shell command with `rtk`.

---

### Task 1: Deterministic four-stage road cycle

**Files:**

- Modify: `submissions/LAB2/hide_agent/roads.py`
- Create temporarily:
  `submissions/LAB2/revision_road_cycle_check.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RoadCycleStage:
    index: int
    label: str
    road_ids: tuple


@dataclass(frozen=True)
class RoadCycle:
    ghost_side: str
    stage_turns: int
    stages: tuple

    def requested_index(self, elapsed_turns): ...
    def stage(self, index): ...


def build_road_cycle(
    road_visibility,
    ghost_spawn,
    map_shape,
    stage_turns=5,
) -> RoadCycle: ...
```

- [ ] Write a failing temporary check with crossing vertical roads and top and
  bottom horizontal roads. Assert top order
  `vertical_up → top_horizontal → vertical_down → bottom_horizontal`, bottom
  order is mirrored, connected road IDs are correct, and requested indices for
  elapsed turns `0, 4, 5, 9, 10, 14, 15, 19, 20` are
  `0, 0, 1, 1, 2, 2, 3, 3, 0`.

- [ ] Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_cycle_check.py
```

Expected: import failure for `RoadCycle` or `build_road_cycle`.

- [ ] Implement direct road-cell intersection:

```python
def _roads_intersect(first, second):
    return bool(set(first.cells).intersection(second.cells))
```

Build stage 0 from every vertical record. Select the Ghost-side horizontal
record from those intersecting stage 0 using smallest row for top or largest
row for bottom, then road ID. Build stage 2 from vertical records intersecting
that horizontal. Select the opposite-side horizontal from those intersecting
stage 2 using the mirrored row order. Use an empty `road_ids` tuple when a
connected stage does not exist.

- [ ] Implement timing:

```python
def requested_index(self, elapsed_turns):
    return (max(0, int(elapsed_turns)) // self.stage_turns) % 4
```

- [ ] Rerun the temporary check. Expected: all Task 1 checks pass.

---

### Task 2: Active filtering and closest-safe relocation

**Files:**

- Modify: `submissions/LAB2/hide_agent/roads.py`
- Modify: `submissions/LAB2/hide_agent/controller.py`
- Extend temporarily:
  `submissions/LAB2/revision_road_cycle_check.py`

**Interfaces:**

Change filtering to accept an explicit active set:

```python
def filter_hideout_candidates(
    candidates,
    road_visibility,
    active_road_ids=None,
) -> tuple[tuple, tuple]: ...
```

When `active_road_ids is None`, retain R08-B compatibility by using all records
where `is_approach` is true.

Add:

```python
def select_closest_safe_hideout(
    map_state,
    ghost_position,
    candidates,
    compromised,
) -> HideoutSelection: ...
```

- [ ] Add failing checks proving explicit horizontal IDs exclude only that
  horizontal road, the previous vertical set is released, compromised and
  unreachable candidates are rejected, shorter distance wins, R07 static
  quality breaks an equal-distance tie, and coordinates break a complete tie.

- [ ] Run the temporary checker and confirm failure for the missing explicit
  filter argument or closest selector.

- [ ] Implement explicit active filtering by looking up only records whose
  `record.road.road_id` occurs in the supplied ID set.

- [ ] Implement closest-safe selection using
  `structural_shortest_paths()`, `reconstruct_path()`, `HideoutSelection`, and
  this maximum rank:

```python
(
    -distances[candidate.position],
    *hideout_quality_rank(candidate),
    -candidate.position[0],
    -candidate.position[1],
)
```

Return an empty `HideoutSelection` when no uncompromised reachable candidate
exists.

- [ ] Add controller fields and reset them:

```python
self._road_cycle = None
self._active_road_stage = None
self._active_road_ids = ()
self._match_start_step = None
```

- [ ] At match scan, build the cycle and activate stage 0. Filter candidates
  using stage 0 IDs.

- [ ] During normal unseen play calculate elapsed turns and requested stage.
  Switch only when the requested stage differs and has non-empty road IDs.
  Refilter from the complete `_hideout_candidates`. If the previous target is
  still eligible, preserve it through the existing preferred-target behavior.
  If it is exposed, call `select_closest_safe_hideout`; otherwise use the normal
  R07 selector.

- [ ] A requested empty stage retains `_active_road_stage`,
  `_active_road_ids`, eligible candidates, and excluded candidates unchanged.

- [ ] Extend the temporary controller check across steps 1, 5, 6, 11, 16, and
  21. Assert safe-target preservation and exposed-target distance-first
  reselection. Rerun and expect all Task 1–2 checks to pass.

---

### Task 3: Cycle and active-stage diagnostics

**Files:**

- Modify: `submissions/LAB2/hide_agent/controller.py`
- Modify: `submissions/LAB2/hide_agent/diagnostics.py`
- Extend temporarily:
  `submissions/LAB2/revision_road_cycle_check.py`

**Interfaces:**

- Add `road_cycle_built` once per match.
- Add `road_stage_changed` for initial activation and valid switches.
- Extend map snapshots with:

```python
road_cycle=None,
active_road_stage=None,
active_road_ids=(),
active_road_excluded_cells=(),
```

- [ ] Add failing diagnostic checks for one cycle event, ordered stage-change
  events, released/active road IDs, current excluded hideouts, and matching
  human/machine snapshot state.

- [ ] Serialize the cycle as Ghost side, stage duration, and four stage records.
  Serialize the active excluded-cell union from cached records selected by
  `_active_road_ids`.

- [ ] Emit `road_stage_changed` with previous, requested, and active stage,
  elapsed turns, released and active IDs, excluded counts, selected position,
  and `selected_hideout_safe`.

- [ ] Human snapshots list cycle stage and active road/excluded coordinates
  below the matrix without overlays. Machine snapshots store the same values.

- [ ] Rerun the temporary checker. Expected: every Task 1–3 assertion passes.

---

### Task 4: Arena verification and phase record

**Files:**

- Modify: `submissions/LAB2/HIDE-AGENT-PHASES.md`
- Delete: `submissions/LAB2/revision_road_cycle_check.py`

- [ ] Compile `submissions/LAB2/agent.py` and all
  `submissions/LAB2/hide_agent/*.py` files from source. Expected: 13 files,
  zero failures.

- [ ] Run a stochastic 25-or-more-step Arena game with diagnostics enabled so
  turns 6, 11, 16, and 21 occur unless the game ends earlier. If it ends early,
  run another game.

- [ ] Parse the retained logs and assert one cycle event, valid five-turn
  switches, previous road release, selected hideout outside the active mask,
  legal decisions, synchronized snapshots, and no legacy
  `road_phase_changed` or `road_fallback` event.

- [ ] Update the phase board and add actual cycle verification evidence. Do not
  mark any route-exposure, pursuit-timing, or fallback behavior complete.

- [ ] Delete only the temporary checker, rerun source compilation, and rerun
  inline cycle assertions against the exact remaining source.

