# Road-Aware Hideout Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or `executing-plans` to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude strategic hideouts visible while Pacman traverses any
vertical major road, without adding road timing, route exclusion, relocation,
or fallback behavior.

**Architecture:** Extend the diagnostic-only R08-A road model with one cached
visibility record per road. Build those records once per match from the same
exact visibility footprints used by hideout analysis, combine only vertical
road footprints for selection, and pass the filtered candidate tuple to the
unchanged R07 selector.

**Tech Stack:** Python 3, NumPy map arrays, existing Hide geometry and
diagnostic modules.

## Global Constraints

- Treat `0` and `-1` identically as structurally traversable.
- Use configured cardinal, wall-blocked line of sight and observation radius.
- Horizontal roads never exclude hideouts.
- Filter hideout destinations only; do not filter routes.
- Do not add road timing, phase switching, relocation, or fallback behavior.
- Visible and `HOT_UNSEEN` survival logic retains priority.
- All state resets between matches.
- Do not commit; project control rules require explicit user authorization.
- Use `rtk` at the start of every shell command.

---

### Task 1: Per-road cached visibility

**Files:**

- Modify: `submissions/LAB2/hide_agent/roads.py`
- Create temporarily:
  `submissions/LAB2/revision_road_filter_check.py`

**Interfaces:**

- Consumes:
  `MajorRoad` and a mapping
  `Mapping[tuple[int, int], tuple[tuple[int, int], ...]]` containing exact
  visibility footprints.
- Produces:

```python
@dataclass(frozen=True)
class RoadVisibility:
    road: MajorRoad
    visible_cells: tuple
    is_approach: bool

    def to_log_record(self) -> dict: ...


def build_road_visibility(roads, footprints) -> tuple[RoadVisibility, ...]: ...


def filter_hideout_candidates(
    candidates,
    road_visibility,
) -> tuple[tuple, tuple]:
    """Return (eligible_candidates, rejected_candidates)."""
```

- [ ] **Step 1: Write focused failing checks**

Create `submissions/LAB2/revision_road_filter_check.py` with deterministic
checks equivalent to:

```python
from types import SimpleNamespace

from hide_agent.roads import (
    MajorRoad,
    build_road_visibility,
    filter_hideout_candidates,
)

vertical = MajorRoad(
    road_id=0,
    orientation="vertical",
    start=(0, 2),
    end=(3, 2),
    cells=((0, 2), (1, 2), (2, 2), (3, 2)),
    length=4,
)
horizontal = MajorRoad(
    road_id=1,
    orientation="horizontal",
    start=(2, 0),
    end=(2, 3),
    cells=((2, 0), (2, 1), (2, 2), (2, 3)),
    length=4,
)
footprints = {
    (0, 2): ((0, 1), (0, 2), (0, 3)),
    (1, 2): ((1, 1), (1, 2), (1, 3)),
    (2, 2): ((2, 1), (2, 2), (2, 3)),
    (3, 2): ((3, 1), (3, 2), (3, 3)),
    (2, 0): ((2, 0),),
    (2, 1): ((2, 0), (2, 1), (2, 2)),
    (2, 3): ((2, 2), (2, 3), (2, 4)),
}

records = build_road_visibility((vertical, horizontal), footprints)
assert records[0].is_approach is True
assert records[1].is_approach is False
assert records[0].visible_cells == (
    (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
)

candidates = (
    SimpleNamespace(position=(1, 1)),
    SimpleNamespace(position=(2, 4)),
    SimpleNamespace(position=(4, 4)),
)
eligible, rejected = filter_hideout_candidates(candidates, records)
assert tuple(item.position for item in rejected) == ((1, 1),)
assert tuple(item.position for item in eligible) == ((2, 4), (4, 4))
```

- [ ] **Step 2: Run the check and confirm the missing interface**

Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_filter_check.py
```

Expected: import failure for `build_road_visibility` or `RoadVisibility`.

- [ ] **Step 3: Implement the minimal road visibility model**

Add to `roads.py`:

```python
@dataclass(frozen=True)
class RoadVisibility:
    road: MajorRoad
    visible_cells: tuple
    is_approach: bool

    def to_log_record(self):
        return {
            "road_id": self.road.road_id,
            "orientation": self.road.orientation,
            "is_approach": self.is_approach,
            "visible_cells": [
                list(position) for position in self.visible_cells
            ],
        }


def build_road_visibility(roads, footprints):
    records = []
    for road in roads:
        visible_cells = tuple(
            sorted(
                {
                    visible
                    for road_cell in road.cells
                    for visible in footprints.get(road_cell, ())
                }
            )
        )
        records.append(
            RoadVisibility(
                road=road,
                visible_cells=visible_cells,
                is_approach=road.orientation == "vertical",
            )
        )
    return tuple(records)


def filter_hideout_candidates(candidates, road_visibility):
    excluded = {
        position
        for record in road_visibility
        if record.is_approach
        for position in record.visible_cells
    }
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.position not in excluded
    )
    rejected = tuple(
        candidate
        for candidate in candidates
        if candidate.position in excluded
    )
    return eligible, rejected
```

- [ ] **Step 4: Run the focused check**

Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_filter_check.py
```

Expected: the Task 1 checks exit successfully with no assertion failure.

---

### Task 2: One shared visibility calculation and filtered R07 selection

**Files:**

- Modify: `submissions/LAB2/hide_agent/hideout.py`
- Modify: `submissions/LAB2/hide_agent/controller.py`
- Extend temporarily:
  `submissions/LAB2/revision_road_filter_check.py`

**Interfaces:**

- Consumes:
  `visibility_footprints(map_state, observation_radius)`,
  `build_road_visibility()`, and `filter_hideout_candidates()`.
- Produces:
  `_road_visibility`, `_eligible_hideouts`, and
  `_road_excluded_hideouts` as match-local controller state.
- Changes `scan_hideouts()` by adding an optional final keyword:

```python
def scan_hideouts(
    map_state,
    observation_radius,
    pacman_spawn=None,
    ghost_spawn=None,
    pacman_speed=2,
    footprints=None,
):
```

- [ ] **Step 1: Add failing controller-boundary checks**

Extend the temporary checker to verify:

```python
import inspect

from hide_agent.hideout import scan_hideouts

assert "footprints" in inspect.signature(scan_hideouts).parameters
```

Add a deterministic controller call using a bordered NumPy map. Patch
`controller.select_hideout` with a recording wrapper and assert:

```python
assert all(
    candidate.position
    not in controller_instance._road_excluded_hideouts
    for candidate in candidates_received_by_select_hideout
)
assert controller_instance._road_visibility
assert controller_instance._eligible_hideouts
```

Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_filter_check.py
```

Expected: failure because `scan_hideouts` has no `footprints` parameter or the
controller has no road-filter state.

- [ ] **Step 2: Share the exact footprint mapping**

In `hideout.py`, replace the unconditional footprint calculation with:

```python
if footprints is None:
    footprints = visibility_footprints(map_state, observation_radius)
```

Keep the old behavior for every caller that does not supply the new keyword.

- [ ] **Step 3: Add and reset match-local controller fields**

Import:

```python
from .hideout import scan_hideouts, select_hideout, visibility_footprints
from .roads import (
    build_road_visibility,
    detect_major_roads,
    filter_hideout_candidates,
    road_thresholds,
)
```

Initialize and reset:

```python
self._road_visibility = ()
self._eligible_hideouts = ()
self._road_excluded_hideouts = ()
```

- [ ] **Step 4: Build the cached mapping once per match**

Immediately after R08-A road detection:

```python
footprints = visibility_footprints(
    current_map,
    self._observation_radius,
)
self._road_visibility = build_road_visibility(
    self._major_roads,
    footprints,
)
```

Pass `footprints=footprints` into `scan_hideouts`, then filter once:

```python
(
    self._eligible_hideouts,
    self._road_excluded_hideouts,
) = filter_hideout_candidates(
    self._hideout_candidates,
    self._road_visibility,
)
```

- [ ] **Step 5: Filter destinations without altering routing**

Change only the normal unseen selector input:

```python
selection = select_hideout(
    current_map,
    my_position,
    self._eligible_hideouts,
    self._compromised_hideouts,
    preferred_position=previous_selected,
)
```

Do not change `RouteTarget`, route planning, visible escape, belief tracking,
or `HOT_UNSEEN` ordering.

- [ ] **Step 6: Run the focused checker**

Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_filter_check.py
```

Expected: Task 1 and Task 2 checks pass.

---

### Task 3: Auditable per-road and excluded-hideout diagnostics

**Files:**

- Modify: `submissions/LAB2/hide_agent/controller.py`
- Modify: `submissions/LAB2/hide_agent/diagnostics.py`
- Extend temporarily:
  `submissions/LAB2/revision_road_filter_check.py`

**Interfaces:**

- `main_road_scan` gains:
  `road_visibility`, `approach_road_ids`,
  `excluded_cell_count`, and `excluded_hideout_count`.
- `MapDiagnostics.write_snapshot()` gains:

```python
road_visibility=(),
road_excluded_hideouts=(),
```

- Machine snapshots gain `road_visibility` and
  `road_excluded_hideouts`.
- Human snapshots list road visibility and road-excluded hideout coordinates
  below the matrix; they do not overlay matrix symbols.

- [ ] **Step 1: Add failing diagnostic checks**

Extend the temporary checker to run one diagnostic-enabled controller step and
assert:

```python
scan = next(row for row in log_rows if row["event"] == "main_road_scan")
assert scan["approach_road_ids"] == [
    record.road.road_id
    for record in controller_instance._road_visibility
    if record.is_approach
]
assert scan["excluded_hideout_count"] == len(
    controller_instance._road_excluded_hideouts
)

snapshot = map_rows[0]
assert snapshot["road_visibility"]
assert snapshot["road_excluded_hideouts"] == [
    list(candidate.position)
    for candidate in controller_instance._road_excluded_hideouts
]
```

Also assert the human snapshot contains:

```python
assert "Approach road visibility" in human_text
assert "Road-excluded hideouts" in human_text
```

Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_filter_check.py
```

Expected: failure because the new diagnostic fields are absent.

- [ ] **Step 2: Extend `main_road_scan`**

Move the existing once-per-match `main_road_scan` write to immediately after
`filter_hideout_candidates()` so its exclusion counts are available. Write:

```python
road_visibility=[
    record.to_log_record()
    for record in self._road_visibility
],
approach_road_ids=[
    record.road.road_id
    for record in self._road_visibility
    if record.is_approach
],
excluded_cell_count=len(
    {
        position
        for record in self._road_visibility
        if record.is_approach
        for position in record.visible_cells
    }
),
excluded_hideout_count=len(self._road_excluded_hideouts),
```

Keep `main_road_scan` once per match.

- [ ] **Step 3: Extend machine and human map snapshots**

Serialize road records with:

```python
road_records = [
    record.to_log_record() for record in road_visibility
]
excluded_positions = [
    list(candidate.position)
    for candidate in road_excluded_hideouts
]
```

Add those lists to the machine snapshot. In the human snapshot, list each
approach road ID followed by its complete visible coordinate list, then list
all excluded hideout coordinates. Do not modify the matrix symbols.

- [ ] **Step 4: Pass diagnostics from the controller**

Add:

```python
road_visibility=self._road_visibility,
road_excluded_hideouts=self._road_excluded_hideouts,
```

to `write_snapshot()`.

- [ ] **Step 5: Run all focused checks**

Run:

```powershell
rtk python -X utf8 submissions/LAB2/revision_road_filter_check.py
```

Expected: all road visibility, filtering, controller, and diagnostics checks
pass.

---

### Task 4: Integration verification and phase record

**Files:**

- Modify:
  `submissions/LAB2/HIDE-AGENT-PHASES.md`
- Delete:
  `submissions/LAB2/revision_road_filter_check.py`

**Interfaces:**

- No new runtime interfaces.
- The phase board must describe this increment as the only road-aware behavior
  beyond R08-A.

- [ ] **Step 1: Compile the exact source**

Run:

```powershell
rtk python -X utf8 -c "from pathlib import Path; files=[Path(r'submissions/LAB2/agent.py'),*Path(r'submissions/LAB2/hide_agent').glob('*.py')]; [compile(p.read_text(encoding='utf-8'),str(p),'exec') for p in files]; print(len(files))"
```

Expected: `13` and exit code zero.

- [ ] **Step 2: Run one Arena game**

Use the project’s existing LAB2 game command with stochastic starts and
diagnostics enabled. Do not add flags not documented in `LAB2-INFO.md`.

Expected log evidence:

- exactly one `main_road_scan`;
- at least one vertical approach road on the standard 21×21 map;
- every road-excluded hideout absent from the selector input;
- one legal `decision` and one map snapshot per Ghost step;
- no `road_phase_changed`, `road_fallback`, timing, or relocation event.

- [ ] **Step 3: Inspect behavior, not only diagnostics**

Parse the retained game and report:

```text
detected road count
vertical approach road IDs
excluded hideout count
selected hideout
whether selected hideout occurs in any approach-road visible set
first returned move
maximum runtime_ms
```

Required result: the selected hideout is not in any vertical approach-road
visibility set.

- [ ] **Step 4: Update the phase document**

Replace the obsolete next R08 phase description with this implemented scope:

```text
Compute exact cached visibility per road and exclude from normal strategic
selection every hideout visible from a vertical DOWN↔TOP approach road.
Horizontal roads do not exclude hideouts; routing, timing, relocation, and
fallback remain unchanged.
```

Record the actual focused-check and Arena evidence. Do not mark any later
timing, relocation, or fallback phase complete.

- [ ] **Step 5: Remove the temporary checker and recompile**

Delete only:

```text
submissions/LAB2/revision_road_filter_check.py
```

Then repeat the source-only compilation command from Step 1.

Expected: compilation succeeds and no temporary checker remains.
