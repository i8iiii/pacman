"""Temporary focused checks for the five-turn road-switching cycle."""

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hide_agent.controller import HideController
from hide_agent.roads import (
    MajorRoad,
    RoadVisibility,
    build_road_cycle,
    filter_hideout_candidates,
    select_closest_safe_hideout,
)


def road(road_id, orientation, cells):
    return MajorRoad(
        road_id=road_id,
        orientation=orientation,
        start=cells[0],
        end=cells[-1],
        cells=tuple(cells),
        length=len(cells),
    )


top = road(0, "horizontal", [(2, column) for column in range(1, 20)])
bottom = road(
    1,
    "horizontal",
    [(18, column) for column in range(1, 20)],
)
left = road(2, "vertical", [(row, 5) for row in range(2, 19)])
right = road(3, "vertical", [(row, 15) for row in range(2, 19)])
records = tuple(
    RoadVisibility(
        road=item,
        visible_cells=item.cells,
        is_approach=item.orientation == "vertical",
    )
    for item in (top, bottom, left, right)
)

top_cycle = build_road_cycle(records, (1, 10), (21, 21))
assert top_cycle.ghost_side == "top"
assert tuple(stage.road_ids for stage in top_cycle.stages) == (
    (2, 3),
    (0,),
    (2, 3),
    (1,),
)
assert tuple(stage.label for stage in top_cycle.stages) == (
    "vertical_up",
    "top_horizontal",
    "vertical_down",
    "bottom_horizontal",
)

bottom_cycle = build_road_cycle(records, (19, 10), (21, 21))
assert bottom_cycle.ghost_side == "bottom"
assert tuple(stage.road_ids for stage in bottom_cycle.stages) == (
    (2, 3),
    (1,),
    (2, 3),
    (0,),
)
assert tuple(stage.label for stage in bottom_cycle.stages) == (
    "vertical_down",
    "bottom_horizontal",
    "vertical_up",
    "top_horizontal",
)

elapsed_turns = (0, 4, 5, 9, 10, 14, 15, 19, 20)
assert tuple(
    top_cycle.requested_index(elapsed)
    for elapsed in elapsed_turns
) == (0, 0, 1, 1, 2, 2, 3, 3, 0)

vertical_only = build_road_cycle(records[2:], (1, 10), (21, 21))
assert vertical_only.stages[0].road_ids == (2, 3)
assert vertical_only.stages[1].road_ids == ()
assert vertical_only.stages[2].road_ids == ()
assert vertical_only.stages[3].road_ids == ()

partition_candidates = (
    SimpleNamespace(position=(2, 5)),
    SimpleNamespace(position=(2, 6)),
    SimpleNamespace(position=(10, 10)),
)
partition_visibility = (
    RoadVisibility(
        road=top,
        visible_cells=((2, 5),),
        is_approach=False,
    ),
    RoadVisibility(
        road=left,
        visible_cells=((2, 6),),
        is_approach=True,
    ),
)
eligible, rejected = filter_hideout_candidates(
    partition_candidates,
    partition_visibility,
    active_road_ids=(0,),
)
assert tuple(item.position for item in eligible) == (
    (2, 6),
    (10, 10),
)
assert tuple(item.position for item in rejected) == ((2, 5),)


def candidate(position, inspection_depth):
    return SimpleNamespace(
        position=position,
        gate_depth=1,
        inspection_depth=inspection_depth,
        visibility_footprint=3,
        must_backtrack=True,
        spawn_discovery_distance=2,
        opposite_vertical_band=True,
    )


grid = np.zeros((7, 7), dtype=int)
near = candidate((3, 2), inspection_depth=0)
far_better_quality = candidate((3, 5), inspection_depth=9)
closest = select_closest_safe_hideout(
    grid,
    (3, 3),
    (far_better_quality, near),
    (),
)
assert closest.candidate is near
assert closest.route_distance == 1

equal_low = candidate((2, 3), inspection_depth=1)
equal_high = candidate((4, 3), inspection_depth=2)
quality_tie_break = select_closest_safe_hideout(
    grid,
    (3, 3),
    (equal_low, equal_high),
    (),
)
assert quality_tie_break.candidate is equal_high

compromised_result = select_closest_safe_hideout(
    grid,
    (3, 3),
    (near, far_better_quality),
    (near.position,),
)
assert compromised_result.candidate is far_better_quality

controller_grid = np.ones((21, 21), dtype=int)
controller_grid[4, 1:20] = 0
controller_grid[19, 1:20] = 0
controller_grid[1:20, 5] = 0
controller_grid[1:20, 15] = 0
controller_grid[1, 5:11] = 0

with tempfile.TemporaryDirectory() as temporary_directory:
    debug_root = Path(temporary_directory)
    controller = HideController(
        log_path=debug_root / "hide-agent.log",
        map_text_path=debug_root / "hide-agent-map.txt",
        map_jsonl_path=debug_root / "hide-agent-map.jsonl",
        diagnostics_enabled=True,
    )
    observed_stages = []
    for step_number in (1, 5, 6, 10, 11, 15, 16, 20, 21):
        controller.step(
            controller_grid,
            (1, 10),
            None,
            step_number,
        )
        observed_stages.append(controller._active_road_stage.index)
    controller_log = [
        json.loads(line)
        for line in (debug_root / "hide-agent.log")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    controller_maps = [
        json.loads(line)
        for line in (debug_root / "hide-agent-map.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    controller_human_map = (
        debug_root / "hide-agent-map.txt"
    ).read_text(encoding="utf-8")

assert observed_stages == [0, 0, 1, 1, 2, 2, 3, 3, 0]
assert controller._road_cycle.stage_turns == 5
assert controller._active_road_ids == (2, 3)

cycle_events = [
    row
    for row in controller_log
    if row.get("event") == "road_cycle_built"
]
stage_events = [
    row
    for row in controller_log
    if row.get("event") == "road_stage_changed"
]
assert len(cycle_events) == 1
assert [row["active_stage"] for row in stage_events] == [
    0,
    1,
    2,
    3,
    0,
]
assert [row["elapsed_turns"] for row in stage_events] == [
    0,
    5,
    10,
    15,
    20,
]
assert stage_events[1]["released_road_ids"] == [2, 3]
assert stage_events[1]["active_road_ids"] == [0]

assert [
    row["active_road_stage"]["index"]
    for row in controller_maps
] == observed_stages
assert controller_maps[-1]["active_road_ids"] == [2, 3]
assert controller_maps[-1]["active_road_excluded_cells"]
assert "Road cycle stage:" in controller_human_map
assert "Active road IDs" in controller_human_map
assert "Active road excluded cells" in controller_human_map

print("road cycle checks passed")
