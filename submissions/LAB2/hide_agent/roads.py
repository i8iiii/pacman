"""Deterministic major-road detection for Hide diagnostics."""

from dataclasses import dataclass

from .hideout import HideoutSelection, hideout_quality_rank
from .navigation import reconstruct_path, structural_shortest_paths


@dataclass(frozen=True)
class MajorRoad:
    """One maximal straight traversable run meeting the major-road threshold."""

    road_id: int
    orientation: str
    start: tuple
    end: tuple
    cells: tuple
    length: int

    def to_log_record(self):
        return {
            "road_id": self.road_id,
            "orientation": self.orientation,
            "start": list(self.start),
            "end": list(self.end),
            "length": self.length,
            "cells": [list(position) for position in self.cells],
        }


@dataclass(frozen=True)
class RoadVisibility:
    """The cells Pacman can see while traversing one detected road."""

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


@dataclass(frozen=True)
class RoadCycleStage:
    """One five-turn road set in the repeating search prediction."""

    index: int
    label: str
    road_ids: tuple

    def to_log_record(self):
        return {
            "index": self.index,
            "label": self.label,
            "road_ids": list(self.road_ids),
        }


@dataclass(frozen=True)
class RoadCycle:
    """The spawn-mirrored four-stage road prediction."""

    ghost_side: str
    stage_turns: int
    stages: tuple

    def requested_index(self, elapsed_turns):
        return (
            max(0, int(elapsed_turns))
            // max(1, int(self.stage_turns))
        ) % len(self.stages)

    def stage(self, index):
        return self.stages[int(index) % len(self.stages)]

    def to_log_record(self):
        return {
            "ghost_side": self.ghost_side,
            "stage_turns": self.stage_turns,
            "stages": [
                stage.to_log_record() for stage in self.stages
            ],
        }


def detect_major_roads(map_state):
    """Return deterministic maximal roads spanning at least two-thirds."""

    rows, columns = map_state.shape
    horizontal_threshold = _two_thirds_ceiling(columns)
    vertical_threshold = _two_thirds_ceiling(rows)
    records = []

    for row in range(rows):
        records.extend(
            _line_roads(
                map_state,
                orientation="horizontal",
                fixed=row,
                length=columns,
                threshold=horizontal_threshold,
            )
        )

    for column in range(columns):
        records.extend(
            _line_roads(
                map_state,
                orientation="vertical",
                fixed=column,
                length=rows,
                threshold=vertical_threshold,
            )
        )

    records.sort(
        key=lambda record: (
            record[0],
            record[1][0],
            record[1][1],
            record[2][0],
            record[2][1],
        )
    )
    return tuple(
        MajorRoad(
            road_id=road_id,
            orientation=orientation,
            start=start,
            end=end,
            cells=cells,
            length=len(cells),
        )
        for road_id, (orientation, start, end, cells) in enumerate(records)
    )


def build_road_visibility(roads, footprints):
    """Cache the complete visible-cell set for every detected road."""

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


def build_road_cycle(
    road_visibility,
    ghost_spawn,
    map_shape,
    stage_turns=20,
):
    """Build the connected four-stage cycle for one match."""

    rows = int(map_shape[0])
    ghost_side = (
        "top"
        if int(ghost_spawn[0]) < rows / 2.0
        else "bottom"
    )
    vertical = tuple(
        record
        for record in road_visibility
        if record.road.orientation == "vertical"
    )
    horizontal = tuple(
        record
        for record in road_visibility
        if record.road.orientation == "horizontal"
    )

    ghost_horizontal = _select_connected_horizontal(
        horizontal,
        vertical,
        side=ghost_side,
    )
    reverse_vertical = (
        ()
        if ghost_horizontal is None
        else tuple(
            record
            for record in vertical
            if _roads_intersect(
                record.road,
                ghost_horizontal.road,
            )
        )
    )
    opposite_side = (
        "bottom" if ghost_side == "top" else "top"
    )
    opposite_horizontal = _select_connected_horizontal(
        horizontal,
        reverse_vertical,
        side=opposite_side,
    )

    if ghost_side == "top":
        labels = (
            "vertical_up",
            "top_horizontal",
            "vertical_down",
            "bottom_horizontal",
        )
    else:
        labels = (
            "vertical_down",
            "bottom_horizontal",
            "vertical_up",
            "top_horizontal",
        )

    road_sets = (
        vertical,
        () if ghost_horizontal is None else (ghost_horizontal,),
        reverse_vertical,
        (
            ()
            if opposite_horizontal is None
            else (opposite_horizontal,)
        ),
    )
    stages = tuple(
        RoadCycleStage(
            index=index,
            label=labels[index],
            road_ids=tuple(
                record.road.road_id
                for record in road_sets[index]
            ),
        )
        for index in range(4)
    )
    return RoadCycle(
        ghost_side=ghost_side,
        stage_turns=max(1, int(stage_turns)),
        stages=stages,
    )


def filter_hideout_candidates(
    candidates,
    road_visibility,
    active_road_ids=None,
):
    """Return candidates outside and inside the active road visibility."""

    active_ids = (
        {
            record.road.road_id
            for record in road_visibility
            if record.is_approach
        }
        if active_road_ids is None
        else {int(road_id) for road_id in active_road_ids}
    )
    excluded = {
        position
        for record in road_visibility
        if record.road.road_id in active_ids
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


def select_closest_safe_hideout(
    map_state,
    ghost_position,
    candidates,
    compromised,
):
    """Choose the closest reachable safe hideout deterministically."""

    distances, parents = structural_shortest_paths(
        map_state,
        ghost_position,
    )
    compromised_positions = {
        tuple(position) for position in compromised
    }
    rejections = {"compromised": 0, "unreachable": 0}
    reachable = []
    for candidate in candidates:
        if candidate.position in compromised_positions:
            rejections["compromised"] += 1
        elif candidate.position not in distances:
            rejections["unreachable"] += 1
        else:
            reachable.append(candidate)

    if not reachable:
        return HideoutSelection(
            candidate=None,
            path=(),
            route_distance=None,
            rank=(),
            admitted_count=0,
            rejections=rejections,
        )

    selected = max(
        reachable,
        key=lambda candidate: (
            -distances[candidate.position],
            *hideout_quality_rank(candidate),
            -candidate.position[0],
            -candidate.position[1],
        ),
    )
    route_distance = distances[selected.position]
    rank = (
        -route_distance,
        *hideout_quality_rank(selected),
        -selected.position[0],
        -selected.position[1],
    )
    return HideoutSelection(
        candidate=selected,
        path=tuple(reconstruct_path(parents, selected.position)),
        route_distance=route_distance,
        rank=rank,
        admitted_count=len(reachable),
        rejections=rejections,
    )


def _select_connected_horizontal(horizontal, vertical, side):
    connected = tuple(
        record
        for record in horizontal
        if any(
            _roads_intersect(record.road, other.road)
            for other in vertical
        )
    )
    if not connected:
        return None
    if side == "top":
        return min(
            connected,
            key=lambda record: (
                record.road.start[0],
                record.road.road_id,
            ),
        )
    return min(
        connected,
        key=lambda record: (
            -record.road.start[0],
            record.road.road_id,
        ),
    )


def _roads_intersect(first, second):
    return bool(set(first.cells).intersection(second.cells))


def road_thresholds(map_state):
    """Return the horizontal and vertical major-road length thresholds."""

    rows, columns = map_state.shape
    return {
        "horizontal": _two_thirds_ceiling(columns),
        "vertical": _two_thirds_ceiling(rows),
    }


def _two_thirds_ceiling(dimension):
    return (2 * int(dimension) + 2) // 3


def _line_roads(map_state, orientation, fixed, length, threshold):
    roads = []
    run_start = None

    for offset in range(length + 1):
        traversable = (
            offset < length
            and _line_cell_is_traversable(
                map_state,
                orientation,
                fixed,
                offset,
            )
        )
        if traversable and run_start is None:
            run_start = offset
            continue
        if traversable or run_start is None:
            continue

        run_length = offset - run_start
        if run_length >= threshold:
            cells = tuple(
                _line_position(orientation, fixed, position)
                for position in range(run_start, offset)
            )
            roads.append((orientation, cells[0], cells[-1], cells))
        run_start = None

    return roads


def _line_cell_is_traversable(map_state, orientation, fixed, offset):
    position = _line_position(orientation, fixed, offset)
    return int(map_state[position]) != 1


def _line_position(orientation, fixed, offset):
    if orientation == "horizontal":
        return (fixed, offset)
    return (offset, fixed)
