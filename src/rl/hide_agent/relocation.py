"""Deterministic major-road detection for Hide diagnostics."""

from dataclasses import dataclass

from rl.hide_agent.concealment import (
    HideoutCandidate,
    HideoutSelection,
    hideout_quality_rank,
    is_strategic_hideout,
)
from rl.hide_agent.spatial import reconstruct_path, structural_shortest_paths


ROAD_STAGE_TURNS = 10


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
    """One timed road set in the repeating search prediction."""

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


@dataclass(frozen=True)
class RelocationDecision:
    """A safe-hideout choice plus its gradual-relocation evidence."""

    selection: HideoutSelection
    mode: str
    previous_main_junction_distance: object
    selected_main_junction_distance: object
    improving_candidate_count: int

    def to_log_fields(self):
        return {
            "relocation_mode": self.mode,
            "previous_main_junction_distance": (
                self.previous_main_junction_distance
            ),
            "selected_main_junction_distance": (
                self.selected_main_junction_distance
            ),
            "improving_candidate_count": self.improving_candidate_count,
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
    stage_turns=ROAD_STAGE_TURNS,
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
        map_rows=rows,
    )
    reverse_vertical = (
        vertical
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
        map_rows=rows,
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


def main_junction_manhattan_distance(
    position,
    intersections,
):
    """Return distance from a position to its nearest major-road junction."""

    if position is None or not intersections:
        return None
    row, column = tuple(position)
    return min(
        abs(row - junction[0]) + abs(column - junction[1])
        for junction in intersections
    )


def select_reachable_component_fallback(
    map_state,
    ghost_position,
    footprints,
    active_road_excluded_cells,
    intersections,
    compromised=(),
    preferred_position=None,
    allowed_positions=None,
):
    """Choose a concealed waiting cell when no certified hideout is reachable.

    This does not relax hideout certification. It selects only from Ghost's
    structural BFS component and excludes every cell visible from the active
    predicted roads.
    """

    distances, parents = structural_shortest_paths(
        map_state,
        ghost_position,
    )
    road_excluded = {
        tuple(position) for position in active_road_excluded_cells
    }
    compromised_positions = {
        tuple(position) for position in compromised
    }
    allowed = (
        None
        if allowed_positions is None
        else {tuple(position) for position in allowed_positions}
    )
    admissible = [
        position
        for position in distances
        if position not in road_excluded
        and position not in compromised_positions
        and (allowed is None or position in allowed)
    ]
    rejections = {
        "road_exposed": sum(
            position in road_excluded for position in distances
        ),
        "compromised": sum(
            position in compromised_positions for position in distances
        ),
        "outside_required_band": (
            0
            if allowed is None
            else sum(position not in allowed for position in distances)
        ),
    }
    if not admissible:
        return HideoutSelection(
            candidate=None,
            path=(),
            route_distance=None,
            rank=(),
            admitted_count=0,
            rejections=rejections,
        )

    preferred_position = (
        None
        if preferred_position is None
        else tuple(preferred_position)
    )

    def rank(position):
        junction_distance = main_junction_manhattan_distance(
            position,
            intersections,
        )
        return (
            -len(footprints.get(position, ())),
            0 if junction_distance is None else junction_distance,
            -distances[position],
            -position[0],
            -position[1],
        )

    selected_position = (
        preferred_position
        if preferred_position in admissible
        else max(admissible, key=rank)
    )
    selected_rank = rank(selected_position)
    selected = HideoutCandidate(
        position=selected_position,
        kind="reachable_fallback",
        entrance=None,
        gate_depth=0,
        must_backtrack=False,
        entrance_hidden=False,
        inspection_depth=0,
        visibility_footprint=len(
            footprints.get(selected_position, ())
        ),
    )
    return HideoutSelection(
        candidate=selected,
        path=tuple(reconstruct_path(parents, selected_position)),
        route_distance=distances[selected_position],
        rank=selected_rank,
        admitted_count=len(admissible),
        rejections=rejections,
    )


def select_gradual_relocation(
    map_state,
    ghost_position,
    candidates,
    compromised,
    intersections,
    previous_hideout,
    route_slack=4,
):
    """Choose the best nearby hideout that improves junction distance."""

    distances, parents = structural_shortest_paths(
        map_state,
        ghost_position,
    )
    compromised_positions = {
        tuple(position) for position in compromised
    }
    rejections = {
        "unsafe": 0,
        "compromised": 0,
        "unreachable": 0,
        "outside_route_window": 0,
    }
    reachable = []
    for candidate in candidates:
        if not is_strategic_hideout(candidate):
            rejections["unsafe"] += 1
        elif candidate.position in compromised_positions:
            rejections["compromised"] += 1
        elif candidate.position not in distances:
            rejections["unreachable"] += 1
        else:
            reachable.append(candidate)

    previous_distance = main_junction_manhattan_distance(
        previous_hideout,
        intersections,
    )
    improving = (
        []
        if previous_distance is None
        else [
            candidate
            for candidate in reachable
            if main_junction_manhattan_distance(
                candidate.position,
                intersections,
            )
            > previous_distance
        ]
    )
    if improving:
        nearest_distance = min(
            distances[candidate.position]
            for candidate in improving
        )
        route_limit = nearest_distance + max(0, int(route_slack))
        pool = [
            candidate
            for candidate in improving
            if distances[candidate.position] <= route_limit
        ]
        rejections["outside_route_window"] = (
            len(improving) - len(pool)
        )
        mode = "junction_improving"
    else:
        pool = reachable
        mode = "closest_safe_fallback"

    if not pool:
        selection = HideoutSelection(
            candidate=None,
            path=(),
            route_distance=None,
            rank=(),
            admitted_count=0,
            rejections=rejections,
        )
        return RelocationDecision(
            selection=selection,
            mode=mode,
            previous_main_junction_distance=previous_distance,
            selected_main_junction_distance=None,
            improving_candidate_count=len(improving),
        )

    if mode == "junction_improving":
        selected = max(
            pool,
            key=lambda candidate: (
                *hideout_quality_rank(candidate),
                -distances[candidate.position],
                -candidate.position[0],
                -candidate.position[1],
            ),
        )
    else:
        selected = max(
            pool,
            key=lambda candidate: (
                -distances[candidate.position],
                *hideout_quality_rank(candidate),
                -candidate.position[0],
                -candidate.position[1],
            ),
        )
    route_distance = distances[selected.position]
    if mode == "junction_improving":
        rank = (
            *hideout_quality_rank(selected),
            -route_distance,
            -selected.position[0],
            -selected.position[1],
        )
    else:
        rank = (
            -route_distance,
            *hideout_quality_rank(selected),
            -selected.position[0],
            -selected.position[1],
        )
    selection = HideoutSelection(
        candidate=selected,
        path=tuple(reconstruct_path(parents, selected.position)),
        route_distance=route_distance,
        rank=rank,
        admitted_count=len(pool),
        rejections=rejections,
    )
    return RelocationDecision(
        selection=selection,
        mode=mode,
        previous_main_junction_distance=previous_distance,
        selected_main_junction_distance=(
            main_junction_manhattan_distance(
                selected.position,
                intersections,
            )
        ),
        improving_candidate_count=len(improving),
    )


def _select_connected_horizontal(
    horizontal,
    vertical,
    side,
    map_rows,
):
    midpoint = int(map_rows) / 2.0
    connected = tuple(
        record
        for record in horizontal
        if (
            (
                record.road.start[0] < midpoint
                if side == "top"
                else record.road.start[0] >= midpoint
            )
            and any(
                _roads_intersect(record.road, other.road)
                for other in vertical
            )
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


def main_road_intersections(roads):
    """Return cells shared by horizontal and vertical major roads."""

    horizontal_cells = {
        position
        for road in roads
        if road.orientation == "horizontal"
        for position in road.cells
    }
    vertical_cells = {
        position
        for road in roads
        if road.orientation == "vertical"
        for position in road.cells
    }
    return tuple(sorted(horizontal_cells.intersection(vertical_cells)))


def active_road_visibility_cells(
    road_visibility,
    active_road_ids,
):
    """Return the cached visibility union for the active roads."""

    active_ids = {int(road_id) for road_id in active_road_ids}
    return tuple(
        sorted(
            {
                position
                for record in road_visibility
                if record.road.road_id in active_ids
                for position in record.visible_cells
            }
        )
    )


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


"""Spawn-relative progressive hideout migration."""

from dataclasses import dataclass

from rl.hide_agent.spatial import opposite_outer_band, vertical_band
from rl.hide_agent.concealment import (
    HideoutCandidate,
    HideoutSelection,
    hideout_quality_rank,
    is_strategic_hideout,
)
from rl.hide_agent.spatial import reconstruct_path, structural_shortest_paths


TO_MIDDLE = "TO_MIDDLE"
MIDDLE_HOLD = "MIDDLE_HOLD"
TO_OPPOSITE = "TO_OPPOSITE"
DEEPEN_OPPOSITE = "DEEPEN_OPPOSITE"
OPPOSITE_HOLD = "OPPOSITE_HOLD"

MIDDLE_HOLD_TURNS = 5
OPPOSITE_ROAD_SWITCH_TURNS = 10


@dataclass
class MigrationState:
    """The only cross-map phase and hold-counter authority."""

    phase: str = TO_MIDDLE
    middle_hold_turns: int = 0
    opposite_hold_turns: int = 0

    def reset(self):
        self.phase = TO_MIDDLE
        self.middle_hold_turns = 0
        self.opposite_hold_turns = 0


@dataclass(frozen=True)
class MigrationSelection:
    """One waypoint result plus the reason it was selected or rejected."""

    selection: HideoutSelection
    reason: str
    required_band: str
    progress_delta: int | None


def empty_hideout_selection(reason):
    return HideoutSelection(
        candidate=None,
        path=(),
        route_distance=None,
        rank=(),
        admitted_count=0,
        rejections={reason: 1},
    )


def migration_direction(ghost_spawn, rows):
    """Return +1 for top spawns and -1 for bottom spawns."""

    return 1 if int(ghost_spawn[0]) < int(rows) / 2.0 else -1


def destination_band(phase, ghost_spawn, rows):
    """Return the band required by a migration phase."""

    if phase in (TO_MIDDLE, MIDDLE_HOLD):
        return "middle"
    return opposite_outer_band(ghost_spawn, rows)


def select_progressive_waypoint(
    map_state,
    ghost_position,
    ghost_spawn,
    phase,
    certified_candidates,
    fallback_candidates,
    compromised,
    road_excluded,
    intersections,
    route_slack=4,
):
    """Choose the next nearby hiding waypoint that makes strict progress."""

    ghost_position = tuple(ghost_position)
    rows = map_state.shape[0]
    required = destination_band(phase, ghost_spawn, rows)
    distances, parents = structural_shortest_paths(
        map_state,
        ghost_position,
    )
    compromised = {tuple(position) for position in compromised}
    road_excluded = {tuple(position) for position in road_excluded}
    direction = migration_direction(ghost_spawn, rows)
    current_junction_distance = main_junction_manhattan_distance(
        ghost_position,
        intersections,
    )
    rejections = {
        "unsafe": 0,
        "compromised": 0,
        "road_exposed": 0,
        "unreachable": 0,
        "wrong_band": 0,
        "no_progress": 0,
        "outside_route_window": 0,
    }

    def progress(candidate):
        position = tuple(candidate.position)
        band = vertical_band(position, rows)
        if phase == TO_MIDDLE:
            spawn_band = vertical_band(ghost_spawn, rows)
            if band not in (spawn_band, "middle"):
                return None
            delta = direction * (
                int(position[0]) - int(ghost_position[0])
            )
            return delta if delta > 0 else None
        if phase == TO_OPPOSITE:
            if band not in ("middle", required):
                return None
            delta = direction * (
                int(position[0]) - int(ghost_position[0])
            )
            return delta if delta > 0 else None
        if phase == DEEPEN_OPPOSITE:
            if band != required or current_junction_distance is None:
                return None
            candidate_distance = main_junction_manhattan_distance(
                position,
                intersections,
            )
            if candidate_distance is None:
                return None
            delta = candidate_distance - current_junction_distance
            return delta if delta > 0 else None
        return None

    def admitted(candidates, require_strategic):
        result = []
        for candidate in candidates:
            position = tuple(candidate.position)
            if require_strategic and not is_strategic_hideout(candidate):
                rejections["unsafe"] += 1
            elif position in compromised:
                rejections["compromised"] += 1
            elif position in road_excluded:
                rejections["road_exposed"] += 1
            elif position not in distances:
                rejections["unreachable"] += 1
            else:
                delta = progress(candidate)
                if delta is None:
                    rejections["no_progress"] += 1
                else:
                    result.append((candidate, delta))
        return result

    pool = admitted(certified_candidates, require_strategic=True)
    reason = "certified_progressive_waypoint"
    if not pool:
        pool = admitted(fallback_candidates, require_strategic=False)
        reason = "fallback_progressive_waypoint"
    if not pool:
        return MigrationSelection(
            selection=empty_hideout_selection(
                "no_progressive_waypoint"
            ),
            reason="no_progressive_waypoint",
            required_band=required,
            progress_delta=None,
        )

    nearest = min(distances[item.position] for item, _ in pool)
    limit = nearest + max(0, int(route_slack))
    window = [
        (item, delta)
        for item, delta in pool
        if distances[item.position] <= limit
    ]
    rejections["outside_route_window"] = len(pool) - len(window)
    selected, selected_progress = max(
        window,
        key=lambda pair: (
            -int(pair[0].visibility_footprint),
            *hideout_quality_rank(pair[0]),
            pair[1],
            -distances[pair[0].position],
            -pair[0].position[0],
            -pair[0].position[1],
        ),
    )
    return _selection_from_candidate(
        selected,
        distances,
        parents,
        reason=reason,
        required_band=required,
        progress_delta=selected_progress,
        admitted_count=len(window),
        rejections=rejections,
    )


def select_required_band_rescue(
    map_state,
    ghost_position,
    ghost_spawn,
    phase,
    certified_candidates,
    fallback_candidates,
    compromised,
    road_excluded,
    footprints,
):
    """Select a safe reachable cell in the required band without phase drift."""

    required = destination_band(
        phase,
        ghost_spawn,
        map_state.shape[0],
    )
    distances, parents = structural_shortest_paths(
        map_state,
        ghost_position,
    )
    compromised = {tuple(position) for position in compromised}
    road_excluded = {tuple(position) for position in road_excluded}
    required_cells = [
        position
        for position in distances
        if vertical_band(position, map_state.shape[0]) == required
    ]
    if not required_cells:
        return MigrationSelection(
            selection=empty_hideout_selection(
                "required_band_unreachable"
            ),
            reason="required_band_unreachable",
            required_band=required,
            progress_delta=None,
        )

    certified = [
        candidate
        for candidate in certified_candidates
        if is_strategic_hideout(candidate)
        and candidate.position in distances
        and candidate.position not in compromised
        and candidate.position not in road_excluded
        and vertical_band(
            candidate.position,
            map_state.shape[0],
        )
        == required
    ]
    fallback = [
        candidate
        for candidate in fallback_candidates
        if candidate.position in distances
        and candidate.position not in compromised
        and candidate.position not in road_excluded
        and vertical_band(
            candidate.position,
            map_state.shape[0],
        )
        == required
    ]
    known_fallback = {candidate.position for candidate in fallback}
    fallback.extend(
        _fallback_candidate(position, footprints)
        for position in required_cells
        if position not in known_fallback
        and position not in compromised
        and position not in road_excluded
    )
    pool = certified or fallback
    if not pool:
        return MigrationSelection(
            selection=empty_hideout_selection(
                "required_band_no_safe_target"
            ),
            reason="required_band_no_safe_target",
            required_band=required,
            progress_delta=None,
        )

    selected = min(
        pool,
        key=lambda candidate: (
            distances[candidate.position],
            int(candidate.visibility_footprint),
            -int(candidate.gate_depth),
            candidate.position,
        ),
    )
    return _selection_from_candidate(
        selected,
        distances,
        parents,
        reason="required_band_rescue",
        required_band=required,
        progress_delta=None,
        admitted_count=len(pool),
        rejections={},
    )


def fallback_candidates_for_cells(positions, footprints):
    """Create non-certified hiding-cell records for structural fallback."""

    return tuple(
        _fallback_candidate(tuple(position), footprints)
        for position in positions
    )


def _fallback_candidate(position, footprints):
    return HideoutCandidate(
        position=tuple(position),
        kind="reachable_fallback",
        entrance=None,
        gate_depth=0,
        must_backtrack=False,
        entrance_hidden=False,
        inspection_depth=0,
        visibility_footprint=len(footprints.get(tuple(position), ())),
    )


def _selection_from_candidate(
    candidate,
    distances,
    parents,
    reason,
    required_band,
    progress_delta,
    admitted_count,
    rejections,
):
    route_distance = distances[candidate.position]
    rank = (
        -int(candidate.visibility_footprint),
        *hideout_quality_rank(candidate),
        0 if progress_delta is None else progress_delta,
        -route_distance,
        -candidate.position[0],
        -candidate.position[1],
    )
    return MigrationSelection(
        selection=HideoutSelection(
            candidate=candidate,
            path=tuple(
                reconstruct_path(parents, candidate.position)
            ),
            route_distance=route_distance,
            rank=rank,
            admitted_count=admitted_count,
            rejections=dict(rejections),
        ),
        reason=reason,
        required_band=required_band,
        progress_delta=progress_delta,
    )
