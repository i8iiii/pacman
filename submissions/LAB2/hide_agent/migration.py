"""Spawn-relative progressive hideout migration."""

from dataclasses import dataclass

from .cross_map import opposite_outer_band, vertical_band
from .hideout import (
    HideoutCandidate,
    HideoutSelection,
    hideout_quality_rank,
    is_strategic_hideout,
)
from .navigation import reconstruct_path, structural_shortest_paths
from .roads import main_junction_manhattan_distance


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
