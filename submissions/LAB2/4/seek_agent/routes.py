"""Turn-aware local search routes for seeker areas."""

from dataclasses import dataclass
from itertools import permutations
from time import perf_counter

from .spatial import (
    concatenate_paths,
    is_traversable,
    minimum_turn_path,
    normalize_position,
    path_to_actions,
    visible_cells_for_actions,
)


@dataclass(frozen=True)
class RouteResult:
    """An immutable local-search route and the coverage it obtains."""

    cells: tuple
    actions: tuple
    turns: int
    cell_steps: int
    covered_required_cells: frozenset
    entry: tuple
    exit: tuple
    used_viewpoints: tuple
    complete: bool


def plan_area_route(
    topology,
    area,
    entry,
    exit=None,
    required_cells=(),
    pacman_speed=2,
    deadline=None,
):
    """Plan a bounded local coverage sweep inside one analysed ``Area``.

    Viewpoints are tried as every ordered subset of the area's (at most five)
    stored viewpoints.  This is deliberately exhaustive: it allows a direct
    route to win when a viewpoint is redundant, while retaining deterministic
    tie breaking for routes with equal game-turn cost.

    ``deadline`` is an absolute ``perf_counter`` deadline.  On expiry the
    best already-evaluated route is returned; an already expired deadline
    produces the safe one-cell fallback at ``entry``.
    """
    entry = normalize_position(entry)
    area_cells = frozenset(normalize_position(cell) for cell in area.cells)
    if entry not in area_cells:
        raise ValueError("Area-route entry must be inside the area's cells")
    if not is_traversable(topology, entry):
        raise ValueError("Area-route entry must be traversable")

    if exit is not None:
        exit = normalize_position(exit)
        if exit not in area_cells:
            raise ValueError("Area-route exit must be inside the area's cells")
        if not is_traversable(topology, exit):
            raise ValueError("Area-route exit must be traversable")

    required = frozenset(normalize_position(cell) for cell in required_cells)
    speed = _normalise_speed(pacman_speed)
    until = _resolve_deadline(deadline)
    fallback = _route_result(
        topology=topology,
        cells=(entry,),
        entry=entry,
        required=required,
        used_viewpoints=(),
        pacman_speed=speed,
    )

    best_complete = None
    best_incomplete = fallback
    viewpoints = _ordered_viewpoints(area.viewpoints, area_cells)
    for viewpoint_count in range(len(viewpoints) + 1):
        for selected in permutations(viewpoints, viewpoint_count):
            if _deadline_expired(until):
                return best_complete or best_incomplete

            candidate = _candidate_route(
                topology,
                area_cells,
                entry,
                exit,
                selected,
                required,
                speed,
            )
            if candidate is None:
                continue
            if candidate.complete:
                if (
                    best_complete is None
                    or _complete_key(candidate) < _complete_key(best_complete)
                ):
                    best_complete = candidate
            elif _incomplete_key(candidate) < _incomplete_key(best_incomplete):
                best_incomplete = candidate

    return best_complete or best_incomplete


def _candidate_route(
    topology,
    area_cells,
    entry,
    exit,
    viewpoints,
    required,
    pacman_speed,
):
    checkpoints = (entry,) + viewpoints
    if exit is not None:
        checkpoints += (exit,)

    segments = []
    segment_actions = []
    for start, goal in zip(checkpoints, checkpoints[1:]):
        path = minimum_turn_path(
            topology,
            start,
            goal,
            pacman_speed=pacman_speed,
            allowed_cells=area_cells,
        )
        if path is None:
            return None
        segments.append(path)
        # Each checkpoint is an intentional observation stop.  Keep the
        # segment's own actions instead of recompressing the final cell path:
        # two same-direction segments may be adjacent yet must remain two
        # turns when the checkpoint lies between them.
        segment_actions.append(path_to_actions(path, pacman_speed))

    cells = tuple(concatenate_paths(*segments)) if segments else (entry,)
    return _route_result(
        topology=topology,
        cells=cells,
        entry=entry,
        required=required,
        used_viewpoints=viewpoints,
        pacman_speed=pacman_speed,
        actions=tuple(
            action
            for segment_action_list in segment_actions
            for action in segment_action_list
        ),
    )


def _route_result(
    topology,
    cells,
    entry,
    required,
    used_viewpoints,
    pacman_speed,
    actions=None,
):
    actions = (
        path_to_actions(cells, pacman_speed)
        if actions is None else tuple(actions)
    )
    visible = visible_cells_for_actions(topology, cells, actions)
    covered = frozenset(required & visible)
    return RouteResult(
        cells=tuple(cells),
        actions=actions,
        turns=len(actions),
        cell_steps=max(0, len(cells) - 1),
        covered_required_cells=covered,
        entry=entry,
        exit=cells[-1],
        used_viewpoints=tuple(used_viewpoints),
        complete=covered == required,
    )


def _ordered_viewpoints(viewpoints, area_cells):
    """Keep stored order, discard duplicates, and never exceed five points."""
    ordered = []
    for viewpoint in viewpoints:
        viewpoint = normalize_position(viewpoint)
        if viewpoint in area_cells and viewpoint not in ordered:
            ordered.append(viewpoint)
        if len(ordered) == 5:
            break
    return tuple(ordered)


def _complete_key(route):
    return (
        route.turns,
        route.cell_steps,
        len(route.used_viewpoints),
        route.cells,
        route.used_viewpoints,
    )


def _incomplete_key(route):
    return (
        -len(route.covered_required_cells),
        route.turns,
        route.cell_steps,
        len(route.used_viewpoints),
        route.cells,
        route.used_viewpoints,
    )


def _normalise_speed(pacman_speed):
    try:
        return max(1, int(pacman_speed))
    except (TypeError, ValueError) as exc:
        raise ValueError("pacman_speed must be a positive integer") from exc


def _resolve_deadline(deadline):
    if deadline is None:
        return None
    try:
        return float(deadline)
    except (TypeError, ValueError) as exc:
        raise ValueError("deadline must be an absolute perf_counter value") from exc


def _deadline_expired(deadline):
    return deadline is not None and perf_counter() >= deadline
