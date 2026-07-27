"""Direct structural routing to a safe campsite."""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush

from .geometry import (
    CARDINAL_MOVES,
    apply_move,
    is_structurally_traversable,
)


@dataclass(frozen=True)
class RouteTarget:
    """A reachable no-sight target and its structural route."""

    kind: str
    position: tuple
    path: tuple


@dataclass(frozen=True)
class ConcealedRoute:
    """One structural route ranked by predicted sight exposure."""

    path: tuple
    mode: str
    road_exposed_steps: int
    footprint_cost: int


def concealment_route(
    map_state,
    start,
    target,
    footprints,
    active_road_visible,
):
    """Choose zero-road-exposure, least-exposed, then shortest routing."""

    start = tuple(start)
    target = tuple(target)
    if footprints is None or active_road_visible is None:
        return _shortest_route_fallback(map_state, start, target)
    if not is_structurally_traversable(map_state, start):
        return _shortest_route_fallback(map_state, start, target)
    if not is_structurally_traversable(map_state, target):
        return _shortest_route_fallback(map_state, start, target)

    exposed = {tuple(position) for position in active_road_visible}
    best = {start: (0, 0, 0)}
    parents = {start: None}
    queue = [(0, 0, 0, start)]

    while queue:
        road_cost, footprint_cost, steps, position = heappop(queue)
        cost = (road_cost, footprint_cost, steps)
        if best.get(position) != cost:
            continue
        if position == target:
            return ConcealedRoute(
                path=tuple(reconstruct_path(parents, target)),
                mode=(
                    "concealed"
                    if road_cost == 0
                    else "least_exposed"
                ),
                road_exposed_steps=road_cost,
                footprint_cost=footprint_cost,
            )

        for move in CARDINAL_MOVES:
            neighbor = apply_move(position, move)
            if not is_structurally_traversable(map_state, neighbor):
                continue
            next_cost = (
                road_cost + int(neighbor in exposed),
                footprint_cost
                + len(footprints.get(neighbor, ())),
                steps + 1,
            )
            if next_cost >= best.get(
                neighbor,
                (10**9, 10**9, 10**9),
            ):
                continue
            best[neighbor] = next_cost
            parents[neighbor] = position
            heappush(queue, (*next_cost, neighbor))

    return _shortest_route_fallback(map_state, start, target)


def _shortest_route_fallback(map_state, start, target):
    distances, parents = structural_shortest_paths(map_state, start)
    path = (
        tuple(reconstruct_path(parents, target))
        if target in distances
        else ()
    )
    return ConcealedRoute(
        path=path,
        mode="shortest_fallback",
        road_exposed_steps=0,
        footprint_cost=0,
    )


def choose_no_sight_target(
    map_state,
    start,
    safe_campsites,
    preferred_position=None,
):
    """Select a reachable safe campsite, preserving an existing selection."""
    distances, parents = structural_shortest_paths(map_state, start)
    reachable_campsites = [
        campsite
        for campsite in safe_campsites
        if campsite.position in distances
    ]

    if not reachable_campsites:
        return None, None

    preferred_position = (
        None if preferred_position is None else tuple(preferred_position)
    )
    selected = next(
        (
            campsite
            for campsite in reachable_campsites
            if campsite.position == preferred_position
        ),
        None,
    )
    if selected is None:
        selected = max(
            reachable_campsites,
            key=lambda campsite: (
                campsite.score,
                -distances[campsite.position],
                -campsite.position[0],
                -campsite.position[1],
            ),
        )

    return (
        RouteTarget(
            kind="safe_campsite",
            position=selected.position,
            path=tuple(reconstruct_path(parents, selected.position)),
        ),
        selected,
    )


def structural_shortest_paths(map_state, start):
    """Return BFS paths through every non-wall cell, including fog."""
    start = tuple(start)
    if not is_structurally_traversable(map_state, start):
        return {}, {}

    distances = {start: 0}
    parents = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for move in CARDINAL_MOVES:
            neighbor = apply_move(current, move)
            if neighbor in distances:
                continue
            if not is_structurally_traversable(map_state, neighbor):
                continue
            distances[neighbor] = distances[current] + 1
            parents[neighbor] = current
            queue.append(neighbor)

    return distances, parents


def reconstruct_path(parents, target):
    """Return positions after the route start, ending at target."""
    target = tuple(target)
    if target not in parents:
        return []

    path = []
    current = target
    while parents[current] is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def route_moves(start, path):
    """Convert an adjacent position path to Move values."""
    moves = []
    current = tuple(start)
    for position in path:
        move = move_between(current, position)
        if move is None:
            return []
        moves.append(move)
        current = tuple(position)
    return moves


def move_between(start, end):
    """Return the cardinal move between adjacent cells, if one exists."""
    start = tuple(start)
    end = tuple(end)
    for move in CARDINAL_MOVES:
        if apply_move(start, move) == end:
            return move
    return None


def route_is_structural(map_state, start, path):
    """Return whether a path is adjacent and entirely non-wall."""
    current = tuple(start)
    for position in path:
        position = tuple(position)
        if move_between(current, position) is None:
            return False
        if not is_structurally_traversable(map_state, position):
            return False
        current = position
    return True
