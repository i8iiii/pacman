"""Direct structural routing to a safe campsite."""

from collections import deque
from dataclasses import dataclass

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
