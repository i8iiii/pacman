"""Movement, visibility, and capture geometry shared by Hide behaviors."""

from environment import Move


CARDINAL_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def in_bounds(map_state, position):
    row, column = position
    rows, columns = map_state.shape
    return 0 <= row < rows and 0 <= column < columns


def is_structurally_traversable(map_state, position):
    """Walls are blocked; observed and unseen empty cells are traversable."""
    return in_bounds(map_state, position) and int(map_state[position]) != 1


def is_observed_traversable(map_state, position):
    """Return whether Hide has directly observed the cell as open."""
    return in_bounds(map_state, position) and int(map_state[position]) == 0


def apply_move(position, move, distance=1):
    row_delta, column_delta = move.value
    return (
        position[0] + row_delta * distance,
        position[1] + column_delta * distance,
    )


def ghost_move_options(map_state, position):
    """Return confirmed and structurally possible one-step Ghost moves."""
    confirmed = [Move.STAY]
    possible = [Move.STAY]

    for move in CARDINAL_MOVES:
        endpoint = apply_move(position, move)
        if not is_structurally_traversable(map_state, endpoint):
            continue
        possible.append(move)
        if is_observed_traversable(map_state, endpoint):
            confirmed.append(move)

    return confirmed, possible


def pacman_endpoints(map_state, position, speed=2):
    """Enumerate Pacman's legal endpoints after one straight action."""
    endpoints = {tuple(position)}
    max_speed = max(1, int(speed))

    for move in CARDINAL_MOVES:
        for distance in range(1, max_speed + 1):
            endpoint = apply_move(position, move, distance)
            if not is_structurally_traversable(map_state, endpoint):
                break
            endpoints.add(endpoint)

    return endpoints


def manhattan_distance(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def is_capture(pacman_position, ghost_position, capture_distance=2):
    """The Arena captures when Manhattan distance is below the threshold."""
    threshold = max(1, int(capture_distance))
    return manhattan_distance(pacman_position, ghost_position) < threshold


def has_line_of_sight(map_state, observer, target, radius=5):
    """Match the Arena's wall-blocked cardinal cross visibility."""
    if not in_bounds(map_state, observer) or not in_bounds(map_state, target):
        return False

    row_difference = target[0] - observer[0]
    column_difference = target[1] - observer[1]
    if row_difference != 0 and column_difference != 0:
        return False

    distance = abs(row_difference) + abs(column_difference)
    if distance > max(0, int(radius)):
        return False
    if distance == 0:
        return True

    row_step = 0 if row_difference == 0 else (1 if row_difference > 0 else -1)
    column_step = (
        0 if column_difference == 0 else (1 if column_difference > 0 else -1)
    )

    for offset in range(1, distance):
        cell = (
            observer[0] + row_step * offset,
            observer[1] + column_step * offset,
        )
        if int(map_state[cell]) == 1:
            return False

    return True


def geometry_summary(
    map_state,
    ghost_position,
    enemy_position,
    pacman_speed=2,
    capture_distance=2,
    observation_radius=5,
):
    confirmed_moves, possible_moves = ghost_move_options(map_state, ghost_position)
    summary = {
        "confirmed_ghost_moves": [move.name for move in confirmed_moves],
        "possible_ghost_moves": [move.name for move in possible_moves],
        "enemy_visible": enemy_position is not None,
        "enemy_line_of_sight": None,
        "current_capture": None,
        "pacman_endpoints": [],
        "capturing_pacman_endpoints": [],
    }

    if enemy_position is None:
        return summary

    endpoints = sorted(pacman_endpoints(map_state, enemy_position, pacman_speed))
    capturing_endpoints = [
        endpoint
        for endpoint in endpoints
        if is_capture(endpoint, ghost_position, capture_distance)
    ]
    summary.update(
        {
            "enemy_line_of_sight": has_line_of_sight(
                map_state,
                ghost_position,
                enemy_position,
                observation_radius,
            ),
            "current_capture": is_capture(
                enemy_position,
                ghost_position,
                capture_distance,
            ),
            "pacman_endpoints": [list(endpoint) for endpoint in endpoints],
            "capturing_pacman_endpoints": [
                list(endpoint) for endpoint in capturing_endpoints
            ],
        }
    )
    return summary


"""Direct structural routing to a safe campsite."""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush



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


"""Spawn-relative vertical band helpers."""


def vertical_band(position, rows):
    """Return the proportional vertical band containing position."""

    row = int(position[0])
    rows = max(1, int(rows))
    scaled = 3 * row
    if scaled < rows:
        return "top"
    if scaled < 2 * rows:
        return "middle"
    return "bottom"


def opposite_outer_band(ghost_spawn, rows):
    """Return the outer third opposite the original spawn half."""

    return (
        "bottom"
        if int(ghost_spawn[0]) < int(rows) / 2.0
        else "top"
    )
