"""Shared spatial operations for seek-agent modules."""

from collections import deque

from environment import Move


CARDINAL_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


def normalize_position(value):
    return int(value[0]), int(value[1])


def optional_position(value):
    return None if value is None else normalize_position(value)


def apply_move(position, move):
    return (
        position[0] + move.value[0],
        position[1] + move.value[1],
    )


def movement_delta(start, end):
    return end[0] - start[0], end[1] - start[1]


def move_for_delta(delta):
    for move in CARDINAL_MOVES:
        if move.value == delta:
            return move
    raise ValueError(f"Path contains invalid movement delta: {delta}")


def is_traversable(topology, position):
    row, column = position
    return (
        0 <= row < topology.shape[0]
        and 0 <= column < topology.shape[1]
        and bool(topology[row, column])
    )


def traversable_neighbors(topology, position):
    for move in CARDINAL_MOVES:
        neighbor = apply_move(position, move)
        if is_traversable(topology, neighbor):
            yield neighbor


def shortest_path(topology, start, goal):
    if not is_traversable(topology, start):
        return None
    if not is_traversable(topology, goal):
        return None
    if start == goal:
        return [start]

    frontier = deque([start])
    parents = {start: None}

    while frontier:
        current = frontier.popleft()
        for neighbor in traversable_neighbors(topology, current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == goal:
                return _reconstruct_path(parents, goal)
            frontier.append(neighbor)

    return None


def visibility_footprint(topology, position, radius=5):
    """Return traversable cells visible on four wall-blocked cardinal rays."""
    if not is_traversable(topology, position):
        return frozenset()

    visible = {position}
    for move in CARDINAL_MOVES:
        row, column = position
        for _ in range(radius):
            row += move.value[0]
            column += move.value[1]
            target = (row, column)
            if not is_traversable(topology, target):
                break
            visible.add(target)
    return frozenset(visible)


def _reconstruct_path(parents, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path
