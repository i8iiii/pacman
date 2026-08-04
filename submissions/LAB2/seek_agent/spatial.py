"""Shared spatial operations for seek-agent modules."""

from collections import deque
from heapq import heappop, heappush

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


def minimum_turn_path(
    topology,
    start,
    goal,
    pacman_speed=2,
    allowed_cells=None,
):
    """Find a legal path which minimises Pacman game turns.

    A turn may travel one to ``pacman_speed`` consecutive cells in one
    cardinal direction.  The returned result remains cell-by-cell so it can
    be inspected for visibility and later converted back into game actions.
    ``allowed_cells`` limits *every* visited cell, including both endpoints.
    """
    start = normalize_position(start)
    goal = normalize_position(goal)
    speed = _normalise_speed(pacman_speed)
    allowed = None
    if allowed_cells is not None:
        allowed = frozenset(normalize_position(cell) for cell in allowed_cells)

    if not _permitted_cell(topology, start, allowed):
        return None
    if not _permitted_cell(topology, goal, allowed):
        return None
    if start == goal:
        return [start]

    # Priority is game turns, then moved cells, then a stable cell/action
    # spelling.  Keeping the full spelling in the label makes ties independent
    # of heap insertion order.
    start_path = (start,)
    frontier = [(0, 0, start_path, (), start)]
    best = {start: (0, 0, start_path, ())}

    while frontier:
        turns, cell_steps, cells, actions, current = heappop(frontier)
        label = (turns, cell_steps, cells, actions)
        if best.get(current) != label:
            continue
        if current == goal:
            return list(cells)

        for move_index, move in enumerate(CARDINAL_MOVES):
            segment = []
            position = current
            for step_count in range(1, speed + 1):
                position = apply_move(position, move)
                if not _permitted_cell(topology, position, allowed):
                    break
                segment.append(position)
                candidate_cells = cells + tuple(segment)
                candidate_actions = actions + ((move_index, step_count),)
                candidate = (
                    turns + 1,
                    cell_steps + step_count,
                    candidate_cells,
                    candidate_actions,
                )
                previous = best.get(position)
                if previous is not None and previous <= candidate:
                    continue
                best[position] = candidate
                heappush(frontier, candidate + (position,))

    return None


def path_to_actions(path, pacman_speed=2):
    """Compact a cell-by-cell path into legal ``(Move, steps)`` actions."""
    if path is None:
        raise ValueError("Path must be an iterable of cells, not None")
    cells = tuple(normalize_position(cell) for cell in path)
    if len(cells) < 2:
        return ()

    speed = _normalise_speed(pacman_speed)
    actions = []
    index = 0
    while index < len(cells) - 1:
        move = move_for_delta(movement_delta(cells[index], cells[index + 1]))
        run_length = 1
        while (
            index + run_length < len(cells) - 1
            and movement_delta(
                cells[index + run_length],
                cells[index + run_length + 1],
            )
            == move.value
        ):
            run_length += 1

        remaining = run_length
        while remaining:
            action_steps = min(speed, remaining)
            actions.append((move, action_steps))
            remaining -= action_steps
        index += run_length

    return tuple(actions)


def count_path_turns(path, pacman_speed=2):
    """Return the number of game actions required to follow ``path``."""
    return len(path_to_actions(path, pacman_speed))


def concatenate_paths(*paths):
    """Join cell paths without repeating a common boundary cell."""
    combined = []
    for path in paths:
        if path is None:
            raise ValueError("Cannot concatenate a missing path")
        cells = [normalize_position(cell) for cell in path]
        if not cells:
            continue
        if combined and combined[-1] == cells[0]:
            cells = cells[1:]
        combined.extend(cells)
    return combined


def visible_cells_for_actions(topology, path, actions, radius=5):
    """Simulate visibility at the initial cell and each action endpoint.

    Pacman observes before its first action and after each game action.  A
    two-cell action therefore does not create a separate observation from its
    intermediate crossed cell.
    """
    cells = tuple(normalize_position(cell) for cell in path)
    if not cells:
        return frozenset()

    observation_indices = [0]
    index = 0
    for _, steps in actions:
        try:
            steps = int(steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action steps must be positive integers") from exc
        if steps < 1:
            raise ValueError("Action steps must be positive integers")
        index += steps
        if index >= len(cells):
            raise ValueError("Actions extend beyond the supplied path")
        observation_indices.append(index)
    if index != len(cells) - 1:
        raise ValueError("Actions do not cover the supplied path")

    visible = set()
    for index in observation_indices:
        visible.update(visibility_footprint(topology, cells[index], radius))
    return frozenset(visible)


def visible_cells_for_path(topology, path, pacman_speed=2, radius=5):
    """Simulate path visibility at initial and action-endpoint positions."""
    return visible_cells_for_actions(
        topology,
        path,
        path_to_actions(path, pacman_speed),
        radius,
    )


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


def _normalise_speed(pacman_speed):
    try:
        return max(1, int(pacman_speed))
    except (TypeError, ValueError) as exc:
        raise ValueError("pacman_speed must be a positive integer") from exc


def _permitted_cell(topology, position, allowed_cells):
    return is_traversable(topology, position) and (
        allowed_cells is None or position in allowed_cells
    )
