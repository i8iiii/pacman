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
