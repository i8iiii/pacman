"""Visible escape selection at four-way and directionally useful T-junctions."""

from collections import deque
from dataclasses import asdict, dataclass
from random import choice

from environment import Move

from .geometry import (
    CARDINAL_MOVES,
    apply_move,
    is_capture,
    is_observed_traversable,
    is_structurally_traversable,
    manhattan_distance,
    pacman_endpoints,
)


@dataclass(frozen=True)
class EscapeBranch:
    move: Move
    endpoint: tuple
    guaranteed_safe: bool
    worst_case_distance: int
    trapped: bool
    reconnects: bool
    next_junction_distance: int | None
    escape_depth: int
    region_size: int

    @property
    def rank(self):
        next_junction_exists = self.next_junction_distance is not None
        return (
            int(self.guaranteed_safe),
            self.worst_case_distance,
            int(not self.trapped),
            int(self.reconnects),
            int(next_junction_exists),
            (
                -self.next_junction_distance
                if next_junction_exists
                else 0
            ),
            self.escape_depth,
            self.region_size,
        )

    def to_log_record(self):
        record = asdict(self)
        record["move"] = self.move.name
        record["endpoint"] = list(self.endpoint)
        record["rank"] = list(self.rank)
        return record


@dataclass(frozen=True)
class EscapeDecision:
    junction_type: str
    approach_direction: Move
    missing_direction: Move | None
    pacman_endpoints: tuple
    branches: tuple
    selected: EscapeBranch
    equivalent_moves: tuple
    mode: str


def choose_visible_junction_escape(
    map_state,
    ghost_position,
    pacman_position,
    pacman_speed=2,
    capture_distance=2,
):
    """Return the best perpendicular escape, or None outside P06's scope."""
    ghost_position = tuple(ghost_position)
    pacman_position = tuple(pacman_position)
    approach = direction_toward(ghost_position, pacman_position)
    if approach is None:
        return None

    exits = tuple(
        move
        for move in CARDINAL_MOVES
        if is_structurally_traversable(
            map_state,
            apply_move(ghost_position, move),
        )
    )
    missing_direction = None
    if len(exits) == 4:
        junction_type = "four_way"
    elif len(exits) == 3:
        junction_type = "t_junction"
        missing_direction = next(
            move for move in CARDINAL_MOVES if move not in exits
        )
        if missing_direction != opposite_move(approach):
            return None
    else:
        return None

    perpendicular = perpendicular_moves(approach)
    candidate_moves = tuple(
        move
        for move in perpendicular
        if move in exits
        and is_observed_traversable(
            map_state,
            apply_move(ghost_position, move),
        )
    )
    if len(candidate_moves) != 2:
        return None

    possible_pacman_endpoints = tuple(
        sorted(
            pacman_endpoints(
                map_state,
                pacman_position,
                pacman_speed,
            )
        )
    )
    escape_endpoints = tuple(
        apply_move(ghost_position, move) for move in candidate_moves
    )
    branches = tuple(
        _evaluate_branch(
            map_state,
            ghost_position,
            move,
            possible_pacman_endpoints,
            escape_endpoints,
            capture_distance,
        )
        for move in candidate_moves
    )
    best_rank = max(branch.rank for branch in branches)
    equivalent = tuple(
        branch for branch in branches if branch.rank == best_rank
    )
    selected = choice(equivalent)
    return EscapeDecision(
        junction_type=junction_type,
        approach_direction=approach,
        missing_direction=missing_direction,
        pacman_endpoints=possible_pacman_endpoints,
        branches=branches,
        selected=selected,
        equivalent_moves=tuple(branch.move for branch in equivalent),
        mode=(
            "guaranteed"
            if selected.guaranteed_safe
            else "forced"
        ),
    )


def direction_toward(start, target):
    """Return the cardinal direction from start toward an aligned target."""
    row_delta = target[0] - start[0]
    column_delta = target[1] - start[1]
    if row_delta and column_delta:
        return None
    if row_delta < 0:
        return Move.UP
    if row_delta > 0:
        return Move.DOWN
    if column_delta < 0:
        return Move.LEFT
    if column_delta > 0:
        return Move.RIGHT
    return None


def opposite_move(move):
    opposites = {
        Move.UP: Move.DOWN,
        Move.DOWN: Move.UP,
        Move.LEFT: Move.RIGHT,
        Move.RIGHT: Move.LEFT,
    }
    return opposites[move]


def perpendicular_moves(move):
    if move in (Move.UP, Move.DOWN):
        return Move.LEFT, Move.RIGHT
    return Move.UP, Move.DOWN


def _evaluate_branch(
    map_state,
    junction,
    move,
    possible_pacman_endpoints,
    escape_endpoints,
    capture_distance,
):
    endpoint = apply_move(junction, move)
    distances = [
        manhattan_distance(pacman_endpoint, endpoint)
        for pacman_endpoint in possible_pacman_endpoints
    ]
    continuation = _continuation_profile(
        map_state,
        junction,
        endpoint,
        escape_endpoints,
    )
    return EscapeBranch(
        move=move,
        endpoint=endpoint,
        guaranteed_safe=not any(
            is_capture(
                pacman_endpoint,
                endpoint,
                capture_distance,
            )
            for pacman_endpoint in possible_pacman_endpoints
        ),
        worst_case_distance=min(distances),
        **continuation,
    )


def _continuation_profile(
    map_state,
    junction,
    start,
    escape_endpoints,
):
    distances = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for move in CARDINAL_MOVES:
            neighbor = apply_move(current, move)
            if neighbor == junction or neighbor in distances:
                continue
            if not is_structurally_traversable(map_state, neighbor):
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    other_escape_endpoints = {
        endpoint for endpoint in escape_endpoints if endpoint != start
    }
    reconnects = any(
        endpoint in distances for endpoint in other_escape_endpoints
    )
    junction_distances = [
        distance
        for position, distance in distances.items()
        if _structural_degree(map_state, position) >= 3
    ]
    next_junction_distance = (
        min(junction_distances) if junction_distances else None
    )
    trapped = not reconnects and next_junction_distance is None
    return {
        "trapped": trapped,
        "reconnects": reconnects,
        "next_junction_distance": next_junction_distance,
        "escape_depth": max(distances.values()),
        "region_size": len(distances),
    }


def _structural_degree(map_state, position):
    return sum(
        is_structurally_traversable(
            map_state,
            apply_move(position, move),
        )
        for move in CARDINAL_MOVES
    )
