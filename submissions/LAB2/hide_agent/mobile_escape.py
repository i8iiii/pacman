"""Survival-first visible escape while Hide is away from a useful junction."""

from collections import deque
from dataclasses import asdict, dataclass
from random import choice

from environment import Move

from .escape import direction_toward, opposite_move
from .geometry import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    is_capture,
    is_structurally_traversable,
    manhattan_distance,
    pacman_endpoints,
)


@dataclass(frozen=True)
class MobileEscapeTarget:
    kind: str
    position: tuple
    distance: int

    def to_log_record(self):
        return {
            "kind": self.kind,
            "position": list(self.position),
            "distance": self.distance,
        }


@dataclass(frozen=True)
class MobileEscapeCandidate:
    move: Move
    endpoint: tuple
    guaranteed_safe: bool
    worst_case_distance: int
    hidden_endpoint_count: int
    trapped: bool
    target: MobileEscapeTarget | None
    continuation_depth: int
    region_size: int

    @property
    def immediate_rank(self):
        return (
            int(self.guaranteed_safe),
            self.worst_case_distance,
            self.hidden_endpoint_count,
            int(not self.trapped),
        )

    @property
    def rank(self):
        target_exists = self.target is not None
        target_is_safe_campsite = (
            target_exists and self.target.kind == "safe_campsite"
        )
        target_distance_rank = (
            -self.target.distance if target_exists else 0
        )
        return (
            *self.immediate_rank,
            int(self.move is not Move.STAY),
            int(target_exists),
            int(target_is_safe_campsite),
            target_distance_rank,
            self.continuation_depth,
        )

    def to_log_record(self):
        record = asdict(self)
        record["move"] = self.move.name
        record["endpoint"] = list(self.endpoint)
        record["target"] = (
            None if self.target is None else self.target.to_log_record()
        )
        record["immediate_rank"] = list(self.immediate_rank)
        record["rank"] = list(self.rank)
        return record


@dataclass(frozen=True)
class MobileEscapeDecision:
    approach_direction: Move | None
    pacman_endpoints: tuple
    candidates: tuple
    selected: MobileEscapeCandidate
    equivalent_moves: tuple
    mode: str


def choose_visible_mobile_escape(
    map_state,
    ghost_position,
    pacman_position,
    safe_campsites,
    pacman_speed=2,
    capture_distance=2,
    observation_radius=5,
):
    """Choose P07's survival-first action and dynamic escape destination."""
    ghost_position = tuple(ghost_position)
    pacman_position = tuple(pacman_position)
    approach = direction_toward(ghost_position, pacman_position)
    possible_pacman_endpoints = tuple(
        sorted(
            pacman_endpoints(
                map_state,
                pacman_position,
                pacman_speed,
            )
        )
    )
    legal_moves = tuple(
        move
        for move in (*CARDINAL_MOVES, Move.STAY)
        if is_structurally_traversable(
            map_state,
            apply_move(ghost_position, move),
        )
    )
    moving_endpoints = tuple(
        apply_move(ghost_position, move)
        for move in legal_moves
        if move is not Move.STAY
    )
    safe_positions = tuple(
        tuple(campsite.position) for campsite in safe_campsites
    )
    useful_t_positions = _useful_t_junctions(map_state, approach)

    candidates = tuple(
        _evaluate_candidate(
            map_state,
            ghost_position,
            move,
            moving_endpoints,
            possible_pacman_endpoints,
            safe_positions,
            useful_t_positions,
            capture_distance,
            observation_radius,
        )
        for move in legal_moves
    )
    best_rank = max(candidate.rank for candidate in candidates)
    equivalent = tuple(
        candidate
        for candidate in candidates
        if candidate.rank == best_rank
    )
    selected = choice(equivalent)
    return MobileEscapeDecision(
        approach_direction=approach,
        pacman_endpoints=possible_pacman_endpoints,
        candidates=candidates,
        selected=selected,
        equivalent_moves=tuple(
            candidate.move for candidate in equivalent
        ),
        mode=(
            "guaranteed"
            if selected.guaranteed_safe
            else "forced"
        ),
    )


def _evaluate_candidate(
    map_state,
    origin,
    move,
    moving_endpoints,
    possible_pacman_endpoints,
    safe_positions,
    useful_t_positions,
    capture_distance,
    observation_radius,
):
    endpoint = apply_move(origin, move)
    distances = [
        manhattan_distance(pacman_endpoint, endpoint)
        for pacman_endpoint in possible_pacman_endpoints
    ]
    continuation_distances, trapped = _continuation_distances(
        map_state,
        origin,
        endpoint,
        moving_endpoints,
    )
    target = _choose_target(
        continuation_distances,
        safe_positions,
        useful_t_positions,
    )
    return MobileEscapeCandidate(
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
        hidden_endpoint_count=sum(
            not has_line_of_sight(
                map_state,
                pacman_endpoint,
                endpoint,
                observation_radius,
            )
            for pacman_endpoint in possible_pacman_endpoints
        ),
        trapped=trapped,
        target=target,
        continuation_depth=max(continuation_distances.values()),
        region_size=len(continuation_distances),
    )


def _continuation_distances(
    map_state,
    origin,
    start,
    moving_endpoints,
):
    blocked = set() if start == origin else {origin}
    distances = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for move in CARDINAL_MOVES:
            neighbor = apply_move(current, move)
            if neighbor in blocked or neighbor in distances:
                continue
            if not is_structurally_traversable(map_state, neighbor):
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    reconnects = any(
        endpoint != start and endpoint in distances
        for endpoint in moving_endpoints
    )
    reaches_junction = any(
        _structural_degree(map_state, position) >= 3
        for position in distances
    )
    return distances, not reconnects and not reaches_junction


def _choose_target(
    distances,
    safe_positions,
    useful_t_positions,
):
    reachable_safe = [
        position for position in safe_positions if position in distances
    ]
    if reachable_safe:
        position = min(
            reachable_safe,
            key=lambda candidate: (
                distances[candidate],
                candidate[0],
                candidate[1],
            ),
        )
        return MobileEscapeTarget(
            kind="safe_campsite",
            position=position,
            distance=distances[position],
        )

    reachable_t = [
        position
        for position in useful_t_positions
        if position in distances
    ]
    if not reachable_t:
        return None

    position = min(
        reachable_t,
        key=lambda candidate: (
            distances[candidate],
            candidate[0],
            candidate[1],
        ),
    )
    return MobileEscapeTarget(
        kind="useful_t_junction",
        position=position,
        distance=distances[position],
    )


def _useful_t_junctions(map_state, approach):
    if approach is None:
        return ()

    required_missing_direction = opposite_move(approach)
    useful = []
    for row in range(map_state.shape[0]):
        for column in range(map_state.shape[1]):
            position = (row, column)
            if not is_structurally_traversable(map_state, position):
                continue
            exits = tuple(
                move
                for move in CARDINAL_MOVES
                if is_structurally_traversable(
                    map_state,
                    apply_move(position, move),
                )
            )
            if len(exits) != 3:
                continue
            missing_direction = next(
                move for move in CARDINAL_MOVES if move not in exits
            )
            if missing_direction == required_missing_direction:
                useful.append(position)
    return tuple(useful)


def _structural_degree(map_state, position):
    return sum(
        is_structurally_traversable(
            map_state,
            apply_move(position, move),
        )
        for move in CARDINAL_MOVES
    )
