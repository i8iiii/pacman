"""Match-local broad Pacman belief and interception calculations."""

from collections import deque
from dataclasses import dataclass
from random import choice

from environment import Move

from .geometry import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    is_capture,
    is_structurally_traversable,
    manhattan_distance,
    pacman_endpoints,
)
from .navigation import reconstruct_path, structural_shortest_paths


@dataclass(frozen=True)
class BeliefUpdate:
    """Result of applying one unseen observation to the broad belief."""

    status: str
    previous_positions: tuple
    positions: tuple
    removed_visibility: tuple
    removed_capture: tuple
    elapsed_unseen: int
    rebuilt: bool
    rebuild_reason: str | None


@dataclass(frozen=True)
class InterceptionAssessment:
    """Pacman's earliest threat against one proposed strategic route."""

    target: tuple
    route: tuple
    ghost_arrival: int
    pacman_threat_arrival: int | None
    first_contested_junction: tuple | None
    junction_ghost_arrival: int | None
    junction_pacman_arrival: int | None
    contested: bool
    reason: str | None

    def to_log_record(self):
        return {
            "target": list(self.target),
            "route": [list(position) for position in self.route],
            "ghost_arrival": self.ghost_arrival,
            "pacman_threat_arrival": self.pacman_threat_arrival,
            "first_contested_junction": (
                None
                if self.first_contested_junction is None
                else list(self.first_contested_junction)
            ),
            "junction_ghost_arrival": self.junction_ghost_arrival,
            "junction_pacman_arrival": self.junction_pacman_arrival,
            "contested": self.contested,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InterceptionPlan:
    """All reachable hideout assessments and the best uncontested one."""

    assessments: tuple
    selected: InterceptionAssessment | None


@dataclass(frozen=True)
class BeliefMoveCandidate:
    """One Hide action ranked first by the broad Pacman belief."""

    move: Move
    endpoint: tuple
    guaranteed_safe: bool
    capturing_endpoint_count: int
    worst_case_distance: int
    hidden_belief_count: int
    interception: InterceptionPlan
    likely_projection: object | None
    trapped: bool
    reverses: bool
    continuation_depth: int
    region_size: int
    next_belief: tuple

    @property
    def target(self):
        return self.interception.selected

    @property
    def rank(self):
        likely = self.likely_projection
        target_exists = self.target is not None
        return (
            int(self.guaranteed_safe),
            -self.capturing_endpoint_count,
            int(target_exists),
            self.worst_case_distance,
            self.hidden_belief_count,
            int(
                likely is not None
                and likely.guaranteed_safe
            ),
            (
                likely.worst_case_distance
                if likely is not None
                else 0
            ),
            (
                likely.hidden_follower_count
                if likely is not None
                else 0
            ),
            int(not self.trapped),
            int(not self.reverses),
            int(self.move is not Move.STAY),
            (
                -self.target.ghost_arrival
                if target_exists
                else 0
            ),
            self.continuation_depth,
        )

    def to_log_record(self):
        return {
            "move": self.move.name,
            "endpoint": list(self.endpoint),
            "guaranteed_safe": self.guaranteed_safe,
            "capturing_endpoint_count": self.capturing_endpoint_count,
            "worst_case_distance": self.worst_case_distance,
            "hidden_belief_count": self.hidden_belief_count,
            "target": (
                None
                if self.target is None
                else self.target.to_log_record()
            ),
            "trapped": self.trapped,
            "reverses": self.reverses,
            "continuation_depth": self.continuation_depth,
            "region_size": self.region_size,
            "next_belief_size": len(self.next_belief),
            "rank": list(self.rank),
        }


@dataclass(frozen=True)
class BeliefMoveDecision:
    candidates: tuple
    selected: BeliefMoveCandidate
    equivalent_moves: tuple
    mode: str


class PacmanBeliefTracker:
    """Track every Pacman cell consistent with this match's observations."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_visible_position = None
        self.last_visible_step = None
        self.positions = ()
        self.elapsed_unseen = 0
        self.absence_history = ()

    @property
    def active(self):
        return self.last_visible_position is not None and bool(self.positions)

    def record_visible(self, pacman_position, step_number):
        """Replace all old belief state with one exact sighting."""
        position = tuple(int(value) for value in pacman_position)
        self.last_visible_position = position
        self.last_visible_step = int(step_number)
        self.positions = (position,)
        self.elapsed_unseen = 0
        self.absence_history = ()
        return self.positions

    def predict_next(self, map_state, pacman_speed=2):
        """Return every structural endpoint reachable in one Pacman turn."""
        if not self.active:
            return ()
        return _expand_positions(
            map_state,
            self.positions,
            pacman_speed,
        )

    def observe_unseen(
        self,
        map_state,
        ghost_position,
        observation_radius,
        capture_distance,
        pacman_speed=2,
    ):
        """Expand one turn and apply absence plus continued-survival evidence."""
        if not self.active:
            return BeliefUpdate(
                status="inactive",
                previous_positions=(),
                positions=(),
                removed_visibility=(),
                removed_capture=(),
                elapsed_unseen=self.elapsed_unseen,
                rebuilt=False,
                rebuild_reason=None,
            )

        ghost_position = tuple(int(value) for value in ghost_position)
        previous_positions = self.positions
        expanded = _expand_positions(
            map_state,
            previous_positions,
            pacman_speed,
        )
        filtered, removed_visibility, removed_capture = _filter_evidence(
            map_state,
            expanded,
            ghost_position,
            observation_radius,
            capture_distance,
        )

        self.elapsed_unseen += 1
        self.absence_history = (*self.absence_history, ghost_position)
        rebuilt = self.elapsed_unseen == 1
        rebuild_reason = (
            "initial_from_last_sighting" if rebuilt else None
        )

        if not filtered:
            filtered = self._replay_filtered(
                map_state,
                observation_radius,
                capture_distance,
                pacman_speed,
            )
            rebuilt = True
            rebuild_reason = "evidence_replay"

        if not filtered:
            filtered = self._replay_structural(
                map_state,
                pacman_speed,
            )
            rebuilt = True
            rebuild_reason = "structural_fallback"

        self.positions = tuple(sorted(set(filtered)))
        return BeliefUpdate(
            status="updated",
            previous_positions=previous_positions,
            positions=self.positions,
            removed_visibility=removed_visibility,
            removed_capture=removed_capture,
            elapsed_unseen=self.elapsed_unseen,
            rebuilt=rebuilt,
            rebuild_reason=rebuild_reason,
        )

    def _replay_filtered(
        self,
        map_state,
        observation_radius,
        capture_distance,
        pacman_speed,
    ):
        positions = (self.last_visible_position,)
        for ghost_position in self.absence_history:
            positions = _expand_positions(
                map_state,
                positions,
                pacman_speed,
            )
            positions, _, _ = _filter_evidence(
                map_state,
                positions,
                ghost_position,
                observation_radius,
                capture_distance,
            )
            if not positions:
                return ()
        return positions

    def _replay_structural(self, map_state, pacman_speed):
        positions = (self.last_visible_position,)
        for _ in self.absence_history:
            positions = _expand_positions(
                map_state,
                positions,
                pacman_speed,
            )
        if positions:
            return positions
        return (self.last_visible_position,)


def _expand_positions(map_state, positions, pacman_speed):
    expanded = set()
    for position in positions:
        for endpoint in pacman_endpoints(
            map_state,
            position,
            speed=pacman_speed,
        ):
            if is_structurally_traversable(map_state, endpoint):
                expanded.add(tuple(endpoint))
    return tuple(sorted(expanded))


def _filter_evidence(
    map_state,
    positions,
    ghost_position,
    observation_radius,
    capture_distance,
):
    kept = []
    removed_visibility = []
    removed_capture = []
    for position in positions:
        if is_capture(position, ghost_position, capture_distance):
            removed_capture.append(position)
        elif has_line_of_sight(
            map_state,
            ghost_position,
            position,
            observation_radius,
        ):
            removed_visibility.append(position)
        else:
            kept.append(position)
    return (
        tuple(sorted(kept)),
        tuple(sorted(removed_visibility)),
        tuple(sorted(removed_capture)),
    )


def pacman_threat_time(
    map_state,
    belief_positions,
    target,
    capture_distance,
    pacman_speed=2,
):
    """Return exact Pacman turns needed to enter target capture range."""
    distances = pacman_turn_distances(
        map_state,
        belief_positions,
        pacman_speed,
    )
    return _threat_time_from_distances(
        map_state,
        distances,
        target,
        capture_distance,
    )


def pacman_turn_distances(
    map_state,
    belief_positions,
    pacman_speed=2,
):
    """Return exact multi-source Pacman action-graph distances."""
    starts = tuple(
        sorted(
            {
                tuple(position)
                for position in belief_positions
                if is_structurally_traversable(map_state, position)
            }
        )
    )
    if not starts:
        return {}

    distances = {position: 0 for position in starts}
    queue = deque(starts)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for endpoint in pacman_endpoints(
            map_state,
            current,
            speed=pacman_speed,
        ):
            endpoint = tuple(endpoint)
            if endpoint in distances:
                continue
            distances[endpoint] = next_distance
            queue.append(endpoint)
    return distances


def _threat_time_from_distances(
    map_state,
    distances,
    target,
    capture_distance,
):
    target = tuple(target)
    threats = [
        turns
        for position, turns in distances.items()
        if is_structurally_traversable(map_state, position)
        and is_capture(position, target, capture_distance)
    ]
    return min(threats) if threats else None


def assess_interceptions(
    map_state,
    ghost_position,
    first_move,
    belief_positions,
    safe_campsites,
    capture_distance,
    pacman_speed=2,
    pacman_distances=None,
):
    """Assess every reachable hideout after one proposed Hide action."""
    ghost_position = tuple(ghost_position)
    endpoint = apply_move(ghost_position, first_move)
    if not is_structurally_traversable(map_state, endpoint):
        return InterceptionPlan(assessments=(), selected=None)

    distances, parents = structural_shortest_paths(map_state, endpoint)
    if pacman_distances is None:
        pacman_distances = pacman_turn_distances(
            map_state,
            belief_positions,
            pacman_speed,
        )
    assessments = []
    for campsite in safe_campsites:
        target = tuple(campsite.position)
        if target not in distances:
            continue

        path_after_endpoint = tuple(reconstruct_path(parents, target))
        route = (endpoint, *path_after_endpoint)
        ghost_arrival = 1 + distances[target]
        target_threat = _threat_time_from_distances(
            map_state,
            pacman_distances,
            target,
            capture_distance,
        )

        first_contested_junction = None
        junction_ghost_arrival = None
        junction_pacman_arrival = None
        for arrival, position in enumerate(route[:-1], start=1):
            if _structural_degree(map_state, position) < 3:
                continue
            threat_arrival = _threat_time_from_distances(
                map_state,
                pacman_distances,
                position,
                capture_distance,
            )
            if threat_arrival is not None and threat_arrival <= arrival:
                first_contested_junction = position
                junction_ghost_arrival = arrival
                junction_pacman_arrival = threat_arrival
                break

        if first_contested_junction is not None:
            contested = True
            reason = "route_junction_contested"
        elif (
            target_threat is not None
            and target_threat <= ghost_arrival
        ):
            contested = True
            reason = "hideout_arrival_threat"
        elif (
            target_threat is not None
            and target_threat <= ghost_arrival + 1
        ):
            contested = True
            reason = "hideout_next_turn_threat"
        else:
            contested = False
            reason = None

        assessments.append(
            InterceptionAssessment(
                target=target,
                route=route,
                ghost_arrival=ghost_arrival,
                pacman_threat_arrival=target_threat,
                first_contested_junction=first_contested_junction,
                junction_ghost_arrival=junction_ghost_arrival,
                junction_pacman_arrival=junction_pacman_arrival,
                contested=contested,
                reason=reason,
            )
        )

    assessments = tuple(
        sorted(
            assessments,
            key=lambda assessment: (
                assessment.ghost_arrival,
                assessment.target,
            ),
        )
    )
    selected = next(
        (
            assessment
            for assessment in assessments
            if not assessment.contested
        ),
        None,
    )
    return InterceptionPlan(
        assessments=assessments,
        selected=selected,
    )


def _structural_degree(map_state, position):
    return sum(
        is_structurally_traversable(
            map_state,
            apply_move(position, move),
        )
        for move in CARDINAL_MOVES
    )


def choose_belief_hot_move(
    map_state,
    ghost_position,
    belief_positions,
    safe_campsites,
    observation_radius,
    capture_distance,
    pacman_speed=2,
    likely_candidates=(),
    previous_ghost_position=None,
):
    """Choose a HOT_UNSEEN action with broad safety before P08 likelihood."""
    ghost_position = tuple(ghost_position)
    belief_positions = tuple(
        sorted({tuple(position) for position in belief_positions})
    )
    next_belief = _expand_positions(
        map_state,
        belief_positions,
        pacman_speed,
    )
    pacman_distances = pacman_turn_distances(
        map_state,
        belief_positions,
        pacman_speed,
    )
    likely_by_move = {
        candidate.move: candidate for candidate in likely_candidates
    }
    previous_ghost_position = (
        None
        if previous_ghost_position is None
        else tuple(previous_ghost_position)
    )

    legal_moves = [
        move
        for move in CARDINAL_MOVES
        if is_structurally_traversable(
            map_state,
            apply_move(ghost_position, move),
        )
    ]
    legal_moves.append(Move.STAY)

    candidates = tuple(
        _evaluate_belief_move(
            map_state,
            ghost_position,
            move,
            belief_positions,
            next_belief,
            safe_campsites,
            observation_radius,
            capture_distance,
            pacman_speed,
            likely_by_move.get(move),
            previous_ghost_position,
            pacman_distances,
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
    return BeliefMoveDecision(
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


def _evaluate_belief_move(
    map_state,
    ghost_position,
    move,
    belief_positions,
    next_belief,
    safe_campsites,
    observation_radius,
    capture_distance,
    pacman_speed,
    likely_projection,
    previous_ghost_position,
    pacman_distances,
):
    endpoint = apply_move(ghost_position, move)
    capturing_positions = tuple(
        position
        for position in next_belief
        if is_capture(position, endpoint, capture_distance)
    )
    distances = tuple(
        manhattan_distance(position, endpoint)
        for position in next_belief
    )
    continuation_distances, trapped = _continuation_profile(
        map_state,
        ghost_position,
        endpoint,
    )
    interception = assess_interceptions(
        map_state,
        ghost_position,
        move,
        belief_positions,
        safe_campsites,
        capture_distance,
        pacman_speed,
        pacman_distances,
    )
    reverses = (
        move is not Move.STAY
        and previous_ghost_position is not None
        and endpoint == previous_ghost_position
    )
    if previous_ghost_position is None and likely_projection is not None:
        reverses = likely_projection.reverses

    return BeliefMoveCandidate(
        move=move,
        endpoint=endpoint,
        guaranteed_safe=not capturing_positions,
        capturing_endpoint_count=len(capturing_positions),
        worst_case_distance=(
            min(distances)
            if distances
            else map_state.shape[0] + map_state.shape[1]
        ),
        hidden_belief_count=sum(
            not has_line_of_sight(
                map_state,
                endpoint,
                position,
                observation_radius,
            )
            for position in next_belief
        ),
        interception=interception,
        likely_projection=likely_projection,
        trapped=trapped,
        reverses=reverses,
        continuation_depth=max(continuation_distances.values()),
        region_size=len(continuation_distances),
        next_belief=next_belief,
    )


def _continuation_profile(map_state, origin, start):
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

    reaches_junction = any(
        _structural_degree(map_state, position) >= 3
        for position in distances
    )
    reconnects = any(
        apply_move(origin, move) in distances
        for move in CARDINAL_MOVES
        if apply_move(origin, move) != start
    )
    return distances, not reaches_junction and not reconnects
