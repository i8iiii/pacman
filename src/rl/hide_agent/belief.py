"""Match-local broad Pacman belief and interception calculations."""

from collections import deque
from dataclasses import dataclass
from random import choice

from environment import Move

from rl.hide_agent.spatial import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    is_capture,
    is_structurally_traversable,
    manhattan_distance,
    pacman_endpoints,
)
from rl.hide_agent.spatial import reconstruct_path, structural_shortest_paths


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


"""Match-local HOT_UNSEEN follower tracking and escape selection."""

from collections import deque
from dataclasses import asdict, dataclass
from random import choice

from environment import Move

from rl.hide_agent.spatial import (
    CARDINAL_MOVES,
    apply_move,
    has_line_of_sight,
    is_capture,
    is_structurally_traversable,
    manhattan_distance,
)


@dataclass(frozen=True)
class PursuitSeed:
    step_number: int
    pacman_position: tuple
    ghost_position: tuple
    escape_move: Move
    expected_ghost_position: tuple
    trail: tuple


@dataclass(frozen=True)
class FollowerPrediction:
    trail_index: int
    position: tuple

    def to_log_record(self):
        return _prediction_log_record(self)


@dataclass(frozen=True)
class HotPursuitState:
    seed_step_number: int
    trail: tuple
    follower_indices: tuple


@dataclass(frozen=True)
class PendingHotTransition:
    expected_ghost_position: tuple
    trail: tuple
    follower_indices: tuple
    removals: dict


@dataclass(frozen=True)
class PursuitUpdate:
    status: str
    entered: bool
    seed: PursuitSeed | None
    trail: tuple
    previous_followers: tuple
    followers: tuple
    removals: dict
    reason: str | None

    def removals_log_record(self):
        return _removals_log_record(self.removals)


@dataclass(frozen=True)
class HotMoveTarget:
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
class HotMoveCandidate:
    move: Move
    endpoint: tuple
    guaranteed_safe: bool
    worst_case_distance: int
    hidden_follower_count: int
    trapped: bool
    reverses: bool
    target: HotMoveTarget | None
    continuation_depth: int
    region_size: int
    trail: tuple
    next_followers: tuple
    transition_removals: dict

    @property
    def rank(self):
        target_exists = self.target is not None
        target_distance_rank = (
            -self.target.distance if target_exists else 0
        )
        return (
            int(self.guaranteed_safe),
            self.worst_case_distance,
            self.hidden_follower_count,
            int(not self.trapped),
            int(not self.reverses),
            int(self.move is not Move.STAY),
            int(target_exists),
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
        record["trail"] = [list(position) for position in self.trail]
        record["next_followers"] = [
            _prediction_log_record(follower)
            for follower in self.next_followers
        ]
        record["transition_removals"] = _removals_log_record(
            self.transition_removals
        )
        record["rank"] = list(self.rank)
        return record


@dataclass(frozen=True)
class HotMoveDecision:
    candidates: tuple
    selected: HotMoveCandidate
    equivalent_moves: tuple
    mode: str


class PursuitTracker:
    """Own the current match's visible seed and unseen follower model."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.seed = None
        self.active = None
        self.pending = None

    def record_visible_escape(
        self,
        map_state,
        ghost_position,
        pacman_position,
        escape_move,
        step_number,
    ):
        """Replace pursuit state with a validated visible-escape seed."""
        self.reset()
        ghost_position = tuple(ghost_position)
        pacman_position = tuple(pacman_position)
        if escape_move not in (*CARDINAL_MOVES, Move.STAY):
            return None

        visible_line = _cardinal_segment(
            pacman_position,
            ghost_position,
        )
        if not visible_line or not all(
            is_structurally_traversable(map_state, position)
            for position in visible_line
        ):
            return None

        expected_ghost_position = apply_move(
            ghost_position,
            escape_move,
        )
        if not is_structurally_traversable(
            map_state,
            expected_ghost_position,
        ):
            return None

        trail = list(visible_line)
        if expected_ghost_position != trail[-1]:
            trail.append(expected_ghost_position)

        self.seed = PursuitSeed(
            step_number=int(step_number),
            pacman_position=pacman_position,
            ghost_position=ghost_position,
            escape_move=escape_move,
            expected_ghost_position=expected_ghost_position,
            trail=tuple(trail),
        )
        return self.seed

    def observe_unseen(
        self,
        map_state,
        ghost_position,
        observation_radius,
        capture_distance,
    ):
        """Commit the last transition, then apply current absence evidence."""
        ghost_position = tuple(ghost_position)
        entered = self.seed is not None and self.pending is None
        entry_seed = self.seed if entered else None

        if entered:
            expected_position = self.seed.expected_ghost_position
            trail = self.seed.trail
            previous_indices = (0,)
            follower_indices, transition_removals = (
                _advance_follower_indices(
                    trail,
                    previous_indices,
                    allow_stay=True,
                )
            )
            seed_step_number = self.seed.step_number
        elif self.pending is not None and self.active is not None:
            expected_position = self.pending.expected_ghost_position
            trail = self.pending.trail
            previous_indices = self.active.follower_indices
            follower_indices = self.pending.follower_indices
            transition_removals = self.pending.removals
            seed_step_number = self.active.seed_step_number
        else:
            return PursuitUpdate(
                status="inactive",
                entered=False,
                seed=None,
                trail=(),
                previous_followers=(),
                followers=(),
                removals=_empty_removals(),
                reason="no_pursuit_seed",
            )

        previous_followers = _predictions(trail, previous_indices)
        if ghost_position != expected_position:
            self.reset()
            return PursuitUpdate(
                status="invalidated",
                entered=entered,
                seed=entry_seed,
                trail=trail,
                previous_followers=previous_followers,
                followers=(),
                removals=transition_removals,
                reason="endpoint_mismatch",
            )

        followers, evidence_removals = _filter_follower_indices(
            map_state,
            trail,
            follower_indices,
            ghost_position,
            observation_radius,
            capture_distance,
        )
        removals = _merge_removals(
            transition_removals,
            evidence_removals,
        )
        self.seed = None
        self.pending = None
        if not followers:
            self.active = None
            return PursuitUpdate(
                status="invalidated",
                entered=entered,
                seed=entry_seed,
                trail=trail,
                previous_followers=previous_followers,
                followers=(),
                removals=removals,
                reason="empty_follower_set",
            )

        self.active = HotPursuitState(
            seed_step_number=seed_step_number,
            trail=trail,
            follower_indices=tuple(
                follower.trail_index for follower in followers
            ),
        )
        return PursuitUpdate(
            status="entered" if entered else "updated",
            entered=entered,
            seed=entry_seed,
            trail=trail,
            previous_followers=previous_followers,
            followers=followers,
            removals=removals,
            reason=None,
        )

    def invalidate(self, reason):
        """Discard pursuit state and return the discarded follower evidence."""
        if self.pending is not None:
            trail = self.pending.trail
            followers = _predictions(
                trail,
                self.pending.follower_indices,
            )
        elif self.active is not None:
            trail = self.active.trail
            followers = _predictions(
                trail,
                self.active.follower_indices,
            )
        elif self.seed is not None:
            trail = self.seed.trail
            followers = _predictions(trail, (0,))
        else:
            trail = ()
            followers = ()
        seed = self.seed
        self.reset()
        return PursuitUpdate(
            status="invalidated",
            entered=False,
            seed=seed,
            trail=trail,
            previous_followers=followers,
            followers=(),
            removals=_empty_removals(),
            reason=reason,
        )

    def choose_hot_move(
        self,
        map_state,
        ghost_position,
        safe_campsites,
        observation_radius,
        capture_distance,
    ):
        """Choose and stage one P08-only HOT_UNSEEN action."""
        candidates = self.project_hot_moves(
            map_state,
            ghost_position,
            safe_campsites,
            observation_radius,
            capture_distance,
        )
        if not candidates:
            return None

        best_rank = max(candidate.rank for candidate in candidates)
        equivalent = tuple(
            candidate
            for candidate in candidates
            if candidate.rank == best_rank
        )
        selected = choice(equivalent)
        self.stage_hot_candidate(selected)
        return HotMoveDecision(
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

    def project_hot_moves(
        self,
        map_state,
        ghost_position,
        safe_campsites,
        observation_radius,
        capture_distance,
    ):
        """Return P08 likely-follower projections without selecting a move."""
        if self.active is None:
            return ()

        ghost_position = tuple(ghost_position)
        if self.active.trail[-1] != ghost_position:
            return ()

        legal_moves = tuple(
            move
            for move in (*CARDINAL_MOVES, Move.STAY)
            if is_structurally_traversable(
                map_state,
                apply_move(ghost_position, move),
            )
        )
        candidates = tuple(
            _evaluate_hot_candidate(
                map_state,
                self.active,
                ghost_position,
                move,
                safe_campsites,
                observation_radius,
                capture_distance,
            )
            for move in legal_moves
        )
        return candidates

    def stage_hot_candidate(self, selected):
        """Stage exactly the P08 projection selected by the broad model."""
        self.pending = PendingHotTransition(
            expected_ghost_position=selected.endpoint,
            trail=selected.trail,
            follower_indices=tuple(
                follower.trail_index
                for follower in selected.next_followers
            ),
            removals=selected.transition_removals,
        )
        return self.pending


def _cardinal_segment(start, end):
    start = tuple(start)
    end = tuple(end)
    row_delta = end[0] - start[0]
    column_delta = end[1] - start[1]
    if row_delta and column_delta:
        return ()

    if row_delta:
        step = (1 if row_delta > 0 else -1, 0)
        length = abs(row_delta)
    elif column_delta:
        step = (0, 1 if column_delta > 0 else -1)
        length = abs(column_delta)
    else:
        return (start,)

    return tuple(
        (
            start[0] + step[0] * distance,
            start[1] + step[1] * distance,
        )
        for distance in range(length + 1)
    )


def _advance_follower_indices(
    trail,
    follower_indices,
    allow_stay,
):
    """Advance along ordered indices without permitting a speed-two turn."""
    resulting_indices = set()
    removed = _empty_removal_sets()
    advances = (0, 1, 2) if allow_stay else (1, 2)

    for follower_index in follower_indices:
        if not allow_stay:
            removed["stationary_after_entry"].add(follower_index)
        for distance in advances:
            target_index = follower_index + distance
            if target_index >= len(trail):
                removed["illegal_advance"].add(follower_index)
                continue
            if (
                distance == 2
                and _step_direction(
                    trail[follower_index],
                    trail[follower_index + 1],
                )
                != _step_direction(
                    trail[follower_index + 1],
                    trail[target_index],
                )
            ):
                removed["illegal_advance"].add(follower_index)
                continue
            resulting_indices.add(target_index)

    return (
        tuple(sorted(resulting_indices)),
        _removal_predictions(trail, removed),
    )


def _evaluate_hot_candidate(
    map_state,
    active,
    ghost_position,
    move,
    safe_campsites,
    observation_radius,
    capture_distance,
):
    endpoint = apply_move(ghost_position, move)
    candidate_trail = list(active.trail)
    if endpoint != candidate_trail[-1]:
        candidate_trail.append(endpoint)
    candidate_trail = tuple(candidate_trail)

    next_indices, transition_removals = _advance_follower_indices(
        candidate_trail,
        active.follower_indices,
        allow_stay=False,
    )
    next_followers = _predictions(candidate_trail, next_indices)
    distances = [
        manhattan_distance(follower.position, endpoint)
        for follower in next_followers
    ]
    continuation_distances, trapped = _continuation_distances(
        map_state,
        ghost_position,
        endpoint,
    )
    target = _nearest_safe_campsite(
        continuation_distances,
        safe_campsites,
    )
    previous_trail_position = (
        active.trail[-2] if len(active.trail) >= 2 else None
    )
    return HotMoveCandidate(
        move=move,
        endpoint=endpoint,
        guaranteed_safe=not any(
            is_capture(
                follower.position,
                endpoint,
                capture_distance,
            )
            for follower in next_followers
        ),
        worst_case_distance=(
            min(distances)
            if distances
            else map_state.shape[0] + map_state.shape[1]
        ),
        hidden_follower_count=sum(
            not has_line_of_sight(
                map_state,
                endpoint,
                follower.position,
                observation_radius,
            )
            for follower in next_followers
        ),
        trapped=trapped,
        reverses=(
            move is not Move.STAY
            and endpoint == previous_trail_position
        ),
        target=target,
        continuation_depth=max(continuation_distances.values()),
        region_size=len(continuation_distances),
        trail=candidate_trail,
        next_followers=next_followers,
        transition_removals=transition_removals,
    )


def _continuation_distances(map_state, origin, start):
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


def _nearest_safe_campsite(distances, safe_campsites):
    positions = [
        tuple(campsite.position)
        for campsite in safe_campsites
        if tuple(campsite.position) in distances
    ]
    if not positions:
        return None

    position = min(
        positions,
        key=lambda candidate: (
            distances[candidate],
            candidate[0],
            candidate[1],
        ),
    )
    return HotMoveTarget(
        kind="strategic_hideout",
        position=position,
        distance=distances[position],
    )


def _structural_degree(map_state, position):
    return sum(
        is_structurally_traversable(
            map_state,
            apply_move(position, move),
        )
        for move in CARDINAL_MOVES
    )


def _filter_follower_indices(
    map_state,
    trail,
    follower_indices,
    ghost_position,
    observation_radius,
    capture_distance,
):
    kept = []
    removed = _empty_removal_sets()

    for follower_index in follower_indices:
        position = trail[follower_index]
        if not is_structurally_traversable(map_state, position):
            removed["structural_contradiction"].add(follower_index)
        elif is_capture(
            position,
            ghost_position,
            capture_distance,
        ):
            removed["capture_contradiction"].add(follower_index)
        elif has_line_of_sight(
            map_state,
            ghost_position,
            position,
            observation_radius,
        ):
            removed["visibility_contradiction"].add(follower_index)
        else:
            kept.append(FollowerPrediction(follower_index, position))

    return tuple(kept), _removal_predictions(trail, removed)


def _step_direction(start, end):
    return (
        end[0] - start[0],
        end[1] - start[1],
    )


def _predictions(trail, indices):
    return tuple(
        FollowerPrediction(index, trail[index]) for index in indices
    )


def _empty_removal_sets():
    return {
        "visibility_contradiction": set(),
        "capture_contradiction": set(),
        "structural_contradiction": set(),
        "illegal_advance": set(),
        "stationary_after_entry": set(),
    }


def _empty_removals():
    return {
        reason: () for reason in _empty_removal_sets()
    }


def _removal_predictions(trail, removed_indices):
    return {
        reason: _predictions(trail, sorted(indices))
        for reason, indices in removed_indices.items()
    }


def _merge_removals(*removal_groups):
    merged = _empty_removal_sets()
    prediction_by_index = {}
    for removals in removal_groups:
        for reason, predictions in removals.items():
            for prediction in predictions:
                merged[reason].add(prediction.trail_index)
                prediction_by_index[prediction.trail_index] = prediction

    return {
        reason: tuple(
            prediction_by_index[index]
            for index in sorted(indices)
        )
        for reason, indices in merged.items()
    }


def _prediction_log_record(prediction):
    return {
        "trail_index": prediction.trail_index,
        "position": list(prediction.position),
    }


def _removals_log_record(removals):
    return {
        reason: [
            _prediction_log_record(prediction)
            for prediction in predictions
        ]
        for reason, predictions in removals.items()
    }
