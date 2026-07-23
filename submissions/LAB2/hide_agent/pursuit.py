"""Match-local HOT_UNSEEN follower tracking and escape selection."""

from collections import deque
from dataclasses import asdict, dataclass
from random import choice

from environment import Move

from .geometry import (
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
