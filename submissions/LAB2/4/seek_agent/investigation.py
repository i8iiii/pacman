"""Belief-guided local investigation after the Ghost leaves sight."""

from dataclasses import dataclass
from enum import Enum

from environment import Move

from .spatial import (
    CARDINAL_MOVES,
    apply_move,
    is_traversable,
    normalize_position,
    path_to_actions,
    shortest_path,
    visibility_footprint,
)


INVESTIGATION_TURN_LIMIT = 7


class InvestigationPhase(str, Enum):
    TRANSIT = "transit"
    CANDIDATE_SCAN = "candidate_scan"
    EXPIRED = "expired"
    BELIEF_EMPTY = "belief_empty"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class InvestigationActionScore:
    """Explain how much one legal action can confirm."""

    move: Move
    steps: int
    endpoint: tuple
    confirmable_positions: frozenset
    nearest_candidate_distance: int | None


@dataclass(frozen=True)
class InvestigationDecision:
    """Immutable result of one INVESTIGATING decision."""

    phase: InvestigationPhase
    last_seen_position: tuple
    arrival_step: int | None
    investigation_turn: int | None
    turn_limit: int
    possible_positions: frozenset
    considered_actions: tuple
    chosen_action: tuple
    chosen_endpoint: tuple
    target: tuple | None
    route: tuple
    finished_reason: str | None


class InvestigationPlanner:
    """Pursue current Ghost candidates for a short post-sighting window."""

    def __init__(self, pacman_speed=2, turn_limit=INVESTIGATION_TURN_LIMIT):
        self.pacman_speed = max(1, int(pacman_speed))
        self.turn_limit = max(1, int(turn_limit))
        self.reset()

    def reset(self):
        self._episode_last_seen = None
        self._arrival_step = None

    def decide(
        self,
        topology,
        position,
        last_seen_position,
        possible_positions,
        step_number,
    ):
        position = normalize_position(position)
        last_seen_position = normalize_position(last_seen_position)
        possible = frozenset(
            normalize_position(cell) for cell in possible_positions
        )
        step_number = int(step_number)

        if self._episode_last_seen != last_seen_position:
            self._episode_last_seen = last_seen_position
            self._arrival_step = None

        if not possible:
            return self._finished_decision(
                InvestigationPhase.BELIEF_EMPTY,
                last_seen_position,
                position,
                possible,
                "belief_empty",
            )

        if self._arrival_step is None:
            if position != last_seen_position:
                route = shortest_path(
                    topology,
                    position,
                    last_seen_position,
                )
                if route is None:
                    return self._finished_decision(
                        InvestigationPhase.UNREACHABLE,
                        last_seen_position,
                        position,
                        possible,
                        "last_seen_unreachable",
                    )
                actions = path_to_actions(route, self.pacman_speed)
                chosen = actions[0] if actions else (Move.STAY, 1)
                endpoint = _action_endpoint(position, chosen)
                return InvestigationDecision(
                    phase=InvestigationPhase.TRANSIT,
                    last_seen_position=last_seen_position,
                    arrival_step=None,
                    investigation_turn=None,
                    turn_limit=self.turn_limit,
                    possible_positions=possible,
                    considered_actions=(),
                    chosen_action=chosen,
                    chosen_endpoint=endpoint,
                    target=last_seen_position,
                    route=tuple(route),
                    finished_reason=None,
                )
            self._arrival_step = step_number
        turns_already_used = max(0, step_number - self._arrival_step)
        if turns_already_used >= self.turn_limit:
            return self._finished_decision(
                InvestigationPhase.EXPIRED,
                last_seen_position,
                position,
                possible,
                "turn_limit_reached",
            )

        investigation_turn = turns_already_used + 1
        if len(possible) == 1:
            target = next(iter(possible))
            route = shortest_path(
                topology,
                position,
                target,
            )
            if route is None:
                return self._finished_decision(
                    InvestigationPhase.UNREACHABLE,
                    last_seen_position,
                    position,
                    possible,
                    "candidate_unreachable",
                )
            actions = path_to_actions(route, self.pacman_speed)
            chosen = actions[0] if actions else (Move.STAY, 1)
            endpoint = _action_endpoint(position, chosen)
            score = self._score_action(
                topology,
                endpoint,
                chosen,
                possible,
            )
            return InvestigationDecision(
                phase=InvestigationPhase.CANDIDATE_SCAN,
                last_seen_position=last_seen_position,
                arrival_step=self._arrival_step,
                investigation_turn=investigation_turn,
                turn_limit=self.turn_limit,
                possible_positions=possible,
                considered_actions=(score,),
                chosen_action=chosen,
                chosen_endpoint=endpoint,
                target=target,
                route=tuple(route),
                finished_reason=None,
            )

        scored = self._scored_legal_actions(topology, position, possible)
        if not scored:
            chosen = (Move.STAY, 1)
            endpoint = position
            scored = (
                self._score_action(
                    topology,
                    endpoint,
                    chosen,
                    possible,
                ),
            )
        if all(
            score.nearest_candidate_distance is None
            for score in scored
        ):
            return self._finished_decision(
                InvestigationPhase.UNREACHABLE,
                last_seen_position,
                position,
                possible,
                "candidate_unreachable",
            )
        chosen_score = min(
            scored,
            key=lambda score: (
                -len(score.confirmable_positions),
                _distance_sort_value(score.nearest_candidate_distance),
                -score.steps,
                _move_order(score.move),
            ),
        )
        chosen = chosen_score.move, chosen_score.steps
        route = _action_route(position, chosen)
        target = self._nearest_candidate(
            topology,
            chosen_score.endpoint,
            possible,
        )
        return InvestigationDecision(
            phase=InvestigationPhase.CANDIDATE_SCAN,
            last_seen_position=last_seen_position,
            arrival_step=self._arrival_step,
            investigation_turn=investigation_turn,
            turn_limit=self.turn_limit,
            possible_positions=possible,
            considered_actions=tuple(scored),
            chosen_action=chosen,
            chosen_endpoint=chosen_score.endpoint,
            target=target,
            route=route,
            finished_reason=None,
        )

    def _scored_legal_actions(self, topology, position, possible):
        scored = []
        for move in CARDINAL_MOVES:
            endpoint = position
            for steps in range(1, self.pacman_speed + 1):
                endpoint = apply_move(endpoint, move)
                if not is_traversable(topology, endpoint):
                    break
                scored.append(self._score_action(
                    topology,
                    endpoint,
                    (move, steps),
                    possible,
                ))
        return tuple(scored)

    def _score_action(self, topology, endpoint, action, possible):
        visible = visibility_footprint(topology, endpoint, radius=5)
        confirmable = frozenset(possible & visible)
        distances = []
        for candidate in possible:
            path = shortest_path(topology, endpoint, candidate)
            if path is not None:
                distances.append(len(path) - 1)
        nearest_distance = min(distances) if distances else None
        return InvestigationActionScore(
            move=action[0],
            steps=int(action[1]),
            endpoint=endpoint,
            confirmable_positions=confirmable,
            nearest_candidate_distance=nearest_distance,
        )

    @staticmethod
    def _nearest_candidate(topology, endpoint, possible):
        candidates = []
        for candidate in possible:
            path = shortest_path(topology, endpoint, candidate)
            if path is not None:
                candidates.append((len(path) - 1, candidate))
        return min(candidates)[1] if candidates else None

    def _finished_decision(
        self,
        phase,
        last_seen_position,
        position,
        possible,
        reason,
    ):
        return InvestigationDecision(
            phase=phase,
            last_seen_position=last_seen_position,
            arrival_step=self._arrival_step,
            investigation_turn=None,
            turn_limit=self.turn_limit,
            possible_positions=possible,
            considered_actions=(),
            chosen_action=(Move.STAY, 1),
            chosen_endpoint=position,
            target=None,
            route=(position,),
            finished_reason=reason,
        )


def _action_endpoint(position, action):
    endpoint = position
    for _ in range(int(action[1])):
        endpoint = apply_move(endpoint, action[0])
    return endpoint


def _action_route(position, action):
    cells = [position]
    current = position
    for _ in range(int(action[1])):
        current = apply_move(current, action[0])
        cells.append(current)
    return tuple(cells)


def _move_order(move):
    if move == Move.STAY:
        return len(CARDINAL_MOVES)
    return CARDINAL_MOVES.index(move)


def _distance_sort_value(distance):
    return float("inf") if distance is None else int(distance)
