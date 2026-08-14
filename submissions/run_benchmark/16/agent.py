"""
Hybrid Pacman and Ghost agents for the Hide and Seek Arena.

Main ideas
----------
1. Exact BFS preprocessing on the fixed maze:
   - real shortest-path distances;
   - Pacman's legal one-turn endpoints with straight-line speed;
   - minimum turns to a capture cell around every Ghost position.
2. Clear mode:
   - learn a conservative mixture of likely Ghost behaviours;
   - use a short risk-sensitive Expectiminimax search under simultaneous moves.
3. Blind mode:
   - maintain a Bayesian belief over all possible Ghost positions;
   - remove positions contradicted by the current observation;
   - balance capture probability, pursuit distance, and probability of
     seeing the Ghost again from the next position.
4. Turn-aware A* is retained as a safe fallback.
5. A tiny exact reachability game is solved once per maze.  In Clear mode its
   capture ranks are combined with Team 16's fast first-round chase and the
   learned policy in a conservative portfolio: tactical plateaus are allowed,
   while material losses of the exact safety guarantee are rejected.
6. Ghost combines the dual survival table with online Pacman modelling,
   cycle exploitation, and a separate belief-state policy for Blind mode.
7. A cached-BFS seeker hypothesis is verified from observed transitions; when
   credible, a memoized longest-survival search exploits path-cache lag.

Only Python standard-library modules and NumPy are used.
"""

from __future__ import annotations

import heapq
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Make the file work both inside the provided submissions directory and when
# it is opened directly by a student during local testing.
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


Position = Tuple[int, int]
PacmanAction = Tuple[Move, int]

CARDINAL_MOVES: Tuple[Move, ...] = (
    Move.UP,
    Move.DOWN,
    Move.LEFT,
    Move.RIGHT,
)
GHOST_MOVES: Tuple[Move, ...] = CARDINAL_MOVES + (Move.STAY,)

INF = np.int16(30000)


class PacmanAgent(BasePacmanAgent):
    """
    General-purpose seeker for both full and limited observation.

    The agent does not commit to one fixed Ghost policy. It starts with four
    broad behaviour hypotheses and updates their relative credibility from
    consecutive observations:

    - uniform/unpredictable movement;
    - maximizing maze distance from Pacman;
    - preserving the previous direction;
    - preferring mobile, non-dead-end hiding positions.

    A fixed uniform component is always retained. This prevents the prediction
    model from becoming overconfident after observing only a few turns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "Hybrid Robust Pacman Clear-Safe"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # The official limit is one second. The internal deadline leaves room
        # for framework validation and slower CPU-only tournament machines.
        self.internal_time_budget = 0.72
        self.deadline = 0.0

        # Map and graph data. They are initialized from the first observation.
        self.wall_map: Optional[np.ndarray] = None
        self.map_signature: Optional[bytes] = None
        self.height = 0
        self.width = 0
        self.cells: List[Position] = []
        self.index: Dict[Position, int] = {}
        self.neighbors: List[List[Tuple[Move, int]]] = []
        self.degrees: Optional[np.ndarray] = None
        self.graph_dist: Optional[np.ndarray] = None
        self.turn_dist: Optional[np.ndarray] = None
        self.capture_turns: Optional[np.ndarray] = None
        self.capture_rank: Optional[np.ndarray] = None
        self.pacman_actions: List[List[Tuple[Move, int, int]]] = []
        self.ghost_actions: List[List[Tuple[Move, int]]] = []

        # Partial-observation state.
        self.belief: Optional[np.ndarray] = None
        self.vision_radius_estimate = 5
        self.last_examined: Optional[np.ndarray] = None
        self.hidden_steps = 0
        self.current_step = 0
        self.coverage_target: Optional[int] = None

        # Behaviour learning state.
        # Order: uniform, flee, inertia, mobility. STAY remains present in
        # every policy's legal action set, so camping is still possible without
        # being exaggerated into a dominant assumption.
        self.behaviour_prior = np.array(
            [0.20, 0.35, 0.20, 0.25], dtype=float
        )
        self.behaviour_quality = np.ones(4, dtype=float)
        self.last_ghost_move: Optional[Move] = None
        self.last_observed_enemy: Optional[Position] = None
        self.last_observed_step: Optional[int] = None
        self.previous_my_position: Optional[Position] = None

        # Anti-loop memory. It is a soft penalty rather than a prohibition,
        # because some maze corridors genuinely require backtracking.
        self.position_history: deque[Position] = deque(maxlen=14)
        self.last_action: Optional[PacmanAction] = None

        # Clear-mode portfolio memory.  The first-round Team 16 policy is kept
        # as a fast empirical anchor; the exact-game/learned policy may
        # override it only when there is material evidence that doing so is
        # safer or faster.  These two deques reproduce the useful anti-loop
        # signal of that policy without constructing a second copy of the map.
        self.legacy_recent_positions: deque[Position] = deque(maxlen=8)
        self.legacy_recent_capture_turns: deque[int] = deque(maxlen=6)


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> PacmanAction:
        self.deadline = time.perf_counter() + self.internal_time_budget

        self._ensure_map(map_state)
        self._update_vision_radius(map_state, my_position, enemy_position)
        self._learn_from_visible_transition(
            my_position, enemy_position, step_number
        )
        self._update_belief(
            map_state, my_position, enemy_position, step_number
        )

        self.position_history.append(my_position)
        self.current_step = step_number

        if enemy_position is not None:
            self.hidden_steps = 0
            self.coverage_target = None
            action = self._choose_clear_action(my_position, enemy_position)
        else:
            self.hidden_steps += 1
            action = self._choose_blind_action(my_position, map_state)

        action = self._validate_or_fallback(action, my_position)

        if enemy_position is not None:
            endpoint = self._pacman_endpoint(my_position, action)
            self.legacy_recent_positions.append(endpoint)
            endpoint_index = self.index.get(endpoint)
            enemy_index = self.index.get(enemy_position)
            if endpoint_index is not None and enemy_index is not None:
                self.legacy_recent_capture_turns.append(
                    int(self.capture_turns[endpoint_index, enemy_index])
                )

        if enemy_position is not None:
            self.last_observed_enemy = enemy_position
            self.last_observed_step = step_number

        self.previous_my_position = my_position
        self.last_action = action
        return action

    # ------------------------------------------------------------------
    # Map preprocessing
    # ------------------------------------------------------------------

    def _ensure_map(self, map_state: np.ndarray) -> None:
        """
        Reconstruct and preprocess the structural maze.

        In the supplied framework, walls are always visible even in fog mode.
        Therefore every non-wall cell is traversable, including a cell whose
        current observation value is -1.
        """
        wall_map = (map_state == 1).astype(np.int8)
        signature = wall_map.tobytes()

        if (
            self.wall_map is not None
            and self.wall_map.shape == wall_map.shape
            and self.map_signature == signature
        ):
            return

        self.wall_map = wall_map
        self.map_signature = signature
        self.height, self.width = wall_map.shape

        self.cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if wall_map[r, c] == 0
        ]
        self.index = {pos: i for i, pos in enumerate(self.cells)}

        n = len(self.cells)
        self.neighbors = [[] for _ in range(n)]
        self.ghost_actions = [[] for _ in range(n)]
        self.pacman_actions = [[] for _ in range(n)]

        for i, pos in enumerate(self.cells):
            for move in CARDINAL_MOVES:
                next_pos = self._move_position(pos, move)
                j = self.index.get(next_pos)
                if j is not None:
                    self.neighbors[i].append((move, j))

            self.ghost_actions[i] = list(self.neighbors[i])
            self.ghost_actions[i].append((Move.STAY, i))

            # STAY is legal for Pacman and is useful in rare guaranteed-catch
            # or blocked cases. A STAY action always requests one step.
            self.pacman_actions[i].append((Move.STAY, 1, i))

            for move in CARDINAL_MOVES:
                current = pos
                for steps in range(1, self.pacman_speed + 1):
                    current = self._move_position(current, move)
                    endpoint = self.index.get(current)
                    if endpoint is None:
                        break
                    self.pacman_actions[i].append((move, steps, endpoint))

        self.degrees = np.array(
            [len(items) for items in self.neighbors], dtype=np.int8
        )

        self.graph_dist = self._all_pairs_cell_distances()
        self.turn_dist = self._all_pairs_turn_distances()
        self.capture_turns = self._build_capture_turn_distances()
        self.capture_rank = self._build_capture_game_ranks()

        # Reset probabilistic state if a different map is supplied.
        self.belief = None
        self.last_examined = np.zeros(n, dtype=np.int32)
        self.hidden_steps = 0
        self.coverage_target = None

    def _all_pairs_cell_distances(self) -> np.ndarray:
        """Exact BFS distance for ordinary one-cell movement."""
        n = len(self.cells)
        distances = np.full((n, n), INF, dtype=np.int16)

        for source in range(n):
            distances[source, source] = 0
            queue = deque([source])

            while queue:
                current = queue.popleft()
                next_distance = int(distances[source, current]) + 1
                for _, neighbor in self.neighbors[current]:
                    if distances[source, neighbor] == INF:
                        distances[source, neighbor] = next_distance
                        queue.append(neighbor)

        return distances

    def _all_pairs_turn_distances(self) -> np.ndarray:
        """
        BFS in Pacman's action graph.

        One edge is one legal turn, whether the action advances one cell or
        several cells in the same direction.
        """
        n = len(self.cells)
        distances = np.full((n, n), INF, dtype=np.int16)

        for source in range(n):
            distances[source, source] = 0
            queue = deque([source])

            while queue:
                current = queue.popleft()
                next_distance = int(distances[source, current]) + 1

                for move, _, endpoint in self.pacman_actions[current]:
                    if move == Move.STAY:
                        continue
                    if distances[source, endpoint] == INF:
                        distances[source, endpoint] = next_distance
                        queue.append(endpoint)

        return distances

    def _build_capture_turn_distances(self) -> np.ndarray:
        """
        Minimum Pacman turns needed to reach a capture cell for each Ghost cell.

        Capture is based on final Manhattan distance < 2, so valid capture
        cells are the Ghost cell itself and its traversable cardinal neighbors.
        """
        n = len(self.cells)
        result = np.full((n, n), INF, dtype=np.int16)

        for ghost_index in range(n):
            capture_indices = [ghost_index]
            capture_indices.extend(
                neighbor for _, neighbor in self.neighbors[ghost_index]
            )
            result[:, ghost_index] = np.min(
                self.turn_dist[:, capture_indices], axis=1
            )

        return result

    def _build_capture_game_ranks(self) -> np.ndarray:
        """
        Solve the simultaneous perfect-information reachability game.

        ``rank[p, g] == k`` means Pacman can force a capture from positions
        ``p, g`` in at most ``k`` turns, irrespective of Ghost's legal move.
        A value of -1 means the two positions are in disconnected components.

        This is exact for the supplied rules: Pacman first commits to one
        straight-line endpoint, Ghost simultaneously commits to one adjacent
        endpoint (or STAY), then Manhattan distance < 2 is tested.  NumPy
        evaluates the entire product graph at once; on the official map this
        takes only a few tens of milliseconds and the retained table is below
        100 KB.
        """
        n = len(self.cells)
        if n == 0:
            return np.empty((0, 0), dtype=np.int16)

        max_p = max(len(actions) for actions in self.pacman_actions)
        max_g = max(len(actions) for actions in self.ghost_actions)
        p_endpoints = np.zeros((n, max_p), dtype=np.int16)
        g_endpoints = np.zeros((n, max_g), dtype=np.int16)
        p_mask = np.zeros((n, max_p), dtype=bool)
        g_mask = np.zeros((n, max_g), dtype=bool)

        for index, actions in enumerate(self.pacman_actions):
            count = len(actions)
            p_endpoints[index, :count] = [
                endpoint for _, _, endpoint in actions
            ]
            p_mask[index, :count] = True

        for index, actions in enumerate(self.ghost_actions):
            count = len(actions)
            g_endpoints[index, :count] = [
                endpoint for _, endpoint in actions
            ]
            g_mask[index, :count] = True

        coordinates = np.asarray(self.cells, dtype=np.int16)
        manhattan = np.abs(
            coordinates[:, None, :] - coordinates[None, :, :]
        ).sum(axis=2)
        winning = manhattan < 2
        ranks = np.full((n, n), -1, dtype=np.int16)
        ranks[winning] = 0

        # Attractor recurrence:
        # exists Pacman action for which every legal Ghost reply is already
        # in the previous winning set.
        for turn in range(1, 201):
            safe_endpoint = np.all(
                winning[:, g_endpoints] | ~g_mask[None, :, :],
                axis=2,
            )
            new_winning = np.any(
                safe_endpoint[p_endpoints, :]
                & p_mask[:, :, None],
                axis=1,
            )
            added = new_winning & ~winning
            if not bool(np.any(added)):
                break
            ranks[added] = turn
            winning |= added

        return ranks

    # ------------------------------------------------------------------
    # Ghost behaviour model
    # ------------------------------------------------------------------

    def _learn_from_visible_transition(
        self,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> None:
        """
        Update model credibility only from consecutive exact observations.

        If the Ghost was hidden for one or more steps, the displacement is not
        a valid single Ghost action and must not be learned as one.
        """
        if (
            enemy_position is None
            or self.last_observed_enemy is None
            or self.last_observed_step is None
            or self.previous_my_position is None
            or step_number != self.last_observed_step + 1
        ):
            return

        delta = (
            enemy_position[0] - self.last_observed_enemy[0],
            enemy_position[1] - self.last_observed_enemy[1],
        )
        actual_move = self._move_from_delta(delta)
        if actual_move is None:
            return

        old_index = self.index.get(self.last_observed_enemy)
        if old_index is None:
            return

        legal_moves, components = self._ghost_policy_components(
            old_index,
            self.index.get(self.previous_my_position),
            self.last_ghost_move,
        )

        try:
            actual_index = legal_moves.index(actual_move)
        except ValueError:
            return

        likelihoods = components[:, actual_index]

        # Slow EMA learning limits overreaction to a short deterministic
        # opening sequence.
        self.behaviour_quality = (
            0.90 * self.behaviour_quality + 0.10 * likelihoods
        )
        self.behaviour_quality = np.clip(
            self.behaviour_quality, 0.02, 1.0
        )
        self.last_ghost_move = actual_move

    def _ghost_distribution(
        self,
        ghost_index: int,
        pacman_index: Optional[int],
        inertia_move: Optional[Move],
    ) -> Tuple[List[Move], np.ndarray, List[int]]:
        """
        Return a robust probability distribution over legal Ghost actions.

        Forty-five percent of the weighting may adapt to observed behaviour,
        while the remaining part stays anchored to a broad prior. A final
        uniform floor covers adversarial or deliberately deceptive Ghosts.
        """
        legal_moves, components = self._ghost_policy_components(
            ghost_index, pacman_index, inertia_move
        )

        posterior = self.behaviour_prior * self.behaviour_quality
        posterior_sum = float(posterior.sum())
        if posterior_sum <= 0.0:
            posterior = self.behaviour_prior.copy()
        else:
            posterior /= posterior_sum

        model_weights = 0.55 * self.behaviour_prior + 0.45 * posterior
        probabilities = np.dot(model_weights, components)

        uniform = np.full(
            len(legal_moves), 1.0 / len(legal_moves), dtype=float
        )
        probabilities = 0.86 * probabilities + 0.14 * uniform
        probabilities /= probabilities.sum()

        endpoints = [
            endpoint for _, endpoint in self.ghost_actions[ghost_index]
        ]
        return legal_moves, probabilities, endpoints

    def _ghost_policy_components(
        self,
        ghost_index: int,
        pacman_index: Optional[int],
        inertia_move: Optional[Move],
    ) -> Tuple[List[Move], np.ndarray]:
        actions = self.ghost_actions[ghost_index]
        legal_moves = [move for move, _ in actions]
        endpoints = [endpoint for _, endpoint in actions]
        count = len(actions)

        uniform = np.full(count, 1.0 / count, dtype=float)

        # Flee policy: maximize real maze distance, not Manhattan distance.
        flee_logits = np.zeros(count, dtype=float)
        if pacman_index is not None:
            current_distance = self._finite_distance(
                int(self.graph_dist[ghost_index, pacman_index])
            )
            for k, endpoint in enumerate(endpoints):
                new_distance = self._finite_distance(
                    int(self.graph_dist[endpoint, pacman_index])
                )
                flee_logits[k] = 1.10 * (
                    new_distance - current_distance
                )
        flee = self._softmax(flee_logits)

        # Inertia policy: prefer continuing the last observed direction, but
        # never make it deterministic.
        inertia = uniform.copy()
        if inertia_move in legal_moves:
            remaining = 0.30
            inertia.fill(remaining / max(1, count - 1))
            inertia[legal_moves.index(inertia_move)] = 0.70
            if count == 1:
                inertia[0] = 1.0

        # Mobility policy: prefer junctions and avoid nearby dead ends while
        # still valuing distance from Pacman.
        mobility_logits = np.zeros(count, dtype=float)
        for k, endpoint in enumerate(endpoints):
            degree = int(self.degrees[endpoint])
            mobility_logits[k] = 0.85 * degree
            if degree <= 1:
                mobility_logits[k] -= 2.0
            if pacman_index is not None:
                distance = self._finite_distance(
                    int(self.graph_dist[endpoint, pacman_index])
                )
                mobility_logits[k] += 0.10 * distance

            if legal_moves[k] == Move.STAY:
                mobility_logits[k] -= 0.40

        mobility = self._softmax(mobility_logits)

        components = np.vstack([uniform, flee, inertia, mobility])
        return legal_moves, components

    # ------------------------------------------------------------------
    # Clear-mode decision: risk-sensitive Expectiminimax
    # ------------------------------------------------------------------

    def _choose_clear_action(
        self, my_position: Position, enemy_position: Position
    ) -> PacmanAction:
        pacman_index = self.index.get(my_position)
        ghost_index = self.index.get(enemy_position)
        if pacman_index is None or ghost_index is None:
            return (Move.STAY, 1)

        legal_moves, probabilities, ghost_endpoints = (
            self._ghost_distribution(
                ghost_index, pacman_index, self.last_ghost_move
            )
        )
        all_actions = self.pacman_actions[pacman_index]
        legacy_action = self._legacy_clear_action(
            pacman_index, ghost_index
        )

        # The former version hard-filtered every action that did not reduce
        # the exact adversarial capture rank immediately.  That is a sound
        # worst-case rule, but it is too conservative for the tournament's
        # average-step objective: a temporary rank plateau can cut off a
        # fleeing Ghost one or two turns earlier.  Keep the rank as a safety
        # certificate and portfolio feature instead of a blanket prohibition.
        current_rank = -1
        best_worst_rank = 1000
        action_worst_rank: Dict[PacmanAction, int] = {}
        action_reply_ranks: Dict[PacmanAction, Tuple[int, ...]] = {}
        action_reply_turns: Dict[PacmanAction, Tuple[int, ...]] = {}
        if self.capture_rank is not None:
            current_rank = int(
                self.capture_rank[pacman_index, ghost_index]
            )
            for action in all_actions:
                endpoint = action[2]
                successor_ranks = [
                    int(self.capture_rank[endpoint, g2])
                    for g2 in ghost_endpoints
                ]
                worst_rank = (
                    1000
                    if any(rank < 0 for rank in successor_ranks)
                    else max(successor_ranks)
                )
                key = (action[0], action[1])
                action_worst_rank[key] = worst_rank
                action_reply_ranks[key] = tuple(successor_ranks)
                action_reply_turns[key] = tuple(
                    int(self.capture_turns[endpoint, g2])
                    for g2 in ghost_endpoints
                )
                best_worst_rank = min(best_worst_rank, worst_rank)

        # First look for a one-turn guaranteed capture. This is stronger than
        # simply chasing the current Ghost cell because both sides move
        # simultaneously.
        guaranteed: List[Tuple[float, int, PacmanAction]] = []
        for move, steps, endpoint in all_actions:
            captured = [
                self._is_capture_index(endpoint, g2)
                for g2 in ghost_endpoints
            ]
            if all(captured):
                loop_cost = self._history_penalty(self.cells[endpoint])
                guaranteed.append(
                    (loop_cost, steps, (move, steps))
                )

        if guaranteed:
            guaranteed.sort(key=lambda item: (item[0], item[1]))
            return guaranteed[0][2]

        cache: Dict[Tuple[int, int, int, Optional[Move]], float] = {}
        best_score = -float("inf")
        robust_action: PacmanAction = (Move.STAY, 1)
        action_scores: Dict[PacmanAction, float] = {}

        for move, steps, endpoint in all_actions:
            values = []

            for ghost_move, ghost_endpoint in zip(
                legal_moves, ghost_endpoints
            ):
                if self._is_capture_index(endpoint, ghost_endpoint):
                    value = 10000.0
                else:
                    value = self._clear_state_value(
                        endpoint,
                        ghost_endpoint,
                        depth=1,
                        inertia_move=ghost_move,
                        cache=cache,
                    )
                values.append(value)

            score = self._risk_aggregate(values, probabilities)
            score -= 18.0 * self._history_penalty(
                self.cells[endpoint]
            )

            if move == Move.STAY:
                score -= 8.0

            score += self._action_continuity_bonus(move)
            action_scores[(move, steps)] = score

            if score > best_score + 1e-9:
                best_score = score
                robust_action = (move, steps)
            elif abs(score - best_score) <= 1e-9:
                robust_action = self._tie_break_action(
                    robust_action, (move, steps)
                )

        return self._select_clear_portfolio_action(
            legacy_action=legacy_action,
            robust_action=robust_action,
            action_scores=action_scores,
            action_worst_rank=action_worst_rank,
            action_reply_ranks=action_reply_ranks,
            action_reply_turns=action_reply_turns,
            current_rank=current_rank,
            best_worst_rank=best_worst_rank,
        )

    def _select_clear_portfolio_action(
        self,
        legacy_action: PacmanAction,
        robust_action: PacmanAction,
        action_scores: Dict[PacmanAction, float],
        action_worst_rank: Dict[PacmanAction, int],
        action_reply_ranks: Dict[PacmanAction, Tuple[int, ...]],
        action_reply_turns: Dict[PacmanAction, Tuple[int, ...]],
        current_rank: int,
        best_worst_rank: int,
    ) -> PacmanAction:
        """
        Combine the proven Team 16 chase with the new robust policy.

        The legacy candidate is the empirical anchor because it already
        achieved a very low average capture time on the public field.  The
        learned candidate is allowed to replace it only for a material model
        gain, a clear exact-rank gain, or an observed loop.  This avoids both
        extremes: blindly reverting to the old policy and forcing minimax
        progress on every single turn.
        """
        if legacy_action == robust_action:
            return legacy_action

        legacy_rank = action_worst_rank.get(legacy_action, 1000)
        robust_rank = action_worst_rank.get(robust_action, 1000)

        # Never trade a finite safety certificate for an action that can enter
        # an unreachable/disconnected state in the exact game graph.
        if legacy_rank >= 1000 and robust_rank < legacy_rank:
            return robust_action

        # A move that worsens the guarantee by more than one full turn is not
        # a harmless tactical plateau.  Prefer the robust candidate when it
        # repairs that loss.
        if (
            current_rank > 0
            and legacy_rank > current_rank + 1
            and robust_rank < legacy_rank
        ):
            return robust_action

        legacy_endpoint = self._action_endpoint_index(legacy_action)
        robust_endpoint = self._action_endpoint_index(robust_action)
        if legacy_endpoint is not None and robust_endpoint is not None:
            legacy_pos = self.cells[legacy_endpoint]
            robust_pos = self.cells[robust_endpoint]
            legacy_loop = sum(
                position == legacy_pos
                for position in list(self.legacy_recent_positions)[-5:]
            )
            robust_loop = sum(
                position == robust_pos
                for position in list(self.legacy_recent_positions)[-5:]
            )
            if (
                legacy_loop >= 2
                and robust_loop < legacy_loop
                and robust_rank < legacy_rank
            ):
                return robust_action

        legacy_score = action_scores.get(
            legacy_action, -float("inf")
        )
        robust_score = action_scores.get(
            robust_action, -float("inf")
        )
        model_gain = robust_score - legacy_score
        rank_gain = legacy_rank - robust_rank

        # Roughly half a capture-turn of score is required before the learned
        # model may overrule the empirical anchor.  A two-turn exact rank gain
        # lowers that bar because it is independent of the learned Ghost
        # distribution.
        if model_gain >= 55.0 and robust_rank < legacy_rank:
            return robust_action
        if (
            rank_gain >= 2
            and model_gain >= 18.0
            and robust_rank <= best_worst_rank + 1
        ):
            return robust_action

        # Reply-wise Pareto-safe exact-rank tie-break.  A lower maximum rank
        # alone is not enough: an action may improve the worst case while being
        # slower against one particular legal Ghost reply.  Override the legacy
        # action only when the robust endpoint is no worse for every one-step
        # Ghost reply in both exact game rank and static capture-turn distance,
        # and is strictly better for at least one reply.  This preserves the
        # verified opening cut-off while removing the broader source of local
        # average-step regressions.
        legacy_reply_ranks = action_reply_ranks.get(legacy_action)
        robust_reply_ranks = action_reply_ranks.get(robust_action)
        legacy_reply_turns = action_reply_turns.get(legacy_action)
        robust_reply_turns = action_reply_turns.get(robust_action)
        reply_wise_dominance = (
            legacy_reply_ranks is not None
            and robust_reply_ranks is not None
            and legacy_reply_turns is not None
            and robust_reply_turns is not None
            and len(legacy_reply_ranks) == len(robust_reply_ranks)
            and all(
                robust_value <= legacy_value
                for robust_value, legacy_value in zip(
                    robust_reply_ranks, legacy_reply_ranks
                )
            )
            and all(
                robust_value <= legacy_value
                for robust_value, legacy_value in zip(
                    robust_reply_turns, legacy_reply_turns
                )
            )
            and any(
                robust_value < legacy_value
                for robust_value, legacy_value in zip(
                    robust_reply_ranks, legacy_reply_ranks
                )
            )
        )
        if (
            rank_gain >= 1
            and robust_rank <= best_worst_rank
            and abs(model_gain) <= 1e-9
            and reply_wise_dominance
        ):
            return robust_action
        return legacy_action

    def _legacy_clear_action(
        self, pacman_index: int, ghost_index: int
    ) -> PacmanAction:
        """
        Fast array-backed reproduction of Team 16's first-round Pacman.

        It retains the original forced-capture test, turn-aware A* chase,
        bounded minimax, mobility terms, and anti-loop signal.  All distances
        come from the new agent's precomputed BFS tables, so this oracle is
        considerably cheaper than constructing a second map helper.
        """
        actions = self._legacy_pacman_actions(pacman_index)
        ghost_replies = [
            endpoint
            for _, endpoint in self.ghost_actions[ghost_index]
        ]
        if not actions:
            return (Move.STAY, 1)

        best_forced = actions[0]
        best_cover = -1
        best_worst_distance = int(INF)
        for move, steps, endpoint in actions:
            cover = 0
            worst_distance = 0
            for reply in ghost_replies:
                distance = self._manhattan_indices(endpoint, reply)
                worst_distance = max(worst_distance, distance)
                if distance < 2:
                    cover += 1
            if (
                cover > best_cover
                or (
                    cover == best_cover
                    and worst_distance < best_worst_distance
                )
            ):
                best_cover = cover
                best_worst_distance = worst_distance
                best_forced = (move, steps, endpoint)

        if best_cover == len(ghost_replies):
            return (best_forced[0], best_forced[1])

        fast_action = self._legacy_fast_action(
            pacman_index, ghost_index
        )
        capture_distance = self._finite_distance(
            int(self.capture_turns[pacman_index, ghost_index])
        )
        ghost_mobility = len(self.ghost_actions[ghost_index])
        safe_cache: Dict[Tuple[int, int, int], int] = {}
        ghost_safe_area = self._legacy_safe_area_size(
            ghost_index,
            pacman_index,
            limit=4,
            cache=safe_cache,
        )
        stuck = self._legacy_is_stuck(capture_distance)

        if (
            not stuck
            and (
                (capture_distance > 5 and ghost_mobility <= 3)
                or capture_distance > 8
            )
        ):
            return fast_action

        # The array-backed recursion normally completes in a few milliseconds.
        # The local cap reserves ample time for the learned portfolio member.
        legacy_deadline = min(
            self.deadline, time.perf_counter() + 0.16
        )
        value_cache: Dict[Tuple[int, int, int], float] = {}

        fast_endpoint = self._endpoint_for_action(
            pacman_index, fast_action
        )
        best_action = fast_action
        best_score = self._legacy_evaluate_action(
            fast_endpoint,
            ghost_index,
            depth=2,
            action=fast_action,
            deadline=legacy_deadline,
            value_cache=value_cache,
            safe_cache=safe_cache,
        )

        search_depth = (
            3
            if (
                capture_distance <= 5
                or ghost_mobility >= 4
                or ghost_safe_area >= 10
            )
            else 2
        )

        for move, steps, endpoint in self._legacy_ordered_actions(
            pacman_index, ghost_index
        ):
            action = (move, steps)
            score = self._legacy_evaluate_action(
                endpoint,
                ghost_index,
                depth=search_depth,
                action=action,
                deadline=legacy_deadline,
                value_cache=value_cache,
                safe_cache=safe_cache,
            )
            if score < best_score:
                best_score = score
                best_action = action
            if time.perf_counter() >= legacy_deadline:
                break

        return best_action

    def _legacy_evaluate_action(
        self,
        pacman_index: int,
        ghost_index: int,
        depth: int,
        action: PacmanAction,
        deadline: float,
        value_cache: Dict[Tuple[int, int, int], float],
        safe_cache: Dict[Tuple[int, int, int], int],
    ) -> float:
        ghost_options = self._legacy_ordered_ghost_replies(
            pacman_index, ghost_index, safe_cache
        )
        worst_score = -1_000_000.0
        total_score = 0.0
        captures = 0
        checked = 0

        for reply in ghost_options:
            checked += 1
            if self._is_capture_index(pacman_index, reply):
                score = 0.0
                captures += 1
            else:
                score = 1.0 + self._legacy_search_value(
                    pacman_index,
                    reply,
                    depth - 1,
                    deadline,
                    value_cache,
                    safe_cache,
                )
            worst_score = max(worst_score, score)
            total_score += score
            if time.perf_counter() >= deadline:
                break

        average_score = total_score / max(1, checked)
        capture_distance = self._finite_distance(
            int(self.capture_turns[pacman_index, ghost_index])
        )
        ghost_mobility = len(self.ghost_actions[ghost_index])
        pacman_mobility = len(
            self._legacy_pacman_actions(pacman_index)
        )
        corridor_penalty = (
            0.55
            if int(self.degrees[pacman_index]) <= 2
            and action[1] >= 2
            else 0.0
        )
        speed_bonus = (
            0.08 * action[1]
            if int(self.degrees[pacman_index]) >= 2
            else 0.0
        )
        position = self.cells[pacman_index]
        loop_penalty = (
            0.65
            if position in self.legacy_recent_positions
            else 0.0
        )

        return (
            2.35 * worst_score
            + 0.55 * average_score
            + 0.18 * capture_distance
            + 0.22 * ghost_mobility
            + corridor_penalty
            + loop_penalty
            - 0.05 * pacman_mobility
            - speed_bonus
            - 3.2 * captures
        )

    def _legacy_search_value(
        self,
        pacman_index: int,
        ghost_index: int,
        depth: int,
        deadline: float,
        value_cache: Dict[Tuple[int, int, int], float],
        safe_cache: Dict[Tuple[int, int, int], int],
    ) -> float:
        if self._is_capture_index(pacman_index, ghost_index):
            return 0.0
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._legacy_static_score(
                pacman_index, ghost_index, safe_cache
            )

        key = (pacman_index, ghost_index, depth)
        cached = value_cache.get(key)
        if cached is not None:
            return cached

        best_value = 1_000_000.0
        for _, _, endpoint in self._legacy_ordered_actions(
            pacman_index, ghost_index
        ):
            worst_reply = -1_000_000.0
            for reply in self._legacy_ordered_ghost_replies(
                endpoint, ghost_index, safe_cache
            ):
                if self._is_capture_index(endpoint, reply):
                    value = 1.0
                else:
                    value = 1.0 + self._legacy_search_value(
                        endpoint,
                        reply,
                        depth - 1,
                        deadline,
                        value_cache,
                        safe_cache,
                    )
                worst_reply = max(worst_reply, value)
                if (
                    worst_reply >= best_value
                    or time.perf_counter() >= deadline
                ):
                    break
            best_value = min(best_value, worst_reply)
            if time.perf_counter() >= deadline:
                break

        value_cache[key] = best_value
        return best_value

    def _legacy_static_score(
        self,
        pacman_index: int,
        ghost_index: int,
        safe_cache: Dict[Tuple[int, int, int], int],
    ) -> float:
        capture_turns = self._finite_distance(
            int(self.capture_turns[pacman_index, ghost_index])
        )
        maze_distance = self._finite_distance(
            int(self.graph_dist[pacman_index, ghost_index])
        )
        ghost_safe_area = self._legacy_safe_area_size(
            ghost_index,
            pacman_index,
            limit=4,
            cache=safe_cache,
        )
        return (
            float(capture_turns)
            + 0.08 * self._manhattan_indices(
                pacman_index, ghost_index
            )
            + 0.28 * len(self.ghost_actions[ghost_index])
            + 0.035 * ghost_safe_area
        )

    def _legacy_ordered_actions(
        self, pacman_index: int, ghost_index: int
    ) -> List[Tuple[Move, int, int]]:
        return sorted(
            self._legacy_pacman_actions(pacman_index),
            key=lambda action: (
                int(self.capture_turns[action[2], ghost_index]),
                int(self.graph_dist[action[2], ghost_index]),
                -action[1],
            ),
        )

    def _legacy_ordered_ghost_replies(
        self,
        pacman_index: int,
        ghost_index: int,
        safe_cache: Dict[Tuple[int, int, int], int],
    ) -> List[int]:
        replies = [
            endpoint
            for _, endpoint in self.ghost_actions[ghost_index]
        ]
        return sorted(
            replies,
            key=lambda endpoint: (
                -int(self.capture_turns[pacman_index, endpoint]),
                -self._legacy_safe_area_size(
                    endpoint,
                    pacman_index,
                    limit=4,
                    cache=safe_cache,
                ),
                -int(self.degrees[endpoint]),
            ),
        )

    def _legacy_safe_area_size(
        self,
        start: int,
        pacman_index: int,
        limit: int,
        cache: Dict[Tuple[int, int, int], int],
    ) -> int:
        key = (start, pacman_index, limit)
        cached = cache.get(key)
        if cached is not None:
            return cached

        total = 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            current, depth = queue.popleft()
            if int(self.capture_turns[pacman_index, current]) > 1:
                total += 1
            if depth >= limit:
                continue
            for _, endpoint in self.ghost_actions[current]:
                if endpoint not in seen:
                    seen.add(endpoint)
                    queue.append((endpoint, depth + 1))
        cache[key] = total
        return total

    def _legacy_fast_action(
        self, pacman_index: int, ghost_index: int
    ) -> PacmanAction:
        capture_cells = [ghost_index]
        capture_cells.extend(
            endpoint for _, endpoint in self.neighbors[ghost_index]
        )
        best_goal = capture_cells[0]
        best_turns = int(INF)
        for goal in capture_cells:
            turns = int(self.turn_dist[pacman_index, goal])
            if turns < best_turns:
                best_turns = turns
                best_goal = goal

        if best_turns < int(INF) and best_turns > 0:
            # Reproduce the original turn-aware A* tie-breaking exactly.
            # Merely taking the first endpoint on any shortest path is not
            # equivalent: A*'s Manhattan priority deliberately prefers the
            # branch that makes more geometric progress among equal-cost
            # maze paths.
            frontier: List[Tuple[float, int, int]] = []
            sequence = 0
            start_h = (
                self._manhattan_indices(pacman_index, best_goal)
                / self.pacman_speed
            )
            heapq.heappush(
                frontier, (start_h, sequence, pacman_index)
            )
            cost = {pacman_index: 0}
            parent: Dict[int, Optional[int]] = {
                pacman_index: None
            }

            while frontier:
                _, _, current = heapq.heappop(frontier)
                if current == best_goal:
                    break

                for _, _, endpoint in self._legacy_pacman_actions(
                    current
                ):
                    new_cost = cost[current] + 1
                    if new_cost < cost.get(endpoint, int(INF)):
                        cost[endpoint] = new_cost
                        parent[endpoint] = current
                        sequence += 1
                        priority = (
                            new_cost
                            + self._manhattan_indices(
                                endpoint, best_goal
                            )
                            / self.pacman_speed
                        )
                        heapq.heappush(
                            frontier,
                            (priority, sequence, endpoint),
                        )

            if best_goal in parent:
                path = []
                current: Optional[int] = best_goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                if len(path) >= 2:
                    target = path[1]
                    for move, steps, endpoint in (
                        self._legacy_pacman_actions(pacman_index)
                    ):
                        if endpoint == target:
                            return (move, steps)

        actions = self._legacy_pacman_actions(pacman_index)
        if not actions:
            return (Move.STAY, 1)
        center = (self.height // 2, self.width // 2)
        center_index = self.index.get(center)
        if center_index is None:
            center_index = pacman_index
        best = min(
            actions,
            key=lambda action: int(
                self.graph_dist[action[2], center_index]
            ),
        )
        return (best[0], best[1])

    def _legacy_pacman_actions(
        self, pacman_index: int
    ) -> List[Tuple[Move, int, int]]:
        actions = [
            action
            for action in self.pacman_actions[pacman_index]
            if action[0] != Move.STAY
        ]
        if actions:
            return actions
        return [(Move.STAY, 1, pacman_index)]

    def _legacy_is_stuck(self, capture_distance: int) -> bool:
        if len(self.legacy_recent_capture_turns) < 4:
            return False
        return (
            capture_distance
            >= min(self.legacy_recent_capture_turns)
            and len(set(self.legacy_recent_positions)) <= 4
        )

    def _endpoint_for_action(
        self, pacman_index: int, action: PacmanAction
    ) -> int:
        for move, steps, endpoint in self.pacman_actions[pacman_index]:
            if move == action[0] and steps == action[1]:
                return endpoint
        return pacman_index

    def _action_endpoint_index(
        self, action: PacmanAction
    ) -> Optional[int]:
        if not self.position_history:
            return None
        pacman_index = self.index.get(self.position_history[-1])
        if pacman_index is None:
            return None
        return self._endpoint_for_action(pacman_index, action)

    def _pacman_endpoint(
        self, position: Position, action: PacmanAction
    ) -> Position:
        pacman_index = self.index.get(position)
        if pacman_index is None:
            return position
        endpoint = self._endpoint_for_action(pacman_index, action)
        return self.cells[endpoint]

    def _manhattan_indices(self, first: int, second: int) -> int:
        a = self.cells[first]
        b = self.cells[second]
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _clear_state_value(
        self,
        pacman_index: int,
        ghost_index: int,
        depth: int,
        inertia_move: Optional[Move],
        cache: Dict[Tuple[int, int, int, Optional[Move]], float],
    ) -> float:
        if self._is_capture_index(pacman_index, ghost_index):
            return 10000.0 + 100.0 * depth

        if depth <= 0 or time.perf_counter() >= self.deadline:
            return self._clear_heuristic(pacman_index, ghost_index)

        key = (pacman_index, ghost_index, depth, inertia_move)
        cached = cache.get(key)
        if cached is not None:
            return cached

        legal_moves, probabilities, ghost_endpoints = (
            self._ghost_distribution(
                ghost_index, pacman_index, inertia_move
            )
        )

        best = -float("inf")
        for _, _, pacman_endpoint in self.pacman_actions[pacman_index]:
            outcomes = []

            for ghost_move, ghost_endpoint in zip(
                legal_moves, ghost_endpoints
            ):
                if self._is_capture_index(
                    pacman_endpoint, ghost_endpoint
                ):
                    value = 10000.0 + 100.0 * depth
                else:
                    value = self._clear_state_value(
                        pacman_endpoint,
                        ghost_endpoint,
                        depth - 1,
                        ghost_move,
                        cache,
                    )
                outcomes.append(value)

            action_value = self._risk_aggregate(
                outcomes, probabilities
            )
            if action_value > best:
                best = action_value

            if time.perf_counter() >= self.deadline:
                break

        if best == -float("inf"):
            best = self._clear_heuristic(
                pacman_index, ghost_index
            )

        cache[key] = best
        return best

    def _clear_heuristic(
        self, pacman_index: int, ghost_index: int
    ) -> float:
        capture_turns = self._finite_distance(
            int(self.capture_turns[pacman_index, ghost_index])
        )
        maze_distance = self._finite_distance(
            int(self.graph_dist[pacman_index, ghost_index])
        )
        ghost_degree = int(self.degrees[ghost_index])

        safe_ghost_moves = 0
        for _, endpoint in self.ghost_actions[ghost_index]:
            if not self._is_capture_index(pacman_index, endpoint):
                safe_ghost_moves += 1

        score = -115.0 * capture_turns
        score -= 3.0 * maze_distance
        score -= 6.0 * ghost_degree
        score -= 5.0 * safe_ghost_moves

        pacman_pos = self.cells[pacman_index]
        ghost_pos = self.cells[ghost_index]
        if pacman_pos[0] == ghost_pos[0] or pacman_pos[1] == ghost_pos[1]:
            score += 7.0

        return score

    def _risk_aggregate(
        self, values: Sequence[float], probabilities: np.ndarray
    ) -> float:
        """
        Blend expected value with lower-tail and worst-case performance.

        Pure expectation overfits the learned model. Pure minimax is often too
        pessimistic and ignores the statistical regularities of supplied
        Ghosts. This mixture remains useful against both.
        """
        array = np.asarray(values, dtype=float)
        expected = float(np.dot(probabilities, array))
        worst = float(np.min(array))

        order = np.argsort(array)
        remaining = 0.35
        tail_total = 0.0
        tail_weight = 0.0

        for index in order:
            take = min(remaining, float(probabilities[index]))
            if take > 0:
                tail_total += take * float(array[index])
                tail_weight += take
                remaining -= take
            if remaining <= 1e-12:
                break

        lower_tail = (
            tail_total / tail_weight if tail_weight > 0 else worst
        )
        return 0.66 * expected + 0.20 * lower_tail + 0.14 * worst

    # ------------------------------------------------------------------
    # Blind-mode decision: Bayesian search and information gathering
    # ------------------------------------------------------------------

    def _update_belief(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> None:
        n = len(self.cells)

        if self.last_examined is not None:
            for i, pos in enumerate(self.cells):
                if map_state[pos] == 0:
                    self.last_examined[i] = step_number

        if enemy_position is not None:
            self.belief = np.zeros(n, dtype=float)
            enemy_index = self.index.get(enemy_position)
            if enemy_index is not None:
                self.belief[enemy_index] = 1.0
            return

        if self.belief is None or float(self.belief.sum()) <= 0.0:
            self.belief = self._initial_hidden_belief(
                map_state, my_position
            )
        else:
            reference_pos = (
                self.previous_my_position
                if self.previous_my_position is not None
                else my_position
            )
            reference_index = self.index.get(reference_pos)
            self.belief = self._propagate_belief(
                self.belief, reference_index
            )

        # Negative observation: in the framework, a visible empty cell is 0.
        # If enemy_position is None, the Ghost is definitely not in such cells.
        for i, pos in enumerate(self.cells):
            if map_state[pos] == 0:
                self.belief[i] = 0.0

        total = float(self.belief.sum())
        if total <= 1e-15:
            self.belief = self._initial_hidden_belief(
                map_state, my_position
            )
        else:
            self.belief /= total

    def _initial_hidden_belief(
        self, map_state: np.ndarray, my_position: Position
    ) -> np.ndarray:
        """
        Build a broad, framework-informed prior without hard-coded positions.

        Both official reset modes place the hider above the seeker: stochastic
        starts sample Ghost from the upper part of the maze and Pacman from the
        lower part, while the supplied deterministic starts have the same
        ordering.  We encode that fact only as a gentle vertical gradient,
        retaining substantial probability everywhere that is unseen and
        reachable.  This is much less brittle than assuming one opening cell.

        Cells in disconnected wall components are excluded because neither
        player can cross walls.  A small junction preference represents the
        general tendency of a rational hider to keep escape alternatives.
        """
        n = len(self.cells)
        belief = np.zeros(n, dtype=float)
        pacman_index = self.index.get(my_position)

        candidates = [
            i
            for i, pos in enumerate(self.cells)
            if map_state[pos] == -1
            and (
                pacman_index is None
                or int(self.graph_dist[pacman_index, i]) < int(INF)
            )
        ]
        if not candidates:
            candidates = [
                i
                for i in range(n)
                if pacman_index is None
                or int(self.graph_dist[pacman_index, i]) < int(INF)
            ]
        if not candidates:
            candidates = list(range(n))

        my_row = my_position[0]
        for index in candidates:
            row, _ = self.cells[index]
            vertical = math.exp(
                max(-1.0, min(1.0, 0.07 * (my_row - row)))
            )
            junction = 0.88 + 0.06 * int(self.degrees[index])
            belief[index] = vertical * junction

        total = float(belief.sum())
        if total <= 0.0:
            belief[candidates] = 1.0 / len(candidates)
        else:
            belief /= total
        return belief

    def _propagate_belief(
        self,
        belief: np.ndarray,
        pacman_index: Optional[int],
    ) -> np.ndarray:
        propagated = np.zeros_like(belief)

        significant = np.flatnonzero(belief > 1e-12)
        for ghost_index in significant:
            _, probabilities, endpoints = self._ghost_distribution(
                int(ghost_index),
                pacman_index,
                self.last_ghost_move,
            )
            mass = float(belief[ghost_index])
            for probability, endpoint in zip(
                probabilities, endpoints
            ):
                propagated[endpoint] += mass * float(probability)

        total = float(propagated.sum())
        if total > 0.0:
            propagated /= total
        return propagated

    def _choose_blind_action(
        self, my_position: Position, map_state: np.ndarray
    ) -> PacmanAction:
        pacman_index = self.index.get(my_position)
        if pacman_index is None or self.belief is None:
            return (Move.STAY, 1)

        next_belief = self._propagate_belief(
            self.belief, pacman_index
        )
        if float(next_belief.sum()) <= 0.0:
            next_belief = self.belief.copy()

        entropy = self._normalized_entropy(next_belief)
        information_weight = 70.0 + 150.0 * entropy
        distance_weight = 92.0 - 25.0 * entropy

        best_score = -float("inf")
        best_action: PacmanAction = (Move.STAY, 1)

        for move, steps, endpoint in self.pacman_actions[pacman_index]:
            turn_row = self.capture_turns[endpoint].astype(np.int32)
            finite_turns = np.where(
                turn_row >= int(INF), 40, turn_row
            ).astype(float)

            expected_turns = float(np.dot(next_belief, finite_turns))
            upper_quantile = self._weighted_quantile(
                finite_turns, next_belief, 0.80
            )

            capture_probability = float(
                next_belief[turn_row == 0].sum()
            )

            visible_indices = self._visible_cell_indices(
                endpoint, self.vision_radius_estimate
            )
            reacquire_probability = float(
                next_belief[visible_indices].sum()
            )

            score = 9000.0 * capture_probability
            score -= distance_weight * expected_turns
            score -= 16.0 * upper_quantile
            score += information_weight * reacquire_probability
            score += 5.0 * int(self.degrees[endpoint])
            score -= 15.0 * self._history_penalty(
                self.cells[endpoint]
            )
            score += self._action_continuity_bonus(move)

            if move == Move.STAY:
                score -= 12.0

            if score > best_score + 1e-9:
                best_score = score
                best_action = (move, steps)
            elif abs(score - best_score) <= 1e-9:
                chosen = self._tie_break_action(
                    best_action, (move, steps)
                )
                best_action = chosen

        # A local probability gradient is excellent while the belief is still
        # informative, but it can orbit forever around walls when a hidden
        # Ghost camps. After a moderate unseen interval, switch to a persistent
        # graph-coverage waypoint. The waypoint is selected from observation
        # age and belief mass, then reached with exact turn-aware A*. This
        # guarantees systematic exploration without affecting the normal
        # short pursuit after losing sight of a moving Ghost.
        coverage_delay = (
            24 if self.last_observed_enemy is None else 40
        )
        if self.hidden_steps >= coverage_delay:
            coverage_action = self._coverage_first_action(
                pacman_index, next_belief
            )
            if coverage_action is not None:
                return coverage_action

        # If numerical issues ever make every score invalid, pursue the most
        # probable reachable position with turn-aware A*.
        if not math.isfinite(best_score):
            target_index = int(np.argmax(next_belief))
            fallback = self._astar_first_action(
                pacman_index, target_index
            )
            if fallback is not None:
                return fallback

        return best_action

    def _coverage_first_action(
        self, pacman_index: int, belief: np.ndarray
    ) -> Optional[PacmanAction]:
        """
        Select and persistently approach a stale, informative observation cell.

        The target is not a hand-authored patrol point. It is recomputed from
        the current map graph, current posterior, and the time each cell was
        last observed. Persisting until it becomes visible prevents the
        one-step scoring function from oscillating between equally attractive
        corridors.
        """
        if self.last_examined is None:
            return None

        target_finished = False
        if self.coverage_target is not None:
            target_finished = self.coverage_target == pacman_index

        if self.coverage_target is None or target_finished:
            best_score = -float("inf")
            best_target = None

            for target in range(len(self.cells)):
                turns = int(self.turn_dist[pacman_index, target])
                if turns >= int(INF) or target == pacman_index:
                    continue

                visible = self._visible_cell_indices(
                    target, self.vision_radius_estimate
                )
                never_examined = int(
                    np.count_nonzero(self.last_examined[visible] == 0)
                )
                ages = self.current_step - self.last_examined[visible]
                mean_staleness = float(
                    np.mean(np.minimum(ages, 40))
                )
                posterior_mass = float(belief[visible].sum())

                if int(self.last_examined[target]) == 0:
                    score = 1000.0
                else:
                    score = 0.0
                score += 24.0 * never_examined
                score += 4.0 * mean_staleness
                score += 700.0 * posterior_mass
                score -= 5.0 * turns

                if score > best_score + 1e-9:
                    best_score = score
                    best_target = target
                elif (
                    abs(score - best_score) <= 1e-9
                    and best_target is not None
                    and self.cells[target] < self.cells[best_target]
                ):
                    best_target = target

            self.coverage_target = best_target

        if self.coverage_target is None:
            return None

        action = self._astar_first_action(
            pacman_index, self.coverage_target
        )
        if action is None:
            self.coverage_target = None
        return action

    def _visible_cell_indices(
        self, observer_index: int, radius: int
    ) -> np.ndarray:
        observer = self.cells[observer_index]
        visible = {observer_index}

        for move in CARDINAL_MOVES:
            current = observer
            for _ in range(max(1, radius)):
                current = self._move_position(current, move)
                index = self.index.get(current)
                if index is None:
                    break
                visible.add(index)

        return np.fromiter(visible, dtype=np.int32)

    def _update_vision_radius(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
    ) -> None:
        """
        Infer fog radius from rays that transition from visible 0 to unseen -1.

        In clear mode no inference is needed. If walls block every useful ray,
        the previous estimate (initially 5) is retained.
        """
        if enemy_position is not None and not np.any(map_state == -1):
            return

        candidates = []
        for move in CARDINAL_MOVES:
            current = my_position
            visible_open = 0

            while True:
                current = self._move_position(current, move)
                if not self._in_bounds(current):
                    break
                value = int(map_state[current])
                if value == 1:
                    break
                if value == -1:
                    candidates.append(visible_open)
                    break
                visible_open += 1

        positive = [value for value in candidates if value > 0]
        if positive:
            estimate = int(round(float(np.median(positive))))
            self.vision_radius_estimate = max(1, min(10, estimate))

    # ------------------------------------------------------------------
    # A* fallback
    # ------------------------------------------------------------------

    def _astar_first_action(
        self, start_index: int, goal_index: int
    ) -> Optional[PacmanAction]:
        """
        A* over legal Pacman turns.

        The heuristic ceil(cell_distance / speed) is admissible because one
        turn can traverse at most pacman_speed cells and cannot cross walls.
        """
        if start_index == goal_index:
            return (Move.STAY, 1)

        start_h = self._turn_heuristic(start_index, goal_index)
        frontier = [(start_h, 0, start_index)]
        best_g = {start_index: 0}
        parent: Dict[int, Tuple[int, PacmanAction]] = {}

        while frontier and time.perf_counter() < self.deadline:
            _, g_cost, current = heapq.heappop(frontier)
            if g_cost != best_g.get(current):
                continue
            if current == goal_index:
                break

            for move, steps, endpoint in self.pacman_actions[current]:
                if move == Move.STAY:
                    continue
                new_g = g_cost + 1
                if new_g < best_g.get(endpoint, 10**9):
                    best_g[endpoint] = new_g
                    parent[endpoint] = (
                        current, (move, steps)
                    )
                    priority = new_g + self._turn_heuristic(
                        endpoint, goal_index
                    )
                    heapq.heappush(
                        frontier, (priority, new_g, endpoint)
                    )

        if goal_index not in parent:
            return None

        current = goal_index
        first_action = None
        while current != start_index:
            previous, action = parent[current]
            first_action = action
            current = previous
        return first_action

    # ------------------------------------------------------------------
    # Validation and small helpers
    # ------------------------------------------------------------------

    def _validate_or_fallback(
        self,
        action: Optional[PacmanAction],
        my_position: Position,
    ) -> PacmanAction:
        start_index = self.index.get(my_position)
        if start_index is None:
            return (Move.STAY, 1)

        legal = {
            (move, steps)
            for move, steps, _ in self.pacman_actions[start_index]
        }

        if action in legal:
            return action

        # Prefer a one-cell move into the highest-degree legal neighbor.
        candidates = [
            (int(self.degrees[endpoint]), move, steps)
            for move, steps, endpoint in self.pacman_actions[start_index]
            if move != Move.STAY and steps == 1
        ]
        if candidates:
            candidates.sort(
                key=lambda item: (
                    -item[0],
                    self._move_rank(item[1]),
                )
            )
            _, move, steps = candidates[0]
            return (move, steps)
        return (Move.STAY, 1)

    def _history_penalty(self, position: Position) -> float:
        return float(self.position_history.count(position))

    def _action_continuity_bonus(self, move: Move) -> float:
        if self.last_action is None:
            return 0.0

        last_move = self.last_action[0]
        if move == last_move:
            return 2.0

        opposite = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
        }
        if move == opposite.get(last_move):
            return -2.5
        return 0.0

    def _tie_break_action(
        self, current: PacmanAction, candidate: PacmanAction
    ) -> PacmanAction:
        """
        Deterministic tie-breaking:
        prefer continuing direction, then longer useful straight movement,
        then a stable move order.
        """
        def key(action: PacmanAction):
            move, steps = action
            continuity = (
                0
                if self.last_action is not None
                and move == self.last_action[0]
                else 1
            )
            stay = 1 if move == Move.STAY else 0
            return (
                continuity,
                stay,
                -steps,
                self._move_rank(move),
            )

        return candidate if key(candidate) < key(current) else current

    def _move_rank(self, move: Move) -> int:
        order = {
            Move.UP: 0,
            Move.LEFT: 1,
            Move.DOWN: 2,
            Move.RIGHT: 3,
            Move.STAY: 4,
        }
        return order[move]

    def _turn_heuristic(
        self, current_index: int, goal_index: int
    ) -> int:
        distance = int(self.graph_dist[current_index, goal_index])
        if distance >= int(INF):
            return 1000
        return int(math.ceil(distance / self.pacman_speed))

    def _weighted_quantile(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, quantile, side="left"))
        index = min(index, len(sorted_values) - 1)
        return float(sorted_values[index])

    def _normalized_entropy(self, probabilities: np.ndarray) -> float:
        positive = probabilities[probabilities > 1e-15]
        if len(positive) <= 1:
            return 0.0
        entropy = -float(np.sum(positive * np.log(positive)))
        return entropy / math.log(len(probabilities))

    def _is_capture_index(
        self, pacman_index: int, ghost_index: int
    ) -> bool:
        pacman = self.cells[pacman_index]
        ghost = self.cells[ghost_index]
        return (
            abs(pacman[0] - ghost[0])
            + abs(pacman[1] - ghost[1])
            < 2
        )

    def _finite_distance(self, value: int) -> int:
        return 40 if value >= int(INF) else value

    def _move_position(
        self, position: Position, move: Move
    ) -> Position:
        dr, dc = move.value
        return (position[0] + dr, position[1] + dc)

    def _move_from_delta(
        self, delta: Tuple[int, int]
    ) -> Optional[Move]:
        for move in GHOST_MOVES:
            if move.value == delta:
                return move
        return None

    def _in_bounds(self, position: Position) -> bool:
        return (
            0 <= position[0] < self.height
            and 0 <= position[1] < self.width
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        shifted = np.clip(logits - np.max(logits), -30.0, 30.0)
        exponents = np.exp(shifted)
        total = float(exponents.sum())
        if total <= 0.0:
            return np.full(
                len(logits), 1.0 / len(logits), dtype=float
            )
        return exponents / total


class GhostAgent(BaseGhostAgent):
    """
    Robust hider for Clear and Blind tournament modes.

    The Clear policy combines three layers:

    1. an exact finite-horizon survival value over the complete position
       product graph;
    2. a conservative mixture model of the observed Pacman policy;
    3. stateful cycle detection that safely camps when a deterministic seeker
       has fallen into a remote loop.

    The Blind policy does not chase a stale coordinate.  It maintains a
    probability distribution over Pacman's possible cells, removes locations
    contradicted by negative observations, and selects an action by lower-tail
    survival, expected separation, capture risk, and information denial.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "Belief Maximin Ghost Conservative"
        # Ghost is not passed pacman_speed by the supplied AgentLoader.  The
        # assignment and arena default both specify a two-cell straight move.
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.internal_time_budget = 0.72
        self.deadline = 0.0

        self.wall_map: Optional[np.ndarray] = None
        self.map_signature: Optional[bytes] = None
        self.height = 0
        self.width = 0
        self.cells: List[Position] = []
        self.index: Dict[Position, int] = {}
        self.neighbors: List[List[Tuple[Move, int]]] = []
        self.ghost_actions: List[List[Tuple[Move, int]]] = []
        self.pacman_actions: List[List[Tuple[Move, int, int]]] = []
        self.degrees: Optional[np.ndarray] = None
        self.graph_dist: Optional[np.ndarray] = None
        self.capture_turns: Optional[np.ndarray] = None
        self.capture_rank: Optional[np.ndarray] = None
        self.survival_value: Optional[np.ndarray] = None

        # Pacman behaviour model: uniform, shortest pursuit, inertia, exact
        # rank pursuit.  The broad floor prevents brittle overconfidence.
        self.behaviour_prior = np.array(
            [0.16, 0.42, 0.17, 0.25], dtype=float
        )
        self.behaviour_quality = np.ones(4, dtype=float)
        self.last_pacman_delta: Optional[Tuple[int, int]] = None

        # A common seeker implementation follows a cached BFS path and only
        # replans after its target has moved by two cells.  Keep this as one
        # testable policy hypothesis; its credibility is learned online.
        self.lag_path: Tuple[Move, ...] = ()
        self.lag_target: Optional[int] = None
        self.lag_predicted_pacman: Optional[int] = None
        self.lag_policy_quality = 0.50
        self.lag_policy_observations = 0
        self.lag_distinctive_matches = 0
        self.lag_probe_used = False

        # Observation and belief state.
        self.belief: Optional[np.ndarray] = None
        self.vision_radius_estimate = 5
        self.last_observed_enemy: Optional[Position] = None
        self.last_observed_step: Optional[int] = None
        self.previous_my_position: Optional[Position] = None
        self.current_step = 0

        # Adaptive evasion state.
        self.pacman_history: deque[Position] = deque(maxlen=12)
        self.ghost_history: deque[Position] = deque(maxlen=12)
        self.last_action: Optional[Move] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> Move:
        self.deadline = time.perf_counter() + self.internal_time_budget
        self._ensure_map(map_state)
        self._update_vision_radius(
            map_state, my_position, enemy_position
        )
        self._learn_from_visible_transition(
            my_position, enemy_position, step_number
        )
        self._update_belief(
            map_state, my_position, enemy_position, step_number
        )

        self.current_step = step_number
        self.ghost_history.append(my_position)
        if enemy_position is not None:
            self.pacman_history.append(enemy_position)
            move = self._choose_clear_action(
                my_position, enemy_position
            )
        else:
            move = self._choose_blind_action(my_position)

        move = self._validate_or_fallback(move, my_position)

        if enemy_position is not None:
            self.last_observed_enemy = enemy_position
            self.last_observed_step = step_number
        self.previous_my_position = my_position
        self.last_action = move
        return move

    # ------------------------------------------------------------------
    # Map preprocessing and exact game tables
    # ------------------------------------------------------------------

    def _ensure_map(self, map_state: np.ndarray) -> None:
        wall_map = (map_state == 1).astype(np.int8)
        signature = wall_map.tobytes()
        if (
            self.wall_map is not None
            and self.wall_map.shape == wall_map.shape
            and self.map_signature == signature
        ):
            return

        self.wall_map = wall_map
        self.map_signature = signature
        self.height, self.width = wall_map.shape
        self.cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if wall_map[r, c] == 0
        ]
        self.index = {position: i for i, position in enumerate(self.cells)}

        n = len(self.cells)
        self.neighbors = [[] for _ in range(n)]
        self.ghost_actions = [[] for _ in range(n)]
        self.pacman_actions = [[] for _ in range(n)]

        for i, position in enumerate(self.cells):
            for move in CARDINAL_MOVES:
                endpoint = self.index.get(
                    self._move_position(position, move)
                )
                if endpoint is not None:
                    self.neighbors[i].append((move, endpoint))

            self.ghost_actions[i] = list(self.neighbors[i])
            self.ghost_actions[i].append((Move.STAY, i))

            self.pacman_actions[i].append((Move.STAY, 1, i))
            for move in CARDINAL_MOVES:
                current = position
                for steps in range(1, self.pacman_speed + 1):
                    current = self._move_position(current, move)
                    endpoint = self.index.get(current)
                    if endpoint is None:
                        break
                    self.pacman_actions[i].append(
                        (move, steps, endpoint)
                    )

        self.degrees = np.asarray(
            [len(items) for items in self.neighbors], dtype=np.int8
        )
        self.graph_dist = self._all_pairs_cell_distances()
        turn_dist = self._all_pairs_turn_distances()
        self.capture_turns = self._build_capture_turn_distances(
            turn_dist
        )
        (
            self.capture_rank,
            self.survival_value,
        ) = self._build_exact_game_tables()
        self.lag_path = ()
        self.lag_target = None
        self.lag_predicted_pacman = None
        self.lag_policy_quality = 0.50
        self.lag_policy_observations = 0
        self.lag_distinctive_matches = 0
        self.lag_probe_used = False
        self.belief = None

    def _all_pairs_cell_distances(self) -> np.ndarray:
        n = len(self.cells)
        distances = np.full((n, n), INF, dtype=np.int16)
        for source in range(n):
            distances[source, source] = 0
            queue = deque([source])
            while queue:
                current = queue.popleft()
                next_distance = int(distances[source, current]) + 1
                for _, neighbor in self.neighbors[current]:
                    if distances[source, neighbor] == INF:
                        distances[source, neighbor] = next_distance
                        queue.append(neighbor)
        return distances

    def _all_pairs_turn_distances(self) -> np.ndarray:
        n = len(self.cells)
        distances = np.full((n, n), INF, dtype=np.int16)
        for source in range(n):
            distances[source, source] = 0
            queue = deque([source])
            while queue:
                current = queue.popleft()
                next_distance = int(distances[source, current]) + 1
                for move, _, endpoint in self.pacman_actions[current]:
                    if (
                        move != Move.STAY
                        and distances[source, endpoint] == INF
                    ):
                        distances[source, endpoint] = next_distance
                        queue.append(endpoint)
        return distances

    def _build_capture_turn_distances(
        self, turn_dist: np.ndarray
    ) -> np.ndarray:
        n = len(self.cells)
        result = np.full((n, n), INF, dtype=np.int16)
        for ghost_index in range(n):
            capture_cells = [ghost_index]
            capture_cells.extend(
                endpoint
                for _, endpoint in self.neighbors[ghost_index]
            )
            result[:, ghost_index] = np.min(
                turn_dist[:, capture_cells], axis=1
            )
        return result

    def _build_exact_game_tables(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both sides of the simultaneous reachability game.

        ``capture_rank`` is Pacman's attractor rank. ``survival_value`` is the
        number of robust transitions Ghost can preserve before it is forced
        out of the non-capture set; disconnected invariant states receive 300.
        """
        n = len(self.cells)
        if n == 0:
            empty = np.empty((0, 0), dtype=np.int16)
            return empty, empty

        max_p = max(len(actions) for actions in self.pacman_actions)
        max_g = max(len(actions) for actions in self.ghost_actions)
        p_endpoints = np.zeros((n, max_p), dtype=np.int16)
        g_endpoints = np.zeros((n, max_g), dtype=np.int16)
        p_mask = np.zeros((n, max_p), dtype=bool)
        g_mask = np.zeros((n, max_g), dtype=bool)

        for index, actions in enumerate(self.pacman_actions):
            count = len(actions)
            p_endpoints[index, :count] = [
                endpoint for _, _, endpoint in actions
            ]
            p_mask[index, :count] = True
        for index, actions in enumerate(self.ghost_actions):
            count = len(actions)
            g_endpoints[index, :count] = [
                endpoint for _, endpoint in actions
            ]
            g_mask[index, :count] = True

        coordinates = np.asarray(self.cells, dtype=np.int16)
        capture = (
            np.abs(
                coordinates[:, None, :] - coordinates[None, :, :]
            ).sum(axis=2)
            < 2
        )

        winning = capture.copy()
        capture_rank = np.full((n, n), -1, dtype=np.int16)
        capture_rank[capture] = 0
        for turn in range(1, 201):
            safe_endpoint = np.all(
                winning[:, g_endpoints] | ~g_mask[None, :, :],
                axis=2,
            )
            new_winning = np.any(
                safe_endpoint[p_endpoints, :]
                & p_mask[:, :, None],
                axis=1,
            )
            added = new_winning & ~winning
            if not bool(np.any(added)):
                break
            capture_rank[added] = turn
            winning |= added

        active = ~capture
        survival = np.zeros((n, n), dtype=np.int16)
        for turn in range(1, 201):
            robust_endpoint = np.all(
                active[p_endpoints, :] | ~p_mask[:, :, None],
                axis=1,
            )
            keep = np.any(
                robust_endpoint[:, g_endpoints]
                & g_mask[None, :, :],
                axis=2,
            )
            new_active = active & keep
            removed = active & ~new_active
            survival[removed] = turn
            active = new_active
            if not bool(np.any(removed)):
                break
        survival[active] = 300
        return capture_rank, survival

    # ------------------------------------------------------------------
    # Online Pacman model
    # ------------------------------------------------------------------

    def _learn_from_visible_transition(
        self,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> None:
        if (
            enemy_position is None
            or self.last_observed_enemy is None
            or self.last_observed_step is None
            or self.previous_my_position is None
            or step_number != self.last_observed_step + 1
        ):
            return

        old_pacman = self.index.get(self.last_observed_enemy)
        old_ghost = self.index.get(self.previous_my_position)
        actual_endpoint = self.index.get(enemy_position)
        if (
            old_pacman is None
            or old_ghost is None
            or actual_endpoint is None
        ):
            return

        if self.lag_predicted_pacman is not None:
            fresh_prediction, _, _ = self._lag_predict(
                old_pacman, old_ghost, (), None
            )
            likelihood = (
                1.0
                if actual_endpoint == self.lag_predicted_pacman
                else 0.02
            )
            if (
                actual_endpoint == self.lag_predicted_pacman
                and actual_endpoint != fresh_prediction
            ):
                self.lag_distinctive_matches += 1
            self.lag_policy_quality = (
                0.80 * self.lag_policy_quality
                + 0.20 * likelihood
            )
            self.lag_policy_observations += 1
            if likelihood < 0.5:
                # A mismatching observed transition invalidates the cached
                # route state, but not the entire policy hypothesis.
                self.lag_path = ()
                self.lag_target = None
            self.lag_predicted_pacman = None

        endpoints, components = self._pacman_policy_components(
            old_pacman, old_ghost, self.last_pacman_delta
        )
        try:
            actual_action = endpoints.index(actual_endpoint)
        except ValueError:
            return

        likelihoods = components[:, actual_action]
        self.behaviour_quality = (
            0.90 * self.behaviour_quality + 0.10 * likelihoods
        )
        self.behaviour_quality = np.clip(
            self.behaviour_quality, 0.02, 1.0
        )
        self.last_pacman_delta = (
            enemy_position[0] - self.last_observed_enemy[0],
            enemy_position[1] - self.last_observed_enemy[1],
        )

    def _pacman_distribution(
        self,
        pacman_index: int,
        ghost_index: int,
        inertia_delta: Optional[Tuple[int, int]],
    ) -> Tuple[List[int], np.ndarray]:
        endpoints, components = self._pacman_policy_components(
            pacman_index, ghost_index, inertia_delta
        )
        posterior = self.behaviour_prior * self.behaviour_quality
        total = float(posterior.sum())
        if total <= 0.0:
            posterior = self.behaviour_prior.copy()
        else:
            posterior /= total

        model_weights = 0.58 * self.behaviour_prior + 0.42 * posterior
        probabilities = np.dot(model_weights, components)
        uniform = np.full(
            len(endpoints), 1.0 / len(endpoints), dtype=float
        )
        probabilities = 0.88 * probabilities + 0.12 * uniform
        probabilities /= probabilities.sum()
        return endpoints, probabilities

    def _pacman_policy_components(
        self,
        pacman_index: int,
        ghost_index: int,
        inertia_delta: Optional[Tuple[int, int]],
    ) -> Tuple[List[int], np.ndarray]:
        actions = self.pacman_actions[pacman_index]
        endpoints = [endpoint for _, _, endpoint in actions]
        count = len(endpoints)
        uniform = np.full(count, 1.0 / count, dtype=float)

        pursuit_logits = np.zeros(count, dtype=float)
        for k, endpoint in enumerate(endpoints):
            turns = self._finite_distance(
                int(self.capture_turns[endpoint, ghost_index])
            )
            distance = self._finite_distance(
                int(self.graph_dist[endpoint, ghost_index])
            )
            pursuit_logits[k] = -1.35 * turns - 0.07 * distance
        pursuit = self._softmax(pursuit_logits)

        inertia = uniform.copy()
        if inertia_delta is not None:
            old_position = self.cells[pacman_index]
            predicted = (
                old_position[0] + inertia_delta[0],
                old_position[1] + inertia_delta[1],
            )
            predicted_index = self.index.get(predicted)
            if predicted_index in endpoints:
                inertia.fill(0.28 / max(1, count - 1))
                inertia[endpoints.index(predicted_index)] = 0.72
                if count == 1:
                    inertia[0] = 1.0

        rank_logits = np.zeros(count, dtype=float)
        ghost_replies = [
            endpoint
            for _, endpoint in self.ghost_actions[ghost_index]
        ]
        for k, endpoint in enumerate(endpoints):
            ranks = [
                int(self.capture_rank[endpoint, reply])
                for reply in ghost_replies
            ]
            worst_rank = (
                40 if any(rank < 0 for rank in ranks) else max(ranks)
            )
            rank_logits[k] = -1.10 * worst_rank
        rank_pursuit = self._softmax(rank_logits)

        return endpoints, np.vstack(
            [uniform, pursuit, inertia, rank_pursuit]
        )

    def _lag_bfs_path(
        self, start: int, goal: int
    ) -> Tuple[Move, ...]:
        distance = int(self.graph_dist[start, goal])
        if distance <= 0 or distance >= int(INF):
            return ()

        path: List[Move] = []
        current = start
        while current != goal:
            current_distance = int(
                self.graph_dist[current, goal]
            )
            selected = None
            for move in CARDINAL_MOVES:
                next_index = self.index.get(
                    self._move_position(
                        self.cells[current], move
                    )
                )
                if (
                    next_index is not None
                    and int(
                        self.graph_dist[next_index, goal]
                    )
                    == current_distance - 1
                ):
                    selected = (move, next_index)
                    break
            if selected is None:
                return ()
            move, current = selected
            path.append(move)
        return tuple(path)

    def _lag_predict(
        self,
        pacman_index: int,
        ghost_index: int,
        path: Tuple[Move, ...],
        target: Optional[int],
    ) -> Tuple[int, Tuple[Move, ...], Optional[int]]:
        target_moved = True
        if target is not None:
            old_target = self.cells[target]
            ghost_pos = self.cells[ghost_index]
            target_moved = (
                abs(old_target[0] - ghost_pos[0])
                + abs(old_target[1] - ghost_pos[1])
                >= 2
            )

        if target is None or target_moved or not path:
            path = self._lag_bfs_path(
                pacman_index, ghost_index
            )
            target = ghost_index

        if not path:
            return pacman_index, (), target

        first_move = path[0]
        straight_run = 0
        for move in path:
            if move != first_move:
                break
            straight_run += 1

        steps = min(
            self.pacman_speed, max(1, straight_run)
        )
        current = pacman_index
        actual_steps = 0
        for _ in range(steps):
            next_index = self.index.get(
                self._move_position(
                    self.cells[current], first_move
                )
            )
            if next_index is None:
                break
            current = next_index
            actual_steps += 1

        return current, path[actual_steps:], target

    def _lag_response_move(
        self, pacman_index: int, ghost_index: int
    ) -> Tuple[Move, int, Dict[Move, int]]:
        predicted, path, target = self._lag_predict(
            pacman_index,
            ghost_index,
            self.lag_path,
            self.lag_target,
        )
        self.lag_path = path
        self.lag_target = target
        self.lag_predicted_pacman = predicted

        memo: Dict[
            Tuple[
                int,
                int,
                Tuple[Move, ...],
                Optional[int],
            ],
            int,
        ] = {}
        visiting = set()

        def longest(
            p_index: int,
            g_index: int,
            cached_path: Tuple[Move, ...],
            cached_target: Optional[int],
            depth: int,
        ) -> int:
            state = (
                p_index,
                g_index,
                cached_path,
                cached_target,
            )
            if state in visiting:
                return 300
            if time.perf_counter() >= self.deadline - 0.035:
                return int(self.survival_value[p_index, g_index])
            cached = memo.get(state)
            if cached is not None:
                return cached
            if depth >= 72:
                return int(
                    self.survival_value[p_index, g_index]
                )

            p_next, next_path, next_target = (
                self._lag_predict(
                    p_index,
                    g_index,
                    cached_path,
                    cached_target,
                )
            )
            visiting.add(state)
            best = 0
            for _, g_next in self.ghost_actions[g_index]:
                if self._is_capture_index(p_next, g_next):
                    value = 0
                else:
                    value = 1 + longest(
                        p_next,
                        g_next,
                        next_path,
                        next_target,
                        depth + 1,
                    )
                if value > best:
                    best = value
                if (
                    best >= 300
                    or time.perf_counter() >= self.deadline - 0.035
                ):
                    break
            visiting.remove(state)
            memo[state] = best
            return best

        action_values: Dict[Move, int] = {}
        best_move = Move.STAY
        best_key = (-1, -1, -1, -float("inf"))
        for move, ghost_next in self.ghost_actions[ghost_index]:
            if self._is_capture_index(predicted, ghost_next):
                value = 0
            else:
                value = 1 + longest(
                    predicted,
                    ghost_next,
                    path,
                    target,
                    1,
                )
            action_values[move] = value
            distance = self._finite_distance(
                int(self.graph_dist[predicted, ghost_next])
            )
            key = (
                value,
                distance,
                int(self.degrees[ghost_next]),
                self._continuity_bonus(move),
            )
            if key > best_key:
                best_key = key
                best_move = move

        return best_move, int(best_key[0]), action_values

    # ------------------------------------------------------------------
    # Clear evasion
    # ------------------------------------------------------------------

    def _choose_clear_action(
        self, my_position: Position, enemy_position: Position
    ) -> Move:
        ghost_index = self.index.get(my_position)
        pacman_index = self.index.get(enemy_position)
        if ghost_index is None or pacman_index is None:
            return Move.STAY

        pacman_endpoints, probabilities = self._pacman_distribution(
            pacman_index, ghost_index, self.last_pacman_delta
        )
        lag_move, lag_value, lag_action_values = (
            self._lag_response_move(
                pacman_index, ghost_index
            )
        )

        # A detected remote cycle is exploitable information, not a special
        # case for a named opponent.  Staying prevents the Ghost from walking
        # into a seeker that has already trapped itself in a deterministic
        # oscillation.  The safety checks keep this rule neutral.
        if self._safe_to_camp(
            pacman_index, ghost_index, pacman_endpoints, probabilities
        ):
            return Move.STAY

        legacy_move = self._legacy_race_move(
            my_position, enemy_position
        )
        best_score = -float("inf")
        best_move = Move.STAY
        candidate_data = []

        for move, endpoint in self.ghost_actions[ghost_index]:
            survival_values = np.asarray(
                [
                    int(self.survival_value[pac, endpoint])
                    for pac in pacman_endpoints
                ],
                dtype=float,
            )
            robust_value = float(np.min(survival_values))
            expected_value = float(
                np.dot(probabilities, survival_values)
            )
            lower_tail = self._weighted_quantile(
                survival_values, probabilities, 0.18
            )

            capture_mask = np.asarray(
                [
                    self._is_capture_index(pac, endpoint)
                    for pac in pacman_endpoints
                ],
                dtype=bool,
            )
            capture_probability = float(
                probabilities[capture_mask].sum()
            )

            distances = np.asarray(
                [
                    self._finite_distance(
                        int(self.graph_dist[pac, endpoint])
                    )
                    for pac in pacman_endpoints
                ],
                dtype=float,
            )
            expected_distance = float(
                np.dot(probabilities, distances)
            )
            danger_distance = self._weighted_quantile(
                distances, probabilities, 0.18
            )

            score = 128.0 * robust_value
            score += 20.0 * lower_tail
            score += 8.0 * expected_value
            score += 12.0 * danger_distance
            score += 3.0 * expected_distance
            score += 4.0 * int(self.degrees[endpoint])
            score -= 15000.0 * capture_probability

            if move == Move.STAY:
                score -= 2.0
            score += self._continuity_bonus(move)
            candidate_data.append(
                (
                    move,
                    robust_value,
                    capture_probability,
                    score,
                )
            )

            if score > best_score + 1e-9:
                best_score = score
                best_move = move
            elif (
                abs(score - best_score) <= 1e-9
                and self._ghost_move_key(move)
                < self._ghost_move_key(best_move)
            ):
                best_move = move

        # Group 16's first submission had an unusually strong global
        # time-to-node race planner: it lengthened several games more than
        # shallow minimax agents did.  Retain that move whenever the exact
        # table says it is close to the safest one-step choice and the learned
        # model assigns no immediate capture risk.  This makes the old planner
        # an exploitation layer, while exact survival remains its guardrail.
        if legacy_move is not None:
            legacy = next(
                (
                    item
                    for item in candidate_data
                    if item[0] == legacy_move
                ),
                None,
            )
            if legacy is not None and candidate_data:
                best_robust = max(item[1] for item in candidate_data)
                lag = next(
                    (
                        item
                        for item in candidate_data
                        if item[0] == lag_move
                    ),
                    None,
                )
                # Normal cached-route exploitation still requires an observed
                # transition that distinguishes the lag model from fresh
                # replanning.  In addition, permit at most one conservative
                # late probe when the history is long and highly consistent,
                # the predicted benefit is very large, and the broader Pacman
                # model is not dominated by ordinary shortest-pursuit evidence.
                #
                # The probe may spend at most one exact survival turn.  Because
                # such a probe is intrinsically ambiguous, the first eligible
                # opportunity is marked as considered even when rejected.  This
                # prevents a skipped weak state from enabling a less-audited
                # probe later in the same game.
                pursuit_quality = float(self.behaviour_quality[1])
                rank_quality = float(self.behaviour_quality[3])
                conservative_family = (
                    pursuit_quality <= 0.665
                    and rank_quality >= 0.615
                )
                base_late_probe = (
                    not self.lag_probe_used
                    and self.lag_policy_observations >= 10
                    and self.lag_policy_quality >= 0.90
                    and lag is not None
                    and lag[2] <= legacy[2] + 1e-12
                    and lag[1] >= legacy[1] - 1.0
                    and lag_value
                    >= lag_action_values.get(legacy_move, 0) + 24
                )
                late_probe = base_late_probe and conservative_family
                if base_late_probe:
                    self.lag_probe_used = True

                credible_lag = (
                    late_probe
                    or (
                        self.lag_distinctive_matches > 0
                        and self.lag_policy_quality >= 0.55
                    )
                )
                strong_lag_evidence = (
                    self.lag_distinctive_matches > 0
                    and self.lag_policy_quality >= 0.60
                )
                if (
                    lag is not None
                    and credible_lag
                    # The learned exploit may never lower the exact
                    # one-step survival floor relative to the legacy move.
                    # This guard prevents a statistically credible model from
                    # sacrificing adversarial safety when the opponent changes.
                    and (
                        lag[1] >= legacy[1]
                        or (late_probe and lag[1] >= legacy[1] - 1.0)
                    )
                    and (
                        strong_lag_evidence
                        or lag[2] < 0.35
                    )
                    and (
                        (
                            strong_lag_evidence
                            and lag_value
                            > lag_action_values.get(
                                legacy_move, 0
                            )
                        )
                        or late_probe
                        or (
                            self.lag_policy_observations > 0
                            and self.lag_distinctive_matches > 0
                            and self.lag_policy_quality >= 0.55
                            and lag_value
                            >= lag_action_values.get(
                                legacy_move, 0
                            )
                        )
                    )
                ):
                    return lag_move
                recent_reversal = (
                    self.current_step <= 5
                    and len(self.pacman_history) >= 3
                    and self.pacman_history[-1]
                    == self.pacman_history[-3]
                )
                if (
                    recent_reversal
                    and best_robust >= legacy[1] + 1.0
                    and best_score >= legacy[3] + 100.0
                ):
                    return best_move
                if (
                    legacy[2] < 0.90
                    and legacy[1] >= best_robust - 4.0
                ):
                    return legacy_move

        return best_move

    def _legacy_race_move(
        self, my_position: Position, enemy_position: Position
    ) -> Optional[Move]:
        """
        Global arrival-time target selection from group 16's old Ghost.

        The original idea is preserved, but all formerly unsafe dictionary
        accesses and empty-neighbour minima are guarded.  Unknown cells are
        already reconstructed as open from the always-visible wall map, so
        the same routine also remains structurally valid after a sighting in
        Blind mode.
        """
        direction_order = (
            (1, 0),
            (0, -1),
            (-1, 0),
            (0, 1),
        )

        def distance_map(
            start: Position, speed: int
        ) -> Dict[Position, Tuple[int, Position]]:
            frontier = deque([start])
            result = {start: (0, start)}
            while frontier:
                node = frontier.popleft()
                for dr, dc in direction_order:
                    for stride in range(1, speed + 1):
                        endpoint = (
                            node[0] + stride * dr,
                            node[1] + stride * dc,
                        )
                        if (
                            endpoint not in self.index
                            or endpoint in result
                        ):
                            break
                        result[endpoint] = (
                            result[node][0] + 1, node
                        )
                        frontier.append(endpoint)
            return result

        enemy_bfs = distance_map(
            enemy_position, self.pacman_speed
        )
        my_bfs = distance_map(my_position, 1)

        def neighbor_values(
            position: Position,
        ) -> List[int]:
            values = []
            for dr, dc in (
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
            ):
                item = enemy_bfs.get(
                    (position[0] + dr, position[1] + dc)
                )
                if item is not None:
                    values.append(item[0])
            return values

        current_neighbors = neighbor_values(my_position)
        current_reference = (
            min(current_neighbors)
            if current_neighbors
            else enemy_bfs.get(my_position, (40, my_position))[0]
        )

        best_node: Optional[Position] = None
        best_value: Optional[Tuple[int, int]] = None
        first_turn_multiplier = 0.9 if self.current_step <= 1 else 0.5

        for candidate, (my_time, _) in my_bfs.items():
            trace = candidate
            reachable = True
            while my_bfs[trace][0] != 0:
                enemy_item = enemy_bfs.get(trace)
                if (
                    enemy_item is not None
                    and my_bfs[trace][0] >= enemy_item[0]
                ):
                    reachable = False
                    break
                trace = my_bfs[trace][1]
            if not reachable:
                continue

            neighbors = neighbor_values(candidate)
            if not neighbors:
                best_node = candidate
                break

            spare = min(neighbors) - my_time
            if spare > 0:
                turn_in = self._legacy_time_to_turn(
                    candidate, my_bfs
                )
                if turn_in < 0 or spare < turn_in:
                    spare = 0
                else:
                    spare = math.floor(
                        spare * first_turn_multiplier
                    )

            value = min(neighbors) - current_reference + spare
            value_pair = (value, spare)
            if (
                best_node is None
                or best_value is None
                or value > best_value[0]
                or (
                    value == best_value[0]
                    and self._legacy_estimated_turns(
                        candidate, enemy_position
                    )
                    > self._legacy_estimated_turns(
                        best_node, enemy_position
                    )
                )
            ):
                best_node = candidate
                best_value = value_pair

        if best_node is None or best_node == my_position:
            return Move.STAY

        while (
            best_node in my_bfs
            and my_bfs[best_node][1] != my_position
        ):
            best_node = my_bfs[best_node][1]

        delta = (
            best_node[0] - my_position[0],
            best_node[1] - my_position[1],
        )
        for move in (
            Move.UP,
            Move.LEFT,
            Move.RIGHT,
            Move.DOWN,
        ):
            if move.value == delta:
                return move
        return Move.STAY

    def _legacy_time_to_turn(
        self,
        position: Position,
        distance_map: Dict[Position, Tuple[int, Position]],
    ) -> int:
        parent = distance_map[position][1]
        movement = (
            position[0] - parent[0],
            position[1] - parent[1],
        )
        if movement == (0, 0):
            return 0

        counter = 0
        current = position
        while current in distance_map:
            perpendicular = (movement[1], movement[0])
            if (
                (
                    current[0] + perpendicular[0],
                    current[1] + perpendicular[1],
                )
                in distance_map
                or (
                    current[0] - perpendicular[0],
                    current[1] - perpendicular[1],
                )
                in distance_map
            ):
                return counter
            current = (
                current[0] + movement[0],
                current[1] + movement[1],
            )
            counter += 1
        return -1

    def _legacy_estimated_turns(
        self, start: Position, end: Position
    ) -> int:
        horizontal = abs(start[1] - end[1])
        vertical = abs(start[0] - end[0])
        if vertical == 0:
            return int(math.ceil(horizontal / 2))
        if horizontal == 0:
            return int(math.ceil(vertical / 2))
        return horizontal + vertical

    def _safe_to_camp(
        self,
        pacman_index: int,
        ghost_index: int,
        pacman_endpoints: Sequence[int],
        probabilities: np.ndarray,
    ) -> bool:
        if len(self.pacman_history) < 6:
            return False

        recent = list(self.pacman_history)[-6:]
        two_cycle = (
            recent[-1] == recent[-3] == recent[-5]
            and recent[-2] == recent[-4]
        )
        confined = len(set(recent)) <= 3
        if not (two_cycle or confined):
            return False

        distance = self._finite_distance(
            int(self.graph_dist[pacman_index, ghost_index])
        )
        if distance < 7:
            return False

        values = [
            int(self.survival_value[endpoint, ghost_index])
            for endpoint in pacman_endpoints
        ]
        if min(values) < 3:
            return False

        capture_probability = sum(
            float(probability)
            for endpoint, probability in zip(
                pacman_endpoints, probabilities
            )
            if self._is_capture_index(endpoint, ghost_index)
        )
        return capture_probability <= 1e-12

    # ------------------------------------------------------------------
    # Blind belief update and evasion
    # ------------------------------------------------------------------

    def _update_belief(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
        step_number: int,
    ) -> None:
        n = len(self.cells)
        if enemy_position is not None:
            self.belief = np.zeros(n, dtype=float)
            enemy_index = self.index.get(enemy_position)
            if enemy_index is not None:
                self.belief[enemy_index] = 1.0
            return

        if self.belief is None or float(self.belief.sum()) <= 0.0:
            self.belief = self._initial_hidden_belief(
                map_state, my_position
            )
        else:
            reference = (
                self.previous_my_position
                if self.previous_my_position is not None
                else my_position
            )
            ghost_index = self.index.get(reference)
            self.belief = self._propagate_pacman_belief(
                self.belief, ghost_index
            )

        # Every currently visible empty cell is a negative observation.
        for i, position in enumerate(self.cells):
            if map_state[position] == 0:
                self.belief[i] = 0.0

        total = float(self.belief.sum())
        if total <= 1e-15:
            self.belief = self._initial_hidden_belief(
                map_state, my_position
            )
        else:
            self.belief /= total

    def _initial_hidden_belief(
        self, map_state: np.ndarray, my_position: Position
    ) -> np.ndarray:
        belief = np.zeros(len(self.cells), dtype=float)
        ghost_index = self.index.get(my_position)
        candidates = [
            i
            for i, position in enumerate(self.cells)
            if map_state[position] == -1
            and (
                ghost_index is None
                or int(self.graph_dist[ghost_index, i]) < int(INF)
            )
        ]
        if not candidates:
            candidates = [
                i
                for i in range(len(self.cells))
                if ghost_index is None
                or int(self.graph_dist[ghost_index, i]) < int(INF)
            ]
        if not candidates:
            return np.full(
                len(self.cells), 1.0 / max(1, len(self.cells))
            )

        my_row = my_position[0]
        for index in candidates:
            row, _ = self.cells[index]
            # Reset modes place Pacman below Ghost; retain a broad floor so
            # the prior is useful without becoming a hard-coded start.
            vertical = math.exp(
                max(-1.0, min(1.0, 0.07 * (row - my_row)))
            )
            mobility = 0.90 + 0.05 * int(self.degrees[index])
            belief[index] = vertical * mobility

        belief /= belief.sum()
        return belief

    def _propagate_pacman_belief(
        self,
        belief: np.ndarray,
        ghost_index: Optional[int],
    ) -> np.ndarray:
        propagated = np.zeros_like(belief)
        if ghost_index is None:
            return belief.copy()

        for pacman_index in np.flatnonzero(belief > 1e-12):
            endpoints, probabilities = self._pacman_distribution(
                int(pacman_index),
                ghost_index,
                self.last_pacman_delta,
            )
            mass = float(belief[pacman_index])
            for endpoint, probability in zip(
                endpoints, probabilities
            ):
                propagated[endpoint] += mass * float(probability)

        total = float(propagated.sum())
        if total > 0.0:
            propagated /= total
        return propagated

    def _choose_blind_action(self, my_position: Position) -> Move:
        ghost_index = self.index.get(my_position)
        if ghost_index is None or self.belief is None:
            return Move.STAY

        next_belief = self._propagate_pacman_belief(
            self.belief, ghost_index
        )
        if float(next_belief.sum()) <= 0.0:
            next_belief = self.belief.copy()

        support = np.flatnonzero(next_belief > 1e-12)
        support_probabilities = next_belief[support]
        support_probabilities /= support_probabilities.sum()

        best_score = -float("inf")
        best_move = Move.STAY
        for move, endpoint in self.ghost_actions[ghost_index]:
            values = self.survival_value[support, endpoint].astype(float)
            distances = self.graph_dist[support, endpoint].astype(float)
            distances = np.where(
                distances >= int(INF), 40.0, distances
            )

            expected_value = float(
                np.dot(support_probabilities, values)
            )
            lower_value = self._weighted_quantile(
                values, support_probabilities, 0.12
            )
            expected_distance = float(
                np.dot(support_probabilities, distances)
            )
            danger_distance = self._weighted_quantile(
                distances, support_probabilities, 0.12
            )

            capture_mask = np.asarray(
                [
                    self._is_capture_index(
                        int(pacman), endpoint
                    )
                    for pacman in support
                ],
                dtype=bool,
            )
            capture_probability = float(
                support_probabilities[capture_mask].sum()
            )
            seen_probability = self._visibility_probability(
                endpoint, support, support_probabilities
            )

            score = 54.0 * lower_value
            score += 11.0 * expected_value
            score += 15.0 * danger_distance
            score += 3.0 * expected_distance
            score += 5.0 * int(self.degrees[endpoint])
            score -= 16000.0 * capture_probability
            score -= 34.0 * seen_probability
            score += self._continuity_bonus(move)
            if move == Move.STAY:
                score -= 1.5

            if score > best_score + 1e-9:
                best_score = score
                best_move = move
            elif (
                abs(score - best_score) <= 1e-9
                and self._ghost_move_key(move)
                < self._ghost_move_key(best_move)
            ):
                best_move = move

        return best_move

    def _visibility_probability(
        self,
        ghost_index: int,
        pacman_indices: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        ghost = self.cells[ghost_index]
        visible_probability = 0.0
        for pacman_index, probability in zip(
            pacman_indices, probabilities
        ):
            pacman = self.cells[int(pacman_index)]
            if (
                pacman[0] != ghost[0]
                and pacman[1] != ghost[1]
            ):
                continue
            distance = (
                abs(pacman[0] - ghost[0])
                + abs(pacman[1] - ghost[1])
            )
            if distance > self.vision_radius_estimate:
                continue
            if int(self.graph_dist[int(pacman_index), ghost_index]) == distance:
                visible_probability += float(probability)
        return visible_probability

    def _update_vision_radius(
        self,
        map_state: np.ndarray,
        my_position: Position,
        enemy_position: Optional[Position],
    ) -> None:
        if enemy_position is not None and not np.any(map_state == -1):
            return

        candidates = []
        for move in CARDINAL_MOVES:
            current = my_position
            visible_open = 0
            while True:
                current = self._move_position(current, move)
                if not self._in_bounds(current):
                    break
                value = int(map_state[current])
                if value == 1:
                    break
                if value == -1:
                    candidates.append(visible_open)
                    break
                visible_open += 1
        positive = [value for value in candidates if value > 0]
        if positive:
            estimate = int(round(float(np.median(positive))))
            self.vision_radius_estimate = max(1, min(10, estimate))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _validate_or_fallback(
        self, move: Optional[Move], my_position: Position
    ) -> Move:
        ghost_index = self.index.get(my_position)
        if ghost_index is None:
            return Move.STAY
        legal = {
            action for action, _ in self.ghost_actions[ghost_index]
        }
        if move in legal:
            return move

        candidates = [
            (int(self.degrees[endpoint]), action)
            for action, endpoint in self.ghost_actions[ghost_index]
            if action != Move.STAY
        ]
        if not candidates:
            return Move.STAY
        candidates.sort(
            key=lambda item: (
                -item[0], self._ghost_move_key(item[1])
            )
        )
        return candidates[0][1]

    def _continuity_bonus(self, move: Move) -> float:
        if self.last_action is None:
            return 0.0
        if move == self.last_action:
            return 1.5
        opposite = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
        }
        if move == opposite.get(self.last_action):
            return -2.0
        return 0.0

    def _ghost_move_key(self, move: Move) -> int:
        order = {
            Move.UP: 0,
            Move.LEFT: 1,
            Move.RIGHT: 2,
            Move.DOWN: 3,
            Move.STAY: 4,
        }
        return order[move]

    def _weighted_quantile(
        self,
        values: np.ndarray,
        probabilities: np.ndarray,
        quantile: float,
    ) -> float:
        order = np.argsort(values)
        cumulative = np.cumsum(probabilities[order])
        selected = int(
            np.searchsorted(cumulative, quantile, side="left")
        )
        selected = min(selected, len(order) - 1)
        return float(values[order[selected]])

    def _is_capture_index(
        self, pacman_index: int, ghost_index: int
    ) -> bool:
        pacman = self.cells[pacman_index]
        ghost = self.cells[ghost_index]
        return (
            abs(pacman[0] - ghost[0])
            + abs(pacman[1] - ghost[1])
            < 2
        )

    def _finite_distance(self, value: int) -> int:
        return 40 if value >= int(INF) else value

    def _move_position(
        self, position: Position, move: Move
    ) -> Position:
        dr, dc = move.value
        return (position[0] + dr, position[1] + dc)

    def _in_bounds(self, position: Position) -> bool:
        return (
            0 <= position[0] < self.height
            and 0 <= position[1] < self.width
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        shifted = np.clip(logits - np.max(logits), -30.0, 30.0)
        exponents = np.exp(shifted)
        total = float(exponents.sum())
        if total <= 0.0:
            return np.full(
                len(logits), 1.0 / len(logits), dtype=float
            )
        return exponents / total