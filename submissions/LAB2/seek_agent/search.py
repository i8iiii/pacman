"""Stateful, turn-aware exploration planning for the LAB2 seeker.

This module owns only SEARCHING behaviour.  The controller remains free to
interrupt it for chasing or investigating, while retaining this state for a
later return to searching.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

from environment import Move

from .routes import plan_area_route
from .spatial import (
    count_path_turns,
    minimum_turn_path,
    normalize_position,
    path_to_actions,
    traversable_neighbors,
)


PLANNING_TIME_LIMIT_SECONDS = 0.35
PACMAN_SPEED = 2
_UNREACHABLE_COST = (10 ** 6, 10 ** 6)


class SearchPhase(str, Enum):
    """The lifecycle of a single selected exploration area."""

    SELECT_PRIORITY = "select_priority"
    TRANSIT_TO_AREA = "transit_to_area"
    SWEEP_ACTIVE_AREA = "sweep_active_area"
    AREA_COMPLETE = "area_complete"


@dataclass(frozen=True)
class SearchDecision:
    """An immutable explanation of one SEARCHING decision."""

    phase: SearchPhase
    current_area_id: int | None
    target_area_id: int | None
    planned_area_order: tuple
    entry: tuple | None
    exit: tuple | None
    cells: tuple
    actions: tuple
    chosen_action: tuple
    required_cells: frozenset
    covered_cells: frozenset
    completed_area_ids: frozenset
    # Completion is an event so callers can observe AREA_COMPLETE without
    # forcing an otherwise useless STAY turn before the next transit begins.
    completed_this_step: int | None
    replan_reason: str | None
    planning_seconds: float
    exact: bool
    fallback: bool

    @property
    def route(self):
        """Alias used by diagnostics that call the cell sequence a route."""
        return self.cells


@dataclass(frozen=True)
class _AreaProfile:
    """Concrete local service route used by the global priority heuristic."""

    entry: tuple
    exit: tuple
    turns: int
    cell_steps: int
    complete: bool


class SearchPlanner:
    """Choose and sweep topology-derived areas without random movement.

    The planner deliberately locks an area from selection until its snapshot
    of information requirements has been observed.  Passing through another
    area on the way there is only transit; it never changes that area's state.
    """

    def __init__(self, pacman_speed=PACMAN_SPEED,
                 planning_limit_seconds=PLANNING_TIME_LIMIT_SECONDS):
        self.pacman_speed = max(1, int(pacman_speed))
        self.planning_limit_seconds = max(0.0, float(planning_limit_seconds))
        self.reset()

    def reset(self, analysis=None):
        """Discard all per-match state, optionally binding a known topology."""
        self._fingerprint = None if analysis is None else analysis.fingerprint
        self.phase = SearchPhase.SELECT_PRIORITY
        self.target_area_id = None
        self.planned_area_order = ()
        self.entry = None
        self.exit = None
        self.completed_area_ids = set()
        self.required_cells = frozenset()
        self._snapshot_step = None
        self._route_cells = ()
        self._route_actions = ()
        self._route_index = 0
        self._last_replan_reason = "new_match"

    def decide(self, topology, analysis, belief, position, visible_cells=(),
               step_number=0):
        """Return the next deterministic Search decision.

        ``belief`` must already have been updated with this turn's visible
        cells.  Supplying ``visible_cells`` separately keeps completion safe
        even for callers that update the belief immediately afterwards.
        """
        started_at = perf_counter()
        # Keep a small wall-clock reserve for post-planning route selection,
        # decision assembly, and diagnostics in the controller.
        deadline = started_at + max(0.0, self.planning_limit_seconds - 0.01)
        position = normalize_position(position)
        visible = frozenset(normalize_position(cell) for cell in visible_cells)
        step_number = int(step_number)
        self._ensure_topology(analysis)
        current_area = analysis.area_for(position)

        if current_area is None or not analysis.areas:
            return self._fallback_decision(
                current_area, position, started_at, "position_outside_areas",
            )

        replan_reason = None
        completed_this_step = None
        # Future requirements are deliberately resnapshotted on physical area
        # arrival, so no full future order can be an exact execution plan.
        exact = False
        fallback = False

        if self.phase == SearchPhase.SELECT_PRIORITY or self.target_area_id is None:
            replan_reason = self._last_replan_reason or "select_priority"
            exact, fallback = self._select_priority(
                topology, analysis, belief, position, visible, step_number,
                deadline,
            )

        if self.target_area_id is None:
            return self._fallback_decision(
                current_area, position, started_at, "no_reachable_area",
            )

        # Arrival is physical, not merely reaching a precomputed gateway goal.
        if self.phase == SearchPhase.TRANSIT_TO_AREA and current_area == self.target_area_id:
            self.entry = position
            self._start_sweep(
                topology, analysis, belief, position, visible, step_number,
                deadline,
            )

        if self.phase == SearchPhase.SWEEP_ACTIVE_AREA:
            covered = self._covered_required(belief, visible, step_number)
            if covered == self.required_cells:
                self.phase = SearchPhase.AREA_COMPLETE
                self.completed_area_ids.add(self.target_area_id)
                completed_this_step = self.target_area_id
                self._last_replan_reason = "area_complete"
                # Replan immediately: SEARCHING should not waste a turn at a
                # completed area merely to expose an internal transition.
                self._clear_active_area()
                exact, fallback = self._select_priority(
                    topology, analysis, belief, position, visible, step_number,
                    deadline,
                )
                replan_reason = "area_complete"
                current_area = analysis.area_for(position)
                if self.target_area_id is not None and current_area == self.target_area_id:
                    self.entry = position
                    self._start_sweep(
                        topology, analysis, belief, position, visible,
                        step_number, deadline,
                    )

        if self.phase == SearchPhase.TRANSIT_TO_AREA:
            cells = self._transit_route(topology, position)
        elif self.phase == SearchPhase.SWEEP_ACTIVE_AREA:
            cells = self._sweep_route(
                topology, analysis, position, deadline,
            )
        else:
            cells = (position,)

        actions = path_to_actions(cells, self.pacman_speed)
        chosen = actions[0] if actions else (Move.STAY, 1)
        if actions:
            self._route_index += int(chosen[1])
        covered = self._covered_required(belief, visible, step_number)
        return SearchDecision(
            phase=self.phase,
            current_area_id=current_area,
            target_area_id=self.target_area_id,
            planned_area_order=self.planned_area_order,
            entry=self.entry,
            exit=self.exit,
            cells=tuple(cells),
            actions=tuple(actions),
            chosen_action=chosen,
            required_cells=self.required_cells,
            covered_cells=covered,
            completed_area_ids=frozenset(self.completed_area_ids),
            completed_this_step=completed_this_step,
            replan_reason=replan_reason,
            planning_seconds=perf_counter() - started_at,
            exact=exact,
            fallback=fallback,
        )

    # A concise alias makes eventual controller integration read naturally.
    step = decide

    def _ensure_topology(self, analysis):
        if self._fingerprint != analysis.fingerprint:
            self.reset(analysis)
        self._analysis = analysis

    def _select_priority(self, topology, analysis, belief, position, visible,
                         step_number, deadline):
        remaining = tuple(
            area.area_id
            for area in analysis.areas
            if area.area_id not in self.completed_area_ids
        )
        if not remaining:
            # Areas can become required again as their observations age.  Start
            # a fresh coverage circuit, retaining no accidental transit state.
            self.completed_area_ids.clear()
            remaining = tuple(area.area_id for area in analysis.areas)
            self._last_replan_reason = "coverage_circuit_complete"

        # Reserve a small, hard portion of the turn budget for the local route
        # that may begin immediately when Pacman already occupies the target.
        ordering_deadline = min(
            deadline,
            perf_counter() + max(0.0, self.planning_limit_seconds - 0.06),
        )
        order, exact, fallback = self._global_order(
            topology, analysis, belief, position, visible, step_number,
            remaining, ordering_deadline,
        )
        if not order:
            self.target_area_id = None
            return exact, True

        self.target_area_id = order[0]
        self.planned_area_order = order
        self.entry = self._best_entry_path(
            topology, analysis, position, self.target_area_id,
        )[1]
        self.exit = None
        self.required_cells = frozenset()
        self._snapshot_step = None
        self._route_cells = ()
        self._route_actions = ()
        self._route_index = 0
        self.phase = SearchPhase.TRANSIT_TO_AREA
        return exact, fallback

    def _global_order(self, topology, analysis, belief, position, visible,
                      step_number, remaining, deadline):
        """Route-aware subset DP over concrete area-service endpoints.

        This is a route-aware heuristic, not an exact future execution plan:
        requirements are resnapshotted on arrival and the order is replanned
        after every completed area.  It uses stable concrete endpoints and
        measured routes only to choose the next locked target sensibly.
        """
        ids = tuple(sorted(remaining))
        weights = {
            area_id: len(belief.possible_positions_in_area(
                analysis.areas[area_id],
            ))
            for area_id in ids
        }
        profiles = {}
        initial_costs = {}
        incomplete_profile = False
        for profile_index, area_id in enumerate(ids):
            if perf_counter() >= deadline:
                return self._greedy_order(
                    ids, weights, initial_costs, {}, profiles,
                ), False, True
            area = analysis.areas[area_id]
            entry = self._priority_endpoint(analysis, area_id)
            required = self._snapshot_required(
                topology, area, belief, visible, step_number,
            )
            # ``plan_area_route`` is exhaustive.  Give every profile a fair,
            # small slice instead of letting the first area consume the entire
            # global planning window.
            profiles_left = len(ids) - profile_index
            profile_deadline = min(
                deadline,
                perf_counter() + min(
                    0.025,
                    max(0.0, (deadline - perf_counter()) / profiles_left),
                ),
            )
            route = plan_area_route(
                topology, area, entry, required_cells=required,
                pacman_speed=self.pacman_speed, deadline=profile_deadline,
            )
            profiles[area_id] = _AreaProfile(
                entry=entry,
                exit=route.exit,
                turns=route.turns,
                cell_steps=route.cell_steps,
                complete=route.complete,
            )
            incomplete_profile = incomplete_profile or not route.complete
            if perf_counter() >= deadline:
                return self._greedy_order(
                    ids, weights, initial_costs, {}, profiles,
                ), False, True
            initial_costs[area_id] = self._path_cost(minimum_turn_path(
                topology, position, entry, pacman_speed=self.pacman_speed,
            ))

        pair_costs = {}
        for source in ids:
            for target in ids:
                if source == target:
                    pair_costs[source, target] = (0, 0)
                    continue
                if perf_counter() >= deadline:
                    return self._greedy_order(
                        ids, weights, initial_costs, pair_costs, profiles,
                    ), False, True
                pair_costs[source, target] = self._path_cost(minimum_turn_path(
                    topology, profiles[source].exit, profiles[target].entry,
                    pacman_speed=self.pacman_speed,
                ))

        states = {}
        for area_id in ids:
            turns, cells = initial_costs[area_id]
            service = profiles[area_id]
            elapsed_turns = turns + service.turns
            states[1 << area_id, area_id] = (
                weights[area_id] * elapsed_turns,
                elapsed_turns,
                cells + service.cell_steps,
                (area_id,),
            )
        full_mask = sum(1 << area_id for area_id in ids)
        for cardinality in range(1, len(ids)):
            for (mask, last), state in tuple(states.items()):
                if mask.bit_count() != cardinality:
                    continue
                if perf_counter() >= deadline:
                    return self._greedy_order(
                        ids, weights, initial_costs, pair_costs, profiles,
                    ), False, True
                for target in ids:
                    if mask & (1 << target):
                        continue
                    move_turns, move_cells = pair_costs[last, target]
                    service = profiles[target]
                    elapsed_turns = state[1] + move_turns + service.turns
                    candidate = (
                        state[0] + weights[target] * elapsed_turns,
                        elapsed_turns,
                        state[2] + move_cells + service.cell_steps,
                        state[3] + (target,),
                    )
                    key = mask | (1 << target), target
                    if key not in states or candidate < states[key]:
                        states[key] = candidate
        best = min(state for (mask, _), state in states.items() if mask == full_mask)
        return best[3], False, incomplete_profile

    def _greedy_order(self, ids, weights, initial_costs, pair_costs, profiles):
        unvisited = set(ids)
        order = []
        previous = None
        while unvisited:
            def key(area_id):
                turns, cells = (
                    initial_costs.get(area_id, _UNREACHABLE_COST)
                    if previous is None
                    else pair_costs.get((previous, area_id), _UNREACHABLE_COST)
                )
                # Higher remaining belief is served first; travel time is the
                # deterministic tie breaker when the belief is equal.
                return (-weights[area_id], turns, cells, area_id)

            choice = min(unvisited, key=key)
            order.append(choice)
            unvisited.remove(choice)
            previous = choice
        return tuple(order)

    def _start_sweep(self, topology, analysis, belief, position, visible,
                     step_number, deadline):
        area = analysis.areas[self.target_area_id]
        self.required_cells = self._snapshot_required(
            topology, area, belief, visible, step_number,
        )
        self._snapshot_step = step_number
        self.exit = self._exit_toward_next(analysis, self.target_area_id)
        route = plan_area_route(
            topology,
            area,
            position,
            exit=self.exit,
            required_cells=self.required_cells,
            pacman_speed=self.pacman_speed,
            deadline=deadline,
        )
        self._route_cells = route.cells
        self._route_actions = route.actions
        self._route_index = 0
        if not route.complete and len(route.cells) < 2:
            self._route_cells = self._safe_required_route(
                topology, area, position,
            )
            self._route_actions = path_to_actions(
                self._route_cells, self.pacman_speed,
            )
        self.phase = SearchPhase.SWEEP_ACTIVE_AREA

    def _snapshot_required(self, topology, area, belief, visible, step_number):
        required = set(belief.never_observed_in_area(area))
        required.update(belief.possible_positions_in_area(area))
        possible = tuple(sorted(belief.possible_positions))
        ghost_distances = self._ghost_distance_map(topology, possible)
        for cell in area.cells:
            cell = normalize_position(cell)
            age = belief.last_observed_age(cell, step_number)
            if age is None:
                continue
            if age >= ghost_distances.get(cell, float("inf")):
                required.add(cell)
        # A current observation is already valid at the snapshot instant.
        required.intersection_update(normalize_position(cell) for cell in area.cells)
        return frozenset(required)

    @staticmethod
    def _ghost_distance_map(topology, possible):
        """One multi-source BFS gives all earliest Ghost arrival times."""
        distances = {normalize_position(source): 0 for source in possible}
        frontier = deque(sorted(distances))
        while frontier:
            current = frontier.popleft()
            for neighbor in traversable_neighbors(topology, current):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
        return distances

    def _covered_required(self, belief, visible, step_number):
        if self._snapshot_step is None:
            return frozenset()
        return frozenset(
            cell for cell in self.required_cells
            if cell in visible
            or belief.last_observed_step.get(cell, -1) >= self._snapshot_step
        )

    def _transit_route(self, topology, position):
        path, entry = self._best_entry_path(
            topology, self._analysis, position, self.target_area_id,
        )
        if path is None:
            return (position,)
        self.entry = entry
        return tuple(path)

    def _sweep_route(self, topology, analysis, position, deadline):
        if self._route_cells and self._route_index < len(self._route_cells):
            expected = self._route_cells[self._route_index]
            if position == expected:
                return self._route_cells[self._route_index:]
        # A chase interruption or altered action result invalidated the stored
        # suffix.  Rebuild only the locked local route; never choose a new area.
        area = analysis.areas[self.target_area_id]
        route = plan_area_route(
            topology, area, position, exit=self.exit,
            required_cells=self.required_cells,
            pacman_speed=self.pacman_speed, deadline=deadline,
        )
        self._route_cells = route.cells
        self._route_actions = route.actions
        self._route_index = 0
        if not route.complete and len(route.cells) < 2:
            self._route_cells = self._safe_required_route(
                topology, area, position,
            )
            self._route_actions = path_to_actions(
                self._route_cells, self.pacman_speed,
            )
        return self._route_cells

    def _safe_required_route(self, topology, area, position):
        """Deterministic one-goal fallback when local enumeration expires."""
        outstanding = sorted(self.required_cells)
        candidates = []
        for goal in outstanding:
            path = minimum_turn_path(
                topology, position, goal, pacman_speed=self.pacman_speed,
                allowed_cells=area.cells,
            )
            if path is not None:
                candidates.append((
                    count_path_turns(path, self.pacman_speed),
                    len(path) - 1,
                    tuple(path),
                ))
        return min(candidates)[2] if candidates else (position,)

    def _best_entry_path(self, topology, analysis, position, target_area_id):
        """Return a shortest route and its first physical cell in the target."""
        position = normalize_position(position)
        target = analysis.areas[target_area_id]
        if analysis.area_for(position) == target_area_id:
            return (position,), position
        candidates = set(target.viewpoints)
        for gateway in analysis.gateways:
            if gateway.area_a == target_area_id:
                candidates.update(first for first, _ in gateway.connections)
            elif gateway.area_b == target_area_id:
                candidates.update(second for _, second in gateway.connections)
        candidates.add(min(target.cells))
        best = None
        for candidate in sorted(normalize_position(cell) for cell in candidates):
            path = minimum_turn_path(
                topology, position, candidate, pacman_speed=self.pacman_speed,
            )
            if path is None:
                continue
            first_target_index = next(
                (index for index, cell in enumerate(path)
                 if analysis.area_for(cell) == target_area_id),
                None,
            )
            if first_target_index is None:
                continue
            path = tuple(path[:first_target_index + 1])
            key = (count_path_turns(path, self.pacman_speed), len(path) - 1, path)
            if best is None or key < best[0]:
                best = (key, path, path[-1])
        if best is None:
            return None, None
        return best[1], best[2]

    def _exit_toward_next(self, analysis, current_area_id):
        try:
            index = self.planned_area_order.index(current_area_id)
        except ValueError:
            return None
        if index + 1 >= len(self.planned_area_order):
            return None
        next_area_id = self.planned_area_order[index + 1]
        next_hop = self._next_area_hop(analysis, current_area_id, next_area_id)
        if next_hop is None:
            return None
        choices = []
        for gateway in analysis.gateways:
            if {gateway.area_a, gateway.area_b} != {current_area_id, next_hop}:
                continue
            for first, second in gateway.connections:
                choices.append(
                    first if gateway.area_a == current_area_id else second,
                )
        return min(choices) if choices else None

    @staticmethod
    def _next_area_hop(analysis, start, target):
        if start == target:
            return start
        frontier = [start]
        parents = {start: None}
        for current in frontier:
            if current == target:
                break
            for neighbor in analysis.areas[current].neighbors:
                if neighbor not in parents:
                    parents[neighbor] = current
                    frontier.append(neighbor)
        if target not in parents:
            return None
        current = target
        while parents[current] != start:
            current = parents[current]
            if current is None:
                return None
        return current

    @staticmethod
    def _priority_endpoint(analysis, area_id):
        """Choose a stable concrete gateway/viewpoint cell for DP costing."""
        area = analysis.areas[area_id]
        endpoints = set(normalize_position(cell) for cell in area.viewpoints)
        for gateway in analysis.gateways:
            if gateway.area_a == area_id:
                endpoints.update(first for first, _ in gateway.connections)
            elif gateway.area_b == area_id:
                endpoints.update(second for _, second in gateway.connections)
        return min(endpoints) if endpoints else min(area.cells)

    def _path_cost(self, path):
        if path is None:
            return _UNREACHABLE_COST
        return (
            count_path_turns(path, self.pacman_speed),
            max(0, len(path) - 1),
        )

    def _clear_active_area(self):
        self.phase = SearchPhase.SELECT_PRIORITY
        self.target_area_id = None
        self.planned_area_order = ()
        self.entry = None
        self.exit = None
        self.required_cells = frozenset()
        self._snapshot_step = None
        self._route_cells = ()
        self._route_actions = ()
        self._route_index = 0

    def _fallback_decision(self, current_area, position, started_at, reason):
        return SearchDecision(
            phase=self.phase,
            current_area_id=current_area,
            target_area_id=self.target_area_id,
            planned_area_order=self.planned_area_order,
            entry=self.entry,
            exit=self.exit,
            cells=(position,),
            actions=(),
            chosen_action=(Move.STAY, 1),
            required_cells=self.required_cells,
            covered_cells=frozenset(),
            completed_area_ids=frozenset(self.completed_area_ids),
            completed_this_step=None,
            replan_reason=reason,
            planning_seconds=perf_counter() - started_at,
            exact=False,
            fallback=True,
        )
