"""Chasing-first controller for the LAB2 Pacman seeker."""

from enum import Enum
from time import perf_counter

import numpy as np

from environment import Move

from .areas import AreaAnalyzer
from .belief import GhostBelief
from .diagnostics import SeekDiagnostics
from .search import SearchPlanner
from .spatial import (
    move_for_delta,
    movement_delta,
    normalize_position,
    optional_position,
    reachable_component,
    shortest_path,
    visibility_footprint,
)


class SeekMode(str, Enum):
    SEARCHING = "searching"
    CHASING = "chasing"
    INVESTIGATING = "investigating"


class SeekController:
    """Coordinate the seeker's three initial behavior modes."""

    def __init__(
        self,
        pacman_speed=2,
        diagnostics_enabled=None,
        diagnostic_log_path=None,
        diagnostic_area_path=None,
    ):
        self.pacman_speed = max(1, int(pacman_speed))
        self.diagnostics = SeekDiagnostics(
            enabled=diagnostics_enabled,
            log_path=diagnostic_log_path,
            area_path=diagnostic_area_path,
        )
        self.area_analyzer = AreaAnalyzer()
        self.area_analysis = None
        self.area_cache_hit = False
        self._area_diagnostics_pending = True
        self._last_step_number = None
        self.reachable_component = frozenset()
        self.ghost_belief = GhostBelief()
        self.search_planner = SearchPlanner(pacman_speed=self.pacman_speed)
        self._reset_match_state()

    def _reset_match_state(self, analysis=None):
        """Clear all state whose lifetime is one arena match."""
        self.mode = SeekMode.SEARCHING
        self.last_seen_position = None
        self.last_seen_step = None
        self.reachable_component = frozenset()
        self.ghost_belief = GhostBelief()
        self.search_planner.reset(analysis)

    def step(self, map_state, my_position, enemy_position, step_number):
        started_at = perf_counter()
        previous_mode = self.mode
        transition_reasons = []
        target = None
        path = None
        search_decision = None
        visible_cells = frozenset()
        error = None
        move, move_steps = Move.STAY, 1

        try:
            observation = np.asarray(map_state)
            topology = observation != 1
            my_position = normalize_position(my_position)
            enemy_position = optional_position(enemy_position)
            step_number = int(step_number)

            is_new_match = (
                self._last_step_number is None
                or step_number <= self._last_step_number
            )
            if is_new_match:
                self._reset_match_state(self.area_analysis)
                self.reachable_component = reachable_component(
                    topology,
                    my_position,
                )
                self.diagnostics.reset_for_match()
                self._area_diagnostics_pending = True
                transition_reasons.append("new_match")

            self._last_step_number = step_number

            # Search decisions and the area diagnostic report both use the
            # same topology.  AreaAnalyzer caches its expensive analysis.
            self._ensure_area_analysis(observation)
            visible_cells = visibility_footprint(topology, my_position, radius=5)
            if is_new_match:
                self.ghost_belief.reset(
                    observation,
                    my_position,
                    visible_cells=visible_cells,
                    step_number=step_number,
                    reachable_cells=self.reachable_component,
                )
            self.ghost_belief.update(
                observation,
                visible_cells,
                enemy_position,
                step_number,
            )

            if enemy_position is not None:
                if self.mode != SeekMode.CHASING:
                    transition_reasons.append("ghost_visible")
                self.mode = SeekMode.CHASING
                self.last_seen_position = enemy_position
                self.last_seen_step = step_number
                target = enemy_position
                path = shortest_path(topology, my_position, target)
                move, move_steps = self._action_from_path(path)
            elif self.last_seen_position is not None:
                if self.mode != SeekMode.INVESTIGATING:
                    transition_reasons.append("ghost_lost")
                self.mode = SeekMode.INVESTIGATING
                target = self.last_seen_position
                path = shortest_path(topology, my_position, target)
                move, move_steps = self._action_from_path(path)
            else:
                if self.mode != SeekMode.SEARCHING:
                    transition_reasons.append("no_ghost_information")
                self.mode = SeekMode.SEARCHING
                search_decision = self.search_planner.decide(
                    topology,
                    self.area_analysis,
                    self.ghost_belief,
                    my_position,
                    visible_cells=visible_cells,
                    step_number=step_number,
                    reachable_cells=self.reachable_component,
                )
                target = search_decision.entry
                path = list(search_decision.route)
                move, move_steps = search_decision.chosen_action

            if target is not None and path is None:
                error = f"no path from {my_position} to {target}"
                move, move_steps = Move.STAY, 1
        except Exception as exc:
            observation = np.asarray(map_state)
            topology = observation != 1
            error = f"{type(exc).__name__}: {exc}"
            move, move_steps = Move.STAY, 1

        duration = perf_counter() - started_at
        current_area_id = self._area_for(my_position)
        target_area_id = (
            search_decision.target_area_id
            if search_decision is not None
            else self._area_for(target)
        )
        self.diagnostics.record_decision(
            step_number=step_number,
            mode=self.mode,
            previous_mode=previous_mode,
            transition_reasons=transition_reasons,
            map_state=observation,
            topology=topology,
            my_position=my_position,
            enemy_position=enemy_position,
            last_seen_position=self.last_seen_position,
            last_seen_step=self.last_seen_step,
            target=target,
            path=path,
            move=move,
            move_steps=move_steps,
            duration_seconds=duration,
            area_analysis=self.area_analysis,
            area_cache_hit=self.area_cache_hit,
            current_area_id=current_area_id,
            target_area_id=target_area_id,
            visible_cells=visible_cells,
            ghost_belief=self.ghost_belief,
            search_decision=search_decision,
            error=error,
        )
        return move, move_steps

    def _ensure_area_analysis(self, observation):
        analysis, cache_hit = self.area_analyzer.analyze(observation)
        changed = (
            self.area_analysis is None
            or self.area_analysis.fingerprint != analysis.fingerprint
        )
        self.area_analysis = analysis
        self.area_cache_hit = cache_hit
        if changed or self._area_diagnostics_pending:
            self.diagnostics.write_area_analysis(
                analysis,
                observation,
                cache_hit,
            )
            self._area_diagnostics_pending = False

    def _area_for(self, position):
        if self.area_analysis is None or position is None:
            return None
        try:
            return self.area_analysis.area_for(normalize_position(position))
        except (IndexError, TypeError, ValueError):
            return None

    def _action_from_path(self, path):
        if not path or len(path) < 2:
            return Move.STAY, 1

        first_delta = movement_delta(path[0], path[1])
        move = move_for_delta(first_delta)
        straight_steps = 1
        limit = min(self.pacman_speed, len(path) - 1)

        for index in range(2, limit + 1):
            if movement_delta(path[index - 1], path[index]) != first_delta:
                break
            straight_steps += 1

        return move, straight_steps
