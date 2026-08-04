"""Chasing-first controller for the LAB2 Pacman seeker."""

import random
from enum import Enum
from time import perf_counter

import numpy as np

from environment import Move

from .areas import AreaAnalyzer
from .diagnostics import SeekDiagnostics
from .spatial import (
    CARDINAL_MOVES,
    apply_move,
    is_traversable,
    move_for_delta,
    movement_delta,
    normalize_position,
    optional_position,
    shortest_path,
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
        self._reset_match_state()

    def _reset_match_state(self):
        self.mode = SeekMode.SEARCHING
        self.last_seen_position = None
        self.last_seen_step = None

    def step(self, map_state, my_position, enemy_position, step_number):
        started_at = perf_counter()
        previous_mode = self.mode
        transition_reasons = []
        target = None
        path = None
        error = None
        move, move_steps = Move.STAY, 1

        try:
            observation = np.asarray(map_state)
            topology = observation != 1
            my_position = normalize_position(my_position)
            enemy_position = optional_position(enemy_position)
            step_number = int(step_number)

            if (
                self._last_step_number is not None
                and step_number <= self._last_step_number
            ):
                self._reset_match_state()
                self.diagnostics.reset_for_match()
                self._area_diagnostics_pending = True
                transition_reasons.append("new_match")

            self._last_step_number = step_number

            if self.diagnostics.enabled or (
                enemy_position is None
                and self.last_seen_position is None
            ):
                self._ensure_area_analysis(observation)

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
                move, move_steps, path = self._random_search_action(
                    topology,
                    my_position,
                )

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
        target_area_id = self._area_for(target)
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

    @staticmethod
    def _random_search_action(topology, my_position):
        choices = []
        for move in CARDINAL_MOVES:
            next_position = apply_move(my_position, move)
            if is_traversable(topology, next_position):
                choices.append((move, next_position))

        if not choices:
            return Move.STAY, 1, [my_position]

        move, next_position = random.choice(choices)
        return move, 1, [my_position, next_position]
