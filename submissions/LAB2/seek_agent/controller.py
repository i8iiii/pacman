"""Chasing-first controller for the LAB2 Pacman seeker."""

import random
from collections import deque
from enum import Enum
from time import perf_counter

import numpy as np

from environment import Move

from .diagnostics import SeekDiagnostics


CARDINAL_MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)


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
    ):
        self.pacman_speed = max(1, int(pacman_speed))
        self.diagnostics = SeekDiagnostics(
            enabled=diagnostics_enabled,
            log_path=diagnostic_log_path,
        )
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
            my_position = _position(my_position)
            enemy_position = _optional_position(enemy_position)
            step_number = int(step_number)

            if (
                self._last_step_number is not None
                and step_number < self._last_step_number
            ):
                self._reset_match_state()
                self.diagnostics.reset_for_match()
                transition_reasons.append("new_match")

            self._last_step_number = step_number

            if enemy_position is not None:
                if self.mode != SeekMode.CHASING:
                    transition_reasons.append("ghost_visible")
                self.mode = SeekMode.CHASING
                self.last_seen_position = enemy_position
                self.last_seen_step = step_number
                target = enemy_position
                path = _shortest_path(topology, my_position, target)
                move, move_steps = self._action_from_path(path)
            elif self.last_seen_position is not None:
                if self.mode != SeekMode.INVESTIGATING:
                    transition_reasons.append("ghost_lost")
                self.mode = SeekMode.INVESTIGATING
                target = self.last_seen_position
                path = _shortest_path(topology, my_position, target)
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
            error=error,
        )
        return move, move_steps

    def _action_from_path(self, path):
        if not path or len(path) < 2:
            return Move.STAY, 1

        first_delta = _delta(path[0], path[1])
        move = _move_for_delta(first_delta)
        straight_steps = 1
        limit = min(self.pacman_speed, len(path) - 1)

        for index in range(2, limit + 1):
            if _delta(path[index - 1], path[index]) != first_delta:
                break
            straight_steps += 1

        return move, straight_steps

    @staticmethod
    def _random_search_action(topology, my_position):
        choices = []
        for move in CARDINAL_MOVES:
            next_position = _apply_move(my_position, move)
            if _is_traversable(topology, next_position):
                choices.append((move, next_position))

        if not choices:
            return Move.STAY, 1, [my_position]

        move, next_position = random.choice(choices)
        return move, 1, [my_position, next_position]


def _shortest_path(topology, start, goal):
    if not _is_traversable(topology, start):
        return None
    if not _is_traversable(topology, goal):
        return None
    if start == goal:
        return [start]

    frontier = deque([start])
    parents = {start: None}

    while frontier:
        current = frontier.popleft()
        for move in CARDINAL_MOVES:
            neighbor = _apply_move(current, move)
            if neighbor in parents or not _is_traversable(topology, neighbor):
                continue
            parents[neighbor] = current
            if neighbor == goal:
                return _reconstruct_path(parents, goal)
            frontier.append(neighbor)

    return None


def _reconstruct_path(parents, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def _is_traversable(topology, position):
    row, column = position
    return (
        0 <= row < topology.shape[0]
        and 0 <= column < topology.shape[1]
        and bool(topology[row, column])
    )


def _apply_move(position, move):
    return (
        position[0] + move.value[0],
        position[1] + move.value[1],
    )


def _delta(start, end):
    return end[0] - start[0], end[1] - start[1]


def _move_for_delta(delta):
    for move in CARDINAL_MOVES:
        if move.value == delta:
            return move
    raise ValueError(f"Path contains invalid movement delta: {delta}")


def _position(value):
    return int(value[0]), int(value[1])


def _optional_position(value):
    return None if value is None else _position(value)
