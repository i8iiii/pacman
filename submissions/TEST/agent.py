"""Deterministic Hide agent that occupies a fixed map coordinate."""

from collections import deque

from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class GhostAgent(BaseGhostAgent):
    """Follow a shortest route to the ``(10, 9)`` dead end and remain there."""

    TARGET = (10, 9)
    _MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TEST-Fixed-Position-Hide-Agent"

    def step(self, map_state, my_position, enemy_position, step_number):
        start = int(my_position[0]), int(my_position[1])
        if start == self.TARGET:
            return Move.STAY
        if not self._is_traversable(map_state, start):
            return Move.STAY
        if not self._is_traversable(map_state, self.TARGET):
            return Move.STAY

        frontier = deque([start])
        parents = {start: None}
        arrival_moves = {}

        while frontier:
            current = frontier.popleft()
            for move in self._MOVES:
                candidate = (
                    current[0] + move.value[0],
                    current[1] + move.value[1],
                )
                if candidate in parents:
                    continue
                if not self._is_traversable(map_state, candidate):
                    continue

                parents[candidate] = current
                arrival_moves[candidate] = move
                if candidate == self.TARGET:
                    return self._first_move(
                        start,
                        candidate,
                        parents,
                        arrival_moves,
                    )
                frontier.append(candidate)

        return Move.STAY

    @staticmethod
    def _first_move(start, target, parents, arrival_moves):
        current = target
        while parents[current] != start:
            current = parents[current]
        return arrival_moves[current]

    @staticmethod
    def _is_traversable(map_state, position):
        row, column = position
        rows = len(map_state)
        columns = len(map_state[0]) if rows else 0
        return (
            0 <= row < rows
            and 0 <= column < columns
            and int(map_state[row][column]) != 1
        )
