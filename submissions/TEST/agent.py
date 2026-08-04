"""Simple deterministic Hide agent for movement checks."""

from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class GhostAgent(BaseGhostAgent):
    """Move down twice, right three times, then remain still."""

    _MOVES = (
        Move.DOWN,
        Move.DOWN,
        Move.RIGHT,
        Move.RIGHT,
        Move.RIGHT,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TEST-Hide-Agent"

    def step(self, map_state, my_position, enemy_position, step_number):
        try:
            move_index = int(step_number) - 1
        except (TypeError, ValueError):
            return Move.STAY
        if 0 <= move_index < len(self._MOVES):
            return self._MOVES[move_index]
        return Move.STAY
