"""Ghost policy classes for training opponent behavior.

Provides RandomGhostPolicy, GreedyGhostPolicy, and a factory function
get_ghost_policy() for creating policy instances by name.

All policies expose a step() method compatible with GhostAgent.step().
"""

import random
from typing import List, Optional, Tuple
import numpy as np
from environment import Move


def _get_valid_moves(map_state: np.ndarray, pos: Tuple[int, int]) -> List[Move]:
    """Return a list of Move enums representing valid moves from pos.

    A move is valid if it stays within bounds and does not enter a wall.
    STAY is always considered valid.

    Args:
        map_state: 2D numpy array where 1 = wall, anything else is traversable.
        pos: Current position as (row, col).

    Returns:
        List of Move enum values for valid actions.
    """
    height, width = map_state.shape
    r, c = pos
    valid = [Move.STAY]  # STAY is always valid

    for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
        dr, dc = move.value
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] != 1:
            valid.append(move)

    return valid


class RandomGhostPolicy:
    """Ghost policy that selects uniformly from valid moves."""

    def step(
        self,
        map_state: np.ndarray,
        my_position: Tuple[int, int],
        enemy_position: Optional[Tuple[int, int]],
        step_number: int,
    ) -> Move:
        """Select a uniformly random valid move.

        Args:
            map_state: 2D numpy array (1 = wall).
            my_position: Ghost position as (row, col).
            enemy_position: Pacman position or None (ignored).
            step_number: Current step number (ignored).

        Returns:
            A Move enum value.
        """
        valid = _get_valid_moves(map_state, my_position)
        return random.choice(valid)


class GreedyGhostPolicy:
    """Ghost policy that runs away from Pacman (maximizes Manhattan distance).

    When enemy_position is None (ghost cannot see Pacman), falls back to
    uniform random behavior.
    """

    def step(
        self,
        map_state: np.ndarray,
        my_position: Tuple[int, int],
        enemy_position: Optional[Tuple[int, int]],
        step_number: int,
    ) -> Move:
        """Select the move that maximizes Manhattan distance from enemy.

        If enemy_position is None, falls through to random selection.
        If no valid moves, returns Move.STAY.

        Args:
            map_state: 2D numpy array (1 = wall).
            my_position: Ghost position as (row, col).
            enemy_position: Pacman position or None if not visible.
            step_number: Current step number (ignored).

        Returns:
            A Move enum value.
        """
        valid = _get_valid_moves(map_state, my_position)

        if enemy_position is None:
            # Fall through to random behavior
            return random.choice(valid)

        # Pick move maximizing Manhattan distance from enemy
        er, ec = enemy_position
        best_move = random.choice(valid)  # default fallback
        best_dist = -1

        for move in valid:
            dr, dc = move.value
            nr, nc = my_position[0] + dr, my_position[1] + dc
            dist = abs(nr - er) + abs(nc - ec)
            if dist > best_dist:
                best_dist = dist
                best_move = move

        return best_move


def get_ghost_policy(name: str):
    """Factory function returning a ghost policy instance by name.

    Args:
        name: Policy name -- 'random', 'greedy', or 'minimax'.
              'minimax' currently falls back to GreedyGhostPolicy.

    Returns:
        A ghost policy instance with a step() method.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name == "random":
        return RandomGhostPolicy()
    elif name == "greedy":
        return GreedyGhostPolicy()
    elif name == "minimax":
        # Falls back to greedy for now
        return GreedyGhostPolicy()
    else:
        raise ValueError(f"Unknown ghost policy: {name!r}. "
                         f"Expected one of: 'random', 'greedy', 'minimax'.")
