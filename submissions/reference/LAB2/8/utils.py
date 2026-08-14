"""
utils.py — Shared utilities for Pacman (Hide) and Ghost (Seek) agents.
Provides:
1. process_vision(map_state, my_position) -> bool mask of visible cells (cross-shaped, range 5, wall-blocked).
2. MemoryMap class: maintains explored while accumulating observed map over time.
3. BeliefTracker class: maintains probability distribution over opponent position.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
from collections import deque


def process_vision(map_state: np.ndarray, my_position: Tuple[int, int]) -> np.ndarray:
    """
    Return a boolean mask same shape as map_state where True indicates the cell
    is visible according to cross-shaped vision (up to 5 cells in each cardinal
    direction) and blocked immediately by walls (value 1).

    Parameters
    ----------
    map_state: np.ndarray (21,21) with values:
        1 = wall, 0 = empty/path, -1 = fog/unseen.
    my_position: (row, col) of the agent.

    Returns
    -------
    np.ndarray of bool, True for visible cells.
    """
    h, w = map_state.shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0 = my_position

    # own cell is always visible (if not a wall; but agent never stands on wall)
    if 0 <= r0 < h and 0 <= c0 < w and map_state[r0, c0] != 1:
        mask[r0, c0] = True

    # four cardinal directions
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for step in range(1, 6):  # vision range 5
            r, c = r0 + dr * step, c0 + dc * step
            if not (0 <= r < h and 0 <= c < w):
                break
            if map_state[r, c] == 1:  # wall blocks vision
                break
            mask[r, c] = True
    return mask


class MemoryMap:
    """
    Persistent map that accumulates observations over time.
    Unknown cells are marked -1, known empty 0, wall 1.
    """

    def __init__(self, shape: Tuple[int, int] = (21, 21)):
        self.shape = shape
        # -1 unknown, 0 empty/path, 1 wall
        self._map: np.ndarray = np.full(shape, -1, dtype=int)

    def update(self, observation: np.ndarray) -> None:
        """
        Incorporate a new observation.
        Known cells (!= -1) overwrite previous unknowns; known cells are trusted.
        """
        if observation.shape != self.shape:
            raise ValueError(f"Observation shape {observation.shape} != expected {self.shape}")
        visible_mask = observation != -1
        self._map[visible_mask] = observation[visible_mask]

    def get_map(self) -> np.ndarray:
        """Return a copy of the current known map."""
        return self._map.copy()

    def is_walkable(self, pos: Tuple[int, int]) -> bool:
        """True if inside bounds and not a known wall (unknown treated as walkable)."""
        r, c = pos
        if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
            return False
        return self._map[r, c] != 1  # -1 or 0 => walkable

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]


class BeliefTracker:
    """
    Maintains a probability distribution over the opponent's position.
    When opponent is seen, belief becomes a delta at that location.
    When unseen, belief is diffused from the last known position over
    reachable cells within a distance proportional to elapsed steps.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (21, 21),
        enemy_speed: int = 1,
        decay: float = 0.8,
    ):
        self.shape = shape
        self.enemy_speed = enemy_speed
        self.decay = decay
        # start uniform over all cells (treated as equally likely)
        self._belief: np.ndarray = np.full(shape, 1.0 / np.prod(shape), dtype=float)
        self._last_seen: Optional[Tuple[int, int]] = None
        self._steps_since_seen: int = 0

    def update(
        self,
        enemy_pos: Optional[Tuple[int, int]],
        my_pos: Tuple[int, int],
        known_map: np.ndarray,
    ) -> None:
        """
        Update belief state.

        Parameters
        ----------
        enemy_pos: (r, c) if opponent observed this step, else None.
        my_pos:    agent's own position (unused but kept for potential extensions).
        known_map: map produced by MemoryMap (values -1/0/1).
        """
        if enemy_pos is not None:
            # Direct observation: certainty
            self._belief.fill(0.0)
            self._belief[enemy_pos] = 1.0
            self._last_seen = enemy_pos
            self._steps_since_seen = 0
            return

        # No observation this step
        if self._last_seen is None:
            # No prior info; keep uniform
            return

        self._steps_since_seen += 1

        # Apply decay to increase uncertainty
        self._belief *= self.decay

        # Compute reachable cells from last seen position within max distance
        max_dist = self._steps_since_seen * self.enemy_speed
        reachable = self._bfs_reachable(self._last_seen, max_dist, known_map)

        # Build uniform distribution over reachable cells
        uniform = np.zeros_like(self._belief)
        if reachable:
            for r, c in reachable:
                self._belief[r, c] = 1.0
            self._belief /= self._belief.sum()
        else:
            # Fallback: uniform over whole map (should not happen if map has open space)
            self._belief.fill(1.0 / np.prod(self.shape))

        # Ensure normalized (defensive)
        total = self._belief.sum()
        if total > 0:
            self._belief /= total
        else:
            valid_positions = (known_map == 0)
            self._belief[valid_positions] = 1.0 / np.sum(valid_positions)

    def get_belief(self) -> np.ndarray:
        """Return a copy of the current belief distribution."""
        return self._belief.copy()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _bfs_reachable(
        self, start: Tuple[int, int], max_dist: int, known_map: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Return list of coordinates reachable from start within max_dist steps,
        treating walls (value 1) as blocked."""
        if not self._in_bounds(start) or known_map[start] == 1:
            return []
        q = deque()
        q.append((start, 0))
        visited = {start}
        reachable = [start]

        while q:
            (r, c), dist = q.popleft()
            if dist >= max_dist:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.shape[0]
                    and 0 <= nc < self.shape[1]
                    and (nr, nc) not in visited
                    and known_map[nr, nc] != 1
                ):
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))
                    reachable.append((nr, nc))
        return reachable

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]


# ----------------------------------------------------------------------
# Example usage (to be adapted by agents)
# ----------------------------------------------------------------------
#
# In agent.py (__init__):
#   from utils import process_vision, MemoryMap, BeliefTracker
#   self.memory = MemoryMap()
#   self.belief = BeliefTracker(enemy_speed=1, decay=0.9)
#
# In agent.step(...):
#   visible = process_vision(map_state, my_position)   # bool mask
#   self.memory.update(map_state)                      # incorporate new observation
#   self.belief.update(enemy_pos, my_position, self.memory.get_map())
#   prob_map = self.belief.get_belief()                # (21,21) probability
#
# The belief can be used to select a target (e.g., argmax) or for sampling.
