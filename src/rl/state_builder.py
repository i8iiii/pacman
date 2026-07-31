"""StateBuilder — multi-channel frame stacking for DRL Pacman Seeker.

Converts raw game observations into a stacked tensor suitable for a DRL agent.
Each frame has n_channels binary channels encoding walls, visible empty space,
unseen/fog, pacman position, ghost position, and last-known ghost position.
Consecutive frames are stacked along the channel axis to provide temporal context.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np

from rl.config import Config


class StateBuilder:
    """Builds stacked multi-channel state representations from game observations.

    Observations come as (map_state, my_position, enemy_position, step).
    The builder converts each observation into a multi-channel frame and
    stacks the most recent n_frames frames for temporal context.

    Channels (per frame):
        0: walls         — cells where map == 1
        1: visible_empty — cells where map == 0
        2: unseen        — cells where map == -1 (fog / outside observation radius)
        3: pacman_pos    — binary mask with 1 at the agent's position
        4: ghost_pos     — binary mask with 1 at visible enemy position (all zeros if None)
        5: last_known    — binary mask retaining the last seen enemy position
    """

    def __init__(self, config: Config):
        """Initialise the builder from a Config dataclass.

        Args:
            config: Centralised configuration with map dimensions, frame count,
                    and channel count.
        """
        self.config = config
        self.H: int = config.map_height
        self.W: int = config.map_width
        self.n_frames: int = config.n_frames
        self.n_channels: int = config.n_channels
        self._frame_shape = (self.n_channels, self.H, self.W)

        # Internal state
        self.last_known_enemy_pos: Optional[Tuple[int, int]] = None
        self._buffer: deque = deque(maxlen=self.n_frames)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the frame buffer and last-known enemy position.

        After reset the buffer is pre-filled with zero frames so that
        the first call to build() already produces a valid stacked output.
        """
        self.last_known_enemy_pos = None
        self._buffer.clear()
        zero_frame = np.zeros(self._frame_shape, dtype=np.float32)
        for _ in range(self.n_frames):
            self._buffer.append(zero_frame.copy())

    def build(
        self,
        map_state: np.ndarray,
        my_pos: Tuple[int, int],
        enemy_pos: Optional[Tuple[int, int]],
        step: int,
    ) -> np.ndarray:
        """Convert the current observation into a multi-channel frame, append it
        to the frame buffer, and return the concatenated stacked state.

        Args:
            map_state: 2-D numpy array (H, W) with values 0 (empty), 1 (wall),
                       -1 (unseen/fog).
            my_pos: (row, col) of the Pacman agent.
            enemy_pos: (row, col) of the ghost, or None if not visible.
            step: Current environment step (unused; available for future use).

        Returns:
            np.ndarray of shape (n_frames * n_channels, H, W) — the stacked
            state ready for consumption by a DRL network.
        """
        # Update last-known enemy position when the enemy is visible
        if enemy_pos is not None:
            self.last_known_enemy_pos = enemy_pos

        frame = self._build_single_frame(map_state, my_pos, enemy_pos)
        self._buffer.append(frame)

        # Concatenate frames along the channel axis (axis 0)
        return np.concatenate(list(self._buffer), axis=0)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_single_frame(
        self,
        map_state: np.ndarray,
        my_pos: Tuple[int, int],
        enemy_pos: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """Build a single (n_channels, H, W) frame from one observation.

        Args:
            map_state: 2-D array (H, W) with 0/1/-1 cell values.
            my_pos: Current Pacman position.
            enemy_pos: Current enemy position, or None.

        Returns:
            np.ndarray of shape (n_channels, H, W), dtype float32.
        """
        frame = np.zeros(self._frame_shape, dtype=np.float32)

        # Channel 0: walls  (map == 1)
        frame[0] = (map_state == 1).astype(np.float32)

        # Channel 1: visible empty  (map == 0)
        frame[1] = (map_state == 0).astype(np.float32)

        # Channel 2: unseen / fog  (map == -1)
        frame[2] = (map_state == -1).astype(np.float32)

        # Channel 3: pacman position (binary mask)
        r, c = my_pos
        if 0 <= r < self.H and 0 <= c < self.W:
            frame[3, r, c] = 1.0

        # Channel 4: ghost position (all zeros if enemy_pos is None)
        if enemy_pos is not None:
            gr, gc = enemy_pos
            if 0 <= gr < self.H and 0 <= gc < self.W:
                frame[4, gr, gc] = 1.0

        # Channel 5: last-known enemy position (persists when enemy not visible)
        if self.last_known_enemy_pos is not None:
            lr, lc = self.last_known_enemy_pos
            if 0 <= lr < self.H and 0 <= lc < self.W:
                frame[5, lr, lc] = 1.0

        return frame
