"""PacmanEnv — Gym-style wrapper for the DRL Pacman Seeker agent.

Wraps the Environment, StateBuilder, RewardCalculator, and ghost policies
into a single step()/reset() interface compatible with RL training loops.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from environment import Environment, Move
from rl.config import Config
from rl.ghost_policies import get_ghost_policy
from rl.reward import RewardCalculator
from rl.state_builder import StateBuilder


class PacmanEnv:
    """Gym-style RL environment wrapping the Pacman vs Ghost game.

    Provides reset() -> state and step(action) -> (next_state, reward, done, info)
    using the standard Gym interface convention.

    Attributes:
        ACTION_MAP: Mapping from discrete action indices (0-4) to Move enums.
    """

    ACTION_MAP = {
        0: Move.UP,
        1: Move.DOWN,
        2: Move.LEFT,
        3: Move.RIGHT,
        4: Move.STAY,
    }

    def __init__(self, config: Config, ghost_policy_name: str = "random"):
        """Initialise the Pacman environment wrapper.

        Args:
            config: A Config dataclass with all hyperparameters.
            ghost_policy_name: Name of the ghost policy to use
                               ("random", "greedy", or "minimax").
        """
        self._env = Environment(
            max_steps=config.max_steps,
            deterministic_starts=True,
            capture_distance_threshold=config.capture_distance,
            pacman_speed=1,
        )
        self.state_builder = StateBuilder(config)
        self.reward_calc = RewardCalculator(config)
        self.config = config
        self.ghost_policy_name = ghost_policy_name
        self.ghost_policy = None

        # Track previous state for reward computation
        self._prev_pacman_pos: Optional[Tuple[int, int]] = None
        self._prev_enemy_pos: Optional[Tuple[int, int]] = None
        self._prev_enemy_visible: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial stacked state.

        Returns:
            np.ndarray of shape (n_frames * n_channels, H, W) — the initial
            stacked state tensor with dtype float32.
        """
        self._env.reset()
        self.state_builder.reset()
        self.reward_calc.reset()
        self.ghost_policy = get_ghost_policy(self.ghost_policy_name)

        map_state, pacman_pos, ghost_pos = self._env.get_state()
        pacman_obs, my_pos, visible_enemy = self._env.get_observation(
            "pacman", self.config.pacman_obs_radius, self.config.ghost_obs_radius
        )

        self._prev_pacman_pos = my_pos
        self._prev_enemy_pos = visible_enemy
        self._prev_enemy_visible = visible_enemy is not None

        return self.state_builder.build(pacman_obs, my_pos, visible_enemy, 1)

    def step(self, action: int):
        """Execute one step in the environment.

        Args:
            action: Discrete action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY).
                    Values outside [0,4] are clamped.

        Returns:
            Tuple of (next_state, reward, done, info):
                - next_state: stacked state ndarray
                - reward: float reward for the transition
                - done: True if the episode has terminated
                - info: dict with result, step, positions, and visibility
        """
        action = max(0, min(4, int(action)))
        pacman_move = self.ACTION_MAP[action]

        # Get ghost observation and produce ghost move
        ghost_obs, ghost_pos, ghost_visible_enemy = self._env.get_observation(
            "ghost", self.config.pacman_obs_radius, self.config.ghost_obs_radius
        )
        ghost_move = self.ghost_policy.step(
            ghost_obs, ghost_pos, ghost_visible_enemy, self._env.current_step + 1
        )

        # Step the underlying environment
        game_over, result, new_state = self._env.step(pacman_move, ghost_move)
        map_state, pacman_pos, ghost_pos = new_state

        # Get Pacman observation for the new state
        pacman_obs, my_pos, visible_enemy = self._env.get_observation(
            "pacman", self.config.pacman_obs_radius, self.config.ghost_obs_radius
        )

        curr_visible = visible_enemy is not None

        # Compute reward
        reward = self.reward_calc.compute(
            prev_pos=self._prev_pacman_pos,
            prev_enemy=self._prev_enemy_pos,
            prev_visible=self._prev_enemy_visible,
            curr_pos=my_pos,
            curr_enemy=visible_enemy,
            curr_visible=curr_visible,
            done=game_over,
            action=action,
        )

        # Build next state
        next_state = self.state_builder.build(
            pacman_obs, my_pos, visible_enemy, self._env.current_step
        )

        # Update tracking for next step
        self._prev_pacman_pos = my_pos
        self._prev_enemy_pos = visible_enemy
        self._prev_enemy_visible = curr_visible

        info: Dict = {
            "result": result,
            "step": self._env.current_step,
            "pacman_pos": pacman_pos,
            "ghost_pos": ghost_pos,
            "enemy_visible": curr_visible,
        }

        return next_state, reward, game_over, info