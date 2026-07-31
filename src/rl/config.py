"""Configuration dataclass for DRL Pacman Seeker agent training.

All hyperparameters and environment settings are centralized here.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Config:
    """Centralized configuration for DRL training.

    Attributes:
        map_height: Grid height (rows).
        map_width: Grid width (columns).
        max_steps: Maximum steps per episode before truncation.
        n_actions: Number of discrete actions (UP, DOWN, LEFT, RIGHT, STAY).
        n_channels: Number of input channels per frame.
        n_frames: Number of stacked frames for frame-stacking.
        pacman_obs_radius: Observation radius for Pacman agent.
        ghost_obs_radius: Observation radius for ghost agents.
        capture_distance: Manhattan distance threshold for capture.
        gamma: Discount factor.
        epsilon_start: Initial epsilon for exploration.
        epsilon_end: Minimum epsilon after decay.
        epsilon_decay_steps: Steps over which epsilon decays linearly.
        learning_rate: Optimizer learning rate.
        batch_size: Minibatch size for training.
        sequence_length: DRQN sequence length (unrolled LSTM steps).
        target_sync_steps: Steps between target network syncs.
        replay_buffer_capacity: Maximum size of replay buffer.
        per_alpha: PER alpha - prioritization exponent.
        per_beta_start: PER beta initial value - importance-sampling correction.
        per_beta_end: PER beta final value.
        per_epsilon: PER epsilon - small constant to avoid zero priorities.
        total_training_steps: Total environment steps for training.
        eval_interval: Steps between evaluation runs.
        eval_episodes: Number of episodes per evaluation run.
        curriculum_stages: Tuple of (obs_radius, ghost_policy_weights) stages.
        reward_capture: Reward for capturing a ghost.
        reward_step: Per-step reward (small negative to encourage efficiency).
        reward_stay_penalty: Penalty for choosing the STAY action.
        reward_exploration_base: Base exploration reward for visiting new cells.
        reward_lost_sight: Penalty when a ghost leaves the observation radius.
        reward_regained_sight: Bonus when a ghost re-enters the observation radius.
        device: Compute device for PyTorch tensors.
    """

    # --- Environment ---
    map_height: int = 21
    map_width: int = 19
    max_steps: int = 200
    n_actions: int = 5
    n_channels: int = 6
    n_frames: int = 4

    # --- Observation ---
    pacman_obs_radius: int = 5
    ghost_obs_radius: int = 5
    capture_distance: int = 1

    # --- DQN hyperparameters ---
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    learning_rate: float = 2.5e-4
    batch_size: int = 32
    sequence_length: int = 8
    target_sync_steps: int = 2000
    replay_buffer_capacity: int = 100000

    # --- Prioritized Experience Replay (PER) ---
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-6

    # --- Training loop ---
    total_training_steps: int = 100000
    eval_interval: int = 500
    eval_episodes: int = 100

    # --- Curriculum stages ---
    curriculum_stages: Tuple[Tuple[int, Dict[str, float]], ...] = (
        (10, {"random": 0.7, "greedy": 0.3, "minimax": 0.0}),
        (7, {"random": 0.5, "greedy": 0.5, "minimax": 0.0}),
        (5, {"random": 0.0, "greedy": 0.7, "minimax": 0.3}),
    )

    # --- Reward weights ---
    reward_capture: float = 10.0
    reward_step: float = -0.01
    reward_stay_penalty: float = -0.05
    reward_exploration_base: float = 0.1
    reward_lost_sight: float = -0.3
    reward_regained_sight: float = 0.3

    # --- Device ---
    device: str = "cpu"
