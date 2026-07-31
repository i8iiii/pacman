"""RewardCalculator -- potential-based reward shaping for DRL Pacman Seeker.

Computes shaped rewards using potential-based reward shaping (PBRS) to provide
dense learning signals. Rewards combine:
    - Terminal capture / timeout
    - Step cost (with extra penalty for STAY)
    - Potential shaping: gamma * Phi(curr) - Phi(prev)
    - Exploration bonus for first-time cell visits
    - Visibility change events (lost/regained sight)
"""

from typing import Dict, Optional, Tuple

from rl.config import Config


class RewardCalculator:
    """Potential-based reward calculator for Pacman Seeker DRL agent.

    Uses potential-based reward shaping with Manhattan distance as the
    potential function to guide the agent toward the enemy. Also provides
    step costs, exploration bonuses, and visibility-change rewards.

    Attributes:
        config: The shared Config dataclass.
        visit_counts: Dict mapping (row, col) to visit count for exploration.
        _prev_enemy_pos: Last known enemy position (persists when invisible).
        _prev_visible: Whether the enemy was visible on the previous step.
    """

    def __init__(self, config: Config):
        """Initialise the reward calculator from a Config."""
        self.config = config
        self.H: int = config.map_height
        self.W: int = config.map_width
        self._D_max: float = float(self.H + self.W)
        self.gamma: float = config.gamma
        self.reward_capture: float = config.reward_capture
        self.reward_step: float = config.reward_step
        self.reward_stay_penalty: float = config.reward_stay_penalty
        self.reward_exploration_base: float = config.reward_exploration_base
        self.reward_lost_sight: float = config.reward_lost_sight
        self.reward_regained_sight: float = config.reward_regained_sight

        # Mutable state
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self._prev_enemy_pos: Optional[Tuple[int, int]] = None
        self._prev_visible: bool = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state for a new episode.

        Clears visit counts, last-known enemy position, and visibility flag.
        """
        self.visit_counts.clear()
        self._prev_enemy_pos = None
        self._prev_visible = False

    def compute(
        self,
        prev_pos: Tuple[int, int],
        prev_enemy: Optional[Tuple[int, int]],
        prev_visible: bool,
        curr_pos: Tuple[int, int],
        curr_enemy: Optional[Tuple[int, int]],
        curr_visible: bool,
        done: bool,
        action: int,
    ) -> float:
        """Compute the shaped reward for a transition.

        Args:
            prev_pos: Agent position before the action (row, col).
            prev_enemy: Enemy position before the action, or None if invisible.
            prev_visible: Whether the enemy was visible before the action.
            curr_pos: Agent position after the action (row, col).
            curr_enemy: Enemy position after the action, or None if invisible.
            curr_visible: Whether the enemy is visible after the action.
            done: True if the episode terminated (capture or timeout).
            action: The action index that was taken (0=UP, 1=DOWN, 2=LEFT,
                    3=RIGHT, 4=STAY).

        Returns:
            The total shaped reward as a float.
        """
        # --- Terminal rewards ---
        if done:
            if curr_visible and curr_enemy is not None:
                self._update_state(curr_enemy, curr_visible)
                return self.reward_capture
            else:
                self._update_state(curr_enemy, curr_visible)
                return 0.0

        reward = 0.0

        # --- Step cost ---
        if action == 4:  # STAY
            reward += self.reward_stay_penalty
        else:
            reward += self.reward_step

        # --- Potential-based shaping ---
        phi_curr = self._potential(curr_pos, curr_enemy)
        phi_prev = self._potential(prev_pos, prev_enemy)
        reward += self.gamma * phi_curr - phi_prev

        # --- Exploration bonus ---
        if curr_pos not in self.visit_counts:
            reward += self.reward_exploration_base
        self.visit_counts[curr_pos] = self.visit_counts.get(curr_pos, 0) + 1

        # --- Visibility events ---
        if prev_visible and not curr_visible:
            reward += self.reward_lost_sight
        if not prev_visible and curr_visible:
            reward += self.reward_regained_sight

        # --- Update internal state ---
        self._update_state(curr_enemy, curr_visible)

        return reward

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _potential(
        self,
        pos: Tuple[int, int],
        enemy_pos: Optional[Tuple[int, int]],
    ) -> float:
        """Compute the potential function Phi for a given state.

        Phi = -manhattan_distance(pos, enemy_pos) / (H + W)

        If enemy_pos is None, the last known enemy position is used as a
        fallback.  If no last-known position exists, the maximum possible
        distance (H + W) is used as a worst-case estimate.
        """
        if enemy_pos is not None:
            d = abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1])
        elif self._prev_enemy_pos is not None:
            d = abs(pos[0] - self._prev_enemy_pos[0]) + abs(pos[1] - self._prev_enemy_pos[1])
        else:
            d = self._D_max
        return -d / self._D_max

    def _update_state(
        self,
        curr_enemy: Optional[Tuple[int, int]],
        curr_visible: bool,
    ) -> None:
        """Update internal tracking after a transition.

        The last-known enemy position is only overwritten when the enemy
        is currently visible; otherwise the previous value is retained.
        """
        if curr_enemy is not None:
            self._prev_enemy_pos = curr_enemy
        self._prev_visible = curr_visible
