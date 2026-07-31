"""Tests for PacmanEnv -- Gym-style wrapper for the DRL Pacman Seeker agent."""

import sys
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))

import unittest
import numpy as np
from rl.config import Config
from rl.pacman_env import PacmanEnv


class TestPacmanEnv(unittest.TestCase):
    """Verify reset/step interface, max_steps termination, and ghost policy integration."""

    def _make_config(self, **overrides):
        """Create a Config with overridden fields for testing.

        Sets map_width=21 to match the actual Environment map dimensions.
        """
        cfg = Config()
        cfg.map_width = 21  # Environment produces 21x21 maps
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    # ------------------------------------------------------------------
    # Test 1: reset() shape and dtype
    # ------------------------------------------------------------------

    def test_reset_shape_and_dtype(self):
        """reset() returns ndarray of shape (24, 21, 21) with dtype float32."""
        cfg = self._make_config()
        env = PacmanEnv(cfg, ghost_policy_name="random")
        state = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (24, 21, 21),
                         f"Expected shape (24,21,21), got {state.shape}")
        self.assertEqual(state.dtype, np.float32)

    # ------------------------------------------------------------------
    # Test 2: step() return types
    # ------------------------------------------------------------------

    def test_step_return_types(self):
        """step(0) returns (ndarray, float, bool, dict)."""
        cfg = self._make_config()
        env = PacmanEnv(cfg, ghost_policy_name="random")
        env.reset()
        next_state, reward, done, info = env.step(0)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        # Verify info keys
        expected_keys = {"result", "step", "pacman_pos", "ghost_pos", "enemy_visible"}
        self.assertTrue(expected_keys.issubset(set(info.keys())),
                        f"Missing info keys: {expected_keys - set(info.keys())}")

    # ------------------------------------------------------------------
    # Test 3: max_steps termination
    # ------------------------------------------------------------------

    def test_terminates_within_max_steps(self):
        """env terminates before or at max_steps (use config.max_steps=10)."""
        cfg = self._make_config(max_steps=10)
        env = PacmanEnv(cfg, ghost_policy_name="random")
        env.reset()
        step_count = 0
        done = False
        while not done:
            _, _, done, _ = env.step(0)  # always move UP
            step_count += 1
        self.assertLessEqual(step_count, 10,
                             f"Expected <=10 steps, got {step_count}")
        # Should have terminated
        self.assertTrue(done)

    # ------------------------------------------------------------------
    # Test 4: ghost policy integration
    # ------------------------------------------------------------------

    def test_works_with_random_ghost_policy(self):
        """PacmanEnv works with ghost_policy_name='random'."""
        cfg = self._make_config(max_steps=20)
        env = PacmanEnv(cfg, ghost_policy_name="random")
        state = env.reset()
        self.assertEqual(state.shape, (24, 21, 21))
        for _ in range(10):
            next_state, reward, done, info = env.step(0)
            self.assertIsInstance(next_state, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            if done:
                break

    def test_works_with_greedy_ghost_policy(self):
        """PacmanEnv works with ghost_policy_name='greedy'."""
        cfg = self._make_config(max_steps=20)
        env = PacmanEnv(cfg, ghost_policy_name="greedy")
        state = env.reset()
        self.assertEqual(state.shape, (24, 21, 21))
        for _ in range(10):
            next_state, reward, done, info = env.step(1)  # DOWN
            self.assertIsInstance(next_state, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            if done:
                break

    # ------------------------------------------------------------------
    # Test 5: multiple resets work correctly
    # ------------------------------------------------------------------

    def test_multiple_resets(self):
        """Calling reset() multiple times produces consistent shapes."""
        cfg = self._make_config(max_steps=10)
        env = PacmanEnv(cfg, ghost_policy_name="random")
        for _ in range(3):
            state = env.reset()
            self.assertEqual(state.shape, (24, 21, 21))
            self.assertEqual(state.dtype, np.float32)
            # Step a few times
            for _ in range(5):
                ns, r, d, info = env.step(2)  # LEFT
                if d:
                    break


if __name__ == "__main__":
    unittest.main()