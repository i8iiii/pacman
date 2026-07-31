"""Tests for RewardCalculator -- potential-based reward shaping for DRL Pacman Seeker."""

import sys
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))

import unittest
from rl.config import Config
from rl.reward import RewardCalculator

class TestRewardCalculator(unittest.TestCase):
    """Verify capture, step costs, potential shaping, exploration, and visibility rewards."""

    def setUp(self):
        self.config = Config()
        self.calc = RewardCalculator(self.config)

    def test_capture_reward(self):
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(4, 4), curr_enemy=(4, 4), curr_visible=True,
            done=True, action=2,
        )
        self.assertEqual(reward, 10.0)

    def test_terminal_timeout_not_visible(self):
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(1, 1), curr_enemy=None, curr_visible=False,
            done=True, action=2,
        )
        self.assertEqual(reward, 0.0)

    def test_terminal_timeout_enemy_none(self):
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(1, 1), curr_enemy=None, curr_visible=True,
            done=True, action=2,
        )
        self.assertEqual(reward, 0.0)

    def test_step_cost(self):
        self.calc.visit_counts[(0, 0)] = 1
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(0, 0), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=0,
        )
        self.assertAlmostEqual(reward, -0.01 + (0.99 * (-10 / 40) - (-10 / 40)), places=6)

    def test_stay_penalty(self):
        self.calc.visit_counts[(0, 0)] = 1
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(0, 0), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=4,
        )
        self.assertAlmostEqual(reward, -0.05 + (0.99 * (-10 / 40) - (-10 / 40)), places=6)

    def test_approach_positive_shaping(self):
        self.calc.visit_counts[(2, 2)] = 1
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(2, 2), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=1,
        )
        expected = -0.01 + (0.99 * (-6 / 40) - (-10 / 40))
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertGreater(reward, 0.0)

    def test_retreat_negative_shaping(self):
        self.calc.visit_counts[(0, 2)] = 1
        reward = self.calc.compute(
            prev_pos=(2, 2), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(0, 2), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=0,
        )
        expected = -0.01 + (0.99 * (-8 / 40) - (-6 / 40))
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertLess(reward, 0.0)

    def test_first_visit_exploration_bonus(self):
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(2, 2), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=1,
        )
        self.assertIn((2, 2), self.calc.visit_counts)
        self.assertEqual(self.calc.visit_counts[(2, 2)], 1)
        expected = -0.01 + (0.99 * (-6 / 40) - (-10 / 40)) + 0.1
        self.assertAlmostEqual(reward, expected, places=6)

    def test_revisit_no_exploration_bonus(self):
        self.calc.visit_counts[(2, 2)] = 1
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(2, 2), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=1,
        )
        self.assertEqual(self.calc.visit_counts[(2, 2)], 2)
        expected = -0.01 + (0.99 * (-6 / 40) - (-10 / 40))
        self.assertAlmostEqual(reward, expected, places=6)

    def test_lost_sight_penalty(self):
        self.calc.visit_counts[(1, 1)] = 1
        self.calc._prev_enemy_pos = (5, 5)
        self.calc._prev_visible = True
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(5, 5), prev_visible=True,
            curr_pos=(1, 1), curr_enemy=None, curr_visible=False,
            done=False, action=1,
        )
        expected = -0.01 + (0.99 * (-8 / 40) - (-10 / 40)) - 0.3
        self.assertAlmostEqual(reward, expected, places=6)

    def test_regained_sight_bonus(self):
        self.calc.visit_counts[(1, 1)] = 1
        self.calc._prev_visible = False
        reward = self.calc.compute(
            prev_pos=(0, 0), prev_enemy=None, prev_visible=False,
            curr_pos=(1, 1), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=1,
        )
        expected = -0.01 + (0.99 * (-8 / 40) - (-40 / 40)) + 0.3
        self.assertAlmostEqual(reward, expected, places=6)

    def test_reset_clears_state(self):
        self.calc.visit_counts[(0, 0)] = 5
        self.calc._prev_enemy_pos = (3, 3)
        self.calc._prev_visible = True
        self.calc.reset()
        self.assertEqual(len(self.calc.visit_counts), 0)
        self.assertIsNone(self.calc._prev_enemy_pos)
        self.assertFalse(self.calc._prev_visible)

    def test_state_updated_after_compute(self):
        self.calc.compute(
            prev_pos=(0, 0), prev_enemy=None, prev_visible=False,
            curr_pos=(1, 1), curr_enemy=(5, 5), curr_visible=True,
            done=False, action=1,
        )
        self.assertEqual(self.calc._prev_enemy_pos, (5, 5))
        self.assertTrue(self.calc._prev_visible)

    def test_prev_enemy_pos_persists_when_enemy_none(self):
        self.calc._prev_enemy_pos = (3, 3)
        self.calc.compute(
            prev_pos=(0, 0), prev_enemy=(3, 3), prev_visible=True,
            curr_pos=(1, 1), curr_enemy=None, curr_visible=False,
            done=False, action=1,
        )
        self.assertEqual(self.calc._prev_enemy_pos, (3, 3))
        self.assertFalse(self.calc._prev_visible)

if __name__ == "__main__":
    unittest.main()
