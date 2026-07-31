"""Tests for Prioritized Replay Buffer with SumTree -- DRQN-compatible."""

import sys
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))

import unittest
import numpy as np
from rl.config import Config
from rl.replay_buffer import SumTree, ReplayBuffer


class TestSumTree(unittest.TestCase):
    """Verify SumTree: add, update, get, total."""

    def setUp(self):
        self.capacity = 8
        self.tree = SumTree(self.capacity)

    def test_initial_total_zero(self):
        self.assertEqual(self.tree.total(), 0.0)
        self.assertEqual(self.tree.size, 0)

    def test_add_increases_total_and_size(self):
        self.tree.add(5.0, "a")
        self.assertEqual(self.tree.total(), 5.0)
        self.assertEqual(self.tree.size, 1)

        self.tree.add(3.0, "b")
        self.assertEqual(self.tree.total(), 8.0)
        self.assertEqual(self.tree.size, 2)

    def test_add_over_capacity_maintains_size(self):
        for i in range(10):
            self.tree.add(float(i + 1), f"data_{i}")
        self.assertEqual(self.tree.size, self.capacity)
        self.assertEqual(self.tree.total(), 52.0)

    def test_get_returns_correct_data(self):
        priorities = [1.0, 4.0, 2.0, 3.0]
        data_items = ["a", "b", "c", "d"]
        for p, d in zip(priorities, data_items):
            self.tree.add(p, d)

        total = self.tree.total()
        self.assertEqual(total, 10.0)

        _, _, d0 = self.tree.get(0.0)
        self.assertEqual(d0, "a")

        _, _, d1 = self.tree.get(0.5)
        self.assertEqual(d1, "a")

        _, _, d2 = self.tree.get(1.5)
        self.assertEqual(d2, "b")

        _, _, d3 = self.tree.get(5.5)
        self.assertEqual(d3, "c")

        _, _, d4 = self.tree.get(9.9)
        self.assertEqual(d4, "d")

    def test_get_boundary_safety(self):
        for i in range(3):
            self.tree.add(1.0, f"x{i}")
        _, _, d = self.tree.get(self.tree.total())
        self.assertEqual(d, "x2")

    def test_update_changes_priority(self):
        self.tree.add(2.0, "first")
        self.tree.add(3.0, "second")
        self.assertEqual(self.tree.total(), 5.0)

        tree_idx = self.capacity - 1
        self.tree.update(tree_idx, 10.0)
        self.assertEqual(self.tree.total(), 13.0)

    def test_update_propagates_correctly(self):
        for i in range(4):
            self.tree.add(1.0, f"d{i}")
        self.assertEqual(self.tree.total(), 4.0)

        first_leaf = self.capacity - 1
        self.tree.update(first_leaf, 5.0)
        self.assertEqual(self.tree.total(), 8.0)


class TestReplayBuffer(unittest.TestCase):
    """Verify ReplayBuffer: push, sample, update_priorities, beta annealing."""

    def setUp(self):
        self.config = Config()
        self.config.replay_buffer_capacity = 64
        self.config.sequence_length = 4
        self.config.batch_size = 4
        self.buffer = ReplayBuffer(self.config)

    def _dummy_state(self):
        return np.zeros((1, 5, 5), dtype=np.float32)

    def _dummy_hidden(self, dim=64):
        return (np.zeros(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))

    def _push_episode(self, length, start_val=0):
        """Push a full episode of given length. Returns ending index."""
        for t in range(length):
            state = np.full((1, 5, 5), float(start_val + t), dtype=np.float32)
            next_state = np.full((1, 5, 5), float(start_val + t + 1), dtype=np.float32)
            done = (t == length - 1)
            self.buffer.push(state, t % 5, 1.0, next_state, done, self._dummy_hidden())

    def test_empty_buffer_len_zero(self):
        self.assertEqual(len(self.buffer), 0)

    def test_push_increases_len(self):
        self.buffer.push(self._dummy_state(), 0, 1.0, self._dummy_state(), False, self._dummy_hidden())
        self.assertEqual(len(self.buffer), 1)

        self.buffer.push(self._dummy_state(), 1, 1.0, self._dummy_state(), True, self._dummy_hidden())
        self.assertEqual(len(self.buffer), 2)

    def test_push_over_capacity_evicts_oldest(self):
        cap = self.config.replay_buffer_capacity
        for i in range(cap + 10):
            self.buffer.push(
                np.full((1, 5, 5), float(i), dtype=np.float32),
                0, 1.0,
                np.full((1, 5, 5), float(i + 1), dtype=np.float32),
                False, self._dummy_hidden(),
            )
        self.assertEqual(len(self.buffer), cap)

    def test_push_done_marks_episode_boundary(self):
        self.assertEqual(len(self.buffer.episode_boundaries), 0)
        self._push_episode(5)
        self.assertEqual(len(self.buffer.episode_boundaries), 1)

        self._push_episode(3, start_val=100)
        self.assertEqual(len(self.buffer.episode_boundaries), 2)

    def test_sample_returns_correct_shapes(self):
        for _ in range(4):
            self._push_episode(10)

        result = self.buffer.sample(batch_size=4)
        self.assertIsNotNone(result, "sample() returned None despite enough data")

        states, actions, rewards, next_states, dones, hiddens, indices, weights = result

        seq_len = self.config.sequence_length
        self.assertEqual(states.shape, (4, seq_len, 1, 5, 5))
        self.assertEqual(actions.shape, (4, seq_len))
        self.assertEqual(rewards.shape, (4, seq_len))
        self.assertEqual(next_states.shape, (4, seq_len, 1, 5, 5))
        self.assertEqual(dones.shape, (4, seq_len))
        self.assertEqual(len(hiddens), 4)
        self.assertEqual(len(indices), 4)
        self.assertEqual(weights.shape, (4,))

    def test_sample_sequence_length(self):
        for _ in range(4):
            self._push_episode(10)

        result = self.buffer.sample(batch_size=4)
        self.assertIsNotNone(result)
        states, actions, rewards, next_states, dones, hiddens, indices, weights = result

        self.assertEqual(states.shape[1], self.config.sequence_length)
        self.assertEqual(actions.shape[1], self.config.sequence_length)

    def test_sample_not_enough_data_returns_none(self):
        self.buffer.push(self._dummy_state(), 0, 1.0, self._dummy_state(), False, self._dummy_hidden())
        self.buffer.push(self._dummy_state(), 1, 1.0, self._dummy_state(), False, self._dummy_hidden())

        result = self.buffer.sample(batch_size=4)
        self.assertIsNone(result, "sample() should return None when not enough data")

    def test_update_priorities_no_crash(self):
        for _ in range(4):
            self._push_episode(10)

        result = self.buffer.sample(batch_size=4)
        self.assertIsNotNone(result)
        indices = result[-2]

        td_errors = np.array([0.5, 1.0, 0.2, 2.0], dtype=np.float32)
        self.buffer.update_priorities(indices, td_errors)

    def test_update_priorities_changes_distribution(self):
        for _ in range(20):
            self._push_episode(10)

        result = self.buffer.sample(batch_size=4)
        self.assertIsNotNone(result)
        indices = result[-2]

        old_total = self.buffer.tree.total()

        td_errors = np.ones(4, dtype=np.float32)
        td_errors[0] = 100.0
        self.buffer.update_priorities(indices, td_errors)

        new_total = self.buffer.tree.total()
        self.assertGreater(new_total, old_total)

    def test_update_beta_linear_anneal(self):
        initial = self.buffer.beta
        self.assertAlmostEqual(initial, self.config.per_beta_start)

        self.buffer.update_beta(0, 100)
        self.assertAlmostEqual(self.buffer.beta, self.config.per_beta_start)

        self.buffer.update_beta(50, 100)
        mid = self.config.per_beta_start + (self.config.per_beta_end - self.config.per_beta_start) * 0.5
        self.assertAlmostEqual(self.buffer.beta, mid, places=5)

        self.buffer.update_beta(100, 100)
        self.assertAlmostEqual(self.buffer.beta, self.config.per_beta_end)

    def test_hidden_state_stored_and_returned(self):
        h = np.ones(64, dtype=np.float32) * 0.5
        c = np.ones(64, dtype=np.float32) * 0.3
        hidden = (h, c)

        # Push 2 full episodes of length 10 so that sequences of length 4 can be sampled
        for ep in range(2):
            for t in range(10):
                state = self._dummy_state()
                done = (t == 9)
                self.buffer.push(state, 0, 1.0, state, done, hidden)

        result = self.buffer.sample(batch_size=2)
        self.assertIsNotNone(result)
        hiddens = result[-3]

        self.assertEqual(len(hiddens), 2)
        for hid in hiddens:
            self.assertIsInstance(hid, tuple)
            self.assertEqual(len(hid), 2)
            self.assertIsInstance(hid[0], np.ndarray)
            self.assertIsInstance(hid[1], np.ndarray)

    def test_importance_weights_are_reasonable(self):
        for _ in range(10):
            self._push_episode(10)

        result = self.buffer.sample(batch_size=4)
        self.assertIsNotNone(result)
        weights = result[-1]

        self.assertEqual(weights.shape, (4,))
        self.assertTrue(np.all(weights > 0), "All weights should be positive")
        self.assertTrue(np.all(np.isfinite(weights)), "All weights should be finite")


if __name__ == "__main__":
    unittest.main()
