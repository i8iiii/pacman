'''Tests for DRQNNetwork and DQNAgent (Task 7).

Note: PyTorch is not yet a required dependency (Task 10 adds it).
Tests are skipped gracefully if torch is unavailable.
'''

import sys
from pathlib import Path

PACMAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACMAN_ROOT / "src"))

import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDRQNNetwork(unittest.TestCase):
    """Verify DRQNNetwork forward pass shapes and values."""

    @classmethod
    def setUpClass(cls):
        if not TORCH_AVAILABLE:
            raise unittest.SkipTest("PyTorch not installed; skipping DQN tests")

        from rl.config import Config
        from rl.dqn_agent import DRQNNetwork

        cls.config = Config()
        cls.net = DRQNNetwork(cls.config)
        cls.net.eval()

    def _get_dummy_hidden(self, batch_size=1):
        """Return zero-initialized hidden state matching LSTM shape."""
        h = torch.zeros(1, batch_size, 128)
        c = torch.zeros(1, batch_size, 128)
        return (h, c)

    def test_forward_single_step_output_shape(self):
        """DRQNNetwork produces (batch, 5) for a single (batch, C, H, W) input."""
        x = torch.randn(3, 24, 21, 21)
        hidden = self._get_dummy_hidden(batch_size=3)
        q, new_hidden = self.net(x, hidden)
        self.assertEqual(q.shape, (3, 5))
        self.assertIsInstance(new_hidden, tuple)
        self.assertEqual(len(new_hidden), 2)
        self.assertEqual(new_hidden[0].shape, (1, 3, 128))
        self.assertEqual(new_hidden[1].shape, (1, 3, 128))

    def test_forward_single_batch_output_shape(self):
        """DRQNNetwork produces (1, 5) for batch_size=1."""
        x = torch.randn(1, 24, 21, 21)
        hidden = self._get_dummy_hidden(batch_size=1)
        q, new_hidden = self.net(x, hidden)
        self.assertEqual(q.shape, (1, 5))

    def test_forward_sequence_output_shape(self):
        """DRQNNetwork handles (seq_len, batch, C, H, W) input."""
        x = torch.randn(8, 4, 24, 21, 21)
        hidden = self._get_dummy_hidden(batch_size=4)
        q, new_hidden = self.net(x, hidden)
        self.assertEqual(q.shape, (8, 4, 5))
        self.assertEqual(new_hidden[0].shape, (1, 4, 128))
        self.assertEqual(new_hidden[1].shape, (1, 4, 128))

    def test_forward_deterministic_no_grad(self):
        """Forward pass produces identical output with same input (eval mode)."""
        x = torch.randn(2, 24, 21, 21)
        hidden = self._get_dummy_hidden(batch_size=2)
        with torch.no_grad():
            q1, _ = self.net(x, hidden)
            q2, _ = self.net(x, hidden)
        self.assertTrue(torch.allclose(q1, q2))

    def test_has_trainable_parameters(self):
        """DRQNNetwork has a reasonable number of trainable parameters."""
        total = sum(p.numel() for p in self.net.parameters())
        self.assertGreater(total, 100_000)
        self.assertLess(total, 20_000_000)


class TestDQNAgent(unittest.TestCase):
    """Verify DQNAgent: action selection, target sync, save/load."""

    @classmethod
    def setUpClass(cls):
        if not TORCH_AVAILABLE:
            raise unittest.SkipTest("PyTorch not installed; skipping DQN tests")

        from rl.config import Config
        from rl.dqn_agent import DQNAgent

        cls.config = Config()
        cls.agent = DQNAgent(cls.config)

    def _dummy_state(self):
        """Return a dummy stacked state: (24, 21, 21) float32 ndarray."""
        return np.random.randn(24, 21, 21).astype(np.float32)

    def test_get_action_returns_valid_action_and_hidden(self):
        """get_action returns an int action within [0, n_actions) and new hidden."""
        state = self._dummy_state()
        hidden = self.agent.get_initial_hidden(batch_size=1)
        action, new_hidden = self.agent.get_action(state, hidden, epsilon=0.0)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.config.n_actions)
        self.assertIsInstance(new_hidden, tuple)
        self.assertEqual(len(new_hidden), 2)
        self.assertIsInstance(new_hidden[0], np.ndarray)
        self.assertIsInstance(new_hidden[1], np.ndarray)

    def test_get_action_epsilon_zero_returns_argmax(self):
        """With epsilon=0, get_action is fully greedy (deterministic)."""
        state = self._dummy_state()
        hidden = self.agent.get_initial_hidden(batch_size=1)
        actions = []
        for _ in range(10):
            a, hidden = self.agent.get_action(state, hidden, epsilon=0.0)
            actions.append(a)
        self.assertEqual(len(set(actions)), 1)

    def test_get_action_epsilon_one_produces_diverse_actions(self):
        """With epsilon=1, get_action produces varied random actions."""
        state = self._dummy_state()
        hidden = self.agent.get_initial_hidden(batch_size=1)
        actions = set()
        for _ in range(50):
            a, hidden = self.agent.get_action(state, hidden, epsilon=1.0)
            actions.add(a)
        self.assertGreater(len(actions), 1)

    def test_get_action_hidden_state_evolves(self):
        """Hidden state changes across sequential calls (LSTM state updates)."""
        state = self._dummy_state()
        hidden = self.agent.get_initial_hidden(batch_size=1)
        initial_h = hidden[0].copy()
        for _ in range(5):
            _, hidden = self.agent.get_action(state, hidden, epsilon=0.0)
        self.assertFalse(np.allclose(initial_h, hidden[0]))

    def test_get_initial_hidden_shape(self):
        """get_initial_hidden returns correctly-shaped zero tensors."""
        hidden = self.agent.get_initial_hidden(batch_size=4)
        self.assertIsInstance(hidden, tuple)
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].shape, (1, 4, 128))
        self.assertEqual(hidden[1].shape, (1, 4, 128))
        self.assertTrue(np.allclose(hidden[0], 0.0))
        self.assertTrue(np.allclose(hidden[1], 0.0))

    def test_sync_target_copies_weights(self):
        """sync_target copies online network weights to target network."""
        with torch.no_grad():
            for param in self.agent.online_net.parameters():
                param.add_(0.1)
        online_params = list(self.agent.online_net.parameters())
        target_params = list(self.agent.target_net.parameters())
        weights_differ = any(
            not torch.allclose(o, t)
            for o, t in zip(online_params, target_params)
        )
        self.assertTrue(weights_differ)
        self.agent.sync_target()
        for o, t in zip(self.agent.online_net.parameters(),
                         self.agent.target_net.parameters()):
            self.assertTrue(torch.allclose(o, t))

    def test_sync_target_idempotent(self):
        """Calling sync_target twice doesn't change anything on second call."""
        self.agent.sync_target()
        target_weights_before = {
            name: param.clone()
            for name, param in self.agent.target_net.named_parameters()
        }
        self.agent.sync_target()
        for name, param in self.agent.target_net.named_parameters():
            self.assertTrue(torch.allclose(target_weights_before[name], param))

    def test_save_load_roundtrip_preserves_weights(self):
        """save + load roundtrip preserves online network weights."""
        import tempfile
        import os
        online_weights_before = {
            name: param.clone()
            for name, param in self.agent.online_net.named_parameters()
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            self.agent.save(path)
            self.assertTrue(os.path.isfile(path))
            with torch.no_grad():
                for param in self.agent.online_net.parameters():
                    param.add_(0.5)
            weights_changed = any(
                not torch.allclose(online_weights_before[name], param)
                for name, param in self.agent.online_net.named_parameters()
            )
            self.assertTrue(weights_changed)
            self.agent.load(path)
            for name, param in self.agent.online_net.named_parameters():
                self.assertTrue(
                    torch.allclose(online_weights_before[name], param, atol=1e-6)
                )

    def test_save_load_preserves_train_steps(self):
        """save/load roundtrip preserves train_steps counter."""
        import tempfile
        import os
        self.agent.train_steps = 42
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            self.agent.save(path)
            self.agent.train_steps = 0
            self.agent.load(path)
        self.assertEqual(self.agent.train_steps, 42)

    def test_save_load_preserves_optimizer_state(self):
        """save/load roundtrip preserves optimizer state."""
        import tempfile
        import os
        state = torch.randn(1, 24, 21, 21)
        hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
        q_values, _ = self.agent.online_net(state, hidden)
        loss = q_values.sum()
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            self.agent.save(path)
            from rl.dqn_agent import DQNAgent
            agent2 = DQNAgent(self.config)
            agent2.load(path)
        self.assertGreater(len(agent2.optimizer.state), 0)

    def test_train_step_returns_float_loss(self):
        """train_step runs a forward/backward pass and returns a float loss."""
        from rl.replay_buffer import ReplayBuffer
        config = self.config
        config.replay_buffer_capacity = 500
        config.sequence_length = 4
        config.batch_size = 4
        buffer = ReplayBuffer(config)
        hidden_size = self.agent.lstm_hidden_size
        for ep in range(10):
            for t in range(10):
                state = np.random.randn(24, 21, 21).astype(np.float32)
                next_state = np.random.randn(24, 21, 21).astype(np.float32)
                done = (t == 9)
                h = np.zeros(hidden_size, dtype=np.float32)
                c = np.zeros(hidden_size, dtype=np.float32)
                buffer.push(state, t % 5, 1.0, next_state, done, (h, c))
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(config)
        loss = agent.train_step(buffer)
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))

    def test_train_step_increments_train_steps(self):
        """Each train_step call increments self.train_steps."""
        from rl.replay_buffer import ReplayBuffer
        config = self.config
        config.replay_buffer_capacity = 500
        config.sequence_length = 4
        config.batch_size = 4
        buffer = ReplayBuffer(config)
        hidden_size = self.agent.lstm_hidden_size
        for ep in range(10):
            for t in range(10):
                state = np.random.randn(24, 21, 21).astype(np.float32)
                next_state = np.random.randn(24, 21, 21).astype(np.float32)
                done = (t == 9)
                h = np.zeros(hidden_size, dtype=np.float32)
                c = np.zeros(hidden_size, dtype=np.float32)
                buffer.push(state, t % 5, 1.0, next_state, done, (h, c))
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(config)
        before = agent.train_steps
        agent.train_step(buffer)
        after = agent.train_steps
        self.assertEqual(after, before + 1)

    def test_train_step_handles_no_sample(self):
        """train_step returns 0.0 when buffer doesn't have enough data."""
        from rl.replay_buffer import ReplayBuffer
        config = self.config
        config.replay_buffer_capacity = 500
        config.sequence_length = 4
        config.batch_size = 4
        buffer = ReplayBuffer(config)
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(config)
        loss = agent.train_step(buffer)
        self.assertEqual(loss, 0.0)

    def test_double_dqn_loss_computation(self):
        """Verify Double DQN loss is computed without NaN in a controlled scenario."""
        from rl.replay_buffer import ReplayBuffer
        config = self.config
        config.replay_buffer_capacity = 500
        config.sequence_length = 4
        config.batch_size = 4
        buffer = ReplayBuffer(config)
        hidden_size = self.agent.lstm_hidden_size
        for ep in range(5):
            for t in range(10):
                state = np.zeros((24, 21, 21), dtype=np.float32)
                next_state = np.zeros((24, 21, 21), dtype=np.float32)
                done = (t == 9)
                h = np.zeros(hidden_size, dtype=np.float32)
                c = np.zeros(hidden_size, dtype=np.float32)
                buffer.push(state, 0, 0.0, next_state, done, (h, c))
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(config)
        for _ in range(3):
            loss = agent.train_step(buffer)
            self.assertTrue(np.isfinite(loss))

    def test_gradient_clipping_applied(self):
        """Verify gradient clipping is active (no NaN after many steps)."""
        from rl.replay_buffer import ReplayBuffer
        config = self.config
        config.replay_buffer_capacity = 500
        config.sequence_length = 4
        config.batch_size = 4
        buffer = ReplayBuffer(config)
        hidden_size = self.agent.lstm_hidden_size
        for ep in range(10):
            for t in range(10):
                state = np.random.randn(24, 21, 21).astype(np.float32) * 5.0
                next_state = np.random.randn(24, 21, 21).astype(np.float32) * 5.0
                done = (t == 9)
                h = np.zeros(hidden_size, dtype=np.float32)
                c = np.zeros(hidden_size, dtype=np.float32)
                buffer.push(state, 0, 1.0, next_state, done, (h, c))
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(config)
        for _ in range(5):
            loss = agent.train_step(buffer)
            self.assertTrue(np.isfinite(loss))


class TestDQNAgentEdgeCases(unittest.TestCase):
    """Edge cases for DQNAgent."""

    @classmethod
    def setUpClass(cls):
        if not TORCH_AVAILABLE:
            raise unittest.SkipTest("PyTorch not installed; skipping DQN tests")

    def test_agent_initial_train_steps_zero(self):
        """New agent starts with train_steps = 0."""
        from rl.config import Config
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(Config())
        self.assertEqual(agent.train_steps, 0)

    def test_agent_target_starts_identical_to_online(self):
        """Target network weights match online at initialization."""
        from rl.config import Config
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(Config())
        for o, t in zip(agent.online_net.parameters(),
                         agent.target_net.parameters()):
            self.assertTrue(torch.allclose(o, t))


if __name__ == "__main__":
    unittest.main()
