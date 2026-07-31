"""DRQN (Deep Recurrent Q-Network) and Double DQN Agent for Pacman Seeker.

Implements a CNN+LSTM architecture for partially observable multi-agent
Pacman environment. Uses Double DQN with clipped importance-sampling
weights for stable training.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    raise ImportError(
        "PyTorch is required for DQN agent. "
        "Install with: pip install torch"
    )


class DRQNNetwork(nn.Module):
    """CNN + LSTM network for processing sequences of game frames.

    Architecture:
        Conv2d(24->32, 3x3) -> ReLU
        Conv2d(32->64, 3x3) -> ReLU
        Conv2d(64->64, 3x3) -> ReLU
        Flatten -> Linear(28224, 128) -> ReLU
        LSTM(128, 128, batch_first=False)
        Linear(128, 64) -> ReLU
        Linear(64, n_actions)

    Input shapes:
        Single step: (batch, C, H, W) -> output (batch, n_actions)
        Sequence:    (seq_len, batch, C, H, W) -> output (seq_len, batch, n_actions)
    """

    def __init__(self, config):
        super().__init__()
        in_channels = config.n_frames * config.n_channels  # 24
        h, w = config.map_height, config.map_width  # 21, 21

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        conv_out = 64 * h * w  # 64 * 21 * 21 = 28224
        self.fc1 = nn.Linear(conv_out, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=False)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, config.n_actions)

    def forward(self, x, hidden):
        """Forward pass through CNN + LSTM.

        Args:
            x: Tensor of shape (batch, C, H, W) or (seq_len, batch, C, H, W).
            hidden: Tuple of (h, c) each of shape (1, batch, 128).

        Returns:
            q_values: (batch, n_actions) or (seq_len, batch, n_actions).
            new_hidden: Tuple (h, c) each (1, batch, 128).
        """
        orig_ndim = x.dim()

        if orig_ndim == 4:
            # Single step: (batch, C, H, W)
            batch_size = x.shape[0]

            feats = torch.relu(self.conv1(x))
            feats = torch.relu(self.conv2(feats))
            feats = torch.relu(self.conv3(feats))
            feats = feats.reshape(batch_size, -1)
            feats = torch.relu(self.fc1(feats))
            # Add sequence dimension for LSTM
            feats = feats.unsqueeze(0)  # (1, batch, 128)
            lstm_out, new_hidden = self.lstm(feats, hidden)
            lstm_out = lstm_out.squeeze(0)  # (batch, 128)
            out = torch.relu(self.fc2(lstm_out))
            q = self.out(out)  # (batch, n_actions)
            return q, new_hidden

        elif orig_ndim == 5:
            # Sequence: (seq_len, batch, C, H, W)
            seq_len, batch_size = x.shape[0], x.shape[1]

            # Merge seq_len and batch for CNN processing
            x_flat = x.reshape(seq_len * batch_size, *x.shape[2:])
            feats = torch.relu(self.conv1(x_flat))
            feats = torch.relu(self.conv2(feats))
            feats = torch.relu(self.conv3(feats))
            feats = feats.reshape(seq_len * batch_size, -1)
            feats = torch.relu(self.fc1(feats))
            # Reshape for LSTM
            feats = feats.reshape(seq_len, batch_size, -1)  # (seq_len, batch, 128)
            lstm_out, new_hidden = self.lstm(feats, hidden)
            # Reshape for FC layers
            lstm_out = lstm_out.reshape(seq_len * batch_size, -1)  # (seq_len*batch, 128)
            out = torch.relu(self.fc2(lstm_out))
            q = self.out(out)  # (seq_len*batch, n_actions)
            q = q.reshape(seq_len, batch_size, -1)  # (seq_len, batch, n_actions)
            return q, new_hidden

        else:
            raise ValueError(
                f"Expected 4D or 5D input, got {orig_ndim}D tensor of shape {x.shape}"
            )


class DQNAgent:
    """Double DQN agent with DRQN (CNN+LSTM) architecture.

    Uses online and target networks with soft/hard target updates.
    Supports prioritized experience replay and importance-sampling weights.

    Attributes:
        online_net: DRQNNetwork used for action selection and gradient updates.
        target_net: DRQNNetwork used for Double DQN target computation.
        optimizer: Adam optimizer.
        n_actions: Number of discrete actions.
        gamma: Discount factor.
        lstm_hidden_size: LSTM hidden state dimension (128).
        train_steps: Counter of training steps taken.
    """

    def __init__(self, config):
        """Initialize DQNAgent with config.

        Args:
            config: Config dataclass with DQN hyperparameters.
        """
        self.online_net = DRQNNetwork(config)
        self.target_net = DRQNNetwork(config)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net always in eval mode

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.n_actions = config.n_actions
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.lstm_hidden_size = 128
        self.train_steps = 0
        self.device = config.device

        # Move networks to device
        self.online_net.to(self.device)
        self.target_net.to(self.device)

    def get_initial_hidden(self, batch_size=1):
        """Return zero-initialized LSTM hidden state as numpy arrays.

        Args:
            batch_size: Number of parallel environments.

        Returns:
            Tuple of (h, c) numpy arrays, each shape (1, batch_size, 128).
        """
        h = np.zeros((1, batch_size, self.lstm_hidden_size), dtype=np.float32)
        c = np.zeros((1, batch_size, self.lstm_hidden_size), dtype=np.float32)
        return (h, c)

    def get_action(self, state, hidden, epsilon):
        """Select action using epsilon-greedy policy.

        Args:
            state: numpy array of shape (C, H, W) — single stacked observation.
            hidden: Tuple of (h, c) numpy arrays each (1, batch, 128).
            epsilon: Exploration probability in [0, 1].

        Returns:
            action: int in [0, n_actions).
            new_hidden: Tuple of (h, c) numpy arrays each (1, batch, 128).
        """
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        # state_t: (1, C, H, W)

        h_t = torch.from_numpy(np.asarray(hidden[0], dtype=np.float32)).to(self.device)
        c_t = torch.from_numpy(np.asarray(hidden[1], dtype=np.float32)).to(self.device)
        hidden_t = (h_t, c_t)

        self.online_net.eval()
        with torch.no_grad():
            q_values, new_hidden_t = self.online_net(state_t, hidden_t)
        self.online_net.train()

        q_values = q_values.squeeze(0)  # (batch, n_actions) or (n_actions,) for batch=1

        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            if q_values.dim() == 1:
                action = int(q_values.argmax(dim=-1).item())
            else:
                action = int(q_values[0].argmax(dim=-1).item())

        new_hidden = (
            new_hidden_t[0].cpu().numpy(),
            new_hidden_t[1].cpu().numpy(),
        )

        return action, new_hidden

    def train_step(self, replay_buffer):
        """Sample a batch from replay buffer and perform one Double DQN update.

        Uses DRQN sequence unrolling: the hidden state stored in the buffer
        (captured at the start of the sequence) is used to initialise the
        online network LSTM. Target network uses zero hidden state (standard
        DRQN approach).

        Args:
            replay_buffer: ReplayBuffer instance with .sample(batch_size) method.

        Returns:
            float: Scalar loss value for this training step, or 0.0 if
                   the buffer does not have enough data.
        """
        batch = replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones, hiddens, indices, weights = batch

        # Convert to tensors
        # states: (batch, seq_len, C, H, W) -> (seq_len, batch, C, H, W)
        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).permute(1, 0, 2, 3, 4).to(self.device)
        actions_t = torch.from_numpy(np.asarray(actions, dtype=np.int64)).T.to(self.device)  # (seq_len, batch)
        rewards_t = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).T.to(self.device)
        next_states_t = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).permute(1, 0, 2, 3, 4).to(self.device)
        dones_t = torch.from_numpy(np.asarray(dones, dtype=np.float32)).T.to(self.device)
        weights_t = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(self.device)

        seq_len = states_t.shape[0]
        actual_batch = states_t.shape[1]

        # Build initial hidden state for online network from stored hiddens
        # Each stored hidden is (h, c) where h/c shape is (1, 1, hidden_size)
        h_list = []
        c_list = []
        for hid in hiddens:
            h_arr = np.asarray(hid[0], dtype=np.float32)
            c_arr = np.asarray(hid[1], dtype=np.float32)
            # Handle both (hidden_size,) and (1, 1, hidden_size) shapes
            if h_arr.ndim == 1:
                h_arr = h_arr.reshape(1, 1, -1)
            if c_arr.ndim == 1:
                c_arr = c_arr.reshape(1, 1, -1)
            h_list.append(torch.from_numpy(h_arr))
            c_list.append(torch.from_numpy(c_arr))
        init_h = torch.cat(h_list, dim=1).to(self.device)  # (1, batch, hidden_size)
        init_c = torch.cat(c_list, dim=1).to(self.device)
        init_hidden = (init_h, init_c)

        # Zero hidden state for target network (standard DRQN approach)
        zero_hidden = (
            torch.zeros(1, actual_batch, self.lstm_hidden_size, device=self.device),
            torch.zeros(1, actual_batch, self.lstm_hidden_size, device=self.device),
        )

        # ---- Forward pass: Q_online for the sequence ----
        q_online_seq, _ = self.online_net(states_t, init_hidden)
        # q_online_seq: (seq_len, batch, n_actions)

        # Q-values for the taken actions
        q_online = q_online_seq.gather(dim=-1, index=actions_t.unsqueeze(-1)).squeeze(-1)
        # q_online: (seq_len, batch)

        # ---- Double DQN targets ----
        with torch.no_grad():
            # Online network selects actions on next states (zero hidden for DRQN)
            q_online_next, _ = self.online_net(next_states_t, zero_hidden)
            best_actions = q_online_next.argmax(dim=-1)  # (seq_len, batch)

            # Target network evaluates those actions
            q_target_next, _ = self.target_net(next_states_t, zero_hidden)
            q_target_best = q_target_next.gather(
                dim=-1, index=best_actions.unsqueeze(-1)
            ).squeeze(-1)  # (seq_len, batch)

            # Bellman target: r + gamma * Q_target(s', a*) * (1 - done)
            y = rewards_t + self.gamma * q_target_best * (1.0 - dones_t)

        # ---- Loss with importance-sampling weights ----
        td_errors = y - q_online  # (seq_len, batch)
        loss_per_step = td_errors ** 2
        # Mean over sequence dimension, weight each sample
        loss_per_sample = loss_per_step.mean(dim=0)  # (batch,)
        weighted_loss = (loss_per_sample * weights_t).mean()

        # ---- Backward pass ----
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping (max norm 10.0)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- Update priorities ----
        td_abs = td_errors.abs().mean(dim=0).detach().cpu().numpy()  # (batch,)
        replay_buffer.update_priorities(indices, td_abs)

        self.train_steps += 1
        return float(weighted_loss.item())

    def sync_target(self):
        """Hard-copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        """Save agent state to a checkpoint file.

        Args:
            path: File path to save the checkpoint.
        """
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path):
        """Load agent state from a checkpoint file.

        Args:
            path: File path to load the checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
