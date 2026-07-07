"""
train.py -- DQN Training Script for Pacman CNN Agent (Submission 1.2)
=====================================================================
Trains the PacmanCNN model using Deep Q-Learning with:
  - Experience Replay (ReplayBuffer)
  - Target Network (soft updates)
  - Double DQN (online selects, target evaluates)
  - Epsilon-Greedy exploration with decay
  - Fog-of-war training support
  - CPU-only execution

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 200             # Custom epochs
    python train.py --epochs 500 --lr 0.0005 # Custom epochs + learning rate
    python train.py --obs-radius 5           # Train with fog of war
"""

import sys
import argparse
import random
import time
from pathlib import Path
from collections import deque, namedtuple

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.optim as optim

from model import PacmanCNN
from environment import Environment, Move


# ============================================================
# Configuration
# ============================================================

Transition = namedtuple('Transition', (
    'state', 'last_move', 'action', 'reward',
    'next_state', 'next_last_move', 'done'
))


class TrainingConfig:
    """All hyperparameters in one place."""

    def __init__(self, args=None):
        # -- Training --
        self.epochs = getattr(args, 'epochs', 100)
        self.episodes_per_epoch = getattr(args, 'episodes_per_epoch', 20)
        self.max_steps_per_episode = 200

        # -- DQN Hyperparameters --
        self.batch_size = getattr(args, 'batch_size', 64)
        self.gamma = 0.99
        self.lr = getattr(args, 'lr', 1e-3)
        self.tau = 0.005
        self.replay_buffer_size = 50000
        self.min_replay_size = 1000

        # -- Epsilon-Greedy --
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995

        # -- Model --
        self.input_shape = (1, 21, 21)
        self.n_actions = 4

        # -- Fog of War --
        self.obs_radius = getattr(args, 'obs_radius', 0)  # 0 = full visibility

        # -- Device --
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Saving --
        self.save_dir = Path(__file__).parent
        self.save_every = 10
        self.model_filename = "pacman_dqn.pt"


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Ghost Opponent (rule-based, for self-play training)
# ============================================================

class SimpleGhostOpponent:
    """
    A rule-based ghost opponent for training.
    Uses BFS to flee from Pacman, inspired by submission 1's GhostAgent logic.
    """

    def step(self, map_state, my_pos, enemy_pos, step_number):
        """Move away from Pacman using BFS distance maximization."""
        if enemy_pos is None:
            return self._random_move(my_pos, map_state)

        best_move = Move.STAY
        best_dist = -1

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if self._is_valid(nr, nc, map_state):
                dist = self._bfs_distance((nr, nc), enemy_pos, map_state)
                if dist is not None and dist > best_dist:
                    best_dist = dist
                    best_move = move

        return best_move

    def _random_move(self, pos, map_state):
        moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = pos[0] + dr, pos[1] + dc
            if self._is_valid(nr, nc, map_state):
                moves.append(move)
        return random.choice(moves) if moves else Move.STAY

    def _bfs_distance(self, start, goal, map_state):
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        h, w = map_state.shape
        while queue:
            curr, dist = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if (0 <= nr < h and 0 <= nc < w
                        and map_state[nr, nc] != 1
                        and nxt not in visited):
                    if nxt == goal:
                        return dist + 1
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return None

    def _is_valid(self, r, c, map_state):
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1


# ============================================================
# Training Environment Wrapper
# ============================================================

class TrainingEnv:
    """
    Wraps the game Environment for DQN training.
    Manages state encoding, reward computation, and fog-of-war observations.
    """

    ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    def __init__(self, pacman_speed=2, obs_radius=0):
        self.env = Environment(pacman_speed=pacman_speed)
        self.ghost = SimpleGhostOpponent()
        self.obs_radius = obs_radius
        self.last_pacman_move = None
        self.step_count = 0
        self.prev_distance = None

    def reset(self):
        """Reset environment and return initial state."""
        self.env.reset()
        self.last_pacman_move = None
        self.step_count = 0
        self.prev_distance = self._manhattan(self.env.pacman_pos, self.env.ghost_pos)
        state = self._encode_state()
        last_move_vec = self._encode_last_move()
        return state, last_move_vec

    def step(self, action_idx):
        """
        Execute one step: Pacman moves, then Ghost moves.

        Returns:
            (next_state, next_last_move, reward, done)
        """
        pacman_move = self.ALL_MOVES[action_idx]
        self.step_count += 1

        # -- Apply Pacman move --
        old_pac_pos = self.env.pacman_pos
        dr, dc = pacman_move.value
        new_r, new_c = old_pac_pos[0] + dr, old_pac_pos[1] + dc

        if self._is_valid_pos(new_r, new_c):
            self.env.pacman_pos = (new_r, new_c)
            self.last_pacman_move = pacman_move

        # -- Check capture after Pacman move --
        if self._is_caught():
            reward = 100.0
            done = True
            state = self._encode_state()
            last_move_vec = self._encode_last_move()
            return state, last_move_vec, reward, done

        # -- Apply Ghost move --
        ghost_move = self.ghost.step(
            self.env.map, self.env.ghost_pos, self.env.pacman_pos, self.step_count
        )
        gdr, gdc = ghost_move.value
        new_gr, new_gc = self.env.ghost_pos[0] + gdr, self.env.ghost_pos[1] + gdc
        if self._is_valid_pos(new_gr, new_gc):
            self.env.ghost_pos = (new_gr, new_gc)

        # -- Check capture after Ghost move --
        if self._is_caught():
            reward = 100.0
            done = True
            state = self._encode_state()
            last_move_vec = self._encode_last_move()
            return state, last_move_vec, reward, done

        # -- Compute reward --
        reward = self._compute_reward(old_pac_pos)

        # -- Check timeout --
        done = self.step_count >= self.env.max_steps
        if done:
            reward -= 50.0

        state = self._encode_state()
        last_move_vec = self._encode_last_move()
        return state, last_move_vec, reward, done

    def _compute_reward(self, old_pac_pos):
        """Shaped reward: closer=positive, farther=negative, wall bump=penalty, time=penalty."""
        current_dist = self._manhattan(self.env.pacman_pos, self.env.ghost_pos)
        reward = 0.0

        if self.prev_distance is not None:
            dist_delta = self.prev_distance - current_dist
            reward += dist_delta * 2.0

        if self.env.pacman_pos == old_pac_pos:
            reward -= 1.0

        reward -= 0.1
        self.prev_distance = current_dist
        return reward

    def _is_caught(self):
        return (self.env.pacman_pos == self.env.ghost_pos or
                self._manhattan(self.env.pacman_pos, self.env.ghost_pos) < self.env.capture_distance_threshold)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid_pos(self, r, c):
        h, w = self.env.map.shape
        return 0 <= r < h and 0 <= c < w and self.env.map[r, c] != 1

    def _encode_state(self):
        """Encode game state as float32 array [21, 21]."""
        state_map = self.env.map.copy().astype(np.float32)
        state_map[self.env.pacman_pos] = 2.0
        state_map[self.env.ghost_pos] = 3.0

        # Apply fog of war if obs_radius > 0
        if self.obs_radius > 0:
            visible = self.env.get_visible_cells_cross(
                self.env.pacman_pos, self.obs_radius
            )
            for r in range(state_map.shape[0]):
                for c in range(state_map.shape[1]):
                    if (r, c) not in visible and state_map[r, c] != 1.0:
                        state_map[r, c] = -1.0

        return state_map

    def _encode_last_move(self):
        """One-hot encode the last Pacman move [4]."""
        vec = np.zeros(4, dtype=np.float32)
        if self.last_pacman_move is not None and self.last_pacman_move in self.ALL_MOVES:
            idx = self.ALL_MOVES.index(self.last_pacman_move)
            vec[idx] = 1.0
        return vec


# ============================================================
# DQN Trainer
# ============================================================

class DQNTrainer:
    """Complete DQN training pipeline with online + target networks."""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        print(f"Pacman CNN-DQN Training Pipeline")
        print(f"  Device: {config.device}")
        print(f"  Obs radius: {config.obs_radius}")

        # -- Build Networks --
        self.online_net = PacmanCNN(config.input_shape, config.n_actions).to(self.device)
        self.target_net = PacmanCNN(config.input_shape, config.n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        total_params = sum(p.numel() for p in self.online_net.parameters())
        print(f"  Model params: {total_params:,}")
        print(f"  Est. file size: {total_params * 4 / (1024*1024):.1f} MB")

        # -- Optimizer & Loss --
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # -- Replay Buffer --
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # -- Exploration --
        self.epsilon = config.epsilon_start

        # -- Training Environment --
        self.env = TrainingEnv(pacman_speed=2, obs_radius=config.obs_radius)

        # -- Metrics --
        self.epoch_rewards = []
        self.epoch_catches = []
        self.epoch_losses = []

    def select_action(self, state, last_move_vec):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.config.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            move_t = torch.FloatTensor(last_move_vec).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t, move_t)
            return torch.argmax(q_values, dim=1).item()

    def train_step(self):
        """Sample batch and perform one gradient update."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return 0.0

        batch = self.replay_buffer.sample(self.config.batch_size)
        batch = Transition(*zip(*batch))

        states = torch.FloatTensor(np.array(batch.state)).unsqueeze(1).to(self.device)
        last_moves = torch.FloatTensor(np.array(batch.last_move)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).unsqueeze(1).to(self.device)
        next_last_moves = torch.FloatTensor(np.array(batch.next_last_move)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q = self.online_net(states, last_moves).gather(1, actions).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_q_online = self.online_net(next_states, next_last_moves)
            best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states, next_last_moves)
            next_q = next_q_target.gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self):
        """Soft update: theta_target = tau * theta_online + (1 - tau) * theta_target."""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                self.config.tau * online_param.data + (1.0 - self.config.tau) * target_param.data
            )

    def run_episode(self):
        """Run one full episode. Returns (total_reward, caught, steps, avg_loss)."""
        state, last_move_vec = self.env.reset()
        total_reward = 0.0
        losses = []
        caught = False

        for step in range(self.config.max_steps_per_episode):
            action = self.select_action(state, last_move_vec)
            next_state, next_last_move, reward, done = self.env.step(action)

            self.replay_buffer.push(
                state, last_move_vec, action, reward,
                next_state, next_last_move, float(done)
            )

            loss = self.train_step()
            if loss > 0:
                losses.append(loss)

            self.soft_update_target()

            total_reward += reward
            state = next_state
            last_move_vec = next_last_move

            if done:
                if reward > 50:
                    caught = True
                break

        avg_loss = np.mean(losses) if losses else 0.0
        return total_reward, caught, step + 1, avg_loss

    def train(self):
        """Main training loop."""
        print(f"\nStarting training: {self.config.epochs} epochs x {self.config.episodes_per_epoch} episodes")
        print(f"  Batch: {self.config.batch_size} | LR: {self.config.lr} | gamma: {self.config.gamma}")
        print(f"  epsilon: {self.config.epsilon_start} -> {self.config.epsilon_end} (decay: {self.config.epsilon_decay})")
        print("-" * 80)
        print(f"{'Epoch':>6} | {'eps':>6} | {'Avg Reward':>11} | {'Catches':>8} | {'Avg Steps':>10} | {'Avg Loss':>10} | {'Time':>6}")
        print("-" * 80)

        total_start = time.time()
        best_catch_rate = 0.0

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            rewards = []
            catches = 0
            steps_list = []
            losses = []

            for ep in range(self.config.episodes_per_epoch):
                reward, caught, steps, avg_loss = self.run_episode()
                rewards.append(reward)
                if caught:
                    catches += 1
                steps_list.append(steps)
                losses.append(avg_loss)

            # Decay epsilon
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

            # Epoch metrics
            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps_list)
            avg_loss = np.mean(losses)
            catch_rate = catches / self.config.episodes_per_epoch
            elapsed = time.time() - epoch_start

            self.epoch_rewards.append(avg_reward)
            self.epoch_catches.append(catch_rate)
            self.epoch_losses.append(avg_loss)

            print(f"{epoch:>6} | {self.epsilon:>6.3f} | {avg_reward:>+11.2f} | {catches:>4}/{self.config.episodes_per_epoch:<3} | {avg_steps:>10.1f} | {avg_loss:>10.4f} | {elapsed:>5.1f}s")

            # Save checkpoints
            if catch_rate > best_catch_rate:
                best_catch_rate = catch_rate
                self._save_model(f"best_{self.config.model_filename}")
                print(f"       -> New best catch rate: {catch_rate:.1%}")

            if epoch % self.config.save_every == 0:
                self._save_model(self.config.model_filename)

        # Final save
        self._save_model(self.config.model_filename)
        total_time = time.time() - total_start

        print("-" * 80)
        print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Best catch rate: {best_catch_rate:.1%}")
        print(f"  Final epsilon: {self.epsilon:.4f}")
        print(f"  Model saved to: {self.config.save_dir / self.config.model_filename}")

    def _save_model(self, filename):
        """Save model state dict."""
        save_path = self.config.save_dir / filename
        torch.save(self.online_net.state_dict(), save_path)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pacman CNN-DQN agent (submission 1.2)")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--episodes-per-epoch', type=int, default=20, help='Episodes per epoch (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for DQN updates (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--obs-radius', type=int, default=0, help='Pacman observation radius for fog-of-war training (0 = full visibility)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(args)
    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
