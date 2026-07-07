"""
train.py — DQN Training Script for Pacman CNN-DQN Agent
========================================================
Trains the PacmanCNNDQN model using Deep Q-Learning with:
  - Experience Replay (ReplayBuffer)
  - Target Network (soft updates)
  - Epsilon-Greedy exploration with decay
  - Epoch-based training loop
  - CPU-only execution

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 200             # Custom epochs
    python train.py --epochs 500 --lr 0.0005 # Custom epochs + learning rate
"""

import sys
import os
import argparse
import random
import time
import math
from pathlib import Path
from collections import deque, namedtuple

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.optim as optim

from model import PacmanCNNDQN, PacmanCNNDQN_Small
from environment import Environment, Move, CellType


# ============================================================
# Configuration
# ============================================================

Transition = namedtuple('Transition', ('state', 'last_move', 'action', 'reward', 'next_state', 'next_last_move', 'done'))


class TrainingConfig:
    """All hyperparameters in one place."""

    def __init__(self, args=None):
        # ── Training ──
        self.epochs = getattr(args, 'epochs', 100)
        self.episodes_per_epoch = getattr(args, 'episodes_per_epoch', 20)
        self.max_steps_per_episode = 200

        # ── DQN Hyperparameters ──
        self.batch_size = getattr(args, 'batch_size', 64)
        self.gamma = 0.99                      # Discount factor
        self.lr = getattr(args, 'lr', 1e-3)    # Learning rate
        self.tau = 0.005                        # Soft update coefficient
        self.replay_buffer_size = 50000
        self.min_replay_size = 1000            # Min experiences before training starts

        # ── Epsilon-Greedy ──
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995             # Per-epoch decay

        # ── Model ──
        self.model_variant = getattr(args, 'model', 'large')  # 'large' or 'small'
        self.input_shape = (1, 21, 21)
        self.n_actions = 4

        # ── Device ──
        self.device = torch.device("cpu")

        # ── Saving ──
        self.save_dir = Path(__file__).parent
        self.save_every = 10  # Save checkpoint every N epochs
        self.model_filename = "pacman_cnn_dqn.pt"


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

    def __init__(self):
        self.name = "TrainingGhost"

    def step(self, map_state, my_pos, enemy_pos, step_number):
        """Move away from Pacman using BFS distance maximization."""
        if enemy_pos is None:
            # Random walk when Pacman is not visible
            return self._random_move(my_pos, map_state)

        # Find the move that maximizes distance from Pacman
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
    Manages state encoding and reward computation.
    """

    ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    def __init__(self, pacman_speed=2):
        self.env = Environment(pacman_speed=pacman_speed)
        self.ghost = SimpleGhostOpponent()
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

        Args:
            action_idx: Index into ALL_MOVES (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)

        Returns:
            (next_state, next_last_move, reward, done)
        """
        pacman_move = self.ALL_MOVES[action_idx]
        self.step_count += 1

        # ── Apply Pacman move ──
        old_pac_pos = self.env.pacman_pos
        dr, dc = pacman_move.value
        new_r, new_c = old_pac_pos[0] + dr, old_pac_pos[1] + dc

        if self._is_valid_pos(new_r, new_c):
            self.env.pacman_pos = (new_r, new_c)
            self.last_pacman_move = pacman_move
        # else: invalid move, Pacman stays put (penalized via reward)

        # ── Check capture after Pacman move ──
        if self._is_caught():
            reward = 100.0  # Big reward for catching the ghost
            done = True
            state = self._encode_state()
            last_move_vec = self._encode_last_move()
            return state, last_move_vec, reward, done

        # ── Apply Ghost move ──
        # Ghost sees full map for simplicity during training
        ghost_move = self.ghost.step(
            self.env.map, self.env.ghost_pos, self.env.pacman_pos, self.step_count
        )
        gdr, gdc = ghost_move.value
        new_gr, new_gc = self.env.ghost_pos[0] + gdr, self.env.ghost_pos[1] + gdc
        if self._is_valid_pos(new_gr, new_gc):
            self.env.ghost_pos = (new_gr, new_gc)

        # ── Check capture after Ghost move ──
        if self._is_caught():
            reward = 100.0
            done = True
            state = self._encode_state()
            last_move_vec = self._encode_last_move()
            return state, last_move_vec, reward, done

        # ── Compute reward ──
        reward = self._compute_reward(old_pac_pos)

        # ── Check timeout ──
        done = self.step_count >= self.env.max_steps

        if done:
            reward -= 50.0  # Penalty for not catching ghost

        state = self._encode_state()
        last_move_vec = self._encode_last_move()
        return state, last_move_vec, reward, done

    def _compute_reward(self, old_pac_pos):
        """
        Shaped reward to guide learning:
        - Getting closer to ghost → positive
        - Moving away → negative
        - Hitting a wall (stayed in place) → penalty
        - Time penalty each step
        """
        current_dist = self._manhattan(self.env.pacman_pos, self.env.ghost_pos)

        reward = 0.0

        # Distance-based shaping
        if self.prev_distance is not None:
            dist_delta = self.prev_distance - current_dist
            reward += dist_delta * 2.0  # Reward getting closer

        # Wall bump penalty
        if self.env.pacman_pos == old_pac_pos:
            reward -= 1.0

        # Small time penalty to encourage fast captures
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
        """Encode the game state as a float32 tensor [1, 21, 21]."""
        state_map = self.env.map.copy().astype(np.float32)
        # Mark Pacman and Ghost positions
        state_map[self.env.pacman_pos] = 2.0
        state_map[self.env.ghost_pos] = 3.0
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
    """
    Complete DQN training pipeline with:
    - Online + Target networks
    - Experience replay
    - Epsilon-greedy exploration
    - Epoch-based training with periodic evaluation
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        print(f"╔══════════════════════════════════════════════╗")
        print(f"║      Pacman CNN-DQN Training Pipeline       ║")
        print(f"║         Device: {str(config.device):>10}                 ║")
        print(f"╚══════════════════════════════════════════════╝")

        # ── Build Networks ──
        if config.model_variant == 'small':
            self.online_net = PacmanCNNDQN_Small(config.input_shape, config.n_actions).to(self.device)
            self.target_net = PacmanCNNDQN_Small(config.input_shape, config.n_actions).to(self.device)
        else:
            self.online_net = PacmanCNNDQN(config.input_shape, config.n_actions).to(self.device)
            self.target_net = PacmanCNNDQN(config.input_shape, config.n_actions).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Count and display parameters
        total_params = sum(p.numel() for p in self.online_net.parameters())
        trainable_params = sum(p.numel() for p in self.online_net.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"\n📊 Model: {config.model_variant.upper()}")
        print(f"   Total params:     {total_params:>12,}")
        print(f"   Trainable params: {trainable_params:>12,}")
        print(f"   Est. file size:   {model_size_mb:>10.1f} MB\n")

        # ── Optimizer ──
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)

        # ── Loss Function ──
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for DQN stability

        # ── Replay Buffer ──
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # ── Exploration ──
        self.epsilon = config.epsilon_start

        # ── Training environment ──
        self.env = TrainingEnv(pacman_speed=2)

        # ── Metrics ──
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
        """Sample a batch from replay buffer and perform one gradient update."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return 0.0

        batch = self.replay_buffer.sample(self.config.batch_size)
        batch = Transition(*zip(*batch))

        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).unsqueeze(1).to(self.device)       # [B, 1, 21, 21]
        last_moves = torch.FloatTensor(np.array(batch.last_move)).to(self.device)             # [B, 4]
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)                 # [B, 1]
        rewards = torch.FloatTensor(batch.reward).to(self.device)                             # [B]
        next_states = torch.FloatTensor(np.array(batch.next_state)).unsqueeze(1).to(self.device)
        next_last_moves = torch.FloatTensor(np.array(batch.next_last_move)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)                                 # [B]

        # ── Current Q values ──
        current_q = self.online_net(states, last_moves).gather(1, actions).squeeze(1)

        # ── Target Q values (Double DQN style) ──
        with torch.no_grad():
            # Use online net to select best action
            next_q_online = self.online_net(next_states, next_last_moves)
            best_next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

            # Use target net to evaluate
            next_q_target = self.target_net(next_states, next_last_moves)
            next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # ── Compute loss and update ──
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self):
        """Soft update target network: θ_target ← τ·θ_online + (1-τ)·θ_target."""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                self.config.tau * online_param.data + (1.0 - self.config.tau) * target_param.data
            )

    def run_episode(self):
        """Run one full episode and return (total_reward, caught, steps, avg_loss)."""
        state, last_move_vec = self.env.reset()
        total_reward = 0.0
        losses = []
        caught = False

        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.select_action(state, last_move_vec)

            # Step environment
            next_state, next_last_move, reward, done = self.env.step(action)

            # Store transition
            self.replay_buffer.push(
                state, last_move_vec, action, reward,
                next_state, next_last_move, float(done)
            )

            # Train
            loss = self.train_step()
            if loss > 0:
                losses.append(loss)

            # Soft update target network
            self.soft_update_target()

            total_reward += reward
            state = next_state
            last_move_vec = next_last_move

            if done:
                if reward > 50:  # Caught ghost
                    caught = True
                break

        avg_loss = np.mean(losses) if losses else 0.0
        return total_reward, caught, step + 1, avg_loss

    def train(self):
        """Main training loop: epochs × episodes."""
        print(f"🚀 Starting training: {self.config.epochs} epochs × {self.config.episodes_per_epoch} episodes/epoch")
        print(f"   Batch size: {self.config.batch_size} | LR: {self.config.lr} | γ: {self.config.gamma}")
        print(f"   ε: {self.config.epsilon_start} → {self.config.epsilon_end} (decay: {self.config.epsilon_decay})")
        print("─" * 80)
        print(f"{'Epoch':>6} │ {'ε':>6} │ {'Avg Reward':>11} │ {'Catches':>8} │ {'Avg Steps':>10} │ {'Avg Loss':>10} │ {'Time':>6}")
        print("─" * 80)

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
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )

            # Epoch metrics
            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps_list)
            avg_loss = np.mean(losses)
            catch_rate = catches / self.config.episodes_per_epoch
            elapsed = time.time() - epoch_start

            self.epoch_rewards.append(avg_reward)
            self.epoch_catches.append(catch_rate)
            self.epoch_losses.append(avg_loss)

            # Print progress
            print(f"{epoch:>6} │ {self.epsilon:>6.3f} │ {avg_reward:>+11.2f} │ {catches:>4}/{self.config.episodes_per_epoch:<3} │ {avg_steps:>10.1f} │ {avg_loss:>10.4f} │ {elapsed:>5.1f}s")

            # Save checkpoint
            if epoch % self.config.save_every == 0 or catch_rate > best_catch_rate:
                if catch_rate > best_catch_rate:
                    best_catch_rate = catch_rate
                    self._save_model(f"best_{self.config.model_filename}")
                    print(f"       └── 🏆 New best catch rate: {catch_rate:.1%} — model saved!")

                if epoch % self.config.save_every == 0:
                    self._save_model(self.config.model_filename)

        # ── Final save ──
        self._save_model(self.config.model_filename)
        total_time = time.time() - total_start

        print("─" * 80)
        print(f"✅ Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   Best catch rate: {best_catch_rate:.1%}")
        print(f"   Final ε: {self.epsilon:.4f}")
        print(f"   Model saved to: {self.config.save_dir / self.config.model_filename}")
        print(f"   Buffer size: {len(self.replay_buffer)} transitions")

        self._print_summary()

    def _save_model(self, filename):
        """Save model state dict."""
        save_path = self.config.save_dir / filename
        torch.save(self.online_net.state_dict(), save_path)

    def _print_summary(self):
        """Print training summary with ASCII chart."""
        print("\n📈 Training Summary")
        print("═" * 60)

        # Show last 10 epoch stats
        n = min(10, len(self.epoch_rewards))
        print(f"\nLast {n} epochs:")
        print(f"  {'Epoch':>6} │ {'Reward':>10} │ {'Catch Rate':>11} │ {'Loss':>10}")
        print(f"  {'─'*6}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*10}")

        start_idx = len(self.epoch_rewards) - n
        for i in range(start_idx, len(self.epoch_rewards)):
            epoch = i + 1
            print(f"  {epoch:>6} │ {self.epoch_rewards[i]:>+10.2f} │ {self.epoch_catches[i]:>10.1%} │ {self.epoch_losses[i]:>10.4f}")

        # Simple ASCII bar chart for catch rate
        print(f"\n📊 Catch Rate Trend (last {n} epochs):")
        for i in range(start_idx, len(self.epoch_catches)):
            bar_len = int(self.epoch_catches[i] * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  E{i+1:>4} │{bar}│ {self.epoch_catches[i]:.0%}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Pacman CNN-DQN agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                          # Default: 100 epochs, large model
  python train.py --epochs 200             # More epochs
  python train.py --model small --epochs 50 --lr 0.001
  python train.py --epochs 500 --batch-size 128 --episodes 30
        """
    )
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--episodes-per-epoch', type=int, default=20,
                        help='Episodes per epoch (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for DQN updates (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--model', type=str, choices=['large', 'small'], default='large',
                        help='Model variant: large (~12MB) or small (~7.6MB)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(args)
    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
