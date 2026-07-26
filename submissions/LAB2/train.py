"""
train.py -- DQN Training Script for Pacman CNN Agent
=====================================================================
Trains the PacmanCNNv2 model using Deep Q-Learning with:
  - Experience Replay (ReplayBuffer)
  - Target Network (soft updates)
  - Double DQN (online selects, target evaluates)
  - Epsilon-Greedy exploration with decay
  - Fog-of-war training support
  - GPU-accelerated training (CUDA)
  - Curriculum training (SimpleGhost → Mixed → GhostAgent phases)
  - Upper-half explore reward for fog-of-war mode
  - Stochastic start positions support

Usage:
    python train.py --curriculum --stochastic   # Full curriculum training
    python train.py --phase 1                   # Run phase 1 only
    python train.py --epochs 200                # Custom epochs (standalone)
    python train.py --obs-radius 5              # Train with fog of war
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

from model import PacmanCNN, PacmanCNNv2
from environment import Environment, Move
from agent import GhostAgent

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
        self.obs_radius = getattr(args, 'obs_radius', 0)
        
        # -- Start mode --
        self.stochastic = getattr(args, 'stochastic', False)

        # -- Curriculum phases --
        self.simple_epochs = getattr(args, 'simple_epochs', 200)
        self.mixed_epochs = getattr(args, 'mixed_epochs', 100)
        self.ghost_epochs = getattr(args, 'ghost_epochs', 50)

        # -- Device --
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Saving --
        self.save_dir = Path(__file__).parent
        self.save_every = 10
        self.model_filename = "best_pacman_dqn_v2.pt"

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
# Ghost Opponents
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
    Supports opponent switching for curriculum training.
    """

    ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    def __init__(self, pacman_speed=2, obs_radius=0, stochastic=False, opponent="simple"):
        self.env = Environment(pacman_speed=pacman_speed, deterministic_starts=not stochastic)
        self.obs_radius = obs_radius
        self.last_pacman_move = None
        self.step_count = 0
        self.prev_distance = None
        self._prev_fog_mask = None
        
        # Opponent management
        self._opponent_mode = opponent
        self._ghost_agent = None
        self._simple_ghost = SimpleGhostOpponent()
        self._init_ghost(pacman_speed)

    def _init_ghost(self, pacman_speed):
        if self._opponent_mode == "simple":
            self.ghost = self._simple_ghost
        elif self._opponent_mode == "ghost":
            if self._ghost_agent is None:
                self._ghost_agent = GhostAgent(
                    log_path=None,
                    diagnostics_enabled=False,
                    pacman_speed=pacman_speed,
                )
            self.ghost = self._ghost_agent
        elif self._opponent_mode == "mixed":
            if self._ghost_agent is None:
                self._ghost_agent = GhostAgent(
                    log_path=None,
                    diagnostics_enabled=False,
                    pacman_speed=pacman_speed,
                )

    def reset(self):
        """Reset environment and return initial state."""
        self.env.reset()
        self.last_pacman_move = None
        self.step_count = 0
        self.prev_distance = self._manhattan(self.env.pacman_pos, self.env.ghost_pos)
        
        # Mixed mode: choose opponent per episode (70% simple, 30% ghost)
        if self._opponent_mode == "mixed":
            self.ghost = self._simple_ghost if random.random() < 0.7 else self._ghost_agent
        
        state = self._encode_state()
        last_move_vec = self._encode_last_move()
        self._prev_fog_mask = state.copy()
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

        # -- Compute reward (including explore bonus) --
        explore_bonus = self._compute_explore_reward()
        reward = self._compute_reward(old_pac_pos) + explore_bonus

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

    def _compute_explore_reward(self):
        """Reward for uncovering fog cells, with 2x bonus for upper half."""
        if self.obs_radius <= 0 or self._prev_fog_mask is None:
            return 0.0

        current_state = self._encode_state()
        h, w = current_state.shape
        mid_row = h // 2
        total_bonus = 0.0

        for r in range(h):
            for c in range(w):
                was_fog = self._prev_fog_mask[r, c] == -1.0
                is_known = current_state[r, c] != -1.0
                if was_fog and is_known and current_state[r, c] != 1.0:
                    bonus = 2.0 if r < mid_row else 1.0
                    total_bonus += bonus * 0.1

        self._prev_fog_mask = current_state
        return total_bonus

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
# A* Pathfinding Helpers (for guided exploration)
# ============================================================

def _train_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _train_is_valid(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1

def _train_astar_path(start, goal, map_state):
    """A-star pathfinding from start to goal. Returns list of Move enums."""
    if start == goal:
        return []
    from heapq import heappush, heappop
    open_set = [(0, 0, start, [])]
    g_score = {start: 0}
    closed = set()
    counter = 0
    while open_set:
        _, _, current, path = heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            return path
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in closed or not _train_is_valid(nxt, map_state):
                continue
            tg = g_score[current] + 1
            if tg < g_score.get(nxt, float('inf')):
                g_score[nxt] = tg
                h = _train_manhattan(nxt, goal)
                counter += 1
                heappush(open_set, (tg + h, counter, nxt, path + [move]))
    return []

def _train_greedy_toward(my_pos, target, map_state):
    """Greedy move toward target using Manhattan distance."""
    best_move = Move.STAY
    best_dist = _train_manhattan(my_pos, target)
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        dr, dc = move.value
        nxt = (my_pos[0] + dr, my_pos[1] + dc)
        if _train_is_valid(nxt, map_state):
            d = _train_manhattan(nxt, target)
            if d < best_dist:
                best_dist = d
                best_move = move
    return best_move

# Known hiding pockets in the classic 21x21 map (upper half)
_TRAIN_UPPER_POCKETS = {
    (1, 1), (1, 2), (1, 3),
    (1, 17), (1, 18), (1, 19),
    (5, 5), (5, 6),
    (5, 14), (5, 15),
    (9, 8), (9, 9), (9, 10), (9, 11), (9, 12),
}

def _train_find_search_target(my_pos, map_state):
    """Find the best cell in the upper half to search when ghost is hidden."""
    h, w = map_state.shape
    mid_row = h // 2
    best = None
    best_score = -1.0

    for r in range(h):
        for c in range(w):
            # Only consider walkable upper-half cells
            if not _train_is_valid((r, c), map_state):
                continue
            if r >= mid_row:
                continue
            dist = _train_manhattan(my_pos, (r, c))
            if dist == 0:
                continue
            # Base score: prefer closer cells
            score = 1.0 / dist
            # Pocket bonus: 5x for known hiding spots
            if (r, c) in _TRAIN_UPPER_POCKETS:
                score *= 5.0
            if score > best_score:
                best_score = score
                best = (r, c)
    return best


# ============================================================
# DQN Trainer
# ============================================================

class DQNTrainer:
    """Complete DQN training pipeline with curriculum phases."""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        print(f"Pacman CNN-DQN v2 Training Pipeline")
        print(f"  Device: {config.device}")
        print(f"  Obs radius: {config.obs_radius}")

        # -- Build Networks (v1 for Phase 1, v2 for later phases) --
        self._use_v2 = False
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

        # -- Phase tracking --
        self.current_phase = 0
        self.cum_epoch = 0

        # -- Training Environment --
        self.env = TrainingEnv(
            pacman_speed=2,
            obs_radius=config.obs_radius,
            stochastic=config.stochastic,
            opponent="simple",
        )

        # -- Metrics --
        self.epoch_rewards = []
        self.epoch_catches = []
        self.epoch_losses = []

    def select_action(self, state, last_move_vec, ghost_pos=None):
        """Epsilon-greedy with guided exploration.

        When exploring and ghost is visible: use A-star/greedy toward ghost
        instead of purely random. This ensures the agent catches ghosts
        during training, providing positive reinforcement.
        """
        if random.random() < self.epsilon:
            if ghost_pos is not None:
                # Ghost visible: A* chase toward ghost
                pac_pos = self._get_pacman_pos(state)
                path = _train_astar_path(pac_pos, ghost_pos, state)
                if path:
                    return self.env.ALL_MOVES.index(path[0])
                return self.env.ALL_MOVES.index(_train_greedy_toward(pac_pos, ghost_pos, state))
            else:
                # Ghost hidden: systematically search upper half, prioritize corners
                pac_pos = self._get_pacman_pos(state)
                search_target = _train_find_search_target(pac_pos, state)
                if search_target is not None:
                    path = _train_astar_path(pac_pos, search_target, state)
                    if path:
                        return self.env.ALL_MOVES.index(path[0])
                    return self.env.ALL_MOVES.index(_train_greedy_toward(pac_pos, search_target, state))
                return random.randint(0, self.config.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            move_t = torch.FloatTensor(last_move_vec).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t, move_t)
            return torch.argmax(q_values, dim=1).item()

    @staticmethod
    def _get_pacman_pos(state):
        """Extract Pacman position from state array."""
        coords = __import__('numpy').argwhere(state == 2.0)
        if len(coords) > 0:
            return tuple(coords[0])
        coords = __import__('numpy').argwhere((state != 1.0) & (state != -1.0))
        return tuple(coords[0]) if len(coords) > 0 else (0, 0)

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

        current_q = self.online_net(states, last_moves).gather(1, actions).squeeze(1)

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
            ghost_pos = self.env.env.ghost_pos
            action = self.select_action(state, last_move_vec, ghost_pos)
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

    def _run_phase(self, epochs, phase_name, opponent_mode):
        """Run one curriculum phase."""
        self.current_phase += 1
        print(f"\n{'='*60}")
        print(f"Phase {self.current_phase}: {phase_name}")
        print(f"  Opponent: {opponent_mode}")
        print(f"  Epochs: {epochs} | Current epsilon: {self.epsilon:.4f}")
        print(f"{'='*60}")

        # Upgrade to v2 model at Phase 2 (when facing GhostAgent)
        if self.current_phase >= 2 and not self._use_v2:
            print("  Upgrading model to PacmanCNNv2 (BatchNorm + 3-layer)...")
            old_state = self.online_net.state_dict()
            self.online_net = PacmanCNNv2(self.config.input_shape, self.config.n_actions).to(self.device)
            self.target_net = PacmanCNNv2(self.config.input_shape, self.config.n_actions).to(self.device)
            # Copy compatible weights from v1 to v2
            # v1 has: conv1.*, conv2.*, fc1.*, fc2.*
            # v2 has: conv1.*, bn1.*, conv2.*, bn2.*, conv3.*, bn3.*, fc1.*, fc2.*
            # We can copy: conv1.*, conv2.* (only weight shape matches), fc1.*, fc2.*
            v2_sd = self.online_net.state_dict()
            for key in ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                        'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']:
                if key in old_state and key in v2_sd:
                    v2_sd[key] = old_state[key]
            self.online_net.load_state_dict(v2_sd)
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()
            self._use_v2 = True
            # Re-create optimizer with reduced learning rate for v2 stability
            v2_lr = self.config.lr * 0.5
            print(f"  Reducing LR: {self.config.lr} -> {v2_lr}")
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=v2_lr)
            print(f"  Model upgraded. Params: {sum(p.numel() for p in self.online_net.parameters()):,}")

        self.env._opponent_mode = opponent_mode
        self.env._init_ghost(pacman_speed=2)

        total_start = time.time()
        best_catch_rate = 0.0

        for epoch in range(1, epochs + 1):
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

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            self.cum_epoch += 1

            avg_reward = np.mean(rewards)
            avg_steps = np.mean(steps_list)
            avg_loss = np.mean(losses)
            catch_rate = catches / self.config.episodes_per_epoch
            elapsed = time.time() - epoch_start

            self.epoch_rewards.append(avg_reward)
            self.epoch_catches.append(catch_rate)
            self.epoch_losses.append(avg_loss)

            print(f"  Epoch {self.cum_epoch:>4} | eps {self.epsilon:.3f} | {avg_reward:>+11.2f} | {catches:>4}/{self.config.episodes_per_epoch:<3} | {avg_steps:>10.1f} | {avg_loss:>10.4f} | {elapsed:>5.1f}s")

            if catch_rate > best_catch_rate:
                best_catch_rate = catch_rate
                self._save_checkpoint(f"best_{self.config.model_filename}", best=True)

            if self.cum_epoch % self.config.save_every == 0:
                self._save_checkpoint(self.config.model_filename)

        total_time = time.time() - total_start
        print(f"  Phase complete in {total_time:.1f}s | Best catch rate: {best_catch_rate:.1%}")
        return best_catch_rate

    def train(self):
        """Main curriculum training loop."""
        total_epochs = self.config.simple_epochs + self.config.mixed_epochs + self.config.ghost_epochs
        print(f"\nCurriculum Training: {total_epochs} total epochs")
        print(f"  Phase 1 (SimpleGhost): {self.config.simple_epochs} epochs")
        print(f"  Phase 2 (Mixed 70/30): {self.config.mixed_epochs} epochs")
        print(f"  Phase 3 (GhostAgent):  {self.config.ghost_epochs} epochs")
        print(f"  Batch: {self.config.batch_size} | LR: {self.config.lr} | gamma: {self.config.gamma}")
        print(f"  epsilon: {self.config.epsilon_start} -> {self.config.epsilon_end}")
        print("-" * 60)

        total_start = time.time()

        # Phase 1: SimpleGhost only
        self._run_phase(self.config.simple_epochs, "SimpleGhost BFS Flee", "simple")

        # Phase 2: Mixed 70% SimpleGhost / 30% GhostAgent
        self._run_phase(self.config.mixed_epochs, "Mixed 70/30", "mixed")

        # Phase 3: GhostAgent only
        self._run_phase(self.config.ghost_epochs, "GhostAgent Hide-Agent", "ghost")

        # Final save
        self._save_checkpoint(self.config.model_filename)

        total_time = time.time() - total_start
        print("-" * 60)
        print(f"Curriculum training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Total epochs: {self.cum_epoch}")
        print(f"  Final epsilon: {self.epsilon:.4f}")
        print(f"  Model saved to: {self.config.save_dir / self.config.model_filename}")

    def _save_checkpoint(self, filename, best=False):
        """Save model checkpoint with metadata."""
        save_path = self.config.save_dir / filename
        checkpoint = {
            'model_state_dict': self.online_net.state_dict(),
            'epoch': self.cum_epoch,
            'epsilon': self.epsilon,
            'phase': self.current_phase,
            'training_epochs': self.cum_epoch,
        }
        torch.save(checkpoint, save_path)
        if best:
            print(f"       -> New best: {self.epoch_catches[-1]:.1%}")

# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pacman CNN-DQN v2 agent")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--episodes-per-epoch', type=int, default=20, help='Episodes per epoch (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for DQN updates (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--obs-radius', type=int, default=0, help='Pacman observation radius for fog-of-war training (0 = full visibility)')
    parser.add_argument('--stochastic', action='store_true', default=False, help='Use random (non-deterministic) start positions')
    parser.add_argument('--curriculum', action='store_true', default=False, help='Run all 3 curriculum phases sequentially')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None, help='Run a single training phase')
    parser.add_argument('--simple-epochs', type=int, default=200, help='Phase 1 epochs (default: 200)')
    parser.add_argument('--mixed-epochs', type=int, default=100, help='Phase 2 epochs (default: 100)')
    parser.add_argument('--ghost-epochs', type=int, default=50, help='Phase 3 epochs (default: 50)')
    return parser.parse_args()

def main():
    args = parse_args()
    config = TrainingConfig(args)
    trainer = DQNTrainer(config)

    if args.phase is not None:
        # Single phase mode
        phase_configs = {
            1: (config.simple_epochs, "SimpleGhost BFS Flee", "simple"),
            2: (config.mixed_epochs, "Mixed 70/30", "mixed"),
            3: (config.ghost_epochs, "GhostAgent Hide-Agent", "ghost"),
        }
        epochs, name, opponent = phase_configs[args.phase]
        # Load existing checkpoint if available
        model_path = config.save_dir / config.model_filename
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=config.device, weights_only=True)
            sd = ckpt['model_state_dict']
            is_v2 = any('conv3' in k for k in sd.keys())
            if is_v2 and args.phase is not None and args.phase >= 2:
                # Resume with v2 model
                trainer.online_net = PacmanCNNv2(config.input_shape, config.n_actions).to(config.device)
                trainer.target_net = PacmanCNNv2(config.input_shape, config.n_actions).to(config.device)
                trainer._use_v2 = True
            trainer.online_net.load_state_dict(sd)
            trainer.target_net.load_state_dict(sd)
            trainer.epsilon = ckpt.get('epsilon', config.epsilon_start)
            trainer.cum_epoch = ckpt.get('epoch', 0)
            trainer.current_phase = ckpt.get('phase', 0)
            print(f"Resumed from checkpoint (epoch {trainer.cum_epoch}, epsilon {trainer.epsilon:.4f})")
        trainer._run_phase(epochs, name, opponent)
        trainer._save_checkpoint(config.model_filename)
    else:
        trainer.train()

if __name__ == '__main__':
    main()
