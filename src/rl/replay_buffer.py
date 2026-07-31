"""Prioritized Experience Replay buffer with SumTree for DRQN Pacman Seeker agent.

Supports:
- O(log N) weighted sampling via SumTree
- DRQN sequence sampling (contiguous transitions within episodes)
- Importance sampling weights (PER beta correction)
- Linear beta annealing
"""

import numpy as np
import random
from typing import List, Optional, Tuple


class SumTree:
    """Binary tree for O(log N) weighted sampling.

    Tree is stored as a 1D array of length 2*capacity-1.
    Leaves are at indices [capacity-1 .. 2*capacity-2].
    Internal nodes store cumulative sums of child priorities.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity  # pre-allocated circular buffer
        self.ptr = 0  # next write position
        self.size = 0  # number of stored elements (0..capacity)

    def add(self, priority: float, data):
        """Store data at current write position with given priority."""
        tree_idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(tree_idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at tree_idx and propagate delta up the tree."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, s: float) -> Tuple[int, float, object]:
        """Return (tree_idx, priority, data) where cumulative sum reaches s.

        Args:
            s: value in [0, total_priority]. The cumulative sum threshold.

        Returns:
            Tuple of (leaf tree index, leaf priority, stored data).
        """
        idx = 0  # root
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                # Leaf node
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1  # right child
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def total(self) -> float:
        """Total priority sum (root node value)."""
        return self.tree[0]


class ReplayBuffer:
    """Prioritized Experience Replay buffer for DRQN training.

    Stores (state, action, reward, next_state, done, hidden) transitions,
    grouped by episode for contiguous sequence sampling.

    Attributes:
        capacity: Maximum number of stored transitions.
        tree: SumTree for O(log N) priority-weighted sampling.
        alpha: PER prioritization exponent.
        beta: Current IS correction exponent (annealed during training).
        epsilon: Small constant to avoid zero priorities.
        max_priority: Upper bound priority (updated on new pushes/updates).
        sequence_length: Number of unrolled LSTM steps per sample.
        episode_boundaries: List of (start_logical, end_logical) per completed episode.
    """

    def __init__(self, config):
        self.capacity = config.replay_buffer_capacity
        self.tree = SumTree(self.capacity)
        self.alpha = config.per_alpha
        self.beta = config.per_beta_start
        self.beta_start = config.per_beta_start
        self.beta_end = config.per_beta_end
        self.epsilon = config.per_epsilon
        self.max_priority = 1.0
        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size

        # Episode tracking
        # episode_boundaries stores (start_gidx, end_gidx) for completed episodes,
        # where gidx is a monotonically increasing global insertion index.
        self.episode_boundaries: List[Tuple[int, int]] = []
        self._next_idx: int = 0  # global insertion counter
        self._episode_start: int = 0  # gidx where current episode started
        # Maps logical global index to physical position in tree data buffer.
        # When data is evicted, the corresponding entry is removed.
        self._gidx_to_ptr: dict = {}  # gidx -> physical_ptr

    def __len__(self) -> int:
        return self.tree.size

    def push(self, state, action, reward, next_state, done, hidden):
        """Store a transition.

        New transitions receive max_priority^alpha as initial priority.

        Args:
            state: ndarray (C, H, W) observation.
            action: int, discrete action index.
            reward: float.
            next_state: ndarray (C, H, W) next observation.
            done: bool, whether episode ended.
            hidden: tuple (h, c) of LSTM hidden states (ndarrays).
        """
        gidx = self._next_idx
        self._next_idx += 1

        transition = (
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done,
            (hidden[0].copy(), hidden[1].copy()),
        )

        # If buffer is full, the oldest transition is about to be overwritten
        if self.tree.size == self.capacity:
            # The physical position being overwritten is self.tree.ptr
            overwrite_ptr = self.tree.ptr
            # Remove any mapping pointing to this position
            stale_keys = [k for k, v in self._gidx_to_ptr.items() if v == overwrite_ptr]
            for k in stale_keys:
                del self._gidx_to_ptr[k]
            # Clean up episode boundaries that reference evicted transitions
            self._prune_episode_boundaries()

        # Record physical position before add
        phys_ptr = self.tree.ptr
        self.tree.add(self.max_priority ** self.alpha, transition)
        # After add, data was written at phys_ptr; ptr has advanced
        self._gidx_to_ptr[gidx] = phys_ptr

        if done:
            self.episode_boundaries.append((self._episode_start, gidx))
            self._episode_start = gidx + 1

    def _prune_episode_boundaries(self):
        """Remove episode boundaries for episodes fully evicted from buffer."""
        if not self._gidx_to_ptr:
            self.episode_boundaries.clear()
            return
        oldest_active = min(self._gidx_to_ptr.keys())
        self.episode_boundaries = [
            (s, e) for s, e in self.episode_boundaries if e >= oldest_active
        ]

    def _find_episode(self, gidx: int) -> Optional[int]:
        """Return the index in episode_boundaries containing gidx, or None."""
        for i, (start, end) in enumerate(self.episode_boundaries):
            if start <= gidx <= end:
                return i
        return None

    def _get_transition_at_gidx(self, gidx: int) -> Optional[tuple]:
        """Retrieve transition by global index, or None if evicted."""
        phys = self._gidx_to_ptr.get(gidx)
        if phys is None:
            return None
        return self.tree.data[phys]

    def _get_valid_episodes(self) -> List[Tuple[int, int, int]]:
        """Return episodes with >= sequence_length active transitions.

        Returns:
            List of (ep_idx, start_gidx, end_gidx) tuples.
        """
        valid = []
        for ep_idx, (start, end) in enumerate(self.episode_boundaries):
            # Clamp to active range
            active_start = start
            active_end = end
            # Check if all transitions in [start, end] are still active
            if start not in self._gidx_to_ptr or end not in self._gidx_to_ptr:
                # At least partially evicted - clamp to active range
                active_keys = sorted(self._gidx_to_ptr.keys())
                active_start = max(start, active_keys[0])
                active_end = min(end, active_keys[-1])
            if active_end - active_start + 1 >= self.sequence_length:
                valid.append((ep_idx, active_start, active_end))
        return valid

    def sample(self, batch_size: int):
        """Sample batch_size sequences of length sequence_length.

        Uses proportional prioritization from the SumTree, then locates
        each sampled transition within its episode and extracts a contiguous
        sequence that does not cross episode boundaries.

        Returns:
            If enough data: tuple of (
                states:    (batch, seq_len, C, H, W) ndarray,
                actions:   (batch, seq_len) ndarray,
                rewards:   (batch, seq_len) ndarray,
                next_states: (batch, seq_len, C, H, W) ndarray,
                dones:     (batch, seq_len) ndarray,
                hiddens:   list of (h, c) initial hidden state tuples,
                indices:   list of tree leaf indices for priority update,
                weights:   (batch,) IS weights ndarray
            )
            If not enough data: None.
        """
        # Check we have enough data
        valid_eps = self._get_valid_episodes()
        if not valid_eps:
            return None

        # We need batch_size sequences
        # Calculate how many we can actually get
        total_transitions = len(self._gidx_to_ptr)
        if total_transitions < self.sequence_length:
            return None

        states_list, actions_list, rewards_list = [], [], []
        next_states_list, dones_list = [], []
        hiddens_list = []
        indices_list = []
        weights_list = []

        total_prio = self.tree.total()
        if total_prio <= 0:
            return None

        segment = total_prio / batch_size
        beta_weight = 0.0

        for i in range(batch_size):
            # Proportional prioritization sampling
            s_low = i * segment
            s_high = (i + 1) * segment
            s = random.uniform(s_low, s_high)

            tree_idx, priority, data = self.tree.get(s)

            # data is a tuple: (state, action, reward, next_state, done, hidden)
            # Find its global index
            phys_idx = tree_idx - (self.tree.capacity - 1)
            # Find gidx for this physical position
            gidx = None
            for g, p in self._gidx_to_ptr.items():
                if p == phys_idx:
                    gidx = g
                    break
            if gidx is None:
                continue  # data was evicted between get and now (rare race)

            # Find episode containing this transition
            ep_idx = self._find_episode(gidx)
            if ep_idx is None:
                continue

            ep_start, ep_end = self.episode_boundaries[ep_idx]

            # Determine valid sequence range within this episode
            # We want sequence_length transitions ending at most at ep_end
            # and starting at least at ep_start
            seq_start_gidx = max(ep_start, gidx)
            seq_end_gidx = seq_start_gidx + self.sequence_length - 1

            if seq_end_gidx > ep_end:
                # Shift left
                seq_end_gidx = ep_end
                seq_start_gidx = seq_end_gidx - self.sequence_length + 1
                if seq_start_gidx < ep_start:
                    continue  # Episode too short

            # Check all transitions in sequence are still active
            all_active = True
            for offset in range(self.sequence_length):
                if (seq_start_gidx + offset) not in self._gidx_to_ptr:
                    all_active = False
                    break
            if not all_active:
                continue

            # Extract sequence
            seq_states = []
            seq_actions = []
            seq_rewards = []
            seq_next_states = []
            seq_dones = []

            for offset in range(self.sequence_length):
                sg = seq_start_gidx + offset
                trans = self._get_transition_at_gidx(sg)
                if trans is None:
                    all_active = False
                    break
                st, ac, rw, ns, dn, _ = trans
                seq_states.append(st)
                seq_actions.append(ac)
                seq_rewards.append(rw)
                seq_next_states.append(ns)
                seq_dones.append(dn)

            if not all_active:
                continue

            # Get initial hidden state for this sequence
            first_trans = self._get_transition_at_gidx(seq_start_gidx)
            if first_trans is None:
                continue
            init_hidden = first_trans[5]  # (h, c) from the transition

            states_list.append(np.stack(seq_states, axis=0))
            actions_list.append(np.array(seq_actions, dtype=np.int64))
            rewards_list.append(np.array(seq_rewards, dtype=np.float32))
            next_states_list.append(np.stack(seq_next_states, axis=0))
            dones_list.append(np.array(seq_dones, dtype=np.float32))
            hiddens_list.append(init_hidden)
            indices_list.append(tree_idx)

            # Importance sampling weight
            # w = (N * P(i))^(-beta) / max_w
            N = self.tree.size
            p_i = priority / total_prio if total_prio > 0 else 1.0
            weight = (N * p_i) ** (-self.beta)
            beta_weight = max(beta_weight, weight)
            weights_list.append(weight)

        if len(states_list) < batch_size:
            return None

        # Normalize weights
        weights = np.array(weights_list, dtype=np.float32)
        if beta_weight > 0:
            weights = weights / beta_weight  # max-weight normalization

        return (
            np.stack(states_list, axis=0),      # (batch, seq_len, C, H, W)
            np.stack(actions_list, axis=0),      # (batch, seq_len)
            np.stack(rewards_list, axis=0),      # (batch, seq_len)
            np.stack(next_states_list, axis=0),  # (batch, seq_len, C, H, W)
            np.stack(dones_list, axis=0),        # (batch, seq_len)
            hiddens_list,                         # list of (h, c)
            indices_list,                         # list of tree_idx
            weights,                              # (batch,)
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities given new TD errors.

        Args:
            indices: list of SumTree leaf indices.
            td_errors: ndarray of TD errors (absolute values used).
        """
        td_abs = np.abs(td_errors)
        for idx, td in zip(indices, td_abs):
            priority = (float(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority

    def update_beta(self, step: int, total_steps: int):
        """Linear anneal beta from beta_start to beta_end.

        Args:
            step: Current training step (0-indexed).
            total_steps: Total training steps.
        """
        progress = min(step / max(total_steps, 1), 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
