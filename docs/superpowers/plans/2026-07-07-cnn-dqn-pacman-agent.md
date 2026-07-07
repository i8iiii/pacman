# CNN-DQN Pacman Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CNN-DQN Pacman agent in `submissions/1.2/` that uses DQN as primary decision-maker with A* fallback, plus a rule-based GhostAgent from submission 1.

**Architecture:** Small 2-layer CNN-DQN model outputs Q-values for 4 actions. Agent uses DQN when confident and enemy is visible; falls back to A* pathfinding when DQN is uncertain or enemy is hidden by fog. GhostAgent inlines submission 1's rule-based logic (BFS flee, dead-end detection, simulation).

**Tech Stack:** Python 3.14, PyTorch (CPU), NumPy

## Global Constraints

- Class names must be exactly `PacmanAgent` and `GhostAgent` (case-sensitive)
- Both classes must inherit from base classes in `agent_interface.py`
- `PacmanAgent.__init__` receives `pacman_speed` kwarg; `GhostAgent.__init__` receives no special kwargs
- Agent must return `Move` enum or `(Move, steps)` tuple for Pacman
- Agent step must complete well under 1.0s timeout
- Imports from `src/` use path manipulation: `sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))`
- Map values: `0` = empty, `1` = wall, `-1` = unseen (fog of war)
- All code in flat `submissions/1.2/` folder — no subdirectories
- GhostAgent helper functions must be inlined (loader only adds agent's own folder to `sys.path`)
- Model file `pacman_dqn.pt` must be loadable on CPU with `torch.load(map_location='cpu')`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `submissions/1.2/model.py` | `PacmanCNN` PyTorch model class — CNN feature extractor + DQN head |
| `submissions/1.2/agent.py` | `PacmanAgent` (DQN + A* + fallback logic) and `GhostAgent` (rule-based from submission 1) + all inline helpers |
| `submissions/1.2/train.py` | DQN training pipeline — replay buffer, training env, trainer, CLI |
| `submissions/1.2/README.md` | Documentation of architecture, training, and usage |

---

### Task 1: Create model.py — PacmanCNN

**Files:**
- Create: `submissions/1.2/model.py`

**Interfaces:**
- Produces: `PacmanCNN(input_shape=(1, 21, 21), n_actions=4)` class
  - `forward(self, x: Tensor[B, 1, 21, 21], last_move_vec: Tensor[B, 4]) -> Tensor[B, 4]`

- [ ] **Step 1: Create `submissions/1.2/` directory and write `model.py`**

```python
# model.py — CNN-DQN Model for Pacman Agent (Submission 1.2)
# Small 2-layer CNN feature extractor + DQN head
# Designed for CPU inference under 1ms

import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanCNN(nn.Module):
    """
    Small CNN-DQN for Pacman agent.

    Architecture:
        Input: map_state [B, 1, 21, 21] + last_move one-hot [B, 4]

        CNN Feature Extractor:
            Conv2d(1→32, 3×3, stride=1, pad=1) → ReLU  → [B, 32, 21, 21]
            Conv2d(32→64, 3×3, stride=2, pad=1) → ReLU  → [B, 64, 11, 11]
            Flatten                                        → [B, 7744]

        DQN Head:
            Linear(7744 + 4 → 256) → ReLU → Dropout(0.1)
            Linear(256 → 4)        → Linear (raw Q-values)

        Output: Q-values for [UP, DOWN, LEFT, RIGHT]
    """

    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanCNN, self).__init__()

        # ── CNN Feature Extractor ──
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   # → [B, 32, 21, 21]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # → [B, 64, 11, 11]

        # 64 * 11 * 11 = 7744
        self.feature_size = 64 * 11 * 11

        # ── DQN Head ──
        # Concatenate CNN features (7744) + last_move one-hot (4) = 7748
        self.fc1 = nn.Linear(self.feature_size + n_actions, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, n_actions)  # Output: Q-values (linear activation)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for ReLU layers, Xavier for the output layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc2:
                    # Output layer: Xavier init for linear activation
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Hidden layers: Kaiming init for ReLU activation
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, last_move_vec):
        """
        Forward pass.

        Args:
            x:             Map state tensor [B, 1, 21, 21]
            last_move_vec: One-hot encoded last move [B, 4]

        Returns:
            Q-values for each action [B, 4]
        """
        # ── CNN Feature Extraction ──
        x = F.relu(self.conv1(x))  # [B, 32, 21, 21]
        x = F.relu(self.conv2(x))  # [B, 64, 11, 11]

        # Flatten spatial features
        x = x.view(x.size(0), -1)  # [B, 7744]

        # ── Concatenate with last move info ──
        combined = torch.cat((x, last_move_vec), dim=1)  # [B, 7748]

        # ── DQN Head ──
        x = F.relu(self.fc1(combined))  # [B, 256]
        x = self.dropout(x)
        q_values = self.fc2(x)  # [B, 4] — Linear (no activation)

        return q_values
```

- [ ] **Step 2: Verify model can be instantiated and produces correct output shape**

Run:
```bash
cd /home/ntdat/Documents/pacman && uv run python -c "
import torch
import sys
sys.path.insert(0, 'submissions/1.2')
from model import PacmanCNN
model = PacmanCNN()
# Count params
total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total:,}')
print(f'Est file size: {total * 4 / (1024*1024):.1f} MB')
# Test forward pass
x = torch.randn(1, 1, 21, 21)
last_move = torch.zeros(1, 4)
last_move[0, 0] = 1.0
q = model(x, last_move)
print(f'Output shape: {q.shape}')
assert q.shape == (1, 4), f'Expected (1, 4), got {q.shape}'
print('Model OK')
"
```
Expected: `Total params: ~1,990,660`, `Est file size: ~7.6 MB`, `Output shape: torch.Size([1, 4])`, `Model OK`

- [ ] **Step 3: Commit**

```bash
git add submissions/1.2/model.py
git commit -m "feat(1.2): add PacmanCNN model definition"
```

---

### Task 2: Create agent.py — GhostAgent + inline helpers

**Files:**
- Create: `submissions/1.2/agent.py`

**Interfaces:**
- Produces: `GhostAgent` class (ready for arena to load)
- Produces inline helper functions used by GhostAgent:
  - `_ghost_is_valid_position(pos, map_state) -> bool`
  - `_ghost_get_neighbors(pos, map_state) -> list[tuple]`
  - `_ghost_translate_move(current, next_pos) -> Move`
  - `_ghost_bfs(start, destination, map_state) -> list[tuple]`
  - `_ghost_enemy_next_position(my_pos, enemy, map_state) -> tuple`
  - `_ghost_simulate(my_pos, threat, map_state, turns=3) -> int`
  - `_ghost_list_dead_end_cells(map_state) -> list[tuple]`
  - `_ghost_build_dead_end_exit_map(map_state) -> dict[tuple, tuple]`
  - `_ghost_find_safest_junction(my_pos, threat, range, dead_end_exit, map_state) -> tuple | None`

This task creates the full `agent.py` but with the `PacmanAgent.step()` returning a placeholder — the DQN + A* logic comes in Task 3.

- [ ] **Step 1: Write `agent.py` with GhostAgent + all inline helpers + PacmanAgent skeleton**

```python
# agent.py — Submission 1.2: CNN-DQN Pacman + Rule-Based Ghost
# PacmanAgent: DQN primary, A* fallback, fog-of-war aware
# GhostAgent: BFS flee, dead-end detection, simulation (from submission 1)

import sys
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
import random
import numpy as np

# Add src to path to import framework classes
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

# ============================================================
# GHOST HELPER FUNCTIONS (inlined from submission 1)
# ============================================================

def _ghost_is_valid_position(pos, map_state):
    """Check if position is valid (not wall, within bounds)."""
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] == 0


def _ghost_get_neighbors(pos, map_state):
    """Get all valid neighboring positions."""
    neighbors = []
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        dr, dc = move.value
        next_pos = (pos[0] + dr, pos[1] + dc)
        if _ghost_is_valid_position(next_pos, map_state):
            neighbors.append(next_pos)
    return neighbors


def _ghost_translate_move(current, next_pos):
    """Convert position delta to Move enum."""
    dr = next_pos[0] - current[0]
    dc = next_pos[1] - current[1]
    return {
        (-1, 0): Move.UP,
        (1, 0): Move.DOWN,
        (0, -1): Move.LEFT,
        (0, 1): Move.RIGHT,
    }.get((dr, dc), Move.STAY)


def _ghost_bfs(start, destination, map_state):
    """Return shortest path from start to destination, inclusive. Returns [] if no path."""
    if tuple(start) == tuple(destination):
        return [start]
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor in _ghost_get_neighbors(current, map_state):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor == destination:
                path = [destination]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(neighbor)
    return []


def _ghost_enemy_next_position(my_pos, enemy, map_state):
    """Predict where enemy moves next based on BFS approach."""
    path = _ghost_bfs(enemy, my_pos, map_state)
    if len(path) < 2:
        return enemy
    next_pos = path[1]
    if len(path) < 3:
        return next_pos
    d1 = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    d2 = (path[2][0] - path[1][0], path[2][1] - path[1][1])
    if d1 == d2:
        return path[2]
    return next_pos


def _ghost_simulate(my_pos, threat, map_state, turns=3):
    """Simulate N turns of chase, return final BFS distance."""
    pacman = threat
    for _ in range(turns):
        pacman = _ghost_enemy_next_position(my_pos, pacman, map_state)
    path = _ghost_bfs(my_pos, pacman, map_state)
    return len(path) - 1 if path else 0


def _ghost_list_dead_end_cells(map_state):
    """Find all dead-end cells in the map."""
    dead_end_cells = set()
    rows, cols = map_state.shape
    for row in range(rows):
        for col in range(cols):
            start = (row, col)
            if not _ghost_is_valid_position(start, map_state):
                continue
            if len(_ghost_get_neighbors(start, map_state)) != 1:
                continue
            # Found a dead end, walk until we hit a junction
            prev, cur = None, start
            while True:
                dead_end_cells.add(cur)
                neighbors = [n for n in _ghost_get_neighbors(cur, map_state) if n != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                if len(_ghost_get_neighbors(nxt, map_state)) >= 3:
                    break  # nxt is a junction
                prev, cur = cur, nxt
    return list(dead_end_cells)


def _ghost_build_dead_end_exit_map(map_state):
    """Build mapping from dead-end cells to their exit gates."""
    dead_end_cells = set(_ghost_list_dead_end_cells(map_state))
    exit_map = {}
    for cell in dead_end_cells:
        for neighbor in _ghost_get_neighbors(cell, map_state):
            if neighbor not in dead_end_cells:
                gate = neighbor
                stack = [cell]
                visited = set()
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    exit_map[cur] = gate
                    for nxt in _ghost_get_neighbors(cur, map_state):
                        if nxt in dead_end_cells:
                            stack.append(nxt)
    return exit_map


def _ghost_find_safest_junction(my_pos, threat, range_val, dead_end_exit, map_state):
    """Find junction farthest from threat within range."""
    junctions = set(dead_end_exit.values())
    junctions.discard(my_pos)
    if not junctions:
        return None

    def dist(a, b):
        path = _ghost_bfs(a, b, map_state)
        return len(path) - 1 if path else float("inf")

    candidates = [j for j in junctions if dist(my_pos, j) <= range_val]
    if not candidates:
        return None
    return max(candidates, key=lambda j: dist(j, threat))


# ============================================================
# GHOST AGENT — Rule-based (from submission 1)
# ============================================================

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) agent using BFS flee, dead-end avoidance, and simulation.
    Ported from submission 1's hide_agent/ modules.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost-1.2"
        self.last_known_enemy_pos = None
        self.run_once = False
        self.dead_end_exit = {}
        self.forced_exit_path = []
        self.odd_step_position = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if not self.run_once:
            self.dead_end_exit = _ghost_build_dead_end_exit_map(map_state)
            self.run_once = True

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position if enemy_position is not None else self.last_known_enemy_pos
        return self._best_move(my_position, threat, map_state, step_number)

    def _best_move(self, my_pos, threat, map_state, step_number):
        neighbors = _ghost_get_neighbors(my_pos, map_state)

        if threat is not None:
            remaining_steps = (len(_ghost_bfs(threat, my_pos, map_state)) - 1) // 2
        else:
            remaining_steps = 999

        def __fall_back_move():
            if not neighbors:
                return Move.STAY
            best_cell = max(neighbors, key=lambda cell: len(_ghost_bfs(cell, threat, map_state)) - 1 if threat else 0)
            return _ghost_translate_move(my_pos, best_cell)

        if remaining_steps < 5:
            return __fall_back_move()

        if self.forced_exit_path:
            if self.forced_exit_path[0] == my_pos:
                self.forced_exit_path.pop(0)
            if self.forced_exit_path:
                return _ghost_translate_move(my_pos, self.forced_exit_path[0])

        if my_pos in self.dead_end_exit and threat is not None:
            closest_exit = self.dead_end_exit[my_pos]
            exit_steps = len(_ghost_bfs(my_pos, closest_exit, map_state)) - 1
            threat_to_exit_steps = (len(_ghost_bfs(threat, closest_exit, map_state)) - 1) // 2

            if 2 * exit_steps <= threat_to_exit_steps:
                self.forced_exit_path = _ghost_bfs(my_pos, closest_exit, map_state)
                if len(self.forced_exit_path) > 1:
                    next_cell = self.forced_exit_path[1]
                    return _ghost_translate_move(my_pos, next_cell)
                return Move.STAY
            else:
                return __fall_back_move()

        candidates = {}

        if remaining_steps > 8:
            neighbors.append(my_pos)

        for cell in neighbors:
            if remaining_steps > 4 and cell in self.dead_end_exit:
                continue
            if threat is not None:
                score = _ghost_simulate(cell, threat, map_state, min(remaining_steps, 3))
            else:
                score = 0
            candidates[cell] = score

        if not candidates:
            return __fall_back_move()

        best_candidate = max(candidates, key=candidates.get)

        if step_number % 2 == 1:
            self.odd_step_position = my_pos

        # Anti-fidgeting
        if step_number != 0 and step_number % 2 == 0:
            if best_candidate == self.odd_step_position:
                if threat is not None:
                    closest_junction = _ghost_find_safest_junction(
                        my_pos, threat, remaining_steps, self.dead_end_exit, map_state
                    )
                else:
                    closest_junction = None

                if closest_junction is None:
                    return __fall_back_move()

                path = _ghost_bfs(my_pos, closest_junction, map_state)
                if not path or len(path) < 2:
                    return __fall_back_move()

                self.forced_exit_path = path
                return _ghost_translate_move(my_pos, self.forced_exit_path[1])

        return _ghost_translate_move(my_pos, best_candidate)


# ============================================================
# PACMAN HELPER FUNCTIONS
# ============================================================

def _pacman_is_valid_position(pos, map_state):
    """Check if position is valid on internal map (not wall, within bounds)."""
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] != 1


def _pacman_get_neighbors(pos, map_state):
    """Get valid neighboring positions and their moves."""
    neighbors = []
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        dr, dc = move.value
        next_pos = (pos[0] + dr, pos[1] + dc)
        if _pacman_is_valid_position(next_pos, map_state):
            neighbors.append((next_pos, move))
    return neighbors


def _pacman_manhattan(a, b):
    """Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pacman_astar(start, goal, map_state):
    """A* pathfinding from start to goal. Returns list of Move enums, or empty list if no path."""
    if start == goal:
        return []

    open_set = [(0, 0, start, [])]  # (f_score, counter, pos, path_of_moves)
    g_score = {start: 0}
    closed_set = set()
    counter = 0

    while open_set:
        _, _, current, path = heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            return path

        for next_pos, move in _pacman_get_neighbors(current, map_state):
            if next_pos in closed_set:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(next_pos, float('inf')):
                g_score[next_pos] = tentative_g
                h = _pacman_manhattan(next_pos, goal)
                counter += 1
                heappush(open_set, (tentative_g + h, counter, next_pos, path + [move]))

    return []  # No path found


def _pacman_find_frontier(my_pos, internal_map):
    """Find the best frontier cell (boundary between known and unknown) for exploration."""
    if internal_map is None:
        return None
    h, w = internal_map.shape
    best = None
    best_score = -1
    for r in range(h):
        for c in range(w):
            if internal_map[r, c] != 0:
                continue
            # Check if this cell borders an unseen cell
            has_unknown = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and internal_map[nr, nc] == -1:
                    has_unknown = True
                    break
            if has_unknown:
                dist = _pacman_manhattan(my_pos, (r, c))
                if dist == 0:
                    continue
                score = 1.0 / dist
                if score > best_score:
                    best_score = score
                    best = (r, c)
    return best


# ============================================================
# PACMAN AGENT — DQN primary + A* fallback
# ============================================================

# Attempt to import torch and model; gracefully degrade if unavailable
try:
    import torch
    from model import PacmanCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PacmanCNN = None


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) agent with CNN-DQN primary + A* fallback.

    Decision flow:
    1. If enemy visible and DQN confident → use DQN move
    2. If enemy visible but DQN uncertain → A* to enemy
    3. If enemy hidden but recently seen → A* to last known position
    4. If enemy hidden too long → A* to frontier (explore)
    """

    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get('pacman_speed', 1)))
        self.name = "Pacman-1.2-DQN"

        # ── State tracking ──
        self.internal_map = None
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.last_move = None

        # ── Load DQN model ──
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None
        self.model = None
        if TORCH_AVAILABLE and PacmanCNN is not None:
            try:
                self.model = PacmanCNN()
                current_dir = Path(__file__).parent
                # Try multiple possible model file names
                for model_name in ["pacman_dqn.pt", "best_pacman_dqn.pt"]:
                    model_path = current_dir / model_name
                    if model_path.exists():
                        self.model.load_state_dict(
                            torch.load(model_path, map_location=self.device, weights_only=True)
                        )
                        self.model.eval()
                        break
                else:
                    # No trained weights found — DQN will return None
                    self.model = None
            except Exception:
                self.model = None

    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Update internal map memory
        self._update_map_memory(map_state)

        # 2. Track enemy visibility
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        # 3. Choose move
        chosen_move = Move.STAY
        path = None

        if enemy_position is not None:
            # Enemy visible — try DQN first
            dqn_move = self._get_dqn_move(map_state, my_position, enemy_position)
            if dqn_move is not None:
                chosen_move = dqn_move
            else:
                # DQN not confident or unavailable — A* to enemy
                path = _pacman_astar(my_position, enemy_position, self.internal_map)
                if path:
                    chosen_move = path[0]
                else:
                    chosen_move = self._greedy_toward(my_position, enemy_position)
        elif self.last_known_enemy_pos is not None and self.steps_since_seen <= 10:
            # Lost sight recently — A* to last known position
            path = _pacman_astar(my_position, self.last_known_enemy_pos, self.internal_map)
            if path:
                chosen_move = path[0]
            else:
                chosen_move = self._greedy_toward(my_position, self.last_known_enemy_pos)
        else:
            # Lost sight too long — explore frontier
            frontier = _pacman_find_frontier(my_position, self.internal_map)
            if frontier:
                path = _pacman_astar(my_position, frontier, self.internal_map)
                if path:
                    chosen_move = path[0]
            if chosen_move == Move.STAY:
                chosen_move = self._random_valid_move(my_position)

        # 4. Compute speed steps
        steps = self._compute_steps(my_position, chosen_move, path)

        self.last_move = chosen_move
        return (chosen_move, steps)

    # ── DQN Inference ──

    def _get_dqn_move(self, map_state, my_pos, enemy_pos):
        """Run DQN forward pass. Returns Move if confident, None otherwise."""
        if self.model is None or not TORCH_AVAILABLE:
            return None

        try:
            # Encode state
            input_map = self.internal_map.copy().astype(np.float32)
            input_map[input_map == -1] = -1.0  # Fog cells stay -1
            input_map[my_pos] = 2.0
            input_map[enemy_pos] = 3.0

            state_tensor = torch.FloatTensor(input_map).unsqueeze(0).unsqueeze(0).to(self.device)

            # Encode last move
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            last_move_vec = np.zeros(4, dtype=np.float32)
            if self.last_move in all_moves:
                last_move_vec[all_moves.index(self.last_move)] = 1.0
            move_tensor = torch.FloatTensor(last_move_vec).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                q_values = self.model(state_tensor, move_tensor).squeeze(0)  # [4]

            # Confidence check
            confidence = q_values.max().item() - q_values.mean().item()
            if confidence < self.CONFIDENCE_THRESHOLD:
                return None

            # Get best valid move
            best_idx = q_values.argmax().item()
            predicted_move = all_moves[best_idx]

            # Validate: must not reverse into wall
            if self._can_move(my_pos, predicted_move):
                return predicted_move

        except Exception:
            pass

        return None

    # ── Map Memory ──

    def _update_map_memory(self, map_state):
        """Build and maintain internal map by merging fog observations over time."""
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.internal_map[map_state == 1] = 1
            self.map_initialized = True

        # Merge visible cells into memory
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]

        # Restore agent position markers back to empty (they move)
        self.internal_map[(self.internal_map == 2) | (self.internal_map == 3)] = 0

    # ── Movement Helpers ──

    def _greedy_toward(self, my_pos, target):
        """Move greedily toward target using Manhattan distance."""
        best_move = Move.STAY
        best_dist = _pacman_manhattan(my_pos, target)
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nxt = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid_on_internal(nxt):
                d = _pacman_manhattan(nxt, target)
                if d < best_dist:
                    best_dist = d
                    best_move = move
        return best_move

    def _random_valid_move(self, my_pos):
        """Pick a random valid move."""
        moves = [mv for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                 if self._can_move(my_pos, mv)]
        return random.choice(moves) if moves else Move.STAY

    def _can_move(self, pos, move):
        """Check if a single step in the given direction is valid."""
        dr, dc = move.value
        nxt = (pos[0] + dr, pos[1] + dc)
        return self._is_valid_on_internal(nxt)

    def _is_valid_on_internal(self, pos):
        """Check if position is valid on internal map."""
        if self.internal_map is None:
            return False
        r, c = pos
        h, w = self.internal_map.shape
        return 0 <= r < h and 0 <= c < w and self.internal_map[r, c] != 1

    def _compute_steps(self, my_pos, chosen_move, path):
        """Compute how many speed steps to take (1 or 2) based on path straightness."""
        steps = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            # Only use speed=2 if the path is straight for 2+ tiles
            can_move_2 = self._can_move_n(my_pos, chosen_move, 2)

            # Check path doesn't turn at step 2 (no overshooting corners)
            if path and len(path) >= 2 and path[0] == chosen_move and path[1] != chosen_move:
                can_move_2 = False

            if can_move_2:
                steps = 2
        return steps

    def _can_move_n(self, pos, move, n):
        """Check if we can move n steps in a straight line."""
        r, c = pos
        dr, dc = move.value
        for i in range(1, n + 1):
            nr, nc = r + dr * i, c + dc * i
            if not self._is_valid_on_internal((nr, nc)):
                return False
        return True
```

- [ ] **Step 2: Verify agent can be loaded by the arena and GhostAgent works**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && uv run python -c "
from agent_loader import AgentLoader
loader = AgentLoader(submissions_dir='../submissions')
# Test Ghost loads
ghost = loader.load_agent('1.2', 'ghost')
print(f'Ghost loaded: {ghost.name}')
# Test Pacman loads (no weights yet, falls back to A*)
pacman = loader.load_agent('1.2', 'pacman', init_kwargs={'pacman_speed': 2})
print(f'Pacman loaded: {pacman.name}')
print('Both agents loaded successfully')
"
```
Expected: `Ghost loaded: Ghost-1.2`, `Pacman loaded: Pacman-1.2-DQN`, `Both agents loaded successfully`

- [ ] **Step 3: Run a quick game to verify both agents work end-to-end**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && uv run python arena.py --seek 1.2 --hide 1.2 --no-viz --max-steps 50
```
Expected: Game completes without errors, prints game result.

- [ ] **Step 4: Commit**

```bash
git add submissions/1.2/agent.py
git commit -m "feat(1.2): add PacmanAgent (DQN+A* fallback) and GhostAgent (rule-based)"
```

---

### Task 3: Create train.py — DQN Training Pipeline

**Files:**
- Create: `submissions/1.2/train.py`

**Interfaces:**
- Consumes: `PacmanCNN` from `model.py`, `Environment`/`Move` from `src/environment.py`
- Produces: `pacman_dqn.pt` and `best_pacman_dqn.pt` weight files in `submissions/1.2/`

- [ ] **Step 1: Write `train.py`**

```python
"""
train.py — DQN Training Script for Pacman CNN Agent (Submission 1.2)
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
        # ── Training ──
        self.epochs = getattr(args, 'epochs', 100)
        self.episodes_per_epoch = getattr(args, 'episodes_per_epoch', 20)
        self.max_steps_per_episode = 200

        # ── DQN Hyperparameters ──
        self.batch_size = getattr(args, 'batch_size', 64)
        self.gamma = 0.99
        self.lr = getattr(args, 'lr', 1e-3)
        self.tau = 0.005
        self.replay_buffer_size = 50000
        self.min_replay_size = 1000

        # ── Epsilon-Greedy ──
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995

        # ── Model ──
        self.input_shape = (1, 21, 21)
        self.n_actions = 4

        # ── Fog of War ──
        self.obs_radius = getattr(args, 'obs_radius', 0)  # 0 = full visibility

        # ── Device ──
        self.device = torch.device("cpu")

        # ── Saving ──
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

        # ── Apply Pacman move ──
        old_pac_pos = self.env.pacman_pos
        dr, dc = pacman_move.value
        new_r, new_c = old_pac_pos[0] + dr, old_pac_pos[1] + dc

        if self._is_valid_pos(new_r, new_c):
            self.env.pacman_pos = (new_r, new_c)
            self.last_pacman_move = pacman_move

        # ── Check capture after Pacman move ──
        if self._is_caught():
            reward = 100.0
            done = True
            state = self._encode_state()
            last_move_vec = self._encode_last_move()
            return state, last_move_vec, reward, done

        # ── Apply Ghost move ──
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

        # ── Build Networks ──
        self.online_net = PacmanCNN(config.input_shape, config.n_actions).to(self.device)
        self.target_net = PacmanCNN(config.input_shape, config.n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        total_params = sum(p.numel() for p in self.online_net.parameters())
        print(f"  Model params: {total_params:,}")
        print(f"  Est. file size: {total_params * 4 / (1024*1024):.1f} MB")

        # ── Optimizer & Loss ──
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # ── Replay Buffer ──
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # ── Exploration ──
        self.epsilon = config.epsilon_start

        # ── Training Environment ──
        self.env = TrainingEnv(pacman_speed=2, obs_radius=config.obs_radius)

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
        """Soft update: θ_target ← τ·θ_online + (1-τ)·θ_target."""
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
```

- [ ] **Step 2: Run a short training run to verify the pipeline works**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/1.2 && uv run python train.py --epochs 3 --episodes-per-epoch 5
```
Expected: Training runs for 3 epochs, prints progress table, saves `pacman_dqn.pt` and `best_pacman_dqn.pt`. No errors.

- [ ] **Step 3: Verify the trained weights can be loaded by the agent**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && uv run python -c "
from agent_loader import AgentLoader
loader = AgentLoader(submissions_dir='../submissions')
pacman = loader.load_agent('1.2', 'pacman', init_kwargs={'pacman_speed': 2})
print(f'Pacman loaded: {pacman.name}')
print(f'Model loaded: {pacman.model is not None}')
"
```
Expected: `Pacman loaded: Pacman-1.2-DQN`, `Model loaded: True`

- [ ] **Step 4: Run a game with the trained agent**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && uv run python arena.py --seek 1.2 --hide 1.2 --no-viz --max-steps 50
```
Expected: Game completes without errors or timeouts.

- [ ] **Step 5: Commit**

```bash
git add submissions/1.2/train.py submissions/1.2/pacman_dqn.pt submissions/1.2/best_pacman_dqn.pt
git commit -m "feat(1.2): add DQN training pipeline with replay buffer and Double DQN"
```

---

### Task 4: Create README.md — Documentation

**Files:**
- Create: `submissions/1.2/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Submission 1.2 — CNN-DQN Pacman Agent

## Overview

This agent uses a **CNN-DQN (Convolutional Neural Network + Deep Q-Network)** as its primary decision-maker, with **A\* pathfinding as a fallback** when the DQN is uncertain or the enemy is hidden by fog of war. The Ghost agent uses the proven rule-based logic from submission 1.

## Architecture

### Decision Flow

```
step(map_state, my_position, enemy_position, step_number)
│
├─ Update internal map memory (merge fog observations over time)
│
├─ IF enemy_position is NOT None:
│   ├─ Run DQN forward pass → Q-values for 4 moves
│   ├─ IF max(Q) - mean(Q) > 0.5 (confidence threshold):
│   │   └─ Use DQN move
│   └─ ELSE:
│       └─ A* to enemy_position
│
├─ IF enemy_position is None:
│   ├─ IF last_known_enemy_pos exists and steps_since_seen <= 10:
│   │   └─ A* to last known position
│   └─ ELSE:
│       └─ A* to nearest frontier cell (explore)
│
└─ Compute speed steps (1 or 2) → return (Move, steps)
```

### Model: PacmanCNN

```
Input: map_state [B, 1, 21, 21] + last_move one-hot [B, 4]

CNN Feature Extractor:
  Conv2d(1→32, 3×3, stride=1, pad=1) → ReLU  → [B, 32, 21, 21]
  Conv2d(32→64, 3×3, stride=2, pad=1) → ReLU  → [B, 64, 11, 11]
  Flatten                                        → [B, 7744]

DQN Head:
  Linear(7744 + 4 → 256) → ReLU → Dropout(0.1)
  Linear(256 → 4)        → Linear (raw Q-values)

Output: Q-values for [UP, DOWN, LEFT, RIGHT]
```

**State encoding:**
- `1.0` = wall
- `0.0` = empty
- `2.0` = Pacman position
- `3.0` = Ghost position (if visible)
- `-1.0` = fog/unseen

The CNN learns to distinguish fog cells from known empty cells, enabling partial-observability reasoning.

### GhostAgent

Rule-based agent ported from submission 1:
- BFS-based flee when enemy is close
- Dead-end detection and escape routing
- Multi-turn simulation for move evaluation
- Anti-fidgeting (oscillation detection → redirect to junctions)

## Files

| File | Purpose |
|------|---------|
| `agent.py` | PacmanAgent (DQN + A* fallback) and GhostAgent (rule-based) |
| `model.py` | PacmanCNN PyTorch model definition |
| `train.py` | DQN training pipeline (self-play, replay buffer, Double DQN) |
| `pacman_dqn.pt` | Trained model weights |
| `README.md` | This file |

## Training

### Quick Start

```bash
cd submissions/1.2
python train.py                          # Default: 100 epochs
python train.py --epochs 200             # More epochs
python train.py --epochs 500 --lr 0.0005 # Custom epochs + learning rate
python train.py --obs-radius 5           # Train with fog of war
```

### Training Details

| Setting | Value |
|---------|-------|
| Algorithm | Double DQN |
| Replay buffer | 50,000 transitions |
| Min replay before training | 1,000 |
| Target network update | Soft (τ=0.005) every step |
| Loss | SmoothL1 (Huber) |
| Optimizer | Adam, lr=1e-3 |
| Epsilon schedule | 1.0 → 0.05, decay 0.995/epoch |
| Gradient clipping | max_norm=10.0 |
| Batch size | 64 |
| Discount factor (γ) | 0.99 |
| Ghost opponent | Rule-based (BFS flee) |

### Reward Shaping

| Event | Reward |
|-------|--------|
| Catch ghost | +100.0 |
| Getting closer (per cell) | +2.0 |
| Moving away (per cell) | -2.0 |
| Wall bump | -1.0 |
| Time penalty | -0.1/step |
| Timeout | -50.0 |

### Fog of War Training

Pass `--obs-radius N` to train with limited visibility. The training environment masks cells outside the observation radius as `-1.0` (fog), and the Ghost position is hidden when outside range. This teaches the DQN to handle partial-observability states.

## Running

```bash
cd src
python arena.py --seek 1.2 --hide <any_id>           # As Pacman
python arena.py --seek <any_id> --hide 1.2            # As Ghost
python arena.py --seek 1.2 --hide 1.2                 # Both
python arena.py --seek 1.2 --hide <id> --pacman-obs-radius 5 --ghost-obs-radius 3  # With fog
```

## Performance

- **DQN inference**: single forward pass, ~1ms on CPU
- **A\* pathfinding**: completes in <5ms on 21×21 grid
- **Total step time**: well under 100ms (far below 1.0s timeout)
- **Model size**: ~7.6MB (`pacman_dqn.pt`)
```

- [ ] **Step 2: Commit**

```bash
git add submissions/1.2/README.md
git commit -m "docs(1.2): add README with architecture, training, and usage docs"
```

---

### Task 5: End-to-end verification

**Files:**
- No new files — verifies the complete system works

- [ ] **Step 1: Run a full game with DQN agent vs the submission 1 Ghost**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && uv run python arena.py --seek 1.2 --hide 1 --no-viz
```
Expected: Game completes without errors or timeouts. Prints game result.

- [ ] **Step 2: Run with fog of war to verify partial-observability handling**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && uv run python arena.py --seek 1.2 --hide 1 --no-viz --pacman-obs-radius 5 --ghost-obs-radius 3
```
Expected: Game completes without errors. Agent handles `enemy_position=None` gracefully (falls back to A* to last known position or frontier exploration).

- [ ] **Step 3: Run multiple games to check consistency**

Run:
```bash
cd /home/ntdat/Documents/pacman/src && for i in 1 2 3; do uv run python arena.py --seek 1.2 --hide 1 --no-viz --start-mode stochastic 2>&1 | tail -5; echo "---"; done
```
Expected: 3 games all complete without errors or timeouts.

- [ ] **Step 4: Final commit (if any fixes were needed)**

```bash
git add -A submissions/1.2/
git commit -m "fix(1.2): address end-to-end verification issues" || echo "No fixes needed"
```
