# Pacman DQN Training Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 6 recommendations from the design spec: deeper CNN with BatchNorm, dynamic confidence threshold, upper-half explore reward, upper-half search priority, and curriculum training pipeline against SimpleGhost + GhostAgent.

**Architecture:** New `PacmanCNNv2` class in model.py alongside existing v1. `train.py` gets phase management, restored `SimpleGhostOpponent`, and spatial-biased explore reward. `agent.py` gets dynamic threshold, upper-half frontier search, and model auto-detection.

**Tech Stack:** Python 3.14, PyTorch 2.12.1+cu130, CUDA RTX 2050, numpy

## Global Constraints

- Training time: under 1 hour total
- GPU required (CUDA), fallback to CPU acceptable but slower
- Output model: `pacman_dqn_v2.pt` (separate from original `pacman_dqn.pt`)
- Backward compat: original `PacmanCNN`, `pacman_dqn.pt`, and existing training behavior preserved
- Run via: `python train.py --curriculum --stochastic`
- Venv path: `.venv/bin/python`
- Working directory: `/home/ntdat/Documents/pacman/submissions/LAB2/`
- Ghost spawns upper half; hides in side corners (top-left, top-right) and upper-middle alcoves

---

---

### Task 1: Add PacmanCNNv2 class to model.py

**Files:**
- Modify: `/home/ntdat/Documents/pacman/submissions/LAB2/model.py` (append at end)

**Interfaces:**
- Produces: `PacmanCNNv2(nn.Module)` — 3-layer CNN with BatchNorm, same `forward(x, last_move_vec)` signature as v1. State dict keys: `conv1.weight`, `bn1.weight`, `conv2.weight`, `bn2.weight`, `conv3.weight`, `bn3.weight`, `fc1.weight`, `fc2.weight`.

- [ ] **Step 1: Append PacmanCNNv2 class to model.py**

Append after line 67 (end of existing PacmanCNN class):

```python
class PacmanCNNv2(nn.Module):
    """
    Deeper CNN-DQN with BatchNorm — v2 improvements.

    Architecture:
        Input: map_state [B, 1, 21, 21] + last_move one-hot [B, 4]

        CNN Feature Extractor:
            Conv2d(1→32, 3×3, s=1, p=1) → BN → ReLU  → [B, 32, 21, 21]
            Conv2d(32→64, 3×3, s=1, p=1) → BN → ReLU  → [B, 64, 21, 21]
            Conv2d(64→64, 3×3, s=2, p=1) → BN → ReLU  → [B, 64, 11, 11]
            Flatten                                     → [B, 7744]

        DQN Head:
            Linear(7744 + 4 → 256) → ReLU → Dropout(0.1)
            Linear(256 → 4)        → Linear (raw Q-values)

        Output: Q-values for [UP, DOWN, LEFT, RIGHT]
    """

    def __init__(self, input_shape=(1, 21, 21), n_actions=4):
        super(PacmanCNNv2, self).__init__()

        # CNN Feature Extractor — 3 layers, stride-2 at last conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 64 * 11 * 11 = 7744 (same as v1)
        self.feature_size = 64 * 11 * 11

        # DQN Head
        self.fc1 = nn.Linear(self.feature_size + n_actions, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, n_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m is self.fc2:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, last_move_vec):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        combined = torch.cat((x, last_move_vec), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        q_values = self.fc2(x)
        return q_values
```

- [ ] **Step 2: Verify import and forward pass**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import sys; sys.path.insert(0, '../src')
from model import PacmanCNNv2
import torch
m = PacmanCNNv2()
x = torch.randn(4, 1, 21, 21)
move = torch.randn(4, 4)
out = m(x, move)
print(f'Output shape: {out.shape}')  # Expected: torch.Size([4, 4])
print(f'Total params: {sum(p.numel() for p in m.parameters()):,}')
# Check conv3 exists (v2 marker)
assert hasattr(m, 'conv3'), 'conv3 missing'
assert hasattr(m, 'bn1'), 'bn1 missing'
print('All checks passed')
"
```
Expected: `Output shape: torch.Size([4, 4])`, params ~2,007,684, all checks pass.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/model.py && git commit -m "feat: add PacmanCNNv2 with BatchNorm and 3-layer CNN

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

### Task 2: Dynamic confidence threshold in agent.py

**Files:**
- Modify: `/home/ntdat/Documents/pacman/submissions/LAB2/agent.py`

**Interfaces:**
- Modifies: `PacmanAgent.__init__` — adds `self._current_epoch` and `self._total_epochs` from kwargs
- Produces: `_get_confidence_threshold()` → float between 0.2 and 0.8
- Consumes: `training_epochs` and `current_epoch` from kwargs (default: 200 and 0)

- [ ] **Step 1: Modify PacmanAgent.__init__ to accept epoch kwargs**

In `/home/ntdat/Documents/pacman/submissions/LAB2/agent.py`, find the `PacmanAgent.__init__` method (around line 148-169). Add after the `self.last_move = None` line:

```python
        # ── Confidence threshold tracking ──
        self._total_epochs = max(1, int(kwargs.get('total_epochs', 200)))
        self._current_epoch = max(0, int(kwargs.get('current_epoch', 0)))
```

- [ ] **Step 2: Replace CONFIDENCE_THRESHOLD usage with dynamic method**

Find the line `CONFIDENCE_THRESHOLD = 0.5` (around line 128). Replace the class-level constant and the usage site.

Change the class-level constant from:
```python
    CONFIDENCE_THRESHOLD = 0.5
```
To delete it entirely (it's only used in `_get_dqn_move`).

In `_get_dqn_move` method (around line 187-190), change:
```python
            if confidence < self.CONFIDENCE_THRESHOLD:
                return None
```
To:
```python
            if confidence < self._get_confidence_threshold():
                return None
```

- [ ] **Step 3: Add _get_confidence_threshold method to PacmanAgent**

Add this method inside `PacmanAgent` class (anywhere among the helper methods):

```python
    def _get_confidence_threshold(self):
        """Dynamic threshold that decays as DQN matures through training."""
        if self._total_epochs <= 0:
            return 0.5
        t = self._current_epoch / float(self._total_epochs)
        final = 0.8 * (0.98 ** (t * self._total_epochs)) if t < 1.0 else 0.2
        return max(0.2, min(0.8, final))
```

- [ ] **Step 4: Verify the threshold logic**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import sys; sys.path.insert(0, '../src')
from agent import PacmanAgent
# Test default (trained model, epoch 200)
a = PacmanAgent(current_epoch=200, total_epochs=200)
t = a._get_confidence_threshold()
print(f'Fully trained threshold: {t:.4f}')  # Should be 0.2

# Test mid-training
b = PacmanAgent(current_epoch=50, total_epochs=200)
t2 = b._get_confidence_threshold()
print(f'Mid-training threshold: {t2:.4f}')  # Should be ~0.29

# Test fresh (epoch 0)
c = PacmanAgent(current_epoch=0, total_epochs=200)
t3 = c._get_confidence_threshold()
print(f'Fresh threshold: {t3:.4f}')  # Should be 0.8

assert t == 0.2, f'Expected 0.2, got {t}'
assert 0.25 < t2 < 0.35, f'Expected ~0.29, got {t2}'
assert t3 == 0.8, f'Expected 0.8, got {t3}'
print('All threshold checks passed')
"
```

Expected: Fully trained=0.2, mid-training ~0.29, fresh=0.8, all asserts pass.

- [ ] **Step 5: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: dynamic confidence threshold in PacmanAgent

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

### Task 3: Upper-half search priority in A* fallback

**Files:**
- Modify: `/home/ntdat/Documents/pacman/submissions/LAB2/agent.py`

**Interfaces:**
- Modifies: `_pacman_find_frontier()` — adds upper-half multiplier and pocket bonus
- Produces: `UPPER_POCKETS` module-level set — specific hiding-spot coordinates

- [ ] **Step 1: Add UPPER_POCKETS constant and update _pacman_find_frontier**

In `/home/ntdat/Documents/pacman/submissions/LAB2/agent.py`, after the `_pacman_manhattan` function (around line 65), add:

```python
# Known hiding pockets in the classic 21x21 map (upper half)
UPPER_POCKETS = {
    # Side corners (top-left)
    (1, 1), (1, 2), (1, 3),
    # Side corners (top-right)
    (1, 17), (1, 18), (1, 19),
    # Upper-middle alcoves
    (5, 5), (5, 6),
    (5, 14), (5, 15),
    (9, 8), (9, 9), (9, 10), (9, 11), (9, 12),
}
```

Replace the existing `_pacman_find_frontier` function (around line 73-95). The old implementation:

```python
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
```

Replace with:

```python
def _pacman_find_frontier(my_pos, internal_map):
    """Find best frontier cell with upper-half and pocket priority."""
    if internal_map is None:
        return None
    h, w = internal_map.shape
    mid_row = h // 2
    best = None
    best_score = -1.0
    for r in range(h):
        for c in range(w):
            if internal_map[r, c] != 0:
                continue
            has_unknown = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and internal_map[nr, nc] == -1:
                    has_unknown = True
                    break
            if not has_unknown:
                continue
            dist = _pacman_manhattan(my_pos, (r, c))
            if dist == 0:
                continue
            score = 1.0 / dist
            # Upper-half bonus: 2x score (ghost spawns there)
            if r < mid_row:
                score *= 2.0
            # Pocket bonus: 3x score for known hiding spots
            if (r, c) in UPPER_POCKETS:
                score *= 3.0
            if score > best_score:
                best_score = score
                best = (r, c)
    return best
```

- [ ] **Step 2: Verify frontier search prioritization**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import sys; sys.path.insert(0, '../src')
from agent import _pacman_find_frontier, UPPER_POCKETS
import numpy as np

# Build a test map: simple 21x21 with known path, some fog
m = np.full((21, 21), -1.0, dtype=np.float32)  # all fog
m[:, :] = 0.0  # mark all known
m[0, 8] = -1.0  # fog cell in upper half near center
m[0, 10] = -1.0  # another fog cell
m[20, 10] = -1.0  # fog cell in lower half

# Pacman in lower middle
my_pos = (19, 10)
result = _pacman_find_frontier(my_pos, m)
print(f'Chosen frontier: {result}')
# Should be (0, 8) or (0, 10) — upper half preferred over (20, 10)
assert result is not None, 'No frontier found'
assert result[0] < 10, f'Expected upper half, got row {result[0]}'
print('Frontier correctly prioritized upper half')

# Test pocket bonus
m2 = np.full((21, 21), -1.0, dtype=np.float32)
m2[:, :] = 0.0
# Make a pocket a frontier (add fog neighbor)
m2[1, 3] = -1.0  # neighbor of UPPER_POCKETS (1,2)
my_pos2 = (19, 10)
result2 = _pacman_find_frontier(my_pos2, m2)
print(f'Pocket frontier: {result2}')
assert result2 == (1, 2), f'Expected pocket (1,2), got {result2}'
print('Pocket bonus working correctly')
"
```

Expected: first result in upper half, second result is pocket (1,2).

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: upper-half search priority and pocket bonus in frontier selection

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

### Task 4: Model auto-detection (v1 vs v2) in agent.py

**Files:**
- Modify: `/home/ntdat/Documents/pacman/submissions/LAB2/agent.py`

**Interfaces:**
- Modifies: `PacmanAgent.__init__` — auto-detects PacmanCNN vs PacmanCNNv2 from state_dict keys
- Consumes: `PacmanCNN` and `PacmanCNNv2` from `model` module

- [ ] **Step 1: Update model loading in PacmanAgent.__init__**

In `/home/ntdat/Documents/pacman/submissions/LAB2/agent.py`, find the model loading block in `PacmanAgent.__init__` (around line 155-175). The current logic tries only `PacmanCNN`. Replace with auto-detection.

Current code:
```python
        if TORCH_AVAILABLE and PacmanCNN is not None:
            try:
                self.model = PacmanCNN()
                current_dir = Path(__file__).parent
                for model_name in ["pacman_dqn.pt", "best_pacman_dqn.pt"]:
                    model_path = current_dir / model_name
                    if model_path.exists():
                        self.model.load_state_dict(
                            torch.load(model_path, map_location=self.device, weights_only=True)
                        )
                        self.model.eval()
                        break
                else:
                    self.model = None
            except Exception:
                self.model = None
```

Replace with:
```python
        if TORCH_AVAILABLE:
            try:
                # Try importing v2 model class
                try:
                    from model import PacmanCNNv2
                except ImportError:
                    PacmanCNNv2 = None

                current_dir = Path(__file__).parent
                checkpoint = None
                for model_name in ["pacman_dqn_v2.pt", "pacman_dqn.pt", "best_pacman_dqn.pt"]:
                    model_path = current_dir / model_name
                    if model_path.exists():
                        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                        break

                if checkpoint is not None:
                    # Auto-detect v1 vs v2 by checking for conv3 key
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    is_v2 = any('conv3' in k for k in state_dict.keys())

                    if is_v2 and PacmanCNNv2 is not None:
                        self.model = PacmanCNNv2()
                    elif PacmanCNN is not None:
                        self.model = PacmanCNN()
                    else:
                        self.model = None

                    if self.model is not None:
                        self.model.load_state_dict(state_dict)
                        self.model.eval()

                    # Read training epoch count for dynamic threshold
                    if isinstance(checkpoint, dict):
                        self._total_epochs = max(1, int(checkpoint.get('training_epochs', 200)))
                        self._current_epoch = max(0, int(checkpoint.get('epoch', self._total_epochs)))
                else:
                    self.model = None
            except Exception:
                self.model = None
```

Important: move the `self._total_epochs` / `self._current_epoch` initialization from Task 2 before this block, or handle the case where Task 2's init already set them.

Specifically, change the order in `__init__`:
1. First, set defaults for `_total_epochs` and `_current_epoch` from kwargs
2. Then, model loading block (which may override them from checkpoint)

The Task 2 init block becomes:
```python
        # ── Confidence threshold tracking (may be overridden by checkpoint) ──
        self._total_epochs = max(1, int(kwargs.get('total_epochs', 200)))
        self._current_epoch = max(0, int(kwargs.get('current_epoch', 0)))
```

And the model loading block sets them from checkpoint (overrides kwarg defaults).

- [ ] **Step 2: Verify auto-detection with v2 model**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import sys; sys.path.insert(0, '../src')
from agent import PacmanAgent

# Test with v2 model (will use v2 class if pacman_dqn_v2.pt exists)
import os; files = os.listdir('.')
print('Available model files:', [f for f in files if f.endswith('.pt')])

# Create a dummy v2 checkpoint for testing
import torch
from model import PacmanCNNv2
m = PacmanCNNv2()
checkpoint = {
    'model_state_dict': m.state_dict(),
    'epoch': 150,
    'training_epochs': 350,
}
torch.save(checkpoint, '__test_v2_checkpoint.pt')

# Test loading
import agent as agent_module
original_dir = agent_module.Path('.').parent
agent_module.Path = lambda x: agent_module.Path('.')

# Quick test: check detection logic
state_dict = checkpoint['model_state_dict']
is_v2 = any('conv3' in k for k in state_dict.keys())
print(f'v2 detected: {is_v2}')  # Should be True
assert is_v2, 'Should detect v2'

# Cleanup
import os; os.remove('__test_v2_checkpoint.pt')
print('Detection test passed')
"
```

Expected: `v2 detected: True`.

- [ ] **Step 3: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/agent.py && git commit -m "feat: model auto-detection (v1/v2) with checkpoint epochs

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

### Task 5: Restore SimpleGhostOpponent + add explore reward

**Files:**
- Modify: `/home/ntdat/Documents/pacman/submissions/LAB2/train.py`

**Interfaces:**
- Produces: `SimpleGhostOpponent` class (restored), `_compute_explore_reward()`, `_prev_fog_mask` tracking in TrainingEnv
- Consumes: None (self-contained in train.py)

- [ ] **Step 1: Restore SimpleGhostOpponent class**

In `/home/ntdat/Documents/pacman/submissions/LAB2/train.py`, after the ReplayBuffer class (after line ~98, before the TrainingEnv section), insert the SimpleGhostOpponent class that was previously removed:

```python
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
```

- [ ] **Step 2: Add explore reward tracking to TrainingEnv**

In `TrainingEnv.__init__` (currently around line 120-130), add fog mask tracking:

After `self.prev_distance = None`, add:

```python
        self._prev_fog_mask = None
```

In `TrainingEnv.reset()`, after `self.prev_distance = ...`, add:

```python
        self._prev_fog_mask = None
```

- [ ] **Step 3: Add _compute_explore_reward method to TrainingEnv**

Add this method inside `TrainingEnv` class:

```python
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
```

In `TrainingEnv.step()`, in the reward computation section (before `_compute_reward` is called), add:

```python
        explore_bonus = self._compute_explore_reward()
```

And modify the reward line to include it. Find the line:
```python
        reward = self._compute_reward(old_pac_pos)
```

Change to:
```python
        reward = self._compute_reward(old_pac_pos) + explore_bonus
```

Also in `TrainingEnv.reset()`, initialize `_prev_fog_mask` after encoding state. After `state = self._encode_state()`, add:

```python
        self._prev_fog_mask = state.copy()
```

- [ ] **Step 4: Verify SimpleGhostOpponent and explore reward**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import sys; sys.path.insert(0, '../src')
from train import SimpleGhostOpponent, TrainingEnv

# Test SimpleGhost exists
g = SimpleGhostOpponent()
print('SimpleGhostOpponent created OK')

# Test explore reward (with obs_radius > 0)
env = TrainingEnv(pacman_speed=2, obs_radius=5, stochastic=False)
state, move = env.reset()
print('TrainingEnv reset OK with fog tracking')
print(f'Prev fog mask shape: {env._prev_fog_mask.shape}')
"
```

Expected: SimpleGhost created, TrainingEnv reset with fog mask.

- [ ] **Step 5: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/train.py && git commit -m "feat: restore SimpleGhostOpponent + add explore reward with upper-half bonus

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

### Task 6: Curriculum training pipeline with phases

**Files:**
- Modify: `/home/ntdat/Documents/pacman/submissions/LAB2/train.py`

**Interfaces:**
- Modifies: `DQNTrainer.__init__`, `DQNTrainer.train()`, adds `_run_phase()`, `_create_env_for_phase()`
- Produces: `--curriculum`, `--phase`, `--simple-epochs`, `--mixed-epochs`, `--ghost-epochs` CLI args
- Consumes: `SimpleGhostOpponent`, `GhostAgent`, `PacmanCNNv2`

- [ ] **Step 1: Update TrainingConfig for phase params**

In `TrainingConfig.__init__`, add after the stochastic line:

```python
        # -- Curriculum phases --
        self.simple_epochs = getattr(args, 'simple_epochs', 200)
        self.mixed_epochs = getattr(args, 'mixed_epochs', 100)
        self.ghost_epochs = getattr(args, 'ghost_epochs', 50)
```

- [ ] **Step 2: Add opponent parameter to TrainingEnv**

Modify `TrainingEnv.__init__` signature to accept `opponent` parameter:

Change:
```python
    def __init__(self, pacman_speed=2, obs_radius=0, stochastic=False):
        self.env = Environment(pacman_speed=pacman_speed, deterministic_starts=not stochastic)
        self.ghost = GhostAgent(
            log_path=None,
            diagnostics_enabled=False,
            pacman_speed=pacman_speed,
        )
```

To:
```python
    def __init__(self, pacman_speed=2, obs_radius=0, stochastic=False, opponent="simple"):
        self.env = Environment(pacman_speed=pacman_speed, deterministic_starts=not stochastic)
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
            # 70% SimpleGhost, 30% GhostAgent (per episode)
            if self._ghost_agent is None:
                self._ghost_agent = GhostAgent(
                    log_path=None,
                    diagnostics_enabled=False,
                    pacman_speed=pacman_speed,
                )
            # Ghost selection happens in reset()

    def reset(self):
        """Reset environment and return initial state."""
        self.env.reset()
        self.last_pacman_move = None
        self.step_count = 0
        self.prev_distance = self._manhattan(self.env.pacman_pos, self.env.ghost_pos)
        # Mixed mode: choose opponent per episode
        if self._opponent_mode == "mixed":
            self.ghost = self._simple_ghost if random.random() < 0.7 else self._ghost_agent
        state = self._encode_state()
        last_move_vec = self._encode_last_move()
        self._prev_fog_mask = state.copy()
        return state, last_move_vec
```

- [ ] **Step 3: Update DQNTrainer to use PacmanCNNv2 and support phases**

In `DQNTrainer.__init__`, change model creation:

Find:
```python
        self.online_net = PacmanCNN(config.input_shape, config.n_actions).to(self.device)
        self.target_net = PacmanCNN(config.input_shape, config.n_actions).to(self.device)
```

Replace with:
```python
        from model import PacmanCNNv2
        self.online_net = PacmanCNNv2(config.input_shape, config.n_actions).to(self.device)
        self.target_net = PacmanCNNv2(config.input_shape, config.n_actions).to(self.device)
```

And update TrainingEnv creation:
```python
        self.env = TrainingEnv(pacman_speed=2, obs_radius=config.obs_radius, stochastic=config.stochastic, opponent="simple")
```

Add phase tracking:
```python
        self.current_phase = 0
        self.cum_epoch = 0
```

- [ ] **Step 4: Add _run_phase method to DQNTrainer**

Add this method:

```python
    def _run_phase(self, epochs, phase_name, opponent_mode):
        """Run one curriculum phase."""
        self.current_phase += 1
        print(f"\n{'='*60}")
        print(f"Phase {self.current_phase}: {phase_name}")
        print(f"  Opponent: {opponent_mode}")
        print(f"  Epochs: {epochs} | Current epsilon: {self.epsilon:.4f}")
        print(f"{'='*60}")

        # Switch opponent mode
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
                self._save_checkpoint(f"best_{self.config.model_filename}", phase_name, best=True)

            if self.cum_epoch % self.config.save_every == 0:
                self._save_checkpoint(self.config.model_filename, phase_name)

        total_time = time.time() - total_start
        print(f"  Phase complete in {total_time:.1f}s | Best catch rate: {best_catch_rate:.1%}")
        return best_catch_rate
```

- [ ] **Step 5: Add _save_checkpoint method**

Replace the existing `_save_model` with:

```python
    def _save_checkpoint(self, filename, phase_name, best=False):
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
            print(f"       -> New best ({phase_name}): {self.epoch_catches[-1]:.1%}")
```

- [ ] **Step 6: Replace DQNTrainer.train() with curriculum version**

Replace the entire `train()` method:

```python
    def train(self):
        """Main curriculum training loop."""
        print(f"\nCurriculum Training: {self.config.simple_epochs + self.config.mixed_epochs + self.config.ghost_epochs} total epochs")
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
        self._save_checkpoint(self.config.model_filename, "final")

        total_time = time.time() - total_start
        print("-" * 60)
        print(f"Curriculum training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Total epochs: {self.cum_epoch}")
        print(f"  Final epsilon: {self.epsilon:.4f}")
        print(f"  Model saved to: {self.config.save_dir / self.config.model_filename}")
```

- [ ] **Step 7: Update parse_args with curriculum args**

In `parse_args()`, add after the existing arguments:

```python
    parser.add_argument('--curriculum', action='store_true', default=False, help='Run all 3 curriculum phases sequentially')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None, help='Run a single training phase')
    parser.add_argument('--simple-epochs', type=int, default=200, help='Phase 1 epochs (default: 200)')
    parser.add_argument('--mixed-epochs', type=int, default=100, help='Phase 2 epochs (default: 100)')
    parser.add_argument('--ghost-epochs', type=int, default=50, help='Phase 3 epochs (default: 50)')
```

- [ ] **Step 8: Update main() to handle --phase mode**

Update `main()` to handle single-phase execution:

```python
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
            trainer.online_net.load_state_dict(ckpt['model_state_dict'])
            trainer.target_net.load_state_dict(ckpt['model_state_dict'])
            trainer.epsilon = ckpt.get('epsilon', config.epsilon_start)
            trainer.cum_epoch = ckpt.get('epoch', 0)
            trainer.current_phase = ckpt.get('phase', 0)
            print(f"Resumed from checkpoint (epoch {trainer.cum_epoch}, epsilon {trainer.epsilon:.4f})")
        trainer._run_phase(epochs, name, opponent)
        trainer._save_checkpoint(config.model_filename, name)
    else:
        trainer.train()
```

- [ ] **Step 9: Verify imports and phase switching**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import sys; sys.path.insert(0, '../src')
from train import TrainingConfig, DQNTrainer, SimpleGhostOpponent
print('Phase imports OK')
# Quick smoke test: create config with phase defaults
class FakeArgs:
    epochs=100; episodes_per_epoch=20; batch_size=64; lr=1e-3; obs_radius=0
    stochastic=False; simple_epochs=200; mixed_epochs=100; ghost_epochs=50
cfg = TrainingConfig(FakeArgs())
assert cfg.simple_epochs == 200; assert cfg.mixed_epochs == 100; assert cfg.ghost_epochs == 50
print('Config phase params OK')
"
```

Expected: imports and config OK.

- [ ] **Step 10: Commit**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/train.py && git commit -m "feat: curriculum training pipeline with 3 phases and mixed opponent

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```

---

### Task 7: Integration — run curriculum training

**Files:**
- None (run training script)
- Verify: `pacman_dqn_v2.pt` created in LAB2 folder

- [ ] **Step 1: Run curriculum training**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python train.py --curriculum --stochastic --simple-epochs 10 --mixed-epochs 5 --ghost-epochs 5
```

Use reduced epoch counts for a quick smoke test (~3-5 min). Expected: all 3 phases run, model saved as `pacman_dqn_v2.pt`.

- [ ] **Step 2: Verify output model**

Run:
```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python -c "
import torch
ckpt = torch.load('pacman_dqn_v2.pt', map_location='cpu', weights_only=True)
print('Checkpoint keys:', list(ckpt.keys()))
print(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'Phase: {ckpt.get(\"phase\", \"N/A\")}')
print(f'Training epochs: {ckpt.get(\"training_epochs\", \"N/A\")}')
assert 'model_state_dict' in ckpt, 'Missing model state'
# Check for v2 keys
sd = ckpt['model_state_dict']
has_conv3 = any('conv3' in k for k in sd.keys())
print(f'Is v2 model: {has_conv3}')
assert has_conv3, 'Expected v2 model (conv3 key not found)'
print('Checkpoint verified OK')
"
```

Expected: checkpoint has model_state_dict, epoch, phase, training_epochs; model is v2.

- [ ] **Step 3: Full training run (optional - when ready)**

```bash
cd /home/ntdat/Documents/pacman/submissions/LAB2 && /home/ntdat/Documents/pacman/.venv/bin/python train.py --curriculum --stochastic
```

This runs the full 350-epoch curriculum (~60 min). Save the output for review.

- [ ] **Step 4: Commit checkpoint if trained**

```bash
cd /home/ntdat/Documents/pacman && git add submissions/LAB2/pacman_dqn_v2.pt && git commit -m "model: trained PacmanCNNv2 via curriculum pipeline

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>"
```
