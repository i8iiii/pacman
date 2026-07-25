# Pacman DQN Training Improvements — Design Spec

**Date**: 2026-07-25
**Status**: Design approved
**Based on**: ASSESSMENT.md (Section 8: Summary & Key Findings)

## Overview

Implement all 5 recommendations from the Pacman CNN-DQN assessment to produce a more effective seeker agent. Target: under 1 hour total training time. Output model: `pacman_dqn_v2.pt`. Additionally, incorporate domain knowledge about ghost spawning behavior (upper half) and hiding patterns (side corners, upper-middle alcoves) to optimize search efficiency.

## Recommendations Implemented

| # | Recommendation | Implementation | Where |
|---|---------------|---------------|-------|
| 1 | Train against GhostAgent | Curriculum phases (SimpleGhost → Mixed → GhostAgent) | `train.py` |
| 2 | Add batch normalization | BatchNorm2d after each conv layer before ReLU | `model.py` |
| 3 | Deeper CNN (3 layers) | Third conv layer (64→64, s2 moved to last) | `model.py` |
| 4 | Dynamic confidence threshold | Epoch-aware threshold that decays from 0.8 to 0.2 | `agent.py` |
| 5 | Explore reward (fog training) | +0.1 per newly revealed fog cell, 2x bonus in upper half | `train.py` |
| 6 | Upper-half search priority | Frontier search biased toward upper half, side corners, alcoves | `agent.py` |

---

## Section 1: Model Architecture (`model.py`)

### New class: `PacmanCNNv2`

Keep existing `PacmanCNN` for backward compatibility. Create new class with:

**Architecture:**
```
Input: map_state [B, 1, 21, 21] + last_move [B, 4]

CNN Feature Extractor:
  Conv2d(1→32, 3×3, s=1, p=1) → BatchNorm2d(32) → ReLU  → [B, 32, 21, 21]
  Conv2d(32→64, 3×3, s=1, p=1) → BatchNorm2d(64) → ReLU  → [B, 64, 21, 21]
  Conv2d(64→64, 3×3, s=2, p=1) → BatchNorm2d(64) → ReLU  → [B, 64, 11, 11]
  Flatten                                                 → [B, 7744]

DQN Head:
  Concat(features, last_move)                             → [B, 7748]
  Linear(7748→256) → ReLU → Dropout(0.1)
  Linear(256→4)                                            → Q-values [B, 4]
```

**Key changes from v1:**
- Added third conv layer (64→64): deeper spatial reasoning without changing flattened size
- Stride-2 moved from second to third conv: first two layers preserve full 21×21 resolution
- BatchNorm2d after each conv: stabilizes training, reduces initialization sensitivity

**Size**: ~2,007,684 params. File: ~7.8 MB.

---

## Section 2: Dynamic Confidence Threshold (`agent.py`)

### Change in `PacmanAgent` class

**Current (hardcoded):** `CONFIDENCE_THRESHOLD = 0.5`

**New (schedule-based):**
```python
def __init__(self, **kwargs):
    self.training_epochs = kwargs.get('training_epochs', 200)

def _get_confidence_threshold(self):
    t = self.training_epochs / 200.0
    threshold = 0.8 * (0.98 ** (t * 200))
    return max(0.2, min(0.8, threshold))
```

**Behavior:**
- Epoch 0: threshold = 0.8 (almost always falls back to A*)
- Epoch 50: threshold ≈ 0.29 (DQN takes over frequently)
- Epoch 100+: threshold = 0.2 (DQN dominates)

The `training_epochs` value is stored in the model checkpoint and loaded by the PacmanAgent. Defaults to 200 if not found.

---

## Section 3: Explore Reward with Spatial Bias (`train.py`)

### New behavior in `TrainingEnv`

Only active when `obs_radius > 0`. Add to `_compute_reward()`:

```python
def _count_newly_revealed(self):
    """Count fog cells that became visible this step, with upper-half bonus."""
    count = 0
    mid_row = self.env.height // 2
    for r in range(self.env.height):
        for c in range(self.env.width):
            if self._prev_fog_mask[r,c] == -1 and self._current_obs[r,c] != -1:
                count += 1.0 if r >= mid_row else 2.0  # 2x for upper half
    return count * 0.1
```

**Reward scale**: Upper-half cells are worth 2x, encouraging the agent to search there first. Typical step: 1-5 cells revealed → 0.1-1.0 reward depending on location.

---

## Section 4: Upper-Half Search Priority in A* Fallback (`agent.py`)

### Change in `_pacman_find_frontier()`

The frontier selection function (used when ghost is not visible) is updated with spatial priorities:

```python
UPPER_POCKETS = {
    # Side corners (highest priority - isolated, hard to escape)
    (1, 1), (1, 2), (1, 3),              # top-left corner area
    (1, 17), (1, 18), (1, 19),           # top-right corner area
    # Upper-middle alcoves (high priority - wall-protected pockets)
    (5, 5), (5, 6),                       # left alcove
    (5, 14), (5, 15),                      # right alcove
    (9, 8), (9, 9), (9, 10), (9, 11),    # center alcove pockets
    (9, 12),
}

def _pacman_find_frontier(my_pos, internal_map):
    """Find best frontier cell with upper-half and pocket priority."""
    mid_row = internal_map.shape[0] // 2
    best = None
    best_score = -1
    for r in range(h):
        for c in range(w):
            if internal_map[r, c] != 0:
                continue
            if not has_unknown_neighbor(r, c, internal_map):
                continue
            dist = manhattan(my_pos, (r, c))
            if dist == 0:
                continue
            # Base score: inverse distance
            score = 1.0 / dist
            # Upper-half bonus: 2x score
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

**Search priority order:**
1. Upper-half hidden pockets (score × 6) — highest priority
2. Upper-half general area (score × 2)
3. Lower-half frontier (score × 1) — lowest priority

---

## Section 5: Curriculum Training Pipeline (`train.py`)

### Phase Structure

| Phase | Opponent | Epochs | Est. time | Gate |
|-------|----------|--------|-----------|------|
| 1 | SimpleGhostOpponent (100%) | 200 | ~15 min | epsilon <= 0.3 and catch rate >= 10%, or 200 epochs |
| 2 | Mixed (70% SimpleGhost / 30% GhostAgent) | 100 | ~25 min | epsilon <= 0.05, or 100 epochs |
| 3 | GhostAgent (100%) | 50 | ~20 min | 50 epochs |

Total: ~60 minutes. Epsilon decays continuously across phases.

### New CLI args:
```
--curriculum          Run all 3 phases sequentially
--phase {1,2,3}       Run a single phase
--simple-epochs N     Override phase 1 epochs (default: 200)
--mixed-epochs N      Override phase 2 epochs (default: 100)
--ghost-epochs N      Override phase 3 epochs (default: 50)
```

### Checkpoint format:
```python
checkpoint = {
    'model_state_dict': ...,
    'epoch': cum_epoch,
    'epsilon': ...,
    'phase': current_phase,
    'training_epochs': cum_epoch,  # for dynamic confidence threshold
}
torch.save(checkpoint, 'pacman_dqn_v2.pt')
```

### Model auto-detection (`agent.py`):
PacmanAgent loads either PacmanCNN or PacmanCNNv2 automatically: if state_dict contains "conv3.weight" keys, it is v2; otherwise v1.

---

## Section 6: Files Modified

| File | Changes |
|------|---------|
| `model.py` | Add PacmanCNNv2 class with BatchNorm + 3-layer CNN |
| `train.py` | Phase management, mixed opponent, upper-half explore reward, checkpoint format, restore SimpleGhostOpponent |
| `agent.py` | Dynamic confidence threshold, model auto-detection (v1/v2), upper-half frontier search priority, pocket bonus |

### Training command:
```
python train.py --curriculum --stochastic
```

---

## Constraints

- **Time**: Under 1 hour
- **GPU**: CUDA RTX 2050, model ~8 MB VRAM
- **Backward compat**: Original model and training script preserved
- **Output**: `pacman_dqn_v2.pt` (separate from original)
- **Ghost behavior**: Ghost spawns in upper half, hides in side corners and upper-middle alcoves — search prioritized accordingly
