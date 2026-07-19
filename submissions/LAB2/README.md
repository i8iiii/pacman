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
| Target network update | Soft (tau=0.005) every step |
| Loss | SmoothL1 (Huber) |
| Optimizer | Adam, lr=1e-3 |
| Epsilon schedule | 1.0 → 0.05, decay 0.995/epoch |
| Gradient clipping | max_norm=10.0 |
| Batch size | 64 |
| Discount factor (gamma) | 0.99 |
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
- **A\* pathfinding**: completes in <5ms on 21x21 grid
- **Total step time**: well under 100ms (far below 1.0s timeout)
- **Model size**: ~7.6MB (`pacman_dqn.pt`)
