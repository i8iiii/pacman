# Pacman CNN-DQN Agent: Model, Training & Data Assessment

## 1. CNN Architecture

### 1.1 Overview

The `PacmanCNN` is a lightweight 2-layer convolutional neural network designed for **CPU inference under 1ms**. It takes a 21x21 grid map state plus a 4-element last-move one-hot vector and outputs Q-values for the four possible actions: `[UP, DOWN, LEFT, RIGHT]`.

### 1.2 Layer-by-Layer Breakdown

```
Input: [B, 1, 21, 21] + [B, 4] last_move one-hot

CNN Feature Extractor:
  Conv2d(1 -> 32, kernel=3x3, stride=1, padding=1)  -> [B, 32, 21, 21]
  ReLU
  Conv2d(32 -> 64, kernel=3x3, stride=2, padding=1)  -> [B, 64, 11, 11]
  ReLU
  Flatten                                            -> [B, 7744]

DQN Head:
  Concat(flattened_features, last_move_vec)          -> [B, 7748]
  Linear(7748 -> 256)
  ReLU
  Dropout(p=0.1)
  Linear(256 -> 4)                                    -> Q-values [B, 4]
```

### 1.3 Parameter Count

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Conv1 (1x32x3x3) | 288 | 32 | 320 |
| Conv2 (32x64x3x3) | 18,432 | 64 | 18,496 |
| FC1 (7748->256) | 1,983,488 | 256 | 1,983,744 |
| FC2 (256->4) | 1,024 | 4 | 1,028 |
| **Total** | **2,003,232** | **356** | **~2,003,588** |

Model file size on disk: **7.7 MB** (float32 x ~2M parameters ~ 8 MB).

### 1.4 State Encoding

Each cell in the 21x21 grid is encoded as a float32:

| Value | Meaning |
|-------|---------|
| `1.0` | Wall (always visible) |
| `0.0` | Known empty path |
| `2.0` | Pacman's current position |
| `3.0` | Ghost's current position (if visible) |
| `-1.0` | Fog / unknown territory |

The CNN distinguishes fog cells (`-1.0`) from known empty cells (`0.0`), enabling partial-observability reasoning.

### 1.5 Design Observations

- **Shallow CNN**: Only 2 convolutional layers -- minimal for speed but limited in spatial reasoning depth.
- **Stride-2 downsampling**: Reduces the 21x21 feature map to 11x11 in the second conv layer (halving resolution). This may lose fine-grained spatial detail.
- **No batch normalization**: The network relies solely on ReLU activations and dropout regularization.
- **Last-move concatenation**: The previous action is injected as a 4-dim one-hot vector at the FC layer, not at the input. This means the CNN feature extractor operates purely on the current map state.
- **Small feature space**: 7744 CNN features + 4 last-move = 7748 dimensions before the DQN head -- relatively compact.

---

## 2. Weight Initialization Strategy

The model uses **different initialization schemes for different layers**:

### Conv2d Layers
```python
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(m.bias, 0)
```
- **Kaiming/He normal** initialization with `fan_out` mode preserves variance through ReLU activations.
- Appropriate for the two convolutional layers followed by ReLU.

### Hidden FC Layer (fc1)
```python
nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
nn.init.constant_(m.bias, 0)
```
- Same Kaiming initialization, also followed by ReLU.

### Output FC Layer (fc2)
```python
nn.init.xavier_uniform_(m.weight)
nn.init.constant_(m.bias, 0)
```
- **Xavier/Glorot uniform** initialization for the output layer (no activation function after it -- raw Q-values).
- This is a deliberate choice: Xavier uniform ensures the initial Q-values are centered around zero with controlled variance, preventing the agent from starting with strong biases toward any action.

### Assessment
The initialization strategy is sound:
- Kaiming for ReLU-activated layers prevents vanishing/exploding gradients.
- Xavier for the linear output layer avoids initial Q-value skew.
- All biases initialized to zero -- standard practice.

---

## 3. DQN Training Pipeline

### 3.1 Algorithm Configuration

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| Algorithm | **Double DQN** | Online net selects action, target net evaluates -- reduces overestimation bias |
| Loss function | **SmoothL1Loss (Huber)** | Robust to outliers, smoother gradients than MSE |
| Optimizer | **Adam** | Adaptive learning rates, momentum |
| Learning rate | **1 x 10^-3** (default) | Moderate, adjustable via `--lr` |
| Discount factor (gamma) | **0.99** | Strong emphasis on long-term rewards |
| Batch size | **64** | Standard for small-scale RL |
| Replay buffer capacity | **50,000** transitions | Circular deque, FIFO eviction |
| Min replay before training | **1,000** transitions | Ensures diverse initial samples |
| Target network update | **Soft update, tau = 0.005** | Every step: theta_target <- tau * theta_online + (1 - tau) * theta_target |
| Gradient clipping | **max_norm = 10.0** | Prevents exploding gradients |

### 3.2 Exploration Strategy

| Parameter | Value |
|-----------|-------|
| Initial epsilon | **1.0** (fully random) |
| Final epsilon | **0.05** (5% random) |
| Decay rate | **0.995 per epoch** |
| Decay schedule | After all episodes in an epoch complete |

With 100 epochs (default), epsilon reaches approximately 0.05 x 0.995^100 ~ 0.030 -- slightly below the configured minimum (clamped at 0.05).

### 3.3 Training Loop

```
For each epoch (1..N):
    For each episode (1..M):
        Reset environment
        For each step (max 200):
            Select action via epsilon-greedy
            Execute action -> (next_state, reward, done)
            Push transition to replay buffer
            Sample batch -> train_step (if buffer >= min_replay_size)
            Soft-update target network
        Decay epsilon at end of epoch
    Save checkpoint every 10 epochs and on best catch rate
```

Each epoch runs **20 episodes**, each capped at **200 steps**. Total training environment steps per epoch: up to 4,000.

### 3.4 Reward Shaping

| Event | Reward | Rationale |
|-------|--------|-----------|
| Catch Ghost | **+100.0** | Terminal success -- large positive signal |
| Distance delta (closer) | **+2.0 per cell** | Shaped reward for reducing Manhattan distance |
| Distance delta (farther) | **-2.0 per cell** | Penalty for moving away |
| Wall bump (no movement) | **-1.0** | Discourage invalid moves |
| Time penalty | **-0.1 per step** | Encourage fast capture |
| Timeout (200 steps) | **-50.0** | Heavy penalty for failure to catch |

The shaped distance reward (`dist_delta * 2.0`) provides a dense learning signal, helping the agent learn even when catches are rare early in training. The timeout penalty ensures the agent is incentivized to finish quickly.

---

## 4. Training Data & Opponent

### 4.1 Training Opponent: `SimpleGhostOpponent`

The training script uses `SimpleGhostOpponent` -- a **rule-based, non-learning opponent**:

- **Decision logic**: BFS flee -- picks the move that maximizes BFS distance from Pacman.
- **Behavior**: Always runs away; no hiding, no camping, no strategic concealment.
- **Fallback**: Random valid move if BFS path is unavailable.
- **No memory or fog handling**: Always acts with full map knowledge (from the training env perspective).

```python
class SimpleGhostOpponent:
    def step(self, map_state, my_pos, enemy_pos, step_number):
        best_move = Move.STAY
        best_dist = -1
        for move in [UP, DOWN, LEFT, RIGHT]:
            if valid(nr, nc, map_state):
                dist = bfs_distance((nr, nc), enemy_pos, map_state)
                if dist > best_dist:
                    best_dist = dist
                    best_move = move
        return best_move
```

### 4.2 Training Data Characteristics

- **Self-play** format: Pacman CNN-DQN vs SimpleGhostOpponent (BFS flee).
- **State distribution**: The training ghost always flees -- Pacman learns to chase a fleeing target.
- **No hiding behavior**: The ghost never camps, uses hideouts, or breaks line of sight strategically.
- **No pursuit modeling**: The ghost doesn't simulate being chased or tracked.
- **Fog-of-war optional**: `--obs-radius N` flag enables limited-visibility training, but the ghost opponent itself doesn't use fog mechanics strategically.

### 4.3 What the DQN Actually Learns

Given this training setup, the model learns:
1. **Map navigation**: Moving through corridors without hitting walls.
2. **Greedy pursuit**: Closing Manhattan distance to a visible ghost.
3. **Basic obstacle avoidance**: Wall bumps incur penalties.
4. **Speed-2 path checking**: When fog-of-war is enabled, handling partial observability.

What it does NOT learn:
- Dealing with a ghost that stays still and hides
- Dealing with a ghost that breaks line of sight intentionally
- Pursuing through complex maps where BFS flee leads to dead ends
- Strategic search patterns when the ghost is not visible

---

## 5. Will Training Against the GhostAgent Help?

### 5.1 The GhostAgent (`agent.py`)

The `GhostAgent` in the same folder is a **sophisticated, multi-phase Hide agent** developed across 20+ implementation phases (P00-P09, R00-R08B). Its capabilities include:

| Phase | Capability |
|-------|------------|
| P00-P02 | Legal movement, logging, geometry, line-of-sight |
| P03-P04 | Campsite scanning, scout/camp controller |
| P06-P07 | Visible escape (at campsite and mobile) |
| P08-P09 | Pursuit tracking, broad belief model, interception |
| R00-R07 | Concealed hideout strategy (visibility footprints, terminal pockets, deterministic selection, compromise lifecycle, retargeting) |
| R08-A/B | Road-aware hiding (detect major roads, exclude hideouts visible from vertical approach roads) |

The `GhostAgent` is a comprehensive implementation with:
- **Concealed hideout selection**: Finds structurally safe hiding spots based on visibility footprints, gate depth, inspection depth, and backtracking requirements.
- **Road cycle awareness**: Avoids hideouts visible from vertical approach roads.
- **Pursuit belief tracking**: Maintains a broad belief set of possible Pacman positions and plans evasively.
- **Visible escape**: Uses tactical junction geometry to escape when spotted.
- **Deterministic hideout ranking**: Selects the best hideout by concealment class, gate depth, inspection depth, visibility footprint, backtracking, spawn distance, and ghost route distance.

### 5.2 Comparison: Training Opponents

| Capability | SimpleGhostOpponent (training) | GhostAgent (same folder) |
|------------|-------------------------------|--------------------------|
| Flee behavior | BFS flee (always) | Strategic hide + escape |
| Hide/camp | Never | Yes, at selected hideouts |
| Line-of-sight tactics | None | Breaks LOS, uses occlusion |
| Dead-end avoidance | BFS natural | Explicit detection + routing |
| Persistence | Always moves | Stays still when hidden |
| Partial observability | None | Full fog reasoning |
| Pursuit counter-play | None | Belief tracking, interception avoidance |
| Road awareness | None | Detects and avoids road-visible hideouts |

### 5.3 Assessment: Would Training Against GhostAgent Help?

**Yes, significantly.** Training the Pacman CNN-DQN against the full GhostAgent would produce a more intelligent seeker agent. Here's why:

1. **Diverse opponent behavior**: The GhostAgent hides, flees, and uses concealment -- Pacman would learn to handle all three modes rather than just chasing a perpetually fleeing target.

2. **Realistic hide-and-seek dynamics**: The GhostAgent actually tries to break line of sight and reach concealed positions. The Pacman would need to learn search patterns, not just pursuit.

3. **Strategic depth**: The Pacman would face situations where:
   - The ghost is not visible (must search/explore)
   - The ghost is stationary in a hideout (must learn to inspect)
   - The ghost escapes when spotted (must learn to cut off escape routes)
   - The ghost uses road-awareness (must learn to patrol roads)

4. **Better generalization**: Training against a sophisticated opponent produces more robust Q-value estimates that generalize better to unknown opponents in the Arena.

5. **Escape counter-play**: The GhostAgent's P06/P07 escape mechanisms would teach the Pacman to anticipate and counter escape routes.

However, there are practical challenges:
- **Training speed**: The GhostAgent is significantly more computationally expensive per step (up to ~80ms) than the SimpleGhostOpponent (~0.01ms). Training would be ~1000x slower.
- **Integration effort**: The GhostAgent requires specific initialization (log paths, map diagnostics) that the training pipeline doesn't currently support.
- **Stability**: The GhostAgent's deterministic hideout selection might create repetitive training scenarios; some randomization would be beneficial.

### 5.4 Suggested Approach

For maximum training benefit, consider:

1. **Curriculum learning**: Start training against SimpleGhostOpponent (current approach), then progressively introduce the GhostAgent once the Pacman achieves a reasonable catch rate.

2. **Mixed training**: Randomly select between SimpleGhostOpponent and GhostAgent (e.g., 70/30 split) to maintain training speed while exposing the model to sophisticated behavior.

3. **Pre-extracted datasets**: Generate a dataset of GhostAgent behaviors offline (positions, escape moves, beliefs), then sample from it during training to avoid the per-step complexity.

4. **Simplified GhostAgent variant**: Create a "Lite" version of the GhostAgent that uses only the core hide/escape logic without the full belief tracking and road awareness, balancing realism with training speed.

---

## 6. PacmanAgent Runtime Architecture

At inference time, the `PacmanAgent` uses a **hybrid DQN + A* fallback** strategy:

```
step(map_state, my_position, enemy_position, step_number)
|
+-- Update internal map memory (merge fog observations over time)
|
+-- IF enemy_position is NOT None (Ghost visible):
|   +-- DQN forward pass -> Q-values for 4 moves
|   +-- IF confidence = max(Q) - mean(Q) > 0.5:
|   |   +-- Use DQN move (must be a valid non-wall move)
|   +-- ELSE (low confidence):
|       +-- A* pathfinding to enemy_position
|
+-- IF enemy_position is None (Ghost hidden):
|   +-- IF last_known_enemy_pos exists AND steps_since_seen <= 10:
|   |   +-- A* to last known position
|   +-- ELSE (lost track):
|       +-- A* to nearest frontier cell (boundary between known/unknown)
|
+-- Compute speed steps (1 or 2 based on path straightness) -> return (Move, steps)
```

### DQN Confidence Threshold Analysis

The confidence metric is `max(Q) - mean(Q)` with a threshold of **0.5**. This means:
- If Q-values are [2.1, 1.0, 1.2, 0.9]: max=2.1, mean=1.3, confidence=0.8 -> **DQN used**
- If Q-values are [0.5, 0.3, 0.4, 0.2]: max=0.5, mean=0.35, confidence=0.15 -> **A* fallback**

This is a reasonable heuristic, but has limitations:
- It assumes the DQN's Q-value spread correlates with action quality -- not always true in partially-trained models.
- At 0.5, roughly half of uncertain states will fall back to A*, which limits the DQN's influence on actual gameplay.

---

## 7. Model File & Deployment

| Property | Value |
|----------|-------|
| File | `pacman_dqn.pt` |
| Format | PyTorch `state_dict` (no optimizer state) |
| Size | 7.7 MB |
| Loading | `torch.load(model_path, map_location='cpu', weights_only=True)` |
| Fallback names | Tries `pacman_dqn.pt` first, then `best_pacman_dqn.pt` |

The agent also stores a `best_pacman_dqn.pt` checkpoint whenever a new best catch rate is achieved during training.

---

## 8. Summary & Key Findings

### Strengths
- **Clean, well-structured architecture**: Clear separation of CNN feature extractor and DQN head.
- **Sound initialization**: Kaiming for ReLU layers, Xavier for output.
- **Double DQN**: Reduces Q-value overestimation bias.
- **Dense reward shaping**: Distance-based rewards provide learning signal even without catches.
- **Graceful degradation**: A* fallback when DQN is uncertain or ghost is hidden.
- **Fog-of-war support**: Both training and inference handle partial observability.
- **Model is deployable**: Single `.pt` file, CPU-friendly inference, well under 1ms.

### Key Gaps
1. **Training opponent mismatch**: The Pacman is trained against a simple BFS-flee ghost, but deployed against sophisticated multi-phase Hide agents in the Arena. The DQN has never seen a ghost that camps, hides behind walls, or uses belief-based evasion.
2. **No curriculum or multi-opponent training**: Training exclusively against SimpleGhostOpponent limits generalization.
3. **Confidence threshold is heuristic**: The `max(Q) - mean(Q) > 0.5` threshold may filter out correct DQN decisions in edge cases.
4. **Shallow CNN**: Only 2 conv layers with stride-2 downsampling -- may miss fine-grained spatial patterns.
5. **No batch normalization**: Training stability depends entirely on initialization and gradient clipping.

### Recommendations
1. **Train against the GhostAgent** (curriculum or mixed approach) to improve generalization.
2. **Lower the confidence threshold** gradually as training progresses to rely more on the DQN.
3. **Add batch normalization** after conv layers for more stable training.
4. **Increase CNN depth** to 3-4 layers if inference time budget allows.
5. **Add a search/explore reward** during fog-of-war training to encourage the agent to actively seek the ghost when it's not visible.
