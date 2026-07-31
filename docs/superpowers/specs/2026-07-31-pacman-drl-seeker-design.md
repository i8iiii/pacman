# Pacman DRL Seeker Agent — Design Spec

**Date**: 2026-07-31  
**Status**: Approved  
**Algorithm**: Double DQN + DRQN (CNN + LSTM)  
**Environment**: Partially Observable Pacman vs Ghost Arena  

---

## 1. Overview

Design a Deep Reinforcement Learning agent for the **Pacman (Seeker)** role in the Pacman vs Ghost Arena. The agent uses Double DQN with an LSTM layer (DRQN) to handle partial observability (fog of war, observation radius = 5). Training uses multiple ghost policies for robustness, with a curriculum that gradually increases difficulty.

### 1.1 Constraints
- **Pacman only** — ghost agents use pre-existing rule-based policies, not modified
- Observation radius: **5** for both Pacman and Ghost
- Map size: 21 × 19 grid
- Max steps per episode: 200 (configurable)

---

## 2. State Representation

### 2.1 Single Frame (6 channels, 21×19 each)

| Channel | Name | Encoding |
|---------|------|----------|
| 0 | Walls | `map == 1` → 1.0, else 0.0 |
| 1 | Visible empty | `map == 0` → 1.0, else 0.0 |
| 2 | Unseen/Fog | `map == -1` → 1.0, else 0.0 |
| 3 | Pacman position | Binary mask at `(row, col)` |
| 4 | Ghost position | Binary mask if visible, else all zeros |
| 5 | Last known ghost | Binary mask at last seen ghost position |

### 2.2 Frame Stacking

Stack **4 consecutive frames** → input tensor shape `(21, 19, 24)`.

4 frames are sufficient for the agent to infer movement direction (velocity) of both itself and the ghost, which is critical in a synchronous-move environment.

---

## 3. Action Space

**Discrete 5 actions**: UP (0), DOWN (1), LEFT (2), RIGHT (3), STAY (4)

DQN outputs 5 Q-values, one per action. Selection via ε-greedy during training, argmax during evaluation.

> **Note**: Pacman speed multiplier `(Move, steps)` is not used in this version to keep the action space simple. Can be added as a future extension.

---

## 4. Reward Function — Potential-Based Shaping

### 4.1 Core Formula

```
R(s, a, s') = R_terminal + R_step + R_shaped + R_exploration + R_visibility
```

### 4.2 Terminal Rewards

| Event | Reward |
|-------|--------|
| Capture ghost (distance < threshold) | **+10** |
| Episode ends without capture | **0** (no penalty, avoids bias) |

### 4.3 Step Cost

| Event | Reward |
|-------|--------|
| Each step taken | **-0.01** |
| STAY action | **-0.05** (extra penalty for standing still) |

### 4.4 Potential-Based Shaping

```
Φ(s) = -d / D_max
  where d = Manhattan distance to ghost (or last known position if not visible)
        D_max = height + width (max possible distance on grid)

R_shaped = γ · Φ(s') - Φ(s)
  where γ = 0.99 (discount factor)
```

This shaping is **policy-invariant**: it does not change the optimal policy.

### 4.5 Exploration Bonus

| Event | Reward | Formula |
|-------|--------|---------|
| First visit to a cell | Positive | `+0.1 / sqrt(N_visits)` |
| Revisit | 0 | — |

Decays naturally as cells are visited more often. Encourages systematic exploration when ghost is not visible.

### 4.6 Visibility Events

| Event | Reward |
|-------|--------|
| Lost sight of ghost | **-0.3** |
| Regained sight of ghost (found in previously unseen area) | **+0.3** |

---

## 5. Network Architecture (DRQN)

```
Input: (21, 19, 24)  [4 frames × 6 channels]
        │
        ▼
┌──────────────────────────┐
│ Conv2D(24→32, 3×3, s=1)  │ + ReLU
│ Conv2D(32→64, 3×3, s=1)  │ + ReLU
│ Conv2D(64→64, 3×3, s=1)  │ + ReLU
│ Flatten                   │
│ Dense(64→128)             │ + ReLU
└──────────┬───────────────┘
           │ feature vector (128d)
           ▼
┌──────────────────────────┐
│ LSTM(128→128)             │
│ Dense(128→64)             │ + ReLU
│ Dense(64→5)               │ Q-values
└──────────────────────────┘
```

### 5.1 DRQN Details
- LSTM hidden state `h_t` is reset to zero at episode start
- During training: sample **sequences of 8 transitions** from replay buffer, unroll LSTM through the sequence
- During inference: maintain hidden state across steps within an episode

---

## 6. Curriculum Learning

3 stages, advancing when win rate > 60% over 100 eval episodes or reaching step cap.

| Stage | Steps | Obs Radius | Ghost Policies | Description |
|-------|-------|-----------|----------------|-------------|
| 1 (Easy) | 0 – 30K | 10 | Random 70%, Greedy 30% | Learn basic approach behavior |
| 2 (Medium) | 30K – 60K | 5 | Greedy 50%, Random 50% | Reduced vision, smarter ghost |
| 3 (Hard) | 60K – 100K | 3 | Greedy 70%, Minimax 30% | Realistic conditions |

---

## 7. Training Details

### 7.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| γ (discount) | 0.99 |
| ε_start | 1.0 |
| ε_end | 0.05 |
| ε_decay steps | 50,000 |
| Learning rate | 2.5e-4 |
| Batch size | 32 |
| Sequence length (DRQN) | 8 |
| Target network sync | every 2,000 steps |
| Replay buffer size | 100,000 transitions |
| Prioritization α | 0.6 |
| Prioritization β | 0.4 → 1.0 (annealed) |
| Optimizer | Adam |

### 7.2 Per-Episode Loop
1. Reset environment, select ghost policy per curriculum stage
2. Reset LSTM hidden state
3. For t = 1 … max_steps:
   - Build stacked state tensor from frame buffer
   - ε-greedy action selection
   - Step environment → (next_obs, reward, done)
   - Store transition in prioritized replay buffer
   - If buffer has enough: sample batch, train one step
   - Decay ε, sync target network periodically
4. Log: episode reward, steps, win/loss, ε

### 7.3 Evaluation
- Every **500 training episodes**: run 100 eval games (ε=0.05)
- Record: win rate, average steps to capture, average episode reward

---

## 8. Ghost Policies (Training Opponents)

| Policy | Behavior | Purpose |
|--------|----------|---------|
| **Random** | Uniform random from valid moves | Baseline exploration |
| **Greedy Evasion** | Move to adjacent cell maximizing Manhattan distance from Pacman | Basic evasion |
| **Minimax** | 2-ply minimax search (if implemented) | Advanced adversarial opponent |

All ghost policies use the same `GhostAgent` interface without modification.

---

## 9. File Structure

```
src/
├── rl/
│   ├── __init__.py
│   ├── pacman_env.py        # Gym-style env wrapping Arena/Environment
│   ├── state_builder.py     # Multi-channel frame builder + stack
│   ├── dqn_agent.py         # CNN+LSTM network + Double DQN agent
│   ├── replay_buffer.py     # Prioritized Experience Replay
│   ├── reward.py            # Potential-based reward shaping
│   ├── ghost_policies.py    # Training ghost policy implementations
│   ├── train.py             # Training loop with curriculum
│   └── config.py            # Hyperparameters and constants
```

---

## 10. Dependencies

- `numpy` (already required)
- `torch` (PyTorch for neural networks)
- `gymnasium` (Gym-style environment interface, optional — can implement minimal interface directly)

---

## 11. Success Criteria

- [ ] Agent achieves > 60% win rate against greedy ghost in Stage 1
- [ ] Agent achieves > 40% win rate against mixed ghost policies in Stage 3
- [ ] Training is stable (no NaN losses, no reward collapse)
- [ ] Agent generalizes: performs above random against unseen ghost policies
- [ ] Code is compatible with existing Arena infrastructure (no changes to `environment.py`, `arena.py`, `agent_interface.py`)
