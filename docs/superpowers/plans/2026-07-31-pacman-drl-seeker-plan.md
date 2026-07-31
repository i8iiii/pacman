# Pacman DRL Seeker Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Double DQN + DRQN agent that learns to catch the ghost in a partially observable maze.

**Architecture:** Gym-style `PacmanEnv` wraps the existing `Environment` class, a `StateBuilder` produces stacked multi-channel tensors, a `RewardCalculator` computes potential-based shaped rewards, `GhostPolicy` classes provide training opponents, a `DQNAgent` with CNN+LSTM learns via prioritized experience replay, and `train.py` orchestrates curriculum learning.

**Tech Stack:** Python 3.7+, PyTorch, NumPy, existing Arena framework

## Global Constraints

- No modifications to existing source files (`environment.py`, `arena.py`, `agent_interface.py`, `agent_loader.py`, `visualizer.py`)
- Pacman only — ghost agents use pre-existing rule-based policies
- Observation radius: 5 for both agents (configurable)
- Max steps per episode: 200 (configurable)
- Map: 21×19 grid, 1=wall, 0=empty, -1=unseen
- 5 discrete actions: UP, DOWN, LEFT, RIGHT, STAY
- Training via Double DQN with DRQN (frame stacking + LSTM)
- Curriculum: obs radius 10→7→5

---

### Task 1: Config Module

**Files:**
- Create: `src/rl/__init__.py`
- Create: `src/rl/config.py`

**Interfaces:**
- Produces: `Config` dataclass with all hyperparameters

- [ ] **Step 1: Create package init and config module**

Create `src/rl/__init__.py`:
```python
"""Deep Reinforcement Learning module for Pacman Seeker agent."""
```

Create `src/rl/config.py` with a `Config` dataclass containing fields from the spec:
- Environment: `map_height=21`, `map_width=19`, `max_steps=200`, `n_actions=5`, `n_channels=6`, `n_frames=4`
- Observation: `pacman_obs_radius=5`, `ghost_obs_radius=5`, `capture_distance=1`
- DQN: `gamma=0.99`, `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay_steps=50000`, `learning_rate=2.5e-4`, `batch_size=32`, `sequence_length=8`, `target_sync_steps=2000`, `replay_buffer_capacity=100000`
- PER: `per_alpha=0.6`, `per_beta_start=0.4`, `per_beta_end=1.0`, `per_epsilon=1e-6`
- Training: `total_training_steps=100000`, `eval_interval=500`, `eval_episodes=100`
- Curriculum stages: `((10, {"random":0.7,"greedy":0.3,"minimax":0.0}), (7, {"random":0.5,"greedy":0.5,"minimax":0.0}), (5, {"random":0.0,"greedy":0.7,"minimax":0.3}))`
- Reward weights: `reward_capture=10.0`, `reward_step=-0.01`, `reward_stay_penalty=-0.05`, `reward_exploration_base=0.1`, `reward_lost_sight=-0.3`, `reward_regained_sight=0.3`
- Device: `"cpu"`

Full implementation in spec section 7.1.

- [ ] **Step 2: Verify import works**

Run: `python -c "from rl.config import Config; c = Config(); print(c.n_actions)"`
Expected: prints `5`

- [ ] **Step 3: Commit**

```bash
git add src/rl/__init__.py src/rl/config.py
git commit -m "feat: add DRL config module with hyperparameters"
```


### Task 2: State Builder

**Files:**
- Create: `src/rl/state_builder.py`
- Create: `tests/test_state_builder.py`

**Interfaces:**
- Consumes: `Config` from Task 1
- Produces: `StateBuilder(config)`, `.reset()`, `.build(map_state, my_pos, enemy_pos, step) -> np.ndarray` shape `(n_frames*n_channels, H, W)`

- [ ] **Step 1: Write test in tests/test_state_builder.py**
  - Test output shape = (24, 21, 19)
  - Test channel content: wall channel marks walls, visible empty channel marks 0s, unseen channel marks -1s, pacman channel has 1 at position, ghost channel marks visible ghost
  - Test enemy=None: ghost channel all zeros, last_known retains previous value
  - Test frame stacking: after 4+ frames, buffer contains 4 frames

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement StateBuilder**
  - Internal deque(frame_buffer) with maxlen=n_frames, pre-filled with zeros
  - `_build_single_frame()` creates (n_channels, H, W) array:
    - Ch0: map==1, Ch1: map==0, Ch2: map==-1
    - Ch3: binary at my_pos, Ch4: binary at enemy_pos if visible
    - Ch5: binary at last_known_enemy_pos (tracked internally)
  - `build()` appends current frame, returns `np.concatenate(list(buffer), axis=0)`
  - `reset()` clears buffer, resets last_known

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**


### Task 3: Reward Calculator

**Files:**
- Create: `src/rl/reward.py`
- Create: `tests/test_reward.py`

**Interfaces:**
- Consumes: `Config` from Task 1
- Produces: `RewardCalculator(config)`, `.reset()`, `.compute(prev_pos, prev_enemy, prev_visible, curr_pos, curr_enemy, curr_visible, done, action) -> float`

- [ ] **Step 1: Write tests** — capture=+10, step cost=-0.01, stay=-0.05, approach>0, retreat<0, first visit>revisit, lost sight penalty, regained sight bonus

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement RewardCalculator**
  - `_potential(pos, enemy_pos)`: Φ = -manhattan_distance / (H+W)
  - `compute()`: shaped_reward = γ·Φ(s'') - Φ(s)
  - Terminal: capture=+10, timeout=0
  - Step cost: -0.01, stay extra=-0.05
  - Exploration: first visit bonus, tracked via visit_counts dict
  - Visibility: lost_sight=-0.3, regained_sight=+0.3
  - Tracks _prev_enemy_pos and _prev_visible internally

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

---

### Task 4: Ghost Policies

**Files:**
- Create: `src/rl/ghost_policies.py`
- Create: `tests/test_ghost_policies.py`

**Interfaces:**
- Produces: `RandomGhostPolicy().step(map_state, my_pos, enemy_pos, step) -> Move`
- Produces: `GreedyGhostPolicy().step(map_state, my_pos, enemy_pos, step) -> Move`
- Produces: `get_ghost_policy(name) -> policy instance`

- [ ] **Step 1: Write tests** — random returns valid Move, greedy moves away from Pacman, handles enemy=None

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement**
  - `RandomGhostPolicy`: uniform random from valid moves (not walls), helpers: `_get_valid_moves(map_state, pos)`
  - `GreedyGhostPolicy`: pick move maximizing Manhattan distance from enemy_pos; fall through to random if enemy is None
  - `get_ghost_policy(name)`: factory returning instance by name

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

---

### Task 5: PacmanEnv (Gym-style Wrapper)

**Files:**
- Create: `src/rl/pacman_env.py`
- Create: `tests/test_pacman_env.py`

**Interfaces:**
- Consumes: `Environment`, `StateBuilder`, `RewardCalculator`, `Config`
- Produces: `PacmanEnv(config, ghost_policy_name).reset() -> state`, `.step(action) -> (next_state, reward, done, info)`

- [ ] **Step 1: Write tests** — reset returns (24,21,19) float32, step returns valid tuple, handles max_steps termination, works with random/greedy ghost

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement PacmanEnv**
  - Wraps `Environment`, uses `StateBuilder` for state, `RewardCalculator` for reward
  - `ACTION_MAP = {0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY}`
  - `reset()`: reset env, state builder, reward calc, ghost policy; return initial stacked state
  - `step(action)`: sanitize action→pacman_move, get ghost obs→ghost move via policy, env.step(), compute reward, build next state, track prev positions/visibility
  - Info dict: result, step, positions, enemy_visible

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

---

### Task 6: Prioritized Replay Buffer

**Files:**
- Create: `src/rl/replay_buffer.py`
- Create: `tests/test_replay_buffer.py`

**Interfaces:**
- Consumes: `Config` from Task 1
- Produces: `ReplayBuffer(config)`, `.push(state, action, reward, next_state, done, hidden)`, `.sample(batch_size) -> (states, actions, rewards, next_states, dones, hiddens, indices, weights)`, `.update_priorities(indices, td_errors)`

- [ ] **Step 1: Write tests** — push+sample, sequence sampling (seq_len=4), priority updates

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement**
  - `SumTree` class: binary tree for O(log N) weighted sampling
  - `ReplayBuffer`: stores (state, action, reward, next_state, done, hidden) tuples
  - `push()`: adds with max_priority^alpha
  - `sample()`: returns sequences of length `sequence_length`, importance sampling weights
  - `update_priorities()`: `(abs(td_error) + epsilon)^alpha`
  - `update_beta()`: linear anneal from beta_start to beta_end

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

---

### Task 7: DQN Agent (CNN+LSTM + Double DQN)

**Files:**
- Create: `src/rl/dqn_agent.py`
- Create: `tests/test_dqn_agent.py`

**Interfaces:**
- Consumes: `Config` from Task 1, `ReplayBuffer` from Task 6
- Produces: `DRQNNetwork(config)` (nn.Module), `DQNAgent(config)`, `.get_action(state, hidden, epsilon) -> (action, new_hidden)`, `.train_step(replay_buffer) -> loss`, `.sync_target()`, `.save(path)`, `.load(path)`

- [ ] **Step 1: Write tests** — network output shape (batch,5), epsilon-greedy exploration, target sync copies weights, save/load roundtrip

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement DRQNNetwork**
  - Conv2D(24→32,3×3,s=1,p=1), Conv2D(32→64,3×3,s=1,p=1), Conv2D(64→64,3×3,s=1,p=1)
  - All with ReLU; Flatten→Dense(21×19×64,128)→ReLU
  - LSTM(128→128, batch_first=False)
  - Dense(128→64)+ReLU, Dense(64→5)
  - `forward(x, hidden)`: handles both (B,C,H,W) and (S,B,C,H,W) shapes

- [ ] **Step 4: Implement DQNAgent**
  - Online + target DRQNNetwork, Adam optimizer
  - `get_action()`: epsilon-greedy, returns action + new hidden (detached to numpy)
  - `train_step()`: sample sequences from replay, unroll LSTM (seq_len, batch, ...), Double DQN target (online selects action, target evaluates), MSE loss with IS weights, gradient clipping, update priorities
  - `sync_target()`: copy online→target
  - `save()`/`load()`: torch save/load with optimizer state

- [ ] **Step 5: Run test, verify PASS**

- [ ] **Step 6: Commit**

---

### Task 8: Training Loop + Curriculum

**Files:**
- Create: `src/rl/train.py`

**Interfaces:**
- Consumes: All modules from Tasks 1-7
- Produces: `train(config)` callable, `python -m rl.train` entry point

- [ ] **Step 1: Implement train()**
  - `get_curriculum_stage(total_steps)`: returns 0/1/2 based on thresholds (30000, 60000)
  - `select_ghost_policy(stage)`: weighted random from stage config
  - `evaluate(agent, n_episodes)`: runs eval with ε=0.05, returns {win_rate, avg_steps, avg_reward}
  - Main loop:
    - Per episode: set obs radius + ghost policy from curriculum stage
    - Per step: ε-greedy (exponential decay), store transition, train if buffer is full enough, sync target periodically
    - Log every 10 episodes, eval every 500 episodes
    - Save final model to `models/pacman_dqn_final.pth`

- [ ] **Step 2: Verify imports and basic structure**

Run: `python -c "from rl.train import train, Config; print('OK')"`
Expected: prints OK

- [ ] **Step 3: Commit**

---

### Task 9: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write test** — minimal training loop (200 steps), verify no NaN losses, agent trained, buffer populated

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

---

### Task 10: Add PyTorch dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add torch to requirements.txt**

Append `torch>=2.0.0` to requirements.txt

- [ ] **Step 2: Install**

Run: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

- [ ] **Step 3: Commit**

---

## Summary

| Task | Files Created | Description |
|------|-------------|-------------|
| 1 | `config.py`, `__init__.py` | Hyperparameters |
| 2 | `state_builder.py`, test | Multi-channel frame stacking |
| 3 | `reward.py`, test | Potential-based reward shaping |
| 4 | `ghost_policies.py`, test | Training opponents |
| 5 | `pacman_env.py`, test | Gym-style env wrapper |
| 6 | `replay_buffer.py`, test | Prioritized experience replay |
| 7 | `dqn_agent.py`, test | CNN+LSTM + Double DQN |
| 8 | `train.py` | Curriculum training loop |
| 9 | `test_integration.py` | End-to-end test |
| 10 | `requirements.txt` (modify) | Add torch dependency |
