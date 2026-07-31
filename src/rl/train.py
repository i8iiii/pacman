"""Training loop with curriculum learning for DRL Pacman Seeker agent.

Provides train(config) callable and python -m rl.train entry point.
"""
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import random
from collections import deque

from rl.config import Config
from rl.pacman_env import PacmanEnv
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer

_CURRICULUM_STEP_THRESHOLDS = (30000, 60000)

def get_curriculum_stage(config, total_steps):
    for i, threshold in enumerate(_CURRICULUM_STEP_THRESHOLDS):
        if total_steps < threshold:
            return i
    return len(_CURRICULUM_STEP_THRESHOLDS)

def select_ghost_policy(config, stage):
    obs_radius, weights = config.curriculum_stages[stage]
    policies = list(weights.keys())
    probs = list(weights.values())
    return random.choices(policies, weights=probs, k=1)[0], obs_radius

def evaluate(config, agent, n_episodes=100):
    wins = 0; total_steps = 0; total_reward = 0.0
    for _ in range(n_episodes):
        env = PacmanEnv(config, ghost_policy_name="greedy")
        state = env.reset()
        hidden = agent.get_initial_hidden()
        done = False; ep_reward = 0.0; ep_steps = 0
        while not done:
            action, hidden = agent.get_action(state, hidden, epsilon=0.05)
            state, reward, done, info = env.step(action)
            ep_reward += reward; ep_steps += 1
            if done and info.get("result") == "pacman_wins":
                wins += 1
        total_steps += ep_steps; total_reward += ep_reward
    return {"win_rate": wins / n_episodes, "avg_steps": total_steps / n_episodes,
            "avg_reward": total_reward / n_episodes}

def train(config=None):
    if config is None:
        config = Config()
    agent = DQNAgent(config)
    replay_buffer = ReplayBuffer(config)
    total_steps, episode = 0, 0
    episode_rewards = deque(maxlen=100)

    while total_steps < config.total_training_steps:
        episode += 1
        stage = get_curriculum_stage(config, total_steps)
        ghost_policy, obs_radius = select_ghost_policy(config, stage)
        config.pacman_obs_radius = obs_radius
        config.ghost_obs_radius = obs_radius

        env = PacmanEnv(config, ghost_policy_name=ghost_policy)
        state = env.reset()
        hidden = agent.get_initial_hidden()
        done = False; ep_reward = 0.0

        while not done and total_steps < config.total_training_steps:
            epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * np.exp(
                -total_steps / config.epsilon_decay_steps)
            old_hidden = (hidden[0].copy(), hidden[1].copy())
            action, hidden = agent.get_action(state, hidden, epsilon)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done, old_hidden)
            if len(replay_buffer) >= config.batch_size * 2:
                loss = agent.train_step(replay_buffer)
                if agent.train_steps % config.target_sync_steps == 0:
                    agent.sync_target()
            state = next_state
            ep_reward += reward
            total_steps += 1

        episode_rewards.append(ep_reward)
        if episode % 10 == 0:
            avg_r = np.mean(episode_rewards) if episode_rewards else 0.0
            print(f"Ep {episode:5d} | Steps {total_steps:7d} | AvgR {avg_r:7.2f} | Stage {stage} | eps {epsilon:.3f}")

        if episode % config.eval_interval == 0:
            results = evaluate(config, agent, config.eval_episodes)
            print(f"Eval: Win {results['win_rate']:.2%} | Steps {results['avg_steps']:.1f} | Reward {results['avg_reward']:.2f}")

    model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    agent.save(str(model_dir / "pacman_dqn_final.pth"))
    print(f"Model saved to {model_dir / 'pacman_dqn_final.pth'}")
    return agent

if __name__ == "__main__":
    train()
