#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import os
import time
import pickle
import logging
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecNormalize, unwrap_vec_normalize

from common.buffers import *
from sac_mamba_ensemble import SAC_Ensemble


logger = logging.getLogger(__name__)

# =========================
# Environment: Pendulum-v1
# =========================
# NOTE: Pendulum-v1 has continuous action in [-2, 2] and obs dim = 3
def make_env():
    # Gym wraps Pendulum with a TimeLimit (200 steps)
    env = gym.make('Pendulum-v1')
    return env

mode = 'train'  # set to 'test' if you want to load a saved VecNormalize and evaluate

base_save_dir = './pendulum_runs'
os.makedirs(base_save_dir, exist_ok=True)

env_raw = make_env()
env = DummyVecEnv([make_env])

if mode == 'train':
    #env = VecNormalize(env, gamma=0.99)
    print(mode)

# =========================
# Spaces
# =========================
# Use env spaces directly to configure SAC_Trainer
# Pendulum obs: Box(3,), act: Box(1,) in [-2, 2]
gym_obs_space  = env_raw.observation_space
gym_act_space  = env_raw.action_space

# Provide these to your trainer as Box spaces
state_space  = spaces.Box(low=-np.inf, high=np.inf, shape=gym_obs_space.shape, dtype=np.float32)
action_space = spaces.Box(low=gym_act_space.low, high=gym_act_space.high, shape=gym_act_space.shape, dtype=np.float32)

action_dim   = action_space.shape[0]

action_range = float(np.max(np.abs(action_space.high)))  

# =========================
# Replay buffer & Trainer
# =========================
replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer_compressed(replay_buffer_size)

hidden_dim = 32
sac_trainer = SAC_Ensemble(replay_buffer, state_space, action_space, hidden_dim=hidden_dim, action_range=action_range)

# =========================
# Hyperparameters
# =========================
max_episodes   = 2000
# Pendulum’s TimeLimit is 200; keep max_steps >= 200 to let env terminate naturally
max_steps      = 200
frame_idx      = 0
batch_size     = 64
explore_steps  = 0
update_itr     = 40
AUTO_ENTROPY   = True
DETERMINISTIC  = False

rewards_log = []

if __name__ == '__main__':
    if mode == 'train':
        # Reset and flatten VecEnv obs to list for your policy
        state = env.reset()
        state = state.ravel().tolist()

        for eps in range(max_episodes):
            last_action = env.action_space.sample()  # shape (n_envs, act_dim) == (1,1)
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []

            # LSTM hidden state
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            hidden_out = (
                torch.zeros([1, 1, hidden_dim], dtype=torch.float, device=device),
                torch.zeros([1, 1, hidden_dim], dtype=torch.float, device=device),
            )

            ini_hidden_in = None
            ini_hidden_out = None

            for step in range(max_steps):
                hidden_in = hidden_out

                # Query action from policy (expects flattened state + last_action)
                action, hidden_out = sac_trainer.policy_net.get_action(
                    state,
                    last_action,
                    hidden_in,
                    deterministic=DETERMINISTIC
                )
                # Ensure action respects env bounds
                action = np.clip(np.array(action, dtype=np.float32), action_space.low, action_space.high)

                # VecEnv step expects shape (n_envs, act_dim), we have 1 env -> wrap
                next_state, reward, done, info = env.step(np.array([action], dtype=np.float32))
                next_state_list = next_state.ravel().tolist()

                if step == 0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out

                # Log transitions (single-env indexing)
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(float(reward[0]))
                episode_next_state.append(next_state_list)
                episode_done.append(bool(done[0]))

                state = next_state_list
                last_action = action
                frame_idx += 1

                if done[0]:
                    break

            # Episode end
            # Reset for next episode
            state = env.reset()
            state = state.ravel().tolist()

            # Push episode sequence to replay buffer
            replay_buffer.push(
                ini_hidden_in, ini_hidden_out,
                episode_state, episode_action, episode_last_action,
                episode_reward, episode_next_state, episode_done
            )

            # Updates
            if len(replay_buffer) > 2:
                for _ in range(update_itr):
                    _ = sac_trainer.update(
                        min(len(replay_buffer), batch_size),
                        reward_scale=1.0,              # Pendulum scale is fine at 1.0
                        auto_entropy=AUTO_ENTROPY,
                        target_entropy=-1.0 * action_dim, offline_training=False)

            # Print episodic return for easy monitoring
            ep_return = np.sum(episode_reward)
            rewards_log.append(ep_return)
            print(f'Episode: {eps:4d} | Return: {ep_return:8.2f}')
            
            if eps % 30 == 0:
                model_path = './model/saver_data_{}'.format(eps)
                sac_trainer.save_model(model_path)


                experience_filepath = './model/replaybuffer_{}.pkl'.format(eps)
                with open(experience_filepath, 'wb') as f:
                    pickle.dump({
                        'capacity': replay_buffer.maxlen,
                        'position': replay_buffer.write_idx,
                        'buffer': replay_buffer.store
                    }, f)
                
    if mode == 'test':
        model_path = r'./model/'
        sac_trainer.load_model(model_path)
        state = env.reset()
        state = state.ravel().tolist()
        
        action_list = []

        for eps in range(10):
            last_action = env.action_space.sample()
            episode_reward = 0.0
            hidden_out = (
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(),
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda()
            )

            for step in range(200):
                hidden_in = hidden_out
                action, hidden_out = sac_trainer.policy_net.get_action(
                    state, last_action, hidden_in, deterministic=True
                )
                action_list.append(action)

                next_state, reward, done, _ = env.step(np.array([action]))
                next_state = next_state.ravel().tolist()

                last_action = action
                episode_reward += reward[0]
                state = next_state

            state = env.reset()
            state = state.ravel().tolist()
            print('Evaluate Episode: ', eps, '| Eval Episode Reward: ', episode_reward)
            



