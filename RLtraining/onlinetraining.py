#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import gym
import numpy as np
import torch
from gym import spaces

import time
import logging
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym.envs
import pickle

from DrlPlatform import UdpServer
from sac_mamba_ensemble import SAC_Ensemble
from common.buffers import ReplayBuffer_compressed


# -------------------------
# Replay buffer
# -------------------------
replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer_compressed(replay_buffer_size)

logger = logging.getLogger(__name__)

# -------------------------
# Env registration & server
# -------------------------
OBSERVATION_TYPE = "All"
if OBSERVATION_TYPE == "All":
    gym.envs.register(
        id='AhmedBody_AllObservations-v0',
        entry_point='Case.LABVIEW_Environment:AhmedBody_AllObservations'
    )

    env: gym.Env = gym.make('AhmedBody_AllObservations-v0')  # type: ignore

    env_server = UdpServer(
        server_host="192.168.1.192",
        server_port=16388,
        client_host="192.168.1.183",  # REPLACE WITH IP FROM PXI
        client_port=16385,
        package_timeout=5.0,
        max_package_size=16384
    )
    env_server.start_server()

# attach server handle into underlying env
env.env.env_server = env_server  # type: ignore


mode = 'train'
env = DummyVecEnv([lambda: env])


# -------------------------
# Spaces for trainer
# -------------------------
action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
state_space  = spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32)

action_dim   = action_space.shape[0]
action_range = 2.5

# -------------------------
# Hyperparameters
# -------------------------
max_episodes = 2000
max_steps    = int(4096)
batch_size   = 6
update_itr   = 40
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim   = 512

# -------------------------
# Trainer
# -------------------------
sac_trainer = SAC_Ensemble(replay_buffer, state_space, action_space,
                           hidden_dim=hidden_dim, action_range=action_range)

if __name__ == '__main__':
    if mode == 'train':
        state = env.reset()
        state = state.ravel().tolist()
        for eps in range(max_episodes):

            last_action = env.action_space.sample()
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []

            hidden_out = (
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(),
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda()
            )

            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = sac_trainer.policy_net.get_action(
                    state, last_action, hidden_in, deterministic=DETERMINISTIC
                )
                next_state, reward, done, info = env.step(np.array([action]))
                next_state = next_state.ravel().tolist()

                if step == 0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out

                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward[0])
                episode_next_state.append(next_state)
                episode_done.append(done[0])

                state = next_state
                last_action = action

                if done:
                    break

            state = env.reset()
            state = state.ravel().tolist()

            if len(replay_buffer) > 2:
                for _ in range(update_itr):
                    _ = sac_trainer.update(
                        min(len(replay_buffer), batch_size),
                        reward_scale=10.,
                        auto_entropy=AUTO_ENTROPY,
                        target_entropy=-1. * action_dim
                    )
                    

            replay_buffer.push(
                ini_hidden_in, ini_hidden_out,
                episode_state, episode_action, episode_last_action,
                episode_reward, episode_next_state, episode_done
            )

            print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))

            if eps % 10 == 0:
                model_path = './model/saver_data_{}'.format(eps)
                sac_trainer.save_model(model_path)


                experience_filepath = './model/replaybuffer_.pkl'
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

            for step in range(4096 * 16):
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

            time.sleep(30)
            print('Start repeat')
            episode_reward = 0.0

            for step in range(4096 * 16):
                hidden_in = hidden_out
                action = action_list[step]

                next_state, reward, done, _ = env.step(np.array([action]))
                next_state = next_state.ravel().tolist()

                last_action = action
                episode_reward += reward[0]
                state = next_state

            state = env.reset()
            state = state.ravel().tolist()
            print('Repeated Evaluate Episode: ', eps, '| Eval Episode Reward: ', episode_reward)
