#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from common.buffers import *
from gym import spaces
from sac_mamba_ensemble import SAC_Ensemble
import numpy as np

experience_filepath = './model/replaybuffer_60.pkl'
with open(experience_filepath, 'rb') as f:
    data = pickle.load(f)

experience_filepath = './model/replaybuffer_90.pkl'
with open(experience_filepath, 'rb') as f:
    data2 = pickle.load(f)

replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer_compressed(replay_buffer_size)
# Create (or already have) a new replay_buffer object of the same class
#replay_buffer.capacity = data['capacity']

for index in range(len(data['buffer'])):
    transition = data['buffer'][index]
    replay_buffer.push(transition[0],transition[1],transition[2],transition[3],transition[4],transition[5],transition[6],transition[7])  # or .store(), .push(), etc.

for index in range(len(data2['buffer'])):
    transition = data2['buffer'][index]
    replay_buffer.push(transition[0],transition[1],transition[2],transition[3],transition[4],transition[5],transition[6],transition[7])  # or .store(), .push(), etc.
    
    
    
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
batch_size   = 6
update_itr   = 5000
hidden_dim   = 512



sac_trainer = SAC_Ensemble(replay_buffer, state_space, action_space,
                           hidden_dim=hidden_dim, action_range=action_range)

for upd in range(update_itr):
    _ = sac_trainer.update(
        min(len(replay_buffer), batch_size),
        target_entropy=-1. * action_dim, offline_training=True
    )
    
    if upd % 1000 == 0:
        model_path = './model/saver_data_{}'.format(upd)
        sac_trainer.save_model(model_path)

    
    
                    
    
    
    