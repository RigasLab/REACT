#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

def lstm_weights_init(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                gate_size = param.size(0) // 4
                # forget gate bias = 1, others = 0
                param.data[:gate_size].fill_(1.0)
                param.data[gate_size:].zero_()

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:
            pass  # high-dim state (not used here)

        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]

        self.action_range = action_range

    def forward(self):
        pass

    def evaluate(self):
        pass

    def get_action(self):
        pass

    def sample_action(self):
        a = torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range * a.numpy()


class SAC_PolicyNetwork_LSTMbased(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)
        print('policy network v2')

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size

        # layers (names unchanged to keep checkpoint compatibility)
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim + self._action_dim, hidden_size)
        self.lstm1   = nn.LSTM(hidden_size, hidden_size, num_layers=1)
        self.linear3 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear    = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)

        # init
        self.linear1.apply(linear_weights_init)
        self.linear2.apply(linear_weights_init)
        self.linear3.apply(linear_weights_init)
        self.linear4.apply(linear_weights_init)
        self.mean_linear.apply(linear_weights_init)
        self.log_std_linear.apply(linear_weights_init)
        self.lstm1.apply(lstm_weights_init)

    def forward(self, state, last_action, hidden_in):
        
        obs_seq  = state.permute(1, 0, 2)        # (seq, batch, feat)
        prev_act = last_action.permute(1, 0, 2)  # (seq, batch, act)
        mlp_branch = F.relu(self.linear1(obs_seq))
        lstm_in = torch.cat([obs_seq, prev_act], dim=-1)
        lstm_in = F.relu(self.linear2(lstm_in))
        lstm_out, lstm_hidden = self.lstm1(lstm_in, hidden_in) 
        fused = torch.cat([mlp_branch, lstm_out], dim=-1)
        h = F.relu(self.linear3(fused))
        h = F.relu(self.linear4(h))
        h = h.permute(1, 0, 2)  
        mean    = self.mean_linear(h)
        log_std = self.log_std_linear(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std, lstm_hidden

    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()  # keep unclipped for gradient flow
        device = mean.device
        std_normal = Normal(torch.zeros_like(mean, device=device),
                            torch.ones_like(mean, device=device))
        noise = std_normal.rsample()  # rsample for reparameterization
        pre_tanh = mean + std * noise
        tanh_act = torch.tanh(pre_tanh)
        action = self.action_range * tanh_act
        base_log_prob = Normal(mean, std).log_prob(pre_tanh)
        squash_correction = torch.log(1. - tanh_act.pow(2) + epsilon)
        log_prob = base_log_prob - squash_correction - np.log(self.action_range)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, noise, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        
        obs_t  = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
        prev_a = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        mean, log_std, hidden_out = self.forward(obs_t, prev_a, hidden_in)
        std = log_std.exp()
        device = mean.device
        std_normal = Normal(torch.zeros_like(mean, device=device),
                            torch.ones_like(mean, device=device))
        noise = std_normal.rsample()
        sampled = self.action_range * torch.tanh(mean + std * noise)
        action_np = (self.action_range * torch.tanh(mean)).detach().cpu().numpy() if deterministic \
                    else sampled.detach().cpu().numpy()

        return action_np[0][0], hidden_out
