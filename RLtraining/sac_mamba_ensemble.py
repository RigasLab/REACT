import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from common.buffers import *
from common.value_networks import *
from common.policy_networksv2 import *



device_idx = 0
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
print(device)


class SAC_Ensemble():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range,
                 num_critics: int = 2):
        
        self.replay_buffer = replay_buffer
        self.num_critics = num_critics

        # ---- ENSEMBLE CRITICS ----
        self.soft_q_nets = nn.ModuleList(
            [MambaQNetwork(state_space, action_space, hidden_dim).to(device)
             for _ in range(num_critics)]
        )
        self.target_soft_q_nets = nn.ModuleList(
            [MambaQNetwork(state_space, action_space, hidden_dim).to(device)
             for _ in range(num_critics)]
        )


        for tgt, src in zip(self.target_soft_q_nets, self.soft_q_nets):
            for tparam, sparam in zip(tgt.parameters(), src.parameters()):
                tparam.data.copy_(sparam.data)


        self.policy_net = SAC_PolicyNetwork_LSTMbased(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        print(f'Soft Q Networks (ensemble={num_critics}): ', self.soft_q_nets)
        print('Policy Network: ', self.policy_net)

        # losses & optimizers
        self.soft_q_criterion = nn.MSELoss()
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizers = [
            optim.Adam(q.parameters(), lr=soft_q_lr) for q in self.soft_q_nets
        ]
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.update_count = 0


    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-1,
               gamma=0.99, soft_tau=0.05, print_every = 100, offline_training=False):

        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state      = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        action     = torch.FloatTensor(np.array(action)).to(device)
        last_action= torch.FloatTensor(np.array(last_action)).to(device)
        reward     = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)
        done       = torch.FloatTensor(np.float32(np.array(done))).unsqueeze(-1).to(device)

        # forward Q ensemble for current (s,a)
        q_preds = [qnet(state, action, last_action, hidden_in)[0] for qnet in self.soft_q_nets]

        # policy eval at s and s'
        new_action, log_prob, _, _, _, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)

        # normalize reward
        if not offline_training:
            reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)



        if auto_entropy and not offline_training:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(0.1, device=device)
            alpha_loss = 0.0
        self.update_count += 1

        # ----- TARGETS: min over target ensemble -----
        target_q_preds = [tq(next_state, new_next_action, action, hidden_out)[0]
                          for tq in self.target_soft_q_nets]
        # stack to shape (Ncrit, B, T, 1) and min over Ncrit
        target_q_min = torch.stack(target_q_preds, dim=0).min(dim=0).values - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min

        # ----- Q losses: sum (or mean) over ensemble -----
        total_q_loss = 0.0
        for opt, q_pred in zip(self.soft_q_optimizers, q_preds):
            q_loss = self.soft_q_criterion(q_pred, target_q_value.detach())
            opt.zero_grad()
            q_loss.backward()
            opt.step()
            total_q_loss += q_loss

        # ----- POLICY loss: min over current ensemble -----
        policy_q_preds = [qnet(state, new_action, last_action, hidden_in)[0]
                          for qnet in self.soft_q_nets]
        q_min_current = torch.stack(policy_q_preds, dim=0).min(dim=0).values
        
        
        if offline_training:
            bc_weight = 2.5  # tune via validation
            bc_loss = ((new_action - action).pow(2)).mean()
            policy_loss = (self.alpha * log_prob - q_min_current).mean() + bc_weight * bc_loss
        else:
            policy_loss = (self.alpha * log_prob - q_min_current).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ----- SOFT UPDATES -----
        for tgt, src in zip(self.target_soft_q_nets, self.soft_q_nets):
            for tparam, sparam in zip(tgt.parameters(), src.parameters()):
                tparam.data.copy_(tparam.data * (1.0 - soft_tau) + sparam.data * soft_tau)
        
        if offline_training and (self.update_count % print_every == 0):
            q_stack = torch.stack(policy_q_preds, dim=0) 
            q_disagree = float(q_stack.std(dim=0).mean().detach().cpu().item()) if q_stack.shape[0] >= 2 else 0.0
            mean_q_loss = float(total_q_loss / max(1, len(self.soft_q_nets)))
            q_pi_mean = float(q_min_current.mean().detach().cpu().item())
            
            print(
                f"[upd {float(self.update_count)}] "
                f"Qloss: {mean_q_loss:.4f} | "
                f"Qπ: {float(q_pi_mean):.4f} | "
                f"Disagree: {q_disagree:.4f} | "
                f"BC: {float(bc_loss):.4f}"
            )

        # keep your original return semantics (mean new Q under policy)
        return q_min_current.mean()

    def save_model(self, path):
        # Save each critic with an index; keep backward-compat for first two.
        for i, q in enumerate(self.soft_q_nets, start=1):
            torch.save(q.state_dict(), f"{path}_q{i}")
        torch.save(self.policy_net.state_dict(), f"{path}_policy")

    def load_model(self, path_base):
        # Try to load q1..qN; if only q1/q2 exist, that’s fine for num_critics=2
        for i, q in enumerate(self.soft_q_nets, start=1):
            q.load_state_dict(torch.load(f"{path_base}saver_data_90_q{i}", map_location=device))
            q.eval()
        self.policy_net.load_state_dict(torch.load(f"{path_base}saver_data_90_policy", map_location=device))
        self.policy_net.eval()


