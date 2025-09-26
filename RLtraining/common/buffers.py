import random
import torch
from typing import List, Tuple, Any

class ReplayBuffer_compressed:


    def __init__(self, capacity: int):
        self.maxlen = capacity
        self.store: List[Any] = []
        self.write_idx = 0  # ring buffer cursor

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        """Append one episode tuple into the ring buffer."""
        if len(self.store) < self.maxlen:
            self.store.append(None)
        self.store[self.write_idx] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.write_idx = int((self.write_idx + 1) % self.maxlen)  # ring advance

    def sample(self, batch_size: int):
        """Uniformly sample episodes; return (hidden_in_batch, hidden_out_batch, s, a, last_a, r, s_next, done)."""
        s_buf, a_buf, prev_a_buf, r_buf, s_next_buf, d_buf = [], [], [], [], [], []
        h0_list, c0_list, h1_list, c1_list = [], [], [], []

        batch = random.sample(self.store, batch_size)
        for ep in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = ep

            s_buf.append(state)
            a_buf.append(action)
            prev_a_buf.append(last_action)
            r_buf.append(reward)
            s_next_buf.append(next_state)
            d_buf.append(done)

            # hidden states (each is shaped (1, batch=1, hidden_size))
            h0_list.append(h_in)
            c0_list.append(c_in)
            h1_list.append(h_out)
            c1_list.append(c_out)

        # Concatenate along the batch dimension (-2) to form (1, B, hidden_size)
        h0 = torch.cat(h0_list, dim=-2).detach()
        h1 = torch.cat(h1_list, dim=-2).detach()
        c0 = torch.cat(c0_list, dim=-2).detach()
        c1 = torch.cat(c1_list, dim=-2).detach()

        hidden_in_batch = (h0, c0)
        hidden_out_batch = (h1, c1)

        return hidden_in_batch, hidden_out_batch, s_buf, a_buf, prev_a_buf, r_buf, s_next_buf, d_buf

    def __len__(self):
        
        return len(self.store)

    def get_length(self):
        return len(self.store)


    

