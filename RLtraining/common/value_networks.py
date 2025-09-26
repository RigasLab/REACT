import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from mamba_ssm import Mamba


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
                # Optionally, set forget gate bias to 1
                hidden_size = param.size(0) // 4
                param.data[:hidden_size].fill_(1.0)  # Set forget gate bias to 1
                param.data[hidden_size:].zero_()     # Initialize other biases to 0

class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  

        self.activation = activation

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation ):
        super().__init__( state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]


class MambaQNetwork(QNetworkBase):

    
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.mamba1 = Mamba(d_model=hidden_dim, # Model dimension d_model
                            d_state=64,  # SSM state expansion factor
                            d_conv=4,    # Local convolution width
                            expand=2,    # Block expansion factor
                            ).to("cuda")
        
        
        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim,hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear4.apply(linear_weights_init)
        
        self.layernorm1 = LayerNorm(hidden_dim)
        self.layernorm2 = LayerNorm(hidden_dim)
        
        
    def forward(self, state, action, last_action, hidden_in):

        fc_branch = torch.cat([state, action], -1) 
        fc_branch = F.relu(self.linear1(fc_branch))

        mamba_head = torch.cat([state, last_action], -1)
        mamba_head = F.silu(self.linear2(mamba_head))
        mamba_head = self.mamba1(mamba_head)
        mamba_head = self.layernorm1(mamba_head)
        
        fc_branch = self.layernorm2(fc_branch)

        merged_branch=torch.cat([fc_branch, mamba_head], -1) 

        x = F.relu(self.linear3(merged_branch))
        x = self.linear4(x)

        lstm_hidden = hidden_in
        
        return x, lstm_hidden    
    


