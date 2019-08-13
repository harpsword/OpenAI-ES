

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(AbstractPolicy):

    def __init__(self, state_dim, action_dim, action_lim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_lim

    def get_action(self, obs):
        return self.forward(torch.from_numpy(obs).float()).numpy().flatten()

def build_model(args):
    return Policy(args.state_dim, args.action_dim, args.action_lim)
