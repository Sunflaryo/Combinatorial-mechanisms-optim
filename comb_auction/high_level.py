# high_level.py
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical, Normal

class HighPolicy(nn.Module):
    def __init__(self, obs_dim, param_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,128), nn.Tanh(), nn.Linear(128,param_dim))
    def forward(self, x):
        return self.net(x)

class PPOController:
    def __init__(self, obs_dim, param_dim, lr=3e-4):
        self.policy = HighPolicy(obs_dim, param_dim)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
    def get_params(self, obs):
        # obs: macro state -> return mechanism params (e.g., weights)
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        params = torch.tanh(self.policy(x)).detach().squeeze(0).numpy()
        return params
    def update(self, trajs):
        # trajs: collected high-level transitions; implement PPO update here
        pass
