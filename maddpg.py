# maddpg.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid()  # scale later
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, full_obs_dim, full_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(full_obs_dim + full_act_dim,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, maxsize=100000):
        self.buf = deque(maxlen=maxsize)
    def add(self, sample):
        self.buf.append(sample)
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return batch

class MADDPG:
    def __init__(self, n_agents, obs_dims, act_dims, lr=1e-3, gamma=0.99):
        self.n = n_agents
        self.actors = [Actor(obs_dims[i], act_dims[i]) for i in range(self.n)]
        self.critics = [Critic(sum(obs_dims), sum(act_dims)) for _ in range(self.n)]
        self.target_actors = [Actor(obs_dims[i], act_dims[i]) for i in range(self.n)]
        self.target_critics = [Critic(sum(obs_dims), sum(act_dims)) for _ in range(self.n)]
        self.optimizers = [optim.Adam(list(self.actors[i].parameters())+list(self.critics[i].parameters()), lr=lr) for i in range(self.n)]
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        # TODO: copy parameters to targets

    def act(self, obs_list):
        actions = []
        for i,obs in enumerate(obs_list):
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = self.actors[i](o).squeeze(0).numpy()
            actions.append(a)
        return actions

    def update(self, batch_size=64):
        if len(self.buffer.buf) < batch_size: 
            return
        batch = self.buffer.sample(batch_size)
        # batch items: (obs_list, action_list, reward_list, next_obs_list, done)
        # TODO: implement critic/actor update using centralized critic
        pass
