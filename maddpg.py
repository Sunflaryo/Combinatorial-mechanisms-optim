# maddpg.py
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


########################################
# 基础网络：Actor / Critic
########################################

class Actor(nn.Module):
    """
    单个智能体的策略网络：
    输入：obs (batch, obs_dim)
    输出：动作 a \in [0,1]^act_dim （之后可按需求缩放到出价区间）
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid()  # 输出限制在 [0,1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """
    集中式 Critic：
    输入：拼接后的全局 obs 与全局 act
        obs_full: (batch, sum_obs_dim)
        act_full: (batch, sum_act_dim)
    输出：Q(s, a) 标量
    """
    def __init__(self, full_obs_dim: int, full_act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(full_obs_dim + full_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_full: torch.Tensor, act_full: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_full, act_full], dim=-1)
        return self.net(x)


########################################
# 回放缓冲区：结构化存储多智能体数据
########################################

class ReplayBuffer:
    """
    多智能体经验回放：
    每个 transition 存：
        obs_list:      List[np.ndarray]，长度 n_agents，每个 shape (obs_dim_i,)
        act_list:      List[np.ndarray]，长度 n_agents，每个 shape (act_dim_i,)
        rew_list:      np.ndarray，shape (n_agents,) 或 list
        next_obs_list: 同 obs_list
        done:          bool / 0-1
    sample() 返回：
        obs_batch:      List[torch.Tensor]，长度 n_agents，每个 shape (batch, obs_dim_i)
        act_batch:      List[torch.Tensor]，长度 n_agents，每个 shape (batch, act_dim_i)
        rew_batch:      torch.Tensor，shape (batch, n_agents)
        next_obs_batch: 同 obs_batch
        done_batch:     torch.Tensor，shape (batch, 1)
    """
    def __init__(self,
                 n_agents: int,
                 obs_dims: List[int],
                 act_dims: List[int],
                 max_size: int = 100000):
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.max_size = max_size
        self.storage = deque(maxlen=max_size)

    def __len__(self):
        return len(self.storage)

    def add(self,
            obs_list: List[np.ndarray],
            act_list: List[np.ndarray],
            rew_list,
            next_obs_list: List[np.ndarray],
            done: bool):
        """
        obs_list / next_obs_list: [agent_i_obs]
        act_list:                 [agent_i_action]
        rew_list:                 list 或 np.ndarray，长度 n_agents
        done:                     bool
        """
        rew_arr = np.asarray(rew_list, dtype=np.float32)
        self.storage.append((obs_list, act_list, rew_arr, next_obs_list, float(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.storage, batch_size)
        # 解包
        obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)  # 每个长度 batch_size

        # 逐 agent 堆叠
        obs_batch = []
        next_obs_batch = []
        act_batch = []

        for i in range(self.n_agents):
            obs_i = np.stack([obs_b[k][i] for k in range(batch_size)], axis=0)
            next_obs_i = np.stack([next_obs_b[k][i] for k in range(batch_size)], axis=0)
            act_i = np.stack([act_b[k][i] for k in range(batch_size)], axis=0)

            obs_batch.append(torch.as_tensor(obs_i, dtype=torch.float32, device=device))
            next_obs_batch.append(torch.as_tensor(next_obs_i, dtype=torch.float32, device=device))
            act_batch.append(torch.as_tensor(act_i, dtype=torch.float32, device=device))

        rew_batch = torch.as_tensor(np.stack(rew_b, axis=0), dtype=torch.float32, device=device)
        done_batch = torch.as_tensor(np.array(done_b).reshape(-1, 1),
                                     dtype=torch.float32,
                                     device=device)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch


########################################
# MADDPG 主类
########################################

def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau)
        tp.data.add_(sp.data, alpha=tau)


class MADDPG:
    """
    多智能体 DDPG:
    - 集中 critic：每个 agent 拥有独立 critic，但都看全局 obs + 全局 actions
    - 分散 actor：每个 agent 各自一个 actor，只用自身 obs
    使用方式：
        maddpg = MADDPG(n_agents, obs_dims, act_dims, ...)
        actions = maddpg.act(obs_list, explore=True)
        buffer.add(obs_list, actions, rewards, next_obs_list, done)
        maddpg.update()
    """
    def __init__(self,
                 n_agents: int,
                 obs_dims: List[int],
                 act_dims: List[int],
                 actor_hidden_dim: int = 128,
                 critic_hidden_dim: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 lr_actor: float = 1e-3,
                 lr_critic: float = 1e-3,
                 max_buffer_size: int = 100000,
                 device: str = None):
        assert len(obs_dims) == n_agents
        assert len(act_dims) == n_agents

        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.gamma = gamma
        self.tau = tau

        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.target_actors: List[Actor] = []
        self.target_critics: List[Critic] = []
        self.actor_optimizers: List[optim.Optimizer] = []
        self.critic_optimizers: List[optim.Optimizer] = []

        full_obs_dim = sum(obs_dims)
        full_act_dim = sum(act_dims)

        for i in range(n_agents):
            actor = Actor(obs_dims[i], act_dims[i], hidden_dim=actor_hidden_dim).to(self.device)
            critic = Critic(full_obs_dim, full_act_dim, hidden_dim=critic_hidden_dim).to(self.device)

            target_actor = Actor(obs_dims[i], act_dims[i], hidden_dim=actor_hidden_dim).to(self.device)
            target_critic = Critic(full_obs_dim, full_act_dim, hidden_dim=critic_hidden_dim).to(self.device)

            hard_update(target_actor, actor)
            hard_update(target_critic, critic)

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)

            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic))

        self.buffer = ReplayBuffer(n_agents, obs_dims, act_dims, max_size=max_buffer_size)

        # 探索噪声参数
        self.noise_std = 0.2

    def act(self,
            obs_list: List[np.ndarray],
            explore: bool = False,
            action_low: float = 0.0,
            action_high: float = 1.0) -> List[np.ndarray]:
        """
        obs_list: [obs_i]，每个 obs_i shape (obs_dims[i],)
        返回：动作 list，每个 shape (act_dims[i],)，已缩放到 [action_low, action_high]
        """
        actions = []
        for i in range(self.n_agents):
            obs_i = torch.as_tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a_i = self.actors[i](obs_i).squeeze(0).cpu().numpy()  # in [0,1]
            if explore:
                noise = np.random.normal(0.0, self.noise_std, size=a_i.shape)
                a_i = np.clip(a_i + noise, 0.0, 1.0)
            # 缩放到真实动作区间（例如出价区间）
            a_scaled = action_low + (action_high - action_low) * a_i
            actions.append(a_scaled.astype(np.float32))
        return actions

    def update(self, batch_size: int = 64, actor_loss_coef: float = 1.0):
        """
        在外部训练循环中，每步/每若干步调用一次。
        buffer 中样本数不足时自动跳过。
        """
        if len(self.buffer) < batch_size:
            return

        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = \
            self.buffer.sample(batch_size, device=self.device)

        # 构造全局 obs/action：拼接各 agent 分量
        obs_full = torch.cat(obs_batch, dim=-1)          # (B, sum_obs_dim)
        act_full = torch.cat(act_batch, dim=-1)          # (B, sum_act_dim)
        next_obs_full = torch.cat(next_obs_batch, dim=-1)

        # 对每个智能体分别更新
        for i in range(self.n_agents):
            # ========= Critic 更新 =========
            with torch.no_grad():
                # 目标动作：来自 target actors
                next_act_list = []
                for j in range(self.n_agents):
                    a_j = self.target_actors[j](next_obs_batch[j])
                    next_act_list.append(a_j)
                next_act_full = torch.cat(next_act_list, dim=-1)

                target_q = self.target_critics[i](next_obs_full, next_act_full)
                # reward_i shape: (B, 1)
                r_i = rew_batch[:, i:i+1]
                # y = r_i + gamma * (1 - done) * target_q
                y = r_i + self.gamma * (1.0 - done_batch) * target_q

            current_q = self.critics[i](obs_full, act_full)
            critic_loss = nn.MSELoss()(current_q, y)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=1.0)
            self.critic_optimizers[i].step()

            # ========= Actor 更新 =========
            # 构造当前 actor_i 的动作，其他 agent 用 buffer 中的旧动作
            curr_act_list = []
            for j in range(self.n_agents):
                if j == i:
                    a_j = self.actors[j](obs_batch[j])
                else:
                    a_j = act_batch[j].detach()
                curr_act_list.append(a_j)
            curr_act_full = torch.cat(curr_act_list, dim=-1)

            actor_q = self.critics[i](obs_full, curr_act_full)
            actor_loss = -actor_q.mean() * actor_loss_coef

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=1.0)
            self.actor_optimizers[i].step()

            # ========= 软更新 target 网络 =========
            soft_update(self.target_actors[i], self.actors[i], self.tau)
            soft_update(self.target_critics[i], self.critics[i], self.tau)

    def store_transition(self,
                         obs_list: List[np.ndarray],
                         act_list: List[np.ndarray],
                         rew_list,
                         next_obs_list: List[np.ndarray],
                         done: bool):
        """
        方便调用的封装：直接把一次 step 的数据写入 buffer。
        """
        self.buffer.add(obs_list, act_list, rew_list, next_obs_list, done)

    def to(self, device: str):
        """
        手动迁移到指定 device（一般不需要手动调用）
        """
        self.device = torch.device(device)
        for i in range(self.n_agents):
            self.actors[i].to(self.device)
            self.critics[i].to(self.device)
            self.target_actors[i].to(self.device)
            self.target_critics[i].to(self.device)
