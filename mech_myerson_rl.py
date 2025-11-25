# mech_myerson_rl.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mech_env import SingleItemAuctionMechEnv


class ReservePolicy(nn.Module):
    """
    机制策略：只学一个参数 mu，通过 sigmoid 限制在 (0,1)，
    并作为 Normal( mu, sigma ) 的均值做探索。
    """

    def __init__(self, init_logit: float = 0.0, sigma: float = 0.1):
        super().__init__()
        # 直接把 logit 作为一个可学习参数；mu = sigmoid(logit)
        self.logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.sigma = sigma

    def forward(self, obs: torch.Tensor):
        """
        obs: (B, 1)，这里其实用不到 obs，只是保持接口一致
        返回:
            r: (B, 1)，采样的保留价 ∈ (0,1)
            log_prob: (B, 1)，对应的 log π(r | theta)
        """
        batch_size = obs.shape[0]
        mu = torch.sigmoid(self.logit)  # ∈ (0,1)
        mu_expand = mu.expand(batch_size, 1)

        dist = torch.distributions.Normal(mu_expand, self.sigma)
        # 先在 R 上采样，再 clip 到 [0,1]
        raw_sample = dist.rsample()  # rsample 支持 reparameterization
        r = torch.clamp(raw_sample, 0.0, 1.0)

        # 注意：严格来说 clip 会破坏 log_prob 的精确性，这里做近似用
        log_prob = dist.log_prob(raw_sample)  # shape (B, 1)
        return r, log_prob

    def get_deterministic_reserve(self) -> float:
        """
        评估时用的确定性保留价（不用探索噪声）。
        """
        with torch.no_grad():
            mu = torch.sigmoid(self.logit).item()
        return float(mu)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mechanism(
    n_agents: int = 5,
    batch_size: int = 512,
    num_episodes: int = 5000,
    lr: float = 1e-2,
    gamma: float = 1.0,
    seed: int = 42,
    log_dir: str = "./logs_mech",
    sigma: float = 0.1,
):
    """
    使用 REINFORCE 训练保留价 r 的机制。
    每个 episode:
        - reset -> sample 一批估值
        - 从 policy 采样 r
        - env.step(r) 得到收入 reward
        - 更新 policy 使得期望 revenue 最大
    """
    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    env = SingleItemAuctionMechEnv(n_agents=n_agents, batch_size=batch_size, seed=seed)
    policy = ReservePolicy(sigma=sigma)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # 简单 moving baseline，减少方差
    baseline = 0.0
    baseline_momentum = 0.9

    rewards_hist = []
    reserves_hist = []

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        obs_t = torch.tensor(obs[None, :], dtype=torch.float32)  # (1,1)

        r_t, log_prob = policy(obs_t)  # (1,1)
        action = r_t.detach().cpu().numpy()[0]  # np.array shape (1,)

        _, reward, _, info = env.step(action)

        reward_t = torch.tensor([reward], dtype=torch.float32)

        # 更新 baseline
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
        advantage = reward_t - baseline

        loss = -log_prob.mean() * advantage  # REINFORCE: -E[adv * log pi]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_hist.append(float(reward))
        reserves_hist.append(float(action[0]))

        if ep % 100 == 0:
            avg_R = np.mean(rewards_hist[-100:])
            avg_r = np.mean(reserves_hist[-100:])
            print(f"[Ep {ep:5d}] avg_reward={avg_R:.4f}, reserve~{avg_r:.3f}")

    # 训练完成后保存 policy
    ckpt_path = os.path.join(log_dir, "reserve_policy.pt")
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Training finished. Policy saved to {ckpt_path}")
    
    # 同时保存一份到根目录，方便 notebook 加载
    torch.save(policy.state_dict(), "policy.pt")
    print("Policy also saved to policy.pt")

    return policy, rewards_hist, reserves_hist


if __name__ == "__main__":
    train_mechanism()
