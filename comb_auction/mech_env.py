# mech_env.py
import numpy as np
import gym
from gym import spaces


class SingleItemAuctionMechEnv(gym.Env):
    """
    单物品 + i.i.d. U[0,1] 估值。
    RL agent = 机制设计者：
        动作：保留价 r ∈ [0,1]
    内部机制：单物品二价拍卖 + 保留价 r
        - 只要最高估值 >= r 才成交
        - 成交价 = max(r, second_highest_v)
    奖励：批量样本下的平均收入（可以理解为一次“周期”的收益）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_agents: int = 5, batch_size: int = 512, seed: int = 42):
        super().__init__()
        self.n_agents = n_agents
        self.batch_size = batch_size
        self._rng = np.random.RandomState(seed)

        # 观测空间：简单起见，就给一个 dummy 标量（甚至可以全 0）
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # 动作空间：保留价 r ∈ [0,1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._sample_valuations()

    def _sample_valuations(self):
        # i.i.d. U[0,1] 估值
        self._vals = self._rng.rand(self.batch_size, self.n_agents)

    def reset(self):
        # 每个 episode 重新 sample 一批估值
        self._sample_valuations()
        obs = np.array([0.0], dtype=np.float32)  # dummy obs
        return obs

    def step(self, action):
        """
        action: np.ndarray, shape (1,) in [0,1]
        """
        r = float(np.clip(action[0], 0.0, 1.0))  # reserve

        vals = self._vals  # (B, n)
        B, n = vals.shape

        # 排序得到 top1 / top2
        order = np.argsort(-vals, axis=1)
        top1_idx = order[:, 0]
        top1_val = vals[np.arange(B), top1_idx]
        top2_val = np.where(
            n >= 2, vals[np.arange(B), order[:, 1]], 0.0
        )

        # 是否成交：最高估值 >= r
        sale_mask = top1_val >= r  # (B,)
        # 成交价 = max(r, second_highest_v)
        prices = np.maximum(r, top2_val) * sale_mask.astype(float)

        revenue = prices.mean()  # 平均收入作为 reward
        reward = float(revenue)

        # 我们把每一步都当作一个完整 episode，done=True
        done = True
        info = {
            "reserve": r,
            "revenue": revenue,
            "sale_rate": float(sale_mask.mean()),
        }

        # 为下一步准备新样本，这里依然生成，但下一次 reset() 也会重新生成
        self._sample_valuations()
        obs_next = np.array([0.0], dtype=np.float32)

        return obs_next, reward, done, info

    def render(self, mode="human"):
        pass