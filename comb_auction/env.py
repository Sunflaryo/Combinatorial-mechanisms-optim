# env.py
import numpy as np
from typing import List, Dict, Tuple, Optional
import gym
from gym import spaces

class CombinatorialAuctionEnv(gym.Env):
    """
    组合拍卖环境
    
    环境描述：
    - N个智能体（竞拍者），M个物品
    - 每个智能体对物品组合有私有估值（通过价值函数表示）
    - 每个智能体提交一个出价向量（对每个物品的出价）
    - 分配规则：将每个物品分配给对其出价最高的智能体
    - 支付规则：次价支付（每个获胜者支付对应物品的第二高出价）
    - 智能体效用：分配到的物品组合的价值减去总支付
    
    观察空间：
    每个智能体观察到：
    - 自己的私有估值向量 (M维)
    - 上一轮所有物品的最高出价 (M维)
    - 当前回合数 (1维)
    
    动作空间：
    每个智能体对每个物品的出价 (M维)，范围在 [0, max_bid]
    
    奖励：
    每个智能体的奖励是其效用（价值减去支付）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_agents: int = 3, n_items: int = 5, max_bid: float = 10.0, max_steps: int = 20):
        super().__init__()
        self.n_agents = n_agents
        self.n_items = n_items
        self.max_bid = max_bid
        self.max_steps = max_steps

        # 定义每个智能体的观察空间
        # 观察包括: 自己的估值(M) + 上一轮最高出价(M) + 回合数(1)
        self.obs_dim = n_items * 2 + 1
        self.observation_space = spaces.Box(
            low=0.0, 
            high=max_bid, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        
        # 定义每个智能体的动作空间 (对每个物品的出价)
        self.action_space = spaces.Box(
            low=0.0, 
            high=max_bid, 
            shape=(n_items,), 
            dtype=np.float32
        )

        # 环境状态
        self.timestep = 0
        self.true_valuations = None  # 智能体对物品的真实估值 (n_agents, n_items)
        self.last_max_bids = None    # 上一轮各物品的最高出价 (n_items,)
        
        # 用于记录历史信息
        self.history = {
            'allocations': [],
            'payments': [],
            'utilities': [],
            'bids': []
        }

    def reset(self) -> List[np.ndarray]:
        """重置环境状态并返回初始观察"""
        self.timestep = 0
        
        # 生成新的私有估值 - 每个智能体对每个物品的估值
        self.true_valuations = np.random.rand(self.n_agents, self.n_items) * self.max_bid
        
        # 初始化上一轮最高出价为0
        self.last_max_bids = np.zeros(self.n_items)
        
        # 清空历史
        self.history = {
            'allocations': [],
            'payments': [],
            'utilities': [],
            'bids': []
        }
        
        # 返回每个智能体的初始观察
        return self._get_obs()

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, bool, Dict]:
        """
        执行一个时间步
        
        参数:
            actions: 每个智能体的动作 (出价向量) 列表
            
        返回:
            obs_list: 每个智能体的新观察
            rewards: 每个智能体的奖励
            done: 是否结束
            info: 附加信息
        """
        self.timestep += 1
        actions = np.asarray(actions)  # 形状: (n_agents, n_items)
        
        # 记录本轮出价
        self.history['bids'].append(actions.copy())
        
        # 分配物品给最高出价者
        winners = np.argmax(actions, axis=0)  # 每个物品的获胜者索引
        
        # 计算支付 (次价支付规则)
        payments = np.zeros(self.n_agents)
        allocations = np.zeros((self.n_agents, self.n_items), dtype=bool)
        
        for item in range(self.n_items):
            item_bids = actions[:, item]
            winner = winners[item]
            
            # 分配物品给获胜者
            allocations[winner, item] = True
            
            # 计算支付 (第二高出价)
            if self.n_agents > 1:
                # 获取第二高的出价
                second_price = np.partition(item_bids, -2)[-2]
                payments[winner] += second_price
            # 如果只有一个竞拍者，支付0或保留价 (这里设为0)
        
        # 计算每个智能体的效用 (价值 - 支付)
        # 注意: 这里假设价值是可加的 (没有组合效应)
        utilities = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            # 智能体i获得物品的总价值
            value = np.sum(self.true_valuations[i] * allocations[i])
            utilities[i] = value - payments[i]
        
        # 更新上一轮最高出价
        self.last_max_bids = np.max(actions, axis=0)
        
        # 记录历史
        self.history['allocations'].append(allocations)
        self.history['payments'].append(payments)
        self.history['utilities'].append(utilities)
        
        # 检查是否结束
        done = self.timestep >= self.max_steps
        
        # 获取新观察
        next_obs = self._get_obs()
        
        # 信息字典
        info = {
            "allocations": allocations,
            "payments": payments,
            "utilities": utilities,
            "winners": winners
        }
        
        return next_obs, utilities, done, info

    def _get_obs(self) -> List[np.ndarray]:
        """获取每个智能体的观察"""
        obs_list = []
        for i in range(self.n_agents):
            # 观察包括: 自己的估值 + 上一轮最高出价 + 当前回合数(归一化)
            agent_obs = np.concatenate([
                self.true_valuations[i].copy(),  # 自己的估值
                self.last_max_bids.copy(),       # 上一轮最高出价
                [self.timestep / self.max_steps] # 归一化的回合数
            ])
            obs_list.append(agent_obs)
        return obs_list

    def render(self, mode: str = 'human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== 回合 {self.timestep} ===")
            print("真实估值:")
            for i in range(self.n_agents):
                print(f"  智能体 {i}: {self.true_valuations[i]}")
            
            if self.timestep > 0:
                last_bids = self.history['bids'][-1]
                print("出价:")
                for i in range(self.n_agents):
                    print(f"  智能体 {i}: {last_bids[i]}")
                
                print("分配结果:")
                allocations = self.history['allocations'][-1]
                for i in range(self.n_agents):
                    items_won = np.where(allocations[i])[0]
                    print(f"  智能体 {i} 获得物品: {items_won}")
                
                print("支付:")
                payments = self.history['payments'][-1]
                for i in range(self.n_agents):
                    print(f"  智能体 {i}: {payments[i]:.2f}")
                
                print("效用:")
                utilities = self.history['utilities'][-1]
                for i in range(self.n_agents):
                    print(f"  智能体 {i}: {utilities[i]:.2f}")

    def get_global_state(self) -> Dict:
        """获取全局状态 (用于高层控制器或MCTS)"""
        return {
            "timestep": self.timestep,
            "true_valuations": self.true_valuations.copy(),
            "last_max_bids": self.last_max_bids.copy(),
            "history": self.history.copy()
        }