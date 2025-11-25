# 组合拍卖与机制设计：强化学习方法

本项目实现了基于多智能体强化学习（MARL）的组合拍卖机制设计，包含两个主要研究方向：

1. **组合拍卖中的智能体学习**：使用 MADDPG 算法训练智能体在组合拍卖中的竞价策略
2. **Myerson 最优机制学习**：使用强化学习学习单物品拍卖中的最优保留价

## 项目结构

```
comb/
├── env.py                      # 组合拍卖环境实现
├── maddpg.py                   # MADDPG 多智能体强化学习算法
├── train01.py                  # 组合拍卖训练脚本
├── mech_env.py                 # Myerson 单物品拍卖环境
├── mech_myerson_rl.py          # Myerson 保留价策略训练
├── myerson_check_single_item.py # Myerson 理论验证脚本
├── 智能体测试.ipynb             # MADDPG 交互式测试与可视化
├── Myerson核对.ipynb           # Myerson 机制验证与对比
├── logs/                       # MADDPG 训练日志和检查点
├── logs_mech/                  # Myerson 训练日志
├── models/                     # 保存的模型文件
└── README.md                   # 项目文档
```

## 环境说明

### 1. 组合拍卖环境 (CombinatorialAuctionEnv)

**环境特点：**
- N 个智能体（竞拍者），M 个物品
- 每个智能体对物品有私有估值
- 多轮拍卖，每轮智能体同时出价
- 次价支付规则（第二高价）

**观察空间：** 每个智能体观察到：
- 自己的私有估值向量 (M 维)
- 上一轮所有物品的最高出价 (M 维)
- 当前回合数 (1 维)

**动作空间：**
- 每个智能体对每个物品的出价 (M 维)
- 范围：[0, max_bid]

**分配规则：**
- 每个物品分配给对其出价最高的智能体

**支付规则：**
- 次价支付：获胜者支付对应物品的第二高出价

**奖励：**
- 智能体效用 = 获得物品的价值总和 - 支付总和

### 2. Myerson 单物品拍卖环境 (SingleItemAuctionMechEnv)

**环境特点：**
- 单物品，多个竞拍者
- 竞拍者估值服从 i.i.d. U[0,1] 分布
- 机制设计者学习最优保留价

**目标：**
- 最大化拍卖收入
- 理论最优保留价：r* = 0.5
- 理论最优收入：约 0.672

## 依赖安装

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install numpy torch matplotlib gym
```

**主要依赖：**
- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Gym >= 0.17.0
- Matplotlib >= 3.3.0

## 使用方法

### 1. 训练组合拍卖智能体（MADDPG）

```bash
python train01.py --num_episodes 2000 --n_agents 3 --n_items 5
```

**主要参数：**
- `--num_episodes`: 训练回合数（默认：2000）
- `--n_agents`: 智能体数量（默认：3）
- `--n_items`: 物品数量（默认：5）
- `--batch_size`: 批次大小（默认：64）
- `--lr_actor`: Actor 学习率（默认：1e-3）
- `--lr_critic`: Critic 学习率（默认：1e-3）
- `--gamma`: 折扣因子（默认：0.99）
- `--tau`: 软更新系数（默认：0.01）
- `--save_interval`: 模型保存间隔（默认：200）

**训练日志：**
- 日志保存在 `logs/train_log.csv`
- 模型检查点保存在 `logs/checkpoints_epXXX/`

### 2. 训练 Myerson 保留价策略

```bash
python mech_myerson_rl.py
```

训练完成后，模型保存为：
- `policy.pt`（供 notebook 使用）
- `logs_mech/reserve_policy.pt`（完整保存）

### 3. 交互式测试与可视化

#### MADDPG 测试：

打开 `智能体测试.ipynb`，依次运行各个 cell：
- Cell 1: 环境测试
- Cell 2: MADDPG 初始化
- Cell 3: 单回合训练
- Cell 4-6: 可视化（奖励曲线、效用分布等）
- Cell 8: 多回合训练
- Cell 9: 动作分布分析
- Cell 10: 模型保存与加载

#### Myerson 验证：

打开 `Myerson核对.ipynb`：
- Cell 2: 理论 Myerson 机制评估
- Cell 4: RL 学习的机制评估
- Cell 6: 价格分布对比可视化

**注意：** 运行 Cell 4 前需要先训练 Myerson 模型（见上一节）

## 实验结果

### MADDPG 组合拍卖训练

- **训练回合数**：2000
- **智能体数量**：3-5
- **收敛情况**：约 500-1000 回合后收敛
- **性能**：智能体学会接近真实估值的竞价策略

### Myerson 保留价学习

- **训练回合数**：5000
- **学习到的保留价**：约 0.496-0.510（理论最优：0.5）
- **平均收入**：约 0.667-0.670（理论最优：0.672）
- **收敛精度**：与理论值误差 < 1%

## 文件说明

### 核心代码

- **env.py**: 组合拍卖环境，实现 Gym 接口
- **maddpg.py**: MADDPG 算法实现，包含 Actor/Critic 网络和经验回放
- **train01.py**: 完整的训练流程，包含日志记录和模型保存
- **mech_env.py**: 单物品拍卖环境，用于 Myerson 机制学习
- **mech_myerson_rl.py**: REINFORCE 算法训练保留价策略

### 辅助文件

- **demo_run.py**: 简单的演示脚本
- **debug_env.py**: 环境调试脚本
- **myerson_check_single_item.py**: Myerson 理论公式验证
- **mcts.py**: MCTS 算法实现（预留接口）

### Notebook

- **智能体测试.ipynb**: MADDPG 交互式测试，包含丰富的可视化
- **Myerson核对.ipynb**: Myerson 机制理论与 RL 结果对比

### 其他

- **notes.md**: 开发笔记
- **logs/**: 训练日志和检查点
- **models/**: 保存的模型文件

## 技术细节

### MADDPG 算法

- **算法类型**：多智能体深度确定性策略梯度（Multi-Agent DDPG）
- **网络结构**：
  - Actor: 128 → 64 → act_dim (Tanh)
  - Critic: (obs_dim × N + act_dim × N) → 128 → 64 → 1
- **经验回放**：容量 100,000
- **探索策略**：OU 噪声 + ε-greedy

### Myerson 机制学习

- **算法类型**：REINFORCE with baseline
- **策略网络**：单参数 logit → sigmoid → μ ∈ (0,1)
- **探索**：Normal(μ, σ) 采样，σ = 0.1
- **基准线**：指数移动平均，momentum = 0.9

## 注意事项

1. **Notebook Kernel 重启**：修改 Python 代码后，需要重启 notebook kernel 以加载最新代码
2. **模型路径**：确保训练脚本和 notebook 中的模型路径一致
3. **依赖版本**：Gym 已停止维护，建议迁移到 Gymnasium（向后兼容）
4. **随机种子**：已设置随机种子保证可复现性

## 未来改进

- [ ] 支持物品组合的互补估值
- [ ] 实现 VCG 机制作为对比基准
- [ ] 集成 MCTS 到训练流程
- [ ] 支持更复杂的拍卖规则（如保留价、打包拍卖）
- [ ] 添加更多评估指标（效率、公平性等）

## 参考文献

1. Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
2. Myerson, R. (1981). Optimal Auction Design.
3. Cramton, P., et al. (2006). Combinatorial Auctions.

## 许可

本项目仅供学术研究使用。
