自适应奖励（reward shaping）与冷启动实现细节

动态 reward 设计要点

把 reward 分为：即时收益（agent utility）、长期指标（社会福利）、惩罚项（偏离、违规、流拍）。

state-aware scaling：当竞争强度高（更多 active bidders）时，把“避免流拍”权重提高。

使用 MCTS 估计的 long-term value V_mcts 来修正即时 reward：r' = r + λ * (V_mcts - baseline)。

冷启动（预训练）

从历史数据用行为克隆（supervised）先训练 actor：训练目标最小化 L = ||π(o)-a_demo||^2。

将行为克隆策略作为 MADDPG 的初始 actor 权重（减少探索期浪费）。

把 MCTS 结果注入训练

把 MCTS 推荐的（state, action, value) 放入 prioritized replay；训练时采样优先级更高的数据用于 critic 更新。

5 评价指标、对照组与消融实验

指标：社会福利(sum utilities)、拍卖收入(organizer revenue)、公平性（Gini/多人标准差）、收敛速度（episodes to threshold）、鲁棒性（面对新 agents 的表现）。

对照组：VCG、Myerson-like static mechanism、standard MADDPG（无 MCTS）、无高层调整（固定机制）。

消融：去掉 MCTS / 去掉高层 / 去掉自适应 reward，分别比较上面指标。

2025/9/5 23:08

一旦基础数据流打通，就可以按优先级逐一实现核心功能：

实现 MADDPG.update() 方法：这是最大的任务。你需要参考MADDPG论文中的损失函数，用PyTorch实现集中式Critic和策略梯度的计算。

为MCTS提供策略函数：将 maddpg.actors[agent_id] 包装成一个可以用于MCTS policy_func 的函数。

实现高层控制器的更新逻辑：设计高层策略的奖励信号（如回合总收益），实现PPO的更新步骤。

连接高层与中层：定义高层输出的参数（如θ_payment）如何影响环境的奖励计算或智能体的动作。