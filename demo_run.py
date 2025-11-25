# demo_run.py
from env import CombinatorialAuctionEnv
from maddpg import MADDPG
import numpy as np

env = CombinatorialAuctionEnv(n_agents=5, n_items=3) # 先用简单参数
n_agents = env.n_agents
obs_dims = [env.obs_dim] * n_agents
act_dims = [env.n_items] * n_agents

maddpg = MADDPG(n_agents, obs_dims, act_dims)

for episode in range(100):
    obs_list = env.reset() # 现在reset返回obs_list
    done = False
    episode_rewards = []

    while not done:
        # 1. MADDPG产生动作
        actions = maddpg.act(obs_list) # 现在输入和输出都是列表，维度对了

        # 2. 环境执行动作
        next_obs_list, rewards, done, info = env.step(actions)
        episode_rewards.append(sum(rewards))

        # 3. 存储经验 (state是上一个obs_list, next_state是next_obs_list)
        maddpg.store_transition(obs_list, actions, rewards, next_obs_list, done)

        obs_list = next_obs_list

    # 4. 更新MADDPG
    maddpg.update(batch_size=32)

    print(f"Episode {episode}, Total Reward: {sum(episode_rewards):.2f}")

print("Demo run finished!")