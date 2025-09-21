# debug_env.py
import numpy as np
from env import CombinatorialAuctionEnv

def test_env_basic():
    """测试环境的基本功能"""
    print("=== 测试环境基本功能 ===")
    
    # 创建环境
    env = CombinatorialAuctionEnv(n_agents=2, n_items=3, max_steps=5)
    print(f"环境创建成功: {env.n_agents}个智能体, {env.n_items}个物品")
    
    # 测试重置
    try:
        obs = env.reset()
        print(f"重置成功，观察形状: {[o.shape for o in obs]}")
        print(f"第一个智能体的观察: {obs[0]}")
    except Exception as e:
        print(f"重置失败: {e}")
        return False
    
    # 测试随机动作
    try:
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        print(f"生成随机动作: {[a.shape for a in actions]}")
        
        # 执行一步
        next_obs, rewards, done, info = env.step(actions)
        print(f"执行步骤成功")
        print(f"奖励: {rewards}")
        print(f"完成状态: {done}")
        print(f"信息键: {info.keys()}")
        
        # 渲染
        env.render()
    except Exception as e:
        print(f"执行步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_multiple_steps():
    """测试多步执行"""
    print("\n=== 测试多步执行 ===")
    
    env = CombinatorialAuctionEnv(n_agents=2, n_items=3, max_steps=3)
    obs = env.reset()
    
    for step in range(5):  # 尝试执行5步，即使max_steps=3
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)
        
        print(f"步骤 {step+1}: 奖励={rewards}, 完成={done}")
        env.render()
        
        if done:
            print("环境已结束")
            break
            
        obs = next_obs

if __name__ == "__main__":
    # 运行基本测试
    success = test_env_basic()
    
    if success:
        # 运行多步测试
        test_multiple_steps()
        print("\n=== 所有测试完成 ===")
    else:
        print("\n=== 测试失败 ===")