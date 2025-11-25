#log：20251124
#当前版本在 train 里只预留了接入位置，cfg.use_mcts 目前默认 False。
#
# train.py
import os
import csv
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch

from env import CombinatorialAuctionEnv
from maddpg import MADDPG
# from mcts import SimpleMCTS  # 如果以后要真用 MCTS 再打开


# =========================
# 配置
# =========================

@dataclass
class Config:
    seed: int = 42

    # 训练设置
    num_episodes: int = 2000
    max_episode_steps: int = 20
    batch_size: int = 64
    update_interval: int = 1          # 每多少个 env step 调一次 update()
    updates_per_step: int = 1         # 每次调用 update() 连续做几步梯度更新

    # 环境 / 动作设置
    max_bid: float = 10.0             # 出价上限（需和 env 保持一致）
    use_mcts: bool = False            # 预留 MCTS（默认关闭）

    # 日志与模型
    log_dir: str = "./logs"
    eval_interval: int = 50
    eval_episodes: int = 10
    save_interval: int = 200

    # MADDPG 超参
    gamma: float = 0.99
    tau: float = 0.01
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    max_buffer_size: int = 100000


# =========================
# 工具函数
# =========================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_obs_list(obs) -> list:
    """
    把 env 返回的 obs 转成 [obs_i] 的 list，提供给 MADDPG.act/存入 buffer.

    env.reset() / env.step() 返回的是一个长度为 n_agents 的 list，
    其中每个元素是 shape (obs_dim,) 的 numpy array。
    这种情况下我们直接把它转换成 float32 即可。
    """
    return [np.asarray(o, dtype=np.float32) for o in obs]

def log_csv_init(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "train_return_mean",
                "train_social_welfare",
                "eval_return_mean",
                "eval_social_welfare"
            ])


def log_csv_append(csv_path: str,
                   episode: int,
                   train_ret: float,
                   train_welfare: float,
                   eval_ret: float,
                   eval_welfare: float):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            train_ret,
            train_welfare,
            eval_ret,
            eval_welfare
        ])


# =========================
# 评估函数
# =========================

def evaluate(env: CombinatorialAuctionEnv,
             maddpg: MADDPG,
             cfg: Config) -> tuple:
    """
    在当前策略下评估若干 episode（无探索噪声，不更新参数）.
    返回：
        (mean_return_per_agent, mean_social_welfare)
    """
    n_agents = env.n_agents
    returns = []
    welfare_list = []

    for _ in range(cfg.eval_episodes):
        obs = env.reset()
        obs_list = extract_obs_list(obs)
        done = False
        steps = 0

        # 每个 agent 的 episode return
        ret_agents = np.zeros(n_agents, dtype=np.float64)

        while not done and steps < cfg.max_episode_steps:
            actions = maddpg.act(
                obs_list,
                explore=False,
                action_low=0.0,
                action_high=cfg.max_bid
            )

            obs_next, rewards, done, info = env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            ret_agents += rewards

            obs_list = extract_obs_list(obs_next)
            steps += 1

        # 记录指标
        returns.append(ret_agents.mean())            # 每个 episode 的“平均单 agent 回报”
        welfare_list.append(ret_agents.sum())        # 每个 episode 的“社会福利（sum reward）”

    return float(np.mean(returns)), float(np.mean(welfare_list))


# =========================
# 主训练循环
# =========================

def train(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.log_dir)

    # 实例化环境
    env = CombinatorialAuctionEnv()
    n_agents = env.n_agents
    # 这里假设每个 agent 的 obs_dim = n_items（只看自己的 valuations）
    obs_dims = [env.n_items for _ in range(n_agents)]
    act_dims = [env.n_items for _ in range(n_agents)]  # 每个 item 出一个 bid

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    maddpg = MADDPG(
        n_agents=n_agents,
        obs_dims=obs_dims,
        act_dims=act_dims,
        actor_hidden_dim=cfg.actor_hidden_dim,
        critic_hidden_dim=cfg.critic_hidden_dim,
        gamma=cfg.gamma,
        tau=cfg.tau,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        max_buffer_size=cfg.max_buffer_size,
        device=device
    )

    # MCTS 预留（默认不用）
    # if cfg.use_mcts:
    #     def policy_func_for_mcts(state):
    #         # TODO: 根据 state 调用 maddpg 的某个 actor 生成候选动作
    #         raise NotImplementedError
    #     mcts = SimpleMCTS(policy_func_for_mcts)

    # CSV 日志初始化
    csv_path = os.path.join(cfg.log_dir, "train_log.csv")
    log_csv_init(csv_path)

    best_eval_welfare = -1e9

    start_time = time.time()
    for ep in range(1, cfg.num_episodes + 1):
        obs = env.reset()
        obs_list = extract_obs_list(obs)
        done = False
        steps = 0

        # 每个 agent 的 episode return
        ret_agents = np.zeros(n_agents, dtype=np.float64)

        while not done and steps < cfg.max_episode_steps:
            # 1. 根据当前策略选择动作（带探索噪声）
            actions = maddpg.act(
                obs_list,
                explore=True,
                action_low=0.0,
                action_high=cfg.max_bid
            )

            # 如果以后要用 MCTS refine 某些 agent 的动作，可以在这里加：
            # if cfg.use_mcts and (steps % some_interval == 0):
            #     actions[0] = mcts_refine(...)

            # 2. 交互环境
            next_obs, rewards, done, info = env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32)

            next_obs_list = extract_obs_list(next_obs)

            # 3. 写入 buffer
            maddpg.store_transition(
                obs_list,
                actions,
                rewards,
                next_obs_list,
                done
            )

            # 4. 训练更新
            if steps % cfg.update_interval == 0:
                for _ in range(cfg.updates_per_step):
                    maddpg.update(batch_size=cfg.batch_size)

            # 5. 统计
            ret_agents += rewards
            obs_list = next_obs_list
            steps += 1

        train_return_mean = float(ret_agents.mean())
        train_social_welfare = float(ret_agents.sum())

        # ========= 周期性评估 =========
        if ep % cfg.eval_interval == 0:
            eval_ret, eval_welfare = evaluate(env, maddpg, cfg)
        else:
            eval_ret, eval_welfare = np.nan, np.nan

        # 写日志
        log_csv_append(
            csv_path,
            ep,
            train_return_mean,
            train_social_welfare,
            eval_ret,
            eval_welfare
        )

        # 打印进度
        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - start_time
            print(
                f"[Ep {ep:4d}] "
                f"TrainRet(mean)={train_return_mean:.3f}, "
                f"TrainWelfare={train_social_welfare:.3f}, "
                f"EvalRet={eval_ret:.3f}, "
                f"EvalWelfare={eval_welfare:.3f}, "
                f"Steps={steps}, "
                f"Elapsed={elapsed/60:.1f} min"
            )

        # 保存最优 checkpoint（按 eval welfare）
        if ep % cfg.eval_interval == 0 and not np.isnan(eval_welfare):
            if eval_welfare > best_eval_welfare:
                best_eval_welfare = eval_welfare
                ckpt_dir = os.path.join(cfg.log_dir, "checkpoints_best")
                ensure_dir(ckpt_dir)
                for i in range(n_agents):
                    torch.save(
                        maddpg.actors[i].state_dict(),
                        os.path.join(ckpt_dir, f"actor_agent{i}.pth")
                    )
                    torch.save(
                        maddpg.critics[i].state_dict(),
                        os.path.join(ckpt_dir, f"critic_agent{i}.pth")
                    )
                print(f"  >> New best eval welfare = {best_eval_welfare:.3f}, models saved.")

        # 周期性保存当前模型（非 best）
        if ep % cfg.save_interval == 0:
            ckpt_dir = os.path.join(cfg.log_dir, f"checkpoints_ep{ep}")
            ensure_dir(ckpt_dir)
            for i in range(n_agents):
                torch.save(
                    maddpg.actors[i].state_dict(),
                    os.path.join(ckpt_dir, f"actor_agent{i}.pth")
                )
                torch.save(
                    maddpg.critics[i].state_dict(),
                    os.path.join(ckpt_dir, f"critic_agent{i}.pth")
                )
            print(f"  >> Checkpoints saved at episode {ep}.")

    print("Training finished.")


# =========================
# CLI 入口
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--use_mcts", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        num_episodes=args.episodes,
        log_dir=args.log_dir,
        use_mcts=args.use_mcts
    )
    train(cfg)
