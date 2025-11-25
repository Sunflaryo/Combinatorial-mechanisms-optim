# myerson_check_single_item.py
import numpy as np
import torch

from mech_myerson_rl import ReservePolicy
from mech_env import SingleItemAuctionMechEnv


def sample_uniform(n_agents: int, n_samples: int, seed: int = 123):
    rng = np.random.RandomState(seed)
    return rng.rand(n_samples, n_agents)  # U[0,1]


def myerson_single_item_uniform(valuations: np.ndarray, reserve: float = 0.5):
    """
    单物品 Myerson 机制 (U[0,1]):
      - 分配：max(v) >= reserve 才给最高估值者
      - 支付：max(reserve, second_highest_v)
    返回:
      alloc: (B, n) one-hot winner（或全 0 流拍）
      pay:   (B, n) 支付（只有 winner 有）
    """
    B, n = valuations.shape
    alloc = np.zeros((B, n), dtype=float)
    pay = np.zeros((B, n), dtype=float)

    order = np.argsort(-valuations, axis=1)
    top1_idx = order[:, 0]
    top1_val = valuations[np.arange(B), top1_idx]
    top2_val = np.where(n >= 2, valuations[np.arange(B), order[:, 1]], 0.0)

    sale_mask = top1_val >= reserve
    alloc[np.arange(B), top1_idx] = sale_mask.astype(float)

    prices = np.maximum(reserve, top2_val) * sale_mask.astype(float)
    pay[np.arange(B), top1_idx] = prices
    return alloc, pay


def revenue(pay: np.ndarray) -> float:
    return float(pay.sum(axis=1).mean())


def efficiency(valuations: np.ndarray, alloc: np.ndarray) -> float:
    B, n = valuations.shape
    top1_true = np.argmax(valuations, axis=1)
    winner = np.argmax(alloc, axis=1)
    sale_mask = alloc.sum(axis=1) > 0.5
    return float(((sale_mask) & (winner == top1_true)).mean())


def main():
    n_agents = 5
    n_samples = 100000
    vals = sample_uniform(n_agents, n_samples)

    # ---- 理论 Myerson ----
    alloc_m, pay_m = myerson_single_item_uniform(vals, reserve=0.5)
    rev_m = revenue(pay_m)
    eff_m = efficiency(vals, alloc_m)

    print("=== Theoretical Myerson (reserve=0.5) ===")
    print(f"Revenue:    {rev_m:.4f}")
    print(f"Efficiency: {eff_m:.4f}")

    # ---- 加载 RL 学出的机制 ----
    # 与 mech_myerson_rl 中保存的路径保持一致
    ckpt_path = "./logs_mech/reserve_policy.pt"
    policy = ReservePolicy()
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.eval()

    r_hat = policy.get_deterministic_reserve()
    print(f"\nRL learned reserve ≈ {r_hat:.4f}")

    alloc_rl, pay_rl = myerson_single_item_uniform(vals, reserve=r_hat)
    rev_rl = revenue(pay_rl)
    eff_rl = efficiency(vals, alloc_rl)

    print("=== RL Mechanism (plug-in reserve) ===")
    print(f"Revenue:    {rev_rl:.4f}")
    print(f"Efficiency: {eff_rl:.4f}")


if __name__ == "__main__":
    main()
