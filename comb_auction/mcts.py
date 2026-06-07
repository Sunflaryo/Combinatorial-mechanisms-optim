# mcts.py
import math, random
from collections import defaultdict, namedtuple

TreeNode = namedtuple("TreeNode", ["visits", "value", "children", "prior"])

class SimpleMCTS:
    def __init__(self, policy_func, rollout_steps=3, c_puct=1.0):
        """
        policy_func(state) -> action distribution or single action
        policy_func should be deterministic or stochastic model (actor) used for rollouts
        """
        self.policy_func = policy_func
        self.c_puct = c_puct
        self.rollout_steps = rollout_steps

    def search(self, root_state, simulate_env_fn, n_sim=50):
        """
        simulate_env_fn(state, action) -> next_state, reward, done, info
        root_state: environment state representation
        returns: best_action
        """
        root = {"N":0, "W":0, "Q":0, "children":{}, "prior":1.0}
        for _ in range(n_sim):
            self._simulate(root, root_state, simulate_env_fn)
        # pick child with highest Q/N
        best_action = max(root["children"].items(), key=lambda kv: kv[1]["Q"]/max(1,kv[1]["N"]))[0]
        return best_action

    def _simulate(self, node, state, simulate_fn):
        # select/expand/rollout/backprop simplified
        # selection
        if len(node["children"])==0:
            # expand using policy
            acts = self.policy_func(state)  # returns list of candidate actions
            for a,prior in acts:
                node["children"][a] = {"N":0,"W":0,"Q":0,"prior":prior}
        # pick child with highest UCB
        best_a, best_child = None, None
        best_score = -float("inf")
        for a,child in node["children"].items():
            U = self.c_puct * child["prior"] * math.sqrt(node["N"]+1)/(1+child["N"])
            Q = child["Q"]/max(1,child["N"])
            score = Q + U
            if score>best_score:
                best_score=score
                best_a=a
                best_child=child
        # simulate one step
        next_state, reward, done, _ = simulate_fn(state, best_a)
        # rollout
        total_reward = reward
        if not done:
            # micro-rollout using policy
            s = next_state
            for _ in range(self.rollout_steps):
                acts = self.policy_func(s)
                if len(acts)==0: break
                a = acts[0][0]
                s, r, done, _ = simulate_fn(s, a)
                total_reward += r
                if done: break
        # backpropagate: update child and node (simple)
        best_child["N"] += 1
        best_child["W"] += total_reward
        best_child["Q"] = best_child["W"]
        node["N"] += 1
        node["W"] = node.get("W",0)+total_reward
##上面 policy_func 可以由已训练的 actor 网络（mid-level）提供；simulate_env_fn 可以用 env 的复制/fast-forward 来进行。MCTS 的输出（高价值 action paths）可以放入中层的经验池或作为 reward 修正因子。