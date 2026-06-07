# Combinatorial Mechanisms Optim

This repository is being organized as the code-first workspace for a project on **HRL-AMA: Hierarchical Reinforcement Learning for Dynamic Affine Maximizer Auctions**. The current codebase contains earlier baselines and sanity checks; the main HRL-AMA implementation will be merged later from teammate code.

The research direction has been narrowed from "bidders learn bidding strategies" to a mechanism-design framing:

```text
state -> RL policy -> AMA parameters -> deterministic AMA solver -> revenue/budget feedback -> next state
```

The intended learning agent is the **auctioneer**, which adjusts parameters inside an Affine Maximizer Auction (AMA). Bidders are not the primary learning agents in the new pipeline; truthful reporting should come from the mechanism structure.

## Repository Layout

```text
comb/
├── comb_auction/          # Core Python package and legacy baseline implementations
├── scripts/               # Script entry points for current runnable checks
├── notebooks/             # Interactive notebooks from earlier experiments
├── docs/research/         # Literature review, experiment design, prompts, and reports
├── README.md
├── requirements.txt
└── .gitignore
```

Runtime outputs such as `logs/`, `logs_mech/`, `models/`, `outputs/`, `*.pt`, and `*.pth` are intentionally ignored. Existing local files are kept on disk, but future cloud commits should avoid tracking generated checkpoints and caches.

## Current Status

Implemented and preserved:

- Historical MADDPG bidder-learning baseline for a fixed second-price combinatorial auction.
- Single-item Myerson reserve-price learning sanity check.
- Myerson theoretical comparison script.
- Research positioning materials under `docs/research/`, especially the HRL-AMA literature review and experiment design.

Not implemented in this stage:

- Full deterministic AMA solver.
- Static AMA optimizer.
- Flat DRL + AMA controller.
- HRL-AMA controller and evaluation pipeline.

Those pieces should be added after the teammate code is available.

## Installation

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Core dependencies are NumPy, PyTorch, Gym, Matplotlib, Notebook, and ReportLab. ReportLab is used by the research-document generation script in `docs/research/`.

## Current Runnable Commands

The old root-level commands are preserved for compatibility:

```bash
python train01.py --episodes 2000 --log_dir ./logs
python mech_myerson_rl.py
python myerson_check_single_item.py
python debug_env.py
```

Equivalent script-style entry points:

```bash
python scripts/train_maddpg_baseline.py --episodes 2000 --log_dir ./logs
python scripts/train_myerson.py
python scripts/evaluate_myerson.py
python scripts/debug_env.py
```

The README previously documented `python train01.py --num_episodes 2000 --n_agents 3 --n_items 5`, but the current CLI does not support those flags. The correct current options are `--episodes`, `--log_dir`, and `--use_mcts`.

## HRL-AMA Target Pipeline

The planned HRL-AMA pipeline follows the experiment design in `docs/research/srtp结题/`:

- Setting: repeated combinatorial auction.
- Initial scale: `n = 3` bidders, `m in {2, 3}` items, `T = 30` rounds.
- State features: time progress, budget ratio, bid mean/std, and recent revenue.
- Action: AMA bidder weights and item/allocation boosts.
- Clearing: deterministic AMA solver computes allocation and payment.
- Feedback: revenue and budget updates become the next state.

Planned future commands, documented here only as interface targets:

```bash
python scripts/train_hrl_ama.py
python scripts/evaluate_hrl_ama.py
```

Do not add placeholder implementations for these until the AMA/HRL code is ready.

## Baselines And Evaluation

The planned comparison groups are:

- `B0` random allocation or random feasible AMA parameters as a sanity check.
- `B1` VCG as the exact DSIC/IR static baseline.
- `B2` Static AMA to test offline/static AMA parameter tuning.
- `B3` Flat DRL + AMA as a single-level learning baseline.
- `B4` HRL-AMA with high-level goals and low-level AMA parameters.

Primary metrics:

- Cumulative revenue.
- Per-round IC regret.
- IR violation.
- Adaptation lag after non-stationary shifts.
- Budget exhaustion pattern.
- Runtime per round and training time.
- Parameter interpretability through weight/boost trajectories.

The important claim boundary is: AMA can provide **per-round** DSIC/IR when parameters are committed before current-round bids and bidder weights remain positive. This is not the same as full dynamic incentive compatibility across rounds.

## Notes For Future Cloud Updates

- Keep code in `comb_auction/` and runnable commands in `scripts/`.
- Keep notebooks in `notebooks/`.
- Keep research documents in `docs/research/`.
- Do not commit generated logs, checkpoints, model weights, Python cache files, or Jupyter checkpoints.
- Treat the existing MADDPG bidder-learning code as historical baseline material, not as the main HRL-AMA research story.
