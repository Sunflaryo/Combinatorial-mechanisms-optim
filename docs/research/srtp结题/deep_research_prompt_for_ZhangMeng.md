# 给张萌的 Deep Research Prompt

```text
We are working on a literature review and research positioning for a project tentatively titled "HRL-AMA: Hierarchical Reinforcement Learning for Dynamic Affine Maximizer Auctions." The intended problem is repeated combinatorial auction revenue optimization. Our current idea is not to let bidders learn bidding strategies; instead, the auctioneer fixes the mechanism class to Affine Maximizer Auctions (AMAs), which are DSIC/IR by construction when bidder weights are positive, and uses reinforcement learning to tune AMA parameters across rounds based on historical market states such as budgets, bid statistics, time, and recent revenue.

Please search for papers on reinforcement learning, online learning, differentiable economics, or automated mechanism design for combinatorial auction/mechanism design with exact DSIC/IR guarantees. Prioritize mechanisms based on affine maximizer auctions, virtual valuations, dynamic VCG, sequential combinatorial auctions, or structured neural architectures that guarantee truthfulness by design. Compare these papers with an approach that fixes AMA structure and uses RL to tune AMA parameters over repeated rounds.

For each paper, summarize: auction setting, mechanism class, whether it is static or dynamic, IC/IR guarantee, learning method, whether it handles combinatorial items, main limitation, and how it overlaps with or differs from our HRL-AMA idea. Pay special attention to whether any prior work already studies dynamic or repeated revenue optimization within an exact-DSIC AMA parameter space.
```

## 中文说明版

我们现在想做的是 HRL-AMA 的文献综述和研究定位。核心问题不是让 bidder 学会报价，而是由 auctioneer 固定 AMA 机制类来继承 DSIC/IR，再用 RL 在重复组合拍卖中根据历史市场状态调节 AMA 参数。请重点检索：

- reinforcement learning / online learning for mechanism design
- combinatorial auction design with exact DSIC/IR
- affine maximizer auctions / virtual valuations / dynamic VCG
- sequential combinatorial auctions
- structured neural architectures that guarantee truthfulness

最终希望得到一张对比表，说明每篇文章的 setting、mechanism class、IC/IR guarantee、learning method、是否 dynamic、是否处理 combinatorial items、主要限制，以及它和 HRL-AMA 的重合/差异。最关键的问题是：有没有已有工作已经研究了 **dynamic or repeated revenue optimization within an exact-DSIC AMA parameter space**。
