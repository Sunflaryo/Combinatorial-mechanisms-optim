# HRL-AMA 文献综述与研究定位

## 0. 这篇综述真正要回答的问题

这篇 literature review 不应该写成“我们找到了很多论文，所以我们知道这个领域很复杂”。更重要的是，它要回答张萌老师在会议里反复追问的那个问题：

> 现有机制设计和学习机制文献已经做到哪里了？在这些已有结果之上，我们现在这件事到底还新增了什么？

围绕这个问题，HRL-AMA 比较稳妥的定位是：我们并不是重新证明 truthful bidding，也不是让每个 bidder 通过学习来找到自己的报价策略；更准确地说，我们想让 auctioneer 在一个已经有激励保证的机制类里学习调参。具体地，机制被限制在 **Affine Maximizer Auction (AMA)** 这个结构内；只要 bidder weights 保持为正，每一轮机制就继承 **dominant-strategy incentive compatibility (DSIC)** 和 **individual rationality (IR)**。在这个安全边界内，强化学习负责根据历史市场状态、预算变化、近期 revenue 和时间进度来调节 AMA 参数。

换句话说，文献综述需要帮我们把叙事从“我们用 RL 模拟 bidder 学习报价”调整为：

> bidder 的 truthful reporting 由机制结构保证；RL 学的是机制设计者这一侧的参数策略。

这句话非常关键。它会直接影响 Related Work 的写法，也会影响老师怎么看这个工作是不是一个机制设计问题，而不只是一个多智能体强化学习模拟。

## 1. Classical Foundations: AMA 为什么是合适的起点

机制设计的经典起点是单物品最优拍卖和 Vickrey-Clarke-Groves (VCG) 机制。Myerson (1981) 解决了单物品 revenue-optimal auction 的刻画问题，说明在一类标准假设下，卖方可以通过 virtual valuation 来构造最优拍卖。VCG 机制则从另一个方向给出一个非常强的基准：在准线性偏好和私有信息环境中，它可以让 truthful reporting 成为 dominant strategy，并且满足 individual rationality。

这两条线索共同说明了一件事：机制设计最难的地方之一，不只是“怎么提高收入”，而是在提高收入的同时不要破坏参与者如实报告的激励。VCG 的强项在于激励性质非常干净，但它通常服务于效率或社会福利最大化，未必是 revenue-optimal。Myerson 的结论很漂亮，但它主要解决单物品场景。到了 multi-bidder, multi-item 的组合拍卖，最优收入机制的精确刻画就困难得多。RegretNet 等后续机器学习机制设计工作也常常以这一点作为出发点：多物品最优拍卖在理论上并没有被完全解决。

因此，我们不应该把 HRL-AMA 写成“解决了多物品最优拍卖”。更稳妥的说法是：我们选择了一个可解释、可保证激励性质的机制类，然后在这个机制类内部做动态 revenue optimization。

Affine Maximizer Auction (AMA) 正好提供了这样的结构。AMA 可以理解成 VCG 的加权和加 boost 版本：分配时不是直接最大化原始 welfare，而是最大化经过 bidder weights 和 allocation boosts 变换后的 affine welfare；支付规则仍然采用 VCG-style 的 externality payment。它最适合我们工作的地方在于，DSIC 和 IR 不是训练出来的经验结果，而是机制形式本身带来的结构性质。只要 bidder weights 为正，机制每一轮就有 exact DSIC/IR。

这也是为什么 AMA 比一个任意 neural network auction 更适合作为我们当前项目的骨架。一个完全自由的神经网络机制也许能拟合出很高 revenue，但它往往需要额外的 regret penalty 来逼近 incentive compatibility。AMA 则反过来：它先把 incentive compatibility 放在结构里，再把学习能力留给参数调节。

不过，已有文献已经研究过如何优化 AMA 参数。Likhodedov and Sandholm (2004, 2005) 以及 Sandholm and Likhodedov (2015) 都属于 automated mechanism design 的经典线索：在已知或采样得到的估值分布下，搜索或优化 AMA 参数来提高 revenue。这个事实提醒我们：如果只说“优化 AMA 参数”，并不新。我们的潜在 gap 应该放在 repeated 或 non-stationary 环境中，让 RL 根据动态市场状态持续选择 AMA 参数。

因此，本节对 ourwork 的启发可以概括为：

- AMA 是“激励保证的安全壳”，负责 dominant-strategy incentive compatibility (DSIC) 和 individual rationality (IR)。
- RL 是“动态调参策略”，负责在 repeated auction horizon 中提高 cumulative revenue。
- AMA 保证 DSIC 和 IR 不是我们的新发现；我们要强调的是如何把这个结构和动态 revenue optimization 结合起来。

## 2. Differentiable and Automated Mechanism Design: 灵活性和精确激励保证之间的取舍

Dutting et al. (2019) 的 RegretNet 是和我们最需要比较的 deep learning auction 工作之一。它把 auction mechanism 表示成神经网络，输入 bidder valuations，输出 allocation 和 payment，然后用 expected revenue 作为主要优化目标。为了处理 incentive compatibility，它把 ex-post regret 写进训练目标，并通过 augmented Lagrangian 方法让 regret 尽量趋近于零。

RegretNet 的意义在于，它把原本很难解析求解的多物品拍卖设计问题转化成了一个可训练的学习问题。它可以在很多未知最优机制的场景里找到高 revenue 的机制，也能恢复一些已有理论解。这是它的强项。但从我们项目的角度看，它的弱点也正好是我们的切入点：RegretNet 通常得到的是 approximate IC，也就是经验 regret 很小，而不是由机制形式直接保证的 exact DSIC。

RochetNet 和 MenuNet 则代表另一种思路：通过机制结构或凸性刻画来保证 strategyproofness。这类方法说明，“architecture-level truthfulness”是可以做到的。但它们多集中在 single-bidder 或菜单式机制设置中，和 general multi-bidder combinatorial auction 还有距离。它们对我们的启发是：如果能把学习模型限制在一个天然 truthful 的结构里，训练就不必一直和 regret penalty 拉扯。

Curry, Sandholm, and Dickerson (2022) 的 **Differentiable Economics for Randomized Affine Maximizer Auctions** 与我们的工作尤其接近。该文把 AMA 放进 differentiable economics 框架中，学习 bidder weights、allocation boosts，甚至 lottery allocations。它非常明确地强调了一个我们也想强调的点：AMA 支持 multiple bidders and items，并且 strategyproof by construction。换句话说，这篇文献已经做到了“学习 AMA，并保留 exact strategyproofness”。

所以，我们在写文献综述时必须承认：HRL-AMA 不是第一个把 AMA 和学习方法结合起来的工作。我们应该把差异说得更细：Lottery AMA 主要是静态机制训练，也就是在一个分布下训练出一个机制；而我们的目标是 repeated auction setting，在每一轮根据历史状态选择 AMA 参数。这不是简单的“谁更好”，而是研究问题的时间结构不同。

Duan et al. (2023) 的 AMenuNet 和 Duan et al. (2024) 的 OD-VVCA 进一步说明，exact DSIC 和 IR 的可学习机制已经形成了一条相当清楚的研究线。AMenuNet 用神经网络生成 AMA 参数和 allocation menu，从而改善 AMA 在大规模候选 allocation 上的可扩展性；OD-VVCA 则把机制限制在 Virtual Valuations Combinatorial Auctions 这类本身满足 DSIC 和 IR 的 deterministic AMA 子类中，并通过 objective decomposition 提升训练效率和 revenue。它们都提醒我们：如果论文只写“我们用神经网络设计 DSIC combinatorial auction”，会显得和已有工作重合太多。

Sun et al. (2026) 的 **Correlation-Aware AMA (CA-AMA)** 则指出 AMA 的另一个重要限制：经典 AMA 的 VCG-style payment 在 bidder valuations 相关时可能表达能力不足。CA-AMA 通过加入只依赖其他 bidders' valuations 的 correlation-aware payment 项来增强支付表达能力。因为这个新支付项不依赖 bidder 自己的 bid，所以 DSIC 仍然能保留；但 IR 需要额外约束或后处理。

这篇 CA-AMA 对我们尤其有提醒意义。如果我们的实验主要是 independent valuations、LLG 或简单 non-stationary distributions，就不能声称解决了 correlated valuation 下 AMA 表达能力不足的问题。更好的写法是：correlated valuations 是 AMA-based dynamic learning 的一个自然扩展方向，CA-AMA 可以作为 future work 的参考。

这一组文献最后给我们的定位是：

- RegretNet 类方法很灵活，revenue 表现强，但 incentive compatibility 多为 approximate。
- AMA/AMenuNet/OD-VVCA 类方法把 exact DSIC 和 IR 放进结构里，但主要研究静态机制学习。
- CA-AMA 增强了 AMA 在相关估值分布下的 payment expressiveness，但不是动态 RL 框架。
- HRL-AMA 可以尝试站在它们之间：用 AMA 结构保留每一轮的 exact DSIC 和 IR，再用 RL 做 repeated environment 中的动态 revenue tuning。

## 3. Dynamic and RL-based Mechanisms: 动态机制已有，但问题侧重点不同

如果只看“动态机制设计”，我们的工作也不是从零开始。Athey and Segal (2013) 的 **An Efficient Dynamic Mechanism** 在动态私有信息环境中构造 efficient mechanism，讨论 ex post incentive compatibility、Bayesian incentive compatibility、budget balance 和 self-enforcement 等问题。它说明动态机制设计本身已经有成熟理论，也提醒我们一个重要事实：一旦进入多轮环境，incentive compatibility 会比单轮更微妙。

这一点和 ourwork 直接相关。我们目前比较稳的理论说法是：每一轮参数先由 RL 根据历史状态确定，然后 bidder 在当轮提交 bid；在这个顺序下，当轮机制就是一个固定参数的 AMA，所以有每轮的 exact DSIC 和 IR。可是这并不等于完整的 dynamic incentive compatibility。一个非常耐心、非常会算的 bidder 也许可以在早期故意压低 bid，影响后续状态统计，从而改变未来 AMA 参数和未来 payment。这个问题应该写在 limitation 里，而不能被一句“dynamic DSIC”带过去。

Reinforcement Mechanism Design (Shen et al., 2017) 把 reinforcement learning 用到 sponsored search auctions 和 dynamic reserve pricing 中。它说明 RL 早就可以被用来调节拍卖机制参数，尤其是在 repeated auction environment 中。但它主要关注 sponsored search 和 GSP-like settings，不是 exact-DSIC combinatorial AMA。因此，我们不能说“第一个用 RL 做拍卖机制设计”，但可以说我们的机制类和激励保证不同。

Qiu et al. (2022) 的 **Learning Dynamic Mechanisms in Unknown Environments** 和 Leon and Etesami (2025) 的 **Online Learning for Dynamic VCG Mechanism** 更接近动态机制理论与在线学习/RL 的交叉。它们研究 unknown MDP 环境中的 dynamic VCG-like mechanisms，并给出 regret、truthfulness 或 efficiency 相关保证。这类工作对我们尤其重要，因为它们说明“dynamic VCG + learning”已经存在。我们的区别应当放在：它们多围绕 efficiency、dynamic VCG 和 unknown environment learning，而我们关注的是 repeated combinatorial auction 中，在 AMA parameter space 里做 revenue-oriented tuning。

Ravindranath et al. (2024) 的 **Deep Reinforcement Learning for Sequential Combinatorial Auctions** 是另一个必须认真比较的近邻。它已经把 deep reinforcement learning 和 sequential combinatorial auctions 联系起来，并且展示了在较大规模场景中的 revenue improvement。因此，我们也不能说“没人用 deep RL 做 sequential combinatorial auction”。我们更应该强调：ourwork 不是任意 sequential auction rule 的学习，而是固定 per-round AMA structure，让每轮机制通过 AMA 继承 exact DSIC 和 IR，再学习动态参数策略。

因此，动态/RL 文献给出的结论不是“这块没人做”，而是：

> 已有工作分别研究了 dynamic mechanism、dynamic VCG、RL pricing、sequential combinatorial auctions 和 static exact-DSIC auction learning；但把 repeated combinatorial auction、exact-DSIC AMA structure 和 RL-based revenue parameter tuning 明确放在一起研究的工作相对少。

这句话比“目前没有相关文献”更诚实，也更容易经得住老师追问。

## 4. 建议写进 Related Work 的 gap paragraph

英文版可以这样写：

> Existing work has made substantial progress along three separate axes. Classical VCG and Affine Maximizer Auction (AMA) mechanisms provide exact dominant-strategy incentive compatibility (DSIC) and individual rationality (IR), but are often static and may be revenue-suboptimal. Differentiable mechanism design methods such as RegretNet learn high-revenue multi-item auctions, but typically enforce incentive compatibility only approximately through regret penalties. Recent AMA-based neural architectures restore exact DSIC and IR mostly in static auction design settings. Meanwhile, dynamic mechanism design and reinforcement-learning-based auction design study repeated environments, dynamic VCG, reserve pricing, or sequential combinatorial auctions, but do not directly focus on dynamic revenue optimization within an exact-DSIC AMA parameter space. Our work positions HRL-AMA in this gap: we fix the AMA structure to inherit per-round exact DSIC and IR and use reinforcement learning to adapt AMA parameters over repeated rounds, with non-stationarity and budget dynamics motivating the hierarchical architecture.

中文解释版可以这样写：

> 现有文献已经分别推进了三个方向：VCG 和 Affine Maximizer Auction (AMA) 提供了 exact dominant-strategy incentive compatibility (DSIC) 和 individual rationality (IR) 的结构保证，但通常是静态机制，且 revenue 不一定高；RegretNet 等 differentiable mechanism design 方法可以学习高收入的多物品拍卖，但 incentive compatibility 多依赖 regret penalty，只能做到近似；近年的 AMA-based neural architectures 又把 exact DSIC 和 IR 带回了学习机制设计，但主要仍是静态拍卖设计。另一方面，动态机制设计和 reinforcement learning 拍卖文献研究了 repeated environments、dynamic VCG、reserve pricing 或 sequential combinatorial auctions，却较少直接研究在 exact-DSIC AMA 参数空间中进行动态 revenue optimization。HRL-AMA 可以定位在这个交叉缺口上：固定 AMA 结构以继承每轮 exact DSIC 和 IR，再用 RL 根据历史市场状态调节 AMA 参数；非平稳估值和预算动态则是 hierarchical architecture 的主要动机。

这段话的语气比较稳，因为它没有过度宣称：

- 没有说我们第一个保证 DSIC。
- 没有说我们第一个用 RL 做拍卖。
- 没有说我们解决了完整 dynamic incentive compatibility。
- 清楚说明我们是在已有线索之间寻找一个交叉问题。

## 5. 对 ourwork 当前稿件的修改建议

### 5.1 Abstract 和 Introduction

当前 `OURWORK_HRL_for_Dynamic_Affine_Maximizer_Auctions__AMAs.pdf` 的标题和摘要主打 HRL-AMA，但实验部分主要展示 VCG、Static AMA 和 Flat DRL。也就是说，稿件的“理论/架构口号”和“实验支撑”之间还有一点距离。

如果短期内不能补完整 HRL 实验，建议把语气从“we empirically confirm HRL-AMA outperforms...”改成更保守的版本：

- “We propose a preliminary HRL-AMA framework...”
- “Flat DRL results motivate a hierarchical extension under non-stationarity...”
- “A full HRL evaluation is the next experimental milestone.”

这样不是削弱工作，而是让工作更可信。老师通常不是怕你们结果还 preliminary，而是怕 claim 比证据走得太快。

### 5.2 Contributions

贡献可以写成三点：

1. We formulate repeated combinatorial auction revenue optimization within the Affine Maximizer Auction (AMA) parameter space.
2. We show that if AMA parameters are committed before current-round bids and bidder weights remain positive, per-round dominant-strategy incentive compatibility (DSIC) and individual rationality (IR) follow from the AMA structure for any reinforcement learning policy.
3. We provide preliminary experiments showing that adaptive AMA parameter tuning can outperform static baselines in several valuation settings, while non-stationarity motivates hierarchical control.

这三点比“我们提出第一个 exact-DSIC dynamic auction”稳得多。它们把理论保证、学习目标和实验状态分别说清楚，也给后续补实验留下空间。

### 5.3 Limitations

limitations 不要写得像道歉，而要写成研究边界。建议明确保留三点：

**Cross-round incentive compatibility.** 当前论证主要保证 per-round DSIC。由于 RL state 会使用历史 bids、revenue 或 budget 信息，strategic bidders 可能通过早期报告影响后续机制参数。这是 dynamic mechanism design 中真实存在的问题，值得作为 future work。

**AMA expressiveness.** 经典 AMA 的 payment form 在 correlated valuation distributions 下可能不够灵活。CA-AMA 已经指出了这一点，并给出 correlation-aware payment 的方向。如果我们的当前实验没有覆盖相关估值结构，就应把它写成 future work，而不是隐含声称已经解决。

**Scalability.** exact AMA solver 需要枚举 allocation，随着 items 数量增长会指数爆炸。当前小规模实验可以说明机制逻辑，但如果要走更大规模组合拍卖，需要 mixed-integer programming、dynamic programming、column generation 或 learned candidate allocations。

## 6. 开题思路回溯：哪些内容仍然可以放进 ourwork

回看开题答辩稿，一个比较清楚的变化是：项目最初的表述很大，像是在讲“用分层深度强化学习解决动态组合机制设计”；现在的 ourwork 更收敛，变成“在 repeated combinatorial auction 中，用 reinforcement learning 调节 Affine Maximizer Auction (AMA) 参数，同时保留每轮 exact DSIC 和 IR”。这个变化不是坏事。相反，它让工作更像一个可以被文献定位、可以被证明、也可以被实验检验的问题。

开题稿里最值得保留的是三个关键词：动态、分层、反馈。它们仍然是 HRL-AMA 的动机来源，只是需要换成更准确的研究语言。

第一，开题时强调的“动态”仍然重要。真实的组合拍卖或资源分配不是一次性的：估值分布会变，bidder budget 会消耗，竞争强度也会随时间变化。静态机制参数可能在某个固定分布下表现不错，但遇到 non-stationarity 时就不一定稳。现在的 HRL-AMA 可以把这个动机说得更具体：auctioneer 根据历史 bid statistics、budget ratio、time progress 和 recent revenue 来选择当轮 AMA 参数，而不是一直使用同一套静态参数。

第二，开题时强调的“分层”也可以保留，但分层对象要重新解释。早期稿件里有“高层机制优化-中层个体决策-低层策略执行”的三层结构，也提到 MADDPG 和 MCTS。现在不建议继续把重点放在 bidder 自己学习报价策略上，因为机制设计很多时候恰恰希望 bidder 的策略简单到 truthful reporting。更适合当前 ourwork 的说法是：分层发生在 auctioneer 侧。高层 controller 决定当前阶段的宏观目标或机制倾向，低层 worker 把这个目标翻译成具体 AMA parameters，最后 deterministic AMA solver 清算当轮拍卖。这样既保留了开题时“分解复杂决策”的直觉，又不会把问题带偏到多智能体 strategic bidding。

第三，开题时的“反馈闭环”和“动态奖励”可以转化成 RL state/reward 设计的动机。重复拍卖里，auctioneer 关心的是 cumulative revenue，而不是孤立的一轮收入；同时还要考虑预算消耗和非平稳变化。因此，current framework 中的闭环可以写成：state -> RL policy -> AMA parameters -> deterministic AMA outcome -> revenue/budget feedback -> next state。这个说法比“智能体实时调整报价”更贴近现在的机制设计主线。

同时，开题稿里有两类内容不适合照搬。第一类是“bidder 学会怎么报价”的表述。它可以作为早期探索背景，但不应该成为现在论文的核心叙事。现在更好的说法是：学习发生在 auctioneer 侧，bidder 侧通过 AMA 结构获得 truthful reporting 的激励。第二类是没有当前实验支撑的强效果数字，比如收益提升百分比、效率提升倍数、响应时间、代码规模、数据规模或顶会投稿承诺。这些可以被理解为开题时的愿景，但不应写成 current result。

因此，开题思路可以作为 ourwork 的补充背景写进 literature review：它说明项目从一开始就关注动态性、分层决策和反馈优化；而当前 HRL-AMA 是对这些早期直觉的收束。我们不再声称要做一个通用 dynamic CMD 求解器，而是把问题落到更具体、更可验证的方向：在 exact-DSIC AMA 结构中进行 repeated auction revenue optimization。

可以写进论文或结题材料的一段自然表述是：

> 开题阶段的项目设想强调了三个问题：组合机制设计面对高维分配空间，传统静态机制难以适应动态市场变化，而纯深度学习方法虽然灵活，却容易失去可解释性和激励保证。当前的 HRL-AMA 可以看作对这一早期设想的收束：我们不再把重点放在 bidder 学习复杂报价策略上，而是把学习移动到 auctioneer 侧，让机制设计者在 AMA 这一 exact-DSIC 结构内动态调节参数。这样既保留了开题时关于动态性、分层决策和反馈闭环的直觉，也让问题更贴近机制设计文献中的核心约束。

## 7. 给张萌的 deep research prompt

下面这段可以直接发给张萌老师，用来让 deep research 帮你们找近邻文献：

```text
We are working on a literature review and research positioning for a project tentatively titled "HRL-AMA: Hierarchical Reinforcement Learning for Dynamic Affine Maximizer Auctions." The intended problem is repeated combinatorial auction revenue optimization. Our current idea is not to let bidders learn bidding strategies; instead, the auctioneer fixes the mechanism class to Affine Maximizer Auctions (AMAs), which are dominant-strategy incentive compatible (DSIC) and individually rational (IR) by construction when bidder weights are positive, and uses reinforcement learning to tune AMA parameters across rounds based on historical market states such as budgets, bid statistics, time, and recent revenue.

Please search for papers on reinforcement learning, online learning, differentiable economics, or automated mechanism design for combinatorial auction/mechanism design with exact dominant-strategy incentive compatibility (DSIC) and individual rationality (IR) guarantees. Prioritize mechanisms based on affine maximizer auctions, virtual valuations, dynamic VCG, sequential combinatorial auctions, or structured neural architectures that guarantee truthfulness by design. Compare these papers with an approach that fixes AMA structure and uses RL to tune AMA parameters over repeated rounds.

For each paper, summarize: auction setting, mechanism class, whether it is static or dynamic, IC/IR guarantee, learning method, whether it handles combinatorial items, main limitation, and how it overlaps with or differs from our HRL-AMA idea. Pay special attention to whether any prior work already studies dynamic or repeated revenue optimization within an exact-DSIC AMA parameter space.
```

## 8. Key References

- Myerson, R. B. (1981). Optimal Auction Design.
- Vickrey (1961), Clarke (1971), Groves (1973). VCG foundations.
- Roberts, K. (1979). The characterization of implementable choice rules.
- Likhodedov and Sandholm (2004, 2005); Sandholm and Likhodedov (2015). Revenue boosting / automated mechanism design with AMAs.
- Dutting et al. (2019). [Optimal Auctions through Deep Learning / RegretNet](https://arxiv.org/abs/1706.03459).
- Curry, Sandholm, and Dickerson (2022). [Differentiable Economics for Randomized Affine Maximizer Auctions](https://arxiv.org/abs/2202.02872).
- Duan et al. (2023). [A Scalable Neural Network for DSIC Affine Maximizer Auction Design](https://arxiv.org/abs/2305.12162).
- Duan et al. (2024). [Automated Deterministic Auction Design with Objective Decomposition](https://arxiv.org/abs/2402.11904).
- Sun et al. (2026). [Enhancing Affine Maximizer Auctions with Correlation-Aware Payment](https://arxiv.org/pdf/2602.09455).
- Athey and Segal (2013). [An Efficient Dynamic Mechanism](https://web.stanford.edu/~isegal/agv.pdf).
- Shen et al. (2017). [Reinforcement Mechanism Design, with Applications to Dynamic Pricing in Sponsored Search Auctions](https://arxiv.org/abs/1711.10279).
- Qiu et al. (2022). [Learning Dynamic Mechanisms in Unknown Environments](https://arxiv.org/abs/2202.12797).
- Leon and Etesami (2025). [Online Learning for Dynamic Vickrey-Clarke-Groves Mechanism in Unknown Environments](https://arxiv.org/abs/2506.19038).
- Ravindranath et al. (2024). [Deep Reinforcement Learning for Sequential Combinatorial Auctions](https://arxiv.org/abs/2407.08022).
