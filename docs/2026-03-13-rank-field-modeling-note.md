# Rank Field Modeling: 从 R3GAN 的 rank 实现到新的生成范式

Date: 2026-03-13

Status: Research note / design memo

这份笔记不是在复述当前代码，而是借当前仓库里的 rank 相关实现，把一个更本质的问题写清楚：

> 最好的学习 rank 的方式，一定是 GAN 吗？如果不是，那么“最能学到 rank 的生成模型”到底是什么？我们是否应该把 rank 从一个 GAN trick，提升成新的 generative modeling 第一对象？

本文基于当前仓库里的三条实现线索：

- `training/loss.py` 里的 `pairwise_delta()` / `pairwise_{discriminator,generator}_loss()`：当前主干仍然是 delta-centric 的相对比较博弈。
- `training/loss.py` 里的 `build_local_coupling()` / `local_*`：我们已经在尝试把“一个 fake 相对于一个 real 的比较”推广成“相对于局部 real frontier 的比较”。
- `training/loss.py` 里的 `_compute_path_rank_loss()` 与 `make_rank_list()`：我们还保留了一条被标记为 deprecated 的 interpolation path-rank prior，它显式在 real-to-fake chord 上施加单调排序约束。

README 当前也已经把这三层结构说得很清楚：

- `RankGAN-v1` 是 pairwise delta core。
- `RankGAN-v2/v3` 是 pairwise core 之上叠加 listwise / local prior。
- `path_rank_reg` 被明确定义为 deprecated path prior，而不是主博弈。

这恰好暴露了一个重要事实：**我们现在的系统已经不只是“GAN 加一个 rank loss”，而是在无意中摸到了 comparator、path 和 field 三种不同对象。**

## 1. 当前实现已经告诉了我们什么

### 1.1 R3GAN 主干最擅长的是 comparator，不是 path

当前主干的第一性原理仍然是

$$
\Delta_{ij} = s_\psi(x_i^r) - s_\psi(\hat x_j),
$$

也就是一个单一 scalar critic 所定义的相对差值。无论是 `pairwise_delta()` 还是 local-coupled 的 `local_delta()`，本质都还是：

- 哪个 real 比哪个 fake 更高；
- 某个 fake 相对于一组邻近 real 的差距有多大。

这非常强，也非常重要。但它本质上学习的是 **comparison rank**，也就是“谁比谁更好”。

### 1.2 local coupling 把 comparator 变得更局部，但还没有变成 path

`build_local_coupling()` 用 clean feature 在 fake 和 real 之间建稀疏邻域，再用 `local_pairwise_*` 或 `local_listwise_*` 聚合局部比较。这个设计把 one-anchor boundary 推向了 many-anchor local frontier，是非常有价值的一步。

但要注意：它依然没有直接回答下面两个问题：

- 当前 fake 处在“由差到好”的哪一段？
- 从当前状态出发，下一步应该往哪个方向、以什么速度移动？

local coupling 仍然是在做 **局部比较结构化**，而不是 **全局可积路径建模**。

### 1.3 path-rank prior 是一个很关键的信号

`_compute_path_rank_loss()` 会构造

$$
[x_r, x_{\lambda_1}, \ldots, x_{\lambda_{K-2}}, x_f]
$$

这样的 real-to-fake 插值链，并要求 D 在这条链上满足单调排序。这里最有研究价值的点不是它现在是不是 canonical loss，而是它暴露出了一件事：

> 一旦我们开始在一条连续路径上要求 critic 单调，我们问的就不再只是“谁更好”，而是在问“怎样沿着一条路径变好”。

这已经开始触碰 path learning，而不是单纯 comparator learning。

### 1.4 当前实验信号支持“path prior 有研究价值”，但还不是 clean proof

现有训练结果里：

- baseline `00007-cifar10-gpus2-batch512` 的最佳 FID 是 `2.3307`。
- 带 path-rank 的 `00023-cifar10-gpus2-batch512-pair1m1-pathrank-pairwise_hinge-pair1_m1_pathrank_hinge_k3_dbgpu128` 的最佳 FID 是 `2.2285`。

这说明 path-rank 至少不是一个明显错误的方向。但这个比较还不是严格对照实验，因为 `00023` 同时改了 `pair_margin=1.0`，并且 `d_batch_gpu` 也从 `256` 变成了 `128`。因此更准确的表述是：

> path-rank 提供了一个值得认真对待的研究信号，但它目前还不是“path 范式优于 baseline GAN”的干净证明。

## 2. rank 至少要拆成三类

我们关心的 rank，至少有三种彼此不同的对象：

### 2.1 比较型 rank

定义：谁比谁更好。

形式上最像

$$
C(x_i, x_j) = r(x_i) - r(x_j),
$$

或者更弱一点，只要求能判断 `x_i` 是否优于 `x_j`。

R3GAN / RpGAN / relativistic loss 天生最擅长这一类，因为它们一开始就是围绕 real-vs-fake difference 来构造的。

### 2.2 路径型 rank

定义：一个样本当前位于“由差到好”的哪一段，以及沿着这条路径是否单调上升。

这时我们需要的不只是两两比较，而是某种连续坐标或序参量，例如

$$
r_\phi(x) \in \mathbb{R},
$$

使得它可以充当 path coordinate，而不是只做 binary comparator。

当前 `path_rank_reg` 已经在这件事上迈出了一小步，因为它显式地在 real-fake chord 上要求单调。

### 2.3 动力学型 rank

定义：给定当前样本或当前分布，下一步该往哪个方向走才会更好。

这时仅有 `r(x)` 还不够，还需要一个 update field / transport field：

$$
V_\theta(x, r, t)
$$

或者一个离散更新算子：

$$
x' = T_\theta(x, r, t).
$$

只有这时，rank 才从“描述顺序”变成“驱动生成”。

## 3. 为什么 GAN 不是学习 rank 的本质形式

一旦把 rank 拆成上面三类，结论就会很清楚：

> GAN 天生强在第 1 类，也就是比较型 rank；它并不天然解决第 2、3 类。

R3GAN 的优势在于：

- 它把 pointwise 真伪判断改成了 relative comparison。
- 它用 R1/R2 把 equilibrium 压成 flat constant-score equilibrium，避免在最优点附近继续振荡。

但这依然意味着：

- 它给我们的是局部比较信号；
- 它没有直接给出一个全局可积的 rank path；
- 它也没有显式给出一个“如何从低 rank 往高 rank 运输”的动力学对象。

因此，**GAN 不是学 rank 的本质形式；GAN 只是学局部比较 rank 的一种高效实现。**

更尖锐地说：

> 当我们把主对象定义成对抗比较时，我们其实是在学 comparator；当我们把主对象定义成 potential + transport 时，我们才开始真正学 path 和 field。

## 4. 从本体上，最适合学习 rank 的对象是什么

如果问题不是“哪种现成模型最流行”，而是“什么对象最符合 rank 的数学本性”，那么更合理的答案应该是：

> 最适合学习 rank 的，不是纯 GAN，而是 potential / field-based generative model。

原因有两个。

### 4.1 rank 的最干净对象是标量势函数，而不是 binary 判别

如果我们真的想表达“什么叫更好”，最自然的对象是一个标量势函数

$$
r_\phi(x) \in \mathbb{R},
$$

它允许我们写出：

- 全局可比较性：`r(x_1) < r(x_2)`；
- 局部差值：`C(x_i, x_j) = r(x_i) - r(x_j)`；
- 路径坐标：样本位于从低 rank 到高 rank 的哪一段。

和 critic 相比，这个对象更本质，因为它不依赖于 real-vs-fake 二元博弈本身。

### 4.2 生成要求的不只是排序，还要有可执行更新

只有一个 rank potential 还不够，生成还需要一个可以执行的 drift / transport law：

$$
x' = x + V_\theta(x, r_\phi(x), t).
$$

否则我们只能说“哪个点更好”，却没法说“怎么从当前点变得更好”。

所以，一个真正围绕 rank 构建的生成模型，至少要同时拥有：

- 一个全局 rank potential；
- 一个沿 rank 上升的 transport / drift field；
- 一个在 top rank 处稳定归零的 fixed point 机制。

## 5. 现有工作里，谁更接近这个本质

下面这段不是文献综述，而是当前研究方向中的概念对照。

### 5.1 R3GAN 最像局部比较型 rank learner

在我们当前代码里，R3GAN/RankGAN 主干仍然是最强的 comparator：

- `pairwise_delta` 定义局部比较 primitive；
- `local coupling` 提升局部 frontier 的表达能力；
- `R1/R2` 保证比较 game 在 equilibrium 附近不会乱推。

如果问题是“谁比谁更好”，这条线已经很强。

### 5.2 iMF / pMF 更像 sample-level 的 rank path learner

从我们的研究框架看，这类方法的重要启发在于：

- 它们显式建模两个时刻之间的演化；
- 它们把问题从真假对抗，转成了状态在路径上的推进；
- 它们强调可学习的状态表示必须更靠近数据流形。

这和当前 `path_rank_reg` 的启发是同向的：真正重要的不是只会比较，而是要能表达“如何沿路径变好”。

### 5.3 Drifting 更像 distribution-level 的 rank field 雏形

如果我们把问题提升到整个分布，最关键的对象就不再是单样本得分，而是训练诱导出的分布演化场。这个视角提醒我们：

- 真正的 top rank 不应只是“高分”，而应是一个稳定平衡；
- 真正的生成过程不一定发生在 inference-time，也可以体现在 training-time 的分布推进中。

这和当前 R3GAN 里用 R1/R2 保证 equilibrium 的思路是兼容的，只是对象从 sample comparator 进一步提升到了 distribution field。

## 6. 新范式提案：Rank Field Modeling

如果沿着上面的逻辑继续推进，一个更自然的新范式不是“更会排序的 GAN”，而是：

## Rank Field Modeling (RFM)

它的中心对象不是 adversarial game，也不是预设噪声时间表，而是下面三个统一对象：

### 6.1 全局 rank potential

$$
r_\phi(x) \in \mathbb{R}
$$

它定义样本在“由差到好”序上的位置。

### 6.2 局部比较算子

$$
C_\phi(x_i, x_j) = r_\phi(x_i) - r_\phi(x_j)
$$

它继承 R3GAN 的 pairwise relative ranking 优点，为局部监督提供 comparator。

### 6.3 rank-conditioned transport / drift

$$
x' = x + V_\theta(x, r_\phi(x), t)
$$

或者

$$
x_r = T_\theta(x_t, r, t).
$$

它让 rank 不再只是静态分数，而成为可以驱动生成过程的更新规律。

从这个角度看，一个统一的 rank 生成范式应该满足：

- 样本可以被全局排序，而不只是局部比较。
- 样本可以沿 rank field 单调改善，而不只是被判为更真或更假。
- top rank 必须是稳定 fixed point，在最优处更新应当归零。

## 7. 这个范式如何统一当前 repo 里的几条线

如果用 RFM 的眼光回看当前实现，我们可以更清楚地理解每一块代码的理论角色。

### 7.1 `pairwise_delta` 给出局部比较 primitive

这是 RFM 里的 comparator 层。它对应

$$
C_\phi(x_i, x_j).
$$

我们不该放弃这层，因为它仍然是最干净、最稳定、最 identifiable 的局部监督来源。

### 7.2 `local coupling` 给出 manifold-aware 局部 frontier

这是从单一 anchor 到局部 frontier 的推广。它提示我们：

- rank 监督不该只靠全局 all-vs-all；
- 更合理的是在 clean feature 上找可信的 local neighborhood。

这为后续的 rank path / transport 提供了 neighborhood geometry。

### 7.3 `path_rank_reg` 给出 chord-monotonic path prior

当前实现虽然把它标成 deprecated，但从研究角度看，它极其重要，因为它说明：

- 我们已经开始在一条连续路径上施加序关系；
- 这件事更像 path-shaping，而不是单纯 comparator regularization；
- 它更适合作为 transient geometric prior，而不是 equilibrium-defining main game。

也就是说，它不一定适合永远做 canonical 主损失，但它非常适合作为下一代 rank path 建模的原型。

### 7.4 R1 / R2 给出 top-rank equilibrium 的雏形

RFM 不能只有上升方向，还必须有“到顶后归零”的机制。当前 R3GAN 的强点就在于：

- 最优处要求 constant-score equilibrium；
- R1/R2 把判别器梯度压平；
- 这让 top rank 不只是高分区，而是一个稳定平衡。

这部分应该被保留，并升级成 Rank Field 范式里的 fixed-point requirement。

## 8. 一个更准确的判断

如果必须用一句话来下判断，我会写成：

> GAN 不是学习 rank 的本质形式；GAN 只是学习局部比较 rank 的高效实现。真正从本质上最适合学习 rank 的生成模型，应当是一个 potential-guided transport / drift model，也就是把全局顺序、局部比较、连续路径与稳定平衡统一起来的 Rank Field Model。

对应到我们现在的四条线索，可以压缩成：

- R3GAN 让我们学会“谁更好”。
- path-rank 让我们学会“沿着哪条 chord 变好”。
- local coupling 让我们学会“在哪个局部 frontier 上比较”。
- 下一步真正该做的，是把这三件事统一成“如何在一个 rank field 中持续变好”。

## 9. 对当前 repo 最重要的四条设计原则

如果这个方向继续推进，我会坚持下面四条原则。

### 9.1 rank 不能只停留在 pairwise comparator

pairwise comparator 必须保留，但不能把它当成最终对象。rank 必须被提升成连续变量、势函数或路径坐标。

### 9.2 rank path 必须可积

如果只有局部比较，没有全局一致路径，那么系统学到的只是很多碎片化 boundary，而不是一个可以推进生成的 order field。

### 9.3 状态表示必须 manifold-aware

无论是 local coupling 还是未来的 transport，状态都不应只在 raw critic residual 上表达，更合理的是在 clean feature 或更接近 data manifold 的 `x-like` 表示空间里学习。

### 9.4 top rank 必须是 fixed point

最优点处更新应当归零。否则 rank 只会一直推高分，而不会形成真正稳定的生成平衡。

## 10. 面向实现的下一步建议

这部分只谈本仓库里最现实、最有连续性的推进路径。

### 10.1 重新定位 `path_rank_reg`

不要把它视作失败的旧功能，而应把它重命名为：

- transient path prior
- chord-monotonic geometric regularizer
- pathwise ordinal critic regularization

它的研究角色是“早中期的路径整形项”，而不是最终 equilibrium objective。

### 10.2 保留 delta-centric 主博弈

`pairwise_delta + R1/R2` 仍然应该是系统骨架。因为 comparator 层目前最稳定、最可控、最接近现有理论保证。

### 10.3 把 local coupling 从 loss trick 升级成 rank neighborhood

下一步不应只是“继续加大 local-listwise 权重”，而应让 local coupling 成为 rank path / transport 的 neighborhood oracle，也就是：

- 先决定“我应该参考哪个局部 frontier”；
- 再决定“我沿着哪条 rank path 更新”。

### 10.4 尝试显式 rank potential + update field

最值得探索的新模块不是再加一个 discriminator head，而是显式学习：

$$
r_\phi(x), \qquad V_\theta(x, r, t).
$$

其中 `r_\phi` 可以与当前 scalar critic 共享主体表示，而 `V_\theta` 则负责把 rank 变成真正的生成更新。

## 11. 结论

当前仓库里的 rank 相关实现，实际上已经给出了一条很清晰的研究轨迹：

- 从 `pairwise_delta` 开始，我们学到了局部比较型 rank。
- 从 `local coupling` 开始，我们学到了 rank 必须尊重局部流形结构。
- 从 `path_rank_reg` 开始，我们第一次看到了“沿路径排序”比“只比较端点”更接近生成的本质。

因此，这份笔记最终想记录的不是一句“GAN 不行”，而是一个更积极的判断：

> GAN 不是 rank 的终极形式，但它帮助我们识别出了 rank 的第一个原子对象：local comparator。真正的新范式应当在这个原子之上，进一步学习全局 rank potential、可积 path，以及在 top rank 处归零的 transport / drift field。

如果这个方向成立，那么我们接下来要做的，就不只是继续做 RankGAN 变体，而是尝试定义一种新的 generative modeling language：

## Rank Field Modeling

