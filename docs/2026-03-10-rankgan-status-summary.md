# RankGAN / Local-Coupled Design And Analysis Summary

Date: 2026-03-10

This document summarizes the current code design, the main experiments that were run, what has already been fixed, what is still failing, and the most likely next directions. It is written so that an external model can reason about the system without reading the code directly.

## 1. Executive Summary

- The repository has been refactored from a simple R3GAN baseline into a delta-centric RankGAN framework.
- The new framework supports:
  - pairwise delta adversarial loss
  - listwise InfoNCE-style delta loss
  - local kNN coupling between fake and real samples
  - D-only semantic-local rank prior
  - deprecated interpolation path-rank prior
  - full-batch replay for losses that require global batch structure
- A correctness bug in conditional CIFAR-10 coupling was found and fixed:
  - old coupling did not enforce same-class neighbors
  - new coupling is class-aware and logs detailed coupling statistics
- Even after the class-aware fix, the current coupled local-listwise setup still does not work well on CIFAR-10.
- The strongest current evidence is that the coupled local-listwise objective itself is the problem, not the implementation bug anymore.
- The sharper diagnosis is that we turned what should have been a weak D-side geometric prior into a high-entropy local transport target induced by live discriminator features.
- The local coupling weights are almost uniform across neighbors, so the method is averaging same-class neighbors instead of constructing informative local matches.
- This is best understood as critic blurring rather than critic sharpening.
- A separate legacy path-rank configuration is theoretically more plausible, but currently OOMs under the default per-GPU microbatch because D-phase path-rank adds a large extra discriminator forward/backward graph.
- The best current research direction is not "stronger coupling", but an Anchor-Path-Rank design:
  - keep the global pairwise delta game unchanged
  - use local structure only to select a trusted anchor
  - apply a weak D-only 1D monotonic prior along the fake-to-anchor path

## 2. Project Context

Base repository:

- Official R3GAN / modern GAN baseline, adapted from StyleGAN3 training infrastructure.

Baseline run used for comparison:

- `00007-cifar10-gpus2-batch512`
- This is plain R3GAN with no added rank/coupling machinery.

Important files in the current implementation:

- `training/loss.py`
- `training/training_loop.py`
- `training/networks.py`
- `R3GAN/Networks.py`
- `train.py`
- `tests/test_adv_losses.py`

## 3. Current Loss Design

### 3.1 Main design idea

The codebase now uses the score gap

`delta = s(real) - s(fake)`

as the central primitive.

From that primitive, it builds several loss families:

1. Pairwise delta loss
2. Listwise delta loss
3. Local-coupled pair/list losses
4. D-only local semantic rank prior
5. Deprecated interpolation path-rank prior

### 3.2 Pairwise delta loss

This is the main adversarial objective.

- Generator and discriminator both operate on a pairwise real-vs-fake score gap.
- `pair_margin=0` reproduces the RpGAN-like soft-margin case.
- Positive margin makes the discriminator try to maintain a larger score gap.

This is the most stable and best-understood part of the new framework.

### 3.3 Listwise delta loss

This is an InfoNCE-style listwise loss over real-fake comparisons.

- It can be symmetric via `lambda_list`
- Or asymmetric via `lambda_list_d` and `lambda_list_g`

The original intent was:

- pairwise loss sets the primary adversarial game
- listwise loss refines ranking geometry

In practice, large local listwise weights on CIFAR-10 appear to hurt.

### 3.4 Local-coupled losses

The local-coupled mechanism builds a fake-to-real neighborhood using discriminator features.

Current intended pipeline:

1. Run D and extract clean, non-augmented penultimate features
2. L2-normalize features
3. For each fake, find k nearest reals
4. Build a sparse coupling distribution over those real neighbors
5. Use local pairwise or local listwise delta losses based on these coupled neighbors

Important implementation detail:

- This is full-batch only
- It does not operate independently per microbatch
- It requires metadata collection and replay

### 3.5 D-only local semantic rank prior

This is separate from coupling.

- It uses local neighborhoods as a D-side prior only
- It does not directly give G-side gradients
- It is conceptually safer than changing the main real/fake pairing structure

### 3.6 Deprecated interpolation path-rank prior

This is the old rank regularizer.

- It builds interpolation chains between real and fake
- D scores are forced to follow an ordering along the path
- It acts as a D-side prior, not as a full replacement for the main adversarial game

Empirically, this family appears more promising than the local-coupled listwise formulation.

## 4. Key Implementation Details

### 4.1 Full-batch buffering and replay

The most important architectural change is that certain losses now use a full-batch replay pipeline.

This is necessary because:

- coupling needs global feature neighborhoods
- InfoNCE/listwise terms need global batch structure
- naive microbatch backward would compute the wrong objective

Current behavior:

1. During microbatch accumulation, batches are buffered instead of immediately backpropagated
2. Full-batch metadata is collected
3. Metadata may be all-gathered across distributed ranks
4. Global losses and replay gradients are computed
5. Microbatches are replayed to produce exact parameter gradients

This logic lives mainly in:

- `training/loss.py`, around `accumulate_gradients()`, `finalize_accumulation()`, `_collect_coupled_metadata()`, `_prepare_coupled_replay()`, `_replay_coupled_gradients()`

### 4.2 Discriminator features

The discriminator was extended to optionally return penultimate features in addition to scores.

This is used for:

- coupling
- local semantic priors

Feature extraction is based on the clean, non-augmented discriminator pathway.

### 4.3 Class-aware coupling fix

An important conditional-training bug was identified:

- in the original coupled implementation, fake samples could couple to real samples from the wrong class
- on CIFAR-10 conditional training, this makes the local delta target semantically inconsistent

The fix that is now in the code:

- same-class masking is applied during coupling
- if a fake sample has no same-class reals available in the current batch, the row falls back to the unmasked coupling
- the fallback is logged explicitly

### 4.4 Logging improvements

The current code now logs the new machinery much more explicitly.

Added logging categories include:

- `Loss/weights/*`
- `Loss/config/*`
- `Loss/coupling/*`
- `Loss/D/pair`
- `Loss/D/list`
- `Loss/G/pair`
- `Loss/G/list`
- `Progress/lambda_pair`
- `Progress/pair_margin`
- `Progress/lambda_list`
- `Progress/lambda_list_d`
- `Progress/lambda_list_g`
- `Progress/lambda_local_rank`
- `Progress/lambda_path_rank`
- `Progress/list_tau`
- `Progress/coupling_k`
- `Progress/local_rank_k`
- `Progress/path_rank_k`

Before this patch, the coupled pathway incorrectly collapsed pair and list contributions in the stats. That has been fixed.

### 4.5 Tests added for the new logic

New tests were added for:

- class-aware coupling
- coupling fallback behavior
- coupled stats reporting

The targeted unit test file passed after the patch.

## 5. Historical And Current Runs

### 5.1 Baseline: `00007`

Run:

- `00007-cifar10-gpus2-batch512`

Config:

- plain R3GAN
- no explicit extra rank/coupling config in `loss_kwargs`

FID:

- best overall: `2.3307` at `59597 kimg`
- `1024 kimg`: `23.6962`
- `2048 kimg`: `10.2974`
- `3072 kimg`: `6.9210`
- `4096 kimg`: `5.4647`
- `5120 kimg`: `4.7181`
- `10035 kimg`: `4.1785`

Representative score statistics:

- at `3071.5 kimg`
  - `Loss/scores/real ≈ 0.7973`
  - `Loss/scores/fake ≈ -1.0143`
  - `Loss/D/loss ≈ 0.4420`
  - `Loss/G/loss ≈ 1.3910`

Interpretation:

- Baseline R3GAN produces strong real/fake score separation early.
- This is the reference trajectory the new methods should at least match.

### 5.2 Historical rank run: `00015`

Run:

- `00015-cifar10-gpus2-batch512-rank-pairwise_hinge`

Stored loss config:

- `rank_loss=True`
- `rank_K=3`
- `rank_loss_type=pairwise_hinge`
- `lambda_rank=0.1`
- `lambda_adv=1.0`

Best FID:

- `1.9599` at `222413 kimg`

Interpretation:

- This indicates that a D-side interpolation path-rank prior can be productive in at least one historical configuration.
- It is not an apples-to-apples comparison with `00021`, because:
  - it was trained much longer
  - it is from an older configuration style
- Still, it is evidence that "ranking as an auxiliary D prior" is more promising than the current local-coupled listwise formulation.

### 5.3 Coupled run before class-aware fix: `00018`

Run:

- `00018-cifar10-gpus2-batch512-pair1m1-infonce-tau0.07-coupled4-listD0.5-listG0.1-...`

Config:

- `lambda_pair=1.0`
- `pair_margin=1.0`
- `lambda_list=0.5`
- `list_tau=0.07`
- `coupling_k=4`
- `lambda_list_d=0.5`
- `lambda_list_g=0.1`

FID:

- best overall: `7.6740` at `5325 kimg`
- `1024 kimg`: `27.7394`
- `2048 kimg`: `14.2569`
- `3072 kimg`: `9.9100`
- `4096 kimg`: `8.2614`
- `5120 kimg`: `7.7368`
- `10035 kimg`: `8.9982`

Observed failure pattern:

- The run improved at first, then degraded.
- It was much worse than plain R3GAN at the same early/mid training points.

Most likely causes identified at that time:

1. coupling was not class-aware for conditional CIFAR-10
2. `pair_margin=1.0` was too aggressive
3. local listwise weights were too large
4. old logging hid the true listwise contribution

### 5.4 Class-aware coupled run after fix: `00021`

Run:

- `00021-cifar10-gpus2-batch512-pair1-coupled4-listD0.5-listG0-...`

Config:

- `lambda_pair=1.0`
- `pair_margin=0.0`
- `lambda_list=0.0`
- `list_tau=0.07`
- `coupling_k=4`
- `lambda_list_d=0.5`
- `lambda_list_g=0.0`
- no path-rank
- no local-rank prior

FID:

- current best: `9.6226` at `3482 kimg`
- `1024 kimg`: `22.4830`
- `2048 kimg`: `12.6436`
- `3072 kimg`: `9.9981`

This is still much worse than `00007`:

- `00007 @ 3072 kimg = 6.9210`
- `00021 @ 3072 kimg = 9.9981`

## 6. What The New Logs Tell Us About `00021`

### 6.1 The class-aware fix is working

At representative training points:

- `Loss/coupling/same_class_mass = 1.0`
- `Loss/coupling/fallback_rows = 0.0`

Interpretation:

- Every coupling row is effectively same-class
- No fallback to cross-class behavior is happening

So the old conditional coupling bug is not the reason `00021` is failing.

### 6.2 The D-side listwise term is genuinely active

Representative values:

- at `511.5 kimg`
  - `lambda_list_d ≈ 0.0511`
  - `Loss/D/list_weighted ≈ 0.0456`
- at `1024.5 kimg`
  - `lambda_list_d ≈ 0.1024`
  - `Loss/D/list_weighted ≈ 0.0632`
- at `2047.5 kimg`
  - `lambda_list_d ≈ 0.2047`
  - `Loss/D/list_weighted ≈ 0.0794`
- at `3071.5 kimg`
  - `lambda_list_d ≈ 0.3071`
  - `Loss/D/list_weighted ≈ 0.1332`

Interpretation:

- The local listwise term is not "inactive"
- The logging fix worked
- The objective itself is active and affecting training

### 6.3 Coupling weights are almost uniform

This is one of the most important findings.

For `coupling_k=4`, the theoretical entropy of a uniform distribution is:

- `ln(4) = 1.386294`

Observed in `00021`:

- `weight_entropy ≈ 1.38622`
- `top1_weight ≈ 0.2535`

Those numbers are extremely close to uniform over 4 neighbors.

Interpretation:

- The coupling is not acting like a sharp local assignment
- It is effectively averaging over four same-class neighbors almost uniformly
- This likely destroys the intended advantage of local matching

### 6.4 The discriminator is much weaker than baseline

At `3071.5 kimg`:

Baseline `00007`:

- `real score ≈ 0.7973`
- `fake score ≈ -1.0143`

Coupled `00021`:

- `real score ≈ 0.2271`
- `fake score ≈ -0.0368`

Interpretation:

- The score ordering is still correct
- But the discriminator gap is much smaller
- The new objective seems to flatten or weaken D instead of improving the adversarial geometry

## 7. Current Diagnosis

### 7.1 What has been fixed already

Already fixed in code:

1. class-aware coupling for conditional tasks
2. fallback handling when no same-class neighbor exists
3. detailed coupling stats in `stats.jsonl` and TensorBoard
4. separate reporting of pair and list losses in coupled mode

### 7.2 What is no longer the main issue

These are no longer the leading explanation for failure:

- simple cross-class coupling bug
- missing logs
- uncertainty about whether listwise is even active

### 7.3 Most likely current explanation

The strongest current hypothesis is:

- local-coupled listwise D training on CIFAR-10 is conceptually too strong and too diffuse
- same-class kNN neighborhoods in live discriminator feature space are not providing useful transport structure
- with `k=4`, the coupling weights are nearly uniform, so the model receives a blurred local target rather than an informative matched target
- this weakens the discriminator gap instead of improving it

More aggressive formulation of the same conclusion:

- the main failure is no longer "coupling still has a bug"
- the main failure is "the objective uses live D features to induce a high-entropy local transport target"
- that target then partially rewrites the geometry of the adversarial game instead of acting like a weak prior

This matters because the substrate that should be protected is:

- global pairwise delta adversarial game
- plus zero-centered regularization

The Mescheder-style perspective is that GAN instability is especially dangerous when D keeps the wrong slopes near the data manifold, especially orthogonal slopes. The R3GAN perspective is that relative pairwise games are useful, but need zero-centered regularization to become locally convergent. Put differently:

- the most valuable object in this codebase is not "local coupling"
- it is the stabilized substrate of pairwise delta plus R1/R2

The current coupled local-listwise design is risky because it is not merely regularizing that substrate; it is trying to redefine the pairing geometry using a self-generated, drifting metric.

### 7.4 Why the entropy numbers matter

The `00021` coupling statistics are strong evidence that the current method is not actually selecting informative neighbors.

For `k=4`:

- uniform entropy is `ln(4) = 1.38629`
- observed `weight_entropy ≈ 1.38622`
- observed `top1_weight ≈ 0.2535`

This is extremely close to the high-temperature entropic limit.

Interpretation:

- the coupling is almost not choosing at all
- it is effectively doing same-class class-average smoothing
- on conditional CIFAR-10, class labels already provide the coarse semantic partition, so same-class averaging adds blur but not useful fine structure

This also explains the score-gap collapse:

- baseline `00007` has strong separation (`real ≈ 0.797`, `fake ≈ -1.014` at `3071.5 kimg`)
- coupled `00021` has weak separation (`real ≈ 0.227`, `fake ≈ -0.037`)

This is most naturally read as:

- local listwise is not sharpening the critic
- local listwise is blurring the critic

### 7.5 Why sharpening alone is not the first fix

It is tempting to immediately try:

- hard assignment
- lower temperature
- top-1 coupling

These may still be useful later, but they are not first-order fixes if the metric itself is unreliable.

Reason:

- in the current regime, the method is near-uniform
- making the softmax harder in that regime may only convert "blurred averaging" into "almost random top-1 selection"

So the correct order is:

1. make the metric more trustworthy
2. then sharpen the assignment if needed

Not:

1. sharpen first
2. hope the metric becomes meaningful later

In short:

- the implementation bug is fixed
- the method still underperforms because the objective itself appears wrong for this setting

## 8. Theory: What Is More Likely To Work

Given the current evidence, the most plausible design principles are:

### 8.1 Keep the main adversarial game simple

The main real/fake game should remain the global pairwise delta objective.

Reason:

- this is the part that is already stable
- it produces strong score separation
- the new failures happen when local coupled structure becomes too influential

### 8.2 Treat extra ranking structure as a weak D-side prior

Additional ranking should probably be:

- D-only
- auxiliary
- low weight
- not allowed to redefine the main pairing geometry

This aligns better with the success pattern of the historical path-rank run.

### 8.3 If local coupling is used, it must be sharp

Possible requirements:

- smaller `k`, such as `1` or `2`
- explicit coupling temperature
- hard top-1 assignment
- delayed start after D features become semantically meaningful
- preferably use EMA D features before trying live D features
- possibly use a separate semantic encoder if EMA still fails

The current `k=4` softmax coupling is too close to uniform.

Important caveat:

- sharpening is only meaningful after the metric becomes trustworthy
- if the metric is not trustworthy, sharpening can collapse into noisy or near-random anchor selection

### 8.4 Conditional tasks probably require stricter structure

For CIFAR-10 conditional training, local structure probably needs:

- same-class constraint
- maybe class-conditional normalization or per-class neighbor pools
- maybe anchor-based local ranking rather than weighted multi-real averaging

### 8.5 Late-onset local priors may be safer than early coupling

Online D features are noisy early in training.

This suggests:

- warm up with pure R3GAN first
- only later turn on local or path priors
- or schedule them much more slowly

### 8.6 Anchor selection is safer than local transport

The local structure should probably be downgraded from:

- "a multi-real soft transport plan that changes who each fake is responsible to"

to:

- "a trusted anchor-selection mechanism that only chooses a plausible direction"

This is a major conceptual shift:

- local structure chooses an anchor
- local structure does not define a new adversarial game
- the regularizer then acts only along a 1D fake-to-anchor path

### 8.7 Anchor-Path-Rank is the most promising next design

The currently most compelling design is:

1. Keep the main game unchanged:
   - global pairwise delta for G and D
   - keep R1/R2 as the stability substrate
2. Use local structure only for anchor selection:
   - top-1 same-class anchor
   - anchor chosen using EMA discriminator features first
   - optionally add a confidence gate
3. Apply only a weak D-side monotonic prior:
   - path-rank or monotonicity along the fake-to-anchor interpolation path
   - no local gradient to G
   - low weight

This is appealing because:

- local information only selects a direction
- it does not rewrite the main real/fake pairing geometry
- the actual prior is 1D, interpretable, and low-entropy
- it is much closer to a geometric prior than to a second adversarial game

## 9. Path-Rank Command: Semantics And OOM

The command under discussion was:

```bash
CUDA_VISIBLE_DEVICES=0,1 python /workspace/R3GAN/train.py \
  --outdir=/workspace/training-runs \
  --data=/workspace/datasets/cifar10.zip \
  --gpus=2 --batch=512 --mirror=1 --aug=1 --cond=1 \
  --preset=CIFAR10 \
  --adv-loss-type=softmargin --adv-margin=1.0 --lambda-adv=1.0 \
  --rank-loss=1 --rank-k=3 --rank-loss-type=pairwise_hinge --lambda-rank=0.1 \
  --metrics=fid50k_full --tick=1 --snap=200 \
  --desc=softmargin_m1_pathrank_hinge_k3
```

### 9.1 What this command really means in current code

This is a legacy CLI spelling. In the current code it maps to:

- `lambda_pair = 1.0`
- `pair_margin = 1.0`
- `lambda_list = 0.0`
- `path_rank_reg = True`
- `path_rank_k = 3`
- `path_rank_loss_type = pairwise_hinge`
- `lambda_path_rank = 0.1`
- `path_rank_mode = intrpl`
- `path_rank_alpha_dist = linear`
- `path_rank_margin = 1.0`
- `path_rank_score_reg = 0.0`
- `coupling_k = 0`

So it is not a local-coupled experiment.

It is:

- pairwise soft-margin main loss
- plus a D-only interpolation path-rank prior

### 9.2 Why it OOMs

With:

- `batch=512`
- `gpus=2`
- default `d_batch_gpu = batch / gpus = 256`

The D phase already processes:

- 256 real samples
- 256 fake samples

Then path-rank adds:

- `k=3` rank images per sample
- so `256 * 3 = 768` extra D inputs per GPU

That means the D phase is effectively carrying activations for roughly:

- `256 real + 256 fake + 768 path = 1280` images per GPU

And path-rank is not a no-grad side computation:

- it contributes to the D backward graph
- it coexists with R1 and R2 penalties

That is why the stack trace shows an OOM during backward inside a fused bias-act op:

- the plugin is not the root cause
- it is just where the graph finally requested another large gradient tensor

### 9.3 Immediate mitigation

The most direct mitigation is to lower D microbatch using the existing CLI:

- `--d-batch-gpu=128`
- or `--d-batch-gpu=64`

The code already supports this.

Additional mitigation:

- lower `path_rank_k` from `3` to `2`
- optionally set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 10. Commands Of Interest

### 10.1 Original problematic coupled run

```bash
CUDA_VISIBLE_DEVICES=0,1 python /workspace/R3GAN/train.py \
  --outdir=/workspace/training-runs \
  --data=/workspace/datasets/cifar10.zip \
  --gpus=2 --batch=512 --mirror=1 --aug=1 --cond=1 \
  --preset=CIFAR10 \
  --lambda-pair=1.0 --pair-margin=1.0 \
  --lambda-list=0.5 --list-tau=0.07 \
  --coupling-k=4 \
  --lambda-list-d=0.5 --lambda-list-g=0.1 \
  --metrics=fid50k_full --tick=1 --snap=200 \
  --desc=coupled4_pair1_m1_listD05_listG01_tau007
```

### 10.2 Revised class-aware coupled run

```bash
CUDA_VISIBLE_DEVICES=0,1 python /workspace/R3GAN/train.py \
  --outdir=/workspace/training-runs \
  --data=/workspace/datasets/cifar10.zip \
  --gpus=2 --batch=512 --mirror=1 --aug=1 --cond=1 \
  --preset=CIFAR10 \
  --lambda-pair=1.0 --pair-margin=0.0 \
  --list-tau=0.07 \
  --coupling-k=4 \
  --lambda-list-d=0.5 --lambda-list-g=0.0 \
  --metrics=fid50k_full --tick=1 --snap=200 \
  --desc=coupled4_classaware_pair1_m0_listD05_listG0_tau007
```

### 10.3 Safer path-rank command to avoid OOM

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 python /workspace/R3GAN/train.py \
  --outdir=/workspace/training-runs \
  --data=/workspace/datasets/cifar10.zip \
  --gpus=2 --batch=512 --d-batch-gpu=128 --mirror=1 --aug=1 --cond=1 \
  --preset=CIFAR10 \
  --lambda-pair=1.0 --pair-margin=1.0 \
  --path-rank-reg=1 --path-rank-k=3 --path-rank-loss-type=pairwise_hinge \
  --lambda-path-rank=0.1 --path-rank-mode=intrpl --path-rank-alpha-dist=linear \
  --path-rank-margin=1.0 \
  --metrics=fid50k_full --tick=1 --snap=200 \
  --desc=pair1_m1_pathrank_hinge_k3_dbgpu128
```

## 11. Recommended Next Experiments

If the goal is to find something that is genuinely more likely to work, the recommended order is:

1. Most conservative path-rank run:
   - `lambda_pair=1`
   - `pair_margin=0`
   - `lambda_path_rank=0.05`
   - `path_rank_k=2`
   - `d_batch_gpu=128`
2. Anchor-Path-Rank:
   - local structure only selects a same-class top-1 anchor
   - anchor comes from EMA discriminator features
   - only high-confidence anchors activate the prior
   - the prior is weak D-only path-rank / monotonicity
3. Two decisive ablations:
   - `EMA top-1 anchor` vs `live D top-1 anchor`
   - `EMA top-1 anchor` vs `random same-class anchor`

Avoid for now:

- `coupling_k=4` with nearly uniform weights
- large local D-list weights like `0.5`
- G-side local listwise before D-side behavior is validated
- relying on sharpened live-D coupling before metric quality is established

## 12. Critical Diagnostics To Add

These diagnostics would immediately tell whether the local prior is helping or fighting the main game:

1. `coupling/top1_minus_top2`
2. `coupling/temporal_jaccard`
3. `aux_vs_pair_grad_cosine`
4. `local_active_ratio`

Interpretation:

- `top1_minus_top2` measures confidence margin of the anchor choice
- `temporal_jaccard` measures whether local neighborhoods are stable across nearby steps
- `aux_vs_pair_grad_cosine` measures whether the auxiliary prior is aligned with or opposed to the main pairwise D gradient
- `local_active_ratio` measures how often the confidence gate is actually allowing the prior to fire

Of these, `aux_vs_pair_grad_cosine` is arguably the most important:

- if it stays negative for long periods, the auxiliary prior is systematically fighting the main game rather than regularizing it

## 13. Open Questions For External Discussion

These are the highest-value questions to discuss with another model:

1. Is local coupling from the live discriminator feature space fundamentally too unstable early in GAN training?
2. If the metric is untrustworthy, does sharpening help at all, or does it merely convert blur into noisy top-1 anchors?
3. Should the main adversarial game remain strictly diagonal/global, with local structure allowed only to choose anchors for a D-side prior?
4. Would EMA discriminator features already be stable enough to separate "metric drift problem" from "objective design problem"?
5. Is same-class kNN in CIFAR-10 inherently too coarse, so that the local geometry collapses into class-average smoothing?
6. Is path-rank a better inductive bias than local-coupled listwise because it constrains score monotonicity without redefining the real/fake transport structure?
7. Should local structure be treated as anchor selection only, rather than as a soft transport plan?

## 14. Bottom Line

The current state is:

- the framework implementation is substantially richer and now instrumented well enough to reason about
- the main conditional coupling correctness bug has been fixed
- the coupled local-listwise idea still fails on CIFAR-10 even after the fix
- the strongest evidence points to an objective-design problem, not a remaining implementation bug
- the more precise diagnosis is that local coupling currently acts like a high-entropy, self-induced local transport target from live D features
- this target appears to blur the critic instead of sharpening it
- the best next direction is not "stronger coupling", but "weaker, one-way local geometry"
- concretely: keep global pairwise delta unchanged, let local structure only choose anchors, and apply only a weak D-only path-rank prior along the fake-to-anchor direction
