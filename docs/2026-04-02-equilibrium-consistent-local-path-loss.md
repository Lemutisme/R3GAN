# Equilibrium-Consistent Local Path Loss (ECLP)

Date: 2026-04-02

Status: reviewed design note / not yet implemented

This note refines the earlier `local-coupled RankGAN` direction using the later evidence in:

- [docs/design.md](./design.md)
- [docs/2026-03-10-rankgan-status-summary.md](./2026-03-10-rankgan-status-summary.md)
- [docs/2026-03-13-rank-field-modeling-note.md](./2026-03-13-rank-field-modeling-note.md)

The main update is a narrowing of scope:

- keep the main adversarial game as the stable `delta-centric pairwise core + R1/R2`
- use `local coupling` to choose a plausible local path or anchor
- use the path object only inside a weak `D-side auxiliary geometry prior`
- schedule that prior on in early/mid training and back off late, so it does not fight the flat equilibrium

This note therefore partially supersedes the more aggressive suggestion of letting local structure redefine the main pairing geometry.

## 1. Review Summary

The current evidence points to four conclusions.

### 1.1 The protected substrate is still pairwise delta plus R1/R2

The most reliable object in this codebase is the stabilized adversarial substrate:

\[
\Delta_j = s_\psi(a(x_j^r)) - s_\psi(a(\hat x_j)),
\]

together with zero-centered regularization.

This is the part that already has the clearest stability story and the strongest empirical support. The main failure mode of the more aggressive local-coupled variants was not "ranking is useless", but "live local structure became strong enough to partially rewrite the adversarial geometry".

### 1.2 Arbitrary real-fake interpolation is too biased to be a final objective

The older `path_rank_reg` was useful because it added directional supervision in the interior between endpoints. But if a loss keeps requiring

\[
s(x_r) > s(x_\lambda) > s(x_f)
\]

for arbitrary `real-fake` pairs even near convergence, then it conflicts with the desired flat equilibrium at `p_\theta = p_D`.

So the right interpretation is:

- not equilibrium-defining main game
- but early/mid-training transient geometry prior

### 1.3 Local coupling is still valuable, but first as a neighborhood oracle

The `2026-03-10` review showed that high-entropy coupling from live discriminator features can become too blurry and can weaken score separation. That does not mean local structure is useless. It means local structure should be demoted from:

- "new adversarial geometry"

to:

- "trusted neighborhood or anchor selector for a weak prior"

This is a major conceptual change.

### 1.4 The next clean object is local path consistency, not global total rank

The next object to try should not be "a globally correct rank for every real/fake pair". That object is not identifiable at equilibrium. The next object should be:

- local
- coupling-conditioned
- one-way
- equilibrium-consistent

That is what this note calls `ECLP`.

## 2. Proposed Object

### 2.1 Ingredients

Let `s_\psi(x)` be the scalar critic.

Let clean features be

\[
e_i^r = \operatorname{sg}(\operatorname{norm}(\phi(x_i^r))), \qquad
e_j^f = \operatorname{sg}(\operatorname{norm}(\phi(\hat x_j))).
\]

Using clean features, build a sparse neighborhood and coupling for each fake:

\[
\mathcal N(j) \subset \{1,\dots,B\}, \qquad
\pi_{ij} \ge 0, \qquad
\sum_{i \in \mathcal N(j)} \pi_{ij} = 1.
\]

For v1, `kNN + softmax` or `top-1` anchor selection is preferred over a high-entropy transport plan.

If `kNN + softmax` is used:

\[
\mathcal N(j) = \operatorname{TopK}_i \langle e_i^r, e_j^f \rangle,
\]

\[
\pi_{ij}
=
\frac{\exp(\beta \langle e_i^r, e_j^f \rangle)}
{\sum_{i' \in \mathcal N(j)} \exp(\beta \langle e_{i'}^r, e_j^f \rangle)}.
\]

### 2.2 Path Target

Define a local target `T_j` from the neighborhood of fake `j`.

Recommended low-risk v1:

\[
i^\*(j) = \arg\max_{i \in \mathcal N(j)} \pi_{ij}, \qquad
T_j = x_{i^\*(j)}^r.
\]

Higher-risk later variant:

\[
T_j = \sum_{i \in \mathcal N(j)} \pi_{ij} x_i^r.
\]

Then define the local path vector:

\[
v_j = T_j - \hat x_j.
\]

This is the key change relative to old path-rank:

- the path is no longer an arbitrary real-fake chord
- the path is induced by a local frontier estimate

### 2.3 Confidence Gate

Because the `2026-03-10` review showed that coupling entropy can become almost uniform, the local path prior should include a confidence gate.

Define

\[
H_j = - \sum_{i \in \mathcal N(j)} \pi_{ij} \log(\pi_{ij} + \varepsilon),
\]

\[
c_j
=
\operatorname{sg}\!\left(
1 - \frac{H_j}{\log |\mathcal N(j)|}
\right).
\]

This is near zero when the neighborhood is close to uniform, and near one when the coupling is sharp.

Also define a simple path-magnitude gate:

\[
r_j
=
\operatorname{sg}\!\left(
\frac{\|v_j\|_2}{\|v_j\|_2 + \tau_v}
\right).
\]

The final gate is

\[
\alpha_j = c_j \cdot r_j.
\]

Interpretation:

- if the neighborhood is untrustworthy, do not force geometry
- if the local target is already too close, do not force slope

## 3. Main Game And Auxiliary Geometry

### 3.1 Main Adversarial Game

The main game remains the standard pairwise delta objective:

\[
\mathcal L_D^{\mathrm{pair}}
=
\frac1B \sum_{j=1}^B
\operatorname{softplus}\!\left(
m - \bigl[s_\psi(a(x_j^r)) - s_\psi(a(\hat x_j))\bigr]
\right),
\]

\[
\mathcal L_G^{\mathrm{pair}}
=
\frac1B \sum_{j=1}^B
\operatorname{softplus}\!\left(
m + \bigl[s_\psi(a(x_j^r)) - s_\psi(a(\hat x_j))\bigr]
\right).
\]

This is the protected substrate.

Local structure is not allowed to redefine this main game in the first implementation of `ECLP`.

### 3.2 Local Clean Gap

Using clean views only, define the local average critic gap:

\[
\bar\Delta_j^{\mathrm{loc}}
=
\sum_{i \in \mathcal N(j)}
\pi_{ij}
\bigl[s_\psi(x_i^r) - s_\psi(\hat x_j)\bigr].
\]

For the `top-1 anchor` variant, this reduces to:

\[
\bar\Delta_j^{\mathrm{loc}}
=
s_\psi(T_j) - s_\psi(\hat x_j).
\]

### 3.3 Equilibrium-Consistent Local Path Loss

Define the local path-consistency term:

\[
\mathcal L_{\mathrm{ECLP}}^D
=
\frac1B \sum_{j=1}^B
\alpha_j\,
\rho\!\left(
\bar\Delta_j^{\mathrm{loc}}
-
\nabla_x s_\psi(\hat x_j)^\top v_j
\right),
\]

where `rho` is a smooth robust penalty, for example smooth L1 or Charbonnier:

\[
\rho(z) = \sqrt{z^2 + \varepsilon^2} - \varepsilon.
\]

Interpretation:

- `\bar\Delta_j^{loc}` says how much better the local frontier currently scores than the fake
- `\nabla s(\hat x_j)^\top v_j` says how much first-order improvement the critic predicts along the local path
- the loss asks these two local objects to agree

This is not asking for a global total order. It is asking for local consistency between:

- comparison gap
- and local improvement direction

### 3.4 Optional One-Sided Monotonicity

If a weaker directional term is preferred, add:

\[
\mathcal L_{\mathrm{mono}}^D
=
\frac1B \sum_{j=1}^B
\alpha_j
\operatorname{softplus}\!\left(
m_p \alpha_j
-
\nabla_x s_\psi(\hat x_j)^\top
\frac{v_j}{\|v_j\|_2 + \varepsilon}
\right).
\]

But this should remain optional. The main proposed object is `gap consistency`, not hard monotonic ranking everywhere.

### 3.5 Total Objective

\[
\mathcal L_D
=
\mathcal L_D^{\mathrm{pair}}
+ \lambda_{\mathrm{list}}^D(t)\,\mathcal L_D^{\mathrm{list,loc}}
+ \lambda_{\mathrm{geo}}(t)\,\mathcal L_{\mathrm{ECLP}}^D
+ \frac{\gamma_1}{2}R_1
+ \frac{\gamma_2}{2}R_2,
\]

\[
\mathcal L_G
=
\mathcal L_G^{\mathrm{pair}}
+ \lambda_{\mathrm{list}}^G(t)\,\mathcal L_G^{\mathrm{list,loc}},
\qquad
\lambda_{\mathrm{list}}^G(t) \ll \lambda_{\mathrm{list}}^D(t).
\]

For the first implementation, an even safer variant is:

- `lambda_list^D = 0`
- `lambda_list^G = 0`
- only `pairwise core + ECLP + R1/R2`

## 4. Why This Is Equilibrium-Consistent

The key requirement near convergence is not "maintain a visible path slope forever". It is:

- when `p_\theta = p_D`, the critic should become locally flat
- the generator should stop receiving a spurious push

Under the intended equilibrium,

\[
\nabla_x s_\psi(\hat x_j) \to 0,
\]

and the local clean gap should also vanish in expectation because fake and real come from the same local distribution:

\[
\bar\Delta_j^{\mathrm{loc}} \to 0.
\]

So in the flat-equilibrium regime,

\[
\mathcal L_{\mathrm{ECLP}}^D \to 0.
\]

This is the sense in which the loss is `equilibrium-consistent`:

- it does not insist on a global total order that should not exist at equilibrium
- it only asks local gap and local field to agree when there is still a meaningful local discrepancy

## 5. Why This Is Safer Than Old Path-Rank

Relative to interpolation-based path-rank, `ECLP` changes four things.

### 5.1 Path source

Old:

- arbitrary real-fake interpolation chord

New:

- local coupling chooses a plausible local target first

### 5.2 Bias type

Old:

- strong structural bias on every chosen pair

New:

- conditional bias only where the local metric is confident

### 5.3 Role in the game

Old:

- easy to over-interpret as an alternative main objective

New:

- explicitly auxiliary and D-only

### 5.4 Late-training behavior

Old:

- risks conflicting with flat equilibrium if kept active forever

New:

- built from a term that should naturally shrink as local gap and local slope both vanish

## 6. Recommended Training Curriculum

The geometry prior should not be active from step zero and should not remain strong forever.

### Phase 0: Bootstrap

Suggested range:

- first `10%` to `15%` of training

Objective:

- `pairwise delta + R1/R2`
- no local geometry prior

Reason:

- early discriminator features are noisy
- local coupling quality is not yet trustworthy

### Phase 1: Neighborhood warmup

Suggested range:

- roughly `15%` to `35%`

Objective:

- build clean local coupling
- log coupling quality
- optionally keep `lambda_geo = 0` or very small

Reason:

- first make the metric trustworthy
- only then let it shape the critic

### Phase 2: Active geometry shaping

Suggested range:

- roughly `35%` to `75%`

Objective:

- keep pairwise core unchanged
- turn on `lambda_geo(t)` on the D side
- optionally allow a small `lambda_list^D(t)`
- keep `lambda_list^G(t)` at `0` or near `0`

Suggested v1 settings:

- `top-1` or `k=2`
- same-class constraint on CIFAR-10
- `lambda_geo` small, for example `0.02 -> 0.05`

### Phase 3: Handoff to flat equilibrium

Suggested range:

- last `25%`

Objective:

- cosine decay `lambda_geo(t)` toward `0`
- reduce or remove `lambda_list^D(t)`
- return emphasis to `pairwise core + R1/R2`

Reason:

- late training should prioritize stable flat equilibrium over continued path shaping

## 7. Practical Safeguards

### 7.1 Use clean features and clean scores

The geometry prior should be built on non-augmented views, just like the current path-rank auxiliary.

### 7.2 Prefer anchor selection before soft transport

Given the earlier entropy failure, the safer order is:

1. trustworthy anchor
2. then maybe weighted multi-real path

Not the other way around.

### 7.3 Delay start

Local geometry should not be active at the beginning of training.

### 7.4 Add confidence gating

If coupling entropy is near uniform, the regularizer should effectively switch off.

### 7.5 Consider EMA features before live features

If live discriminator features remain unstable, an EMA feature extractor or lagged feature bank is a safer source for neighborhood construction.

### 7.6 For conditional CIFAR-10, keep class structure strict

At minimum:

- same-class neighborhood
- per-class anchor selection

This keeps local targets semantically plausible.

## 8. Mapping To Current Code

This note is intentionally close to the current implementation vocabulary.

- `pairwise_delta()` remains the main adversarial primitive.
- `build_local_coupling()` remains the neighborhood builder.
- existing coupling stats such as `weight_entropy`, `top1_weight`, and `same_class_mass` are directly useful for confidence gating.
- `path_rank_reg` should be conceptually demoted in favor of `ECLP`, not promoted into a stronger main game.
- existing `lambda_list_d` / `lambda_list_g` scheduling hooks suggest the natural place to later add a `lambda_geo` scheduler.

## 9. What This Note Does And Does Not Claim

This note does claim:

- the best next design is a weak, local, D-side geometry prior
- the main adversarial game should stay simple
- local coupling should first act as a neighborhood oracle, not a new game definition

This note does not claim:

- that global total rank is identifiable
- that barycentric multi-real targets are already safe
- that local-coupled main-game replacement is currently justified by evidence

## 10. Immediate Next-Step Recommendation

The cleanest next experiment is not "full local-coupled RankGAN v2". It is:

1. keep the main pairwise delta game unchanged
2. use same-class clean-feature coupling only to select a local anchor
3. apply a weak D-only `ECLP` prior along the fake-to-anchor direction
4. turn it on after warmup
5. decay it late

This preserves the strongest parts of the current system while upgrading path-rank from:

- arbitrary chord regularization

to:

- confidence-gated local path consistency

That is the most conservative way to test whether the `1.90`-style benefit is really coming from better critic geometry rather than from a brittle path bias.
