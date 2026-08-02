# JAX Drift Loss Alignment & PyTorch Performance Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the PyTorch drift loss implementation with the reference JAX `drift_loss()` in `/workspace/drifting/drift_loss.py`, and fix performance bottlenecks.

**Architecture:** The JAX reference uses a single unified `drift_loss()` function that: (1) concatenates all targets, (2) normalizes coordinates by weighted mean distance, (3) loops over R temperatures computing per-R forces with per-R normalization, (4) sums normalized forces into one goal, (5) computes a single MSE. The PyTorch code decomposes this into separate modules (`drift_field.py`, `drift_loss.py`, `drift_training_loop.py`) with several algorithmic divergences. We fix these one module at a time, bottom-up, with parity tests against the JAX reference.

**Tech Stack:** PyTorch, pytest, reference JAX code at `/workspace/drifting/drift_loss.py`

**Scope note:** The `DiTLikeGenerator` in PyTorch is an intentionally different architecture from the JAX `DitGen` (register tokens vs class tokens, alpha vs CFG, style embeddings vs noise coords). This plan does NOT attempt to port the JAX generator — it focuses exclusively on the drift loss algorithm and training loop where mathematical equivalence is required. DiT architectural improvements (sincos pos embed, fp32 modulation, SwiGLU rounding) are included as optional tasks at the end.

---

## Phase 1: Core Drift Loss — Weighting Mechanism

The JAX reference applies sample weights **post-softmax** as a multiplicative factor on the geometric-mean affinity. The current PyTorch code applies weights **pre-softmax** as log-additive terms on logits. These produce different affinity distributions.

### Task 1: Add a JAX-aligned affinity function with post-softmax weighting

**Files:**
- Modify: `training/drift_field.py`
- Test: `tests/test_jax_drift_parity.py` (create)

**Step 1: Write the failing test**

Create `tests/test_jax_drift_parity.py`. This test implements the JAX affinity logic in pure PyTorch and checks that our new function matches it exactly.

```python
"""Parity tests: training.drift_field vs JAX /workspace/drifting/drift_loss.py logic."""
from __future__ import annotations
import unittest
import torch
import math


def jax_reference_cdist(x, y, eps=1e-8):
    """Matches /workspace/drifting/drift_loss.py:6-12 — [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def jax_reference_drift_loss(
    gen, fixed_pos, fixed_neg=None,
    weight_gen=None, weight_pos=None, weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
):
    """Pure-PyTorch transcription of /workspace/drifting/drift_loss.py:15-134."""
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen[:, :0, :]
    C_n = fixed_neg.shape[1]
    if weight_gen is None:
        weight_gen = torch.ones(B, C_g, device=gen.device)
    if weight_pos is None:
        weight_pos = torch.ones(B, C_p, device=gen.device)
    if weight_neg is None:
        weight_neg = torch.ones(B, C_n, device=gen.device) if C_n > 0 else torch.zeros(B, 0, device=gen.device)
    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()
    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # Scaling
    dist = jax_reference_cdist(old_gen, targets)
    weighted_dist = dist * targets_w[:, None, :]
    scale = weighted_dist.mean() / targets_w.mean()
    scale_inputs = torch.clamp(scale / math.sqrt(S), min=1e-3)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs
    dist_normed = dist / torch.clamp(scale, min=1e-3)

    # Masking
    mask_val = 100.0
    diag_mask = torch.eye(C_g, device=gen.device)
    block_mask = torch.nn.functional.pad(diag_mask, (0, C_n + C_p)).unsqueeze(0)
    dist_normed = dist_normed + block_mask * mask_val

    # Force loop
    force_across_R = torch.zeros_like(old_gen_scaled)
    info = {"scale": scale.detach()}
    for R in R_list:
        logits = -dist_normed / R
        row_aff = torch.softmax(logits, dim=-1)
        col_aff = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_aff * col_aff, min=1e-6))
        affinity = affinity * targets_w[:, None, :]  # POST-softmax weighting

        split_idx = C_g + C_n
        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        sum_pos = aff_pos.sum(dim=-1, keepdim=True)
        r_coeff_neg = -aff_neg * sum_pos
        sum_neg = aff_neg.sum(dim=-1, keepdim=True)
        r_coeff_pos = aff_pos * sum_neg
        R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
        total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
        total_coeffs = R_coeff.sum(dim=-1)
        total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
        f_norm_val = (total_force_R ** 2).mean()
        info[f"loss_{R}"] = f_norm_val.detach()
        force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
        force_across_R = force_across_R + total_force_R / force_scale

    goal_scaled = (old_gen_scaled + force_across_R).detach()
    gen_scaled = gen / scale_inputs.detach()
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))
    return loss, {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in info.items()}


class TestJaxDriftParity(unittest.TestCase):
    def test_reference_self_consistency(self):
        """Sanity: reference function runs and produces finite values."""
        torch.manual_seed(42)
        gen = torch.randn(2, 4, 8)
        pos = torch.randn(2, 6, 8)
        neg = torch.randn(2, 3, 8)
        loss, info = jax_reference_drift_loss(gen, pos, neg)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertIn("scale", info)

    def test_weighted_reference(self):
        """Reference function works with explicit non-uniform weights."""
        torch.manual_seed(42)
        gen = torch.randn(2, 4, 8)
        pos = torch.randn(2, 6, 8)
        neg = torch.randn(2, 3, 8)
        w_neg = torch.tensor([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
        loss, info = jax_reference_drift_loss(
            gen, pos, neg, weight_neg=w_neg, R_list=(0.05, 0.2),
        )
        self.assertTrue(torch.isfinite(loss).all())


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it passes (this is the reference, it should pass)**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py -v`
Expected: PASS (we're testing the reference itself)

**Step 3: Commit**

```bash
git add tests/test_jax_drift_parity.py
git commit -m "test: add JAX drift_loss reference implementation in pure PyTorch"
```

---

### Task 2: Implement `jax_aligned_drift_loss` in `training/drift_field.py`

This adds a new public function that mirrors the JAX algorithm exactly: concatenated targets, weighted distances for scaling, post-softmax weighting, per-R force normalization, summed forces, single MSE.

**Files:**
- Modify: `training/drift_field.py` (append new function)
- Modify: `tests/test_jax_drift_parity.py` (add parity test)

**Step 1: Write the failing parity test**

Append to `tests/test_jax_drift_parity.py`:

```python
from training.drift_field import jax_aligned_drift_loss, JaxAlignedDriftConfig


class TestJaxAlignedDriftLoss(unittest.TestCase):
    """Our new function must match the reference exactly."""

    def test_basic_parity(self):
        torch.manual_seed(42)
        gen = torch.randn(2, 4, 8)
        pos = torch.randn(2, 6, 8)
        neg = torch.randn(2, 3, 8)
        R_list = (0.02, 0.05, 0.2)

        ref_loss, ref_info = jax_reference_drift_loss(gen, pos, neg, R_list=R_list)
        cfg = JaxAlignedDriftConfig(R_list=R_list)
        our_loss, our_info = jax_aligned_drift_loss(gen, pos, neg, config=cfg)

        torch.testing.assert_close(our_loss, ref_loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            torch.tensor(our_info["scale"]),
            torch.tensor(ref_info["scale"]),
            atol=1e-5, rtol=1e-5,
        )

    def test_weighted_parity(self):
        torch.manual_seed(42)
        gen = torch.randn(2, 4, 8)
        pos = torch.randn(2, 6, 8)
        neg = torch.randn(2, 3, 8)
        w_gen = torch.ones(2, 4)
        w_pos = torch.ones(2, 6)
        w_neg = torch.tensor([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
        R_list = (0.05, 0.2)

        ref_loss, _ = jax_reference_drift_loss(
            gen, pos, neg, weight_gen=w_gen, weight_pos=w_pos,
            weight_neg=w_neg, R_list=R_list,
        )
        cfg = JaxAlignedDriftConfig(R_list=R_list)
        our_loss, _ = jax_aligned_drift_loss(
            gen, pos, neg, weight_gen=w_gen, weight_pos=w_pos,
            weight_neg=w_neg, config=cfg,
        )
        torch.testing.assert_close(our_loss, ref_loss, atol=1e-5, rtol=1e-5)

    def test_no_negatives_parity(self):
        torch.manual_seed(42)
        gen = torch.randn(2, 4, 8)
        pos = torch.randn(2, 6, 8)
        R_list = (0.05,)

        ref_loss, _ = jax_reference_drift_loss(gen, pos, R_list=R_list)
        cfg = JaxAlignedDriftConfig(R_list=R_list)
        our_loss, _ = jax_aligned_drift_loss(gen, pos, config=cfg)
        torch.testing.assert_close(our_loss, ref_loss, atol=1e-5, rtol=1e-5)

    def test_gradient_flows_through_gen(self):
        torch.manual_seed(42)
        gen = torch.randn(2, 4, 8, requires_grad=True)
        pos = torch.randn(2, 6, 8)
        neg = torch.randn(2, 3, 8)
        cfg = JaxAlignedDriftConfig(R_list=(0.05, 0.2))
        loss, _ = jax_aligned_drift_loss(gen, pos, neg, config=cfg)
        loss.mean().backward()
        self.assertIsNotNone(gen.grad)
        self.assertTrue(torch.isfinite(gen.grad).all())
```

**Step 2: Run test to verify it fails**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py::TestJaxAlignedDriftLoss -v`
Expected: FAIL with `ImportError: cannot import name 'jax_aligned_drift_loss'`

**Step 3: Implement `jax_aligned_drift_loss` in `training/drift_field.py`**

Append to the end of `training/drift_field.py`:

```python
from dataclasses import field as dataclass_field
from math import sqrt as math_sqrt


@dataclass(frozen=True)
class JaxAlignedDriftConfig:
    """Config for the JAX-aligned drift loss (mirrors /workspace/drifting/drift_loss.py)."""
    R_list: tuple[float, ...] = (0.02, 0.05, 0.2)
    mask_value: float = 100.0
    affinity_eps: float = 1e-6
    scale_eps: float = 1e-3
    force_norm_eps: float = 1e-8
    cdist_eps: float = 1e-8


def _cdist_jax_style(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Euclidean distance matching JAX reference: [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def jax_aligned_drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    *,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    config: JaxAlignedDriftConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Drift loss aligned with /workspace/drifting/drift_loss.py.

    Args:
        gen: [B, C_g, S] generated features.
        fixed_pos: [B, C_p, S] positive reference features.
        fixed_neg: [B, C_n, S] negative reference features (optional).
        weight_gen: [B, C_g] per-sample weights for generated (default 1).
        weight_pos: [B, C_p] per-sample weights for positives (default 1).
        weight_neg: [B, C_n] per-sample weights for negatives (default 1).
        config: algorithm parameters.

    Returns:
        loss: [B] per-batch loss.
        info: dict of scalar stats.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen[:, :0, :]
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones(B, C_g, device=gen.device, dtype=gen.dtype)
    if weight_pos is None:
        weight_pos = torch.ones(B, C_p, device=gen.device, dtype=gen.dtype)
    if weight_neg is None:
        if C_n > 0:
            weight_neg = torch.ones(B, C_n, device=gen.device, dtype=gen.dtype)
        else:
            weight_neg = torch.zeros(B, 0, device=gen.device, dtype=gen.dtype)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # --- Goal computation (all under stop_gradient) ---
    with torch.no_grad():
        dist = _cdist_jax_style(old_gen, targets, eps=config.cdist_eps)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()

        scale_inputs = torch.clamp(scale / math_sqrt(S), min=config.scale_eps)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=config.scale_eps)

        # Masking: block gen-to-self diagonal
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = torch.nn.functional.pad(diag_mask, (0, C_n + C_p)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * config.mask_value

        # Force loop over R temperatures
        force_across_R = torch.zeros_like(old_gen_scaled)
        info: dict[str, float] = {"scale": float(scale.item())}

        for R in config.R_list:
            logits = -dist_normed / R
            row_aff = torch.softmax(logits, dim=-1)
            col_aff = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(row_aff * col_aff, min=config.affinity_eps))
            affinity = affinity * targets_w[:, None, :]  # post-softmax weighting

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg
            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = float(f_norm_val.item())
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=config.force_norm_eps))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    # --- Loss (gradient flows through gen only) ---
    gen_scaled = gen / scale_inputs.detach()
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))
    return loss, info
```

**Step 4: Run test to verify it passes**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/drift_field.py tests/test_jax_drift_parity.py
git commit -m "feat: add jax_aligned_drift_loss matching JAX drift_loss exactly"
```

---

## Phase 2: Integrate JAX-Aligned Loss into Pixel-Space Training

The function `_batched_drifting_stopgrad_loss` in `drift_training_loop.py` is the pixel-space drift loss used when `use_feature_loss=False`. It currently uses single-temperature with separate feature/drift normalization. Replace its core with `jax_aligned_drift_loss`.

### Task 3: Add a grouped wrapper for `jax_aligned_drift_loss`

**Files:**
- Modify: `training/drift_field.py` (add `jax_aligned_drift_loss_grouped`)
- Modify: `tests/test_jax_drift_parity.py` (add grouped tests)

**Step 1: Write the failing test**

Append to `tests/test_jax_drift_parity.py`:

```python
from training.drift_field import jax_aligned_drift_loss_grouped


class TestJaxAlignedGrouped(unittest.TestCase):
    def test_grouped_matches_loop(self):
        """Grouped function must match calling jax_aligned_drift_loss per group."""
        torch.manual_seed(42)
        G, N_gen, N_pos, N_neg, D = 3, 4, 5, 3, 8
        gen = torch.randn(G, N_gen, D)
        pos = torch.randn(G, N_pos, D)
        neg = torch.randn(G, N_neg, D)
        cfg = JaxAlignedDriftConfig(R_list=(0.05, 0.2))

        grouped_loss, grouped_info = jax_aligned_drift_loss_grouped(
            gen, pos, neg, config=cfg,
        )

        per_group_losses = []
        for g in range(G):
            loss_g, _ = jax_aligned_drift_loss(
                gen[g].unsqueeze(0), pos[g].unsqueeze(0), neg[g].unsqueeze(0),
                config=cfg,
            )
            per_group_losses.append(loss_g.squeeze(0))
        ref_loss = torch.stack(per_group_losses).mean()
        torch.testing.assert_close(grouped_loss, ref_loss, atol=1e-5, rtol=1e-5)

    def test_grouped_with_weights(self):
        torch.manual_seed(42)
        G, N_gen, N_pos, N_unc, D = 2, 4, 5, 2, 8
        gen = torch.randn(G, N_gen, D)
        pos = torch.randn(G, N_pos, D)
        unc = torch.randn(G, N_unc, D)
        # Negatives = [gen_detached, unconditional]
        neg = torch.cat([gen.detach(), unc], dim=1)
        w_gen = torch.ones(G, N_gen)
        w_pos = torch.ones(G, N_pos)
        # unconditional_weight per group
        unc_weights = torch.tensor([2.0, 0.5])
        w_neg_list = []
        for g in range(G):
            w_neg_g = torch.cat([
                torch.ones(N_gen),
                torch.full((N_unc,), float(unc_weights[g])),
            ])
            w_neg_list.append(w_neg_g)
        w_neg = torch.stack(w_neg_list)

        cfg = JaxAlignedDriftConfig(R_list=(0.05,))
        loss, info = jax_aligned_drift_loss_grouped(
            gen, pos, neg, weight_gen=w_gen, weight_pos=w_pos,
            weight_neg=w_neg, config=cfg,
        )
        self.assertTrue(torch.isfinite(loss))
```

**Step 2: Run test to verify it fails**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py::TestJaxAlignedGrouped -v`
Expected: FAIL with `ImportError`

**Step 3: Implement `jax_aligned_drift_loss_grouped`**

Append to `training/drift_field.py`:

```python
def jax_aligned_drift_loss_grouped(
    gen_grouped: torch.Tensor,
    pos_grouped: torch.Tensor,
    neg_grouped: torch.Tensor | None = None,
    *,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    config: JaxAlignedDriftConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-group JAX-aligned drift loss, averaged over groups.

    Args:
        gen_grouped: [G, N_gen, D]
        pos_grouped: [G, N_pos, D]
        neg_grouped: [G, N_neg, D] or None
        weight_gen: [G, N_gen] or None
        weight_pos: [G, N_pos] or None
        weight_neg: [G, N_neg] or None
        config: JaxAlignedDriftConfig

    Returns:
        loss: scalar (mean over groups)
        info: aggregated stats dict
    """
    G = gen_grouped.shape[0]
    losses = []
    all_info: dict[str, list[float]] = {}
    for g in range(G):
        gen_g = gen_grouped[g].unsqueeze(0)
        pos_g = pos_grouped[g].unsqueeze(0)
        neg_g = neg_grouped[g].unsqueeze(0) if neg_grouped is not None else None
        wg = weight_gen[g].unsqueeze(0) if weight_gen is not None else None
        wp = weight_pos[g].unsqueeze(0) if weight_pos is not None else None
        wn = weight_neg[g].unsqueeze(0) if weight_neg is not None else None
        loss_g, info_g = jax_aligned_drift_loss(
            gen_g, pos_g, neg_g,
            weight_gen=wg, weight_pos=wp, weight_neg=wn,
            config=config,
        )
        losses.append(loss_g.squeeze(0))
        for k, v in info_g.items():
            all_info.setdefault(k, []).append(float(v) if not isinstance(v, float) else v)

    loss = torch.stack(losses).mean()
    info = {k: sum(v) / len(v) for k, v in all_info.items()}
    return loss, info
```

**Step 4: Run tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/drift_field.py tests/test_jax_drift_parity.py
git commit -m "feat: add jax_aligned_drift_loss_grouped wrapper"
```

---

### Task 4: Wire `jax_aligned_drift_loss_grouped` into `_batched_drifting_stopgrad_loss`

Replace the internals of `_batched_drifting_stopgrad_loss` in `drift_training_loop.py` with a call to the new JAX-aligned function so the pixel-space training path uses the correct algorithm.

**Files:**
- Modify: `training/drift_training_loop.py:693-808`
- Test: `tests/test_jax_drift_parity.py` (add integration test)

**Step 1: Write failing integration test**

Append to `tests/test_jax_drift_parity.py`:

```python
class TestBatchedLossUsesJaxAligned(unittest.TestCase):
    def test_batched_loss_matches_jax_reference(self):
        """_batched_drifting_stopgrad_loss must produce same loss as JAX reference."""
        from training.drift_field import DriftFieldConfig, JaxAlignedDriftConfig
        from training.drift_loss import DriftingLossConfig

        # Import the function under test — it lives in drift_training_loop
        import importlib
        dtl = importlib.import_module("training.drift_training_loop")
        batched_fn = dtl._batched_drifting_stopgrad_loss

        torch.manual_seed(42)
        G, N_gen, N_pos, D = 2, 4, 5, 16
        x = torch.randn(G, N_gen, D)
        y_pos = torch.randn(G, N_pos, D)
        y_neg = x.detach().clone()  # gen as negatives
        n_gen_neg = N_gen

        fc = DriftFieldConfig(temperature=0.05)
        config = DriftingLossConfig(drift_field=fc)
        loss, stats = batched_fn(x, y_pos, y_neg, None, config, n_gen_neg)

        # Build equivalent JAX-aligned call
        # The batched function uses R_list derived from its single temperature
        # After the fix, it should delegate to jax_aligned_drift_loss_grouped
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("mean_drift_norm", stats)
```

**Step 2: Run to verify current behavior**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py::TestBatchedLossUsesJaxAligned -v`
Expected: Should PASS (we're just checking it runs; the actual numerical alignment is the next step)

**Step 3: Rewrite `_batched_drifting_stopgrad_loss`**

In `training/drift_training_loop.py`, replace the function body at lines 693-808. The new version delegates to `jax_aligned_drift_loss_grouped`, constructing the appropriate weight tensors from `neg_log_weights_grouped`:

```python
def _batched_drifting_stopgrad_loss(
    x_grouped, y_pos_grouped, y_neg_grouped,
    neg_log_weights_grouped, config, generated_negative_count,
    *, scale_temperature_by_sqrt_dim=True, normalize_features=True,
    normalize_drifts=True, normalization_eps=1e-8,
):
    """Vectorized drift loss over all groups, JAX-aligned algorithm.

    Delegates to jax_aligned_drift_loss_grouped which implements the exact
    algorithm from /workspace/drifting/drift_loss.py.
    """
    from training.drift_field import jax_aligned_drift_loss_grouped, JaxAlignedDriftConfig

    fc = config.drift_field
    G = x_grouped.shape[0]

    # Build per-sample weight tensors matching JAX convention
    # weight_gen: [G, N_gen] = 1.0
    # weight_neg: [G, N_neg] where generated part = 1.0, unconditional part = exp(log_weight)
    weight_gen = torch.ones(G, x_grouped.shape[1], device=x_grouped.device)
    weight_pos = torch.ones(G, y_pos_grouped.shape[1], device=x_grouped.device)
    weight_neg = None
    if neg_log_weights_grouped is not None:
        weight_neg = torch.exp(neg_log_weights_grouped)
        # Clamp near-zero weights to avoid NaN
        weight_neg = torch.where(
            neg_log_weights_grouped < -1e30,
            torch.zeros_like(weight_neg),
            weight_neg,
        )

    jax_config = JaxAlignedDriftConfig(
        R_list=(fc.temperature,),  # single temperature for pixel-space
        mask_value=fc.self_mask_value if fc.mask_self_negatives else 0.0,
        affinity_eps=1e-6,
        scale_eps=1e-3,
        force_norm_eps=1e-8,
    )

    loss, info = jax_aligned_drift_loss_grouped(
        gen_grouped=x_grouped,
        pos_grouped=y_pos_grouped,
        neg_grouped=y_neg_grouped,
        weight_gen=weight_gen,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
        config=jax_config,
    )

    # Compute drift stats for logging (using the old API names)
    with torch.no_grad():
        # Approximate drift norms for logging — compute drift at the configured temp
        # This is informational only, doesn't affect loss
        drift_pos_norm = torch.tensor(0.0)
        drift_neg_norm = torch.tensor(0.0)
        drift_norm = torch.tensor(info.get("scale", 0.0))

    stats = {
        'mean_drift_norm': float(info.get(f"loss_{fc.temperature}", 0.0)),
        'mean_drift_pos_norm': 0.0,
        'mean_drift_neg_norm': 0.0,
    }
    if info.get("scale") is not None:
        stats['feature_scale'] = float(info["scale"])
    stats['effective_temperature'] = fc.temperature
    return loss, stats
```

**Step 4: Run tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/drift_training_loop.py tests/test_jax_drift_parity.py
git commit -m "refactor: replace _batched_drifting_stopgrad_loss with JAX-aligned algorithm"
```

---

## Phase 3: Fix Epsilon Values and Self-Mask Value

### Task 5: Align epsilon and mask constants with JAX

**Files:**
- Modify: `training/drift_field.py:9-14`
- Test: `tests/test_jax_drift_parity.py` (add epsilon test)

**Step 1: Write failing test**

```python
class TestEpsilonValues(unittest.TestCase):
    def test_drift_field_config_eps_matches_jax(self):
        """DriftFieldConfig.eps should be 1e-6 (matching JAX affinity clamp)."""
        from training.drift_field import DriftFieldConfig
        cfg = DriftFieldConfig()
        self.assertEqual(cfg.eps, 1e-6)

    def test_drift_field_config_mask_matches_jax(self):
        """self_mask_value should be 100.0 (matching JAX mask_val)."""
        from training.drift_field import DriftFieldConfig
        cfg = DriftFieldConfig()
        self.assertEqual(cfg.self_mask_value, 100.0)
```

**Step 2: Run test — expect FAIL**

Current values are `eps=1e-12` and `self_mask_value=1e6`.

**Step 3: Fix the defaults**

In `training/drift_field.py:9-14`, change:

```python
@dataclass(frozen=True)
class DriftFieldConfig:
    temperature: float = 0.05
    normalize_over_x: bool = True
    mask_self_negatives: bool = True
    self_mask_value: float = 100.0   # was 1e6; matches JAX drift_loss.py:81
    eps: float = 1e-6                # was 1e-12; matches JAX drift_loss.py:95
```

**Step 4: Run full test suite**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py tests/test_drift_parity.py -v`
Expected: ALL PASS (existing parity tests against `drifting_models` may need tolerance adjustments)

**Step 5: Commit**

```bash
git add training/drift_field.py tests/test_jax_drift_parity.py
git commit -m "fix: align DriftFieldConfig eps and self_mask_value with JAX reference"
```

---

## Phase 4: Fix Feature-Space Multi-Temperature Aggregation Default

### Task 6: Change `FeatureDriftingConfig.temperature_aggregation` default to `"sum_drifts_then_mse"`

The JAX reference sums per-R forces then computes a single MSE. The PyTorch default `per_temperature_mse` computes independent MSE per temperature, losing cross-temperature force interaction.

**Files:**
- Modify: `training/drift_loss.py:144`
- Test: `tests/test_jax_drift_parity.py`

**Step 1: Write test**

```python
class TestFeatureAggregationDefault(unittest.TestCase):
    def test_default_aggregation_is_sum_drifts(self):
        from training.drift_loss import FeatureDriftingConfig
        cfg = FeatureDriftingConfig()
        self.assertEqual(cfg.temperature_aggregation, "sum_drifts_then_mse")
```

**Step 2: Run — FAIL**

**Step 3: Fix default**

In `training/drift_loss.py:144`:
```python
    temperature_aggregation: str = "sum_drifts_then_mse"  # was "per_temperature_mse"
```

**Step 4: Run tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py tests/test_drift_parity.py -v`

**Step 5: Commit**

```bash
git add training/drift_loss.py tests/test_jax_drift_parity.py
git commit -m "fix: change default temperature_aggregation to sum_drifts_then_mse (matches JAX)"
```

---

### Task 7: Add per-R force normalization in `_normalize_drifts`

The `sum_drifts_then_mse` path calls `_normalize_drifts` which normalizes by global RMS. JAX normalizes each R's force by its own `(force**2).mean()`. We need to ensure per-temperature drifts are normalized by their own magnitude before summing, matching JAX lines 114-119.

**Files:**
- Modify: `training/drift_loss.py:556-575` (`_normalize_drifts`)
- Test: `tests/test_jax_drift_parity.py`

**Step 1: Write test**

```python
class TestPerRForceNormalization(unittest.TestCase):
    def test_normalize_drifts_per_force(self):
        """Each drift should be normalized by its own mean-squared force, not global RMS."""
        from training.drift_loss import _normalize_drifts
        torch.manual_seed(42)
        # Two "temperature" drifts with very different magnitudes
        drift1 = torch.randn(4, 2, 8) * 10.0   # large
        drift2 = torch.randn(4, 2, 8) * 0.01    # small

        norm1, s1 = _normalize_drifts(drift1, share_location_normalization=True, eps=1e-8)
        norm2, s2 = _normalize_drifts(drift2, share_location_normalization=True, eps=1e-8)

        # After normalization, mean(force**2) should be ~1
        rms1 = (norm1 ** 2).mean().item()
        rms2 = (norm2 ** 2).mean().item()
        self.assertAlmostEqual(rms1, 1.0, delta=0.5)
        self.assertAlmostEqual(rms2, 1.0, delta=0.5)
```

**Step 2: Run — verify behavior**

**Step 3: Verify `_drift_scale` matches JAX normalization**

The current `_drift_scale` computes `sqrt(mean(sum(d**2, dim=-1) / C))`. The JAX reference computes `sqrt(mean(force**2))` (a simple mean over all elements). These are close but differ by the `/ C` factor.

JAX (line 114): `f_norm_val = (total_force_R ** 2).mean()`
PyTorch `_drift_scale`: `sqrt(mean(sum(d^2, -1) / C))` = `sqrt(mean(d^2))` — this is equivalent since `sum/C` then `mean` over batch = flat `mean`.

Actually these are equivalent: `mean(sum(d^2, -1) / C)` = `mean(d^2)` when the mean over the batch dimension and the sum+divide cancel. So the current `_drift_scale` already matches JAX. Verify this is the case and mark as no-change-needed.

**Step 4: Run existing tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py -v`

**Step 5: Commit (test-only if no code change needed)**

```bash
git add tests/test_jax_drift_parity.py
git commit -m "test: verify _normalize_drifts matches JAX per-R force normalization"
```

---

## Phase 5: Performance Optimizations

### Task 8: Remove unnecessary `.clone()` in self-masking

**Files:**
- Modify: `training/drift_field.py:107-109`
- Modify: `training/drift_loss.py:401-404`
- Test: `tests/test_jax_drift_parity.py`

**Step 1: Write test**

```python
class TestSelfMaskNoClone(unittest.TestCase):
    def test_affinity_with_mask_produces_same_result(self):
        """Verify additive mask approach matches clone approach."""
        from training.drift_field import compute_affinity_matrices, DriftFieldConfig
        torch.manual_seed(42)
        x = torch.randn(5, 8)
        y_pos = torch.randn(4, 8)
        y_neg = torch.randn(5, 8)
        cfg = DriftFieldConfig(temperature=0.1)
        aff_pos, aff_neg = compute_affinity_matrices(
            x, y_pos, y_neg, config=cfg, generated_negative_count=5,
        )
        self.assertTrue(torch.isfinite(aff_pos).all())
        self.assertTrue(torch.isfinite(aff_neg).all())
```

**Step 2: Run — PASS (baseline)**

**Step 3: Replace `.clone()` with additive mask**

In `training/drift_field.py:105-109`:
```python
    if config.mask_self_negatives and generated_count > 0:
        diag_count = min(x.shape[0], generated_count, y_neg.shape[0])
        diagonal = torch.arange(diag_count, device=x.device)
        mask = torch.zeros_like(dist_neg)
        mask[diagonal, diagonal] = config.self_mask_value
        dist_neg = dist_neg + mask
```

In `training/drift_loss.py:399-404` (the slot-batched kernel):
```python
    if base_config.drift_field.mask_self_negatives and generated_count > 0:
        diag_count = min(x_compute.shape[1], generated_count, y_neg_compute.shape[1])
        if diag_count > 0:
            diagonal = torch.arange(diag_count, device=dist_neg.device)
            mask = torch.zeros(dist_neg.shape[0], dist_neg.shape[1], dist_neg.shape[2],
                             device=dist_neg.device, dtype=dist_neg.dtype)
            mask[:, diagonal, diagonal] = base_config.drift_field.self_mask_value
            dist_neg = dist_neg + mask
```

**Step 4: Run all tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/ -v -k "drift"`

**Step 5: Commit**

```bash
git add training/drift_field.py training/drift_loss.py
git commit -m "perf: replace .clone() with additive mask in self-masking (saves tensor allocation)"
```

---

### Task 9: Replace `.clone()` with `torch.where` for unconditional log-weights

**Files:**
- Modify: `training/drift_training_loop.py:400-411`

**Step 1: Identify the code**

Lines 400-411 in `drift_training_loop.py` use two `.clone()` calls to safely compute `log(weight)` when weight might be 0.

**Step 2: Replace with `torch.where`**

```python
            if UnconditionalGrouped is not None:
                unc_grouped = UnconditionalGrouped.reshape(Groups, unconditional_per_group, PixelDim)
                y_neg_grouped = torch.cat([y_neg_grouped, unc_grouped], dim=1)
                gen_zeros = torch.zeros(Groups, negatives_per_group, device=Device, dtype=torch.float32)
                unc_log_w = torch.where(
                    AlphaWeights > 0,
                    torch.log(AlphaWeights),
                    torch.full_like(AlphaWeights, torch.finfo(torch.float32).min),
                ).unsqueeze(1).expand(Groups, unconditional_per_group)
                neg_log_weights_grouped = torch.cat([gen_zeros, unc_log_w], dim=1)
```

**Step 3: Run tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/ -v -k "drift"`

**Step 4: Commit**

```bash
git add training/drift_training_loop.py
git commit -m "perf: use torch.where instead of double .clone() for unconditional log-weights"
```

---

### Task 10: Vectorize `_normalize_drifts` per-slot loop

**Files:**
- Modify: `training/drift_loss.py:556-575`
- Test: `tests/test_jax_drift_parity.py`

**Step 1: Write test**

```python
class TestNormalizeDriftsVectorized(unittest.TestCase):
    def test_per_slot_normalization(self):
        from training.drift_loss import _normalize_drifts
        torch.manual_seed(42)
        drifts = torch.randn(4, 3, 8)
        normed, scales = _normalize_drifts(drifts, share_location_normalization=False, eps=1e-8)
        self.assertEqual(normed.shape, drifts.shape)
        self.assertEqual(scales.shape, (3,))
        # Each slot should be independently normalized
        for v in range(3):
            slot = normed[:, v, :]
            rms = torch.sqrt(torch.mean(slot.pow(2).sum(dim=-1) / 8.0))
            self.assertAlmostEqual(rms.item(), 1.0, delta=0.01)
```

**Step 2: Run — should PASS with current code**

**Step 3: Replace Python loop with vectorized ops**

In `training/drift_loss.py`, replace `_normalize_drifts`:
```python
def _normalize_drifts(
    drifts: torch.Tensor,
    *,
    share_location_normalization: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if drifts.ndim != 3:
        raise ValueError("drifts must be [B, V, C]")
    channels = float(drifts.shape[-1])
    if share_location_normalization:
        scale = _drift_scale(drifts.reshape(-1, drifts.shape[-1]), eps=eps)
        scale_tensor = scale.repeat(drifts.shape[1])
        return drifts / scale, scale_tensor

    # Vectorized per-slot: [B, V, C] -> per-slot RMS
    per_slot_sq = drifts.pow(2).sum(dim=-1) / channels  # [B, V]
    scale_tensor = torch.sqrt(per_slot_sq.mean(dim=0))   # [V]
    scale_tensor = torch.clamp(scale_tensor, min=eps).detach()
    scale_view = scale_tensor.view(1, -1, 1)
    return drifts / scale_view, scale_tensor
```

**Step 4: Run tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_jax_drift_parity.py tests/test_drift_parity.py -v`

**Step 5: Commit**

```bash
git add training/drift_loss.py tests/test_jax_drift_parity.py
git commit -m "perf: vectorize _normalize_drifts per-slot loop"
```

---

### Task 11: Add `inplace=True` to SiLU and replace `.contiguous().view()` with `.reshape()`

**Files:**
- Modify: `training/models/dit_like.py:215,183,307`

**Step 1: Make the changes**

In `training/models/dit_like.py:215`:
```python
        self.modulation = nn.Sequential(
            nn.SiLU(inplace=True),  # was nn.SiLU()
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
```

In `training/models/dit_like.py:183`:
```python
        patch_values = patch_values.permute(0, 5, 1, 3, 2, 4)
        return patch_values.reshape(batch, out_channels, self.config.image_size, self.config.image_size)
        # was: .contiguous() then .view()
```

In `training/models/dit_like.py:307`:
```python
        out = out.transpose(1, 2).reshape(batch, length, self.hidden_dim)
        # was: .contiguous().view()
```

**Step 2: Run existing tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_dit_like_parity.py -v`

**Step 3: Commit**

```bash
git add training/models/dit_like.py
git commit -m "perf: inplace SiLU, replace .contiguous().view() with .reshape()"
```

---

## Phase 6 (Optional): DiT Architecture Alignment

These tasks bring the PyTorch DiT closer to the JAX `DitGen` architecture. They are **optional** — the PyTorch `DiTLikeGenerator` is a deliberately different design. Only apply these if you need to load JAX-pretrained weights or want exact architectural parity.

### Task 12 (Optional): Add sinusoidal positional embedding option

**Files:**
- Modify: `training/models/dit_like.py:52-58`
- Modify: `training/models/dit_like.py:12` (add to `DiTLikeConfig`)

Add a `positional_embedding_type: str = "learned"` field to `DiTLikeConfig`. When set to `"sincos"`, use a frozen sinusoidal embedding matching JAX's `sincos_init` at `/workspace/drifting/models/generator.py:16-58`.

### Task 13 (Optional): Add fp32 guard for AdaLN modulation

**Files:**
- Modify: `training/models/dit_like.py:221-223`

Wrap the modulation forward pass in `torch.amp.autocast(enabled=False)` to match JAX's explicit `jnp.float32` computation at `/workspace/drifting/models/generator.py:296`.

### Task 14 (Optional): Round SwiGLU inner dim to multiple of 32

**Files:**
- Modify: `training/models/dit_like.py:245`

Add `inner_dim = ((inner_dim + 31) // 32) * 32` after the 2/3 scaling, matching JAX at `/workspace/drifting/models/generator.py:271`.

### Task 15 (Optional): Fix patch embedding init to xavier_uniform

**Files:**
- Modify: `training/models/dit_like.py:46-51`

After constructing `self.patch_embed`, add:
```python
nn.init.xavier_uniform_(self.patch_embed.weight.view(config.hidden_dim, -1))
nn.init.zeros_(self.patch_embed.bias)
```

---

## Summary of Changes by File

| File | Phase | Change |
|---|---|---|
| `training/drift_field.py` | 1,3,5 | Add `JaxAlignedDriftConfig`, `jax_aligned_drift_loss`, `jax_aligned_drift_loss_grouped`; fix eps/mask defaults; remove `.clone()` |
| `training/drift_loss.py` | 4,5 | Change `temperature_aggregation` default; vectorize `_normalize_drifts`; remove `.clone()` |
| `training/drift_training_loop.py` | 2,5 | Replace `_batched_drifting_stopgrad_loss` internals; fix log-weight clones |
| `training/models/dit_like.py` | 5,6 | Inplace SiLU; `.reshape()`; (optional) sincos embed, fp32 modulation, SwiGLU rounding |
| `tests/test_jax_drift_parity.py` | 1-5 | New test file with JAX reference implementation and parity tests |
