# Faithful drift_models Reproduction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Inline drift_models into R3GAN's drift branch as a fully self-contained implementation, verified by three-way parity tests against toy.py and drift_models.

**Architecture:** Mirror drift_models' module structure flat under `training/`, with `features/` and `models/` subdirectories. Each file maps 1:1 to its drift_models counterpart. Delete sibling repo wrappers. Verify with toy 2D training + function-level parity.

**Tech Stack:** PyTorch, frozen dataclasses for configs, pytest for tests.

**Design doc:** `docs/plans/2026-03-16-faithful-drift-reproduction-design.md`

---

## Task 1: Create `training/drift_field.py` — Core drift field math

**Files:**
- Create: `training/drift_field.py`
- Test: `tests/test_drift_parity.py`

**Step 1: Write the parity test file skeleton + drift_field kernel test**

```python
# tests/test_drift_parity.py
"""Three-way parity tests: toy.py ↔ drift_models ↔ R3GAN inlined."""

import sys
import os
import unittest

import torch
import torch.nn.functional as F

# Reference: drift_models
DRIFT_MODELS_ROOT = "/workspace/drift_models"
if os.path.isdir(DRIFT_MODELS_ROOT) and DRIFT_MODELS_ROOT not in sys.path:
    sys.path.insert(0, DRIFT_MODELS_ROOT)

from drifting_models.drift_field import (
    DriftFieldConfig as RefDriftFieldConfig,
    cfg_alpha_to_unconditional_weight as ref_cfg_alpha,
    build_negative_log_weights as ref_build_neg_weights,
    compute_affinity_matrices as ref_compute_affinity,
    compute_drift_components as ref_compute_drift_components,
    compute_v as ref_compute_v,
)

# Toy reference (standalone)
def toy_compute_drift(gen, pos, temp=0.05):
    """Exact copy from toy.py lines 135-163."""
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]
    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)
    kernel = (-dist / temp).exp()
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer
    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]
    return pos_V - neg_V


@unittest.skipIf(not torch.is_available() if hasattr(torch, 'is_available') else False, "no torch")
class TestDriftFieldParity(unittest.TestCase):

    def test_cfg_alpha_to_unconditional_weight_parity(self):
        from training.drift_field import cfg_alpha_to_unconditional_weight
        cases = [(1.0, 4, 2), (3.0, 4, 2), (2.5, 8, 3), (1.0, 2, 1)]
        for alpha, n_gen, n_unc in cases:
            r3gan = cfg_alpha_to_unconditional_weight(alpha, n_gen, n_unc)
            ref = ref_cfg_alpha(alpha, n_gen, n_unc)
            self.assertAlmostEqual(r3gan, ref, places=10,
                msg=f"Mismatch for alpha={alpha}, n_gen={n_gen}, n_unc={n_unc}")

    def test_build_negative_log_weights_parity(self):
        from training.drift_field import build_negative_log_weights
        cases = [
            (4, 2, 1.0),
            (4, 2, 0.0),
            (4, 0, 0.0),
            (8, 3, 0.5),
        ]
        for n_gen, n_unc, w in cases:
            r3gan = build_negative_log_weights(n_gen, n_unc, w, device=torch.device('cpu'), dtype=torch.float32)
            ref = ref_build_neg_weights(n_gen, n_unc, w, device=torch.device('cpu'), dtype=torch.float32)
            torch.testing.assert_close(r3gan, ref, atol=1e-7, rtol=1e-7,
                msg=f"Mismatch for n_gen={n_gen}, n_unc={n_unc}, w={w}")

    def test_compute_affinity_matrices_parity(self):
        from training.drift_field import DriftFieldConfig, compute_affinity_matrices
        torch.manual_seed(42)
        x = torch.randn(5, 8)
        y_pos = torch.randn(4, 8)
        y_neg = torch.randn(5, 8)
        config = DriftFieldConfig(temperature=0.1, normalize_over_x=True, mask_self_negatives=True)
        ref_config = RefDriftFieldConfig(temperature=0.1, normalize_over_x=True, mask_self_negatives=True)
        aff_pos, aff_neg = compute_affinity_matrices(x, y_pos, y_neg, config=config)
        ref_pos, ref_neg = ref_compute_affinity(x, y_pos, y_neg, config=ref_config)
        torch.testing.assert_close(aff_pos, ref_pos, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(aff_neg, ref_neg, atol=1e-6, rtol=1e-6)

    def test_compute_v_parity(self):
        from training.drift_field import DriftFieldConfig, compute_v
        torch.manual_seed(42)
        x = torch.randn(5, 8)
        y_pos = torch.randn(4, 8)
        y_neg = torch.randn(5, 8)
        config = DriftFieldConfig(temperature=0.1)
        ref_config = RefDriftFieldConfig(temperature=0.1)
        v = compute_v(x, y_pos, y_neg, config=config)
        v_ref = ref_compute_v(x, y_pos, y_neg, config=ref_config)
        torch.testing.assert_close(v, v_ref, atol=1e-6, rtol=1e-6)

    def test_compute_v_matches_toy_drift(self):
        """Three-way: R3GAN compute_v should match toy.py compute_drift."""
        from training.drift_field import DriftFieldConfig, compute_v
        torch.manual_seed(42)
        gen = torch.randn(10, 2)
        pos = torch.randn(20, 2)
        temp = 0.2
        # toy.py uses gen as negatives (with self-masking), pos as positives
        # drift_field.compute_v: x=gen, y_pos=pos, y_neg=gen (self-neg masked)
        config = DriftFieldConfig(temperature=temp, normalize_over_x=True, mask_self_negatives=True)
        v_r3gan = compute_v(gen, pos, gen, config=config, generated_negative_count=gen.shape[0])
        v_toy = toy_compute_drift(gen, pos, temp=temp)
        torch.testing.assert_close(v_r3gan, v_toy, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py::TestDriftFieldParity -v 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.drift_field'`

**Step 3: Create `training/drift_field.py`**

Port from `/workspace/drift_models/drifting_models/drift_field.py` — exact same logic, exact same function signatures and config field names.

```python
# training/drift_field.py
"""Core drift field computation — self-contained port of drift_models.drift_field."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DriftFieldConfig:
    temperature: float = 0.05
    normalize_over_x: bool = True
    mask_self_negatives: bool = True
    self_mask_value: float = 1e6
    eps: float = 1e-12


def cfg_alpha_to_unconditional_weight(
    alpha: float,
    n_generated_negatives: int,
    n_unconditional_negatives: int,
) -> float:
    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.0")
    if n_generated_negatives <= 1:
        raise ValueError("n_generated_negatives must be > 1")
    if n_unconditional_negatives <= 0:
        raise ValueError("n_unconditional_negatives must be > 0")
    return ((alpha - 1.0) * (n_generated_negatives - 1)) / n_unconditional_negatives


def build_negative_log_weights(
    n_generated_negatives: int,
    n_unconditional_negatives: int,
    unconditional_weight: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_generated_negatives <= 0:
        raise ValueError("n_generated_negatives must be > 0")
    if n_unconditional_negatives < 0:
        raise ValueError("n_unconditional_negatives must be >= 0")
    if unconditional_weight < 0.0:
        raise ValueError("unconditional_weight must be >= 0")
    generated = torch.zeros(n_generated_negatives, device=device, dtype=dtype)
    if n_unconditional_negatives == 0:
        return generated
    if unconditional_weight == 0.0:
        unconditional = torch.full(
            (n_unconditional_negatives,),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
    else:
        unconditional = torch.full(
            (n_unconditional_negatives,),
            torch.log(torch.tensor(unconditional_weight, device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )
    return torch.cat([generated, unconditional], dim=0)


def compute_affinity_matrices(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(
        x=x, y_pos=y_pos, y_neg=y_neg,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    generated_count = generated_negative_count if generated_negative_count is not None else y_neg.shape[0]
    if config.mask_self_negatives and generated_count > 0:
        diag_count = min(x.shape[0], generated_count, y_neg.shape[0])
        diagonal = torch.arange(diag_count, device=x.device)
        dist_neg = dist_neg.clone()
        dist_neg[diagonal, diagonal] = dist_neg[diagonal, diagonal] + config.self_mask_value
    logit_pos = -(dist_pos / config.temperature)
    logit_neg = -(dist_neg / config.temperature)
    if negative_log_weights is not None:
        logit_neg = logit_neg + negative_log_weights.view(1, -1)
    logits = torch.cat([logit_pos, logit_neg], dim=1)
    row_affinity = torch.softmax(logits, dim=-1)
    if config.normalize_over_x:
        col_affinity = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=config.eps))
    else:
        affinity = row_affinity
    n_pos = y_pos.shape[0]
    return affinity[:, :n_pos], affinity[:, n_pos:]


def compute_drift_components(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    affinity_pos, affinity_neg = compute_affinity_matrices(
        x=x, y_pos=y_pos, y_neg=y_neg, config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    weight_pos = affinity_pos * affinity_neg.sum(dim=1, keepdim=True)
    weight_neg = affinity_neg * affinity_pos.sum(dim=1, keepdim=True)
    drift_pos = weight_pos @ y_pos
    drift_neg = weight_neg @ y_neg
    return drift_pos, drift_neg


def compute_v(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    *,
    config: DriftFieldConfig,
    negative_log_weights: torch.Tensor | None = None,
    generated_negative_count: int | None = None,
) -> torch.Tensor:
    drift_pos, drift_neg = compute_drift_components(
        x=x, y_pos=y_pos, y_neg=y_neg, config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    return drift_pos - drift_neg


def _validate_inputs(
    *,
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    negative_log_weights: torch.Tensor | None,
    generated_negative_count: int | None,
) -> None:
    for name, value in (("x", x), ("y_pos", y_pos), ("y_neg", y_neg)):
        if value.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {tuple(value.shape)}")
    if x.shape[1] != y_pos.shape[1] or x.shape[1] != y_neg.shape[1]:
        raise ValueError("x, y_pos, y_neg must share feature dimension")
    if negative_log_weights is not None:
        if negative_log_weights.ndim != 1:
            raise ValueError("negative_log_weights must be 1D")
        if negative_log_weights.shape[0] != y_neg.shape[0]:
            raise ValueError("negative_log_weights size must match y_neg count")
    if generated_negative_count is not None:
        if generated_negative_count < 0:
            raise ValueError("generated_negative_count must be >= 0")
        if generated_negative_count > y_neg.shape[0]:
            raise ValueError("generated_negative_count cannot exceed y_neg count")
```

**Step 4: Run parity tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py::TestDriftFieldParity -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/drift_field.py tests/test_drift_parity.py
git commit -m "feat: add training/drift_field.py — self-contained drift field kernel with parity tests"
```

---

## Task 2: Rewrite `training/drift_loss.py` — Loss functions

**Files:**
- Rewrite: `training/drift_loss.py`
- Create: `training/features/__init__.py` (empty, needed for FeatureVectorizationConfig import)
- Create: `training/features/vectorize.py` (stub with just FeatureVectorizationConfig, full impl in Task 7)
- Test: `tests/test_drift_parity.py` (add TestDriftLossParity class)

**Step 1: Add parity tests for drift_loss functions**

Append to `tests/test_drift_parity.py`:

```python
from drifting_models.drift_loss import (
    DriftingLossConfig as RefDriftingLossConfig,
    drifting_stopgrad_loss as ref_stopgrad_loss,
    drifting_stopgrad_loss_multi_temperature as ref_multi_temp_loss,
)
from drifting_models.drift_field import DriftFieldConfig as RefDriftFieldConfig


class TestDriftLossParity(unittest.TestCase):

    def test_drifting_stopgrad_loss_parity(self):
        from training.drift_loss import DriftingLossConfig, drifting_stopgrad_loss
        from training.drift_field import DriftFieldConfig
        torch.manual_seed(42)
        x = torch.randn(5, 16)
        y_pos = torch.randn(4, 16)
        y_neg = torch.randn(5, 16)

        config = DriftingLossConfig(drift_field=DriftFieldConfig(temperature=0.1))
        ref_config = RefDriftingLossConfig(drift_field=RefDriftFieldConfig(temperature=0.1))

        loss, drift, stats = drifting_stopgrad_loss(x, y_pos, y_neg, config=config)
        ref_loss, ref_drift, ref_stats = ref_stopgrad_loss(x, y_pos, y_neg, config=ref_config)

        torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(drift, ref_drift, atol=1e-6, rtol=1e-6)

    def test_multi_temperature_loss_parity(self):
        from training.drift_loss import DriftingLossConfig, drifting_stopgrad_loss_multi_temperature
        from training.drift_field import DriftFieldConfig
        torch.manual_seed(42)
        x = torch.randn(5, 16)
        y_pos = torch.randn(4, 16)
        y_neg = torch.randn(5, 16)
        temps = (0.01, 0.05, 0.1)

        config = DriftingLossConfig(drift_field=DriftFieldConfig(temperature=0.05))
        ref_config = RefDriftingLossConfig(drift_field=RefDriftFieldConfig(temperature=0.05))

        loss, stats = drifting_stopgrad_loss_multi_temperature(
            x, y_pos, y_neg, temperatures=temps, config=config)
        ref_loss, ref_stats = ref_multi_temp_loss(
            x, y_pos, y_neg, temperatures=temps, config=ref_config)

        torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-6)
        for key in ["loss", "mean_drift_norm", "temperature_count"]:
            self.assertAlmostEqual(stats[key], ref_stats[key], places=5,
                msg=f"Stats mismatch for key={key}")

    def test_compute_weighted_drift_parity(self):
        from training.drift_loss import DriftingLossConfig, compute_weighted_drift
        from training.drift_field import DriftFieldConfig
        from drifting_models.drift_loss import compute_weighted_drift as ref_compute_weighted_drift
        torch.manual_seed(42)
        x = torch.randn(5, 16)
        y_pos = torch.randn(4, 16)
        y_neg = torch.randn(5, 16)

        config = DriftingLossConfig(drift_field=DriftFieldConfig(temperature=0.1))
        ref_config = RefDriftingLossConfig(drift_field=RefDriftFieldConfig(temperature=0.1))

        drift, stats = compute_weighted_drift(x, y_pos, y_neg, config=config)
        ref_drift, ref_stats = ref_compute_weighted_drift(x, y_pos, y_neg, config=ref_config)

        torch.testing.assert_close(drift, ref_drift, atol=1e-6, rtol=1e-6)
```

**Step 2: Run to verify failure**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py::TestDriftLossParity -v 2>&1 | head -20`
Expected: FAIL — `ImportError` or `AttributeError` because DriftingLossConfig doesn't exist yet

**Step 3: Create stub `training/features/__init__.py` and `training/features/vectorize.py`**

The drift_loss.py imports `FeatureVectorizationConfig` from features. Create minimal stubs first:

```python
# training/features/__init__.py
```

```python
# training/features/vectorize.py
"""Feature vectorization — stub, full implementation in Task 7."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureVectorizationConfig:
    include_per_location: bool = True
    include_global_stats: bool = True
    include_patch2_stats: bool = True
    include_patch4_stats: bool = True
    include_input_x2_mean: bool = False
    selected_stages: tuple[int, ...] | None = None
```

**Step 4: Rewrite `training/drift_loss.py`**

Port from `/workspace/drift_models/drifting_models/drift_loss.py`. This replaces the existing file entirely. Key changes from old R3GAN version:
- Uses `DriftingLossConfig` (with nested `DriftFieldConfig`) instead of flat `DriftLossConfig`
- Adds `compute_weighted_drift`, `drifting_stopgrad_loss_multi_temperature`
- Adds `FeatureDriftingConfig`, `feature_space_drifting_loss` with all normalization helpers
- Removes old `grouped_drifting_stopgrad_loss` (moved to drift_stage2 later)

The full file should be an exact port of `/workspace/drift_models/drifting_models/drift_loss.py`, changing only the import path:
```python
from training.features.vectorize import FeatureVectorizationConfig
from training.drift_field import (
    DriftFieldConfig,
    build_negative_log_weights,
    compute_drift_components,
)
```

**Step 5: Run parity tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py::TestDriftLossParity -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add training/drift_loss.py training/features/__init__.py training/features/vectorize.py tests/test_drift_parity.py
git commit -m "feat: rewrite training/drift_loss.py — full DriftingLossConfig + multi-temp + feature loss with parity tests"
```

---

## Task 3: Create `training/drift_grouped.py` — Grouped batch utilities

**Files:**
- Create: `training/drift_grouped.py`
- Test: `tests/test_drift_parity.py` (add TestDriftGroupedParity)

**Step 1: Add parity test**

```python
from drifting_models.train.grouped import (
    GroupedBatchShapes as RefGroupedBatchShapes,
    infer_grouped_shapes as ref_infer_grouped_shapes,
    compute_grouped_v as ref_compute_grouped_v,
)

class TestDriftGroupedParity(unittest.TestCase):

    def test_infer_grouped_shapes_parity(self):
        from training.drift_grouped import infer_grouped_shapes
        torch.manual_seed(42)
        x = torch.randn(3, 4, 16)
        y_pos = torch.randn(3, 5, 16)
        y_neg = torch.randn(3, 4, 16)
        shapes = infer_grouped_shapes(x, y_pos, y_neg)
        ref_shapes = ref_infer_grouped_shapes(x, y_pos, y_neg)
        self.assertEqual(shapes.groups, ref_shapes.groups)
        self.assertEqual(shapes.negatives_per_group, ref_shapes.negatives_per_group)
        self.assertEqual(shapes.positives_per_group, ref_shapes.positives_per_group)
        self.assertEqual(shapes.feature_dim, ref_shapes.feature_dim)

    def test_compute_grouped_v_parity(self):
        from training.drift_grouped import compute_grouped_v
        from training.drift_field import DriftFieldConfig
        torch.manual_seed(42)
        x = torch.randn(3, 4, 16)
        y_pos = torch.randn(3, 5, 16)
        y_neg = torch.randn(3, 4, 16)
        config = DriftFieldConfig(temperature=0.1)
        ref_config = RefDriftFieldConfig(temperature=0.1)
        v = compute_grouped_v(x, y_pos, y_neg, config=config)
        v_ref = ref_compute_grouped_v(x, y_pos, y_neg, config=ref_config)
        torch.testing.assert_close(v, v_ref, atol=1e-6, rtol=1e-6)
```

**Step 2: Run test — fails**

**Step 3: Create `training/drift_grouped.py`**

Port from `/workspace/drift_models/drifting_models/train/grouped.py` (88 lines). Change imports to:
```python
from training.drift_field import DriftFieldConfig, compute_v
```

**Step 4: Run test — passes**

**Step 5: Commit**

```bash
git add training/drift_grouped.py tests/test_drift_parity.py
git commit -m "feat: add training/drift_grouped.py — grouped batch utilities with parity tests"
```

---

## Task 4: Create `training/models/dit_like.py` — DiT generator

**Files:**
- Create: `training/models/__init__.py`
- Create: `training/models/dit_like.py`
- Test: `tests/test_drift_parity.py` (add TestDiTLikeParity)

**Step 1: Add parity test**

```python
class TestDiTLikeParity(unittest.TestCase):

    def test_dit_like_forward_shape(self):
        from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator
        config = DiTLikeConfig(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=10,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        )
        model = DiTLikeGenerator(config)
        noise = torch.randn(2, 4, 8, 8)
        labels = torch.tensor([0, 3])
        alpha = torch.tensor([1.0, 2.0])
        out = model(noise, labels, alpha)
        self.assertEqual(out.shape, (2, 4, 8, 8))

    def test_dit_like_alpha_changes_output(self):
        from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator
        config = DiTLikeConfig(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=10,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        )
        model = DiTLikeGenerator(config)
        torch.manual_seed(0)
        noise = torch.randn(1, 4, 8, 8)
        labels = torch.tensor([0])
        out_a1 = model(noise, labels, torch.tensor([1.0]))
        out_a3 = model(noise, labels, torch.tensor([3.0]))
        self.assertFalse(torch.allclose(out_a1, out_a3),
            "Different alpha should produce different output")

    def test_dit_like_weight_parity_with_reference(self):
        """Same config + same weights → same output."""
        from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator
        from drifting_models.models.dit_like import (
            DiTLikeConfig as RefConfig, DiTLikeGenerator as RefGenerator,
        )
        cfg_kwargs = dict(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=10,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        )
        r3gan_model = DiTLikeGenerator(DiTLikeConfig(**cfg_kwargs))
        ref_model = RefGenerator(RefConfig(**cfg_kwargs))
        # Copy weights from r3gan to ref
        ref_model.load_state_dict(r3gan_model.state_dict())

        torch.manual_seed(7)
        noise = torch.randn(2, 4, 8, 8)
        labels = torch.tensor([0, 5])
        alpha = torch.tensor([1.0, 2.5])
        with torch.no_grad():
            out_r3gan = r3gan_model(noise, labels, alpha)
            out_ref = ref_model(noise, labels, alpha)
        torch.testing.assert_close(out_r3gan, out_ref, atol=1e-5, rtol=1e-5)
```

**Step 2: Run test — fails**

**Step 3: Create `training/models/__init__.py`** (empty)

**Step 4: Create `training/models/dit_like.py`**

Port from `/workspace/drift_models/drifting_models/models/dit_like.py` (469 lines). No import changes needed — the file is self-contained (only uses torch, math).

**Step 5: Run test — passes**

**Step 6: Commit**

```bash
git add training/models/__init__.py training/models/dit_like.py tests/test_drift_parity.py
git commit -m "feat: add training/models/dit_like.py — self-contained DiT generator with parity tests"
```

---

## Task 5: Update `training/networks.py` — Use inlined DiT

**Files:**
- Modify: `training/networks.py`
- Test: `tests/test_drift_parity.py` (add TestNetworkWrapperParity)

**Step 1: Add test**

```python
class TestNetworkWrapperParity(unittest.TestCase):

    def test_dit_like_drift_generator_forward(self):
        from training.networks import DiTLikeDriftGenerator
        kw = dict(
            c_dim=10, img_resolution=8, ImageChannels=4, PatchSize=2,
            HiddenDim=32, Depth=2, NumHeads=4, RegisterTokens=4,
            StyleVocabSize=4, StyleTokenCount=2, AlphaHiddenDim=16,
            EvalAlpha=1.0,
        )
        model = DiTLikeDriftGenerator(**kw)
        noise = torch.randn(2, 4, 8, 8)
        labels = torch.tensor([0, 5])
        out = model(noise, labels)  # no alpha → uses EvalAlpha
        self.assertEqual(out.shape, (2, 4, 8, 8))

    def test_dit_like_drift_generator_with_alpha(self):
        from training.networks import DiTLikeDriftGenerator
        kw = dict(
            c_dim=10, img_resolution=8, ImageChannels=4, PatchSize=2,
            HiddenDim=32, Depth=2, NumHeads=4, RegisterTokens=4,
            StyleVocabSize=4, StyleTokenCount=2, AlphaHiddenDim=16,
            EvalAlpha=1.0,
        )
        model = DiTLikeDriftGenerator(**kw)
        noise = torch.randn(2, 4, 8, 8)
        labels = torch.tensor([0, 5])
        alpha = torch.tensor([1.5, 3.0])
        out = model(noise, labels, alpha=alpha)
        self.assertEqual(out.shape, (2, 4, 8, 8))
```

**Step 2: Modify `training/networks.py`**

Change the import on line 6 from:
```python
from training.drift_reference import DiTLikeConfig, DiTLikeGenerator as ReferenceDiTLikeGenerator
```
to:
```python
from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator as ReferenceDiTLikeGenerator
```

**Step 3: Run test — passes**

**Step 4: Commit**

```bash
git add training/networks.py tests/test_drift_parity.py
git commit -m "feat: update networks.py to import from inlined dit_like instead of drift_reference"
```

---

## Task 6: Complete `training/features/vectorize.py` and create `training/features/extractors.py`

**Files:**
- Rewrite: `training/features/vectorize.py` (replace stub with full implementation)
- Create: `training/features/extractors.py`
- Test: `tests/test_drift_parity.py` (add TestFeaturesParity)

**Step 1: Add parity tests**

```python
class TestFeaturesParity(unittest.TestCase):

    def test_vectorize_feature_maps_parity(self):
        from training.features.vectorize import (
            FeatureVectorizationConfig, vectorize_feature_maps,
        )
        from drifting_models.features.vectorize import (
            FeatureVectorizationConfig as RefConfig,
            vectorize_feature_maps as ref_vectorize,
        )
        torch.manual_seed(42)
        fmaps = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]
        config = FeatureVectorizationConfig()
        ref_config = RefConfig()
        result = vectorize_feature_maps(fmaps, config=config)
        ref_result = ref_vectorize(fmaps, config=ref_config)
        self.assertEqual(sorted(result.keys()), sorted(ref_result.keys()))
        for key in result:
            torch.testing.assert_close(result[key], ref_result[key], atol=1e-6, rtol=1e-6,
                msg=f"Mismatch for key={key}")

    def test_tiny_feature_encoder_shape(self):
        from training.features.extractors import TinyFeatureEncoderConfig, TinyFeatureEncoder
        config = TinyFeatureEncoderConfig(in_channels=4, base_channels=16, stages=3)
        encoder = TinyFeatureEncoder(config)
        images = torch.randn(2, 4, 16, 16)
        features = encoder(images)
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0].shape[0], 2)
        self.assertEqual(features[0].ndim, 4)

    def test_tiny_feature_encoder_weight_parity(self):
        from training.features.extractors import TinyFeatureEncoderConfig, TinyFeatureEncoder
        from drifting_models.features.extractors import (
            TinyFeatureEncoderConfig as RefConfig,
            TinyFeatureEncoder as RefEncoder,
        )
        cfg_kwargs = dict(in_channels=4, base_channels=16, stages=2)
        r3gan_enc = TinyFeatureEncoder(TinyFeatureEncoderConfig(**cfg_kwargs))
        ref_enc = RefEncoder(RefConfig(**cfg_kwargs))
        ref_enc.load_state_dict(r3gan_enc.state_dict())
        torch.manual_seed(7)
        images = torch.randn(2, 4, 16, 16)
        with torch.no_grad():
            r3gan_out = r3gan_enc(images)
            ref_out = ref_enc(images)
        for i in range(len(r3gan_out)):
            torch.testing.assert_close(r3gan_out[i], ref_out[i], atol=1e-5, rtol=1e-5)
```

**Step 2: Run test — fails**

**Step 3: Rewrite `training/features/vectorize.py`**

Port from `/workspace/drift_models/drifting_models/features/vectorize.py` (109 lines). No import changes — file is self-contained.

**Step 4: Create `training/features/extractors.py`**

Port from `/workspace/drift_models/drifting_models/features/extractors.py` (56 lines). No import changes — file is self-contained.

**Step 5: Run test — passes**

**Step 6: Commit**

```bash
git add training/features/vectorize.py training/features/extractors.py tests/test_drift_parity.py
git commit -m "feat: add feature vectorization and TinyFeatureEncoder with parity tests"
```

---

## Task 7: Create `training/drift_stage2.py` — Grouped training step

**Files:**
- Create: `training/drift_stage2.py`
- Test: `tests/test_drift_parity.py` (add TestDriftStage2Parity)

**Step 1: Add parity test**

```python
class TestDriftStage2Parity(unittest.TestCase):

    def test_grouped_drift_step_raw_loss_parity(self):
        """Raw path (no feature extractor): R3GAN vs drift_models."""
        from training.drift_stage2 import GroupedDriftStepConfig, grouped_drift_training_step
        from training.drift_loss import DriftingLossConfig
        from training.drift_field import DriftFieldConfig
        from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator

        from drifting_models.train.stage2 import (
            GroupedDriftStepConfig as RefStepConfig,
            grouped_drift_training_step as ref_step,
        )
        from drifting_models.drift_loss import DriftingLossConfig as RefLossConfig
        from drifting_models.drift_field import DriftFieldConfig as RefFieldConfig
        from drifting_models.models.dit_like import DiTLikeConfig as RefDiTConfig, DiTLikeGenerator as RefDiTGen

        cfg_kwargs = dict(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=10,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        )
        torch.manual_seed(42)
        r3gan_gen = DiTLikeGenerator(DiTLikeConfig(**cfg_kwargs))
        ref_gen = RefDiTGen(RefDiTConfig(**cfg_kwargs))
        ref_gen.load_state_dict(r3gan_gen.state_dict())

        r3gan_opt = torch.optim.Adam(r3gan_gen.parameters(), lr=1e-4)
        ref_opt = torch.optim.Adam(ref_gen.parameters(), lr=1e-4)

        torch.manual_seed(99)
        noise = torch.randn(2, 3, 4, 8, 8)  # [G=2, N=3, C=4, H=8, W=8]
        labels = torch.tensor([0, 5])
        alpha = torch.tensor([1.5, 2.5])
        positives = torch.randn(2, 4, 4, 8, 8)
        unconditional = torch.randn(2, 2, 4, 8, 8)
        unc_weights = torch.tensor([1.0, 0.5])

        loss_config = DriftingLossConfig(drift_field=DriftFieldConfig(temperature=0.1))
        ref_loss_config = RefLossConfig(drift_field=RefFieldConfig(temperature=0.1))

        config = GroupedDriftStepConfig(loss_config=loss_config)
        ref_config = RefStepConfig(loss_config=ref_loss_config)

        r3gan_stats = grouped_drift_training_step(
            generator=r3gan_gen, optimizer=r3gan_opt,
            noise_grouped=noise, class_labels_grouped=labels,
            alpha_grouped=alpha, positives_grouped=positives,
            style_indices_grouped=None,
            unconditional_grouped=unconditional,
            unconditional_weight_grouped=unc_weights,
            config=config,
        )
        ref_stats = ref_step(
            generator=ref_gen, optimizer=ref_opt,
            noise_grouped=noise, class_labels_grouped=labels,
            alpha_grouped=alpha, positives_grouped=positives,
            style_indices_grouped=None,
            unconditional_grouped=unconditional,
            unconditional_weight_grouped=unc_weights,
            config=ref_config,
        )
        self.assertAlmostEqual(r3gan_stats["loss"], ref_stats["loss"], places=4)
        self.assertAlmostEqual(r3gan_stats["mean_drift_norm"], ref_stats["mean_drift_norm"], places=4)
```

**Step 2: Run test — fails**

**Step 3: Create `training/drift_stage2.py`**

Port from `/workspace/drift_models/drifting_models/train/stage2.py` (400 lines). Change imports:
```python
from training.drift_field import build_negative_log_weights
from training.drift_loss import (
    DriftingLossConfig,
    FeatureDriftingConfig,
    drifting_stopgrad_loss,
    drifting_stopgrad_loss_multi_temperature,
    feature_space_drifting_loss,
)
from training.features.vectorize import extract_feature_maps, vectorize_feature_maps
from training.drift_grouped import infer_grouped_shapes
```

**Step 4: Run test — passes**

**Step 5: Commit**

```bash
git add training/drift_stage2.py tests/test_drift_parity.py
git commit -m "feat: add training/drift_stage2.py — grouped training step with parity test"
```

---

## Task 8: Align `training/drift_queue.py` with drift_models

**Files:**
- Modify: `training/drift_queue.py`
- Test: `tests/test_drift_parity.py` (add TestQueueParity)

**Step 1: Add parity test**

```python
class TestQueueParity(unittest.TestCase):

    def test_queue_push_and_sample(self):
        from training.drift_queue import ClassConditionalSampleQueue, QueueConfig
        config = QueueConfig(num_classes=3, per_class_capacity=10, global_capacity=30)
        queue = ClassConditionalSampleQueue(config)
        images = torch.randn(6, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        queue.push(images, labels)
        self.assertEqual(queue.global_count(), 6)
        self.assertEqual(queue.class_count(0), 2)
        positives = queue.sample_positive_grouped(
            torch.tensor([0, 1]), 2, device=torch.device('cpu'))
        self.assertEqual(positives.shape, (2, 2, 3, 4, 4))

    def test_queue_state_dict_roundtrip(self):
        from training.drift_queue import ClassConditionalSampleQueue, QueueConfig
        config = QueueConfig(num_classes=3, per_class_capacity=10, global_capacity=30)
        queue = ClassConditionalSampleQueue(config)
        images = torch.randn(6, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        queue.push(images, labels)
        state = queue.state_dict()
        queue2 = ClassConditionalSampleQueue(config)
        queue2.load_state_dict(state)
        self.assertEqual(queue2.global_count(), queue.global_count())
        for label in range(3):
            self.assertEqual(queue2.class_count(label), queue.class_count(label))

    def test_queue_version_field(self):
        """drift_models queue state_dict includes version field."""
        from training.drift_queue import ClassConditionalSampleQueue, QueueConfig
        config = QueueConfig(num_classes=3, per_class_capacity=10, global_capacity=30)
        queue = ClassConditionalSampleQueue(config)
        state = queue.state_dict()
        self.assertIn("version", state)
        self.assertEqual(state["version"], 1)
```

**Step 2: Align drift_queue.py**

The current R3GAN queue is already very close to drift_models. Key changes:
1. Add `version: 1` to `state_dict()` output
2. Add `GroupedSamplingConfig` and `sample_grouped_real_batches` helper
3. Handle `strict_without_replacement` None check in `load_state_dict`

Keep `ensure_class_coverage` (R3GAN-specific utility not in drift_models but still needed by training loop).

**Step 3: Run test — passes**

**Step 4: Commit**

```bash
git add training/drift_queue.py tests/test_drift_parity.py
git commit -m "feat: align drift_queue.py with drift_models contract (version field, GroupedSamplingConfig)"
```

---

## Task 9: Delete legacy files

**Files:**
- Delete: `training/drift_reference.py`
- Delete: `training/drift_research.py`
- Delete: `training/drift_diagnostics.py`

**Step 1: Verify no remaining imports**

Run: `cd /workspace/R3GAN && grep -r 'drift_reference\|drift_research\|drift_diagnostics' training/ --include='*.py' | grep -v '__pycache__'`

Expected: Only hits in the files being deleted (and possibly old tests). The `networks.py` import was already updated in Task 5.

**Step 2: Delete the files**

```bash
rm training/drift_reference.py training/drift_research.py training/drift_diagnostics.py
```

**Step 3: Update test file**

Remove any imports of `drift_diagnostics_impl` and `drift_research_impl` from `tests/test_drift_training.py`.

**Step 4: Run existing tests to check nothing breaks**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -u training/drift_reference.py training/drift_research.py training/drift_diagnostics.py
git add tests/test_drift_training.py
git commit -m "chore: delete drift_reference, drift_research, drift_diagnostics — replaced by inlined code"
```

---

## Task 10: Rewrite `training/drift_training_loop.py` — Delegate to drift_stage2

**Files:**
- Rewrite: `training/drift_training_loop.py`

**Step 1: Rewrite the training loop**

Key changes from current version:
1. Remove the `drift_research` delegation block at top (lines 74-100)
2. Import from new modules instead of old `drift_loss.py`:
   ```python
   from training.drift_field import DriftFieldConfig, cfg_alpha_to_unconditional_weight
   from training.drift_loss import DriftingLossConfig
   from training.drift_stage2 import GroupedDriftStepConfig, grouped_drift_training_step
   from training.drift_queue import ClassConditionalSampleQueue, QueueConfig, ensure_class_coverage
   ```
3. Replace inline loss computation (lines 327-343) with call to `grouped_drift_training_step()`
4. Support both raw (DriftGenerator) and faithful (DiTLikeDriftGenerator) via `drift_config.backbone`

The training loop structure (dataset loading, queue priming, EMA, checkpointing, metrics) stays the same — only the loss computation block changes.

**Step 2: Verify the loop can still be called**

This requires a full dataset, so we defer full integration testing to Task 12. For now, verify imports work:

Run: `cd /workspace/R3GAN && python -c "from training.drift_training_loop import training_loop; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add training/drift_training_loop.py
git commit -m "feat: rewrite drift_training_loop to delegate to drift_stage2"
```

---

## Task 11: Toy training smoke test

**Files:**
- Create: `tests/test_drift_integration.py`

**Step 1: Write toy 2D training test using R3GAN's inlined drift functions**

```python
# tests/test_drift_integration.py
"""Integration tests: toy training, image smoke, checkpoint round-trip."""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_checkerboard(n, seed=None):
    """Minimal checkerboard sampler from toy.py."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    return pts / 2.0


class ToyMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z):
        return self.net(z)


class TestToyTrainingSmoke(unittest.TestCase):

    def test_toy_checkerboard_loss_decreases(self):
        """Train MLP on checkerboard using R3GAN's drift_field.compute_v, verify loss drops."""
        from training.drift_field import DriftFieldConfig, compute_v

        torch.manual_seed(42)
        model = ToyMLP(in_dim=16, hidden=128, out_dim=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = DriftFieldConfig(temperature=0.1)

        losses = []
        for step in range(200):
            pos = sample_checkerboard(512, seed=step)
            gen = model(torch.randn(256, 16))
            with torch.no_grad():
                v = compute_v(
                    gen.detach(), pos, gen.detach(),
                    config=config, generated_negative_count=gen.shape[0],
                )
                target = (gen.detach() + v).detach()
            loss = F.mse_loss(gen, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should decrease significantly over 200 steps
        early_avg = sum(losses[:20]) / 20
        late_avg = sum(losses[-20:]) / 20
        self.assertLess(late_avg, early_avg * 0.5,
            f"Loss did not decrease enough: early={early_avg:.4f}, late={late_avg:.4f}")


class TestImageSmokeIntegration(unittest.TestCase):

    def test_dit_feature_smoke(self):
        """Tiny DiT + TinyFeatureEncoder + queue: 3 steps, finite loss, non-zero grads."""
        from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator
        from training.features.extractors import TinyFeatureEncoderConfig, TinyFeatureEncoder
        from training.features.vectorize import FeatureVectorizationConfig
        from training.drift_field import DriftFieldConfig, cfg_alpha_to_unconditional_weight
        from training.drift_loss import DriftingLossConfig, FeatureDriftingConfig
        from training.drift_stage2 import GroupedDriftStepConfig, grouped_drift_training_step
        from training.drift_queue import ClassConditionalSampleQueue, QueueConfig

        torch.manual_seed(42)
        gen = DiTLikeGenerator(DiTLikeConfig(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=5,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        ))
        encoder = TinyFeatureEncoder(TinyFeatureEncoderConfig(in_channels=4, base_channels=8, stages=2))
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
        queue = ClassConditionalSampleQueue(QueueConfig(num_classes=5, per_class_capacity=20, global_capacity=100))

        # Prime queue with synthetic data
        for label in range(5):
            images = torch.randn(10, 4, 8, 8)
            labels = torch.full((10,), label, dtype=torch.long)
            queue.push(images, labels)

        feature_config = FeatureDriftingConfig(
            temperatures=(0.05, 0.1),
            vectorization=FeatureVectorizationConfig(
                include_per_location=True, include_global_stats=True,
                include_patch2_stats=False, include_patch4_stats=False,
            ),
            normalize_features=True,
            normalize_drifts=True,
            scale_temperature_by_sqrt_channels=True,
            detach_positive_features=True,
            detach_negative_features=True,
        )
        loss_config = DriftingLossConfig(drift_field=DriftFieldConfig(temperature=0.05))
        step_config = GroupedDriftStepConfig(
            loss_config=loss_config,
            feature_config=feature_config,
            clip_grad_norm=2.0,
        )

        prev_loss = None
        for step in range(3):
            groups = 2
            neg_per_group = 3
            class_ids = torch.randint(0, 5, (groups,))
            noise = torch.randn(groups, neg_per_group, 4, 8, 8)
            alpha = torch.tensor([1.5, 2.5])
            positives = queue.sample_positive_grouped(class_ids, 4, device=torch.device('cpu'))
            unconditional = queue.sample_unconditional_grouped(groups, 2, device=torch.device('cpu'))
            unc_weights = torch.tensor([
                cfg_alpha_to_unconditional_weight(float(a), neg_per_group, 2)
                for a in alpha
            ])

            stats = grouped_drift_training_step(
                generator=gen, optimizer=opt,
                noise_grouped=noise, class_labels_grouped=class_ids,
                alpha_grouped=alpha, positives_grouped=positives,
                style_indices_grouped=None,
                unconditional_grouped=unconditional,
                unconditional_weight_grouped=unc_weights,
                feature_extractor=encoder,
                config=step_config,
            )
            self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])),
                f"Non-finite loss at step {step}: {stats['loss']}")
            if prev_loss is not None:
                # Just check it's finite, not necessarily decreasing in 3 steps
                pass
            prev_loss = stats["loss"]

    def test_checkpoint_roundtrip(self):
        """Queue state_dict save/restore preserves counts."""
        from training.drift_queue import ClassConditionalSampleQueue, QueueConfig
        config = QueueConfig(num_classes=3, per_class_capacity=10, global_capacity=30)
        queue = ClassConditionalSampleQueue(config)
        images = torch.randn(9, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        queue.push(images, labels)

        state = queue.state_dict()
        queue2 = ClassConditionalSampleQueue(config)
        queue2.load_state_dict(state)

        self.assertEqual(queue2.global_count(), 9)
        for label in range(3):
            self.assertEqual(queue2.class_count(label), 3)

        # Sample from restored queue should work
        positives = queue2.sample_positive_grouped(
            torch.tensor([0, 1]), 2, device=torch.device('cpu'))
        self.assertEqual(positives.shape, (2, 2, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_integration.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_drift_integration.py
git commit -m "test: add integration tests — toy training smoke + DiT feature smoke + checkpoint roundtrip"
```

---

## Task 12: Run full test suite and verify

**Step 1: Run all parity tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_parity.py -v`
Expected: ALL PASS

**Step 2: Run all integration tests**

Run: `cd /workspace/R3GAN && python -m pytest tests/test_drift_integration.py -v`
Expected: ALL PASS

**Step 3: Verify imports are clean (no drift_reference/drift_research)**

Run: `cd /workspace/R3GAN && grep -rn 'drift_reference\|drift_research\|drift_diagnostics' training/ --include='*.py' | grep -v __pycache__`
Expected: No output

**Step 4: Verify all new modules import cleanly**

Run: `cd /workspace/R3GAN && python -c "
from training.drift_field import DriftFieldConfig, compute_v, cfg_alpha_to_unconditional_weight
from training.drift_loss import DriftingLossConfig, FeatureDriftingConfig, feature_space_drifting_loss
from training.drift_grouped import GroupedBatchShapes, infer_grouped_shapes, compute_grouped_v
from training.drift_stage2 import GroupedDriftStepConfig, grouped_drift_training_step
from training.drift_queue import ClassConditionalSampleQueue, QueueConfig
from training.features.vectorize import FeatureVectorizationConfig, vectorize_feature_maps, extract_feature_maps
from training.features.extractors import TinyFeatureEncoder, TinyFeatureEncoderConfig
from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator
print('All imports OK')
"`
Expected: `All imports OK`

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify full test suite passes — faithful drift reproduction phases 1-5 complete"
```
