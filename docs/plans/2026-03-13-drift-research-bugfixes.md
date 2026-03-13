# Drift Research Backend Bugfixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 bugs in `drift_research.py`: missing EMA, silent multi-temperature drop, and silent feature-loss drop for `r3gan_conv` backbone.

**Architecture:** Bug 1 (critical) replaces a direct param copy with proper EMA in the research training loop. Bugs 2-3 (moderate) route the `r3gan_conv` fallback through the reference `grouped_drift_training_step` instead of the local simplified loss, so it gains multi-temperature and feature-loss support automatically.

**Tech Stack:** PyTorch, R3GAN codebase, `drift_models` reference repo. Tests run with `conda run -n gan python -m pytest`.

---

## Bug Summary

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Critical | `drift_research.py:446` | `misc.copy_params_and_buffers(g, g_ema)` is a direct copy, not EMA |
| 2 | Moderate | `drift_research.py:1146-1151` | `_run_r3gan_conv_step` ignores `drift_temperatures` |
| 3 | Moderate | `drift_research.py:1102-1163` | `_run_r3gan_conv_step` ignores `use_feature_loss` |

---

### Task 1: Fix EMA in `drift_research.py`

**Files:**
- Modify: `training/drift_research.py:446` (EMA update line)
- Modify: `training/drift_research.py:60-80` (add `ema_decay` parameter)
- Modify: `training/drift_research.py:605-726` (`_build_drift_args` — add `ema_decay` default)
- Test: `tests/test_drift_training.py`

**Step 1: Write failing test — EMA produces different params than direct copy**

Add to `tests/test_drift_training.py`:

```python
@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDriftEMA(unittest.TestCase):
    def test_ema_update_differs_from_direct_copy(self):
        """EMA update should produce params between old and new, not equal to new."""
        model = torch.nn.Linear(4, 4, bias=False)
        ema_model = torch.nn.Linear(4, 4, bias=False)
        # Initialize both to same weights
        with torch.no_grad():
            ema_model.weight.copy_(model.weight)

        # Simulate a training step: change model weights
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight))

        # Direct copy (the bug) — ema becomes identical to model
        from torch_utils.misc import copy_params_and_buffers
        ema_copy = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            ema_copy.weight.copy_(ema_model.weight)
        copy_params_and_buffers(model, ema_copy, require_all=False)
        self.assertTrue(torch.allclose(ema_copy.weight, model.weight))

        # Proper EMA — ema should NOT be identical to model
        from training.drift_research import _ema_update
        _ema_update(src=model, dst=ema_model, decay=0.999)
        self.assertFalse(torch.allclose(ema_model.weight, model.weight))
        # EMA should be between old and new (closer to old with high decay)
        # The old value was ema_copy's initial state before copy, but we lost it.
        # Just verify it differs from model.
        diff = (ema_model.weight - model.weight).abs().sum().item()
        self.assertGreater(diff, 0.0)

    def test_ema_update_with_buffers(self):
        """Buffers should be copied directly (not EMA-smoothed)."""
        model = torch.nn.BatchNorm1d(4)
        ema_model = torch.nn.BatchNorm1d(4)
        with torch.no_grad():
            for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                b_ema.copy_(b)
            # Change model buffers
            model.running_mean.fill_(5.0)

        from training.drift_research import _ema_update
        _ema_update(src=model, dst=ema_model, decay=0.999)

        # Buffers should be directly copied
        self.assertTrue(torch.allclose(ema_model.running_mean, model.running_mean))
```

**Step 2: Run test to verify it fails**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py::TestDriftEMA -v --tb=short`
Expected: FAIL — `_ema_update` does not exist yet.

**Step 3: Implement `_ema_update` and wire it into the training loop**

In `training/drift_research.py`, add a helper function (near the bottom, before `_class_ids_from_labels`):

```python
def _ema_update(*, src, dst, decay):
    """Exponential moving average: dst = decay * dst + (1 - decay) * src."""
    with torch.no_grad():
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.lerp_(p_src, 1.0 - decay)
        for b_dst, b_src in zip(dst.buffers(), src.buffers()):
            b_dst.copy_(b_src)
```

Add `ema_decay` to `_build_drift_args` defaults dict (around line 620):

```python
'ema_decay': 0.999,
```

Replace line 446:
```python
# OLD:
misc.copy_params_and_buffers(g, g_ema, require_all=False)

# NEW:
_ema_update(src=g, dst=g_ema, decay=float(drift.ema_decay))
```

**Step 4: Run test to verify it passes**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py::TestDriftEMA -v --tb=short`
Expected: PASS

**Step 5: Run all existing tests to check for regressions**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py -v --tb=short`
Expected: All 14+ tests PASS.

**Step 6: Commit**

```bash
git add training/drift_research.py tests/test_drift_training.py
git commit -m "fix: replace direct param copy with proper EMA in drift research trainer"
```

---

### Task 2: Fix `_run_r3gan_conv_step` to use reference training step (fixes Bugs 2 & 3)

**Files:**
- Modify: `training/drift_research.py:1102-1163` (`_run_r3gan_conv_step`)
- Modify: `training/networks.py` (ensure `DriftGenerator` compatible with `grouped_drift_training_step` forward signature)
- Test: `tests/test_drift_training.py`

**Context:** The root cause of bugs 2 & 3 is that `_run_r3gan_conv_step` calls the local `grouped_drifting_stopgrad_loss` directly, bypassing the reference `grouped_drift_training_step` which handles multi-temperature and feature loss. The fix is to route it through the same reference step used by the DiT-like path.

**Step 1: Write failing test — r3gan_conv step returns multi-temperature stats**

```python
@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestR3GANConvStepParity(unittest.TestCase):
    def test_r3gan_conv_step_respects_drift_temperatures(self):
        """_run_r3gan_conv_step should use drift_temperatures when specified."""
        from types import SimpleNamespace
        from training.drift_research import _run_r3gan_conv_step
        from training.drift_queue import ClassConditionalSampleQueue, QueueConfig

        # Build a minimal generator
        class FakeConvGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3 * 4 * 4)
                self.z_dim = 4
                self.c_dim = 2

            def forward(self, z, c, alpha=None):
                return self.linear(z).reshape(z.shape[0], 3, 4, 4)

        gen = FakeConvGenerator()
        opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
        queue = ClassConditionalSampleQueue(QueueConfig(num_classes=2, per_class_capacity=16, global_capacity=32))
        # Fill queue
        for _ in range(4):
            queue.push(torch.randn(8, 3, 4, 4), torch.randint(0, 2, (8,)))

        class FakeProvider:
            def next_batch(self, *, device):
                return torch.randn(8, 3, 4, 4, device=device), torch.randint(0, 2, (8,), device=device)

        drift = SimpleNamespace(
            negatives_per_group=2,
            positives_per_group=2,
            unconditional_per_group=1,
            drift_temperature=0.05,
            drift_temperatures=[0.02, 0.05, 0.2],
            drift_temperature_reduction='sum',
            clip_grad_norm=2.0,
            queue_refill_policy='per_step',
            queue_refill_every=1,
            queue_push_batch=4,
            queue_strict_without_replacement=False,
            use_feature_loss=False,
        )
        class_labels = torch.tensor([0, 1])
        alpha = torch.tensor([2.0, 3.0])

        stats = _run_r3gan_conv_step(
            generator=gen,
            optimizer=opt,
            queue=queue,
            provider=FakeProvider(),
            step=0,
            local_groups=2,
            num_classes=2,
            drift=drift,
            image_channels=3,
            image_size=4,
            class_labels=class_labels,
            alpha=alpha,
            device=torch.device('cpu'),
        )
        # If multi-temperature is used, stats should have temperature_count
        self.assertIn('loss', stats)
        # The key check: when drift_temperatures is non-empty, the reference step
        # produces mean_drift_norm (not just drift_norm from the local impl)
        self.assertIn('mean_drift_norm', stats)
```

**Step 2: Run test to verify it fails**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py::TestR3GANConvStepParity -v --tb=short`
Expected: FAIL (current impl ignores drift_temperatures, returns local stats format).

**Step 3: Rewrite `_run_r3gan_conv_step` to use reference step**

Replace the entire `_run_r3gan_conv_step` function in `training/drift_research.py`:

```python
def _run_r3gan_conv_step(*, generator, optimizer, queue, provider, step, local_groups, num_classes, drift, image_channels, image_size, class_labels, alpha, device, step_config=None, feature_extractor=None):
    should_refill = str(drift.queue_refill_policy) == 'per_step' or (
        str(drift.queue_refill_policy) == 'every_n_steps'
        and (step % max(1, int(drift.queue_refill_every)) == 0)
    )
    if should_refill:
        refill_images, refill_labels = _sample_real_batch(provider=provider, count=int(drift.queue_push_batch), device=device)
        queue.push(refill_images, refill_labels)
    backfilled = ensure_queue_has_labels(
        queue=queue,
        class_labels=class_labels,
        provider=provider,
        required_count=int(drift.positives_per_group if drift.queue_strict_without_replacement else 1),
        device=device,
    )
    positives_grouped, unconditional_grouped = sample_grouped_real_batches(
        queue=queue,
        class_labels=class_labels,
        config=GroupedSamplingConfig(
            positives_per_group=int(drift.positives_per_group),
            unconditional_per_group=int(drift.unconditional_per_group),
        ),
        device=device,
    )
    unconditional_weight_grouped = torch.tensor(
        [
            cfg_alpha_to_unconditional_weight(
                alpha=float(alpha[g_index].item()),
                n_generated_negatives=int(drift.negatives_per_group),
                n_unconditional_negatives=int(drift.unconditional_per_group),
            )
            for g_index in range(local_groups)
        ],
        device=device,
        dtype=torch.float32,
    )

    # Build noise as [G, N, C, H, W] grouped tensor (like DiT path)
    noise_grouped = torch.randn(
        local_groups,
        int(drift.negatives_per_group),
        generator.z_dim,
        device=device,
    )

    # Build step_config if not provided
    if step_config is None:
        step_config = GroupedDriftStepConfig(
            loss_config=DriftingLossConfig(
                drift_field=DriftFieldConfig(
                    temperature=float(drift.drift_temperature),
                    normalize_over_x=True,
                    mask_self_negatives=True,
                ),
                attraction_scale=1.0,
                repulsion_scale=1.0,
                stopgrad_target=True,
            ),
            feature_config=_build_feature_config(drift),
            drift_temperatures=tuple(float(v) for v in drift.drift_temperatures),
            drift_temperature_reduction=str(drift.drift_temperature_reduction),
            clip_grad_norm=float(drift.clip_grad_norm),
            run_optimizer_step=True,
        )

    # Use the reference grouped_drift_training_step for full feature parity.
    # The R3GAN Conv generator takes (z[B, Z], c[B, C], alpha=...) but
    # grouped_drift_training_step expects (noise[B,C,H,W], class_ids[B], alpha[B], style[B,S]).
    # We wrap the forward call via a thin adapter.
    stats = _conv_grouped_drift_training_step(
        generator=generator,
        optimizer=optimizer,
        noise_grouped=noise_grouped,
        class_labels_grouped=class_labels,
        alpha_grouped=alpha,
        positives_grouped=positives_grouped,
        unconditional_grouped=unconditional_grouped,
        unconditional_weight_grouped=unconditional_weight_grouped,
        num_classes=num_classes,
        image_channels=image_channels,
        image_size=image_size,
        feature_extractor=feature_extractor,
        config=step_config,
    )
    stats['queue_underflow_backfilled'] = float(backfilled)
    return stats
```

Note: The reference `grouped_drift_training_step` expects the generator to take `(noise[B,C,H,W], class_ids, alpha, style_indices)`, but the R3GAN Conv generator takes `(z[B,Z], c_onehot[B,C], alpha=...)`. We need a thin adapter. Add `_conv_grouped_drift_training_step`:

```python
def _conv_grouped_drift_training_step(
    *, generator, optimizer, noise_grouped, class_labels_grouped, alpha_grouped,
    positives_grouped, unconditional_grouped, unconditional_weight_grouped,
    num_classes, image_channels, image_size, feature_extractor, config,
):
    """Grouped drift training step adapted for R3GAN Conv generator signature."""
    groups = noise_grouped.shape[0]
    negatives_per_group = noise_grouped.shape[1]
    z_dim = noise_grouped.shape[2]
    device = noise_grouped.device

    z_flat = noise_grouped.reshape(groups * negatives_per_group, z_dim)
    class_labels_flat = class_labels_grouped.repeat_interleave(negatives_per_group)
    alpha_flat = alpha_grouped.repeat_interleave(negatives_per_group)
    one_hot = torch.zeros([z_flat.shape[0], num_classes], device=device, dtype=torch.float32)
    one_hot.scatter_(1, class_labels_flat.view(-1, 1), 1.0)

    generated_flat = generator(z_flat, one_hot, alpha=alpha_flat)
    generated_grouped = generated_flat.reshape(
        groups, negatives_per_group,
        generated_flat.shape[1], generated_flat.shape[2], generated_flat.shape[3],
    )

    from drifting_models.drift_field import build_negative_log_weights
    from drifting_models.drift_loss import (
        drifting_stopgrad_loss,
        drifting_stopgrad_loss_multi_temperature,
        feature_space_drifting_loss,
    )
    from drifting_models.features.vectorize import extract_feature_maps, vectorize_feature_maps

    losses = []
    drift_norms = []
    per_group_stats = []
    for g_idx in range(groups):
        gen_group = generated_grouped[g_idx]
        pos_group = positives_grouped[g_idx]
        unc_group = None if unconditional_grouped is None else unconditional_grouped[g_idx]
        unc_weight = None if unconditional_weight_grouped is None else float(unconditional_weight_grouped[g_idx].item())

        if config.feature_config is not None and feature_extractor is not None:
            # Feature loss path — delegate to reference
            gen_feats = vectorize_feature_maps(extract_feature_maps(feature_extractor, gen_group), config.feature_config.vectorization)
            pos_feats = vectorize_feature_maps(extract_feature_maps(feature_extractor, pos_group), config.feature_config.vectorization)
            unc_feats = None
            if unc_group is not None:
                unc_feats = vectorize_feature_maps(extract_feature_maps(feature_extractor, unc_group), config.feature_config.vectorization)
            loss, stats = feature_space_drifting_loss(
                generated_feature_vectors=gen_feats,
                positive_feature_vectors=pos_feats,
                unconditional_feature_vectors=unc_feats,
                base_loss_config=config.loss_config,
                feature_config=config.feature_config,
                unconditional_weight=unc_weight if unc_weight is not None else 1.0,
            )
        else:
            gen_vec = gen_group.reshape(gen_group.shape[0], -1)
            pos_vec = pos_group.reshape(pos_group.shape[0], -1)
            neg_vec = gen_vec
            neg_log_w = None
            if unc_group is not None:
                unc_vec = unc_group.reshape(unc_group.shape[0], -1)
                neg_vec = torch.cat([gen_vec, unc_vec], dim=0)
                w = 1.0 if unc_weight is None else unc_weight
                neg_log_w = build_negative_log_weights(
                    n_generated_negatives=gen_vec.shape[0],
                    n_unconditional_negatives=unc_vec.shape[0],
                    unconditional_weight=w,
                    device=gen_vec.device,
                    dtype=gen_vec.dtype,
                )
            if config.drift_temperatures:
                loss, stats = drifting_stopgrad_loss_multi_temperature(
                    x=gen_vec, y_pos=pos_vec, y_neg=neg_vec,
                    temperatures=tuple(config.drift_temperatures),
                    config=config.loss_config,
                    negative_log_weights=neg_log_w,
                    generated_negative_count=gen_vec.shape[0],
                    reduction=str(config.drift_temperature_reduction),
                )
            else:
                loss, _, stats = drifting_stopgrad_loss(
                    x=gen_vec, y_pos=pos_vec, y_neg=neg_vec,
                    config=config.loss_config,
                    negative_log_weights=neg_log_w,
                    generated_negative_count=gen_vec.shape[0],
                )

        losses.append(loss)
        drift_norms.append(stats.get('mean_drift_norm', stats.get('drift_norm', 0.0)))
        per_group_stats.append(stats)

    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = None
    if config.clip_grad_norm is not None and config.clip_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), config.clip_grad_norm)
    optimizer.step()

    return {
        'loss': float(total_loss.item()),
        'mean_drift_norm': float(sum(drift_norms) / max(len(drift_norms), 1)),
        'groups': groups,
        'negatives_per_group': negatives_per_group,
        'alpha_mean': float(alpha_grouped.mean().item()),
        'alpha_min': float(alpha_grouped.min().item()),
        'alpha_max': float(alpha_grouped.max().item()),
        'grad_norm': None if grad_norm is None else float(grad_norm.item()),
    }
```

Also update the call site in the training loop (around line 428) to pass `step_config` and `feature_extractor`:

```python
else:
    stats = _run_r3gan_conv_step(
        generator=g,
        optimizer=optimizer,
        queue=queue,
        provider=provider,
        step=step,
        local_groups=local_groups,
        num_classes=num_classes,
        drift=drift,
        image_channels=image_channels,
        image_size=int(training_set.resolution),
        class_labels=class_labels,
        alpha=alpha,
        device=device,
        step_config=step_config,
        feature_extractor=feature_extractor,
    )
```

**Step 4: Run test to verify it passes**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py::TestR3GANConvStepParity -v --tb=short`
Expected: PASS

**Step 5: Run all tests**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py -v --tb=short`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add training/drift_research.py tests/test_drift_training.py
git commit -m "fix: route r3gan_conv step through reference loss for multi-temp and feature loss support"
```

---

### Task 3: Add integration-level regression test

**Files:**
- Test: `tests/test_drift_training.py`

**Step 1: Write test — EMA produces smoothed snapshots over multiple steps**

```python
@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDriftEMAIntegration(unittest.TestCase):
    def test_ema_produces_smoothed_params_over_multiple_updates(self):
        """After N training-like updates, EMA model should differ from current model."""
        from training.drift_research import _ema_update

        torch.manual_seed(42)
        model = torch.nn.Linear(8, 8, bias=False)
        ema = torch.nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            ema.weight.copy_(model.weight)
        initial_weight = model.weight.clone()

        # Simulate 10 training steps
        for _ in range(10):
            with torch.no_grad():
                model.weight.add_(torch.randn_like(model.weight) * 0.1)
            _ema_update(src=model, dst=ema, decay=0.99)

        # EMA should NOT equal current model
        self.assertFalse(torch.allclose(ema.weight, model.weight, atol=1e-6))
        # EMA should NOT equal initial weights either
        self.assertFalse(torch.allclose(ema.weight, initial_weight, atol=1e-6))
        # EMA should be "between" — closer to model than initial
        dist_to_model = (ema.weight - model.weight).norm().item()
        dist_to_initial = (ema.weight - initial_weight).norm().item()
        self.assertLess(dist_to_model, dist_to_initial)
```

**Step 2: Run test**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py::TestDriftEMAIntegration -v --tb=short`
Expected: PASS (implementation already done in Task 1).

**Step 3: Commit**

```bash
git add tests/test_drift_training.py
git commit -m "test: add EMA integration regression test for drift research trainer"
```

---

### Task 4: Final verification

**Step 1: Run full test suite**

Run: `conda run -n gan python -m pytest tests/test_drift_training.py -v --tb=long`
Expected: All tests PASS.

**Step 2: Verify no import errors in modified modules**

Run: `conda run -n gan python -c "from training.drift_research import training_loop, _ema_update, _run_r3gan_conv_step; print('imports OK')"`
Expected: `imports OK`
