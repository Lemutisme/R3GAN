# First-Principles Ranking Loss Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the GAN loss system so that full-batch InfoNCE and symmetric Soft-Margin are first-class citizens, with micro-batch-correct gradient accumulation for InfoNCE.

**Architecture:** The core change is a buffering mechanism in `R3GANLoss` that collects all micro-batch data for InfoNCE mode and processes it as a single full-batch forward/backward, ensuring `logsumexp` sees all negatives. Pairwise losses (softmargin) remain micro-batch-compatible with no change. The old interpolation-based rank losses are kept as experimental. Code quality issues from the prior review are fixed.

**Tech Stack:** PyTorch, unittest

---

### Task 1: Return fake samples from AccumulateDiscriminatorGradients

Eliminates the double G forward pass in the D phase. Currently, `AccumulateDiscriminatorGradients` generates fake samples internally but doesn't expose them. The rank loss block in `loss.py` calls G again on the same noise. This wastes one full G forward per D step.

**Files:**
- Modify: `R3GAN/Trainer.py:81-118`
- Modify: `training/loss.py:159-173`
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

Add to `tests/test_adv_losses.py`:

```python
def test_discriminator_returns_fake_samples(self):
    """AccumulateDiscriminatorGradients should return 5 values including FakeSamples."""
    class SimpleG(torch.nn.Module):
        def __init__(self):
            super(SimpleG, self).__init__()
            self.fc = torch.nn.Linear(8, 3 * 4 * 4)
        def forward(self, z, c):
            return self.fc(z).reshape(z.shape[0], 3, 4, 4)
    class SimpleD(torch.nn.Module):
        def __init__(self):
            super(SimpleD, self).__init__()
            self.fc = torch.nn.Linear(3 * 4 * 4, 1)
        def forward(self, x, c):
            return self.fc(x.reshape(x.shape[0], -1)).squeeze(-1)

    g = SimpleG()
    d = SimpleD()
    trainer = AdversarialTraining(g, d)
    noise = torch.randn(4, 8)
    real = torch.randn(4, 3, 4, 4)
    cond = torch.zeros(4, 0)

    results = trainer.AccumulateDiscriminatorGradients(
        Noise=noise, RealSamples=real, Conditions=cond, Gamma=0.1, Scale=1.0,
    )
    self.assertEqual(len(results), 5)
    fake_samples = results[4]
    self.assertEqual(fake_samples.shape, (4, 3, 4, 4))
    self.assertFalse(fake_samples.requires_grad)  # Should be detached
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestAdversarialLosses.test_discriminator_returns_fake_samples -v`
Expected: FAIL -- currently returns 4 values, not 5.

**Step 3: Modify AccumulateDiscriminatorGradients to return FakeSamples**

In `R3GAN/Trainer.py`, change the return statement at end of `AccumulateDiscriminatorGradients`:

```python
# Before:
return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]

# After:
return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty, FakeSamples]]
```

**Step 4: Update the call site in loss.py**

In `training/loss.py`, change the D phase unpacking (around line 159):

```python
# Before:
AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty = self.trainer.AccumulateDiscriminatorGradients(...)

# After:
AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty, FakeSamples = self.trainer.AccumulateDiscriminatorGradients(...)
```

And replace the double G forward in the rank loss block:

```python
# Before:
with torch.no_grad():
    rank_fake_img = self.G(gen_z, real_c).detach()

# After:
rank_fake_img = FakeSamples
```

**Step 5: Run tests to verify they pass**

Run: `python -m unittest discover -s tests -v`
Expected: All tests PASS (including the new one).

**Step 6: Commit**

```bash
git add R3GAN/Trainer.py training/loss.py tests/test_adv_losses.py
git commit -m "fix: return fake samples from AccumulateDiscriminatorGradients to eliminate double G forward"
```

---

### Task 2: Add full-batch InfoNCE buffering mechanism

The core architectural change. When `adv_loss_type='infonce'`, micro-batch data is buffered in `R3GANLoss` and processed as a single full-batch after all micro-batches are collected. This ensures `logsumexp` sees all B negatives (not just `d_batch_gpu`). For `softmargin`, behavior is unchanged -- gradients accumulate per micro-batch as before.

**Gain math:** When N micro-batches each have `gain = num_gpus * batch_gpu / batch_size`, the merged batch uses `sum(gains) = N * gain = 1.0`. So `backward(1.0 * merged_loss.mean())` equals `backward(mean over full batch)`.

**Memory note:** Full-batch InfoNCE holds the entire per-GPU batch in GPU memory simultaneously (no micro-batch savings). Users needing micro-batching for memory should use `--adv-loss-type=softmargin` instead.

**Files:**
- Modify: `training/loss.py:89-237`
- Modify: `training/training_loop.py:345-347`
- Test: `tests/test_adv_losses.py`

**Step 1: Write the property test confirming micro-batch != full-batch for InfoNCE**

Add to `tests/test_adv_losses.py`:

```python
class TestInfoNCEBuffering(unittest.TestCase):
    """Test that InfoNCE processes the full batch, not per micro-batch."""

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_infonce_logsumexp_uses_full_batch(self):
        """Verify that InfoNCE D loss logsumexp denominator differs between micro and full batch."""
        real = torch.tensor([2.0, 1.5, 1.0, 0.5])
        fake = torch.tensor([-0.5, -1.0, -1.5, -2.0])
        tau = 0.1

        # Full-batch InfoNCE
        d_loss_full, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real, FakeLogits=fake, LossType='infonce', Tau=tau,
        )

        # Micro-batch InfoNCE (split into 2 micro-batches of 2)
        d_loss_micro1, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real[:2], FakeLogits=fake[:2], LossType='infonce', Tau=tau,
        )
        d_loss_micro2, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real[2:], FakeLogits=fake[2:], LossType='infonce', Tau=tau,
        )
        d_loss_micro_avg = 0.5 * d_loss_micro1.mean() + 0.5 * d_loss_micro2.mean()

        # They should NOT be equal -- micro-batch logsumexp misses half the negatives
        self.assertFalse(
            torch.allclose(d_loss_full.mean(), d_loss_micro_avg, atol=1e-4),
            'InfoNCE micro-batch should differ from full-batch (logsumexp scope differs)',
        )
```

**Step 2: Run test to verify it passes (property test)**

Run: `python -m unittest tests.test_adv_losses.TestInfoNCEBuffering.test_infonce_logsumexp_uses_full_batch -v`
Expected: PASS -- this confirms the mathematical property motivating the buffering.

**Step 3: Refactor R3GANLoss to separate the core logic**

Rename the current body of `accumulate_gradients` into `_accumulate_gradients_impl`. Make `accumulate_gradients` a dispatcher that either buffers (InfoNCE) or delegates immediately (softmargin).

In `training/loss.py`, the refactored class structure:

```python
class R3GANLoss:
    def __init__(self, G, D, augment_pipe=None, ...):
        # ... existing init ...
        self._infonce_buffer = None

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gamma, gain):
        # InfoNCE requires full-batch logits for correct logsumexp normalization.
        # When using gradient accumulation (d_batch_gpu < batch_size // num_gpus),
        # micro-batch data is buffered here and processed in finalize_accumulation().
        # For softmargin (pairwise) losses, micro-batch processing is mathematically
        # equivalent to full-batch, so no buffering is needed.
        if self.adv_loss_type == 'infonce':
            if self._infonce_buffer is None:
                self._infonce_buffer = {
                    'phase': phase, 'gamma': gamma,
                    'real_imgs': [], 'real_cs': [], 'gen_zs': [], 'gains': [],
                }
            self._infonce_buffer['real_imgs'].append(real_img)
            self._infonce_buffer['real_cs'].append(real_c)
            self._infonce_buffer['gen_zs'].append(gen_z)
            self._infonce_buffer['gains'].append(gain)
            return
        self._accumulate_gradients_impl(phase, real_img, real_c, gen_z, gamma, gain)

    def finalize_accumulation(self):
        """Flush buffered micro-batches for full-batch InfoNCE. No-op for pairwise losses."""
        if self._infonce_buffer is None:
            return
        buf = self._infonce_buffer
        self._infonce_buffer = None
        merged_real = torch.cat(buf['real_imgs'])
        merged_c = torch.cat(buf['real_cs'])
        merged_z = torch.cat(buf['gen_zs'])
        merged_gain = sum(buf['gains'])
        self._accumulate_gradients_impl(
            buf['phase'], merged_real, merged_c, merged_z, buf['gamma'], merged_gain,
        )

    def _accumulate_gradients_impl(self, phase, real_img, real_c, gen_z, gamma, gain):
        # ... existing accumulate_gradients body, unchanged ...
```

**Step 4: Add finalize_accumulation call in training_loop.py**

In `training/training_loop.py`, after the micro-batch loop (line 346), add one line before `requires_grad_(False)`:

```python
# Before (lines 345-347):
            for real_img, real_c, gen_z in zip(phase_real_img, phase_real_c, phase_gen_z):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gamma=cur_gamma, gain=num_gpus * phase.batch_gpu / batch_size)
            phase.module.requires_grad_(False)

# After:
            for real_img, real_c, gen_z in zip(phase_real_img, phase_real_c, phase_gen_z):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gamma=cur_gamma, gain=num_gpus * phase.batch_gpu / batch_size)
            loss.finalize_accumulation()
            phase.module.requires_grad_(False)
```

**Step 5: Write integration test for the buffering mechanism**

Add to `tests/test_adv_losses.py`:

```python
@unittest.skipIf(torch is None, 'PyTorch not available')
class TestR3GANLossBuffering(unittest.TestCase):
    """Integration test for R3GANLoss InfoNCE buffering."""

    def _make_loss(self, adv_loss_type='infonce', tau=0.07):
        from training.loss import R3GANLoss

        class TinyG(torch.nn.Module):
            def __init__(self):
                super(TinyG, self).__init__()
                self.fc = torch.nn.Linear(4, 3 * 4 * 4)
            def forward(self, z, c):
                return self.fc(z).reshape(z.shape[0], 3, 4, 4)

        class TinyD(torch.nn.Module):
            def __init__(self):
                super(TinyD, self).__init__()
                self.fc = torch.nn.Linear(3 * 4 * 4, 1)
            def forward(self, x, c):
                return self.fc(x.reshape(x.shape[0], -1)).squeeze(-1)

        g = TinyG()
        d = TinyD()
        return R3GANLoss(G=g, D=d, adv_loss_type=adv_loss_type, adv_tau=tau)

    def test_finalize_noop_for_softmargin(self):
        """finalize_accumulation is a no-op when adv_loss_type is softmargin."""
        loss_obj = self._make_loss(adv_loss_type='softmargin')
        loss_obj.finalize_accumulation()  # Should not raise

    def test_infonce_buffers_then_flushes(self):
        """InfoNCE should buffer micro-batches and flush on finalize."""
        loss_obj = self._make_loss(adv_loss_type='infonce')
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        # First call: should buffer, no backward yet
        loss_obj.accumulate_gradients('D', real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        # D should have no gradients yet
        for p in loss_obj.D.parameters():
            self.assertIsNone(p.grad)

        # Second call: still buffering
        loss_obj.accumulate_gradients('D', real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        for p in loss_obj.D.parameters():
            self.assertIsNone(p.grad)

        # Finalize: should process full batch and produce gradients
        loss_obj.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss_obj.D.parameters())
        self.assertTrue(has_grad, 'D should have gradients after finalize_accumulation')
```

**Step 6: Run all tests**

Run: `python -m unittest discover -s tests -v`
Expected: All tests PASS.

**Step 7: Commit**

```bash
git add training/loss.py training/training_loop.py tests/test_adv_losses.py
git commit -m "feat: add full-batch InfoNCE buffering for micro-batch-correct contrastive loss"
```

---

### Task 3: Mark old rank loss system as experimental

The interpolation-based rank losses (listmle, pairwise_logistic, pairwise_hinge) are kept functional but clearly marked as experimental. They are not the recommended path per the first-principles analysis.

**Files:**
- Modify: `training/loss.py:19-86` (rank loss functions)
- Modify: `train.py:173-186` (CLI options)

**Step 1: Add experimental markers to rank loss functions in loss.py**

Above the `listmle_loss` function, replace the existing section divider with:

```python
#----------------------------------------------------------------------------
# EXPERIMENTAL: Interpolation-based ranking losses.
# These auxiliary losses build an interpolation chain between real and fake
# images and teach D to rank them.  They are NOT the recommended approach;
# prefer --adv-loss-type=softmargin (with --adv-margin>0) or infonce instead.
# Kept for reproducibility of earlier experiments.
#----------------------------------------------------------------------------
```

**Step 2: Add CLI warning in train.py**

Inside the `if opts.rank_loss:` block, before the `desc +=` lines, add:

```python
if opts.rank_loss:
    click.echo('NOTE: --rank-loss is experimental. Prefer --adv-loss-type=infonce or --adv-loss-type=softmargin --adv-margin=1.0')
```

**Step 3: Run tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS (no functional change).

**Step 4: Commit**

```bash
git add training/loss.py train.py
git commit -m "docs: mark interpolation-based rank losses as experimental"
```

---

### Task 4: Clean up code quality issues

Fixes naming, dead code, redundant stats, and shape inconsistencies identified in the code review.

**Files:**
- Modify: `training/loss.py`
- Test: `tests/test_adv_losses.py` (verify no regressions)

**Step 4.1: Fix variable naming**

In `training/loss.py`, within the rank loss block of `_accumulate_gradients_impl`, rename all occurrences of `loss_Drank` to `loss_d_rank` (NVIDIA-derived file uses snake_case).

**Step 4.2: Remove dead else branch**

```python
# Before:
                elif self.rank_loss_type == 'pairwise_hinge':
                    loss_d_rank = pairwise_hinge_loss(rank_scores, margin=self.rank_margin)
                else:
                    loss_d_rank = listmle_loss(rank_scores)

# After:
                elif self.rank_loss_type == 'pairwise_hinge':
                    loss_d_rank = pairwise_hinge_loss(rank_scores, margin=self.rank_margin)
                else:
                    raise ValueError(f'Unknown rank_loss_type: {self.rank_loss_type}')
```

**Step 4.3: Remove redundant training stats**

`Loss/D/loss` and `Loss/D/adv` report the identical `AdversarialLoss` tensor. Remove the duplicate:

```python
# Remove this line:
training_stats.report('Loss/D/adv', AdversarialLoss)
# Keep:
training_stats.report('Loss/D/loss', AdversarialLoss)
```

**Step 4.4: Simplify run_D method**

```python
# Before:
def run_D(self, img, c, augment=True):
    if augment and self.augment_pipe is not None:
        img = self.augment_pipe(img.to(torch.float32)).to(img.dtype)
    return self.D(img, c)

# After:
def run_D(self, img, c, augment=True):
    if augment:
        img = self.preprocessor(img)
    return self.D(img, c)
```

This reuses the already-defined `self.preprocessor` (which is identity when no augmentation).

**Step 4.5: Fix d_rank_term shape**

```python
# Before:
d_rank_term = torch.zeros([], device=real_img.device, dtype=AdversarialLoss.dtype)

# After:
d_rank_term = torch.zeros_like(AdversarialLoss)
```

This makes `d_rank_term` batch-shaped `[B]`, consistent with `d_base_total`.

**Step 5: Run all tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS.

**Step 6: Commit**

```bash
git add training/loss.py
git commit -m "refactor: fix naming, remove dead code, clean up training stats in loss.py"
```

---

### Task 5: Add comprehensive tests for all loss variants

Adds coverage for softmargin with margin, InfoNCE properties, ranking loss functions, and make_rank_list.

**Files:**
- Modify: `tests/test_adv_losses.py`

**Step 1: Add softmargin with margin tests**

```python
def test_softmargin_positive_margin_penalizes_small_gap(self):
    """With margin > 0, even a correct ordering with small gap should have high loss."""
    real = torch.tensor([1.0] * 32)
    fake = torch.tensor([0.5] * 32)  # gap = 0.5, less than margin = 2.0

    d_loss, _ = AdversarialTraining._discriminator_adv_loss(
        RealLogits=real, FakeLogits=fake, LossType='softmargin', Margin=2.0,
    )
    d_loss_no_margin, _ = AdversarialTraining._discriminator_adv_loss(
        RealLogits=real, FakeLogits=fake, LossType='softmargin', Margin=0.0,
    )
    # With margin, loss should be significantly higher
    self.assertGreater(d_loss.mean().item(), d_loss_no_margin.mean().item())

def test_softmargin_gradient_never_zero(self):
    """Softmargin should always have non-zero gradient (unlike hard hinge)."""
    real = torch.tensor([10.0], requires_grad=True)
    fake = torch.tensor([0.0])

    d_loss, _ = AdversarialTraining._discriminator_adv_loss(
        RealLogits=real, FakeLogits=fake, LossType='softmargin', Margin=1.0,
    )
    d_loss.backward()
    # Even with a huge gap (10 > 1), gradient should be non-zero
    self.assertGreater(real.grad.abs().item(), 0.0)
```

**Step 2: Add InfoNCE property tests**

```python
def test_infonce_harder_negatives_increase_loss(self):
    """InfoNCE should produce higher loss when fakes are closer to reals."""
    real = torch.tensor([1.0, 1.0, 1.0, 1.0])

    easy_fake = torch.tensor([-5.0, -5.0, -5.0, -5.0])
    hard_fake = torch.tensor([0.9, 0.8, 0.7, 0.6])

    d_loss_easy, _ = AdversarialTraining._discriminator_adv_loss(
        RealLogits=real, FakeLogits=easy_fake, LossType='infonce', Tau=0.1,
    )
    d_loss_hard, _ = AdversarialTraining._discriminator_adv_loss(
        RealLogits=real, FakeLogits=hard_fake, LossType='infonce', Tau=0.1,
    )
    self.assertGreater(d_loss_hard.mean().item(), d_loss_easy.mean().item())

def test_infonce_symmetry(self):
    """G and D InfoNCE losses should be symmetric mirrors."""
    real = torch.randn(16)
    fake = torch.randn(16)
    tau = 0.1

    d_loss, d_rel = AdversarialTraining._discriminator_adv_loss(
        RealLogits=real, FakeLogits=fake, LossType='infonce', Tau=tau,
    )
    g_loss, g_rel = AdversarialTraining._generator_adv_loss(
        FakeLogits=fake, RealLogits=real, LossType='infonce', Tau=tau,
    )
    # Relativistic logits should be mirror images
    self.assertTrue(torch.allclose(d_rel, -g_rel))
    # Both should be finite
    self.assertTrue(torch.isfinite(d_loss).all())
    self.assertTrue(torch.isfinite(g_loss).all())
```

**Step 3: Add ranking loss function tests**

```python
class TestRankingLosses(unittest.TestCase):
    """Tests for experimental interpolation-based ranking losses."""

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_listmle_correct_order_low_loss(self):
        from training.loss import listmle_loss
        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        loss = listmle_loss(scores)
        scores_rev = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss_rev = listmle_loss(scores_rev)
        self.assertGreater(loss_rev.item(), loss.item())

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_pairwise_hinge_zero_loss_when_margin_satisfied(self):
        from training.loss import pairwise_hinge_loss
        scores = torch.tensor([[10.0, 5.0, 0.0]])  # gaps all > margin=1
        loss = pairwise_hinge_loss(scores, margin=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_pairwise_logistic_matches_rpgan_for_k2(self):
        """With K=2, pairwise_logistic should match the base RpGAN loss."""
        from training.loss import pairwise_logistic_loss
        real_score = torch.tensor([2.0])
        fake_score = torch.tensor([-1.0])
        scores = torch.stack([real_score, fake_score], dim=-1)  # [1, 2]

        rank_loss = pairwise_logistic_loss(scores)
        rpgan_loss = F.softplus(-(real_score - fake_score)).mean()
        self.assertTrue(torch.allclose(rank_loss, rpgan_loss, atol=1e-5))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_make_rank_list_endpoints(self):
        from training.loss import make_rank_list
        real = torch.randn(2, 3, 4, 4)
        fake = torch.randn(2, 3, 4, 4)
        result = make_rank_list(real, fake, k=4, mode='intrpl', alpha_dist='linear')
        self.assertEqual(result.shape, (2, 4, 3, 4, 4))
        self.assertTrue(torch.allclose(result[:, 0], real))
        self.assertTrue(torch.allclose(result[:, -1], fake))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_make_rank_list_alpha_monotonic(self):
        from training.loss import make_rank_list
        real = torch.ones(1, 1, 2, 2)
        fake = torch.zeros(1, 1, 2, 2)
        for dist in ['linear', 'cosine', 'random']:
            result = make_rank_list(real, fake, k=8, mode='intrpl', alpha_dist=dist)
            means = result[0, :, 0, 0, 0]
            for i in range(len(means) - 1):
                self.assertGreaterEqual(means[i].item(), means[i + 1].item(),
                                        f'alpha_dist={dist}: not monotonic at position {i}')
```

**Step 4: Run all tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS.

**Step 5: Commit**

```bash
git add tests/test_adv_losses.py
git commit -m "test: add comprehensive tests for softmargin, infonce, ranking losses, and make_rank_list"
```

---

### Task 6: Document InfoNCE micro-batch limitation and non-aug GP recommendation

Add inline documentation for the two key design decisions.

**Files:**
- Modify: `training/loss.py` (add comments)
- Modify: `train.py` (update help text)

**Step 1: Add InfoNCE documentation in loss.py**

The comment from Step 3 of Task 2 (added to the `accumulate_gradients` method) serves as the primary documentation. Verify it exists.

**Step 2: Update non-aug GP help text in train.py**

```python
# Before:
@click.option('--non-aug-gp', help='Compute R1/R2 on non-augmented samples', metavar='BOOL', type=bool, default=False, show_default=True)

# After:
@click.option('--non-aug-gp', help='Compute R1/R2 on non-augmented samples (recommended for convergence guarantees; adds extra D forward passes)', metavar='BOOL', type=bool, default=False, show_default=True)
```

**Step 3: Run tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add training/loss.py train.py
git commit -m "docs: document InfoNCE full-batch requirement and non-aug GP recommendation"
```

---

## Summary

| Task | Files | What changes |
|------|-------|-------------|
| 1. Return fake samples | `Trainer.py`, `loss.py`, tests | Eliminate double G forward in D phase |
| 2. Full-batch InfoNCE | `loss.py`, `training_loop.py`, tests | Buffering mechanism for micro-batch-correct InfoNCE |
| 3. Experimental markers | `loss.py`, `train.py` | Mark old rank losses as experimental |
| 4. Code quality | `loss.py` | Fix naming, dead code, redundant stats, shape issues |
| 5. Tests | `test_adv_losses.py` | Comprehensive coverage for all loss variants |
| 6. Documentation | `loss.py`, `train.py` | Inline docs for design decisions |

## What is NOT changed

- `R3GAN/Trainer.py` loss implementations (softmargin, infonce) -- already correct and symmetric
- Training loop structure (phases, optimizer steps, EMA) -- only one line added
- Experimental rank loss system -- kept functional, just marked
- Default hyperparameters -- no defaults changed
- Non-aug GP default -- kept as False per user decision
