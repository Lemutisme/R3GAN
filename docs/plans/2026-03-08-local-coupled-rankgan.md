# Local-Coupled RankGAN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the local-coupled, delta-centric RankGAN design from `docs/design.md` — replacing global all-pairs InfoNCE with local kNN-coupled pairwise + listwise losses, adding asymmetric D/G listwise weights, and a 3-phase training curriculum.

**Architecture:** A sparse coupling `π_ij` is built via kNN on stop-gradient, L2-normalized clean (non-augmented) discriminator features. Both pairwise and listwise losses then use this coupling to weight delta comparisons `Δ_ij = s(aug(real_i)) - s(aug(fake_j))` over each fake's local real neighborhood. The existing full-batch buffering/replay infrastructure handles gradient accumulation. When `coupling_k=0`, behavior falls back to current diagonal pairing + global InfoNCE (full backward compatibility).

**Tech Stack:** PyTorch, unittest

---

## Design Review Summary

### What's already implemented (no changes needed)

1. Single scalar critic `s_ψ(x) ∈ R` — no independent rank head
2. Pairwise delta primitive `Δ = s(real) - s(fake)` with softplus losses
3. Global all-pairs InfoNCE with full-batch micro-batch buffering + gradient replay
4. R1/R2 zero-centered gradient penalties (including non-aug GP option)
5. Local-rank gap prior (kNN feature ordering among fakes)
6. Path-rank (interpolation-based) marked as experimental auxiliary
7. Clean feature extraction via `run_D(augment=False, return_features=True)`
8. Multi-GPU all-gather for distributed training

### What needs to be built

| # | Component | Design Doc Section |
|---|-----------|-------------------|
| 1 | Sparse kNN coupling `π_ij` on clean features | "局部 real 邻域" |
| 2 | Local-coupled pairwise loss `Σ π̃_ij softplus(m - Δ_ij)` | `L_D^pair`, `L_G^pair` |
| 3 | Local-coupled listwise loss `log(1 + Σ π̃_ij exp(-Δ/τ))` | `L_D^list`, `L_G^list` |
| 4 | Asymmetric D/G listwise weights `λ_ℓ^G ≤ λ_ℓ^D` | Total loss equations |
| 5 | 3-phase training curriculum (pairwise warmup → D-list → G-list) | Phases 1-3 |
| 6 | CLI flags + preset updates | — |

### Design decisions made in this plan

- **Coupling normalization:** softmax of cosine similarities (proximity-weighted, not uniform)
- **kNN first, OT later:** Interface supports future OT coupling, but v1 uses kNN
- **`coupling_k=0` as backward-compat:** No coupling = current behavior (diagonal pair + global InfoNCE)
- **Numerically stable listwise:** log-sum-exp trick for `log(1 + Σ π̃ exp(−Δ/τ))`
- **Mobility-aware R2:** Deferred to optional Task 12 (unproven, not canonical)

---

## Phase A: Pure Loss Functions

### Task 1: Coupling builder and local delta

**Files:**
- Modify: `training/loss.py` (add after `infonce_generator_loss`, ~line 153)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
class TestLocalCoupling(unittest.TestCase):

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_build_local_coupling_shapes(self):
        from training.loss import build_local_coupling
        real_feat = torch.randn(8, 16)
        fake_feat = torch.randn(4, 16)
        indices, weights = build_local_coupling(real_feat, fake_feat, k=3)
        self.assertEqual(indices.shape, (4, 3))
        self.assertEqual(weights.shape, (4, 3))
        self.assertTrue((indices >= 0).all() and (indices < 8).all())
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_build_local_coupling_k_clamped_to_num_reals(self):
        from training.loss import build_local_coupling
        real_feat = torch.randn(2, 8)
        fake_feat = torch.randn(5, 8)
        indices, weights = build_local_coupling(real_feat, fake_feat, k=10)
        self.assertEqual(indices.shape[1], 2)  # clamped to num reals

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_build_local_coupling_nearest_neighbor_is_first(self):
        from training.loss import build_local_coupling
        real_feat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        fake_feat = torch.tensor([[0.9, 0.1]])  # closest to real[0]
        indices, weights = build_local_coupling(real_feat, fake_feat, k=2)
        self.assertEqual(indices[0, 0].item(), 0)  # nearest = real[0]
        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_delta_values(self):
        from training.loss import local_delta
        real_scores = torch.tensor([3.0, 1.0, 5.0])
        fake_scores = torch.tensor([2.0, 4.0])
        neighbor_indices = torch.tensor([[0, 2], [1, 0]])  # fake0→real{0,2}, fake1→real{1,0}
        delta = local_delta(real_scores, fake_scores, neighbor_indices)
        expected = torch.tensor([[3.0 - 2.0, 5.0 - 2.0], [1.0 - 4.0, 3.0 - 4.0]])
        self.assertTrue(torch.allclose(delta, expected))
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling -v`
Expected: FAIL with ImportError (functions not defined)

**Step 3: Write implementation**

In `training/loss.py`, after `infonce_generator_loss` (line 153), add:

```python
def build_local_coupling(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    k: int,
) -> tuple:
    """Build sparse kNN coupling from fakes to nearest reals in feature space.

    Returns (neighbor_indices [B_f, k], coupling_weights [B_f, k]).
    """
    real_norm = F.normalize(real_features.detach().to(torch.float32), dim=1)
    fake_norm = F.normalize(fake_features.detach().to(torch.float32), dim=1)
    sim = torch.matmul(fake_norm, real_norm.t())
    k = min(k, real_norm.shape[0])
    topk_sim, topk_idx = sim.topk(k, dim=1)
    coupling_weights = F.softmax(topk_sim, dim=1)
    return topk_idx, coupling_weights


def local_delta(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """Local critic differences Δ_ij = s(real_{N(j,i)}) - s(fake_j). Returns [B_f, k]."""
    return real_scores[neighbor_indices] - fake_scores.unsqueeze(1)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: add build_local_coupling and local_delta functions"
```

---

### Task 2: Local-coupled pairwise loss functions

**Files:**
- Modify: `training/loss.py` (add after `local_delta`)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_pairwise_d_loss_correct_ordering_low(self):
        from training.loss import local_delta, local_pairwise_discriminator_loss
        real_scores = torch.tensor([5.0, 4.0])
        fake_scores = torch.tensor([0.0])
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.6, 0.4]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss = local_pairwise_discriminator_loss(delta, weights, margin=0.0)
        self.assertEqual(loss.shape, (1,))
        self.assertGreater(loss.item(), 0)  # softplus always > 0 but small

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_pairwise_g_loss_mirror(self):
        from training.loss import local_delta, local_pairwise_discriminator_loss, local_pairwise_generator_loss
        real_scores = torch.tensor([2.0, 3.0])
        fake_scores = torch.tensor([1.0, 2.5])
        indices = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([[0.5, 0.5], [0.7, 0.3]])
        delta = local_delta(real_scores, fake_scores, indices)
        d_loss = local_pairwise_discriminator_loss(delta, weights, margin=0.0)
        g_loss = local_pairwise_generator_loss(delta, weights, margin=0.0)
        # With large positive delta, D loss small and G loss large
        # With delta~0, both ~log(2)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_pairwise_margin_increases_loss(self):
        from training.loss import local_delta, local_pairwise_discriminator_loss
        real_scores = torch.tensor([3.0, 4.0])
        fake_scores = torch.tensor([1.0])
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss_m0 = local_pairwise_discriminator_loss(delta, weights, margin=0.0)
        loss_m2 = local_pairwise_discriminator_loss(delta, weights, margin=2.0)
        self.assertGreater(loss_m2.item(), loss_m0.item())
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling.test_local_pairwise_d_loss_correct_ordering_low -v`
Expected: FAIL

**Step 3: Write implementation**

```python
def local_pairwise_discriminator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Coupling-weighted pairwise D loss. Returns [B_f] per-fake losses."""
    return (coupling_weights * F.softplus(margin - delta)).sum(dim=1)


def local_pairwise_generator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Coupling-weighted pairwise G loss. Returns [B_f] per-fake losses."""
    return (coupling_weights * F.softplus(margin + delta)).sum(dim=1)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: add local-coupled pairwise D/G loss functions"
```

---

### Task 3: Local-coupled listwise loss functions

**Files:**
- Modify: `training/loss.py` (add after local pairwise losses)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_d_loss_correct_ordering_low(self):
        from training.loss import local_delta, local_listwise_discriminator_loss
        real_scores = torch.tensor([10.0, 8.0])
        fake_scores = torch.tensor([0.0])
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss = local_listwise_discriminator_loss(delta, weights, tau=0.1)
        self.assertLess(loss.item(), 0.01)  # large positive delta → loss ≈ 0

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_wrong_ordering_high(self):
        from training.loss import local_delta, local_listwise_discriminator_loss
        real_scores = torch.tensor([0.0, -1.0])
        fake_scores = torch.tensor([5.0])  # fake higher than reals
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss = local_listwise_discriminator_loss(delta, weights, tau=0.1)
        self.assertGreater(loss.item(), 1.0)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_numerical_stability(self):
        from training.loss import local_delta, local_listwise_discriminator_loss
        real_scores = torch.tensor([-100.0, -200.0])
        fake_scores = torch.tensor([100.0])  # extreme negative delta
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss = local_listwise_discriminator_loss(delta, weights, tau=0.01)
        self.assertTrue(torch.isfinite(loss).all(), 'Loss must be finite even with extreme values')

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_reduces_to_global_with_all_neighbors(self):
        """With k=B_r and uniform weights, local listwise ≈ global InfoNCE row."""
        from training.loss import (
            local_delta, local_listwise_discriminator_loss,
            allpairs_delta, infonce_discriminator_loss,
        )
        torch.manual_seed(42)
        real_scores = torch.randn(4)
        fake_scores = torch.randn(4)
        tau = 0.1
        # Global InfoNCE (all-pairs)
        global_loss = infonce_discriminator_loss(allpairs_delta(real_scores, fake_scores), tau)
        # Local with k=4 (all reals) and uniform weights
        indices = torch.arange(4).unsqueeze(0).expand(4, -1)
        weights = torch.ones(4, 4) / 4.0
        delta = local_delta(real_scores, fake_scores, indices)
        local_loss = local_listwise_discriminator_loss(delta, weights, tau)
        # They should NOT be identical (weighting differs) but should have similar
        # directional behavior - both should be small when D wins
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling.test_local_listwise_d_loss_correct_ordering_low -v`
Expected: FAIL

**Step 3: Write implementation**

```python
def local_listwise_discriminator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Local-coupled listwise D loss: log(1 + Σ π̃_ij exp(-Δ_ij/τ)). Returns [B_f]."""
    if tau <= 0:
        raise ValueError(f"Temperature must be positive, got {tau}")
    neg_scaled = -delta / tau
    max_val = neg_scaled.detach().max(dim=1, keepdim=True).values.clamp(min=0)
    stable_sum = torch.exp(-max_val.squeeze(1)) + (
        coupling_weights * torch.exp(neg_scaled - max_val)
    ).sum(dim=1)
    return max_val.squeeze(1) + torch.log(stable_sum)


def local_listwise_generator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Local-coupled listwise G loss: log(1 + Σ π̃_ij exp(+Δ_ij/τ)). Returns [B_f]."""
    if tau <= 0:
        raise ValueError(f"Temperature must be positive, got {tau}")
    pos_scaled = delta / tau
    max_val = pos_scaled.detach().max(dim=1, keepdim=True).values.clamp(min=0)
    stable_sum = torch.exp(-max_val.squeeze(1)) + (
        coupling_weights * torch.exp(pos_scaled - max_val)
    ).sum(dim=1)
    return max_val.squeeze(1) + torch.log(stable_sum)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: add local-coupled listwise D/G loss functions with numerical stability"
```

---

### Task 4: Combined gradient helper for replay

**Files:**
- Modify: `training/loss.py` (add after local listwise losses)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_coupled_d_grads_match_autograd(self):
        from training.loss import local_coupled_discriminator_loss_with_grads
        from training.loss import (
            build_local_coupling, local_delta,
            local_pairwise_discriminator_loss, local_listwise_discriminator_loss,
        )
        torch.manual_seed(42)
        real_scores = torch.randn(6, requires_grad=True)
        fake_scores = torch.randn(4, requires_grad=True)
        real_feat = torch.randn(6, 8)
        fake_feat = torch.randn(4, 8)
        indices, weights = build_local_coupling(real_feat, fake_feat, k=3)

        # Autograd reference
        delta = local_delta(real_scores, fake_scores, indices)
        loss = (
            1.0 * local_pairwise_discriminator_loss(delta, weights, margin=0.5).mean()
            + 0.5 * local_listwise_discriminator_loss(delta, weights, tau=0.1).mean()
        )
        grad_real_ref, grad_fake_ref = torch.autograd.grad(loss, [real_scores, fake_scores])

        # Function under test
        _, _, grad_real, grad_fake = local_coupled_discriminator_loss_with_grads(
            real_scores.detach(), fake_scores.detach(), indices, weights,
            lambda_pair=1.0, pair_margin=0.5, lambda_list=0.5, list_tau=0.1,
        )
        self.assertTrue(torch.allclose(grad_real, grad_real_ref.detach(), atol=1e-5))
        self.assertTrue(torch.allclose(grad_fake, grad_fake_ref.detach(), atol=1e-5))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_coupled_g_grads_match_autograd(self):
        from training.loss import local_coupled_generator_loss_with_grads
        from training.loss import (
            build_local_coupling, local_delta,
            local_pairwise_generator_loss, local_listwise_generator_loss,
        )
        torch.manual_seed(42)
        real_scores = torch.randn(6)
        fake_scores = torch.randn(4, requires_grad=True)
        real_feat = torch.randn(6, 8)
        fake_feat = torch.randn(4, 8)
        indices, weights = build_local_coupling(real_feat, fake_feat, k=3)

        delta = local_delta(real_scores, fake_scores, indices)
        loss = (
            1.0 * local_pairwise_generator_loss(delta, weights, margin=0.5).mean()
            + 0.5 * local_listwise_generator_loss(delta, weights, tau=0.1).mean()
        )
        (grad_fake_ref,) = torch.autograd.grad(loss, [fake_scores])

        _, _, grad_fake = local_coupled_generator_loss_with_grads(
            real_scores.detach(), fake_scores.detach(), indices, weights,
            lambda_pair=1.0, pair_margin=0.5, lambda_list=0.5, list_tau=0.1,
        )
        self.assertTrue(torch.allclose(grad_fake, grad_fake_ref.detach(), atol=1e-5))
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling.test_local_coupled_d_grads_match_autograd -v`
Expected: FAIL

**Step 3: Write implementation**

```python
def local_coupled_discriminator_loss_with_grads(
    real_scores, fake_scores, neighbor_indices, coupling_weights,
    lambda_pair, pair_margin, lambda_list, list_tau,
):
    """Return (loss_scalar, loss_vector, grad_real, grad_fake) for local-coupled D loss."""
    real_var = real_scores.detach().to(torch.float32).requires_grad_(True)
    fake_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    delta = local_delta(real_var, fake_var, neighbor_indices)
    loss_terms = torch.zeros(fake_var.shape[0], device=fake_var.device)
    if lambda_pair > 0:
        loss_terms = loss_terms + lambda_pair * local_pairwise_discriminator_loss(
            delta, coupling_weights, margin=pair_margin
        )
    if lambda_list > 0:
        loss_terms = loss_terms + lambda_list * local_listwise_discriminator_loss(
            delta, coupling_weights, tau=list_tau
        )
    loss_value = loss_terms.mean()
    grad_real, grad_fake = torch.autograd.grad(loss_value, [real_var, fake_var])
    return loss_value.detach(), loss_terms.detach(), grad_real.detach(), grad_fake.detach()


def local_coupled_generator_loss_with_grads(
    real_scores, fake_scores, neighbor_indices, coupling_weights,
    lambda_pair, pair_margin, lambda_list, list_tau,
):
    """Return (loss_scalar, loss_vector, grad_fake) for local-coupled G loss."""
    real_var = real_scores.detach().to(torch.float32)
    fake_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    delta = local_delta(real_var, fake_var, neighbor_indices)
    loss_terms = torch.zeros(fake_var.shape[0], device=fake_var.device)
    if lambda_pair > 0:
        loss_terms = loss_terms + lambda_pair * local_pairwise_generator_loss(
            delta, coupling_weights, margin=pair_margin
        )
    if lambda_list > 0:
        loss_terms = loss_terms + lambda_list * local_listwise_generator_loss(
            delta, coupling_weights, tau=list_tau
        )
    loss_value = loss_terms.mean()
    (grad_fake,) = torch.autograd.grad(loss_value, [fake_var])
    return loss_value.detach(), loss_terms.detach(), grad_fake.detach()
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupling -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: add local-coupled loss gradient helpers for replay"
```

---

## Phase B: R3GANLoss Integration

### Task 5: Add coupling and asymmetric weight parameters to R3GANLoss

**Files:**
- Modify: `training/loss.py:247-378` (R3GANLoss `__init__`)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
class TestLocalCoupledR3GANLoss(unittest.TestCase):
    """Tests for R3GANLoss with local coupling."""

    def _make_loss(self, **kwargs):
        from training.loss import R3GANLoss

        class TinyG(torch.nn.Module):
            def __init__(self):
                super(TinyG, self).__init__()
                self.fc = torch.nn.Linear(4, 3 * 4 * 4)
            def forward(self, z, c):
                return self.fc(z).reshape(z.shape[0], 3, 4, 4)

        class TinyFeatureD(torch.nn.Module):
            def __init__(self):
                super(TinyFeatureD, self).__init__()
                self.feat = torch.nn.Linear(3 * 4 * 4, 8)
                self.head = torch.nn.Linear(8, 1)
            def forward(self, x, c, return_features=False):
                f = self.feat(x.reshape(x.shape[0], -1))
                s = self.head(f).squeeze(-1)
                if return_features:
                    return s, f
                return s

        defaults = dict(G=TinyG(), D=TinyFeatureD(), lambda_pair=1.0)
        defaults.update(kwargs)
        return R3GANLoss(**defaults)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_k_zero_is_backward_compatible(self):
        loss = self._make_loss(coupling_k=0)
        self.assertEqual(loss.coupling_k, 0)
        self.assertFalse(loss._requires_coupling())

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_k_positive_enables_coupling(self):
        loss = self._make_loss(coupling_k=4)
        self.assertEqual(loss.coupling_k, 4)
        self.assertTrue(loss._requires_coupling())

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_asymmetric_list_weights(self):
        loss = self._make_loss(lambda_list_d=1.0, lambda_list_g=0.1, coupling_k=4)
        self.assertEqual(loss.lambda_list_d, 1.0)
        self.assertEqual(loss.lambda_list_g, 0.1)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_lambda_list_maps_to_symmetric(self):
        loss = self._make_loss(lambda_list=0.5, coupling_k=4)
        self.assertEqual(loss.lambda_list_d, 0.5)
        self.assertEqual(loss.lambda_list_g, 0.5)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss -v`
Expected: FAIL

**Step 3: Modify R3GANLoss.__init__**

Add new parameters to `__init__`:

```python
def __init__(
    self,
    G, D, augment_pipe=None,
    # ... existing params ...
    coupling_k=0,            # 0 = no coupling (backward compat); >0 = kNN coupling
    lambda_list_d=None,      # D-side listwise weight (overrides lambda_list)
    lambda_list_g=None,      # G-side listwise weight (overrides lambda_list)
    # ... rest unchanged ...
):
```

After the existing `lambda_list` setup, add:

```python
    self.coupling_k = int(coupling_k)
    if lambda_list_d is not None or lambda_list_g is not None:
        self.lambda_list_d = float(lambda_list_d if lambda_list_d is not None else self.lambda_list)
        self.lambda_list_g = float(lambda_list_g if lambda_list_g is not None else 0.0)
    else:
        self.lambda_list_d = self.lambda_list
        self.lambda_list_g = self.lambda_list

    # ... validation ...
    if self.coupling_k < 0:
        raise ValueError("coupling_k must be non-negative")
    if self.lambda_list_d < 0:
        raise ValueError("lambda_list_d must be non-negative")
    if self.lambda_list_g < 0:
        raise ValueError("lambda_list_g must be non-negative")
```

Add helper method:

```python
def _requires_coupling(self):
    return self.coupling_k > 0
```

Update `_requires_full_batch` to also check coupling:

```python
def _requires_full_batch(self, phase=None):
    if self._requires_coupling():
        return True  # coupling needs full-batch features
    if phase == "G":
        return self.lambda_list > 0
    if phase == "D":
        return self.lambda_list > 0 or self.lambda_local_rank > 0
    return self.lambda_list > 0 or self.lambda_local_rank > 0
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: add coupling_k and asymmetric lambda_list_d/g to R3GANLoss"
```

---

### Task 6: Integrate coupling into metadata collection

**Files:**
- Modify: `training/loss.py:577-642` (`_collect_coupled_metadata`)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_metadata_collects_clean_features(self):
        loss = self._make_loss(coupling_k=2, lambda_pair=1.0)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        self.assertIsNotNone(loss._coupled_phase_buffer)
        loss.finalize_accumulation()
        # Should complete without error; features collected for coupling
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss.test_coupling_metadata_collects_clean_features -v`
Expected: FAIL (coupling path not wired yet)

**Step 3: Modify `_collect_coupled_metadata`**

The key change: collect clean features when `_requires_coupling()` is True (not just when `lambda_local_rank > 0`):

```python
need_clean_features = (phase == "D" and self.lambda_local_rank > 0) or self._requires_coupling()
```

Replace `if phase == "D" and self.lambda_local_rank > 0:` with `if need_clean_features:` in both the per-micro-batch loop and the concatenation block.

For G phase with coupling, also collect clean features:
```python
if need_clean_features:
    _, real_features = self.run_D(real_img.detach(), real_c, augment=False, return_features=True)
    clean_fake_scores, fake_features = self.run_D(fake_img, real_c, augment=False, return_features=True)
    local["clean_real_features"].append(real_features.to(torch.float32))
    local["clean_fake_features"].append(fake_features.to(torch.float32))
    local["clean_fake_scores"].append(clean_fake_scores.to(torch.float32))
    class_ids = self._class_ids(real_c)
    if class_ids is not None:
        local["class_ids"].append(class_ids.to(torch.long))
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: collect clean features for coupling in metadata"
```

---

### Task 7: Integrate coupling into replay preparation

**Files:**
- Modify: `training/loss.py:644-718` (`_prepare_coupled_replay`)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupled_replay_produces_gradients_d(self):
        loss = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_d=0.5)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        loss.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss.D.parameters())
        self.assertTrue(has_grad, 'D should have gradients after coupled replay')

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupled_replay_produces_gradients_g(self):
        loss = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_g=0.3)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('G', real, cond, noise, gamma=0.1, gain=1.0)
        loss.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss.G.parameters())
        self.assertTrue(has_grad, 'G should have gradients after coupled replay')
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss.test_coupled_replay_produces_gradients_d -v`
Expected: FAIL

**Step 3: Modify `_prepare_coupled_replay`**

When `_requires_coupling()`, build the coupling and use local-coupled losses instead of global InfoNCE.

Add coupling construction at the top of `_prepare_coupled_replay`:

```python
coupling = None
if self._requires_coupling():
    from training.loss import build_local_coupling
    coupling = build_local_coupling(
        global_meta["clean_real_features"],
        global_meta["clean_fake_features"],
        k=self.coupling_k,
    )
```

Then branch on `self._requires_coupling()` for loss computation:

For **G phase with coupling**:
```python
if self._requires_coupling():
    neighbor_indices, coupling_weights = coupling
    lambda_list_g = self.lambda_list_g
    _, global_loss, grad_fake = local_coupled_generator_loss_with_grads(
        global_meta["aug_real_scores"],
        global_meta["aug_fake_scores"],
        neighbor_indices, coupling_weights,
        lambda_pair=self.lambda_pair, pair_margin=self.pair_margin,
        lambda_list=lambda_list_g, list_tau=self.list_tau,
    )
    replay["pair_loss"] = global_loss[local_slice]  # combined loss vector
    replay["list_loss"] = torch.zeros_like(replay["pair_loss"])  # included in pair_loss
    replay["grad_fake_aug"] = grad_fake[local_slice]
    return replay
```

For **D phase with coupling** (analogous, using `local_coupled_discriminator_loss_with_grads`):
```python
if self._requires_coupling():
    neighbor_indices, coupling_weights = coupling
    lambda_list_d = self.lambda_list_d
    _, global_loss, grad_real, grad_fake = local_coupled_discriminator_loss_with_grads(
        global_meta["aug_real_scores"],
        global_meta["aug_fake_scores"],
        neighbor_indices, coupling_weights,
        lambda_pair=self.lambda_pair, pair_margin=self.pair_margin,
        lambda_list=lambda_list_d, list_tau=self.list_tau,
    )
    replay["pair_loss"] = global_loss[local_slice]
    replay["list_loss"] = torch.zeros_like(replay["pair_loss"])
    replay["grad_real_aug"] = grad_real[local_slice]
    replay["grad_fake_aug"] = grad_fake[local_slice]
    # local_rank still computed separately (if enabled)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: integrate local coupling into replay preparation"
```

---

### Task 8: Wire coupling into gradient replay

**Files:**
- Modify: `training/loss.py:775-922` (`_replay_coupled_gradients`)
- Test: `tests/test_adv_losses.py`

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupled_replay_matches_full_batch_d_gradients(self):
        """Verify that buffered coupled replay produces same D param gradients as full-batch."""
        torch.manual_seed(42)
        loss_ref = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_d=0.5)
        loss_buf = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_d=0.5)
        # Share weights
        loss_buf.G.load_state_dict(loss_ref.G.state_dict())
        loss_buf.D.load_state_dict(loss_ref.D.state_dict())

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        # Reference: single full batch
        loss_ref.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        loss_ref.finalize_accumulation()
        ref_grads = {n: p.grad.clone() for n, p in loss_ref.D.named_parameters() if p.grad is not None}

        # Buffered: two micro-batches of 2
        loss_buf.D.zero_grad()
        loss_buf.accumulate_gradients('D', real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        loss_buf.accumulate_gradients('D', real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        loss_buf.finalize_accumulation()
        buf_grads = {n: p.grad.clone() for n, p in loss_buf.D.named_parameters() if p.grad is not None}

        for name in ref_grads:
            self.assertTrue(
                torch.allclose(ref_grads[name], buf_grads[name], atol=1e-4),
                f'D grad mismatch for {name}',
            )
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss.test_coupled_replay_matches_full_batch_d_gradients -v`
Expected: FAIL (may already pass if Task 7 was done correctly — verify)

**Step 3: Verify/fix `_replay_coupled_gradients`**

The replay mechanism already handles `grad_real_aug` and `grad_fake_aug` correctly. The key verification is that:
1. When coupling is enabled, pairwise loss in the replay loop should NOT double-backward through pairwise (since it's already included in the coupled loss gradients)
2. The `scalar_loss` path should be skipped for the coupled portion

When `_requires_coupling()` and the D phase: the pairwise backward in the replay loop must be disabled (gradients come from the replay coefficients instead). Modify the D replay loop:

```python
if self._requires_coupling():
    # Pairwise + listwise already covered by coupled replay gradients.
    # Only compute scalar terms for path_rank and gradient penalties.
    scalar_terms = []
else:
    if self.lambda_pair > 0:
        pair_loss = pairwise_discriminator_loss(...)
        scalar_terms.append(self.lambda_pair * pair_loss)
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_adv_losses.TestLocalCoupledR3GANLoss -v`
Expected: PASS

**Step 5: Run all existing tests for regression**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add training/loss.py tests/test_adv_losses.py
git commit -m "feat: wire local coupling into gradient replay with regression tests"
```

---

## Phase C: Training Infrastructure

### Task 9: CLI flags for coupling and asymmetric weights

**Files:**
- Modify: `train.py` (Click options + config assembly)
- Test: `tests/test_adv_losses.py` (TestCliMappings)

**Step 1: Write the failing test**

```python
    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_k_flag_maps_to_loss_kwargs(self):
        """--coupling-k should map to loss_kwargs.coupling_k."""
        # This tests the config assembly logic in train.py
        # by directly verifying R3GANLoss accepts the parameter
        from training.loss import R3GANLoss
        # Minimal construction with coupling
        class TinyG(torch.nn.Module):
            def __init__(self):
                super(TinyG, self).__init__()
                self.fc = torch.nn.Linear(4, 48)
            def forward(self, z, c):
                return self.fc(z).reshape(z.shape[0], 3, 4, 4)
        class TinyD(torch.nn.Module):
            def __init__(self):
                super(TinyD, self).__init__()
                self.fc = torch.nn.Linear(48, 1)
            def forward(self, x, c, return_features=False):
                f = x.reshape(x.shape[0], -1)
                s = self.fc(f).squeeze(-1)
                return (s, f) if return_features else s
        loss = R3GANLoss(G=TinyG(), D=TinyD(), coupling_k=4, lambda_list_d=0.5, lambda_list_g=0.1)
        self.assertEqual(loss.coupling_k, 4)
        self.assertEqual(loss.lambda_list_d, 0.5)
        self.assertEqual(loss.lambda_list_g, 0.1)
```

**Step 2: Run test to verify it passes** (should already pass from Task 5)

**Step 3: Add CLI flags to train.py**

After the existing `--local-rank-k` option:

```python
@click.option('--coupling-k', help='kNN coupling neighbors (0=no coupling, backward compat)', metavar='INT', type=int, default=0, show_default=True)
@click.option('--lambda-list-d', help='D-side local-listwise weight (overrides --lambda-list for D)', metavar='FLOAT', type=float, default=None)
@click.option('--lambda-list-g', help='G-side local-listwise weight (overrides --lambda-list for G)', metavar='FLOAT', type=float, default=None)
```

In the config assembly section (around line 936), add:

```python
c.loss_kwargs.coupling_k = opts.coupling_k
if opts.lambda_list_d is not None:
    c.loss_kwargs.lambda_list_d = opts.lambda_list_d
if opts.lambda_list_g is not None:
    c.loss_kwargs.lambda_list_g = opts.lambda_list_g
```

Update the description string:

```python
if opts.coupling_k > 0:
    desc += f'-coupled{opts.coupling_k:d}'
if opts.lambda_list_d is not None:
    desc += f'-listD{opts.lambda_list_d:g}'
if opts.lambda_list_g is not None:
    desc += f'-listG{opts.lambda_list_g:g}'
```

**Step 4: Run tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add train.py tests/test_adv_losses.py
git commit -m "feat: add CLI flags for coupling_k, lambda_list_d, lambda_list_g"
```

---

### Task 10: Curriculum schedulers for listwise weights

**Files:**
- Modify: `training/training_loop.py:154-167` (add scheduler params)
- Modify: `training/training_loop.py:417-447` (use schedulers)
- Modify: `training/loss.py` (add method to update weights)
- Modify: `train.py` (preset scheduler definitions)

**Step 1: Add scheduler params to training_loop.py**

In `training_loop()` signature, add:

```python
list_d_scheduler=None,
list_g_scheduler=None,
```

**Step 2: Add weight update method to R3GANLoss**

```python
def set_list_weights(self, lambda_list_d=None, lambda_list_g=None):
    """Update listwise weights (called by training loop scheduler)."""
    if lambda_list_d is not None:
        self.lambda_list_d = float(lambda_list_d)
    if lambda_list_g is not None:
        self.lambda_list_g = float(lambda_list_g)
```

**Step 3: Use schedulers in training loop**

After `cur_aug_p = cosine_decay_with_warmup(...)` (line 421), add:

```python
if list_d_scheduler is not None:
    cur_list_d = cosine_decay_with_warmup(cur_nimg, **list_d_scheduler)
else:
    cur_list_d = None
if list_g_scheduler is not None:
    cur_list_g = cosine_decay_with_warmup(cur_nimg, **list_g_scheduler)
else:
    cur_list_g = None
loss.set_list_weights(lambda_list_d=cur_list_d, lambda_list_g=cur_list_g)
```

**Step 4: Define schedulers in presets**

Example for CIFAR10 preset with coupling (users opt in via CLI):

```python
if opts.coupling_k > 0:
    list_d_target = opts.lambda_list_d if opts.lambda_list_d is not None else 0.5
    list_g_target = opts.lambda_list_g if opts.lambda_list_g is not None else 0.1
    c.list_d_scheduler = {
        'base_value': list_d_target, 'total_nimg': total_nimg,
        'final_value': list_d_target,
        'warmup_value': 0.0, 'warmup_nimg': total_nimg // 4,
    }
    c.list_g_scheduler = {
        'base_value': list_g_target, 'total_nimg': total_nimg,
        'final_value': list_g_target,
        'warmup_value': 0.0, 'warmup_nimg': total_nimg // 2,
    }
```

This gives:
- Phase 1 (0 → 25% training): pairwise warmup (both λ_ℓ = 0)
- Phase 2 (25% → 50%): D-side ramps 0 → target; G-side still 0
- Phase 3 (50% →): both at target (G ≤ D)

**Step 5: Run tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add training/training_loop.py training/loss.py train.py
git commit -m "feat: add curriculum schedulers for asymmetric listwise weights"
```

---

### Task 11: Update presets with recommended coupling defaults

**Files:**
- Modify: `train.py` (preset definitions)

**Step 1: Add coupling defaults to presets**

For each preset (CIFAR10, FFHQ-64, FFHQ-256, ImageNet-32, ImageNet-64), add coupling-aware defaults that only activate when `--coupling-k > 0`:

The presets should NOT change defaults for users who don't use `--coupling-k`. They only define scheduler configs when coupling is explicitly enabled.

**Step 2: Run tests**

Run: `python -m unittest discover -s tests -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add coupling-aware scheduler defaults to presets"
```

---

## Phase D: Optional Extensions

### Task 12 (Optional): Mobility-aware R2 penalty

**Files:**
- Modify: `training/loss.py`
- Test: `tests/test_adv_losses.py`

Implements the optional field-aware fake-side damping from the design doc:

```
R_{2,μ} = (γ/2) Σ_j sg(μ̄_j) · |∇_x s(x̂_j)|²
```

where `μ̄_j` is the truncated local-listwise mobility for fake j.

This reweights the R2 penalty so that fakes under more listwise pressure get stronger gradient damping. Implementation:

1. Compute mobility `μ_j = Σ_i π̃_ij exp(-Δ_ij/τ)` from the coupling (available after `_prepare_coupled_replay`)
2. Truncate and normalize: `μ̄_j = clamp(μ_j / μ_j.mean(), 0.1, 10.0)`
3. Replace `r2_penalty` with `sg(μ̄_j) * r2_penalty` in the D loss

**Status:** Not recommended for canonical method. Implement only after ablation experiments validate the base system.

---

## Summary

| Phase | Task | Files | What changes |
|-------|------|-------|-------------|
| A | 1. Coupling builder + local delta | `loss.py`, tests | `build_local_coupling`, `local_delta` functions |
| A | 2. Local-coupled pairwise losses | `loss.py`, tests | `local_pairwise_{d,g}_loss` functions |
| A | 3. Local-coupled listwise losses | `loss.py`, tests | `local_listwise_{d,g}_loss` with numerical stability |
| A | 4. Gradient helpers for replay | `loss.py`, tests | `local_coupled_{d,g}_loss_with_grads` functions |
| B | 5. R3GANLoss coupling params | `loss.py`, tests | `coupling_k`, `lambda_list_d/g`, `_requires_coupling()` |
| B | 6. Metadata collection | `loss.py`, tests | Clean features collected when coupling enabled |
| B | 7. Replay preparation | `loss.py`, tests | Local-coupled losses in `_prepare_coupled_replay` |
| B | 8. Gradient replay wiring | `loss.py`, tests | Coupled path in `_replay_coupled_gradients` |
| C | 9. CLI flags | `train.py`, tests | `--coupling-k`, `--lambda-list-d/g` |
| C | 10. Curriculum schedulers | `training_loop.py`, `loss.py`, `train.py` | `list_d_scheduler`, `list_g_scheduler` |
| C | 11. Preset updates | `train.py` | Coupling-aware defaults |
| D | 12. Mobility-aware R2 (optional) | `loss.py`, tests | Field-aware fake-side damping |

## What is NOT changed

- R3GAN/Trainer.py — base pairwise/softmargin logic untouched
- Generator/Discriminator architecture — no architectural changes
- Augmentation pipeline — unchanged
- Default behavior without `--coupling-k` — full backward compatibility
- Path-rank experimental system — kept as-is
- Training loop structure (phases, optimizer steps, EMA) — minimal additions

## Usage Examples

```bash
# Canonical RankGAN: local-coupled pairwise + D-side listwise + R1/R2
python train.py --preset=CIFAR10 --coupling-k=4 \
    --lambda-pair=1.0 --pair-margin=0.0 \
    --lambda-list-d=0.5 --lambda-list-g=0.1 --list-tau=0.07

# Pairwise warmup only (Phase 1)
python train.py --preset=CIFAR10 --coupling-k=4 \
    --lambda-pair=1.0 --lambda-list-d=0.0 --lambda-list-g=0.0

# With curriculum (auto ramp-up)
python train.py --preset=CIFAR10 --coupling-k=4 \
    --lambda-pair=1.0 --lambda-list-d=0.5 --lambda-list-g=0.1

# Backward compat: no coupling (current behavior)
python train.py --preset=CIFAR10 --lambda-pair=1.0
```
