import unittest

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

if torch is not None:
    from R3GAN.Trainer import AdversarialTraining
else:  # pragma: no cover
    AdversarialTraining = None


@unittest.skipIf(
    torch is None or AdversarialTraining is None,
    "PyTorch is not available in this environment",
)
class TestAdversarialLosses(unittest.TestCase):
    def test_softmargin_margin_zero_matches_rpgan(self):
        real = torch.randn(32)
        fake = torch.randn(32)

        d_loss, d_rel = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="softmargin",
            Margin=0.0,
            Tau=0.07,
        )
        g_loss, g_rel = AdversarialTraining._generator_adv_loss(
            FakeLogits=fake,
            RealLogits=real,
            LossType="softmargin",
            Margin=0.0,
            Tau=0.07,
        )

        self.assertTrue(torch.allclose(d_rel, real - fake))
        self.assertTrue(torch.allclose(g_rel, fake - real))
        self.assertTrue(torch.allclose(d_loss, F.softplus(-(real - fake))))
        self.assertTrue(torch.allclose(g_loss, F.softplus(-(fake - real))))

    def test_infonce_is_finite_and_vectorized(self):
        real = torch.randn(64)
        fake = torch.randn(64)

        d_loss, d_rel = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="infonce",
            Margin=0.0,
            Tau=0.07,
        )
        g_loss, g_rel = AdversarialTraining._generator_adv_loss(
            FakeLogits=fake,
            RealLogits=real,
            LossType="infonce",
            Margin=0.0,
            Tau=0.07,
        )

        self.assertEqual(d_loss.shape, real.shape)
        self.assertEqual(g_loss.shape, fake.shape)
        self.assertEqual(d_rel.shape, real.shape)
        self.assertEqual(g_rel.shape, fake.shape)
        self.assertTrue(torch.isfinite(d_loss).all())
        self.assertTrue(torch.isfinite(g_loss).all())

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
            Noise=noise,
            RealSamples=real,
            Conditions=cond,
            Gamma=0.1,
            Scale=1.0,
        )
        self.assertEqual(len(results), 5)
        fake_samples = results[4]
        self.assertEqual(fake_samples.shape, (4, 3, 4, 4))
        self.assertFalse(fake_samples.requires_grad)  # Should be detached

    def test_infonce_requires_positive_tau(self):
        real = torch.randn(8)
        fake = torch.randn(8)

        with self.assertRaises(ValueError):
            AdversarialTraining._discriminator_adv_loss(
                RealLogits=real,
                FakeLogits=fake,
                LossType="infonce",
                Margin=0.0,
                Tau=0.0,
            )

        with self.assertRaises(ValueError):
            AdversarialTraining._generator_adv_loss(
                FakeLogits=fake,
                RealLogits=real,
                LossType="infonce",
                Margin=0.0,
                Tau=-1.0,
            )


@unittest.skipIf(
    torch is None or AdversarialTraining is None,
    "PyTorch is not available in this environment",
)
class TestInfoNCEBuffering(unittest.TestCase):
    def test_infonce_logsumexp_uses_full_batch(self):
        real = torch.tensor([1.0, 0.5, -0.5, -1.0])
        fake = torch.tensor([0.8, 0.3, -0.3, -0.8])
        tau = 1.0
        d_loss_full, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="infonce",
            Tau=tau,
        )
        d_loss_micro1, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real[:2],
            FakeLogits=fake[:2],
            LossType="infonce",
            Tau=tau,
        )
        d_loss_micro2, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real[2:],
            FakeLogits=fake[2:],
            LossType="infonce",
            Tau=tau,
        )
        d_loss_micro_avg = 0.5 * d_loss_micro1.mean() + 0.5 * d_loss_micro2.mean()
        self.assertFalse(
            torch.allclose(d_loss_full.mean(), d_loss_micro_avg, atol=1e-4),
            "InfoNCE micro-batch should differ from full-batch (logsumexp scope differs)",
        )


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestR3GANLossBuffering(unittest.TestCase):
    def _make_loss(self, adv_loss_type="infonce", tau=0.07):
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
        loss_obj = self._make_loss(adv_loss_type="softmargin")
        loss_obj.finalize_accumulation()  # Should not raise

    def test_infonce_buffers_then_flushes(self):
        loss_obj = self._make_loss(adv_loss_type="infonce")
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        # First call: should buffer, no backward yet
        loss_obj.accumulate_gradients(
            "D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5
        )
        for p in loss_obj.D.parameters():
            self.assertIsNone(p.grad)
        # Second call: still buffering
        loss_obj.accumulate_gradients(
            "D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5
        )
        for p in loss_obj.D.parameters():
            self.assertIsNone(p.grad)
        # Finalize: should process full batch and produce gradients
        loss_obj.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss_obj.D.parameters())
        self.assertTrue(has_grad, "D should have gradients after finalize_accumulation")


@unittest.skipIf(
    torch is None or AdversarialTraining is None,
    "PyTorch is not available in this environment",
)
class TestSoftMarginWithMargin(unittest.TestCase):
    def test_positive_margin_penalizes_small_gap(self):
        """With margin > 0, even a correct ordering with small gap should have high loss."""
        real = torch.tensor([1.0] * 32)
        fake = torch.tensor([0.5] * 32)  # gap = 0.5, less than margin = 2.0

        d_loss, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="softmargin",
            Margin=2.0,
        )
        d_loss_no_margin, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="softmargin",
            Margin=0.0,
        )
        self.assertGreater(d_loss.mean().item(), d_loss_no_margin.mean().item())

    def test_gradient_never_zero(self):
        """Softmargin should always have non-zero gradient (unlike hard hinge)."""
        real = torch.tensor([10.0], requires_grad=True)
        fake = torch.tensor([0.0])

        d_loss, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="softmargin",
            Margin=1.0,
        )
        d_loss.backward()
        # Even with a huge gap (10 > 1), gradient should be non-zero
        self.assertGreater(real.grad.abs().item(), 0.0)


@unittest.skipIf(
    torch is None or AdversarialTraining is None,
    "PyTorch is not available in this environment",
)
class TestInfoNCEProperties(unittest.TestCase):
    def test_harder_negatives_increase_loss(self):
        """InfoNCE should produce higher loss when fakes are closer to reals."""
        real = torch.tensor([1.0, 1.0, 1.0, 1.0])
        easy_fake = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        hard_fake = torch.tensor([0.9, 0.8, 0.7, 0.6])

        d_loss_easy, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=easy_fake,
            LossType="infonce",
            Tau=0.1,
        )
        d_loss_hard, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=hard_fake,
            LossType="infonce",
            Tau=0.1,
        )
        self.assertGreater(d_loss_hard.mean().item(), d_loss_easy.mean().item())

    def test_symmetry(self):
        """G and D InfoNCE losses should have mirrored relativistic logits."""
        real = torch.randn(16)
        fake = torch.randn(16)
        tau = 0.1

        d_loss, d_rel = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="infonce",
            Tau=tau,
        )
        g_loss, g_rel = AdversarialTraining._generator_adv_loss(
            FakeLogits=fake,
            RealLogits=real,
            LossType="infonce",
            Tau=tau,
        )
        self.assertTrue(torch.allclose(d_rel, -g_rel))
        self.assertTrue(torch.isfinite(d_loss).all())
        self.assertTrue(torch.isfinite(g_loss).all())


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestRankingLosses(unittest.TestCase):
    """Tests for experimental interpolation-based ranking losses."""

    def test_listmle_correct_order_low_loss(self):
        from training.loss import listmle_loss

        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        loss = listmle_loss(scores)
        scores_rev = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss_rev = listmle_loss(scores_rev)
        self.assertGreater(loss_rev.item(), loss.item())

    def test_pairwise_hinge_zero_loss_when_margin_satisfied(self):
        from training.loss import pairwise_hinge_loss

        scores = torch.tensor([[10.0, 5.0, 0.0]])  # gaps all > margin=1
        loss = pairwise_hinge_loss(scores, margin=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_pairwise_logistic_matches_rpgan_for_k2(self):
        """With K=2, pairwise_logistic should match the base RpGAN loss."""
        from training.loss import pairwise_logistic_loss

        real_score = torch.tensor([2.0])
        fake_score = torch.tensor([-1.0])
        scores = torch.stack([real_score, fake_score], dim=-1)  # [1, 2]

        rank_loss = pairwise_logistic_loss(scores)
        rpgan_loss = F.softplus(-(real_score - fake_score)).mean()
        self.assertTrue(torch.allclose(rank_loss, rpgan_loss, atol=1e-5))

    def test_make_rank_list_endpoints(self):
        from training.loss import make_rank_list

        real = torch.randn(2, 3, 4, 4)
        fake = torch.randn(2, 3, 4, 4)
        result = make_rank_list(real, fake, k=4, mode="intrpl", alpha_dist="linear")
        self.assertEqual(result.shape, (2, 4, 3, 4, 4))
        self.assertTrue(torch.allclose(result[:, 0], real))
        self.assertTrue(torch.allclose(result[:, -1], fake))

    def test_make_rank_list_alpha_monotonic(self):
        from training.loss import make_rank_list

        real = torch.ones(1, 1, 2, 2)
        fake = torch.zeros(1, 1, 2, 2)
        for dist in ["linear", "cosine", "random"]:
            result = make_rank_list(real, fake, k=8, mode="intrpl", alpha_dist=dist)
            means = result[0, :, 0, 0, 0]
            for i in range(len(means) - 1):
                self.assertGreaterEqual(
                    means[i].item(),
                    means[i + 1].item(),
                    f"alpha_dist={dist}: not monotonic at position {i}",
                )


if __name__ == "__main__":
    unittest.main()
