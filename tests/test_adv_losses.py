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


if __name__ == "__main__":
    unittest.main()
