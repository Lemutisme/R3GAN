import unittest
import warnings
from unittest import mock

try:
    import dnnlib
    import torch
    import torch.nn.functional as F
    from click.testing import CliRunner
except Exception:  # pragma: no cover
    dnnlib = None
    torch = None
    F = None
    CliRunner = None

if torch is not None:
    import train as train_cli
    from R3GAN.Trainer import AdversarialTraining
    from training.loss import (
        R3GANLoss,
        allpairs_delta,
        infonce_discriminator_loss,
        infonce_generator_loss,
        listmle_loss,
        make_rank_list,
        pairwise_delta,
        pairwise_discriminator_loss,
        pairwise_generator_loss,
        pairwise_hinge_loss,
        pairwise_logistic_loss,
    )
else:  # pragma: no cover
    train_cli = None
    AdversarialTraining = None
    R3GANLoss = None


if torch is not None:
    class TinyG(torch.nn.Module):
        def __init__(self, z_dim=4):
            super(TinyG, self).__init__()
            self.fc = torch.nn.Linear(z_dim, 3 * 4 * 4)

        def forward(self, z, c):
            return self.fc(z).reshape(z.shape[0], 3, 4, 4)


    class TinyD(torch.nn.Module):
        def __init__(self):
            super(TinyD, self).__init__()
            self.fc = torch.nn.Linear(3 * 4 * 4, 1)

        def forward(self, x, c):
            return self.fc(x.reshape(x.shape[0], -1)).squeeze(-1)


    class TinyFeatureD(torch.nn.Module):
        def __init__(self):
            super(TinyFeatureD, self).__init__()
            self.feature = torch.nn.Linear(3 * 4 * 4, 8)
            self.score = torch.nn.Linear(3 * 4 * 4, 1)

        def forward(self, x, c, return_features=False):
            flat = x.reshape(x.shape[0], -1)
            score = self.score(flat).squeeze(-1)
            if return_features:
                return score, self.feature(flat)
            return score


    class MeanScoreD(torch.nn.Module):
        def forward(self, x, c, return_features=False):
            score = x.mean(dim=(1, 2, 3))
            if return_features:
                return score, x.mean(dim=(2, 3))
            return score


    class OffsetAugment(torch.nn.Module):
        def __init__(self, offset):
            super(OffsetAugment, self).__init__()
            self.offset = offset

        def forward(self, x):
            return x + self.offset


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
        self.assertTrue(torch.allclose(g_loss, F.softplus(real - fake)))

    def test_pairwise_softmargin_matches_legacy_gradients(self):
        margin = 0.35
        real = torch.randn(16, requires_grad=True)
        fake = torch.randn(16, requires_grad=True)
        delta = pairwise_delta(real, fake)
        new_loss = pairwise_discriminator_loss(delta, margin=margin).mean()
        new_loss.backward()
        new_real_grad = real.grad.detach().clone()
        new_fake_grad = fake.grad.detach().clone()

        legacy_real = real.detach().clone().requires_grad_(True)
        legacy_fake = fake.detach().clone().requires_grad_(True)
        legacy_loss, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=legacy_real,
            FakeLogits=legacy_fake,
            LossType="softmargin",
            Margin=margin,
        )
        legacy_loss.mean().backward()
        self.assertTrue(torch.allclose(new_loss.detach(), legacy_loss.mean(), atol=1e-6))
        self.assertTrue(torch.allclose(new_real_grad, legacy_real.grad, atol=1e-6))
        self.assertTrue(torch.allclose(new_fake_grad, legacy_fake.grad, atol=1e-6))

    def test_delta_symmetry(self):
        real = torch.randn(20)
        fake = torch.randn(20)
        delta = pairwise_delta(real, fake)
        self.assertTrue(torch.allclose(delta, -(pairwise_delta(fake, real))))
        d_loss = pairwise_discriminator_loss(delta)
        g_loss = pairwise_generator_loss(delta)
        self.assertTrue(torch.allclose(d_loss, pairwise_generator_loss(-delta)))
        self.assertTrue(torch.allclose(g_loss, pairwise_discriminator_loss(-delta)))

    def test_discriminator_returns_fake_samples(self):
        g = TinyG(z_dim=8)
        d = TinyD()
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
        self.assertFalse(fake_samples.requires_grad)

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
class TestListwiseLosses(unittest.TestCase):
    def test_explicit_delta_matrix_matches_legacy_infonce(self):
        real = torch.randn(7)
        fake = torch.randn(7)
        tau = 0.23
        delta_matrix = allpairs_delta(real, fake)
        d_list = infonce_discriminator_loss(delta_matrix, tau=tau)
        g_list = infonce_generator_loss(delta_matrix, tau=tau)
        d_legacy, _ = AdversarialTraining._discriminator_adv_loss(
            RealLogits=real,
            FakeLogits=fake,
            LossType="infonce",
            Tau=tau,
        )
        g_legacy, _ = AdversarialTraining._generator_adv_loss(
            FakeLogits=fake,
            RealLogits=real,
            LossType="infonce",
            Tau=tau,
        )
        self.assertTrue(torch.allclose(d_list, d_legacy, atol=1e-6))
        self.assertTrue(torch.allclose(g_list, g_legacy, atol=1e-6))

    def test_harder_negatives_increase_list_loss(self):
        real = torch.tensor([1.0, 1.0, 1.0, 1.0])
        easy_fake = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        hard_fake = torch.tensor([0.9, 0.8, 0.7, 0.6])
        tau = 0.1
        d_easy = infonce_discriminator_loss(allpairs_delta(real, easy_fake), tau=tau)
        d_hard = infonce_discriminator_loss(allpairs_delta(real, hard_fake), tau=tau)
        self.assertGreater(d_hard.mean().item(), d_easy.mean().item())

    def test_listwise_symmetry(self):
        real = torch.randn(16)
        fake = torch.randn(16)
        tau = 0.1
        delta_matrix = allpairs_delta(real, fake)
        d_list = infonce_discriminator_loss(delta_matrix, tau=tau)
        g_list = infonce_generator_loss(delta_matrix, tau=tau)
        self.assertTrue(torch.isfinite(d_list).all())
        self.assertTrue(torch.isfinite(g_list).all())
        self.assertTrue(torch.allclose(d_list, infonce_generator_loss(-delta_matrix.T, tau=tau)))


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestR3GANLossBuffering(unittest.TestCase):
    def _make_loss(self, **kwargs):
        g = TinyG()
        d = kwargs.pop("D", TinyFeatureD())
        return R3GANLoss(G=g, D=d, **kwargs)

    def test_finalize_noop_for_pairwise(self):
        loss_obj = self._make_loss(lambda_pair=1.0, lambda_list=0.0)
        loss_obj.finalize_accumulation()

    def test_listwise_buffers_then_flushes(self):
        loss_obj = self._make_loss(lambda_pair=0.0, lambda_list=1.0, list_tau=0.07)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss_obj.accumulate_gradients("D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        for p in loss_obj.D.parameters():
            self.assertIsNone(p.grad)
        loss_obj.accumulate_gradients("D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        for p in loss_obj.D.parameters():
            self.assertIsNone(p.grad)
        loss_obj.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss_obj.D.parameters())
        self.assertTrue(has_grad)


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDiscriminatorFeatureInterface(unittest.TestCase):
    def test_return_features_false_preserves_scores(self):
        from training.networks import Discriminator

        d = Discriminator(
            WidthPerStage=[16, 16, 16, 16],
            CardinalityPerStage=[1, 1, 1, 1],
            BlocksPerStage=[1, 1, 1, 1],
            ExpansionFactor=2,
            FP16Stages=[],
            c_dim=0,
            img_resolution=32,
        )
        x = torch.randn(2, 3, 32, 32)
        c = torch.zeros(2, 0)
        score_default = d(x, c)
        score_plain = d(x, c, return_features=False)
        score_feat, features = d(x, c, return_features=True)
        self.assertTrue(torch.allclose(score_default, score_plain))
        self.assertTrue(torch.allclose(score_default, score_feat))
        self.assertEqual(features.shape[0], x.shape[0])
        self.assertEqual(features.ndim, 2)
        self.assertGreater(features.shape[1], 0)

    def test_return_features_conditional_model(self):
        from training.networks import Discriminator

        d = Discriminator(
            WidthPerStage=[16, 16, 16, 16],
            CardinalityPerStage=[1, 1, 1, 1],
            BlocksPerStage=[1, 1, 1, 1],
            ExpansionFactor=2,
            ConditionEmbeddingDimension=8,
            FP16Stages=[],
            c_dim=5,
            img_resolution=32,
        )
        x = torch.randn(2, 3, 32, 32)
        c = F.one_hot(torch.tensor([1, 3]), num_classes=5).to(torch.float32)
        score, features = d(x, c, return_features=True)
        self.assertEqual(score.shape, (2,))
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.ndim, 2)


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestLocalRankPrior(unittest.TestCase):
    def test_local_rank_is_deterministic_and_blocks_feature_gradients(self):
        loss_obj = R3GANLoss(
            G=TinyG(),
            D=TinyFeatureD(),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=2,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        real_scores = torch.tensor([5.0, 4.0], requires_grad=True)
        fake_scores = torch.tensor([4.2, 3.0], requires_grad=True)
        real_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        fake_features = torch.tensor([[0.9, 0.1], [0.8, 0.2]], requires_grad=True)
        cond = torch.zeros(2, 0)
        loss_1 = loss_obj._compute_local_rank_loss(
            real_scores, fake_scores, real_features, fake_features, cond
        )
        loss_2 = loss_obj._compute_local_rank_loss(
            real_scores, fake_scores, real_features, fake_features, cond
        )
        self.assertAlmostEqual(loss_1.item(), loss_2.item(), places=7)
        loss_1.backward()
        self.assertIsNone(real_features.grad)
        self.assertIsNone(fake_features.grad)
        self.assertIsNotNone(real_scores.grad)
        self.assertIsNotNone(fake_scores.grad)

    def test_local_rank_class_masking_skips_cross_class_neighbors(self):
        loss_obj = R3GANLoss(
            G=TinyG(),
            D=TinyFeatureD(),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=2,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        real_scores = torch.tensor([3.0, 1.0])
        fake_scores = torch.tensor([2.0, 0.0])
        real_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        fake_features = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        unconditional = torch.zeros(2, 0)
        conditional = F.one_hot(torch.tensor([0, 1]), num_classes=2).to(torch.float32)
        loss_unconditional = loss_obj._compute_local_rank_loss(
            real_scores, fake_scores, real_features, fake_features, unconditional
        )
        loss_conditional = loss_obj._compute_local_rank_loss(
            real_scores, fake_scores, real_features, fake_features, conditional
        )
        self.assertGreater(loss_unconditional.item(), 0.0)
        self.assertAlmostEqual(loss_conditional.item(), 0.0, places=6)

    def test_local_rank_updates_only_discriminator(self):
        loss_obj = R3GANLoss(
            G=TinyG(),
            D=TinyFeatureD(),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=2,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss_obj.accumulate_gradients("D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        loss_obj.accumulate_gradients("D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        loss_obj.finalize_accumulation()
        self.assertTrue(any(p.grad is not None for p in loss_obj.D.parameters()))
        self.assertTrue(all(p.grad is None for p in loss_obj.G.parameters()))


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestPathRankingLosses(unittest.TestCase):
    def test_listmle_correct_order_low_loss(self):
        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        loss = listmle_loss(scores)
        scores_rev = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss_rev = listmle_loss(scores_rev)
        self.assertGreater(loss_rev.item(), loss.item())

    def test_pairwise_hinge_zero_loss_when_margin_satisfied(self):
        scores = torch.tensor([[10.0, 5.0, 0.0]])
        loss = pairwise_hinge_loss(scores, margin=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_pairwise_logistic_matches_rpgan_for_k2(self):
        real_score = torch.tensor([2.0])
        fake_score = torch.tensor([-1.0])
        scores = torch.stack([real_score, fake_score], dim=-1)
        rank_loss = pairwise_logistic_loss(scores)
        rpgan_loss = F.softplus(-(real_score - fake_score)).mean()
        self.assertTrue(torch.allclose(rank_loss, rpgan_loss, atol=1e-5))

    def test_make_rank_list_endpoints(self):
        real = torch.randn(2, 3, 4, 4)
        fake = torch.randn(2, 3, 4, 4)
        result = make_rank_list(real, fake, k=4, mode="intrpl", alpha_dist="linear")
        self.assertEqual(result.shape, (2, 4, 3, 4, 4))
        self.assertTrue(torch.allclose(result[:, 0], real))
        self.assertTrue(torch.allclose(result[:, -1], fake))

    def test_make_rank_list_alpha_monotonic(self):
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

    def test_path_rank_loss_ignores_augmentation(self):
        real = torch.zeros(2, 3, 4, 4)
        fake = torch.ones(2, 3, 4, 4)
        cond = torch.zeros(2, 0)
        loss_plain = R3GANLoss(
            G=TinyG(),
            D=MeanScoreD(),
            path_rank_reg=True,
            path_rank_k=3,
            augment_pipe=None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss_aug = R3GANLoss(
                G=TinyG(),
                D=MeanScoreD(),
                path_rank_reg=True,
                path_rank_k=3,
                augment_pipe=OffsetAugment(5.0),
                rank_augment=True,
            )
        self.assertTrue(any("deprecated and ignored" in str(w.message) for w in caught))
        loss_plain_value, _ = loss_plain._compute_path_rank_loss(real, fake, cond)
        loss_aug_value, _ = loss_aug._compute_path_rank_loss(real, fake, cond)
        self.assertTrue(torch.allclose(loss_plain_value, loss_aug_value, atol=1e-6))


@unittest.skipIf(
    torch is None or CliRunner is None or dnnlib is None,
    "Test dependencies are not available in this environment",
)
class TestCliMappings(unittest.TestCase):
    def _invoke_train(self, extra_args):
        dataset_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path="dummy.zip",
            use_labels=False,
            max_size=8,
            xflip=False,
            resolution=32,
            random_seed=0,
        )
        runner = CliRunner()
        with mock.patch.object(
            train_cli, "init_dataset_kwargs", return_value=(dataset_kwargs, "dummy")
        ), mock.patch.object(
            train_cli.metric_main, "is_valid_metric", return_value=True
        ), mock.patch.object(train_cli, "launch_training") as launch_training:
            result = runner.invoke(
                train_cli.main,
                [
                    "--outdir=/tmp/out",
                    "--data=dummy.zip",
                    "--gpus=1",
                    "--batch=1",
                    "--preset=CIFAR10",
                    *extra_args,
                ],
            )
        return result, launch_training

    def test_legacy_adv_flags_map_to_delta_centric_config(self):
        result, launch_training = self._invoke_train(
            ["--adv-loss-type=infonce", "--adv-tau=0.2"]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        kwargs = launch_training.call_args.kwargs
        loss_kwargs = kwargs["c"].loss_kwargs
        self.assertEqual(loss_kwargs.lambda_pair, 0.0)
        self.assertEqual(loss_kwargs.lambda_list, 1.0)
        self.assertEqual(loss_kwargs.list_tau, 0.2)
        self.assertEqual(loss_kwargs.list_loss_type, "infonce")

    def test_legacy_rank_flags_map_to_path_rank_and_ignore_rank_augment(self):
        result, launch_training = self._invoke_train(
            ["--rank-loss=1", "--rank-k=3", "--rank-augment=1"]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        kwargs = launch_training.call_args.kwargs
        loss_kwargs = kwargs["c"].loss_kwargs
        self.assertTrue(loss_kwargs.path_rank_reg)
        self.assertEqual(loss_kwargs.path_rank_k, 3)
        self.assertIn("deprecated; use --path-rank-* instead", result.output)
        self.assertIn("deprecated and ignored", result.output)


if __name__ == "__main__":
    unittest.main()
