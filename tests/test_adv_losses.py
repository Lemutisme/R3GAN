import copy
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
        build_local_coupling,
        infonce_discriminator_loss,
        infonce_discriminator_loss_with_grads,
        infonce_generator_loss,
        infonce_generator_loss_with_grads,
        listmle_loss,
        local_delta,
        local_rank_loss_with_grads,
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
class TestCoupledLossHelpers(unittest.TestCase):
    def test_infonce_discriminator_grad_coefficients_match_autograd(self):
        real = torch.randn(5)
        fake = torch.randn(5)
        tau = 0.19
        loss_value, loss_vector, grad_real, grad_fake = infonce_discriminator_loss_with_grads(
            real, fake, tau=tau
        )

        real_var = real.detach().clone().requires_grad_(True)
        fake_var = fake.detach().clone().requires_grad_(True)
        direct_loss = infonce_discriminator_loss(allpairs_delta(real_var, fake_var), tau=tau).mean()
        direct_loss.backward()

        self.assertTrue(torch.allclose(loss_value, direct_loss.detach(), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                loss_vector,
                infonce_discriminator_loss(allpairs_delta(real, fake), tau=tau),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(grad_real, real_var.grad, atol=1e-6))
        self.assertTrue(torch.allclose(grad_fake, fake_var.grad, atol=1e-6))

    def test_local_rank_grad_coefficients_match_autograd(self):
        real_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
        fake_features = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
        fake_scores = torch.tensor([2.0, 1.5, 0.5])
        class_ids = torch.tensor([0, 0, 1])

        loss_value, adjacent_losses, grad_fake = local_rank_loss_with_grads(
            real_features=real_features,
            fake_features=fake_features,
            fake_scores=fake_scores,
            class_ids=class_ids,
            k=2,
        )

        score_var = fake_scores.detach().clone().requires_grad_(True)
        direct_loss = R3GANLoss(
            G=TinyG(),
            D=TinyFeatureD(),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=2,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )._compute_local_rank_loss(
            real_scores=torch.zeros_like(score_var),
            fake_scores=score_var,
            real_features=real_features,
            fake_features=fake_features,
            real_c=F.one_hot(class_ids, num_classes=2).to(torch.float32),
        )
        direct_loss.backward()

        self.assertGreater(adjacent_losses.numel(), 0)
        self.assertTrue(torch.allclose(loss_value, direct_loss.detach(), atol=1e-6))
        self.assertTrue(torch.allclose(grad_fake, score_var.grad, atol=1e-6))


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestR3GANLossBuffering(unittest.TestCase):
    def _make_loss(self, **kwargs):
        g = TinyG()
        d = kwargs.pop("D", TinyFeatureD())
        return R3GANLoss(G=g, D=d, **kwargs)

    def _set_phase_requires_grad(self, loss_obj, phase):
        loss_obj.G.requires_grad_(phase == "G")
        loss_obj.D.requires_grad_(phase == "D")

    def _grads(self, module):
        return [None if p.grad is None else p.grad.detach().clone() for p in module.parameters()]

    def _assert_grad_lists_close(self, got, expected, atol=1e-6):
        self.assertEqual(len(got), len(expected))
        for grad_got, grad_expected in zip(got, expected):
            if grad_expected is None:
                self.assertIsNone(grad_got)
            else:
                self.assertIsNotNone(grad_got)
                self.assertTrue(torch.allclose(grad_got, grad_expected, atol=atol))

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

    def test_local_rank_only_does_not_buffer_generator_phase(self):
        loss_obj = self._make_loss(
            lambda_pair=1.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        self._set_phase_requires_grad(loss_obj, "G")
        real = torch.randn(2, 3, 4, 4)
        cond = torch.zeros(2, 0)
        noise = torch.randn(2, 4)
        loss_obj.accumulate_gradients("G", real, cond, noise, gamma=0.1, gain=1.0)
        self.assertIsNone(loss_obj._coupled_phase_buffer)
        self.assertTrue(any(p.grad is not None for p in loss_obj.G.parameters()))

    def test_listwise_replay_matches_full_batch_generator_gradients(self):
        torch.manual_seed(3)
        base_g = TinyG()
        base_d = TinyFeatureD()
        replay_loss = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=1.0,
            lambda_list=1.0,
            list_tau=0.11,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        direct_loss = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=1.0,
            lambda_list=1.0,
            list_tau=0.11,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        self._set_phase_requires_grad(replay_loss, "G")
        self._set_phase_requires_grad(direct_loss, "G")

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        fake = direct_loss.G(noise, cond)
        real_scores = direct_loss.run_D(real.detach(), cond, augment=True)
        fake_scores = direct_loss.run_D(fake, cond, augment=True)
        pair_term = pairwise_generator_loss(
            pairwise_delta(real_scores, fake_scores), margin=direct_loss.pair_margin
        ).mean()
        list_term = infonce_generator_loss(
            allpairs_delta(real_scores, fake_scores), tau=direct_loss.list_tau
        ).mean()
        (pair_term + list_term).backward()
        expected_grads = self._grads(direct_loss.G)

        replay_loss.accumulate_gradients("G", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        replay_loss.accumulate_gradients("G", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        replay_loss.finalize_accumulation()
        replay_grads = self._grads(replay_loss.G)

        self._assert_grad_lists_close(replay_grads, expected_grads)
        self.assertTrue(all(p.grad is None for p in replay_loss.D.parameters()))

    def test_listwise_replay_matches_full_batch_discriminator_gradients(self):
        torch.manual_seed(7)
        base_g = TinyG()
        base_d = TinyFeatureD()
        replay_loss = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=1.0,
            lambda_list=1.0,
            list_tau=0.23,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        direct_loss = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=1.0,
            lambda_list=1.0,
            list_tau=0.23,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        self._set_phase_requires_grad(replay_loss, "D")
        self._set_phase_requires_grad(direct_loss, "D")

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        fake = direct_loss.G(noise, cond).detach()
        real_scores = direct_loss.run_D(real, cond, augment=True)
        fake_scores = direct_loss.run_D(fake, cond, augment=True)
        pair_term = pairwise_discriminator_loss(
            pairwise_delta(real_scores, fake_scores), margin=direct_loss.pair_margin
        ).mean()
        list_term = infonce_discriminator_loss(
            allpairs_delta(real_scores, fake_scores), tau=direct_loss.list_tau
        ).mean()
        (pair_term + list_term).backward()
        expected_grads = self._grads(direct_loss.D)

        replay_loss.accumulate_gradients("D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        replay_loss.accumulate_gradients("D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        replay_loss.finalize_accumulation()
        replay_grads = self._grads(replay_loss.D)

        self._assert_grad_lists_close(replay_grads, expected_grads)
        self.assertTrue(all(p.grad is None for p in replay_loss.G.parameters()))


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
        self.assertIsNone(real_scores.grad)
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

    def test_local_rank_replay_matches_full_batch_gradients(self):
        torch.manual_seed(11)
        base_g = TinyG()
        base_d = TinyFeatureD()
        replay_loss = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=3,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        direct_loss = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=3,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        replay_loss.G.requires_grad_(False)
        replay_loss.D.requires_grad_(True)
        direct_loss.G.requires_grad_(False)
        direct_loss.D.requires_grad_(True)

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        fake = direct_loss.G(noise, cond).detach()
        _, real_features = direct_loss.run_D(real, cond, augment=False, return_features=True)
        clean_fake_scores, fake_features = direct_loss.run_D(
            fake, cond, augment=False, return_features=True
        )
        local_rank_loss = direct_loss._compute_local_rank_loss(
            real_scores=torch.zeros_like(clean_fake_scores),
            fake_scores=clean_fake_scores,
            real_features=real_features,
            fake_features=fake_features,
            real_c=cond,
        )
        local_rank_loss.backward()
        expected_grads = [
            None if p.grad is None else p.grad.detach().clone() for p in direct_loss.D.parameters()
        ]

        replay_loss.accumulate_gradients("D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        replay_loss.accumulate_gradients("D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        replay_loss.finalize_accumulation()
        replay_grads = [
            None if p.grad is None else p.grad.detach().clone() for p in replay_loss.D.parameters()
        ]

        self.assertEqual(len(replay_grads), len(expected_grads))
        for grad_replay, grad_expected in zip(replay_grads, expected_grads):
            if grad_expected is None:
                self.assertIsNone(grad_replay)
            else:
                self.assertTrue(torch.allclose(grad_replay, grad_expected, atol=1e-6))

    def test_local_rank_replay_ignores_augmentation(self):
        torch.manual_seed(13)
        base_g = TinyG()
        base_d = TinyFeatureD()
        loss_plain = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=2,
            use_r1_penalty=False,
            use_r2_penalty=False,
            augment_pipe=None,
        )
        loss_aug = R3GANLoss(
            G=copy.deepcopy(base_g),
            D=copy.deepcopy(base_d),
            lambda_pair=0.0,
            lambda_list=0.0,
            lambda_local_rank=1.0,
            local_rank_k=2,
            use_r1_penalty=False,
            use_r2_penalty=False,
            augment_pipe=OffsetAugment(5.0),
        )
        loss_plain.G.requires_grad_(False)
        loss_plain.D.requires_grad_(True)
        loss_aug.G.requires_grad_(False)
        loss_aug.D.requires_grad_(True)

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        loss_plain.accumulate_gradients("D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        loss_plain.accumulate_gradients("D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        loss_plain.finalize_accumulation()

        loss_aug.accumulate_gradients("D", real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        loss_aug.accumulate_gradients("D", real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        loss_aug.finalize_accumulation()

        for param_plain, param_aug in zip(loss_plain.D.parameters(), loss_aug.D.parameters()):
            if param_plain.grad is None or param_aug.grad is None:
                self.assertIsNone(param_plain.grad)
                self.assertIsNone(param_aug.grad)
            else:
                self.assertTrue(torch.allclose(param_plain.grad, param_aug.grad, atol=1e-6))


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
    torch is None,
    "PyTorch is not available in this environment",
)
class TestStatsSemantics(unittest.TestCase):
    def test_score_stats_report_raw_scores_and_delta(self):
        loss_obj = R3GANLoss(
            G=TinyG(),
            D=TinyD(),
            lambda_pair=1.0,
            lambda_list=0.0,
            use_r1_penalty=False,
            use_r2_penalty=False,
        )
        loss_obj.G.requires_grad_(False)
        loss_obj.D.requires_grad_(True)
        real = torch.randn(2, 3, 4, 4)
        cond = torch.zeros(2, 0)
        noise = torch.randn(2, 4)

        with torch.no_grad():
            fake = loss_obj.G(noise, cond).detach()
            expected_real_scores = loss_obj.run_D(real, cond, augment=True)
            expected_fake_scores = loss_obj.run_D(fake, cond, augment=True)
            expected_delta = pairwise_delta(expected_real_scores, expected_fake_scores)

        reports = {}

        def _capture(name, value):
            if torch.is_tensor(value):
                reports[name] = value.detach().clone()
            else:
                reports[name] = torch.as_tensor(value)

        with mock.patch("training.loss.training_stats.report", side_effect=_capture):
            loss_obj.accumulate_gradients("D", real, cond, noise, gamma=0.1, gain=1.0)

        self.assertTrue(torch.allclose(reports["Loss/scores/real"], expected_real_scores))
        self.assertTrue(torch.allclose(reports["Loss/scores/fake"], expected_fake_scores))
        self.assertTrue(torch.allclose(reports["Loss/delta/pair"], expected_delta))


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

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_k_creates_r3ganloss_with_coupling(self):
        """Verify that coupling_k parameter is accepted by R3GANLoss."""
        from training.loss import R3GANLoss

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

        loss = R3GANLoss(
            G=TinyG(), D=TinyD(),
            coupling_k=4, lambda_list_d=0.5, lambda_list_g=0.1, lambda_pair=1.0,
        )
        self.assertEqual(loss.coupling_k, 4)
        self.assertEqual(loss.lambda_list_d, 0.5)
        self.assertEqual(loss.lambda_list_g, 0.1)


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
        neighbor_indices = torch.tensor([[0, 2], [1, 0]])
        delta = local_delta(real_scores, fake_scores, neighbor_indices)
        expected = torch.tensor([[3.0 - 2.0, 5.0 - 2.0], [1.0 - 4.0, 3.0 - 4.0]])
        self.assertTrue(torch.allclose(delta, expected))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_build_local_coupling_no_feature_gradients(self):
        from training.loss import build_local_coupling
        real_feat = torch.randn(4, 8, requires_grad=True)
        fake_feat = torch.randn(3, 8, requires_grad=True)
        indices, weights = build_local_coupling(real_feat, fake_feat, k=2)
        self.assertFalse(weights.requires_grad, 'Coupling weights must not track gradients')

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
        self.assertGreater(loss.item(), 0)  # softplus always > 0 but small for correct ordering

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
        self.assertEqual(d_loss.shape, (2,))
        self.assertEqual(g_loss.shape, (2,))
        # Both should be finite
        self.assertTrue(torch.isfinite(d_loss).all())
        self.assertTrue(torch.isfinite(g_loss).all())

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

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_d_loss_correct_ordering_low(self):
        from training.loss import local_delta, local_listwise_discriminator_loss
        real_scores = torch.tensor([10.0, 8.0])
        fake_scores = torch.tensor([0.0])
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss = local_listwise_discriminator_loss(delta, weights, tau=0.1)
        self.assertEqual(loss.shape, (1,))
        self.assertLess(loss.item(), 0.01)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_wrong_ordering_high(self):
        from training.loss import local_delta, local_listwise_discriminator_loss
        real_scores = torch.tensor([0.0, -1.0])
        fake_scores = torch.tensor([5.0])
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        loss = local_listwise_discriminator_loss(delta, weights, tau=0.1)
        self.assertGreater(loss.item(), 1.0)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_numerical_stability(self):
        from training.loss import local_delta, local_listwise_discriminator_loss, local_listwise_generator_loss
        real_scores = torch.tensor([-100.0, -200.0])
        fake_scores = torch.tensor([100.0])
        indices = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.5, 0.5]])
        delta = local_delta(real_scores, fake_scores, indices)
        d_loss = local_listwise_discriminator_loss(delta, weights, tau=0.01)
        g_loss = local_listwise_generator_loss(delta, weights, tau=0.01)
        self.assertTrue(torch.isfinite(d_loss).all(), 'D loss must be finite with extreme values')
        self.assertTrue(torch.isfinite(g_loss).all(), 'G loss must be finite with extreme values')

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_tau_validation(self):
        from training.loss import local_listwise_discriminator_loss
        delta = torch.tensor([[1.0, 2.0]])
        weights = torch.tensor([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            local_listwise_discriminator_loss(delta, weights, tau=0.0)
        with self.assertRaises(ValueError):
            local_listwise_discriminator_loss(delta, weights, tau=-1.0)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_listwise_g_loss_symmetric_to_d(self):
        from training.loss import local_delta, local_listwise_discriminator_loss, local_listwise_generator_loss
        torch.manual_seed(42)
        real_scores = torch.randn(4)
        fake_scores = torch.randn(3)
        indices = torch.tensor([[0, 1], [2, 3], [1, 2]])
        weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])
        delta = local_delta(real_scores, fake_scores, indices)
        d_loss = local_listwise_discriminator_loss(delta, weights, tau=0.1)
        g_loss = local_listwise_generator_loss(delta, weights, tau=0.1)
        # Both should be finite and non-negative (log(1 + positive) >= 0)
        self.assertTrue(torch.isfinite(d_loss).all())
        self.assertTrue(torch.isfinite(g_loss).all())
        self.assertTrue((d_loss >= 0).all())
        self.assertTrue((g_loss >= 0).all())

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_coupled_d_grads_match_autograd(self):
        from training.loss import (
            local_coupled_discriminator_loss_with_grads,
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
        from training.loss import (
            local_coupled_generator_loss_with_grads,
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

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_local_coupled_pairwise_only_d(self):
        from training.loss import local_coupled_discriminator_loss_with_grads, build_local_coupling
        torch.manual_seed(42)
        real_feat = torch.randn(4, 8)
        fake_feat = torch.randn(3, 8)
        indices, weights = build_local_coupling(real_feat, fake_feat, k=2)
        loss_val, loss_vec, grad_real, grad_fake = local_coupled_discriminator_loss_with_grads(
            torch.randn(4), torch.randn(3), indices, weights,
            lambda_pair=1.0, pair_margin=0.0, lambda_list=0.0, list_tau=0.07,
        )
        self.assertGreater(loss_val.item(), 0)
        self.assertEqual(loss_vec.shape, (3,))
        self.assertEqual(grad_real.shape, (4,))
        self.assertEqual(grad_fake.shape, (3,))


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
        # With no coupling and pairwise only, should not require full batch
        self.assertFalse(loss._requires_full_batch('D'))
        self.assertFalse(loss._requires_full_batch('G'))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_k_positive_enables_coupling(self):
        loss = self._make_loss(coupling_k=4)
        self.assertEqual(loss.coupling_k, 4)
        self.assertTrue(loss._requires_coupling())
        # Coupling always requires full batch
        self.assertTrue(loss._requires_full_batch('D'))
        self.assertTrue(loss._requires_full_batch('G'))

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_asymmetric_list_weights(self):
        loss = self._make_loss(lambda_list_d=1.0, lambda_list_g=0.1, coupling_k=4)
        self.assertEqual(loss.lambda_list_d, 1.0)
        self.assertEqual(loss.lambda_list_g, 0.1)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_lambda_list_maps_to_symmetric(self):
        loss = self._make_loss(lambda_list=0.5)
        self.assertEqual(loss.lambda_list_d, 0.5)
        self.assertEqual(loss.lambda_list_g, 0.5)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_lambda_list_d_only_defaults_g_to_zero(self):
        loss = self._make_loss(lambda_list_d=0.8)
        self.assertEqual(loss.lambda_list_d, 0.8)
        self.assertEqual(loss.lambda_list_g, 0.0)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_set_list_weights(self):
        loss = self._make_loss(lambda_list_d=0.5, lambda_list_g=0.1)
        loss.set_list_weights(lambda_list_d=0.8, lambda_list_g=0.2)
        self.assertEqual(loss.lambda_list_d, 0.8)
        self.assertEqual(loss.lambda_list_g, 0.2)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_set_list_weights_partial(self):
        loss = self._make_loss(lambda_list_d=0.5, lambda_list_g=0.1)
        loss.set_list_weights(lambda_list_d=0.9)
        self.assertEqual(loss.lambda_list_d, 0.9)
        self.assertEqual(loss.lambda_list_g, 0.1)  # unchanged

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_k_negative_raises(self):
        with self.assertRaises(ValueError):
            self._make_loss(coupling_k=-1)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_buffers_d_phase(self):
        """With coupling_k > 0, D phase should buffer and produce gradients."""
        loss = self._make_loss(coupling_k=2, lambda_pair=1.0)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        self.assertIsNotNone(loss._coupled_phase_buffer)
        loss.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss.D.parameters())
        self.assertTrue(has_grad, 'D should have gradients after coupled finalize')

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupling_buffers_g_phase(self):
        """With coupling_k > 0, G phase should buffer and produce gradients."""
        loss = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_g=0.3)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('G', real, cond, noise, gamma=0.1, gain=1.0)
        self.assertIsNotNone(loss._coupled_phase_buffer)
        loss.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss.G.parameters())
        self.assertTrue(has_grad, 'G should have gradients after coupled finalize')

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupled_replay_matches_full_batch_d_gradients(self):
        """Buffered coupled replay should produce same D param gradients as single full batch."""
        torch.manual_seed(42)
        loss_ref = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_d=0.5)
        loss_buf = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_d=0.5)
        loss_buf.G.load_state_dict(loss_ref.G.state_dict())
        loss_buf.D.load_state_dict(loss_ref.D.state_dict())

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        # Reference: single full batch
        torch.manual_seed(99)
        loss_ref.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        loss_ref.finalize_accumulation()
        ref_grads = {n: p.grad.clone() for n, p in loss_ref.D.named_parameters() if p.grad is not None}

        # Buffered: two micro-batches
        loss_buf.D.zero_grad()
        loss_buf.G.zero_grad()
        torch.manual_seed(99)
        loss_buf.accumulate_gradients('D', real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        loss_buf.accumulate_gradients('D', real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        loss_buf.finalize_accumulation()
        buf_grads = {n: p.grad.clone() for n, p in loss_buf.D.named_parameters() if p.grad is not None}

        for name in ref_grads:
            self.assertTrue(
                torch.allclose(ref_grads[name], buf_grads[name], atol=1e-4),
                f'D grad mismatch for {name}: max diff = {(ref_grads[name] - buf_grads[name]).abs().max().item():.6f}',
            )

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_coupled_replay_matches_full_batch_g_gradients(self):
        """Buffered coupled replay should produce same G param gradients as single full batch."""
        torch.manual_seed(42)
        loss_ref = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_g=0.3)
        loss_buf = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_g=0.3)
        loss_buf.G.load_state_dict(loss_ref.G.state_dict())
        loss_buf.D.load_state_dict(loss_ref.D.state_dict())

        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)

        torch.manual_seed(99)
        loss_ref.accumulate_gradients('G', real, cond, noise, gamma=0.1, gain=1.0)
        loss_ref.finalize_accumulation()
        ref_grads = {n: p.grad.clone() for n, p in loss_ref.G.named_parameters() if p.grad is not None}

        loss_buf.G.zero_grad()
        loss_buf.D.zero_grad()
        torch.manual_seed(99)
        loss_buf.accumulate_gradients('G', real[:2], cond[:2], noise[:2], gamma=0.1, gain=0.5)
        loss_buf.accumulate_gradients('G', real[2:], cond[2:], noise[2:], gamma=0.1, gain=0.5)
        loss_buf.finalize_accumulation()
        buf_grads = {n: p.grad.clone() for n, p in loss_buf.G.named_parameters() if p.grad is not None}

        for name in ref_grads:
            self.assertTrue(
                torch.allclose(ref_grads[name], buf_grads[name], atol=1e-4),
                f'G grad mismatch for {name}',
            )

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_pairwise_only_coupling_produces_d_gradients(self):
        """Coupling with lambda_list_d=0 (pairwise only) should still produce D gradients."""
        loss = self._make_loss(coupling_k=2, lambda_pair=1.0, lambda_list_d=0.0, lambda_list_g=0.0)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        loss.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss.D.parameters())
        self.assertTrue(has_grad)

    @unittest.skipIf(torch is None, 'PyTorch not available')
    def test_existing_noncoupled_listwise_still_works(self):
        """With coupling_k=0 and lambda_list > 0, existing global InfoNCE should still work."""
        loss = self._make_loss(coupling_k=0, lambda_pair=0.0, lambda_list=1.0, list_tau=0.1)
        real = torch.randn(4, 3, 4, 4)
        cond = torch.zeros(4, 0)
        noise = torch.randn(4, 4)
        loss.accumulate_gradients('D', real, cond, noise, gamma=0.1, gain=1.0)
        loss.finalize_accumulation()
        has_grad = any(p.grad is not None for p in loss.D.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
