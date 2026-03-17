"""Parity tests: training.drift_field vs drifting_models.drift_field and toy.py."""

from __future__ import annotations

import sys
import unittest

import torch

# Make the reference drift_models package importable.
sys.path.insert(0, "/workspace/drift_models")

from training.drift_field import (
    DriftFieldConfig as NewDriftFieldConfig,
    cfg_alpha_to_unconditional_weight as new_cfg_alpha,
    build_negative_log_weights as new_build_neg,
    compute_affinity_matrices as new_compute_affinity,
    compute_v as new_compute_v,
)
from drifting_models.drift_field import (
    DriftFieldConfig as RefDriftFieldConfig,
    cfg_alpha_to_unconditional_weight as ref_cfg_alpha,
    build_negative_log_weights as ref_build_neg,
    compute_affinity_matrices as ref_compute_affinity,
    compute_v as ref_compute_v,
)

from training.drift_loss import (
    DriftingLossConfig as NewDriftingLossConfig,
    drifting_stopgrad_loss as new_stopgrad_loss,
    drifting_stopgrad_loss_multi_temperature as new_multi_temp_loss,
    compute_weighted_drift as new_compute_weighted_drift,
)
from drifting_models.drift_loss import (
    DriftingLossConfig as RefDriftingLossConfig,
    drifting_stopgrad_loss as ref_stopgrad_loss,
    drifting_stopgrad_loss_multi_temperature as ref_multi_temp_loss,
    compute_weighted_drift as ref_compute_weighted_drift,
)
from drifting_models.drift_field import DriftFieldConfig as RefDriftFieldConfig2

from training.drift_grouped import (
    infer_grouped_shapes as new_infer_grouped_shapes,
    compute_grouped_v as new_compute_grouped_v,
)
from drifting_models.train.grouped import (
    infer_grouped_shapes as ref_infer_grouped_shapes,
    compute_grouped_v as ref_compute_grouped_v,
)
from drifting_models.drift_field import DriftFieldConfig as RefDriftFieldConfig3


def toy_compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    """
    Standalone compute_drift copied from R3GAN/toy.py lines 135-163.

    Compute drift field V with attention-based kernel.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for softmax kernel

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()  # unnormalized kernel

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V


class TestDriftFieldParity(unittest.TestCase):
    """Verify training.drift_field matches drifting_models.drift_field exactly."""

    # ------------------------------------------------------------------
    # cfg_alpha_to_unconditional_weight
    # ------------------------------------------------------------------
    def test_cfg_alpha_to_unconditional_weight_parity(self) -> None:
        cases = [
            (1.0, 4, 2),
            (3.0, 4, 2),
            (2.5, 8, 3),
            (1.0, 2, 1),
        ]
        for alpha, n_gen, n_unc in cases:
            with self.subTest(alpha=alpha, n_gen=n_gen, n_unc=n_unc):
                new_val = new_cfg_alpha(alpha, n_gen, n_unc)
                ref_val = ref_cfg_alpha(alpha, n_gen, n_unc)
                self.assertEqual(new_val, ref_val)

    # ------------------------------------------------------------------
    # build_negative_log_weights
    # ------------------------------------------------------------------
    def test_build_negative_log_weights_parity(self) -> None:
        cases = [
            (4, 2, 1.0),
            (4, 2, 0.0),
            (4, 0, 0.0),
            (8, 3, 0.5),
        ]
        device = torch.device("cpu")
        dtype = torch.float32
        for n_gen, n_unc, w in cases:
            with self.subTest(n_gen=n_gen, n_unc=n_unc, w=w):
                new_t = new_build_neg(n_gen, n_unc, w, device=device, dtype=dtype)
                ref_t = ref_build_neg(n_gen, n_unc, w, device=device, dtype=dtype)
                torch.testing.assert_close(new_t, ref_t)

    # ------------------------------------------------------------------
    # compute_affinity_matrices
    # ------------------------------------------------------------------
    def test_compute_affinity_matrices_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(5, 8)
        y_pos = torch.randn(4, 8)
        y_neg = torch.randn(5, 8)
        temp = 0.1

        new_cfg = NewDriftFieldConfig(temperature=temp)
        ref_cfg = RefDriftFieldConfig(temperature=temp)

        new_aff_pos, new_aff_neg = new_compute_affinity(x, y_pos, y_neg, config=new_cfg)
        ref_aff_pos, ref_aff_neg = ref_compute_affinity(x, y_pos, y_neg, config=ref_cfg)

        torch.testing.assert_close(new_aff_pos, ref_aff_pos)
        torch.testing.assert_close(new_aff_neg, ref_aff_neg)

    # ------------------------------------------------------------------
    # compute_v
    # ------------------------------------------------------------------
    def test_compute_v_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(5, 8)
        y_pos = torch.randn(4, 8)
        y_neg = torch.randn(5, 8)
        temp = 0.1

        new_cfg = NewDriftFieldConfig(temperature=temp)
        ref_cfg = RefDriftFieldConfig(temperature=temp)

        new_v = new_compute_v(x, y_pos, y_neg, config=new_cfg)
        ref_v = ref_compute_v(x, y_pos, y_neg, config=ref_cfg)

        torch.testing.assert_close(new_v, ref_v)

    # ------------------------------------------------------------------
    # Three-way parity: compute_v vs toy_compute_drift
    # ------------------------------------------------------------------
    def test_compute_v_matches_toy_drift(self) -> None:
        torch.manual_seed(42)
        gen = torch.randn(10, 2)
        pos = torch.randn(20, 2)
        temp = 0.2

        # toy.py compute_drift uses gen as both x and negatives,
        # with self-masking via fill_diagonal_.
        toy_v = toy_compute_drift(gen, pos, temp=temp)

        # The drift_field module equivalent: x=gen, y_pos=pos, y_neg=gen,
        # with generated_negative_count=gen.shape[0] to enable self-mask.
        new_cfg = NewDriftFieldConfig(temperature=temp)
        ref_cfg = RefDriftFieldConfig(temperature=temp)

        new_v = new_compute_v(
            gen, pos, gen,
            config=new_cfg,
            generated_negative_count=gen.shape[0],
        )
        ref_v = ref_compute_v(
            gen, pos, gen,
            config=ref_cfg,
            generated_negative_count=gen.shape[0],
        )

        # All three should agree.
        torch.testing.assert_close(new_v, ref_v)
        torch.testing.assert_close(new_v, toy_v, atol=1e-5, rtol=1e-5)


class TestDriftLossParity(unittest.TestCase):
    """Verify training.drift_loss matches drifting_models.drift_loss exactly."""

    # ------------------------------------------------------------------
    # drifting_stopgrad_loss
    # ------------------------------------------------------------------
    def test_drifting_stopgrad_loss_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(5, 16)
        y_pos = torch.randn(4, 16)
        y_neg = torch.randn(5, 16)
        temp = 0.1

        new_cfg = NewDriftingLossConfig(drift_field=NewDriftFieldConfig(temperature=temp))
        ref_cfg = RefDriftingLossConfig(drift_field=RefDriftFieldConfig(temperature=temp))

        new_loss, new_drift, new_stats = new_stopgrad_loss(
            x, y_pos, y_neg, config=new_cfg,
        )
        ref_loss, ref_drift, ref_stats = ref_stopgrad_loss(
            x, y_pos, y_neg, config=ref_cfg,
        )

        torch.testing.assert_close(new_loss, ref_loss)
        torch.testing.assert_close(new_drift, ref_drift)
        for key in ref_stats:
            self.assertAlmostEqual(new_stats[key], ref_stats[key], places=6, msg=f"stat '{key}' mismatch")

    # ------------------------------------------------------------------
    # drifting_stopgrad_loss_multi_temperature
    # ------------------------------------------------------------------
    def test_multi_temperature_loss_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(5, 16)
        y_pos = torch.randn(4, 16)
        y_neg = torch.randn(5, 16)
        temps = (0.01, 0.05, 0.1)

        new_cfg = NewDriftingLossConfig(drift_field=NewDriftFieldConfig(temperature=0.05))
        ref_cfg = RefDriftingLossConfig(drift_field=RefDriftFieldConfig(temperature=0.05))

        new_loss, new_stats = new_multi_temp_loss(
            x, y_pos, y_neg, temperatures=temps, config=new_cfg,
        )
        ref_loss, ref_stats = ref_multi_temp_loss(
            x, y_pos, y_neg, temperatures=temps, config=ref_cfg,
        )

        torch.testing.assert_close(new_loss, ref_loss)
        for key in ref_stats:
            self.assertAlmostEqual(new_stats[key], ref_stats[key], places=6, msg=f"stat '{key}' mismatch")

    # ------------------------------------------------------------------
    # compute_weighted_drift
    # ------------------------------------------------------------------
    def test_compute_weighted_drift_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(5, 16)
        y_pos = torch.randn(4, 16)
        y_neg = torch.randn(5, 16)

        new_cfg = NewDriftingLossConfig(drift_field=NewDriftFieldConfig(temperature=0.1))
        ref_cfg = RefDriftingLossConfig(drift_field=RefDriftFieldConfig(temperature=0.1))

        new_drift, new_stats = new_compute_weighted_drift(
            x, y_pos, y_neg, config=new_cfg,
        )
        ref_drift, ref_stats = ref_compute_weighted_drift(
            x, y_pos, y_neg, config=ref_cfg,
        )

        torch.testing.assert_close(new_drift, ref_drift)
        for key in ref_stats:
            self.assertAlmostEqual(new_stats[key], ref_stats[key], places=6, msg=f"stat '{key}' mismatch")


class TestDriftGroupedParity(unittest.TestCase):
    """Verify training.drift_grouped matches drifting_models.train.grouped exactly."""

    # ------------------------------------------------------------------
    # infer_grouped_shapes
    # ------------------------------------------------------------------
    def test_infer_grouped_shapes_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(3, 4, 16)
        y_pos = torch.randn(3, 5, 16)
        y_neg = torch.randn(3, 4, 16)

        new_shapes = new_infer_grouped_shapes(x, y_pos, y_neg)
        ref_shapes = ref_infer_grouped_shapes(x, y_pos, y_neg)

        self.assertEqual(new_shapes.groups, ref_shapes.groups)
        self.assertEqual(new_shapes.negatives_per_group, ref_shapes.negatives_per_group)
        self.assertEqual(new_shapes.positives_per_group, ref_shapes.positives_per_group)
        self.assertEqual(new_shapes.feature_dim, ref_shapes.feature_dim)

    # ------------------------------------------------------------------
    # compute_grouped_v
    # ------------------------------------------------------------------
    def test_compute_grouped_v_parity(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(3, 4, 16)
        y_pos = torch.randn(3, 5, 16)
        y_neg = torch.randn(3, 4, 16)
        temp = 0.1

        new_cfg = NewDriftFieldConfig(temperature=temp)
        ref_cfg = RefDriftFieldConfig3(temperature=temp)

        new_v = new_compute_grouped_v(x, y_pos, y_neg, config=new_cfg)
        ref_v = ref_compute_grouped_v(x, y_pos, y_neg, config=ref_cfg)

        torch.testing.assert_close(new_v, ref_v)


class TestFeaturesParity(unittest.TestCase):
    """Verify training.features matches drifting_models.features exactly."""

    # ------------------------------------------------------------------
    # vectorize_feature_maps
    # ------------------------------------------------------------------
    def test_vectorize_feature_maps_parity(self) -> None:
        from training.features.vectorize import (
            FeatureVectorizationConfig as NewVecConfig,
            vectorize_feature_maps as new_vectorize,
        )
        from drifting_models.features.vectorize import (
            FeatureVectorizationConfig as RefVecConfig,
            vectorize_feature_maps as ref_vectorize,
        )

        torch.manual_seed(42)
        fmaps = [torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)]

        new_cfg = NewVecConfig()
        ref_cfg = RefVecConfig()

        new_result = new_vectorize(fmaps, config=new_cfg)
        ref_result = ref_vectorize(fmaps, config=ref_cfg)

        self.assertEqual(set(new_result.keys()), set(ref_result.keys()))
        for key in ref_result:
            torch.testing.assert_close(new_result[key], ref_result[key], msg=f"key '{key}' mismatch")

    # ------------------------------------------------------------------
    # TinyFeatureEncoder shape
    # ------------------------------------------------------------------
    def test_tiny_feature_encoder_shape(self) -> None:
        from training.features.extractors import (
            TinyFeatureEncoderConfig as NewEncoderConfig,
            TinyFeatureEncoder as NewEncoder,
        )

        config = NewEncoderConfig(in_channels=4, base_channels=16, stages=3)
        encoder = NewEncoder(config)
        images = torch.randn(2, 4, 16, 16)
        features = encoder(images)
        self.assertEqual(len(features), 3)

    # ------------------------------------------------------------------
    # TinyFeatureEncoder weight parity
    # ------------------------------------------------------------------
    def test_tiny_feature_encoder_weight_parity(self) -> None:
        from training.features.extractors import (
            TinyFeatureEncoderConfig as NewEncoderConfig,
            TinyFeatureEncoder as NewEncoder,
        )
        from drifting_models.features.extractors import (
            TinyFeatureEncoderConfig as RefEncoderConfig,
            TinyFeatureEncoder as RefEncoder,
        )

        new_cfg = NewEncoderConfig(in_channels=4, base_channels=16, stages=3)
        ref_cfg = RefEncoderConfig(in_channels=4, base_channels=16, stages=3)

        torch.manual_seed(42)
        new_encoder = NewEncoder(new_cfg)
        torch.manual_seed(42)
        ref_encoder = RefEncoder(ref_cfg)

        # Copy weights from ref to new to ensure exact parity
        new_encoder.load_state_dict(ref_encoder.state_dict())

        torch.manual_seed(99)
        images = torch.randn(2, 4, 16, 16)

        new_features = new_encoder(images)
        ref_features = ref_encoder(images)

        self.assertEqual(len(new_features), len(ref_features))
        for i, (nf, rf) in enumerate(zip(new_features, ref_features)):
            torch.testing.assert_close(nf, rf, msg=f"stage {i} mismatch")


class TestQueueParity(unittest.TestCase):
    """Verify training.drift_queue queue contract (push, sample, state_dict, version)."""

    # ------------------------------------------------------------------
    # test_queue_push_and_sample
    # ------------------------------------------------------------------
    def test_queue_push_and_sample(self) -> None:
        from training.drift_queue import QueueConfig, ClassConditionalSampleQueue

        config = QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64)
        queue = ClassConditionalSampleQueue(config)

        # Push 6 images with labels [0, 1, 2, 0, 1, 2]
        images = torch.randn(6, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        queue.push(images, labels)

        # Verify counts
        self.assertEqual(queue.class_count(0), 2)
        self.assertEqual(queue.class_count(1), 2)
        self.assertEqual(queue.class_count(2), 2)
        self.assertEqual(queue.global_count(), 6)
        self.assertEqual(queue.class_counts(), [2, 2, 2])

        # Sample positives — should not raise
        class_labels = torch.tensor([0, 1, 2])
        positives = queue.sample_positive_grouped(class_labels, 2, torch.device("cpu"))
        self.assertEqual(positives.shape, (3, 2, 3, 4, 4))

    # ------------------------------------------------------------------
    # test_queue_state_dict_roundtrip
    # ------------------------------------------------------------------
    def test_queue_state_dict_roundtrip(self) -> None:
        from training.drift_queue import QueueConfig, ClassConditionalSampleQueue

        config = QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64)
        queue = ClassConditionalSampleQueue(config)

        images = torch.randn(6, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        queue.push(images, labels)

        state = queue.state_dict()

        # Restore into a fresh queue
        queue2 = ClassConditionalSampleQueue(config)
        queue2.load_state_dict(state)

        self.assertEqual(queue2.class_counts(), queue.class_counts())
        self.assertEqual(queue2.global_count(), queue.global_count())

    # ------------------------------------------------------------------
    # test_queue_version_field
    # ------------------------------------------------------------------
    def test_queue_version_field(self) -> None:
        from training.drift_queue import QueueConfig, ClassConditionalSampleQueue

        config = QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64)
        queue = ClassConditionalSampleQueue(config)

        images = torch.randn(2, 3, 4, 4)
        labels = torch.tensor([0, 1])
        queue.push(images, labels)

        state = queue.state_dict()
        self.assertIn("version", state)
        self.assertEqual(state["version"], 1)


class TestDriftStage2Parity(unittest.TestCase):
    """Verify training.drift_stage2 matches drifting_models.train.stage2 exactly."""

    def test_grouped_drift_step_raw_loss_parity(self) -> None:
        from training.drift_field import DriftFieldConfig as NewFieldConfig
        from training.drift_loss import DriftingLossConfig as NewLossConfig
        from training.drift_stage2 import (
            GroupedDriftStepConfig as NewStepConfig,
            grouped_drift_training_step as new_step,
        )
        from training.models.dit_like import DiTLikeConfig as NewDiTConfig, DiTLikeGenerator as NewDiTGen

        from drifting_models.train.stage2 import (
            GroupedDriftStepConfig as RefStepConfig,
            grouped_drift_training_step as ref_step,
        )
        from drifting_models.drift_loss import DriftingLossConfig as RefLossConfig
        from drifting_models.drift_field import DriftFieldConfig as RefFieldConfig
        from drifting_models.models.dit_like import DiTLikeConfig as RefDiTConfig, DiTLikeGenerator as RefDiTGen

        # Build identical generators
        new_dit_cfg = NewDiTConfig(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=10,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        )
        ref_dit_cfg = RefDiTConfig(
            image_size=8, in_channels=4, out_channels=4, patch_size=2,
            hidden_dim=32, depth=2, num_heads=4, num_classes=10,
            register_tokens=4, style_vocab_size=4, style_token_count=2,
            alpha_hidden_dim=16,
        )

        torch.manual_seed(42)
        new_gen = NewDiTGen(new_dit_cfg)
        torch.manual_seed(42)
        ref_gen = RefDiTGen(ref_dit_cfg)

        # Copy weights from new to ref to ensure exact parity
        ref_gen.load_state_dict(new_gen.state_dict())

        # Create separate optimizers
        new_opt = torch.optim.Adam(new_gen.parameters(), lr=1e-4)
        ref_opt = torch.optim.Adam(ref_gen.parameters(), lr=1e-4)

        # Fixed seed=99 for inputs
        torch.manual_seed(99)
        noise = torch.randn(2, 3, 4, 8, 8)
        labels = torch.tensor([0, 5])
        alpha = torch.tensor([1.5, 2.5])
        positives = torch.randn(2, 4, 4, 8, 8)
        unconditional = torch.randn(2, 2, 4, 8, 8)
        unc_weights = torch.tensor([1.0, 0.5])

        # Configs
        new_loss_cfg = NewLossConfig(drift_field=NewFieldConfig(temperature=0.1))
        ref_loss_cfg = RefLossConfig(drift_field=RefFieldConfig(temperature=0.1))

        new_step_cfg = NewStepConfig(loss_config=new_loss_cfg)
        ref_step_cfg = RefStepConfig(loss_config=ref_loss_cfg)

        # Call both
        new_stats = new_step(
            generator=new_gen,
            optimizer=new_opt,
            noise_grouped=noise,
            class_labels_grouped=labels,
            alpha_grouped=alpha,
            positives_grouped=positives,
            style_indices_grouped=None,
            unconditional_grouped=unconditional,
            unconditional_weight_grouped=unc_weights,
            config=new_step_cfg,
        )
        ref_stats = ref_step(
            generator=ref_gen,
            optimizer=ref_opt,
            noise_grouped=noise,
            class_labels_grouped=labels,
            alpha_grouped=alpha,
            positives_grouped=positives,
            style_indices_grouped=None,
            unconditional_grouped=unconditional,
            unconditional_weight_grouped=unc_weights,
            config=ref_step_cfg,
        )

        self.assertAlmostEqual(new_stats["loss"], ref_stats["loss"], places=4)
        self.assertAlmostEqual(new_stats["mean_drift_norm"], ref_stats["mean_drift_norm"], places=4)


if __name__ == "__main__":
    unittest.main()
