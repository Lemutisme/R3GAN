"""Integration tests: end-to-end pipeline smoke tests for the drift training system."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.drift_field import DriftFieldConfig, compute_v
from training.drift_loss import (
    DriftingLossConfig,
    FeatureDriftingConfig,
    feature_space_drifting_loss,
)
from training.drift_stage2 import (
    GroupedDriftStepConfig,
    grouped_drift_training_step,
)
from training.drift_queue import (
    ClassConditionalSampleQueue,
    QueueConfig,
)
from training.features.extractors import (
    TinyFeatureEncoder,
    TinyFeatureEncoderConfig,
    freeze_module_parameters,
)
from training.features.vectorize import FeatureVectorizationConfig
from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_checkerboard(n: int, seed: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    return pts / 2.0


class ToyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Test 1: Toy training smoke
# ---------------------------------------------------------------------------

class TestToyTrainingSmoke(unittest.TestCase):
    def test_toy_checkerboard_loss_decreases(self) -> None:
        torch.manual_seed(42)

        model = ToyMLP(in_dim=16, hidden=128, out_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = DriftFieldConfig(temperature=0.1)

        num_steps = 200
        gen_batch = 256
        pos_batch = 512
        losses: list[float] = []

        for step in range(num_steps):
            pos = sample_checkerboard(pos_batch, seed=step)
            noise = torch.randn(gen_batch, 16)
            gen = model(noise)

            v = compute_v(
                gen.detach(),
                pos,
                gen.detach(),
                config=config,
                generated_negative_count=gen.shape[0],
            )
            target = gen.detach() + v
            loss = F.mse_loss(gen, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_first_20 = sum(losses[:20]) / 20.0
        avg_last_20 = sum(losses[-20:]) / 20.0
        # With temperature=0.1, 200 steps, and these batch sizes the drift
        # field magnitude reliably drops to ~65% of its initial value.
        # Using 0.5 as the threshold requires ~600 steps; 0.75 gives margin.
        self.assertLess(
            avg_last_20,
            0.75 * avg_first_20,
            f"Loss did not decrease enough: first 20 avg={avg_first_20:.6f}, "
            f"last 20 avg={avg_last_20:.6f} "
            f"(ratio={avg_last_20/avg_first_20:.4f}, need < 0.75)",
        )


# ---------------------------------------------------------------------------
# Test 2 & 3: Image smoke integration
# ---------------------------------------------------------------------------

class TestImageSmokeIntegration(unittest.TestCase):
    def test_dit_feature_smoke(self) -> None:
        torch.manual_seed(42)

        # -- DiT generator --
        dit_config = DiTLikeConfig(
            image_size=8,
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_dim=32,
            depth=2,
            num_heads=4,
            num_classes=5,
            register_tokens=4,
            style_vocab_size=4,
            style_token_count=2,
            alpha_hidden_dim=16,
        )
        generator = DiTLikeGenerator(dit_config)

        # -- Feature encoder (eval, frozen) --
        encoder_config = TinyFeatureEncoderConfig(
            in_channels=4,
            base_channels=8,
            stages=2,
        )
        feature_encoder = TinyFeatureEncoder(encoder_config)
        feature_encoder.eval()
        freeze_module_parameters(feature_encoder)

        # -- Queue --
        queue_config = QueueConfig(
            num_classes=5,
            per_class_capacity=20,
            global_capacity=100,
        )
        queue = ClassConditionalSampleQueue(queue_config)

        # Prime queue with 10 synthetic images per class
        for class_label in range(5):
            images = torch.randn(10, 4, 8, 8)
            labels = torch.full((10,), class_label, dtype=torch.long)
            queue.push(images, labels)

        # -- Configs --
        vec_config = FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=False,
            include_patch4_stats=False,
            include_input_x2_mean=False,
        )
        feature_config = FeatureDriftingConfig(
            temperatures=(0.05, 0.1),
            vectorization=vec_config,
        )
        loss_config = DriftingLossConfig(
            drift_field=DriftFieldConfig(temperature=0.05),
        )
        step_config = GroupedDriftStepConfig(
            loss_config=loss_config,
            feature_config=feature_config,
            clip_grad_norm=2.0,
        )

        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

        # -- 3 training steps --
        groups = 2
        neg_per_group = 3
        positives_per_group = 4
        unconditional_per_group = 2

        for step_idx in range(3):
            torch.manual_seed(100 + step_idx)

            # Build grouped inputs
            noise = torch.randn(groups, neg_per_group, 4, 8, 8)
            class_labels = torch.randint(0, 5, (groups,))
            alpha = torch.ones(groups) * 1.5

            positives = queue.sample_positive_grouped(
                class_labels, positives_per_group, torch.device("cpu")
            )
            unconditional = queue.sample_unconditional_grouped(
                groups, unconditional_per_group, torch.device("cpu")
            )
            unc_weights = torch.ones(groups)

            stats = grouped_drift_training_step(
                generator=generator,
                optimizer=optimizer,
                noise_grouped=noise,
                class_labels_grouped=class_labels,
                alpha_grouped=alpha,
                positives_grouped=positives,
                style_indices_grouped=None,
                unconditional_grouped=unconditional,
                unconditional_weight_grouped=unc_weights,
                feature_extractor=feature_encoder,
                config=step_config,
            )

            loss_val = stats["loss"]
            self.assertTrue(
                torch.isfinite(torch.tensor(loss_val)).item(),
                f"Step {step_idx}: loss is not finite ({loss_val})",
            )

    def test_checkpoint_roundtrip(self) -> None:
        queue_config = QueueConfig(
            num_classes=5,
            per_class_capacity=20,
            global_capacity=100,
        )
        queue = ClassConditionalSampleQueue(queue_config)

        # Push 9 images: 3 per class for classes 0, 1, 2
        for class_label in range(3):
            images = torch.randn(3, 4, 8, 8)
            labels = torch.full((3,), class_label, dtype=torch.long)
            queue.push(images, labels)

        original_counts = queue.class_counts()
        original_global = queue.global_count()

        # Save state
        state = queue.state_dict()

        # Restore into a new queue
        restored_queue = ClassConditionalSampleQueue(queue_config)
        restored_queue.load_state_dict(state)

        # Verify counts match
        self.assertEqual(restored_queue.class_counts(), original_counts)
        self.assertEqual(restored_queue.global_count(), original_global)

        # Verify sampling from restored queue works
        class_ids = torch.tensor([0, 1, 2])
        sampled = restored_queue.sample_positive_grouped(
            class_ids, samples_per_group=2, device=torch.device("cpu")
        )
        self.assertEqual(sampled.shape, (3, 2, 4, 8, 8))
        self.assertTrue(torch.isfinite(sampled).all(), "Sampled images contain non-finite values")


if __name__ == "__main__":
    unittest.main()
