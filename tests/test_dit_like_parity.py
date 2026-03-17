"""Parity tests between R3GAN's dit_like.py and the reference drift_models implementation."""

from __future__ import annotations

import os
import sys
import unittest

DRIFT_MODELS_ROOT = "/workspace/drift_models"
if os.path.isdir(DRIFT_MODELS_ROOT) and DRIFT_MODELS_ROOT not in sys.path:
    sys.path.insert(0, DRIFT_MODELS_ROOT)

import torch

from training.models.dit_like import DiTLikeConfig, DiTLikeGenerator

# Reference imports from drift_models
from drifting_models.models.dit_like import (
    DiTLikeConfig as RefConfig,
    DiTLikeGenerator as RefGenerator,
)


def _make_test_config_kwargs() -> dict:
    return dict(
        image_size=8,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        register_tokens=4,
        style_vocab_size=4,
        style_token_count=2,
        alpha_hidden_dim=16,
    )


class TestDiTLikeParity(unittest.TestCase):

    def test_dit_like_forward_shape(self) -> None:
        """Verify output shape is (2, 4, 8, 8) for a small config."""
        config = DiTLikeConfig(**_make_test_config_kwargs())
        model = DiTLikeGenerator(config)
        model.eval()

        batch = 2
        noise = torch.randn(batch, 4, 8, 8)
        class_labels = torch.randint(0, 10, (batch,))
        alpha = torch.rand(batch)

        with torch.no_grad():
            output = model(noise, class_labels, alpha)

        self.assertEqual(output.shape, (2, 4, 8, 8))

    def test_dit_like_alpha_changes_output(self) -> None:
        """Same config, same noise/labels, different alpha values -> different output.

        AdaLN-zero init zeroes the modulation weights, so we perturb them
        to verify the alpha conditioning path actually works.
        """
        config = DiTLikeConfig(**_make_test_config_kwargs())
        model = DiTLikeGenerator(config)
        # Perturb modulation weights so conditioning has effect
        for block in model.blocks:
            with torch.no_grad():
                block.modulation[-1].weight.normal_(std=0.1)
        model.eval()

        batch = 2
        torch.manual_seed(42)
        noise = torch.randn(batch, 4, 8, 8)
        class_labels = torch.randint(0, 10, (batch,))

        alpha_a = torch.tensor([1.0, 1.0])
        alpha_b = torch.tensor([5.0, 5.0])

        with torch.no_grad():
            out_a = model(noise, class_labels, alpha_a)
            out_b = model(noise, class_labels, alpha_b)

        self.assertFalse(
            torch.allclose(out_a, out_b, atol=1e-6),
            "Outputs should differ when alpha values differ",
        )

    def test_dit_like_weight_parity_with_reference(self) -> None:
        """Create R3GAN and reference models with same config, copy weights, verify same output."""
        kwargs = _make_test_config_kwargs()
        our_config = DiTLikeConfig(**kwargs)
        ref_config = RefConfig(**kwargs)

        torch.manual_seed(123)
        our_model = DiTLikeGenerator(our_config)
        torch.manual_seed(123)
        ref_model = RefGenerator(ref_config)

        # Copy weights from reference to our model to ensure exact match
        our_model.load_state_dict(ref_model.state_dict())
        our_model.eval()
        ref_model.eval()

        batch = 2
        torch.manual_seed(999)
        noise = torch.randn(batch, 4, 8, 8)
        class_labels = torch.randint(0, 10, (batch,))
        alpha = torch.rand(batch)

        with torch.no_grad():
            our_output = our_model(noise, class_labels, alpha)
            ref_output = ref_model(noise, class_labels, alpha)

        self.assertTrue(
            torch.allclose(our_output, ref_output, atol=1e-5),
            f"Max difference: {(our_output - ref_output).abs().max().item():.2e}",
        )


if __name__ == "__main__":
    unittest.main()
