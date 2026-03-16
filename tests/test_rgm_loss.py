import unittest

import torch

from training.drift_loss import DriftLossConfig, grouped_drifting_stopgrad_loss
from training.rgm_loss import RGMLossConfig, fixed_point_loss, grouped_rank_drifting_stopgrad_loss, rank_order_loss


class TestRgmLoss(unittest.TestCase):
    def test_single_rank_matches_grouped_drift(self):
        torch.manual_seed(7)
        x_grouped = torch.randn(2, 1, 3, 1, 4, 4)
        y_pos_grouped = torch.randn(2, 5, 1, 4, 4)
        unconditional_grouped = torch.randn(2, 2, 1, 4, 4)
        unconditional_weights = torch.tensor([1.0, 1.0])

        rgm_loss, rgm_stats = grouped_rank_drifting_stopgrad_loss(
            x_grouped=x_grouped,
            y_pos_grouped=y_pos_grouped,
            unconditional_grouped=unconditional_grouped,
            unconditional_weight_grouped=unconditional_weights,
            rank_levels=[1.0],
            config=RGMLossConfig(temperature=0.05),
        )
        drift_loss, drift_stats = grouped_drifting_stopgrad_loss(
            x_grouped=x_grouped[:, 0],
            y_pos_grouped=y_pos_grouped,
            unconditional_grouped=unconditional_grouped,
            unconditional_weight_grouped=unconditional_weights,
            config=DriftLossConfig(temperature=0.05),
        )
        self.assertTrue(torch.allclose(rgm_loss, drift_loss))
        self.assertAlmostEqual(rgm_stats["mean_drift_norm"], drift_stats["mean_drift_norm"], places=6)

    def test_order_loss_zero_when_monotone(self):
        drift_norms = torch.tensor([[3.0, 2.0, 1.0]])
        self.assertEqual(float(rank_order_loss(drift_norms).item()), 0.0)

    def test_order_loss_positive_when_reversed(self):
        drift_norms = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertGreater(float(rank_order_loss(drift_norms).item()), 0.0)

    def test_fixed_point_loss_zero_for_zero_drift(self):
        x_best = torch.randn(4, 8)
        drift_best = torch.zeros_like(x_best)
        self.assertEqual(float(fixed_point_loss(x_best, drift_best).item()), 0.0)

    def test_multi_rank_loss_is_finite(self):
        torch.manual_seed(11)
        loss, stats = grouped_rank_drifting_stopgrad_loss(
            x_grouped=torch.randn(2, 3, 4, 1, 2, 2),
            y_pos_grouped=torch.randn(2, 5, 1, 2, 2),
            unconditional_grouped=torch.randn(2, 2, 1, 2, 2),
            unconditional_weight_grouped=torch.ones(2),
            rank_levels=[1.0, 0.5, 0.0],
            config=RGMLossConfig(temperature=0.1),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("transport_loss", stats)
        self.assertIn("order_loss", stats)
        self.assertIn("eq_loss", stats)
