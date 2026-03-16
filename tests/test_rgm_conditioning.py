import sys
import types
import unittest
from unittest import mock

import torch

if "pkg_resources" not in sys.modules:
    pkg_resources_stub = types.ModuleType("pkg_resources")
    pkg_resources_stub.parse_version = lambda value: value
    sys.modules["pkg_resources"] = pkg_resources_stub

from training import networks as training_networks
from training.rgm_conditioning import RankConditionSchema, normalize_scalar_condition, schema_extra_dim


class TestRgmConditioning(unittest.TestCase):
    def test_schema_extra_dim(self):
        self.assertEqual(schema_extra_dim(RankConditionSchema(use_alpha=True, use_rank=False, use_rank_pair=True)), 3)

    def test_normalize_scalar_condition(self):
        values = torch.tensor([[0.0], [0.5], [1.0]])
        normalized = normalize_scalar_condition(values, 0.0, 1.0)
        self.assertTrue(torch.allclose(normalized, values))

    def test_rank_conditioned_generator_uses_eval_rank_by_default(self):
        class FakeInnerGenerator(torch.nn.Module):
            def __init__(self):
                super(FakeInnerGenerator, self).__init__()
                self.last_condition = None

            def forward(self, z, cond):
                self.last_condition = cond.detach().clone()
                return cond[:, -1].view(-1, 1, 1, 1).expand(z.shape[0], 3, 4, 4)

        inner = FakeInnerGenerator()
        with mock.patch.object(training_networks.R3GAN.Networks, "Generator", return_value=inner):
            generator = training_networks.RankConditionedGenerator(
                NoiseDimension=4,
                WidthPerStage=[8],
                CardinalityPerStage=[1],
                BlocksPerStage=[1],
                ExpansionFactor=2,
                ConditionEmbeddingDimension=4,
                FP16Stages=[],
                c_dim=3,
                img_resolution=4,
                UseRank=True,
                RankMin=0.0,
                RankMax=1.0,
                EvalRank=0.25,
            )

        z = torch.randn(2, 4)
        c = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out_default = generator(z, c)
        out_explicit = generator(z, c, rank=torch.full([2], 0.25))
        self.assertTrue(torch.allclose(out_default, out_explicit))
        self.assertTrue(torch.allclose(inner.last_condition[:, -1], torch.full([2], 0.25)))

    def test_drift_generator_still_uses_eval_alpha(self):
        class FakeInnerGenerator(torch.nn.Module):
            def __init__(self):
                super(FakeInnerGenerator, self).__init__()
                self.last_condition = None

            def forward(self, z, cond):
                self.last_condition = cond.detach().clone()
                return cond[:, -1].view(-1, 1, 1, 1).expand(z.shape[0], 3, 4, 4)

        inner = FakeInnerGenerator()
        with mock.patch.object(training_networks.R3GAN.Networks, "Generator", return_value=inner):
            generator = training_networks.DriftGenerator(
                NoiseDimension=4,
                WidthPerStage=[8],
                CardinalityPerStage=[1],
                BlocksPerStage=[1],
                ExpansionFactor=2,
                ConditionEmbeddingDimension=4,
                FP16Stages=[],
                c_dim=3,
                img_resolution=4,
                AlphaMin=1.0,
                AlphaMax=4.0,
                EvalAlpha=2.5,
            )

        z = torch.randn(2, 4)
        c = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out_default = generator(z, c)
        out_explicit = generator(z, c, alpha=torch.full([2], 2.5))
        self.assertTrue(torch.allclose(out_default, out_explicit))
        self.assertTrue(torch.allclose(inner.last_condition[:, -1], torch.full([2], 0.5)))
