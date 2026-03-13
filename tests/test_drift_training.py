import io
import pickle
import unittest
from unittest import mock

try:
    import dnnlib
    import torch
    from click.testing import CliRunner
except Exception:  # pragma: no cover
    dnnlib = None
    torch = None
    CliRunner = None

if torch is not None:
    import legacy
    import train as train_cli
    from training import networks as training_networks
    from training.drift_loss import (
        DriftLossConfig,
        build_negative_log_weights,
        cfg_alpha_to_unconditional_weight,
        drifting_stopgrad_loss,
        grouped_drifting_stopgrad_loss,
    )
    from training.drift_queue import (
        ClassConditionalSampleQueue,
        QueueConfig,
        ensure_class_coverage,
    )
else:  # pragma: no cover
    legacy = None
    train_cli = None
    training_networks = None


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDriftLoss(unittest.TestCase):
    def test_cfg_alpha_to_unconditional_weight(self):
        weight = cfg_alpha_to_unconditional_weight(
            alpha=3.0,
            n_generated_negatives=4,
            n_unconditional_negatives=2,
        )
        self.assertAlmostEqual(weight, 3.0)

    def test_grouped_drift_matches_scalar_reference(self):
        torch.manual_seed(7)
        config = DriftLossConfig(temperature=0.2)
        x_grouped = torch.randn(3, 2, 3, 4, 4)
        y_pos_grouped = torch.randn(3, 4, 3, 4, 4)
        unconditional_grouped = torch.randn(3, 1, 3, 4, 4)
        unconditional_weight_grouped = torch.tensor([0.5, 1.0, 2.0])

        grouped_loss, grouped_stats = grouped_drifting_stopgrad_loss(
            x_grouped=x_grouped,
            y_pos_grouped=y_pos_grouped,
            unconditional_grouped=unconditional_grouped,
            unconditional_weight_grouped=unconditional_weight_grouped,
            config=config,
        )

        scalar_losses = []
        scalar_norms = []
        for group_idx in range(x_grouped.shape[0]):
            x = x_grouped[group_idx].reshape(x_grouped.shape[1], -1)
            y_pos = y_pos_grouped[group_idx].reshape(y_pos_grouped.shape[1], -1)
            y_unc = unconditional_grouped[group_idx].reshape(unconditional_grouped.shape[1], -1)
            y_neg = torch.cat([x, y_unc], dim=0)
            negative_log_weights = build_negative_log_weights(
                n_generated_negatives=x_grouped.shape[1],
                n_unconditional_negatives=unconditional_grouped.shape[1],
                unconditional_weight=float(unconditional_weight_grouped[group_idx].item()),
                device=x.device,
                dtype=x.dtype,
            )
            scalar_loss, _drift, scalar_stats = drifting_stopgrad_loss(
                x=x,
                y_pos=y_pos,
                y_neg=y_neg,
                config=config,
                negative_log_weights=negative_log_weights,
                generated_negative_count=x_grouped.shape[1],
            )
            scalar_losses.append(scalar_loss.detach())
            scalar_norms.append(scalar_stats["drift_norm"])

        reference_loss = torch.stack(scalar_losses).mean()
        reference_norm = sum(scalar_norms) / len(scalar_norms)
        self.assertTrue(torch.allclose(grouped_loss.detach(), reference_loss, atol=1e-6))
        self.assertAlmostEqual(grouped_stats["mean_drift_norm"], reference_norm, places=6)


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDriftQueue(unittest.TestCase):
    def test_positive_sampling_preserves_class_identity(self):
        queue = ClassConditionalSampleQueue(
            QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=16)
        )
        images = torch.stack(
            [
                torch.full([3, 4, 4], 0.0),
                torch.full([3, 4, 4], 0.0),
                torch.full([3, 4, 4], 1.0),
                torch.full([3, 4, 4], 1.0),
                torch.full([3, 4, 4], 2.0),
                torch.full([3, 4, 4], 2.0),
            ],
            dim=0,
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        queue.push(images, labels)

        sampled = queue.sample_positive_grouped(
            class_ids=torch.tensor([0, 2]),
            samples_per_group=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(sampled.shape, (2, 2, 3, 4, 4))
        self.assertTrue(torch.allclose(sampled[0], torch.zeros_like(sampled[0])))
        self.assertTrue(torch.allclose(sampled[1], torch.full_like(sampled[1], 2.0)))

    def test_ensure_class_coverage_backfills_missing_labels(self):
        queue = ClassConditionalSampleQueue(
            QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=16)
        )
        refill_batches = iter(
            [
                (
                    torch.stack([torch.ones(3, 4, 4), torch.ones(3, 4, 4)], dim=0),
                    torch.tensor([0, 0]),
                ),
                (
                    torch.stack([torch.full([3, 4, 4], 2.0), torch.full([3, 4, 4], 2.0)], dim=0),
                    torch.tensor([2, 2]),
                ),
            ]
        )

        attempts = ensure_class_coverage(
            queue,
            class_ids=torch.tensor([0, 2]),
            refill_fn=lambda: next(refill_batches),
            required_count=1,
            max_attempts=4,
        )
        self.assertEqual(attempts, 2)
        self.assertGreaterEqual(queue.class_count(0), 1)
        self.assertGreaterEqual(queue.class_count(2), 1)


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDriftGenerator(unittest.TestCase):
    def test_alpha_default_matches_eval_alpha(self):
        class FakeInnerGenerator(torch.nn.Module):
            def __init__(self):
                super(FakeInnerGenerator, self).__init__()
                self.last_condition = None

            def forward(self, z, cond):
                self.last_condition = cond.detach().clone()
                value = cond[:, -1].view(-1, 1, 1, 1)
                return value.expand(z.shape[0], 3, 4, 4)

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


@unittest.skipIf(torch is None, "PyTorch is not available in this environment")
class TestDriftLegacyCompatibility(unittest.TestCase):
    def test_legacy_loader_accepts_drift_snapshot_without_discriminator(self):
        payload = {
            "G": torch.nn.Linear(2, 2),
            "D": None,
            "G_ema": torch.nn.Linear(2, 2),
            "training_set_kwargs": None,
            "augment_pipe": None,
            "trainer": "drift",
        }
        buffer = io.BytesIO()
        pickle.dump(payload, buffer)
        buffer.seek(0)

        loaded = legacy.load_network_pkl(buffer)
        self.assertIsNone(loaded["D"])
        self.assertIsInstance(loaded["G_ema"], torch.nn.Module)


@unittest.skipIf(
    torch is None or CliRunner is None or dnnlib is None,
    "Test dependencies are not available in this environment",
)
class TestDriftCliMappings(unittest.TestCase):
    def _invoke_train(self, extra_args, use_labels=True):
        dataset_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path="dummy.zip",
            use_labels=use_labels,
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
                    "--batch=4",
                    "--preset=CIFAR10",
                    *extra_args,
                ],
            )
        return result, launch_training

    def test_drift_cli_maps_to_drift_generator_config(self):
        result, launch_training = self._invoke_train(
            [
                "--trainer=drift",
                "--cond=1",
                "--negatives-per-group=2",
                "--positives-per-group=3",
                "--unconditional-per-group=1",
                "--alpha-min=1.0",
                "--alpha-max=3.0",
                "--queue-push-batch=4",
                "--queue-warmup-batches=2",
                "--aug=0",
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        kwargs = launch_training.call_args.kwargs
        config = kwargs["c"]
        self.assertEqual(config.trainer, "drift")
        self.assertEqual(config.G_kwargs.class_name, "training.networks.DriftGenerator")
        self.assertEqual(config.negatives_per_group, 2)
        self.assertEqual(config.positives_per_group, 3)
        self.assertEqual(config.unconditional_per_group, 1)
        self.assertEqual(config.G_kwargs.AlphaMax, 3.0)
        self.assertIsNone(config.D_kwargs)

    def test_drift_cli_requires_conditional_labels(self):
        result, _launch_training = self._invoke_train(["--trainer=drift", "--cond=0"], use_labels=True)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--trainer=drift requires --cond=1", result.output)

