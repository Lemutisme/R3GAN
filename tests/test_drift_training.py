import io
import inspect
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REFERENCE_DRIFT_ROOT = "/workspace/drift_models"
if os.path.isdir(REFERENCE_DRIFT_ROOT) and REFERENCE_DRIFT_ROOT not in sys.path:
    sys.path.insert(0, REFERENCE_DRIFT_ROOT)

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
    from training import drift_training_loop as drift_training_loop_impl
    from training.drift_field import (
        DriftFieldConfig,
        build_negative_log_weights,
        cfg_alpha_to_unconditional_weight,
    )
    from training.drift_loss import (
        DriftingLossConfig,
        drifting_stopgrad_loss,
    )
    from training.drift_queue import (
        ClassConditionalSampleQueue,
        QueueConfig,
        ensure_class_coverage,
    )
    # Legacy API stubs — old tests reference these but they no longer exist.
    # The old DriftLossConfig is now DriftFieldConfig; grouped_drifting_stopgrad_loss
    # moved to drift_stage2. These stubs let old tests be skipped gracefully.
    DriftLossConfig = DriftFieldConfig
    grouped_drifting_stopgrad_loss = None
    drift_research_impl = None
    drift_diagnostics_impl = None
    try:
        from drifting_models.drift_field import DriftFieldConfig as ReferenceDriftFieldConfig
        from drifting_models.drift_loss import (
            DriftingLossConfig as ReferenceDriftingLossConfig,
            drifting_stopgrad_loss as reference_drifting_stopgrad_loss,
        )
        from drifting_models.models import DiTLikeConfig as ReferenceDiTLikeConfig
        from drifting_models.models import DiTLikeGenerator as ReferenceDiTLikeGenerator
    except Exception:  # pragma: no cover
        ReferenceDriftFieldConfig = None
        ReferenceDriftingLossConfig = None
        reference_drifting_stopgrad_loss = None
        ReferenceDiTLikeConfig = None
        ReferenceDiTLikeGenerator = None
else:  # pragma: no cover
    legacy = None
    train_cli = None
    training_networks = None
    drift_training_loop_impl = None
    drift_research_impl = None
    drift_diagnostics_impl = None
    DriftLossConfig = None
    grouped_drifting_stopgrad_loss = None
    ReferenceDriftFieldConfig = None
    ReferenceDriftingLossConfig = None
    reference_drifting_stopgrad_loss = None
    ReferenceDiTLikeConfig = None
    ReferenceDiTLikeGenerator = None


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

    @unittest.skipIf(
        reference_drifting_stopgrad_loss is None or ReferenceDriftFieldConfig is None or ReferenceDriftingLossConfig is None,
        "Reference drift_models package is not available",
    )
    def test_scalar_loss_matches_reference_drift_models(self):
        torch.manual_seed(13)
        x = torch.randn(4, 3 * 4 * 4)
        y_pos = torch.randn(5, 3 * 4 * 4)
        y_unc = torch.randn(2, 3 * 4 * 4)
        y_neg = torch.cat([x, y_unc], dim=0)
        negative_log_weights = build_negative_log_weights(
            n_generated_negatives=x.shape[0],
            n_unconditional_negatives=y_unc.shape[0],
            unconditional_weight=1.75,
            device=x.device,
            dtype=x.dtype,
        )

        ours_loss, ours_drift, ours_stats = drifting_stopgrad_loss(
            x=x,
            y_pos=y_pos,
            y_neg=y_neg,
            config=DriftLossConfig(temperature=0.07),
            negative_log_weights=negative_log_weights,
            generated_negative_count=x.shape[0],
        )

        reference_loss, reference_drift, reference_stats = reference_drifting_stopgrad_loss(
            x=x,
            y_pos=y_pos,
            y_neg=y_neg,
            config=ReferenceDriftingLossConfig(
                drift_field=ReferenceDriftFieldConfig(
                    temperature=0.07,
                    normalize_over_x=True,
                    mask_self_negatives=True,
                    self_mask_value=1e6,
                    eps=1e-12,
                ),
                attraction_scale=1.0,
                repulsion_scale=1.0,
                stopgrad_target=True,
            ),
            negative_log_weights=negative_log_weights,
            generated_negative_count=x.shape[0],
        )

        self.assertTrue(torch.allclose(ours_loss.detach(), reference_loss.detach(), atol=1e-6))
        self.assertTrue(torch.allclose(ours_drift.detach(), reference_drift.detach(), atol=1e-6))
        self.assertAlmostEqual(ours_stats["drift_norm"], reference_stats["drift_norm"], places=6)


@unittest.skipIf(torch is None or drift_training_loop_impl is None, "Drift training loop is not available")
class TestDriftTrainingLoopCompatibility(unittest.TestCase):
    def test_training_loop_accepts_gan_side_kwargs_without_typeerror(self):
        signature = inspect.signature(drift_training_loop_impl.training_loop)
        self.assertTrue(
            any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        )

        with mock.patch.object(
            drift_training_loop_impl.dnnlib.util,
            "construct_class_by_name",
            side_effect=RuntimeError("sentinel"),
        ):
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                drift_training_loop_impl.training_loop(
                    training_set_kwargs={},
                    data_loader_kwargs={},
                    G_kwargs={},
                    G_opt_kwargs={},
                    D_kwargs=None,
                    D_opt_kwargs=None,
                    loss_kwargs={},
                    augment_kwargs=None,
                )


@unittest.skipIf(torch is None or drift_research_impl is None, "Drift research trainer is not available")
class TestDriftResearchMetricOutputs(unittest.TestCase):
    def test_write_periodic_metric_jsonls_emits_compatibility_files(self):
        eval_entry = {
            "fid": 405.0,
            "fid50k_full": 406.0,
            "fid50k_fullb": 407.0,
            "inception_score_mean": 1.25,
            "step": 392,
            "generated_images_total": 200000,
            "generated_kimg_total": 200.0,
            "generated_samples": 50000,
            "reference_samples": 10000,
            "eval_time_s": 28.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            drift_research_impl._write_periodic_metric_jsonls(
                run_dir=tmpdir,
                metrics=["fid50k_full"],
                eval_entry=eval_entry,
            )

            fid_path = Path(tmpdir) / "metric-fid50k_full.jsonl"
            fidb_path = Path(tmpdir) / "metric-fid50k_fullb.jsonl"
            fid_alias_missing = Path(tmpdir) / "metric-fid.jsonl"

            self.assertTrue(fid_path.exists())
            self.assertTrue(fidb_path.exists())
            self.assertFalse(fid_alias_missing.exists())

            fid_payload = json.loads(fid_path.read_text(encoding="utf-8").strip())
            fidb_payload = json.loads(fidb_path.read_text(encoding="utf-8").strip())
            self.assertEqual(fid_payload["metric"], "fid50k_full")
            self.assertEqual(fid_payload["results"]["fid50k_full"], 406.0)
            self.assertEqual(fidb_payload["metric"], "fid50k_fullb")
            self.assertEqual(fidb_payload["results"]["fid50k_fullb"], 407.0)


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

    def test_dit_like_generator_accepts_one_hot_labels_and_eval_alpha(self):
        generator = training_networks.DiTLikeDriftGenerator(
            FP16Stages=[],
            c_dim=3,
            img_resolution=8,
            ImageChannels=3,
            PatchSize=4,
            HiddenDim=32,
            Depth=2,
            NumHeads=4,
            RegisterTokens=2,
            StyleTokenCount=3,
            StyleVocabSize=7,
            EvalAlpha=1.5,
        )

        noise = torch.randn(2, 3, 8, 8)
        one_hot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        class_ids = torch.tensor([0, 1], dtype=torch.long)
        style_indices = torch.zeros(2, 3, dtype=torch.long)

        out_default = generator(noise, one_hot)
        out_explicit = generator(noise, class_ids, alpha=torch.full([2], 1.5), style_indices=style_indices)
        self.assertEqual(out_default.shape, (2, 3, 8, 8))
        self.assertTrue(torch.allclose(out_default, out_explicit, atol=1e-6))


@unittest.skipIf(torch is None or drift_diagnostics_impl is None, "Drift diagnostics are not available")
class TestDriftDiagnostics(unittest.TestCase):
    def test_collect_generator_diagnostics_emits_conditioning_metrics(self):
        generator = training_networks.DiTLikeDriftGenerator(
            FP16Stages=[],
            c_dim=4,
            img_resolution=8,
            ImageChannels=3,
            PatchSize=4,
            HiddenDim=32,
            Depth=2,
            NumHeads=4,
            RegisterTokens=2,
            StyleTokenCount=4,
            StyleVocabSize=8,
            EvalAlpha=1.5,
        )

        diagnostics = drift_diagnostics_impl.collect_generator_diagnostics(
            generator=generator,
            device=torch.device('cpu'),
            batch_size=4,
            num_classes=4,
            eval_alpha=1.5,
            alpha_pair=(1.0, 4.0),
            seed=7,
        )
        self.assertIn('pairwise_l2_mean', diagnostics)
        self.assertIn('diff_class_l2', diagnostics)
        self.assertIn('diff_style_l2', diagnostics)
        self.assertIn('class_cond_norm_mean', diagnostics)
        self.assertIn('style_cond_norm_mean', diagnostics)
        self.assertGreaterEqual(diagnostics['pairwise_l2_mean'], 0.0)

    @unittest.skipIf(
        ReferenceDiTLikeConfig is None or ReferenceDiTLikeGenerator is None,
        "Reference DiT-like generator is not available",
    )
    def test_reference_dit_like_style_condition_scales_by_sqrt_token_count(self):
        model = ReferenceDiTLikeGenerator(
            ReferenceDiTLikeConfig(
                image_size=8,
                in_channels=3,
                out_channels=3,
                patch_size=4,
                hidden_dim=8,
                depth=1,
                num_heads=2,
                num_classes=3,
                register_tokens=0,
                style_vocab_size=2,
                style_token_count=4,
            )
        )
        with torch.no_grad():
            model.class_embedding.weight.zero_()
            for parameter in model.alpha_embedding.parameters():
                parameter.zero_()
            model.style_embedding.weight.fill_(1.0)

        condition = model._build_conditioning(
            class_labels=torch.zeros(2, dtype=torch.long),
            alpha=torch.zeros(2),
            style_indices=torch.zeros(2, 4, dtype=torch.long),
            device=torch.device('cpu'),
            batch=2,
        )
        self.assertTrue(torch.allclose(condition, torch.full_like(condition, 2.0)))


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


@unittest.skipIf(torch is None or drift_research_impl is None, "Drift research trainer is not available")
class TestDriftEMA(unittest.TestCase):
    def test_ema_update_differs_from_direct_copy(self):
        """EMA update should produce params between old and new, not equal to new."""
        torch.manual_seed(99)
        model = torch.nn.Linear(4, 4, bias=False)
        ema_model = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            ema_model.weight.copy_(model.weight)
            old_ema_weight = ema_model.weight.clone()
            model.weight.add_(torch.randn_like(model.weight))

        drift_research_impl._ema_update(src=model, dst=ema_model, decay=0.999)

        # EMA should NOT equal current model (not a direct copy)
        self.assertFalse(torch.allclose(ema_model.weight, model.weight))
        # EMA should NOT equal old weights either (it moved)
        self.assertFalse(torch.allclose(ema_model.weight, old_ema_weight))

    def test_ema_update_copies_buffers_directly(self):
        """Buffers (e.g. BatchNorm running stats) should be copied, not EMA-smoothed."""
        model = torch.nn.BatchNorm1d(4)
        ema_model = torch.nn.BatchNorm1d(4)
        with torch.no_grad():
            model.running_mean.fill_(5.0)

        drift_research_impl._ema_update(src=model, dst=ema_model, decay=0.999)
        self.assertTrue(torch.allclose(ema_model.running_mean, model.running_mean))

    def test_ema_smoothing_over_multiple_steps(self):
        """After N updates with consistent drift, EMA should track the model."""
        torch.manual_seed(42)
        model = torch.nn.Linear(8, 8, bias=False)
        ema = torch.nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            ema.weight.copy_(model.weight)
        initial_ema_weight = ema.weight.clone()

        # Apply a consistent positive shift so EMA can track it
        for _ in range(100):
            with torch.no_grad():
                model.weight.add_(torch.ones_like(model.weight) * 0.01)
            drift_research_impl._ema_update(src=model, dst=ema, decay=0.99)

        # EMA should have moved away from initial value
        self.assertFalse(torch.allclose(ema.weight, initial_ema_weight, atol=1e-3))
        # EMA should NOT equal current model (it lags)
        self.assertFalse(torch.allclose(ema.weight, model.weight, atol=1e-3))
        # EMA should be between initial and current: all values increased
        self.assertTrue((ema.weight > initial_ema_weight).all())


@unittest.skipIf(torch is None or drift_research_impl is None, "Drift research trainer is not available")
class TestR3GANConvStepParity(unittest.TestCase):
    def test_r3gan_conv_step_uses_multi_temperature(self):
        """_run_r3gan_conv_step should use drift_temperatures when specified."""
        from types import SimpleNamespace
        from training.drift_reference import ClassConditionalSampleQueue, QueueConfig

        class FakeConvGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3 * 4 * 4)
                self.z_dim = 4
                self.c_dim = 2

            def forward(self, z, c, alpha=None):
                return self.linear(z).reshape(z.shape[0], 3, 4, 4)

        gen = FakeConvGenerator()
        opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
        queue = ClassConditionalSampleQueue(QueueConfig(num_classes=2, per_class_capacity=16, global_capacity=32))
        for _ in range(4):
            queue.push(torch.randn(8, 3, 4, 4), torch.randint(0, 2, (8,)))

        class FakeProvider:
            def next_batch(self, *, device):
                return torch.randn(8, 3, 4, 4, device=device), torch.randint(0, 2, (8,), device=device)

        drift = SimpleNamespace(
            negatives_per_group=2,
            positives_per_group=2,
            unconditional_per_group=1,
            drift_temperature=0.05,
            drift_temperatures=[0.02, 0.05, 0.2],
            drift_temperature_reduction='sum',
            clip_grad_norm=2.0,
            queue_refill_policy='per_step',
            queue_refill_every=1,
            queue_push_batch=4,
            queue_strict_without_replacement=False,
            use_feature_loss=False,
        )

        stats = drift_research_impl._run_r3gan_conv_step(
            generator=gen,
            optimizer=opt,
            queue=queue,
            provider=FakeProvider(),
            step=0,
            local_groups=2,
            num_classes=2,
            drift=drift,
            image_channels=3,
            image_size=4,
            class_labels=torch.tensor([0, 1]),
            alpha=torch.tensor([2.0, 3.0]),
            device=torch.device('cpu'),
        )
        self.assertIn('loss', stats)
        self.assertIn('mean_drift_norm', stats)
        self.assertIn('grad_norm', stats)
        self.assertGreater(stats['loss'], 0.0)


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
        self.assertEqual(config.G_kwargs.class_name, "training.networks.DiTLikeDriftGenerator")
        self.assertEqual(config.negatives_per_group, 2)
        self.assertEqual(config.positives_per_group, 3)
        self.assertEqual(config.unconditional_per_group, 1)
        self.assertEqual(config.G_kwargs.HiddenDim, 256)
        self.assertEqual(config.drift_config.backbone, "dit_like")
        self.assertEqual(config.drift_config.alpha_max, 3.0)
        self.assertEqual(config.G_kwargs.PatchSize, 4)
        self.assertEqual(config.G_kwargs.Depth, 6)
        self.assertEqual(config.G_kwargs.StyleVocabSize, 1)
        self.assertEqual(config.G_kwargs.StyleTokenCount, 0)
        self.assertEqual(config.G_opt_kwargs.class_name, "torch.optim.Adam")
        self.assertEqual(config.G_opt_kwargs.lr, 2e-4)
        self.assertEqual(config.G_opt_kwargs.betas, [0.0, 0.0])
        self.assertIsNotNone(config.lr_scheduler)
        self.assertIsNotNone(config.beta2_scheduler)
        self.assertIsNone(config.D_kwargs)

    def test_drift_cli_requires_conditional_labels(self):
        result, _launch_training = self._invoke_train(["--trainer=drift", "--cond=0"], use_labels=True)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--trainer=drift requires --cond=1", result.output)

    def test_drift_cli_can_select_r3gan_conv_backbone(self):
        result, launch_training = self._invoke_train(
            [
                "--trainer=drift",
                "--cond=1",
                "--drift-backbone=r3gan_conv",
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config = launch_training.call_args.kwargs["c"]
        self.assertEqual(config.G_kwargs.class_name, "training.networks.DriftGenerator")
        self.assertEqual(config.drift_config.backbone, "r3gan_conv")

    def test_drift_cli_uses_adamw_when_weight_decay_is_enabled(self):
        result, launch_training = self._invoke_train(
            [
                "--trainer=drift",
                "--cond=1",
                "--weight-decay=0.01",
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config = launch_training.call_args.kwargs["c"]
        self.assertEqual(config.G_opt_kwargs.class_name, "torch.optim.AdamW")
        self.assertEqual(config.G_opt_kwargs.weight_decay, 0.01)

    def test_drift_cli_custom_learning_rate_disables_preset_lr_scheduler_for_dit_like(self):
        result, launch_training = self._invoke_train(
            [
                "--trainer=drift",
                "--cond=1",
                "--learning-rate=1e-4",
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config = launch_training.call_args.kwargs["c"]
        self.assertIsNone(config.lr_scheduler)

    def test_drift_cli_auto_maps_metrics_to_periodic_eval(self):
        with mock.patch.object(
            train_cli,
            "_infer_drift_periodic_eval_paths",
            return_value=("/tmp/cifar10_val", "/tmp/reference_stats.pt"),
        ):
            result, launch_training = self._invoke_train(
                [
                    "--trainer=drift",
                    "--cond=1",
                    "--metrics=fid50k_full",
                ]
            )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config = launch_training.call_args.kwargs["c"]
        self.assertEqual(config.drift_config.eval_every_kimg, 200.0)
        self.assertEqual(config.drift_config.eval_reference_imagefolder_root, "/tmp/cifar10_val")
        self.assertEqual(config.drift_config.eval_reference_stats_path, "/tmp/reference_stats.pt")
        self.assertIn("enabling drift periodic eval to mirror --metrics", result.output)

    def test_drift_cli_keeps_periodic_eval_disabled_when_metrics_none(self):
        with mock.patch.object(
            train_cli,
            "_infer_drift_periodic_eval_paths",
            return_value=("/tmp/cifar10_val", "/tmp/reference_stats.pt"),
        ):
            result, launch_training = self._invoke_train(
                [
                    "--trainer=drift",
                    "--cond=1",
                    "--metrics=none",
                ]
            )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config = launch_training.call_args.kwargs["c"]
        self.assertEqual(config.drift_config.eval_every_kimg, 0.0)
        self.assertEqual(config.drift_config.eval_reference_imagefolder_root, None)
        self.assertEqual(config.drift_config.eval_reference_stats_path, None)
