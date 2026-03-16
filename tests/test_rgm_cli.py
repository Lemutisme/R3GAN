import sys
import types
import unittest
from unittest import mock

import dnnlib
from click.testing import CliRunner

if "pkg_resources" not in sys.modules:
    pkg_resources_stub = types.ModuleType("pkg_resources")
    pkg_resources_stub.parse_version = lambda value: value
    sys.modules["pkg_resources"] = pkg_resources_stub

import train as train_cli


class TestRgmCli(unittest.TestCase):
    def _invoke_train(self, extra_args, *, batch="6", use_labels=True):
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
        fake_metric_main = mock.Mock()
        fake_metric_main.is_valid_metric.return_value = True
        fake_metric_main.list_valid_metrics.return_value = ["fid50k_full"]
        with mock.patch.object(train_cli, "metric_main", fake_metric_main), mock.patch.object(
            train_cli, "_load_metric_main", return_value=fake_metric_main
        ), mock.patch.object(
            train_cli, "init_dataset_kwargs", return_value=(dataset_kwargs, "dummy")
        ), mock.patch.object(train_cli, "launch_training") as launch_training:
            result = runner.invoke(
                train_cli.main,
                [
                    "--outdir=/tmp/out",
                    "--data=dummy.zip",
                    "--gpus=1",
                    f"--batch={batch}",
                    "--preset=CIFAR10",
                    *extra_args,
                ],
            )
        return result, launch_training

    def test_rgm_cli_maps_to_rank_generator_config(self):
        result, launch_training = self._invoke_train(
            [
                "--trainer=rgm",
                "--cond=1",
                "--rank-levels=1.0,0.5,0.0",
                "--negatives-per-group=2",
                "--positives-per-group=3",
                "--unconditional-per-group=1",
                "--learning-rate=1e-4",
                "--aug=0",
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        config = launch_training.call_args.kwargs["c"]
        self.assertEqual(config.trainer, "rgm")
        self.assertEqual(config.G_kwargs.class_name, "training.networks.RankConditionedGenerator")
        self.assertEqual(config.G_kwargs.EvalRank, 0.0)
        self.assertEqual(config.rank_levels, [1.0, 0.5, 0.0])
        self.assertEqual(config.loss_kwargs.lambda_transport, 1.0)
        self.assertEqual(config.D_kwargs, None)

    def test_rgm_requires_conditional_labels(self):
        result, _launch_training = self._invoke_train(["--trainer=rgm", "--cond=0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--trainer=rgm requires --cond=1", result.output)

    def test_rgm_rank_levels_must_descend(self):
        result, _launch_training = self._invoke_train(
            ["--trainer=rgm", "--cond=1", "--rank-levels=0.0,0.5,1.0"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--rank-levels must be strictly descending", result.output)

    def test_rgm_batch_must_match_rank_expansion(self):
        result, _launch_training = self._invoke_train(
            ["--trainer=rgm", "--cond=1", "--rank-levels=1.0,0.5,0.0", "--negatives-per-group=2"],
            batch="4",
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--batch / --gpus must be divisible", result.output)
