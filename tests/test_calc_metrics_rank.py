import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from click.testing import CliRunner

import calc_metrics


class FakeRgmGenerator(torch.nn.Module):
    def __init__(self):
        super(FakeRgmGenerator, self).__init__()
        self.z_dim = 4
        self.c_dim = 2
        self.img_resolution = 4

    def forward(self, z, c, rank=None):
        if rank is None:
            rank = torch.zeros([z.shape[0]], device=z.device)
        return rank.view(-1, 1, 1, 1).expand(z.shape[0], 3, 4, 4)


class TestCalcMetricsRank(unittest.TestCase):
    def _run_calc_metrics(self, extra_args):
        runner = CliRunner()
        real_torch_device = torch.device
        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            fake_metric_main = mock.Mock()
            fake_metric_main.is_valid_metric.return_value = True
            fake_metric_main.list_valid_metrics.return_value = ["fid50k_full"]
            fake_metric_main.calc_metric.side_effect = self._fake_calc_metric
            fake_metric_main.report_metric.return_value = None
            fake_metric_utils = mock.Mock()
            fake_metric_utils.ProgressMonitor.return_value = object()
            with mock.patch.object(calc_metrics, "metric_main", fake_metric_main), mock.patch.object(
                calc_metrics,
                "metric_utils",
                fake_metric_utils,
            ), mock.patch.object(
                calc_metrics,
                "_load_metric_modules",
                return_value=(fake_metric_main, fake_metric_utils),
            ), mock.patch.object(
                calc_metrics,
                "_load_conv2d_gradfix",
                return_value=SimpleNamespace(enabled=False),
            ), mock.patch.object(
                calc_metrics.dnnlib.util,
                "open_url",
                return_value=io.BytesIO(),
            ), mock.patch.object(
                calc_metrics.legacy,
                "load_network_pkl",
                return_value={
                    "G_ema": FakeRgmGenerator(),
                    "trainer": "rgm",
                    "rank_levels": [1.0, 0.5, 0.0],
                    "training_set_kwargs": {"class_name": "training.dataset.ImageFolderDataset", "path": "dummy.zip"},
                },
            ), mock.patch.object(
                calc_metrics.torch,
                "device",
                side_effect=lambda *args, **kwargs: real_torch_device("cpu"),
            ), mock.patch.object(
                calc_metrics.torch.multiprocessing,
                "set_start_method",
            ):
                result = runner.invoke(
                    calc_metrics.calc_metrics,
                    [
                        f"--network={handle.name}",
                        "--metrics=fid50k_full",
                        "--data=dummy.zip",
                        "--gpus=1",
                        *extra_args,
                    ],
                )
        return result

    def _fake_calc_metric(self, metric, G, dataset_kwargs, num_gpus, rank, device, progress):
        z = torch.randn(1, G.z_dim, device=device)
        c = torch.zeros([1, G.c_dim], device=device)
        out = G(z, c)
        self.observed_value = float(out.mean().item())
        return SimpleNamespace(results={metric: self.observed_value})

    def test_eval_best_rank_uses_snapshot_best_rank(self):
        self.observed_value = None
        result = self._run_calc_metrics(["--eval-best-rank", "--verbose=0"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertAlmostEqual(self.observed_value, 0.0, places=6)

    def test_eval_rank_override_uses_adapter(self):
        self.observed_value = None
        result = self._run_calc_metrics(["--eval-rank=0.5", "--verbose=0"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertAlmostEqual(self.observed_value, 0.5, places=6)
