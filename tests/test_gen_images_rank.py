import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from click.testing import CliRunner

import gen_images


class FakeRgmGenerator(torch.nn.Module):
    def __init__(self):
        super(FakeRgmGenerator, self).__init__()
        self.z_dim = 4
        self.c_dim = 2
        self.img_resolution = 4

    def forward(self, z, c, rank=None):
        if rank is None:
            value = torch.zeros([z.shape[0], 1, 1, 1], device=z.device)
        else:
            value = rank.reshape(z.shape[0], 1, 1, 1)
        return value.expand(z.shape[0], 3, 4, 4)


class FakeGanGenerator(torch.nn.Module):
    def __init__(self):
        super(FakeGanGenerator, self).__init__()
        self.z_dim = 4
        self.c_dim = 2
        self.img_resolution = 4

    def forward(self, z, c):
        return torch.zeros([z.shape[0], 3, 4, 4], device=z.device)


class TestGenImagesRank(unittest.TestCase):
    def test_rank_grid_generates_path_strip(self):
        runner = CliRunner()
        real_torch_device = torch.device
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(gen_images.dnnlib.util, "open_url", return_value=io.BytesIO()), mock.patch.object(
                gen_images.legacy,
                "load_network_pkl",
                return_value={"G_ema": FakeRgmGenerator(), "trainer": "rgm", "rank_levels": [1.0, 0.5, 0.0]},
            ), mock.patch.object(gen_images.torch, "device", return_value=real_torch_device("cpu")):
                result = runner.invoke(
                    gen_images.generate_images,
                    [
                        "--network=fake.pkl",
                        "--seeds=0",
                        "--class=0",
                        "--rank-grid=1.0,0.5,0.0",
                        f"--outdir={tmpdir}",
                    ],
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue((Path(tmpdir) / "seed0000_path.png").exists())

    def test_explicit_rank_generates_rank_file(self):
        runner = CliRunner()
        real_torch_device = torch.device
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(gen_images.dnnlib.util, "open_url", return_value=io.BytesIO()), mock.patch.object(
                gen_images.legacy,
                "load_network_pkl",
                return_value={"G_ema": FakeRgmGenerator(), "trainer": "rgm", "rank_levels": [1.0, 0.5, 0.0]},
            ), mock.patch.object(gen_images.torch, "device", return_value=real_torch_device("cpu")):
                result = runner.invoke(
                    gen_images.generate_images,
                    [
                        "--network=fake.pkl",
                        "--seeds=0",
                        "--class=0",
                        "--rank=0.5",
                        f"--outdir={tmpdir}",
                    ],
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue((Path(tmpdir) / "seed0000_rank0p5.png").exists())

    def test_gan_snapshot_keeps_old_output_name(self):
        runner = CliRunner()
        real_torch_device = torch.device
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(gen_images.dnnlib.util, "open_url", return_value=io.BytesIO()), mock.patch.object(
                gen_images.legacy,
                "load_network_pkl",
                return_value={"G_ema": FakeGanGenerator(), "trainer": "gan"},
            ), mock.patch.object(gen_images.torch, "device", return_value=real_torch_device("cpu")):
                result = runner.invoke(
                    gen_images.generate_images,
                    [
                        "--network=fake.pkl",
                        "--seeds=0",
                        "--class=0",
                        f"--outdir={tmpdir}",
                    ],
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue((Path(tmpdir) / "seed0000.png").exists())
