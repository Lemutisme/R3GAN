import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from training.toy_datasets import sample_checkerboard, sample_swiss_roll
from training.toy_models import ToyRankMLP


class TestToyRgm(unittest.TestCase):
    def test_samplers_return_2d_points(self):
        self.assertEqual(sample_checkerboard(16).shape, (16, 2))
        self.assertEqual(sample_swiss_roll(16).shape, (16, 2))

    def test_same_latent_changes_with_rank(self):
        torch.manual_seed(3)
        model = ToyRankMLP()
        z = torch.randn(8, 32)
        low_rank = model(z, torch.zeros(8))
        high_rank = model(z, torch.ones(8))
        self.assertFalse(torch.allclose(low_rank, high_rank))

    def test_toy_script_smoke(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_toy_rgm.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--dataset=checkerboard",
                    "--toy-mode=rank_drift",
                    "--rank-levels=1.0,0.5,0.0",
                    "--steps=2",
                    "--plot-every=1",
                    f"--outdir={tmpdir}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            summary = json.loads((Path(tmpdir) / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["final"]["loss"] >= 0.0)
