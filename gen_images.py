# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for part in s.split(","):
        match = range_re.match(part)
        if match:
            ranges.extend(range(int(match.group(1)), int(match.group(2)) + 1))
        else:
            ranges.append(int(part))
    return ranges


def parse_float_list(s: str) -> List[float]:
    if s is None or s == "":
        return []
    return [float(value) for value in s.split(",")]


@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--seeds", type=parse_range, help="List of random seeds (e.g., '0,1,4-6')", required=True)
@click.option("--class", "class_idx", type=int, help="Class label (unconditional if not specified)")
@click.option("--rank", "rank_value", type=float, default=None, help="Explicit rank for RGM snapshots")
@click.option("--rank-grid", type=str, default="", help="Comma-separated ranks to render as a horizontal strip")
@click.option("--outdir", help="Where to save the output images", type=str, required=True, metavar="DIR")
def generate_images(network_pkl: str, seeds: List[int], outdir: str, class_idx: Optional[int], rank_value: Optional[float], rank_grid: str):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
    G = network_dict["G_ema"].to(device)
    trainer = str(network_dict.get("trainer", "gan"))

    os.makedirs(outdir, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException("Must specify class label with --class when using a conditional network")
        label[:, class_idx] = 1
    elif class_idx is not None:
        print("warn: --class ignored when running on an unconditional network")

    rank_grid_values = parse_float_list(rank_grid)
    if rank_value is not None and len(rank_grid_values) > 0:
        raise click.ClickException("Use either --rank or --rank-grid, not both")

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        if len(rank_grid_values) > 0:
            if trainer != "rgm":
                raise click.ClickException("--rank-grid is only supported for RGM snapshots")
            strip = []
            for rank_item in rank_grid_values:
                strip.append(_tensor_to_uint8(_call_generator(G, z, label, rank=rank_item))[0].cpu().numpy())
            panel = np.concatenate(strip, axis=1)
            PIL.Image.fromarray(panel, "RGB").save(f"{outdir}/seed{seed:04d}_path.png")
        else:
            image = _call_generator(G, z, label, rank=rank_value)
            image = _tensor_to_uint8(image)
            if rank_value is None:
                file_name = f"{outdir}/seed{seed:04d}.png"
            else:
                file_name = f"{outdir}/seed{seed:04d}_rank{_format_rank_tag(rank_value)}.png"
            PIL.Image.fromarray(image[0].cpu().numpy(), "RGB").save(file_name)


def _call_generator(G, z, label, rank=None):
    if rank is None:
        return G(z, label)
    try:
        return G(z, label, rank=torch.full([z.shape[0]], float(rank), device=z.device, dtype=torch.float32))
    except TypeError as err:
        raise click.ClickException("This snapshot does not accept --rank") from err


def _tensor_to_uint8(img):
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


def _format_rank_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
