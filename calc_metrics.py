# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import copy
import json
import os
import tempfile

import click
import torch

import dnnlib
import legacy
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils import training_stats

metric_main = None
metric_utils = None
conv2d_gradfix = None


def _load_metric_modules():
    global metric_main, metric_utils
    if metric_main is None or metric_utils is None:
        from metrics import metric_main as metric_main_impl
        from metrics import metric_utils as metric_utils_impl

        metric_main = metric_main_impl
        metric_utils = metric_utils_impl
    return metric_main, metric_utils


def _load_conv2d_gradfix():
    global conv2d_gradfix
    if conv2d_gradfix is None:
        from torch_utils.ops import conv2d_gradfix as conv2d_gradfix_impl

        conv2d_gradfix = conv2d_gradfix_impl
    return conv2d_gradfix


class ConditionedGeneratorAdapter(torch.nn.Module):
    def __init__(self, base_G, *, fixed_rank=None):
        super(ConditionedGeneratorAdapter, self).__init__()
        self.base_G = base_G
        self.fixed_rank = fixed_rank
        self.z_dim = base_G.z_dim
        self.c_dim = base_G.c_dim
        self.img_resolution = base_G.img_resolution

    def forward(self, z, c):
        if self.fixed_rank is None:
            return self.base_G(z, c)
        rank = torch.full([z.shape[0]], float(self.fixed_rank), device=z.device, dtype=torch.float32)
        return self.base_G(z, c, rank=rank)


def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)
    metric_main_impl, metric_utils_impl = _load_metric_modules()
    conv2d_gradfix_impl = _load_conv2d_gradfix()

    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(backend="nccl", init_method=init_method, rank=rank, world_size=args.num_gpus)

    sync_device = torch.device("cuda", rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = "none"

    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix_impl.enabled = True

    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])

    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f"Calculating {metric}...")
        progress = metric_utils_impl.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main_impl.calc_metric(
            metric=metric,
            G=G,
            dataset_kwargs=args.dataset_kwargs,
            num_gpus=args.num_gpus,
            rank=rank,
            device=device,
            progress=progress,
        )
        if rank == 0:
            metric_main_impl.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    if rank == 0 and args.verbose:
        print("Exiting...")


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


@click.command()
@click.pass_context
@click.option("network_pkl", "--network", help="Network pickle filename or URL", metavar="PATH", required=True)
@click.option("--metrics", help="Quality metrics", metavar="[NAME|A,B,C|none]", type=parse_comma_separated_list, default="fid50k_full", show_default=True)
@click.option("--data", help="Dataset to evaluate against  [default: look up]", metavar="[ZIP|DIR]")
@click.option("--mirror", help="Enable dataset x-flips  [default: look up]", type=bool, metavar="BOOL")
@click.option("--gpus", help="Number of GPUs to use", type=int, default=1, metavar="INT", show_default=True)
@click.option("--verbose", help="Print optional information", type=bool, default=True, metavar="BOOL", show_default=True)
@click.option("--eval-rank", type=float, default=None, help="Explicit rank override for RGM snapshots")
@click.option("--eval-best-rank", is_flag=True, help="Evaluate the best/default rank for an RGM snapshot")
def calc_metrics(ctx, network_pkl, metrics, data, mirror, gpus, verbose, eval_rank, eval_best_rank):
    dnnlib.util.Logger(should_flush=True)
    metric_main_impl, _metric_utils_impl = _load_metric_modules()

    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose)
    if not all(metric_main_impl.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail("\n".join(["--metrics can only contain the following values:"] + metric_main_impl.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail("--gpus must be at least 1")

    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail("--network must point to a file or URL")
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)

    base_G = network_dict["G_ema"]
    fixed_rank = None
    if eval_rank is not None:
        fixed_rank = float(eval_rank)
    elif eval_best_rank:
        rank_levels = network_dict.get("rank_levels")
        if rank_levels is None:
            ctx.fail("--eval-best-rank requires an RGM snapshot with rank_levels metadata")
        fixed_rank = float(rank_levels[-1])
    args.G = ConditionedGeneratorAdapter(base_G, fixed_rank=fixed_rank) if fixed_rank is not None else base_G

    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", path=data)
    elif network_dict["training_set_kwargs"] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict["training_set_kwargs"])
    else:
        ctx.fail("Could not look up dataset options; please specify --data")

    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = args.G.c_dim != 0
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    if args.verbose:
        print("Dataset options:")
        print(json.dumps(args.dataset_kwargs, indent=2))

    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, "training_options.json")):
            args.run_dir = pkl_dir

    if args.verbose:
        print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn")
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


if __name__ == "__main__":
    calc_metrics()  # pylint: disable=no-value-for-parameter
