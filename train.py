# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import click
import re
import json
import tempfile
from pathlib import Path
import torch
from torch.utils.cpp_extension import verify_ninja_availability

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

# ----------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(
        file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True
    )

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=c.num_gpus,
            )
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=c.num_gpus,
            )

    # Init torch_utils.
    sync_device = torch.device("cuda", rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = "none"

    # Execute training loop.
    loop_kwargs = {k: v for k, v in c.items() if k != "trainer"}
    if getattr(c, "trainer", "gan") == "drift":
        from training import drift_training_loop

        drift_training_loop.training_loop(rank=rank, **loop_kwargs)
    else:
        training_loop.training_loop(rank=rank, **loop_kwargs)


# ----------------------------------------------------------------------------


def build_custom_ops_or_die():
    """Build required fused CUDA ops once before spawning worker processes."""
    # Use a writable cache directory by default so extension build artifacts
    # are stable across runs and do not depend on the host user's home config.
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(
            repo_root, ".cache", "torch_extensions"
        )
    os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

    try:
        verify_ninja_availability()

        from torch_utils.ops import bias_act
        from torch_utils.ops import upfirdn2d

        if not bias_act._init():
            raise RuntimeError('Failed to build or load "bias_act_plugin".')
        if not upfirdn2d._init():
            raise RuntimeError('Failed to build or load "upfirdn2d_plugin".')
    except Exception as err:
        raise click.ClickException(
            "Custom CUDA ops prebuild failed. "
            "Please run `bash scripts/build_custom_ops.sh` to diagnose and fix the toolchain, "
            f"then retry training.\nOriginal error: {err}"
        ) from err


# ----------------------------------------------------------------------------


def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [
            x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))
        ]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print("Training options:")
    print(json.dumps(c, indent=2))
    print()
    print(f"Output directory:    {c.run_dir}")
    print(f"Number of GPUs:      {c.num_gpus}")
    print(f"Batch size:          {c.batch_size} images")
    print(f"Training duration:   {c.total_kimg} kimg")
    print(f"Dataset path:        {c.training_set_kwargs.path}")
    print(f"Dataset size:        {c.training_set_kwargs.max_size} images")
    print(f"Dataset resolution:  {c.training_set_kwargs.resolution}")
    print(f"Dataset labels:      {c.training_set_kwargs.use_labels}")
    print(f"Dataset x-flips:     {c.training_set_kwargs.xflip}")
    print()

    # Dry run?
    if dry_run:
        print("Dry run; exiting.")
        return

    # Create output directory.
    print("Creating output directory...")
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
        json.dump(c, f, indent=2)

    # Build fused CUDA ops once in the parent process when the selected trainer
    # actually depends on them. The parity-oriented drift DiT backend is pure
    # PyTorch and should not be blocked by missing local extension toolchains.
    drift_backbone = None
    if getattr(c, "trainer", "gan") == "drift":
        drift_config = getattr(c, "drift_config", None)
        if drift_config is not None:
            drift_backbone = getattr(drift_config, "backbone", None)
    if getattr(c, "trainer", "gan") == "drift" and drift_backbone == "dit_like":
        print("Skipping custom CUDA op prebuild for drift/dit_like backend...")
    else:
        print("Prebuilding custom CUDA ops...")
        build_custom_ops_or_die()

    # Launch processes.
    print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn")
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus
            )


# ----------------------------------------------------------------------------


def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path=data,
            use_labels=True,
            max_size=None,
            xflip=False,
        )
        dataset_obj = dnnlib.util.construct_class_by_name(
            **dataset_kwargs
        )  # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = (
            dataset_obj.resolution
        )  # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f"--data: {err}")


# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


def parse_float_list(s):
    return [float(x) for x in parse_comma_separated_list(s)]


def parse_int_list(s):
    return [int(x) for x in parse_comma_separated_list(s)]


# ----------------------------------------------------------------------------


def _canonical_dataset_slug(dataset_name, data_path):
    candidates = []
    if dataset_name is not None:
        candidates.append(str(dataset_name))
    if data_path is not None:
        candidates.append(Path(str(data_path)).stem)

    for raw_value in candidates:
        normalized = re.sub(r"[^a-z0-9]+", "", str(raw_value).lower())
        if "cifar10" in normalized:
            return "cifar10"
        if "imagenet1k" in normalized:
            return "imagenet1k"
        if normalized.startswith("imagenet"):
            return "imagenet1k"
        if "ffhq" in normalized:
            return "ffhq"
    return None


def _infer_drift_periodic_eval_paths(*, dataset_name, data_path, inception_weights):
    dataset_slug = _canonical_dataset_slug(dataset_name=dataset_name, data_path=data_path)
    if dataset_slug is None:
        return None, None

    workspace_root = Path(__file__).resolve().parent.parent
    outputs_roots = [
        workspace_root / "outputs",
        workspace_root / "drift_models" / "outputs",
    ]

    root_candidates = []
    stats_candidates = []

    if dataset_slug == "imagenet1k":
        for base in outputs_roots:
            root_candidates.extend(
                [
                    base / "datasets" / "imagenet1k_val",
                    base / "datasets" / "imagenet1k_raw" / "val",
                ]
            )
            stats_candidates.extend(
                [
                    base / "datasets" / f"imagenet1k_val_reference_stats_{inception_weights}.pt",
                    base / "imagenet_eval" / f"reference_stats_{inception_weights}.pt",
                ]
            )
    else:
        for base in outputs_roots:
            root_candidates.append(base / "datasets" / f"{dataset_slug}_val")
            stats_candidates.extend(
                [
                    base / f"{dataset_slug}_eval" / f"reference_stats_{inception_weights}.pt",
                    base / "datasets" / f"{dataset_slug}_val_reference_stats_{inception_weights}.pt",
                ]
            )

    resolved_root = next((str(path) for path in root_candidates if path.is_dir()), None)
    resolved_stats = next((str(path) for path in stats_candidates if path.is_file()), None)
    return resolved_root, resolved_stats


# ----------------------------------------------------------------------------


@click.command()
# Required.
@click.option(
    "--outdir", help="Where to save the results", metavar="DIR", required=True
)
@click.option(
    "--data", help="Training data", metavar="[ZIP|DIR]", type=str, required=True
)
@click.option(
    "--gpus",
    help="Number of GPUs to use",
    metavar="INT",
    type=click.IntRange(min=1),
    required=True,
)
@click.option(
    "--batch",
    help="Total batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    required=True,
)
@click.option("--preset", help="Preset configs", metavar="STR", type=str, required=True)
@click.option(
    "--trainer",
    help="Training mode",
    type=click.Choice(["gan", "drift"]),
    default="gan",
    show_default=True,
)
# Optional features.
@click.option(
    "--cond",
    help="Train conditional model",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--mirror",
    help="Enable dataset x-flips",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--aug",
    help="Enable Augmentation",
    metavar="BOOL",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--resume", help="Resume from given network pickle", metavar="[PATH|URL]", type=str
)
@click.option(
    "--disable-r1", help="Disable R1 gradient penalty on real images", is_flag=True
)
@click.option(
    "--disable-r2", help="Disable R2 gradient penalty on generated images", is_flag=True
)
@click.option(
    "--non-aug-gp",
    help="Compute R1/R2 on non-augmented samples (recommended for convergence guarantees; adds extra D forward passes)",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
# Rank loss options.
@click.option(
    "--rank-loss",
    help="Legacy: enable deprecated interpolation path prior for D",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--rank-k",
    help="Legacy: number of interpolation steps for deprecated path prior",
    metavar="INT",
    type=click.IntRange(min=2),
    default=8,
    show_default=True,
)
@click.option(
    "--rank-loss-type",
    help="Legacy: ranking loss type for deprecated path prior",
    type=click.Choice(["listmle", "pairwise_logistic", "pairwise_hinge"]),
    default="listmle",
    show_default=True,
)
@click.option(
    "--lambda-rank",
    help="Legacy: weight for deprecated path prior",
    type=float,
    default=0.1,
    show_default=True,
)
@click.option(
    "--lambda-adv",
    help="Legacy: weight for mapped adversarial loss",
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    "--adv-loss-type",
    help="Legacy: adversarial loss type mapped to delta-centric pair/list weights",
    type=click.Choice(["softmargin", "infonce"]),
    default="softmargin",
    show_default=True,
)
@click.option(
    "--adv-margin",
    help="Legacy: pairwise soft-margin (0 reproduces RpGAN)",
    type=float,
    default=0.0,
    show_default=True,
)
@click.option(
    "--adv-tau",
    help="Legacy: temperature for mapped listwise InfoNCE loss",
    type=float,
    default=0.07,
    show_default=True,
)
@click.option(
    "--rank-mode",
    help="Interpolation mode for rank list",
    type=click.Choice(["intrpl", "noise", "add_mix"]),
    default="intrpl",
    show_default=True,
)
@click.option(
    "--rank-alpha-dist",
    help="Alpha distribution for rank list",
    type=click.Choice(["linear", "cosine", "random"]),
    default="linear",
    show_default=True,
)
@click.option(
    "--rank-augment",
    help="Legacy no-op: deprecated rank augmentation flag",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--rank-margin",
    help="Margin for pairwise hinge rank loss",
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    "--rank-score-reg",
    help="Score regularization weight for rank loss (lambda*mean(scores^2))",
    type=float,
    default=0.0,
    show_default=True,
)
# Delta-centric RankGAN options.
@click.option(
    "--lambda-pair",
    help="Weight for pairwise delta loss",
    type=float,
)
@click.option(
    "--pair-margin",
    help="Soft-margin for pairwise delta loss",
    type=float,
)
@click.option(
    "--lambda-list",
    help="Weight for listwise delta loss",
    type=float,
)
@click.option(
    "--list-loss-type",
    help="Listwise delta aggregator",
    type=click.Choice(["infonce"]),
    default="infonce",
    show_default=True,
)
@click.option(
    "--list-tau",
    help="Temperature for listwise InfoNCE delta loss",
    type=float,
)
@click.option(
    "--lambda-local-rank",
    help="Weight for semantic-local gap-rank prior on D",
    type=float,
    default=0.0,
    show_default=True,
)
@click.option(
    "--local-rank-k",
    help="Number of nearest fake neighbors for semantic-local gap-rank prior",
    metavar="INT",
    type=click.IntRange(min=2),
    default=4,
    show_default=True,
)
@click.option("--coupling-k",      help="kNN coupling neighbors (0=no coupling, backward compat)",    metavar="INT",   type=int,   default=0,    show_default=True)
@click.option("--lambda-list-d",   help="D-side local-listwise weight (overrides --lambda-list for D)", metavar="FLOAT", type=float, default=None)
@click.option("--lambda-list-g",   help="G-side local-listwise weight (overrides --lambda-list for G)", metavar="FLOAT", type=float, default=None)
@click.option(
    "--path-rank-reg",
    help="Enable deprecated interpolation-based path prior for D",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--path-rank-k",
    help="Number of interpolation steps for deprecated path prior",
    metavar="INT",
    type=click.IntRange(min=2),
    default=8,
    show_default=True,
)
@click.option(
    "--path-rank-loss-type",
    help="Deprecated path prior loss type",
    type=click.Choice(["listmle", "pairwise_logistic", "pairwise_hinge"]),
    default="listmle",
    show_default=True,
)
@click.option(
    "--lambda-path-rank",
    help="Weight for deprecated path prior",
    type=float,
    default=0.1,
    show_default=True,
)
@click.option(
    "--path-rank-mode",
    help="Interpolation mode for deprecated path prior",
    type=click.Choice(["intrpl", "noise", "add_mix"]),
    default="intrpl",
    show_default=True,
)
@click.option(
    "--path-rank-alpha-dist",
    help="Alpha distribution for deprecated path prior",
    type=click.Choice(["linear", "cosine", "random"]),
    default="linear",
    show_default=True,
)
@click.option(
    "--path-rank-margin",
    help="Margin for deprecated pairwise hinge path prior",
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    "--path-rank-score-reg",
    help="Score regularization weight for deprecated path prior",
    type=float,
    default=0.0,
    show_default=True,
)
# Misc hyperparameters.
@click.option(
    "--g-batch-gpu",
    help="Limit batch size per GPU for G",
    metavar="INT",
    type=click.IntRange(min=1),
)
@click.option(
    "--d-batch-gpu",
    help="Limit batch size per GPU for D",
    metavar="INT",
    type=click.IntRange(min=1),
)
@click.option(
    "--negatives-per-group",
    help="Generated negatives per drift group",
    metavar="INT",
    type=click.IntRange(min=2),
    default=4,
    show_default=True,
)
@click.option(
    "--positives-per-group",
    help="Positive real samples per drift group",
    metavar="INT",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
)
@click.option(
    "--unconditional-per-group",
    help="Unconditional real negatives per drift group",
    metavar="INT",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
)
@click.option(
    "--alpha-min",
    help="Minimum alpha for drift CFG weighting",
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    "--alpha-max",
    help="Maximum alpha for drift CFG weighting",
    type=float,
    default=4.0,
    show_default=True,
)
@click.option(
    "--queue-capacity-per-class",
    help="Per-class drift queue capacity",
    metavar="INT",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
)
@click.option(
    "--queue-capacity-global",
    help="Global drift queue capacity",
    metavar="INT",
    type=click.IntRange(min=1),
    default=4096,
    show_default=True,
)
@click.option(
    "--queue-push-batch",
    help="Total real samples pushed into the drift queue per step",
    metavar="INT",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
)
@click.option(
    "--queue-warmup-batches",
    help="Number of queue warmup batches before drift training starts",
    metavar="INT",
    type=click.IntRange(min=0),
    default=4,
    show_default=True,
)
@click.option(
    "--drift-temperature",
    help="Temperature for pixel-space drift affinity",
    type=float,
    default=0.05,
    show_default=True,
)
@click.option(
    "--eval-alpha",
    help="Default alpha used when sampling/evaluating a drift generator without explicit alpha",
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    "--drift-backbone",
    help="Drift generator backend",
    type=click.Choice(["dit_like", "r3gan_conv"]),
    default="dit_like",
    show_default=True,
)
@click.option("--alpha-fixed", help="Fixed training alpha for drift", type=float, default=None)
@click.option(
    "--alpha-dist",
    help="Alpha sampling distribution for drift training",
    type=click.Choice(["uniform", "powerlaw", "mixture_point_powerlaw", "table8_l2_latent"]),
    default="uniform",
    show_default=True,
)
@click.option("--alpha-power", help="Power-law exponent for alpha sampling", type=float, default=3.0, show_default=True)
@click.option("--alpha-point", help="Point-mass alpha for mixture sampling", type=float, default=1.0, show_default=True)
@click.option("--alpha-point-prob", help="Probability of sampling alpha-point", type=float, default=0.5, show_default=True)
@click.option(
    "--drift-temperatures",
    help="Comma-separated extra drift temperatures for multi-temperature raw drift",
    type=str,
    default="",
    show_default=True,
)
@click.option(
    "--drift-temperature-reduction",
    help="Reduction over multiple raw drift temperatures",
    type=click.Choice(["mean", "sum"]),
    default="sum",
    show_default=True,
)
@click.option("--learning-rate", help="Drift learning rate", type=float, default=2e-4, show_default=True)
@click.option("--adam-beta1", help="Drift Adam/AdamW beta1", type=float, default=0.0, show_default=True)
@click.option("--adam-beta2", help="Drift Adam/AdamW beta2", type=float, default=0.0, show_default=True)
@click.option("--weight-decay", help="Drift optimizer weight decay", type=float, default=0.0, show_default=True)
@click.option(
    "--scheduler",
    help="Drift learning-rate scheduler",
    type=click.Choice(["none", "constant", "cosine", "warmup_cosine"]),
    default="none",
    show_default=True,
)
@click.option("--warmup-steps", help="Warmup steps for warmup_cosine drift scheduler", type=int, default=0, show_default=True)
@click.option("--clip-grad-norm", help="Gradient clipping norm for drift training", type=float, default=2.0, show_default=True)
@click.option("--use-bf16/--no-bf16", help="Use bfloat16 autocast for generator forward pass", default=True, show_default=True)
@click.option("--compile-generator", help="Compile the drift generator forward", is_flag=True)
@click.option("--compile-backend", help="torch.compile backend for generator", type=str, default="inductor", show_default=True)
@click.option("--compile-mode", help="torch.compile mode for generator", type=str, default="reduce-overhead", show_default=True)
@click.option("--compile-dynamic", help="Enable dynamic torch.compile for generator", is_flag=True)
@click.option("--compile-fullgraph", help="Enable fullgraph torch.compile for generator", is_flag=True)
@click.option(
    "--compile-fail-action",
    help="How drift generator compile failures are handled",
    type=click.Choice(["warn", "raise", "disable"]),
    default="warn",
    show_default=True,
)
@click.option("--patch-size", help="DiT-like patch size for drift backend", type=int, default=4, show_default=True)
@click.option("--hidden-dim", help="DiT-like hidden dimension for drift backend", type=int, default=256, show_default=True)
@click.option("--depth", help="DiT-like depth for drift backend", type=int, default=6, show_default=True)
@click.option("--num-heads", help="DiT-like attention heads for drift backend", type=int, default=8, show_default=True)
@click.option("--mlp-ratio", help="DiT-like MLP ratio for drift backend", type=float, default=4.0, show_default=True)
@click.option("--ffn-inner-dim", help="Optional explicit DiT-like FFN inner dimension", type=int, default=None)
@click.option("--register-tokens", help="DiT-like register token count", type=int, default=16, show_default=True)
@click.option("--style-vocab-size", help="Style vocabulary size for drift backend", type=int, default=1, show_default=True)
@click.option("--style-token-count", help="Style token count for drift backend", type=int, default=0, show_default=True)
@click.option("--alpha-hidden-dim", help="Hidden dimension for alpha embedding MLP", type=int, default=128, show_default=True)
@click.option(
    "--norm-type",
    help="Normalization type for drift DiT blocks",
    type=click.Choice(["layernorm", "rmsnorm"]),
    default="layernorm",
    show_default=True,
)
@click.option("--use-qk-norm", help="Enable QK normalization in drift DiT attention", is_flag=True)
@click.option("--use-rope", help="Enable rotary embeddings in drift DiT attention", is_flag=True)
@click.option(
    "--alpha-embedding-type",
    help="Alpha embedding type for drift DiT backend",
    type=click.Choice(["mlp", "fourier_mlp"]),
    default="mlp",
    show_default=True,
)
@click.option(
    "--qk-norm-mode",
    help="QK normalization mode for drift DiT attention",
    type=click.Choice(["auto", "none", "l2"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--rope-mode",
    help="Rotary embedding mode for drift DiT attention",
    type=click.Choice(["auto", "none", "1d_flat", "2d_axial"]),
    default="auto",
    show_default=True,
)
@click.option("--disable-patch-positional-embedding", help="Disable DiT patch positional embedding", is_flag=True)
@click.option("--disable-rmsnorm-affine", help="Disable RMSNorm affine parameters in drift DiT", is_flag=True)
@click.option("--use-feature-loss", help="Enable feature-space drift loss", is_flag=True)
@click.option(
    "--feature-encoder",
    help="Feature encoder used for drift feature loss",
    type=click.Choice(["tiny", "mae", "convnext_tiny", "convnextv2_tiny", "mae_convnextv2"]),
    default="tiny",
    show_default=True,
)
@click.option("--convnext-weights", help="Weights preset for convnext_tiny feature encoder", type=click.Choice(["none", "imagenet1k_v1"]), default="none", show_default=True)
@click.option("--convnextv2-weights", help="Weights preset for convnextv2_tiny feature encoder", type=click.Choice(["none", "imagenet1k_v1"]), default="none", show_default=True)
@click.option("--mae-encoder-path", help="Optional MAE encoder checkpoint for drift feature loss", type=str, default=None)
@click.option(
    "--mae-encoder-arch",
    help="MAE encoder architecture for drift feature loss",
    type=click.Choice(["resnet_unet", "legacy_conv", "paper_resnet34_unet"]),
    default="resnet_unet",
    show_default=True,
)
@click.option("--mae-input-patchify-size", help="MAE input patchify size", type=int, default=1, show_default=True)
@click.option("--feature-base-channels", help="Base channels for tiny/MAE feature encoders", type=int, default=16, show_default=True)
@click.option("--feature-stages", help="Number of stages for tiny/MAE feature encoders", type=int, default=3, show_default=True)
@click.option(
    "--feature-temperatures",
    help="Comma-separated feature drift temperatures",
    type=str,
    default="0.02,0.05,0.2",
    show_default=True,
)
@click.option(
    "--feature-temperature-aggregation",
    help="How to aggregate feature drift temperatures",
    type=click.Choice(["per_temperature_mse", "sum_drifts_then_mse"]),
    default="sum_drifts_then_mse",
    show_default=True,
)
@click.option(
    "--feature-loss-term-reduction",
    help="Reduction over feature loss terms",
    type=click.Choice(["sum", "mean"]),
    default="sum",
    show_default=True,
)
@click.option(
    "--feature-selected-stages",
    help="Comma-separated selected feature stages; empty means all",
    type=str,
    default="",
    show_default=True,
)
@click.option("--include-patch4-stats", help="Include patch4 feature vector statistics", is_flag=True)
@click.option("--include-input-x2-mean", help="Include input x2 mean feature statistics", is_flag=True)
@click.option("--disable-shared-location-normalization", help="Disable shared location normalization for feature loss", is_flag=True)
@click.option("--disable-feature-temperature-sqrt-scaling", help="Disable sqrt(channel) scaling for feature temperatures", is_flag=True)
@click.option("--feature-include-raw-drift-loss", help="Mix raw pixel drift into feature loss", is_flag=True)
@click.option("--feature-raw-drift-loss-weight", help="Weight for raw drift loss mixed into feature loss", type=float, default=1.0, show_default=True)
@click.option("--feature-compile-drift-kernel", help="Compile the feature drift kernel", is_flag=True)
@click.option("--feature-compile-backend", help="torch.compile backend for feature drift kernel", type=str, default="inductor", show_default=True)
@click.option("--feature-compile-mode", help="torch.compile mode for feature drift kernel", type=str, default="reduce-overhead", show_default=True)
@click.option("--feature-compile-dynamic", help="Enable dynamic torch.compile for feature drift kernel", is_flag=True)
@click.option("--feature-compile-fullgraph", help="Enable fullgraph torch.compile for feature drift kernel", is_flag=True)
@click.option(
    "--feature-compile-fail-action",
    help="How feature drift compile failures are handled",
    type=click.Choice(["warn", "raise"]),
    default="warn",
    show_default=True,
)
@click.option("--queue-prime-samples", help="Real samples used to warm the drift queue", type=int, default=200, show_default=True)
@click.option(
    "--queue-warmup-mode",
    help="Drift queue warmup mode",
    type=click.Choice(["random", "class_balanced"]),
    default="random",
    show_default=True,
)
@click.option("--queue-warmup-min-per-class", help="Minimum per-class count for class_balanced warmup", type=int, default=1, show_default=True)
@click.option("--queue-strict-without-replacement", help="Disallow queue sampling with replacement", is_flag=True)
@click.option(
    "--queue-refill-policy",
    help="Queue refill policy during drift training",
    type=click.Choice(["per_step", "every_n_steps"]),
    default="per_step",
    show_default=True,
)
@click.option("--queue-refill-every", help="Number of steps between queue refills when using every_n_steps", type=int, default=1, show_default=True)
@click.option(
    "--queue-report-level",
    help="Queue report detail level",
    type=click.Choice(["basic", "full"]),
    default="basic",
    show_default=True,
)
@click.option(
    "--real-batch-source",
    help="Source used for drift real batches",
    type=click.Choice(["dataset_loader", "synthetic_dataset", "imagefolder", "tensor_file", "tensor_shards", "webdataset"]),
    default="dataset_loader",
    show_default=True,
)
@click.option("--real-dataset-size", help="Synthetic drift real dataset size", type=int, default=4096, show_default=True)
@click.option("--real-loader-batch-size", help="Batch size for real-batch provider", type=int, default=128, show_default=True)
@click.option("--real-num-workers", help="Worker count for real-batch provider", type=int, default=0, show_default=True)
@click.option("--disable-real-shuffle", help="Disable shuffling in real-batch provider", is_flag=True)
@click.option("--real-pin-memory", help="Enable pin_memory in real-batch provider", is_flag=True)
@click.option("--real-persistent-workers", help="Enable persistent workers in real-batch provider", is_flag=True)
@click.option("--real-prefetch-factor", help="Prefetch factor for real-batch provider", type=int, default=0, show_default=True)
@click.option("--real-sanity-sample-batches", help="Sample this many batches to build a real-provider sanity report", type=int, default=0, show_default=True)
@click.option("--real-imagefolder-root", help="ImageFolder root for drift real-batch provider", type=str, default=None)
@click.option("--real-webdataset-urls", help="WebDataset URL(s) for drift real-batch provider", type=str, default=None)
@click.option("--real-tensor-file-path", help="Tensor file path for drift real-batch provider", type=str, default=None)
@click.option("--real-tensor-shards-manifest-path", help="Tensor shards manifest path for drift real-batch provider", type=str, default=None)
@click.option("--real-transform-resize", help="Resize for imagefolder real-batch provider", type=int, default=None)
@click.option("--disable-real-center-crop", help="Disable center crop in imagefolder real-batch provider", is_flag=True)
@click.option("--real-horizontal-flip", help="Enable horizontal flip in imagefolder real-batch provider", is_flag=True)
@click.option("--real-transform-normalize", help="Normalize imagefolder real batches to [0,1]", is_flag=True)
@click.option("--resume-model-only", help="Load only model weights from a drift checkpoint", is_flag=True)
@click.option("--resume-reset-scheduler", help="Reset drift scheduler state on resume", is_flag=True)
@click.option("--resume-reset-optimizer-lr", help="Reset drift optimizer LR on resume", is_flag=True)
@click.option("--allow-resume-config-mismatch", help="Allow drift checkpoint config hash mismatch", is_flag=True)
@click.option("--save-every", help="Save a research checkpoint every N steps; 0 disables", type=int, default=0, show_default=True)
@click.option("--checkpoint-dir", help="Directory for drift checkpoints; defaults under run dir", type=str, default=None)
@click.option("--keep-last-k-checkpoints", help="Keep only the latest K drift checkpoints; 0 keeps all", type=int, default=0, show_default=True)
@click.option("--eval-every-kimg", help="Run reference-style periodic eval every N kimg; 0 disables", type=float, default=0.0, show_default=True)
@click.option("--eval-reference-imagefolder-root", help="Reference image folder used to build eval stats", type=str, default=None)
@click.option("--eval-reference-stats-path", help="Reference stats path for drift periodic eval", type=str, default=None)
@click.option("--eval-samples", help="Generated sample count for each periodic eval", type=int, default=50000, show_default=True)
@click.option("--eval-sample-batch-size", help="Generator batch size used during periodic eval sampling", type=int, default=128, show_default=True)
@click.option("--eval-batch-size", help="Batch size used when building reference eval stats", type=int, default=128, show_default=True)
@click.option("--eval-num-workers", help="Worker count used during periodic eval", type=int, default=0, show_default=True)
@click.option("--eval-inception-weights", help="Inception weights preset for periodic eval", type=click.Choice(["pretrained", "none"]), default="pretrained", show_default=True)
@click.option(
    "--eval-postprocess-mode",
    help="Postprocess mode for periodic eval samples",
    type=click.Choice(["clamp_0_1", "tanh_to_0_1", "sigmoid", "identity"]),
    default="clamp_0_1",
    show_default=True,
)
# Misc settings.
@click.option(
    "--desc", help="String to include in result dir name", metavar="STR", type=str
)
@click.option(
    "--metrics",
    help="Quality metrics",
    metavar="[NAME|A,B,C|none]",
    type=parse_comma_separated_list,
    default="fid50k_full",
    show_default=True,
)
@click.option(
    "--kimg",
    help="Total training duration",
    metavar="KIMG",
    type=click.IntRange(min=1),
    default=10000000,
    show_default=True,
)
@click.option(
    "--tick",
    help="How often to print progress",
    metavar="KIMG",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
)
@click.option(
    "--snap",
    help="How often to save snapshots",
    metavar="TICKS",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
)
@click.option(
    "--snapshot-policy",
    help="Snapshot retention policy",
    type=click.Choice(["all", "latest-best"]),
    default="latest-best",
    show_default=True,
)
@click.option(
    "--seed",
    help="Random seed",
    metavar="INT",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--nobench",
    help="Disable cuDNN benchmarking",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--workers",
    help="DataLoader worker processes",
    metavar="INT",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.

    c.G_kwargs = dnnlib.EasyDict(class_name="training.networks.Generator")
    c.D_kwargs = dnnlib.EasyDict(class_name="training.networks.Discriminator")

    c.G_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", betas=[0.0, 0.0], eps=1e-8
    )
    c.D_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", betas=[0.0, 0.0], eps=1e-8
    )

    c.loss_kwargs = dnnlib.EasyDict(class_name="training.loss.R3GANLoss")
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException(
            "--cond=True requires labels specified in dataset.json"
        )
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.g_batch_gpu = opts.g_batch_gpu or opts.batch // opts.gpus
    c.d_batch_gpu = opts.d_batch_gpu or opts.batch // opts.gpus

    if opts.preset == "CIFAR10":
        WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32]]
        FP16Stages = [-1, -2, -3]
        NoiseDimension = 64

        if opts.cond:
            c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
            c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]

        ema_nimg = 5000 * 1000
        decay_nimg = 2e7

        c.ema_scheduler = {
            "base_value": 0,
            "final_value": ema_nimg,
            "total_nimg": decay_nimg,
        }
        c.aug_scheduler = {
            "base_value": 0,
            "final_value": 0.55,
            "total_nimg": decay_nimg,
        }
        c.lr_scheduler = {
            "base_value": 2e-4,
            "final_value": 5e-5,
            "total_nimg": decay_nimg,
        }
        c.gamma_scheduler = {
            "base_value": 0.05,
            "final_value": 0.005,
            "total_nimg": decay_nimg,
        }
        c.beta2_scheduler = {
            "base_value": 0.9,
            "final_value": 0.99,
            "total_nimg": decay_nimg,
        }

    if opts.preset == "FFHQ-64":
        WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024, 512]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 16]]
        FP16Stages = [-1, -2, -3, -4]
        NoiseDimension = 64

        ema_nimg = 500 * 1000
        decay_nimg = 2e7

        c.ema_scheduler = {
            "base_value": 0,
            "final_value": ema_nimg,
            "total_nimg": decay_nimg,
        }
        c.aug_scheduler = {
            "base_value": 0,
            "final_value": 0.3,
            "total_nimg": decay_nimg,
        }
        c.lr_scheduler = {
            "base_value": 2e-4,
            "final_value": 5e-5,
            "total_nimg": decay_nimg,
        }
        c.gamma_scheduler = {
            "base_value": 2,
            "final_value": 0.2,
            "total_nimg": decay_nimg,
        }
        c.beta2_scheduler = {
            "base_value": 0.9,
            "final_value": 0.99,
            "total_nimg": decay_nimg,
        }

    if opts.preset == "FFHQ-256":
        WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024, 512, 256, 128]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 16, 8, 4]]
        FP16Stages = [-1, -2, -3, -4]
        NoiseDimension = 64

        ema_nimg = 500 * 1000
        decay_nimg = 2e7

        c.ema_scheduler = {
            "base_value": 0,
            "final_value": ema_nimg,
            "total_nimg": decay_nimg,
        }
        c.aug_scheduler = {
            "base_value": 0,
            "final_value": 0.3,
            "total_nimg": decay_nimg,
        }
        c.lr_scheduler = {
            "base_value": 2e-4,
            "final_value": 5e-5,
            "total_nimg": decay_nimg,
        }
        c.gamma_scheduler = {
            "base_value": 150,
            "final_value": 15,
            "total_nimg": decay_nimg,
        }
        c.beta2_scheduler = {
            "base_value": 0.9,
            "final_value": 0.99,
            "total_nimg": decay_nimg,
        }

    if opts.preset == "ImageNet-32":
        WidthPerStage = [6 * x // 4 for x in [1024, 1024, 1024, 1024]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32]]
        FP16Stages = [-1, -2, -3]
        NoiseDimension = 64

        c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
        c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]

        ema_nimg = 50000 * 1000
        decay_nimg = 2e8

        c.ema_scheduler = {
            "base_value": 0,
            "final_value": ema_nimg,
            "total_nimg": decay_nimg,
        }
        c.aug_scheduler = {
            "base_value": 0,
            "final_value": 0.5,
            "total_nimg": decay_nimg,
        }
        c.lr_scheduler = {
            "base_value": 2e-4,
            "final_value": 5e-5,
            "total_nimg": decay_nimg,
        }
        c.gamma_scheduler = {
            "base_value": 0.5,
            "final_value": 0.05,
            "total_nimg": decay_nimg,
        }
        c.beta2_scheduler = {
            "base_value": 0.9,
            "final_value": 0.99,
            "total_nimg": decay_nimg,
        }

    if opts.preset == "ImageNet-64":
        WidthPerStage = [6 * x // 4 for x in [1024, 1024, 1024, 1024, 1024]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 32]]
        FP16Stages = [-1, -2, -3, -4]
        NoiseDimension = 64

        c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
        c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]

        ema_nimg = 50000 * 1000
        decay_nimg = 2e8

        c.ema_scheduler = {
            "base_value": 0,
            "final_value": ema_nimg,
            "total_nimg": decay_nimg,
        }
        c.aug_scheduler = {
            "base_value": 0,
            "final_value": 0.4,
            "total_nimg": decay_nimg,
        }
        c.lr_scheduler = {
            "base_value": 2e-4,
            "final_value": 5e-5,
            "total_nimg": decay_nimg,
        }
        c.gamma_scheduler = {
            "base_value": 1,
            "final_value": 0.1,
            "total_nimg": decay_nimg,
        }
        c.beta2_scheduler = {
            "base_value": 0.9,
            "final_value": 0.99,
            "total_nimg": decay_nimg,
        }

    c.G_kwargs.NoiseDimension = NoiseDimension
    c.G_kwargs.WidthPerStage = WidthPerStage
    c.G_kwargs.CardinalityPerStage = CardinalityPerStage
    c.G_kwargs.BlocksPerStage = BlocksPerStage
    c.G_kwargs.ExpansionFactor = 2
    c.G_kwargs.FP16Stages = FP16Stages

    c.D_kwargs.WidthPerStage = [*reversed(WidthPerStage)]
    c.D_kwargs.CardinalityPerStage = [*reversed(CardinalityPerStage)]
    c.D_kwargs.BlocksPerStage = [*reversed(BlocksPerStage)]
    c.D_kwargs.ExpansionFactor = 2
    c.D_kwargs.FP16Stages = [x + len(FP16Stages) for x in FP16Stages]

    # Coupling-aware listwise curriculum (only when --coupling-k > 0)
    if opts.coupling_k > 0:
        total_nimg = c.lr_scheduler['total_nimg']
        list_d_target = opts.lambda_list_d if opts.lambda_list_d is not None else 0.0
        list_g_target = opts.lambda_list_g if opts.lambda_list_g is not None else 0.0
        if list_d_target > 0:
            c.list_d_scheduler = dict(
                base_value=list_d_target, total_nimg=total_nimg, final_value=list_d_target,
                warmup_value=0.0, warmup_nimg=total_nimg // 4,
            )
        if list_g_target > 0:
            c.list_g_scheduler = dict(
                base_value=list_g_target, total_nimg=total_nimg, final_value=list_g_target,
                warmup_value=0.0, warmup_nimg=total_nimg // 2,
            )

    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.snapshot_policy = opts.snapshot_policy
    c.trainer = opts.trainer
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.data_loader_kwargs.persistent_workers = opts.workers > 0

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException("--batch must be a multiple of --gpus")
    if (
        c.batch_size % (c.num_gpus * c.g_batch_gpu) != 0
        or c.batch_size % (c.num_gpus * c.d_batch_gpu) != 0
    ):
        raise click.ClickException(
            "--batch must be a multiple of --gpus times --batch-gpu"
        )
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException(
            "\n".join(
                ["--metrics can only contain the following values:"]
                + metric_main.list_valid_metrics()
            )
        )
    if opts.lambda_rank < 0:
        raise click.ClickException("--lambda-rank must be non-negative")
    if opts.lambda_adv < 0:
        raise click.ClickException("--lambda-adv must be non-negative")
    if opts.adv_margin < 0:
        raise click.ClickException("--adv-margin must be non-negative")
    if opts.adv_tau <= 0:
        raise click.ClickException("--adv-tau must be positive")
    if opts.rank_margin <= 0:
        raise click.ClickException("--rank-margin must be positive")
    if opts.rank_score_reg < 0:
        raise click.ClickException("--rank-score-reg must be non-negative")
    if opts.lambda_local_rank < 0:
        raise click.ClickException("--lambda-local-rank must be non-negative")
    if opts.coupling_k < 0:
        raise click.ClickException('--coupling-k must be non-negative')
    if opts.coupling_k > 0 and opts.lambda_pair is not None and opts.lambda_pair <= 0 and (opts.lambda_list_d is None or opts.lambda_list_d <= 0) and (opts.lambda_list_g is None or opts.lambda_list_g <= 0):
        click.echo('WARNING: coupling_k > 0 but no pairwise or listwise loss enabled')
    if opts.path_rank_margin <= 0:
        raise click.ClickException("--path-rank-margin must be positive")
    if opts.lambda_path_rank < 0:
        raise click.ClickException("--lambda-path-rank must be non-negative")
    if opts.path_rank_score_reg < 0:
        raise click.ClickException("--path-rank-score-reg must be non-negative")

    # Augmentation.
    if opts.aug:
        c.augment_kwargs = dnnlib.EasyDict(
            class_name="training.augment.AugmentPipe",
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=0.5,
            contrast=0.5,
            lumaflip=0.5,
            hue=0.5,
            saturation=0.5,
            cutout=1,
        )

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume

    # Performance-related toggles.
    if opts.nobench:
        c.cudnn_benchmark = False

    if opts.trainer == "drift":
        if not opts.cond:
            raise click.ClickException("--trainer=drift requires --cond=1")
        if not c.training_set_kwargs.use_labels:
            raise click.ClickException("--trainer=drift requires a labeled dataset")
        if opts.alpha_fixed is not None and opts.alpha_fixed < 1.0:
            raise click.ClickException("--alpha-fixed must be >= 1.0")
        if opts.alpha_min < 1.0:
            raise click.ClickException("--alpha-min must be >= 1.0")
        if opts.alpha_max < opts.alpha_min:
            raise click.ClickException("--alpha-max must be >= --alpha-min")
        if not 0.0 <= opts.alpha_point_prob <= 1.0:
            raise click.ClickException("--alpha-point-prob must be in [0, 1]")
        if opts.eval_alpha <= 0:
            raise click.ClickException("--eval-alpha must be positive")
        if opts.drift_temperature <= 0:
            raise click.ClickException("--drift-temperature must be positive")
        if opts.clip_grad_norm <= 0:
            raise click.ClickException("--clip-grad-norm must be positive")
        if c.batch_size % (c.num_gpus * opts.negatives_per_group) != 0:
            raise click.ClickException(
                "--batch / --gpus must be divisible by --negatives-per-group for drift training"
            )
        if opts.queue_push_batch % c.num_gpus != 0:
            raise click.ClickException("--queue-push-batch must be divisible by --gpus")
        if opts.queue_warmup_min_per_class <= 0:
            raise click.ClickException("--queue-warmup-min-per-class must be positive")
        if opts.queue_refill_every <= 0:
            raise click.ClickException("--queue-refill-every must be positive")
        if opts.real_loader_batch_size <= 0:
            raise click.ClickException("--real-loader-batch-size must be positive")
        if opts.learning_rate <= 0:
            raise click.ClickException("--learning-rate must be positive")
        if opts.adam_beta1 < 0 or opts.adam_beta1 >= 1:
            raise click.ClickException("--adam-beta1 must be in [0, 1)")
        if opts.adam_beta2 < 0 or opts.adam_beta2 >= 1:
            raise click.ClickException("--adam-beta2 must be in [0, 1)")
        if opts.weight_decay < 0:
            raise click.ClickException("--weight-decay must be non-negative")
        if opts.style_vocab_size < 1:
            raise click.ClickException("--style-vocab-size must be >= 1")
        if opts.style_token_count < 0:
            raise click.ClickException("--style-token-count must be non-negative")
        if opts.eval_every_kimg < 0:
            raise click.ClickException("--eval-every-kimg must be non-negative")
        if opts.eval_samples <= 0:
            raise click.ClickException("--eval-samples must be positive")
        if opts.eval_sample_batch_size <= 0 or opts.eval_batch_size <= 0:
            raise click.ClickException("--eval-sample-batch-size and --eval-batch-size must be positive")
        if opts.feature_raw_drift_loss_weight < 0:
            raise click.ClickException("--feature-raw-drift-loss-weight must be non-negative")

        if opts.aug:
            click.echo("NOTE: --aug is ignored for --trainer=drift.")
        if opts.disable_r1 or opts.disable_r2 or opts.non_aug_gp:
            click.echo("NOTE: R1/R2 and non-aug GP flags are ignored for --trainer=drift.")
        if (
            opts.lambda_pair is not None
            or opts.pair_margin is not None
            or opts.lambda_list is not None
            or opts.list_tau is not None
            or opts.lambda_local_rank != 0.0
            or opts.coupling_k != 0
            or opts.path_rank_reg
            or opts.rank_loss
        ):
            click.echo("NOTE: adversarial and discriminator-side ranking options are ignored for --trainer=drift.")

        drift_temperatures = parse_float_list(opts.drift_temperatures)
        feature_temperatures = parse_float_list(opts.feature_temperatures)
        feature_selected_stages = parse_int_list(opts.feature_selected_stages)
        resolved_eval_every_kimg = float(opts.eval_every_kimg)
        resolved_eval_reference_imagefolder_root = opts.eval_reference_imagefolder_root
        resolved_eval_reference_stats_path = opts.eval_reference_stats_path

        if opts.drift_backbone == "dit_like" and (len(c.metrics) > 0 or resolved_eval_every_kimg > 0.0):
            inferred_eval_root, inferred_eval_stats = _infer_drift_periodic_eval_paths(
                dataset_name=dataset_name,
                data_path=opts.data,
                inception_weights=str(opts.eval_inception_weights),
            )
            if resolved_eval_reference_imagefolder_root is None:
                resolved_eval_reference_imagefolder_root = inferred_eval_root
            if resolved_eval_reference_stats_path is None:
                resolved_eval_reference_stats_path = inferred_eval_stats

        if opts.drift_backbone == "dit_like" and len(c.metrics) > 0 and resolved_eval_every_kimg <= 0.0:
            if resolved_eval_reference_imagefolder_root is not None:
                resolved_eval_every_kimg = float(c.kimg_per_tick * c.network_snapshot_ticks)
                click.echo(
                    "NOTE: enabling drift periodic eval to mirror --metrics; "
                    f"using --eval-every-kimg={resolved_eval_every_kimg:g}."
                )
            else:
                click.echo(
                    "NOTE: drift periodic eval was not auto-enabled because no reference imagefolder "
                    "could be inferred. Set --eval-reference-imagefolder-root to enable eval."
                )

        if opts.drift_backbone == "dit_like":
            c.G_kwargs.class_name = "training.networks.DiTLikeDriftGenerator"
            c.G_kwargs.ImageChannels = 3
            c.G_kwargs.EvalAlpha = opts.eval_alpha
            c.G_kwargs.PatchSize = opts.patch_size
            c.G_kwargs.HiddenDim = opts.hidden_dim
            c.G_kwargs.Depth = opts.depth
            c.G_kwargs.NumHeads = opts.num_heads
            c.G_kwargs.MlpRatio = opts.mlp_ratio
            c.G_kwargs.FfnInnerDim = opts.ffn_inner_dim
            c.G_kwargs.RegisterTokens = opts.register_tokens
            c.G_kwargs.StyleVocabSize = opts.style_vocab_size
            c.G_kwargs.StyleTokenCount = opts.style_token_count
            c.G_kwargs.AlphaHiddenDim = opts.alpha_hidden_dim
            c.G_kwargs.NormType = opts.norm_type
            c.G_kwargs.UseQkNorm = bool(opts.use_qk_norm)
            c.G_kwargs.UseRope = bool(opts.use_rope)
            c.G_kwargs.AlphaEmbeddingType = opts.alpha_embedding_type
            c.G_kwargs.QkNormMode = opts.qk_norm_mode
            c.G_kwargs.RopeMode = opts.rope_mode
            c.G_kwargs.DisablePatchPositionalEmbedding = bool(opts.disable_patch_positional_embedding)
            c.G_kwargs.DisableRmsNormAffine = bool(opts.disable_rmsnorm_affine)
            optimizer_class_name = "torch.optim.AdamW" if float(opts.weight_decay) > 0 else "torch.optim.Adam"
            c.G_opt_kwargs = dnnlib.EasyDict(
                class_name=optimizer_class_name,
                lr=float(opts.learning_rate),
                betas=[float(opts.adam_beta1), float(opts.adam_beta2)],
                eps=1e-8,
            )
            if optimizer_class_name.endswith("AdamW"):
                c.G_opt_kwargs.weight_decay = float(opts.weight_decay)
        else:
            c.G_kwargs.class_name = "training.networks.DriftGenerator"
            c.G_kwargs.AlphaMin = opts.alpha_min
            c.G_kwargs.AlphaMax = opts.alpha_max
            c.G_kwargs.EvalAlpha = opts.eval_alpha
            c.G_opt_kwargs = dnnlib.EasyDict(
                class_name="torch.optim.Adam",
                lr=float(opts.learning_rate),
                betas=[0.0, 0.0],
                eps=1e-8,
            )
        c.D_kwargs = None
        c.D_opt_kwargs = None
        c.loss_kwargs = dnnlib.EasyDict()
        c.augment_kwargs = None
        c.aug_scheduler = None
        c.gamma_scheduler = None
        if opts.drift_backbone == "dit_like":
            user_lr = float(opts.learning_rate)
            total_nimg = opts.kimg * 1000
            if opts.scheduler == "cosine":
                c.lr_scheduler = dict(base_value=user_lr, final_value=user_lr * 0.25, total_nimg=total_nimg)
            elif opts.scheduler == "warmup_cosine":
                c.lr_scheduler = dict(base_value=user_lr, final_value=user_lr * 0.25, total_nimg=total_nimg, warmup_nimg=opts.warmup_steps * c.batch_size)
            elif user_lr != 2e-4:
                c.lr_scheduler = None
            user_beta2 = float(opts.adam_beta2)
            if user_beta2 != 0.0:
                c.beta2_scheduler = None
        c.use_bf16 = opts.use_bf16
        c.negatives_per_group = opts.negatives_per_group
        c.positives_per_group = opts.positives_per_group
        c.unconditional_per_group = opts.unconditional_per_group
        c.alpha_min = opts.alpha_min
        c.alpha_max = opts.alpha_max
        c.drift_temperature = opts.drift_temperature
        c.queue_capacity_per_class = opts.queue_capacity_per_class
        c.queue_capacity_global = opts.queue_capacity_global
        c.queue_push_batch = opts.queue_push_batch
        c.queue_warmup_batches = opts.queue_warmup_batches
        c.clip_grad_norm = float(opts.clip_grad_norm)
        c.drift_config = dnnlib.EasyDict(
            backbone=opts.drift_backbone,
            alpha_fixed=opts.alpha_fixed,
            alpha_min=float(opts.alpha_min),
            alpha_max=float(opts.alpha_max),
            alpha_dist=opts.alpha_dist,
            alpha_power=float(opts.alpha_power),
            alpha_point=float(opts.alpha_point),
            alpha_point_prob=float(opts.alpha_point_prob),
            drift_temperature=float(opts.drift_temperature),
            drift_temperatures=drift_temperatures,
            drift_temperature_reduction=opts.drift_temperature_reduction,
            learning_rate=float(opts.learning_rate),
            adam_beta1=float(opts.adam_beta1),
            adam_beta2=float(opts.adam_beta2),
            weight_decay=float(opts.weight_decay),
            scheduler=opts.scheduler,
            warmup_steps=int(opts.warmup_steps),
            clip_grad_norm=float(opts.clip_grad_norm),
            use_bf16=bool(opts.use_bf16),
            compile_generator=bool(opts.compile_generator),
            compile_backend=str(opts.compile_backend),
            compile_mode=str(opts.compile_mode),
            compile_dynamic=bool(opts.compile_dynamic),
            compile_fullgraph=bool(opts.compile_fullgraph),
            compile_fail_action=str(opts.compile_fail_action),
            use_feature_loss=bool(opts.use_feature_loss),
            feature_encoder=str(opts.feature_encoder),
            convnext_weights=str(opts.convnext_weights),
            convnextv2_weights=str(opts.convnextv2_weights),
            mae_encoder_path=opts.mae_encoder_path,
            mae_encoder_arch=str(opts.mae_encoder_arch),
            mae_input_patchify_size=int(opts.mae_input_patchify_size),
            feature_base_channels=int(opts.feature_base_channels),
            feature_stages=int(opts.feature_stages),
            feature_temperatures=feature_temperatures,
            feature_temperature_aggregation=str(opts.feature_temperature_aggregation),
            feature_loss_term_reduction=str(opts.feature_loss_term_reduction),
            feature_selected_stages=feature_selected_stages,
            include_patch4_stats=bool(opts.include_patch4_stats),
            include_input_x2_mean=bool(opts.include_input_x2_mean),
            disable_shared_location_normalization=bool(opts.disable_shared_location_normalization),
            disable_feature_temperature_sqrt_scaling=bool(opts.disable_feature_temperature_sqrt_scaling),
            feature_include_raw_drift_loss=bool(opts.feature_include_raw_drift_loss),
            feature_raw_drift_loss_weight=float(opts.feature_raw_drift_loss_weight),
            feature_compile_drift_kernel=bool(opts.feature_compile_drift_kernel),
            feature_compile_backend=str(opts.feature_compile_backend),
            feature_compile_mode=str(opts.feature_compile_mode),
            feature_compile_dynamic=bool(opts.feature_compile_dynamic),
            feature_compile_fullgraph=bool(opts.feature_compile_fullgraph),
            feature_compile_fail_action=str(opts.feature_compile_fail_action),
            queue_prime_samples=int(opts.queue_prime_samples),
            queue_warmup_mode=str(opts.queue_warmup_mode),
            queue_warmup_min_per_class=int(opts.queue_warmup_min_per_class),
            queue_strict_without_replacement=bool(opts.queue_strict_without_replacement),
            queue_refill_policy=str(opts.queue_refill_policy),
            queue_refill_every=int(opts.queue_refill_every),
            queue_report_level=str(opts.queue_report_level),
            real_batch_source=str(opts.real_batch_source),
            real_dataset_size=int(opts.real_dataset_size),
            real_loader_batch_size=int(opts.real_loader_batch_size),
            real_num_workers=int(opts.real_num_workers),
            disable_real_shuffle=bool(opts.disable_real_shuffle),
            real_pin_memory=bool(opts.real_pin_memory),
            real_persistent_workers=bool(opts.real_persistent_workers),
            real_prefetch_factor=int(opts.real_prefetch_factor),
            real_sanity_sample_batches=int(opts.real_sanity_sample_batches),
            real_imagefolder_root=opts.real_imagefolder_root,
            real_webdataset_urls=opts.real_webdataset_urls,
            real_tensor_file_path=opts.real_tensor_file_path,
            real_tensor_shards_manifest_path=opts.real_tensor_shards_manifest_path,
            real_transform_resize=opts.real_transform_resize,
            disable_real_center_crop=bool(opts.disable_real_center_crop),
            real_horizontal_flip=bool(opts.real_horizontal_flip),
            real_transform_normalize=bool(opts.real_transform_normalize),
            resume_model_only=bool(opts.resume_model_only),
            resume_reset_scheduler=bool(opts.resume_reset_scheduler),
            resume_reset_optimizer_lr=bool(opts.resume_reset_optimizer_lr),
            allow_resume_config_mismatch=bool(opts.allow_resume_config_mismatch),
            save_every=int(opts.save_every),
            checkpoint_dir=opts.checkpoint_dir,
            keep_last_k_checkpoints=int(opts.keep_last_k_checkpoints),
            eval_every_kimg=resolved_eval_every_kimg,
            eval_reference_imagefolder_root=resolved_eval_reference_imagefolder_root,
            eval_reference_stats_path=resolved_eval_reference_stats_path,
            eval_samples=int(opts.eval_samples),
            eval_sample_batch_size=int(opts.eval_sample_batch_size),
            eval_batch_size=int(opts.eval_batch_size),
            eval_num_workers=int(opts.eval_num_workers),
            eval_inception_weights=str(opts.eval_inception_weights),
            eval_postprocess_mode=str(opts.eval_postprocess_mode),
            eval_alpha=float(opts.eval_alpha),
            patch_size=int(opts.patch_size),
            hidden_dim=int(opts.hidden_dim),
            depth=int(opts.depth),
            num_heads=int(opts.num_heads),
            mlp_ratio=float(opts.mlp_ratio),
            ffn_inner_dim=opts.ffn_inner_dim,
            register_tokens=int(opts.register_tokens),
            style_vocab_size=int(opts.style_vocab_size),
            style_token_count=int(opts.style_token_count),
            alpha_hidden_dim=int(opts.alpha_hidden_dim),
            norm_type=str(opts.norm_type),
            use_qk_norm=bool(opts.use_qk_norm),
            use_rope=bool(opts.use_rope),
            alpha_embedding_type=str(opts.alpha_embedding_type),
            qk_norm_mode=str(opts.qk_norm_mode),
            rope_mode=str(opts.rope_mode),
            disable_patch_positional_embedding=bool(opts.disable_patch_positional_embedding),
            disable_rmsnorm_affine=bool(opts.disable_rmsnorm_affine),
        )

        desc = f"{dataset_name:s}-drift-gpus{c.num_gpus:d}-batch{c.batch_size:d}"
        desc += f"-{opts.drift_backbone}"
        desc += f"-neg{opts.negatives_per_group:d}-pos{opts.positives_per_group:d}"
        desc += f"-alpha{opts.alpha_min:g}to{opts.alpha_max:g}"
        if opts.desc is not None:
            desc += f"-{opts.desc}"

        launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)
        return

    # Description string.
    desc = f"{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}"
    c.loss_kwargs.use_r1_penalty = not opts.disable_r1
    c.loss_kwargs.use_r2_penalty = not opts.disable_r2
    c.loss_kwargs.use_non_aug_gp = opts.non_aug_gp

    using_new_main = any(
        value is not None
        for value in [opts.lambda_pair, opts.pair_margin, opts.lambda_list, opts.list_tau]
    ) or opts.list_loss_type != "infonce"
    if using_new_main:
        lambda_pair = 1.0 if opts.lambda_pair is None else opts.lambda_pair
        pair_margin = 0.0 if opts.pair_margin is None else opts.pair_margin
        lambda_list = 0.0 if opts.lambda_list is None else opts.lambda_list
        list_tau = 0.07 if opts.list_tau is None else opts.list_tau
        if (
            opts.lambda_adv != 1.0
            or opts.adv_loss_type != "softmargin"
            or opts.adv_margin != 0.0
            or opts.adv_tau != 0.07
        ):
            click.echo(
                "NOTE: legacy --adv-* options are ignored because delta-centric pair/list options were provided."
            )
    else:
        lambda_pair = opts.lambda_adv if opts.adv_loss_type == "softmargin" else 0.0
        pair_margin = opts.adv_margin
        lambda_list = opts.lambda_adv if opts.adv_loss_type == "infonce" else 0.0
        list_tau = opts.adv_tau

    if lambda_pair < 0:
        raise click.ClickException("--lambda-pair must be non-negative")
    if pair_margin < 0:
        raise click.ClickException("--pair-margin must be non-negative")
    if lambda_list < 0:
        raise click.ClickException("--lambda-list must be non-negative")
    if list_tau <= 0:
        raise click.ClickException("--list-tau must be positive")

    legacy_rank_used = (
        opts.rank_loss
        or opts.rank_k != 8
        or opts.rank_loss_type != "listmle"
        or opts.lambda_rank != 0.1
        or opts.rank_mode != "intrpl"
        or opts.rank_alpha_dist != "linear"
        or opts.rank_augment
        or opts.rank_margin != 1.0
        or opts.rank_score_reg != 0.0
    )
    using_new_path = (
        opts.path_rank_reg
        or opts.path_rank_k != 8
        or opts.path_rank_loss_type != "listmle"
        or opts.lambda_path_rank != 0.1
        or opts.path_rank_mode != "intrpl"
        or opts.path_rank_alpha_dist != "linear"
        or opts.path_rank_margin != 1.0
        or opts.path_rank_score_reg != 0.0
    )
    if using_new_path:
        path_rank_reg = opts.path_rank_reg
        path_rank_k = opts.path_rank_k
        path_rank_loss_type = opts.path_rank_loss_type
        lambda_path_rank = opts.lambda_path_rank
        path_rank_mode = opts.path_rank_mode
        path_rank_alpha_dist = opts.path_rank_alpha_dist
        path_rank_margin = opts.path_rank_margin
        path_rank_score_reg = opts.path_rank_score_reg
        if legacy_rank_used:
            click.echo(
                "NOTE: legacy --rank-* options are ignored because --path-rank-* options were provided."
            )
    else:
        path_rank_reg = opts.rank_loss
        path_rank_k = opts.rank_k
        path_rank_loss_type = opts.rank_loss_type
        lambda_path_rank = opts.lambda_rank
        path_rank_mode = opts.rank_mode
        path_rank_alpha_dist = opts.rank_alpha_dist
        path_rank_margin = opts.rank_margin
        path_rank_score_reg = opts.rank_score_reg
        if legacy_rank_used:
            click.echo(
                "NOTE: legacy --rank-* options are deprecated; use --path-rank-* instead."
            )
    if opts.rank_augment:
        click.echo(
            "NOTE: --rank-augment is deprecated and ignored; path rank regularization always uses non-augmented views."
        )

    c.loss_kwargs.lambda_pair = lambda_pair
    c.loss_kwargs.pair_margin = pair_margin
    c.loss_kwargs.lambda_list = lambda_list
    c.loss_kwargs.list_loss_type = opts.list_loss_type
    c.loss_kwargs.list_tau = list_tau
    c.loss_kwargs.lambda_local_rank = opts.lambda_local_rank
    c.loss_kwargs.local_rank_k = opts.local_rank_k
    c.loss_kwargs.path_rank_reg = path_rank_reg
    c.loss_kwargs.path_rank_k = path_rank_k
    c.loss_kwargs.path_rank_loss_type = path_rank_loss_type
    c.loss_kwargs.lambda_path_rank = lambda_path_rank
    c.loss_kwargs.path_rank_mode = path_rank_mode
    c.loss_kwargs.path_rank_alpha_dist = path_rank_alpha_dist
    c.loss_kwargs.path_rank_margin = path_rank_margin
    c.loss_kwargs.path_rank_score_reg = path_rank_score_reg
    c.loss_kwargs.coupling_k = opts.coupling_k
    if opts.lambda_list_d is not None:
        c.loss_kwargs.lambda_list_d = opts.lambda_list_d
    if opts.lambda_list_g is not None:
        c.loss_kwargs.lambda_list_g = opts.lambda_list_g

    if opts.disable_r1:
        desc += "-nor1"
    if opts.disable_r2:
        desc += "-nor2"
    if opts.non_aug_gp:
        desc += "-nonauggp"

    if lambda_pair > 0:
        desc += f"-pair{lambda_pair:g}"
        if pair_margin > 0:
            desc += f"m{pair_margin:g}"
    if lambda_list > 0:
        desc += f"-{opts.list_loss_type}-tau{list_tau:g}"
    if opts.coupling_k > 0:
        desc += f'-coupled{opts.coupling_k:d}'
    if opts.lambda_list_d is not None:
        desc += f'-listD{opts.lambda_list_d:g}'
    if opts.lambda_list_g is not None:
        desc += f'-listG{opts.lambda_list_g:g}'
    if opts.lambda_local_rank > 0:
        desc += f"-localrank{opts.lambda_local_rank:g}-k{opts.local_rank_k:d}"
    if path_rank_reg:
        click.echo(
            "NOTE: path rank regularization is a deprecated interpolation-based prior; the main RankGAN game is delta-centric."
        )
        desc += f"-pathrank-{path_rank_loss_type}"
        if path_rank_score_reg > 0:
            desc += f"-scorereg{path_rank_score_reg:g}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
