"""
SRG-R3GAN Training Script.

Wraps the standard R3GAN training with SRG modifications:
- DriftGenerator + DriftDiscriminator (rank-conditioned)  
- SRGLoss (rank-aware adversarial + consistency)
- Evaluation at rank=1 (clean, full quality)

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_srg.py --outdir=runs/srg --data=datasets/cifar10.zip --gpus=1 --batch=64 --kimg=10000
"""

import os
import sys
import click
import torch
import dnnlib

# Import from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import main as r3gan_main, init_dataset_kwargs


@click.command()
@click.option('--outdir', type=str, required=True)
@click.option('--data', type=str, required=True)
@click.option('--gpus', type=int, default=1)
@click.option('--batch', type=int, default=64)
@click.option('--kimg', type=int, default=10000)
@click.option('--snap', type=int, default=200)
@click.option('--seed', type=int, default=0)
@click.option('--resume', type=str, default=None)
@click.option('--srg-coarse-res', type=int, default=8)
@click.option('--srg-rank-prob', type=float, default=0.25)
@click.option('--srg-rank-weight', type=float, default=0.25)
@click.option('--srg-prefix-coarse', type=int, default=16)
@click.option('--srg-prefix-clean', type=int, default=64)
@click.option('--srg-consistency', type=float, default=5.0)
@click.option('--srg-consistency-prob', type=float, default=0.25)
@click.option('--desc', type=str, default='srg')
def main(outdir, data, gpus, batch, kimg, snap, seed, resume,
         srg_coarse_res, srg_rank_prob, srg_rank_weight,
         srg_prefix_coarse, srg_prefix_clean,
         srg_consistency, srg_consistency_prob, desc):

    # Build config exactly like R3GAN's CIFAR10 preset but with SRG modifications
    c = dnnlib.EasyDict()

    # --- Generator: DriftGenerator with alpha = rank ---
    WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024]]
    BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
    CardinalityPerStage = [3 * x for x in [32, 32, 32, 32]]
    NoiseDimension = 64

    c.G_kwargs = dnnlib.EasyDict(
        class_name="training.networks.DriftGenerator",
        NoiseDimension=NoiseDimension,
        WidthPerStage=WidthPerStage,
        BlocksPerStage=BlocksPerStage,
        CardinalityPerStage=CardinalityPerStage,
        ExpansionFactor=2,
        FP16Stages=[-1, -2, -3],
        AlphaMin=0.0,
        AlphaMax=1.0,
        EvalAlpha=1.0,  # At eval, generate at rank=1 (clean)
    )

    # --- Discriminator: DriftDiscriminator (rank-aware) ---
    c.D_kwargs = dnnlib.EasyDict(
        class_name="training.networks.DriftDiscriminator",
        WidthPerStage=WidthPerStage,
        BlocksPerStage=BlocksPerStage,
        CardinalityPerStage=CardinalityPerStage,
        ExpansionFactor=2,
        FP16Stages=[2, 1, 0],
    )

    # --- Loss: SRGLoss ---
    c.loss_kwargs = dnnlib.EasyDict(
        class_name="training.srg_loss.SRGLoss",
        lambda_pair=1.0,
        pair_margin=1.0,
        lambda_list=0.0,
        use_r1_penalty=True,
        use_r2_penalty=True,
        srg_enable=True,
        srg_coarse_res=srg_coarse_res,
        srg_rank_prob_coarse=srg_rank_prob,
        srg_rank_weight_coarse=srg_rank_weight,
        srg_prefix_dim_coarse=srg_prefix_coarse,
        srg_prefix_dim_clean=srg_prefix_clean,
        srg_consistency_weight=srg_consistency,
        srg_consistency_prob=srg_consistency_prob,
    )

    # Conditional
    c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
    c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]

    # Optimizers
    c.G_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0.0, 0.0], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0.0, 0.0], eps=1e-8)

    # Data
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=data)
    c.training_set_kwargs.use_labels = True
    c.training_set_kwargs.xflip = True

    # Training config
    c.num_gpus = gpus
    c.batch_size = batch
    c.g_batch_gpu = batch // gpus
    c.d_batch_gpu = batch // gpus
    c.total_kimg = kimg * 1000
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2, num_workers=3)

    # Scheduling (CIFAR10 preset)
    decay_nimg = 2e7
    c.ema_scheduler = {"base_value": 0, "final_value": 5000 * 1000, "total_nimg": decay_nimg}
    c.aug_scheduler = {"base_value": 0, "final_value": 0.55, "total_nimg": decay_nimg}
    c.lr_scheduler = {"base_value": 0.0002, "final_value": 5e-05, "total_nimg": decay_nimg}
    c.gamma_scheduler = {"base_value": 0.05, "final_value": 0.005, "total_nimg": decay_nimg}
    c.beta2_scheduler = {"base_value": 0.9, "final_value": 0.99, "total_nimg": decay_nimg}

    # Resolution and dims
    c.G_kwargs.c_dim = c.training_set_kwargs.get('max_label_size', 10)
    c.D_kwargs.c_dim = c.G_kwargs.c_dim
    c.G_kwargs.img_resolution = c.training_set_kwargs.resolution
    c.D_kwargs.img_resolution = c.training_set_kwargs.resolution

    # Output
    c.run_dir = None  # will be set by training_loop
    c.network_snapshot_ticks = snap

    # Metrics
    c.metrics = ['fid50k_full']

    # Resume
    c.resume_pkl = resume

    # Description
    desc_str = f'{dataset_name}-gpus{gpus}-batch{batch}-{desc}'

    # Launch
    from training import training_loop
    print(f'=== SRG-R3GAN Training ===')
    print(f'  SRG: coarse_res={srg_coarse_res}, rank_prob={srg_rank_prob}, '
          f'prefix={srg_prefix_coarse}/{srg_prefix_clean}, '
          f'consistency={srg_consistency}')

    # Create output directory
    import time
    run_dir = os.path.join(outdir, f'{time.strftime("%Y%m%d")}-{desc_str}')
    os.makedirs(run_dir, exist_ok=True)

    training_loop.training_loop(
        run_dir=run_dir,
        training_set_kwargs=dict(c.training_set_kwargs),
        G_kwargs=dict(c.G_kwargs),
        D_kwargs=dict(c.D_kwargs),
        G_opt_kwargs=dict(c.G_opt_kwargs),
        D_opt_kwargs=dict(c.D_opt_kwargs),
        loss_kwargs=dict(c.loss_kwargs),
        total_kimg=kimg,
        batch_size=batch,
        g_batch_gpu=c.g_batch_gpu,
        d_batch_gpu=c.d_batch_gpu,
        metrics=c.metrics,
        network_snapshot_ticks=snap,
        random_seed=seed,
        data_loader_kwargs=dict(c.data_loader_kwargs),
        ema_scheduler=c.ema_scheduler,
        aug_scheduler=c.aug_scheduler,
        lr_scheduler=c.lr_scheduler,
        gamma_scheduler=c.gamma_scheduler,
        beta2_scheduler=c.beta2_scheduler,
        resume_pkl=resume,
    )


if __name__ == '__main__':
    main()
