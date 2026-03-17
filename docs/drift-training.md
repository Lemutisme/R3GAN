# Drift Training Guide

This document covers how to run drift-based generative training in R3GAN.

## Overview

Drift training implements "Generative Modeling via Drifting" — a one-step generative model that moves iteration from inference time to training time. Instead of iterative sampling (like diffusion), a generator learns to produce samples in a single forward pass by following a drifting field during training.

### Two Lanes

| Lane | Space | Generator Input | Feature Loss | Use Case |
|------|-------|----------------|-------------|----------|
| **Pixel** | Raw pixels (C×H×W) | Image-shaped noise | TinyFeatureEncoder | Quick experiments, validation |
| **Latent** | VAE latents (4×32×32) | Latent-shaped noise | MAE or Tiny encoder | Faithful reproduction, best FID |

Both lanes use the DiT-like generator (`--drift-backbone=dit_like`) by default.

## Quick Start

### Smoke Tests (no data needed)

```bash
# CIFAR-10 — ~1 min on CPU
bash scripts/train_drift_cifar10_smoke.sh

# ImageNet — ~2 min on CPU
bash scripts/train_drift_imagenet_smoke.sh
```

### CIFAR-10

```bash
# Pixel-space (~1000 kimg, single GPU)
bash scripts/train_drift_cifar10_pixel.sh

# Latent-space (requires pre-encoded SD-VAE latents)
LATENT_PATH=/path/to/cifar10_latents.pt bash scripts/train_drift_cifar10_latent.sh
```

### ImageNet

```bash
# Pixel-space 32x32 (~5000 kimg)
bash scripts/train_drift_imagenet_pixel.sh

# Latent-space — faithful Table 8 approximation
bash scripts/train_drift_imagenet_latent.sh
```

## Prerequisites

### Data Preparation

**CIFAR-10 pixel:** Use the StyleGAN-format zip:
```bash
python dataset_tool.py --source=path/to/cifar10 --dest=datasets/cifar10.zip
```

**ImageNet 32x32 pixel:** Use the StyleGAN-format zip:
```bash
python dataset_tool.py --source=path/to/imagenet --dest=datasets/imagenet32.zip --resolution=32x32
```

**Latent-space (SD-VAE):** Encode images through Stable Diffusion's VAE encoder to produce `[N, 4, 32, 32]` latent tensors. Save as `.pt` files.

### Feature Encoders

The **TinyFeatureEncoder** (default) requires no pretrained weights — it's a small CNN trained end-to-end.

For the faithful ImageNet latent lane, an **MAE encoder** is recommended. Set `MAE_PATH` when running the latent script:
```bash
MAE_PATH=/path/to/mae_encoder.pt bash scripts/train_drift_imagenet_latent.sh
```

## Architecture

### Training Flow

```
noise → DiTLikeGenerator → generated images
                                    ↓
                           [reshape to groups]
                                    ↓
           ┌─── positives (from queue, same class)
           │
           ├─── unconditional negatives (from queue, any class)
           │
           └─── generated negatives (self, with masking)
                                    ↓
                        compute drift field V
                                    ↓
                    target = stopgrad(x + V)
                                    ↓
                       loss = MSE(x, target)
```

### Key Components

| Module | File | Purpose |
|--------|------|---------|
| `DriftFieldConfig` | `training/drift_field.py` | Core drift math (affinity, v-field) |
| `DriftingLossConfig` | `training/drift_loss.py` | Loss with multi-temp + feature space |
| `GroupedDriftStepConfig` | `training/drift_stage2.py` | Training step orchestration |
| `DiTLikeGenerator` | `training/models/dit_like.py` | Transformer generator |
| `TinyFeatureEncoder` | `training/features/extractors.py` | Lightweight feature extractor |
| `FeatureVectorizationConfig` | `training/features/vectorize.py` | Feature map → vectors |
| `ClassConditionalSampleQueue` | `training/drift_queue.py` | Per-class + global sample queue |

### Config Recipes

All scripts expose key hyperparameters as environment variables:

```bash
# Override any default
BATCH=64 GPUS=2 TOTAL_KIMG=2000 bash scripts/train_drift_cifar10_pixel.sh
```

Pass extra flags via positional args:
```bash
bash scripts/train_drift_cifar10_pixel.sh --snap=10 --metrics=fid50k_full
```

## Hyperparameter Reference

### Grouped Batch

| Flag | Default | Description |
|------|---------|-------------|
| `--negatives-per-group` | 4 | Generated samples per group |
| `--positives-per-group` | 4 | Real same-class samples per group |
| `--unconditional-per-group` | 2 | Real any-class samples per group |

### Alpha (CFG)

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha-min` | 1.0 | Minimum alpha |
| `--alpha-max` | 4.0 | Maximum alpha |
| `--alpha-dist` | uniform | Sampling distribution (uniform, powerlaw) |
| `--alpha-power` | 3.0 | Power-law exponent (when dist=powerlaw) |

### Feature Loss

| Flag | Default | Description |
|------|---------|-------------|
| `--use-feature-loss` | off | Enable feature-space drift |
| `--feature-encoder` | tiny | Encoder type (tiny, mae) |
| `--feature-temperatures` | 0.02,0.05,0.2 | Multi-temperature for features |
| `--feature-include-raw-drift-loss` | off | Add raw drift term to feature loss |

## Checkpoints

Checkpoints include:
- `G`, `G_ema` — generator and EMA weights
- `queue_state` — full queue contents for exact resume
- `G_opt_state` — optimizer state
- `provenance` — training config, git commit, backbone type

Resume training:
```bash
python train.py --trainer=drift ... --resume=path/to/network-snapshot-XXXX.pkl
```

## Provenance

Each checkpoint includes a `provenance` dict with:
- `drift_temperature`, `alpha_min`, `alpha_max`
- `backbone`, `use_feature_loss`, `feature_encoder`
- `queue_capacity_per_class`, `queue_capacity_global`
- `git_commit` (if available)

This allows identifying exactly which config produced a given checkpoint.
