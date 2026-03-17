#!/usr/bin/env bash
# CIFAR-10 latent-space drift training (SD-VAE encoded).
# Based on drift_models/configs/latent/cifar10_sdvae_latents_queue_smoke_mae.yaml.
#
# Prerequisites:
#   - Pre-encoded CIFAR-10 SD-VAE latents at $LATENT_PATH
#     These are [N, 4, 32, 32] tensors saved as a .pt file.
#     To create: encode CIFAR-10 images (upscaled to 256x256) through SD-VAE encoder,
#     then save the [N, 4, 32, 32] latents.
#   - CIFAR-10 dataset at $DATA_PATH for the R3GAN dataset loader
#
# Usage:
#   LATENT_PATH=/path/to/cifar10_latents.pt bash scripts/train_drift_cifar10_latent.sh
set -euo pipefail

# --- Tunable hyperparameters ---
DATA_PATH="${DATA_PATH:-datasets/cifar10.zip}"
LATENT_PATH="${LATENT_PATH:-outputs/datasets/cifar10_sdvae_latents.pt}"
OUTDIR="${OUTDIR:-outputs/drift/cifar10_latent}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-32}"
TOTAL_KIMG="${TOTAL_KIMG:-500}"

# DiT architecture (latent-space: 4 channels, patch_size=2)
HIDDEN_DIM=256
DEPTH=6
NUM_HEADS=8
PATCH_SIZE=2
REGISTER_TOKENS=16

# Drift loss
TEMPERATURE=0.05
ALPHA_MIN=1.0
ALPHA_MAX=4.0

# Grouped batch
NEG_PER_GROUP=4
POS_PER_GROUP=4
UNC_PER_GROUP=2

# Queue
QUEUE_PER_CLASS=128
QUEUE_GLOBAL=2048
QUEUE_PUSH=256
QUEUE_WARMUP=4
QUEUE_PRIME=512

# Optimizer
LR=1e-4

python train.py \
    --outdir="$OUTDIR" \
    --data="$DATA_PATH" \
    --cond=1 \
    --preset=CIFAR10 \
    --gpus="$GPUS" \
    --batch="$BATCH" \
    --trainer=drift \
    --drift-backbone=dit_like \
    --total-kimg="$TOTAL_KIMG" \
    \
    --hidden-dim=$HIDDEN_DIM \
    --depth=$DEPTH \
    --num-heads=$NUM_HEADS \
    --patch-size=$PATCH_SIZE \
    --register-tokens=$REGISTER_TOKENS \
    \
    --negatives-per-group=$NEG_PER_GROUP \
    --positives-per-group=$POS_PER_GROUP \
    --unconditional-per-group=$UNC_PER_GROUP \
    --alpha-min=$ALPHA_MIN \
    --alpha-max=$ALPHA_MAX \
    --drift-temperature=$TEMPERATURE \
    \
    --use-feature-loss \
    --feature-encoder=tiny \
    --feature-base-channels=32 \
    --feature-stages=3 \
    --feature-temperatures=0.02,0.05,0.2 \
    --feature-temperature-aggregation=sum_drifts_then_mse \
    --feature-loss-term-reduction=sum \
    --include-input-x2-mean \
    \
    --queue-capacity-per-class=$QUEUE_PER_CLASS \
    --queue-capacity-global=$QUEUE_GLOBAL \
    --queue-push-batch=$QUEUE_PUSH \
    --queue-warmup-batches=$QUEUE_WARMUP \
    --learning-rate=$LR \
    "$@"

echo "Training complete. Output: $OUTDIR"
