#!/usr/bin/env bash
# ImageNet 32x32 pixel-space drift training.
# Moderate-scale training for quick experiments before committing to latent-space.
#
# Prerequisites:
#   - ImageNet 32x32 dataset at $DATA_PATH (StyleGAN-format .zip)
#     Default: datasets/imagenet32.zip
#
# Usage:
#   bash scripts/train_drift_imagenet_pixel.sh
#   DATA_PATH=/path/to/imagenet32.zip GPUS=4 bash scripts/train_drift_imagenet_pixel.sh
set -euo pipefail

# --- Tunable hyperparameters ---
DATA_PATH="${DATA_PATH:-datasets/imagenet32.zip}"
OUTDIR="${OUTDIR:-outputs/drift/imagenet_pixel}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-64}"
TOTAL_KIMG="${TOTAL_KIMG:-5000}"

# DiT architecture (pixel, 32x32x3)
HIDDEN_DIM=256
DEPTH=6
NUM_HEADS=8
PATCH_SIZE=4
REGISTER_TOKENS=8

# Drift loss
TEMPERATURE=0.05
ALPHA_MIN=1.0
ALPHA_MAX=4.0

# Grouped batch
NEG_PER_GROUP=4
POS_PER_GROUP=4
UNC_PER_GROUP=2

# Queue (larger for 1000-class)
QUEUE_PER_CLASS=64
QUEUE_GLOBAL=8192
QUEUE_PUSH=128
QUEUE_WARMUP=8
QUEUE_PRIME=4000

# Optimizer
LR=2e-4

python train.py \
    --outdir="$OUTDIR" \
    --data="$DATA_PATH" \
    --cond=1 \
    --preset=ImageNet-32 \
    --gpus="$GPUS" \
    --batch="$BATCH" \
    --trainer=drift \
    --drift-backbone=dit_like \
    --kimg="$TOTAL_KIMG" \
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
    --feature-base-channels=16 \
    --feature-stages=3 \
    --feature-temperatures=0.02,0.05,0.2 \
    --include-input-x2-mean \
    \
    --queue-capacity-per-class=$QUEUE_PER_CLASS \
    --queue-capacity-global=$QUEUE_GLOBAL \
    --queue-push-batch=$QUEUE_PUSH \
    --queue-warmup-batches=$QUEUE_WARMUP \
    --learning-rate=$LR \
    "$@"

echo "Training complete. Output: $OUTDIR"
