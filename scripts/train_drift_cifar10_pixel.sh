#!/usr/bin/env bash
# CIFAR-10 pixel-space drift training (~1000 kimg).
# Based on drift_models/configs/pixel/cifar10_queue_1000kimg.yaml.
#
# Prerequisites:
#   - CIFAR-10 dataset at $DATA_PATH (StyleGAN-format .zip)
#     Default: /workspace/datasets/cifar10.zip
#
# Usage:
#   bash scripts/train_drift_cifar10_pixel.sh
#   DATA_PATH=/path/to/cifar10.zip bash scripts/train_drift_cifar10_pixel.sh
#   GPUS=2 BATCH=64 bash scripts/train_drift_cifar10_pixel.sh
set -euo pipefail

# --- Tunable hyperparameters ---
DATA_PATH="${DATA_PATH:-/workspace/datasets/cifar10.zip}"
OUTDIR="${OUTDIR:-outputs/drift/cifar10_pixel}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-32}"
TOTAL_KIMG="${TOTAL_KIMG:-1000}"

# DiT architecture (small, pixel-space)
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

# Queue
QUEUE_PER_CLASS=256
QUEUE_GLOBAL=4000
QUEUE_PUSH=128
QUEUE_WARMUP=4

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
