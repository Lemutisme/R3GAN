#!/usr/bin/env bash
# CIFAR-10 pixel-space drift training with larger DiT model.
# Based on drift_models/configs/pixel/cifar10_queue_5090_100000kimg.yaml.
# Designed for multi-GPU RTX 5090 training with higher GPU utilization.
#
# Prerequisites:
#   - CIFAR-10 dataset at $DATA_PATH (StyleGAN-format .zip)
#     Default: /workspace/datasets/cifar10.zip
#
# Usage:
#   GPUS=2 BATCH=512 bash scripts/train_drift_cifar10_pixel_large.sh
#   CUDA_VISIBLE_DEVICES=2,3 GPUS=2 BATCH=512 bash scripts/train_drift_cifar10_pixel_large.sh
set -euo pipefail

# --- Tunable hyperparameters ---
DATA_PATH="${DATA_PATH:-/workspace/datasets/cifar10.zip}"
OUTDIR="${OUTDIR:-outputs/drift/cifar10_pixel_large}"
GPUS="${GPUS:-2}"
BATCH="${BATCH:-512}"
TOTAL_KIMG="${TOTAL_KIMG:-10000}"

# DiT architecture (larger, pixel-space — matches 5090 config)
HIDDEN_DIM=512
DEPTH=12
NUM_HEADS=16
PATCH_SIZE=4
REGISTER_TOKENS=16

# Drift loss
TEMPERATURE=0.05
ALPHA_MIN=1.0
ALPHA_MAX=4.0

# Grouped batch (larger groups for higher utilization)
NEG_PER_GROUP=8
POS_PER_GROUP=8
UNC_PER_GROUP=4

# Queue (larger capacities)
QUEUE_PER_CLASS=2048
QUEUE_GLOBAL=32768
QUEUE_PUSH=2048
QUEUE_WARMUP=16

# Optimizer
LR=1e-4

python train.py \
    --outdir="$OUTDIR" \
    --data="$DATA_PATH" \
    --cond=1 \
    --preset=CIFAR10 \
    --mirror=True \
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
    --scheduler=cosine \
    "$@"

echo "Training complete. Output: $OUTDIR"
