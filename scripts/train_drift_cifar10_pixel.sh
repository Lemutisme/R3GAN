#!/usr/bin/env bash
# CIFAR-10 drift training with feature-space loss.
# Based on drift_models/configs/pixel/cifar10_queue_1000kimg.yaml.
#
# Usage:
#   bash scripts/train_drift_cifar10_pixel.sh
#   CUDA_VISIBLE_DEVICES=3 bash scripts/train_drift_cifar10_pixel.sh
#   GPUS=2 BATCH=512 bash scripts/train_drift_cifar10_pixel.sh
set -euo pipefail

# --- Tunable hyperparameters ---
DATA_PATH="${DATA_PATH:-/workspace/datasets/cifar10.zip}"
OUTDIR="${OUTDIR:-outputs/drift/cifar10_pixel}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-512}"
TOTAL_KIMG="${TOTAL_KIMG:-100000}"

# DiT architecture — scaled to match R3GAN G capacity (~21.6M params)
HIDDEN_DIM=384
DEPTH=8
NUM_HEADS=8
PATCH_SIZE=4
REGISTER_TOKENS=12
ALPHA_HIDDEN_DIM=128

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

# Monitoring
TICK=50          # print every 50 kimg
SNAP=5           # snapshot + FID every 5 ticks = 250 kimg

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
    --tick=$TICK \
    --snap=$SNAP \
    --metrics=fid50k_full \
    \
    --hidden-dim=$HIDDEN_DIM \
    --depth=$DEPTH \
    --num-heads=$NUM_HEADS \
    --patch-size=$PATCH_SIZE \
    --register-tokens=$REGISTER_TOKENS \
    --alpha-hidden-dim=$ALPHA_HIDDEN_DIM \
    --use-qk-norm \
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
