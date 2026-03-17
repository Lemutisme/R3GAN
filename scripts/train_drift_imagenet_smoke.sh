#!/usr/bin/env bash
# Ultra-fast ImageNet drift smoke test.
# Purpose: verify the drift pipeline compiles and runs for ImageNet-scale configs.
# Expected runtime: ~2 minutes on GPU, ~10 minutes on CPU.
#
# Requires an ImageNet-32 zip. If unavailable, use CIFAR-10 zip as a stand-in
# (the architecture runs regardless of the actual dataset content):
#   DATA_PATH=/workspace/datasets/cifar10.zip PRESET=CIFAR10 bash scripts/train_drift_imagenet_smoke.sh
set -euo pipefail

OUTDIR="${OUTDIR:-outputs/drift_smoke/imagenet}"
DATA_PATH="${DATA_PATH:-/workspace/datasets/cifar10.zip}"
PRESET="${PRESET:-CIFAR10}"

python train.py \
    --outdir="$OUTDIR" \
    --data="$DATA_PATH" \
    --cond=1 \
    --preset="$PRESET" \
    --trainer=drift \
    --drift-backbone=dit_like \
    --gpus=1 \
    --batch=32 \
    --kimg=1 \
    --tick=1 \
    --snap=999 \
    \
    --hidden-dim=64 \
    --depth=2 \
    --num-heads=4 \
    --patch-size=4 \
    --register-tokens=4 \
    --alpha-hidden-dim=32 \
    \
    --negatives-per-group=4 \
    --positives-per-group=4 \
    --unconditional-per-group=2 \
    --alpha-min=1.0 \
    --alpha-max=4.0 \
    --drift-temperature=0.05 \
    \
    --queue-capacity-per-class=16 \
    --queue-capacity-global=320 \
    --queue-push-batch=32 \
    --queue-warmup-batches=2 \
    --learning-rate=1e-4 \
    --metrics none \
    "$@"

echo "Smoke test complete. Output: $OUTDIR"
