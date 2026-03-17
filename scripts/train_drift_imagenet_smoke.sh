#!/usr/bin/env bash
# Ultra-fast ImageNet drift smoke test — runs on CPU with synthetic data.
# Purpose: verify the drift pipeline compiles and runs for ImageNet-scale configs.
# Expected runtime: ~2 minutes on CPU.
#
# Usage:
#   bash scripts/train_drift_imagenet_smoke.sh
set -euo pipefail

OUTDIR="${OUTDIR:-outputs/drift_smoke/imagenet}"

python train.py \
    --outdir="$OUTDIR" \
    --data=datasets/imagenet32.zip \
    --cond=1 \
    --preset=ImageNet-32 \
    --trainer=drift \
    --drift-backbone=dit_like \
    --batch=32 \
    --total-kimg=0.05 \
    --kimg-per-tick=0.01 \
    --image-snapshot-ticks=999 \
    --network-snapshot-ticks=999 \
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
    --metrics=none \
    "$@"

echo "Smoke test complete. Output: $OUTDIR"
