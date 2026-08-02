#!/bin/bash
# MVE (Minimum Viable Experiment) Package for CIFAR-10
# Per GPT-5.4 recommendation: 3-seed matched + controls + lambda sweep
set -euo pipefail
cd /workspace/R3GAN

DATA="datasets/cifar10.zip"
OUTDIR="/workspace/training-runs/mve"
KIMG=5000  # Each run to 5000 kimg (covers best FID window ~1800 kimg)
BATCH=64
SNAP=50

mkdir -p "$OUTDIR"

run_experiment() {
    local GPU=$1
    local DESC=$2
    local SEED=$3
    shift 3
    local EXTRA_ARGS="$@"
    
    echo "[$(date)] Starting: $DESC (seed=$SEED, GPU=$GPU)"
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --outdir="$OUTDIR" \
        --data="$DATA" \
        --gpus=1 \
        --batch=$BATCH \
        --preset=CIFAR10 \
        --cond=1 \
        --mirror=1 \
        --aug=1 \
        --kimg=$KIMG \
        --snap=$SNAP \
        --snapshot-policy=latest-best \
        --metrics=fid50k_full \
        --seed=$SEED \
        --desc="${DESC}_seed${SEED}" \
        $EXTRA_ARGS \
        2>&1 | tee "$OUTDIR/${DESC}_seed${SEED}.log"
    echo "[$(date)] Finished: $DESC (seed=$SEED)"
}

# ═══════════════════════════════════════════════════════════
# Phase 1: 3-seed vanilla vs 3-seed rank (GPU 0, 2, 3)
# Run vanilla seed 0 on GPU 0, rank seed 0 on GPU 2, vanilla seed 1 on GPU 3
# Then rotate
# ═══════════════════════════════════════════════════════════

echo "=== Phase 1: Vanilla seed 0 (GPU 0) ==="
run_experiment 0 "vanilla" 0 \
    --lambda-pair=1.0 --pair-margin=1.0

echo "=== Phase 1: Vanilla seed 1 (GPU 0) ==="
run_experiment 0 "vanilla" 1 \
    --lambda-pair=1.0 --pair-margin=1.0

echo "=== Phase 1: Vanilla seed 2 (GPU 0) ==="
run_experiment 0 "vanilla" 2 \
    --lambda-pair=1.0 --pair-margin=1.0
