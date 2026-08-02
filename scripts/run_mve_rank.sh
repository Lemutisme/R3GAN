#!/bin/bash
set -euo pipefail
cd /workspace/R3GAN

DATA="datasets/cifar10.zip"
OUTDIR="/workspace/training-runs/mve"
KIMG=5000
BATCH=64
SNAP=50

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

echo "=== Rank seed 0 (GPU 2) ==="
run_experiment 2 "rank" 0 \
    --lambda-pair=1.0 --pair-margin=1.0 \
    --path-rank-reg=1 --path-rank-k=3 \
    --path-rank-loss-type=pairwise_hinge \
    --lambda-path-rank=0.1 --path-rank-margin=1.0

echo "=== Rank seed 1 (GPU 2) ==="
run_experiment 2 "rank" 1 \
    --lambda-pair=1.0 --pair-margin=1.0 \
    --path-rank-reg=1 --path-rank-k=3 \
    --path-rank-loss-type=pairwise_hinge \
    --lambda-path-rank=0.1 --path-rank-margin=1.0

echo "=== Rank seed 2 (GPU 2) ==="
run_experiment 2 "rank" 2 \
    --lambda-pair=1.0 --pair-margin=1.0 \
    --path-rank-reg=1 --path-rank-k=3 \
    --path-rank-loss-type=pairwise_hinge \
    --lambda-path-rank=0.1 --path-rank-margin=1.0
