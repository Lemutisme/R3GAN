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

# Lambda sweep (seed 0 only)
for LAM in 0.03 0.3; do
    echo "=== Lambda sweep: λ=${LAM} (GPU 3) ==="
    run_experiment 3 "rank_lam${LAM}" 0 \
        --lambda-pair=1.0 --pair-margin=1.0 \
        --path-rank-reg=1 --path-rank-k=3 \
        --path-rank-loss-type=pairwise_hinge \
        --lambda-path-rank=${LAM} --path-rank-margin=1.0
done
