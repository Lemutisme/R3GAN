#!/usr/bin/env bash
# Train the 00037 family setting from scratch on CIFAR-10.
#
# This reproduces the same hyperparameter family as runs 00023 -> 00037,
# but starts from random initialization instead of resuming from 00023.
#
# Notes:
# - The original 00037 run resumed from 00023 at ~136k kimg and reached its
#   best FID near 240k kimg. If you want a practical repro window, set:
#     KIMG=300000
# - If you want the exact long-run default used by the repo configs, leave:
#     KIMG=10000000
#
# Usage:
#   bash scripts/train_pathrank_cifar10_from_scratch.sh
#   KIMG=300000 GPUS=2 bash scripts/train_pathrank_cifar10_from_scratch.sh
#
set -euo pipefail

OUTDIR="${OUTDIR:-/workspace/training-runs}"
DATA_PATH="${DATA_PATH:-/workspace/datasets/cifar10.zip}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-256}"
G_BATCH_GPU="${G_BATCH_GPU:-128}"
D_BATCH_GPU="${D_BATCH_GPU:-64}"
KIMG="${KIMG:-10000000}"
TICK="${TICK:-1}"
SNAP="${SNAP:-200}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-3}"
METRICS="${METRICS:-fid50k_full}"
DESC="${DESC:-pair1m1_pathrank_hinge_k3_dbgpu128_fresh}"

cd /workspace/R3GAN

python train.py \
  --outdir="${OUTDIR}" \
  --data="${DATA_PATH}" \
  --gpus="${GPUS}" \
  --batch="${BATCH}" \
  --g-batch-gpu="${G_BATCH_GPU}" \
  --d-batch-gpu="${D_BATCH_GPU}" \
  --cond=1 \
  --mirror=1 \
  --aug=1 \
  --preset=CIFAR10 \
  --kimg="${KIMG}" \
  --tick="${TICK}" \
  --snap="${SNAP}" \
  --snapshot-policy=latest-best \
  --metrics="${METRICS}" \
  --workers="${WORKERS}" \
  --seed="${SEED}" \
  --lambda-pair=1.0 \
  --pair-margin=1.0 \
  --lambda-list=0.0 \
  --list-loss-type=infonce \
  --list-tau=0.07 \
  --lambda-local-rank=0.0 \
  --local-rank-k=4 \
  --coupling-k=0 \
  --path-rank-reg=1 \
  --path-rank-k=3 \
  --path-rank-loss-type=pairwise_hinge \
  --lambda-path-rank=0.1 \
  --path-rank-mode=intrpl \
  --path-rank-alpha-dist=linear \
  --path-rank-margin=1.0 \
  --path-rank-score-reg=0.0 \
  --non-aug-gp=0 \
  --desc="${DESC}" \
  "$@"

