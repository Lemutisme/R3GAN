#!/bin/bash
# Train SRG-R3GAN on CIFAR-10
# Uses DriftGenerator + DriftDiscriminator with SRG rank structure
#
# Key difference from vanilla R3GAN:
# - Alpha is used as RANK indicator (0.0 = coarse, 1.0 = clean)
# - Generator sees masked latent at coarse rank
# - Discriminator sees blurred real images at coarse rank
# - Consistency loss: T_0(G(z, rank=1)) ≈ G(z_masked, rank=0)

set -euo pipefail
cd /workspace/R3GAN

OUTDIR="${OUTDIR:-/workspace/training-runs/srg}"
DATA="${DATA:-datasets/cifar10.zip}"
GPU="${GPU:-0}"
KIMG="${KIMG:-10000}"
BATCH="${BATCH:-64}"
SEED="${SEED:-0}"
DESC="${DESC:-srg_k2_res8_fresh}"

# Use DriftGenerator (already supports alpha conditioning) 
# and DriftDiscriminator (newly added, alpha-aware D)
# AlphaMin=0, AlphaMax=1: rank ∈ [0, 1]
# EvalAlpha=1: at eval, generate at clean rank

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
  --tick=1 \
  --snap=200 \
  --snapshot-policy=latest-best \
  --metrics=fid50k_full \
  --seed=$SEED \
  --trainer=drift \
  --lambda-pair=1.0 \
  --pair-margin=1.0 \
  --lambda-list=0.0 \
  --desc="$DESC" \
  "$@"
