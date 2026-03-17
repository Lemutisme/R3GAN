#!/usr/bin/env bash
# ImageNet latent-space drift training — closest feasible single-GPU approximation.
# Based on drift_models Table 8 (B/2 latent) config.
#
# This is the FAITHFUL reproduction target. It uses:
#   - DiT-B/2 architecture (hidden=768, depth=12, heads=12)
#   - RMSNorm + QK-norm + 2D axial RoPE
#   - Feature-space drift loss with MAE encoder
#   - Power-law alpha sampling
#   - Cosine LR schedule with warmup
#
# Prerequisites:
#   - Pre-encoded ImageNet SD-VAE latent shards at $SHARD_MANIFEST
#     Format: directory of .pt shards with a manifest.json index
#   - MAE encoder checkpoint at $MAE_PATH (optional; falls back to tiny encoder)
#   - ImageNet dataset at $DATA_PATH for the R3GAN dataset loader
#
# Usage:
#   bash scripts/train_drift_imagenet_latent.sh
#   MAE_PATH=/path/to/mae_encoder.pt bash scripts/train_drift_imagenet_latent.sh
set -euo pipefail

# --- Tunable hyperparameters ---
DATA_PATH="${DATA_PATH:-datasets/imagenet32.zip}"
SHARD_MANIFEST="${SHARD_MANIFEST:-outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json}"
MAE_PATH="${MAE_PATH:-}"
OUTDIR="${OUTDIR:-outputs/drift/imagenet_latent}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-128}"
TOTAL_KIMG="${TOTAL_KIMG:-25000}"

# DiT-B/2 architecture (latent-space: 4 channels)
HIDDEN_DIM=768
DEPTH=12
NUM_HEADS=12
PATCH_SIZE=2
FFN_INNER_DIM=2184
REGISTER_TOKENS=16

# Drift loss
TEMPERATURE=0.05
ALPHA_MIN=1.0
ALPHA_MAX=4.0

# Grouped batch (Table 8 ratios, scaled for single GPU)
NEG_PER_GROUP=8
POS_PER_GROUP=16
UNC_PER_GROUP=4
GROUPS=16

# Queue
QUEUE_PER_CLASS=128
QUEUE_GLOBAL=16384
QUEUE_PUSH=256
QUEUE_WARMUP=8
QUEUE_PRIME=8192

# Optimizer (paper betas)
LR=4e-4
BETA1=0.9
BETA2=0.95
WARMUP=10000

# Feature encoder selection
if [ -n "$MAE_PATH" ] && [ -f "$MAE_PATH" ]; then
    FEATURE_ENCODER_FLAGS=(
        --feature-encoder=mae
        --mae-encoder-path="$MAE_PATH"
        --mae-encoder-arch=resnet_unet
        --feature-base-channels=64
        --feature-stages=4
    )
    echo "Using MAE feature encoder: $MAE_PATH"
else
    FEATURE_ENCODER_FLAGS=(
        --feature-encoder=tiny
        --feature-base-channels=32
        --feature-stages=3
    )
    echo "MAE encoder not found; using TinyFeatureEncoder fallback."
fi

python train.py \
    --outdir="$OUTDIR" \
    --data="$DATA_PATH" \
    --cond=1 \
    --preset=ImageNet-32 \
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
    --ffn-inner-dim=$FFN_INNER_DIM \
    --register-tokens=$REGISTER_TOKENS \
    --norm-type=rmsnorm \
    --use-qk-norm \
    --use-rope \
    --qk-norm-mode=l2 \
    --rope-mode=2d_axial \
    --disable-patch-positional-embedding \
    --disable-rmsnorm-affine \
    --alpha-embedding-type=mlp \
    \
    --negatives-per-group=$NEG_PER_GROUP \
    --positives-per-group=$POS_PER_GROUP \
    --unconditional-per-group=$UNC_PER_GROUP \
    --alpha-min=$ALPHA_MIN \
    --alpha-max=$ALPHA_MAX \
    --alpha-dist=powerlaw \
    --alpha-power=5.0 \
    --drift-temperature=$TEMPERATURE \
    --drift-temperatures=0.02,0.05,0.2 \
    --drift-temperature-reduction=sum \
    \
    --use-feature-loss \
    "${FEATURE_ENCODER_FLAGS[@]}" \
    --feature-temperatures=0.02,0.05,0.2 \
    --feature-temperature-aggregation=sum_drifts_then_mse \
    --feature-loss-term-reduction=sum \
    --include-input-x2-mean \
    --feature-include-raw-drift-loss \
    --feature-raw-drift-loss-weight=1.0 \
    \
    --queue-capacity-per-class=$QUEUE_PER_CLASS \
    --queue-capacity-global=$QUEUE_GLOBAL \
    --queue-push-batch=$QUEUE_PUSH \
    --queue-warmup-batches=$QUEUE_WARMUP \
    --learning-rate=$LR \
    --adam-beta1=$BETA1 \
    --adam-beta2=$BETA2 \
    --scheduler=warmup_cosine \
    --warmup-steps=$WARMUP \
    --clip-grad-norm=2.0 \
    "$@"

echo "Training complete. Output: $OUTDIR"
