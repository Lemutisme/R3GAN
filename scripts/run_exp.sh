#!/usr/bin/env bash
# ===========================================================================
#  R3GAN / Local-Coupled RankGAN — Experiment Runner
#
#  Usage:
#    bash scripts/run_exp.sh <EXPERIMENT_NAME> [EXTRA_FLAGS...]
#
#  Examples:
#    bash scripts/run_exp.sh baseline_r3gan
#    bash scripts/run_exp.sh rankgan_coupled
#    bash scripts/run_exp.sh rankgan_full --gpus=4 --batch=256
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
#  Shared defaults (override via environment or CLI)
# ---------------------------------------------------------------------------
OUTDIR="${OUTDIR:-runs}"
DATA="${DATA:-datasets/cifar10.zip}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-64}"
PRESET="${PRESET:-CIFAR10}"
MIRROR="${MIRROR:-1}"

# ---------------------------------------------------------------------------
#  Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENT="${1:-help}"
shift || true   # remaining args appended to the command

run_train() {
    echo "============================================================"
    echo "  Experiment : ${EXPERIMENT}"
    echo "  Data       : ${DATA}"
    echo "  GPUs       : ${GPUS}   Batch : ${BATCH}"
    echo "  Preset     : ${PRESET}"
    echo "============================================================"
    echo ""
    echo "  python train.py $*"
    echo ""
    python train.py "$@"
}

case "${EXPERIMENT}" in

# ===== 1. BASELINES =======================================================

baseline_r3gan)
    # -----------------------------------------------------------------------
    #  Original R3GAN: paired pairwise delta + R1+R2.
    #  This is the paper's default — no listwise, no coupling.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --lambda-pair=1.0 --pair-margin=0.0 \
        "$@"
    ;;

baseline_r3gan_margin)
    # -----------------------------------------------------------------------
    #  R3GAN with soft-margin pairwise.
    #  Margin > 0 penalizes small score gaps even if ordering is correct.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --lambda-pair=1.0 --pair-margin=1.0 \
        "$@"
    ;;

baseline_infonce)
    # -----------------------------------------------------------------------
    #  R3GAN with global all-pairs InfoNCE (no coupling).
    #  Full-batch listwise: logsumexp sees all negatives.
    #  Note: needs full batch in memory (no micro-batch savings).
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --lambda-pair=0.0 --lambda-list=1.0 --list-tau=0.07 \
        "$@"
    ;;

baseline_pair_plus_infonce)
    # -----------------------------------------------------------------------
    #  R3GAN with pairwise + global InfoNCE.
    #  Combines diagonal pairing with full all-pairs contrastive.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --lambda-pair=1.0 --lambda-list=1.0 --list-tau=0.07 \
        "$@"
    ;;

# ===== 2. LOCAL-COUPLED RANKGAN (NEW) =====================================

rankgan_pairwise_only)
    # -----------------------------------------------------------------------
    #  Local-coupled pairwise ONLY (no listwise).
    #  Each fake compared to k=4 nearest reals via clean-feature kNN.
    #  Coupling weights = softmax(cosine similarities).
    #  This is the Phase 1 (warmup) config, kept permanently.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.0 --lambda-list-g=0.0 \
        "$@"
    ;;

rankgan_d_list_only)
    # -----------------------------------------------------------------------
    #  Local-coupled pairwise + D-side listwise (no G-side list).
    #  D gets local-listwise for better critic geometry.
    #  G only sees coupled-pairwise gradients.
    #  This is the Phase 2 config, kept permanently.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.5 --lambda-list-g=0.0 \
        --list-tau=0.07 \
        "$@"
    ;;

rankgan_coupled)
    # -----------------------------------------------------------------------
    #  ** Canonical RankGAN **
    #  Local-coupled pairwise + asymmetric D/G listwise.
    #  With 3-phase curriculum (auto):
    #    Phase 1 (0-25%):  pairwise warmup only
    #    Phase 2 (25-50%): D-list ramps 0 → 0.5
    #    Phase 3 (50%+):   G-list ramps 0 → 0.1; both at target
    #
    #  lambda_list_g < lambda_list_d by design — listwise on G
    #  only reshuffles which fakes update more, doesn't create
    #  new directions.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.5 --lambda-list-g=0.1 \
        --list-tau=0.07 \
        "$@"
    ;;

rankgan_aggressive)
    # -----------------------------------------------------------------------
    #  Aggressive variant: higher listwise weights, more neighbors.
    #  Useful for ablation to test sensitivity.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=8 \
        --lambda-pair=1.0 --pair-margin=0.5 \
        --lambda-list-d=1.0 --lambda-list-g=0.3 \
        --list-tau=0.05 \
        "$@"
    ;;

# ===== 3. ABLATIONS =======================================================

ablation_coupling_k)
    # -----------------------------------------------------------------------
    #  Sweep coupling_k = {2, 4, 8, 16} with fixed listwise weights.
    #  Tests sensitivity to neighborhood size.
    # -----------------------------------------------------------------------
    for K in 2 4 8 16; do
        echo ">>> coupling_k=${K}"
        run_train \
            --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
            --preset="${PRESET}" --mirror="${MIRROR}" \
            --coupling-k="${K}" \
            --lambda-pair=1.0 --lambda-list-d=0.5 --lambda-list-g=0.1 \
            --list-tau=0.07 \
            --desc="ablation-k${K}" \
            "$@"
    done
    ;;

ablation_list_tau)
    # -----------------------------------------------------------------------
    #  Sweep list_tau = {0.03, 0.05, 0.07, 0.1, 0.2}.
    #  Lower tau → sharper contrastive, higher tau → smoother.
    # -----------------------------------------------------------------------
    for TAU in 0.03 0.05 0.07 0.1 0.2; do
        echo ">>> list_tau=${TAU}"
        run_train \
            --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
            --preset="${PRESET}" --mirror="${MIRROR}" \
            --coupling-k=4 \
            --lambda-pair=1.0 --lambda-list-d=0.5 --lambda-list-g=0.1 \
            --list-tau="${TAU}" \
            --desc="ablation-tau${TAU}" \
            "$@"
    done
    ;;

ablation_no_coupling_vs_coupling)
    # -----------------------------------------------------------------------
    #  Head-to-head: global InfoNCE vs local-coupled.
    #  Same total listwise weight, only coupling differs.
    # -----------------------------------------------------------------------
    echo ">>> Global InfoNCE (no coupling)"
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --lambda-pair=1.0 --lambda-list=0.5 --list-tau=0.07 \
        --desc="ablation-global-infonce" \
        "$@"

    echo ">>> Local-coupled (k=4)"
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --lambda-list-d=0.5 --lambda-list-g=0.1 \
        --list-tau=0.07 \
        --desc="ablation-local-coupled-k4" \
        "$@"
    ;;

ablation_asymmetric_g_weight)
    # -----------------------------------------------------------------------
    #  Sweep lambda_list_g = {0.0, 0.05, 0.1, 0.2, 0.5} with fixed D.
    #  Tests the design claim that lambda_list_g <= lambda_list_d.
    # -----------------------------------------------------------------------
    for LG in 0.0 0.05 0.1 0.2 0.5; do
        echo ">>> lambda_list_g=${LG}"
        run_train \
            --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
            --preset="${PRESET}" --mirror="${MIRROR}" \
            --coupling-k=4 \
            --lambda-pair=1.0 --lambda-list-d=0.5 --lambda-list-g="${LG}" \
            --list-tau=0.07 \
            --desc="ablation-listG${LG}" \
            "$@"
    done
    ;;

# ===== 4. AUXILIARY LOSSES (OPTIONAL) ======================================

rankgan_with_local_rank)
    # -----------------------------------------------------------------------
    #  Canonical RankGAN + semantic-local gap-rank prior.
    #  Stacks the local-rank prior (fakes near same real should have
    #  monotonic scores) on top of the coupled game.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.5 --lambda-list-g=0.1 \
        --list-tau=0.07 \
        --lambda-local-rank=0.1 --local-rank-k=4 \
        "$@"
    ;;

rankgan_with_path_rank)
    # -----------------------------------------------------------------------
    #  Canonical RankGAN + chord-monotonic path prior (experimental).
    #  Interpolation-based: teaches D that scores increase along
    #  the real-fake chord.  Best used as transient early/mid aid.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.5 --lambda-list-g=0.1 \
        --list-tau=0.07 \
        --path-rank-reg=1 --path-rank-k=3 --lambda-path-rank=0.1 \
        --path-rank-loss-type=listmle \
        "$@"
    ;;

# ===== 5. MULTI-GPU / LARGE SCALE =========================================

rankgan_ffhq256)
    # -----------------------------------------------------------------------
    #  Local-coupled RankGAN on FFHQ 256x256.
    #  Recommended: 4+ GPUs, batch=32-64.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset=FFHQ-256 --mirror="${MIRROR}" \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.5 --lambda-list-g=0.1 \
        --list-tau=0.07 \
        "$@"
    ;;

rankgan_imagenet32)
    # -----------------------------------------------------------------------
    #  Local-coupled RankGAN on ImageNet 32x32 (conditional).
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset=ImageNet-32 --mirror=0 --cond=1 \
        --coupling-k=4 \
        --lambda-pair=1.0 --pair-margin=0.0 \
        --lambda-list-d=0.5 --lambda-list-g=0.1 \
        --list-tau=0.07 \
        "$@"
    ;;

# ===== 6. LEGACY COMPATIBILITY =============================================

legacy_rpgan)
    # -----------------------------------------------------------------------
    #  Legacy RpGAN interface (backward compat).
    #  Maps --adv-loss-type=softmargin to lambda_pair=1.0.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --adv-loss-type=softmargin --adv-margin=0.0 \
        "$@"
    ;;

legacy_infonce)
    # -----------------------------------------------------------------------
    #  Legacy InfoNCE interface (backward compat).
    #  Maps --adv-loss-type=infonce to lambda_list=1.0.
    # -----------------------------------------------------------------------
    run_train \
        --outdir="${OUTDIR}" --data="${DATA}" --gpus="${GPUS}" --batch="${BATCH}" \
        --preset="${PRESET}" --mirror="${MIRROR}" \
        --adv-loss-type=infonce --adv-tau=0.07 \
        "$@"
    ;;

# ===== HELP ================================================================

help|*)
    cat <<'USAGE'
R3GAN / Local-Coupled RankGAN Experiment Runner
================================================

Usage:
    bash scripts/run_exp.sh <EXPERIMENT> [EXTRA_FLAGS...]

Environment variables (override defaults):
    OUTDIR   Output directory      [default: runs]
    DATA     Dataset path          [default: datasets/cifar10.zip]
    GPUS     Number of GPUs        [default: 1]
    BATCH    Total batch size      [default: 64]
    PRESET   Config preset         [default: CIFAR10]
    MIRROR   Dataset x-flips       [default: 1]

Available experiments:

  BASELINES
    baseline_r3gan               Original R3GAN (pairwise + R1+R2)
    baseline_r3gan_margin        R3GAN with soft-margin=1.0
    baseline_infonce             Global all-pairs InfoNCE only
    baseline_pair_plus_infonce   Pairwise + global InfoNCE

  LOCAL-COUPLED RANKGAN
    rankgan_pairwise_only        Coupled pairwise, no listwise (Phase 1)
    rankgan_d_list_only          + D-side listwise (Phase 2)
    rankgan_coupled              ** Canonical: coupled pair + asymmetric list **
    rankgan_aggressive           Higher weights, more neighbors

  ABLATIONS
    ablation_coupling_k          Sweep k = {2, 4, 8, 16}
    ablation_list_tau            Sweep tau = {0.03, 0.05, 0.07, 0.1, 0.2}
    ablation_no_coupling_vs_coupling   Global vs local-coupled head-to-head
    ablation_asymmetric_g_weight Sweep lambda_list_g = {0, 0.05, 0.1, 0.2, 0.5}

  AUXILIARY LOSSES
    rankgan_with_local_rank      + semantic gap-rank prior
    rankgan_with_path_rank       + chord-monotonic path prior (experimental)

  LARGE SCALE
    rankgan_ffhq256              FFHQ 256x256 (set DATA=...)
    rankgan_imagenet32           ImageNet 32x32 conditional (set DATA=...)

  LEGACY
    legacy_rpgan                 Old --adv-loss-type=softmargin interface
    legacy_infonce               Old --adv-loss-type=infonce interface

Examples:
    # Quick CIFAR10 run with canonical RankGAN
    bash scripts/run_exp.sh rankgan_coupled

    # Multi-GPU with custom batch size
    GPUS=4 BATCH=256 bash scripts/run_exp.sh rankgan_coupled

    # FFHQ-256 on 8 GPUs
    DATA=datasets/ffhq256.zip GPUS=8 BATCH=64 bash scripts/run_exp.sh rankgan_ffhq256

    # Dry run (print config, don't train)
    bash scripts/run_exp.sh rankgan_coupled --dry-run

    # Resume from checkpoint
    bash scripts/run_exp.sh rankgan_coupled --resume=runs/00001-xxx/best.pkl

    # Custom metrics
    bash scripts/run_exp.sh rankgan_coupled --metrics=fid50k_full,kid50k_full
USAGE
    ;;

esac
