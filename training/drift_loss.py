# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Drift loss helpers for the prototype drift trainer."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

#----------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftLossConfig:
    temperature: float = 0.05
    normalize_over_x: bool = True
    mask_self_negatives: bool = True
    self_mask_value: float = 1e6
    eps: float = 1e-12


#----------------------------------------------------------------------------


def cfg_alpha_to_unconditional_weight(alpha, n_generated_negatives, n_unconditional_negatives):
    if alpha < 1.0:
        raise ValueError('alpha must be >= 1.0')
    if n_generated_negatives <= 1:
        raise ValueError('n_generated_negatives must be > 1')
    if n_unconditional_negatives <= 0:
        raise ValueError('n_unconditional_negatives must be > 0')
    return ((alpha - 1.0) * (n_generated_negatives - 1)) / n_unconditional_negatives


def build_negative_log_weights(n_generated_negatives, n_unconditional_negatives, unconditional_weight, device, dtype):
    if n_generated_negatives <= 0:
        raise ValueError('n_generated_negatives must be > 0')
    if n_unconditional_negatives < 0:
        raise ValueError('n_unconditional_negatives must be >= 0')
    if unconditional_weight < 0.0:
        raise ValueError('unconditional_weight must be >= 0')

    Generated = torch.zeros([n_generated_negatives], device=device, dtype=dtype)
    if n_unconditional_negatives == 0:
        return Generated

    if unconditional_weight == 0.0:
        Unconditional = torch.full(
            [n_unconditional_negatives],
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
    else:
        Unconditional = torch.full(
            [n_unconditional_negatives],
            torch.log(torch.tensor(unconditional_weight, device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )
    return torch.cat([Generated, Unconditional], dim=0)


#----------------------------------------------------------------------------


def compute_affinity_matrices(x, y_pos, y_neg, config=DriftLossConfig(), negative_log_weights=None, generated_negative_count=None):
    _validate_inputs(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )

    DistPos = torch.cdist(x, y_pos)
    DistNeg = torch.cdist(x, y_neg)

    GeneratedCount = generated_negative_count if generated_negative_count is not None else y_neg.shape[0]
    if config.mask_self_negatives and GeneratedCount > 0:
        DiagonalCount = min(x.shape[0], GeneratedCount, y_neg.shape[0])
        Diagonal = torch.arange(DiagonalCount, device=x.device)
        DistNeg = DistNeg.clone()
        DistNeg[Diagonal, Diagonal] = DistNeg[Diagonal, Diagonal] + config.self_mask_value

    LogitPos = -(DistPos / config.temperature)
    LogitNeg = -(DistNeg / config.temperature)
    if negative_log_weights is not None:
        LogitNeg = LogitNeg + negative_log_weights.view(1, -1)

    Logits = torch.cat([LogitPos, LogitNeg], dim=1)
    RowAffinity = torch.softmax(Logits, dim=-1)
    if config.normalize_over_x:
        ColAffinity = torch.softmax(Logits, dim=-2)
        Affinity = torch.sqrt(torch.clamp(RowAffinity * ColAffinity, min=config.eps))
    else:
        Affinity = RowAffinity

    NumPositives = y_pos.shape[0]
    return Affinity[:, :NumPositives], Affinity[:, NumPositives:]


def compute_drift_components(x, y_pos, y_neg, config=DriftLossConfig(), negative_log_weights=None, generated_negative_count=None):
    AffinityPos, AffinityNeg = compute_affinity_matrices(
        x=x,
        y_pos=y_pos,
        y_neg=y_neg,
        config=config,
        negative_log_weights=negative_log_weights,
        generated_negative_count=generated_negative_count,
    )
    WeightPos = AffinityPos * AffinityNeg.sum(dim=1, keepdim=True)
    WeightNeg = AffinityNeg * AffinityPos.sum(dim=1, keepdim=True)
    DriftPos = WeightPos @ y_pos
    DriftNeg = WeightNeg @ y_neg
    return DriftPos, DriftNeg


def drifting_stopgrad_loss(x, y_pos, y_neg, config=DriftLossConfig(), negative_log_weights=None, generated_negative_count=None):
    with torch.no_grad():
        DriftPos, DriftNeg = compute_drift_components(
            x=x.detach().to(torch.float32),
            y_pos=y_pos.detach().to(torch.float32),
            y_neg=y_neg.detach().to(torch.float32),
            config=config,
            negative_log_weights=None if negative_log_weights is None else negative_log_weights.detach().to(torch.float32),
            generated_negative_count=generated_negative_count,
        )
        Drift = DriftPos - DriftNeg
        Target = x.detach().to(torch.float32) + Drift

    Loss = F.mse_loss(x.to(torch.float32), Target)
    Stats = {
        'loss': float(Loss.detach().item()),
        'drift_norm': float(Drift.norm(dim=-1).mean().item()),
        'drift_pos_norm': float(DriftPos.norm(dim=-1).mean().item()),
        'drift_neg_norm': float(DriftNeg.norm(dim=-1).mean().item()),
    }
    return Loss, Drift, Stats


#----------------------------------------------------------------------------


def grouped_drifting_stopgrad_loss(
    x_grouped,
    y_pos_grouped,
    unconditional_grouped=None,
    unconditional_weight_grouped=None,
    config=DriftLossConfig(),
):
    if x_grouped.ndim != 5:
        raise ValueError(f'x_grouped must be [G, N, C, H, W], got {tuple(x_grouped.shape)}')
    if y_pos_grouped.ndim != 5:
        raise ValueError(f'y_pos_grouped must be [G, P, C, H, W], got {tuple(y_pos_grouped.shape)}')
    if x_grouped.shape[0] != y_pos_grouped.shape[0]:
        raise ValueError('x_grouped and y_pos_grouped must share group dimension')
    if x_grouped.shape[2:] != y_pos_grouped.shape[2:]:
        raise ValueError('x_grouped and y_pos_grouped must share image shape')
    if unconditional_grouped is not None:
        if unconditional_grouped.ndim != 5:
            raise ValueError('unconditional_grouped must be [G, U, C, H, W]')
        if unconditional_grouped.shape[0] != x_grouped.shape[0]:
            raise ValueError('unconditional_grouped must share group dimension')
        if unconditional_grouped.shape[2:] != x_grouped.shape[2:]:
            raise ValueError('unconditional_grouped must share image shape')
    if unconditional_weight_grouped is not None:
        if unconditional_weight_grouped.ndim != 1:
            raise ValueError('unconditional_weight_grouped must be [G]')
        if unconditional_weight_grouped.shape[0] != x_grouped.shape[0]:
            raise ValueError('unconditional_weight_grouped must share group dimension')

    Groups, NegativesPerGroup = x_grouped.shape[:2]
    PositivesPerGroup = y_pos_grouped.shape[1]
    XFlat = x_grouped.to(torch.float32).reshape(Groups, NegativesPerGroup, -1)
    YPosFlat = y_pos_grouped.to(torch.float32).reshape(Groups, PositivesPerGroup, -1)
    YNegFlat = XFlat.detach()
    NegativeLogWeights = None

    if unconditional_grouped is not None:
        UnconditionalFlat = unconditional_grouped.to(torch.float32).reshape(
            Groups, unconditional_grouped.shape[1], -1
        )
        YNegFlat = torch.cat([YNegFlat, UnconditionalFlat], dim=1)
        if unconditional_weight_grouped is None:
            unconditional_weight_grouped = torch.ones([Groups], device=x_grouped.device, dtype=torch.float32)
        NegativeLogWeights = _build_grouped_negative_log_weights(
            n_generated_negatives=NegativesPerGroup,
            n_unconditional_negatives=unconditional_grouped.shape[1],
            unconditional_weight_grouped=unconditional_weight_grouped.to(torch.float32),
            device=x_grouped.device,
            dtype=torch.float32,
        )

    with torch.no_grad():
        DriftPos, DriftNeg = _compute_grouped_drift_components(
            x_grouped=XFlat.detach(),
            y_pos_grouped=YPosFlat.detach(),
            y_neg_grouped=YNegFlat,
            config=config,
            negative_log_weights_grouped=NegativeLogWeights,
            generated_negative_count=NegativesPerGroup,
        )
        Drift = DriftPos - DriftNeg
        Target = XFlat.detach() + Drift

    Loss = F.mse_loss(XFlat, Target)
    Stats = {
        'loss': float(Loss.detach().item()),
        'groups': float(Groups),
        'negatives_per_group': float(NegativesPerGroup),
        'mean_drift_norm': float(Drift.norm(dim=-1).mean().item()),
        'mean_drift_pos_norm': float(DriftPos.norm(dim=-1).mean().item()),
        'mean_drift_neg_norm': float(DriftNeg.norm(dim=-1).mean().item()),
    }
    return Loss, Stats


#----------------------------------------------------------------------------


def _compute_grouped_drift_components(x_grouped, y_pos_grouped, y_neg_grouped, config, negative_log_weights_grouped, generated_negative_count):
    AffinityPos, AffinityNeg = _compute_grouped_affinity_matrices(
        x_grouped=x_grouped,
        y_pos_grouped=y_pos_grouped,
        y_neg_grouped=y_neg_grouped,
        config=config,
        negative_log_weights_grouped=negative_log_weights_grouped,
        generated_negative_count=generated_negative_count,
    )
    WeightPos = AffinityPos * AffinityNeg.sum(dim=-1, keepdim=True)
    WeightNeg = AffinityNeg * AffinityPos.sum(dim=-1, keepdim=True)
    DriftPos = torch.matmul(WeightPos, y_pos_grouped)
    DriftNeg = torch.matmul(WeightNeg, y_neg_grouped)
    return DriftPos, DriftNeg


def _compute_grouped_affinity_matrices(x_grouped, y_pos_grouped, y_neg_grouped, config, negative_log_weights_grouped, generated_negative_count):
    if x_grouped.ndim != 3 or y_pos_grouped.ndim != 3 or y_neg_grouped.ndim != 3:
        raise ValueError('Grouped drift tensors must be [G, N, D]')
    DistPos = torch.cdist(x_grouped, y_pos_grouped)
    DistNeg = torch.cdist(x_grouped, y_neg_grouped)

    if config.mask_self_negatives and generated_negative_count > 0:
        DiagonalCount = min(x_grouped.shape[1], generated_negative_count, y_neg_grouped.shape[1])
        Diagonal = torch.arange(DiagonalCount, device=x_grouped.device)
        DistNeg = DistNeg.clone()
        DistNeg[:, Diagonal, Diagonal] = DistNeg[:, Diagonal, Diagonal] + config.self_mask_value

    LogitPos = -(DistPos / config.temperature)
    LogitNeg = -(DistNeg / config.temperature)
    if negative_log_weights_grouped is not None:
        LogitNeg = LogitNeg + negative_log_weights_grouped.unsqueeze(1)

    Logits = torch.cat([LogitPos, LogitNeg], dim=-1)
    RowAffinity = torch.softmax(Logits, dim=-1)
    if config.normalize_over_x:
        ColAffinity = torch.softmax(Logits, dim=-2)
        Affinity = torch.sqrt(torch.clamp(RowAffinity * ColAffinity, min=config.eps))
    else:
        Affinity = RowAffinity

    NumPositives = y_pos_grouped.shape[1]
    return Affinity[:, :, :NumPositives], Affinity[:, :, NumPositives:]


def _build_grouped_negative_log_weights(n_generated_negatives, n_unconditional_negatives, unconditional_weight_grouped, device, dtype):
    if unconditional_weight_grouped.ndim != 1:
        raise ValueError('unconditional_weight_grouped must be [G]')
    if torch.any(unconditional_weight_grouped < 0):
        raise ValueError('unconditional_weight_grouped must be >= 0')

    Groups = unconditional_weight_grouped.shape[0]
    Generated = torch.zeros([Groups, n_generated_negatives], device=device, dtype=dtype)
    if n_unconditional_negatives == 0:
        return Generated

    UnconditionalWeights = unconditional_weight_grouped.to(device=device, dtype=dtype).unsqueeze(1)
    LogWeights = torch.where(
        UnconditionalWeights == 0,
        torch.full_like(UnconditionalWeights, torch.finfo(dtype).min),
        torch.log(UnconditionalWeights),
    )
    Unconditional = LogWeights.expand(Groups, n_unconditional_negatives)
    return torch.cat([Generated, Unconditional], dim=1)


def _validate_inputs(x, y_pos, y_neg, negative_log_weights, generated_negative_count):
    for Name, Value in [('x', x), ('y_pos', y_pos), ('y_neg', y_neg)]:
        if Value.ndim != 2:
            raise ValueError(f'{Name} must be 2D, got shape {tuple(Value.shape)}')
    if x.shape[1] != y_pos.shape[1] or x.shape[1] != y_neg.shape[1]:
        raise ValueError('x, y_pos, y_neg must share feature dimension')
    if negative_log_weights is not None:
        if negative_log_weights.ndim != 1:
            raise ValueError('negative_log_weights must be 1D')
        if negative_log_weights.shape[0] != y_neg.shape[0]:
            raise ValueError('negative_log_weights size must match y_neg count')
    if generated_negative_count is not None:
        if generated_negative_count < 0:
            raise ValueError('generated_negative_count must be >= 0')
        if generated_negative_count > y_neg.shape[0]:
            raise ValueError('generated_negative_count cannot exceed y_neg count')
