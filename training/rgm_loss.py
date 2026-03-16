"""Rank-aware drifting losses for the prototype RGM trainer."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from training.drift_loss import DriftLossConfig, grouped_drifting_stopgrad_loss


@dataclass(frozen=True)
class RGMLossConfig:
    temperature: float = 0.05
    order_margin: float = 0.0
    lambda_transport: float = 1.0
    lambda_order: float = 0.5
    lambda_eq: float = 0.25
    normalize_over_x: bool = True
    mask_self_negatives: bool = True
    self_mask_value: float = 1e6
    eps: float = 1e-12

    def as_drift_config(self) -> DriftLossConfig:
        return DriftLossConfig(
            temperature=self.temperature,
            normalize_over_x=self.normalize_over_x,
            mask_self_negatives=self.mask_self_negatives,
            self_mask_value=self.self_mask_value,
            eps=self.eps,
        )


def grouped_rank_drifting_stopgrad_loss(
    x_grouped,
    y_pos_grouped,
    unconditional_grouped=None,
    unconditional_weight_grouped=None,
    rank_levels=None,
    config=RGMLossConfig(),
):
    _validate_rank_group_inputs(
        x_grouped=x_grouped,
        y_pos_grouped=y_pos_grouped,
        unconditional_grouped=unconditional_grouped,
        unconditional_weight_grouped=unconditional_weight_grouped,
        rank_levels=rank_levels,
    )
    if x_grouped.shape[1] == 1:
        loss, stats = grouped_drifting_stopgrad_loss(
            x_grouped=x_grouped[:, 0],
            y_pos_grouped=y_pos_grouped,
            unconditional_grouped=unconditional_grouped,
            unconditional_weight_grouped=unconditional_weight_grouped,
            config=config.as_drift_config(),
        )
        stats = dict(stats)
        stats.update(
            transport_loss=float(loss.detach().item()),
            order_loss=0.0,
            eq_loss=0.0,
        )
        return loss, stats

    x_flat = _flatten_grouped_tensor(x_grouped)
    y_pos_flat = _flatten_grouped_tensor(y_pos_grouped)
    unconditional_flat = None if unconditional_grouped is None else _flatten_grouped_tensor(unconditional_grouped)
    drifts = []
    targets = []
    per_rank_norms = []
    negative_log_weights = None

    if unconditional_flat is not None:
        if unconditional_weight_grouped is None:
            unconditional_weight_grouped = torch.ones([x_grouped.shape[0]], device=x_grouped.device, dtype=torch.float32)
        negative_log_weights = _build_grouped_negative_log_weights(
            n_generated_negatives=x_grouped.shape[2],
            n_unconditional_negatives=unconditional_grouped.shape[1],
            unconditional_weight_grouped=unconditional_weight_grouped.to(torch.float32),
            device=x_grouped.device,
            dtype=torch.float32,
        )

    drift_config = config.as_drift_config()
    for rank_index in range(x_grouped.shape[1]):
        x_rank = x_flat[:, rank_index]
        y_neg = x_rank.detach()
        if unconditional_flat is not None:
            y_neg = torch.cat([y_neg, unconditional_flat], dim=1)
        drift = _compute_rank_group_drift(
            x_grouped=x_rank.detach(),
            y_pos_grouped=y_pos_flat.detach(),
            y_neg_grouped=y_neg,
            config=drift_config,
            negative_log_weights_grouped=negative_log_weights,
            generated_negative_count=x_grouped.shape[2],
        )
        drifts.append(drift)
        targets.append(x_rank.detach() + drift)
        per_rank_norms.append(drift.norm(dim=-1).mean(dim=1))

    drifts = torch.stack(drifts, dim=1)
    targets = torch.stack(targets, dim=1)
    drift_norms = torch.stack(per_rank_norms, dim=1)

    transport_loss = F.mse_loss(x_flat[:, 1:], targets[:, :-1])
    order_loss = rank_order_loss(drift_norms, margin=config.order_margin)
    eq_loss = fixed_point_loss(x_flat[:, -1], drifts[:, -1])
    total_loss = (
        (config.lambda_transport * transport_loss)
        + (config.lambda_order * order_loss)
        + (config.lambda_eq * eq_loss)
    )

    stats = {
        "loss": float(total_loss.detach().item()),
        "transport_loss": float(transport_loss.detach().item()),
        "order_loss": float(order_loss.detach().item()),
        "eq_loss": float(eq_loss.detach().item()),
        "groups": float(x_grouped.shape[0]),
        "ranks": float(x_grouped.shape[1]),
        "negatives_per_group": float(x_grouped.shape[2]),
        "mean_drift_norm": float(drifts.norm(dim=-1).mean().item()),
        "best_rank_drift_norm": float(drifts[:, -1].norm(dim=-1).mean().item()),
        "worst_rank_drift_norm": float(drifts[:, 0].norm(dim=-1).mean().item()),
    }
    return total_loss, stats


def rank_order_loss(drift_norms, margin=0.0):
    if drift_norms.ndim != 2:
        raise ValueError("drift_norms must be [G, R]")
    if drift_norms.shape[1] <= 1:
        return torch.zeros([], device=drift_norms.device, dtype=drift_norms.dtype)
    return torch.relu(drift_norms[:, 1:] - drift_norms[:, :-1] + float(margin)).mean()


def fixed_point_loss(x_best, drift_best):
    target = x_best.detach() + drift_best.detach()
    return F.mse_loss(x_best, target)


def _flatten_grouped_tensor(value):
    if value.ndim == 6:
        return value.to(torch.float32).reshape(*value.shape[:3], -1)
    if value.ndim == 5:
        return value.to(torch.float32).reshape(*value.shape[:2], -1)
    raise ValueError(f"unsupported grouped tensor rank: {value.ndim}")


def _compute_rank_group_drift(
    x_grouped,
    y_pos_grouped,
    y_neg_grouped,
    config,
    negative_log_weights_grouped,
    generated_negative_count,
):
    affinity_pos, affinity_neg = _compute_grouped_affinity_matrices(
        x_grouped=x_grouped,
        y_pos_grouped=y_pos_grouped,
        y_neg_grouped=y_neg_grouped,
        config=config,
        negative_log_weights_grouped=negative_log_weights_grouped,
        generated_negative_count=generated_negative_count,
    )
    weight_pos = affinity_pos * affinity_neg.sum(dim=-1, keepdim=True)
    weight_neg = affinity_neg * affinity_pos.sum(dim=-1, keepdim=True)
    drift_pos = torch.matmul(weight_pos, y_pos_grouped)
    drift_neg = torch.matmul(weight_neg, y_neg_grouped)
    return drift_pos - drift_neg


def _compute_grouped_affinity_matrices(
    x_grouped,
    y_pos_grouped,
    y_neg_grouped,
    config,
    negative_log_weights_grouped,
    generated_negative_count,
):
    if x_grouped.ndim != 3 or y_pos_grouped.ndim != 3 or y_neg_grouped.ndim != 3:
        raise ValueError("grouped rank drift tensors must be [G, N, D]")
    dist_pos = torch.cdist(x_grouped, y_pos_grouped)
    dist_neg = torch.cdist(x_grouped, y_neg_grouped)

    if config.mask_self_negatives and generated_negative_count > 0:
        diagonal_count = min(x_grouped.shape[1], generated_negative_count, y_neg_grouped.shape[1])
        diagonal = torch.arange(diagonal_count, device=x_grouped.device)
        dist_neg = dist_neg.clone()
        dist_neg[:, diagonal, diagonal] = dist_neg[:, diagonal, diagonal] + config.self_mask_value

    logit_pos = -(dist_pos / config.temperature)
    logit_neg = -(dist_neg / config.temperature)
    if negative_log_weights_grouped is not None:
        logit_neg = logit_neg + negative_log_weights_grouped.unsqueeze(1)

    logits = torch.cat([logit_pos, logit_neg], dim=-1)
    row_affinity = torch.softmax(logits, dim=-1)
    if config.normalize_over_x:
        col_affinity = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=config.eps))
    else:
        affinity = row_affinity

    num_positives = y_pos_grouped.shape[1]
    return affinity[:, :, :num_positives], affinity[:, :, num_positives:]


def _build_grouped_negative_log_weights(
    n_generated_negatives,
    n_unconditional_negatives,
    unconditional_weight_grouped,
    device,
    dtype,
):
    if unconditional_weight_grouped.ndim != 1:
        raise ValueError("unconditional_weight_grouped must be [G]")
    if torch.any(unconditional_weight_grouped < 0):
        raise ValueError("unconditional_weight_grouped must be >= 0")

    groups = unconditional_weight_grouped.shape[0]
    generated = torch.zeros([groups, n_generated_negatives], device=device, dtype=dtype)
    if n_unconditional_negatives == 0:
        return generated

    unconditional_weights = unconditional_weight_grouped.to(device=device, dtype=dtype).unsqueeze(1)
    log_weights = torch.where(
        unconditional_weights == 0,
        torch.full_like(unconditional_weights, torch.finfo(dtype).min),
        torch.log(unconditional_weights),
    )
    unconditional = log_weights.expand(groups, n_unconditional_negatives)
    return torch.cat([generated, unconditional], dim=1)


def _validate_rank_group_inputs(
    *,
    x_grouped,
    y_pos_grouped,
    unconditional_grouped,
    unconditional_weight_grouped,
    rank_levels,
):
    if x_grouped.ndim != 6:
        raise ValueError(f"x_grouped must be [G, R, N, C, H, W], got {tuple(x_grouped.shape)}")
    if y_pos_grouped.ndim != 5:
        raise ValueError(f"y_pos_grouped must be [G, P, C, H, W], got {tuple(y_pos_grouped.shape)}")
    if x_grouped.shape[0] != y_pos_grouped.shape[0]:
        raise ValueError("x_grouped and y_pos_grouped must share group dimension")
    if x_grouped.shape[3:] != y_pos_grouped.shape[2:]:
        raise ValueError("x_grouped and y_pos_grouped must share image shape")
    if unconditional_grouped is not None:
        if unconditional_grouped.ndim != 5:
            raise ValueError("unconditional_grouped must be [G, U, C, H, W]")
        if unconditional_grouped.shape[0] != x_grouped.shape[0]:
            raise ValueError("unconditional_grouped must share group dimension")
        if unconditional_grouped.shape[2:] != x_grouped.shape[3:]:
            raise ValueError("unconditional_grouped must share image shape")
    if unconditional_weight_grouped is not None:
        if unconditional_weight_grouped.ndim != 1:
            raise ValueError("unconditional_weight_grouped must be [G]")
        if unconditional_weight_grouped.shape[0] != x_grouped.shape[0]:
            raise ValueError("unconditional_weight_grouped must share group dimension")
    if rank_levels is not None:
        if not isinstance(rank_levels, torch.Tensor):
            rank_levels = torch.as_tensor(rank_levels, dtype=torch.float32)
        if rank_levels.ndim != 1 or rank_levels.shape[0] != x_grouped.shape[1]:
            raise ValueError("rank_levels must be [R]")
        if rank_levels.shape[0] > 1 and torch.any(rank_levels[1:] >= rank_levels[:-1]):
            raise ValueError("rank_levels must be strictly descending")
