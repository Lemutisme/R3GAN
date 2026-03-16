"""Condition helpers for rank-conditioned generators."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RankConditionSchema:
    use_alpha: bool = False
    use_rank: bool = False
    use_rank_pair: bool = False
    alpha_min: float = 1.0
    alpha_max: float = 4.0
    rank_min: float = 0.0
    rank_max: float = 1.0


def schema_extra_dim(schema: RankConditionSchema) -> int:
    return int(schema.use_alpha) + int(schema.use_rank) + (2 * int(schema.use_rank_pair))


def normalize_scalar_condition(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    if hi < lo:
        raise ValueError("condition max must be >= min")
    if hi == lo:
        return torch.zeros_like(x, dtype=torch.float32)
    return ((x.to(torch.float32) - float(lo)) / float(hi - lo)).clamp(0.0, 1.0)


def build_extra_condition(
    c: torch.Tensor | None,
    *,
    alpha: torch.Tensor | None = None,
    rank: torch.Tensor | None = None,
    rank_from: torch.Tensor | None = None,
    rank_to: torch.Tensor | None = None,
    schema: RankConditionSchema,
) -> torch.Tensor:
    batch_size, device = _resolve_batch_size_and_device(c, alpha, rank, rank_from, rank_to)
    pieces = []

    if schema.use_alpha:
        alpha_value = _coerce_scalar(alpha, batch_size=batch_size, device=device, name="alpha")
        pieces.append(normalize_scalar_condition(alpha_value, schema.alpha_min, schema.alpha_max))
    if schema.use_rank:
        rank_value = _coerce_scalar(rank, batch_size=batch_size, device=device, name="rank")
        pieces.append(normalize_scalar_condition(rank_value, schema.rank_min, schema.rank_max))
    if schema.use_rank_pair:
        rank_from_value = _coerce_scalar(rank_from, batch_size=batch_size, device=device, name="rank_from")
        rank_to_value = _coerce_scalar(rank_to, batch_size=batch_size, device=device, name="rank_to")
        pieces.append(normalize_scalar_condition(rank_from_value, schema.rank_min, schema.rank_max))
        pieces.append(normalize_scalar_condition(rank_to_value, schema.rank_min, schema.rank_max))

    if c is not None:
        condition = c.to(device=device, dtype=torch.float32)
        if condition.ndim != 2 or condition.shape[0] != batch_size:
            raise ValueError("c must be [B, C] and aligned with extra conditions")
    else:
        condition = torch.zeros([batch_size, 0], device=device, dtype=torch.float32)

    if len(pieces) == 0:
        return condition
    return torch.cat([condition, *pieces], dim=1)


def _resolve_batch_size_and_device(*values: torch.Tensor | None) -> tuple[int, torch.device]:
    for value in values:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim == 0:
            continue
        return int(value.shape[0]), value.device
    raise ValueError("could not infer batch size from conditioning inputs")


def _coerce_scalar(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} must be provided by the active condition schema")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
    else:
        value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must be [B] or scalar, got {tuple(value.shape)}")
    return value.reshape(batch_size, 1)
