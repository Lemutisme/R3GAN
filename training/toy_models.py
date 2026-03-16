"""Toy models for 2D RGM experiments."""

import torch
import torch.nn as nn


class ToyMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=256, out_dim=2):
        super(ToyMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class ToyRankMLP(nn.Module):
    def __init__(self, z_dim=32, hidden=256, out_dim=2):
        super(ToyRankMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, rank):
        if not isinstance(rank, torch.Tensor):
            rank = torch.as_tensor(rank, device=z.device, dtype=torch.float32)
        else:
            rank = rank.to(device=z.device, dtype=torch.float32)
        if rank.ndim == 0:
            rank = rank.expand(z.shape[0])
        if rank.ndim != 1 or rank.shape[0] != z.shape[0]:
            raise ValueError("rank must be [B] or scalar")
        return self.net(torch.cat([z, rank.reshape(z.shape[0], 1)], dim=1))


class ToyTransportMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=256, out_dim=2):
        super(ToyTransportMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_t, rank_from, rank_to):
        rank_from = _coerce_scalar(rank_from, batch_size=x_t.shape[0], device=x_t.device)
        rank_to = _coerce_scalar(rank_to, batch_size=x_t.shape[0], device=x_t.device)
        return self.net(torch.cat([x_t, rank_from, rank_to], dim=1))


def _coerce_scalar(value, batch_size, device):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
    else:
        value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError("scalar input must be [B] or scalar")
    return value.reshape(batch_size, 1)
