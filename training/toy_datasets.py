"""Toy 2D samplers for RGM experiments."""

import math

import torch


def sample_checkerboard(n: int, noise: float = 0.05, seed: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=generator)
    i = torch.randint(0, 2, (n,), generator=generator) * 2 + b
    j = torch.randint(0, 2, (n,), generator=generator) * 2 + b
    u = torch.rand(n, generator=generator)
    v = torch.rand(n, generator=generator)
    points = torch.stack([i + u, j + v], dim=1) - 2.0
    points = points / 2.0
    if noise > 0:
        points = points + noise * torch.randn(points.shape, generator=generator)
    return points


def sample_swiss_roll(n: int, noise: float = 0.03, seed: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand(n, generator=generator)
    t = 0.5 * math.pi + 4.0 * math.pi * u
    points = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    points = points / (points.abs().max() + 1e-8)
    if noise > 0:
        points = points + noise * torch.randn(points.shape, generator=generator)
    return points


def sample_four_gaussians(n: int, noise: float = 0.1, seed: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    centers = torch.tensor(
        [[-0.75, -0.75], [-0.75, 0.75], [0.75, -0.75], [0.75, 0.75]],
        dtype=torch.float32,
    )
    indices = torch.randint(0, centers.shape[0], (n,), generator=generator)
    points = centers[indices]
    if noise > 0:
        points = points + noise * torch.randn(points.shape, generator=generator)
    return points
