"""
SRG (Successive Refinement Generator) helpers for R3GAN.

Implements:
- Rank-dependent target transforms T_k(x)
- Latent prefix masking
- Consistency loss
- Rank sampling
"""

import torch
import torch.nn.functional as F


def transform_real(real_img, rank, resolutions=(8, 32)):
    """
    Transform real images to target distribution p_k.
    k=0: antialiased downsample to low_res then upsample back
    k=1: identity (clean data)
    """
    if rank == 1:
        return real_img
    low_res = resolutions[0]
    # Antialiased downsample then upsample
    down = F.interpolate(real_img, size=low_res, mode='bilinear', align_corners=False, antialias=True)
    up = F.interpolate(down, size=real_img.shape[-1], mode='bilinear', align_corners=False)
    return up


def mask_latent(z, rank, prefix_dims=(16, 64)):
    """
    Apply prefix mask to latent z.
    At rank k, only the first prefix_dims[k] dimensions are used.
    """
    total_dim = z.shape[1]
    active = prefix_dims[min(rank, len(prefix_dims) - 1)]
    if active >= total_dim:
        return z
    mask = torch.zeros_like(z)
    mask[:, :active] = 1.0
    return z * mask


def sample_rank(batch_size, rank_probs=(0.25, 0.75), device='cpu'):
    """Sample rank indices according to rank_probs."""
    probs = torch.tensor(rank_probs, device=device)
    return torch.multinomial(probs.expand(batch_size, -1), 1).squeeze(1)


def consistency_loss(G, z, c, alpha_clean=1.0, alpha_coarse=0.0,
                     prefix_dims=(16, 64), resolutions=(8, 32)):
    """
    Consistency: T_0(G(z, rank=1)) ≈ G(z_masked, rank=0)
    The clean-rank output, when degraded, should match coarse-rank output.
    """
    # Clean rank generation (full latent)
    with torch.no_grad():
        clean_out = G(z, c, alpha=alpha_clean)
        clean_degraded = transform_real(clean_out, rank=0, resolutions=resolutions)

    # Coarse rank generation (masked latent)
    z_masked = mask_latent(z, rank=0, prefix_dims=prefix_dims)
    coarse_out = G(z_masked, c, alpha=alpha_coarse)

    return F.l1_loss(coarse_out, clean_degraded)
