"""
Symmetric Pushforward Matching (SPM) for R3GAN.

Core idea: Instead of ranking samples, match transformed distributions.
For each transform T_k, match T_k#p_G against T_k#p_data.

This is:
- Equilibrium-compatible: at p_G=p_data, T_k#p_G = T_k#p_data → loss = 0
- Symmetric: same transform applied to both real and fake
- G-preserving: G generates one image, no rank conditioning

Transforms:
  T_0 = identity (standard adversarial game)
  T_1 = antialiased downsample 32→8→32 (coarse structure)
  T_2 = Gaussian blur σ=2 (smooth approximation)
"""

import torch
import torch.nn.functional as F


def transform_downsample(x, low_res=8):
    """Antialiased downsample then upsample."""
    orig_size = x.shape[-1]
    down = F.interpolate(x, size=low_res, mode='bilinear', align_corners=False, antialias=True)
    return F.interpolate(down, size=orig_size, mode='bilinear', align_corners=False)


def transform_blur(x, kernel_size=5, sigma=2.0):
    """Gaussian blur."""
    C = x.shape[1]
    # Create Gaussian kernel
    ax = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(C, 1, -1, -1)
    pad = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=pad, groups=C)


TRANSFORMS = {
    'identity': lambda x: x,
    'downsample8': lambda x: transform_downsample(x, low_res=8),
    'downsample16': lambda x: transform_downsample(x, low_res=16),
    'blur2': lambda x: transform_blur(x, sigma=2.0),
    'blur4': lambda x: transform_blur(x, sigma=4.0),
}


def compute_spm_loss(D, real_img, fake_img, real_c, preprocessor,
                     transforms=('downsample8',), weights=(0.25,),
                     margin=1.0):
    """
    Compute SPM auxiliary loss.
    
    For each transform T_k:
      D sees T_k(real) vs T_k(fake)
      Pairwise delta loss: softplus(margin - (D(T_k(real)) - D(T_k(fake))))
    
    Args:
        D: discriminator (standard, no rank conditioning needed)
        real_img: real images
        fake_img: G(z) output (detached for D phase, with grad for G phase)
        real_c: class labels
        preprocessor: augmentation pipeline
        transforms: tuple of transform names
        weights: loss weight per transform
        margin: pairwise margin
    
    Returns:
        spm_loss: scalar
        info: dict with per-transform losses
    """
    total_loss = 0.0
    info = {}
    
    for k, (t_name, w) in enumerate(zip(transforms, weights)):
        T = TRANSFORMS[t_name]
        
        # Apply SAME transform to both real and fake (symmetric!)
        real_t = T(real_img)
        fake_t = T(fake_img)
        
        # Augment then score
        real_scores = D(preprocessor(real_t), real_c)
        fake_scores = D(preprocessor(fake_t), real_c)
        
        # Flatten scores
        if real_scores.ndim > 1:
            real_scores = real_scores.squeeze()
        if fake_scores.ndim > 1:
            fake_scores = fake_scores.squeeze()
        
        delta = real_scores - fake_scores
        loss_k = F.softplus(margin - delta).mean()
        
        total_loss = total_loss + w * loss_k
        info[f'spm_{t_name}'] = loss_k.item()
        info[f'spm_delta_{t_name}'] = delta.mean().item()
    
    info['spm_total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    return total_loss, info
