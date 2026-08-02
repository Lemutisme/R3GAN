"""
SRG Loss wrapper for R3GAN.

Wraps R3GANLoss to add:
1. Rank sampling per batch (k=0 coarse, k=1 clean)
2. Real image transformation at coarse rank (downsample-upsample)
3. Latent prefix masking at coarse rank
4. Consistency loss: T_0(G(z, 1)) ≈ G(z_masked, 0)
5. Rank-aware D forwarding

This is designed as MINIMAL modification to R3GANLoss.
"""

import torch
import torch.nn.functional as F
from training.loss import R3GANLoss
from training.srg import transform_real, mask_latent, consistency_loss
import dnnlib.util as training_stats_module

# Monkey-patch: we need to intercept G/D calls in the loss


class SRGLoss(R3GANLoss):
    """R3GANLoss + SRG rank structure."""

    def __init__(self, *args,
                 srg_enable=False,
                 srg_coarse_res=8,
                 srg_rank_prob_coarse=0.25,
                 srg_rank_weight_coarse=0.25,
                 srg_prefix_dim_coarse=16,
                 srg_prefix_dim_clean=64,
                 srg_consistency_weight=5.0,
                 srg_consistency_prob=0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.srg_enable = bool(srg_enable)
        self.srg_coarse_res = int(srg_coarse_res)
        self.srg_rank_prob_coarse = float(srg_rank_prob_coarse)
        self.srg_rank_weight_coarse = float(srg_rank_weight_coarse)
        self.srg_prefix_dim_coarse = int(srg_prefix_dim_coarse)
        self.srg_prefix_dim_clean = int(srg_prefix_dim_clean)
        self.srg_consistency_weight = float(srg_consistency_weight)
        self.srg_consistency_prob = float(srg_consistency_prob)

    def _srg_transform_real(self, real_img, rank):
        """Transform real images based on rank."""
        if rank == 0:
            return transform_real(real_img, rank=0,
                                  resolutions=(self.srg_coarse_res, real_img.shape[-1]))
        return real_img

    def _srg_mask_latent(self, z, rank):
        """Mask latent based on rank."""
        if rank == 0:
            return mask_latent(z, rank=0,
                               prefix_dims=(self.srg_prefix_dim_coarse, self.srg_prefix_dim_clean))
        return z

    def _accumulate_gradients_impl(self, phase, real_img, real_c, gen_z, gamma, gain):
        """Override to inject SRG rank logic."""
        if not self.srg_enable:
            return super()._accumulate_gradients_impl(phase, real_img, real_c, gen_z, gamma, gain)

        device = real_img.device
        B = real_img.shape[0]

        # Sample rank for this batch
        is_coarse = torch.rand(1).item() < self.srg_rank_prob_coarse
        rank = 0 if is_coarse else 1
        alpha = torch.full([B], float(rank), device=device)
        weight = self.srg_rank_weight_coarse if rank == 0 else 1.0

        # Transform real images for this rank
        real_img_rank = self._srg_transform_real(real_img, rank)

        # Mask latent for this rank
        gen_z_rank = self._srg_mask_latent(gen_z, rank)

        # Generate fake images with rank conditioning
        gen_img = self.G(gen_z_rank, real_c, alpha=alpha)

        # Run D with rank conditioning
        if hasattr(self.D, 'forward'):
            # Check if D accepts alpha
            import inspect
            sig = inspect.signature(self.D.forward)
            if 'alpha' in sig.parameters:
                real_scores = self._as_scores(
                    self.D(self.preprocessor(real_img_rank), real_c, alpha=alpha))
                fake_scores = self._as_scores(
                    self.D(self.preprocessor(gen_img), real_c, alpha=alpha))
            else:
                real_scores = self.run_D(real_img_rank, real_c)
                fake_scores = self.run_D(gen_img, real_c)
        else:
            real_scores = self.run_D(real_img_rank, real_c)
            fake_scores = self.run_D(gen_img, real_c)

        # Pairwise delta loss (core R3GAN)
        paired_delta = real_scores - fake_scores

        if phase == "D":
            pair_loss = F.softplus(self.pair_margin - paired_delta)
            loss = weight * self.lambda_pair * pair_loss.mean()

            # R1/R2 gradient penalties
            if self.use_r1_penalty and gamma > 0:
                real_img_gp = real_img_rank.detach().requires_grad_(True)
                if hasattr(self.D, 'forward') and 'alpha' in inspect.signature(self.D.forward).parameters:
                    r1_scores = self._as_scores(self.D(real_img_gp, real_c, alpha=alpha))
                else:
                    r1_scores = self._as_scores(self.D(real_img_gp, real_c))
                r1_grads = torch.autograd.grad(r1_scores.sum(), real_img_gp, create_graph=True)[0]
                r1_penalty = (r1_grads ** 2).sum(dim=[1, 2, 3]).mean()
                loss = loss + gamma / 2 * r1_penalty

            if self.use_r2_penalty and gamma > 0:
                fake_img_gp = gen_img.detach().requires_grad_(True)
                if hasattr(self.D, 'forward') and 'alpha' in inspect.signature(self.D.forward).parameters:
                    r2_scores = self._as_scores(self.D(fake_img_gp, real_c, alpha=alpha))
                else:
                    r2_scores = self._as_scores(self.D(fake_img_gp, real_c))
                r2_grads = torch.autograd.grad(r2_scores.sum(), fake_img_gp, create_graph=True)[0]
                r2_penalty = (r2_grads ** 2).sum(dim=[1, 2, 3]).mean()
                loss = loss + gamma / 2 * r2_penalty

            loss.mul(gain).backward()

        elif phase == "G":
            pair_loss = F.softplus(self.pair_margin + paired_delta)
            loss = weight * self.lambda_pair * pair_loss.mean()

            # Consistency loss (only on a fraction of G batches)
            if self.srg_consistency_weight > 0 and torch.rand(1).item() < self.srg_consistency_prob:
                cons = consistency_loss(
                    self.G, gen_z, real_c,
                    alpha_clean=1.0, alpha_coarse=0.0,
                    prefix_dims=(self.srg_prefix_dim_coarse, self.srg_prefix_dim_clean),
                    resolutions=(self.srg_coarse_res, real_img.shape[-1]),
                )
                loss = loss + self.srg_consistency_weight * cons

            loss.mul(gain).backward()
