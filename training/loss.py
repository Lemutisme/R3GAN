# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from R3GAN.Trainer import AdversarialTraining

#----------------------------------------------------------------------------

def listmle_loss(scores_sorted: torch.Tensor) -> torch.Tensor:
    """ListMLE loss for [B, K] scores ordered best->worst."""
    rev = torch.flip(scores_sorted, dims=[-1])
    rev_lse = torch.logcumsumexp(rev, dim=-1)
    suffix_lse = torch.flip(rev_lse, dims=[-1])
    loss = (suffix_lse - scores_sorted).sum(dim=-1)
    return loss.mean()


def pairwise_logistic_loss(scores_sorted: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic loss for [B, K] scores ordered best->worst."""
    bsz, k = scores_sorted.shape
    s_i = scores_sorted.unsqueeze(2)
    s_j = scores_sorted.unsqueeze(1)
    diff = s_i - s_j
    mask = torch.triu(torch.ones(k, k, device=scores_sorted.device, dtype=scores_sorted.dtype), diagonal=1)
    loss_all = F.softplus(-diff)
    return (loss_all * mask).sum() / (bsz * mask.sum())


def pairwise_hinge_loss(scores_sorted: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Pairwise hinge loss for [B, K] scores ordered best->worst."""
    bsz, k = scores_sorted.shape
    s_i = scores_sorted.unsqueeze(2)
    s_j = scores_sorted.unsqueeze(1)
    diff = s_i - s_j
    mask = torch.triu(torch.ones(k, k, device=scores_sorted.device, dtype=scores_sorted.dtype), diagonal=1)
    loss_all = F.relu(margin - diff)
    return (loss_all * mask).sum() / (bsz * mask.sum())


def make_rank_list(real_imgs: torch.Tensor, fake_imgs: torch.Tensor, k: int,
                   mode: str = 'intrpl', alpha_dist: str = 'linear') -> torch.Tensor:
    """
    Build [B, K, C, H, W] list from real (index 0) to fake (index K-1).
    """
    device = real_imgs.device
    if alpha_dist == 'linear':
        alphas = torch.linspace(1.0, 0.0, k, device=device)
    elif alpha_dist == 'cosine':
        alphas = 0.5 * (1.0 + torch.cos(torch.linspace(0, np.pi, k, device=device)))
    elif alpha_dist == 'random':
        if k <= 2:
            alphas = torch.linspace(1.0, 0.0, k, device=device)
        else:
            alphas = torch.cat([
                torch.ones(1, device=device),
                torch.rand(k - 2, device=device),
                torch.zeros(1, device=device),
            ])
            alphas = torch.sort(alphas, descending=True)[0]
    else:
        alphas = torch.linspace(1.0, 0.0, k, device=device)

    alphas = alphas.view(1, k, 1, 1, 1)
    interp = alphas * real_imgs.unsqueeze(1) + (1.0 - alphas) * fake_imgs.unsqueeze(1)
    if mode == 'intrpl':
        return interp
    if mode == 'noise':
        noise = torch.randn_like(real_imgs).unsqueeze(1) * 0.01
        noise_gain = alphas * (1.0 - alphas)
        return interp + noise_gain * noise
    if mode == 'add_mix':
        noise = torch.randn_like(real_imgs).unsqueeze(1) * 0.01
        noise_gain = 2.0 * alphas * (1.0 - alphas)
        return interp + noise_gain * noise
    return interp

#----------------------------------------------------------------------------

class R3GANLoss:
    def __init__(self, G, D, augment_pipe=None, rank_loss=False, rank_K=8, rank_loss_type='listmle',
                 lambda_rank=0.1, lambda_adv=1.0, rank_mode='intrpl', rank_alpha_dist='linear',
                 rank_augment=False, rank_margin=1.0, rank_score_reg=0.0,
                 use_r1_penalty=True, use_r2_penalty=True):
        self.G = G
        self.D = D
        self.trainer = AdversarialTraining(G, D)
        self.augment_pipe = augment_pipe
        if augment_pipe is not None:
            self.preprocessor = lambda x: augment_pipe(x.to(torch.float32)).to(x.dtype)
        else:
            self.preprocessor = lambda x: x

        self.rank_loss = rank_loss
        self.rank_K = rank_K
        self.rank_loss_type = rank_loss_type
        self.lambda_rank = lambda_rank
        self.lambda_adv = lambda_adv
        self.rank_mode = rank_mode
        self.rank_alpha_dist = rank_alpha_dist
        self.rank_augment = rank_augment
        self.rank_margin = rank_margin
        self.rank_score_reg = rank_score_reg
        self.use_r1_penalty = use_r1_penalty
        self.use_r2_penalty = use_r2_penalty

    def run_D(self, img, c, augment=True):
        if augment and self.augment_pipe is not None:
            img = self.augment_pipe(img.to(torch.float32)).to(img.dtype)
        return self.D(img, c)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gamma, gain):
        # G
        if phase == 'G':
            AdversarialLoss, RelativisticLogits = self.trainer.AccumulateGeneratorGradients(gen_z, real_img, real_c, gain, self.preprocessor)

            training_stats.report('Loss/scores/fake', RelativisticLogits)
            training_stats.report('Loss/signs/fake', RelativisticLogits.sign())
            training_stats.report('Loss/G/loss', AdversarialLoss)

        # D
        if phase == 'D':
            AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty = self.trainer.AccumulateDiscriminatorGradients(
                gen_z, real_img, real_c, gamma, gain, self.preprocessor, self.lambda_adv,
                self.use_r1_penalty, self.use_r2_penalty
            )

            # Report decomposed D terms for easier diagnostics.
            adv_term = self.lambda_adv * AdversarialLoss
            r1_term = (gamma / 2) * R1Penalty if self.use_r1_penalty else torch.zeros_like(AdversarialLoss)
            r2_term = (gamma / 2) * R2Penalty if self.use_r2_penalty else torch.zeros_like(AdversarialLoss)
            d_base_total = adv_term + r1_term + r2_term
            d_rank_term = torch.zeros([], device=real_img.device, dtype=AdversarialLoss.dtype)

            training_stats.report('Loss/scores/real', RelativisticLogits)
            training_stats.report('Loss/signs/real', RelativisticLogits.sign())
            training_stats.report('Loss/D/loss', AdversarialLoss)
            training_stats.report('Loss/r1_penalty', R1Penalty)
            training_stats.report('Loss/r2_penalty', R2Penalty)
            training_stats.report('Loss/D/adv', AdversarialLoss)
            training_stats.report('Loss/D/adv_weighted', adv_term)
            training_stats.report('Loss/D/r1_weighted', r1_term)
            training_stats.report('Loss/D/r2_weighted', r2_term)
            training_stats.report('Loss/D/base_total', d_base_total)

            if self.rank_loss:
                with torch.no_grad():
                    rank_fake_img = self.G(gen_z, real_c).detach()
                rank_imgs = make_rank_list(
                    real_img, rank_fake_img,
                    k=self.rank_K, mode=self.rank_mode, alpha_dist=self.rank_alpha_dist
                )
                bsz, k, c, h, w = rank_imgs.shape
                rank_imgs_flat = rank_imgs.reshape(bsz * k, c, h, w)
                rank_c = real_c.repeat_interleave(k, dim=0)
                rank_logits = self.run_D(rank_imgs_flat, rank_c, augment=self.rank_augment)
                rank_logits = rank_logits.reshape(bsz, k, -1)
                rank_scores = rank_logits.squeeze(-1) if rank_logits.shape[-1] == 1 else rank_logits.mean(dim=-1)

                if self.rank_loss_type == 'listmle':
                    loss_Drank = listmle_loss(rank_scores)
                elif self.rank_loss_type == 'pairwise_logistic':
                    loss_Drank = pairwise_logistic_loss(rank_scores)
                elif self.rank_loss_type == 'pairwise_hinge':
                    loss_Drank = pairwise_hinge_loss(rank_scores, margin=self.rank_margin)
                else:
                    loss_Drank = listmle_loss(rank_scores)

                if self.rank_score_reg > 0:
                    score_reg = rank_scores.square().mean()
                    loss_Drank = loss_Drank + self.rank_score_reg * score_reg
                    training_stats.report('Loss/D/score_reg', score_reg)

                training_stats.report('Loss/D/rank', loss_Drank)
                d_rank_term = self.lambda_rank * loss_Drank
                training_stats.report('Loss/D/rank_weighted', d_rank_term)
                (gain * self.lambda_rank * loss_Drank).backward()
            else:
                training_stats.report('Loss/D/rank_weighted', d_rank_term)

            training_stats.report('Loss/D/total', d_base_total + d_rank_term)
#----------------------------------------------------------------------------
