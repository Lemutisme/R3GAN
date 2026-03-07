# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from R3GAN.Trainer import AdversarialTraining

# ----------------------------------------------------------------------------
# Deprecated interpolation-based path ranking losses.
# These auxiliary losses build an interpolation chain between real and fake
# images and teach D to rank them. They are kept only as optional path priors;
# the main game is defined on pairwise critic differences.
# ----------------------------------------------------------------------------


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
    mask = torch.triu(
        torch.ones(k, k, device=scores_sorted.device, dtype=scores_sorted.dtype),
        diagonal=1,
    )
    loss_all = F.softplus(-diff)
    return (loss_all * mask).sum() / (bsz * mask.sum())


def pairwise_hinge_loss(
    scores_sorted: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Pairwise hinge loss for [B, K] scores ordered best->worst."""
    bsz, k = scores_sorted.shape
    s_i = scores_sorted.unsqueeze(2)
    s_j = scores_sorted.unsqueeze(1)
    diff = s_i - s_j
    mask = torch.triu(
        torch.ones(k, k, device=scores_sorted.device, dtype=scores_sorted.dtype),
        diagonal=1,
    )
    loss_all = F.relu(margin - diff)
    return (loss_all * mask).sum() / (bsz * mask.sum())


def make_rank_list(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    k: int,
    mode: str = "intrpl",
    alpha_dist: str = "linear",
) -> torch.Tensor:
    """
    Build [B, K, C, H, W] list from real (index 0) to fake (index K-1).
    """
    device = real_imgs.device
    if alpha_dist == "linear":
        alphas = torch.linspace(1.0, 0.0, k, device=device)
    elif alpha_dist == "cosine":
        alphas = 0.5 * (1.0 + torch.cos(torch.linspace(0, np.pi, k, device=device)))
    elif alpha_dist == "random":
        if k <= 2:
            alphas = torch.linspace(1.0, 0.0, k, device=device)
        else:
            alphas = torch.cat(
                [
                    torch.ones(1, device=device),
                    torch.rand(k - 2, device=device),
                    torch.zeros(1, device=device),
                ]
            )
            alphas = torch.sort(alphas, descending=True)[0]
    else:
        alphas = torch.linspace(1.0, 0.0, k, device=device)

    alphas = alphas.view(1, k, 1, 1, 1)
    interp = alphas * real_imgs.unsqueeze(1) + (1.0 - alphas) * fake_imgs.unsqueeze(1)
    if mode == "intrpl":
        return interp
    if mode == "noise":
        noise = torch.randn_like(real_imgs).unsqueeze(1) * 0.01
        noise_gain = alphas * (1.0 - alphas)
        return interp + noise_gain * noise
    if mode == "add_mix":
        noise = torch.randn_like(real_imgs).unsqueeze(1) * 0.01
        noise_gain = 2.0 * alphas * (1.0 - alphas)
        return interp + noise_gain * noise
    return interp


def pairwise_delta(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Paired critic differences delta_i = s(real_i) - s(fake_i)."""
    return AdversarialTraining._pairwise_delta(real_scores, fake_scores)


def allpairs_delta(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """All-pairs critic differences Delta_ij = s(real_i) - s(fake_j)."""
    return real_scores.unsqueeze(1) - fake_scores.unsqueeze(0)


def pairwise_discriminator_loss(
    delta: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """Pairwise discriminator loss on delta = s(real) - s(fake)."""
    return F.softplus(margin - delta)


def pairwise_generator_loss(
    delta: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """Pairwise generator loss on delta = s(real) - s(fake)."""
    return F.softplus(margin + delta)


def infonce_discriminator_loss(
    delta_matrix: torch.Tensor, tau: float = 0.07
) -> torch.Tensor:
    """Row-wise listwise loss on Delta[i, j] = s(real_i) - s(fake_j)."""
    if tau <= 0:
        raise ValueError(f"InfoNCE temperature must be positive, got {tau}")
    row_lse = torch.logsumexp(-delta_matrix / tau, dim=1)
    return torch.logaddexp(torch.zeros_like(row_lse), row_lse)


def infonce_generator_loss(
    delta_matrix: torch.Tensor, tau: float = 0.07
) -> torch.Tensor:
    """Column-wise listwise loss on Delta[i, j] = s(real_i) - s(fake_j)."""
    if tau <= 0:
        raise ValueError(f"InfoNCE temperature must be positive, got {tau}")
    col_lse = torch.logsumexp(delta_matrix / tau, dim=0)
    return torch.logaddexp(torch.zeros_like(col_lse), col_lse)


# ----------------------------------------------------------------------------


class R3GANLoss:
    def __init__(
        self,
        G,
        D,
        augment_pipe=None,
        lambda_pair=None,
        pair_margin=None,
        lambda_list=None,
        list_loss_type="infonce",
        list_tau=None,
        lambda_local_rank=0.0,
        local_rank_k=4,
        path_rank_reg=False,
        path_rank_k=8,
        path_rank_loss_type="listmle",
        lambda_path_rank=0.1,
        path_rank_mode="intrpl",
        path_rank_alpha_dist="linear",
        path_rank_margin=1.0,
        path_rank_score_reg=0.0,
        use_r1_penalty=True,
        use_r2_penalty=True,
        use_non_aug_gp=False,
        rank_loss=False,
        rank_K=8,
        rank_loss_type="listmle",
        lambda_rank=0.1,
        lambda_adv=1.0,
        adv_loss_type="softmargin",
        adv_margin=0.0,
        adv_tau=0.07,
        rank_mode="intrpl",
        rank_alpha_dist="linear",
        rank_augment=False,
        rank_margin=1.0,
        rank_score_reg=0.0,
    ):
        self.G = G
        self.D = D
        self.trainer = AdversarialTraining(G, D)
        self.augment_pipe = augment_pipe
        if augment_pipe is not None:
            self.preprocessor = lambda x: augment_pipe(x.to(torch.float32)).to(x.dtype)
        else:
            self.preprocessor = lambda x: x

        self.use_r1_penalty = use_r1_penalty
        self.use_r2_penalty = use_r2_penalty
        self.use_non_aug_gp = use_non_aug_gp

        using_new_main = any(
            value is not None
            for value in [lambda_pair, pair_margin, lambda_list, list_tau]
        )
        if using_new_main:
            self.lambda_pair = 1.0 if lambda_pair is None else float(lambda_pair)
            self.pair_margin = 0.0 if pair_margin is None else float(pair_margin)
            self.lambda_list = 0.0 if lambda_list is None else float(lambda_list)
            self.list_tau = 0.07 if list_tau is None else float(list_tau)
        else:
            if adv_loss_type not in ["softmargin", "infonce"]:
                raise ValueError(f"Unknown adversarial loss type: {adv_loss_type}")
            self.lambda_pair = float(lambda_adv if adv_loss_type == "softmargin" else 0.0)
            self.pair_margin = float(adv_margin)
            self.lambda_list = float(lambda_adv if adv_loss_type == "infonce" else 0.0)
            self.list_tau = float(adv_tau)
            list_loss_type = "infonce"

        self.list_loss_type = list_loss_type
        self.lambda_local_rank = float(lambda_local_rank)
        self.local_rank_k = int(local_rank_k)

        using_new_path = (
            path_rank_reg
            or path_rank_k != 8
            or path_rank_loss_type != "listmle"
            or lambda_path_rank != 0.1
            or path_rank_mode != "intrpl"
            or path_rank_alpha_dist != "linear"
            or path_rank_margin != 1.0
            or path_rank_score_reg != 0.0
        )
        if using_new_path:
            self.path_rank_reg = bool(path_rank_reg)
            self.path_rank_k = int(path_rank_k)
            self.path_rank_loss_type = path_rank_loss_type
            self.lambda_path_rank = float(lambda_path_rank)
            self.path_rank_mode = path_rank_mode
            self.path_rank_alpha_dist = path_rank_alpha_dist
            self.path_rank_margin = float(path_rank_margin)
            self.path_rank_score_reg = float(path_rank_score_reg)
        else:
            self.path_rank_reg = bool(rank_loss)
            self.path_rank_k = int(rank_K)
            self.path_rank_loss_type = rank_loss_type
            self.lambda_path_rank = float(lambda_rank)
            self.path_rank_mode = rank_mode
            self.path_rank_alpha_dist = rank_alpha_dist
            self.path_rank_margin = float(rank_margin)
            self.path_rank_score_reg = float(rank_score_reg)

        if self.list_loss_type != "infonce":
            raise ValueError(f"Unknown list_loss_type: {self.list_loss_type}")
        if self.lambda_pair < 0:
            raise ValueError("lambda_pair must be non-negative")
        if self.pair_margin < 0:
            raise ValueError("pair_margin must be non-negative")
        if self.lambda_list < 0:
            raise ValueError("lambda_list must be non-negative")
        if self.list_tau <= 0:
            raise ValueError("list_tau must be positive")
        if self.lambda_local_rank < 0:
            raise ValueError("lambda_local_rank must be non-negative")
        if self.local_rank_k < 2:
            raise ValueError("local_rank_k must be at least 2")
        if self.lambda_path_rank < 0:
            raise ValueError("lambda_path_rank must be non-negative")
        if self.path_rank_k < 2:
            raise ValueError("path_rank_k must be at least 2")
        if self.path_rank_margin <= 0:
            raise ValueError("path_rank_margin must be positive")
        if self.path_rank_score_reg < 0:
            raise ValueError("path_rank_score_reg must be non-negative")

        if rank_augment:
            warnings.warn(
                "rank_augment is deprecated and ignored; path_rank_reg always uses non-augmented views.",
                stacklevel=2,
            )

        self._full_batch_buffer = None

    def _requires_full_batch(self):
        return self.lambda_list > 0 or self.lambda_local_rank > 0

    def _as_scores(self, logits):
        return self.trainer._as_vector(logits)

    def run_D(self, img, c, augment=True, return_features=False):
        if augment:
            img = self.preprocessor(img)
        if return_features:
            try:
                scores, features = self.D(img, c, return_features=True)
            except TypeError as err:
                raise ValueError(
                    "Discriminator must support return_features=True when lambda_local_rank > 0."
                ) from err
            scores = self._as_scores(scores)
            features = features.to(torch.float32)
            if features.ndim != 2:
                features = features.reshape(features.shape[0], -1)
            return scores, features
        return self._as_scores(self.D(img, c))

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gamma, gain):
        if self._requires_full_batch():
            if self._full_batch_buffer is None:
                self._full_batch_buffer = {
                    "phase": phase,
                    "gamma": gamma,
                    "real_imgs": [],
                    "real_cs": [],
                    "gen_zs": [],
                    "gains": [],
                }
            self._full_batch_buffer["real_imgs"].append(real_img)
            self._full_batch_buffer["real_cs"].append(real_c)
            self._full_batch_buffer["gen_zs"].append(gen_z)
            self._full_batch_buffer["gains"].append(gain)
            return
        self._accumulate_gradients_impl(phase, real_img, real_c, gen_z, gamma, gain)

    def finalize_accumulation(self):
        """Flush buffered micro-batches for full-batch coupled losses."""
        if self._full_batch_buffer is None:
            return
        buf = self._full_batch_buffer
        self._full_batch_buffer = None
        merged_real = torch.cat(buf["real_imgs"])
        merged_c = torch.cat(buf["real_cs"])
        merged_z = torch.cat(buf["gen_zs"])
        merged_gain = sum(buf["gains"])
        self._accumulate_gradients_impl(
            buf["phase"],
            merged_real,
            merged_c,
            merged_z,
            buf["gamma"],
            merged_gain,
        )

    def _report_common_stats(self, phase, pair_loss, list_loss, paired_delta, device):
        zero_scalar = torch.zeros([], device=device)
        zero_vector = torch.zeros_like(paired_delta)
        if phase == "G":
            training_stats.report("Loss/scores/fake", -paired_delta)
            training_stats.report("Loss/signs/fake", (-paired_delta).sign())
            training_stats.report("Loss/G/pair", pair_loss)
            training_stats.report("Loss/G/list", list_loss)
            training_stats.report(
                "Loss/G/infonce",
                list_loss if self.lambda_list > 0 else zero_vector,
            )
            training_stats.report(
                "Loss/list_tau",
                torch.as_tensor(self.list_tau if self.lambda_list > 0 else 0.0, device=device),
            )
            training_stats.report(
                "Loss/list_type_code",
                torch.as_tensor(1.0 if self.lambda_list > 0 else 0.0, device=device),
            )
            training_stats.report("Loss/G/local_rank", zero_scalar)
            training_stats.report("Loss/G/path_rank", zero_scalar)
        else:
            training_stats.report("Loss/scores/real", paired_delta)
            training_stats.report("Loss/signs/real", paired_delta.sign())
            training_stats.report("Loss/D/pair", pair_loss)
            training_stats.report("Loss/D/list", list_loss)
            training_stats.report(
                "Loss/D/infonce",
                list_loss if self.lambda_list > 0 else zero_vector,
            )
            training_stats.report(
                "Loss/list_tau",
                torch.as_tensor(self.list_tau if self.lambda_list > 0 else 0.0, device=device),
            )
            training_stats.report(
                "Loss/list_type_code",
                torch.as_tensor(1.0 if self.lambda_list > 0 else 0.0, device=device),
            )

    def _class_ids(self, c):
        if c is None or c.ndim == 0 or (c.ndim == 2 and c.shape[1] == 0):
            return None
        if c.ndim == 1:
            return c.to(torch.long)
        return c.argmax(dim=1)

    def _compute_local_rank_loss(self, real_scores, fake_scores, real_features, fake_features, real_c):
        real_features = F.normalize(real_features.detach(), dim=1)
        fake_features = F.normalize(fake_features.detach(), dim=1)
        class_ids = self._class_ids(real_c)
        losses = []
        for fake_idx in range(fake_scores.shape[0]):
            if class_ids is None:
                same_real = torch.ones(
                    real_scores.shape[0], dtype=torch.bool, device=real_scores.device
                )
                same_fake = torch.ones(
                    fake_scores.shape[0], dtype=torch.bool, device=fake_scores.device
                )
            else:
                same_real = class_ids == class_ids[fake_idx]
                same_fake = class_ids == class_ids[fake_idx]
            real_indices = torch.where(same_real)[0]
            fake_indices = torch.where(same_fake)[0]
            if real_indices.numel() == 0 or fake_indices.numel() < 2:
                continue
            real_distances = 1.0 - torch.matmul(
                real_features[real_indices], fake_features[fake_idx]
            )
            anchor_index = real_indices[real_distances.argmin()]
            fake_distances = 1.0 - torch.matmul(
                fake_features[fake_indices], real_features[anchor_index]
            )
            topk = min(self.local_rank_k, fake_indices.numel())
            if topk < 2:
                continue
            ordered_fake = fake_indices[torch.argsort(fake_distances)[:topk]]
            deltas = real_scores[anchor_index] - fake_scores[ordered_fake]
            losses.append(F.softplus(-(deltas[1:] - deltas[:-1])))
        if len(losses) == 0:
            return real_scores.new_zeros([])
        return torch.cat(losses).mean()

    def _compute_path_rank_loss(self, real_img, fake_img, real_c):
        rank_imgs = make_rank_list(
            real_img,
            fake_img.detach(),
            k=self.path_rank_k,
            mode=self.path_rank_mode,
            alpha_dist=self.path_rank_alpha_dist,
        )
        bsz, k, c, h, w = rank_imgs.shape
        rank_imgs_flat = rank_imgs.reshape(bsz * k, c, h, w)
        rank_c = real_c.repeat_interleave(k, dim=0)
        rank_scores = self.run_D(rank_imgs_flat, rank_c, augment=False)
        rank_scores = rank_scores.reshape(bsz, k)

        if self.path_rank_loss_type == "listmle":
            path_loss = listmle_loss(rank_scores)
        elif self.path_rank_loss_type == "pairwise_logistic":
            path_loss = pairwise_logistic_loss(rank_scores)
        elif self.path_rank_loss_type == "pairwise_hinge":
            path_loss = pairwise_hinge_loss(rank_scores, margin=self.path_rank_margin)
        else:
            raise ValueError(f"Unknown path_rank_loss_type: {self.path_rank_loss_type}")

        score_reg = rank_scores.square().mean()
        if self.path_rank_score_reg > 0:
            path_loss = path_loss + self.path_rank_score_reg * score_reg
        return path_loss, score_reg

    def _accumulate_gradients_impl(self, phase, real_img, real_c, gen_z, gamma, gain):
        if phase == "G":
            fake_img = self.G(gen_z, real_c)
            real_scores = self.run_D(real_img.detach(), real_c, augment=True)
            fake_scores = self.run_D(fake_img, real_c, augment=True)
            paired_delta = pairwise_delta(real_scores, fake_scores)
            zero_vector = torch.zeros_like(paired_delta)
            pair_loss = (
                pairwise_generator_loss(paired_delta, margin=self.pair_margin)
                if self.lambda_pair > 0
                else zero_vector
            )
            list_loss = zero_vector
            if self.lambda_list > 0:
                delta_matrix = allpairs_delta(real_scores, fake_scores)
                list_loss = infonce_generator_loss(delta_matrix, tau=self.list_tau)

            pair_term = self.lambda_pair * pair_loss.mean()
            list_term = self.lambda_list * list_loss.mean()
            total_loss = pair_term + list_term
            (gain * total_loss).backward()

            self._report_common_stats("G", pair_loss, list_loss, paired_delta, real_img.device)
            training_stats.report("Loss/G/pair_weighted", pair_term)
            training_stats.report("Loss/G/list_weighted", list_term)
            training_stats.report("Loss/G/loss", pair_term + list_term)
            training_stats.report("Loss/G/adv_weighted", pair_term + list_term)
            training_stats.report("Loss/G/total", total_loss)
            return

        if phase == "D":
            real_img = real_img.detach().requires_grad_(self.use_r1_penalty)
            fake_img = self.G(gen_z, real_c).detach().requires_grad_(self.use_r2_penalty)
            need_features = self.lambda_local_rank > 0
            if need_features:
                real_scores, real_features = self.run_D(
                    real_img, real_c, augment=True, return_features=True
                )
                fake_scores, fake_features = self.run_D(
                    fake_img, real_c, augment=True, return_features=True
                )
            else:
                real_scores = self.run_D(real_img, real_c, augment=True)
                fake_scores = self.run_D(fake_img, real_c, augment=True)
                real_features = None
                fake_features = None

            paired_delta = pairwise_delta(real_scores, fake_scores)
            zero_vector = torch.zeros_like(paired_delta)
            zero_scalar = torch.zeros([], device=real_img.device)
            pair_loss = (
                pairwise_discriminator_loss(paired_delta, margin=self.pair_margin)
                if self.lambda_pair > 0
                else zero_vector
            )
            list_loss = zero_vector
            if self.lambda_list > 0:
                delta_matrix = allpairs_delta(real_scores, fake_scores)
                list_loss = infonce_discriminator_loss(delta_matrix, tau=self.list_tau)

            local_rank_loss = zero_scalar
            if self.lambda_local_rank > 0:
                local_rank_loss = self._compute_local_rank_loss(
                    real_scores, fake_scores, real_features, fake_features, real_c
                )

            path_rank_loss = zero_scalar
            path_score_reg = zero_scalar
            if self.path_rank_reg:
                path_rank_loss, path_score_reg = self._compute_path_rank_loss(
                    real_img.detach(), fake_img.detach(), real_c
                )

            r1_penalty = torch.zeros_like(real_scores)
            r2_penalty = torch.zeros_like(fake_scores)
            if self.use_r1_penalty:
                real_scores_for_penalty = real_scores
                if self.use_non_aug_gp:
                    real_scores_for_penalty = self.run_D(
                        real_img, real_c, augment=False, return_features=False
                    )
                r1_penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
                    real_img, real_scores_for_penalty
                )
            if self.use_r2_penalty:
                fake_scores_for_penalty = fake_scores
                if self.use_non_aug_gp:
                    fake_scores_for_penalty = self.run_D(
                        fake_img, real_c, augment=False, return_features=False
                    )
                r2_penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
                    fake_img, fake_scores_for_penalty
                )

            pair_term = self.lambda_pair * pair_loss.mean()
            list_term = self.lambda_list * list_loss.mean()
            adv_term = pair_term + list_term
            local_term = self.lambda_local_rank * local_rank_loss
            path_term = self.lambda_path_rank * path_rank_loss
            r1_term = (gamma / 2) * r1_penalty.mean() if self.use_r1_penalty else zero_scalar
            r2_term = (gamma / 2) * r2_penalty.mean() if self.use_r2_penalty else zero_scalar
            base_total = adv_term + r1_term + r2_term
            total_loss = base_total + local_term + path_term
            (gain * total_loss).backward()

            self._report_common_stats("D", pair_loss, list_loss, paired_delta, real_img.device)
            training_stats.report("Loss/r1_penalty", r1_penalty)
            training_stats.report("Loss/r2_penalty", r2_penalty)
            training_stats.report("Loss/D/local_rank", local_rank_loss)
            training_stats.report("Loss/D/path_rank", path_rank_loss)
            training_stats.report("Loss/D/path_score_reg", path_score_reg)
            training_stats.report("Loss/D/pair_weighted", pair_term)
            training_stats.report("Loss/D/list_weighted", list_term)
            training_stats.report("Loss/D/loss", adv_term)
            training_stats.report("Loss/D/adv_weighted", adv_term)
            training_stats.report("Loss/D/local_rank_weighted", local_term)
            training_stats.report("Loss/D/path_rank_weighted", path_term)
            training_stats.report("Loss/D/r1_weighted", r1_term)
            training_stats.report("Loss/D/r2_weighted", r2_term)
            training_stats.report("Loss/D/base_total", base_total)
            training_stats.report("Loss/D/total", total_loss)
            return

        raise ValueError(f"Unknown phase: {phase}")


# ----------------------------------------------------------------------------
