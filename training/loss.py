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
import torch.distributed
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


def build_local_coupling(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    k: int,
    real_class_ids: torch.Tensor | None = None,
    fake_class_ids: torch.Tensor | None = None,
    return_info: bool = False,
) -> tuple:
    """Build sparse kNN coupling from fakes to nearest reals in feature space.

    Returns (neighbor_indices [B_f, k], coupling_weights [B_f, k]).
    """
    real_norm = F.normalize(real_features.detach().to(torch.float32), dim=1)
    fake_norm = F.normalize(fake_features.detach().to(torch.float32), dim=1)
    sim = torch.matmul(fake_norm, real_norm.t())
    k = min(k, real_norm.shape[0])
    masked_sim = sim
    available_neighbors = torch.full(
        [fake_norm.shape[0]], fill_value=k, device=fake_norm.device, dtype=torch.int64
    )
    fallback_rows = torch.zeros(
        [fake_norm.shape[0]], device=fake_norm.device, dtype=torch.bool
    )
    class_masked = False
    same_class = None
    if real_class_ids is not None and fake_class_ids is not None:
        real_class_ids = real_class_ids.detach().to(fake_norm.device).reshape(-1)
        fake_class_ids = fake_class_ids.detach().to(fake_norm.device).reshape(-1)
        if real_class_ids.shape[0] != real_norm.shape[0]:
            raise ValueError("real_class_ids must align with real_features")
        if fake_class_ids.shape[0] != fake_norm.shape[0]:
            raise ValueError("fake_class_ids must align with fake_features")
        class_masked = True
        same_class = fake_class_ids.unsqueeze(1).eq(real_class_ids.unsqueeze(0))
        fallback_rows = ~same_class.any(dim=1)
        available_neighbors = same_class.sum(dim=1).clamp(max=k)
        masked_sim = sim.masked_fill(~same_class, float("-inf"))
        if fallback_rows.any():
            masked_sim = torch.where(fallback_rows.unsqueeze(1), sim, masked_sim)
            available_neighbors = torch.where(
                fallback_rows,
                torch.full_like(available_neighbors, k),
                available_neighbors,
            )

    topk_sim, topk_idx = masked_sim.topk(k, dim=1)
    coupling_weights = F.softmax(topk_sim, dim=1)
    info = None
    if return_info:
        topk_sim_unmasked = sim.gather(1, topk_idx)
        effective_neighbors = (coupling_weights > 0).sum(dim=1).to(torch.float32)
        weight_entropy = -(
            coupling_weights * coupling_weights.clamp_min(1e-12).log()
        ).sum(dim=1)
        info = dict(
            class_masked=torch.full(
                [fake_norm.shape[0]],
                fill_value=1.0 if class_masked else 0.0,
                device=fake_norm.device,
                dtype=torch.float32,
            ),
            available_neighbors=available_neighbors.to(torch.float32),
            effective_neighbors=effective_neighbors,
            fallback_rows=fallback_rows.to(torch.float32),
            top1_similarity=topk_sim_unmasked[:, 0],
            weighted_similarity=(coupling_weights * topk_sim_unmasked).sum(dim=1),
            top1_weight=coupling_weights[:, 0],
            max_weight=coupling_weights.max(dim=1).values,
            weight_entropy=weight_entropy,
        )
        if same_class is not None:
            neighbor_same_class = same_class.gather(1, topk_idx).to(torch.float32)
            info["same_class_mass"] = (coupling_weights * neighbor_same_class).sum(dim=1)
            info["cross_class_mass"] = (
                coupling_weights * (1.0 - neighbor_same_class)
            ).sum(dim=1)
            info["top1_same_class"] = neighbor_same_class[:, 0]
        else:
            ones = torch.ones([fake_norm.shape[0]], device=fake_norm.device, dtype=torch.float32)
            zeros = torch.zeros_like(ones)
            info["same_class_mass"] = ones
            info["cross_class_mass"] = zeros
            info["top1_same_class"] = ones
    if return_info:
        return topk_idx, coupling_weights, info
    return topk_idx, coupling_weights


def local_delta(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """Local critic differences Δ_ij = s(real_{N(j,i)}) - s(fake_j). Returns [B_f, k]."""
    return real_scores[neighbor_indices] - fake_scores.unsqueeze(1)


def local_pairwise_discriminator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Coupling-weighted pairwise D loss. Returns [B_f] per-fake losses."""
    return (coupling_weights * F.softplus(margin - delta)).sum(dim=1)


def local_pairwise_generator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Coupling-weighted pairwise G loss. Returns [B_f] per-fake losses."""
    return (coupling_weights * F.softplus(margin + delta)).sum(dim=1)


def local_listwise_discriminator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Local-coupled listwise D loss: log(1 + Σ π̃_ij exp(-Δ_ij/τ)). Returns [B_f]."""
    if tau <= 0:
        raise ValueError(f"Temperature must be positive, got {tau}")
    neg_scaled = -delta / tau
    max_val = neg_scaled.detach().max(dim=1, keepdim=True).values.clamp(min=0)
    stable_sum = torch.exp(-max_val.squeeze(1)) + (
        coupling_weights * torch.exp(neg_scaled - max_val)
    ).sum(dim=1)
    return max_val.squeeze(1) + torch.log(stable_sum)


def local_listwise_generator_loss(
    delta: torch.Tensor,
    coupling_weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Local-coupled listwise G loss: log(1 + Σ π̃_ij exp(+Δ_ij/τ)). Returns [B_f]."""
    if tau <= 0:
        raise ValueError(f"Temperature must be positive, got {tau}")
    pos_scaled = delta / tau
    max_val = pos_scaled.detach().max(dim=1, keepdim=True).values.clamp(min=0)
    stable_sum = torch.exp(-max_val.squeeze(1)) + (
        coupling_weights * torch.exp(pos_scaled - max_val)
    ).sum(dim=1)
    return max_val.squeeze(1) + torch.log(stable_sum)


def local_coupled_discriminator_loss_with_grads(
    real_scores, fake_scores, neighbor_indices, coupling_weights,
    lambda_pair, pair_margin, lambda_list, list_tau,
):
    """Return (loss_scalar, loss_vector, grad_real, grad_fake) for local-coupled D loss."""
    real_var = real_scores.detach().to(torch.float32).requires_grad_(True)
    fake_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    delta = local_delta(real_var, fake_var, neighbor_indices)
    loss_terms = torch.zeros(fake_var.shape[0], device=fake_var.device)
    if lambda_pair > 0:
        loss_terms = loss_terms + lambda_pair * local_pairwise_discriminator_loss(
            delta, coupling_weights, margin=pair_margin
        )
    if lambda_list > 0:
        loss_terms = loss_terms + lambda_list * local_listwise_discriminator_loss(
            delta, coupling_weights, tau=list_tau
        )
    loss_value = loss_terms.mean()
    grad_real, grad_fake = torch.autograd.grad(loss_value, [real_var, fake_var])
    return loss_value.detach(), loss_terms.detach(), grad_real.detach(), grad_fake.detach()


def local_coupled_generator_loss_with_grads(
    real_scores, fake_scores, neighbor_indices, coupling_weights,
    lambda_pair, pair_margin, lambda_list, list_tau,
):
    """Return (loss_scalar, loss_vector, grad_fake) for local-coupled G loss."""
    real_var = real_scores.detach().to(torch.float32)
    fake_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    delta = local_delta(real_var, fake_var, neighbor_indices)
    loss_terms = torch.zeros(fake_var.shape[0], device=fake_var.device)
    if lambda_pair > 0:
        loss_terms = loss_terms + lambda_pair * local_pairwise_generator_loss(
            delta, coupling_weights, margin=pair_margin
        )
    if lambda_list > 0:
        loss_terms = loss_terms + lambda_list * local_listwise_generator_loss(
            delta, coupling_weights, tau=list_tau
        )
    loss_value = loss_terms.mean()
    (grad_fake,) = torch.autograd.grad(loss_value, [fake_var])
    return loss_value.detach(), loss_terms.detach(), grad_fake.detach()


# ----------------------------------------------------------------------------


def infonce_discriminator_loss_with_grads(
    real_scores: torch.Tensor, fake_scores: torch.Tensor, tau: float = 0.07
):
    """Return scalar D list loss, per-real losses, and score gradients."""
    real_scores_var = real_scores.detach().to(torch.float32).requires_grad_(True)
    fake_scores_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    loss_vector = infonce_discriminator_loss(
        allpairs_delta(real_scores_var, fake_scores_var), tau=tau
    )
    loss_value = loss_vector.mean()
    grad_real, grad_fake = torch.autograd.grad(loss_value, [real_scores_var, fake_scores_var])
    return (
        loss_value.detach(),
        loss_vector.detach(),
        grad_real.detach(),
        grad_fake.detach(),
    )


def infonce_generator_loss_with_grads(
    real_scores: torch.Tensor, fake_scores: torch.Tensor, tau: float = 0.07
):
    """Return scalar G list loss, per-fake losses, and fake-score gradients."""
    real_scores_var = real_scores.detach().to(torch.float32)
    fake_scores_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    loss_vector = infonce_generator_loss(
        allpairs_delta(real_scores_var, fake_scores_var), tau=tau
    )
    loss_value = loss_vector.mean()
    (grad_fake,) = torch.autograd.grad(loss_value, [fake_scores_var])
    return loss_value.detach(), loss_vector.detach(), grad_fake.detach()


def local_rank_loss_with_grads(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    fake_scores: torch.Tensor,
    class_ids: torch.Tensor | None,
    k: int,
):
    """Return scalar local-rank loss, adjacent losses, and fake-score gradients."""
    fake_scores_var = fake_scores.detach().to(torch.float32).requires_grad_(True)
    real_features = F.normalize(real_features.detach().to(torch.float32), dim=1)
    fake_features = F.normalize(fake_features.detach().to(torch.float32), dim=1)
    adjacent_losses = []

    for fake_idx in range(fake_scores_var.shape[0]):
        if class_ids is None:
            same_real = torch.ones(
                real_features.shape[0], dtype=torch.bool, device=real_features.device
            )
            same_fake = torch.ones(
                fake_features.shape[0], dtype=torch.bool, device=fake_features.device
            )
        else:
            same_real = class_ids == class_ids[fake_idx]
            same_fake = class_ids == class_ids[fake_idx]

        real_indices = torch.where(same_real)[0]
        fake_indices = torch.where(same_fake)[0]
        if real_indices.numel() == 0 or fake_indices.numel() < 2:
            continue

        real_distances = 1.0 - torch.matmul(real_features[real_indices], fake_features[fake_idx])
        anchor_index = real_indices[real_distances.argmin()]
        fake_distances = 1.0 - torch.matmul(fake_features[fake_indices], real_features[anchor_index])
        topk = min(k, fake_indices.numel())
        if topk < 2:
            continue

        ordered_fake = fake_indices[torch.argsort(fake_distances)[:topk]]
        ordered_scores = fake_scores_var[ordered_fake]
        adjacent_losses.append(F.softplus(ordered_scores[1:] - ordered_scores[:-1]))

    if len(adjacent_losses) == 0:
        zero_scalar = fake_scores.new_zeros([], dtype=torch.float32)
        zero_vector = fake_scores.new_zeros([0], dtype=torch.float32)
        zero_grad = fake_scores.new_zeros(fake_scores.shape, dtype=torch.float32)
        return zero_scalar, zero_vector, zero_grad

    adjacent_loss_vector = torch.cat(adjacent_losses)
    loss_value = adjacent_loss_vector.mean()
    (grad_fake,) = torch.autograd.grad(loss_value, [fake_scores_var])
    return loss_value.detach(), adjacent_loss_vector.detach(), grad_fake.detach()


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
        coupling_k=0,            # 0 = no coupling (backward compat); >0 = kNN coupling
        lambda_list_d=None,      # D-side listwise weight (overrides lambda_list for D)
        lambda_list_g=None,      # G-side listwise weight (overrides lambda_list for G)
        path_rank_reg=False,
        path_rank_k=8,
        path_rank_loss_type="listmle",
        lambda_path_rank=0.1,
        path_rank_mode="intrpl",
        path_rank_alpha_dist="linear",
        path_rank_margin=1.0,
        path_rank_score_reg=0.0,
        # SPM (Symmetric Pushforward Matching) parameters
        spm_enable=False,
        spm_transforms=("downsample8",),
        spm_weights=(0.25,),
        spm_margin=1.0,
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

        self.coupling_k = int(coupling_k)
        if lambda_list_d is not None or lambda_list_g is not None:
            self.lambda_list_d = float(lambda_list_d if lambda_list_d is not None else self.lambda_list)
            self.lambda_list_g = float(lambda_list_g if lambda_list_g is not None else 0.0)
        else:
            self.lambda_list_d = self.lambda_list
            self.lambda_list_g = self.lambda_list

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

        # SPM storage
        self.spm_enable = bool(spm_enable)
        self.spm_transforms = tuple(spm_transforms) if spm_transforms else ()
        self.spm_weights = tuple(float(w) for w in spm_weights) if spm_weights else ()
        self.spm_margin = float(spm_margin)

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
        if self.coupling_k < 0:
            raise ValueError("coupling_k must be non-negative")
        if self.lambda_list_d < 0:
            raise ValueError("lambda_list_d must be non-negative")
        if self.lambda_list_g < 0:
            raise ValueError("lambda_list_g must be non-negative")

        if rank_augment:
            warnings.warn(
                "rank_augment is deprecated and ignored; path_rank_reg always uses non-augmented views.",
                stacklevel=2,
            )

        self._coupled_phase_buffer = None

    def _requires_coupling(self):
        return self.coupling_k > 0

    def _requires_full_batch(self, phase=None):
        if self._requires_coupling():
            return True  # coupling needs full-batch features
        if phase == "G":
            return self.lambda_list > 0
        if phase == "D":
            return self.lambda_list > 0 or self.lambda_local_rank > 0
        return self.lambda_list > 0 or self.lambda_local_rank > 0

    def set_list_weights(self, lambda_list_d=None, lambda_list_g=None):
        """Update listwise weights (called by training loop scheduler)."""
        if lambda_list_d is not None:
            self.lambda_list_d = float(lambda_list_d)
        if lambda_list_g is not None:
            self.lambda_list_g = float(lambda_list_g)

    def _phase_list_lambda(self, phase):
        if self._requires_coupling():
            return self.lambda_list_g if phase == "G" else self.lambda_list_d
        return self.lambda_list

    def _as_scores(self, logits):
        return self.trainer._as_vector(logits)

    def _world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def _rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _get_layout(self, local_count, device):
        if self._world_size() == 1:
            return [int(local_count)], slice(0, int(local_count))
        local_size = torch.as_tensor([int(local_count)], device=device, dtype=torch.int64)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(self._world_size())]
        torch.distributed.all_gather(gathered_sizes, local_size)
        sizes = [int(x.item()) for x in gathered_sizes]
        offset = sum(sizes[: self._rank()])
        return sizes, slice(offset, offset + int(local_count))

    def _all_gather_rows(self, tensor, sizes):
        if tensor is None or self._world_size() == 1:
            return tensor
        max_size = max(sizes)
        if tensor.shape[0] < max_size:
            padding = torch.zeros(
                (max_size - tensor.shape[0],) + tensor.shape[1:],
                device=tensor.device,
                dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, padding], dim=0)
        gathered = [torch.zeros_like(tensor) for _ in sizes]
        torch.distributed.all_gather(gathered, tensor.contiguous())
        return torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)

    def _weighted_mean_scalar(self, values, counts, device):
        if len(values) == 0:
            return torch.zeros([], device=device)
        numerator = torch.zeros([], device=device)
        denominator = 0
        for value, count in zip(values, counts):
            numerator = numerator + value * float(count)
            denominator += int(count)
        if denominator == 0:
            return torch.zeros([], device=device)
        return numerator / denominator

    def _capture_rng_state(self, device):
        state = {"cpu": torch.random.get_rng_state()}
        if device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(device)
        return state

    def _restore_rng_state(self, device, state):
        torch.random.set_rng_state(state["cpu"])
        if device.type == "cuda" and "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], device)

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
        if self._requires_full_batch(phase):
            if self._coupled_phase_buffer is None:
                self._coupled_phase_buffer = {
                    "phase": phase,
                    "gamma": gamma,
                    "real_imgs": [],
                    "real_cs": [],
                    "gen_zs": [],
                    "gains": [],
                }
            self._coupled_phase_buffer["real_imgs"].append(real_img)
            self._coupled_phase_buffer["real_cs"].append(real_c)
            self._coupled_phase_buffer["gen_zs"].append(gen_z)
            self._coupled_phase_buffer["gains"].append(gain)
            return
        self._accumulate_gradients_impl(phase, real_img, real_c, gen_z, gamma, gain)

    def finalize_accumulation(self):
        """Flush buffered micro-batches for coupled losses with replay."""
        if self._coupled_phase_buffer is None:
            return
        buf = self._coupled_phase_buffer
        self._coupled_phase_buffer = None
        rng_state = self._capture_rng_state(buf["real_imgs"][0].device)
        metadata = self._collect_coupled_metadata(buf)
        self._restore_rng_state(buf["real_imgs"][0].device, rng_state)
        replay = self._prepare_coupled_replay(buf["phase"], metadata, sum(buf["gains"]))
        replay_stats = self._replay_coupled_gradients(buf, replay)
        self._report_coupled_stats(buf["phase"], metadata, replay, replay_stats)

    def _report_weight_stats(self, phase, device):
        list_lambda = self._phase_list_lambda(phase)
        training_stats.report(
            "Loss/weights/lambda_pair",
            torch.as_tensor(self.lambda_pair, device=device),
        )
        training_stats.report(
            "Loss/weights/pair_margin",
            torch.as_tensor(self.pair_margin, device=device),
        )
        training_stats.report(
            "Loss/weights/lambda_list",
            torch.as_tensor(self.lambda_list, device=device),
        )
        training_stats.report(
            "Loss/weights/lambda_list_d",
            torch.as_tensor(self.lambda_list_d, device=device),
        )
        training_stats.report(
            "Loss/weights/lambda_list_g",
            torch.as_tensor(self.lambda_list_g, device=device),
        )
        training_stats.report(
            "Loss/weights/active_lambda_list",
            torch.as_tensor(list_lambda, device=device),
        )
        training_stats.report(
            "Loss/weights/lambda_local_rank",
            torch.as_tensor(self.lambda_local_rank, device=device),
        )
        training_stats.report(
            "Loss/weights/lambda_path_rank",
            torch.as_tensor(self.lambda_path_rank, device=device),
        )
        training_stats.report(
            "Loss/config/coupling_enabled",
            torch.as_tensor(1.0 if self._requires_coupling() else 0.0, device=device),
        )
        training_stats.report(
            "Loss/config/coupling_k",
            torch.as_tensor(float(self.coupling_k), device=device),
        )
        training_stats.report(
            "Loss/config/local_rank_enabled",
            torch.as_tensor(1.0 if self.lambda_local_rank > 0 else 0.0, device=device),
        )
        training_stats.report(
            "Loss/config/local_rank_k",
            torch.as_tensor(float(self.local_rank_k), device=device),
        )
        training_stats.report(
            "Loss/config/path_rank_enabled",
            torch.as_tensor(1.0 if self.path_rank_reg else 0.0, device=device),
        )
        training_stats.report(
            "Loss/config/path_rank_k",
            torch.as_tensor(float(self.path_rank_k), device=device),
        )

    def _report_common_stats(
        self, phase, real_scores, fake_scores, pair_loss, list_loss, paired_delta, device
    ):
        self._report_weight_stats(phase, device)
        zero_scalar = torch.zeros([], device=device)
        zero_vector = torch.zeros_like(paired_delta)
        list_lambda = self._phase_list_lambda(phase)
        training_stats.report("Loss/scores/real", real_scores)
        training_stats.report("Loss/scores/fake", fake_scores)
        training_stats.report("Loss/signs/real", real_scores.sign())
        training_stats.report("Loss/signs/fake", fake_scores.sign())
        training_stats.report("Loss/delta/pair", paired_delta)
        training_stats.report("Loss/signs/delta", paired_delta.sign())
        training_stats.report(
            "Loss/list_tau",
            torch.as_tensor(self.list_tau if list_lambda > 0 else 0.0, device=device),
        )
        training_stats.report(
            "Loss/list_type_code",
            torch.as_tensor(1.0 if list_lambda > 0 else 0.0, device=device),
        )
        if phase == "G":
            training_stats.report("Loss/G/pair", pair_loss)
            training_stats.report("Loss/G/list", list_loss)
            training_stats.report(
                "Loss/G/infonce",
                list_loss if list_lambda > 0 else zero_vector,
            )
            training_stats.report("Loss/G/local_rank", zero_scalar)
            training_stats.report("Loss/G/path_rank", zero_scalar)
        else:
            training_stats.report("Loss/D/pair", pair_loss)
            training_stats.report("Loss/D/list", list_loss)
            training_stats.report(
                "Loss/D/infonce",
                list_loss if list_lambda > 0 else zero_vector,
            )

    def _report_coupling_stats(self, coupling_info):
        if coupling_info is None:
            return
        training_stats.report("Loss/coupling/class_masked", coupling_info["class_masked"])
        training_stats.report(
            "Loss/coupling/available_neighbors", coupling_info["available_neighbors"]
        )
        training_stats.report(
            "Loss/coupling/effective_neighbors", coupling_info["effective_neighbors"]
        )
        training_stats.report("Loss/coupling/fallback_rows", coupling_info["fallback_rows"])
        training_stats.report("Loss/coupling/top1_similarity", coupling_info["top1_similarity"])
        training_stats.report(
            "Loss/coupling/weighted_similarity", coupling_info["weighted_similarity"]
        )
        training_stats.report("Loss/coupling/top1_weight", coupling_info["top1_weight"])
        training_stats.report("Loss/coupling/max_weight", coupling_info["max_weight"])
        training_stats.report("Loss/coupling/weight_entropy", coupling_info["weight_entropy"])
        training_stats.report(
            "Loss/coupling/same_class_mass", coupling_info["same_class_mass"]
        )
        training_stats.report(
            "Loss/coupling/cross_class_mass", coupling_info["cross_class_mass"]
        )
        training_stats.report(
            "Loss/coupling/top1_same_class", coupling_info["top1_same_class"]
        )

    def _class_ids(self, c):
        if c is None or c.ndim == 0 or (c.ndim == 2 and c.shape[1] == 0):
            return None
        if c.ndim == 1:
            return c.to(torch.long)
        return c.argmax(dim=1)

    def _compute_local_rank_loss(self, real_scores, fake_scores, real_features, fake_features, real_c):
        del real_scores
        real_features = F.normalize(real_features.detach().to(torch.float32), dim=1)
        fake_features = F.normalize(fake_features.detach().to(torch.float32), dim=1)
        class_ids = self._class_ids(real_c)
        adjacent_losses = []
        for fake_idx in range(fake_scores.shape[0]):
            if class_ids is None:
                same_real = torch.ones(
                    real_features.shape[0], dtype=torch.bool, device=real_features.device
                )
                same_fake = torch.ones(
                    fake_features.shape[0], dtype=torch.bool, device=fake_features.device
                )
            else:
                same_real = class_ids == class_ids[fake_idx]
                same_fake = class_ids == class_ids[fake_idx]

            real_indices = torch.where(same_real)[0]
            fake_indices = torch.where(same_fake)[0]
            if real_indices.numel() == 0 or fake_indices.numel() < 2:
                continue

            real_distances = 1.0 - torch.matmul(real_features[real_indices], fake_features[fake_idx])
            anchor_index = real_indices[real_distances.argmin()]
            fake_distances = 1.0 - torch.matmul(fake_features[fake_indices], real_features[anchor_index])
            topk = min(self.local_rank_k, fake_indices.numel())
            if topk < 2:
                continue

            ordered_fake = fake_indices[torch.argsort(fake_distances)[:topk]]
            ordered_scores = fake_scores[ordered_fake]
            adjacent_losses.append(F.softplus(ordered_scores[1:] - ordered_scores[:-1]))

        if len(adjacent_losses) == 0:
            return fake_scores.new_zeros([])
        return torch.cat(adjacent_losses).mean()

    def _collect_coupled_metadata(self, buf):
        phase = buf["phase"]
        local = dict(
            aug_real_scores=[],
            aug_fake_scores=[],
            clean_real_features=[],
            clean_fake_features=[],
            clean_fake_scores=[],
            class_ids=[],
        )

        with torch.no_grad():
            for real_img, real_c, gen_z in zip(buf["real_imgs"], buf["real_cs"], buf["gen_zs"]):
                if phase == "G":
                    fake_img = self.G(gen_z, real_c)
                else:
                    fake_img = self.G(gen_z, real_c).detach()

                local["aug_real_scores"].append(
                    self.run_D(real_img.detach(), real_c, augment=True).to(torch.float32)
                )
                local["aug_fake_scores"].append(
                    self.run_D(fake_img, real_c, augment=True).to(torch.float32)
                )

                need_clean_features = (phase == "D" and self.lambda_local_rank > 0) or self._requires_coupling()
                if need_clean_features:
                    _, real_features = self.run_D(
                        real_img.detach(), real_c, augment=False, return_features=True
                    )
                    clean_fake_scores, fake_features = self.run_D(
                        fake_img, real_c, augment=False, return_features=True
                    )
                    local["clean_real_features"].append(real_features.to(torch.float32))
                    local["clean_fake_features"].append(fake_features.to(torch.float32))
                    local["clean_fake_scores"].append(clean_fake_scores.to(torch.float32))
                    class_ids = self._class_ids(real_c)
                    if class_ids is not None:
                        local["class_ids"].append(class_ids.to(torch.long))

        local["aug_real_scores"] = torch.cat(local["aug_real_scores"], dim=0)
        local["aug_fake_scores"] = torch.cat(local["aug_fake_scores"], dim=0)
        need_clean_features = (phase == "D" and self.lambda_local_rank > 0) or self._requires_coupling()
        if need_clean_features:
            local["clean_real_features"] = torch.cat(local["clean_real_features"], dim=0)
            local["clean_fake_features"] = torch.cat(local["clean_fake_features"], dim=0)
            local["clean_fake_scores"] = torch.cat(local["clean_fake_scores"], dim=0)
            local["class_ids"] = (
                torch.cat(local["class_ids"], dim=0) if len(local["class_ids"]) > 0 else None
            )
        else:
            local["clean_real_features"] = None
            local["clean_fake_features"] = None
            local["clean_fake_scores"] = None
            local["class_ids"] = None

        sizes, local_slice = self._get_layout(
            local["aug_real_scores"].shape[0], local["aug_real_scores"].device
        )
        global_meta = dict(
            aug_real_scores=self._all_gather_rows(local["aug_real_scores"], sizes),
            aug_fake_scores=self._all_gather_rows(local["aug_fake_scores"], sizes),
            clean_real_features=self._all_gather_rows(local["clean_real_features"], sizes),
            clean_fake_features=self._all_gather_rows(local["clean_fake_features"], sizes),
            clean_fake_scores=self._all_gather_rows(local["clean_fake_scores"], sizes),
            class_ids=self._all_gather_rows(local["class_ids"], sizes),
        )
        return dict(local=local, global_meta=global_meta, sizes=sizes, local_slice=local_slice)

    def _prepare_coupled_replay(self, phase, metadata, total_gain):
        local = metadata["local"]
        global_meta = metadata["global_meta"]
        local_slice = metadata["local_slice"]
        device = local["aug_real_scores"].device
        zero_scalar = torch.zeros([], device=device, dtype=torch.float32)
        zero_vector = torch.zeros_like(local["aug_fake_scores"])

        replay = dict(
            total_gain=float(total_gain),
            replay_scale=float(total_gain) * float(self._world_size()),
            pair_loss=zero_vector,
            pair_loss_value=zero_scalar,
            list_loss=zero_vector,
            list_loss_value=zero_scalar,
            grad_real_aug=None,
            grad_fake_aug=None,
            local_rank_loss=zero_scalar,
            local_rank_adjacent_losses=torch.zeros([0], device=device, dtype=torch.float32),
            grad_fake_clean=None,
            coupling_info=None,
        )

        coupling = None
        if self._requires_coupling():
            coupling = build_local_coupling(
                global_meta["clean_real_features"],
                global_meta["clean_fake_features"],
                k=self.coupling_k,
                real_class_ids=global_meta["class_ids"],
                fake_class_ids=global_meta["class_ids"],
                return_info=True,
            )

        paired_delta = pairwise_delta(local["aug_real_scores"], local["aug_fake_scores"])
        if phase == "G":
            if self._requires_coupling():
                neighbor_indices, coupling_weights, coupling_info = coupling
                global_delta = local_delta(
                    global_meta["aug_real_scores"],
                    global_meta["aug_fake_scores"],
                    neighbor_indices,
                )
                global_pair_loss = (
                    local_pairwise_generator_loss(
                        global_delta, coupling_weights, margin=self.pair_margin
                    )
                    if self.lambda_pair > 0
                    else torch.zeros_like(global_meta["aug_fake_scores"])
                )
                global_list_loss = (
                    local_listwise_generator_loss(
                        global_delta, coupling_weights, tau=self.list_tau
                    )
                    if self.lambda_list_g > 0
                    else torch.zeros_like(global_meta["aug_fake_scores"])
                )
                _, loss_vector, grad_fake = local_coupled_generator_loss_with_grads(
                    global_meta["aug_real_scores"],
                    global_meta["aug_fake_scores"],
                    neighbor_indices, coupling_weights,
                    lambda_pair=self.lambda_pair, pair_margin=self.pair_margin,
                    lambda_list=self.lambda_list_g, list_tau=self.list_tau,
                )
                del loss_vector
                replay["pair_loss"] = global_pair_loss[local_slice]
                replay["pair_loss_value"] = global_pair_loss.mean()
                replay["list_loss"] = global_list_loss[local_slice]
                replay["list_loss_value"] = global_list_loss.mean()
                replay["grad_fake_aug"] = grad_fake[local_slice]
                replay["coupling_info"] = {
                    name: value[local_slice] for name, value in coupling_info.items()
                }
                return replay
            replay["pair_loss"] = (
                pairwise_generator_loss(paired_delta, margin=self.pair_margin)
                if self.lambda_pair > 0
                else zero_vector
            )
            replay["pair_loss_value"] = replay["pair_loss"].mean()
            if self.lambda_list > 0:
                list_loss_value, global_list_loss, grad_fake = infonce_generator_loss_with_grads(
                    global_meta["aug_real_scores"],
                    global_meta["aug_fake_scores"],
                    tau=self.list_tau,
                )
                replay["list_loss"] = global_list_loss[local_slice]
                replay["list_loss_value"] = list_loss_value
                replay["grad_fake_aug"] = grad_fake[local_slice]
            return replay

        if self._requires_coupling():
            neighbor_indices, coupling_weights, coupling_info = coupling
            global_delta = local_delta(
                global_meta["aug_real_scores"],
                global_meta["aug_fake_scores"],
                neighbor_indices,
            )
            global_pair_loss = (
                local_pairwise_discriminator_loss(
                    global_delta, coupling_weights, margin=self.pair_margin
                )
                if self.lambda_pair > 0
                else torch.zeros_like(global_meta["aug_fake_scores"])
            )
            global_list_loss = (
                local_listwise_discriminator_loss(
                    global_delta, coupling_weights, tau=self.list_tau
                )
                if self.lambda_list_d > 0
                else torch.zeros_like(global_meta["aug_fake_scores"])
            )
            _, loss_vector, grad_real, grad_fake = local_coupled_discriminator_loss_with_grads(
                global_meta["aug_real_scores"],
                global_meta["aug_fake_scores"],
                neighbor_indices, coupling_weights,
                lambda_pair=self.lambda_pair, pair_margin=self.pair_margin,
                lambda_list=self.lambda_list_d, list_tau=self.list_tau,
            )
            del loss_vector
            replay["pair_loss"] = global_pair_loss[local_slice]
            replay["pair_loss_value"] = global_pair_loss.mean()
            replay["list_loss"] = global_list_loss[local_slice]
            replay["list_loss_value"] = global_list_loss.mean()
            replay["grad_real_aug"] = grad_real[local_slice]
            replay["grad_fake_aug"] = grad_fake[local_slice]
            replay["coupling_info"] = {
                name: value[local_slice] for name, value in coupling_info.items()
            }
            # local_rank computed separately if enabled (existing code still runs below)
        else:
            replay["pair_loss"] = (
                pairwise_discriminator_loss(paired_delta, margin=self.pair_margin)
                if self.lambda_pair > 0
                else zero_vector
            )
            replay["pair_loss_value"] = replay["pair_loss"].mean()
            if self.lambda_list > 0:
                (
                    list_loss_value,
                    global_list_loss,
                    grad_real,
                    grad_fake,
                ) = infonce_discriminator_loss_with_grads(
                    global_meta["aug_real_scores"],
                    global_meta["aug_fake_scores"],
                    tau=self.list_tau,
                )
                replay["list_loss"] = global_list_loss[local_slice]
                replay["list_loss_value"] = list_loss_value
                replay["grad_real_aug"] = grad_real[local_slice]
                replay["grad_fake_aug"] = grad_fake[local_slice]
        if self.lambda_local_rank > 0:
            (
                local_rank_loss,
                adjacent_losses,
                grad_fake_clean,
            ) = local_rank_loss_with_grads(
                real_features=global_meta["clean_real_features"],
                fake_features=global_meta["clean_fake_features"],
                fake_scores=global_meta["clean_fake_scores"],
                class_ids=global_meta["class_ids"],
                k=self.local_rank_k,
            )
            replay["local_rank_loss"] = local_rank_loss
            replay["local_rank_adjacent_losses"] = adjacent_losses
            replay["grad_fake_clean"] = grad_fake_clean[local_slice]
        return replay

    def _report_coupled_stats(self, phase, metadata, replay, replay_stats):
        local = metadata["local"]
        paired_delta = pairwise_delta(local["aug_real_scores"], local["aug_fake_scores"])
        self._report_coupling_stats(replay["coupling_info"])
        self._report_common_stats(
            phase,
            real_scores=local["aug_real_scores"],
            fake_scores=local["aug_fake_scores"],
            pair_loss=replay["pair_loss"],
            list_loss=replay["list_loss"],
            paired_delta=paired_delta,
            device=local["aug_real_scores"].device,
        )

        pair_term = self.lambda_pair * replay["pair_loss"].mean()
        list_lambda = self._phase_list_lambda(phase)
        list_term = list_lambda * replay["list_loss"].mean()
        if phase == "G":
            total_loss = pair_term + list_term
            training_stats.report("Loss/G/pair_weighted", pair_term)
            training_stats.report("Loss/G/list_weighted", list_term)
            training_stats.report("Loss/G/loss", pair_term + list_term)
            training_stats.report("Loss/G/adv_weighted", pair_term + list_term)
            training_stats.report("Loss/G/total", total_loss)
            return

        local_term = self.lambda_local_rank * replay["local_rank_loss"]
        path_term = self.lambda_path_rank * replay_stats["path_rank_loss"]
        r1_term = (
            (replay_stats["gamma"] / 2) * replay_stats["r1_penalty"].mean()
            if self.use_r1_penalty
            else torch.zeros([], device=local["aug_real_scores"].device)
        )
        r2_term = (
            (replay_stats["gamma"] / 2) * replay_stats["r2_penalty"].mean()
            if self.use_r2_penalty
            else torch.zeros([], device=local["aug_real_scores"].device)
        )
        adv_term = pair_term + list_term
        base_total = adv_term + r1_term + r2_term
        total_loss = base_total + local_term + path_term
        training_stats.report("Loss/r1_penalty", replay_stats["r1_penalty"])
        training_stats.report("Loss/r2_penalty", replay_stats["r2_penalty"])
        training_stats.report("Loss/D/local_rank", replay["local_rank_loss"])
        training_stats.report("Loss/D/path_rank", replay_stats["path_rank_loss"])
        training_stats.report("Loss/D/path_score_reg", replay_stats["path_score_reg"])
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

    def _replay_coupled_gradients(self, buf, replay):
        phase = buf["phase"]
        offset = 0
        device = buf["real_imgs"][0].device
        zero_scalar = torch.zeros([], device=device)
        r1_penalties = []
        r2_penalties = []
        path_losses = []
        path_score_regs = []
        sample_counts = []

        for real_img, real_c, gen_z, gain in zip(
            buf["real_imgs"], buf["real_cs"], buf["gen_zs"], buf["gains"]
        ):
            batch_size = real_img.shape[0]
            sample_counts.append(batch_size)
            if phase == "G":
                fake_img = self.G(gen_z, real_c)
                real_scores = self.run_D(real_img.detach(), real_c, augment=True)
                fake_scores = self.run_D(fake_img, real_c, augment=True)
                scalar_loss = None
                if not self._requires_coupling() and self.lambda_pair > 0:
                    pair_loss = pairwise_generator_loss(
                        pairwise_delta(real_scores, fake_scores), margin=self.pair_margin
                    ).mean()
                    scalar_loss = gain * self.lambda_pair * pair_loss
                if scalar_loss is not None:
                    scalar_loss.backward(retain_graph=self.lambda_list > 0 or self._requires_coupling())
                if self._requires_coupling() or self.lambda_list > 0:
                    grad_fake = replay["grad_fake_aug"][offset : offset + batch_size].to(
                        fake_scores.dtype
                    )
                    torch.autograd.backward(
                        fake_scores,
                        grad_tensors=replay["replay_scale"] * grad_fake,
                    )
                offset += batch_size
                continue

            real_img = real_img.detach().requires_grad_(self.use_r1_penalty)
            fake_img = self.G(gen_z, real_c).detach().requires_grad_(self.use_r2_penalty)
            real_scores = self.run_D(real_img, real_c, augment=True)
            fake_scores = self.run_D(fake_img, real_c, augment=True)
            clean_fake_scores = None

            scalar_terms = []
            if not self._requires_coupling() and self.lambda_pair > 0:
                pair_loss = pairwise_discriminator_loss(
                    pairwise_delta(real_scores, fake_scores), margin=self.pair_margin
                ).mean()
                scalar_terms.append(self.lambda_pair * pair_loss)

            if self.path_rank_reg:
                path_rank_loss, path_score_reg = self._compute_path_rank_loss(
                    real_img.detach(), fake_img.detach(), real_c
                )
                path_losses.append(path_rank_loss.detach())
                path_score_regs.append(path_score_reg.detach())
                scalar_terms.append(self.lambda_path_rank * path_rank_loss)

            real_scores_for_penalty = real_scores
            fake_scores_for_penalty = fake_scores
            if self.use_non_aug_gp:
                real_scores_for_penalty = self.run_D(
                    real_img, real_c, augment=False, return_features=False
                )
                if self.lambda_local_rank > 0:
                    clean_fake_scores = self.run_D(
                        fake_img, real_c, augment=False, return_features=False
                    )
                    fake_scores_for_penalty = clean_fake_scores
                else:
                    fake_scores_for_penalty = self.run_D(
                        fake_img, real_c, augment=False, return_features=False
                    )

            r1_penalty = torch.zeros_like(real_scores)
            r2_penalty = torch.zeros_like(fake_scores)
            if self.use_r1_penalty:
                r1_penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
                    real_img, real_scores_for_penalty
                )
                scalar_terms.append((buf["gamma"] / 2) * r1_penalty.mean())
            if self.use_r2_penalty:
                r2_penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
                    fake_img, fake_scores_for_penalty
                )
                scalar_terms.append((buf["gamma"] / 2) * r2_penalty.mean())

            r1_penalties.append(r1_penalty.detach())
            r2_penalties.append(r2_penalty.detach())

            scalar_loss = None
            if len(scalar_terms) > 0:
                scalar_loss = gain * torch.stack(scalar_terms).sum()
                scalar_loss.backward(
                    retain_graph=self.lambda_list > 0 or self.lambda_local_rank > 0 or self._requires_coupling()
                )

            replay_tensors = []
            replay_grads = []
            if self._requires_coupling() or self.lambda_list > 0:
                replay_tensors.extend([real_scores, fake_scores])
                replay_grads.extend(
                    [
                        replay["replay_scale"]
                        * replay["grad_real_aug"][offset : offset + batch_size].to(
                            real_scores.dtype
                        ),
                        replay["replay_scale"]
                        * replay["grad_fake_aug"][offset : offset + batch_size].to(
                            fake_scores.dtype
                        ),
                    ]
                )
            if self.lambda_local_rank > 0:
                if clean_fake_scores is None:
                    clean_fake_scores = self.run_D(
                        fake_img, real_c, augment=False, return_features=False
                    )
                replay_tensors.append(clean_fake_scores)
                replay_grads.append(
                    replay["replay_scale"]
                    * replay["grad_fake_clean"][offset : offset + batch_size].to(
                        clean_fake_scores.dtype
                    )
                )
            if len(replay_tensors) > 0:
                torch.autograd.backward(replay_tensors, grad_tensors=replay_grads)
            offset += batch_size

        if len(r1_penalties) == 0:
            r1_penalty = torch.zeros([0], device=device)
        else:
            r1_penalty = torch.cat(r1_penalties, dim=0)
        if len(r2_penalties) == 0:
            r2_penalty = torch.zeros([0], device=device)
        else:
            r2_penalty = torch.cat(r2_penalties, dim=0)
        path_rank_loss = self._weighted_mean_scalar(path_losses, sample_counts, device)
        path_score_reg = self._weighted_mean_scalar(path_score_regs, sample_counts, device)
        return dict(
            gamma=buf["gamma"],
            r1_penalty=r1_penalty,
            r2_penalty=r2_penalty,
            path_rank_loss=path_rank_loss,
            path_score_reg=path_score_reg,
        )

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

            # SPM: symmetric pushforward matching (G phase)
            spm_term = torch.zeros([], device=real_img.device)
            if self.spm_enable and len(self.spm_transforms) > 0:
                from training.spm import compute_spm_loss
                spm_loss_g, spm_info = compute_spm_loss(
                    D=self.D, real_img=real_img.detach(), fake_img=fake_img,
                    real_c=real_c, preprocessor=self.preprocessor,
                    transforms=self.spm_transforms, weights=self.spm_weights,
                    margin=self.spm_margin,
                )
                # For G phase: G wants to MINIMIZE D(T_k(real)) - D(T_k(fake))
                # i.e. make fake look real at all scales
                # Use generator-side loss: softplus(margin + delta)
                spm_g = torch.zeros([], device=real_img.device)
                from training.spm import TRANSFORMS
                for k, (t_name, w) in enumerate(zip(self.spm_transforms, self.spm_weights)):
                    T = TRANSFORMS[t_name]
                    real_t = T(real_img.detach())
                    fake_t = T(fake_img)
                    rs = self._as_scores(self.D(self.preprocessor(real_t), real_c))
                    fs = self._as_scores(self.D(self.preprocessor(fake_t), real_c))
                    delta = rs - fs
                    g_loss_k = F.softplus(self.spm_margin + delta).mean()
                    spm_g = spm_g + w * g_loss_k
                spm_term = spm_g
                total_loss = total_loss + spm_term
                training_stats.report("Loss/G/spm", spm_term)

            (gain * total_loss).backward()

            self._report_common_stats(
                "G",
                real_scores=real_scores,
                fake_scores=fake_scores,
                pair_loss=pair_loss,
                list_loss=list_loss,
                paired_delta=paired_delta,
                device=real_img.device,
            )
            training_stats.report("Loss/G/pair_weighted", pair_term)
            training_stats.report("Loss/G/list_weighted", list_term)
            training_stats.report("Loss/G/loss", pair_term + list_term + spm_term)
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
            # SPM: symmetric pushforward matching (D phase)
            spm_d_term = zero_scalar
            if self.spm_enable and len(self.spm_transforms) > 0:
                from training.spm import TRANSFORMS
                for k, (t_name, w) in enumerate(zip(self.spm_transforms, self.spm_weights)):
                    T = TRANSFORMS[t_name]
                    real_t = T(real_img.detach())
                    fake_t = T(fake_img.detach())
                    rs = self._as_scores(self.D(self.preprocessor(real_t), real_c))
                    fs = self._as_scores(self.D(self.preprocessor(fake_t), real_c))
                    delta_k = rs - fs
                    d_loss_k = F.softplus(self.spm_margin - delta_k).mean()
                    spm_d_term = spm_d_term + w * d_loss_k
                training_stats.report("Loss/D/spm", spm_d_term)

            base_total = adv_term + r1_term + r2_term
            total_loss = base_total + local_term + path_term + spm_d_term
            (gain * total_loss).backward()

            self._report_common_stats(
                "D",
                real_scores=real_scores,
                fake_scores=fake_scores,
                pair_loss=pair_loss,
                list_loss=list_loss,
                paired_delta=paired_delta,
                device=real_img.device,
            )
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
