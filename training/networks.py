import copy

import torch
import torch.nn as nn

import R3GAN.Networks
from training.drift_reference import DiTLikeConfig, DiTLikeGenerator as ReferenceDiTLikeGenerator
from training.rgm_conditioning import RankConditionSchema, build_extra_condition, schema_extra_dim


def _build_r3gan_generator_config(kw):
    config = copy.deepcopy(kw)
    del config["FP16Stages"]
    del config["c_dim"]
    del config["img_resolution"]
    return config


def _init_main_layers_fp16(model, fp16_stages):
    for stage_index in fp16_stages:
        model.MainLayers[stage_index].DataType = torch.bfloat16


class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()

        config = _build_r3gan_generator_config(kw)
        if kw["c_dim"] != 0:
            config["ConditionDimension"] = kw["c_dim"]

        self.Model = R3GAN.Networks.Generator(*args, **config)
        self.z_dim = kw["NoiseDimension"]
        self.c_dim = kw["c_dim"]
        self.img_resolution = kw["img_resolution"]

        _init_main_layers_fp16(self.Model, kw["FP16Stages"])

    def forward(self, x, c):
        return self.Model(x, c)


class RankConditionedGenerator(nn.Module):
    def __init__(self, *args, **kw):
        super(RankConditionedGenerator, self).__init__()

        config = _build_r3gan_generator_config(kw)
        self.c_dim = kw["c_dim"]
        self.z_dim = kw["NoiseDimension"]
        self.img_resolution = kw["img_resolution"]
        self.ConditionSchema = RankConditionSchema(
            use_alpha=bool(config.pop("UseAlpha", False)),
            use_rank=bool(config.pop("UseRank", True)),
            use_rank_pair=bool(config.pop("UseRankPair", False)),
            alpha_min=float(config.pop("AlphaMin", 1.0)),
            alpha_max=float(config.pop("AlphaMax", 4.0)),
            rank_min=float(config.pop("RankMin", 0.0)),
            rank_max=float(config.pop("RankMax", 1.0)),
        )
        self.EvalAlpha = float(config.pop("EvalAlpha", 1.0))
        self.EvalRank = float(config.pop("EvalRank", self.ConditionSchema.rank_min))

        extra_dim = schema_extra_dim(self.ConditionSchema)
        if extra_dim <= 0:
            raise ValueError("RankConditionedGenerator requires at least one active extra condition")
        config["ConditionDimension"] = self.c_dim + extra_dim if self.c_dim != 0 else extra_dim

        self.Model = R3GAN.Networks.Generator(*args, **config)
        _init_main_layers_fp16(self.Model, kw["FP16Stages"])

    def forward(self, x, c, alpha=None, rank=None, rank_from=None, rank_to=None):
        condition_labels = _coerce_condition_labels(c, batch_size=x.shape[0], c_dim=self.c_dim, device=x.device)
        if self.ConditionSchema.use_alpha and alpha is None:
            alpha = torch.full([x.shape[0]], self.EvalAlpha, device=x.device, dtype=torch.float32)
        if self.ConditionSchema.use_rank and rank is None:
            rank = torch.full([x.shape[0]], self.EvalRank, device=x.device, dtype=torch.float32)

        condition = build_extra_condition(
            condition_labels,
            alpha=alpha,
            rank=rank,
            rank_from=rank_from,
            rank_to=rank_to,
            schema=self.ConditionSchema,
        )
        return self.Model(x, condition)


class DriftGenerator(RankConditionedGenerator):
    def __init__(self, *args, **kw):
        super(DriftGenerator, self).__init__(*args, UseAlpha=True, UseRank=False, UseRankPair=False, **kw)

    def forward(self, x, c, alpha=None):
        return super(DriftGenerator, self).forward(x, c, alpha=alpha)


class DiTLikeDriftGenerator(nn.Module):
    def __init__(self, *args, **kw):
        super(DiTLikeDriftGenerator, self).__init__()

        self.c_dim = int(kw.get("c_dim", 0))
        self.img_resolution = int(kw.get("img_resolution", 0))
        image_channels = int(kw.get("ImageChannels", 3))
        eval_alpha = float(kw.get("EvalAlpha", 1.0))
        self.EvalAlpha = eval_alpha
        self.StyleTokenCount = int(kw.get("StyleTokenCount", 0))

        config = DiTLikeConfig(
            image_size=self.img_resolution,
            in_channels=image_channels,
            out_channels=image_channels,
            patch_size=int(kw.get("PatchSize", 4)),
            hidden_dim=int(kw.get("HiddenDim", 256)),
            depth=int(kw.get("Depth", 6)),
            num_heads=int(kw.get("NumHeads", 8)),
            mlp_ratio=float(kw.get("MlpRatio", 4.0)),
            ffn_inner_dim=kw.get("FfnInnerDim", None),
            num_classes=max(self.c_dim, 1),
            register_tokens=int(kw.get("RegisterTokens", 16)),
            style_vocab_size=int(kw.get("StyleVocabSize", 1)),
            style_token_count=self.StyleTokenCount,
            alpha_hidden_dim=int(kw.get("AlphaHiddenDim", 128)),
            norm_type=str(kw.get("NormType", "layernorm")),
            use_qk_norm=bool(kw.get("UseQkNorm", False)),
            use_rope=bool(kw.get("UseRope", False)),
            alpha_embedding_type=str(kw.get("AlphaEmbeddingType", "mlp")),
            qk_norm_mode=str(kw.get("QkNormMode", "auto")),
            rope_mode=str(kw.get("RopeMode", "auto")),
            use_patch_positional_embedding=not bool(kw.get("DisablePatchPositionalEmbedding", False)),
            rmsnorm_affine=not bool(kw.get("DisableRmsNormAffine", False)),
        )
        self.ModelConfig = config
        self.Model = ReferenceDiTLikeGenerator(config)
        self.noise_channels = int(config.in_channels)
        self.style_vocab_size = int(config.style_vocab_size)

    def forward(self, noise, class_labels, alpha=None, style_indices=None):
        if noise.ndim != 4:
            raise ValueError(f"noise must be [B, C, H, W], got {tuple(noise.shape)}")
        if alpha is None:
            alpha = torch.full([noise.shape[0]], self.EvalAlpha, device=noise.device, dtype=torch.float32)
        elif not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha, device=noise.device, dtype=torch.float32)
        else:
            alpha = alpha.to(device=noise.device, dtype=torch.float32)
        if alpha.ndim == 0:
            alpha = alpha.expand(noise.shape[0])
        alpha = alpha.reshape(noise.shape[0])

        class_ids = _coerce_class_labels_to_ids(
            class_labels=class_labels,
            batch_size=noise.shape[0],
            device=noise.device,
        )
        if style_indices is None:
            style_indices = torch.zeros(
                noise.shape[0],
                self.StyleTokenCount,
                device=noise.device,
                dtype=torch.long,
            )
        return self.Model(noise, class_ids, alpha, style_indices)


class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()

        config = _build_r3gan_generator_config(kw)
        if kw["c_dim"] != 0:
            config["ConditionDimension"] = kw["c_dim"]

        self.Model = R3GAN.Networks.Discriminator(*args, **config)
        _init_main_layers_fp16(self.Model, kw["FP16Stages"])

    def forward(self, x, c, return_features=False):
        return self.Model(x, c, return_features=return_features)


def _coerce_condition_labels(c, batch_size, c_dim, device):
    if c is None:
        if c_dim != 0:
            raise ValueError("conditional generator requires label inputs")
        return torch.zeros([batch_size, 0], device=device, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c, device=device, dtype=torch.float32)
    else:
        c = c.to(device=device, dtype=torch.float32)
    if c.ndim != 2 or c.shape[0] != batch_size:
        raise ValueError("c must be [B, C] and aligned with z")
    if c.shape[1] != c_dim:
        raise ValueError(f"c must have {c_dim} channels, got {c.shape[1]}")
    return c


def _coerce_class_labels_to_ids(class_labels, batch_size, device):
    if class_labels is None:
        raise ValueError("class_labels must be provided for drift generation")
    if not isinstance(class_labels, torch.Tensor):
        class_labels = torch.as_tensor(class_labels, device=device)
    else:
        class_labels = class_labels.to(device=device)
    if class_labels.ndim == 1:
        if class_labels.shape[0] != batch_size:
            raise ValueError("class_labels batch dimension mismatch")
        return class_labels.long()
    if class_labels.ndim == 2:
        if class_labels.shape[0] != batch_size:
            raise ValueError("class_labels batch dimension mismatch")
        return class_labels.argmax(dim=1).long()
    raise ValueError(f"class_labels must be [B] or [B, C], got {tuple(class_labels.shape)}")
