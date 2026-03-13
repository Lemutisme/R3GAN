import torch
import torch.nn as nn
import copy
import R3GAN.Networks

from training.drift_reference import DiTLikeConfig, DiTLikeGenerator as ReferenceDiTLikeGenerator

class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Generator(*args, **config)
        self.z_dim = kw['NoiseDimension']
        self.c_dim = kw['c_dim']
        self.img_resolution = kw['img_resolution']
        
        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16
        
    def forward(self, x, c):
        return self.Model(x, c)

class DriftGenerator(nn.Module):
    def __init__(self, *args, **kw):
        super(DriftGenerator, self).__init__()

        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        self.AlphaMin = float(config.pop('AlphaMin', 1.0))
        self.AlphaMax = float(config.pop('AlphaMax', 4.0))
        self.EvalAlpha = float(config.pop('EvalAlpha', 1.0))

        if self.AlphaMax < self.AlphaMin:
            raise ValueError('AlphaMax must be >= AlphaMin')

        InternalConditionDimension = kw['c_dim'] + 1 if kw['c_dim'] != 0 else 1
        config['ConditionDimension'] = InternalConditionDimension

        self.Model = R3GAN.Networks.Generator(*args, **config)
        self.z_dim = kw['NoiseDimension']
        self.c_dim = kw['c_dim']
        self.img_resolution = kw['img_resolution']

        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16

    def forward(self, x, c, alpha=None):
        if alpha is None:
            alpha = torch.full([x.shape[0]], self.EvalAlpha, device=x.device, dtype=torch.float32)
        elif not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha, device=x.device, dtype=torch.float32)
        else:
            alpha = alpha.to(device=x.device, dtype=torch.float32)

        if alpha.ndim == 0:
            alpha = alpha.expand(x.shape[0])
        alpha = alpha.reshape(x.shape[0], 1)
        if self.AlphaMax == self.AlphaMin:
            AlphaCondition = torch.zeros_like(alpha)
        else:
            AlphaCondition = (alpha - self.AlphaMin) / (self.AlphaMax - self.AlphaMin)
        AlphaCondition = AlphaCondition.clamp(0.0, 1.0)

        if self.c_dim == 0:
            Condition = AlphaCondition
        else:
            if c is None:
                raise ValueError('Conditional drift generator requires label inputs')
            Condition = torch.cat([c.to(device=x.device, dtype=torch.float32), AlphaCondition], dim=1)
        return self.Model(x, Condition)


class DiTLikeDriftGenerator(nn.Module):
    def __init__(self, *args, **kw):
        super(DiTLikeDriftGenerator, self).__init__()

        self.c_dim = int(kw.get('c_dim', 0))
        self.img_resolution = int(kw.get('img_resolution', 0))
        image_channels = int(kw.get('ImageChannels', 3))
        eval_alpha = float(kw.get('EvalAlpha', 1.0))
        self.EvalAlpha = eval_alpha
        self.StyleTokenCount = int(kw.get('StyleTokenCount', 32))

        config = DiTLikeConfig(
            image_size=self.img_resolution,
            in_channels=image_channels,
            out_channels=image_channels,
            patch_size=int(kw.get('PatchSize', 8)),
            hidden_dim=int(kw.get('HiddenDim', 256)),
            depth=int(kw.get('Depth', 4)),
            num_heads=int(kw.get('NumHeads', 8)),
            mlp_ratio=float(kw.get('MlpRatio', 4.0)),
            ffn_inner_dim=kw.get('FfnInnerDim', None),
            num_classes=max(self.c_dim, 1),
            register_tokens=int(kw.get('RegisterTokens', 16)),
            style_vocab_size=int(kw.get('StyleVocabSize', 64)),
            style_token_count=self.StyleTokenCount,
            alpha_hidden_dim=int(kw.get('AlphaHiddenDim', 128)),
            norm_type=str(kw.get('NormType', 'layernorm')),
            use_qk_norm=bool(kw.get('UseQkNorm', False)),
            use_rope=bool(kw.get('UseRope', False)),
            alpha_embedding_type=str(kw.get('AlphaEmbeddingType', 'mlp')),
            qk_norm_mode=str(kw.get('QkNormMode', 'auto')),
            rope_mode=str(kw.get('RopeMode', 'auto')),
            use_patch_positional_embedding=not bool(kw.get('DisablePatchPositionalEmbedding', False)),
            rmsnorm_affine=not bool(kw.get('DisableRmsNormAffine', False)),
        )
        self.ModelConfig = config
        self.Model = ReferenceDiTLikeGenerator(config)
        self.noise_channels = int(config.in_channels)
        self.style_vocab_size = int(config.style_vocab_size)

    def forward(self, noise, class_labels, alpha=None, style_indices=None):
        if noise.ndim != 4:
            raise ValueError(f'noise must be [B, C, H, W], got {tuple(noise.shape)}')
        if alpha is None:
            alpha = torch.full([noise.shape[0]], self.EvalAlpha, device=noise.device, dtype=torch.float32)
        elif not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha, device=noise.device, dtype=torch.float32)
        else:
            alpha = alpha.to(device=noise.device, dtype=torch.float32)
        if alpha.ndim == 0:
            alpha = alpha.expand(noise.shape[0])
        alpha = alpha.reshape(noise.shape[0])

        ClassIds = _coerce_class_labels_to_ids(
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
        return self.Model(noise, ClassIds, alpha, style_indices)
    
class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Discriminator(*args, **config)
        
        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16
        
    def forward(self, x, c, return_features=False):
        return self.Model(x, c, return_features=return_features)


def _coerce_class_labels_to_ids(class_labels, batch_size, device):
    if class_labels is None:
        raise ValueError('class_labels must be provided for drift generation')
    if not isinstance(class_labels, torch.Tensor):
        class_labels = torch.as_tensor(class_labels, device=device)
    else:
        class_labels = class_labels.to(device=device)
    if class_labels.ndim == 1:
        if class_labels.shape[0] != batch_size:
            raise ValueError('class_labels batch dimension mismatch')
        return class_labels.long()
    if class_labels.ndim == 2:
        if class_labels.shape[0] != batch_size:
            raise ValueError('class_labels batch dimension mismatch')
        return class_labels.argmax(dim=1).long()
    raise ValueError(f'class_labels must be [B] or [B, C], got {tuple(class_labels.shape)}')
