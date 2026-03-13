import torch
import torch.nn as nn
import copy
import R3GAN.Networks

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
