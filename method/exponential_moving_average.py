

import numpy as np
import operator
from collections import OrderedDict
from copy import deepcopy


import torch
import torch.nn as nn

'''
# https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/utils/model_ema.py
class EMA(nn.Module):
    """ Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
        self.backup = None

    def _update(self, model, update_fn):
        with torch.no_grad():
            msd = model.state_dict()
            modulesd = self.module.state_dict()
            # for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            for k, ema_v in modulesd.items():
                model_v = msd[k].detach()
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                modulesd[k].copy_(0.5*update_fn(ema_v, model_v))

    def update(self, model):
        # print('before up', self.module.state_dict()) 
        print('before up', model.state_dict()) 
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        # print('after up', self.module.state_dict()) # no change 
        print('after up', model.state_dict())


    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def apply_shadow(self,model):
        self.backup = deepcopy(model)
        return deepcopy(self.module)
    
    def restore(self):
        return self.backup
'''

# # work but not include mean and var of BN
# # https://fyubang.com/2019/06/01/ema/
# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}

#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def update(self):
#         # print('before up', self.shadow) 
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()
#         # print('after up', self.shadow) 
    
#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]
    
#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}




# from cdd
class EMA:
    def __init__(self, model, decay=0.999, device=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_is_dp = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        assert isinstance(checkpoint, dict)
        if 'model_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_ema'].items():
                if self.ema_is_dp:
                    name = k if k.startswith('module') else 'module.' + k
                else:
                    name = k.replace('module.', '') if k.startswith('module') else k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)

    def state_dict(self):
        return self.ema.state_dict()

    def update(self, model):
        pre_module = hasattr(model, 'module') and not self.ema_is_dp
        with torch.no_grad():
            curr_msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                k = 'module.' + k if pre_module else k
                model_v = curr_msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


# # Training Sample
# model_ema = None
# if cfg.SOLVER.MODEL_EMA>0:
#     model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)

# for iteration, (images, targets, idxs) in enumerate(data_loader, start_iter):
#     if model_ema is not None:
#         model_ema.update(model)
#         arguments["model_ema"] = model_ema.state_dict()

# for data in val_dataloader:
#     if model_ema is not None:
#         model_ema.ema.eval()        




'''
# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()
'''