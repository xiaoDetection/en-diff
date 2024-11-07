import torch
import math

from mmdet.models.builder import MODELS
from mmcv.runner import BaseModule

@MODELS.register_module()
class CosSchedule(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.f_0 = torch.square(torch.cos(torch.tensor(0.008/1.008 * math.pi / 2)))

    def get_alpha_bar(self, t):
        f_t = torch.square(torch.cos((t + 0.008) / 1.008 * math.pi / 2))
        return f_t / f_t.new_full((1,), self.f_0)[:, None, None, None]