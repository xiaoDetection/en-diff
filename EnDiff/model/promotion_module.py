import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmdet.models.builder import MODELS
from mmcv.runner import BaseModule, ModuleList

# from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(BaseModule):
    def __init__(self, n_channels, init_cfg=None):
        super().__init__(init_cfg)
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        t = t.reshape(-1)
        d = self.n_channels // 8
        emb = math.log(1e4) / (d - 1)
        emb = torch.exp(torch.arange(d, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        return self.lin2(emb)

class PMBlock(BaseModule):
    def __init__(self, in_channels, out_channels, time_channels, is_output_block=False, init_cfg=None):
        super().__init__(init_cfg)
        self.is_out = is_output_block
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.time_layer = nn.Sequential(nn.Linear(time_channels, out_channels), Swish())

    def forward(self, x, t):
        y1 = F.leaky_relu(self.bn1(self.conv1(x)))
        y1 += self.time_layer(t)[:, :, None, None]
        y2 = self.conv2(y1)
        if not self.is_out:
            y2 = F.leaky_relu(self.bn2(y2))
        return y1 + y2


@MODELS.register_module()
class PM(BaseModule):
    def __init__(self, channels, time_channels, num_mid_block=1, init_cfg=None):
        super().__init__(init_cfg)
        self.time_emb = TimeEmbedding(time_channels)
        self.block1 = PMBlock(3, channels, time_channels)
        self.block3 = PMBlock(channels, 3,time_channels, is_output_block=True)

        self.block2 = ModuleList()
        for _ in range(num_mid_block):
            self.block2.append(PMBlock(channels, channels, time_channels))

    def forward(self, x, t, c):
        t = t.type_as(x)
        t = self.time_emb(t)
        y = self.block1(x, t)
        for b in self.block2:
            y = b(y, t)
        y = self.block3(y, t)
        return y + c