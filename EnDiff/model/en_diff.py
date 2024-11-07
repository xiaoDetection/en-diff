import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS


@MODELS.register_module()
class EnDiff(BaseModule):
    def __init__(
            self,
            net,
            T=1000,
            diffuse_ratio=0.6,
            sample_times=10,
            land_loss_weight=1,
            uw_loss_weight=1,
            init_cfg=None,
    ):
        super(EnDiff, self).__init__(init_cfg)
        self.net = MODELS.build(net)
        self.T = T
        self.diffuse_ratio = diffuse_ratio
        self.sample_times = sample_times
        self.t_end = int(self.T * self.diffuse_ratio)
        self.land_loss_weight = land_loss_weight
        self.uw_loss_weight = uw_loss_weight

        self.f_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        self.t_list = list(range(0, self.t_end, self.t_end // self.sample_times)) + [self.t_end]
    
    def get_alpha_cumprod(self, t: torch.Tensor):
        return (torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2 / self.f_0)[:, None, None, None] 

    def q_diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        t = t.reshape(-1)
        alpha_cumprod = self.get_alpha_cumprod(t).to(x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        xt =  torch.sqrt(alpha_cumprod) * x0 + torch.sqrt(1 - alpha_cumprod) * noise
        return xt, noise

    def predict(self, et: torch.Tensor, u0: torch.Tensor, t: torch.Tensor):
        e0 = self.net(et, t, u0)
        alpha_cumprod = self.get_alpha_cumprod(t)
        alpha_cumprod_prev = self.get_alpha_cumprod(t - (self.t_end // self.sample_times))
        noise = (et - torch.sqrt(alpha_cumprod) * e0) / torch.sqrt(1 - alpha_cumprod)
        e_prev = torch.sqrt(alpha_cumprod_prev) * e0 + torch.sqrt(1 - alpha_cumprod_prev) * noise
        return e0, e_prev, noise
    
    def loss(self, r_prev: torch.Tensor, h_prev: torch.Tensor, noise_pred: torch.Tensor, noise_gt: torch.Tensor):
        r = torch.flatten(r_prev, 1)
        h = torch.flatten(h_prev, 1)
        r = F.log_softmax(r, dim=-1)
        h = F.log_softmax(h, dim=-1)
        land_loss =  F.kl_div(r, h, log_target=True, reduction='batchmean') * self.land_loss_weight

        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight
        return dict(land_loss = land_loss, uw_loss = uw_loss)
    
    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        train_idx = random.randint(1, len(self.t_list) - 1)
        rs, noise_gt = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))

        rt_prev = rs
        for i, t in enumerate(self.t_list[:train_idx - 1:-1]):
            _, rt_prev, noise_pred = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))

        ht_prev, _ = self.q_diffuse(h0, torch.full((1,), self.t_list[train_idx - 1], device=u0.device))
        return self.loss(rt_prev, ht_prev, noise_pred, noise_gt)
    
    def forward_test(self, u0):
        rs, _ = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))

        rt_prev = rs
        for i, t in enumerate(self.t_list[:1:-1]):
            r0, rt_prev, _ = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))
        return r0

    
    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        if return_loss:
            assert h0 is not None
            return self.forward_train(u0, h0)
        else:
            return self.forward_test(u0)
