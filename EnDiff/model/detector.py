import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS, MODELS
from mmdet.models.detectors import CascadeRCNN

@DETECTORS.register_module()
class EnDiffDet(CascadeRCNN):
    def __init__(self, backbone, diff_cfg, **kwargs):
        super().__init__(backbone, **kwargs)
        self.diffusion = MODELS.build(diff_cfg)
        self.train_mode = 'det'

    def train_mode_control(self, train_mode):
        def freeze_module(module):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()
        def unfreeze_module(module):
            for p in module.parameters():
                p.requires_grad = True
            module.train()

        assert train_mode in ['det', 'sample']

        self.train_mode = train_mode
        if train_mode == 'det': # train detection
            unfreeze_module(self)
            freeze_module(self.diffusion)
        elif train_mode == 'sample':
            freeze_module(self)
            unfreeze_module(self.diffusion.net)
        return 

    def extract_feat(self, x):
        x = self.diffusion(x, return_loss=False)
        x = self.backbone(x)

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      hq_img=None,
                      **kwargs):
        if self.train_mode == 'det':
            return super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, proposals, **kwargs)
        elif self.train_mode == 'sample':
            loss = self.diffusion(img, hq_img)
            return loss

    def forward_test(self, imgs, img_metas, hq_img=None, **kwargs):
        _ = hq_img
        return super().forward_test(imgs, img_metas, **kwargs)