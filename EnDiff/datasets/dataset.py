from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
import os
import os.path as osp
from random import randint

@DATASETS.register_module()
class HqLqCocoDataset(CocoDataset):
    def __init__(
            self,
            ann_file,
            pipeline,
            classes=None,
            data_root=None,
            img_prefix='',
            hq_img_prefix=None,
            seg_prefix=None,
            seg_suffix='.png',
            proposal_file=None,
            test_mode=False,
            filter_empty_gt=True,
            file_client_args=dict(backend='disk')
        ):
        super().__init__(
                ann_file,
                pipeline,
                classes,
                data_root,
                img_prefix,
                seg_prefix,
                seg_suffix,
                proposal_file,
                test_mode,
                filter_empty_gt,
                file_client_args
            )
        self.hq_img_dir = hq_img_prefix if data_root is None else osp.join(data_root, hq_img_prefix)
        self.hq_img_names = [name for name in sorted(os.listdir(self.hq_img_dir)) if name.endswith('.jpg')]
        self.hq_img_num = len(self.hq_img_names)

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        idx = randint(0, self.hq_img_num - 1)
        results['hq_img_filename'] = osp.join(self.hq_img_dir, self.hq_img_names[idx])
    
