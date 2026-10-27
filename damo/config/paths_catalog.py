# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    _CUSTOM_DATA_OUT = os.environ.get('DATA_OUT', '/content/datasets/custom')
    DATASETS = {
        'custom_train_coco': {
            'img_dir': f'{_CUSTOM_DATA_OUT}/train',
            'ann_file': f'{_CUSTOM_DATA_OUT}/annotations/train.json',
        },
        'custom_val_coco': {
            'img_dir': f'{_CUSTOM_DATA_OUT}/val',
            'ann_file': f'{_CUSTOM_DATA_OUT}/annotations/val.json',
        },
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
    }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            img_dir = attrs['img_dir']
            ann_file = attrs['ann_file']
            root = img_dir if (os.path.isabs(img_dir) or img_dir.startswith(('/', '\\'))) else os.path.join(data_dir, img_dir)
            ann = ann_file if (os.path.isabs(ann_file) or ann_file.startswith(('/', '\\'))) else os.path.join(data_dir, ann_file)
            args = dict(
                root=root,
                ann_file=ann,
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
