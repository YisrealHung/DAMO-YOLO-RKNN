# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import cv2
import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection

from damo.structures.bounding_box import BoxList

cv2.setNumThreads(0)


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, transforms=None, class_names=None):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        assert (class_names is not None), 'plz provide class_names'

        self.contiguous_class2id = {
            class_name: i
            for i, class_name in enumerate(class_names)
        }
        self.contiguous_id2class = {
            i: class_name
            for i, class_name in enumerate(class_names)
        }

        categories = self.coco.dataset['categories']
        cat_names = [cat['name'] for cat in categories]
        cat_ids = [cat['id'] for cat in categories]
        self.ori_class2id = {
            class_name: i
            for class_name, i in zip(cat_names, cat_ids)
        }
        self.ori_id2class = {
            i: class_name
            for class_name, i in zip(cat_names, cat_ids)
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        # DAMO-YOLO-Lite-Train: RAM Cache for high-speed training on Colab
        self.cache_ram = bool(int(os.environ.get('DAMO_CACHE_RAM', '1')))
        self._pull_cache = {}

    def preload_cache(self):
        """Pre-cache all images and annotations in RAM for zero disk I/O."""
        if not self.cache_ram:
            return
        total = len(self.ids)
        if len(self._pull_cache) == total:
            return
        print(f"[DAMO-YOLO-Lite-Train] Pre-caching {total} samples into RAM...", flush=True)
        for i in range(total):
            self.pull_item(i)
        try:
            import psutil
            ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            print(f"[DAMO-YOLO-Lite-Train] Pre-caching completed! System RAM used: {ram_gb:.2f} GB", flush=True)
        except Exception:
            print(f"[DAMO-YOLO-Lite-Train] Pre-caching completed ({total} items in RAM)!", flush=True)

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp

        if self.cache_ram and idx in self._pull_cache:
            img_c, res_c, _, _ = self._pull_cache[idx]
            img = img_c.copy()
            h, w = img.shape[:2]
            boxes = torch.as_tensor(res_c[:, :4], dtype=torch.float32).reshape(-1, 4)
            target = BoxList(boxes, (w, h), mode='xyxy')
            classes = torch.as_tensor(res_c[:, 4], dtype=torch.int64)
            target.add_field('labels', classes)
            target = target.clip_to_image(remove_empty=True)
        else:
            img, anno = super(COCODataset, self).__getitem__(idx)
            # filter crowd annotations
            # TODO might be better to add an extra field
            anno = [obj for obj in anno if obj['iscrowd'] == 0]

            boxes = [obj['bbox'] for obj in anno]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

            classes = [obj['category_id'] for obj in anno]
            classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                       for c in classes]

            classes = torch.tensor(classes)
            target.add_field('labels', classes)
            target = target.clip_to_image(remove_empty=True)

            # PIL to numpy array
            img = np.asarray(img)  # rgb

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def pull_item(self, idx):
        if self.cache_ram and idx in self._pull_cache:
            img, res, seg_masks, i = self._pull_cache[idx]
            return img.copy(), res.copy(), seg_masks, i

        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for i_box in range(len(target.bbox)):
            res[i_box, 0:4] = target.bbox[i_box]
            res[i_box, 4] = classes[i_box]

        img = np.asarray(img)  # rgb

        ret = (img, res, seg_masks, idx)
        if self.cache_ram:
            self._pull_cache[idx] = ret

        return img.copy(), res.copy(), seg_masks, idx

    def load_anno(self, idx):
        if self.cache_ram and idx in self._pull_cache:
            _, res, _, _ = self._pull_cache[idx]
            return [int(c) for c in res[:, 4]]

        _, anno = super(COCODataset, self).__getitem__(idx)
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        return classes

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

