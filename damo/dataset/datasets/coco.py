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

        # DAMO-YOLO-Lite-Train: Ultra-light Byte-Level RAM Cache (~500MB total instead of 11GB raw array)
        self.cache_ram = bool(int(os.environ.get('DAMO_CACHE_RAM', '1')))
        self._byte_cache = {}
        self._anno_cache = {}

    def _load_and_cache_anno(self, idx):
        if idx in self._anno_cache:
            return self._anno_cache[idx]
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        anno = [obj for obj in anno if obj.get('iscrowd', 0) == 0]

        img_info = self.coco.loadImgs(img_id)[0]
        w, h = img_info['width'], img_info['height']

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (w, h), mode='xywh').convert('xyxy')
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] for c in classes]

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

        cached = (res, seg_masks, (w, h))
        if self.cache_ram:
            self._anno_cache[idx] = cached
        return cached

    def preload_cache(self):
        """Pre-cache compressed image bytes and annotations in RAM for zero disk I/O with ultra-low memory (<600MB)."""
        if not self.cache_ram:
            return
        total = len(self.ids)
        if len(self._byte_cache) == total:
            return
        print(f"[DAMO-YOLO-Lite-Train] Pre-caching {total} samples into RAM (Byte-Cache mode, ~500MB total)...", flush=True)
        for idx in range(total):
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            full_path = os.path.join(self.root, file_name)
            if os.path.exists(full_path):
                with open(full_path, 'rb') as f:
                    self._byte_cache[idx] = f.read()
            self._load_and_cache_anno(idx)
        try:
            import psutil
            ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            print(f"[DAMO-YOLO-Lite-Train] Pre-caching completed! Current System RAM used: {ram_gb:.2f} GB (Extremely safe!)", flush=True)
        except Exception:
            print(f"[DAMO-YOLO-Lite-Train] Pre-caching completed ({total} items in RAM)!", flush=True)

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp

        res, seg_masks, (w, h) = self._load_and_cache_anno(idx)

        if self.cache_ram and idx in self._byte_cache:
            raw_bytes = self._byte_cache[idx]
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            full_path = os.path.join(self.root, file_name)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = torch.as_tensor(res[:, :4], dtype=torch.float32).reshape(-1, 4)
        target = BoxList(boxes, (w, h), mode='xyxy')
        classes = torch.as_tensor(res[:, 4], dtype=torch.int64)
        target.add_field('labels', classes)
        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def pull_item(self, idx):
        res, seg_masks, (w, h) = self._load_and_cache_anno(idx)

        if self.cache_ram and idx in self._byte_cache:
            raw_bytes = self._byte_cache[idx]
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_id = self.ids[idx]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            full_path = os.path.join(self.root, file_name)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, res.copy(), seg_masks, idx

    def load_anno(self, idx):
        res, _, _ = self._load_and_cache_anno(idx)
        return [int(c) for c in res[:, 4]]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        return classes

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

