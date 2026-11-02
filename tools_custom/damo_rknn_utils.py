"""DAMO-YOLO pre/post-processing shared by ONNX, the RKNN simulator and the board.

Model contract (as exported by export_onnx.py):
  input : 1 x 3 x S x S, RGB, 0-255 float (RKNN: 1 x S x S x 3 uint8 RGB),
          plain resize to S x S (no letterbox, no mean/std)
  output: 6 tensors, for stride 8, 16, 32 in this order:
          cls_<s>: 1 x K x H x W   class scores, sigmoid already applied
          reg_<s>: 1 x 4*(R+1) x H x W   DFL logits (l, t, r, b), R = reg_max
Only numpy (and cv2 for resize / drawing) is needed, so this file can be
copied to the RK3588 board as-is.
"""
import numpy as np

STRIDES = (8, 16, 32)


def preprocess(img_bgr, size):
    """BGR image (cv2.imread) -> RGB uint8 HWC resized to size x size."""
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def to_nchw(img_rgb):
    """RGB uint8 HWC -> 1x3xHxW float32 (ONNX input)."""
    return img_rgb.transpose(2, 0, 1)[None].astype(np.float32)


def nms(boxes, scores, iou_thr):
    """Plain numpy NMS, boxes: Nx4 xyxy. Returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def decode(outputs, img_size, conf_thr, reg_max=16, strides=STRIDES):
    """Raw outputs -> (boxes xyxy in model input pixels, scores N x K).

    Only cells whose best class score passes conf_thr are decoded, which
    keeps the DFL softmax cheap on the board CPU.
    """
    all_boxes, all_scores = [], []
    proj = np.arange(reg_max + 1, dtype=np.float32)
    for i, s in enumerate(strides):
        cls = np.asarray(outputs[2 * i], dtype=np.float32)[0]
        reg = np.asarray(outputs[2 * i + 1], dtype=np.float32)[0]
        k, h, w = cls.shape
        assert reg.shape[0] == 4 * (reg_max + 1), \
            f'unexpected reg channels {reg.shape}, check output order'
        cls = cls.reshape(k, -1).T                       # HW x K
        keep = np.nonzero(cls.max(axis=1) > conf_thr)[0]
        if keep.size == 0:
            continue
        reg = reg.reshape(4, reg_max + 1, -1)[:, :, keep]  # 4 x (R+1) x n
        reg = np.exp(reg - reg.max(axis=1, keepdims=True))
        reg /= reg.sum(axis=1, keepdims=True)
        dist = (reg * proj[None, :, None]).sum(axis=1).T * s  # n x 4
        px = (keep % w).astype(np.float32) * s
        py = (keep // w).astype(np.float32) * s
        boxes = np.stack([px - dist[:, 0], py - dist[:, 1],
                          px + dist[:, 2], py + dist[:, 3]], axis=1)
        all_boxes.append(boxes)
        all_scores.append(cls[keep])
    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0, 0), np.float32)
    return np.concatenate(all_boxes), np.concatenate(all_scores)


def postprocess(outputs, img_size, orig_wh, conf_thr=0.25, iou_thr=0.45,
                reg_max=16, max_det=300, multi_label=False):
    """Raw outputs -> boxes (xyxy, original image pixels), scores, class ids.

    multi_label=True keeps every (box, class) pair above conf_thr (use for
    mAP evaluation); False keeps only the best class per box (deployment).
    """
    boxes, scores = decode(outputs, img_size, conf_thr, reg_max)
    if boxes.shape[0] == 0:
        return boxes, np.zeros(0, np.float32), np.zeros(0, np.int64)
    if multi_label:
        idx, cls_ids = np.nonzero(scores > conf_thr)
        boxes, scores = boxes[idx], scores[idx, cls_ids]
    else:
        cls_ids = scores.argmax(axis=1)
        scores = scores[np.arange(len(cls_ids)), cls_ids]
        m = scores > conf_thr
        boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]
    if scores.size > 3000:  # cap NMS input
        top = scores.argsort()[::-1][:3000]
        boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]
    # class-aware NMS by offsetting boxes of different classes
    offset = cls_ids[:, None].astype(np.float32) * (img_size * 4)
    keep = nms(boxes + offset, scores, iou_thr)[:max_det]
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    sx, sy = orig_wh[0] / img_size, orig_wh[1] / img_size
    boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_wh[0])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_wh[1])
    return boxes, scores, cls_ids.astype(np.int64)


def draw(img_bgr, boxes, scores, cls_ids, class_names, conf_thr=0.0):
    import cv2
    out = img_bgr.copy()
    rng = np.random.RandomState(3)
    colors = rng.randint(64, 255, size=(max(len(class_names), 1), 3))
    for b, s, c in zip(boxes, scores, cls_ids):
        if s < conf_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        color = tuple(int(v) for v in colors[c % len(colors)])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{class_names[c] if c < len(class_names) else c} {s:.2f}'
        cv2.putText(out, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def coco_map(predict_fn, ann_file, img_dir, max_images=None, log_every=50):
    """Evaluate predict_fn(img_bgr) -> (boxes, scores, cls_ids) with COCO mAP.

    Category ids in ann_file must be 1..K in class order (prepare_dataset
    writes them that way). Returns (mAP@0.5:0.95, mAP@0.5).
    """
    import contextlib
    import io
    import os

    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_file)
    img_ids = sorted(gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]
    results = []
    for n, img_id in enumerate(img_ids, 1):
        info = gt.loadImgs(img_id)[0]
        img = cv2.imread(os.path.join(img_dir, info['file_name']))
        boxes, scores, cls_ids = predict_fn(img)
        for b, s, c in zip(boxes, scores, cls_ids):
            results.append({'image_id': img_id, 'category_id': int(c) + 1,
                            'bbox': [float(b[0]), float(b[1]),
                                     float(b[2] - b[0]), float(b[3] - b[1])],
                            'score': float(s)})
        if log_every and n % log_every == 0:
            print(f'  {n}/{len(img_ids)} images')
    if not results:
        print('no detections')
        return 0.0, 0.0
    dt = gt.loadRes(results)
    ev = COCOeval(gt, dt, 'bbox')
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])