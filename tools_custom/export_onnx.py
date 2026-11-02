"""Export a trained DAMO-YOLO checkpoint to an RKNN-friendly ONNX model.

Differences from tools/converter.py:
  * no post-processing in the graph (no concat / DFL softmax / NMS)
  * 6 separate outputs: cls_8, reg_8, cls_16, reg_16, cls_32, reg_32
  * static 1 x 3 x S x S input, TorchScript-based exporter (dynamo=False)
Run from the DAMO-YOLO repo root.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.getcwd())
from damo.base_models.core.ops import RepConv, SiLU  # noqa: E402
from damo.config.base import parse_config  # noqa: E402
from damo.detectors.detector import build_local_model  # noqa: E402
from damo.utils.model_utils import replace_module  # noqa: E402


class RawHeadModel(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.backbone = detector.backbone
        self.neck = detector.neck
        self.head = detector.head

    def forward(self, x):
        cls_scores, bbox_preds = self.head(self.neck(self.backbone(x)))
        outs = []
        for c, r in zip(cls_scores, bbox_preds):
            outs += [c, r]
        return tuple(outs)


def build_model(config_file, ckpt, device='cpu'):
    config = parse_config(config_file)
    config.model.head.export_with_post = False
    model = build_local_model(config, device)
    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        state = state.get('model', state)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    model.head.nms = False
    return RawHeadModel(model).eval(), config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--config_file', required=True)
    p.add_argument('-c', '--ckpt', default=None)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--opset', type=int, default=12)
    p.add_argument('-o', '--output', required=True)
    args = p.parse_args()

    model, config = build_model(args.config_file, args.ckpt)
    strides = model.head.strides
    names = []
    for s in strides:
        names += [f'cls_{s}', f'reg_{s}']
    dummy = torch.rand(1, 3, args.img_size, args.img_size) * 255
    with torch.no_grad():
        ref = [t.numpy() for t in model(dummy)]
    kwargs = dict(input_names=['images'], output_names=names,
                  opset_version=args.opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, args.output, dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5 has no `dynamo` argument
        torch.onnx.export(model, dummy, args.output, **kwargs)

    import onnx
    m = onnx.load(args.output)
    try:
        import onnxsim
        m, ok = onnxsim.simplify(m)
        assert ok
        onnx.save(m, args.output)
        print('onnxsim: ok')
    except Exception as e:  # onnxsim is optional
        print(f'onnxsim skipped: {e}')

    import onnxruntime as ort
    sess = ort.InferenceSession(args.output,
                                providers=['CPUExecutionProvider'])
    outs = sess.run(None, {'images': dummy.numpy()})
    for n, a, b in zip(names, ref, outs):
        print(f'{n:7s} shape={tuple(b.shape)}  '
              f'max|torch-onnx|={np.abs(a - b).max():.2e}')
    print(f'num_classes={config.model.head.num_classes}  '
          f'reg_max={config.model.head.reg_max}  saved: {args.output}')


if __name__ == '__main__':
    main()