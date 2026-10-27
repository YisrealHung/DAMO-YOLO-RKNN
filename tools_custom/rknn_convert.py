"""ONNX -> RKNN (INT8) for RK3588, then check the result on the simulator.

Runs in the separate rknn-toolkit2 environment (x86_64 Linux).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import damo_rknn_utils as U  # noqa: E402


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--out', required=True, help='output .rknn path')
    p.add_argument('--dataset', required=True,
                   help='txt file, one calibration image path per line')
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--platform', default='rk3588')
    p.add_argument('--quant_algo', default='normal',
                   choices=['normal', 'mmse', 'kl_divergence'])
    p.add_argument('--no_quant', action='store_true', help='build FP16')
    p.add_argument('--val_json', default=None)
    p.add_argument('--val_dir', default=None)
    p.add_argument('--eval_images', type=int, default=200)
    p.add_argument('--compare_images', type=int, default=5)
    p.add_argument('--analysis', action='store_true',
                   help='run per-layer accuracy_analysis on one image')
    p.add_argument('--reg_max', type=int, default=16)
    p.add_argument('--report', default=None, help='write metrics json here')
    args = p.parse_args()

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                target_platform=args.platform,
                quantized_algorithm=args.quant_algo,
                quantized_method='channel',
                optimization_level=3)
    assert rknn.load_onnx(model=args.onnx) == 0, 'load_onnx failed'
    ret = rknn.build(do_quantization=not args.no_quant,
                     dataset=args.dataset)
    assert ret == 0, 'build failed'
    assert rknn.export_rknn(args.out) == 0, 'export_rknn failed'
    print(f'[ok] saved {args.out}')

    report = {'rknn': args.out, 'quantized': not args.no_quant,
              'quant_algo': args.quant_algo}
    if args.val_json is None:
        rknn.release()
        return

    # ---- simulator checks --------------------------------------------- #
    assert rknn.init_runtime() == 0, 'simulator init failed'
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    with open(args.val_json, 'r', encoding='utf-8') as f:
        files = [im['file_name'] for im in json.load(f)['images']]
    S = args.img_size

    def run_rknn(img_rgb):
        return rknn.inference(inputs=[img_rgb[None]], data_format='nhwc')

    def run_onnx(img_rgb):
        return sess.run(None, {'images': U.to_nchw(img_rgb)})

    # 1) per-output cosine similarity, ONNX FP32 vs RKNN simulator
    sims = []
    for fn in files[:args.compare_images]:
        img = U.preprocess(cv2.imread(os.path.join(args.val_dir, fn)), S)
        sims.append([cosine(a, b) for a, b in zip(run_onnx(img),
                                                   run_rknn(img))])
    sims = np.array(sims).mean(axis=0)
    names = [o.name for o in sess.get_outputs()]
    print('cosine similarity ONNX vs RKNN (1.0 = identical):')
    for n, s in zip(names, sims):
        flag = '' if s > 0.99 else '   <-- check quantization'
        print(f'  {n:7s} {s:.5f}{flag}')
    report['cosine'] = dict(zip(names, [round(float(s), 5) for s in sims]))

    # 2) mAP on the first N val images for both engines
    def make_pred(run):
        def pred(img_bgr):
            h, w = img_bgr.shape[:2]
            outs = run(U.preprocess(img_bgr, S))
            return U.postprocess(outs, S, (w, h), conf_thr=0.001,
                                 iou_thr=0.65, reg_max=args.reg_max,
                                 multi_label=True)
        return pred

    n = min(args.eval_images, len(files))
    print(f'\n== ONNX FP32 mAP on {n} val images ==')
    m_onnx = U.coco_map(make_pred(run_onnx), args.val_json, args.val_dir, n)
    print(f'\n== RKNN simulator mAP on {n} val images ==')
    m_rknn = U.coco_map(make_pred(run_rknn), args.val_json, args.val_dir, n)
    print(f'\nmAP50-95  ONNX {m_onnx[0]:.4f}  RKNN {m_rknn[0]:.4f}  '
          f'(drop {m_onnx[0] - m_rknn[0]:+.4f})')
    print(f'mAP50     ONNX {m_onnx[1]:.4f}  RKNN {m_rknn[1]:.4f}  '
          f'(drop {m_onnx[1] - m_rknn[1]:+.4f})')
    report.update({'eval_images': n,
                   'map50_95': {'onnx': m_onnx[0], 'rknn': m_rknn[0]},
                   'map50': {'onnx': m_onnx[1], 'rknn': m_rknn[1]}})

    # 3) optional per-layer analysis (writes ./snapshot)
    if args.analysis:
        img_path = os.path.join(args.val_dir, files[0])
        tmp = os.path.join(os.path.dirname(args.out), 'analysis_input.jpg')
        img = U.preprocess(cv2.imread(img_path), S)
        cv2.imwrite(tmp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out_dir = os.path.join(os.path.dirname(args.out), 'snapshot')
        try:
            rknn.accuracy_analysis(inputs=[tmp], output_dir=out_dir)
            print(f'[ok] per-layer analysis written to {out_dir}')
        except Exception as e:
            print(f'[warn] accuracy_analysis failed: {e}')

    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    rknn.release()


if __name__ == '__main__':
    main()