"""DAMO-YOLO inference on RK3588 with rknn-toolkit-lite2.

Copy the export folder (model .rknn, model_info.json, labels.txt,
damo_rknn_utils.py, this file) to the board, then:
    python3 board_infer.py --source test.jpg
    python3 board_infer.py --source /dev/video0 --core 0
    python3 board_infer.py --source video.mp4 --save out.mp4
"""
import argparse
import glob
import json
import os
import time

import cv2

import damo_rknn_utils as U

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='.rknn file')
    p.add_argument('--source', required=True,
                   help='image file, video file or camera (/dev/videoN or N)')
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--core', type=int, default=0, choices=[0, 1, 2],
                   help='NPU core (run 3 processes on cores 0/1/2 '
                        'for more throughput)')
    p.add_argument('--save', default=None, help='save result image/video')
    args = p.parse_args()

    with open(os.path.join(HERE, 'model_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    model = args.model or glob.glob(os.path.join(HERE, '*.rknn'))[0]
    size, reg_max, names = info['img_size'], info['reg_max'], info['class_names']

    from rknnlite.api import RKNNLite
    rknn = RKNNLite()
    assert rknn.load_rknn(model) == 0, 'load_rknn failed'
    core = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1,
            RKNNLite.NPU_CORE_2][args.core]
    assert rknn.init_runtime(core_mask=core) == 0, 'init_runtime failed'

    def detect(frame):
        h, w = frame.shape[:2]
        t0 = time.time()
        outs = rknn.inference(inputs=[U.preprocess(frame, size)[None]],
                              data_format='nhwc')
        t1 = time.time()
        res = U.postprocess(outs, size, (w, h), args.conf, args.iou,
                            reg_max=reg_max)
        t2 = time.time()
        return res, (t1 - t0) * 1000, (t2 - t1) * 1000

    is_image = os.path.isfile(args.source) and \
        args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    if is_image:
        frame = cv2.imread(args.source)
        (b, s, c), t_npu, t_post = detect(frame)
        for bb, ss, cc in zip(b, s, c):
            print(f'{names[cc]:15s} {ss:.2f} {bb.round().astype(int).tolist()}')
        print(f'npu {t_npu:.1f} ms, post {t_post:.1f} ms')
        out = U.draw(frame, b, s, c, names)
        cv2.imwrite(args.save or 'result.jpg', out)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            (b, s, c), t_npu, t_post = detect(frame)
            out = U.draw(frame, b, s, c, names)
            cv2.putText(out, f'npu {t_npu:.1f}ms post {t_post:.1f}ms',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            if args.save:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                        (out.shape[1], out.shape[0]))
                writer.write(out)
            else:
                print(f'{len(b)} objects, npu {t_npu:.1f} ms, '
                      f'post {t_post:.1f} ms')
        cap.release()
        if writer is not None:
            writer.release()
    rknn.release()


if __name__ == '__main__':
    main()