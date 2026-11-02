"""DAMO-YOLO inference on RK3588 (Orange Pi 5 Plus) with rknn-toolkit-lite2.

Supports:
- Multi-NPU core selection: 1, 2, or 3 NPU cores (--cores 1, 2, or 3)
- Dedicated single core binding (--core 0, 1, or 2)
- High-throughput multi-threaded NPU pipeline for video/webcam/RTSP stream
- Single image inference and benchmark mode (--benchmark)
- Real-time HUD stats overlay (FPS, NPU latency, object count)
- Video / image saving (--save) and GUI display (--show)

Usage examples on Orange Pi 5 Plus:
  # 1. Single image with 3 NPU cores
  python3 board_infer.py --source test.jpg --cores 3 --save result.jpg

  # 2. Benchmark latency (1 vs 2 vs 3 NPU cores) on an image
  python3 board_infer.py --source test.jpg --cores 1 --benchmark
  python3 board_infer.py --source test.jpg --cores 3 --benchmark

  # 3. Live USB camera with 3 NPU cores (maximum FPS)
  python3 board_infer.py --source 0 --cores 3 --show

  # 4. Video file with 3 NPU cores, save annotated video
  python3 board_infer.py --source video.mp4 --cores 3 --save out.mp4

  # 5. Run only on Core 2 (leaving Core 0 & 1 for other tasks)
  python3 board_infer.py --source /dev/video0 --core 2
"""

import argparse
import glob
import json
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# Add local directory to path for damo_rknn_utils
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import damo_rknn_utils as U
except ImportError:
    from tools_custom import damo_rknn_utils as U

# Try importing RKNNLite (available on RK3588 board)
try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None


# Core mask constants for RK3588 (3 NPU cores)
def get_rknn_core_masks():
    if RKNNLite is not None:
        c0 = getattr(RKNNLite, 'NPU_CORE_0', 1)
        c1 = getattr(RKNNLite, 'NPU_CORE_1', 2)
        c2 = getattr(RKNNLite, 'NPU_CORE_2', 4)
        c01 = getattr(RKNNLite, 'NPU_CORE_0_1', c0 | c1)
        c012 = getattr(RKNNLite, 'NPU_CORE_0_1_2', getattr(RKNNLite, 'NPU_CORE_ALL', c0 | c1 | c2))
        c_auto = getattr(RKNNLite, 'NPU_CORE_AUTO', 0)
    else:
        # Fallback values when RKNNLite is not installed (e.g. testing on PC)
        c0, c1, c2 = 1, 2, 4
        c01 = 3
        c012 = 7
        c_auto = 0
    return {
        'single_cores': [c0, c1, c2],
        'multi_core_map': {1: c0, 2: c01, 3: c012},
        'names': {c0: 'Core 0', c1: 'Core 1', c2: 'Core 2', c01: 'Core 0+1', c012: 'Core 0+1+2 (All 3)', c_auto: 'Auto'}
    }


class FPSCounter:
    """Sliding-window FPS counter for real-time video."""
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)

    def update(self):
        now = time.time()
        self.timestamps.append(now)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def draw_hud(frame, fps, t_prep, t_npu, t_post, num_dets, core_label):
    """Draw a clean, semi-transparent HUD banner with performance metrics."""
    h, w = frame.shape[:2]
    total_lat = t_prep + t_npu + t_post
    
    # Overlay banner background
    banner_h = 56
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Line 1: NPU Cores, FPS, Detections
    text1 = f"NPU: {core_label} | FPS: {fps:5.1f} | Objects: {num_dets}"
    cv2.putText(frame, text1, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    # Line 2: Latency breakdown
    text2 = f"Latency: {total_lat:4.1f} ms (Prep: {t_prep:4.1f}ms | NPU: {t_npu:4.1f}ms | Post: {t_post:4.1f}ms)"
    cv2.putText(frame, text2, (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Worker Thread for Multi-Core Pipeline
# ---------------------------------------------------------------------------
class NPUWorker(threading.Thread):
    def __init__(self, worker_id, core_mask, core_name, model_path,
                 img_size, reg_max, conf_thr, iou_thr,
                 input_queue, output_queue, stop_event):
        super().__init__(name=f"NPUWorker-{worker_id}")
        self.worker_id = worker_id
        self.core_mask = core_mask
        self.core_name = core_name
        self.model_path = model_path
        self.img_size = img_size
        self.reg_max = reg_max
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.daemon = True

    def run(self):
        if RKNNLite is None:
            raise RuntimeError("rknnlite is not installed. Please run on the RK3588 board.")

        rknn = RKNNLite()
        ret = rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"[{self.name}] Failed to load RKNN model: {self.model_path}")
        ret = rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(f"[{self.name}] Failed to init runtime on {self.core_name}")

        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                # Sentinel reached
                self.input_queue.task_done()
                break

            frame_id, frame = task
            h, w = frame.shape[:2]

            # 1. Preprocess
            t0 = time.perf_counter()
            inp = U.preprocess(frame, self.img_size)[None]
            t1 = time.perf_counter()

            # 2. NPU Inference
            outs = rknn.inference(inputs=[inp], data_format='nhwc')
            t2 = time.perf_counter()

            # 3. Postprocess
            boxes, scores, cls_ids = U.postprocess(
                outs, self.img_size, (w, h),
                conf_thr=self.conf_thr, iou_thr=self.iou_thr,
                reg_max=self.reg_max
            )
            t3 = time.perf_counter()

            t_prep = (t1 - t0) * 1000.0
            t_npu = (t2 - t1) * 1000.0
            t_post = (t3 - t2) * 1000.0

            self.output_queue.put((frame_id, frame, (boxes, scores, cls_ids),
                                   t_prep, t_npu, t_post, self.core_name))
            self.input_queue.task_done()

        rknn.release()


# ---------------------------------------------------------------------------
# Pipeline Runner for Video / Camera / Streams
# ---------------------------------------------------------------------------
def run_pipeline(source, model_path, num_workers, specific_core,
                 img_size, reg_max, names, conf_thr, iou_thr,
                 save_path=None, show=True):
    core_info = get_rknn_core_masks()
    single_cores = core_info['single_cores']

    # Determine worker core assignments
    worker_configs = []
    if specific_core is not None:
        c_mask = single_cores[specific_core]
        c_name = f"Core {specific_core}"
        worker_configs.append((0, c_mask, c_name))
    else:
        for i in range(num_workers):
            c_mask = single_cores[i % 3]
            c_name = f"Core {i % 3}"
            worker_configs.append((i, c_mask, c_name))

    print(f"\n[Info] Launching {len(worker_configs)} NPU worker thread(s):")
    for w_id, _, c_name in worker_configs:
        print(f"  - Worker {w_id}: Assigned to {c_name}")

    # Queues: Keep small to minimize buffer lag
    input_queue = queue.Queue(maxsize=len(worker_configs) * 2)
    output_queue = queue.Queue()
    stop_event = threading.Event()

    # Start workers
    workers = []
    for w_id, c_mask, c_name in worker_configs:
        w = NPUWorker(w_id, c_mask, c_name, model_path,
                      img_size, reg_max, conf_thr, iou_thr,
                      input_queue, output_queue, stop_event)
        w.start()
        workers.append(w)

    # Open Video Source
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Error] Failed to open source: {source}")
        stop_event.set()
        return

    # Reader thread
    def reader_fn():
        fid = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            while not stop_event.is_set():
                try:
                    input_queue.put((fid, frame), timeout=0.1)
                    break
                except queue.Full:
                    continue
            fid += 1

        # Send sentinels to all workers
        for _ in workers:
            while not stop_event.is_set():
                try:
                    input_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    reader_thread = threading.Thread(target=reader_fn, name="FrameReader", daemon=True)
    reader_thread.start()

    writer = None
    fps_counter = FPSCounter()
    reorder_buffer = {}
    next_frame_id = 0
    total_processed = 0

    mode_label = f"{len(worker_configs)} Worker(s) [{', '.join(c[2] for c in worker_configs)}]"

    print("[Info] Inference pipeline running. Press 'q' or Ctrl+C to stop.")
    try:
        while not stop_event.is_set():
            try:
                item = output_queue.get(timeout=0.2)
            except queue.Empty:
                if not reader_thread.is_alive() and all(not w.is_alive() for w in workers):
                    break
                continue

            fid, frame, (boxes, scores, cls_ids), t_prep, t_npu, t_post, core_name = item
            reorder_buffer[fid] = (frame, boxes, scores, cls_ids, t_prep, t_npu, t_post, core_name)
            output_queue.task_done()

            # Process completed frames in strict sequential order
            while next_frame_id in reorder_buffer:
                frame, boxes, scores, cls_ids, t_prep, t_npu, t_post, core_name = reorder_buffer.pop(next_frame_id)
                next_frame_id += 1
                total_processed += 1
                fps_counter.update()
                cur_fps = fps_counter.get_fps()

                # Draw boxes
                out_frame = U.draw(frame, boxes, scores, cls_ids, names, conf_thr=conf_thr)
                # Draw HUD
                out_frame = draw_hud(out_frame, cur_fps, t_prep, t_npu, t_post, len(boxes), mode_label)

                # Save video
                if save_path:
                    if writer is None:
                        src_fps = cap.get(cv2.CAP_PROP_FPS)
                        if src_fps <= 0 or src_fps > 120 or np.isnan(src_fps):
                            src_fps = 30.0
                        h, w = out_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(save_path, fourcc, src_fps, (w, h))
                    writer.write(out_frame)

                # Show GUI
                if show:
                    cv2.imshow("DAMO-YOLO RK3588 Multi-NPU", out_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        stop_event.set()
                        break
                elif total_processed % 30 == 0:
                    print(f"  Frame {total_processed:5d} | FPS: {cur_fps:5.1f} | NPU: {t_npu:4.1f} ms | Objs: {len(boxes)}")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    finally:
        stop_event.set()
        reader_thread.join(timeout=1.0)
        for w in workers:
            w.join(timeout=1.0)
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[ok] Video saved to {save_path}")
        if show:
            cv2.destroyAllWindows()
        print(f"[Done] Total frames processed: {total_processed}")


# ---------------------------------------------------------------------------
# Single Image Inference & Benchmark
# ---------------------------------------------------------------------------
def run_single_image(source, model_path, core_mask, core_name,
                     img_size, reg_max, names, conf_thr, iou_thr,
                     save_path=None, show=False, benchmark=False,
                     warmup_loops=10, bench_loops=100):
    if RKNNLite is None:
        raise RuntimeError("rknnlite is not installed. Please run on the RK3588 board.")

    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    h, w = frame.shape[:2]

    print(f"\n[Info] Loading model: {model_path}")
    rknn = RKNNLite()
    assert rknn.load_rknn(model_path) == 0, "load_rknn failed"
    print(f"[Info] Initializing runtime on {core_name} (core_mask={core_mask})")
    assert rknn.init_runtime(core_mask=core_mask) == 0, "init_runtime failed"

    def detect_once(img):
        t0 = time.perf_counter()
        inp = U.preprocess(img, img_size)[None]
        t1 = time.perf_counter()
        outs = rknn.inference(inputs=[inp], data_format='nhwc')
        t2 = time.perf_counter()
        res = U.postprocess(outs, img_size, (w, h), conf_thr=conf_thr,
                            iou_thr=iou_thr, reg_max=reg_max)
        t3 = time.perf_counter()
        return res, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0

    # 1. Warmup
    print(f"[Info] Warming up ({warmup_loops} iterations)...")
    for _ in range(warmup_loops):
        detect_once(frame)

    # 2. Benchmark or single test
    if benchmark:
        print(f"[Info] Running benchmark ({bench_loops} iterations)...")
        prep_times, npu_times, post_times, total_times = [], [], [], []
        for _ in range(bench_loops):
            _, tp, tn, tpo = detect_once(frame)
            prep_times.append(tp)
            npu_times.append(tn)
            post_times.append(tpo)
            total_times.append(tp + tn + tpo)

        p_m, n_m, po_m, tot_m = np.mean(prep_times), np.mean(npu_times), np.mean(post_times), np.mean(total_times)
        tot_std = np.std(total_times)
        fps = 1000.0 / tot_m if tot_m > 0 else 0

        print("\n" + "=" * 54)
        print(f"       RK3588 NPU Benchmark ({core_name})")
        print("=" * 54)
        print(f"Model          : {os.path.basename(model_path)}")
        print(f"Image Resolution: {w}x{h} -> Model Input: {img_size}x{img_size}")
        print(f"NPU Core Config: {core_name}")
        print(f"Loops / Warmup : {bench_loops} / {warmup_loops}")
        print("-" * 54)
        print(f"Pre-process    : {p_m:6.2f} ms  (min: {min(prep_times):.2f}, max: {max(prep_times):.2f})")
        print(f"NPU Inference  : {n_m:6.2f} ms  (min: {min(npu_times):.2f}, max: {max(npu_times):.2f})")
        print(f"Post-process   : {po_m:6.2f} ms  (min: {min(post_times):.2f}, max: {max(post_times):.2f})")
        print(f"Total Latency  : {tot_m:6.2f} ms  (std: {tot_std:.2f} ms)")
        print(f"Equivalent FPS : {fps:6.1f} FPS")
        print("=" * 54 + "\n")

    # Run one final detection for display/saving
    (boxes, scores, cls_ids), tp, tn, tpo = detect_once(frame)
    print(f"[Result] Detected {len(boxes)} object(s):")
    for b, s, c in zip(boxes, scores, cls_ids):
        cname = names[c] if c < len(names) else str(c)
        print(f"  - {cname:16s} score: {s:.2f}  box: {b.round().astype(int).tolist()}")
    print(f"Latency: Pre {tp:.1f}ms, NPU {tn:.1f}ms, Post {tpo:.1f}ms, Total {tp+tn+tpo:.1f}ms")

    out_img = U.draw(frame, boxes, scores, cls_ids, names, conf_thr=conf_thr)
    out_img = draw_hud(out_img, 1000.0 / (tp + tn + tpo), tp, tn, tpo, len(boxes), core_name)

    out_file = save_path or "result.jpg"
    cv2.imwrite(out_file, out_img)
    print(f"[ok] Result image saved to: {out_file}")

    if show:
        cv2.imshow("DAMO-YOLO RK3588", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    rknn.release()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="DAMO-YOLO inference on RK3588 (Orange Pi 5 Plus) supporting 1-3 NPU cores."
    )
    p.add_argument('--source', required=True,
                   help='Image file, video file, or camera index/device (/dev/video0, 0, or rtsp://)')
    p.add_argument('--model', default=None,
                   help='Path to .rknn file (auto-detected if omitted)')
    p.add_argument('--info', default=None,
                   help='Path to model_info.json (auto-detected if omitted)')
    p.add_argument('--cores', type=int, default=3, choices=[1, 2, 3],
                   help='Number of NPU cores to use: 1, 2, or 3 (default: 3)')
    p.add_argument('--core', type=int, default=None, choices=[0, 1, 2],
                   help='Bind to a single specific core: 0, 1, or 2 (overrides --cores)')
    p.add_argument('--conf', type=float, default=0.3,
                   help='Confidence threshold (default: 0.3)')
    p.add_argument('--iou', type=float, default=0.45,
                   help='NMS IoU threshold (default: 0.45)')
    p.add_argument('--save', default=None,
                   help='Path to save output result (image or video)')
    p.add_argument('--show', action='store_true',
                   help='Display real-time window with cv2.imshow')
    p.add_argument('--benchmark', action='store_true',
                   help='Run benchmark and report latency/FPS')
    p.add_argument('--bench_loops', type=int, default=100,
                   help='Number of iterations for image benchmark (default: 100)')
    p.add_argument('--warmup', type=int, default=10,
                   help='Warmup iterations before benchmark (default: 10)')
    args = p.parse_args()

    # Locate model_info.json
    info_path = args.info or os.path.join(HERE, 'model_info.json')
    if not os.path.isfile(info_path):
        # Look in workspace
        alt = glob.glob(os.path.join(HERE, '**', 'model_info.json'), recursive=True)
        if alt:
            info_path = alt[0]
        else:
            raise FileNotFoundError("model_info.json not found. Please provide with --info.")

    with open(info_path, encoding='utf-8') as f:
        info = json.load(f)

    # Locate .rknn model
    if args.model:
        model_path = args.model
    else:
        # Check current folder or info['model']
        cands = [os.path.join(HERE, info.get('model', '')),
                 *glob.glob(os.path.join(HERE, '*.rknn'))]
        valid = [c for c in cands if os.path.isfile(c)]
        if not valid:
            raise FileNotFoundError("Could not find .rknn file. Please specify with --model.")
        model_path = valid[0]

    img_size = info['img_size']
    reg_max = info['reg_max']
    names = info['class_names']

    core_data = get_rknn_core_masks()
    if args.core is not None:
        core_mask = core_data['single_cores'][args.core]
        core_name = f"Core {args.core}"
    else:
        core_mask = core_data['multi_core_map'][args.cores]
        core_name = core_data['names'][core_mask]

    is_image = os.path.isfile(args.source) and \
        args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

    if is_image:
        run_single_image(
            source=args.source,
            model_path=model_path,
            core_mask=core_mask,
            core_name=core_name,
            img_size=img_size,
            reg_max=reg_max,
            names=names,
            conf_thr=args.conf,
            iou_thr=args.iou,
            save_path=args.save,
            show=args.show,
            benchmark=args.benchmark,
            warmup_loops=args.warmup,
            bench_loops=args.bench_loops
        )
    else:
        # Video, Camera, or Stream -> Multi-threaded Pipeline
        run_pipeline(
            source=args.source,
            model_path=model_path,
            num_workers=args.cores,
            specific_core=args.core,
            img_size=img_size,
            reg_max=reg_max,
            names=names,
            conf_thr=args.conf,
            iou_thr=args.iou,
            save_path=args.save,
            show=args.show
        )


if __name__ == '__main__':
    main()