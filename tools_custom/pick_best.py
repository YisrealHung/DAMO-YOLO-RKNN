"""Utility to automatically select or evaluate the best checkpoint for DAMO-YOLO.

Features:
1. Fast rule-based selection:
   - Priority 1: best_ckpt.pth (saved dynamically during training)
   - Priority 2: Match existing checkpoints against train_log.txt mAP scores
   - Priority 3: latest_ckpt.pth or largest epoch checkpoint
2. Offline batch evaluation (--eval):
   - Evaluates all candidate checkpoints on the validation dataset
   - Displays a ranked leaderboard by mAP50-95
   - Copies or links the best checkpoint to best_ckpt.pth
"""

import argparse
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def parse_train_log(log_path):
    """Parse train_log.txt to extract mAP50-95 by epoch."""
    if not log_path or not os.path.exists(log_path):
        return {}
    pat_ep = re.compile(r'epoch: (\d+)/(\d+), iter: \d+/\d+')
    pat_ap = re.compile(
        r'Average Precision\s+\(AP\) @\[ IoU=(0\.50:0\.95|0\.50)\s+\| area=\s+all \| maxDets=100 \] = (-?[\d.]+)'
    )
    by_epoch = {}
    cur_ep = None
    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat_ep.search(line)
            if m:
                cur_ep = int(m.group(1))
                continue
            m = pat_ap.search(line)
            if m and cur_ep is not None and m.group(1) == '0.50:0.95':
                by_epoch[cur_ep] = float(m.group(2))
    return by_epoch


def pick_best_checkpoint(ckpt_dir, train_log_path=None):
    """Automatically select the best checkpoint from ckpt_dir.

    Returns:
        (best_ckpt_path, reason_description)
    """
    # 1. Check if best_ckpt.pth exists (dynamically saved during training)
    best_path = os.path.join(ckpt_dir, 'best_ckpt.pth')
    if os.path.exists(best_path):
        try:
            import torch
            info = torch.load(best_path, map_location='cpu', weights_only=False)
            ep = info.get('epoch', '?')
            score = info.get('best_ap', None)
            score_str = f', mAP50-95 = {score:.4f}' if score is not None else ''
            return best_path, f'採用訓練中自動保存的最佳權重：best_ckpt.pth (epoch {ep}{score_str})'
        except Exception:
            return best_path, '採用訓練中自動保存的最佳權重：best_ckpt.pth'

    # Gather all existing checkpoints
    ckpt_map = {}
    for p in glob.glob(os.path.join(ckpt_dir, 'epoch_*_ckpt.pth')):
        m = re.search(r'epoch_(\d+)_ckpt', p)
        if m:
            ckpt_map[int(m.group(1))] = p
    latest_path = os.path.join(ckpt_dir, 'latest_ckpt.pth')
    if os.path.exists(latest_path):
        try:
            import torch
            ep = torch.load(latest_path, map_location='cpu', weights_only=False).get('epoch')
            if ep is not None:
                ckpt_map[int(ep)] = latest_path
        except Exception:
            pass

    # 2. Parse train_log.txt and match against existing checkpoints
    if train_log_path and os.path.exists(train_log_path):
        by_epoch = parse_train_log(train_log_path)
        matched = [(ep, by_epoch[ep], ckpt_map[ep]) for ep in ckpt_map if ep in by_epoch]
        if matched:
            best_ep, best_score, best_p = max(matched, key=lambda x: x[1])
            return best_p, f'由 train_log.txt 篩選出最佳存檔權重：{os.path.basename(best_p)} (epoch {best_ep}, mAP50-95 = {best_score:.4f})'

    # 3. Fallback
    if os.path.exists(latest_path):
        return latest_path, '無評估記錄，改用最後訓練完成之權重：latest_ckpt.pth'
    if ckpt_map:
        max_ep = max(ckpt_map.keys())
        return ckpt_map[max_ep], f'無評估記錄，改用 epoch 數最大之 checkpoint：epoch_{max_ep}_ckpt.pth'

    raise FileNotFoundError(f'{ckpt_dir} 內找不到任何 checkpoint！')


def evaluate_all_checkpoints(config_file, ckpt_dir, device='cuda'):
    """Offline evaluation of all checkpoints in ckpt_dir on the validation set."""
    from damo.apis.detector_inference import inference
    from damo.base_models.core.ops import RepConv
    from damo.config.base import parse_config
    from damo.dataset import build_dataloader, build_dataset
    from damo.detectors.detector import build_local_model

    config = parse_config(config_file)
    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(
        val_dataset,
        config.test.augment,
        batch_size=config.test.batch_size,
        num_workers=config.miscs.num_workers,
        is_train=False,
        size_div=32,
    )

    candidates = glob.glob(os.path.join(ckpt_dir, 'epoch_*_ckpt.pth'))
    latest = os.path.join(ckpt_dir, 'latest_ckpt.pth')
    if os.path.exists(latest) and latest not in candidates:
        candidates.append(latest)

    if not candidates:
        raise FileNotFoundError(f'在 {ckpt_dir} 內找不到任何可評估的 checkpoint。')

    def sort_key(p):
        m = re.search(r'epoch_(\d+)_ckpt', p)
        return int(m.group(1)) if m else 999999

    candidates = sorted(candidates, key=sort_key)
    results = []

    print(f'\n🔍 開始對 {len(candidates)} 個 checkpoint 進行離線驗證評估...')
    for ckpt_path in candidates:
        fname = os.path.basename(ckpt_path)
        print(f'--> 正在評估：{fname} ...')
        model = build_local_model(config, device)
        model.head.nms = True
        model.eval()

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state_dict, strict=False)

        for layer in model.modules():
            if isinstance(layer, RepConv):
                layer.switch_to_deploy()

        output_folders = [None] * len(config.dataset.val_ann)
        ap_score = -1.0
        for output_folder, dataset_name, data_loader_val in zip(
                output_folders, config.dataset.val_ann, val_loader):
            res = inference(
                model,
                data_loader_val,
                dataset_name,
                iou_types=('bbox', ),
                box_only=False,
                device=device,
                output_folder=output_folder,
            )
            if res and isinstance(res, tuple) and len(res) > 0:
                coco_res = res[0]
                if hasattr(coco_res, 'results') and 'bbox' in coco_res.results:
                    ap_score = coco_res.results['bbox'].get('AP', -1.0)

        ep = ckpt.get('epoch', '?')
        results.append((ckpt_path, ep, ap_score))
        print(f'    結果: epoch {ep} | mAP50-95 = {ap_score:.4f}')

    ranked = sorted(results, key=lambda x: x[2], reverse=True)
    best_path, best_ep, best_ap = ranked[0]

    print('\n🏆 Checkpoint 評估排行榜：')
    print(f'{"Rank":<6} {"Checkpoint":<24} {"Epoch":<8} {"mAP50-95":<10}')
    print('-' * 50)
    for r, (p, ep, score) in enumerate(ranked, 1):
        print(f'{r:<6} {os.path.basename(p):<24} {str(ep):<8} {score:.4f}')

    best_out = os.path.join(ckpt_dir, 'best_ckpt.pth')
    shutil.copyfile(best_path, best_out)
    print(f'\n✅ 已將最佳權重複製為：{best_out}')
    return best_out, f'離線評估選出：{os.path.basename(best_path)} (epoch {best_ep}, mAP={best_ap:.4f})'


def main():
    parser = argparse.ArgumentParser(description='Auto-select or evaluate best DAMO-YOLO checkpoint')
    parser.add_argument('-c', '--ckpt_dir', required=True, help='Checkpoint directory')
    parser.add_argument('-l', '--log', default=None, help='Path to train_log.txt')
    parser.add_argument('-f', '--config', default=None, help='Config file path (required for --eval)')
    parser.add_argument('--eval', action='store_true', help='Perform offline evaluation on all checkpoints')
    parser.add_argument('--device', default='cuda', help='Compute device (cuda / cpu)')
    args = parser.parse_args()

    if args.eval:
        if not args.config:
            parser.error('--eval 需要指定 --config 設定檔路徑')
        best_path, msg = evaluate_all_checkpoints(args.config, args.ckpt_dir, device=args.device)
    else:
        best_path, msg = pick_best_checkpoint(args.ckpt_dir, args.log)

    print('\n🎯 結果：', msg)
    print('📁 最佳權重：', best_path)


if __name__ == '__main__':
    main()
