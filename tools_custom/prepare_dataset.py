"""Normalize a COCO / YOLO / Roboflow dataset into the layout DAMO-YOLO expects.

Output layout (out_dir):
    train/<images>            (symlinks, or copies if symlinks are unavailable)
    val/<images>
    annotations/train.json    COCO json, category ids 1..K in class_names order
    annotations/val.json
    classes.json              {"class_names": [...]}
"""
import glob
import json
import os
import random
import shutil

from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
SPLIT_ALIASES = {
    'train': 'train', 'training': 'train',
    'val': 'val', 'valid': 'val', 'validation': 'val', 'val2017': 'val',
    'test': 'test', 'testing': 'test', 'train2017': 'train',
}


def _split_of(path, root, use_name=False):
    """Guess the split from the folders below root (None if unknown).

    use_name: also look for split keywords in the file name itself
    (for json files such as instances_train2017.json).
    """
    rel = os.path.relpath(path, root).replace('\\', '/').lower()
    for part in rel.split('/')[:-1][::-1]:
        if part in SPLIT_ALIASES:
            return SPLIT_ALIASES[part]
    if use_name:
        name = os.path.basename(rel)
        for key, split in SPLIT_ALIASES.items():
            if key in name:
                return split
    return None


def _walk_files(root, exts):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith('.') and d != '__MACOSX']
        for f in filenames:
            if f.lower().endswith(exts) and not f.startswith('._'):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _descend_single_dir(root):
    """Zips often contain a single top-level folder; step into it."""
    while True:
        entries = [e for e in os.listdir(root)
                   if not e.startswith('.') and e != '__MACOSX']
        if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
            root = os.path.join(root, entries[0])
        else:
            return root


def _image_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def _is_coco_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            head = f.read(4096)
        return '"images"' in head or '"annotations"' in head or \
            '"categories"' in head
    except (OSError, UnicodeDecodeError):
        return False


def _load_coco(root):
    jsons = [j for j in _walk_files(root, ('.json', ))
             if _is_coco_json(j)]
    if not jsons:
        raise RuntimeError(f'No COCO json found under {root}')

    image_index = {}
    for p in _walk_files(root, IMG_EXTS):
        image_index.setdefault(os.path.basename(p), []).append(p)

    raw_splits = {}
    for jp in jsons:
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'images' not in data or 'categories' not in data:
            continue
        split = _split_of(jp, root, use_name=True) or 'train'
        if split in raw_splits:
            print(f'[warn] more than one json for split "{split}", '
                  f'merging {jp}')
        raw_splits.setdefault(split, []).append((jp, data))
    if not raw_splits:
        raise RuntimeError('COCO json files found, but none has '
                           '"images" and "categories".')

    # category ids -> names, ordered by original id (first json wins)
    cat_order = []
    for split in ('train', 'val', 'test'):
        for _, data in raw_splits.get(split, []):
            for c in sorted(data['categories'], key=lambda c: c['id']):
                if c['name'] not in cat_order:
                    cat_order.append(c['name'])

    splits = {}
    for split, items in raw_splits.items():
        records = []
        for jp, data in items:
            jdir = os.path.dirname(jp)
            id2name = {c['id']: c['name'] for c in data['categories']}
            anns_by_img = {}
            for a in data.get('annotations', []):
                anns_by_img.setdefault(a['image_id'], []).append(a)
            missing = 0
            for im in data['images']:
                fn = im['file_name']
                cands = [os.path.join(jdir, fn), os.path.join(root, fn),
                         os.path.join(jdir, 'images', fn)]
                path = next((c for c in cands if os.path.isfile(c)), None)
                if path is None:
                    hits = image_index.get(os.path.basename(fn), [])
                    same_split = [h for h in hits
                                  if _split_of(h, root) == split]
                    hits = same_split or hits
                    path = hits[0] if hits else None
                if path is None:
                    missing += 1
                    continue
                anns = []
                for a in anns_by_img.get(im['id'], []):
                    if a.get('iscrowd', 0):
                        continue
                    x, y, w, h = a['bbox']
                    anns.append((id2name[a['category_id']], x, y, w, h))
                records.append({'path': path, 'width': im.get('width'),
                                'height': im.get('height'), 'anns': anns})
            if missing:
                print(f'[warn] {missing} images listed in {jp} were not '
                      f'found on disk and are skipped')
        splits[split] = records
    return splits, cat_order


# --------------------------------------------------------------------------- #
# YOLO txt
# --------------------------------------------------------------------------- #
def _read_class_names_yolo(root):
    for pattern in ('data.yaml', 'data.yml', '*.yaml', '*.yml'):
        for yp in glob.glob(os.path.join(root, '**', pattern),
                            recursive=True):
            try:
                import yaml
                with open(yp, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                continue
            names = cfg.get('names') if isinstance(cfg, dict) else None
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=int)]
            if isinstance(names, list):
                return [str(n) for n in names]
    for fname in ('classes.txt', 'obj.names', 'classes.names',
                  '_darknet.labels', 'labels.txt'):
        for p in glob.glob(os.path.join(root, '**', fname), recursive=True):
            with open(p, 'r', encoding='utf-8') as f:
                names = [l.strip() for l in f if l.strip()]
            if names:
                return names
    return None


def _label_path_for(img_path):
    p = img_path.replace('\\', '/')
    idx = p.rfind('/images/')
    stem = os.path.splitext(p)[0]
    if idx >= 0:
        stem = os.path.splitext(p[:idx] + '/labels/' + p[idx + 8:])[0]
    return stem + '.txt'


def _load_yolo(root):
    names = _read_class_names_yolo(root)
    splits = {}
    max_cls = -1
    no_label = 0
    for img in _walk_files(root, IMG_EXTS):
        lp = _label_path_for(img)
        anns = []
        if os.path.isfile(lp):
            w, h = _image_size(img)
            with open(lp, 'r', encoding='utf-8') as f:
                for line in f:
                    v = line.split()
                    if len(v) < 5:
                        continue
                    c = int(float(v[0]))
                    nums = [float(t) for t in v[1:]]
                    if len(nums) == 4:
                        cx, cy, bw, bh = nums
                        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                        bw, bh = bw * w, bh * h
                    else:  # polygon (segmentation export) -> bbox
                        xs, ys = nums[0::2], nums[1::2]
                        x1, y1 = min(xs) * w, min(ys) * h
                        bw, bh = max(xs) * w - x1, max(ys) * h - y1
                    anns.append((c, x1, y1, bw, bh))
                    max_cls = max(max_cls, c)
            rec = {'path': img, 'width': w, 'height': h, 'anns': anns}
        else:
            no_label += 1
            rec = {'path': img, 'width': None, 'height': None, 'anns': []}
        split = _split_of(img, root) or 'train'
        splits.setdefault(split, []).append(rec)
    if not splits:
        raise RuntimeError(f'No images found under {root}')
    if no_label:
        print(f'[info] {no_label} images have no label file '
              f'(kept as background images)')
    if names is None:
        names = [f'class_{i}' for i in range(max_cls + 1)]
        print('[warn] no data.yaml / classes.txt found, using generic names')
    if max_cls >= len(names):
        raise RuntimeError(f'Label uses class id {max_cls}, but only '
                           f'{len(names)} class names were found')
    for recs in splits.values():
        for r in recs:
            r['anns'] = [(names[c], x, y, w, h) for c, x, y, w, h in r['anns']]
    return splits, names


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def _link_or_copy(src, dst):
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _write_split(records, class_names, out_dir, split):
    img_dir = os.path.join(out_dir, split)
    os.makedirs(img_dir, exist_ok=True)
    name2id = {n: i + 1 for i, n in enumerate(class_names)}
    images, annotations, used = [], [], set()
    dropped = 0
    for img_id, r in enumerate(records, start=1):
        base = os.path.basename(r['path'])
        fname, k = base, 1
        while fname.lower() in used:
            stem, ext = os.path.splitext(base)
            fname, k = f'{stem}_{k}{ext}', k + 1
        used.add(fname.lower())
        _link_or_copy(r['path'], os.path.join(img_dir, fname))
        w, h = r['width'], r['height']
        if not w or not h:
            w, h = _image_size(r['path'])
        images.append({'id': img_id, 'file_name': fname,
                       'width': int(w), 'height': int(h)})
        for name, x, y, bw, bh in r['anns']:
            if name not in name2id:
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
            if x2 - x1 < 1 or y2 - y1 < 1:
                dropped += 1
                continue
            annotations.append({
                'id': len(annotations) + 1, 'image_id': img_id,
                'category_id': name2id[name],
                'bbox': [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                'area': round((x2 - x1) * (y2 - y1), 2),
                'iscrowd': 0, 'segmentation': []})
    if dropped:
        print(f'[info] {split}: dropped {dropped} boxes smaller than 1 px '
              f'after clipping')
    coco = {'images': images, 'annotations': annotations,
            'categories': [{'id': i + 1, 'name': n, 'supercategory': 'none'}
                           for i, n in enumerate(class_names)]}
    os.makedirs(os.path.join(out_dir, 'annotations'), exist_ok=True)
    with open(os.path.join(out_dir, 'annotations', f'{split}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)
    return len(images), len(annotations)


def prepare_dataset(src, out_dir, fmt='auto', val_ratio=0.15, seed=0):
    """Convert `src` (folder) into DAMO-YOLO layout in `out_dir`.

    fmt: 'auto' | 'coco' | 'yolo'.  Returns the list of class names.
    """
    root = _descend_single_dir(src)
    if fmt == 'auto':
        has_coco = any(_is_coco_json(j)
                       for j in _walk_files(root, ('.json', )))
        fmt = 'coco' if has_coco else 'yolo'
    print(f'[info] dataset root: {root}  (format: {fmt})')
    splits, cat_order = _load_coco(root) if fmt == 'coco' else _load_yolo(root)

    # drop categories that never occur (e.g. Roboflow's id-0 super category)
    counts = {n: 0 for n in cat_order}
    for recs in splits.values():
        for r in recs:
            for a in r['anns']:
                counts[a[0]] = counts.get(a[0], 0) + 1
    class_names = [n for n in cat_order if counts.get(n, 0) > 0]
    unused = [n for n in cat_order if counts.get(n, 0) == 0]
    if unused:
        print(f'[info] categories without any box are removed: {unused}')
    if not class_names:
        raise RuntimeError('No bounding boxes found in the dataset.')

    val_source = 'val'
    if 'val' not in splits:
        if 'test' in splits:
            print('[info] no val split, using the test split as val')
            splits['val'] = splits.pop('test')
            val_source = 'test'
        else:
            val_source = 'random_split'
            recs = splits['train'][:]
            random.Random(seed).shuffle(recs)
            n_val = max(1, int(len(recs) * val_ratio))
            splits['val'], splits['train'] = recs[:n_val], recs[n_val:]
            print(f'[info] no val split, randomly moved {n_val} images '
                  f'({val_ratio:.0%}) from train to val')
    if 'train' not in splits or not splits['train']:
        raise RuntimeError('No training images found.')

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for split in ('train', 'val'):
        n_img, n_ann = _write_split(splits[split], class_names, out_dir,
                                    split)
        print(f'[ok] {split}: {n_img} images, {n_ann} boxes')
    with open(os.path.join(out_dir, 'classes.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'class_names': class_names, 'val_source': val_source}, f,
                  ensure_ascii=False, indent=2)
    print(f'[ok] {len(class_names)} classes: {class_names}')
    return class_names