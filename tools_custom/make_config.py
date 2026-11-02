"""Write a DAMO-YOLO config for the custom dataset, based on an official one."""
import os

TEMPLATES = {
    # size: (official config, Google Drive id of COCO weights, file name)
    # The AliYun OSS links in the README return 404 (checked 2026-10); these
    # are the README's Google Drive mirrors of the T*/S*/M* models. Their
    # backbone/neck match the current configs; only the class head differs,
    # which finetuning re-initializes anyway.
    'T': ('damoyolo_tinynasL20_T.py', '1-6fBf_oe9vITSTYgQkaYklL94REz2zCh',
          'damoyolo_tinynasL20_T.pth'),
    'S': ('damoyolo_tinynasL25_S.py', '1-O-ObHN970GRVKkL1TiAxfoMCpYGJS6B',
          'damoyolo_tinynasL25_S.pth'),
    'M': ('damoyolo_tinynasL35_M.py', '1-RoKaO7U9U1UrweJb7c4Hs_S_qKFDExc',
          'damoyolo_tinynasL35_M.pth'),
}


def make_config(repo, model_size, exp_name, class_names, output_dir,
                img_size=640, epochs=100, batch_size=16, eval_interval=5,
                ckpt_interval=10, finetune_path=None, resume_path=None,
                num_workers=2):
    assert img_size % 32 == 0, 'IMG_SIZE must be a multiple of 32'
    template = os.path.join(repo, 'configs', TEMPLATES[model_size][0])
    with open(template, 'r', encoding='utf-8') as f:
        src = f.read().rstrip() + '\n'
    no_aug = min(16, max(1, epochs // 6))
    warmup = min(5, max(1, epochs // 20))
    overrides = {
        'self.miscs.exp_name': exp_name,
        'self.miscs.output_dir': output_dir,
        'self.miscs.eval_interval_epochs': eval_interval,
        'self.miscs.ckpt_interval_epochs': ckpt_interval,
        'self.miscs.num_workers': num_workers,
        'self.train.batch_size': batch_size,
        'self.train.total_epochs': epochs,
        'self.train.no_aug_epochs': no_aug,
        'self.train.warmup_epochs': warmup,
        'self.train.finetune_path': None if resume_path else finetune_path,
        'self.train.resume_path': resume_path,
        'self.test.batch_size': 8,
        'self.dataset.train_ann': ('custom_train_coco', ),
        'self.dataset.val_ann': ('custom_val_coco', ),
        "self.model.head['num_classes']": len(class_names),
        'self.dataset.class_names': list(class_names),
        'self.train.augment.transform.image_max_range': (img_size, img_size),
        'self.train.augment.mosaic_mixup.mosaic_size': (img_size, img_size),
        'self.test.augment.transform.image_max_range': (img_size, img_size),
    }
    lines = ['', '        # ---- custom dataset overrides (generated) ----']
    lines += [f'        {k} = {v!r}' for k, v in overrides.items()]
    out = os.path.join(repo, 'configs', f'{exp_name}.py')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(src + '\n'.join(lines) + '\n')
    return out