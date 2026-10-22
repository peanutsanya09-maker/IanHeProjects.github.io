"""训练实验日志：逐轮 CSV、独立权重目录、配置与实验汇总。"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parent
EPOCH_FIELDS = ['epoch', 'lr', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'epoch_seconds', 'elapsed_seconds']
RESULT_FIELDS = ['run_id', 'status', 'optimizer', 'lr', 'weight_decay', 'batch_size', 'image_size', 'seed', 'epochs_requested', 'epochs_completed', 'best_epoch', 'best_val_dice', 'val_loss_at_best', 'elapsed_seconds', 'peak_gpu_memory_mb', 'source', 'run_dir']


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkLog:
    def __init__(self, args, train_set, val_set, device):
        self.args = args
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6]
        self.directory = Path(args.run_dir).resolve() if getattr(args, 'run_dir', None) else ROOT / 'benchmark' / 'runs' / self.run_id
        self.run_id = self.directory.name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.best = None
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else '',
                      init='random', split_seed=0, loss='CrossEntropyLoss(ignore_index=255)',
                      augmentation='none', precision='float32', dice='global foreground; ignore 255',
                      timing='train+validation+checkpoint; excludes initialization; includes logging overhead')
        config['source_sha256'] = {}
        for name in ('unet.py', 'pet_dataset.py', 'train_step.py', 'benchmark_log.py'):
            data = (ROOT / name).read_bytes()
            (self.directory / name).write_bytes(data)
            config['source_sha256'][name] = hashlib.sha256(data).hexdigest()
        dataset = train_set.dataset
        split_ids = {'train': [dataset.ids[i] for i in train_set.indices],
                     'validation': [dataset.ids[i] for i in val_set.indices]}
        (self.directory / 'split.json').write_text(json.dumps(split_ids, indent=2), encoding='utf-8')
        (self.directory / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        self.summary('running')

    def add_epoch(self, row):
        self.rows.append(row)
        if self.best is None or row['val_dice'] > self.best['val_dice']:
            self.best = row
        write_csv(self.directory / 'epochs.csv', EPOCH_FIELDS, self.rows)
        self.summary('running')

    def summary(self, status):
        row = dict(run_id=self.run_id, status=status, optimizer=self.args.optimizer, lr=self.args.lr,
                   weight_decay=self.args.weight_decay, batch_size=self.args.batch_size,
                   image_size=self.args.image_size, seed=self.args.seed, epochs_requested=self.args.epochs,
                   epochs_completed=len(self.rows), best_epoch=self.best['epoch'] if self.best else '',
                   best_val_dice=self.best['val_dice'] if self.best else '',
                   val_loss_at_best=self.best['val_loss'] if self.best else '',
                   elapsed_seconds=self.rows[-1]['elapsed_seconds'] if self.rows else '',
                   peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else '',
                   source='automatic_training_log', run_dir=str(self.directory.relative_to(ROOT)))
        write_csv(self.directory / 'result.csv', RESULT_FIELDS, [row])
        # 单进程串行运行实验；从每个独立实验目录重建汇总，保留历史记录。
        rows = []
        for path in sorted((ROOT / 'benchmark' / 'runs').glob('*/result.csv')):
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows.extend(csv.DictReader(f))
        write_csv(ROOT / 'benchmark' / 'results.csv', RESULT_FIELDS, rows)
