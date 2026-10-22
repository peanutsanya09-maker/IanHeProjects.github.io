"""串行执行固定预算学习率实验、预测、CSV 汇总与绘图。"""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmark_log import ROOT, write_csv

LEARNING_RATES = [('lr_1e3', 1e-3), ('lr_3e4', 3e-4), ('lr_1e4', 1e-4)]


def read_csv(path: Path) -> list:
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def execute(command: list, log_path: Path) -> None:
    """将子进程输出同时写入终端和日志；失败立即停止实验。"""
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f'Command failed; see {log_path}')


def plot_results(rows: list, output: Path) -> None:
    labels = [f"{float(row['lr']):.0e}" for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, key, title in zip(axes, ['best_val_dice', 'test_global_dice', 'elapsed_seconds'],
                              ['Best validation Dice', 'Test Dice (descriptive only)', 'Training + validation time (min)']):
        values = [float(row[key])/(60 if key == 'elapsed_seconds' else 1) for row in rows]
        bars = ax.bar(labels, values, color=['#2563eb', '#0d9488', '#ea580c'])
        ax.bar_label(bars, fmt='%.4f' if key != 'elapsed_seconds' else '%.1f', padding=4)
        ax.set(xlabel='Learning rate', title=title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.08 if key != 'elapsed_seconds' else max(values)*1.18)
    fig.suptitle('Fixed U-Net | Adam | 5 epochs | batch=2 | 256x256 | seed=0 (one run/config)')
    fig.savefig(output / 'learning_rate_bars.png', dpi=180)
    fig.savefig(output / 'learning_rate_bars.svg')
    plt.close(fig)


def main():
    suite = ROOT / 'benchmark' / 'suites' / datetime.now().strftime('learning_rates_%Y%m%d_%H%M%S')
    suite.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'rates': LEARNING_RATES, 'epochs': 5, 'seed': 0,
                'selection': 'highest validation global Dice; test not used for selection',
                'unet_sha256': hashlib.sha256((ROOT / 'unet.py').read_bytes()).hexdigest(), 'runs': {}}
    manifest_path = suite / 'manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    print(f'SUITE: {suite}', flush=True)
    rows = []
    try:
        for name, lr in LEARNING_RATES:
            run_dir = ROOT / 'benchmark' / 'runs' / f'{suite.name}_{name}'
            manifest['runs'][name] = str(run_dir.relative_to(ROOT))
            save_manifest()
            execute([sys.executable, '-X', 'utf8', '-u', 'train_step.py', '--train', '--lr', str(lr),
                     '--epochs', '5', '--seed', '0', '--batch-size', '2', '--image-size', '256',
                     '--optimizer', 'adam', '--weight-decay', '0', '--run-dir', str(run_dir)], suite / f'{name}_train.log')
            execute([sys.executable, '-X', 'utf8', '-u', '-m', 'benchmark.evaluate',
                     '--model', str(run_dir / 'best.pth'), '--output-dir', str(run_dir / 'predictions')],
                    suite / f'{name}_predict.log')
            row = read_csv(run_dir / 'result.csv')[0]
            row.update(read_csv(run_dir / 'predictions' / 'summary.csv')[0])
            row['checkpoint_sha256'] = hashlib.sha256((run_dir / 'best.pth').read_bytes()).hexdigest()
            rows.append(row)
            write_csv(suite / 'results.csv', list(row), rows)
        # 确认三个实验模型代码和数据划分完全一致。
        paths = [ROOT / p for p in manifest['runs'].values()]
        for filename in ('unet.py', 'split.json'):
            assert len({hashlib.sha256((p / filename).read_bytes()).hexdigest() for p in paths}) == 1
        plot_results(rows, suite)
        manifest['status'] = 'completed'
        manifest['best_by_validation'] = max(rows, key=lambda r: float(r['best_val_dice']))['run_id']
        save_manifest()
        print(f'COMPLETE: {suite}', flush=True)
    except BaseException:
        manifest['status'] = 'failed_or_interrupted'
        save_manifest()
        raise


if __name__ == '__main__':
    main()
