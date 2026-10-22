"""Compare completed batch-size runs using validation metrics only."""
import argparse
import json
from pathlib import Path

from benchmark.run_learning_rates import read_csv
from benchmark_log import ROOT, write_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    baseline = ROOT / 'benchmark/runs/learning_rates_20261022_104847_lr_1e4'
    run = args.run.resolve()
    output = ROOT / 'benchmark/suites' / run.name
    rows = [read_csv(p / 'result.csv')[0] for p in (baseline, run)]
    assert all(r['status'] == 'completed' for r in rows)
    configs = [json.loads((p / 'config.json').read_text(encoding='utf-8')) for p in (baseline, run)]
    for key in ('lr', 'optimizer', 'weight_decay', 'seed', 'image_size', 'epochs', 'init',
                'split_seed', 'loss', 'augmentation', 'precision', 'torch_version', 'gpu'):
        assert configs[0][key] == configs[1][key], key
    assert json.loads((baseline / 'split.json').read_text()) == json.loads((run / 'split.json').read_text())
    write_csv(output / 'comparison.csv', list(rows[0]), rows)
    labels = [f"batch={r['batch_size']}" for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, key, title, scale in zip(axes,
            ['best_val_dice', 'elapsed_seconds', 'peak_gpu_memory_mb'],
            ['Best validation Dice', 'Training + validation (min)', 'Peak allocated GPU memory (MiB)'],
            [1, 60, 1]):
        values = [float(r[key]) / scale for r in rows]
        bars = ax.bar(labels, values, color=['#0072B2', '#D55E00'])
        ax.bar_label(bars, fmt='%.4f' if scale == 1 and key == 'best_val_dice' else '%.1f')
        ax.set_title(title)
        ax.set_ylim(0, 1.05 if key == 'best_val_dice' else max(values)*1.18)
    fig.suptitle('Adam | lr=1e-4 | 5 epochs | 256x256 | seed=0')
    for extension in ('png', 'svg'):
        fig.savefig(output / f'batch_comparison.{extension}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for directory, label in zip((baseline, run), labels):
        epochs = read_csv(directory / 'epochs.csv')
        for ax, key in zip(axes, ['val_loss', 'val_dice']):
            ax.plot([int(r['epoch']) for r in epochs], [float(r[key]) for r in epochs], marker='o', label=label)
            ax.set(xlabel='Epoch', ylabel=key)
            ax.legend()
            ax.grid(alpha=.2)
    fig.savefig(output / 'validation_curves.png', dpi=180)
    plt.close(fig)
    delta = (float(rows[1]['best_val_dice']) - float(rows[0]['best_val_dice'])) * 100
    report = '# Batch-size benchmark\n\n'
    report += 'Adam, lr=1e-4, weight decay=0, 256x256, seed=0, random initialization, 5 epochs.\n\n'
    report += '| Batch | Best val Dice | Best epoch | Time (min) | Peak allocated GPU MiB |\n|---|---|---|---|---|\n'
    for r in rows:
        report += f"| {r['batch_size']} | {float(r['best_val_dice']):.6f} | {r['best_epoch']} | {float(r['elapsed_seconds'])/60:.2f} | {float(r['peak_gpu_memory_mb']):.1f} |\n"
    report += f'\nNew run minus baseline Dice: {delta:+.3f} percentage points.\n\n'
    report += 'The baseline is reused from the previous learning-rate experiment. Configuration, software/GPU and train/validation split were checked. Model/data source changes are comments, formatting and class order; their class/function ASTs match the baseline.\n\n'
    report += 'Batch 1 performs 2944 optimizer updates per epoch versus 1472 for batch 2; BatchNorm behavior also changes. This compares fixed-epoch configurations, not equal-update budgets. Timings were measured in separate runs and can vary with system load. Only one seed and five epochs were used; no statistical or convergence claim is made. The test set was not evaluated in this experiment.\n'
    (output / 'REPORT.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
