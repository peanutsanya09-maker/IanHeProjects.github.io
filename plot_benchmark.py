"""绘制已完成实验；不执行训练。历史记录与正式实验分开显示。"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'benchmark'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    output = ROOT / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    for directory in sorted((ROOT / 'runs').iterdir()):
        path = directory / 'epochs.csv'
        if not path.is_file():
            continue
        rows = read(path)
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        epochs = [int(r['epoch']) for r in rows]
        for ax, metric in zip(axes, ['loss', 'dice']):
            for split in ['train', 'val']:
                ax.plot(epochs, [float(r[f'{split}_{metric}']) for r in rows], marker='o', label=split)
            ax.set(xlabel='Epoch', ylabel=metric.title(), title=f'{metric.title()} vs epoch')
            ax.grid(alpha=.25)
            ax.legend()
        axes[1].set_ylim(0, 1)
        fig.suptitle(directory.name + (' (imported user log)' if directory.name.startswith('historical') else ''))
        fig.savefig(output / f'{directory.name}_curves.png', dpi=160)
        plt.close(fig)

    rows = read(ROOT / 'results.csv')
    # 不混入缺乏环境和耗时信息的历史记录，也不比较未完成实验。
    completed = [r for r in rows if r['status'] == 'completed' and r['source'] == 'automatic_training_log']
    if completed:
        labels = [f"{r['run_id']}\nlr={r['lr']}, bs={r['batch_size']}, {r['optimizer']}, wd={r['weight_decay']}" for r in completed]
        fig, ax = plt.subplots(figsize=(max(9, len(completed)*2), 5), constrained_layout=True)
        ax.bar(range(len(completed)), [float(r['best_val_dice']) for r in completed])
        ax.set_xticks(range(len(completed)), labels, rotation=30, ha='right', fontsize=8)
        ax.set(ylabel='Best validation Dice', ylim=(0, 1), title='Completed runs (check configs before causal comparisons)')
        fig.savefig(output / 'run_comparison.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for r in completed:
            x, y = float(r['elapsed_seconds'])/60, float(r['best_val_dice'])
            ax.scatter(x, y)
            ax.annotate(r['run_id'], (x, y), fontsize=7, xytext=(4, 4), textcoords='offset points')
        ax.set(xlabel='Training + validation time (minutes)', ylabel='Best validation Dice', title='Quality vs measured run time')
        ax.grid(alpha=.25)
        fig.savefig(output / 'quality_vs_time.png', dpi=160)
        plt.close(fig)
    else:
        print('尚无正式 benchmark 结果；仅绘制真实历史曲线，不生成虚假的实验对比。')
    print(f'图表目录：{output}')


if __name__ == '__main__':
    main()
