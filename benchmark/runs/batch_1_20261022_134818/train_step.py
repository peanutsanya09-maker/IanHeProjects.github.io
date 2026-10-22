"""默认检查数据；加 --train 才训练。原矩形练习保存在 train_rectangle.py。"""
import argparse
import time
import random
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from pet_dataset import DEFAULT_ROOT, IGNORE_INDEX, make_train_val
from unet import UNet
from benchmark_log import BenchmarkLog


def dice_counts(logits, masks):
    valid = masks != IGNORE_INDEX
    predicted = (logits.argmax(dim=1) == 1) & valid
    actual = (masks == 1) & valid
    return (predicted & actual).sum().item(), predicted.sum().item() + actual.sum().item()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = valid_count = intersection = foreground_count = 0
    with torch.set_grad_enabled(training):
        for batch_index, (images, masks) in enumerate(loader, start=1):
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            n_valid = (masks != IGNORE_INDEX).sum().item()
            loss_sum += loss.item() * n_valid
            valid_count += n_valid
            overlap, total = dice_counts(logits.detach(), masks)
            intersection += overlap
            foreground_count += total
            if training and batch_index % 200 == 0:
                print(f'  batch {batch_index}/{len(loader)} | Loss {loss_sum / valid_count:.4f}', flush=True)
    return loss_sum / valid_count, (2 * intersection + 1e-6) / (foreground_count + 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='显式启动训练')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--run-dir', type=Path, help='指定新的独立实验目录，不允许覆盖')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.lr <= 0:
        parser.error('batch-size、epochs、lr 必须为正数')
    if args.weight_decay < 0:
        parser.error('weight-decay 不能为负数')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_set, val_set = make_train_val(args.data_root, args.image_size)
    # Windows 下先使用单进程加载，便于调试。
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, masks = next(iter(val_loader))
    print(f'训练集：{len(train_set)}；验证集：{len(val_set)}')
    print(f'图片：{tuple(images.shape)}，{images.dtype}，范围 [{images.min():.3f}, {images.max():.3f}]')
    print(f'标签：{tuple(masks.shape)}，{masks.dtype}，类别值 {masks.unique().tolist()}')
    print('类别：0=背景，1=宠物，255=忽略；官方测试集未参与划分。')
    if not args.train:
        print('数据检查完成，未创建模型、未训练、未保存权重。')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'开始训练，设备：{device}')
    model = UNet(in_channels=3, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer_class = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[args.optimizer]
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    log = BenchmarkLog(args, train_set, val_set, device)
    checkpoint_path = log.directory / 'best.pth'
    print(f'实验目录：{log.directory}', flush=True)
    best_dice = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, device)
        print(
            f'Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} Dice {train_dice:.4f} | '
            f'Val Loss {val_loss:.4f} Dice {val_dice:.4f}', flush=True
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f'最佳验证权重已保存：{checkpoint_path}')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        log.add_epoch(dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'],
                           train_loss=train_loss, train_dice=train_dice,
                           val_loss=val_loss, val_dice=val_dice,
                           epoch_seconds=time.perf_counter()-epoch_started,
                           elapsed_seconds=time.perf_counter()-started))
    log.summary('completed')
    print('实验汇总：benchmark/results.csv；绘图：python plot_benchmark.py')


if __name__ == '__main__':
    main()
