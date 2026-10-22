"""评估固定权重，保存完整测试集逐图成绩和固定样本对比图。"""

import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from benchmark_log import write_csv
from pet_dataset import IGNORE_INDEX, PetDataset
from predict import save_comparison
from unet import UNet


def evaluate(checkpoint: Path, output_dir: Path) -> dict:
    """测试集不参与权重选择；使用所有有效像素累计全局指标。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PetDataset(split='test', image_size=256)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = UNet(3, 2)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
    selected = set(random.Random(0).sample(range(len(dataset)), 6))
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    intersection = predicted_count = actual_count = valid_count = 0
    loss_sum = 0.0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(loader):
            logits = model(images.to(device))
            targets = masks.to(device)
            loss_sum += criterion(logits, targets).item()
            predictions = logits.argmax(1).cpu()
            for j in range(len(images)):
                index = batch_index * loader.batch_size + j
                target, prediction = masks[j], predictions[j]
                valid = target != IGNORE_INDEX
                p, t = (prediction == 1) & valid, (target == 1) & valid
                overlap, npred, ntrue = (p & t).sum().item(), p.sum().item(), t.sum().item()
                dice = (2 * overlap + 1e-6) / (npred + ntrue + 1e-6)
                iou = (overlap + 1e-6) / (npred + ntrue - overlap + 1e-6)
                name = dataset.ids[index]
                rows.append(dict(image_id=name, dice=dice, iou=iou))
                intersection += overlap
                predicted_count += npred
                actual_count += ntrue
                valid_count += valid.sum().item()
                if index in selected:
                    save_comparison(images[j], prediction, target, name, dice, output_dir / f'{name}_comparison.png')
            if (batch_index + 1) % 400 == 0:
                print(f'Test images: {len(rows)}/{len(dataset)}', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result = dict(test_count=len(rows), test_loss=loss_sum / valid_count,
                  test_global_dice=(2 * intersection + 1e-6) / (predicted_count + actual_count + 1e-6),
                  test_global_iou=(intersection + 1e-6) / (predicted_count + actual_count - intersection + 1e-6),
                  test_mean_image_dice=sum(r['dice'] for r in rows)/len(rows),
                  prediction_seconds=time.perf_counter()-started)
    write_csv(output_dir / 'per_image.csv', ['image_id', 'dice', 'iou'], rows)
    write_csv(output_dir / 'summary.csv', list(result), [result])
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.output_dir)


if __name__ == '__main__':
    main()
