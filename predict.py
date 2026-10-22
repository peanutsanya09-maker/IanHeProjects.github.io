"""在官方测试集上推理并保存对比图；不会训练或修改权重。"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from pet_dataset import DEFAULT_ROOT, IGNORE_INDEX, PetDataset
from unet import UNet

ROOT = Path(__file__).resolve().parent


def save_comparison(image, prediction, target, name, dice, path):
    # 所有面板均为模型输入的缩放尺寸，方便逐像素比较。
    rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    predicted = prediction.numpy().astype(np.uint8) * 255
    truth = np.zeros(target.shape, dtype=np.uint8)
    truth[target.numpy() == 1] = 255
    truth[target.numpy() == IGNORE_INDEX] = 128
    panels = [Image.fromarray(rgb), Image.fromarray(predicted).convert("RGB"), Image.fromarray(truth).convert("RGB")]
    size = image.shape[-1]
    canvas = Image.new("RGB", (size * 3, size + 76), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"{name} | Dice: {dice:.4f} | resized to {size}x{size}", fill="black")
    for i, (panel, title) in enumerate(zip(panels, ["Input photo", "Prediction", "Ground truth"])):
        draw.text((i * size + 8, 25), title, fill="black")
        canvas.paste(panel, (i * size, 44))
    draw.text((8, size + 52), "White: pet | Black: background | Gray in ground truth: ignored", fill="black")
    canvas.save(path)
    Image.fromarray(predicted).save(path.with_name(f"{name}_mask.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "unet_pets.pth")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "predictions")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    dataset = PetDataset(args.data_root, split="test", image_size=args.image_size)
    if not 1 <= args.count <= len(dataset):
        parser.error(f"count 必须在 1 到 {len(dataset)} 之间")
    # 在查看预测之前随机选样本，固定种子以便重复查看。
    indices = random.Random(args.seed).sample(range(len(dataset)), args.count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()  # BatchNorm 使用训练积累的统计量。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"设备：{device}；仅预测官方测试集中的 {args.count} 张样本")
    with torch.inference_mode():  # 不记录梯度，不更新任何参数。
        for index in indices:
            image, target = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1)[0].cpu()
            valid = target != IGNORE_INDEX
            pred_fg = (prediction == 1) & valid
            true_fg = (target == 1) & valid
            intersection = (pred_fg & true_fg).sum().item()
            total = pred_fg.sum().item() + true_fg.sum().item()
            dice = (2 * intersection + 1e-6) / (total + 1e-6)
            name = dataset.ids[index]
            save_comparison(image, prediction, target, name, dice, args.output_dir / f"{name}_comparison.png")
            rows.append({"image_id": name, "test_index": index, "dice": dice})
            print(f"{name}: Dice={dice:.4f}")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "test_index", "dice"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"对比图和黑白掩码已保存：{args.output_dir}")
    print("这些是抽样结果，不代表完整测试集的评估分数。输出图片均为缩放后的尺寸。")


if __name__ == "__main__":
    main()
