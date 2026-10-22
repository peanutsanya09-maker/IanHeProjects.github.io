"""读取宠物原图与分割标签。"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split

IGNORE_INDEX = 255
DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "oxford-iiit-pet"


class PetDataset(Dataset):
    def __init__(self, root=DEFAULT_ROOT, split="trainval", image_size=256):
        if split not in ("trainval", "test"):
            raise ValueError("split 必须为 trainval 或 test")
        if image_size < 32 or image_size % 16:
            raise ValueError("尺寸必须至少为32且为16的倍数")
        self.root, self.image_size = Path(root), image_size
        self.ids = [line.split()[0] for line in (self.root / "annotations" / f"{split}.txt").read_text().splitlines() if line.strip()]
        if not self.ids:
            raise ValueError("数据列表为空")
        for name in self.ids:
            for path in self.paths(name):
                if not path.is_file():
                    raise FileNotFoundError(path)

    def paths(self, name):
        return self.root / "images" / f"{name}.jpg", self.root / "annotations" / "trimaps" / f"{name}.png"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path, mask_path = self.paths(self.ids[index])
        size = (self.image_size, self.image_size)
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"原图与标签尺寸不匹配：{image_path}")
            # 原图双线性插值，标签最近邻插值，避免产生新的类别值。
            pixels = np.array(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
            raw = np.array(mask.resize(size, Image.Resampling.NEAREST))
        if not np.isin(raw, [1, 2, 3]).all():
            raise ValueError(f"标签值异常：{mask_path}")
        labels = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
        labels[raw == 1] = 1  # 宠物
        labels[raw == 2] = 0  # 背景；未确定区域3保留为255
        if not (labels != IGNORE_INDEX).any():
            raise ValueError(f"没有有效标签：{mask_path}")
        return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0, torch.from_numpy(labels)


def make_train_val(root=DEFAULT_ROOT, image_size=256):
    dataset = PetDataset(root, "trainval", image_size)
    if len(dataset) < 2:
        raise ValueError("至少需要两个样本")
    n_val = max(1, int(len(dataset) * 0.2))
    # 官方 test 留作最终测试；固定种子保证训练/验证划分一致。
    return random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(0))
