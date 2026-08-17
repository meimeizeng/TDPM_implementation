"""FFHQ-256 box inpainting dataset.

Images are read from local parquet shards (the on-disk layout produced by
``huggingface-cli download merkol/ffhq-256 --local-dir <dir>``) so that
training runs without network access; if no shards are found the dataset falls
back to the Hub.

Each item is (masked image, mask, original image) with the mask equal to 1
inside the missing box.
"""

import glob
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _load_raw(root, split="train"):
    from datasets import load_dataset

    parquet_dir = os.path.join(root, "data")
    shards = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if shards:
        print(f"[data] loading {len(shards)} local parquet shard(s) from {parquet_dir}")
        return load_dataset("parquet", data_files=shards, split="train")
    print(f"[data] no shards under {parquet_dir}; falling back to the HuggingFace Hub")
    return load_dataset("merkol/ffhq-256", split=split)


class FFHQInpaintDataset(Dataset):
    def __init__(self, hf_dataset, image_size=256, mask_min_size=32, mask_max_size=128,
                 fixed_masks=False, seed=0):
        self.dataset = hf_dataset
        self.image_size = image_size
        self.mask_min_size = mask_min_size
        self.mask_max_size = mask_max_size
        self.fixed_masks = fixed_masks
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.dataset)

    def _random_box(self, height, width, index):
        rng = random.Random(self.seed + index) if self.fixed_masks else random
        box_h = rng.randint(self.mask_min_size, self.mask_max_size)
        box_w = rng.randint(self.mask_min_size, self.mask_max_size)
        top = rng.randint(0, height - box_h)
        left = rng.randint(0, width - box_w)
        mask = torch.zeros(1, height, width)
        mask[:, top:top + box_h, left:left + box_w] = 1.0
        return mask

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        original = self.transform(image)
        mask = self._random_box(original.shape[1], original.shape[2], index)
        masked = original * (1.0 - mask)
        return masked, mask, original


def build_inpaint_dataloaders(cfg):
    data = cfg.data
    raw = _load_raw(data.root)
    split = raw.train_test_split(test_size=data.get("val_ratio", 0.02), seed=cfg.runtime.seed)

    train_set = FFHQInpaintDataset(split["train"], data.image_size,
                                   data.mask_min_size, data.mask_max_size)
    val_set = FFHQInpaintDataset(split["test"], data.image_size,
                                 data.mask_min_size, data.mask_max_size,
                                 fixed_masks=True, seed=cfg.runtime.seed)

    train_loader = DataLoader(train_set, batch_size=data.batch_size, shuffle=True,
                              num_workers=cfg.runtime.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=data.get("val_batch_size", data.batch_size),
                            shuffle=False, num_workers=2, pin_memory=True)
    print(f"[data] inpainting: {len(train_set)} train / {len(val_set)} val images "
          f"(box size {data.mask_min_size}-{data.mask_max_size})")
    return train_loader, val_loader
