"""DIV2K deblurring dataset.

Returns (clean patch, blur kernel, sigma). The observation itself is produced
on the GPU during training by ``tdpm.degradations.deblur.degrade`` so that a
fresh noise realisation is drawn every step.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..degradations.deblur import KERNEL_SIZE, gaussian_kernel, sample_sigma

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(root):
    paths = []
    for directory, _, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(directory, name))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


class DIV2KDeblurDataset(Dataset):
    def __init__(self, paths, patch_size=256, sigma_low=2.5, sigma_high=10.0,
                 kernel_size=KERNEL_SIZE, repeat=1, train=True, seed=0):
        self.paths = paths
        self.patch_size = patch_size
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.kernel_size = kernel_size
        self.repeat = max(1, repeat)
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.paths) * self.repeat

    def _crop(self, image, rng):
        width, height = image.size
        size = self.patch_size
        if width < size or height < size:
            scale = size / min(width, height)
            image = image.resize((max(size, int(round(width * scale))),
                                  max(size, int(round(height * scale)))), Image.BICUBIC)
            width, height = image.size
        if self.train:
            left = rng.randint(0, width - size + 1)
            top = rng.randint(0, height - size + 1)
        else:
            left = (width - size) // 2
            top = (height - size) // 2
        return image.crop((left, top, left + size, top + size))

    def __getitem__(self, index):
        path = self.paths[index % len(self.paths)]
        rng = np.random if self.train else np.random.RandomState(self.seed + index)

        image = Image.open(path).convert("RGB")
        patch = self._crop(image, rng)
        array = np.asarray(patch, dtype=np.float32) / 255.0
        clean = torch.from_numpy(np.transpose(array, (2, 0, 1))) * 2.0 - 1.0

        sigma = sample_sigma(rng, self.sigma_low, self.sigma_high)
        kernel = torch.from_numpy(gaussian_kernel(sigma, self.kernel_size))
        return clean, kernel, torch.tensor(sigma, dtype=torch.float32)


def build_deblur_dataloaders(cfg):
    data = cfg.data
    train_root = data.get("train_root", os.path.join(data.root, "DIV2K_train_HR"))
    val_root = data.get("val_root", os.path.join(data.root, "DIV2K_valid_HR"))

    train_paths = list_images(train_root)
    val_paths = list_images(val_root)[:data.get("num_val_images", 32)]

    train_set = DIV2KDeblurDataset(train_paths, data.patch_size, data.sigma_low,
                                   data.sigma_high, repeat=data.get("repeat", 20),
                                   train=True)
    val_set = DIV2KDeblurDataset(val_paths, data.patch_size, data.sigma_low,
                                 data.sigma_high, train=False, seed=cfg.runtime.seed)

    train_loader = DataLoader(train_set, batch_size=data.batch_size, shuffle=True,
                              num_workers=cfg.runtime.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=data.get("val_batch_size", data.batch_size),
                            shuffle=False, num_workers=2, pin_memory=True)
    print(f"[data] deblurring: {len(train_paths)} train / {len(val_paths)} val images "
          f"(patch {data.patch_size}, sigma in [{data.sigma_low}, {data.sigma_high}])")
    return train_loader, val_loader
