"""Paired super-resolution dataset built from a folder of images.

Each item is (high resolution image, bicubic-upsampled low resolution image),
both normalised to [-1, 1] and of size ``image_size``.
"""

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_images(root):
    paths = []
    for directory, _, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(directory, name))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


class SuperResolutionDataset(Dataset):
    def __init__(self, paths, image_size=256, scale_factor=4):
        self.paths = paths
        self.image_size = image_size
        self.lr_size = image_size // scale_factor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        hr = transforms.Resize((self.image_size, self.image_size),
                               Image.BICUBIC)(image)
        hr = transforms.CenterCrop((self.image_size, self.image_size))(hr)
        lr = transforms.Resize((self.lr_size, self.lr_size), Image.BICUBIC)(hr)
        lr_up = transforms.Resize((self.image_size, self.image_size), Image.BICUBIC)(lr)
        return self.to_tensor(hr), self.to_tensor(lr_up)


def build_sr_dataloaders(cfg):
    data = cfg.data
    paths = list_images(data.root)
    dataset = SuperResolutionDataset(paths, data.image_size, data.scale_factor)

    val_size = max(1, int(len(dataset) * data.get("val_ratio", 0.02)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.runtime.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=data.batch_size, shuffle=True,
                              num_workers=cfg.runtime.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=data.get("val_batch_size", data.batch_size),
                            shuffle=False, num_workers=2, pin_memory=True)
    print(f"[data] super-resolution: {train_size} train / {val_size} val images")
    return train_loader, val_loader
