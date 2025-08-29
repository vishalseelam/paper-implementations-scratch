from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int, grayscale: bool) -> transforms.Compose:
    tfms = [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if grayscale:
        tfms.insert(1, transforms.Grayscale(num_output_channels=1))
    tfms.append(transforms.Normalize(mean=[0.5] * (1 if grayscale else 3), std=[0.5] * (1 if grayscale else 3)))
    return transforms.Compose(tfms)


def get_dataset(name: str, data_dir: str, image_size: int):
    name = name.lower()
    if name == "cifar10":
        train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(image_size, grayscale=False))
        return train, 3
    if name == "mnist":
        # Keep as 1 channel
        train = datasets.MNIST(root=data_dir, train=True, download=True, transform=get_transforms(image_size, grayscale=True))
        return train, 1
    raise ValueError(f"Unsupported dataset: {name}")


def get_dataloader(dataset, batch_size: int, num_workers: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

