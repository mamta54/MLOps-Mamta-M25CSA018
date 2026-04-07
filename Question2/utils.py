"""
utils.py
--------
Shared helpers used across all scripts:
  - CIFAR-10 data loaders (with proper augmentation)
  - Accuracy computation
  - Tensor ↔ Numpy conversion (needed because IBM ART works with numpy)
  - Denormalization (needed before passing images to ART attacks)
  - Saving / loading model checkpoints
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms


# ─── CIFAR-10 Statistics ──────────────────────────────────────────────────────
# Pre-computed mean and std per channel on the CIFAR-10 training set.
# Used to normalize images so all pixel values are on a similar scale.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
):
    """
    Returns train and test DataLoaders for CIFAR-10.

    Train transforms include augmentation (random crop + flip) to prevent overfitting.
    Test transforms only normalize — no augmentation (we want deterministic eval).

    Args:
        data_dir:    where to download/cache the dataset
        batch_size:  images per batch
        num_workers: parallel workers for data loading (2 works well on Kaggle)

    Returns:
        (train_loader, test_loader)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # random crop with 4px padding
        transforms.RandomHorizontalFlip(),           # 50% chance of horizontal mirror
        transforms.ToTensor(),                       # PIL Image → float tensor [0,1]
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar10_test_images(
    data_dir: str = "./data",
    n_samples: int = None,
):
    """
    Returns the raw CIFAR-10 test images as numpy arrays (no normalization).

    Why raw / unnormalized?
    IBM ART attacks require images in [0, 1] range — NOT normalized with mean/std.
    ART handles the normalization internally via its preprocessing pipeline.

    Args:
        data_dir:  dataset cache directory
        n_samples: if set, returns only first n_samples images (for quick testing)

    Returns:
        x_test: np.float32 array of shape (N, 3, 32, 32), values in [0, 1]
        y_test: np.int64  array of shape (N,)
    """
    raw_transform = transforms.Compose([
        transforms.ToTensor(),  # → float32 [0, 1], shape (3, 32, 32)
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=raw_transform
    )

    if n_samples is not None:
        indices = list(range(n_samples))
        test_set = torch.utils.data.Subset(test_set, indices)

    loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    x_test, y_test = next(iter(loader))

    return x_test.numpy().astype(np.float32), y_test.numpy()


# ─── Accuracy ─────────────────────────────────────────────────────────────────

def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Computes classification accuracy of model on a DataLoader.

    Returns:
        accuracy as a float between 0 and 100
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def compute_accuracy_numpy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 256) -> float:
    """
    Computes accuracy from raw numpy arrays (used after attack generation).

    Args:
        x: float32 numpy array (N, C, H, W), values in [0, 1] — unnormalized
        y: int numpy array (N,)

    Returns:
        accuracy as a float between 0 and 100
    """
    model.eval()
    correct, total = 0, 0
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(device)

    for i in range(0, len(x), batch_size):
        batch_x = torch.tensor(x[i:i+batch_size]).to(device)
        batch_y = torch.tensor(y[i:i+batch_size]).to(device)

        # Normalize before passing to model
        batch_x = (batch_x - mean) / std

        with torch.no_grad():
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    return 100.0 * correct / total


# ─── Tensor ↔ Numpy Conversion ────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses CIFAR-10 normalization: goes from normalized tensor back to [0, 1].

    Why needed: when we want to visualize images or pass them to ART,
    we need them in [0,1] range, not normalized.

    Args:
        tensor: normalized tensor of shape (N, 3, H, W) or (3, H, W)

    Returns:
        denormalized tensor clipped to [0, 1]
    """
    mean = torch.tensor(CIFAR10_MEAN, device=tensor.device)
    std  = torch.tensor(CIFAR10_STD,  device=tensor.device)

    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std  = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std  = std.view(3, 1, 1)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to numpy float32 array.
    Handles detach + CPU move automatically.
    """
    return tensor.detach().cpu().numpy().astype(np.float32)


# ─── Detector Dataset Builder ─────────────────────────────────────────────────

def build_detector_dataset(
    clean_images: np.ndarray,
    adv_images: np.ndarray,
    val_split: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 2,
):
    """
    Builds a binary classification dataset: clean (0) vs adversarial (1).

    Args:
        clean_images: np.float32 (N, 3, 32, 32), values in [0, 1]
        adv_images:   np.float32 (N, 3, 32, 32), values in [0, 1]
        val_split:    fraction of data for validation
        batch_size:   batch size for returned loaders
        num_workers:  parallel workers

    Returns:
        (train_loader, val_loader)
    """
    assert len(clean_images) == len(adv_images), \
        "clean and adversarial arrays must have the same number of samples"

    labels_clean = torch.zeros(len(clean_images), dtype=torch.long)  # 0 = clean
    labels_adv   = torch.ones(len(adv_images),   dtype=torch.long)   # 1 = adversarial

    all_images = torch.cat([
        torch.tensor(clean_images),
        torch.tensor(adv_images),
    ], dim=0)
    all_labels = torch.cat([labels_clean, labels_adv], dim=0)

    # Shuffle
    perm = torch.randperm(len(all_images))
    all_images = all_images[perm]
    all_labels = all_labels[perm]

    dataset = TensorDataset(all_images, all_labels)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Detector dataset — Train: {train_size} | Val: {val_size}")
    print(f"  Clean samples: {len(clean_images)} | Adversarial samples: {len(adv_images)}")

    return train_loader, val_loader


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, epoch: int, val_acc: float):
    """
    Saves model weights + metadata to disk.

    Args:
        model:   the nn.Module to save
        path:    full file path (e.g. 'weights/resnet18_clean.pth')
        epoch:   current epoch number (saved for reference)
        val_acc: current validation accuracy (saved for reference)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "val_acc":      val_acc,
        "model_state":  model.state_dict(),
    }, path)
    print(f"  Saved checkpoint → {path}  (epoch={epoch}, val_acc={val_acc:.2f}%)")


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """
    Loads model weights from a checkpoint file.

    Args:
        model:  the nn.Module to load weights into
        path:   checkpoint file path
        device: torch device to map weights to

    Returns:
        checkpoint dict (contains epoch, val_acc, etc.)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Loaded checkpoint ← {path}  (epoch={checkpoint['epoch']}, val_acc={checkpoint['val_acc']:.2f}%)")
    return checkpoint
