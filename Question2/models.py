"""
models.py
---------
Model definitions for Q2:
  - ResNet18 (CIFAR-10 image classifier, 10 classes)
  - ResNet34 (Adversarial detector, 2 classes: clean vs adversarial)

Both models are adapted from the standard ImageNet versions to work
with CIFAR-10's 32x32 images (instead of 224x224).

Changes made from default torchvision ResNet:
  1. conv1: 7x7 stride-2 → 3x3 stride-1  (preserves spatial resolution on small images)
  2. maxpool: removed (replaced with Identity) (avoids halving 32x32 too early)
  3. fc: 1000 → num_classes (10 for classifier, 2 for detector)
"""

import torch.nn as nn
import torchvision.models as models


def get_resnet18_classifier(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    ResNet18 adapted for CIFAR-10 (32x32 images).
    trained from scratch (pretrained=False as per assignment requirements).

    Args:
        num_classes: number of output classes (10 for CIFAR-10)
        pretrained:  False = train from scratch (required by assignment)

    Returns:
        nn.Module: modified ResNet18
    """
    weights = None  # always train from scratch for Q2
    model = models.resnet18(weights=weights)

    # Fix 1: replace 7x7 stride-2 conv → 3x3 stride-1
    # Reason: 7x7 with stride-2 on a 32x32 image → 16x16 immediately, losing detail
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Fix 2: remove MaxPool (would reduce 32x32 → 16x16, too aggressive)
    model.maxpool = nn.Identity()

    # Fix 3: output head → 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet34_detector(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """
    ResNet34 adapted as a binary adversarial detector for CIFAR-10 images.
    Output: 0 = clean image, 1 = adversarial image

    Same CIFAR-10 adaptations as ResNet18 above.

    Args:
        num_classes: 2 (binary: clean vs adversarial)
        pretrained:  False = train from scratch

    Returns:
        nn.Module: modified ResNet34
    """
    weights = None
    model = models.resnet34(weights=weights)

    # Same CIFAR-10 fixes as ResNet18
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # Binary output: clean (0) vs adversarial (1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    import torch

    # Quick sanity check — verify output shapes
    dummy = torch.randn(4, 3, 32, 32)  # batch of 4 CIFAR-10 images

    classifier = get_resnet18_classifier(num_classes=10)
    out = classifier(dummy)
    print(f"ResNet18 classifier output shape: {out.shape}")  # expect (4, 10)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"ResNet18 total parameters: {total_params:,}")

    detector = get_resnet34_detector(num_classes=2)
    out = detector(dummy)
    print(f"ResNet34 detector output shape:   {out.shape}")  # expect (4, 2)
    total_params = sum(p.numel() for p in detector.parameters())
    print(f"ResNet34 total parameters: {total_params:,}")
