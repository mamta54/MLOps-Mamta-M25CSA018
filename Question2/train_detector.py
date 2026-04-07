"""
train_detector.py
-----------------
Step 3: Train ResNet34 as a binary adversarial detector.

This trains TWO detectors:
  - Detector A: detects PGD adversarial images  (trained on clean + PGD)
  - Detector B: detects BIM adversarial images  (trained on clean + BIM)

The detector is a binary classifier:
  - Output 0 = clean image
  - Output 1 = adversarial image

Think of it as a WAF (Web Application Firewall) that scans inputs and
flags suspicious ones before they reach the main model.

Prerequisites:
  Run generate_attacks.py first to produce the .npy files.

Usage:
    python train_detector.py

Outputs:
    weights/detector_pgd_best.pth
    weights/detector_bim_best.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from models import get_resnet34_detector
from utils  import build_detector_dataset, save_checkpoint


# ─── Argument Parser ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet34 adversarial detector")
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--weights_dir", type=str,   default="./weights")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--wandb_project", type=str, default="dlops-q2-adversarial")
    parser.add_argument("--no_wandb",    action="store_true")
    return parser.parse_args()


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += images.size(0)

    return running_loss / total, 100.0 * correct / total


def evaluate_detector(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    # Track per-class (clean vs adv) accuracy
    class_correct = [0, 0]
    class_total   = [0, 0]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)

            # Per-class breakdown
            for cls in [0, 1]:
                mask = labels == cls
                class_correct[cls] += predicted[mask].eq(labels[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    clean_acc = 100.0 * class_correct[0] / max(class_total[0], 1)
    adv_acc   = 100.0 * class_correct[1] / max(class_total[1], 1)

    return avg_loss, accuracy, clean_acc, adv_acc


# ─── Train One Detector ───────────────────────────────────────────────────────
def train_detector(
    clean_images: np.ndarray,
    adv_images: np.ndarray,
    attack_name: str,
    args,
    device: torch.device,
):
    """
    Trains a ResNet34 binary detector for one attack type.

    Args:
        clean_images: float32 numpy (N, 3, 32, 32), values in [0, 1]
        adv_images:   float32 numpy (N, 3, 32, 32), values in [0, 1]
        attack_name:  "PGD" or "BIM" (used for logging and saving)
        args:         parsed CLI arguments
        device:       torch device

    Returns:
        best_val_acc: best validation accuracy achieved
    """
    print(f"\n{'='*60}")
    print(f"Training Detector for {attack_name} attack")
    print(f"{'='*60}")

    # ── WandB run ──
    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=f"detector-{attack_name.lower()}",
            config={
                "architecture":  "ResNet34",
                "task":          f"adversarial-detection-{attack_name}",
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "n_clean":       len(clean_images),
                "n_adversarial": len(adv_images),
                "attack_type":   attack_name,
            },
            reinit=True,
        )

    # ── Build dataset ──
    train_loader, val_loader = build_detector_dataset(
        clean_images=clean_images,
        adv_images=adv_images,
        val_split=0.2,
        batch_size=args.batch_size,
    )

    # ── Model ──
    model = get_resnet34_detector(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet34 | Parameters: {total_params:,}")

    # ── Loss + Optimizer + Scheduler ──
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Halve learning rate every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training ──
    best_val_acc = 0.0
    best_weights_path = os.path.join(args.weights_dir, f"detector_{attack_name.lower()}_best.pth")

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Clean Det':>9} | {'Adv Det':>7}")
    print("-"*80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, clean_det_acc, adv_det_acc = evaluate_detector(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
            f"{val_loss:>8.4f} | {val_acc:>6.2f}% | "
            f"{clean_det_acc:>8.2f}% | {adv_det_acc:>6.2f}%"
        )

        if not args.no_wandb:
            wandb.log({
                "epoch":                        epoch,
                f"{attack_name}/train/loss":    train_loss,
                f"{attack_name}/train/acc":     train_acc,
                f"{attack_name}/val/loss":      val_loss,
                f"{attack_name}/val/acc":       val_acc,
                f"{attack_name}/val/clean_det": clean_det_acc,
                f"{attack_name}/val/adv_det":   adv_det_acc,
                f"{attack_name}/lr":            current_lr,
            }, step=epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_weights_path, epoch, val_acc)

    print(f"\n{attack_name} detector — Best val accuracy: {best_val_acc:.2f}%")

    if best_val_acc < 70.0:
        print(f"WARNING: Detection accuracy < 70% for {attack_name}. Try more epochs or larger dataset.")
    else:
        print(f"Assignment requirement met: {attack_name} detection accuracy >= 70%")

    if run is not None:
        wandb.run.summary[f"best_val_acc_{attack_name}"] = best_val_acc
        wandb.finish()

    return best_val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.weights_dir, exist_ok=True)

    # ── Load pre-generated adversarial arrays ──
    print("\nLoading adversarial image arrays...")

    clean_path = os.path.join(args.data_dir, "x_test_clean.npy")
    pgd_path   = os.path.join(args.data_dir, "x_adv_pgd.npy")
    bim_path   = os.path.join(args.data_dir, "x_adv_bim.npy")

    for path in [clean_path, pgd_path, bim_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run generate_attacks.py first to create adversarial image arrays."
            )

    x_clean = np.load(clean_path)
    x_pgd   = np.load(pgd_path)
    x_bim   = np.load(bim_path)

    print(f"  Clean images: {x_clean.shape}")
    print(f"  PGD images:   {x_pgd.shape}")
    print(f"  BIM images:   {x_bim.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # Train Detector A: PGD
    # ─────────────────────────────────────────────────────────────────────────
    best_pgd = train_detector(
        clean_images=x_clean,
        adv_images=x_pgd,
        attack_name="PGD",
        args=args,
        device=device,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Train Detector B: BIM
    # ─────────────────────────────────────────────────────────────────────────
    best_bim = train_detector(
        clean_images=x_clean,
        adv_images=x_bim,
        attack_name="BIM",
        args=args,
        device=device,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("DETECTOR TRAINING SUMMARY")
    print("="*50)
    print(f"  PGD Detector — Best Val Acc: {best_pgd:.2f}%  {'PASS' if best_pgd >= 70 else 'FAIL (<70%)'}")
    print(f"  BIM Detector — Best Val Acc: {best_bim:.2f}%  {'PASS' if best_bim >= 70 else 'FAIL (<70%)'}")
    print("="*50)


if __name__ == "__main__":
    main()
