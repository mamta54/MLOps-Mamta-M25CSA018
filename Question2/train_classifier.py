"""
train_classifier.py
-------------------
Step 1: Train ResNet18 on CIFAR-10 from scratch.

Target: ≥72% test accuracy (assignment requirement).
With 50 epochs + CosineAnnealingLR this typically reaches 85-90%.

Usage (on Kaggle or Docker):
    python train_classifier.py

Outputs:
    weights/resnet18_best.pth    ← best checkpoint (by val accuracy)
    weights/resnet18_final.pth   ← final epoch checkpoint
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from models import get_resnet18_classifier
from utils  import get_cifar10_loaders, compute_accuracy, save_checkpoint


# ─── Argument Parser ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10")
    parser.add_argument("--epochs",     type=int,   default=50,    help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=128,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=0.1,   help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--data_dir",   type=str,   default="./data",    help="CIFAR-10 data directory")
    parser.add_argument("--weights_dir",type=str,   default="./weights", help="Directory to save model weights")
    parser.add_argument("--wandb_project", type=str, default="dlops-q2-adversarial")
    parser.add_argument("--wandb_run",  type=str,   default="resnet18-cifar10-classifier")
    parser.add_argument("--no_wandb",   action="store_true", help="Disable WandB logging")
    return parser.parse_args()


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full pass over the training data. Returns (avg_loss, accuracy)."""
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

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Runs evaluation on validation/test set. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += images.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── WandB init ──
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "architecture":  "ResNet18",
                "dataset":       "CIFAR-10",
                "task":          "classification",
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "optimizer":     "SGD+Nesterov",
                "scheduler":     "CosineAnnealingLR",
            },
        )

    # ── Data ──
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"Dataset: {len(train_loader.dataset)} train | {len(test_loader.dataset)} test")

    # ── Model ──
    model = get_resnet18_classifier(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet18 | Parameters: {total_params:,}")

    # ── Loss + Optimizer + Scheduler ──
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # CosineAnnealingLR: smoothly decays lr from args.lr → 0 over T_max epochs
    # Better than step decay for shorter training runs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    best_val_acc = 0.0
    os.makedirs(args.weights_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}% | {current_lr:>8.6f}")

        # ── WandB logging ──
        if not args.no_wandb:
            wandb.log({
                "epoch":          epoch,
                "train/loss":     train_loss,
                "train/accuracy": train_acc,
                "val/loss":       val_loss,
                "val/accuracy":   val_acc,
                "lr":             current_lr,
            }, step=epoch)

        # ── Save best model ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                path=os.path.join(args.weights_dir, "resnet18_best.pth"),
                epoch=epoch,
                val_acc=val_acc,
            )

    # Save final epoch weights too
    save_checkpoint(
        model,
        path=os.path.join(args.weights_dir, "resnet18_final.pth"),
        epoch=args.epochs,
        val_acc=val_acc,
    )

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

    if not args.no_wandb:
        wandb.run.summary["best_val_accuracy"] = best_val_acc
        wandb.finish()

    if best_val_acc < 72.0:
        print("WARNING: Best accuracy < 72%. Try increasing epochs or adjusting lr.")
    else:
        print("Assignment requirement met: accuracy >= 72%")


if __name__ == "__main__":
    main()
