"""
evaluate.py
-----------
Step 4: Final evaluation and WandB reporting.

This script:
  1. Loads the trained ResNet18 classifier + both ResNet34 detectors
  2. Reports clean and adversarial accuracy for all 4 attacks
  3. Plots epsilon vs accuracy drop curve
  4. Reports PGD vs BIM detection accuracy comparison
  5. Logs everything (metrics + images) to WandB

Prerequisites:
  - train_classifier.py  → weights/resnet18_best.pth
  - generate_attacks.py  → data/*.npy files
  - train_detector.py    → weights/detector_pgd_best.pth, detector_bim_best.pth

Usage:
    python evaluate.py
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from models import get_resnet18_classifier, get_resnet34_detector
from utils  import (
    compute_accuracy_numpy,
    load_checkpoint,
    build_detector_dataset,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


# ─── Argument Parser ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Final evaluation and WandB reporting")
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--weights_dir", type=str, default="./weights")
    parser.add_argument("--output_dir",  type=str, default="./data")
    parser.add_argument("--epsilon",     type=float, default=0.03)
    parser.add_argument("--wandb_project", type=str, default="dlops-q2-adversarial")
    parser.add_argument("--wandb_run",   type=str,   default="final-evaluation")
    parser.add_argument("--no_wandb",    action="store_true")
    return parser.parse_args()


# ─── Epsilon Sweep ────────────────────────────────────────────────────────────
def epsilon_sweep(model, x_test, y_test, device, art_classifier, epsilons):
    """
    Runs FGSM attack at multiple epsilon values and records accuracy.
    Used to plot: epsilon vs accuracy drop.

    Returns:
        list of (epsilon, clean_acc, adv_acc) tuples
    """
    clean_acc = compute_accuracy_numpy(model, x_test, y_test, device)
    results = []

    for eps in epsilons:
        attack = FastGradientMethod(estimator=art_classifier, eps=eps, batch_size=256)
        x_adv  = attack.generate(x=x_test).astype(np.float32)
        adv_acc = compute_accuracy_numpy(model, x_adv, y_test, device)
        results.append((eps, clean_acc, adv_acc))
        print(f"  eps={eps:.3f} | Clean: {clean_acc:.2f}% | Adv: {adv_acc:.2f}% | Drop: {clean_acc-adv_acc:.2f}%")

    return results


def plot_epsilon_sweep(sweep_results, save_path):
    """Plots epsilon vs accuracy and epsilon vs accuracy drop."""
    epsilons  = [r[0] for r in sweep_results]
    clean_acc = [r[1] for r in sweep_results]
    adv_acc   = [r[2] for r in sweep_results]
    drops     = [r[1] - r[2] for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epsilons, clean_acc, "b-o", label="Clean Accuracy")
    ax1.plot(epsilons, adv_acc,   "r-o", label="Adversarial Accuracy")
    ax1.set_xlabel("Epsilon (perturbation strength)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs Epsilon (FGSM ART)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epsilons, drops, "g-o", label="Accuracy Drop")
    ax2.set_xlabel("Epsilon (perturbation strength)")
    ax2.set_ylabel("Accuracy Drop (%)")
    ax2.set_title("Accuracy Drop vs Epsilon (FGSM ART)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved epsilon sweep plot → {save_path}")


# ─── Detector Evaluation ──────────────────────────────────────────────────────
def evaluate_detector_on_test(detector, clean_images, adv_images, device, batch_size=256):
    """
    Evaluates a trained detector on a balanced test set.
    Returns overall accuracy, clean detection rate, adversarial detection rate.
    """
    detector.eval()

    all_images = np.concatenate([clean_images, adv_images], axis=0)
    all_labels = np.array([0] * len(clean_images) + [1] * len(adv_images), dtype=np.int64)

    correct = 0
    class_correct = [0, 0]
    class_total   = [0, 0]

    for i in range(0, len(all_images), batch_size):
        batch_x = torch.tensor(all_images[i:i+batch_size]).to(device)
        batch_y = torch.tensor(all_labels[i:i+batch_size]).to(device)

        with torch.no_grad():
            outputs = detector(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            for cls in [0, 1]:
                mask = batch_y == cls
                class_correct[cls] += predicted[mask].eq(batch_y[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    total_acc  = 100.0 * correct / len(all_images)
    clean_acc  = 100.0 * class_correct[0] / max(class_total[0], 1)
    adv_acc    = 100.0 * class_correct[1] / max(class_total[1], 1)
    return total_acc, clean_acc, adv_acc


def plot_detector_comparison(pgd_results, bim_results, save_path):
    """Bar chart comparing PGD vs BIM detector performance."""
    labels    = ["Overall", "Clean Detection", "Adv Detection"]
    pgd_vals  = list(pgd_results)
    bim_vals  = list(bim_results)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, pgd_vals, width, label="PGD Detector", color="steelblue")
    ax.bar(x + width/2, bim_vals, width, label="BIM Detector", color="tomato")

    ax.set_ylabel("Detection Accuracy (%)")
    ax.set_title("Adversarial Detector Comparison: PGD vs BIM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.axhline(y=70, color="gray", linestyle="--", label="70% threshold")

    for i, (pv, bv) in enumerate(zip(pgd_vals, bim_vals)):
        ax.text(i - width/2, pv + 1, f"{pv:.1f}%", ha="center", fontsize=9)
        ax.text(i + width/2, bv + 1, f"{bv:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved detector comparison plot → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={"epsilon": args.epsilon},
        )

    # ── Load adversarial data ──
    print("\nLoading saved adversarial arrays...")
    x_clean = np.load(os.path.join(args.data_dir, "x_test_clean.npy"))
    y_test  = np.load(os.path.join(args.data_dir, "y_test.npy"))
    x_adv_fgsm_scratch = np.load(os.path.join(args.data_dir, "x_adv_fgsm_scratch.npy"))
    x_adv_fgsm_art     = np.load(os.path.join(args.data_dir, "x_adv_fgsm_art.npy"))
    x_adv_pgd          = np.load(os.path.join(args.data_dir, "x_adv_pgd.npy"))
    x_adv_bim          = np.load(os.path.join(args.data_dir, "x_adv_bim.npy"))

    # ── Load ResNet18 classifier ──
    print("\nLoading ResNet18 classifier...")
    classifier = get_resnet18_classifier(num_classes=10).to(device)
    load_checkpoint(classifier, os.path.join(args.weights_dir, "resnet18_best.pth"), device)
    classifier.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # PART (i) — Attack Accuracy Report
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PART (i): ATTACK EVALUATION")
    print("="*70)

    clean_acc            = compute_accuracy_numpy(classifier, x_clean,             y_test, device)
    adv_acc_fgsm_scratch = compute_accuracy_numpy(classifier, x_adv_fgsm_scratch,  y_test, device)
    adv_acc_fgsm_art     = compute_accuracy_numpy(classifier, x_adv_fgsm_art,      y_test, device)
    adv_acc_pgd          = compute_accuracy_numpy(classifier, x_adv_pgd,           y_test, device)
    adv_acc_bim          = compute_accuracy_numpy(classifier, x_adv_bim,           y_test, device)

    print(f"\n{'Attack':<22} | {'Clean Acc':>9} | {'Adv Acc':>8} | {'Drop':>7}")
    print("-"*60)
    for name, adv_acc in [
        ("No Attack",         clean_acc),
        ("FGSM (Scratch)",    adv_acc_fgsm_scratch),
        ("FGSM (ART)",        adv_acc_fgsm_art),
        ("PGD  (ART)",        adv_acc_pgd),
        ("BIM  (ART)",        adv_acc_bim),
    ]:
        drop = clean_acc - adv_acc if name != "No Attack" else 0.0
        print(f"{name:<22} | {clean_acc:>8.2f}% | {adv_acc:>7.2f}% | {drop:>6.2f}%")

    # ── Epsilon sweep (FGSM ART) ──
    print("\nEpsilon sweep (FGSM ART)...")
    criterion = nn.CrossEntropyLoss()
    optimizer_art = optim.SGD(classifier.parameters(), lr=0.01)
    art_clf = PyTorchClassifier(
        model=classifier,
        loss=criterion,
        optimizer=optimizer_art,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(np.array(CIFAR10_MEAN, dtype=np.float32), np.array(CIFAR10_STD, dtype=np.float32)),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    epsilons = [0.01, 0.02, 0.03, 0.05, 0.1]
    sweep_results = epsilon_sweep(classifier, x_clean, y_test, device, art_clf, epsilons)
    plot_epsilon_sweep(sweep_results, os.path.join(args.output_dir, "epsilon_sweep.png"))

    # ─────────────────────────────────────────────────────────────────────────
    # PART (ii) — Detector Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PART (ii): DETECTOR EVALUATION")
    print("="*70)

    pgd_detector_path = os.path.join(args.weights_dir, "detector_pgd_best.pth")
    bim_detector_path = os.path.join(args.weights_dir, "detector_bim_best.pth")

    print("\nLoading PGD detector...")
    pgd_detector = get_resnet34_detector(num_classes=2).to(device)
    load_checkpoint(pgd_detector, pgd_detector_path, device)

    print("Loading BIM detector...")
    bim_detector = get_resnet34_detector(num_classes=2).to(device)
    load_checkpoint(bim_detector, bim_detector_path, device)

    # Evaluate on test portion (use last 2000 samples for final eval)
    eval_clean = x_clean[-2000:]
    eval_pgd   = x_adv_pgd[-2000:]
    eval_bim   = x_adv_bim[-2000:]

    pgd_total, pgd_clean_det, pgd_adv_det = evaluate_detector_on_test(pgd_detector, eval_clean, eval_pgd, device)
    bim_total, bim_clean_det, bim_adv_det = evaluate_detector_on_test(bim_detector, eval_clean, eval_bim, device)

    print(f"\n{'Detector':<15} | {'Overall':>7} | {'Clean Det':>9} | {'Adv Det':>7} | {'Status':>6}")
    print("-"*60)
    print(f"{'PGD Detector':<15} | {pgd_total:>6.2f}% | {pgd_clean_det:>8.2f}% | {pgd_adv_det:>6.2f}% | {'PASS' if pgd_total >= 70 else 'FAIL':>6}")
    print(f"{'BIM Detector':<15} | {bim_total:>6.2f}% | {bim_clean_det:>8.2f}% | {bim_adv_det:>6.2f}% | {'PASS' if bim_total >= 70 else 'FAIL':>6}")

    plot_detector_comparison(
        (pgd_total, pgd_clean_det, pgd_adv_det),
        (bim_total, bim_clean_det, bim_adv_det),
        os.path.join(args.output_dir, "detector_comparison.png"),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # WandB Final Logging
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        # Scalar metrics
        wandb.log({
            "final/clean_acc":              clean_acc,
            "final/adv_fgsm_scratch":       adv_acc_fgsm_scratch,
            "final/adv_fgsm_art":           adv_acc_fgsm_art,
            "final/adv_pgd":                adv_acc_pgd,
            "final/adv_bim":                adv_acc_bim,
            "final/pgd_detector_acc":       pgd_total,
            "final/bim_detector_acc":       bim_total,
        })

        # Epsilon sweep table
        sweep_table = wandb.Table(
            columns=["epsilon", "clean_acc", "adv_acc", "drop"],
            data=[[r[0], r[1], r[2], r[1]-r[2]] for r in sweep_results],
        )
        wandb.log({"epsilon_sweep_table": sweep_table})

        # Plot images
        wandb.log({
            "plots/epsilon_sweep":       wandb.Image(os.path.join(args.output_dir, "epsilon_sweep.png")),
            "plots/detector_comparison": wandb.Image(os.path.join(args.output_dir, "detector_comparison.png")),
        })

        wandb.finish()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
