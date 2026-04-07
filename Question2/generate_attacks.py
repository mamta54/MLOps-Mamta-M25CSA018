"""
generate_attacks.py
-------------------
Step 2: Generate adversarial images using 4 methods:
  A) FGSM from scratch (manual PyTorch implementation)
  B) FGSM via IBM ART
  C) PGD  via IBM ART
  D) BIM  via IBM ART

All adversarial images are saved as .npy files so they can be:
  - Loaded directly by train_detector.py (no need to regenerate)
  - Evaluated and visualized in evaluate.py

ART IMPORTANT NOTES:
  - ART expects numpy float32 arrays, NOT PyTorch tensors
  - Images must be in [0, 1] range (NOT normalized with CIFAR-10 mean/std)
  - ART handles normalization internally via preprocessing_defences argument
    OR we pass raw [0,1] images and handle normalization inside the model wrapper

Usage:
    python generate_attacks.py --weights ./weights/resnet18_best.pth

Outputs:
    data/x_test_clean.npy       ← clean test images [0,1]
    data/y_test.npy             ← true labels
    data/x_adv_fgsm_scratch.npy
    data/x_adv_fgsm_art.npy
    data/x_adv_pgd.npy
    data/x_adv_bim.npy
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed on Kaggle/Docker)
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    BasicIterativeMethod,
)

from models import get_resnet18_classifier
from utils  import (
    get_cifar10_test_images,
    compute_accuracy_numpy,
    load_checkpoint,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


# ─── Argument Parser ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial attacks")
    parser.add_argument("--weights",     type=str,   default="./weights/resnet18_best.pth")
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--output_dir",  type=str,   default="./data")
    parser.add_argument("--n_samples",   type=int,   default=10000, help="Number of test samples to attack")
    parser.add_argument("--epsilon",     type=float, default=0.03,  help="Attack perturbation budget")
    parser.add_argument("--pgd_steps",   type=int,   default=40,    help="PGD/BIM iterations")
    parser.add_argument("--wandb_project", type=str, default="dlops-q2-adversarial")
    parser.add_argument("--wandb_run",   type=str,   default="attack-generation")
    parser.add_argument("--no_wandb",    action="store_true")
    return parser.parse_args()


# ─── Model Wrapper for ART ────────────────────────────────────────────────────
def build_art_classifier(model: nn.Module, device: torch.device, args) -> PyTorchClassifier:
    """
    Wraps a PyTorch model in ART's PyTorchClassifier.

    ART needs to know:
    - the model
    - the loss function
    - input shape and number of classes
    - pixel value range (clip_values)
    - preprocessing: mean/std normalization (so ART applies it internally)

    By setting preprocessing here, we can pass raw [0,1] images to ART
    and it will normalize them before feeding to the model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # optimizer needed by ART wrapper

    # Preprocessing: ART will normalize inputs before passing to model
    # Format: ([mean_R, mean_G, mean_B], [std_R, std_G, std_B])
    preprocessing = (
        np.array(CIFAR10_MEAN, dtype=np.float32),
        np.array(CIFAR10_STD,  dtype=np.float32),
    )

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=preprocessing,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier


# ─── FGSM From Scratch ────────────────────────────────────────────────────────
def fgsm_scratch(
    model: nn.Module,
    images_np: np.ndarray,
    labels_np: np.ndarray,
    epsilon: float,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Implements FGSM attack manually in PyTorch.

    Formula: x_adv = clip( x + epsilon * sign( ∇_x Loss(x, y) ), 0, 1 )

    Steps:
      1. Enable gradient tracking on input images
      2. Forward pass → compute loss
      3. Backward pass → compute ∂Loss/∂x
      4. Step in sign direction, scale by epsilon
      5. Clip to [0, 1]

    NOTE: Images should be in [0,1] range. We normalize inside before model forward.

    Args:
        model:      trained classifier
        images_np:  float32 numpy (N, 3, 32, 32), values in [0, 1]
        labels_np:  int numpy (N,)
        epsilon:    perturbation budget
        device:     torch device

    Returns:
        adversarial images as float32 numpy (N, 3, 32, 32), values in [0, 1]
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=device).view(1, 3, 1, 1)

    adv_list = []

    for i in range(0, len(images_np), batch_size):
        # Load batch
        batch_x = torch.tensor(images_np[i:i+batch_size], device=device)
        batch_y = torch.tensor(labels_np[i:i+batch_size], device=device)

        # Enable gradient tracking on input (not on model weights)
        batch_x.requires_grad = True

        # Normalize for model
        batch_x_norm = (batch_x - mean) / std

        # Forward pass
        outputs = model(batch_x_norm)
        loss    = criterion(outputs, batch_y)

        # Backward pass — computes ∂Loss/∂x
        model.zero_grad()
        loss.backward()

        # FGSM step: move in the direction that maximizes loss
        data_grad   = batch_x.grad.data         # ∂Loss/∂x, shape (B, 3, 32, 32)
        sign_grad   = data_grad.sign()           # only keep direction, discard magnitude
        adv_batch   = batch_x + epsilon * sign_grad   # single step
        adv_batch   = torch.clamp(adv_batch, 0.0, 1.0)  # stay in valid pixel range

        adv_list.append(adv_batch.detach().cpu().numpy())

    return np.concatenate(adv_list, axis=0).astype(np.float32)


# ─── Accuracy Report ──────────────────────────────────────────────────────────
def report_accuracy(model, x_clean, x_adv, y, device, attack_name):
    """Prints and returns clean vs adversarial accuracy."""
    clean_acc = compute_accuracy_numpy(model, x_clean, y, device)
    adv_acc   = compute_accuracy_numpy(model, x_adv,   y, device)
    drop      = clean_acc - adv_acc
    print(f"  {attack_name:<20} | Clean: {clean_acc:.2f}% | Adversarial: {adv_acc:.2f}% | Drop: {drop:.2f}%")
    return clean_acc, adv_acc


# ─── Visualization ────────────────────────────────────────────────────────────
def save_comparison_plot(x_clean, x_adv_fgsm_scratch, x_adv_fgsm_art, y, n=10, save_path="data/fgsm_comparison.png"):
    """
    Saves a side-by-side comparison: Original | FGSM Scratch | FGSM ART
    for the first n images.
    """
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    titles = ["Original", "FGSM (Scratch)", "FGSM (ART)"]

    for i in range(n):
        for j, x in enumerate([x_clean, x_adv_fgsm_scratch, x_adv_fgsm_art]):
            img = np.transpose(x[i], (1, 2, 0))  # (3,32,32) → (32,32,3)
            img = np.clip(img, 0, 1)
            axes[i][j].imshow(img)
            axes[i][j].axis("off")
            if i == 0:
                axes[i][j].set_title(titles[j], fontsize=12)

    plt.suptitle("Adversarial Attack Comparison (FGSM)", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  Saved comparison plot → {save_path}")


def save_attack_plot(x_clean, x_adv, y, attack_name, n=10, save_path=None):
    """Saves a side-by-side: Original | Adversarial for PGD or BIM."""
    if save_path is None:
        save_path = f"data/{attack_name.lower()}_comparison.png"

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))

    for i in range(n):
        for j, (x, title) in enumerate([(x_clean, "Original"), (x_adv, f"{attack_name}")]):
            img = np.transpose(x[i], (1, 2, 0))
            img = np.clip(img, 0, 1)
            axes[i][j].imshow(img)
            axes[i][j].axis("off")
            if i == 0:
                axes[i][j].set_title(title, fontsize=12)

    plt.suptitle(f"{attack_name} Attack", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  Saved {attack_name} plot → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB init ──
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "epsilon":    args.epsilon,
                "pgd_steps":  args.pgd_steps,
                "n_samples":  args.n_samples,
                "eps_step":   round(args.epsilon / 4, 4),
            },
        )

    # ── Load trained classifier ──
    model = get_resnet18_classifier(num_classes=10).to(device)
    load_checkpoint(model, args.weights, device)
    model.eval()

    # ── Load raw test images (unnormalized, [0,1]) ──
    print(f"\nLoading {args.n_samples} test images...")
    x_test, y_test = get_cifar10_test_images(args.data_dir, n_samples=args.n_samples)
    print(f"  x_test shape: {x_test.shape}, dtype: {x_test.dtype}")
    print(f"  Pixel range:  [{x_test.min():.3f}, {x_test.max():.3f}]")

    # Save clean images for later use
    np.save(os.path.join(args.output_dir, "x_test_clean.npy"), x_test)
    np.save(os.path.join(args.output_dir, "y_test.npy"),       y_test)

    # ── Build ART classifier wrapper ──
    art_classifier = build_art_classifier(model, device, args)

    eps_step = round(args.epsilon / 4, 4)
    print(f"\nAttack config: epsilon={args.epsilon}, eps_step={eps_step}, pgd_steps={args.pgd_steps}")
    print("="*70)

    # ─────────────────────────────────────────────────────────────────────────
    # A) FGSM — From Scratch
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[A] FGSM from scratch...")
    x_adv_fgsm_scratch = fgsm_scratch(model, x_test, y_test, args.epsilon, device)
    np.save(os.path.join(args.output_dir, "x_adv_fgsm_scratch.npy"), x_adv_fgsm_scratch)
    clean_acc, adv_acc_fgsm_scratch = report_accuracy(model, x_test, x_adv_fgsm_scratch, y_test, device, "FGSM (Scratch)")

    # ─────────────────────────────────────────────────────────────────────────
    # B) FGSM — IBM ART
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[B] FGSM via IBM ART...")
    fgsm_art = FastGradientMethod(
        estimator=art_classifier,
        norm=np.inf,
        eps=args.epsilon,
        targeted=False,
        batch_size=256,
    )
    x_adv_fgsm_art = fgsm_art.generate(x=x_test).astype(np.float32)
    np.save(os.path.join(args.output_dir, "x_adv_fgsm_art.npy"), x_adv_fgsm_art)
    _, adv_acc_fgsm_art = report_accuracy(model, x_test, x_adv_fgsm_art, y_test, device, "FGSM (ART)")

    # ─────────────────────────────────────────────────────────────────────────
    # C) PGD — IBM ART
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[C] PGD via IBM ART...")
    pgd_art = ProjectedGradientDescent(
        estimator=art_classifier,
        norm=np.inf,
        eps=args.epsilon,
        eps_step=eps_step,
        max_iter=args.pgd_steps,
        targeted=False,
        num_random_init=1,   # random start = stronger / more realistic attack
        batch_size=256,
        verbose=False,
    )
    x_adv_pgd = pgd_art.generate(x=x_test).astype(np.float32)
    np.save(os.path.join(args.output_dir, "x_adv_pgd.npy"), x_adv_pgd)
    _, adv_acc_pgd = report_accuracy(model, x_test, x_adv_pgd, y_test, device, "PGD (ART)")

    # ─────────────────────────────────────────────────────────────────────────
    # D) BIM — IBM ART
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[D] BIM via IBM ART...")
    bim_art = BasicIterativeMethod(
        estimator=art_classifier,
        eps=args.epsilon,
        eps_step=eps_step,
        max_iter=args.pgd_steps,
        targeted=False,
        batch_size=256,
        verbose=False,
    )
    x_adv_bim = bim_art.generate(x=x_test).astype(np.float32)
    np.save(os.path.join(args.output_dir, "x_adv_bim.npy"), x_adv_bim)
    _, adv_acc_bim = report_accuracy(model, x_test, x_adv_bim, y_test, device, "BIM (ART)")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Attack':<22} | {'Clean Acc':>9} | {'Adv Acc':>7} | {'Drop':>6}")
    print("="*70)
    results = [
        ("No Attack",        clean_acc, clean_acc,            0),
        ("FGSM (Scratch)",   clean_acc, adv_acc_fgsm_scratch, clean_acc - adv_acc_fgsm_scratch),
        ("FGSM (ART)",       clean_acc, adv_acc_fgsm_art,     clean_acc - adv_acc_fgsm_art),
        ("PGD (ART)",        clean_acc, adv_acc_pgd,          clean_acc - adv_acc_pgd),
        ("BIM (ART)",        clean_acc, adv_acc_bim,          clean_acc - adv_acc_bim),
    ]
    for name, ca, aa, drop in results:
        print(f"{name:<22} | {ca:>8.2f}% | {aa:>6.2f}% | {drop:>5.2f}%")
    print("="*70)

    # ─────────────────────────────────────────────────────────────────────────
    # Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots...")
    save_comparison_plot(
        x_test, x_adv_fgsm_scratch, x_adv_fgsm_art, y_test, n=10,
        save_path=os.path.join(args.output_dir, "fgsm_comparison.png"),
    )
    save_attack_plot(x_test, x_adv_pgd, y_test, "PGD", n=10,
                     save_path=os.path.join(args.output_dir, "pgd_comparison.png"))
    save_attack_plot(x_test, x_adv_bim, y_test, "BIM", n=10,
                     save_path=os.path.join(args.output_dir, "bim_comparison.png"))

    # ─────────────────────────────────────────────────────────────────────────
    # WandB Logging
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        # Log accuracy table
        wandb.log({
            "accuracy/clean":             clean_acc,
            "accuracy/fgsm_scratch":      adv_acc_fgsm_scratch,
            "accuracy/fgsm_art":          adv_acc_fgsm_art,
            "accuracy/pgd":               adv_acc_pgd,
            "accuracy/bim":               adv_acc_bim,
            "accuracy_drop/fgsm_scratch": clean_acc - adv_acc_fgsm_scratch,
            "accuracy_drop/fgsm_art":     clean_acc - adv_acc_fgsm_art,
            "accuracy_drop/pgd":          clean_acc - adv_acc_pgd,
            "accuracy_drop/bim":          clean_acc - adv_acc_bim,
        })

        # Log 10 sample images for each attack
        def log_samples(x_clean, x_adv, n, key):
            images = []
            for i in range(n):
                clean_img = np.transpose(np.clip(x_clean[i], 0, 1), (1, 2, 0))
                adv_img   = np.transpose(np.clip(x_adv[i],   0, 1), (1, 2, 0))
                images.append(wandb.Image(clean_img, caption=f"clean_{i}"))
                images.append(wandb.Image(adv_img,   caption=f"adv_{i}"))
            wandb.log({key: images})

        log_samples(x_test, x_adv_fgsm_scratch, 10, "samples/fgsm_scratch")
        log_samples(x_test, x_adv_fgsm_art,     10, "samples/fgsm_art")
        log_samples(x_test, x_adv_pgd,          10, "samples/pgd")
        log_samples(x_test, x_adv_bim,          10, "samples/bim")

        # Log plot images
        wandb.log({
            "plots/fgsm_comparison": wandb.Image(os.path.join(args.output_dir, "fgsm_comparison.png")),
            "plots/pgd_comparison":  wandb.Image(os.path.join(args.output_dir, "pgd_comparison.png")),
            "plots/bim_comparison":  wandb.Image(os.path.join(args.output_dir, "bim_comparison.png")),
        })

        wandb.finish()

    print("\nDone. All adversarial arrays saved to:", args.output_dir)


if __name__ == "__main__":
    main()
