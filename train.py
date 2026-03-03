"""
Training script for the Anatomy-Aware Deep Learning (AADL) neuroprognostication model.

Usage:
    python train.py \\
        --data_dir data/split \\
        --output_dir outputs/run1 \\
        --target_shape 256 256 64 \\
        --batch_size 4 \\
        --epochs 150 \\
        --lr 4e-5 \\
        --dropout 0.05 \\
        --warmup_epochs 5 \\
        --patience 30 \\
        --seed 42 \\
        --use_amp
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from monai.utils import set_determinism
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast

from dataset import compute_class_weights, create_dataloader, get_data_list, seed_worker
from model import DenseNet121WithAtlasAttention


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_determinism(seed=seed)


# ---------------------------------------------------------------------------
# Train / validate helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, correct, total = 0, 0, 0
    amp_device = device.type
    for batch in loader:
        images = batch["image"].to(device)
        atlas = batch["atlas"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with autocast(device_type=amp_device, enabled=use_amp):
            outputs, _, _ = model(images, atlas)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            atlas = batch["atlas"].to(device)
            labels = batch["label"].to(device)
            outputs, _, _ = model(images, atlas)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["val_auc"], label="Val AUC")
    axes[2].set_title("AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_curves.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the AADL neuroprognostication model.")
    parser.add_argument("--data_dir", required=True, help="Root data directory (contains train/val/test)")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and plots")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[256, 256, 64], metavar=("H", "W", "D"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--expand_kernel_size", type=int, default=3)
    parser.add_argument("--expand_iterations", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision (FP16)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    target_shape = tuple(args.target_shape)
    train_loader, train_list = create_dataloader(
        args.data_dir, "train", target_shape, args.batch_size,
        augment=True, seed=args.seed, num_workers=args.num_workers,
    )
    val_loader, _ = create_dataloader(
        args.data_dir, "val", target_shape, args.batch_size,
        augment=False, seed=args.seed, num_workers=args.num_workers,
    )

    class_weights = compute_class_weights(train_list, device)

    # --- Model ---
    model = DenseNet121WithAtlasAttention(
        in_channels=1,
        atlas_channels=1,
        out_channels=2,
        dropout_prob=args.dropout,
        expand_kernel_size=args.expand_kernel_size,
        expand_iterations=args.expand_iterations,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.use_amp)

    # Warmup: linear LR increase for warmup_epochs, then ReduceLROnPlateau
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- Training loop ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_amp
        )
        val_metrics = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f} "
            f"F1={val_metrics['f1']:.4f} AUC={val_metrics['auc']:.4f}"
        )

        # LR scheduling
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_metrics["loss"])

        # Save best models
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), output_dir / "best_model_loss.pth")
            print(f"  → Saved best_model_loss.pth (val_loss={best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), output_dir / "best_model_acc.pth")
            print(f"  → Saved best_model_acc.pth (val_acc={best_val_acc:.4f})")

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    # Save final checkpoint and training curves
    torch.save(model.state_dict(), output_dir / "final_model.pth")
    save_training_curves(history, output_dir)
    print(f"\nTraining complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
