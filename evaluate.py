"""
Evaluation script for the AADL neuroprognostication model.

Computes: Loss, Accuracy, Balanced Accuracy, Sensitivity, Specificity,
          Precision, F1, AUC, PPV, NPV.
Generates: confusion matrix plot, attention map visualizations, metrics.npz.

Usage:
    python evaluate.py \\
        --model_path outputs/run1/best_model_loss.pth \\
        --data_dir data/split \\
        --split test \\
        --batch_size 4 \\
        --output_dir outputs/run1/eval
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from dataset import create_dataloader
from model import DenseNet121WithAtlasAttention


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def run_inference(model, loader, criterion, device):
    """Run model over the full loader; return aggregated predictions and labels."""
    model.eval()
    total_loss, total = 0, 0
    all_preds, all_probs, all_labels = [], [], []
    all_att3, all_att4 = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            atlas = batch["atlas"].to(device)
            labels = batch["label"].to(device)

            outputs, att3, att4 = model(images, atlas)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            total += len(labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_att3.append(att3.cpu())
            all_att4.append(att4.cpu())

    return (
        total_loss / total,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        torch.cat(all_att3, dim=0),
        torch.cat(all_att4, dim=0),
    )


def compute_metrics(labels, preds, probs):
    """Compute classification metrics and return as a dict."""
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
        "ppv": ppv,
        "npv": npv,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(labels, preds, output_path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Good", "Poor"],
        yticklabels=["Good", "Poor"],
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_attention_maps(cases, output_dir, title_prefix=""):
    """
    Plot attention maps for a list of cases.

    Each case is a dict with keys: patient_id, att3 (tensor), att4 (tensor), label, pred.
    """
    for case in cases:
        att3 = case["att3"].squeeze().numpy()
        att4 = case["att4"].squeeze().numpy()
        z_mid = att3.shape[-1] // 2 if att3.ndim == 3 else 0

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, att, name in zip(axes, [att3, att4], ["att3", "att4"]):
            sl = att[:, :, z_mid] if att.ndim == 3 else att
            ax.imshow(sl.T, cmap="hot", origin="lower")
            ax.set_title(f"{name} (z={z_mid})")
            ax.axis("off")
        fig.suptitle(
            f"{title_prefix}{case['patient_id']} | "
            f"True={'Good' if case['label'] == 0 else 'Poor'} "
            f"Pred={'Good' if case['pred'] == 0 else 'Poor'}"
        )
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{case['patient_id']}_attention.png", dpi=80)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the AADL neuroprognostication model.")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation outputs")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("H", "W", "D"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    target_shape = tuple(args.target_shape)
    loader, data_list = create_dataloader(
        args.data_dir, args.split, target_shape, args.batch_size,
        augment=False, num_workers=4,
    )

    # --- Model ---
    model = DenseNet121WithAtlasAttention(in_channels=1, atlas_channels=1, out_channels=2).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model_path}")

    criterion = nn.CrossEntropyLoss()

    # --- Inference ---
    loss, labels, preds, probs, att3_maps, att4_maps = run_inference(model, loader, criterion, device)

    # --- Metrics ---
    metrics = compute_metrics(labels, preds, probs)
    metrics["loss"] = loss

    print(f"\n{'='*50}")
    print(f"Split: {args.split.upper()}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<20s}: {v:.4f}")

    # --- Confusion matrix ---
    plot_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")

    # --- Attention map visualizations: correct good, correct poor, misclassified ---
    cases = []
    for i, d in enumerate(data_list):
        cases.append(
            {
                "patient_id": d["patient_id"],
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "att3": att3_maps[i],
                "att4": att4_maps[i],
            }
        )

    def pick_one(condition):
        return next((c for c in cases if condition(c)), None)

    representative = [
        pick_one(lambda c: c["label"] == 0 and c["pred"] == 0),   # correct good
        pick_one(lambda c: c["label"] == 1 and c["pred"] == 1),   # correct poor
        pick_one(lambda c: c["label"] != c["pred"]),               # misclassified
    ]
    representative = [c for c in representative if c is not None]
    plot_attention_maps(representative, output_dir, title_prefix="")

    # --- Save metrics ---
    np.savez(output_dir / "metrics.npz", **{k: np.array(v) for k, v in metrics.items()})
    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
