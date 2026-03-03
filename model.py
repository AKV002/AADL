"""
GradCAM++ explainability script for the AADL neuroprognostication model.

Generates per-case combined saliency maps from three feature branches:
  - Backbone  : norm5
  - Att3      : norm_att3
  - Att4      : att4.W

Outputs are organized into TP / TN / FP / FN folders.

Usage:
    python explainability.py \\
        --model_path outputs/run1/best_model_loss.pth \\
        --data_dir data/split \\
        --split test \\
        --class_idx 1 \\
        --output_dir outputs/run1/gradcam
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import create_dataloader, get_data_list
from model import DenseNet121WithAtlasAttention


# ---------------------------------------------------------------------------
# GradCAM++ implementation
# ---------------------------------------------------------------------------
class CombinedGradCAMpp3D:
    """
    GradCAM++ over three branches of DenseNet121WithAtlasAttention:
      1. norm5        (backbone, n4 channels)
      2. norm_att3    (atlas attention block 3, n3 channels)
      3. att4.W       (atlas attention block 4, n4 channels)

    The three feature maps are concatenated to form a 3072-channel combined
    representation; GradCAM++ weights are computed over this tensor.
    Per-branch saliency is obtained by splitting the combined map back into
    the three channel groups.

    Activations are captured via forward hooks; gradients are captured via
    full_backward hooks — a single backward pass is sufficient.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._activations = {}
        self._gradients = {}

        for name, module in [
            ("norm5", model.norm5),
            ("norm_att3", model.norm_att3),
            ("att4_W", model.att4.W),
        ]:
            self.hooks.append(module.register_forward_hook(self._make_act_hook(name)))
            self.hooks.append(module.register_full_backward_hook(self._make_grad_hook(name)))

    def _make_act_hook(self, name):
        def hook(module, input, output):
            self._activations[name] = output.detach()
        return hook

    def _make_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the loss w.r.t. the module output
            self._gradients[name] = grad_output[0].detach()
        return hook

    def _compute_gradcampp(self, activations, grads):
        """Compute GradCAM++ saliency map from activations and gradients."""
        grads_sq = grads ** 2
        grads_cub = grads ** 3
        # Sum activations over spatial dims for denominator
        spatial_sum = activations.sum(dim=(2, 3, 4), keepdim=True)
        denom = 2 * grads_sq + grads_cub * spatial_sum
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads_sq / denom
        # Weighted combination
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return cam

    def compute(self, ct, atlas, class_idx):
        """
        Run a forward + backward pass and return saliency maps.

        Args:
            ct:        Tensor (B, 1, H, W, D)
            atlas:     Tensor (B, 1, H, W, D)
            class_idx: Class index to explain (0=good, 1=poor)

        Returns dict with keys:
            combined, backbone, att3, att4  — each a numpy array (H, W, D)
        """
        self.model.eval()
        self._activations.clear()
        self._gradients.clear()

        logits, att3_map, att4_map = self.model(ct, atlas)
        score = logits[0, class_idx]
        self.model.zero_grad()
        # Single backward pass — activations captured in forward hooks,
        # gradients captured in full_backward hooks registered at __init__.
        score.backward()

        ref_size = self._activations["norm5"].shape[2:]

        saliencies = {}
        for key in ("norm5", "norm_att3", "att4_W"):
            act = self._activations[key]
            grad = self._gradients.get(key, torch.zeros_like(act))
            cam = self._compute_gradcampp(act, grad)
            cam_up = F.interpolate(cam, size=ref_size, mode="trilinear", align_corners=False)
            saliencies[key] = cam_up

        combined = sum(saliencies.values())
        combined_np = combined[0, 0].cpu().numpy()
        combined_np = (combined_np - combined_np.min()) / (combined_np.max() - combined_np.min() + 1e-8)

        def _norm(t):
            a = t[0, 0].cpu().numpy()
            return (a - a.min()) / (a.max() - a.min() + 1e-8)

        return {
            "combined": combined_np,
            "backbone": _norm(saliencies["norm5"]),
            "att3": _norm(saliencies["norm_att3"]),
            "att4": _norm(saliencies["att4_W"]),
            "att3_map": att3_map[0, 0].detach().cpu().numpy(),
            "att4_map": att4_map[0, 0].detach().cpu().numpy(),
        }

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_case(ct_np, saliency, patient_id, label, pred, output_path):
    """
    Plot mid-slice views of CT + four saliency branches + attention maps.

    Rows: combined | backbone | att3 | att4 | att3_map | att4_map
    """
    z_mid = ct_np.shape[-1] // 2

    def sl(arr):
        if arr.ndim == 3:
            return arr[:, :, z_mid].T
        return arr.T

    rows = [
        ("CT", sl(ct_np), "gray"),
        ("Combined", sl(saliency["combined"]), "hot"),
        ("Backbone", sl(saliency["backbone"]), "hot"),
        ("Att3", sl(saliency["att3"]), "hot"),
        ("Att4", sl(saliency["att4"]), "hot"),
        ("Att3 map", sl(saliency["att3_map"]), "viridis"),
        ("Att4 map", sl(saliency["att4_map"]), "viridis"),
    ]

    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4))
    for ax, (title, img, cmap) in zip(axes, rows):
        ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"{patient_id} | True={'Good' if label == 0 else 'Poor'} "
        f"Pred={'Good' if pred == 0 else 'Poor'}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GradCAM++ explainability for the AADL model.")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--class_idx", type=int, default=1, help="Class index to explain (0=good, 1=poor)")
    parser.add_argument("--output_dir", required=True, help="Directory to save GradCAM outputs")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 64], metavar=("H", "W", "D"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for folder in ("TP", "TN", "FP", "FN"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data (batch_size=1 for GradCAM) ---
    target_shape = tuple(args.target_shape)
    loader, data_list = create_dataloader(
        args.data_dir, args.split, target_shape, batch_size=1,
        augment=False, num_workers=0,
    )

    # --- Model ---
    model = DenseNet121WithAtlasAttention(in_channels=1, atlas_channels=1, out_channels=2).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {args.model_path}")

    cam = CombinedGradCAMpp3D(model)

    summary = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for i, (batch, meta) in enumerate(zip(loader, data_list)):
        ct = batch["image"].to(device)
        atlas = batch["atlas"].to(device)
        label = int(batch["label"].item())

        saliency = cam.compute(ct, atlas, args.class_idx)

        with torch.no_grad():
            logits, _, _ = model(ct, atlas)
        pred = int(logits.argmax(1).item())

        if label == 1 and pred == 1:
            category = "TP"
        elif label == 0 and pred == 0:
            category = "TN"
        elif label == 0 and pred == 1:
            category = "FP"
        else:
            category = "FN"

        summary[category] += 1
        patient_id = meta["patient_id"]
        ct_np = ct[0, 0].detach().cpu().numpy()

        plot_case(
            ct_np, saliency, patient_id, label, pred,
            output_dir / category / f"{patient_id}.png",
        )
        print(f"[{i+1}/{len(data_list)}] {patient_id}: label={label} pred={pred} → {category}")

    cam.remove_hooks()

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
