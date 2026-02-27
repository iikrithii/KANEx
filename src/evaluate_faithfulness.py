"""
Insertion & Deletion Faithfulness Evaluation for KANEx Heatmaps.

Evaluates how faithfully each heatmap method (gradcam, swam, kancam) reflects
the model's actual decision process, for every model variant.

---- What insertion and deletion measure ----

DELETION: Start with the full image. Progressively mask out pixels in order
of the heatmap's importance (highest-scoring first). At each step, measure
the model's predicted probability for the target class. Plot probability vs
fraction of pixels removed. A LOWER AUC means the heatmap is more faithful:
the prediction drops quickly when you remove the "important" pixels.

INSERTION: Start with a heavily blurred baseline image (all class information
destroyed). Progressively reveal pixels in order of heatmap importance
(highest-scoring first). At each step, measure the model's predicted
probability. A HIGHER AUC means the heatmap is more faithful: the model
recovers its prediction quickly when the right pixels are shown.

Both use the blurred image as the uninformative baseline rather than black
pixels — blurred images are in-distribution for a ResNet trained on real
images; black images are not.

---- Faithfulness score ----

We report:
  - Deletion AUC (lower = better)
  - Insertion AUC (higher = better)
  - Faithfulness = Insertion AUC - Deletion AUC (higher = better)

---- What class to measure ----

For each image, we use the model's own predicted class (argmax over sigmoid
probabilities) as the target. This avoids the confound of GT labels and
measures faithfulness to the model's actual decision, not to ground truth.

---- Implementation detail ----

We use the already-saved .npy heatmaps from generate_heatmaps.py rather than
regenerating them. This keeps evaluation fast and ensures we measure exactly
the same heatmaps we would report in the paper.

Heatmaps are upsampled to 224×224 (image size) using bilinear interpolation
before computing insertion/deletion masks.

Usage:
    python src/evaluate_faithfulness.py \\
        --test_csv    /path/to/test.csv \\
        --image_root  /path/to/files/ \\
        --output_dir  ./outputs \\
        --models      all \\
        --methods     gradcam swam kancam \\
        --steps       50 \\
        --device      cuda
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import build_model, MODEL_REGISTRY
from src.data.dataset import CHEXPERT_LABELS, NUM_CLASSES, get_val_transform


# ── ImageNet normalization constants ─────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Insertion & Deletion faithfulness evaluation for KANEx heatmaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_csv",   required=True,
                        help="Path to test CSV (same one used for heatmap generation)")
    parser.add_argument("--image_root", required=True,
                        help="MIMIC-CXR image root directory")
    parser.add_argument("--output_dir", default="./outputs",
                        help="Root output dir (same as generate_heatmaps.py)")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--methods", nargs="+", default=["gradcam", "kancam", "rollout"],
                        choices=["gradcam", "kancam", "rollout"])
    parser.add_argument("--steps",  type=int, default=50,
                        help="Number of steps in the insertion/deletion curve. "
                             "Each step masks/reveals (100/steps)%% of pixels.")
    parser.add_argument("--blur_radius", type=int, default=51,
                        help="Gaussian blur radius for the uninformative baseline image. "
                             "Should be large enough to destroy pathology detail.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of test samples (useful for quick debugging). "
                             "None = use all samples that have saved heatmaps.")
    return parser.parse_args()


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_model(model_name: str, output_dir: str, device: str):
    """Load best checkpoint for a model. Returns None if checkpoint not found."""
    ckpt_path = os.path.join(output_dir, model_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No checkpoint at {ckpt_path}")
        return None

    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


# ── Image utilities ───────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str, image_size: int) -> torch.Tensor:
    """
    Load an image and apply the standard val transform.
    Returns: (1, 3, H, W) tensor on CPU.
    """
    transform = get_val_transform(image_size)
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def make_blur_baseline(img_tensor: torch.Tensor, blur_radius: int) -> torch.Tensor:
    """
    Create a blurred version of the image as the uninformative baseline.

    We blur in pixel space (before normalization) then renormalize, because
    a Gaussian blur of a normalized tensor is not the same as normalizing
    a blurred image — the former can produce out-of-distribution values.

    Args:
        img_tensor:  (1, 3, H, W) normalized tensor (ImageNet stats)
        blur_radius: Gaussian blur kernel radius

    Returns:
        blurred: (1, 3, H, W) normalized tensor
    """
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std  = IMAGENET_STD.to(img_tensor.device)

    # Denormalize → pixel space [0, 1]
    pixel = img_tensor[0] * std + mean  # (3, H, W)
    pixel = pixel.clamp(0, 1)

    # Convert to PIL, apply Gaussian blur, convert back
    pil = Image.fromarray(
        (pixel.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    # Use a large radius so even large lung fields are smoothed out
    blurred_pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_np  = np.array(blurred_pil).astype(np.float32) / 255.0

    # Re-normalize with ImageNet stats
    blurred_t = torch.from_numpy(blurred_np).permute(2, 0, 1)  # (3, H, W)
    blurred_t = (blurred_t - IMAGENET_MEAN) / IMAGENET_STD
    return blurred_t.unsqueeze(0)  # (1, 3, H, W)


# ── Heatmap loading + upsampling ─────────────────────────────────────────────

def load_heatmap(heatmap_path: str, target_size: int) -> np.ndarray:
    """
    Load a saved .npy heatmap and upsample to (target_size, target_size).
    Returns: (H, W) float32 array normalized to [0, 1].
    """
    hm = np.load(heatmap_path).astype(np.float32)  # (h, w)

    if hm.shape[0] != target_size or hm.shape[1] != target_size:
        # Upsample with bilinear interpolation
        hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        hm_t = F.interpolate(hm_t, size=(target_size, target_size), mode="bilinear",
                              align_corners=False)
        hm   = hm_t.squeeze().numpy()

    # Re-normalize to [0, 1] after upsampling (interpolation can slightly exceed)
    mn, mx = hm.min(), hm.max()
    if mx - mn > 1e-8:
        hm = (hm - mn) / (mx - mn)

    return hm


# ── Pixel ranking from heatmap ────────────────────────────────────────────────

def heatmap_to_sorted_indices(heatmap: np.ndarray):
    """
    Convert a 2D heatmap to flat pixel indices sorted from most to least important.

    Returns: (H*W,) integer array, index [0] = most important pixel
    """
    flat = heatmap.flatten()
    return np.argsort(flat)[::-1].copy()  # descending importance


# ── Single insertion/deletion curve ──────────────────────────────────────────

@torch.no_grad()
def compute_deletion_curve(
    model:          torch.nn.Module,
    img:            torch.Tensor,
    baseline:       torch.Tensor,
    sorted_indices: np.ndarray,
    class_idx:      int,
    n_steps:        int,
    device:         str,
) -> np.ndarray:
    """
    Deletion curve: start with full image, progressively replace pixels with
    baseline (most important first). Record predicted probability at each step.

    Args:
        model:          Trained model in eval mode
        img:            (1, 3, H, W) full image tensor (CPU)
        baseline:       (1, 3, H, W) blurred baseline tensor (CPU)
        sorted_indices: (H*W,) pixel indices sorted most→least important
        class_idx:      Target class to measure
        n_steps:        Number of measurement points
        device:         Compute device

    Returns:
        probs: (n_steps+1,) predicted probabilities at each step
               probs[0] = full image, probs[-1] = fully masked image
    """
    H, W    = img.shape[-2], img.shape[-1]
    n_pixels = H * W
    step_size = max(1, n_pixels // n_steps)

    current = img.clone()  # start with full image
    probs   = []

    for step in range(n_steps + 1):
        # Measure probability at this masking level
        x    = current.to(device)
        logit = model(x)[:, class_idx]
        prob  = torch.sigmoid(logit).item()
        probs.append(prob)

        if step < n_steps:
            # Mask out the next chunk of most-important pixels
            n_masked = min((step + 1) * step_size, n_pixels)
            indices_to_mask = sorted_indices[:n_masked]

            # Apply mask: replace selected pixels with baseline values
            current_flat   = current.reshape(3, -1)
            baseline_flat  = baseline.reshape(3, -1)
            current_flat[:, indices_to_mask] = baseline_flat[:, indices_to_mask]
            current = current_flat.reshape(1, 3, H, W)

    return np.array(probs, dtype=np.float32)


@torch.no_grad()
def compute_insertion_curve(
    model:          torch.nn.Module,
    img:            torch.Tensor,
    baseline:       torch.Tensor,
    sorted_indices: np.ndarray,
    class_idx:      int,
    n_steps:        int,
    device:         str,
) -> np.ndarray:
    """
    Insertion curve: start with blurred baseline, progressively reveal pixels
    from the full image (most important first). Record predicted probability.

    Args: same as compute_deletion_curve

    Returns:
        probs: (n_steps+1,) predicted probabilities at each step
               probs[0] = fully blurred, probs[-1] = full image revealed
    """
    H, W     = img.shape[-2], img.shape[-1]
    n_pixels  = H * W
    step_size = max(1, n_pixels // n_steps)

    current = baseline.clone()  # start with blurred baseline
    probs   = []

    for step in range(n_steps + 1):
        # Measure probability at this revelation level
        x     = current.to(device)
        logit = model(x)[:, class_idx]
        prob  = torch.sigmoid(logit).item()
        probs.append(prob)

        if step < n_steps:
            # Reveal the next chunk of most-important pixels
            n_revealed = min((step + 1) * step_size, n_pixels)
            indices_to_reveal = sorted_indices[:n_revealed]

            current_flat = current.reshape(3, -1)
            img_flat     = img.reshape(3, -1)
            current_flat[:, indices_to_reveal] = img_flat[:, indices_to_reveal]
            current = current_flat.reshape(1, 3, H, W)

    return np.array(probs, dtype=np.float32)


def auc_from_curve(probs: np.ndarray) -> float:
    """
    Compute normalized AUC of a probability curve using the trapezoidal rule.
    Normalized to [0, 1] by dividing by the number of steps.
    """
    return float(np.trapz(probs) / (len(probs) - 1))


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model_method(
    model_name:  str,
    method:      str,
    model:       torch.nn.Module,
    test_df:     pd.DataFrame,
    output_dir:  str,
    args,
) -> dict:
    """
    Compute insertion and deletion AUC for all available heatmaps of one
    model × method combination.

    Returns:
        dict with 'deletion_aucs', 'insertion_aucs', 'faithfulness_scores' (lists),
        and their means/stds.
    """
    heatmap_dir = os.path.join(output_dir, model_name, "heatmaps", method)
    if not os.path.isdir(heatmap_dir):
        print(f"    [SKIP] No heatmap dir: {heatmap_dir}")
        return None

    heatmap_paths = sorted(glob.glob(os.path.join(heatmap_dir, "*.npy")))
    if not heatmap_paths:
        print(f"    [SKIP] No .npy files in {heatmap_dir}")
        return None

    if args.max_samples is not None:
        heatmap_paths = heatmap_paths[:args.max_samples]

    print(f"    {method.upper():<10}: {len(heatmap_paths)} heatmaps")

    # Build dicom_id → image_path lookup from test CSV
    dicom_to_path = {}
    for _, row in test_df.iterrows():
        # test CSV has 'path' column (relative) or 'dicom_id'
        dicom_id  = str(row["dicom_id"])
        if "path" in row and pd.notna(row["path"]):
            img_path = os.path.join(args.image_root, str(row["path"]))
        else:
            # Fallback: construct path from subject/study/dicom structure
            img_path = str(row.get("image_path", ""))
        dicom_to_path[dicom_id] = img_path

    deletion_aucs    = []
    insertion_aucs   = []
    faithfulness_scores = []

    for hm_path in tqdm(heatmap_paths, desc=f"      {method}", leave=False):
        dicom_id = os.path.splitext(os.path.basename(hm_path))[0]

        # Find the image path
        img_path = dicom_to_path.get(dicom_id)
        if img_path is None or not os.path.exists(img_path):
            continue

        # Load and preprocess image
        try:
            img = load_and_preprocess(img_path, args.image_size)  # (1, 3, H, W)
        except Exception as e:
            print(f"      [WARN] Could not load {img_path}: {e}")
            continue

        # Get model's predicted class for this image (what we measure faithfulness for)
        with torch.no_grad():
            logits    = model(img.to(args.device))
            probs_all = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        class_idx = int(np.argmax(probs_all))  # most confident prediction

        # Skip if model has no strong prediction (all probs < 0.1)
        # — the faithfulness measurement is only meaningful when the model
        # has committed to a prediction
        if probs_all[class_idx] < 0.1:
            continue

        # Load saved heatmap and upsample to image size
        try:
            heatmap = load_heatmap(hm_path, args.image_size)
        except Exception as e:
            print(f"      [WARN] Could not load heatmap {hm_path}: {e}")
            continue

        # Build blurred baseline
        baseline = make_blur_baseline(img, args.blur_radius)

        # Sort pixels by heatmap importance (most important first)
        sorted_idx = heatmap_to_sorted_indices(heatmap)

        # Deletion curve: full → baseline
        del_curve = compute_deletion_curve(
            model, img, baseline, sorted_idx, class_idx,
            n_steps=args.steps, device=args.device,
        )
        del_auc = auc_from_curve(del_curve)

        # Insertion curve: baseline → full
        ins_curve = compute_insertion_curve(
            model, img, baseline, sorted_idx, class_idx,
            n_steps=args.steps, device=args.device,
        )
        ins_auc = auc_from_curve(ins_curve)

        faithfulness = ins_auc - del_auc

        deletion_aucs.append(del_auc)
        insertion_aucs.append(ins_auc)
        faithfulness_scores.append(faithfulness)

    if not deletion_aucs:
        return None

    return {
        "n":                    len(deletion_aucs),
        "deletion_auc_mean":    float(np.mean(deletion_aucs)),
        "deletion_auc_std":     float(np.std(deletion_aucs)),
        "insertion_auc_mean":   float(np.mean(insertion_aucs)),
        "insertion_auc_std":    float(np.std(insertion_aucs)),
        "faithfulness_mean":    float(np.mean(faithfulness_scores)),
        "faithfulness_std":     float(np.std(faithfulness_scores)),
        # Keep raw lists for downstream analysis or per-class breakdown
        "deletion_aucs":        deletion_aucs,
        "insertion_aucs":       insertion_aucs,
        "faithfulness_scores":  faithfulness_scores,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_names = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    print(f"\n{'='*70}")
    print(f"  KANEx Insertion & Deletion Faithfulness Evaluation")
    print(f"{'='*70}")
    print(f"  Models:      {model_names}")
    print(f"  Methods:     {args.methods}")
    print(f"  Steps:       {args.steps}  (each step = {100/args.steps:.1f}% of pixels)")
    print(f"  Blur radius: {args.blur_radius}px")
    print(f"  Device:      {args.device}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*70}\n")

    test_df = pd.read_csv(args.test_csv)
    print(f"[Test CSV] {len(test_df)} samples\n")

    # Accumulate all results for the final comparison table
    all_results = {}   # {model_name: {method: stats_dict}}
    all_rows    = []   # for CSV export

    for model_name in model_names:
        print(f"{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        model = load_model(model_name, args.output_dir, args.device)
        if model is None:
            continue

        model.eval()
        all_results[model_name] = {}

        for method in args.methods:
            stats = evaluate_model_method(
                model_name=model_name,
                method=method,
                model=model,
                test_df=test_df,
                output_dir=args.output_dir,
                args=args,
            )

            if stats is None:
                all_results[model_name][method] = None
                continue

            all_results[model_name][method] = stats

            print(
                f"    {method.upper():<10}  "
                f"Deletion AUC: {stats['deletion_auc_mean']:.4f} ± {stats['deletion_auc_std']:.4f}  "
                f"Insertion AUC: {stats['insertion_auc_mean']:.4f} ± {stats['insertion_auc_std']:.4f}  "
                f"Faithfulness: {stats['faithfulness_mean']:.4f}  "
                f"(n={stats['n']})"
            )

            # Save per-model per-method raw scores for detailed analysis
            raw_df = pd.DataFrame({
                "deletion_auc":   stats["deletion_aucs"],
                "insertion_auc":  stats["insertion_aucs"],
                "faithfulness":   stats["faithfulness_scores"],
            })
            raw_path = os.path.join(
                args.output_dir, model_name,
                f"faithfulness_{method}.csv"
            )
            raw_df.to_csv(raw_path, index=False)

            all_rows.append({
                "model":               model_name,
                "method":              method,
                "n":                   stats["n"],
                "deletion_auc_mean":   stats["deletion_auc_mean"],
                "deletion_auc_std":    stats["deletion_auc_std"],
                "insertion_auc_mean":  stats["insertion_auc_mean"],
                "insertion_auc_std":   stats["insertion_auc_std"],
                "faithfulness_mean":   stats["faithfulness_mean"],
                "faithfulness_std":    stats["faithfulness_std"],
            })

        # Free GPU memory between models
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FAITHFULNESS COMPARISON  (Insertion↑  Deletion↓  Faithfulness↑)")
    print(f"{'='*70}")

    header = f"  {'Model':<26} {'Method':<10}  {'Del AUC':>9}  {'Ins AUC':>9}  {'Faithful':>9}  N"
    print(header)
    print(f"  {'─'*65}")

    for row in all_rows:
        print(
            f"  {row['model']:<26} {row['method']:<10}  "
            f"{row['deletion_auc_mean']:>9.4f}  "
            f"{row['insertion_auc_mean']:>9.4f}  "
            f"{row['faithfulness_mean']:>9.4f}  "
            f"{row['n']}"
        )

    print(f"{'='*70}\n")

    # Save full comparison CSV
    if all_rows:
        comparison_df  = pd.DataFrame(all_rows)
        comparison_path = os.path.join(args.output_dir, "faithfulness_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Full comparison saved: {comparison_path}")

    # ── Per-method summary (averaged across models) ───────────────────────────
    print(f"\n{'─'*50}")
    print(f"  PER-METHOD AVERAGE (across all models)")
    print(f"{'─'*50}")
    if all_rows:
        df = pd.DataFrame(all_rows)
        for method in args.methods:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            print(
                f"  {method.upper():<10}  "
                f"Del: {sub['deletion_auc_mean'].mean():.4f}  "
                f"Ins: {sub['insertion_auc_mean'].mean():.4f}  "
                f"Faith: {sub['faithfulness_mean'].mean():.4f}"
            )
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()