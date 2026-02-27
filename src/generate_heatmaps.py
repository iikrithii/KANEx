"""
Heatmap Generation + IoU Evaluation for KANEx.

For each model x each heatmap method:
  1. Generate heatmaps for all test samples
  2. Save heatmaps as .npy  →  {output_dir}/{model}/heatmaps/{method}/{dicom_id}.npy
  3. Compute IoU against ground-truth bounding boxes
  4. Save PNG visualizations (optional)
  5. Print + save IoU comparison table

Methods:
  gradcam  — Grad-CAM (ResNet: GradCAM; ViT: ViTGradCAM)
  kancam   — KAN-CAM  (KAN models only; ResNet+KAN and ViT+KAN)
  rollout  — Gradient-Weighted Attention Rollout (ViT models only)

Usage:
    python src/generate_heatmaps.py \
        --test_csv    /path/to/test.csv \
        --image_root  /path/to/files/ \
        --output_dir  ./outputs \
        --models      all \
        --methods     gradcam kancam rollout
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import build_model, MODEL_REGISTRY
from src.data.dataset import build_test_dataloader, CHEXPERT_LABELS, NUM_CLASSES, get_val_transform
from src.heatmaps.gradcam import GradCAM, ViTGradCAM
from src.heatmaps.attention_rollout import AttentionRollout
from src.heatmaps.kancam import KANCAM
from src.utils.metrics import compute_iou
from src.utils.visualisation import save_heatmap, save_heatmap_visualization, denormalize_image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate heatmaps and compute IoU")
    parser.add_argument("--test_csv",    required=True)
    parser.add_argument("--image_root",  required=True)
    parser.add_argument("--output_dir",  default="./outputs")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--methods", nargs="+", default=["gradcam", "kancam", "rollout"],
                        choices=["gradcam", "kancam", "rollout"])
    parser.add_argument("--metadata_csv",    default=None)
    parser.add_argument("--save_viz",        action="store_true", default=True)
    parser.add_argument("--iou_percentile",  type=float, default=75.0)
    parser.add_argument("--discard_ratio",   type=float, default=0.9,
                        help="Rollout: fraction of lowest-attention values to discard (sharpens map)")
    return parser.parse_args()


def load_best_checkpoint(model_name, output_dir, device):
    checkpoint_path = os.path.join(output_dir, model_name, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] No checkpoint: {checkpoint_path}")
        return None
    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


# ── Heatmap generators ────────────────────────────────────────────────────────

def generate_gradcam_heatmap(model, x, threshold=0.5):
    """
    Grad-CAM for ResNet (GradCAM) or ViT (ViTGradCAM).
    Always applicable — every model has a Grad-CAM target layer.
    """
    is_vit = hasattr(model, "is_vit_model") and model.is_vit_model()
    if is_vit:
        cam = ViTGradCAM(model, grid_size=model.GRID_SIZE)
    else:
        cam = GradCAM(model, target_layer=model.get_cam_target_layer())
    result = cam.generate_multilabel(x, threshold=threshold, labels=CHEXPERT_LABELS)
    cam.remove_hooks()
    return result["aggregate"], result["probs"]


def generate_kancam_heatmap(model, x, threshold=0.5):
    """
    KAN-CAM: spatial spline projection.
    Applicable to all KAN-head models (ResNet+KAN and ViT+KAN).
    Not applicable to: baseline models, ConvKAN.
    """
    if not model.is_kan_model():
        return None, None
    if hasattr(model, "is_conv_kan") and model.is_conv_kan():
        return None, None
    kancam = KANCAM(model)
    result = kancam.generate_multilabel(x, threshold=threshold, labels=CHEXPERT_LABELS)
    return result["aggregate"], result["probs"]


def generate_rollout_heatmap(model, x, threshold=0.5, discard_ratio=0.9):
    """
    Gradient-Weighted Attention Rollout.
    Applicable to ViT models only.
    """
    if not (hasattr(model, "is_vit_model") and model.is_vit_model()):
        return None, None
    rollout = AttentionRollout(model, grid_size=model.GRID_SIZE, discard_ratio=discard_ratio)
    result  = rollout.generate_multilabel(x, threshold=threshold, labels=CHEXPERT_LABELS)
    rollout.remove_hooks()
    return result["aggregate"], result["probs"]


def resize_heatmap_to_standard(heatmap: np.ndarray, size: int = 14) -> np.ndarray:
    """Resize heatmap to standard size for IoU comparison."""
    return cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_LINEAR)


# ── Per-model processing ──────────────────────────────────────────────────────

def process_model_heatmaps(model_name, model, test_loader, test_dataset, args):
    """Generate all heatmaps for one model, save them, compute IoU."""
    model.eval()
    method_ious = {m: [] for m in args.methods}

    # Create output dirs
    for method in args.methods:
        heatmap_dir = os.path.join(args.output_dir, model_name, "heatmaps", method)
        os.makedirs(heatmap_dir, exist_ok=True)
        if args.save_viz:
            os.makedirs(os.path.join(heatmap_dir, "viz"), exist_ok=True)

    print(f"  Processing {len(test_dataset)} test samples...")

    for batch in tqdm(test_loader, desc=f"{model_name}"):
        x        = batch["image"].to(args.device)
        dicom_id = batch["dicom_id"][0]
        img_path = batch["image_path"][0]
        bbox = {k: v[0] if isinstance(v, (list, torch.Tensor)) else v
                for k, v in batch["bbox"].items()}
        # Fix: bbox values come as tensors from dataloader
        bbox = {k: v[0] if isinstance(v, (list, torch.Tensor)) else v
                for k, v in batch["bbox"].items()}
        # Fix: bbox values come as tensors from dataloader
        bbox = {k: float(v) if isinstance(v, torch.Tensor) else v
                for k, v in bbox.items()}

        for method in args.methods:
            heatmap_dir = os.path.join(args.output_dir, model_name, "heatmaps", method)
            npy_path    = os.path.join(heatmap_dir, f"{dicom_id}.npy")

            # Generate
            if method == "gradcam":
                heatmap, probs = generate_gradcam_heatmap(model, x)
            elif method == "kancam":
                heatmap, probs = generate_kancam_heatmap(model, x)
            elif method == "rollout":
                heatmap, probs = generate_rollout_heatmap(
                    model, x, discard_ratio=args.discard_ratio
                )

            if heatmap is None:
                continue

            # Save raw .npy
            save_heatmap(heatmap, npy_path)

            # Save visualization PNG
            if args.save_viz:
                viz_path = os.path.join(heatmap_dir, "viz", f"{dicom_id}.png")
                save_heatmap_visualization(
                    image_path=img_path,
                    heatmap=heatmap,
                    save_path=viz_path,
                    bbox=bbox if bbox.get("w", 0) > 0 else None,
                    title=f"{model_name} | {method.upper()} | {dicom_id}",
                )

            # IoU
            heatmap_std = resize_heatmap_to_standard(heatmap, size=14)
            iou = compute_iou(heatmap_std, bbox, percentile=args.iou_percentile)
            if not np.isnan(iou):
                method_ious[method].append(iou)

    return method_ious


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.batch_size = 1

    model_names = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models

    test_loader, test_dataset = build_test_dataloader(
        test_csv=args.test_csv,
        image_root=args.image_root,
        batch_size=1,
        num_workers=args.num_workers,
        image_size=args.image_size,
        metadata_csv=args.metadata_csv,
    )

    print(f"\n{'='*70}")
    print(f"  KANEx Heatmap Generation + IoU Evaluation")
    print(f"{'='*70}")
    print(f"  Models:   {model_names}")
    print(f"  Methods:  {args.methods}")
    print(f"  Samples:  {len(test_dataset)}")
    print(f"  Device:   {args.device}")
    print(f"{'='*70}\n")

    all_iou_results = {}

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        model = load_best_checkpoint(model_name, args.output_dir, args.device)
        if model is None:
            continue

        method_ious = process_model_heatmaps(
            model_name, model, test_loader, test_dataset, args
        )

        all_iou_results[model_name] = {}
        for method, iou_list in method_ious.items():
            if iou_list:
                mean_iou = float(np.mean(iou_list))
                std_iou  = float(np.std(iou_list))
                all_iou_results[model_name][method] = {"mean": mean_iou, "std": std_iou, "n": len(iou_list)}
                print(f"  {method.upper():<10}: IoU = {mean_iou:.4f} +/- {std_iou:.4f}  (n={len(iou_list)})")
            else:
                all_iou_results[model_name][method] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                print(f"  {method.upper():<10}: N/A (not applicable to this model)")

        # Per-model CSV
        iou_rows = [{"method": m, **s} for m, s in all_iou_results[model_name].items()]
        pd.DataFrame(iou_rows).to_csv(
            os.path.join(args.output_dir, model_name, "iou_results.csv"), index=False
        )

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  IoU COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"  {'Model':<28}" + "".join(f"  {m.upper():<14}" for m in args.methods)
    print(header)
    print(f"  {'─'*65}")

    rows = []
    for model_name, method_results in all_iou_results.items():
        row_str = f"  {model_name:<28}"
        row     = {"model": model_name}
        for method in args.methods:
            stats = method_results.get(method, {})
            val   = stats.get("mean", float("nan"))
            if not np.isnan(val):
                row_str += f"  {val:<14.4f}"
            else:
                row_str += f"  {'N/A':<14}"
            row[f"iou_{method}"] = val
        print(row_str)
        rows.append(row)

    print(f"{'='*70}\n")

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(args.output_dir, "iou_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"IoU comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()