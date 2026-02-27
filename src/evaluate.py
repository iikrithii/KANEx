"""
Evaluation script for KANEx.

Loads the best checkpoint for each model and evaluates on the test set:
  - AUROC per label + macro average
  - F1 per label + macro average
  - Prints a comparison table

Usage:
    python src/evaluate.py \
        --test_csv /path/to/test.csv \
        --image_root /home/gokul/vlm_xray/.../files/ \
        --output_dir ./outputs \
        --models all
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backbones import build_model, MODEL_REGISTRY
from src.data.dataset import build_test_dataloader, CHEXPERT_LABELS, NUM_CLASSES
from src.utils.metrics import compute_auroc, compute_f1, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KANEx models")
    parser.add_argument("--test_csv",    required=True)
    parser.add_argument("--image_root",  required=True)
    parser.add_argument("--output_dir",  default="./outputs")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Run inference on the test set.
    Returns: (all_labels, all_probs, all_metadata)
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_dicoms = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_dicoms.extend(batch["dicom_id"])

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return all_labels, all_probs, all_dicoms


def load_best_checkpoint(model_name, output_dir, device):
    """
    Load the best checkpoint saved during training.
    Returns: model (loaded state), or None if checkpoint not found.
    """
    checkpoint_path = os.path.join(output_dir, model_name, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] No checkpoint found for {model_name}: {checkpoint_path}")
        return None

    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    epoch = checkpoint.get("epoch", "?")
    val_auc = checkpoint.get("val_auc", float("nan"))
    print(f"[Checkpoint] {model_name}: epoch={epoch}, val_auc={val_auc:.4f}")
    return model


def main():
    args = parse_args()

    if "all" in args.models:
        model_names = list(MODEL_REGISTRY.keys())
    else:
        model_names = args.models

    # Build test dataloader
    test_loader, test_dataset = build_test_dataloader(
        test_csv=args.test_csv,
        image_root=args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    all_results = {}

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*60}")

        model = load_best_checkpoint(model_name, args.output_dir, args.device)
        if model is None:
            continue

        # Run inference
        y_true, y_probs, dicoms = evaluate_model(model, test_loader, args.device)

        # Compute metrics
        auc_results = compute_auroc(y_true, y_probs, CHEXPERT_LABELS)
        f1_results = compute_f1(y_true, y_probs, CHEXPERT_LABELS)

        print_results_table(
            {**{f"AUC_{k}": v for k, v in auc_results.items()},
             **{f"F1_{k}": v for k, v in f1_results.items()}},
            title=f"{model_name} — Test Results"
        )

        all_results[model_name] = {
            "macro_auc": auc_results["macro"],
            "macro_f1":  f1_results["macro"],
            "per_label_auc": auc_results,
            "per_label_f1":  f1_results,
        }

        # Save per-model results as CSV
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for label in CHEXPERT_LABELS:
            rows.append({
                "label": label,
                "auc": auc_results.get(label, float("nan")),
                "f1":  f1_results.get(label, float("nan")),
            })
        rows.append({"label": "MACRO", "auc": auc_results["macro"], "f1": f1_results["macro"]})

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "test_results.csv"), index=False)

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE (Test Set)")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Macro AUC':>12} {'Macro F1':>12}")
    print(f"  {'─'*56}")
    for model_name, res in all_results.items():
        print(f"  {model_name:<30} {res['macro_auc']:>12.4f} {res['macro_f1']:>12.4f}")
    print(f"{'='*70}\n")

    # Save comparison CSV
    comparison_df = pd.DataFrame([
        {"model": name, "macro_auc": res["macro_auc"], "macro_f1": res["macro_f1"]}
        for name, res in all_results.items()
    ])
    comparison_path = os.path.join(args.output_dir, "evaluation_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()