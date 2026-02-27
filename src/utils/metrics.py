"""
Evaluation metrics for KANEx.

  - AUROC: per-label and macro-average Area Under ROC Curve
  - F1:    per-label and macro-average F1 at threshold 0.5
  - IoU:   Intersection over Union between heatmap and GT bounding box
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from typing import List, Optional


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray, labels: list):
    """
    Compute per-label and macro-average AUROC.

    Args:
        y_true:   (N, num_classes) binary ground truth
        y_scores: (N, num_classes) predicted probabilities
        labels:   list of label names

    Returns:
        dict: {label: auroc, ..., 'macro': macro_avg}
    """
    results = {}
    valid_aucs = []

    for i, label in enumerate(labels):
        col_true = y_true[:, i]
        col_score = y_scores[:, i]

        # Skip labels with only one class in y_true (AUC undefined)
        if len(np.unique(col_true)) < 2:
            results[label] = float("nan")
            continue

        auc = roc_auc_score(col_true, col_score)
        results[label] = auc
        valid_aucs.append(auc)

    results["macro"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return results


def compute_f1(y_true: np.ndarray, y_scores: np.ndarray, labels: list, threshold: float = 0.5):
    """
    Compute per-label and macro-average F1 score.

    Args:
        y_true:    (N, num_classes)
        y_scores:  (N, num_classes) probabilities
        labels:    label names
        threshold: binarization threshold

    Returns:
        dict: {label: f1, ..., 'macro': macro_avg}
    """
    y_pred = (y_scores >= threshold).astype(int)
    results = {}

    for i, label in enumerate(labels):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        # zero_division=0 prevents warning for empty predictions
        f1 = f1_score(col_true, col_pred, zero_division=0)
        results[label] = f1

    # Macro F1 (only over labels with at least one positive)
    valid_labels = [i for i in range(len(labels)) if y_true[:, i].sum() > 0]
    if valid_labels:
        macro = np.mean([results[labels[i]] for i in valid_labels])
    else:
        macro = float("nan")

    results["macro"] = float(macro)
    return results


def heatmap_to_binary_mask(heatmap: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    """
    Convert a continuous heatmap to a binary mask by thresholding
    at the given percentile of non-zero values.

    Args:
        heatmap:    (H, W) float array in [0, 1]
        percentile: Values above this percentile are considered 'activated'

    Returns:
        mask: (H, W) binary array
    """
    threshold = np.percentile(heatmap, percentile)
    return (heatmap >= threshold).astype(np.float32)


def bbox_to_mask(
    x: float, y: float, w: float, h: float,
    img_w: float, img_h: float,
    mask_h: int, mask_w: int,
) -> np.ndarray:
    """
    Convert a bounding box (in original image coordinates) to a binary
    mask at the heatmap resolution.

    Args:
        x, y, w, h:   Bounding box in original pixel coordinates
        img_w, img_h: Original image size
        mask_h, mask_w: Target mask resolution (heatmap size)

    Returns:
        mask: (mask_h, mask_w) binary array
    """
    # Normalize bbox to [0, 1] relative coordinates
    x1 = x / img_w
    y1 = y / img_h
    x2 = (x + w) / img_w
    y2 = (y + h) / img_h

    # Scale to mask resolution
    mx1 = int(x1 * mask_w)
    my1 = int(y1 * mask_h)
    mx2 = int(x2 * mask_w)
    my2 = int(y2 * mask_h)

    # Clamp to valid range
    mx1 = max(0, min(mx1, mask_w - 1))
    my1 = max(0, min(my1, mask_h - 1))
    mx2 = max(mx1 + 1, min(mx2, mask_w))
    my2 = max(my1 + 1, min(my2, mask_h))

    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    mask[my1:my2, mx1:mx2] = 1.0
    return mask


def compute_iou(heatmap: np.ndarray, bbox_dict: dict, percentile: float = 75.0) -> float:
    """
    Compute IoU between a heatmap and a ground truth bounding box.

    Steps:
      1. Binarize heatmap at `percentile` threshold
      2. Convert bbox to binary mask at heatmap resolution
      3. Compute IoU = |intersection| / |union|

    Args:
        heatmap:      (H, W) float heatmap in [0, 1]
        bbox_dict:    {'x': ..., 'y': ..., 'w': ..., 'h': ...,
                       'img_w': ..., 'img_h': ...}
        percentile:   Binarization percentile

    Returns:
        iou: float in [0, 1]
    """
    H, W = heatmap.shape

    # Skip if bbox is empty (no annotation for this sample)
    if bbox_dict["w"] <= 0 or bbox_dict["h"] <= 0:
        return float("nan")

    # Binarize heatmap
    pred_mask = heatmap_to_binary_mask(heatmap, percentile)

    # Convert bbox to mask at heatmap resolution
    gt_mask = bbox_to_mask(
        x=bbox_dict["x"], y=bbox_dict["y"],
        w=bbox_dict["w"], h=bbox_dict["h"],
        img_w=bbox_dict["img_w"], img_h=bbox_dict["img_h"],
        mask_h=H, mask_w=W,
    )

    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).sum()

    if union == 0:
        return float("nan")

    return float(intersection / union)


def print_results_table(results: dict, title: str = "Results"):
    """Pretty-print a results dictionary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.4f}")
        else:
            print(f"  {k:<35}: {v}")
    print(f"{'='*60}\n")