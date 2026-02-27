"""
Visualization utilities for heatmaps, training curves, and overlays.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import cv2


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = None,
) -> np.ndarray:
    """
    Overlay a heatmap on an image using Jet colormap.

    Args:
        image:   (H, W, 3) uint8 RGB image [0, 255]
        heatmap: (h, w) float in [0, 1]
        alpha:   transparency of heatmap overlay
        colormap: cv2 colormap constant (default: COLORMAP_JET)

    Returns:
        overlay: (H, W, 3) uint8 blended image
    """
    if colormap is None:
        colormap = cv2.COLORMAP_JET

    H, W = image.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(
        (heatmap * 255).astype(np.uint8), (W, H),
        interpolation=cv2.INTER_LINEAR
    )

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_heatmap(
    heatmap: np.ndarray,
    save_path: str,
):
    """Save a raw heatmap as a .npy file for later loading."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, heatmap.astype(np.float32))


def save_heatmap_visualization(
    image_path: str,
    heatmap: np.ndarray,
    save_path: str,
    bbox: dict = None,
    title: str = "",
):
    """
    Save a side-by-side visualization: original | heatmap overlay.
    Optionally draw the ground truth bounding box.

    Args:
        image_path: Path to original image
        heatmap:    (H, W) float heatmap
        save_path:  Where to save the PNG visualization
        bbox:       Optional bounding box dict {x, y, w, h, img_w, img_h}
        title:      Plot title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load original image
    img = np.array(Image.open(image_path).convert("RGB").resize((224, 224)))

    # Create overlay
    overlay = overlay_heatmap_on_image(img, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Heatmap Overlay")
    axes[1].axis("off")

    # Draw GT bounding box if provided
    if bbox is not None and bbox.get("w", 0) > 0:
        # Scale bbox to 224x224
        scale_x = 224 / bbox["img_w"]
        scale_y = 224 / bbox["img_h"]
        rect = patches.Rectangle(
            (bbox["x"] * scale_x, bbox["y"] * scale_y),
            bbox["w"] * scale_x, bbox["h"] * scale_y,
            linewidth=2, edgecolor="lime", facecolor="none",
            label=f"GT: {bbox.get('category', '')}"
        )
        axes[1].add_patch(rect)
        axes[1].legend(loc="upper right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_aucs: list,
    val_aucs: list,
    save_path: str,
    model_name: str = "",
):
    """
    Save training/validation loss and AUC curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=3)
    axes[0].plot(epochs, val_losses, "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_aucs, "b-o", label="Train AUC", markersize=3)
    axes[1].plot(epochs, val_aucs, "r-o", label="Val AUC", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro AUROC")
    axes[1].set_title(f"{model_name} — AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Training curves saved: {save_path}")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor back to uint8 numpy array.
    Reverses ImageNet normalization.

    Args:
        tensor: (3, H, W) normalized tensor
    Returns:
        (H, W, 3) uint8 array
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img