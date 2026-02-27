"""
Grad-CAM for multi-label chest X-ray models.

Standard Grad-CAM:
  1. Forward pass → get class logit for label k
  2. Backward pass → get gradients at target conv layer
  3. Pool gradients spatially → channel weights
  4. Weighted sum of feature maps → heatmap
  5. ReLU + normalize to [0, 1]

For multi-label: generate one heatmap per predicted positive label,
then blend them with prediction confidence as weights.

Reference: Selvaraju et al. (2017) Grad-CAM: Visual Explanations
           from Deep Networks via Gradient-based Localization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class GradCAM:
    """
    Grad-CAM wrapper for any model with a target convolutional layer.

    Usage:
        cam = GradCAM(model, target_layer=model.get_cam_target_layer())
        heatmap = cam.generate(image_tensor, class_idx=3)

    Args:
        model:        nn.Module with a get_cam_target_layer() method
        target_layer: The conv layer to hook (usually last conv)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations = None  # will store forward activations
        self._gradients = None    # will store backward gradients

        # Register hooks on the target layer
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: save the feature map at the target layer."""
        self._activations = output.detach()  # (B, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradient of loss w.r.t. feature map."""
        self._gradients = grad_output[0].detach()  # (B, C, H, W)

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int,
        relu: bool = True,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class.

        Args:
            x:         (B, 3, H, W) input batch (should be B=1 for single images)
            class_idx: Which class/label to explain (0-indexed)
            relu:      Whether to apply ReLU to the CAM (standard practice)

        Returns:
            cam: (B, H_feat, W_feat) numpy array, values in [0, 1]
        """
        self.model.eval()
        x.requires_grad_(True)

        # Forward pass
        logits = self.model(x)  # (B, num_classes)

        # Zero grads, then backprop for the target class only
        self.model.zero_grad()
        target = logits[:, class_idx].sum()  # sum over batch
        target.backward()

        # Grad-CAM: weight channels by global-average-pooled gradient
        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1)               # (B, H, W)

        if relu:
            cam = torch.clamp(cam, min=0)

        # Normalize each sample to [0, 1]
        cam_np = cam.cpu().numpy()
        for b in range(cam_np.shape[0]):
            mn, mx = cam_np[b].min(), cam_np[b].max()
            if mx - mn > 1e-8:
                cam_np[b] = (cam_np[b] - mn) / (mx - mn)

        return cam_np  # (B, H_feat, W_feat)

    def generate_multilabel(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        labels: list = None,
    ) -> dict:
        """
        Generate one Grad-CAM heatmap per predicted positive label,
        and a confidence-weighted aggregate heatmap.

        Args:
            x:          (1, 3, H, W) single image
            threshold:  Confidence threshold to consider a label 'positive'
            labels:     List of label names (for dict keys)

        Returns:
            dict with:
              'per_label': {label_name: (H, W) heatmap}  — one per positive label
              'aggregate': (H, W) heatmap                 — weighted blend
              'probs':     (num_classes,) probabilities
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0)  # (num_classes,)

        probs_np = probs.cpu().numpy()
        positive_indices = np.where(probs_np >= threshold)[0]

        per_label_maps = {}
        aggregate = None
        total_weight = 0.0

        for idx in positive_indices:
            cam = self.generate(x.clone(), class_idx=int(idx))  # (1, H, W)
            cam_2d = cam[0]  # (H, W)

            key = labels[idx] if labels else str(idx)
            per_label_maps[key] = cam_2d

            # Confidence-weighted aggregate
            w = float(probs_np[idx])
            if aggregate is None:
                aggregate = w * cam_2d
            else:
                aggregate += w * cam_2d
            total_weight += w

        if aggregate is not None and total_weight > 0:
            aggregate = aggregate / total_weight
            # Normalize aggregate to [0, 1]
            mn, mx = aggregate.min(), aggregate.max()
            if mx - mn > 1e-8:
                aggregate = (aggregate - mn) / (mx - mn)
        elif aggregate is None:
            # No positive predictions: return zero map
            h, w = x.shape[-2], x.shape[-1]
            aggregate = np.zeros((h // 32, w // 32))  # approximate feature map size

        return {
            "per_label": per_label_maps,
            "aggregate": aggregate,
            "probs": probs_np,
        }

    def remove_hooks(self):
        """Clean up hooks when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class ViTGradCAM:
    """
    Grad-CAM for Vision Transformer models.

    ViT has no convolutional layers, so the standard GradCAM class cannot be
    used directly. Instead, we hook the last transformer block, whose output
    is a token sequence of shape (B, N_tokens, D) where N_tokens = 197 for
    ViT-S/16 (196 patch tokens + 1 CLS token).

    The trick: treat the patch token sequence as a spatial feature map by
    reshaping (B, 196, 384) → (B, 384, 14, 14). Then apply standard Grad-CAM:

        alpha_d  = mean over (H, W) of  d(y_k) / d(F_d)      [channel weight]
        CAM_k    = ReLU( sum_d alpha_d * F_d )                 [spatial map]

    This is equivalent to the formulation in Chefer et al. (2021) without the
    attention propagation step, which is simpler and sufficient for our purposes.

    Because the CLS token is not a spatial token, it is dropped before reshaping.

    Args:
        model:      ViTSmallBaseline (or any model with is_vit_model() == True)
        grid_size:  Number of patches per side (14 for ViT-S/16 at 224×224)
    """

    def __init__(self, model: nn.Module, grid_size: int = 14):
        self.model     = model
        self.grid_size = grid_size
        self.hidden_dim = None   # inferred from first forward pass

        self._activations = None   # (B, N_patches, D), CLS excluded
        self._gradients   = None   # (B, N_patches, D), CLS excluded

        target_layer = model.get_cam_target_layer()
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """
        Hook on the last transformer block.
        output shape: (B, N_tokens, D) where N_tokens = 197 (CLS + 196 patches)
        We drop CLS token (index 0) and save patch tokens only.
        """
        patch_tokens = output[:, 1:, :]           # (B, 196, D) — drop CLS
        self._activations = patch_tokens.detach()
        self.hidden_dim   = patch_tokens.shape[-1] # D = 384 for ViT-S

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Gradient hook — same structure as activation hook.
        grad_output[0] shape: (B, N_tokens, D)
        """
        patch_grad = grad_output[0][:, 1:, :]     # (B, 196, D) — drop CLS grad
        self._gradients = patch_grad.detach()

    def _to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, N_patches, D) → (B, D, grid_size, grid_size).
        This is the spatial feature map analogue for Grad-CAM.
        """
        B, N, D = tokens.shape
        assert N == self.grid_size ** 2, \
            f"Expected {self.grid_size**2} patch tokens, got {N}"
        # Rearrange: (B, N, D) → (B, D, H, W)
        return tokens.permute(0, 2, 1).reshape(B, D, self.grid_size, self.grid_size)

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int,
        relu: bool = True,
    ) -> np.ndarray:
        """
        Generate ViT Grad-CAM heatmap for a specific class.

        Args:
            x:         (1, 3, 224, 224) input tensor
            class_idx: Target class index
            relu:      Apply ReLU to the CAM (standard)

        Returns:
            cam: (1, grid_size, grid_size) numpy array in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)                    # (B, num_classes)
        self.model.zero_grad()
        logits[:, class_idx].sum().backward()

        # Reshape tokens → spatial maps
        act  = self._to_spatial(self._activations)  # (B, D, 14, 14)
        grad = self._to_spatial(self._gradients)    # (B, D, 14, 14)

        # Channel weights: global-average-pool the gradients
        weights = grad.mean(dim=(-2, -1), keepdim=True)  # (B, D, 1, 1)
        cam     = (weights * act).sum(dim=1)              # (B, 14, 14)

        if relu:
            cam = torch.clamp(cam, min=0)

        cam_np = cam.cpu().numpy()
        for b in range(cam_np.shape[0]):
            mn, mx = cam_np[b].min(), cam_np[b].max()
            if mx - mn > 1e-8:
                cam_np[b] = (cam_np[b] - mn) / (mx - mn)

        return cam_np  # (B, 14, 14)

    def generate_multilabel(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        labels: list = None,
    ) -> dict:
        """
        Generate one ViT Grad-CAM heatmap per predicted positive label,
        and a confidence-weighted aggregate heatmap.
        Interface identical to GradCAM.generate_multilabel().
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.sigmoid(logits).squeeze(0)  # (num_classes,)

        probs_np        = probs.cpu().numpy()
        positive_indices = np.where(probs_np >= threshold)[0]

        per_label_maps = {}
        aggregate      = None
        total_weight   = 0.0

        for idx in positive_indices:
            cam    = self.generate(x.clone(), class_idx=int(idx))  # (1, 14, 14)
            cam_2d = cam[0]                                         # (14, 14)

            key = labels[idx] if labels else str(idx)
            per_label_maps[key] = cam_2d

            w = float(probs_np[idx])
            if aggregate is None:
                aggregate = w * cam_2d
            else:
                aggregate += w * cam_2d
            total_weight += w

        if aggregate is not None and total_weight > 0:
            aggregate /= total_weight
            mn, mx = aggregate.min(), aggregate.max()
            if mx - mn > 1e-8:
                aggregate = (aggregate - mn) / (mx - mn)
        else:
            # No positive predictions → zero map at patch resolution
            aggregate = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        return {
            "per_label": per_label_maps,
            "aggregate": aggregate,
            "probs":     probs_np,
        }

    def remove_hooks(self):
        """Clean up hooks when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()