"""
KAN-CAM: Spatial Class Activation Mapping via KAN Spline Projection.

This is the principled KAN-specific spatial interpretability method,
directly analogous to the original CAM paper (Zhou et al., 2016).

--- Motivation ---

CAM for GAP + linear classifier:
    CAM_k(x, y) = Σ_c  w_{k,c}  ×  F_c(x, y)
    where w_{k,c} is the linear classifier weight for class k, channel c.

KAN-CAM for GAP + KAN head:
    KAN_CAM_k(x, y) = Σ_c  φ_{k,c}( F_c(x, y) )
    where φ_{k,c} is the learned spline function connecting channel c to class k.

The key idea: instead of pooling first and then applying splines (the normal
forward pass), we APPLY THE SPLINES POINTWISE across spatial locations and
then pool/aggregate. This recovers a spatial map from the KAN's learned
functions without any backward pass.

For a two-layer KAN head (2048 → hidden → 14):
    1. At each spatial location (x,y): local_feat = F[:, :, x, y]  (C,)
    2. Through kan1 splines:           h_j = Σ_c  φ^1_{j,c}(local_feat_c)  (hidden,)
    3. Through kan2 splines:           out_k = Σ_j  φ^2_{k,j}(h_j)  (num_classes,)
    4. KAN-CAM_k(x, y) = out_k

--- Why this is better than SWAM ---

Current SWAM computes d(output_k)/d(GAP(F)) — the gradient at ONE point (the
globally averaged feature vector). It gives you channel importance weights
but cannot distinguish spatial locations within a channel.

KAN-CAM evaluates the splines at EVERY spatial location, giving a genuine
2D map. It exploits the fact that φ_{k,c} is a function (not just a scalar
derivative) — so asking "what does the KAN think about the local activation
strength at position (x,y)?" is a valid, well-defined operation.

--- Caveat (honest limitation) ---

The KAN splines were fitted on globally-averaged features. Local feature
values at individual spatial positions have a wider distribution than GAP'd
values. B-spline clamping (in efficient_kan.py) handles out-of-range inputs
gracefully. This is the same limitation Grad-CAM has: gradients are computed
at whatever activations occur at inference time.

--- Comparison summary ---

Method      | Spatially aware | KAN-specific | Backward pass | Class-specific
------------|-----------------|--------------|---------------|---------------
Grad-CAM    | Yes             | No           | Yes           | Yes
SWAM        | No (post-GAP)   | Partial      | Yes (autograd)| Yes
KAN-CAM     | Yes             | Yes          | No            | Yes

Reference:
    Zhou et al. (2016) Learning Deep Features for Discriminative Localization.
    (CAM, the linear special case this generalises.)
"""

import torch
import torch.nn as nn
import numpy as np


class KANCAM:
    """
    KAN-CAM: Spatial projection of KAN spline functions for class localization.

    Supports:
      - ResNet50VanillaKAN  (two KANLinear layers: kan1 → kan2)
      - ResNet50GroupKAN    (GroupKANHead: group_kan → output_kan)
      - ResNet50RationalKAN (RationalKANLinear: layer1 → rkan1 → layer2)

    Not applicable to:
      - ResNet50Baseline    (no KAN layers)
      - ResNet50ConvKAN     (KAN is spatial already; use SpatialSWAM instead)
    """

    def __init__(self, model: nn.Module):
        if not model.is_kan_model():
            raise ValueError(
                f"KAN-CAM requires a KAN model, got {type(model).__name__}. "
                f"Use GradCAM for the baseline."
            )
        if hasattr(model, 'is_conv_kan') and model.is_conv_kan():
            raise ValueError(
                "KAN-CAM is not applicable to ResNet50ConvKAN. "
                "Use SpatialSWAM for that model instead."
            )
        self.model = model
        self.model_type = type(model).__name__
        print(f"[KAN-CAM] Initialized for {self.model_type}")

    def _forward_kan_head_spatial(
        self,
        spatial_flat: torch.Tensor,
        class_idx: int,
    ) -> torch.Tensor:
        """
        Run the KAN head on (H*W, C) local features and return
        the class_idx output at each spatial location.

        Args:
            spatial_flat: (H*W, C) — each row is one spatial location's feature vector
            class_idx:    which class to compute

        Returns:
            out: (H*W,) — KAN head output for class_idx at each location
        """
        # Route based on model architecture
        if hasattr(self.model, 'kan1') and hasattr(self.model, 'kan2'):
            # ResNet50VanillaKAN: two explicit KANLinear layers
            h   = self.model.kan1(spatial_flat)   # (H*W, hidden_dim)
            out = self.model.kan2(h)               # (H*W, num_classes)

        elif hasattr(self.model, 'head'):
            head = self.model.head

            if hasattr(head, 'group_kan') and hasattr(head, 'output_kan'):
                # ResNet50GroupKAN: GroupKANHead
                h   = head.group_kan(spatial_flat)   # (H*W, hidden_dim)
                out = head.output_kan(h)              # (H*W, num_classes)

            elif hasattr(head, 'layer1') and hasattr(head, 'rkan1') and hasattr(head, 'layer2'):
                # ResNet50RationalKAN: linear → Jacobi activation → linear
                h     = head.layer1(spatial_flat)    # (H*W, hidden_dim)
                h_act = head.rkan1(h)                # (H*W, hidden_dim)
                out   = head.layer2(h_act)           # (H*W, num_classes)

            else:
                raise RuntimeError(
                    f"KAN-CAM: unrecognised head structure in {self.model_type}. "
                    f"Head has attrs: {list(vars(head).keys())}"
                )
        else:
            raise RuntimeError(
                f"KAN-CAM: cannot find KAN layers in {self.model_type}."
            )

        return out[:, class_idx]  # (H*W,)

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int,
        relu: bool = True,
    ) -> np.ndarray:
        """
        Generate KAN-CAM for a specific class.

        Args:
            x:         (1, 3, H, W) — single preprocessed image on device
            class_idx: class to explain (0-indexed)
            relu:      apply ReLU (keep only positive evidence)

        Returns:
            cam: (H_feat, W_feat) float32 numpy array, normalized to [0, 1]
                 e.g. (7, 7) for ResNet-50 with 224×224 input
        """
        self.model.eval()

        with torch.no_grad():
            # Step 1: extract spatial feature map F — (1, C, H_feat, W_feat)
            spatial = self.model.get_features(x)
            _, C, H_feat, W_feat = spatial.shape

            # Step 2: reshape to (H*W, C) — each row = one spatial location
            # This is the key step: we treat every spatial location as a
            # "pseudo-sample" and evaluate the KAN head on it
            spatial_flat = spatial[0].permute(1, 2, 0).reshape(H_feat * W_feat, C)
            # spatial[0]: (C, H, W) → permute → (H, W, C) → reshape → (H*W, C)

            # Step 3: forward through KAN head at each spatial location
            # out_spatial: (H*W,) — KAN's class_idx output for each location
            out_spatial = self._forward_kan_head_spatial(spatial_flat, class_idx)

            # Step 4: reshape back to spatial (H_feat, W_feat)
            cam = out_spatial.reshape(H_feat, W_feat)

            # Step 5: ReLU — only retain locations where KAN predicts positive
            # contribution to this class
            if relu:
                cam = torch.clamp(cam, min=0)

        cam_np = cam.cpu().float().numpy()

        # Normalize to [0, 1]
        mn, mx = cam_np.min(), cam_np.max()
        if mx - mn > 1e-8:
            cam_np = (cam_np - mn) / (mx - mn)

        return cam_np  # (H_feat, W_feat)

    def generate_multilabel(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        labels: list = None,
    ) -> dict:
        """
        Generate KAN-CAM for all predicted positive classes,
        plus a confidence-weighted aggregate map.

        Args:
            x:         (1, 3, H, W) single image
            threshold: sigmoid probability threshold for 'positive'
            labels:    label name list (for dict keys)

        Returns:
            dict:
              'aggregate': (H_feat, W_feat) confidence-weighted blend
              'per_label': {label_name: (H_feat, W_feat)}
              'probs':     (num_classes,) numpy probabilities
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        positive_idxs = list(np.where(probs >= threshold)[0])
        if len(positive_idxs) == 0:
            # Nothing above threshold: explain the most confident class
            positive_idxs = [int(np.argmax(probs))]

        per_label_maps = {}
        aggregate      = None
        total_weight   = 0.0

        for idx in positive_idxs:
            cam_np = self.generate(x, class_idx=int(idx))
            key    = labels[idx] if labels is not None else str(idx)
            per_label_maps[key] = cam_np

            conf = float(probs[idx])
            if aggregate is None:
                aggregate = conf * cam_np
            else:
                aggregate += conf * cam_np
            total_weight += conf

        if total_weight > 0:
            aggregate /= total_weight

        # Normalize aggregate
        mn, mx = aggregate.min(), aggregate.max()
        if mx - mn > 1e-8:
            aggregate = (aggregate - mn) / (mx - mn)

        return {
            "aggregate": aggregate,
            "per_label": per_label_maps,
            "probs":     probs,
        }