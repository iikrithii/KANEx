"""
Gradient-Weighted Attention Rollout for ViT models.

Plain attention rollout (Abnar & Zuidema, 2020) recursively multiplies attention
matrices across all transformer layers to account for residual connections. It
produces a spatial map showing where the CLS token attends across the image.

However plain rollout is class-agnostic — it doesn't know which class you're
explaining. We add gradient weighting: before rolling out, we weight each
attention head at each layer by the gradient of the target class logit w.r.t.
that head's attention output. This is the method from Chefer et al. (2021)
"Transformer Interpretability Beyond Attention Visualization".

Algorithm for class k:
    For each block l = 1..L:
        A_l   = attention weights (B, heads, tokens, tokens)
        G_l   = gradient of y_k w.r.t. A_l
        R_l   = mean_heads( ReLU(A_l * G_l) )    [class-discriminative]
        R_l   = R_l + I                           [residual connection]
        R_l   = R_l / R_l.sum(dim=-1)            [row-normalize]
    Rollout = R_1 @ R_2 @ ... @ R_L
    Map     = Rollout[0, 1:] reshaped to (14, 14)

For multi-label: one rollout per positive class, confidence-weighted aggregate.
Same interface as GradCAM and ViTGradCAM.

Only applicable to ViT models (is_vit_model() == True).
"""

import torch
import torch.nn as nn
import numpy as np


class AttentionRollout:
    """
    Gradient-Weighted Attention Rollout for ViT-Small/16.

    Hooks the attention dropout layer in every transformer block to capture
    attention weight matrices (forward) and their gradients (backward).

    Args:
        model:         Any ViT model exposing get_attention_blocks()
        grid_size:     Patch grid side length (14 for ViT-S/16 at 224x224)
        discard_ratio: Zero out bottom-k fraction of attention values before
                       rollout. Sharpens the map. 0.9 works well empirically.
    """

    def __init__(self, model: nn.Module, grid_size: int = 14, discard_ratio: float = 0.9):
        if not (hasattr(model, "is_vit_model") and model.is_vit_model()):
            raise ValueError(
                f"AttentionRollout requires a ViT model, got {type(model).__name__}."
            )
        self.model         = model
        self.grid_size     = grid_size
        self.discard_ratio = discard_ratio
        self.n_patches     = grid_size * grid_size   # 196

        self._attentions: list = []
        self._gradients:  list = []
        self._hooks:      list = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Hook the attn_drop (Dropout) module inside each transformer block's
        attention sub-module.

        timm's Attention.forward() flow:
            qkv → q,k,v → attn = softmax(q@k) → attn_drop(attn) → attn @ v

        Hooking attn_drop gives us the attention tensor (B, heads, N, N) as
        the *input* to the dropout, before any values are zeroed. This is
        exactly what we want for rollout.
        """
        blocks = self.model.get_attention_blocks()
        for block in blocks:
            attn_module = block.attn
            fwd = attn_module.attn_drop.register_forward_hook(self._fwd_hook())
            bwd = attn_module.attn_drop.register_full_backward_hook(self._bwd_hook())
            self._hooks.extend([fwd, bwd])

    def _fwd_hook(self):
        def hook(module, inp, out):
            # inp[0]: (B, heads, tokens, tokens) — attention weights before dropout
            self._attentions.append(inp[0].detach())
        return hook

    def _bwd_hook(self):
        def hook(module, grad_in, grad_out):
            # grad_out[0]: gradient w.r.t. the dropout input = attn weights
            g = grad_out[0]
            self._gradients.append(g.detach() if g is not None else None)
        return hook

    def _clear(self):
        self._attentions = []
        self._gradients  = []

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Gradient-weighted attention rollout for one class.

        Args:
            x:         (1, 3, 224, 224) image tensor on the right device
            class_idx: Target class

        Returns:
            (grid_size, grid_size) float32 numpy array in [0, 1]
        """
        self._clear()
        self.model.eval()
        x = x.requires_grad_(True)

        # Forward — hooks collect attention matrices per block
        logits = self.model(x)
        self.model.zero_grad()
        # Backward — hooks collect gradients
        logits[:, class_idx].sum().backward()

        if not self._attentions:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # PyTorch collects backward hooks in reverse layer order, re-reverse
        grads = list(reversed(self._gradients))

        # Safety: align lengths
        n       = min(len(self._attentions), len(grads))
        atts    = self._attentions[:n]
        grads   = grads[:n]

        n_tokens = atts[0].shape[-1]  # 197 for ViT-S/16
        rollout  = torch.eye(n_tokens, device=x.device)

        for A, G in zip(atts, grads):
            A = A.squeeze(0)  # (heads, 197, 197)
            G = G.squeeze(0) if G is not None else torch.zeros_like(A)

            # Gradient-weighted, only positive contributions
            cam = torch.clamp(A * G, min=0).mean(dim=0)  # (197, 197)

            # Discard low-value attention to sharpen
            if self.discard_ratio > 0.0:
                thresh = torch.quantile(cam.flatten(), self.discard_ratio)
                cam    = torch.where(cam >= thresh, cam, torch.zeros_like(cam))

            # Residual + row-normalize
            cam     = cam + torch.eye(n_tokens, device=cam.device)
            cam     = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
            rollout = cam @ rollout

        # CLS-to-patches row: row 0, columns 1: (196 patches)
        spatial = rollout[0, 1:].reshape(self.grid_size, self.grid_size)
        cam_np  = spatial.cpu().float().numpy()

        mn, mx = cam_np.min(), cam_np.max()
        if mx - mn > 1e-8:
            cam_np = (cam_np - mn) / (mx - mn)

        return cam_np   # (14, 14)

    def generate_multilabel(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        labels: list = None,
    ) -> dict:
        """
        Gradient-weighted attention rollout for all predicted positive classes,
        plus a confidence-weighted aggregate map.

        Same return interface as GradCAM.generate_multilabel() and ViTGradCAM.

        Returns:
            dict:
              'aggregate': (grid_size, grid_size) float32 in [0, 1]
              'per_label': {label_name: (grid_size, grid_size)}
              'probs':     (num_classes,) numpy probabilities
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        positive_idxs = list(np.where(probs >= threshold)[0])
        if not positive_idxs:
            positive_idxs = [int(np.argmax(probs))]

        per_label_maps = {}
        aggregate      = None
        total_weight   = 0.0

        for idx in positive_idxs:
            cam_np = self.generate(x.clone(), class_idx=int(idx))
            key    = labels[idx] if labels is not None else str(idx)
            per_label_maps[key] = cam_np

            conf      = float(probs[idx])
            aggregate = conf * cam_np if aggregate is None else aggregate + conf * cam_np
            total_weight += conf

        if total_weight > 0:
            aggregate /= total_weight

        mn, mx = aggregate.min(), aggregate.max()
        if mx - mn > 1e-8:
            aggregate = (aggregate - mn) / (mx - mn)

        return {"aggregate": aggregate, "per_label": per_label_maps, "probs": probs}

    def remove_hooks(self):
        """Remove all registered hooks. Always call when done."""
        for h in self._hooks:
            h.remove()
        self._hooks = []