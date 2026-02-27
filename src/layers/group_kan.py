"""
GroupKAN Layer: Grouped Kolmogorov-Arnold Network.
Inspired by: li2025groupkan (GroupKAN paper) and the group convolution idea.

Core idea: Instead of applying KAN across ALL input-output pairs (which is
O(in * out) splines), split the input into G groups and apply a separate
KANLinear within each group. This reduces parameters by ~G× while keeping
the spline interpretability within each group.

This is analogous to grouped convolutions in CNNs (depthwise separable, etc.),
but applied to the KAN framework.

Architecture:
    input (in_features,)
      ↓
    [group 0: KANLinear(in//G → hidden//G)]
    [group 1: KANLinear(in//G → hidden//G)]  ← G parallel KAN layers
    [... ]
      ↓ concatenate
    (hidden_dim,)
      ↓
    KANLinear(hidden_dim → num_classes)    ← final KAN classifier
"""

import torch
import torch.nn as nn
from .efficient_kan import KANLinear


class GroupKANLinear(nn.Module):
    """
    Grouped KAN layer: splits input channels into G groups,
    applies an independent KANLinear to each group, concatenates.

    This uses the EXACT same B-spline KANLinear as VanillaKAN,
    just restricted to a subset of input features per group.

    Args:
        in_features:  Total input features
        out_features: Total output features
        num_groups:   Number of groups G (must divide both in and out evenly)
        grid_size:    B-spline grid size
        spline_order: B-spline polynomial degree
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int = 8,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()

        # Make sure groups divide evenly
        assert in_features % num_groups == 0, \
            f"in_features ({in_features}) must be divisible by num_groups ({num_groups})"
        assert out_features % num_groups == 0, \
            f"out_features ({out_features}) must be divisible by num_groups ({num_groups})"

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_in = in_features // num_groups
        self.group_out = out_features // num_groups

        # One KANLinear per group — each operates on a slice of features
        self.kan_groups = nn.ModuleList([
            KANLinear(
                in_features=self.group_in,
                out_features=self.group_out,
                grid_size=grid_size,
                spline_order=spline_order,
            )
            for _ in range(num_groups)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        # Split x into G chunks along feature dim
        chunks = x.chunk(self.num_groups, dim=-1)  # list of (batch, group_in)

        # Apply each group's KAN to its corresponding chunk
        out_chunks = [kan(chunk) for kan, chunk in zip(self.kan_groups, chunks)]

        # Concatenate back: (batch, out_features)
        return torch.cat(out_chunks, dim=-1)

    def get_group_importance(self, x: torch.Tensor, out_grad: torch.Tensor = None):
        """
        Intrinsic importance: gradient of group KAN outputs w.r.t. input.

        Args:
            x:        (batch, in_features) — input to this layer
            out_grad: (batch, out_features) — upstream gradient to weight outputs.
                      If None, treats all output channels equally.

        Returns:
            importance: (batch, in_features)
        """
        chunks = x.chunk(self.num_groups, dim=-1)
        importances = []

        for g, (kan, chunk) in enumerate(zip(self.kan_groups, chunks)):
            chunk = chunk.detach().requires_grad_(True)
            out = kan(chunk)  # (batch, group_out)

            if out_grad is not None:
                # Weight by upstream gradient for this group's output slice
                g_start = g * self.group_out
                g_end   = g_start + self.group_out
                target = (out * out_grad[:, g_start:g_end].detach()).sum()
            else:
                target = out.abs().sum()

            grad = torch.autograd.grad(target, chunk, create_graph=False)[0]
            importances.append(grad.abs())  # (batch, group_in)

        return torch.cat(importances, dim=-1)  # (batch, in_features)


class GroupKANHead(nn.Module):
    """
    Full GroupKAN classification head for ResNet.
    Replaces the standard FC head with a two-layer grouped KAN.

    Args:
        in_features:  ResNet output dim (2048 for ResNet-50)
        hidden_dim:   Intermediate dimension (must be divisible by num_groups)
        num_classes:  Number of labels
        num_groups:   Number of KAN groups
        grid_size:    B-spline grid size
        spline_order: B-spline degree
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 14,
        num_groups: int = 8,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()

        # Adjust hidden_dim to be divisible by num_groups
        hidden_dim = (hidden_dim // num_groups) * num_groups

        self.group_kan = GroupKANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            num_groups=num_groups,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        # Final KAN layer maps hidden → classes
        self.output_kan = KANLinear(
            in_features=hidden_dim,
            out_features=num_classes,
            grid_size=grid_size,
            spline_order=spline_order,
        )

        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        """
        x: (batch, in_features)
        returns: (batch, num_classes) logits
        """
        h = self.group_kan(x)   # (batch, hidden_dim)
        out = self.output_kan(h) # (batch, num_classes)
        return out

    def get_feature_importance(self, x: torch.Tensor, class_idx: int = None):
        """
        Intrinsic importance via chain rule through both KAN layers.

        Computes d(output[class_idx]) / d(x_i) for each input feature i.
        If class_idx is None, averages over all classes.

        Returns: (batch, in_features)
        """
        x = x.detach().requires_grad_(True)

        h = self.group_kan(x)   # (batch, hidden_dim)
        out = self.output_kan(h) # (batch, num_classes)

        if class_idx is not None:
            target = out[:, class_idx].sum()
        else:
            target = out.abs().sum()

        grad = torch.autograd.grad(target, x, create_graph=False)[0]
        return grad.abs()  # (batch, in_features)