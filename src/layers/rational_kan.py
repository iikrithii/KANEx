"""
Rational KAN Layer wrapper around rKAN's JacobiRKAN.
Source: https://github.com/alirezaafzalaghaei/rKAN (BSD-3-Clause)

Instead of B-splines, uses Jacobi orthogonal polynomials as basis functions.
These are rational/polynomial activations: non-polynomial behavior,
works well for both positive and negative inputs, faster than B-splines.

Install: pip install rkan
"""

import torch
import torch.nn as nn

try:
    from rkan.torch import JacobiRKAN
    RKAN_AVAILABLE = True
except ImportError:
    RKAN_AVAILABLE = False
    print("WARNING: rKAN not installed. Run: pip install rkan")
    print("Falling back to a manual Jacobi polynomial implementation.")


class JacobiPolynomialActivation(nn.Module):
    """
    Fallback: Jacobi polynomial activation if rkan not installed.
    Computes learnable linear combinations of Jacobi polynomials P_n^(alpha,beta)(x).
    This is mathematically equivalent to what rKAN's JacobiRKAN does.
    """
    def __init__(self, degree: int = 3, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        # Learnable coefficient for each polynomial degree
        self.coeffs = nn.Parameter(torch.randn(degree + 1) * 0.01)

    def jacobi_polynomials(self, x: torch.Tensor):
        """Compute Jacobi polynomials up to self.degree via recurrence."""
        # Clamp x to [-1, 1] for stability
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
        polys = [torch.ones_like(x)]  # P_0 = 1
        if self.degree > 0:
            # P_1^(a,b)(x) = 0.5*(a-b) + (1 + 0.5*(a+b))*x
            p1 = 0.5 * (self.alpha - self.beta) + (1 + 0.5 * (self.alpha + self.beta)) * x
            polys.append(p1)
        for n in range(1, self.degree):
            # Three-term recurrence for Jacobi polynomials
            a, b = self.alpha, self.beta
            n1 = n + 1
            c1 = 2 * n1 * (n1 + a + b) * (2 * n1 + a + b - 2)
            c2 = (2 * n1 + a + b - 1) * (a**2 - b**2)
            c3 = (2 * n1 + a + b - 1) * (2 * n1 + a + b) * (2 * n1 + a + b - 2)
            c4 = 2 * (n + a) * (n + b) * (2 * n1 + a + b)
            pn = ((c2 + c3 * x) * polys[-1] - c4 * polys[-2]) / c1
            polys.append(pn)
        return torch.stack(polys, dim=-1)  # (..., degree+1)

    def forward(self, x):
        polys = self.jacobi_polynomials(x)  # (..., degree+1)
        return (polys * self.coeffs).sum(dim=-1)


class RationalKANLinear(nn.Module):
    """
    Rational KAN head for classification.
    Uses JacobiRKAN (from rkan library) as the activation on each edge.

    Architecture for ResNet head:
        features (2048,) -> RationalKAN layer(s) -> logits (num_classes,)

    The JacobiRKAN layer replaces each linear weight with a learnable
    Jacobi polynomial activation — this is the 'rational' variant
    because rational functions can approximate a wider class of functions
    than B-splines and have better stability.

    Args:
        in_features:  Input feature dimension (2048 for ResNet-50)
        hidden_dim:   Hidden dimension of intermediate KAN layer
        num_classes:  Number of output classes
        degree:       Degree of Jacobi polynomial (default 3)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        degree: int = 3,
    ):
        super().__init__()

        if RKAN_AVAILABLE:
            # Use the official rKAN JacobiRKAN layer
            # It acts as an activation module placed between linear projections
            self.layer1 = nn.Linear(in_features, hidden_dim)
            self.rkan1 = JacobiRKAN(degree)   # Jacobi activation of given degree
            self.layer2 = nn.Linear(hidden_dim, num_classes)
            print(f"[RationalKAN] Using official rKAN JacobiRKAN (degree={degree})")
        else:
            # Fallback with our manual Jacobi implementation
            self.layer1 = nn.Linear(in_features, hidden_dim)
            self.rkan1 = JacobiPolynomialActivation(degree=degree)
            self.layer2 = nn.Linear(hidden_dim, num_classes)
            print(f"[RationalKAN] Using fallback Jacobi polynomial activation (degree={degree})")

        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, x):
        """
        x: (batch, in_features) — flattened ResNet features after global pool
        returns: (batch, num_classes) logits
        """
        x = self.layer1(x)   # (batch, hidden_dim)
        x = self.rkan1(x)    # Jacobi polynomial activation on each element
        x = self.layer2(x)   # (batch, num_classes)
        return x

    def get_feature_importance(self, x: torch.Tensor, class_idx: int = None):
        """
        Intrinsic interpretability: gradient of class output w.r.t. input features.

        Args:
            x:         (batch, in_features) — pooled ResNet features
            class_idx: If given, returns gradient for that class only.
                       If None, returns mean abs gradient across all classes.

        Returns:
            importance: (batch, in_features) — absolute gradient magnitudes
        """
        x = x.detach().requires_grad_(True)
        h = self.layer1(x)
        h_act = self.rkan1(h)
        out = self.layer2(h_act)  # (batch, num_classes)

        if class_idx is not None:
            target = out[:, class_idx].sum()
        else:
            target = out.abs().sum()

        grad = torch.autograd.grad(target, x, create_graph=False)[0]
        return grad.abs()  # (batch, in_features)