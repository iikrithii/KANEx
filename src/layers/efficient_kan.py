"""
Efficient KAN Linear Layer (B-spline based).
Source: https://github.com/Blealtan/efficient-kan (MIT License)
This is the exact B-spline implementation — no approximations.
Each edge has a learnable univariate spline function phi_{i,j}.
"""

import torch
import torch.nn as nn
import math


class KANLinear(nn.Module):
    """
    A single KAN layer where every weight is replaced by a learnable
    univariate B-spline function. This is the exact KAN formulation.

    For in_features inputs and out_features outputs:
      - We have in_features * out_features spline functions
      - Each spline is parameterized by (grid_size + spline_order) B-spline coefficients
      - Plus a base (SiLU) residual connection for stability

    Args:
        in_features:  Number of input dimensions
        out_features: Number of output dimensions
        grid_size:    Number of B-spline grid intervals (more = more expressive)
        spline_order: Degree of B-spline (3 = cubic, standard)
        scale_noise:  Small noise for spline init (prevents dead splines)
        scale_base:   Scale for residual SiLU connection
        scale_spline: Scale for spline activation
        grid_eps:     0=percentile-based grid, 1=uniform grid
        grid_range:   Range of the input grid
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build the uniform B-spline grid: (spline_order + grid_size + 1) knot points
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Learnable B-spline coefficients: shape (out_features, in_features, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # Separate learnable scale per spline (improves training)
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # Residual linear (base) weight for SiLU branch
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for the base (SiLU) linear branch
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # Small noise around zero for spline coefficients
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                self.scale_spline
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input x.
        This is the EXACT Cox-de Boor recursion — no approximation.

        Args:
            x: (batch_size, in_features)
        Returns:
            bases: (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # x: (batch, in_features) -> (batch, in_features, 1)
        # grid: (in_features, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)

        # Ensure grid is on same device and dtype as x (protects against
        # mixed-precision or multi-device edge cases)
        grid = self.grid.unsqueeze(0).to(device=x.device, dtype=x.dtype)  # (1, in_features, knots)

        # Clamp x to [grid_min, grid_max - eps] so that values exactly equal
        # to the last knot still fall inside a valid interval. Without this,
        # the strict < comparison below leaves them with no active basis,
        # producing zero output for that sample.
        x = x.clamp(grid[:, :, :1], grid[:, :, -1:] - 1e-6)

        # Order-0 B-splines: 1 if knot[i] <= x < knot[i+1]
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        # Cox-de Boor recursion up to desired order
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :, : -(k + 1)])
                / (grid[:, :, k:-1] - grid[:, :, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, :, k + 1 :] - x)
                / (grid[:, :, k + 1 :] - grid[:, :, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute B-spline coefficients that interpolate the given (x, y) pairs.
        Used for initialization only.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch, n_basis)
        B = y.transpose(0, 1)                   # (in_features, batch, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, n_basis, out_features)

        result = solution.permute(2, 0, 1)  # (out_features, in_features, n_basis)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """Apply per-spline scale if enabled."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass: output = base_branch(x) + spline_branch(x)
          - base_branch:   linear(SiLU(x))
          - spline_branch: sum of learnable B-spline functions
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base (SiLU residual) branch
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)

        # Spline branch: compute B-spline basis, then linear combination
        spline_output = nn.functional.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def get_spline_derivatives(self, x: torch.Tensor):
        """
        Compute analytic derivative of each spline function phi_{o,i} at x.

        phi_{o,i}(x_i) = sum_k  coeff[o, i, k] * B_k(x_i)
        d phi_{o,i} / d x_i  = autograd through the B-spline basis evaluation

        Since phi_{o,i} depends only on x_i (not on x_j for j≠i), the full
        Jacobian is sparse: grad[b, o, i] = d phi_{o,i} / d x_i.

        Args:
            x: (batch, in_features)
        Returns:
            derivs: (batch, out_features, in_features)
                    derivs[b, o, i] = d phi_{o,i}(x_i) / d x_i
        """
        x = x.detach().clone().requires_grad_(True)  # leaf tensor for grad

        # bases: (batch, in_features, n_basis)
        bases = self.b_splines(x)

        # spline_per_out_in[b, o, i] = phi_{o,i}(x_i) = sum_k coeff[o,i,k] * B_k(x_i)
        # scaled_spline_weight: (out_features, in_features, n_basis)
        spline_per_out_in = torch.einsum(
            "bik,oik->boi", bases, self.scaled_spline_weight
        )  # (batch, out_features, in_features)

        batch   = x.shape[0]
        out_f   = self.out_features
        in_f    = self.in_features
        derivs  = torch.zeros(batch, out_f, in_f, device=x.device, dtype=x.dtype)

        for o in range(out_f):
            # Sum over batch AND over inputs so we get a scalar, then grad.
            # Because phi_{o,i} depends only on x_i, the gradient w.r.t. x[b,i]
            # equals d phi_{o,i}(x_i)/dx_i for each (b, i) pair.
            target = spline_per_out_in[:, o, :].sum()
            grad = torch.autograd.grad(
                target, x,
                create_graph=False,
                retain_graph=(o < out_f - 1),  # keep graph until last output
            )[0]  # (batch, in_features)
            derivs[:, o, :] = grad

        return derivs  # (batch, out_features, in_features)