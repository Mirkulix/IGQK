"""
Laplace-Beltrami operator on statistical manifolds.

    Δ_M f = (1/√g) ∂_i(√g g^{ij} ∂_j f)

For IGQK, H = -Δ_M acts as the Hamiltonian for quantum gradient flow.
"""

import torch
from typing import Optional, Callable


class LaplaceBeltrami:
    """Laplace-Beltrami operator for quantum gradient flow Hamiltonian."""

    def __init__(self, metric_fn: Optional[Callable] = None, eps: float = 1e-4):
        self.metric_fn = metric_fn
        self.eps = eps

    def apply(
        self,
        f: torch.Tensor,
        point: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Δ_M to function values f at point θ.

        For gradient vectors, returns G^{-1} f as approximation.
        """
        dim = point.shape[0]
        device = point.device

        if metric is None:
            if self.metric_fn is not None:
                metric = self.metric_fn(point)
            else:
                metric = torch.eye(dim, device=device)

        G_inv = torch.linalg.inv(metric + 1e-6 * torch.eye(dim, device=device))
        return torch.mv(G_inv, f)

    def hamiltonian(
        self,
        dim: int,
        metric: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Construct Hamiltonian H = -Δ_M as matrix operator.

        For quantum gradient flow: dρ/dt = -i[H, ρ] - γ{G^{-1}∇L, ρ}
        """
        if device is None:
            device = metric.device if metric is not None else torch.device("cpu")

        if metric is None:
            return -torch.eye(dim, device=device)

        reg = 1e-4 * torch.eye(dim, device=device)
        return -torch.linalg.inv(metric + reg)

    def fractional(
        self,
        alpha: float,
        dim: int,
        metric: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Fractional Laplace-Beltrami: (-Δ_M)^α.

        Used in FCHL (Fractional Calculus for Hebbian Learning).
        """
        H = self.hamiltonian(dim, metric, device)
        eigenvalues, eigenvectors = torch.linalg.eigh(-H)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        frac_eigenvalues = eigenvalues.pow(alpha)
        return eigenvectors @ torch.diag(frac_eigenvalues) @ eigenvectors.T
