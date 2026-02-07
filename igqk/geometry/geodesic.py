"""
Geodesic computation on statistical manifolds.

Geodesics are curves γ(t) satisfying:
    d²γ^k/dt² + Γ^k_ij (dγ^i/dt)(dγ^j/dt) = 0
"""

import torch
from typing import Optional, Callable


class GeodesicSolver:
    """Geodesic computation on Riemannian manifolds with Fisher metric."""

    def __init__(self, metric_fn: Optional[Callable] = None):
        """
        Args:
            metric_fn: Function θ -> G(θ) returning the metric tensor.
                       If None, uses Euclidean (identity) metric.
        """
        self.metric_fn = metric_fn

    def exponential_map(
        self,
        point: torch.Tensor,
        tangent: torch.Tensor,
        t: float = 1.0,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Exponential map exp_p(t·v): follow geodesic from point in direction tangent.

        Uses numerical integration (Euler) of geodesic equation.
        """
        if self.metric_fn is None:
            return point + t * tangent

        dt_step = t / num_steps
        pos = point.clone()
        vel = tangent.clone()

        for _ in range(num_steps):
            christoffel = self._christoffel_symbols(pos)
            acc = -torch.einsum("kij,i,j->k", christoffel, vel, vel)
            pos = pos + dt_step * vel
            vel = vel + dt_step * acc

        return pos

    def logarithmic_map(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_iterations: int = 20,
    ) -> torch.Tensor:
        """Logarithmic map log_p(q): find tangent vector v such that exp_p(v) = q."""
        if self.metric_fn is None:
            return end - start

        v = end - start
        for _ in range(num_iterations):
            q_approx = self.exponential_map(start, v)
            error = end - q_approx
            v = v + 0.5 * error
        return v

    def geodesic_path(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_points: int = 20,
    ) -> torch.Tensor:
        """Compute geodesic path between two points as [num_points, dim] tensor."""
        v = self.logarithmic_map(start, end)
        t_values = torch.linspace(0, 1, num_points, device=start.device)
        path = torch.stack([self.exponential_map(start, v, t=t.item()) for t in t_values])
        return path

    def geodesic_distance(self, start: torch.Tensor, end: torch.Tensor) -> float:
        """Compute geodesic distance d(start, end) = ||log_start(end)||_G."""
        v = self.logarithmic_map(start, end)

        if self.metric_fn is not None:
            G = self.metric_fn(start)
            dist_sq = torch.dot(v, torch.mv(G, v))
            return torch.sqrt(torch.clamp(dist_sq, min=0)).item()

        return torch.norm(v).item()

    def parallel_transport(
        self,
        vector: torch.Tensor,
        along_curve: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel transport a vector along a discrete curve [num_points, dim]."""
        if self.metric_fn is None:
            return vector

        v = vector.clone()
        for i in range(len(along_curve) - 1):
            p = along_curve[i]
            dp = along_curve[i + 1] - p
            christoffel = self._christoffel_symbols(p)
            dv = -torch.einsum("kij,i,j->k", christoffel, v, dp)
            v = v + dv
        return v

    def _christoffel_symbols(self, point: torch.Tensor) -> torch.Tensor:
        """Compute Christoffel symbols Γ^k_ij via numerical differentiation."""
        dim = point.shape[0]
        eps = 1e-4
        G = self.metric_fn(point)

        dG = torch.zeros(dim, dim, dim, device=point.device)
        for m in range(dim):
            e_m = torch.zeros(dim, device=point.device)
            e_m[m] = eps
            G_plus = self.metric_fn(point + e_m)
            G_minus = self.metric_fn(point - e_m)
            dG[m] = (G_plus - G_minus) / (2 * eps)

        G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(dim, device=point.device))

        gamma = torch.zeros(dim, dim, dim, device=point.device)
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    val = 0.5 * (dG[j, i, :] + dG[i, j, :] - dG[:, i, j])
                    gamma[k, i, j] = torch.dot(G_inv[k], val)
        return gamma
