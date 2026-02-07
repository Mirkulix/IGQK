"""
Ternary Lie Group Theory (TLGT).

Proposition 6.2: The ternary Lie group G₃ = {-1, 0, +1}^n is a discrete
subgroup of the quantum symmetry group on M.

G₃ acts on the weight space via:
    g · θ = diag(g) θ    where g ∈ {-1, 0, +1}^n

This provides a group-theoretic foundation for ternary compression.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class TernaryLieGroup:
    """
    Ternary Lie Group G₃ for structured weight compression.

    The group structure enables:
    - Invariant compression: preserving symmetries
    - Coset decomposition: efficient representation
    - Group convolutions: structured weight sharing
    """

    def __init__(self, dim: int, scaling: str = "adaptive"):
        """
        Args:
            dim: Dimension of weight space.
            scaling: Scaling strategy ('adaptive', 'fixed', 'per_channel').
        """
        self.dim = dim
        self.scaling = scaling

    def quantize(
        self,
        weights: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project weights onto ternary Lie group G₃ = {-α, 0, +α}.

        Uses optimal scaling factor α to minimize distortion:
            α* = argmin_α ||W - α·Q||²  where Q ∈ {-1,0,+1}

        Args:
            weights: Continuous weight tensor.
            scale: Optional pre-computed scale factor.

        Returns:
            (ternary_weights, scale_factors)
        """
        original_shape = weights.shape
        flat = weights.flatten()

        if scale is None:
            scale = self._compute_optimal_scale(flat)

        # Ternary quantization with optimal thresholds
        threshold = 0.7 * scale  # ~optimal threshold for Gaussian distribution
        ternary = torch.zeros_like(flat)
        ternary[flat > threshold] = 1.0
        ternary[flat < -threshold] = -1.0

        return (ternary * scale).view(original_shape), scale

    def _compute_optimal_scale(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal scale factor α* = E[|w| : w ≠ 0] (for non-zero entries).

        Based on minimizing E[(w - α·sign(w))²].
        """
        mask = weights.abs() > 0.7 * weights.std()
        if mask.sum() > 0:
            return weights[mask].abs().mean()
        return weights.abs().mean()

    def group_action(
        self,
        g: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply group element g ∈ G₃ to weights: g · W = diag(g) @ W.

        Args:
            g: Group element (ternary vector).
            weights: Weight matrix.

        Returns:
            Transformed weights.
        """
        if weights.dim() == 2:
            return g.unsqueeze(1) * weights
        return g * weights

    def coset_decomposition(
        self,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose weights into ternary coset representatives.

        W = α·Q + R where Q ∈ G₃, R is residual.

        Returns:
            (ternary_Q, scale_alpha, residual_R)
        """
        ternary, scale = self.quantize(weights)
        residual = weights - ternary
        q = torch.zeros_like(weights)
        q[ternary > 0] = 1.0
        q[ternary < 0] = -1.0
        return q, scale, residual

    def invariant_compression(
        self,
        weights: torch.Tensor,
        num_iterations: int = 5,
    ) -> torch.Tensor:
        """
        Iterative ternary compression preserving group invariants.

        Alternates between ternary quantization and residual correction.
        """
        result = torch.zeros_like(weights)
        residual = weights.clone()

        for _ in range(num_iterations):
            q, scale, new_residual = self.coset_decomposition(residual)
            result = result + scale * q
            residual = new_residual
            if residual.abs().max() < 1e-6:
                break

        return result

    def compression_stats(self, original: torch.Tensor, compressed: torch.Tensor) -> dict:
        """Compute compression statistics."""
        unique_vals = compressed.unique()
        distortion = torch.norm(original - compressed).item() ** 2
        relative_error = distortion / (torch.norm(original).item() ** 2 + 1e-10)
        sparsity = (compressed == 0).float().mean().item()

        # Bits per weight: ternary = log2(3) ≈ 1.58 bits
        bits_per_weight = np.log2(len(unique_vals)) if len(unique_vals) > 1 else 0
        compression_ratio = bits_per_weight / 32.0

        return {
            "distortion": distortion,
            "relative_error": relative_error,
            "sparsity": sparsity,
            "unique_values": len(unique_vals),
            "bits_per_weight": bits_per_weight,
            "compression_ratio": compression_ratio,
        }
