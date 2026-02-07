"""
Fractional Calculus for Hebbian Learning (FCHL).

Proposition 6.3: FCHL uses the fractional Laplace-Beltrami operator
    H = -(-Δ_M)^α    (0 < α ≤ 1)

The fractional derivative enables long-range memory in the quantum
gradient flow, mimicking Hebbian learning with temporal correlations.

Caputo fractional derivative:
    D^α f(t) = (1/Γ(n-α)) ∫₀ᵗ (t-τ)^{n-α-1} f^{(n)}(τ) dτ
"""

import torch
import numpy as np
from typing import Optional, List
from scipy.special import gamma as gamma_fn


class FractionalHebbianLearning:
    """
    Fractional Hebbian Learning with memory-augmented gradient flow.

    Uses fractional derivatives to incorporate learning history,
    enabling better exploration of the loss landscape.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        memory_length: int = 50,
        hebbian_lr: float = 0.01,
    ):
        """
        Args:
            alpha: Fractional order (0 < α ≤ 1). Lower = more memory.
            memory_length: Number of past gradients to store.
            hebbian_lr: Hebbian learning rate.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.memory_length = memory_length
        self.hebbian_lr = hebbian_lr
        self._gradient_history: List[torch.Tensor] = []
        self._grunwald_weights: Optional[torch.Tensor] = None

    def fractional_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute fractional gradient using Grünwald-Letnikov approximation.

        D^α f ≈ (1/h^α) Σ_k (-1)^k C(α,k) f(t - k·h)

        where C(α,k) = α(α-1)...(α-k+1) / k!

        Args:
            gradient: Current gradient ∇L.

        Returns:
            Fractional gradient D^α(∇L).
        """
        self._gradient_history.append(gradient.detach().clone())
        if len(self._gradient_history) > self.memory_length:
            self._gradient_history.pop(0)

        n = len(self._gradient_history)
        weights = self._grunwald_letnikov_weights(n, gradient.device)

        frac_grad = torch.zeros_like(gradient)
        for k in range(n):
            frac_grad += weights[k] * self._gradient_history[n - 1 - k]

        return frac_grad

    def hebbian_update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Hebbian weight update: ΔW = η · post ⊗ pre.

        Combined with fractional memory for long-range correlations.

        Args:
            pre_activation: Pre-synaptic activations [batch, in_features].
            post_activation: Post-synaptic activations [batch, out_features].

        Returns:
            Weight update [out_features, in_features].
        """
        # Standard Hebbian: ΔW = η · y ⊗ x
        batch_size = pre_activation.shape[0]
        delta_w = torch.mm(post_activation.T, pre_activation) / batch_size
        return self.hebbian_lr * delta_w

    def fractional_hebbian_update(
        self,
        weight: torch.Tensor,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fractional Hebbian update combining Hebb rule with fractional memory.

        ΔW = η · D^α(y ⊗ x) - λ · W  (with weight decay)
        """
        hebb = self.hebbian_update(pre_activation, post_activation)
        frac_hebb = self.fractional_gradient(hebb.flatten()).view_as(hebb)

        # Oja's rule for stability: W_new = W + η·D^α(y⊗x) - η·diag(y²)·W
        decay = 0.001 * weight
        return frac_hebb - decay

    def caputo_derivative(
        self,
        f_values: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Caputo fractional derivative via trapezoidal integration.

        D^α f(t) = (1/Γ(1-α)) ∫₀ᵗ (t-τ)^{-α} f'(τ) dτ
        """
        n = f_values.shape[0]
        if n < 2:
            return torch.zeros_like(f_values[0])

        # Compute f'(τ) via finite differences
        f_prime = (f_values[1:] - f_values[:-1]) / dt

        # Kernel (t-τ)^{-α}
        t = (n - 1) * dt
        tau = torch.arange(n - 1, device=f_values.device).float() * dt
        kernel = (t - tau - dt / 2).clamp(min=1e-10).pow(-self.alpha)

        # Integration
        prefactor = 1.0 / gamma_fn(1 - self.alpha)
        result = prefactor * dt * torch.sum(kernel.unsqueeze(-1) * f_prime, dim=0)
        return result

    def _grunwald_letnikov_weights(self, n: int, device: torch.device) -> torch.Tensor:
        """Compute Grünwald-Letnikov weights for fractional derivative."""
        if self._grunwald_weights is not None and len(self._grunwald_weights) >= n:
            return self._grunwald_weights[:n]

        w = torch.zeros(n, device=device)
        w[0] = 1.0
        for k in range(1, n):
            w[k] = w[k - 1] * (1 - (self.alpha + 1) / k)
        self._grunwald_weights = w
        return w

    def reset_memory(self):
        """Reset gradient history."""
        self._gradient_history.clear()
        self._grunwald_weights = None
