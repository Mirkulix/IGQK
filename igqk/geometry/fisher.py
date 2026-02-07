"""
Fisher Information Metric computation.

The Fisher metric defines the Riemannian structure on the statistical manifold:
    g_ij(θ) = E_θ[∂_i log p(x; θ) · ∂_j log p(x; θ)]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class FisherMetric:
    """Efficient Fisher Information Matrix computation with block-diagonal support."""

    def __init__(self, model: nn.Module, loss_fn: Optional[nn.Module] = None):
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.dim = sum(p.numel() for p in model.parameters())

    def compute_empirical(
        self,
        data_loader: DataLoader,
        num_samples: Optional[int] = None,
        block_diagonal: bool = False,
    ) -> torch.Tensor:
        """
        Compute empirical Fisher: G_ij = (1/N) Σ_n (∂_i L_n)(∂_j L_n).

        Args:
            data_loader: Training data.
            num_samples: Max samples (None = all).
            block_diagonal: Use block-diagonal approximation for scalability.

        Returns:
            Fisher matrix [dim x dim] or list of blocks.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        if block_diagonal:
            return self._compute_block_diagonal(data_loader, num_samples, device)

        fisher = torch.zeros(self.dim, self.dim, device=device)
        n = 0

        for inputs, targets in data_loader:
            if num_samples and n >= num_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            for i in range(batch_size):
                if num_samples and n >= num_samples:
                    break
                self.model.zero_grad()
                output = self.model(inputs[i : i + 1])
                loss = self.loss_fn(output, targets[i : i + 1])
                loss.backward()

                grad = torch.cat([p.grad.flatten() for p in self.model.parameters()])
                fisher += torch.outer(grad, grad)
                n += 1

        if n > 0:
            fisher /= n
        return fisher

    def _compute_block_diagonal(
        self, data_loader: DataLoader, num_samples: Optional[int], device: torch.device
    ) -> list:
        """Compute block-diagonal Fisher (one block per parameter group)."""
        param_list = list(self.model.parameters())
        blocks = [torch.zeros(p.numel(), p.numel(), device=device) for p in param_list]
        n = 0

        for inputs, targets in data_loader:
            if num_samples and n >= num_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            self.model.zero_grad()
            output = self.model(inputs)
            loss = self.loss_fn(output, targets)
            loss.backward()

            for idx, p in enumerate(param_list):
                g = p.grad.flatten()
                blocks[idx] += torch.outer(g, g)
            n += inputs.size(0)

        if n > 0:
            blocks = [b / n for b in blocks]
        return blocks

    def natural_gradient(
        self,
        grad: torch.Tensor,
        fisher: torch.Tensor,
        damping: float = 1e-4,
    ) -> torch.Tensor:
        """Compute natural gradient: G^{-1} ∇L using linear solve."""
        reg = damping * torch.eye(fisher.shape[0], device=fisher.device)
        F_reg = fisher + reg
        nat_grad = torch.linalg.solve(F_reg, grad)
        return nat_grad

    def condition_number(self, fisher: torch.Tensor) -> float:
        """Monitor Fisher metric condition number for numerical stability."""
        eigenvalues = torch.linalg.eigvalsh(fisher)
        positive = eigenvalues[eigenvalues > 1e-10]
        if len(positive) < 2:
            return float("inf")
        return (positive.max() / positive.min()).item()
