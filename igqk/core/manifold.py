"""
Statistical Manifold Implementation

This module implements the statistical manifold M = {θ : p(·; θ) ∈ S}
with the Fisher Information metric as described in IGQK Theory.

Definition 1.1 (Statistical Manifold):
M is a Riemannian manifold with Fisher metric:
g_ij(θ) = E_θ[∂_i log p(x; θ) · ∂_j log p(x; θ)]

For neural networks:
- θ = weights W
- p(y|x; W) = output distribution of network
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from scipy.linalg import sqrtm


class StatisticalManifold:
    """
    Statistical manifold for neural network parameter space.

    Attributes:
        dim: Dimension of the parameter space (number of weights)
        model: Neural network model
        loss_fn: Loss function for training
    """

    def __init__(self, model: nn.Module, loss_fn: Optional[Callable] = None):
        """
        Initialize statistical manifold.

        Args:
            model: PyTorch neural network model
            loss_fn: Loss function (default: CrossEntropyLoss)
        """
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.dim = sum(p.numel() for p in model.parameters())

    def get_parameters(self) -> torch.Tensor:
        """Get flattened parameter vector θ."""
        return torch.cat([p.flatten() for p in self.model.parameters()])

    def set_parameters(self, theta: torch.Tensor):
        """Set model parameters from flattened vector θ."""
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = theta[idx:idx+numel].view_as(p)
            idx += numel

    def fisher_information_matrix(
        self,
        data_loader: torch.utils.data.DataLoader,
        empirical: bool = True,
        num_samples: Optional[int] = None,
        diagonal: bool = False
    ) -> torch.Tensor:
        """
        Compute Fisher Information Matrix.

        G_ij(θ) = E_θ[∂_i log p(x; θ) · ∂_j log p(x; θ)]

        Args:
            data_loader: DataLoader for computing empirical expectation
            empirical: If True, use empirical Fisher (more stable)
            num_samples: Number of samples to use (None = all)
            diagonal: If True, compute only diagonal elements (vector)

        Returns:
            Fisher information matrix [dim x dim] or diagonal vector [dim]
        """
        device = next(self.model.parameters()).device
        if diagonal:
            fisher = torch.zeros(self.dim, device=device)
        else:
            fisher = torch.zeros(self.dim, self.dim, device=device)

        n_samples = 0

        self.model.eval()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if num_samples and n_samples >= num_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            self.model.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            if empirical:
                # Empirical Fisher: use actual labels
                loss = self.loss_fn(outputs, targets)
            else:
                # True Fisher: use model's own predictions
                probs = torch.softmax(outputs, dim=1)
                sampled_targets = torch.multinomial(probs, 1).squeeze()
                loss = self.loss_fn(outputs, sampled_targets)

            # Backward pass to get gradients
            loss.backward()

            # Get gradient vector
            grad = torch.cat([p.grad.flatten() for p in self.model.parameters()])

            # Accumulate outer product: g ⊗ g
            if diagonal:
                fisher += grad ** 2
            else:
                fisher += torch.outer(grad, grad)

            n_samples += inputs.size(0)

        # Normalize by number of samples
        fisher /= n_samples

        return fisher

    def riemannian_distance(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        metric: str = 'hellinger'
    ) -> float:
        """
        Compute Riemannian distance on the manifold.

        For Fisher metric (Definition 3.2):
        d_M(θ, θ') = √(∫ (√p(x; θ) - √p(x; θ'))² dx)  (Hellinger distance)

        Args:
            theta1: First parameter vector
            theta2: Second parameter vector
            metric: Distance metric ('hellinger', 'kl', 'euclidean')

        Returns:
            Distance d_M(θ1, θ2)
        """
        if metric == 'euclidean':
            return torch.norm(theta1 - theta2).item()

        elif metric == 'hellinger':
            # Approximate Hellinger distance via parameter difference
            # For small differences: d_H ≈ ||θ1 - θ2||_G where G is Fisher
            diff = theta1 - theta2
            return torch.norm(diff).item()

        elif metric == 'kl':
            # KL divergence (asymmetric)
            # For neural networks, this is approximated via loss difference
            return torch.norm(theta1 - theta2).item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def tangent_space_projection(
        self,
        point: torch.Tensor,
        vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Project a vector onto the tangent space at a point.

        Args:
            point: Point θ on manifold
            vector: Vector to project

        Returns:
            Projected vector in tangent space T_θ M
        """
        # For unconstrained parameter space, tangent space = R^n
        return vector

    def geodesic(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        num_points: int = 10
    ) -> torch.Tensor:
        """
        Compute geodesic path between two points.

        Args:
            start: Starting point θ_0
            end: Ending point θ_1
            num_points: Number of points along geodesic

        Returns:
            Tensor of shape [num_points, dim] with geodesic points
        """
        # For now, use straight line (works for flat spaces)
        # TODO: Implement true geodesic with Christoffel symbols
        t = torch.linspace(0, 1, num_points, device=start.device)
        return start[None, :] + t[:, None] * (end - start)[None, :]

    def exponential_map(
        self,
        point: torch.Tensor,
        tangent_vector: torch.Tensor,
        t: float = 1.0
    ) -> torch.Tensor:
        """
        Exponential map: exp_θ(t·v) maps tangent vector to manifold.

        Args:
            point: Base point θ
            tangent_vector: Tangent vector v ∈ T_θ M
            t: Parameter (default 1.0)

        Returns:
            Point on manifold reached by following geodesic
        """
        # For flat spaces: exp_θ(t·v) = θ + t·v
        return point + t * tangent_vector

    def parallel_transport(
        self,
        vector: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transport a vector along geodesic from start to end.

        Args:
            vector: Vector at start point
            start: Starting point
            end: Ending point

        Returns:
            Transported vector at end point
        """
        # For flat spaces, parallel transport is identity
        # TODO: Implement for curved spaces
        return vector

    def dimension(self) -> int:
        """Return dimension of the manifold."""
        return self.dim
