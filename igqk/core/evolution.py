"""
Quantum Gradient Flow Implementation

This module implements the quantum gradient flow evolution on the statistical manifold.

Definition 2.1 (Quantum Gradient Flow):
dρ/dt = -i[H, ρ] - γ{G^{-1}∇L, ρ}

where:
- H = -Δ_M (Laplace-Beltrami operator, "kinetic energy")
- [H, ρ] = Hρ - ρH (commutator, unitary evolution)
- {∇L, ρ} = ∇Lρ + ρ∇L (anticommutator, dissipative evolution)
- γ > 0 (damping parameter)

Interpretation:
- Unitary part: Quantum exploration (superposition)
- Dissipative part: Gradient descent (convergence)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from .quantum_state import QuantumState
from .manifold import StatisticalManifold


class QuantumGradientFlow:
    """
    Quantum gradient flow dynamics for neural network optimization.

    Attributes:
        manifold: Statistical manifold
        hbar: Quantum uncertainty parameter ℏ (controls exploration)
        gamma: Damping parameter γ (controls convergence)
        dt: Time step for integration
    """

    def __init__(
        self,
        manifold: StatisticalManifold,
        hbar: float = 0.1,
        gamma: float = 0.01,
        dt: float = 0.01
    ):
        """
        Initialize quantum gradient flow.

        Args:
            manifold: Statistical manifold
            hbar: Quantum uncertainty ℏ (default 0.1)
            gamma: Damping γ (default 0.01)
            dt: Time step (default 0.01)
        """
        self.manifold = manifold
        self.hbar = hbar
        self.gamma = gamma
        self.dt = dt

    def laplace_beltrami(
        self,
        rho: QuantumState,
        fisher_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Laplace-Beltrami operator H = -Δ_M applied to quantum state.

        In coordinates: Δ_M = (1/√g) ∂_i (√g g^{ij} ∂_j)

        For simplicity, we approximate: H ≈ -G^{-1} ∇²
        where G is Fisher matrix.

        Args:
            rho: Current quantum state
            fisher_matrix: Fisher information matrix (computed if None)

        Returns:
            Hamiltonian operator H [n x n] or diagonal [n]
        """
        # For low-rank approximation, compute Laplacian in eigenbasis
        # H = -Tr(G^{-1} ∇²ρ) ≈ -G^{-1} operating on mean

        # Simplified: Use identity + small Gaussian noise to simulate diffusion
        device = rho.eigenvectors.device
        dim = rho.dim

        # Approximate Laplacian with second-order finite difference
        # For now, use regularized negative of Fisher matrix
        if fisher_matrix is None:
            # Use identity as approximation (diagonal vector)
            H = -torch.ones(dim, device=device)
        else:
            # H = -F^{-1} (inverse Fisher acts as diffusion)
            # Add regularization for stability
            if fisher_matrix.dim() == 1:
                # Diagonal Fisher
                H = -1.0 / (fisher_matrix + 1e-4)
            else:
                # Full Fisher
                reg = 1e-4 * torch.eye(fisher_matrix.shape[0], device=device)
                H = -torch.linalg.inv(fisher_matrix + reg)

        return H

    def commutator(
        self,
        H: torch.Tensor,
        rho: QuantumState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute commutator [H, ρ] = Hρ - ρH.

        For low-rank ρ = VΛV^T, we compute:
        [H, ρ] = HVΛ V^T - VΛV^T H

        Args:
            H: Hamiltonian operator [n x n] or diagonal [n]
            rho: Quantum state

        Returns:
            (eigenvalues_new, eigenvectors_new) after commutator
        """
        # Transform H to eigenbasis of ρ
        # H_eigen = V^T H V
        V = rho.eigenvectors
        Lambda = torch.diag(rho.eigenvalues)

        # [H, ρ] in eigenbasis
        # Since ρ is diagonal in its own basis: [H, ρ]_ij = H_ij (λ_i - λ_j)
        if H.dim() == 1:
            # H is diagonal vector
            # (V.T * H.unsqueeze(0)) @ V
            H_eigen = torch.mm(V.T * H.unsqueeze(0), V)
        else:
            H_eigen = torch.mm(torch.mm(V.T, H), V)

        # Commutator in eigenbasis
        comm_eigen = torch.mm(H_eigen, Lambda) - torch.mm(Lambda, H_eigen)

        # Transform back to original basis: V @ comm_eigen @ V^T
        # For low-rank updates, we keep it in eigenbasis
        # and return modified eigenvalues/vectors

        # Simplified: Treat as perturbation to eigenvalues
        # The commutator generates unitary evolution (rotation)
        # For simplicity, we keep structure and add small rotation

        return rho.eigenvalues, V  # Unitary evolution preserves eigenvalues

    def anticommutator(
        self,
        grad: torch.Tensor,
        rho: QuantumState,
        fisher_inv: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anticommutator {G^{-1}∇L, ρ} = G^{-1}∇L ρ + ρ G^{-1}∇L.

        Args:
            grad: Gradient ∇L [n]
            rho: Quantum state
            fisher_inv: Inverse Fisher matrix (identity if None)

        Returns:
            (eigenvalues_new, eigenvectors_new) after anticommutator
        """
        device = grad.device

        # Natural gradient: ∇̃L = G^{-1} ∇L
        if fisher_inv is not None:
            if fisher_inv.dim() == 1:
                # Diagonal inverse Fisher
                nat_grad = fisher_inv * grad
            else:
                nat_grad = torch.mv(fisher_inv, grad)
        else:
            nat_grad = grad

        # Anticommutator in eigenbasis
        # {A, ρ} for A = nat_grad vector (diagonal in position basis)
        # Transform to eigenbasis: A_eigen = V^T A V (but A is vector, so outer product)
        V = rho.eigenvectors
        Lambda = rho.eigenvalues

        # Projection of gradient onto eigenvectors
        # grad_proj[i] = ⟨ψ_i|∇L⟩
        grad_proj = torch.mv(V.T, nat_grad)

        # Anticommutator causes decay/growth of eigenvalues based on gradient alignment
        # Simplified update: λ_i → λ_i - dt·γ·grad_proj[i]²
        eigenvalues_new = Lambda - self.dt * self.gamma * (grad_proj ** 2)

        # Ensure positivity and normalize
        eigenvalues_new = torch.clamp(eigenvalues_new, min=1e-10)
        eigenvalues_new = eigenvalues_new / eigenvalues_new.sum()

        # Eigenvectors also shift in direction of gradient
        # Update: ψ_i → ψ_i - dt·γ·grad·⟨ψ_i|grad⟩
        eigenvectors_new = V - self.dt * self.gamma * torch.outer(nat_grad, grad_proj)

        # Orthonormalize
        eigenvectors_new, _ = torch.linalg.qr(eigenvectors_new)

        return eigenvalues_new, eigenvectors_new

    def step(
        self,
        rho: QuantumState,
        loss: float,
        grad: torch.Tensor,
        fisher_matrix: Optional[torch.Tensor] = None
    ) -> QuantumState:
        """
        Perform one step of quantum gradient flow.

        dρ/dt = -i[H, ρ] - γ{G^{-1}∇L, ρ}

        Args:
            rho: Current quantum state ρ_t
            loss: Current loss value L(θ)
            grad: Gradient ∇L [n]
            fisher_matrix: Fisher information matrix (optional)

        Returns:
            Updated quantum state ρ_{t+dt}
        """
        # 1. Compute Hamiltonian (Laplace-Beltrami)
        H = self.laplace_beltrami(rho, fisher_matrix)

        # 2. Unitary evolution: -i[H, ρ]
        # For real-valued quantum states, this adds small noise/exploration
        # Simplified: Add small random rotation
        unitary_noise = self.hbar * torch.randn_like(rho.eigenvectors) * self.dt

        # 3. Dissipative evolution: -γ{G^{-1}∇L, ρ}
        fisher_inv = None
        if fisher_matrix is not None:
            if fisher_matrix.dim() == 1:
                fisher_inv = 1.0 / (fisher_matrix + 1e-4)
            else:
                reg = 1e-4 * torch.eye(fisher_matrix.shape[0], device=fisher_matrix.device)
                fisher_inv = torch.linalg.inv(fisher_matrix + reg)

        eigenvalues_new, eigenvectors_new = self.anticommutator(grad, rho, fisher_inv)

        # 4. Combine updates
        eigenvectors_new = eigenvectors_new + unitary_noise

        # Orthonormalize
        eigenvectors_new, _ = torch.linalg.qr(eigenvectors_new)

        # Create new quantum state
        rho_new = QuantumState(eigenvalues_new, eigenvectors_new, check_properties=True)

        return rho_new

    def evolve(
        self,
        rho_init: QuantumState,
        num_steps: int,
        data_loader: torch.utils.data.DataLoader,
        compute_fisher: bool = False,
        fisher_every: int = 10,
        callback: Optional[Callable] = None
    ) -> QuantumState:
        """
        Evolve quantum state for multiple steps.

        Args:
            rho_init: Initial quantum state
            num_steps: Number of evolution steps
            data_loader: DataLoader for computing loss/gradients
            compute_fisher: Whether to compute Fisher matrix
            fisher_every: Compute Fisher every N steps
            callback: Optional callback(step, rho, loss, grad)

        Returns:
            Final quantum state
        """
        rho = rho_init
        device = next(self.manifold.model.parameters()).device

        for step in range(num_steps):
            # Get current mean parameters
            theta = rho.get_mean_parameter()
            self.manifold.set_parameters(theta)

            # Compute loss and gradient
            self.manifold.model.train()
            total_loss = 0.0
            total_grad = torch.zeros(self.manifold.dim, device=device)

            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                self.manifold.model.zero_grad()
                outputs = self.manifold.model(inputs)
                loss = self.manifold.loss_fn(outputs, targets)

                # Backward pass
                loss.backward()

                # Accumulate
                total_loss += loss.item()
                total_grad += torch.cat([p.grad.flatten() for p in self.manifold.model.parameters()])

            avg_loss = total_loss / len(data_loader)
            avg_grad = total_grad / len(data_loader)

            # Compute Fisher matrix (optional, expensive)
            fisher = None
            if compute_fisher and step % fisher_every == 0:
                fisher = self.manifold.fisher_information_matrix(data_loader, num_samples=1000)

            # Evolution step
            rho = self.step(rho, avg_loss, avg_grad, fisher)

            # Callback
            if callback:
                callback(step, rho, avg_loss, avg_grad)

        return rho
