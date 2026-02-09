"""
PyTorch Integration for IGQK

This module provides PyTorch optimizer and trainer using IGQK theory.

Algorithm 1: IGQK-Training (from theory document)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
import numpy as np

from ..core.manifold import StatisticalManifold
from ..core.quantum_state import QuantumState
from ..core.evolution import QuantumGradientFlow
from ..core.measurement import MeasurementOperator
from ..compression.projection import OptimalProjection


class IGQKOptimizer:
    """
    IGQK Optimizer for neural network training with compression.

    Implements Algorithm 1 from IGQK Theory.

    Attributes:
        model: Neural network model
        hbar: Quantum uncertainty
        gamma: Damping parameter
        compression_type: Type of compression ('ternary', 'lowrank', 'sparse')
    """

    def __init__(
        self,
        model: nn.Module,
        hbar: float = 0.1,
        gamma: float = 0.01,
        dt: float = 0.01,
        compression_type: str = 'ternary',
        rank: int = 10,
        fisher_diagonal: bool = True,
        **compression_params
    ):
        """
        Initialize IGQK optimizer.

        Args:
            model: PyTorch model to optimize
            hbar: Quantum uncertainty ℏ (default 0.1)
            gamma: Damping γ (default 0.01)
            dt: Time step (default 0.01)
            compression_type: Compression type
            rank: Rank for quantum state
            fisher_diagonal: If True, use diagonal approximation for Fisher matrix
            **compression_params: Additional compression parameters
        """
        self.model = model
        self.hbar = hbar
        self.gamma = gamma
        self.dt = dt
        self.rank = rank
        self.fisher_diagonal = fisher_diagonal

        # Initialize manifold
        self.manifold = StatisticalManifold(model)

        # Initialize quantum gradient flow
        self.flow = QuantumGradientFlow(
            manifold=self.manifold,
            hbar=hbar,
            gamma=gamma,
            dt=dt
        )

        # Initialize measurement operator
        self.measurement = MeasurementOperator()

        # Initialize projection
        self.projection = OptimalProjection(
            submanifold_type=compression_type,
            rank=compression_params.get('rank'),
            sparsity=compression_params.get('sparsity')
        )

        # Initialize quantum state
        theta_init = self.manifold.get_parameters()
        self.rho = QuantumState.from_point(theta_init, rank=rank)

        self.compression_type = compression_type

    def step(
        self,
        data_loader: DataLoader,
        compute_fisher: bool = False
    ) -> Dict[str, float]:
        """
        Perform one optimization step.

        Args:
            data_loader: DataLoader for current batch
            compute_fisher: Whether to compute Fisher matrix

        Returns:
            Dictionary with metrics
        """
        device = next(self.model.parameters()).device

        # Get current parameters
        theta = self.rho.get_mean_parameter()
        self.manifold.set_parameters(theta)

        # Compute loss and gradient
        self.model.train()
        total_loss = 0.0
        total_grad = torch.zeros(self.manifold.dim, device=device)

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.manifold.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Accumulate
            total_loss += loss.item() * inputs.size(0)
            total_grad += torch.cat([p.grad.flatten() for p in self.model.parameters()])

        avg_loss = total_loss / len(data_loader.dataset)
        avg_grad = total_grad / len(data_loader)

        # Compute Fisher matrix (optional)
        fisher = None
        if compute_fisher:
            fisher = self.manifold.fisher_information_matrix(
                data_loader,
                num_samples=1000,
                diagonal=self.fisher_diagonal
            )

        # Quantum evolution step
        self.rho = self.flow.step(self.rho, avg_loss, avg_grad, fisher)

        # Update model with new quantum state
        theta_new = self.rho.get_mean_parameter()
        self.manifold.set_parameters(theta_new)

        # Metrics
        metrics = {
            'loss': avg_loss,
            'grad_norm': torch.norm(avg_grad).item(),
            'entropy': self.rho.entropy(),
            'purity': self.rho.purity(),
            'trace': self.rho.trace()
        }

        return metrics

    def compress_model(self) -> nn.Module:
        """
        Compress model using measurement and projection.

        Returns:
            Compressed model
        """
        # Measure quantum state to get discrete weights
        discrete_weights = self.measurement.measure(self.rho, method='optimal')

        # Project onto compression submanifold
        compressed_weights = self.projection.projector.project(discrete_weights)

        # Set model parameters
        self.manifold.set_parameters(compressed_weights)

        return self.model

    def get_quantum_state(self) -> QuantumState:
        """Get current quantum state."""
        return self.rho


class IGQKTrainer:
    """
    High-level trainer using IGQK optimizer.

    Implements full IGQK training pipeline from Algorithm 1.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        hbar: float = 0.1,
        gamma: float = 0.01,
        compression_type: str = 'ternary',
        fisher_diagonal: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize IGQK trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            hbar: Quantum uncertainty
            gamma: Damping parameter
            compression_type: Compression type
            fisher_diagonal: Use diagonal Fisher approximation
            device: Device ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize optimizer
        self.optimizer = IGQKOptimizer(
            model=model,
            hbar=hbar,
            gamma=gamma,
            compression_type=compression_type,
            fisher_diagonal=fisher_diagonal
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'entropy': [],
            'purity': []
        }

    def train(
        self,
        num_epochs: int,
        compute_fisher_every: Optional[int] = 10,
        callback: Optional[Callable] = None
    ):
        """
        Train model using IGQK.

        Args:
            num_epochs: Number of training epochs
            compute_fisher_every: Compute Fisher every N epochs
            callback: Optional callback(epoch, metrics)
        """
        for epoch in range(num_epochs):
            # Training step
            should_compute_fisher = (
                compute_fisher_every is not None
                and compute_fisher_every > 0
                and (epoch % compute_fisher_every == 0)
            )

            metrics = self.optimizer.step(
                self.train_loader,
                compute_fisher=should_compute_fisher
            )

            # Validation
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate(self.val_loader)
                metrics['val_loss'] = val_loss
                metrics['val_acc'] = val_acc
            else:
                val_loss = None
                val_acc = None

            # Record history
            self.history['train_loss'].append(metrics['loss'])
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
            self.history['entropy'].append(metrics['entropy'])
            self.history['purity'].append(metrics['purity'])

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Entropy: {metrics['entropy']:.4f} | "
                  f"Purity: {metrics['purity']:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Callback
            if callback:
                callback(epoch, metrics)

    def evaluate(self, data_loader: DataLoader) -> tuple:
        """
        Evaluate model on data.

        Args:
            data_loader: Data loader

        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def compress(self) -> nn.Module:
        """
        Compress trained model.

        Returns:
            Compressed model
        """
        compressed_model = self.optimizer.compress_model()

        # Evaluate compressed model
        if self.val_loader is not None:
            val_loss, val_acc = self.evaluate(self.val_loader)
            print(f"\nCompressed Model:")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Compute compression ratio
            compression_ratio = self.optimizer.projection.compression_ratio(
                self.optimizer.manifold.dim
            )
            print(f"  Compression Ratio: {compression_ratio:.4f}")

        return compressed_model
