"""Integration tests for IGQK training pipeline."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from igqk.core.manifold import StatisticalManifold
from igqk.core.quantum_state import QuantumState
from igqk.core.evolution import QuantumGradientFlow
from igqk.core.measurement import MeasurementOperator
from igqk.integration.pytorch import IGQKOptimizer, IGQKTrainer


def _make_toy_data(n_samples=100, in_features=10, num_classes=3):
    X = torch.randn(n_samples, in_features)
    y = torch.randint(0, num_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def _make_model(in_features=10, num_classes=3):
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


class TestFullPipeline:
    def test_manifold_to_quantum_state(self):
        model = _make_model()
        manifold = StatisticalManifold(model)
        theta = manifold.get_parameters()
        rho = QuantumState.from_point(theta, rank=3)
        assert abs(rho.trace() - 1.0) < 1e-5

    def test_quantum_evolution_step(self):
        model = _make_model()
        manifold = StatisticalManifold(model)
        flow = QuantumGradientFlow(manifold, hbar=0.1, gamma=0.01)

        theta = manifold.get_parameters()
        rho = QuantumState.from_point(theta, rank=3)
        grad = torch.randn(manifold.dim)

        rho_new = flow.step(rho, loss=1.0, grad=grad)
        assert abs(rho_new.trace() - 1.0) < 1e-4

    def test_optimizer_step(self):
        model = _make_model()
        loader = _make_toy_data()
        optimizer = IGQKOptimizer(model, hbar=0.1, gamma=0.01, rank=3)
        metrics = optimizer.step(loader)
        assert "loss" in metrics
        assert "entropy" in metrics
        assert "purity" in metrics

    def test_optimizer_compress(self):
        model = _make_model()
        loader = _make_toy_data()
        optimizer = IGQKOptimizer(model, compression_type="ternary", rank=3)
        optimizer.step(loader)
        compressed = optimizer.compress_model()
        assert compressed is not None

    def test_trainer_train_and_compress(self):
        model = _make_model()
        train_loader = _make_toy_data()
        val_loader = _make_toy_data(n_samples=30)

        trainer = IGQKTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hbar=0.1,
            gamma=0.01,
            compression_type="ternary",
            device="cpu",
        )
        trainer.train(num_epochs=2)
        compressed = trainer.compress()
        assert compressed is not None
        assert len(trainer.history["train_loss"]) == 2

    def test_evaluate(self):
        model = _make_model()
        loader = _make_toy_data()
        trainer = IGQKTrainer(model=model, train_loader=loader, device="cpu")
        loss, acc = trainer.evaluate(loader)
        assert loss >= 0
        assert 0 <= acc <= 1


class TestEndToEnd:
    def test_train_compress_evaluate(self):
        """Full end-to-end: train -> compress -> evaluate."""
        model = _make_model(in_features=20, num_classes=5)
        train_loader = _make_toy_data(n_samples=80, in_features=20, num_classes=5)
        val_loader = _make_toy_data(n_samples=20, in_features=20, num_classes=5)

        trainer = IGQKTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hbar=0.1,
            gamma=0.01,
            compression_type="ternary",
            device="cpu",
        )

        # Train
        trainer.train(num_epochs=3)
        assert len(trainer.history["train_loss"]) == 3

        # Compress
        compressed_model = trainer.compress()

        # Evaluate compressed model
        loss, acc = trainer.evaluate(val_loader)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_quantum_state_monitoring(self):
        """Monitor quantum state properties during training."""
        model = _make_model()
        loader = _make_toy_data()
        optimizer = IGQKOptimizer(model, rank=3)

        entropies = []
        purities = []

        for _ in range(3):
            metrics = optimizer.step(loader)
            entropies.append(metrics["entropy"])
            purities.append(metrics["purity"])

        # Entropy and purity should be tracked
        assert len(entropies) == 3
        assert all(isinstance(e, float) for e in entropies)
