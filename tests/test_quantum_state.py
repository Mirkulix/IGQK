"""Tests for quantum state operations."""

import pytest
import torch
from igqk.core.quantum_state import QuantumState


class TestQuantumState:
    def test_from_point_pure_state(self):
        theta = torch.randn(100)
        rho = QuantumState.from_point(theta, rank=1)
        assert rho.rank == 1
        assert abs(rho.trace() - 1.0) < 1e-5
        assert abs(rho.purity() - 1.0) < 1e-5

    def test_from_point_mixed_state(self):
        theta = torch.randn(50)
        rho = QuantumState.from_point(theta, rank=5)
        assert rho.rank == 5
        assert abs(rho.trace() - 1.0) < 1e-5
        assert rho.purity() < 1.0

    def test_eigenvalues_positive(self):
        theta = torch.randn(30)
        rho = QuantumState.from_point(theta, rank=3)
        assert torch.all(rho.eigenvalues >= 0)

    def test_eigenvalues_sum_to_one(self):
        theta = torch.randn(30)
        rho = QuantumState.from_point(theta, rank=3)
        assert abs(rho.eigenvalues.sum().item() - 1.0) < 1e-5

    def test_entropy_pure_state(self):
        theta = torch.randn(20)
        rho = QuantumState.from_point(theta, rank=1)
        assert abs(rho.entropy()) < 1e-5  # Pure state has zero entropy

    def test_entropy_mixed_state(self):
        theta = torch.randn(20)
        rho = QuantumState.from_point(theta, rank=5)
        assert rho.entropy() > 0  # Mixed state has positive entropy

    def test_to_matrix_and_back(self):
        theta = torch.randn(20)
        rho = QuantumState.from_point(theta, rank=3)
        matrix = rho.to_matrix()
        assert matrix.shape == (20, 20)
        assert abs(torch.trace(matrix).item() - 1.0) < 1e-4
        eigenvalues = torch.linalg.eigvalsh(matrix)
        assert torch.all(eigenvalues >= -1e-6)

    def test_from_matrix(self):
        dim = 10
        V = torch.randn(dim, 3)
        V, _ = torch.linalg.qr(V)
        lam = torch.tensor([0.5, 0.3, 0.2])
        matrix = V @ torch.diag(lam) @ V.T
        rho = QuantumState.from_matrix(matrix, rank=3)
        assert rho.rank == 3
        assert abs(rho.trace() - 1.0) < 1e-5

    def test_sample(self):
        theta = torch.randn(50)
        rho = QuantumState.from_point(theta, rank=5)
        samples = rho.sample(num_samples=10)
        assert samples.shape == (10, 50)

    def test_truncate_rank(self):
        theta = torch.randn(30)
        rho = QuantumState.from_point(theta, rank=5)
        rho.truncate_rank(2)
        assert rho.rank == 2
        assert abs(rho.trace() - 1.0) < 1e-5

    def test_fidelity_self(self):
        theta = torch.randn(20)
        rho = QuantumState.from_point(theta, rank=1)
        f = rho.fidelity(rho)
        assert abs(f - 1.0) < 1e-3

    def test_get_mean_parameter(self):
        theta = torch.randn(30)
        rho = QuantumState.from_point(theta, rank=1)
        mean = rho.get_mean_parameter()
        assert mean.shape == (30,)

    def test_expectation_vector(self):
        theta = torch.randn(20)
        rho = QuantumState.from_point(theta, rank=3)
        obs = torch.randn(20)
        val = rho.expectation(obs)
        assert isinstance(val, float)

    def test_renormalize(self):
        evals = torch.tensor([0.3, 0.2, 0.1])  # Sum != 1
        evecs = torch.randn(20, 3)
        evecs, _ = torch.linalg.qr(evecs)
        rho = QuantumState(evals, evecs, check_properties=True)
        assert abs(rho.trace() - 1.0) < 1e-5
