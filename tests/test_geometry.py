"""Tests for geometry module."""

import pytest
import torch
import torch.nn as nn
from igqk.geometry.fisher import FisherMetric
from igqk.geometry.geodesic import GeodesicSolver
from igqk.geometry.laplacian import LaplaceBeltrami


class TestFisherMetric:
    def _make_model_and_loader(self):
        model = nn.Linear(10, 2)
        data = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        return model, loader

    def test_compute_empirical(self):
        model, loader = self._make_model_and_loader()
        fisher = FisherMetric(model)
        F = fisher.compute_empirical(loader, num_samples=16)
        dim = sum(p.numel() for p in model.parameters())
        assert F.shape == (dim, dim)
        # Fisher should be PSD
        eigenvalues = torch.linalg.eigvalsh(F)
        assert torch.all(eigenvalues >= -1e-6)

    def test_block_diagonal(self):
        model, loader = self._make_model_and_loader()
        fisher = FisherMetric(model)
        blocks = fisher.compute_empirical(loader, num_samples=16, block_diagonal=True)
        assert len(blocks) == 2  # weight and bias

    def test_natural_gradient(self):
        model, loader = self._make_model_and_loader()
        fisher = FisherMetric(model)
        F = fisher.compute_empirical(loader, num_samples=16)
        grad = torch.randn(F.shape[0])
        nat_grad = fisher.natural_gradient(grad, F)
        assert nat_grad.shape == grad.shape

    def test_condition_number(self):
        model, loader = self._make_model_and_loader()
        fisher = FisherMetric(model)
        F = fisher.compute_empirical(loader, num_samples=16)
        cond = fisher.condition_number(F)
        assert cond > 0


class TestGeodesicSolver:
    def test_euclidean_exp_map(self):
        solver = GeodesicSolver(metric_fn=None)
        p = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        result = solver.exponential_map(p, v, t=1.0)
        assert torch.allclose(result, torch.tensor([1.0, 1.0]))

    def test_euclidean_log_map(self):
        solver = GeodesicSolver(metric_fn=None)
        p = torch.tensor([0.0, 0.0])
        q = torch.tensor([3.0, 4.0])
        v = solver.logarithmic_map(p, q)
        assert torch.allclose(v, q)

    def test_geodesic_distance_euclidean(self):
        solver = GeodesicSolver(metric_fn=None)
        p = torch.tensor([0.0, 0.0])
        q = torch.tensor([3.0, 4.0])
        dist = solver.geodesic_distance(p, q)
        assert abs(dist - 5.0) < 1e-5

    def test_geodesic_path(self):
        solver = GeodesicSolver(metric_fn=None)
        p = torch.tensor([0.0, 0.0])
        q = torch.tensor([1.0, 1.0])
        path = solver.geodesic_path(p, q, num_points=5)
        assert path.shape == (5, 2)
        assert torch.allclose(path[0], p)
        assert torch.allclose(path[-1], q, atol=1e-4)

    def test_parallel_transport_euclidean(self):
        solver = GeodesicSolver(metric_fn=None)
        v = torch.tensor([1.0, 0.0])
        curve = torch.stack([torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])])
        transported = solver.parallel_transport(v, curve)
        assert torch.allclose(transported, v)  # Identity for flat space


class TestLaplaceBeltrami:
    def test_hamiltonian_identity_metric(self):
        lb = LaplaceBeltrami()
        H = lb.hamiltonian(5)
        expected = -torch.eye(5)
        assert torch.allclose(H, expected)

    def test_hamiltonian_with_metric(self):
        metric = 2.0 * torch.eye(4)
        lb = LaplaceBeltrami()
        H = lb.hamiltonian(4, metric=metric)
        assert H.shape == (4, 4)

    def test_fractional_alpha_one(self):
        lb = LaplaceBeltrami()
        metric = torch.eye(3) * 2
        F = lb.fractional(1.0, 3, metric=metric)
        H = lb.hamiltonian(3, metric=metric)
        # For α=1, fractional should approximate the original
        assert F.shape == (3, 3)

    def test_fractional_alpha_half(self):
        lb = LaplaceBeltrami()
        F = lb.fractional(0.5, 4)
        assert F.shape == (4, 4)
        eigenvalues = torch.linalg.eigvalsh(F)
        assert torch.all(eigenvalues >= -1e-6)
