"""Tests for compression and measurement."""

import pytest
import torch
from igqk.core.quantum_state import QuantumState
from igqk.core.measurement import MeasurementOperator, ProjectiveMeasurement
from igqk.compression.projection import OptimalProjection


class TestMeasurement:
    def test_threshold_measure_ternary(self):
        theta = torch.randn(100)
        rho = QuantumState.from_point(theta, rank=3)
        measurement = MeasurementOperator(weight_values=[-1.0, 0.0, 1.0])
        discrete = measurement.measure(rho, method="threshold")
        assert discrete.shape == (100,)
        unique = set(discrete.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_sample_measure(self):
        theta = torch.randn(50)
        rho = QuantumState.from_point(theta, rank=3)
        measurement = MeasurementOperator()
        discrete = measurement.measure(rho, method="sample")
        assert discrete.shape == (50,)

    def test_optimal_measure(self):
        theta = torch.randn(50)
        rho = QuantumState.from_point(theta, rank=3)
        measurement = MeasurementOperator()
        discrete = measurement.measure(rho, method="optimal")
        assert discrete.shape == (50,)
        unique = set(discrete.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_probability(self):
        theta = torch.randn(20)
        rho = QuantumState.from_point(theta, rank=3)
        measurement = MeasurementOperator()
        config = torch.zeros(20)
        prob = measurement.probability(rho, config)
        assert 0.0 <= prob <= 1.0

    def test_binary_measure(self):
        theta = torch.randn(30)
        rho = QuantumState.from_point(theta, rank=1)
        measurement = MeasurementOperator(weight_values=[-1.0, 1.0])
        discrete = measurement.measure(rho, method="threshold")
        unique = set(discrete.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})


class TestProjectiveMeasurement:
    def test_ternary_projection(self):
        pm = ProjectiveMeasurement(submanifold_type="ternary")
        theta = torch.randn(50)
        projected = pm.project(theta)
        unique = set(projected.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_sparse_projection(self):
        pm = ProjectiveMeasurement(submanifold_type="sparse", sparsity=0.2)
        theta = torch.randn(100)
        projected = pm.project(theta)
        nonzero = (projected != 0).sum().item()
        assert nonzero == 20  # 0.2 * 100

    def test_lowrank_projection(self):
        pm = ProjectiveMeasurement(submanifold_type="lowrank", rank=2)
        theta = torch.randn(16)  # 4x4 matrix
        projected = pm.project(theta)
        assert projected.shape == theta.shape


class TestOptimalProjection:
    def test_project_state(self):
        rho = QuantumState.from_point(torch.randn(50), rank=3)
        proj = OptimalProjection(submanifold_type="ternary")
        compressed = proj.project_state(rho)
        assert compressed.shape == (50,)

    def test_project_model(self):
        import torch.nn as nn
        model = nn.Linear(10, 5)
        proj = OptimalProjection(submanifold_type="ternary")
        compressed_model = proj.project_model(model)
        for param in compressed_model.parameters():
            unique = set(param.flatten().unique().tolist())
            assert unique.issubset({-1.0, 0.0, 1.0})

    def test_compression_ratio_ternary(self):
        proj = OptimalProjection(submanifold_type="ternary")
        ratio = proj.compression_ratio(1000)
        assert ratio < 0.1  # Should be ~1/16

    def test_distortion(self):
        proj = OptimalProjection(submanifold_type="ternary")
        original = torch.randn(100)
        compressed = torch.zeros(100)
        d = proj.distortion(original, compressed)
        assert d > 0
