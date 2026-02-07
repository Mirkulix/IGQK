"""Tests for theory modules (HLWT, TLGT, FCHL)."""

import pytest
import torch
import numpy as np
from igqk.theory.hlwt import HybridLaplaceWavelet
from igqk.theory.tlgt import TernaryLieGroup
from igqk.theory.fchl import FractionalHebbianLearning


class TestHLWT:
    def test_decompose_reconstruct(self):
        hlwt = HybridLaplaceWavelet(num_levels=3)
        signal = torch.randn(64)
        coeffs = hlwt.decompose(signal)
        reconstructed = hlwt.reconstruct(coeffs)
        assert torch.allclose(signal, reconstructed[:64], atol=1e-5)

    def test_decompose_levels(self):
        hlwt = HybridLaplaceWavelet(num_levels=4)
        signal = torch.randn(128)
        coeffs = hlwt.decompose(signal)
        assert len(coeffs) <= 5  # approximation + up to 4 detail levels

    def test_compress(self):
        hlwt = HybridLaplaceWavelet()
        weights = torch.randn(256)
        compressed, ratio = hlwt.compress(weights, keep_ratio=0.5)
        assert compressed.shape == weights.shape
        assert 0 < ratio <= 1.0

    def test_compress_reduces_info(self):
        hlwt = HybridLaplaceWavelet()
        weights = torch.randn(128)
        compressed, _ = hlwt.compress(weights, keep_ratio=0.3)
        # Compressed should have some zeros in wavelet domain
        assert compressed.shape == weights.shape

    def test_analyze_frequency(self):
        hlwt = HybridLaplaceWavelet()
        weights = torch.randn(64)
        analysis = hlwt.analyze_frequency(weights)
        assert "approximation_energy" in analysis
        assert analysis["approximation_energy"] > 0

    def test_db2_wavelet(self):
        hlwt = HybridLaplaceWavelet(num_levels=2, wavelet="db2")
        signal = torch.randn(32)
        coeffs = hlwt.decompose(signal)
        assert len(coeffs) > 1


class TestTLGT:
    def test_quantize_ternary(self):
        tlgt = TernaryLieGroup(100)
        weights = torch.randn(100) * 0.5
        ternary, scale = tlgt.quantize(weights)
        # All values should be -scale, 0, or +scale
        unique = ternary.unique()
        for v in unique:
            assert v.item() == 0.0 or abs(abs(v.item()) - scale.item()) < 1e-5

    def test_quantize_preserves_shape(self):
        tlgt = TernaryLieGroup(48)
        weights = torch.randn(4, 12)
        ternary, scale = tlgt.quantize(weights)
        assert ternary.shape == weights.shape

    def test_group_action(self):
        tlgt = TernaryLieGroup(10)
        g = torch.tensor([1, -1, 0, 1, -1, 1, 0, -1, 1, 0], dtype=torch.float)
        weights = torch.randn(10)
        result = tlgt.group_action(g, weights)
        assert torch.allclose(result, g * weights)

    def test_coset_decomposition(self):
        tlgt = TernaryLieGroup(50)
        weights = torch.randn(50)
        q, scale, residual = tlgt.coset_decomposition(weights)
        # Q should be ternary
        for v in q.unique():
            assert v.item() in [-1.0, 0.0, 1.0]

    def test_invariant_compression(self):
        tlgt = TernaryLieGroup(100)
        weights = torch.randn(100)
        compressed = tlgt.invariant_compression(weights, num_iterations=3)
        assert compressed.shape == weights.shape

    def test_compression_stats(self):
        tlgt = TernaryLieGroup(100)
        weights = torch.randn(100)
        ternary, _ = tlgt.quantize(weights)
        stats = tlgt.compression_stats(weights, ternary)
        assert "distortion" in stats
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] < 1.0  # Actually compressed


class TestFCHL:
    def test_fractional_gradient(self):
        fchl = FractionalHebbianLearning(alpha=0.9, memory_length=10)
        for _ in range(5):
            grad = torch.randn(20)
            frac_grad = fchl.fractional_gradient(grad)
            assert frac_grad.shape == grad.shape

    def test_memory_accumulation(self):
        fchl = FractionalHebbianLearning(alpha=0.9, memory_length=10)
        for i in range(15):
            fchl.fractional_gradient(torch.randn(10))
        assert len(fchl._gradient_history) == 10  # Capped at memory_length

    def test_hebbian_update(self):
        fchl = FractionalHebbianLearning()
        pre = torch.randn(8, 10)  # batch=8, in=10
        post = torch.randn(8, 5)  # batch=8, out=5
        delta_w = fchl.hebbian_update(pre, post)
        assert delta_w.shape == (5, 10)

    def test_fractional_hebbian_update(self):
        fchl = FractionalHebbianLearning(alpha=0.8)
        weight = torch.randn(5, 10)
        pre = torch.randn(8, 10)
        post = torch.randn(8, 5)
        delta = fchl.fractional_hebbian_update(weight, pre, post)
        assert delta.shape == (5, 10)

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            FractionalHebbianLearning(alpha=0.0)
        with pytest.raises(ValueError):
            FractionalHebbianLearning(alpha=1.5)

    def test_caputo_derivative(self):
        fchl = FractionalHebbianLearning(alpha=0.5)
        f_values = torch.randn(10, 5)
        result = fchl.caputo_derivative(f_values, dt=0.1)
        assert result.shape == (5,)

    def test_reset_memory(self):
        fchl = FractionalHebbianLearning()
        fchl.fractional_gradient(torch.randn(10))
        fchl.reset_memory()
        assert len(fchl._gradient_history) == 0
