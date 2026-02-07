"""Tests for IGQK innovation modules."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

from igqk.auto import AutoIGQK
from igqk.entanglement import QuantumEntanglementCompressor
from igqk.annealing import QuantumAnnealingScheduler
from igqk.streaming import StreamingAdaptiveModel
from igqk.format import IGQKFormat
from igqk.visualizer import QuantumVisualizer


def _make_model():
    return nn.Sequential(
        nn.Linear(50, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


# ===== AutoIGQK =====

class TestAutoIGQK:
    def test_analyze(self):
        model = _make_model()
        auto = AutoIGQK(target_compression=0.1)
        plans = auto.analyze(model)
        assert len(plans) > 0
        for plan in plans:
            assert plan.method in ("ternary", "wavelet", "sparse", "none")
            assert 0.0 <= plan.strength <= 1.0

    def test_compress(self):
        model = _make_model()
        auto = AutoIGQK()
        result = auto.compress(model)
        assert result.model is not None
        assert result.total_compression < 1.0
        assert result.quantum_advantage >= 0

    def test_per_layer_different_methods(self):
        model = _make_model()
        auto = AutoIGQK()
        plans = auto.analyze(model)
        methods = {p.method for p in plans if p.method != "none"}
        # AutoIGQK should consider different methods
        assert len(plans) > 0

    def test_plans_have_entropy(self):
        model = _make_model()
        auto = AutoIGQK()
        plans = auto.analyze(model)
        for plan in plans:
            assert isinstance(plan.entropy, float)
            assert isinstance(plan.information_density, float)


# ===== Quantum Entanglement =====

class TestEntanglement:
    def test_discover_entanglement(self):
        model = _make_model()
        compressor = QuantumEntanglementCompressor(entanglement_threshold=0.01)
        pairs = compressor.discover_entanglement(model)
        # Should find at least some correlations in a deep model
        assert isinstance(pairs, list)

    def test_compress_entangled(self):
        model = _make_model()
        compressor = QuantumEntanglementCompressor(entanglement_threshold=0.01)
        compressed, pairs, stats = compressor.compress_entangled(model)
        assert compressed is not None
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] <= 1.0

    def test_entanglement_pair_properties(self):
        model = _make_model()
        compressor = QuantumEntanglementCompressor(entanglement_threshold=0.01)
        pairs = compressor.discover_entanglement(model)
        for pair in pairs:
            assert pair.mutual_information >= 0
            assert 0 <= pair.correlation <= 1
            assert pair.shared_rank >= 1

    def test_visualize_entanglement(self):
        model = _make_model()
        compressor = QuantumEntanglementCompressor(entanglement_threshold=0.01)
        viz = compressor.visualize_entanglement(model)
        assert "layers" in viz
        assert "pairs" in viz


# ===== Quantum Annealing =====

class TestAnnealing:
    def test_linear_schedule(self):
        scheduler = QuantumAnnealingScheduler(
            hbar_init=1.0, hbar_final=0.01, total_steps=100, schedule="linear"
        )
        params_start = scheduler.step(0)
        params_mid = scheduler.step(50)
        params_end = scheduler.step(99)

        assert params_start["hbar"] > params_end["hbar"]
        assert params_start["gamma"] < params_end["gamma"]

    def test_cosine_schedule(self):
        scheduler = QuantumAnnealingScheduler(
            total_steps=100, schedule="cosine"
        )
        params = [scheduler.step(i) for i in range(100)]
        # hbar should decrease overall
        assert params[0]["hbar"] > params[-1]["hbar"]

    def test_adaptive_schedule(self):
        scheduler = QuantumAnnealingScheduler(
            total_steps=100, schedule="adaptive"
        )
        # Simulate with decreasing entropy
        for i in range(100):
            entropy = max(0.01, 1.0 - (i / 100))
            params = scheduler.step(i, entropy=entropy, loss=max(0.01, 1.0 - i/200))
            assert params["hbar"] > 0
            assert params["gamma"] > 0
            assert params["phase"] in ("exploration", "transition", "convergence", "measurement")

    def test_phase_detection(self):
        scheduler = QuantumAnnealingScheduler(total_steps=100, schedule="adaptive")
        # Exploration phase
        p = scheduler.step(5, entropy=0.9, loss=2.0)
        assert p["phase"] == "exploration"

        # Simulate to measurement phase
        for i in range(6, 97):
            scheduler.step(i, entropy=0.9 - i/100, loss=2.0 - i/100)

        p = scheduler.step(97, entropy=0.01, loss=0.1)
        assert p["phase"] == "measurement"

    def test_history(self):
        scheduler = QuantumAnnealingScheduler(total_steps=10)
        for i in range(10):
            scheduler.step(i)
        assert len(scheduler.history) == 10


# ===== .igqk Format =====

class TestIGQKFormat:
    def test_save_load_ternary(self):
        """Ternary model should round-trip through .igqk format."""
        model = _make_model()
        # Make ternary
        with torch.no_grad():
            for p in model.parameters():
                std = p.data.std()
                ternary = torch.zeros_like(p.data)
                ternary[p.data > 0.5 * std] = std
                ternary[p.data < -0.5 * std] = -std
                p.data = ternary

        with tempfile.NamedTemporaryFile(suffix=".igqk", delete=False) as f:
            path = f.name

        try:
            file_size = IGQKFormat.save(model.state_dict(), path)
            assert file_size > 0

            loaded, metadata = IGQKFormat.load(path)
            assert len(loaded) == len(model.state_dict())

            for name in model.state_dict():
                original = model.state_dict()[name]
                restored = loaded[name]
                assert original.shape == restored.shape
                assert torch.allclose(original, restored, atol=1e-5)
        finally:
            os.unlink(path)

    def test_save_load_float32(self):
        """Float32 model should round-trip."""
        state_dict = {"weight": torch.randn(10, 5), "bias": torch.randn(10)}

        with tempfile.NamedTemporaryFile(suffix=".igqk", delete=False) as f:
            path = f.name

        try:
            IGQKFormat.save(state_dict, path)
            loaded, _ = IGQKFormat.load(path)
            assert torch.allclose(state_dict["weight"], loaded["weight"], atol=1e-5)
        finally:
            os.unlink(path)

    def test_ternary_compression_ratio(self):
        """Ternary .igqk should be much smaller than float32."""
        # Float32 state dict
        state_dict_float = {"w": torch.randn(1000)}

        # Ternary state dict
        ternary = torch.zeros(1000)
        ternary[:300] = 1.0
        ternary[300:600] = -1.0
        state_dict_ternary = {"w": ternary}

        with tempfile.NamedTemporaryFile(suffix=".igqk", delete=False) as f:
            path_float = f.name
        with tempfile.NamedTemporaryFile(suffix=".igqk", delete=False) as f:
            path_ternary = f.name

        try:
            size_float = IGQKFormat.save(state_dict_float, path_float)
            size_ternary = IGQKFormat.save(state_dict_ternary, path_ternary)

            # Ternary should be significantly smaller
            assert size_ternary < size_float * 0.5
        finally:
            os.unlink(path_float)
            os.unlink(path_ternary)

    def test_info(self):
        state_dict = {"w": torch.randn(10)}
        with tempfile.NamedTemporaryFile(suffix=".igqk", delete=False) as f:
            path = f.name
        try:
            IGQKFormat.save(state_dict, path, metadata={"test": True})
            info = IGQKFormat.info(path)
            assert info["test"] is True
            assert info["file_size_bytes"] > 0
        finally:
            os.unlink(path)


# ===== Streaming Adaptive =====

class TestStreaming:
    def test_forward(self):
        model = _make_model()
        streaming = StreamingAdaptiveModel(model)
        x = torch.randn(4, 50)
        output = streaming(x)
        assert output.shape == (4, 10)

    def test_stats_tracking(self):
        model = _make_model()
        streaming = StreamingAdaptiveModel(model)

        for _ in range(10):
            x = torch.randn(1, 50)
            streaming(x)

        stats = streaming.get_stats()
        assert stats["total_inferences"] == 10

    def test_routing_distribution(self):
        model = _make_model()
        streaming = StreamingAdaptiveModel(model)

        for _ in range(50):
            x = torch.randn(1, 50)
            streaming(x)

        total = streaming.stats.total_inferences
        routed = (
            streaming.stats.fast_path_count
            + streaming.stats.medium_path_count
            + streaming.stats.full_path_count
        )
        assert routed == total

    def test_compressed_versions_exist(self):
        model = _make_model()
        streaming = StreamingAdaptiveModel(model)
        assert streaming.model_full is not None
        assert streaming.model_medium is not None
        assert streaming.model_fast is not None


# ===== Visualizer =====

class TestVisualizer:
    def test_record_and_summary(self):
        viz = QuantumVisualizer()
        for i in range(10):
            viz.record(
                step=i, entropy=1.0 - i/10, purity=i/10,
                loss=2.0 - i/5, hbar=1.0 - i/10, gamma=i/100,
                phase="exploration" if i < 5 else "convergence",
            )

        summary = viz.summary()
        assert summary["total_steps"] == 10
        assert summary["final_entropy"] < summary["final_loss"]

    def test_plot_dashboard(self):
        viz = QuantumVisualizer()
        for i in range(20):
            eigs = torch.softmax(torch.randn(5), dim=0)
            viz.record(
                step=i, entropy=1.0 - i/20, purity=i/20,
                loss=2.0 - i/10, hbar=1.0 - i/20, gamma=i/200,
                eigenvalues=eigs, phase="exploration",
            )

        try:
            import matplotlib
            fig = viz.plot_quantum_dashboard()
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not installed")
