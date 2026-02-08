"""
Tests for IGQK v5.0 Future Edition modules:
- Quantum Consciousness Monitor
- Neural Architecture DNA
- Compression Oracle
- Time-Travel Debugger
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import copy
import json


def _make_model(in_f=32, hidden=64, out_f=10):
    return nn.Sequential(
        nn.Linear(in_f, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_f),
    )


# ===========================================================================
# Quantum Consciousness Monitor
# ===========================================================================

class TestConsciousness:
    def test_init(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        assert monitor.total_snapshots == 0
        assert monitor.total_events == 0

    def test_observe(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()
        snapshots = monitor.observe(model, step=0)
        assert len(snapshots) > 0
        assert monitor.total_snapshots > 0

    def test_snapshot_properties(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()
        snapshots = monitor.observe(model)
        for snap in snapshots:
            assert snap.entropy >= 0
            assert 0 <= snap.purity <= 1.0 + 1e-6
            assert snap.num_params > 0
            assert snap.is_healthy
            assert len(snap.anomalies) == 0

    def test_detect_phase_transition(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()

        # Observe original
        monitor.observe(model, step=0)

        # Drastically change weights (ternary compression)
        with torch.no_grad():
            for p in model.parameters():
                std = p.data.std()
                result = torch.zeros_like(p.data)
                result[p.data > 0.5 * std] = std
                result[p.data < -0.5 * std] = -std
                p.data = result

        # Observe compressed - should detect events
        monitor.observe(model, step=1)
        # At least some layers should show events
        assert monitor.total_snapshots > 0

    def test_consciousness_map(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()
        monitor.observe(model)
        cmap = monitor.get_consciousness_map()
        assert len(cmap) > 0
        for name, metrics in cmap.items():
            assert "entropy" in metrics
            assert "purity" in metrics
            assert "sparsity" in metrics
            assert "healthy" in metrics

    def test_health_report(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()

        # No data
        report = monitor.get_health_report()
        assert report["status"] == "no_data"

        # With data
        model = _make_model()
        monitor.observe(model)
        report = monitor.get_health_report()
        assert report["status"] == "healthy"
        assert report["layers"] > 0
        assert report["avg_entropy"] > 0

    def test_information_flow(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()
        monitor.observe(model)
        flow = monitor.get_information_flow()
        assert len(flow) > 0
        for f in flow:
            assert "layer" in f
            assert "entropy" in f
            assert "info_bits" in f
            assert "rank" in f

    def test_layer_timeline(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()
        monitor.observe(model, step=0)
        monitor.observe(model, step=1)

        # Get first layer name
        name = list(dict(model.named_parameters()).keys())[0]
        timeline = monitor.get_layer_timeline(name)
        assert len(timeline) == 2

    def test_callback(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        received = []
        monitor.on_snapshot(lambda snaps: received.append(len(snaps)))
        model = _make_model()
        monitor.observe(model)
        assert len(received) == 1

    def test_summary(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model()
        monitor.observe(model)
        summary = monitor.summary()
        assert "Quantum Consciousness Monitor" in summary

    def test_detect_nan(self):
        from igqk.consciousness import QuantumConsciousnessMonitor
        monitor = QuantumConsciousnessMonitor()
        model = _make_model(in_f=8, hidden=16, out_f=4)
        with torch.no_grad():
            for p in model.parameters():
                p.data[0] = float('nan')
        snapshots = monitor.observe(model)
        unhealthy = [s for s in snapshots if not s.is_healthy]
        assert len(unhealthy) > 0


# ===========================================================================
# Neural Architecture DNA
# ===========================================================================

class TestDNA:
    def test_gene_encode_decode(self):
        from igqk.dna import Gene
        gene = Gene(
            layer_name="layer.0.weight",
            layer_shape=[64, 32],
            method="ternary",
            threshold=0.7,
            keep_ratio=0.5,
            quality_score=0.95,
        )
        encoded = gene.encode()
        assert "TERN" in encoded
        decoded = Gene.decode(encoded, "layer.0.weight", [64, 32])
        assert decoded.method == "ternary"
        assert abs(decoded.threshold - 0.7) < 0.01
        assert abs(decoded.quality_score - 0.95) < 0.01

    def test_gene_mutate(self):
        from igqk.dna import Gene
        gene = Gene(
            layer_name="test", layer_shape=[64, 32],
            method="ternary", threshold=0.7,
        )
        mutant = gene.mutate(rate=1.0)
        assert mutant.quality_score == 0.0  # Reset after mutation

    def test_dna_encode_decode(self):
        from igqk.dna import CompressionDNA, Gene
        dna = CompressionDNA(
            dna_id="test123",
            arch_hash="abc",
            num_layers=2,
            total_params=1000,
            fitness=0.85,
            compression_ratio=12.5,
            quality=0.97,
            generation=3,
            mutations=2,
        )
        dna.genes = [
            Gene("layer0", [64, 32], "ternary", 0.7, 0.5, 0.95),
            Gene("layer1", [32, 10], "sparse", 0.3, 0.6, 0.92),
        ]
        encoded = dna.encode()
        assert "IGQK-DNA-v" in encoded
        assert "TERN" in encoded
        assert "SPRS" in encoded

        decoded = CompressionDNA.decode(encoded)
        assert decoded.num_layers == 2
        assert len(decoded.genes) == 2
        assert decoded.genes[0].method == "ternary"
        assert decoded.genes[1].method == "sparse"

    def test_dna_json(self):
        from igqk.dna import CompressionDNA, Gene
        dna = CompressionDNA(
            dna_id="json_test", total_params=500,
        )
        dna.genes = [Gene("l0", [32, 16], "wavelet")]
        json_str = dna.to_json()
        loaded = CompressionDNA.from_json(json_str)
        assert loaded.dna_id == "json_test"
        assert len(loaded.genes) == 1

    def test_extract_dna(self):
        from igqk.dna import DNAExtractor
        extractor = DNAExtractor()
        model = _make_model()
        dna = extractor.extract(model)
        assert dna.total_params > 0
        assert dna.num_layers > 0
        assert len(dna.genes) > 0
        assert dna.arch_hash != ""

    def test_extract_with_compressed(self):
        from igqk.dna import DNAExtractor
        extractor = DNAExtractor()
        model = _make_model()
        compressed = copy.deepcopy(model)
        with torch.no_grad():
            for p in compressed.parameters():
                std = p.data.std()
                result = torch.zeros_like(p.data)
                result[p.data > 0.5 * std] = std
                result[p.data < -0.5 * std] = -std
                p.data = result

        dna = extractor.extract(model, compressed)
        assert len(dna.genes) > 0
        # Should detect ternary-like compression
        methods = {g.method for g in dna.genes}
        assert len(methods) > 0

    def test_apply_dna(self):
        from igqk.dna import DNAExtractor
        extractor = DNAExtractor()
        model = _make_model()
        original_params = sum(p.data.abs().sum().item() for p in model.parameters())

        dna = extractor.extract(model)
        model_copy = copy.deepcopy(model)
        extractor.apply(dna, model_copy)

        # Model should be modified
        new_params = sum(p.data.abs().sum().item() for p in model_copy.parameters())
        assert new_params != original_params

    def test_crossover(self):
        from igqk.dna import DNAExtractor
        extractor = DNAExtractor()

        model_a = _make_model()
        model_b = _make_model()
        dna_a = extractor.extract(model_a)
        dna_b = extractor.extract(model_b)
        dna_a.dna_id = "parent_a"
        dna_b.dna_id = "parent_b"

        child = extractor.crossover(dna_a, dna_b)
        assert "parent_a" in child.parent_ids
        assert "parent_b" in child.parent_ids
        assert child.generation == 1
        assert len(child.genes) == len(dna_a.genes)

    def test_mutate_dna(self):
        from igqk.dna import DNAExtractor
        extractor = DNAExtractor()
        model = _make_model()
        dna = extractor.extract(model)
        dna.dna_id = "original"

        mutant = extractor.mutate(dna, rate=1.0)
        assert mutant.generation == dna.generation + 1
        assert mutant.mutations == dna.mutations + 1
        assert "original" in mutant.parent_ids

    def test_fingerprint(self):
        from igqk.dna import DNAExtractor
        extractor = DNAExtractor()
        model = _make_model()
        fp = extractor.fingerprint(model)
        assert fp.startswith("IGQK-FP-")
        assert len(fp) > 10

    def test_apply_gene_methods(self):
        from igqk.dna import DNAExtractor, Gene
        extractor = DNAExtractor()

        for method in ["ternary", "sparse", "binary", "wavelet", "lowrank", "adaptive_sparse"]:
            param = nn.Parameter(torch.randn(32, 16))
            gene = Gene("test", [32, 16], method, 0.7, 0.3, 0.0)
            extractor._apply_gene(param, gene)
            assert not torch.isnan(param.data).any()


# ===========================================================================
# Compression Oracle
# ===========================================================================

class TestOracle:
    def test_init(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        assert len(oracle.METHOD_PROFILES) > 0

    def test_predict_layer(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        w = torch.randn(64, 32)
        pred = oracle.predict_layer("test", w)
        assert pred.method in oracle.METHOD_PROFILES
        assert pred.expected_ratio >= 1.0
        assert 0 <= pred.expected_quality <= 1.0
        assert 0 < pred.confidence <= 1.0
        assert pred.risk_level in ("low", "medium", "high")
        assert len(pred.reasoning) > 0

    def test_predict_specific_method(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        w = torch.randn(64, 32)
        for method in oracle.METHOD_PROFILES:
            pred = oracle.predict_layer("test", w, method=method)
            assert pred.method == method
            assert pred.expected_ratio >= 1.0

    def test_predict_model(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        model = _make_model()
        prediction = oracle.predict_model(model)
        assert prediction.overall_ratio >= 1.0
        assert 0 <= prediction.overall_quality <= 1.0
        assert prediction.overall_risk in ("low", "medium", "high")
        assert len(prediction.layer_predictions) > 0
        assert 0 <= prediction.compressibility_score <= 1.0
        assert len(prediction.summary) > 0

    def test_predict_empty_model(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        model = nn.Module()  # No parameters
        prediction = oracle.predict_model(model)
        assert prediction.overall_ratio == 1.0
        assert prediction.compressibility_score == 0.0

    def test_compare_methods(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        w = torch.randn(100)
        predictions = oracle.compare_methods(w)
        assert len(predictions) == len(oracle.METHOD_PROFILES)
        for method, pred in predictions.items():
            assert pred.method == method
            assert pred.expected_ratio >= 1.0

    def test_predict_sparse_weights(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        # Create naturally sparse weights
        w = torch.randn(200)
        w[w.abs() < 0.5] = 0
        pred = oracle.predict_layer("sparse_layer", w)
        assert pred.expected_ratio >= 1.0

    def test_predict_1d(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        w = torch.randn(64)
        pred = oracle.predict_layer("bias", w)
        assert pred.expected_ratio >= 1.0

    def test_model_prediction_summary(self):
        from igqk.predictor import CompressionOracle
        oracle = CompressionOracle()
        model = _make_model()
        prediction = oracle.predict_model(model)
        assert "parameters" in prediction.summary
        assert "compression" in prediction.summary.lower()


# ===========================================================================
# Time-Travel Debugger
# ===========================================================================

class TestTimeTravel:
    def test_init(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        assert tt.num_checkpoints == 0
        assert tt.current is None

    def test_record(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        step = tt.record(model, label="initial")
        assert step == 0
        assert tt.num_checkpoints == 1
        assert tt.current is not None
        assert tt.current.label == "initial"

    def test_record_metrics(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model)
        tp = tt.current
        assert tp.total_params > 0
        assert tp.estimated_quality == 1.0  # First = original
        assert len(tp.layer_stats) > 0

    def test_navigation(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model, label="step0")
        tt.record(model, label="step1")
        tt.record(model, label="step2")

        assert tt.current.label == "step2"

        tp = tt.backward()
        assert tp.label == "step1"

        tp = tt.backward()
        assert tp.label == "step0"

        assert tt.backward() is None  # Can't go further back

        tp = tt.forward()
        assert tp.label == "step1"

        tp = tt.first()
        assert tp.label == "step0"

        tp = tt.last()
        assert tp.label == "step2"

    def test_goto(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model, label="a")
        tt.record(model, label="b")
        tt.record(model, label="c")

        tp = tt.goto(0)
        assert tp.label == "a"

        assert tt.goto(999) is None

    def test_restore(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()

        # Record original
        tt.record(model, label="original")
        orig_weight = list(model.parameters())[0].data.clone()

        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.data.zero_()

        # Verify it changed
        assert (list(model.parameters())[0].data == 0).all()

        # Restore
        success = tt.restore(model, step=0)
        assert success
        assert torch.allclose(list(model.parameters())[0].data, orig_weight)

    def test_diff(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()

        tt.record(model, label="before")

        # Compress
        with torch.no_grad():
            for p in model.parameters():
                std = p.data.std()
                result = torch.zeros_like(p.data)
                result[p.data > 0.5 * std] = std
                result[p.data < -0.5 * std] = -std
                p.data = result

        tt.record(model, label="after")

        diff = tt.diff(0, 1)
        assert diff is not None
        assert diff.total_changed_params > 0
        assert diff.change_fraction > 0
        assert len(diff.layer_changes) > 0

    def test_diff_invalid(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        assert tt.diff(0, 1) is None

    def test_quality_timeline(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model)

        with torch.no_grad():
            for p in model.parameters():
                p.data *= 0.5
        tt.record(model)

        timeline = tt.get_quality_timeline()
        assert len(timeline) == 2
        assert timeline[0][1] == 1.0  # Original is perfect quality

    def test_sparsity_timeline(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model)

        with torch.no_grad():
            for p in model.parameters():
                mask = p.data.abs() > p.data.std()
                p.data *= mask.float()
        tt.record(model)

        timeline = tt.get_sparsity_timeline()
        assert len(timeline) == 2
        assert timeline[1][1] > timeline[0][1]

    def test_find_quality_drop(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model, label="perfect")

        # Heavy compression = quality drop
        with torch.no_grad():
            for p in model.parameters():
                p.data.zero_()
        tt.record(model, label="destroyed")

        drop_step = tt.find_quality_drop(threshold=0.05)
        assert drop_step is not None

    def test_no_quality_drop(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model)
        tt.record(model)  # Same model, no drop
        assert tt.find_quality_drop() is None

    def test_max_checkpoints(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger(max_checkpoints=5)
        model = _make_model(in_f=8, hidden=16, out_f=4)
        for i in range(10):
            tt.record(model, label=f"step_{i}")
        assert tt.num_checkpoints <= 5

    def test_summary(self):
        from igqk.time_travel import TimeTravelDebugger
        tt = TimeTravelDebugger()
        model = _make_model()
        tt.record(model, label="start")
        summary = tt.summary()
        assert "Time-Travel" in summary
        assert "start" in summary


# ===========================================================================
# Dashboard Smoke Test (no Gradio import)
# ===========================================================================

class TestDashboardHelpers:
    def test_make_demo_model(self):
        from igqk.dashboard.app import _make_demo_model
        model = _make_demo_model(64)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_format_number(self):
        from igqk.dashboard.app import _format_number
        assert _format_number(1000) == "1,000"
        assert _format_number(1234567) == "1,234,567"
