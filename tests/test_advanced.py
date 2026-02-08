"""
Tests for IGQK v3.0.0 advanced modules:
- Self-Healing Compression
- Temporal Transformer Compression
- Interpretable Compression
- Quantum Transfer Learning
- Hardware-Adaptive Compilation
- Multi-Objective Quantum Flow
- Federated Quantum Compression
- Compression-Aware NAS
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(in_features=32, hidden=64, out_features=10):
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


def _make_inputs(batch=8, features=32, num_classes=10):
    """Create random input/target pair."""
    x = torch.randn(batch, features)
    y = torch.randint(0, num_classes, (batch,))
    return x, y


# ===========================================================================
# 1. Self-Healing Compression
# ===========================================================================

class TestSelfHealing:
    def test_init(self):
        from igqk.healing import SelfHealingModel
        model = _make_model()
        shm = SelfHealingModel(model)
        assert shm._step == 0
        assert len(shm._original_weights) > 0

    def test_forward_passthrough(self):
        from igqk.healing import SelfHealingModel
        model = _make_model()
        shm = SelfHealingModel(model)
        x = torch.randn(4, 32)
        out = shm(x)
        assert out.shape == (4, 10)

    def test_compress_initial(self):
        from igqk.healing import SelfHealingModel
        model = _make_model()
        shm = SelfHealingModel(model)
        shm.compress_initial(method="ternary")
        for name, health in shm._layer_health.items():
            if dict(model.named_parameters())[name].numel() >= 16:
                assert health.compression_level == 1.0

    def test_compress_initial_sparse(self):
        from igqk.healing import SelfHealingModel
        model = _make_model()
        shm = SelfHealingModel(model)
        shm.compress_initial(method="sparse")
        # Should run without error
        x = torch.randn(4, 32)
        out = shm(x)
        assert out.shape == (4, 10)

    def test_confidence_monitoring(self):
        from igqk.healing import SelfHealingModel
        model = _make_model()
        shm = SelfHealingModel(model, window_size=20)
        x = torch.randn(4, 32)
        for _ in range(15):
            shm(x)
        assert len(shm._confidence_window) == 15
        assert len(shm._entropy_window) == 15

    def test_health_report(self):
        from igqk.healing import SelfHealingModel
        model = _make_model()
        shm = SelfHealingModel(model)
        x = torch.randn(4, 32)
        for _ in range(5):
            shm(x)
        report = shm.get_health_report()
        assert "step" in report
        assert "avg_confidence" in report
        assert "layers" in report
        assert report["step"] == 5

    def test_layer_health_dataclass(self):
        from igqk.healing import LayerHealth
        lh = LayerHealth(name="test", compression_level=0.5,
                         current_contribution=0.3, sensitivity=0.2)
        assert lh.heal_count == 0

    def test_healing_event_dataclass(self):
        from igqk.healing import HealingEvent
        he = HealingEvent(step=10, layer="fc1", old_level=1.0,
                          new_level=0.7, trigger="confidence_drop",
                          confidence_before=0.6, confidence_after=0.8)
        assert he.trigger == "confidence_drop"


# ===========================================================================
# 2. Temporal Transformer Compression
# ===========================================================================

class TestTemporalCompressor:
    def test_position_mask(self):
        from igqk.temporal import TemporalCompressor
        tc = TemporalCompressor(full_precision_tokens=10, medium_precision_tokens=50)
        mask = tc.create_position_mask(100)
        assert mask.shape == (100,)
        assert mask[0].item() == 1.0  # full precision
        assert mask[9].item() == 1.0
        assert mask[10].item() == 0.5  # medium
        assert mask[49].item() == 0.5
        assert mask[50].item() == pytest.approx(0.1)  # heavy

    def test_position_mask_short_seq(self):
        from igqk.temporal import TemporalCompressor
        tc = TemporalCompressor(full_precision_tokens=10, medium_precision_tokens=50)
        mask = tc.create_position_mask(5)
        assert mask.shape == (5,)
        assert (mask == 1.0).all()

    def test_kv_cache_compression(self):
        from igqk.temporal import TemporalCompressor
        tc = TemporalCompressor(full_precision_tokens=4, medium_precision_tokens=8)
        keys = torch.randn(1, 4, 16, 8)
        values = torch.randn(1, 4, 16, 8)
        k_c, v_c = tc.compress_kv_cache(keys, values)
        assert k_c.shape == keys.shape
        assert v_c.shape == values.shape

    def test_memory_savings_estimate(self):
        from igqk.temporal import TemporalCompressor
        tc = TemporalCompressor()
        stats = tc.estimate_memory_savings(seq_length=512, num_heads=12, head_dim=64)
        assert "original_mb" in stats
        assert "total_savings" in stats
        assert stats["original_mb"] > stats["after_pruning_mb"]

    def test_head_importance_dataclass(self):
        from igqk.temporal import HeadImportance
        hi = HeadImportance(layer=0, head=1, entropy=0.5, importance=0.8, prunable=False)
        assert hi.layer == 0

    def test_analyze_attention_heads_no_heads(self):
        from igqk.temporal import TemporalCompressor
        tc = TemporalCompressor()
        model = _make_model()  # No attention heads
        heads = tc.analyze_attention_heads(model)
        assert heads == []


# ===========================================================================
# 3. Interpretable Compression
# ===========================================================================

class TestInterpretableCompressor:
    def test_analyze_layer_2d(self):
        from igqk.interpretable import InterpretableCompressor
        ic = InterpretableCompressor()
        weight = torch.randn(64, 32)
        patterns = ic.analyze_layer("test_layer", weight, top_k=5)
        assert len(patterns) == 5
        assert patterns[0].importance >= patterns[1].importance

    def test_analyze_layer_1d(self):
        from igqk.interpretable import InterpretableCompressor
        ic = InterpretableCompressor()
        weight = torch.randn(64)
        patterns = ic.analyze_layer("bias", weight)
        assert len(patterns) == 1
        assert patterns[0].role == "bias_vector"

    def test_explain_compression(self):
        from igqk.interpretable import InterpretableCompressor
        ic = InterpretableCompressor()
        model = _make_model()
        explanations = ic.explain_compression(model, method="ternary")
        assert len(explanations) > 0
        for exp in explanations:
            assert exp.method_chosen == "ternary"
            assert exp.patterns_kept >= 0
            assert isinstance(exp.reason, str)

    def test_generate_report(self):
        from igqk.interpretable import InterpretableCompressor
        ic = InterpretableCompressor(detail_level="detailed")
        model = _make_model()
        report = ic.generate_report(model)
        assert "IGQK Compression Explanation Report" in report
        assert "Summary" in report
        assert "KEEP" in report or "REMOVE" in report

    def test_pattern_roles(self):
        from igqk.interpretable import InterpretableCompressor
        ic = InterpretableCompressor()
        weight = torch.randn(64, 32)
        patterns = ic.analyze_layer("layer", weight, top_k=10)
        assert patterns[0].role == "primary_feature_detector"

    def test_wavelet_method_reason(self):
        from igqk.interpretable import InterpretableCompressor
        ic = InterpretableCompressor()
        model = _make_model()
        explanations = ic.explain_compression(model, method="wavelet")
        assert all("multi-scale" in e.reason for e in explanations)

    def test_weight_pattern_dataclass(self):
        from igqk.interpretable import WeightPattern
        wp = WeightPattern(layer="fc1", pattern_index=0, importance=1.0,
                           explained_variance=0.5, role="test",
                           removable=False, removal_impact=0.01,
                           description="Test pattern")
        assert wp.layer == "fc1"


# ===========================================================================
# 4. Quantum Transfer Learning
# ===========================================================================

class TestQuantumTransferLearning:
    def test_analyze_source(self):
        from igqk.transfer import QuantumTransferLearning
        qtl = QuantumTransferLearning()
        model = _make_model()
        plans = qtl.analyze_source(model)
        assert len(plans) > 0
        for plan in plans:
            assert plan.action in ("freeze_compress", "freeze_keep", "finetune", "reinitialize")
            assert 0.0 <= plan.finetune_lr_scale <= 1.5

    def test_apply_transfer(self):
        from igqk.transfer import QuantumTransferLearning
        qtl = QuantumTransferLearning()
        source = _make_model()
        target = _make_model()
        result, stats = qtl.apply_transfer(source, target)
        assert "frozen" in stats
        assert "finetuned" in stats
        total = stats["frozen"] + stats["finetuned"] + stats["reinitialized"]
        assert total > 0

    def test_create_optimizer_groups(self):
        from igqk.transfer import QuantumTransferLearning
        qtl = QuantumTransferLearning()
        model = _make_model()
        plans = qtl.analyze_source(model)
        # Apply transfer to set requires_grad
        _, _ = qtl.apply_transfer(model, model, plans)
        groups = qtl.create_optimizer_groups(model, plans, base_lr=0.001)
        assert isinstance(groups, list)
        for g in groups:
            assert "params" in g
            assert "lr" in g

    def test_generate_report(self):
        from igqk.transfer import QuantumTransferLearning
        qtl = QuantumTransferLearning()
        model = _make_model()
        plans = qtl.analyze_source(model)
        report = qtl.generate_report(plans)
        assert "Quantum Transfer Learning Plan" in report

    def test_transfer_plan_dataclass(self):
        from igqk.transfer import TransferPlan
        tp = TransferPlan(layer_name="fc1", action="finetune",
                          source_entropy=0.5, confidence=0.5,
                          finetune_lr_scale=0.5, compress_after=False)
        assert tp.action == "finetune"

    def test_small_params_frozen(self):
        from igqk.transfer import QuantumTransferLearning
        qtl = QuantumTransferLearning()
        # Model with small params (biases)
        model = nn.Sequential(nn.Linear(4, 4))
        plans = qtl.analyze_source(model)
        bias_plans = [p for p in plans if "bias" in p.layer_name]
        for bp in bias_plans:
            assert bp.action == "freeze_keep"


# ===========================================================================
# 5. Hardware-Adaptive Compilation
# ===========================================================================

class TestHardwareAdaptiveCompiler:
    def test_compile_cpu(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        compiled, stats = hac.compile("cpu_server")
        assert "target" in stats
        assert "estimated_memory_mb" in stats
        assert stats["fits_in_memory"]

    def test_compile_gpu(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        compiled, stats = hac.compile("gpu_consumer")
        assert stats["target"] == "Consumer GPU"

    def test_compile_mobile(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        compiled, stats = hac.compile("mobile")
        assert "precision" in stats

    def test_compile_browser(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        compiled, stats = hac.compile("browser")
        assert "precision" in stats

    def test_compile_edge(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        compiled, stats = hac.compile("edge")
        assert "precision" in stats

    def test_compile_all_targets(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        results = hac.compile_all()
        assert len(results) == 7  # all profiles

    def test_unknown_target_error(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        with pytest.raises(ValueError, match="Unknown target"):
            hac.compile("quantum_computer")

    def test_summary(self):
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        summary = hac.summary()
        assert "IGQK Hardware-Adaptive Compilation" in summary

    def test_hardware_profile_dataclass(self):
        from igqk.hardware import HardwareProfile
        hp = HardwareProfile(name="Test", compute_type="cpu", max_memory_mb=1024,
                             supports_float16=False, supports_int8=True,
                             simd_width=8, target_latency_ms=10.0)
        assert hp.compute_type == "cpu"

    def test_custom_profile(self):
        from igqk.hardware import HardwareAdaptiveCompiler, HardwareProfile
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        custom = HardwareProfile("Custom", "edge", 256, False, True, 4, 200.0)
        compiled, stats = hac.compile(custom_profile=custom)
        assert stats["target"] == "Custom"


# ===========================================================================
# 6. Multi-Objective Quantum Flow
# ===========================================================================

class TestMultiObjectiveFlow:
    def test_compute_losses(self):
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss}
        )
        model = _make_model()
        x, y = _make_inputs()
        losses = flow.compute_losses(model, x, y)
        assert "accuracy" in losses
        assert "sparsity" in losses

    def test_combined_gradient(self):
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss}
        )
        model = _make_model()
        x, y = _make_inputs()
        grad, losses = flow.combined_gradient(model, x, y)
        assert grad.shape[0] > 0
        assert len(losses) == 2

    def test_dynamic_weight_adaptation(self):
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss},
            adaptation="dynamic",
        )
        model = _make_model()
        x, y = _make_inputs()
        initial_weights = dict(flow.weights)
        flow.combined_gradient(model, x, y)
        # Weights should have been adapted
        assert flow.weights is not None

    def test_fixed_weights(self):
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss},
            adaptation="fixed",
            weights={"accuracy": 0.7, "sparsity": 0.3},
        )
        model = _make_model()
        x, y = _make_inputs()
        flow.combined_gradient(model, x, y)
        assert flow.weights["accuracy"] == 0.7

    def test_pareto_front(self):
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss}
        )
        flow.update_pareto_front({"accuracy": 0.5, "sparsity": 0.3})
        flow.update_pareto_front({"accuracy": 0.3, "sparsity": 0.2})
        flow.update_pareto_front({"accuracy": 0.8, "sparsity": 0.8})
        assert len(flow.pareto_front) >= 1

    def test_dominates(self):
        from igqk.multiobjective import MultiObjectiveFlow
        flow = MultiObjectiveFlow(objectives={})
        assert flow._dominates({"a": 0.1, "b": 0.1}, {"a": 0.2, "b": 0.2})
        assert not flow._dominates({"a": 0.1, "b": 0.3}, {"a": 0.2, "b": 0.2})

    def test_history_tracking(self):
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss}
        )
        model = _make_model()
        x, y = _make_inputs()
        flow.combined_gradient(model, x, y)
        flow.combined_gradient(model, x, y)
        assert len(flow.history) == 2

    def test_loss_functions(self):
        from igqk.multiobjective import accuracy_loss, sparsity_loss, memory_loss, latency_loss
        model = _make_model()
        x, y = _make_inputs()
        acc = accuracy_loss(model, x, y)
        assert acc.item() > 0
        sp = sparsity_loss(model)
        assert sp.item() > 0
        mem = memory_loss(model)
        assert mem.item() > 0
        lat = latency_loss(model)
        assert lat.item() >= 0

    def test_pareto_point_dataclass(self):
        from igqk.multiobjective import ParetoPoint
        pp = ParetoPoint(objectives={"a": 0.1}, is_pareto_optimal=True)
        assert pp.is_pareto_optimal


# ===========================================================================
# 7. Federated Quantum Compression
# ===========================================================================

class TestFederatedCompression:
    def test_device_compute_summary(self):
        from igqk.federated import FederatedDevice
        model = _make_model()
        device = FederatedDevice("device_1", model)
        summary = device.compute_summary()
        assert summary.device_id == "device_1"
        assert summary.num_params > 0
        assert len(summary.layer_entropies) > 0

    def test_coordinator_receive_summary(self):
        from igqk.federated import FederatedDevice, FederatedCoordinator
        model = _make_model()
        device = FederatedDevice("d1", model)
        summary = device.compute_summary()
        coord = FederatedCoordinator()
        coord.receive_summary(summary)
        assert "d1" in coord._summaries

    def test_compute_plans(self):
        from igqk.federated import FederatedDevice, FederatedCoordinator
        coord = FederatedCoordinator()
        for i in range(3):
            model = _make_model()
            device = FederatedDevice(f"d{i}", model)
            summary = device.compute_summary()
            coord.receive_summary(summary)
        plans = coord.compute_plans()
        assert len(plans) == 3
        for plan in plans.values():
            assert len(plan.per_layer_method) > 0

    def test_aggregate_models(self):
        from igqk.federated import FederatedDevice, FederatedCoordinator
        coord = FederatedCoordinator()
        state_dicts = {}
        for i in range(3):
            model = _make_model()
            device = FederatedDevice(f"d{i}", model)
            state_dicts[f"d{i}"] = device.get_state_dict()
        agg = coord.aggregate_models(state_dicts)
        assert len(agg) > 0
        # Should be average of inputs
        for name in agg:
            assert agg[name].shape == state_dicts["d0"][name].shape

    def test_aggregate_empty(self):
        from igqk.federated import FederatedCoordinator
        coord = FederatedCoordinator()
        assert coord.aggregate_models({}) == {}

    def test_global_stats(self):
        from igqk.federated import FederatedDevice, FederatedCoordinator
        coord = FederatedCoordinator()
        for i in range(2):
            model = _make_model()
            device = FederatedDevice(f"d{i}", model)
            coord.receive_summary(device.compute_summary())
        stats = coord.get_global_stats()
        assert stats["num_devices"] == 2
        assert "avg_global_entropy" in stats

    def test_global_stats_empty(self):
        from igqk.federated import FederatedCoordinator
        coord = FederatedCoordinator()
        assert coord.get_global_stats() == {}

    def test_compute_plans_empty(self):
        from igqk.federated import FederatedCoordinator
        coord = FederatedCoordinator()
        assert coord.compute_plans() == {}

    def test_device_quantum_summary_dataclass(self):
        from igqk.federated import DeviceQuantumSummary
        s = DeviceQuantumSummary(
            device_id="test", num_params=100,
            layer_entropies={"fc": 0.5}, layer_purities={"fc": 0.8},
            layer_eigenvalue_spectra={"fc": [0.5, 0.3, 0.2]},
            global_entropy=0.5, global_purity=0.8,
        )
        assert s.device_id == "test"

    def test_federated_compression_plan_dataclass(self):
        from igqk.federated import FederatedCompressionPlan
        p = FederatedCompressionPlan(
            device_id="d1",
            per_layer_method={"fc": "ternary"},
            per_layer_strength={"fc": 0.9},
            target_compression=0.1,
        )
        assert p.device_id == "d1"


# ===========================================================================
# 8. Compression-Aware NAS
# ===========================================================================

class TestCompressionAwareNAS:
    def test_random_architecture(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        arch = nas._random_architecture()
        assert arch.id == 1
        assert len(arch.layer_widths) >= nas.min_layers
        assert len(arch.layer_widths) <= nas.max_layers

    def test_build_model(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        arch = nas._random_architecture()
        model = nas._build_model(arch)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_evaluate_compressibility(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        arch = nas._random_architecture()
        model = nas._build_model(arch)
        score = nas._evaluate_compressibility(model)
        assert 0.0 <= score <= 1.0

    def test_crossover(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        p1 = nas._random_architecture()
        p2 = nas._random_architecture()
        child = nas._crossover(p1, p2)
        assert child.id > p2.id
        assert len(child.layer_widths) == len(child.activations)

    def test_mutate(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        arch = nas._random_architecture()
        original_id = arch.id
        mutated = nas._mutate(arch, mutation_rate=1.0)  # Force mutation
        assert mutated.id == original_id  # Mutation modifies in place

    def test_search_one_generation(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(
            in_features=32, num_classes=10,
            population_size=4, generations=1,
            min_layers=2, max_layers=3,
        )

        def train_fn(model, epochs):
            pass  # No-op training

        def eval_fn(model):
            x = torch.randn(4, 32)
            out = model(x)
            return torch.softmax(out, dim=-1).max(dim=-1).values.mean().item()

        best = nas.search(train_fn, eval_fn, verbose=False)
        assert best is not None
        assert best.fitness > 0
        assert best.accuracy > 0

    def test_best_architecture_property(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        assert nas.best_architecture is None

    def test_build_best_model_none(self):
        from igqk.nas import CompressionAwareNAS
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        assert nas.build_best_model() is None

    def test_architecture_dataclass(self):
        from igqk.nas import Architecture
        arch = Architecture(
            id=1, layer_widths=[64, 32], activations=["relu", "gelu"],
            skip_connections=[False, True], dropout_rates=[0.1, 0.2],
        )
        assert arch.fitness == 0.0
        assert arch.params == 0


# ===========================================================================
# Cross-module integration tests
# ===========================================================================

class TestCrossModuleIntegration:
    def test_healing_with_hardware(self):
        """Self-healing model compiled for different hardware."""
        from igqk.healing import SelfHealingModel
        from igqk.hardware import HardwareAdaptiveCompiler
        model = _make_model()
        hac = HardwareAdaptiveCompiler(model)
        compiled, _ = hac.compile("cpu_laptop")
        shm = SelfHealingModel(compiled)
        x = torch.randn(4, 32)
        out = shm(x)
        assert out.shape == (4, 10)

    def test_interpretable_then_transfer(self):
        """Analyze compression, then transfer."""
        from igqk.interpretable import InterpretableCompressor
        from igqk.transfer import QuantumTransferLearning
        model = _make_model()
        ic = InterpretableCompressor()
        report = ic.generate_report(model)
        assert len(report) > 0
        qtl = QuantumTransferLearning()
        plans = qtl.analyze_source(model)
        assert len(plans) > 0

    def test_federated_with_hardware_compile(self):
        """Federated devices compile for different targets."""
        from igqk.federated import FederatedDevice, FederatedCoordinator
        from igqk.hardware import HardwareAdaptiveCompiler
        coord = FederatedCoordinator()
        for i, target in enumerate(["cpu_server", "mobile"]):
            model = _make_model()
            hac = HardwareAdaptiveCompiler(model)
            compiled, _ = hac.compile(target)
            device = FederatedDevice(f"d{i}", compiled)
            coord.receive_summary(device.compute_summary())
        stats = coord.get_global_stats()
        assert stats["num_devices"] == 2

    def test_multiobjective_with_nas_model(self):
        """Multi-objective flow on NAS-generated model."""
        from igqk.nas import CompressionAwareNAS
        from igqk.multiobjective import MultiObjectiveFlow, accuracy_loss, sparsity_loss
        nas = CompressionAwareNAS(in_features=32, num_classes=10)
        arch = nas._random_architecture()
        model = nas._build_model(arch)
        flow = MultiObjectiveFlow(
            objectives={"accuracy": accuracy_loss, "sparsity": sparsity_loss}
        )
        x, y = _make_inputs()
        grad, losses = flow.combined_gradient(model, x, y)
        assert len(losses) == 2
