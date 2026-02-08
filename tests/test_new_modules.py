"""
Tests for IGQK v3.1 modules:
- ONNX Export
- Metrics Suite
- Plugin System
- LLM Optimizations
- Model Zoo
"""

import pytest
import torch
import torch.nn as nn


def _make_model(in_f=32, hidden=64, out_f=10):
    return nn.Sequential(
        nn.Linear(in_f, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_f),
    )


# ===========================================================================
# ONNX Export
# ===========================================================================

class TestONNXExporter:
    def test_init(self):
        from igqk.export import ONNXExporter
        model = _make_model()
        exporter = ONNXExporter(model, input_shape=(1, 32))
        assert exporter.opset_version == 17

    def test_export(self, tmp_path):
        pytest.importorskip("onnx")
        try:
            import torch.onnx  # noqa: F401
            torch.onnx.export  # Verify it's actually loadable
        except (ImportError, AttributeError):
            pytest.skip("torch.onnx not available in this environment")
        from igqk.export import ONNXExporter
        model = _make_model()
        exporter = ONNXExporter(model, input_shape=(1, 32))
        path = str(tmp_path / "model.onnx")
        result = exporter.export(path)
        assert result == path
        import os
        assert os.path.exists(path)

    def test_summary_without_export(self):
        from igqk.export import ONNXExporter
        model = _make_model()
        exporter = ONNXExporter(model, input_shape=(1, 32))
        summary = exporter.summary()
        assert "IGQK Model Export" in summary
        assert "total_params" in summary


# ===========================================================================
# Metrics Suite
# ===========================================================================

class TestCompressionMetrics:
    def test_evaluate_basic(self):
        from igqk.metrics import CompressionMetrics
        original = _make_model()
        import copy
        compressed = copy.deepcopy(original)
        # Apply simple compression
        with torch.no_grad():
            for p in compressed.parameters():
                p.data = torch.round(p.data * 10) / 10
        metrics = CompressionMetrics()
        report = metrics.evaluate(original, compressed)
        assert report.original_params > 0
        assert report.weight_distortion >= 0
        assert report.output_cosine_sim > 0.5  # Should be somewhat similar

    def test_print_report(self):
        from igqk.metrics import CompressionMetrics
        model = _make_model()
        import copy
        metrics = CompressionMetrics()
        report = metrics.evaluate(model, copy.deepcopy(model))
        text = metrics.print_report(report)
        assert "Compression Metrics Report" in text

    def test_compare(self):
        from igqk.metrics import CompressionMetrics, MetricsReport
        r1 = MetricsReport(model_name="m1", compression_method="ternary")
        r2 = MetricsReport(model_name="m2", compression_method="sparse")
        metrics = CompressionMetrics()
        text = metrics.compare([r1, r2])
        assert "Comparison" in text

    def test_metrics_report_dataclass(self):
        from igqk.metrics import MetricsReport
        r = MetricsReport(model_name="test")
        assert r.compression_ratio == 1.0
        assert r.bits_per_weight == 32.0

    def test_quantum_metrics(self):
        from igqk.metrics import CompressionMetrics
        model = _make_model()
        import copy
        metrics = CompressionMetrics()
        report = metrics.evaluate(model, copy.deepcopy(model))
        assert report.avg_entropy >= 0
        assert report.avg_purity >= 0

    def test_layer_details(self):
        from igqk.metrics import CompressionMetrics
        model = _make_model()
        import copy
        metrics = CompressionMetrics()
        report = metrics.evaluate(model, copy.deepcopy(model))
        assert len(report.layer_details) > 0
        for ld in report.layer_details:
            assert "name" in ld
            assert "params" in ld


# ===========================================================================
# Plugin System
# ===========================================================================

class TestPluginRegistry:
    def test_register_and_get(self):
        from igqk.plugins import PluginRegistry
        # Built-in plugins should exist
        fn = PluginRegistry.get("compression", "ternary_basic")
        assert callable(fn)

    def test_ternary_basic(self):
        from igqk.plugins import PluginRegistry
        fn = PluginRegistry.get("compression", "ternary_basic")
        w = torch.randn(100)
        result = fn(w)
        assert result.shape == w.shape
        assert result.unique().numel() <= 3

    def test_top_k_sparse(self):
        from igqk.plugins import PluginRegistry
        fn = PluginRegistry.get("compression", "top_k_sparse")
        w = torch.randn(100)
        result = fn(w, k=0.3)
        nonzero = (result != 0).sum().item()
        assert nonzero == 30

    def test_uniform_quantize(self):
        from igqk.plugins import PluginRegistry
        fn = PluginRegistry.get("compression", "uniform_quantize")
        w = torch.randn(100)
        result = fn(w, bits=4)
        assert result.unique().numel() <= 16

    def test_register_custom(self):
        from igqk.plugins import PluginRegistry

        @PluginRegistry.register("compression", "test_custom_xyz")
        def custom(weights):
            return weights * 0

        fn = PluginRegistry.get("compression", "test_custom_xyz")
        assert (fn(torch.ones(10)) == 0).all()
        PluginRegistry.unregister("compression", "test_custom_xyz")

    def test_list_plugins(self):
        from igqk.plugins import PluginRegistry
        plugins = PluginRegistry.list_plugins()
        assert "compression" in plugins
        assert len(plugins["compression"]) >= 3

    def test_has(self):
        from igqk.plugins import PluginRegistry
        assert PluginRegistry.has("compression", "ternary_basic")
        assert not PluginRegistry.has("compression", "nonexistent")

    def test_unknown_category(self):
        from igqk.plugins import PluginRegistry
        with pytest.raises(ValueError, match="Unknown category"):
            PluginRegistry.register("invalid_category", "test")

    def test_get_nonexistent(self):
        from igqk.plugins import PluginRegistry
        with pytest.raises(KeyError):
            PluginRegistry.get("compression", "nonexistent_method")

    def test_summary(self):
        from igqk.plugins import PluginRegistry
        summary = PluginRegistry.summary()
        assert "Plugin Registry" in summary

    def test_metric_plugin(self):
        from igqk.plugins import PluginRegistry
        fn = PluginRegistry.get("metric", "model_size_bytes")
        model = _make_model()
        size = fn(model)
        assert size > 0

    def test_sparsity_metric(self):
        from igqk.plugins import PluginRegistry
        fn = PluginRegistry.get("metric", "sparsity_ratio")
        model = _make_model()
        ratio = fn(model)
        assert 0 <= ratio <= 1


# ===========================================================================
# LLM Optimizations
# ===========================================================================

class TestKVCacheCompressor:
    def test_update_basic(self):
        from igqk.llm import KVCacheCompressor, KVCacheConfig
        config = KVCacheConfig(max_cache_size=50, compression_after=10)
        kv = KVCacheCompressor(config)
        for _ in range(20):
            k = torch.randn(1, 4, 1, 32)
            v = torch.randn(1, 4, 1, 32)
            full_k, full_v = kv.update(k, v)
        assert full_k.shape[2] == 20
        assert full_v.shape[2] == 20

    def test_eviction(self):
        from igqk.llm import KVCacheCompressor, KVCacheConfig
        config = KVCacheConfig(max_cache_size=10, compression_after=5, eviction_policy="lru")
        kv = KVCacheCompressor(config)
        for _ in range(15):
            k = torch.randn(1, 2, 1, 16)
            v = torch.randn(1, 2, 1, 16)
            full_k, full_v = kv.update(k, v)
        assert full_k.shape[2] <= 11  # Max cache + 1 tolerance

    def test_stats(self):
        from igqk.llm import KVCacheCompressor
        kv = KVCacheCompressor()
        for _ in range(5):
            kv.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
        stats = kv.stats()
        assert stats["size"] == 5
        assert stats["memory_mb"] > 0

    def test_reset(self):
        from igqk.llm import KVCacheCompressor
        kv = KVCacheCompressor()
        kv.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
        kv.reset()
        assert kv.stats()["size"] == 0


class TestLoRACompressor:
    def test_compress_weights(self):
        from igqk.llm import LoRACompressor
        lc = LoRACompressor()
        lora_A = torch.randn(16, 768)
        lora_B = torch.randn(768, 16)
        comp_A, comp_B, stats = lc.compress_lora_weights(lora_A, lora_B)
        assert comp_A.shape == lora_A.shape
        assert stats["compression_ratio"] > 1

    def test_compress_sparse(self):
        from igqk.llm import LoRACompressor, LoRAConfig
        config = LoRAConfig(compress_method="sparse")
        lc = LoRACompressor(config)
        lora_A = torch.randn(8, 128)
        lora_B = torch.randn(128, 8)
        comp_A, comp_B, stats = lc.compress_lora_weights(lora_A, lora_B)
        assert stats["method"] == "sparse"


class TestSpeculativeDecoder:
    def test_init(self):
        from igqk.llm import SpeculativeDecoder
        model = _make_model()
        sd = SpeculativeDecoder(model, num_speculative=3)
        assert sd.num_speculative == 3
        assert sd.draft_model is not None

    def test_acceptance_rate(self):
        from igqk.llm import SpeculativeDecoder
        sd = SpeculativeDecoder(_make_model())
        assert sd.acceptance_rate() == 0.0


class TestAttentionHeadPruner:
    def test_prune_no_attention(self):
        from igqk.llm import AttentionHeadPruner
        pruner = AttentionHeadPruner(prune_ratio=0.3)
        model = _make_model()
        pruned, stats = pruner.prune(model)
        assert stats["total_heads"] == 0


# ===========================================================================
# Model Zoo
# ===========================================================================

class TestModelZoo:
    def test_list_models(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        models = zoo.list_models()
        assert len(models) >= 8  # Built-in recipes

    def test_get_recipe(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        recipe = zoo.get_recipe("mnist_fc_ternary")
        assert recipe.compression_method == "ternary"
        assert recipe.in_features == 784

    def test_get_recipe_not_found(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        with pytest.raises(KeyError):
            zoo.get_recipe("nonexistent")

    def test_create_compressed_ternary(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        model, stats = zoo.create_compressed("mnist_fc_ternary")
        assert stats["method"] == "ternary"
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_create_compressed_sparse(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        model, stats = zoo.create_compressed("mnist_fc_sparse")
        assert stats["sparsity"] > 0

    def test_create_compressed_wavelet(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        model, stats = zoo.create_compressed("mnist_fc_wavelet")
        assert stats["method"] == "wavelet"

    def test_register_custom_recipe(self):
        from igqk.zoo import ModelZoo, ModelRecipe
        zoo = ModelZoo()
        custom = ModelRecipe(
            name="custom_test", architecture="fc_small",
            dataset="test", compression_method="ternary",
            description="Test", in_features=32, num_classes=5,
            hidden_layers=[64],
        )
        zoo.register_recipe(custom)
        assert zoo.get_recipe("custom_test").name == "custom_test"

    def test_summary(self):
        from igqk.zoo import ModelZoo
        zoo = ModelZoo()
        summary = zoo.summary()
        assert "Model Zoo" in summary
        assert "mnist_fc_ternary" in summary

    def test_model_recipe_dataclass(self):
        from igqk.zoo import ModelRecipe
        r = ModelRecipe(
            name="test", architecture="fc", dataset="test",
            compression_method="ternary", description="test",
        )
        assert r.hbar == 0.1
