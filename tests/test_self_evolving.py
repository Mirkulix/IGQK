"""
Tests for IGQK v4.0 Self-Evolving AI modules:
- Meta-Learning Compression
- Self-Evolving Strategy Engine
- Auto-Discovery Engine
- AI-to-AI Knowledge Transfer Protocol
- Autonomous Pipeline
"""

import pytest
import torch
import torch.nn as nn
import json
import os
import copy


def _make_model(in_f=32, hidden=64, out_f=10):
    return nn.Sequential(
        nn.Linear(in_f, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_f),
    )


# ===========================================================================
# Meta-Learning Compression
# ===========================================================================

class TestMetaLearner:
    def test_init_memory_only(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta.json")
        assert ml.knowledge.total_compressions == 0

    def test_analyze_layer(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta2.json")
        w = torch.randn(64, 32)
        features = ml.analyze_layer("test_layer", w)
        assert "mean" in features
        assert "std" in features
        assert "entropy" in features
        assert "sparsity" in features
        assert "sv_ratio" in features
        assert features["sv_ratio"] >= 1.0

    def test_analyze_layer_1d(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta3.json")
        w = torch.randn(64)
        features = ml.analyze_layer("bias", w)
        assert features["sv_ratio"] == 1.0

    def test_predict_default(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta4.json")
        method, conf, reason = ml.predict_best_method("layer", torch.randn(64, 32))
        assert method in ("ternary", "sparse", "lowrank", "wavelet")
        assert 0 <= conf <= 1
        assert len(reason) > 0

    def test_record_experience(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta5.json")
        model = _make_model()
        w_before = torch.randn(64, 32)
        w_after = torch.zeros(64, 32)  # Fully compressed
        ml.record_experience(model, "test.weight", w_before, w_after, "ternary")
        assert ml.knowledge.total_compressions == 1
        assert len(ml.experiences) == 1
        assert ml.experiences[0].method == "ternary"
        # Cleanup
        if os.path.exists("/tmp/igqk_test_meta5.json"):
            os.remove("/tmp/igqk_test_meta5.json")

    def test_knowledge_persistence(self, tmp_path):
        from igqk.meta_learner import MetaLearner
        path = str(tmp_path / "knowledge.json")

        # Create and save
        ml1 = MetaLearner(knowledge_path=path)
        model = _make_model()
        w = torch.randn(64, 32)
        ml1.record_experience(model, "layer", w, w * 0.5, "ternary")
        assert os.path.exists(path)

        # Load in new instance
        ml2 = MetaLearner(knowledge_path=path)
        assert ml2.knowledge.total_compressions == 1

    def test_predict_after_learning(self, tmp_path):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path=str(tmp_path / "k.json"))
        model = _make_model()

        # Record several experiences
        for _ in range(5):
            w = torch.randn(64, 32)
            ml.record_experience(model, "layer", w, w * 0.1, "sparse")

        method, conf, reason = ml.predict_best_method("layer", torch.randn(64, 32))
        # Should have learned something
        assert ml.knowledge.total_compressions == 5

    def test_shape_class(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta_sc.json")
        assert ml._shape_class([64]) == "bias"
        assert "square" in ml._shape_class([64, 64])
        assert "tall" in ml._shape_class([128, 64])
        assert "wide" in ml._shape_class([64, 128])
        assert "conv" in ml._shape_class([32, 16, 3, 3])

    def test_get_wisdom(self):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path="/tmp/igqk_test_meta_w.json")
        wisdom = ml.get_wisdom()
        assert "Meta-Learning Knowledge" in wisdom

    def test_compress_with_learning(self, tmp_path):
        from igqk.meta_learner import MetaLearner
        ml = MetaLearner(knowledge_path=str(tmp_path / "cwl.json"))
        model = _make_model()
        compressed, stats = ml.compress_with_learning(model)
        assert stats["total_models_seen"] == 1
        assert len(stats["layers"]) > 0
        assert stats["total_experiences"] > 0


# ===========================================================================
# Self-Evolving Strategy Engine
# ===========================================================================

class TestEvolutionEngine:
    def test_init(self):
        from igqk.evolution_engine import EvolutionEngine
        ee = EvolutionEngine(population_size=10, generations=3)
        assert len(ee.population) == 10

    def test_strategy_mutate(self):
        from igqk.evolution_engine import Strategy
        s = Strategy(id=1, name="test")
        child = s.mutate(rate=1.0)  # High rate forces mutations
        assert child.generation == s.generation + 1
        assert child.fitness == 0.0

    def test_strategy_crossover(self):
        from igqk.evolution_engine import Strategy
        a = Strategy(id=1, name="a", blend_ternary=1.0, blend_sparse=0.0)
        b = Strategy(id=2, name="b", blend_ternary=0.0, blend_sparse=1.0)
        child = Strategy.crossover(a, b, child_id=3)
        assert child.id == 3
        assert child.generation == 1
        # Blend should be average
        assert child.blend_ternary == 0.5
        assert child.blend_sparse == 0.5

    def test_evaluate_strategy(self):
        from igqk.evolution_engine import EvolutionEngine, Strategy
        ee = EvolutionEngine(population_size=5, generations=1)
        s = Strategy(id=99, name="test", blend_ternary=1.0)
        w = torch.randn(100)
        fitness = ee.evaluate_strategy(s, w)
        assert fitness > 0
        assert s.compression_ratio >= 1.0
        assert s.distortion >= 0

    def test_evolve(self):
        from igqk.evolution_engine import EvolutionEngine
        ee = EvolutionEngine(population_size=8, generations=3)
        w = torch.randn(200)
        best = ee.evolve(w)
        assert best.fitness > 0
        assert len(ee.hall_of_fame) > 0

    def test_evolve_verbose(self, capsys):
        from igqk.evolution_engine import EvolutionEngine
        ee = EvolutionEngine(population_size=6, generations=2)
        best = ee.evolve(torch.randn(100), verbose=True)
        captured = capsys.readouterr()
        assert "Gen" in captured.out

    def test_best_strategy(self):
        from igqk.evolution_engine import EvolutionEngine
        ee = EvolutionEngine(population_size=6, generations=2)
        assert ee.best_strategy() is None
        ee.evolve(torch.randn(100))
        assert ee.best_strategy() is not None

    def test_summary(self):
        from igqk.evolution_engine import EvolutionEngine
        ee = EvolutionEngine(population_size=6, generations=2)
        ee.evolve(torch.randn(100))
        summary = ee.summary()
        assert "Self-Evolving" in summary
        assert "Hall of Fame" in summary

    def test_apply_strategy_sparse(self):
        from igqk.evolution_engine import EvolutionEngine, Strategy
        ee = EvolutionEngine(population_size=5, generations=1)
        s = Strategy(id=1, name="sparse", blend_ternary=0.0, blend_sparse=1.0,
                     sparse_keep_ratio=0.2)
        w = torch.randn(100)
        result = ee._apply_strategy(s, w)
        # Should have many zeros
        zeros = (result == 0).float().mean().item()
        assert zeros > 0.5

    def test_apply_strategy_wavelet(self):
        from igqk.evolution_engine import EvolutionEngine, Strategy
        ee = EvolutionEngine(population_size=5, generations=1)
        s = Strategy(id=1, name="wavelet", blend_ternary=0.0, blend_sparse=0.0,
                     blend_wavelet=1.0, wavelet_keep_ratio=0.3)
        w = torch.randn(100)
        result = ee._apply_strategy(s, w)
        assert result.shape == w.shape

    def test_two_stage(self):
        from igqk.evolution_engine import EvolutionEngine, Strategy
        ee = EvolutionEngine(population_size=5, generations=1)
        s = Strategy(id=1, name="two_stage", blend_ternary=1.0,
                     use_two_stage=True)
        w = torch.randn(100)
        result = ee._apply_strategy(s, w)
        assert result.shape == w.shape


# ===========================================================================
# Auto-Discovery Engine
# ===========================================================================

class TestAutoDiscovery:
    def test_init(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        assert ad.num_observations == 0
        assert len(ad.patterns) == 0

    def test_observe(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        model = _make_model()
        ad.observe(model, "test_model")
        assert ad.num_observations > 0

    def test_compute_statistics(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        flat = torch.randn(1000)
        stats = ad._compute_statistics(flat)
        assert "mean" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "modality" in stats
        assert "entropy" in stats
        assert "sparsity" in stats
        assert "tail_weight" in stats

    def test_count_modes(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        # Unimodal
        hist_uni = np.array([1, 3, 7, 10, 7, 3, 1], dtype=float)
        assert ad._count_modes(hist_uni) >= 1
        # Short
        assert ad._count_modes(np.array([5, 10])) == 1

    def test_discover_patterns(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        # Observe multiple models
        for _ in range(5):
            model = _make_model(hidden=128)
            ad.observe(model)
        patterns = ad.discover_patterns(min_observations=3)
        assert len(patterns) > 0
        for p in patterns:
            assert p.name != ""
            assert p.frequency >= 3

    def test_discover_no_data(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        patterns = ad.discover_patterns()
        assert len(patterns) == 0

    def test_classify_layer(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        for _ in range(5):
            ad.observe(_make_model(hidden=128))
        ad.discover_patterns()

        w = torch.randn(64, 32)
        pattern = ad.classify_layer(w)
        # May or may not match, but shouldn't crash
        if pattern is not None:
            assert pattern.name != ""

    def test_classify_no_patterns(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        result = ad.classify_layer(torch.randn(64, 32))
        assert result is None

    def test_summary(self):
        from igqk.auto_discovery import AutoDiscovery
        ad = AutoDiscovery()
        for _ in range(5):
            ad.observe(_make_model())
        ad.discover_patterns()
        summary = ad.summary()
        assert "Auto-Discovery" in summary

    def test_discovered_pattern_dataclass(self):
        from igqk.auto_discovery import DiscoveredPattern
        p = DiscoveredPattern(
            name="test", description="test pattern", frequency=10
        )
        assert p.modality == 1
        assert p.best_method == "ternary"


# ===========================================================================
# AI-to-AI Knowledge Transfer Protocol
# ===========================================================================

class TestKnowledgeTransfer:
    def test_init(self):
        from igqk.knowledge_transfer import KnowledgeTransfer
        kt = KnowledgeTransfer()
        assert kt.imported_count == 0
        assert kt.merged_knowledge is None

    def test_knowledge_packet_json(self):
        from igqk.knowledge_transfer import KnowledgePacket
        packet = KnowledgePacket(
            total_compressions=50,
            total_models_seen=5,
            method_scores={"square_medium": {"ternary": 12.0}},
            best_methods={"square_medium": "ternary"},
            confidence=0.5,
        )
        json_str = packet.to_json()
        loaded = KnowledgePacket.from_json(json_str)
        assert loaded.total_compressions == 50
        assert loaded.best_methods["square_medium"] == "ternary"

    def test_knowledge_packet_compact(self):
        from igqk.knowledge_transfer import KnowledgePacket
        packet = KnowledgePacket(total_compressions=25, confidence=0.3)
        compact = packet.to_compact()
        assert isinstance(compact, str)
        loaded = KnowledgePacket.from_compact(compact)
        assert loaded.total_compressions == 25

    def test_export_empty(self):
        from igqk.knowledge_transfer import KnowledgeTransfer
        kt = KnowledgeTransfer()
        packet = kt.export_knowledge()
        assert packet.total_compressions == 0
        assert packet.source_instance == kt.instance_id

    def test_export_with_meta_learner(self, tmp_path):
        from igqk.knowledge_transfer import KnowledgeTransfer
        from igqk.meta_learner import MetaLearner
        kt = KnowledgeTransfer()
        ml = MetaLearner(knowledge_path=str(tmp_path / "kt_ml.json"))
        model = _make_model()
        w = torch.randn(64, 32)
        ml.record_experience(model, "layer", w, w * 0.5, "ternary")
        packet = kt.export_knowledge(meta_learner=ml)
        assert packet.total_compressions == 1

    def test_export_with_auto_discovery(self):
        from igqk.knowledge_transfer import KnowledgeTransfer
        from igqk.auto_discovery import AutoDiscovery
        kt = KnowledgeTransfer()
        ad = AutoDiscovery()
        for _ in range(5):
            ad.observe(_make_model())
        ad.discover_patterns()
        packet = kt.export_knowledge(auto_discovery=ad)
        assert len(packet.discovered_patterns) > 0

    def test_export_with_evolution(self):
        from igqk.knowledge_transfer import KnowledgeTransfer
        from igqk.evolution_engine import EvolutionEngine
        kt = KnowledgeTransfer()
        ee = EvolutionEngine(population_size=6, generations=2)
        ee.evolve(torch.randn(100))
        packet = kt.export_knowledge(evolution_engine=ee)
        assert len(packet.evolved_strategies) > 0

    def test_import_knowledge(self):
        from igqk.knowledge_transfer import KnowledgeTransfer, KnowledgePacket
        kt = KnowledgeTransfer()
        packet = KnowledgePacket(
            total_compressions=100,
            total_models_seen=10,
            method_scores={"bias": {"ternary": 15.0, "sparse": 8.0}},
            best_methods={"bias": "ternary"},
            confidence=0.8,
        )
        stats = kt.import_knowledge(packet)
        assert stats["compressions_imported"] == 100
        assert kt.imported_count == 1
        assert kt.merged_knowledge is not None

    def test_import_updates_meta_learner(self, tmp_path):
        from igqk.knowledge_transfer import KnowledgeTransfer, KnowledgePacket
        from igqk.meta_learner import MetaLearner
        kt = KnowledgeTransfer()
        ml = MetaLearner(knowledge_path=str(tmp_path / "import_ml.json"))
        packet = KnowledgePacket(
            total_compressions=50,
            method_scores={"square_medium": {"sparse": 10.0}},
            best_methods={"square_medium": "sparse"},
            confidence=0.7,
        )
        stats = kt.import_knowledge(packet, meta_learner=ml, trust_weight=0.6)
        assert stats["methods_updated"] > 0
        assert "square_medium" in ml.knowledge.method_scores

    def test_get_recommendation(self):
        from igqk.knowledge_transfer import KnowledgeTransfer, KnowledgePacket
        kt = KnowledgeTransfer()
        # No knowledge yet
        assert kt.get_recommendation({"entropy": 3.0}) is None

        # Import knowledge with pattern
        packet = KnowledgePacket(
            total_compressions=100,
            confidence=0.8,
            discovered_patterns=[{
                "name": "natural_sparsity",
                "frequency": 50,
                "best_method": "sparse",
                "best_params": {},
                "expected_ratio": 5.0,
            }],
        )
        kt.import_knowledge(packet)
        rec = kt.get_recommendation({"sparsity": 0.5})
        assert rec is not None
        assert rec[0] == "sparse"

    def test_multiple_imports_merge(self):
        from igqk.knowledge_transfer import KnowledgeTransfer, KnowledgePacket
        kt = KnowledgeTransfer()
        p1 = KnowledgePacket(
            total_compressions=50, total_models_seen=5, confidence=0.5,
            source_instance="inst_A",
            method_scores={"bias": {"ternary": 10.0}},
        )
        p2 = KnowledgePacket(
            total_compressions=80, total_models_seen=8, confidence=0.8,
            source_instance="inst_B",
            method_scores={"bias": {"sparse": 12.0}},
        )
        kt.import_knowledge(p1)
        kt.import_knowledge(p2)
        assert kt.imported_count == 2
        mk = kt.merged_knowledge
        assert mk.total_compressions == 130

    def test_summary(self):
        from igqk.knowledge_transfer import KnowledgeTransfer, KnowledgePacket
        kt = KnowledgeTransfer()
        kt.import_knowledge(KnowledgePacket(
            total_compressions=10, confidence=0.3, source_instance="test"
        ))
        summary = kt.summary()
        assert "Knowledge Transfer" in summary
        assert "test" in summary


# ===========================================================================
# Autonomous Pipeline
# ===========================================================================

class TestAutonomousPipeline:
    def test_init(self):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path="/tmp/igqk_test_auto_pipe.json",
            enable_evolution=False,
        )
        assert ap.total_compressions == 0

    def test_compress_basic(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap.json"),
            enable_evolution=False,
        )
        model = _make_model()
        result = ap.compress(model)
        assert result.layers_compressed > 0
        assert result.total_time > 0
        assert result.original_params > 0
        assert len(result.decisions) > 0

    def test_compress_with_evolution(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_evo.json"),
            enable_evolution=True,
            evolution_generations=2,
            evolution_population=6,
        )
        model = _make_model(in_f=16, hidden=32, out_f=5)
        result = ap.compress(model)
        assert result.layers_compressed > 0

    def test_compress_verbose(self, tmp_path, capsys):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_v.json"),
            enable_evolution=False,
            verbose=True,
        )
        model = _make_model()
        result = ap.compress(model)
        captured = capsys.readouterr()
        assert "Autonomous Pipeline" in captured.out
        assert "Phase 1" in captured.out

    def test_result_summary(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_s.json"),
            enable_evolution=False,
        )
        result = ap.compress(_make_model())
        summary = result.summary()
        assert "Autonomous Compression Result" in summary
        assert "Layers compressed" in summary

    def test_knowledge_export_import(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        # Instance A compresses
        ap_a = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_a.json"),
            enable_evolution=False,
        )
        ap_a.compress(_make_model())
        packet = ap_a.export_knowledge()

        # Instance B imports
        ap_b = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_b.json"),
            enable_evolution=False,
        )
        ap_b.import_collective_knowledge(packet)
        assert ap_b.knowledge_transfer.imported_count == 1

    def test_pipeline_summary(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_sum.json"),
            enable_evolution=False,
        )
        ap.compress(_make_model())
        summary = ap.summary()
        assert "Autonomous" in summary
        assert "MetaLearner" in summary

    def test_multiple_compressions_improve(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_multi.json"),
            enable_evolution=False,
        )
        # Compress several times to build experience
        for _ in range(3):
            ap.compress(_make_model())
        assert ap.total_compressions == 3
        assert ap.meta_learner.knowledge.total_compressions > 0

    def test_decision_sources(self, tmp_path):
        from igqk.autonomous import AutonomousPipeline
        ap = AutonomousPipeline(
            knowledge_path=str(tmp_path / "ap_ds.json"),
            enable_evolution=False,
        )
        result = ap.compress(_make_model())
        for d in result.decisions:
            assert d.source in (
                "collective", "meta_learner", "discovery",
                "evolution", "heuristic"
            )
            assert d.method in (
                "ternary", "sparse", "wavelet", "binary",
                "adaptive_sparse", "lowrank"
            )
            assert 0 <= d.confidence <= 1


# We need numpy for auto_discovery tests
import numpy as np
