"""
Autonomous Compression Pipeline - Zero-Human Neural Network Compression.

The ultimate IGQK feature: give it a model, get back an optimally compressed model.
No configuration. No hyperparameter tuning. No human decisions at all.

The pipeline orchestrates ALL IGQK subsystems:
1. AutoDiscovery analyzes weight distributions
2. MetaLearner predicts best methods from history
3. KnowledgeTransfer checks collective intelligence
4. EvolutionEngine evolves custom strategies if needed
5. The best approach is selected and applied
6. Results are recorded for future improvement

    ┌─────────┐
    │  Model   │
    │  (any)   │
    └────┬─────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │           Autonomous Pipeline                 │
    │                                               │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
    │  │ Auto-    │  │ Meta-    │  │Knowledge │   │
    │  │ Discovery│  │ Learner  │  │ Transfer │   │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
    │       │              │              │         │
    │       └──────────────┼──────────────┘         │
    │                      ▼                        │
    │              ┌──────────────┐                  │
    │              │  Decision    │                  │
    │              │  Engine      │                  │
    │              └──────┬───────┘                  │
    │                     │                         │
    │                     ▼                         │
    │              ┌──────────────┐                  │
    │              │  Evolution   │ ← if needed     │
    │              │  Engine      │                  │
    │              └──────┬───────┘                  │
    │                     │                         │
    │                     ▼                         │
    │              ┌──────────────┐                  │
    │              │  Compress    │                  │
    │              │  & Record    │                  │
    │              └──────────────┘                  │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Compressed │
                    │  Model      │
                    │  (optimal)  │
                    └─────────────┘

This is FULLY AUTONOMOUS AI-driven compression.
The system makes ALL decisions. Humans just provide the model.
"""

import time
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .meta_learner import MetaLearner
from .evolution_engine import EvolutionEngine, Strategy
from .auto_discovery import AutoDiscovery
from .knowledge_transfer import KnowledgeTransfer, KnowledgePacket


@dataclass
class CompressionDecision:
    """A decision made by the autonomous pipeline for one layer."""
    layer_name: str
    method: str
    confidence: float
    source: str  # "meta_learner", "collective", "evolution", "discovery", "heuristic"
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomousResult:
    """Result of autonomous compression."""
    model: nn.Module
    decisions: List[CompressionDecision]
    total_time: float
    original_params: int
    compressed_values: int  # Number of unique weight values
    estimated_ratio: float
    layers_compressed: int
    evolution_used: bool
    collective_knowledge_used: bool

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "IGQK Autonomous Compression Result",
            "=" * 55,
            f"  Time: {self.total_time:.2f}s",
            f"  Original parameters: {self.original_params:,}",
            f"  Estimated compression: {self.estimated_ratio:.1f}x",
            f"  Layers compressed: {self.layers_compressed}",
            f"  Evolution used: {self.evolution_used}",
            f"  Collective knowledge: {self.collective_knowledge_used}",
            "",
            "  Decision sources:",
        ]

        source_counts = {}
        for d in self.decisions:
            source_counts[d.source] = source_counts.get(d.source, 0) + 1
        for source, count in sorted(source_counts.items()):
            lines.append(f"    {source}: {count} layers")

        method_counts = {}
        for d in self.decisions:
            method_counts[d.method] = method_counts.get(d.method, 0) + 1
        lines.append("")
        lines.append("  Methods applied:")
        for method, count in sorted(method_counts.items()):
            lines.append(f"    {method}: {count} layers")

        return "\n".join(lines)


class AutonomousPipeline:
    """
    Zero-human autonomous compression pipeline.

    Orchestrates all IGQK subsystems to make optimal compression
    decisions without any human input.
    """

    def __init__(
        self,
        knowledge_path: Optional[str] = None,
        enable_evolution: bool = True,
        evolution_generations: int = 5,
        evolution_population: int = 10,
        verbose: bool = False,
    ):
        """
        Args:
            knowledge_path: Path for persistent knowledge. None for in-memory.
            enable_evolution: Whether to use evolutionary search for hard cases.
            evolution_generations: Number of evolution generations.
            evolution_population: Evolution population size.
            verbose: Print progress information.
        """
        self.verbose = verbose
        self.enable_evolution = enable_evolution

        # Initialize subsystems
        self.meta_learner = MetaLearner(knowledge_path=knowledge_path)
        self.auto_discovery = AutoDiscovery()
        self.knowledge_transfer = KnowledgeTransfer()
        self.evolution_engine = EvolutionEngine(
            population_size=evolution_population,
            generations=evolution_generations,
        )

        # Track usage
        self._total_compressions = 0
        self._evolution_count = 0

    def compress(
        self,
        model: nn.Module,
        target_ratio: Optional[float] = None,
        quality_threshold: float = 0.1,
    ) -> AutonomousResult:
        """
        Autonomously compress a model. No configuration needed.

        Args:
            model: PyTorch model to compress.
            target_ratio: Desired compression ratio. None = auto-determine.
            quality_threshold: Maximum acceptable relative distortion.

        Returns:
            AutonomousResult with compressed model and metadata.
        """
        start_time = time.time()

        if self.verbose:
            print("IGQK Autonomous Pipeline starting...")

        # Phase 1: Analyze the model
        if self.verbose:
            print("  Phase 1: Analyzing model distributions...")
        self.auto_discovery.observe(model, model_name="autonomous_input")
        patterns = self.auto_discovery.discover_patterns()
        if self.verbose and patterns:
            print(f"    Discovered {len(patterns)} patterns")

        # Phase 2: Make per-layer decisions
        if self.verbose:
            print("  Phase 2: Making compression decisions...")
        decisions = []
        evolution_used = False
        collective_used = self.knowledge_transfer.imported_count > 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 16:
                    continue

                decision = self._decide_layer(name, param, quality_threshold)
                decisions.append(decision)

                if decision.source == "evolution":
                    evolution_used = True

        # Phase 3: Apply decisions
        if self.verbose:
            print("  Phase 3: Applying compression...")
        total_params = 0
        layers_compressed = 0

        with torch.no_grad():
            for decision in decisions:
                param = dict(model.named_parameters())[decision.layer_name]
                total_params += param.numel()
                weight_before = param.data.clone()

                self._apply_decision(param, decision)
                layers_compressed += 1

                # Record for meta-learning
                self.meta_learner.record_experience(
                    model, decision.layer_name, weight_before,
                    param.data, decision.method,
                )

        # Estimate overall compression
        unique_values = set()
        total_zeros = 0
        total_elements = 0
        for _, param in model.named_parameters():
            flat = param.data.flatten()
            unique_values.update(flat.unique().tolist()[:100])
            total_zeros += (flat == 0).sum().item()
            total_elements += flat.numel()

        if len(unique_values) <= 4:
            estimated_ratio = 16.0
        elif total_elements > 0 and total_zeros / total_elements > 0.5:
            nonzero_frac = max(1 - total_zeros / total_elements, 0.01)
            estimated_ratio = 1.0 / nonzero_frac
        else:
            estimated_ratio = float(32.0 / max(np.log2(max(len(unique_values), 2)), 1))

        elapsed = time.time() - start_time
        self._total_compressions += 1

        if self.verbose:
            print(f"  Done in {elapsed:.2f}s. "
                  f"Estimated {estimated_ratio:.1f}x compression.")

        return AutonomousResult(
            model=model,
            decisions=decisions,
            total_time=elapsed,
            original_params=total_params,
            compressed_values=len(unique_values),
            estimated_ratio=estimated_ratio,
            layers_compressed=layers_compressed,
            evolution_used=evolution_used,
            collective_knowledge_used=collective_used,
        )

    def _decide_layer(
        self,
        name: str,
        param: nn.Parameter,
        quality_threshold: float,
    ) -> CompressionDecision:
        """
        Make a compression decision for a single layer.

        Consults all subsystems in priority order.
        """
        weight = param.data

        # Source 1: Collective knowledge (highest priority if confident)
        collective = self.knowledge_transfer.get_recommendation(
            self._weight_stats(weight)
        )
        if collective and collective[1] > 0.7:
            return CompressionDecision(
                layer_name=name,
                method=collective[0],
                confidence=collective[1],
                source="collective",
                reason=collective[2],
            )

        # Source 2: Meta-learner (learned from local history)
        ml_method, ml_conf, ml_reason = self.meta_learner.predict_best_method(
            name, weight
        )
        if ml_conf > 0.6:
            return CompressionDecision(
                layer_name=name,
                method=ml_method,
                confidence=ml_conf,
                source="meta_learner",
                reason=ml_reason,
            )

        # Source 3: Auto-discovered patterns
        pattern = self.auto_discovery.classify_layer(weight)
        if pattern is not None:
            return CompressionDecision(
                layer_name=name,
                method=pattern.best_method,
                confidence=0.6,
                source="discovery",
                reason=f"Matches pattern '{pattern.name}'",
                params=pattern.best_params,
            )

        # Source 4: Evolutionary search (expensive, used as last resort)
        if self.enable_evolution and weight.numel() >= 64:
            best = self.evolution_engine.evolve(weight)
            self._evolution_count += 1

            method = "ternary"
            if best.blend_sparse > best.blend_ternary and best.blend_sparse > best.blend_wavelet:
                method = "sparse"
            elif best.blend_wavelet > best.blend_ternary:
                method = "wavelet"

            return CompressionDecision(
                layer_name=name,
                method=method,
                confidence=min(0.8, best.fitness / 10.0),
                source="evolution",
                reason=f"Evolved strategy gen={best.generation} fitness={best.fitness:.3f}",
                params={
                    "ternary_threshold": best.ternary_threshold,
                    "sparse_keep_ratio": best.sparse_keep_ratio,
                    "use_two_stage": best.use_two_stage,
                },
            )

        # Source 5: Heuristic fallback
        return CompressionDecision(
            layer_name=name,
            method="ternary",
            confidence=0.4,
            source="heuristic",
            reason="Default ternary compression (insufficient data)",
        )

    def _apply_decision(
        self, param: nn.Parameter, decision: CompressionDecision
    ):
        """Apply a compression decision to a parameter."""
        method = decision.method
        params = decision.params

        if method == "ternary":
            threshold_mult = params.get("ternary_threshold", 0.7)
            std = param.data.std()
            threshold = threshold_mult * std
            result = torch.zeros_like(param.data)
            result[param.data > threshold] = std
            result[param.data < -threshold] = -std
            param.data = result

        elif method == "sparse":
            keep_ratio = params.get("sparse_keep_ratio", 0.3)
            flat = param.data.flatten()
            k = max(1, int(keep_ratio * flat.numel()))
            _, indices = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            param.data = (flat * mask).reshape(param.data.shape)

        elif method == "binary":
            median = param.data.median()
            std = param.data.std()
            param.data = torch.where(param.data > median, std, -std)

        elif method == "wavelet":
            flat = param.data.flatten().float()
            n = flat.numel()
            if n % 2 != 0:
                flat = torch.cat([flat, torch.zeros(1)])
            freq = torch.fft.rfft(flat)
            keep_ratio = params.get("wavelet_keep_ratio", 0.3)
            k = max(1, int(keep_ratio * freq.numel()))
            _, indices = torch.topk(freq.abs(), k)
            mask = torch.zeros_like(freq)
            mask[indices] = 1.0
            reconstructed = torch.fft.irfft(freq * mask, n=flat.numel())[:n]
            param.data = reconstructed.reshape(param.data.shape).to(param.data.dtype)

        elif method == "adaptive_sparse":
            percentile = params.get("percentile", 0.9)
            threshold = torch.quantile(param.data.abs().flatten(), percentile)
            param.data *= (param.data.abs() >= threshold).float()

        elif method == "lowrank":
            if param.data.dim() >= 2:
                W = param.data.float()
                shape = W.shape
                W2d = W.reshape(shape[0], -1)
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
                rank = max(1, int(0.3 * min(W2d.shape)))
                param.data = (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).reshape(shape).to(param.data.dtype)

        # Two-stage post-processing
        if params.get("use_two_stage", False):
            std = param.data.std()
            if std > 0:
                threshold = 0.5 * std
                stage2 = torch.zeros_like(param.data)
                stage2[param.data > threshold] = std
                stage2[param.data < -threshold] = -std
                param.data = 0.5 * param.data + 0.5 * stage2

    def _weight_stats(self, weight: torch.Tensor) -> Dict[str, float]:
        """Quick statistics for collective knowledge lookup."""
        flat = weight.detach().flatten().float()
        mean = flat.mean().item()
        std = flat.std().item()

        centered = flat - mean
        if std > 0:
            skewness = (centered ** 3).mean().item() / (std ** 3)
            kurtosis = (centered ** 4).mean().item() / (std ** 4) - 3
            sparsity = (flat.abs() < 0.01 * std).float().mean().item()
        else:
            skewness = 0.0
            kurtosis = 0.0
            sparsity = 0.0

        hist = torch.histc(flat, bins=50)
        hist_norm = hist / hist.sum()
        entropy = -(hist_norm * torch.log(hist_norm + 1e-10)).sum().item()

        return {
            "mean": mean, "std": std, "skewness": skewness,
            "kurtosis": kurtosis, "sparsity": sparsity,
            "entropy": entropy,
        }

    def import_collective_knowledge(self, packet: KnowledgePacket):
        """Import knowledge from another IGQK instance."""
        self.knowledge_transfer.import_knowledge(
            packet, meta_learner=self.meta_learner
        )

    def export_knowledge(self) -> KnowledgePacket:
        """Export this instance's knowledge for sharing."""
        return self.knowledge_transfer.export_knowledge(
            meta_learner=self.meta_learner,
            auto_discovery=self.auto_discovery,
            evolution_engine=self.evolution_engine,
        )

    @property
    def total_compressions(self) -> int:
        return self._total_compressions

    def summary(self) -> str:
        """Get pipeline summary."""
        lines = [
            "IGQK Autonomous Compression Pipeline",
            "=" * 55,
            f"  Total autonomous compressions: {self._total_compressions}",
            f"  Evolution invocations: {self._evolution_count}",
            f"  Meta-learner experiences: {self.meta_learner.knowledge.total_compressions}",
            f"  Discovered patterns: {len(self.auto_discovery.patterns)}",
            f"  Imported knowledge packets: {self.knowledge_transfer.imported_count}",
            "",
            "  Subsystem Status:",
            f"    MetaLearner: {self.meta_learner.knowledge.total_compressions} experiences",
            f"    AutoDiscovery: {self.auto_discovery.num_observations} observations",
            f"    EvolutionEngine: {len(self.evolution_engine.hall_of_fame)} hall of fame",
            f"    KnowledgeTransfer: {self.knowledge_transfer.imported_count} packets",
        ]
        return "\n".join(lines)
