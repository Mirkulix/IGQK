"""
AI-to-AI Knowledge Transfer Protocol - Compression knowledge sharing between systems.

When one IGQK instance discovers effective compression strategies, it can
EXPORT that knowledge and IMPORT it into other IGQK instances. This creates
a network effect: every IGQK installation benefits from ALL installations.

Transfer Protocol:
1. Instance A compresses many models, learns optimal strategies
2. A exports a KnowledgePacket (compressed representation of its wisdom)
3. Instance B imports the packet and immediately benefits
4. B's own experiences REFINE the imported knowledge
5. B can re-export improved knowledge → knowledge evolves across instances

    ┌───────────┐                     ┌───────────┐
    │ IGQK      │   KnowledgePacket   │ IGQK      │
    │ Instance A│ ──────────────────> │ Instance B│
    │ (expert)  │                     │ (novice)  │
    └─────┬─────┘                     └─────┬─────┘
          │                                 │
          ▼                                 ▼
    ┌───────────┐                     ┌───────────┐
    │ 1000+     │                     │ Instant   │
    │ models    │                     │ expertise │
    │ compressed│                     │ from A    │
    └───────────┘                     └───────────┘

This is the first compression framework with COLLECTIVE INTELLIGENCE.
"""

import json
import time
import hashlib
import base64
import copy
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class KnowledgePacket:
    """
    A transferable unit of compression knowledge.

    Contains everything another IGQK instance needs to benefit
    from this instance's experience, in a compact format.
    """
    # Metadata
    packet_id: str = ""
    created_at: float = 0.0
    source_instance: str = ""
    version: str = "1.0"

    # Experience summary
    total_compressions: int = 0
    total_models_seen: int = 0
    total_layers_analyzed: int = 0

    # Learned method preferences per shape class
    # shape_class -> {method -> score}
    method_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_methods: Dict[str, str] = field(default_factory=dict)

    # Learned thresholds
    ternary_entropy_threshold: float = 0.5
    sparse_sparsity_threshold: float = 0.3
    wavelet_std_threshold: float = 0.1

    # Discovered patterns (from AutoDiscovery)
    discovered_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Evolution results (best strategies from EvolutionEngine)
    evolved_strategies: List[Dict[str, Any]] = field(default_factory=list)

    # Distribution fingerprints: common weight distributions seen
    distribution_signatures: List[Dict[str, float]] = field(default_factory=list)

    # Trust and quality
    confidence: float = 0.0  # 0-1, based on amount of experience
    compatibility_hash: str = ""  # Hash of supported methods

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgePacket":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_compact(self) -> str:
        """Serialize to compact base64 string for easy sharing."""
        json_bytes = self.to_json().encode("utf-8")
        return base64.b64encode(json_bytes).decode("ascii")

    @classmethod
    def from_compact(cls, compact_str: str) -> "KnowledgePacket":
        """Deserialize from compact base64 string."""
        json_bytes = base64.b64decode(compact_str.encode("ascii"))
        return cls.from_json(json_bytes.decode("utf-8"))


class KnowledgeTransfer:
    """
    AI-to-AI Knowledge Transfer Protocol.

    Enables IGQK instances to share compression knowledge, creating
    a collective intelligence that improves with every installation.
    """

    SUPPORTED_METHODS = ["ternary", "sparse", "wavelet", "lowrank", "adaptive_sparse", "binary"]

    def __init__(self, instance_id: Optional[str] = None):
        """
        Args:
            instance_id: Unique identifier for this IGQK instance.
        """
        self.instance_id = instance_id or self._generate_instance_id()
        self._imported_packets: List[KnowledgePacket] = []
        self._merged_knowledge: Optional[KnowledgePacket] = None

    def export_knowledge(
        self,
        meta_learner=None,
        auto_discovery=None,
        evolution_engine=None,
    ) -> KnowledgePacket:
        """
        Export current knowledge as a transferable packet.

        Collects knowledge from all subsystems and packages it.

        Args:
            meta_learner: MetaLearner instance (optional)
            auto_discovery: AutoDiscovery instance (optional)
            evolution_engine: EvolutionEngine instance (optional)

        Returns:
            KnowledgePacket ready for transfer
        """
        packet = KnowledgePacket(
            packet_id=self._generate_packet_id(),
            created_at=time.time(),
            source_instance=self.instance_id,
            compatibility_hash=self._compatibility_hash(),
        )

        # Extract from MetaLearner
        if meta_learner is not None:
            k = meta_learner.knowledge
            packet.total_compressions = k.total_compressions
            packet.total_models_seen = k.total_models
            packet.method_scores = copy.deepcopy(k.method_scores)
            packet.best_methods = copy.deepcopy(k.best_methods)
            packet.ternary_entropy_threshold = k.ternary_entropy_threshold
            packet.sparse_sparsity_threshold = k.sparse_sparsity_threshold
            packet.total_layers_analyzed = len(meta_learner.experiences)

            # Extract distribution signatures from experiences
            for exp in meta_learner.experiences[-100:]:  # Last 100
                packet.distribution_signatures.append({
                    "mean": exp.weight_mean,
                    "std": exp.weight_std,
                    "entropy": exp.weight_entropy,
                    "sparsity": exp.weight_sparsity,
                    "sv_ratio": exp.singular_value_ratio,
                    "best_method": exp.method,
                    "efficiency": exp.efficiency_score,
                })

        # Extract from AutoDiscovery
        if auto_discovery is not None:
            for pattern in auto_discovery.patterns:
                packet.discovered_patterns.append({
                    "name": pattern.name,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "modality": pattern.modality,
                    "best_method": pattern.best_method,
                    "best_params": pattern.best_params,
                    "expected_ratio": pattern.expected_ratio,
                })

        # Extract from EvolutionEngine
        if evolution_engine is not None:
            for strategy in evolution_engine.hall_of_fame[:5]:
                packet.evolved_strategies.append({
                    "name": strategy.name,
                    "generation": strategy.generation,
                    "fitness": strategy.fitness,
                    "ternary_threshold": strategy.ternary_threshold,
                    "sparse_keep_ratio": strategy.sparse_keep_ratio,
                    "wavelet_keep_ratio": strategy.wavelet_keep_ratio,
                    "blend_ternary": strategy.blend_ternary,
                    "blend_sparse": strategy.blend_sparse,
                    "blend_wavelet": strategy.blend_wavelet,
                    "use_two_stage": strategy.use_two_stage,
                    "compression_ratio": strategy.compression_ratio,
                    "distortion": strategy.distortion,
                })

        # Compute confidence based on experience
        packet.confidence = min(1.0, packet.total_compressions / 100.0)

        return packet

    def import_knowledge(
        self,
        packet: KnowledgePacket,
        meta_learner=None,
        trust_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Import knowledge from another IGQK instance.

        Merges external knowledge with local knowledge using
        trust-weighted averaging.

        Args:
            packet: KnowledgePacket from another instance
            meta_learner: Local MetaLearner to update (optional)
            trust_weight: How much to trust imported knowledge (0-1)

        Returns:
            Import statistics
        """
        # Validate compatibility
        if packet.compatibility_hash and packet.compatibility_hash != self._compatibility_hash():
            pass  # Different versions, still try to import what we can

        self._imported_packets.append(packet)

        stats = {
            "source": packet.source_instance,
            "compressions_imported": packet.total_compressions,
            "patterns_imported": len(packet.discovered_patterns),
            "strategies_imported": len(packet.evolved_strategies),
            "trust_weight": trust_weight,
        }

        # Merge into MetaLearner if available
        if meta_learner is not None:
            stats["methods_updated"] = self._merge_into_meta_learner(
                packet, meta_learner, trust_weight
            )

        # Update merged knowledge cache
        self._update_merged_knowledge()

        return stats

    def _merge_into_meta_learner(
        self,
        packet: KnowledgePacket,
        meta_learner,
        trust_weight: float,
    ) -> int:
        """Merge packet knowledge into a MetaLearner."""
        k = meta_learner.knowledge
        updated = 0

        # Merge method scores with trust-weighted averaging
        for shape_class, methods in packet.method_scores.items():
            if shape_class not in k.method_scores:
                k.method_scores[shape_class] = {}

            for method, score in methods.items():
                if method not in k.method_scores[shape_class]:
                    k.method_scores[shape_class][method] = score * trust_weight
                else:
                    local = k.method_scores[shape_class][method]
                    k.method_scores[shape_class][method] = (
                        (1 - trust_weight) * local + trust_weight * score
                    )
                updated += 1

        # Update best methods based on merged scores
        for shape_class in k.method_scores:
            scores = k.method_scores[shape_class]
            if scores:
                k.best_methods[shape_class] = max(scores, key=scores.get)

        # Merge thresholds
        alpha = trust_weight * packet.confidence
        k.ternary_entropy_threshold = (
            (1 - alpha) * k.ternary_entropy_threshold
            + alpha * packet.ternary_entropy_threshold
        )
        k.sparse_sparsity_threshold = (
            (1 - alpha) * k.sparse_sparsity_threshold
            + alpha * packet.sparse_sparsity_threshold
        )

        return updated

    def _update_merged_knowledge(self):
        """Merge all imported packets into a single knowledge view."""
        if not self._imported_packets:
            return

        merged = KnowledgePacket(
            packet_id=self._generate_packet_id(),
            created_at=time.time(),
            source_instance="merged",
        )

        total_weight = 0.0
        for packet in self._imported_packets:
            w = packet.confidence
            total_weight += w

            merged.total_compressions += packet.total_compressions
            merged.total_models_seen += packet.total_models_seen

            # Weighted merge of method scores
            for sc, methods in packet.method_scores.items():
                if sc not in merged.method_scores:
                    merged.method_scores[sc] = {}
                for m, score in methods.items():
                    if m not in merged.method_scores[sc]:
                        merged.method_scores[sc][m] = score * w
                    else:
                        merged.method_scores[sc][m] += score * w

            # Collect all patterns
            merged.discovered_patterns.extend(packet.discovered_patterns)
            merged.evolved_strategies.extend(packet.evolved_strategies)

        # Normalize
        if total_weight > 0:
            for sc in merged.method_scores:
                for m in merged.method_scores[sc]:
                    merged.method_scores[sc][m] /= total_weight

        # Deduplicate patterns by name
        seen_patterns = {}
        for p in merged.discovered_patterns:
            name = p["name"]
            if name not in seen_patterns or p.get("frequency", 0) > seen_patterns[name].get("frequency", 0):
                seen_patterns[name] = p
        merged.discovered_patterns = list(seen_patterns.values())

        # Keep top strategies by fitness
        merged.evolved_strategies.sort(key=lambda s: s.get("fitness", 0), reverse=True)
        merged.evolved_strategies = merged.evolved_strategies[:10]

        merged.confidence = min(1.0, total_weight / len(self._imported_packets))
        self._merged_knowledge = merged

    def get_recommendation(
        self, weight_stats: Dict[str, float]
    ) -> Optional[Tuple[str, float, str]]:
        """
        Get compression recommendation from collective knowledge.

        Uses merged knowledge from all imported packets.

        Args:
            weight_stats: Dict with keys like mean, std, entropy, sparsity

        Returns:
            (method, confidence, reason) or None if no knowledge
        """
        if self._merged_knowledge is None:
            return None

        mk = self._merged_knowledge

        # Check discovered patterns first
        for pattern in mk.discovered_patterns:
            if self._matches_pattern(weight_stats, pattern):
                return (
                    pattern["best_method"],
                    min(0.9, mk.confidence),
                    f"Matches discovered pattern '{pattern['name']}' "
                    f"(seen {pattern.get('frequency', '?')}x across network)",
                )

        # Fall back to method scores
        # Classify the shape if available
        best_method = None
        best_score = -1
        for sc, methods in mk.method_scores.items():
            for method, score in methods.items():
                if score > best_score:
                    best_score = score
                    best_method = method

        if best_method:
            return (
                best_method,
                min(0.7, mk.confidence),
                f"Collective knowledge from {mk.total_compressions} compressions",
            )

        return None

    def _matches_pattern(self, stats: Dict[str, float], pattern: Dict) -> bool:
        """Check if weight stats match a discovered pattern."""
        name = pattern.get("name", "")

        if name == "natural_sparsity" and stats.get("sparsity", 0) > 0.3:
            return True
        if name == "heavy_tail_dominance" and stats.get("kurtosis", 0) > 2:
            return True
        if name == "low_entropy_structured" and stats.get("entropy", 99) < 2.0:
            return True
        if name == "bimodal_collapse" and stats.get("modality", 1) == 2:
            return True
        if name == "symmetric_balanced" and abs(stats.get("skewness", 1)) < 0.1:
            return True
        if name == "gaussian_normal":
            if abs(stats.get("kurtosis", 99)) < 1 and abs(stats.get("skewness", 99)) < 0.5:
                return True

        return False

    @property
    def imported_count(self) -> int:
        return len(self._imported_packets)

    @property
    def merged_knowledge(self) -> Optional[KnowledgePacket]:
        return self._merged_knowledge

    def summary(self) -> str:
        """Get transfer protocol summary."""
        lines = [
            "IGQK AI-to-AI Knowledge Transfer Protocol",
            "=" * 50,
            f"  Instance ID: {self.instance_id}",
            f"  Imported packets: {len(self._imported_packets)}",
        ]

        if self._merged_knowledge:
            mk = self._merged_knowledge
            lines.extend([
                f"  Collective compressions: {mk.total_compressions}",
                f"  Collective models seen: {mk.total_models_seen}",
                f"  Known patterns: {len(mk.discovered_patterns)}",
                f"  Evolved strategies: {len(mk.evolved_strategies)}",
                f"  Collective confidence: {mk.confidence:.2f}",
            ])

        if self._imported_packets:
            lines.append("")
            lines.append("  Sources:")
            for p in self._imported_packets:
                lines.append(
                    f"    {p.source_instance}: "
                    f"{p.total_compressions} compressions, "
                    f"confidence={p.confidence:.2f}"
                )

        return "\n".join(lines)

    def _generate_instance_id(self) -> str:
        """Generate a unique instance ID."""
        raw = f"{time.time()}-{id(self)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _generate_packet_id(self) -> str:
        """Generate a unique packet ID."""
        raw = f"{self.instance_id}-{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _compatibility_hash(self) -> str:
        """Hash of supported methods for compatibility checking."""
        return hashlib.md5(
            ",".join(sorted(self.SUPPORTED_METHODS)).encode()
        ).hexdigest()[:8]
