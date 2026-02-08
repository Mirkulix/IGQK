"""
Meta-Learning Compression - The system learns HOW to compress better.

After every compression, IGQK extracts "compression knowledge":
- Which method worked best for which layer shape?
- What entropy threshold predicts ternary success?
- How does weight distribution correlate with optimal compression?

This knowledge is stored and used to make FUTURE compressions better.
The more you use IGQK, the smarter it gets.

    ┌──────────┐     ┌──────────────┐     ┌──────────────┐
    │  Model   │ ──> │  Compress    │ ──> │  Compressed  │
    │  Input   │     │  (IGQK)      │     │  Model       │
    └──────────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  Extract     │ <── │  Evaluate    │
                     │  Knowledge   │     │  Results     │
                     └──────┬───────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Knowledge   │  ← persists across sessions
                     │  Database    │
                     └──────────────┘

This is the first compression framework that LEARNS FROM ITSELF.
"""

import json
import os
import time
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CompressionExperience:
    """One compression experience the system can learn from."""
    timestamp: float
    model_hash: str
    layer_name: str
    layer_shape: List[int]
    num_params: int

    # Pre-compression analysis
    weight_mean: float
    weight_std: float
    weight_entropy: float
    weight_sparsity: float
    singular_value_ratio: float  # sv[0]/sv[-1] - condition number proxy

    # What was done
    method: str  # ternary, wavelet, sparse, etc.
    strength: float

    # Results
    distortion: float
    compression_ratio: float
    success: bool  # Was this a good compression? (low distortion)

    # Derived
    efficiency_score: float = 0.0  # compression_ratio / (1 + distortion)


@dataclass
class CompressionKnowledge:
    """Accumulated knowledge from many compressions."""
    total_compressions: int = 0
    total_models: int = 0

    # Method success rates per layer type
    method_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # shape_class -> {method -> avg_efficiency}

    # Learned thresholds
    ternary_entropy_threshold: float = 0.5
    sparse_sparsity_threshold: float = 0.3
    wavelet_std_threshold: float = 0.1

    # Best method per shape class
    best_methods: Dict[str, str] = field(default_factory=dict)


class MetaLearner:
    """
    Meta-learning engine that improves compression decisions over time.

    Learns from every compression attempt and builds a knowledge base
    that predicts the best compression method and parameters for any layer.
    """

    def __init__(self, knowledge_path: Optional[str] = None):
        """
        Args:
            knowledge_path: Path to persist knowledge. None for in-memory only.
        """
        self.knowledge_path = knowledge_path or os.path.join(
            os.path.expanduser("~"), ".igqk", "meta_knowledge.json"
        )
        self.experiences: List[CompressionExperience] = []
        self.knowledge = CompressionKnowledge()
        self._load_knowledge()

    def _load_knowledge(self):
        """Load persisted knowledge from disk."""
        try:
            if os.path.exists(self.knowledge_path):
                with open(self.knowledge_path, "r") as f:
                    data = json.load(f)
                self.knowledge.total_compressions = data.get("total_compressions", 0)
                self.knowledge.total_models = data.get("total_models", 0)
                self.knowledge.method_scores = data.get("method_scores", {})
                self.knowledge.best_methods = data.get("best_methods", {})
                self.knowledge.ternary_entropy_threshold = data.get("ternary_entropy_threshold", 0.5)
                self.knowledge.sparse_sparsity_threshold = data.get("sparse_sparsity_threshold", 0.3)
        except (json.JSONDecodeError, IOError):
            pass

    def _save_knowledge(self):
        """Persist knowledge to disk."""
        os.makedirs(os.path.dirname(self.knowledge_path), exist_ok=True)
        data = {
            "total_compressions": self.knowledge.total_compressions,
            "total_models": self.knowledge.total_models,
            "method_scores": self.knowledge.method_scores,
            "best_methods": self.knowledge.best_methods,
            "ternary_entropy_threshold": self.knowledge.ternary_entropy_threshold,
            "sparse_sparsity_threshold": self.knowledge.sparse_sparsity_threshold,
        }
        with open(self.knowledge_path, "w") as f:
            json.dump(data, f, indent=2)

    def analyze_layer(self, name: str, weight: torch.Tensor) -> Dict[str, float]:
        """Extract features from a layer for meta-learning."""
        flat = weight.detach().flatten().float()
        features = {
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "sparsity": (flat.abs() < 0.01 * flat.std()).float().mean().item(),
        }

        # Entropy via histogram
        hist = torch.histc(flat, bins=50)
        hist = hist / hist.sum()
        entropy = -(hist * torch.log(hist + 1e-10)).sum().item()
        features["entropy"] = entropy

        # Singular value ratio (for 2D+ tensors)
        if weight.dim() >= 2:
            W = weight.reshape(weight.shape[0], -1).float()
            sv = torch.linalg.svdvals(W)
            features["sv_ratio"] = (sv[0] / (sv[-1] + 1e-10)).item()
        else:
            features["sv_ratio"] = 1.0

        return features

    def predict_best_method(
        self, name: str, weight: torch.Tensor
    ) -> Tuple[str, float, str]:
        """
        Predict the best compression method for a layer.

        Uses accumulated knowledge to make smart decisions.
        Falls back to heuristics if no knowledge available.

        Returns:
            (method, confidence, reason)
        """
        features = self.analyze_layer(name, weight)
        shape_class = self._shape_class(weight.shape)

        # Check if we have learned knowledge for this shape class
        if shape_class in self.knowledge.best_methods:
            method = self.knowledge.best_methods[shape_class]
            scores = self.knowledge.method_scores.get(shape_class, {})
            confidence = scores.get(method, 0.5)
            return method, confidence, f"Learned from {self.knowledge.total_compressions} experiences"

        # Heuristic fallback (bootstrapping)
        if features["sparsity"] > self.knowledge.sparse_sparsity_threshold:
            return "sparse", 0.6, "High natural sparsity detected"
        elif features["entropy"] < self.knowledge.ternary_entropy_threshold:
            return "ternary", 0.7, "Low entropy suits ternary quantization"
        elif features["sv_ratio"] > 100:
            return "lowrank", 0.5, "High condition number suggests low-rank structure"
        else:
            return "ternary", 0.5, "Default recommendation (insufficient data)"

    def record_experience(
        self,
        model: nn.Module,
        layer_name: str,
        weight_before: torch.Tensor,
        weight_after: torch.Tensor,
        method: str,
        strength: float = 1.0,
    ):
        """Record a compression experience for learning."""
        features = self.analyze_layer(layer_name, weight_before)

        distortion = (weight_before - weight_after).norm().item()
        original_norm = weight_before.norm().item() + 1e-10
        relative_distortion = distortion / original_norm

        # Estimate compression ratio
        unique_vals = weight_after.flatten().unique().numel()
        if unique_vals <= 4:
            ratio = 16.0
        elif (weight_after == 0).float().mean() > 0.5:
            nonzero_frac = (weight_after != 0).float().mean().item()
            ratio = 1.0 / max(nonzero_frac, 0.01)
        else:
            ratio = 1.0

        success = relative_distortion < 0.1  # Less than 10% relative error
        efficiency = ratio / (1.0 + relative_distortion)

        exp = CompressionExperience(
            timestamp=time.time(),
            model_hash=self._model_hash(model),
            layer_name=layer_name,
            layer_shape=list(weight_before.shape),
            num_params=weight_before.numel(),
            weight_mean=features["mean"],
            weight_std=features["std"],
            weight_entropy=features["entropy"],
            weight_sparsity=features["sparsity"],
            singular_value_ratio=features["sv_ratio"],
            method=method,
            strength=strength,
            distortion=relative_distortion,
            compression_ratio=ratio,
            success=success,
            efficiency_score=efficiency,
        )

        self.experiences.append(exp)
        self._update_knowledge(exp)

    def _update_knowledge(self, exp: CompressionExperience):
        """Update knowledge base with new experience."""
        self.knowledge.total_compressions += 1

        shape_class = self._shape_class(exp.layer_shape)

        # Update method scores
        if shape_class not in self.knowledge.method_scores:
            self.knowledge.method_scores[shape_class] = {}

        scores = self.knowledge.method_scores[shape_class]
        if exp.method not in scores:
            scores[exp.method] = exp.efficiency_score
        else:
            # Exponential moving average
            alpha = 0.3
            scores[exp.method] = (1 - alpha) * scores[exp.method] + alpha * exp.efficiency_score

        # Update best method
        best_method = max(scores, key=scores.get)
        self.knowledge.best_methods[shape_class] = best_method

        # Update thresholds based on successful ternary compressions
        if exp.method == "ternary" and exp.success:
            alpha = 0.1
            self.knowledge.ternary_entropy_threshold = (
                (1 - alpha) * self.knowledge.ternary_entropy_threshold
                + alpha * exp.weight_entropy
            )

        self._save_knowledge()

    def compress_with_learning(
        self, model: nn.Module
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress a model while learning from the process.

        This is the main entry point: compress + learn simultaneously.
        """
        from igqk.theory.tlgt import TernaryLieGroup
        from igqk.theory.hlwt import HybridLaplaceWavelet

        self.knowledge.total_models += 1
        stats = {"layers": [], "methods_used": {}}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 16:
                    continue

                weight_before = param.data.clone()
                method, confidence, reason = self.predict_best_method(name, param.data)

                # Apply predicted method
                if method == "ternary":
                    tlgt = TernaryLieGroup(param.numel())
                    compressed, scale = tlgt.quantize(param.data)
                    param.data = compressed
                elif method == "wavelet":
                    hlwt = HybridLaplaceWavelet()
                    compressed, ratio = hlwt.compress(param.data, keep_ratio=0.3)
                    param.data = compressed
                elif method == "sparse":
                    threshold = torch.quantile(param.data.abs().flatten(), 0.7)
                    param.data *= (param.data.abs() >= threshold).float()
                elif method == "lowrank":
                    if param.dim() >= 2:
                        W = param.data.float()
                        U, S, Vh = torch.linalg.svd(W.reshape(W.shape[0], -1), full_matrices=False)
                        rank = max(1, int(0.3 * min(W.shape[0], W.reshape(W.shape[0], -1).shape[1])))
                        param.data = (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).reshape(W.shape).to(param.dtype)

                # Record experience
                self.record_experience(
                    model, name, weight_before, param.data, method
                )

                stats["layers"].append({
                    "name": name,
                    "method": method,
                    "confidence": confidence,
                    "reason": reason,
                })
                stats["methods_used"][method] = stats["methods_used"].get(method, 0) + 1

        stats["total_experiences"] = self.knowledge.total_compressions
        stats["total_models_seen"] = self.knowledge.total_models
        return model, stats

    def get_wisdom(self) -> str:
        """Get human-readable summary of learned knowledge."""
        lines = [
            "IGQK Meta-Learning Knowledge",
            "=" * 50,
            f"  Total compressions: {self.knowledge.total_compressions}",
            f"  Total models seen:  {self.knowledge.total_models}",
            f"  Shape classes known: {len(self.knowledge.best_methods)}",
            "",
            "  Learned best methods:",
        ]
        for shape_class, method in sorted(self.knowledge.best_methods.items()):
            scores = self.knowledge.method_scores.get(shape_class, {})
            score = scores.get(method, 0)
            lines.append(f"    {shape_class}: {method} (score={score:.3f})")

        lines.extend([
            "",
            "  Learned thresholds:",
            f"    Ternary entropy: {self.knowledge.ternary_entropy_threshold:.4f}",
            f"    Sparse sparsity: {self.knowledge.sparse_sparsity_threshold:.4f}",
        ])
        return "\n".join(lines)

    def _shape_class(self, shape) -> str:
        """Classify a tensor shape into a learnable category."""
        if isinstance(shape, torch.Size):
            shape = list(shape)
        if len(shape) == 1:
            return "bias"
        elif len(shape) == 2:
            m, n = shape
            if m == n:
                return f"square_{self._size_bucket(m)}"
            elif m > n:
                return f"tall_{self._size_bucket(m)}x{self._size_bucket(n)}"
            else:
                return f"wide_{self._size_bucket(m)}x{self._size_bucket(n)}"
        elif len(shape) == 4:
            return f"conv_{shape[0]}x{shape[1]}"
        return f"nd{len(shape)}"

    def _size_bucket(self, n: int) -> str:
        """Bucket a dimension size."""
        if n <= 64:
            return "small"
        elif n <= 256:
            return "medium"
        elif n <= 1024:
            return "large"
        else:
            return "xlarge"

    def _model_hash(self, model: nn.Module) -> str:
        """Create a hash for a model architecture."""
        desc = str([(n, list(p.shape)) for n, p in model.named_parameters()])
        return hashlib.md5(desc.encode()).hexdigest()[:12]
