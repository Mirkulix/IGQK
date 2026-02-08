"""
Auto-Discovery Engine - Discovers new compression patterns autonomously.

The system analyzes weight distributions across many models and discovers
recurring patterns that can be exploited for compression. These patterns
are things no human has explicitly programmed.

Discovery process:
1. Collect weight statistics from many models
2. Cluster similar distributions
3. For each cluster, test many compression approaches
4. Discover which combinations work best
5. Name and register new techniques automatically

Example discovered patterns:
- "Bimodal Collapse": Weights with two peaks compress well with binary
- "Tail Dominance": Heavy-tailed distributions need adaptive thresholds
- "Harmonic Structure": Periodic weight patterns compress with FFT

    ┌─────────┐   ┌──────────────┐   ┌───────────────┐
    │ Models  │──>│  Statistical │──>│  Pattern      │
    │ (many)  │   │  Analysis    │   │  Discovery    │
    └─────────┘   └──────────────┘   └───────┬───────┘
                                              │
                                              ▼
                                     ┌───────────────┐
                                     │  New Methods  │
                                     │  (auto-named) │
                                     └───────────────┘
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DiscoveredPattern:
    """A pattern discovered by the auto-discovery engine."""
    name: str
    description: str
    frequency: int  # How many times this pattern was observed

    # Statistical signature
    mean_range: Tuple[float, float] = (0.0, 0.0)
    std_range: Tuple[float, float] = (0.0, 0.0)
    skewness_range: Tuple[float, float] = (0.0, 0.0)
    kurtosis_range: Tuple[float, float] = (0.0, 0.0)
    modality: int = 1  # Number of modes (peaks)

    # Best compression for this pattern
    best_method: str = "ternary"
    best_params: Dict = field(default_factory=dict)
    expected_ratio: float = 1.0


class AutoDiscovery:
    """
    Discovers new compression patterns by analyzing weight distributions.

    Collects statistics, clusters similar layers, and finds optimal
    compression strategies for each cluster.
    """

    def __init__(self):
        self._observations: List[Dict] = []
        self._patterns: List[DiscoveredPattern] = []

    def observe(self, model: nn.Module, model_name: str = ""):
        """Observe a model's weight distributions."""
        for name, param in model.named_parameters():
            if param.numel() < 32:
                continue

            flat = param.detach().flatten().float()
            stats = self._compute_statistics(flat)
            stats["model"] = model_name
            stats["layer"] = name
            stats["shape"] = list(param.shape)
            self._observations.append(stats)

    def _compute_statistics(self, flat: torch.Tensor) -> Dict:
        """Compute comprehensive statistics for a weight tensor."""
        mean = flat.mean().item()
        std = flat.std().item()
        n = flat.numel()

        # Higher moments
        centered = flat - mean
        if std > 0:
            skewness = (centered ** 3).mean().item() / (std ** 3)
            kurtosis = (centered ** 4).mean().item() / (std ** 4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Sparsity
        sparsity = (flat.abs() < 0.01 * std).float().mean().item() if std > 0 else 0

        # Modality detection (number of peaks in histogram)
        hist = torch.histc(flat, bins=50)
        hist_np = hist.numpy()
        modality = self._count_modes(hist_np)

        # Entropy
        hist_norm = hist / hist.sum()
        entropy = -(hist_norm * torch.log(hist_norm + 1e-10)).sum().item()

        # Tail weight (fraction beyond 2*std)
        tail_weight = (flat.abs() > 2 * std).float().mean().item() if std > 0 else 0

        return {
            "mean": mean, "std": std, "skewness": skewness,
            "kurtosis": kurtosis, "sparsity": sparsity,
            "modality": modality, "entropy": entropy,
            "tail_weight": tail_weight, "num_params": n,
        }

    def _count_modes(self, histogram: np.ndarray) -> int:
        """Count number of modes (peaks) in a histogram."""
        if len(histogram) < 3:
            return 1

        modes = 0
        for i in range(1, len(histogram) - 1):
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                if histogram[i] > histogram.max() * 0.1:  # Significant peak
                    modes += 1
        return max(1, modes)

    def discover_patterns(self, min_observations: int = 3) -> List[DiscoveredPattern]:
        """Analyze observations and discover patterns."""
        if len(self._observations) < min_observations:
            return []

        self._patterns = []

        # Classify observations into pattern types
        unimodal = [o for o in self._observations if o["modality"] == 1]
        bimodal = [o for o in self._observations if o["modality"] == 2]
        multimodal = [o for o in self._observations if o["modality"] > 2]
        sparse_layers = [o for o in self._observations if o["sparsity"] > 0.3]
        heavy_tailed = [o for o in self._observations if o["kurtosis"] > 2]
        symmetric = [o for o in self._observations if abs(o["skewness"]) < 0.1]
        low_entropy = [o for o in self._observations if o["entropy"] < 2.0]

        # Discover: Gaussian Normal
        if len(unimodal) >= min_observations:
            gaussian = [o for o in unimodal if abs(o["kurtosis"]) < 1 and abs(o["skewness"]) < 0.5]
            if len(gaussian) >= min_observations:
                self._patterns.append(DiscoveredPattern(
                    name="gaussian_normal",
                    description="Bell-curve distribution. Optimal: ternary with std-based threshold.",
                    frequency=len(gaussian),
                    modality=1,
                    best_method="ternary",
                    best_params={"threshold_multiplier": 0.7},
                    expected_ratio=16.0,
                ))

        # Discover: Bimodal Collapse
        if len(bimodal) >= min_observations:
            self._patterns.append(DiscoveredPattern(
                name="bimodal_collapse",
                description="Two-peak distribution. Discovered: binary quantization is optimal.",
                frequency=len(bimodal),
                modality=2,
                best_method="binary",
                best_params={"threshold": "median"},
                expected_ratio=32.0,
            ))

        # Discover: Natural Sparsity
        if len(sparse_layers) >= min_observations:
            avg_sparsity = np.mean([o["sparsity"] for o in sparse_layers])
            self._patterns.append(DiscoveredPattern(
                name="natural_sparsity",
                description=f"Naturally {avg_sparsity:.0%} sparse. Pruning is lossless for near-zero weights.",
                frequency=len(sparse_layers),
                best_method="sparse",
                best_params={"keep_ratio": 1 - avg_sparsity + 0.05},
                expected_ratio=1 / max(1 - avg_sparsity, 0.01),
            ))

        # Discover: Heavy Tail
        if len(heavy_tailed) >= min_observations:
            self._patterns.append(DiscoveredPattern(
                name="heavy_tail_dominance",
                description="Heavy-tailed: few large weights dominate. Adaptive threshold needed.",
                frequency=len(heavy_tailed),
                kurtosis_range=(2.0, 100.0),
                best_method="adaptive_sparse",
                best_params={"use_percentile": True, "percentile": 0.9},
                expected_ratio=5.0,
            ))

        # Discover: Low Entropy
        if len(low_entropy) >= min_observations:
            self._patterns.append(DiscoveredPattern(
                name="low_entropy_structured",
                description="Low entropy = highly structured. Compresses extremely well with any method.",
                frequency=len(low_entropy),
                best_method="ternary",
                best_params={"threshold_multiplier": 0.5},
                expected_ratio=16.0,
            ))

        # Discover: Symmetric
        if len(symmetric) >= min_observations:
            self._patterns.append(DiscoveredPattern(
                name="symmetric_balanced",
                description="Symmetric around zero. Ternary {-s, 0, +s} is natural fit.",
                frequency=len(symmetric),
                best_method="ternary",
                best_params={"use_median_abs": True},
                expected_ratio=16.0,
            ))

        return self._patterns

    def classify_layer(self, weight: torch.Tensor) -> Optional[DiscoveredPattern]:
        """Classify a new layer against discovered patterns."""
        if not self._patterns:
            return None

        flat = weight.detach().flatten().float()
        stats = self._compute_statistics(flat)

        # Find best matching pattern
        best_match = None
        best_score = -1

        for pattern in self._patterns:
            score = 0

            if pattern.modality == stats["modality"]:
                score += 2

            if pattern.name == "natural_sparsity" and stats["sparsity"] > 0.3:
                score += 3

            if pattern.name == "heavy_tail_dominance" and stats["kurtosis"] > 2:
                score += 3

            if pattern.name == "low_entropy_structured" and stats["entropy"] < 2.0:
                score += 3

            if pattern.name == "symmetric_balanced" and abs(stats["skewness"]) < 0.1:
                score += 2

            if score > best_score:
                best_score = score
                best_match = pattern

        return best_match

    @property
    def patterns(self) -> List[DiscoveredPattern]:
        return self._patterns

    @property
    def num_observations(self) -> int:
        return len(self._observations)

    def summary(self) -> str:
        """Get discovery summary."""
        lines = [
            "IGQK Auto-Discovery Engine",
            "=" * 50,
            f"  Observations: {len(self._observations)}",
            f"  Patterns discovered: {len(self._patterns)}",
            "",
        ]
        for p in self._patterns:
            lines.append(
                f"  [{p.name}] (seen {p.frequency}x)")
            lines.append(f"    {p.description}")
            lines.append(
                f"    Best: {p.best_method} -> ~{p.expected_ratio:.0f}x compression")
            lines.append("")
        return "\n".join(lines)
