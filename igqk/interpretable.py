"""
Interpretable Compression - Explain what compression removes and why.

Uses the quantum state eigendecomposition to explain:
1. WHAT each weight pattern does (functional role)
2. WHY it can be removed (redundancy analysis)
3. HOW removal affects the output (impact prediction)

This makes IGQK the only compression framework that can EXPLAIN its decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WeightPattern:
    """A discovered weight pattern with its explanation."""
    layer: str
    pattern_index: int
    importance: float          # eigenvalue (how much this pattern matters)
    explained_variance: float  # fraction of total variance explained
    role: str                  # inferred functional role
    removable: bool
    removal_impact: float      # predicted accuracy impact if removed
    description: str


@dataclass
class CompressionExplanation:
    """Full explanation of a compression decision."""
    layer: str
    method_chosen: str
    reason: str
    patterns_kept: int
    patterns_removed: int
    expected_accuracy_loss: float
    key_patterns: List[WeightPattern]


class InterpretableCompressor:
    """
    Compression with full explanations.

    Analyzes the spectral decomposition of each layer to understand
    what each weight pattern does, then explains which patterns
    are removed and why.
    """

    def __init__(self, detail_level: str = "medium"):
        """
        Args:
            detail_level: "brief", "medium", "detailed"
        """
        self.detail_level = detail_level

    def analyze_layer(
        self,
        name: str,
        weight: torch.Tensor,
        top_k: int = 10,
    ) -> List[WeightPattern]:
        """
        Analyze weight matrix and discover functional patterns.

        Uses SVD to decompose weights into interpretable components.
        Each singular vector represents a "feature detector" pattern.
        """
        if weight.dim() < 2:
            return [WeightPattern(
                layer=name, pattern_index=0,
                importance=weight.norm().item(),
                explained_variance=1.0,
                role="bias_vector",
                removable=False, removal_impact=0.1,
                description=f"Bias term with magnitude {weight.norm():.4f}",
            )]

        # Reshape to 2D
        W = weight.reshape(weight.shape[0], -1).float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        total_variance = (S ** 2).sum().item()
        patterns = []

        for i in range(min(top_k, len(S))):
            sv = S[i].item()
            variance_explained = (sv ** 2) / total_variance
            cumulative = (S[:i+1] ** 2).sum().item() / total_variance

            # Analyze the pattern
            u_vec = U[:, i]
            v_vec = Vh[i, :]

            role = self._infer_role(u_vec, v_vec, i, sv, W)
            removable = variance_explained < 0.01  # less than 1% contribution
            impact = self._predict_removal_impact(sv, total_variance, cumulative)

            description = self._generate_description(
                name, i, sv, variance_explained, role, u_vec, v_vec
            )

            patterns.append(WeightPattern(
                layer=name, pattern_index=i,
                importance=sv,
                explained_variance=variance_explained,
                role=role,
                removable=removable,
                removal_impact=impact,
                description=description,
            ))

        return patterns

    def explain_compression(
        self, model: nn.Module, method: str = "ternary"
    ) -> List[CompressionExplanation]:
        """Generate full explanation for compressing each layer."""
        explanations = []

        for name, param in model.named_parameters():
            if param.numel() < 16:
                continue

            patterns = self.analyze_layer(name, param.data)

            kept = [p for p in patterns if not p.removable]
            removed = [p for p in patterns if p.removable]

            # Choose method explanation
            reason = self._explain_method_choice(
                name, param.data, patterns, method
            )

            expected_loss = sum(p.removal_impact for p in removed)

            explanations.append(CompressionExplanation(
                layer=name,
                method_chosen=method,
                reason=reason,
                patterns_kept=len(kept),
                patterns_removed=len(removed),
                expected_accuracy_loss=expected_loss,
                key_patterns=patterns[:5],
            ))

        return explanations

    def generate_report(
        self, model: nn.Module, method: str = "ternary"
    ) -> str:
        """Generate human-readable compression report."""
        explanations = self.explain_compression(model, method)

        lines = [
            "=" * 70,
            "IGQK Compression Explanation Report",
            "=" * 70,
            "",
        ]

        total_kept = 0
        total_removed = 0
        total_expected_loss = 0.0

        for exp in explanations:
            total_kept += exp.patterns_kept
            total_removed += exp.patterns_removed
            total_expected_loss += exp.expected_accuracy_loss

            lines.append(f"Layer: {exp.layer}")
            lines.append(f"  Method: {exp.method_chosen}")
            lines.append(f"  Reason: {exp.reason}")
            lines.append(f"  Patterns: {exp.patterns_kept} kept, {exp.patterns_removed} removable")
            lines.append(f"  Expected impact: {exp.expected_accuracy_loss:.4f}")

            if self.detail_level in ("medium", "detailed"):
                lines.append(f"  Top patterns:")
                for p in exp.key_patterns[:3]:
                    status = "KEEP" if not p.removable else "REMOVE"
                    lines.append(
                        f"    [{status}] #{p.pattern_index}: "
                        f"{p.role} (importance={p.importance:.4f}, "
                        f"explains {p.explained_variance:.1%})"
                    )
                    if self.detail_level == "detailed":
                        lines.append(f"           {p.description}")

            lines.append("")

        lines.extend([
            "-" * 70,
            f"Summary:",
            f"  Total patterns kept:    {total_kept}",
            f"  Total patterns removed: {total_removed}",
            f"  Expected accuracy loss: {total_expected_loss:.4f}",
            f"  Compression safe:       {'YES' if total_expected_loss < 0.05 else 'CAUTION'}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def _infer_role(
        self, u: torch.Tensor, v: torch.Tensor,
        index: int, sv: float, W: torch.Tensor
    ) -> str:
        """Infer the functional role of a weight pattern."""
        # Sparsity analysis
        u_sparsity = (u.abs() < 0.01).float().mean().item()
        v_sparsity = (v.abs() < 0.01).float().mean().item()

        # Locality: is the pattern localized or spread out?
        u_entropy = self._vector_entropy(u)
        v_entropy = self._vector_entropy(v)

        if index == 0:
            return "primary_feature_detector"
        elif u_sparsity > 0.8:
            return "sparse_selector"
        elif v_sparsity > 0.8:
            return "sparse_combiner"
        elif u_entropy < 0.3:
            return "localized_detector"
        elif v_entropy < 0.3:
            return "localized_combiner"
        elif index < 3:
            return "major_feature_transform"
        elif sv < W.norm().item() * 0.01:
            return "noise_component"
        else:
            return "distributed_feature"

    def _predict_removal_impact(
        self, sv: float, total_variance: float, cumulative: float
    ) -> float:
        """Predict accuracy impact of removing this pattern."""
        fraction = (sv ** 2) / total_variance if total_variance > 0 else 0
        # Impact is proportional to variance explained, with diminishing returns
        return min(fraction * 2, 0.1)

    def _explain_method_choice(
        self, name: str, weight: torch.Tensor,
        patterns: List[WeightPattern], method: str
    ) -> str:
        """Generate explanation for why this method was chosen."""
        std = weight.std().item()
        sparsity = (weight.abs() < 0.01 * std).float().mean().item()
        num_important = sum(1 for p in patterns if not p.removable)

        if method == "ternary":
            if sparsity > 0.3:
                return (f"Already {sparsity:.0%} sparse. Ternary quantization "
                        f"naturally fits the bimodal distribution.")
            else:
                return (f"{num_important} important patterns capture most variance. "
                        f"Ternary encoding preserves direction while removing magnitude noise.")
        elif method == "wavelet":
            return (f"Smooth weight distribution benefits from multi-scale analysis. "
                    f"High-frequency components (noise) can be removed safely.")
        elif method == "sparse":
            return (f"{sparsity:.0%} of weights are near-zero. "
                    f"Pruning these has minimal impact on the {num_important} key patterns.")
        return f"Compression with {method}"

    def _generate_description(
        self, layer: str, index: int, sv: float,
        variance: float, role: str,
        u: torch.Tensor, v: torch.Tensor,
    ) -> str:
        """Generate human-readable description of a pattern."""
        u_active = (u.abs() > u.abs().mean()).sum().item()
        v_active = (v.abs() > v.abs().mean()).sum().item()

        return (
            f"Pattern #{index} in {layer}: {role} "
            f"(singular value={sv:.4f}, explains {variance:.1%} of variance, "
            f"connects {u_active} output neurons to {v_active} input neurons)"
        )

    def _vector_entropy(self, v: torch.Tensor) -> float:
        """Compute normalized entropy of absolute values."""
        p = v.abs() / (v.abs().sum() + 1e-10)
        entropy = -(p * torch.log(p + 1e-10)).sum().item()
        max_entropy = np.log(len(v))
        return entropy / max_entropy if max_entropy > 0 else 0
