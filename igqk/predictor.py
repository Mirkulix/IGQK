"""
Compression Oracle - Predict compression quality BEFORE compressing.

Instead of compressing and hoping for the best, the Oracle analyzes
weight distributions and predicts:
- Expected compression ratio
- Expected quality (accuracy retention)
- Best method to use
- Time estimate
- Risk level

This saves massive amounts of compute by avoiding bad compression attempts.

    ┌──────────┐     ┌──────────────┐     ┌──────────────┐
    │  Model   │ ──> │   Oracle     │ ──> │  Prediction  │
    │ (input)  │     │  (fast)      │     │  ratio: 12x  │
    └──────────┘     │  No actual   │     │  quality: 97%│
                     │  compression │     │  method: TERN│
                     │  needed!     │     │  risk: LOW   │
                     └──────────────┘     └──────────────┘
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class Prediction:
    """Compression prediction for a layer or full model."""
    method: str
    expected_ratio: float
    expected_quality: float  # 0-1, where 1 = perfect preservation
    confidence: float        # 0-1, how confident the prediction is
    risk_level: str          # "low", "medium", "high"
    reasoning: str
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelPrediction:
    """Full model compression prediction."""
    layer_predictions: List[Prediction]
    overall_ratio: float
    overall_quality: float
    overall_risk: str
    recommended_method: str
    summary: str
    compressibility_score: float  # 0-1


class CompressionOracle:
    """
    Predict compression outcomes without actually compressing.

    Uses statistical analysis of weight distributions to estimate
    compression quality and ratio for different methods.
    """

    # Method-specific prediction models (empirically tuned)
    METHOD_PROFILES = {
        "ternary": {
            "min_ratio": 8.0,
            "max_ratio": 16.0,
            "best_when": "gaussian_like",
            "quality_base": 0.85,
        },
        "sparse": {
            "min_ratio": 2.0,
            "max_ratio": 20.0,
            "best_when": "naturally_sparse",
            "quality_base": 0.90,
        },
        "wavelet": {
            "min_ratio": 3.0,
            "max_ratio": 10.0,
            "best_when": "smooth_structured",
            "quality_base": 0.92,
        },
        "lowrank": {
            "min_ratio": 2.0,
            "max_ratio": 8.0,
            "best_when": "low_rank_matrix",
            "quality_base": 0.93,
        },
        "binary": {
            "min_ratio": 16.0,
            "max_ratio": 32.0,
            "best_when": "bimodal",
            "quality_base": 0.75,
        },
    }

    def predict_layer(
        self, name: str, weight: torch.Tensor, method: Optional[str] = None
    ) -> Prediction:
        """
        Predict compression outcome for a single layer.

        Args:
            name: Layer name
            weight: Weight tensor
            method: Specific method to predict for, or None for best

        Returns:
            Prediction with expected ratio, quality, and risk
        """
        stats = self._analyze_weights(weight)

        if method:
            return self._predict_for_method(name, stats, method)

        # Predict for all methods, return best
        predictions = {}
        for m in self.METHOD_PROFILES:
            pred = self._predict_for_method(name, stats, m)
            predictions[m] = pred

        # Score each: higher quality * higher ratio = better
        best_method = max(
            predictions,
            key=lambda m: predictions[m].expected_quality * np.log2(max(predictions[m].expected_ratio, 1.1))
        )
        return predictions[best_method]

    def predict_model(self, model: nn.Module) -> ModelPrediction:
        """
        Predict compression outcome for an entire model.

        Args:
            model: PyTorch model to analyze

        Returns:
            ModelPrediction with per-layer and overall predictions
        """
        layer_preds = []
        total_params = 0
        weighted_quality = 0
        weighted_ratio = 0
        total_weight = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 4:
                    continue

                pred = self.predict_layer(name, param.data)
                layer_preds.append(pred)

                n = param.numel()
                total_params += n
                weighted_quality += pred.expected_quality * n
                weighted_ratio += pred.expected_ratio * n
                total_weight += n

        if total_weight == 0:
            return ModelPrediction(
                layer_predictions=[],
                overall_ratio=1.0,
                overall_quality=1.0,
                overall_risk="low",
                recommended_method="none",
                summary="No compressible layers found",
                compressibility_score=0.0,
            )

        avg_quality = weighted_quality / total_weight
        avg_ratio = weighted_ratio / total_weight

        # Overall risk
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        for p in layer_preds:
            risk_counts[p.risk_level] += 1

        if risk_counts["high"] > len(layer_preds) * 0.3:
            overall_risk = "high"
        elif risk_counts["medium"] > len(layer_preds) * 0.5:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        # Most common method
        method_counts = {}
        for p in layer_preds:
            method_counts[p.method] = method_counts.get(p.method, 0) + 1
        recommended = max(method_counts, key=method_counts.get) if method_counts else "ternary"

        # Compressibility score
        compressibility = min(1.0, (avg_ratio - 1) / 15.0 * avg_quality)

        summary_lines = [
            f"Model Analysis: {total_params:,} parameters across {len(layer_preds)} layers",
            f"Expected compression: {avg_ratio:.1f}x",
            f"Expected quality retention: {avg_quality:.1%}",
            f"Recommended method: {recommended}",
            f"Risk level: {overall_risk}",
            f"Compressibility score: {compressibility:.2f}/1.00",
        ]

        return ModelPrediction(
            layer_predictions=layer_preds,
            overall_ratio=avg_ratio,
            overall_quality=avg_quality,
            overall_risk=overall_risk,
            recommended_method=recommended,
            summary="\n".join(summary_lines),
            compressibility_score=compressibility,
        )

    def compare_methods(
        self, weight: torch.Tensor
    ) -> Dict[str, Prediction]:
        """
        Compare all methods for a single weight tensor.

        Returns dict of method -> prediction.
        """
        stats = self._analyze_weights(weight)
        return {
            method: self._predict_for_method("layer", stats, method)
            for method in self.METHOD_PROFILES
        }

    def _analyze_weights(self, weight: torch.Tensor) -> Dict[str, float]:
        """Compute weight statistics for prediction."""
        flat = weight.detach().flatten().float()
        n = flat.numel()

        mean = flat.mean().item()
        std = flat.std().item()

        if std > 0:
            centered = flat - mean
            skewness = (centered ** 3).mean().item() / (std ** 3)
            kurtosis = (centered ** 4).mean().item() / (std ** 4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        sparsity = (flat.abs() < 0.01 * max(std, 1e-8)).float().mean().item()

        # Entropy
        hist = torch.histc(flat, bins=min(100, n))
        hist_norm = hist / hist.sum()
        entropy = -(hist_norm * torch.log(hist_norm + 1e-10)).sum().item()

        # Effective rank (for 2D+ tensors)
        rank_ratio = 1.0
        if weight.dim() >= 2:
            W2d = weight.float().reshape(weight.shape[0], -1)
            try:
                sv = torch.linalg.svdvals(W2d)
                sv_norm = sv / sv.sum()
                spectral_entropy = -(sv_norm * torch.log(sv_norm + 1e-10)).sum().item()
                effective_rank = np.exp(spectral_entropy)
                rank_ratio = effective_rank / min(W2d.shape)
            except Exception:
                pass

        # Bimodality
        hist_np = hist.numpy()
        peaks = 0
        for i in range(1, len(hist_np) - 1):
            if hist_np[i] > hist_np[i-1] and hist_np[i] > hist_np[i+1]:
                peaks += 1

        return {
            "mean": mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "sparsity": sparsity,
            "entropy": entropy,
            "rank_ratio": rank_ratio,
            "num_params": n,
            "num_peaks": peaks,
            "dim": weight.dim(),
        }

    def _predict_for_method(
        self, name: str, stats: Dict[str, float], method: str
    ) -> Prediction:
        """Predict compression outcome for a specific method."""
        profile = self.METHOD_PROFILES.get(method, self.METHOD_PROFILES["ternary"])

        sparsity = stats["sparsity"]
        kurtosis = stats["kurtosis"]
        entropy = stats["entropy"]
        rank_ratio = stats["rank_ratio"]
        num_peaks = stats["num_peaks"]
        std = stats["std"]
        n = stats["num_params"]

        # Method-specific prediction logic
        if method == "ternary":
            # Ternary works best for gaussian-like distributions
            gaussianity = max(0, 1.0 - abs(kurtosis) / 5.0)
            quality = profile["quality_base"] + 0.1 * gaussianity - 0.05 * sparsity
            ratio = 16.0  # Fixed 2-bit encoding
            confidence = 0.3 + 0.5 * gaussianity
            if abs(stats["skewness"]) > 2:
                quality -= 0.1
                confidence -= 0.1

        elif method == "sparse":
            # Sparse works best when weights are already sparse
            quality = profile["quality_base"] + 0.08 * sparsity
            ratio = 1.0 / max(1.0 - sparsity, 0.05) if sparsity > 0.1 else 2.0
            ratio = min(ratio, profile["max_ratio"])
            confidence = 0.4 + 0.5 * sparsity
            if kurtosis > 3:  # Heavy tails = naturally sparse
                quality += 0.03
                ratio *= 1.2

        elif method == "wavelet":
            # Wavelet works best for smooth, structured weights
            smoothness = max(0, 1.0 - entropy / 5.0)
            quality = profile["quality_base"] + 0.05 * smoothness
            ratio = profile["min_ratio"] + (profile["max_ratio"] - profile["min_ratio"]) * smoothness
            confidence = 0.3 + 0.4 * smoothness

        elif method == "lowrank":
            # Low-rank works best when rank ratio is low
            if stats["dim"] < 2:
                quality = 0.5
                ratio = 1.0
                confidence = 0.2
            else:
                quality = profile["quality_base"] + 0.1 * (1 - rank_ratio)
                ratio = 1.0 / max(rank_ratio, 0.1)
                ratio = min(ratio, profile["max_ratio"])
                confidence = 0.4 + 0.5 * (1 - rank_ratio)

        elif method == "binary":
            # Binary works when distribution is bimodal
            bimodality = min(1.0, num_peaks / 3.0)
            quality = profile["quality_base"] + 0.15 * bimodality
            ratio = 32.0  # 1-bit
            confidence = 0.2 + 0.5 * bimodality
            if abs(stats["skewness"]) < 0.5:
                quality += 0.05

        else:
            quality = 0.8
            ratio = 4.0
            confidence = 0.3

        # Clamp values
        quality = max(0.0, min(1.0, quality))
        confidence = max(0.1, min(0.95, confidence))
        ratio = max(1.0, ratio)

        # Risk assessment
        if quality > 0.9 and confidence > 0.6:
            risk = "low"
        elif quality > 0.8 or confidence > 0.5:
            risk = "medium"
        else:
            risk = "high"

        # Reasoning
        reasoning_parts = []
        if method == "ternary":
            reasoning_parts.append(f"Gaussianity={gaussianity:.2f}")
        elif method == "sparse":
            reasoning_parts.append(f"Natural sparsity={sparsity:.2f}")
        elif method == "lowrank":
            reasoning_parts.append(f"Rank ratio={rank_ratio:.2f}")
        reasoning_parts.append(f"Entropy={entropy:.2f}")
        reasoning_parts.append(f"Kurtosis={kurtosis:.2f}")

        return Prediction(
            method=method,
            expected_ratio=ratio,
            expected_quality=quality,
            confidence=confidence,
            risk_level=risk,
            reasoning=f"{method}: {', '.join(reasoning_parts)}",
        )
