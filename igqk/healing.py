"""
Self-Healing Compression - Autonomous accuracy recovery in production.

The compressed model monitors its own prediction confidence and automatically
adjusts compression strength per layer when it detects degradation.

Like an immune system: detects threats (accuracy drops) and responds
(decompresses critical layers) without human intervention.

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
    │  Input      │ ──> │  Compressed  │ ──> │  Confidence   │
    │  Stream     │     │  Model       │     │  Monitor      │
    └─────────────┘     └──────────────┘     └───────┬───────┘
                              ▲                      │
                              │                      ▼
                        ┌─────┴──────┐     ┌───────────────┐
                        │  Layer     │ <── │  Healing      │
                        │  Adjuster  │     │  Controller   │
                        └────────────┘     └───────────────┘
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from collections import deque

from igqk.theory.tlgt import TernaryLieGroup


@dataclass
class LayerHealth:
    """Health status of a compressed layer."""
    name: str
    compression_level: float   # 0.0=full, 1.0=max compression
    current_contribution: float  # how much this layer affects output
    sensitivity: float         # how sensitive output is to this layer
    heal_count: int = 0        # times this layer was healed


@dataclass
class HealingEvent:
    """Record of a healing action."""
    step: int
    layer: str
    old_level: float
    new_level: float
    trigger: str  # "confidence_drop", "entropy_spike", "drift_detected"
    confidence_before: float
    confidence_after: float


class SelfHealingModel(nn.Module):
    """
    Self-healing compressed model that maintains accuracy autonomously.

    Monitors prediction confidence in a sliding window and triggers
    per-layer decompression when accuracy degrades.

    Three healing strategies:
    1. Reactive: Decompress when confidence drops below threshold
    2. Predictive: Detect trends and heal BEFORE accuracy drops
    3. Gradual: Slowly re-compress after healing stabilizes
    """

    def __init__(
        self,
        model: nn.Module,
        confidence_threshold: float = 0.7,
        window_size: int = 100,
        heal_cooldown: int = 50,
        auto_recompress: bool = True,
    ):
        super().__init__()
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.heal_cooldown = heal_cooldown
        self.auto_recompress = auto_recompress

        # Store original weights for healing
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._compressed_weights: Dict[str, torch.Tensor] = {}
        self._layer_health: Dict[str, LayerHealth] = {}

        for name, param in model.named_parameters():
            self._original_weights[name] = param.data.clone()
            self._compressed_weights[name] = param.data.clone()
            self._layer_health[name] = LayerHealth(
                name=name, compression_level=0.0,
                current_contribution=0.0, sensitivity=0.0,
            )

        # Monitoring
        self._confidence_window: deque = deque(maxlen=window_size)
        self._entropy_window: deque = deque(maxlen=window_size)
        self._step = 0
        self._last_heal_step = -heal_cooldown
        self._healing_history: List[HealingEvent] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        self._step += 1

        # Monitor confidence
        with torch.no_grad():
            confidence = self._compute_confidence(output)
            self._confidence_window.append(confidence)

            entropy = self._compute_output_entropy(output)
            self._entropy_window.append(entropy)

        # Check if healing is needed
        if self._should_heal():
            self._heal()

        # Check if re-compression is possible
        if self.auto_recompress and self._should_recompress():
            self._recompress()

        return output

    def compress_initial(self, method: str = "ternary"):
        """Apply initial compression to the model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.numel() < 16:
                    continue

                self._original_weights[name] = param.data.clone()

                if method == "ternary":
                    tlgt = TernaryLieGroup(param.numel())
                    compressed, scale = tlgt.quantize(param.data)
                    param.data = compressed
                elif method == "sparse":
                    threshold = torch.quantile(param.data.abs().flatten(), 0.7)
                    param.data *= (param.data.abs() >= threshold).float()

                self._compressed_weights[name] = param.data.clone()
                self._layer_health[name].compression_level = 1.0

    def _compute_confidence(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        return probs.max(dim=-1).values.mean().item()

    def _compute_output_entropy(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy.mean().item()

    def _should_heal(self) -> bool:
        if len(self._confidence_window) < 10:
            return False
        if self._step - self._last_heal_step < self.heal_cooldown:
            return False

        recent = list(self._confidence_window)[-10:]
        avg_confidence = np.mean(recent)

        # Reactive: confidence below threshold
        if avg_confidence < self.confidence_threshold:
            return True

        # Predictive: confidence trending down
        if len(self._confidence_window) >= 20:
            older = list(self._confidence_window)[-20:-10]
            trend = np.mean(recent) - np.mean(older)
            if trend < -0.05:  # 5% drop trend
                return True

        return False

    def _heal(self):
        """Heal the most sensitive layers by reducing compression."""
        sensitivities = self._compute_layer_sensitivities()

        # Sort by sensitivity (most sensitive first)
        sorted_layers = sorted(
            sensitivities.items(), key=lambda x: x[1], reverse=True
        )

        confidence_before = np.mean(list(self._confidence_window)[-10:])
        healed_count = 0

        with torch.no_grad():
            for name, sensitivity in sorted_layers:
                health = self._layer_health[name]
                if health.compression_level <= 0.0:
                    continue
                if healed_count >= 3:  # Max 3 layers per heal
                    break

                # Blend: compressed → original (partial decompression)
                old_level = health.compression_level
                new_level = max(0.0, old_level - 0.3)

                param = dict(self.model.named_parameters())[name]
                original = self._original_weights[name]
                compressed = self._compressed_weights[name]

                # Interpolate between compressed and original
                param.data = new_level * compressed + (1 - new_level) * original

                health.compression_level = new_level
                health.heal_count += 1
                health.sensitivity = sensitivity
                healed_count += 1

                self._healing_history.append(HealingEvent(
                    step=self._step, layer=name,
                    old_level=old_level, new_level=new_level,
                    trigger="confidence_drop",
                    confidence_before=confidence_before,
                    confidence_after=0.0,  # Updated later
                ))

        self._last_heal_step = self._step

    def _should_recompress(self) -> bool:
        if len(self._confidence_window) < 20:
            return False
        if self._step - self._last_heal_step < self.heal_cooldown * 2:
            return False

        recent = list(self._confidence_window)[-20:]
        avg = np.mean(recent)
        std = np.std(recent)

        # Stable and high confidence → safe to recompress
        return avg > self.confidence_threshold + 0.1 and std < 0.02

    def _recompress(self):
        """Gradually re-compress healed layers."""
        with torch.no_grad():
            for name, health in self._layer_health.items():
                if health.compression_level >= 1.0:
                    continue
                if health.compression_level <= 0.0:
                    continue

                # Small step toward more compression
                new_level = min(1.0, health.compression_level + 0.05)

                param = dict(self.model.named_parameters())[name]
                original = self._original_weights[name]
                compressed = self._compressed_weights[name]

                param.data = new_level * compressed + (1 - new_level) * original
                health.compression_level = new_level

    def _compute_layer_sensitivities(self) -> Dict[str, float]:
        """Estimate sensitivity of each layer to compression."""
        sensitivities = {}
        for name, param in self.model.named_parameters():
            if param.numel() < 16:
                continue
            original = self._original_weights[name]
            current = param.data
            # Sensitivity = how much the weights changed from original
            diff = torch.norm(original - current).item()
            magnitude = torch.norm(original).item() + 1e-10
            sensitivities[name] = diff / magnitude
        return sensitivities

    def get_health_report(self) -> dict:
        """Get comprehensive health report."""
        layers = []
        for name, health in self._layer_health.items():
            layers.append({
                "name": name,
                "compression": f"{health.compression_level:.0%}",
                "heal_count": health.heal_count,
                "sensitivity": f"{health.sensitivity:.4f}",
            })

        avg_conf = np.mean(list(self._confidence_window)) if self._confidence_window else 0
        return {
            "step": self._step,
            "avg_confidence": f"{avg_conf:.4f}",
            "healing_events": len(self._healing_history),
            "layers": layers,
        }
