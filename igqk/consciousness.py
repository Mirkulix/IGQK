"""
Quantum Consciousness Monitor - Real-time introspection into compression processes.

Like an MRI for neural networks: observe the quantum state DURING compression,
track how information flows, watch entropy change, see how the network "thinks"
about its own compression.

Features:
- Live quantum state tracking (entropy, purity, eigenvalues)
- Information flow analysis between layers
- Compression "heartbeat" - periodic health checks
- Anomaly detection (when compression goes wrong)
- Historical timeline of all compression events
- Layer-by-layer consciousness map
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class QuantumSnapshot:
    """A snapshot of a layer's quantum state at a point in time."""
    timestamp: float
    layer_name: str
    step: int

    # Weight statistics
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    num_params: int = 0

    # Quantum properties
    entropy: float = 0.0      # Von Neumann entropy approximation
    purity: float = 0.0       # Tr(rho^2)
    rank_estimate: int = 0    # Effective rank
    top_eigenvalue: float = 0.0
    eigenvalue_gap: float = 0.0  # Gap between top two eigenvalues

    # Compression indicators
    sparsity: float = 0.0     # Fraction of near-zero weights
    ternary_fraction: float = 0.0  # Fraction close to {-1, 0, +1}
    information_content: float = 0.0  # Estimated bits needed

    # Health
    is_healthy: bool = True
    anomalies: List[str] = field(default_factory=list)


@dataclass
class ConsciousnessEvent:
    """A notable event during compression."""
    timestamp: float
    event_type: str  # "phase_transition", "anomaly", "convergence", "collapse"
    layer_name: str
    description: str
    severity: float = 0.0  # 0-1


class QuantumConsciousnessMonitor:
    """
    Real-time introspection into IGQK compression processes.

    Observes quantum states during compression, detects anomalies,
    tracks information flow, and provides a "consciousness" view
    of the compression process.
    """

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.snapshots: List[QuantumSnapshot] = []
        self.events: List[ConsciousnessEvent] = []
        self._step = 0
        self._callbacks: List[Callable] = []
        self._prev_snapshots: Dict[str, QuantumSnapshot] = {}

    def observe(self, model: nn.Module, step: Optional[int] = None) -> List[QuantumSnapshot]:
        """
        Take a consciousness snapshot of the entire model.

        Args:
            model: The neural network to observe
            step: Optional step counter

        Returns:
            List of snapshots, one per layer
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        current_snapshots = []
        now = time.time()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 4:
                    continue

                snap = self._create_snapshot(name, param, now)
                current_snapshots.append(snap)

                # Detect anomalies by comparing with previous
                if name in self._prev_snapshots:
                    self._detect_anomalies(snap, self._prev_snapshots[name])

                self._prev_snapshots[name] = snap

        self.snapshots.extend(current_snapshots)
        if len(self.snapshots) > self.history_size:
            self.snapshots = self.snapshots[-self.history_size:]

        for cb in self._callbacks:
            cb(current_snapshots)

        return current_snapshots

    def _create_snapshot(
        self, name: str, param: nn.Parameter, timestamp: float
    ) -> QuantumSnapshot:
        """Create a quantum snapshot for a single layer."""
        w = param.data.flatten().float()
        n = w.numel()

        # Basic statistics
        mean = w.mean().item()
        std = w.std().item()

        # Quantum-inspired properties
        # Estimate entropy from weight distribution
        w_clean = w[torch.isfinite(w)]
        if len(w_clean) > 1:
            hist = torch.histc(w_clean, bins=min(100, len(w_clean)))
            hist_norm = hist / hist.sum()
            entropy = -(hist_norm * torch.log(hist_norm + 1e-10)).sum().item()
        else:
            entropy = 0.0

        # Estimate purity and eigenvalues from correlation structure
        if param.data.dim() >= 2:
            W = param.data.float()
            W2d = W.reshape(W.shape[0], -1)
            try:
                sv = torch.linalg.svdvals(W2d)
                sv_sq = sv ** 2
                sv_sq_norm = sv_sq / sv_sq.sum()
                purity = (sv_sq_norm ** 2).sum().item()
                top_ev = sv_sq_norm[0].item()
                gap = (sv_sq_norm[0] - sv_sq_norm[1]).item() if len(sv_sq_norm) > 1 else 0.0
                rank_est = int((sv > sv[0] * 0.01).sum().item())
            except Exception:
                purity = 1.0
                top_ev = 1.0
                gap = 0.0
                rank_est = 1
        else:
            purity = 1.0
            top_ev = 1.0
            gap = 0.0
            rank_est = 1

        # Compression indicators
        sparsity = (w.abs() < 0.01 * max(std, 1e-8)).float().mean().item()

        # Ternary fraction: how close weights are to {-scale, 0, +scale}
        if std > 0:
            normalized = w / std
            close_to_neg1 = (normalized + 1).abs() < 0.3
            close_to_0 = normalized.abs() < 0.3
            close_to_pos1 = (normalized - 1).abs() < 0.3
            ternary_frac = (close_to_neg1 | close_to_0 | close_to_pos1).float().mean().item()
        else:
            ternary_frac = 1.0

        # Information content (bits needed)
        unique_vals = min(w.unique().numel(), n)
        info_content = np.log2(max(unique_vals, 2))

        # Health check
        anomalies = []
        is_healthy = True
        if torch.isnan(w).any():
            anomalies.append("NaN detected")
            is_healthy = False
        if torch.isinf(w).any():
            anomalies.append("Inf detected")
            is_healthy = False
        if std == 0 and n > 1:
            anomalies.append("Zero variance (collapsed)")
            is_healthy = False
        if std > 100:
            anomalies.append("Extremely high variance")

        return QuantumSnapshot(
            timestamp=timestamp,
            layer_name=name,
            step=self._step,
            mean=mean,
            std=std,
            min_val=w.min().item(),
            max_val=w.max().item(),
            num_params=n,
            entropy=entropy,
            purity=purity,
            rank_estimate=rank_est,
            top_eigenvalue=top_ev,
            eigenvalue_gap=gap,
            sparsity=sparsity,
            ternary_fraction=ternary_frac,
            information_content=info_content,
            is_healthy=is_healthy,
            anomalies=anomalies,
        )

    def _detect_anomalies(self, current: QuantumSnapshot, previous: QuantumSnapshot):
        """Detect anomalies by comparing snapshots."""
        # Sudden entropy change (phase transition)
        entropy_change = abs(current.entropy - previous.entropy)
        if entropy_change > 1.0:
            event = ConsciousnessEvent(
                timestamp=current.timestamp,
                event_type="phase_transition",
                layer_name=current.layer_name,
                description=f"Entropy jumped by {entropy_change:.2f} "
                           f"({previous.entropy:.2f} -> {current.entropy:.2f})",
                severity=min(1.0, entropy_change / 3.0),
            )
            self.events.append(event)

        # Rank collapse
        if (previous.rank_estimate > 1 and
                current.rank_estimate < previous.rank_estimate * 0.5):
            event = ConsciousnessEvent(
                timestamp=current.timestamp,
                event_type="collapse",
                layer_name=current.layer_name,
                description=f"Rank collapsed from {previous.rank_estimate} "
                           f"to {current.rank_estimate}",
                severity=0.7,
            )
            self.events.append(event)

        # Convergence detection
        std_change = abs(current.std - previous.std)
        if std_change < 1e-6 and previous.std > 0:
            event = ConsciousnessEvent(
                timestamp=current.timestamp,
                event_type="convergence",
                layer_name=current.layer_name,
                description="Layer weights have converged (no change)",
                severity=0.2,
            )
            self.events.append(event)

        # NaN/Inf appearance
        if not previous.is_healthy or not current.is_healthy:
            if previous.is_healthy and not current.is_healthy:
                event = ConsciousnessEvent(
                    timestamp=current.timestamp,
                    event_type="anomaly",
                    layer_name=current.layer_name,
                    description=f"Health degraded: {', '.join(current.anomalies)}",
                    severity=1.0,
                )
                self.events.append(event)

    def on_snapshot(self, callback: Callable):
        """Register a callback for new snapshots."""
        self._callbacks.append(callback)

    def get_layer_timeline(self, layer_name: str) -> List[QuantumSnapshot]:
        """Get all snapshots for a specific layer."""
        return [s for s in self.snapshots if s.layer_name == layer_name]

    def get_consciousness_map(self) -> Dict[str, Dict[str, float]]:
        """
        Get a map of the model's "consciousness" - current state of all layers.

        Returns dict: layer_name -> {entropy, purity, sparsity, health, ...}
        """
        result = {}
        for name, snap in self._prev_snapshots.items():
            result[name] = {
                "entropy": snap.entropy,
                "purity": snap.purity,
                "sparsity": snap.sparsity,
                "ternary_readiness": snap.ternary_fraction,
                "rank": snap.rank_estimate,
                "info_bits": snap.information_content,
                "healthy": 1.0 if snap.is_healthy else 0.0,
                "std": snap.std,
            }
        return result

    def get_health_report(self) -> Dict[str, Any]:
        """Get an overall health report."""
        if not self._prev_snapshots:
            return {"status": "no_data", "layers": 0}

        all_healthy = all(s.is_healthy for s in self._prev_snapshots.values())
        unhealthy = [
            name for name, s in self._prev_snapshots.items() if not s.is_healthy
        ]
        avg_entropy = np.mean([s.entropy for s in self._prev_snapshots.values()])
        avg_sparsity = np.mean([s.sparsity for s in self._prev_snapshots.values()])
        recent_events = self.events[-10:]

        return {
            "status": "healthy" if all_healthy else "warning",
            "layers": len(self._prev_snapshots),
            "all_healthy": all_healthy,
            "unhealthy_layers": unhealthy,
            "avg_entropy": float(avg_entropy),
            "avg_sparsity": float(avg_sparsity),
            "total_events": len(self.events),
            "recent_events": [
                {"type": e.event_type, "layer": e.layer_name,
                 "desc": e.description, "severity": e.severity}
                for e in recent_events
            ],
        }

    def get_information_flow(self) -> List[Dict[str, float]]:
        """
        Analyze information flow through the network.
        Returns per-layer information metrics in order.
        """
        layers = sorted(
            self._prev_snapshots.items(),
            key=lambda x: x[0]
        )
        flow = []
        for name, snap in layers:
            flow.append({
                "layer": name,
                "entropy": snap.entropy,
                "info_bits": snap.information_content,
                "rank": snap.rank_estimate,
                "bottleneck_score": snap.purity,  # High purity = bottleneck
            })
        return flow

    @property
    def total_snapshots(self) -> int:
        return len(self.snapshots)

    @property
    def total_events(self) -> int:
        return len(self.events)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Quantum Consciousness Monitor",
            "=" * 50,
            f"  Steps observed: {self._step}",
            f"  Total snapshots: {len(self.snapshots)}",
            f"  Total events: {len(self.events)}",
            f"  Layers tracked: {len(self._prev_snapshots)}",
        ]

        health = self.get_health_report()
        if health["status"] != "no_data":
            lines.extend([
                f"  Status: {health['status']}",
                f"  Avg entropy: {health['avg_entropy']:.3f}",
                f"  Avg sparsity: {health['avg_sparsity']:.3f}",
            ])

        if self.events:
            lines.append("")
            lines.append("  Recent events:")
            for e in self.events[-5:]:
                lines.append(
                    f"    [{e.event_type}] {e.layer_name}: {e.description}"
                )

        return "\n".join(lines)
