"""
Time-Travel Debugging - Step through compression history.

Record every state change during compression and replay them:
- Step forward/backward through the compression process
- Compare any two points in time
- Visualize how weights change over time
- Find the exact step where quality dropped
- Undo compression to any checkpoint

    Timeline:
    t=0          t=5          t=10         t=15
    ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
    │orig │ ──> │step5│ ──> │step10│ ──> │final│
    │model│     │     │     │      │     │     │
    └─────┘     └─────┘     └─────┘     └─────┘
        ▲           ▲           ▲           ▲
        │           │           │           │
    ┌───┴───────────┴───────────┴───────────┴───┐
    │         Time-Travel Debugger               │
    │  ◄◄  ◄  ▮▮  ►  ►►  │  t=7  │  Compare   │
    └────────────────────────────────────────────┘
"""

import time
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TimePoint:
    """A snapshot of model state at a point in time."""
    step: int
    timestamp: float
    label: str
    state_dict: Dict[str, torch.Tensor]

    # Aggregate metrics
    total_params: int = 0
    unique_values: int = 0
    sparsity: float = 0.0
    mean_std: float = 0.0
    estimated_quality: float = 1.0  # vs original

    # Per-layer stats
    layer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TimeDiff:
    """Difference between two time points."""
    from_step: int
    to_step: int
    total_changed_params: int
    total_params: int
    change_fraction: float
    layer_changes: Dict[str, Dict[str, float]]
    quality_change: float  # How much quality changed


class TimeTravelDebugger:
    """
    Record and replay compression history.

    Enables stepping through compression process to understand
    exactly what happened at each stage.
    """

    def __init__(self, max_checkpoints: int = 50):
        self.max_checkpoints = max_checkpoints
        self.timeline: List[TimePoint] = []
        self._current_step = 0
        self._cursor = -1  # Current position in timeline
        self._original_state: Optional[Dict[str, torch.Tensor]] = None

    def record(
        self, model: nn.Module, label: str = "", auto_metrics: bool = True
    ) -> int:
        """
        Record the current model state as a time point.

        Args:
            model: Model to snapshot
            label: Human-readable label for this point

        Returns:
            Step number of this recording
        """
        state = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
        }

        if self._original_state is None:
            self._original_state = {k: v.clone() for k, v in state.items()}

        step = self._current_step
        self._current_step += 1

        tp = TimePoint(
            step=step,
            timestamp=time.time(),
            label=label or f"step_{step}",
            state_dict=state,
        )

        if auto_metrics:
            self._compute_metrics(tp)

        self.timeline.append(tp)
        self._cursor = len(self.timeline) - 1

        # Limit checkpoints
        if len(self.timeline) > self.max_checkpoints:
            # Keep first, last, and evenly spaced
            n = self.max_checkpoints
            indices = [0]
            indices.extend(
                int(i * (len(self.timeline) - 1) / (n - 1))
                for i in range(1, n)
            )
            self.timeline = [self.timeline[i] for i in sorted(set(indices))]
            self._cursor = len(self.timeline) - 1

        return step

    def _compute_metrics(self, tp: TimePoint):
        """Compute metrics for a time point."""
        total_params = 0
        total_zeros = 0
        all_unique = set()
        stds = []

        for name, param in tp.state_dict.items():
            flat = param.flatten().float()
            n = flat.numel()
            total_params += n
            zeros = (flat == 0).sum().item()
            total_zeros += zeros
            unique = flat.unique()
            all_unique.update(unique.tolist()[:50])
            stds.append(flat.std().item())

            tp.layer_stats[name] = {
                "params": n,
                "sparsity": zeros / max(n, 1),
                "std": flat.std().item(),
                "mean": flat.mean().item(),
                "unique_values": len(unique),
            }

        tp.total_params = total_params
        tp.unique_values = len(all_unique)
        tp.sparsity = total_zeros / max(total_params, 1)
        tp.mean_std = float(np.mean(stds)) if stds else 0.0

        # Estimate quality vs original
        if self._original_state:
            total_err = 0.0
            total_norm = 0.0
            for name in tp.state_dict:
                if name in self._original_state:
                    orig = self._original_state[name].float()
                    curr = tp.state_dict[name].float()
                    total_err += (orig - curr).norm().item() ** 2
                    total_norm += orig.norm().item() ** 2

            if total_norm > 0:
                tp.estimated_quality = max(0, 1.0 - np.sqrt(total_err / total_norm))

    @property
    def current(self) -> Optional[TimePoint]:
        """Get the current time point."""
        if 0 <= self._cursor < len(self.timeline):
            return self.timeline[self._cursor]
        return None

    @property
    def num_checkpoints(self) -> int:
        return len(self.timeline)

    def goto(self, step: int) -> Optional[TimePoint]:
        """Jump to a specific step."""
        for i, tp in enumerate(self.timeline):
            if tp.step == step:
                self._cursor = i
                return tp
        return None

    def forward(self) -> Optional[TimePoint]:
        """Move one step forward."""
        if self._cursor < len(self.timeline) - 1:
            self._cursor += 1
            return self.timeline[self._cursor]
        return None

    def backward(self) -> Optional[TimePoint]:
        """Move one step backward."""
        if self._cursor > 0:
            self._cursor -= 1
            return self.timeline[self._cursor]
        return None

    def first(self) -> Optional[TimePoint]:
        """Jump to the first checkpoint."""
        if self.timeline:
            self._cursor = 0
            return self.timeline[0]
        return None

    def last(self) -> Optional[TimePoint]:
        """Jump to the last checkpoint."""
        if self.timeline:
            self._cursor = len(self.timeline) - 1
            return self.timeline[-1]
        return None

    def restore(self, model: nn.Module, step: Optional[int] = None) -> bool:
        """
        Restore model to a specific time point.

        Args:
            model: Model to restore
            step: Step to restore to (None = current cursor position)

        Returns:
            True if successful
        """
        if step is not None:
            tp = self.goto(step)
        else:
            tp = self.current

        if tp is None:
            return False

        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for name, saved in tp.state_dict.items():
                if name in param_dict:
                    device = param_dict[name].data.device
                    param_dict[name].data.copy_(saved.to(device))

        return True

    def diff(self, step_a: int, step_b: int) -> Optional[TimeDiff]:
        """
        Compare two time points.

        Args:
            step_a: First step
            step_b: Second step

        Returns:
            TimeDiff describing the changes
        """
        tp_a = None
        tp_b = None
        for tp in self.timeline:
            if tp.step == step_a:
                tp_a = tp
            if tp.step == step_b:
                tp_b = tp

        if tp_a is None or tp_b is None:
            return None

        total_params = 0
        total_changed = 0
        layer_changes = {}

        for name in tp_a.state_dict:
            if name not in tp_b.state_dict:
                continue

            a = tp_a.state_dict[name].flatten().float()
            b = tp_b.state_dict[name].flatten().float()
            n = a.numel()
            total_params += n

            changed = (a != b).sum().item()
            total_changed += changed

            err = (a - b).norm().item()
            rel_err = err / max(a.norm().item(), 1e-8)

            layer_changes[name] = {
                "changed_params": changed,
                "change_fraction": changed / max(n, 1),
                "absolute_error": err,
                "relative_error": rel_err,
                "sparsity_a": (a == 0).float().mean().item(),
                "sparsity_b": (b == 0).float().mean().item(),
            }

        return TimeDiff(
            from_step=step_a,
            to_step=step_b,
            total_changed_params=int(total_changed),
            total_params=total_params,
            change_fraction=total_changed / max(total_params, 1),
            layer_changes=layer_changes,
            quality_change=tp_b.estimated_quality - tp_a.estimated_quality,
        )

    def find_quality_drop(self, threshold: float = 0.05) -> Optional[int]:
        """
        Find the first step where quality dropped significantly.

        Args:
            threshold: Minimum quality drop to detect

        Returns:
            Step number where quality first dropped, or None
        """
        for i in range(1, len(self.timeline)):
            prev = self.timeline[i - 1]
            curr = self.timeline[i]
            drop = prev.estimated_quality - curr.estimated_quality
            if drop > threshold:
                return curr.step
        return None

    def get_quality_timeline(self) -> List[Tuple[int, float]]:
        """Get quality over time."""
        return [(tp.step, tp.estimated_quality) for tp in self.timeline]

    def get_sparsity_timeline(self) -> List[Tuple[int, float]]:
        """Get sparsity over time."""
        return [(tp.step, tp.sparsity) for tp in self.timeline]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Time-Travel Debugger",
            "=" * 50,
            f"  Checkpoints: {len(self.timeline)}",
            f"  Current position: {self._cursor}",
        ]

        if self.timeline:
            first = self.timeline[0]
            last = self.timeline[-1]
            lines.extend([
                f"  First step: {first.step} ({first.label})",
                f"  Last step: {last.step} ({last.label})",
                f"  Quality: {first.estimated_quality:.4f} -> {last.estimated_quality:.4f}",
                f"  Sparsity: {first.sparsity:.4f} -> {last.sparsity:.4f}",
            ])

            drop = self.find_quality_drop()
            if drop is not None:
                lines.append(f"  First quality drop at step: {drop}")

        if self.timeline:
            lines.append("")
            lines.append("  Timeline:")
            for tp in self.timeline[:20]:
                marker = " >>>" if tp.step == (self.current.step if self.current else -1) else "    "
                lines.append(
                    f"  {marker} step={tp.step} "
                    f"quality={tp.estimated_quality:.4f} "
                    f"sparsity={tp.sparsity:.4f} "
                    f"({tp.label})"
                )
            if len(self.timeline) > 20:
                lines.append(f"    ... and {len(self.timeline) - 20} more")

        return "\n".join(lines)
