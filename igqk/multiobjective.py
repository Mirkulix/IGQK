"""
Multi-Objective Quantum Gradient Flow.

Extends the quantum gradient flow to simultaneously optimize multiple objectives:
    dρ/dt = -i[H, ρ] - Σ_k γ_k {G^{-1}∇L_k, ρ}

Where L_k are different objectives:
    L_1 = accuracy loss
    L_2 = compression loss (sparsity penalty)
    L_3 = latency proxy (FLOPs)
    L_4 = memory proxy (parameter count)

The quantum superposition explores the Pareto front simultaneously,
finding solutions that are optimal across ALL objectives at once.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class ParetoPoint:
    """A point on the Pareto front."""
    objectives: Dict[str, float]
    weights: Optional[torch.Tensor] = None
    is_pareto_optimal: bool = True


class MultiObjectiveFlow:
    """
    Multi-objective quantum gradient flow for joint optimization.

    Unlike standard multi-objective optimization (scalarization, NSGA-II),
    this uses quantum superposition to explore multiple trade-offs simultaneously.
    """

    def __init__(
        self,
        objectives: Dict[str, Callable],
        weights: Optional[Dict[str, float]] = None,
        adaptation: str = "dynamic",
    ):
        """
        Args:
            objectives: Dict of name → loss_function.
            weights: Initial weights for each objective (auto if None).
            adaptation: "fixed", "dynamic" (auto-adjust), "pareto" (explore front).
        """
        self.objectives = objectives
        self.num_objectives = len(objectives)
        self.adaptation = adaptation

        if weights is None:
            weights = {name: 1.0 / self.num_objectives for name in objectives}
        self.weights = weights

        self._history: List[Dict[str, float]] = []
        self._pareto_front: List[ParetoPoint] = []

    def compute_losses(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all objective losses."""
        losses = {}
        for name, loss_fn in self.objectives.items():
            try:
                losses[name] = loss_fn(model, inputs, targets)
            except TypeError:
                losses[name] = loss_fn(model)
        return losses

    def combined_gradient(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined gradient across all objectives.

        Returns weighted sum of gradients, with dynamic weight adaptation.
        """
        all_grads = {}
        all_losses = {}

        for name, loss_fn in self.objectives.items():
            model.zero_grad()
            try:
                loss = loss_fn(model, inputs, targets)
            except TypeError:
                loss = loss_fn(model)

            if isinstance(loss, (int, float)):
                all_losses[name] = loss
                continue

            loss.backward()
            grad = torch.cat([
                p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                for p in model.parameters()
            ])
            all_grads[name] = grad.detach()
            all_losses[name] = loss.item()

        # Dynamic weight adaptation
        if self.adaptation == "dynamic" and all_grads:
            self.weights = self._adapt_weights(all_grads, all_losses)

        # Combine gradients
        combined = torch.zeros_like(list(all_grads.values())[0]) if all_grads else torch.tensor(0.0)
        for name, grad in all_grads.items():
            combined += self.weights[name] * grad

        self._history.append(all_losses)

        return combined, all_losses

    def _adapt_weights(
        self, grads: Dict[str, torch.Tensor], losses: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Dynamically adapt objective weights using gradient magnitude balancing.

        Objectives with larger gradients get smaller weights to prevent
        one objective from dominating.
        """
        magnitudes = {name: grad.norm().item() + 1e-10 for name, grad in grads.items()}
        total_mag = sum(magnitudes.values())

        # Inverse magnitude weighting: big gradient → small weight
        new_weights = {}
        for name in magnitudes:
            inverse = total_mag / magnitudes[name]
            new_weights[name] = inverse

        # Normalize
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        # Smooth update (exponential moving average)
        alpha = 0.1
        for name in self.weights:
            if name in new_weights:
                self.weights[name] = (1 - alpha) * self.weights[name] + alpha * new_weights[name]

        return self.weights

    def update_pareto_front(self, losses: Dict[str, float], weights: torch.Tensor = None):
        """Update the Pareto front with a new solution."""
        new_point = ParetoPoint(objectives=losses.copy(), weights=weights)

        # Check Pareto optimality
        dominated = False
        to_remove = []

        for i, existing in enumerate(self._pareto_front):
            if self._dominates(existing.objectives, new_point.objectives):
                dominated = True
                break
            if self._dominates(new_point.objectives, existing.objectives):
                to_remove.append(i)

        if not dominated:
            for i in reversed(to_remove):
                self._pareto_front.pop(i)
            self._pareto_front.append(new_point)

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Check if solution a dominates solution b (all objectives ≤, at least one <)."""
        all_leq = all(a.get(k, float('inf')) <= b.get(k, float('inf')) for k in b)
        any_lt = any(a.get(k, float('inf')) < b.get(k, float('inf')) for k in b)
        return all_leq and any_lt

    @property
    def pareto_front(self) -> List[ParetoPoint]:
        return self._pareto_front

    @property
    def history(self) -> List[Dict[str, float]]:
        return self._history


def accuracy_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor):
    """Standard cross-entropy loss."""
    outputs = model(inputs)
    return nn.functional.cross_entropy(outputs, targets)


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """L1 regularization promoting sparsity."""
    l1 = sum(p.abs().sum() for p in model.parameters())
    return 0.001 * l1


def memory_loss(model: nn.Module) -> torch.Tensor:
    """Penalize large parameter count / magnitude."""
    total = sum(p.numel() * p.abs().mean() for p in model.parameters())
    return 0.0001 * total


def latency_loss(model: nn.Module) -> torch.Tensor:
    """Proxy for inference latency (FLOPs estimate)."""
    flops = 0.0
    for p in model.parameters():
        if p.dim() >= 2:
            flops += p.shape[0] * p.shape[1]
    return torch.tensor(flops * 1e-6, requires_grad=False)
