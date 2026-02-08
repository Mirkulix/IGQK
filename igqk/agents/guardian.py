"""
Guardian Agent - Quality monitor and safety system.

The Guardian Agent:
- Monitors compression quality in real-time
- Detects quality regressions before they cause problems
- Validates that compression meets minimum standards
- Can VETO compression decisions that are too aggressive
- Maintains quality history and trends

Think of it as the immune system of the compression pipeline.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base import Agent, AgentAction


@dataclass
class QualityReport:
    """Quality assessment of a compression operation."""
    layer_name: str
    method: str
    distortion: float
    relative_error: float
    sparsity: float
    is_acceptable: bool
    verdict: str  # "approved", "warning", "rejected"
    reason: str


@dataclass
class QualityPolicy:
    """Quality thresholds and rules."""
    max_relative_error: float = 0.15
    max_distortion_ratio: float = 0.3
    min_sparsity_improvement: float = 0.01
    max_weight_collapse: float = 0.95  # Max fraction of weights = 0
    require_gradient_preservation: bool = True


class GuardianAgent(Agent):
    """
    Quality guardian that monitors and validates compression operations.

    The Guardian ensures no compression degrades the model beyond
    acceptable limits. It can veto bad compression decisions.
    """

    AGENT_TYPE = "guardian"

    def __init__(self, name: str = "", verbose: bool = False,
                 policy: Optional[QualityPolicy] = None):
        super().__init__(name=name or "guardian", verbose=verbose)
        self.policy = policy or QualityPolicy()
        self.memory.add_goal("Ensure compression quality", priority=10)
        self.memory.add_goal("Detect quality regressions", priority=9)
        self.reports: List[QualityReport] = []
        self._vetoes = 0
        self._approvals = 0

    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor current model state and compression results."""
        observations = {
            "has_model": context.get("model") is not None,
            "compression_results": [],
            "model_health": {},
        }

        # Check for experiment results
        for msg in self.memory.messages_received[-30:]:
            if msg.msg_type == "result":
                observations["compression_results"].append(msg.content)

        # Check model health if available
        model = context.get("model")
        if model is not None:
            observations["model_health"] = self._assess_health(model)

        return observations

    def plan(self, observations: Dict[str, Any],
             context: Dict[str, Any]) -> List[str]:
        """Plan quality checks."""
        plan = []

        if observations["compression_results"]:
            plan.append("validate_results")

        if observations.get("model_health"):
            plan.append("check_health")

        if len(self.reports) > 5:
            plan.append("trend_analysis")

        return plan

    def act(self, plan: List[str], context: Dict[str, Any]) -> List[AgentAction]:
        """Execute quality checks."""
        actions = []

        for task in plan:
            start = time.time()

            if task == "validate_results":
                result = self._validate_results(context)
            elif task == "check_health":
                result = self._check_health(context)
            elif task == "trend_analysis":
                result = self._analyze_trends()
            else:
                result = None

            elapsed = time.time() - start
            action = AgentAction(
                action_type=task,
                result=result,
                success=result is not None,
                duration=elapsed,
            )
            actions.append(action)
            self.log(f"{task}: complete")

        return actions

    def reflect(self, actions: List[AgentAction], context: Dict[str, Any]):
        """Update quality beliefs."""
        super().reflect(actions, context)
        self.memory.believe(
            "quality_summary",
            {
                "total_checks": len(self.reports),
                "approvals": self._approvals,
                "vetoes": self._vetoes,
                "approval_rate": self._approvals / max(self._approvals + self._vetoes, 1),
            },
            source="self_report",
        )

    def validate(self, layer_name: str, original: torch.Tensor,
                 compressed: torch.Tensor, method: str) -> QualityReport:
        """
        Validate a single compression operation.

        This is the main entry point for other agents to check quality.
        """
        orig = original.flatten().float()
        comp = compressed.flatten().float()

        distortion = (orig - comp).norm().item()
        orig_norm = orig.norm().item()
        relative_error = distortion / max(orig_norm, 1e-8)
        sparsity = (comp == 0).float().mean().item()

        # Check against policy
        violations = []

        if relative_error > self.policy.max_relative_error:
            violations.append(
                f"Relative error {relative_error:.3f} > "
                f"max {self.policy.max_relative_error}"
            )

        if sparsity > self.policy.max_weight_collapse:
            violations.append(
                f"Weight collapse: {sparsity:.1%} zeros > "
                f"max {self.policy.max_weight_collapse:.0%}"
            )

        if distortion / max(orig_norm, 1e-8) > self.policy.max_distortion_ratio:
            violations.append(
                f"Distortion ratio {distortion/max(orig_norm,1e-8):.3f} > "
                f"max {self.policy.max_distortion_ratio}"
            )

        # Determine verdict
        if not violations:
            verdict = "approved"
            self._approvals += 1
        elif len(violations) == 1 and relative_error < self.policy.max_relative_error * 1.5:
            verdict = "warning"
            self._approvals += 1
        else:
            verdict = "rejected"
            self._vetoes += 1

        report = QualityReport(
            layer_name=layer_name,
            method=method,
            distortion=distortion,
            relative_error=relative_error,
            sparsity=sparsity,
            is_acceptable=verdict != "rejected",
            verdict=verdict,
            reason="; ".join(violations) if violations else "All checks passed",
        )
        self.reports.append(report)

        # Alert on rejection
        if verdict == "rejected":
            self.send_message(
                receiver="",
                msg_type="alert",
                content={
                    "alert_type": "quality_veto",
                    "layer": layer_name,
                    "method": method,
                    "reason": report.reason,
                    "relative_error": relative_error,
                },
                priority=9,
            )

        return report

    def _validate_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate recent experiment results."""
        results = {
            "checked": 0,
            "approved": 0,
            "rejected": 0,
        }

        for msg in self.memory.messages_received[-30:]:
            if msg.msg_type == "result":
                distortion = msg.content.get("distortion", 0)
                ratio = msg.content.get("ratio", 1)
                method = msg.content.get("method", "unknown")

                results["checked"] += 1
                if distortion < self.policy.max_distortion_ratio:
                    results["approved"] += 1
                else:
                    results["rejected"] += 1
                    self.log(f"VETO: {method} distortion={distortion:.4f}")

        return results

    def _check_health(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall model health."""
        model = context.get("model")
        if model is None:
            return {"status": "no_model"}

        health = self._assess_health(model)

        if health.get("has_nan"):
            self.send_message(
                receiver="",
                msg_type="alert",
                content={"alert_type": "nan_detected", "details": health},
                priority=10,
            )

        return health

    def _assess_health(self, model: nn.Module) -> Dict[str, Any]:
        """Assess model health."""
        total_params = 0
        total_zeros = 0
        has_nan = False
        has_inf = False

        with torch.no_grad():
            for name, param in model.named_parameters():
                flat = param.data.flatten()
                total_params += flat.numel()
                total_zeros += (flat == 0).sum().item()
                if torch.isnan(flat).any():
                    has_nan = True
                if torch.isinf(flat).any():
                    has_inf = True

        return {
            "total_params": total_params,
            "sparsity": total_zeros / max(total_params, 1),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "healthy": not has_nan and not has_inf,
        }

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        if len(self.reports) < 3:
            return {"trend": "insufficient_data"}

        recent = self.reports[-10:]
        errors = [r.relative_error for r in recent]

        trend = "stable"
        if len(errors) > 3:
            if errors[-1] > errors[0] * 1.5:
                trend = "degrading"
            elif errors[-1] < errors[0] * 0.7:
                trend = "improving"

        result = {
            "trend": trend,
            "avg_error": float(np.mean(errors)),
            "min_error": float(min(errors)),
            "max_error": float(max(errors)),
            "approval_rate": self._approvals / max(self._approvals + self._vetoes, 1),
        }

        if trend == "degrading":
            self.send_message(
                receiver="",
                msg_type="alert",
                content={
                    "alert_type": "quality_degradation",
                    "trend": result,
                },
                priority=8,
            )

        return result

    @property
    def approval_rate(self) -> float:
        total = self._approvals + self._vetoes
        return self._approvals / max(total, 1)

    @property
    def veto_count(self) -> int:
        return self._vetoes
