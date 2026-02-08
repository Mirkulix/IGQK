"""
Optimizer Agent - Tunes compression strategies based on experiment results.

The Optimizer Agent:
- Takes experiment results from the Experimenter
- Uses Bayesian-inspired reasoning to refine parameters
- Generates hypotheses about optimal parameters
- Sends tuned strategies back for validation

It maintains a parameter landscape model and narrows down the optimal
configuration over time.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base import Agent, AgentAction


@dataclass
class OptimizationProposal:
    """A proposed parameter configuration."""
    method: str
    layer_name: str
    params: Dict[str, float]
    expected_ratio: float
    expected_quality: float
    reasoning: str


class OptimizerAgent(Agent):
    """
    Optimization agent that tunes compression parameters.

    Uses experiment results to model the parameter-performance landscape
    and proposes optimal configurations.
    """

    AGENT_TYPE = "optimizer"

    def __init__(self, name: str = "", verbose: bool = False):
        super().__init__(name=name or "optimizer", verbose=verbose)
        self.memory.add_goal("Find optimal parameters per method", priority=8)
        self.memory.add_goal("Minimize distortion while maximizing ratio", priority=9)
        self._parameter_history: Dict[str, List[Tuple[float, float]]] = {}
        self.proposals: List[OptimizationProposal] = []

    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect experiment results from messages."""
        observations = {
            "experiment_results": [],
            "has_model": context.get("model") is not None,
        }

        for msg in self.memory.messages_received[-50:]:
            if msg.msg_type == "result":
                observations["experiment_results"].append(msg.content)

        return observations

    def plan(self, observations: Dict[str, Any],
             context: Dict[str, Any]) -> List[str]:
        """Plan optimization strategies."""
        plan = []

        if observations["experiment_results"]:
            plan.append("update_landscape")
            plan.append("generate_proposals")

        if self._parameter_history:
            plan.append("refine_parameters")

        return plan

    def act(self, plan: List[str], context: Dict[str, Any]) -> List[AgentAction]:
        """Execute optimization plan."""
        actions = []

        for task in plan:
            start = time.time()

            if task == "update_landscape":
                result = self._update_landscape()
            elif task == "generate_proposals":
                result = self._generate_proposals(context)
            elif task == "refine_parameters":
                result = self._refine_parameters()
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

            # Send proposals as tasks
            if task == "generate_proposals" and result:
                for proposal in result:
                    self.send_message(
                        receiver="experimenter",
                        msg_type="task",
                        content={
                            f"proposal:{proposal.layer_name}": {
                                "action": f"compress_{proposal.method}",
                                "layer": proposal.layer_name,
                                "params": proposal.params,
                                "data": {
                                    "expected_ratio": proposal.expected_ratio,
                                    "expected_quality": proposal.expected_quality,
                                },
                            }
                        },
                        priority=7,
                    )

            self.log(f"{task}: {'success' if action.success else 'no data'}")

        return actions

    def reflect(self, actions: List[AgentAction], context: Dict[str, Any]):
        """Learn from optimization results."""
        super().reflect(actions, context)

        # Track which proposals led to good results
        if self.proposals:
            self.memory.believe(
                "total_proposals",
                len(self.proposals),
                source="self_report",
            )

    def _update_landscape(self) -> Dict[str, Any]:
        """Update parameter-performance landscape from experiment results."""
        for msg in self.memory.messages_received[-50:]:
            if msg.msg_type == "result":
                content = msg.content
                method = content.get("method", "")
                distortion = content.get("distortion", 0)
                ratio = content.get("ratio", 1)

                if method:
                    key = method
                    if key not in self._parameter_history:
                        self._parameter_history[key] = []
                    score = ratio / max(distortion + 0.01, 0.01)
                    self._parameter_history[key].append((ratio, score))

        return {"methods_tracked": len(self._parameter_history)}

    def _generate_proposals(self, context: Dict[str, Any]) -> List[OptimizationProposal]:
        """Generate new parameter proposals based on landscape."""
        proposals = []
        model = context.get("model")
        if model is None:
            return proposals

        # Analyze what we know
        for method, history in self._parameter_history.items():
            if not history:
                continue

            # Find best performing configuration
            best_ratio, best_score = max(history, key=lambda x: x[1])

            # Propose variations around best
            for name, param in model.named_parameters():
                if param.numel() < 64:
                    continue

                if method == "ternary":
                    # Try thresholds around the optimum
                    for threshold in [0.5, 0.6, 0.7, 0.8]:
                        proposals.append(OptimizationProposal(
                            method=method,
                            layer_name=name,
                            params={"threshold": threshold},
                            expected_ratio=16.0,
                            expected_quality=max(0.8, 1.0 - threshold * 0.15),
                            reasoning=f"Ternary with threshold={threshold} "
                                     f"based on best score={best_score:.2f}",
                        ))
                elif method == "sparse":
                    for keep in [0.15, 0.25, 0.35]:
                        proposals.append(OptimizationProposal(
                            method=method,
                            layer_name=name,
                            params={"keep_ratio": keep},
                            expected_ratio=1.0 / max(keep, 0.01),
                            expected_quality=0.85 + keep * 0.1,
                            reasoning=f"Sparse with keep={keep} "
                                     f"based on landscape analysis",
                        ))

                # Limit proposals per step
                if len(proposals) >= 10:
                    break
            if len(proposals) >= 10:
                break

        self.proposals.extend(proposals)
        return proposals

    def _refine_parameters(self) -> Dict[str, Any]:
        """Refine parameter estimates using history."""
        refined = {}
        for method, history in self._parameter_history.items():
            if len(history) < 3:
                continue

            scores = [s for _, s in history]
            ratios = [r for r, _ in history]

            refined[method] = {
                "avg_score": float(np.mean(scores)),
                "best_score": float(max(scores)),
                "avg_ratio": float(np.mean(ratios)),
                "num_experiments": len(history),
                "trend": "improving" if len(scores) > 2 and scores[-1] > scores[0] else "stable",
            }

            self.memory.believe(
                f"method_performance:{method}",
                refined[method],
                confidence=min(0.9, len(history) / 20),
                source="optimization",
            )

        return refined

    def get_best_config(self, method: str) -> Optional[Dict[str, float]]:
        """Get the best known configuration for a method."""
        belief = self.memory.recall(f"method_performance:{method}")
        if belief:
            return belief
        return None
