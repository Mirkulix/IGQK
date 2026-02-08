"""
Research Agent - Autonomously discovers compression opportunities.

The Research Agent continuously analyzes models to find:
- Weight distribution patterns that suggest specific compression methods
- Layers with unusual properties (outliers)
- Structural redundancies across layers
- Correlations between layer properties and compression success
- New potential compression strategies based on observed patterns

It reports findings to the Swarm for other agents to act on.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .base import Agent, AgentAction, AgentMessage


@dataclass
class Finding:
    """A research finding about a model or layer."""
    finding_type: str  # "pattern", "outlier", "redundancy", "correlation", "opportunity"
    layer_name: str
    description: str
    confidence: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""


class ResearchAgent(Agent):
    """
    Autonomous research agent that discovers compression opportunities.

    Perceives: model weights, distribution statistics, structural properties
    Plans: which layers to analyze, what patterns to look for
    Acts: runs statistical analysis, identifies patterns
    Reflects: learns which findings lead to successful compression
    """

    AGENT_TYPE = "researcher"

    def __init__(self, name: str = "", verbose: bool = False):
        super().__init__(name=name or "researcher", verbose=verbose)
        self.memory.add_goal("Discover compression opportunities", priority=8)
        self.memory.add_goal("Identify weight distribution patterns", priority=7)
        self._analyzed_layers: Dict[str, Dict] = {}

    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model weights and structure."""
        model = context.get("model")
        if model is None:
            return {"has_model": False}

        observations = {
            "has_model": True,
            "layers": {},
            "global": {},
        }

        all_means = []
        all_stds = []
        all_sparsities = []

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 4:
                    continue

                flat = param.data.flatten().float()
                mean = flat.mean().item()
                std = flat.std().item()
                sparsity = (flat.abs() < 0.01 * max(std, 1e-8)).float().mean().item()

                # Entropy
                w_clean = flat[torch.isfinite(flat)]
                if len(w_clean) > 1:
                    hist = torch.histc(w_clean, bins=min(50, len(w_clean)))
                    hist_n = hist / hist.sum()
                    entropy = -(hist_n * torch.log(hist_n + 1e-10)).sum().item()
                else:
                    entropy = 0.0

                # Kurtosis
                if std > 0:
                    centered = flat - mean
                    kurtosis = (centered ** 4).mean().item() / (std ** 4) - 3
                    skewness = (centered ** 3).mean().item() / (std ** 3)
                else:
                    kurtosis = 0.0
                    skewness = 0.0

                # SVD rank (for 2D+)
                rank_ratio = 1.0
                if param.dim() >= 2:
                    W2d = param.data.float().reshape(param.shape[0], -1)
                    try:
                        sv = torch.linalg.svdvals(W2d)
                        effective = (sv > sv[0] * 0.01).sum().item()
                        rank_ratio = effective / min(W2d.shape)
                    except Exception:
                        pass

                layer_obs = {
                    "params": param.numel(),
                    "shape": list(param.shape),
                    "mean": mean, "std": std,
                    "sparsity": sparsity,
                    "entropy": entropy,
                    "kurtosis": kurtosis,
                    "skewness": skewness,
                    "rank_ratio": rank_ratio,
                }
                observations["layers"][name] = layer_obs
                all_means.append(mean)
                all_stds.append(std)
                all_sparsities.append(sparsity)

        if all_stds:
            observations["global"] = {
                "num_layers": len(all_stds),
                "avg_std": float(np.mean(all_stds)),
                "std_of_stds": float(np.std(all_stds)),
                "avg_sparsity": float(np.mean(all_sparsities)),
                "total_params": sum(
                    v["params"] for v in observations["layers"].values()
                ),
            }

        return observations

    def plan(self, observations: Dict[str, Any],
             context: Dict[str, Any]) -> List[str]:
        """Plan analysis based on observations."""
        if not observations.get("has_model"):
            return []

        plan = ["analyze_patterns"]

        # Check if we have enough data for outlier detection
        if len(observations.get("layers", {})) > 2:
            plan.append("detect_outliers")

        # Check for redundancy if multiple layers
        if len(observations.get("layers", {})) > 3:
            plan.append("find_redundancies")

        # Always look for opportunities
        plan.append("identify_opportunities")

        return plan

    def act(self, plan: List[str], context: Dict[str, Any]) -> List[AgentAction]:
        """Execute research plan."""
        model = context.get("model")
        if model is None:
            return []

        actions = []
        observations = self.perceive(context)

        for task in plan:
            start = time.time()
            findings = []

            if task == "analyze_patterns":
                findings = self._analyze_patterns(observations)
            elif task == "detect_outliers":
                findings = self._detect_outliers(observations)
            elif task == "find_redundancies":
                findings = self._find_redundancies(observations, model)
            elif task == "identify_opportunities":
                findings = self._identify_opportunities(observations)

            elapsed = time.time() - start

            action = AgentAction(
                action_type=task,
                parameters={"num_layers": len(observations.get("layers", {}))},
                result=findings,
                success=len(findings) > 0,
                duration=elapsed,
            )
            actions.append(action)

            # Share findings via messages
            for finding in findings:
                self.send_message(
                    receiver="",  # broadcast
                    msg_type="knowledge",
                    content={
                        f"finding:{finding.finding_type}:{finding.layer_name}": {
                            "type": finding.finding_type,
                            "layer": finding.layer_name,
                            "description": finding.description,
                            "confidence": finding.confidence,
                            "action": finding.recommended_action,
                            "data": finding.data,
                        }
                    },
                    priority=int(finding.confidence * 10),
                )

            self.log(f"{task}: found {len(findings)} findings")

        return actions

    def reflect(self, actions: List[AgentAction], context: Dict[str, Any]):
        """Learn from research results."""
        super().reflect(actions, context)

        total_findings = 0
        for action in actions:
            if action.result:
                total_findings += len(action.result)
                for finding in action.result:
                    self.memory.believe(
                        f"layer:{finding.layer_name}:{finding.finding_type}",
                        finding.data,
                        confidence=finding.confidence,
                        source="research",
                    )

        self.memory.believe(
            "total_findings", total_findings,
            source="self_report",
        )

    def _analyze_patterns(self, obs: Dict) -> List[Finding]:
        """Find weight distribution patterns."""
        findings = []
        for name, stats in obs.get("layers", {}).items():
            # Bimodal pattern
            if abs(stats["kurtosis"]) < 0.5 and abs(stats["skewness"]) < 0.3:
                findings.append(Finding(
                    finding_type="pattern",
                    layer_name=name,
                    description="Gaussian-like distribution - ideal for ternary",
                    confidence=0.8,
                    data={"kurtosis": stats["kurtosis"], "skewness": stats["skewness"]},
                    recommended_action="compress_ternary",
                ))

            # Heavy-tailed
            if stats["kurtosis"] > 3:
                findings.append(Finding(
                    finding_type="pattern",
                    layer_name=name,
                    description="Heavy-tailed distribution - good for sparse",
                    confidence=0.7,
                    data={"kurtosis": stats["kurtosis"]},
                    recommended_action="compress_sparse",
                ))

            # Already sparse
            if stats["sparsity"] > 0.3:
                findings.append(Finding(
                    finding_type="pattern",
                    layer_name=name,
                    description=f"Naturally sparse ({stats['sparsity']:.0%})",
                    confidence=0.9,
                    data={"sparsity": stats["sparsity"]},
                    recommended_action="compress_sparse",
                ))

            # Low rank
            if stats["rank_ratio"] < 0.3:
                findings.append(Finding(
                    finding_type="pattern",
                    layer_name=name,
                    description=f"Low effective rank (ratio={stats['rank_ratio']:.2f})",
                    confidence=0.8,
                    data={"rank_ratio": stats["rank_ratio"]},
                    recommended_action="compress_lowrank",
                ))

        return findings

    def _detect_outliers(self, obs: Dict) -> List[Finding]:
        """Detect outlier layers."""
        findings = []
        layers = obs.get("layers", {})
        if len(layers) < 3:
            return findings

        stds = [s["std"] for s in layers.values()]
        mean_std = np.mean(stds)
        std_std = np.std(stds)

        for name, stats in layers.items():
            z_score = abs(stats["std"] - mean_std) / max(std_std, 1e-8)
            if z_score > 2.0:
                findings.append(Finding(
                    finding_type="outlier",
                    layer_name=name,
                    description=f"Outlier std deviation (z={z_score:.1f})",
                    confidence=min(0.9, z_score / 3.0),
                    data={"z_score": z_score, "std": stats["std"]},
                    recommended_action="investigate",
                ))

        return findings

    def _find_redundancies(self, obs: Dict, model: nn.Module) -> List[Finding]:
        """Find structural redundancies."""
        findings = []
        layers = list(obs.get("layers", {}).items())

        # Compare adjacent layers for similarity
        for i in range(len(layers) - 1):
            name_a, stats_a = layers[i]
            name_b, stats_b = layers[i + 1]

            # Check if similar distributions
            std_diff = abs(stats_a["std"] - stats_b["std"])
            mean_diff = abs(stats_a["mean"] - stats_b["mean"])
            entropy_diff = abs(stats_a["entropy"] - stats_b["entropy"])

            similarity = 1.0 / (1.0 + std_diff + mean_diff + entropy_diff)
            if similarity > 0.8:
                findings.append(Finding(
                    finding_type="redundancy",
                    layer_name=f"{name_a}+{name_b}",
                    description=f"Adjacent layers very similar (sim={similarity:.2f})",
                    confidence=similarity,
                    data={
                        "layer_a": name_a, "layer_b": name_b,
                        "similarity": similarity,
                    },
                    recommended_action="share_compression",
                ))

        return findings

    def _identify_opportunities(self, obs: Dict) -> List[Finding]:
        """Identify compression opportunities."""
        findings = []
        global_stats = obs.get("global", {})

        # Overall high sparsity = easy compression
        avg_sparsity = global_stats.get("avg_sparsity", 0)
        if avg_sparsity > 0.2:
            findings.append(Finding(
                finding_type="opportunity",
                layer_name="global",
                description=f"High average sparsity ({avg_sparsity:.0%}) - "
                           f"sparse compression will be very effective",
                confidence=0.8,
                data={"avg_sparsity": avg_sparsity},
                recommended_action="global_sparse",
            ))

        # Low std uniformity = can use same method everywhere
        std_of_stds = global_stats.get("std_of_stds", 0)
        avg_std = global_stats.get("avg_std", 1)
        if avg_std > 0 and std_of_stds / avg_std < 0.3:
            findings.append(Finding(
                finding_type="opportunity",
                layer_name="global",
                description="Uniform weight distributions - "
                           "single method will work for all layers",
                confidence=0.7,
                data={"std_of_stds": std_of_stds, "avg_std": avg_std},
                recommended_action="uniform_compression",
            ))

        return findings

    def get_findings(self) -> List[Finding]:
        """Get all accumulated findings."""
        findings = []
        for action in self.memory.experiences:
            if action.result and isinstance(action.result, list):
                findings.extend(action.result)
        return findings


import time
