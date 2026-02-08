"""
Experiment Agent - Autonomously runs compression experiments.

The Experiment Agent takes findings from the Research Agent and
hypotheses from the Optimizer Agent, then:
- Designs controlled experiments
- Runs compression with different methods
- Measures quality, ratio, and speed
- Reports results back to the swarm

It maintains an experiment log and avoids repeating failed experiments.
"""

import time
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base import Agent, AgentAction


@dataclass
class ExperimentResult:
    """Result of a compression experiment."""
    experiment_id: str
    layer_name: str
    method: str
    params: Dict[str, float]
    original_norm: float
    compressed_norm: float
    distortion: float
    relative_error: float
    sparsity: float
    unique_values: int
    estimated_ratio: float
    duration: float
    success: bool = True


class ExperimentAgent(Agent):
    """
    Autonomous experiment agent that tests compression strategies.

    Takes findings from researchers, designs experiments, executes them,
    and reports back what works and what doesn't.
    """

    AGENT_TYPE = "experimenter"

    # Methods this agent can test
    AVAILABLE_METHODS = {
        "ternary": {"threshold": [0.3, 0.5, 0.7, 1.0]},
        "sparse": {"keep_ratio": [0.1, 0.2, 0.3, 0.5]},
        "wavelet": {"keep_ratio": [0.2, 0.3, 0.5]},
        "lowrank": {"keep_ratio": [0.2, 0.3, 0.5]},
        "binary": {"threshold": [0.5]},
    }

    def __init__(self, name: str = "", verbose: bool = False):
        super().__init__(name=name or "experimenter", verbose=verbose)
        self.memory.add_goal("Find optimal compression per layer", priority=9)
        self.results: List[ExperimentResult] = []
        self._experiment_counter = 0

    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for pending experiment requests."""
        observations = {
            "has_model": context.get("model") is not None,
            "pending_requests": [],
            "past_results": len(self.results),
        }

        # Check messages for experiment requests
        for msg in self.memory.messages_received[-20:]:
            if msg.msg_type in ("task", "knowledge"):
                for key, val in msg.content.items():
                    if isinstance(val, dict) and val.get("action", "").startswith("compress_"):
                        observations["pending_requests"].append({
                            "layer": val.get("layer", ""),
                            "method": val["action"].replace("compress_", ""),
                            "source": msg.sender,
                            "data": val.get("data", {}),
                        })

        # If no specific requests, plan discovery experiments
        if not observations["pending_requests"] and observations["has_model"]:
            observations["discovery_mode"] = True

        return observations

    def plan(self, observations: Dict[str, Any],
             context: Dict[str, Any]) -> List[str]:
        """Plan experiments."""
        if not observations.get("has_model"):
            return []

        plan = []

        # Targeted experiments from requests
        for req in observations.get("pending_requests", []):
            plan.append(f"targeted:{req['layer']}:{req['method']}")

        # Discovery experiments
        if observations.get("discovery_mode") and len(self.results) < 50:
            plan.append("discovery")

        return plan[:10]  # Limit to 10 experiments per step

    def act(self, plan: List[str], context: Dict[str, Any]) -> List[AgentAction]:
        """Execute experiments."""
        model = context.get("model")
        if model is None:
            return []

        actions = []

        for task in plan:
            start = time.time()

            if task.startswith("targeted:"):
                parts = task.split(":")
                layer_name = parts[1]
                method = parts[2]
                results = self._run_targeted(model, layer_name, method)
            elif task == "discovery":
                results = self._run_discovery(model)
            else:
                results = []

            elapsed = time.time() - start
            self.results.extend(results)

            action = AgentAction(
                action_type="experiment",
                parameters={"task": task, "num_results": len(results)},
                result=results,
                success=len(results) > 0,
                duration=elapsed,
            )
            actions.append(action)

            # Report results
            for result in results:
                self.send_message(
                    receiver="",
                    msg_type="result",
                    content={
                        "experiment": result.experiment_id,
                        "layer": result.layer_name,
                        "method": result.method,
                        "distortion": result.distortion,
                        "ratio": result.estimated_ratio,
                        "success": result.success,
                    },
                    priority=5,
                )

            self.log(f"{task}: {len(results)} experiments completed")

        return actions

    def reflect(self, actions: List[AgentAction], context: Dict[str, Any]):
        """Learn from experiment results."""
        super().reflect(actions, context)

        # Update beliefs about best methods
        method_scores = {}
        for result in self.results:
            if result.success:
                key = (result.layer_name, result.method)
                score = result.estimated_ratio / max(result.distortion + 0.01, 0.01)
                if key not in method_scores or score > method_scores[key]:
                    method_scores[key] = score

        for (layer, method), score in method_scores.items():
            self.memory.believe(
                f"best_score:{layer}:{method}",
                score,
                confidence=0.8,
                source="experiment",
            )

    def _run_targeted(self, model: nn.Module, layer_name: str,
                      method: str) -> List[ExperimentResult]:
        """Run experiments for a specific layer and method."""
        results = []
        param_dict = dict(model.named_parameters())
        param = param_dict.get(layer_name)
        if param is None:
            return results

        param_ranges = self.AVAILABLE_METHODS.get(method, {})
        for param_name, values in param_ranges.items():
            for val in values:
                result = self._run_single(
                    layer_name, param.data, method, {param_name: val}
                )
                if result:
                    results.append(result)

        return results

    def _run_discovery(self, model: nn.Module) -> List[ExperimentResult]:
        """Run discovery experiments across the model."""
        results = []
        params = list(model.named_parameters())

        # Pick a few layers to test
        test_layers = [(n, p) for n, p in params if p.numel() >= 64][:3]

        for name, param in test_layers:
            # Quick test with ternary and sparse
            for method in ["ternary", "sparse"]:
                defaults = {"threshold": 0.7} if method == "ternary" else {"keep_ratio": 0.3}
                result = self._run_single(name, param.data, method, defaults)
                if result:
                    results.append(result)

        return results

    def _run_single(self, layer_name: str, weight: torch.Tensor,
                    method: str, params: Dict[str, float]) -> Optional[ExperimentResult]:
        """Run a single compression experiment."""
        self._experiment_counter += 1
        exp_id = f"exp_{self._experiment_counter:04d}"

        try:
            w = weight.clone().float()
            original_norm = w.norm().item()

            # Apply compression
            compressed = self._compress(w, method, params)

            # Measure results
            diff = (w.flatten() - compressed.flatten())
            distortion = diff.norm().item()
            relative_error = distortion / max(original_norm, 1e-8)
            sparsity = (compressed == 0).float().mean().item()
            unique_vals = compressed.unique().numel()

            if unique_vals <= 3:
                ratio = 16.0
            elif sparsity > 0.5:
                ratio = 1.0 / max(1.0 - sparsity, 0.01)
            else:
                ratio = 32.0 / max(np.log2(max(unique_vals, 2)), 1)

            return ExperimentResult(
                experiment_id=exp_id,
                layer_name=layer_name,
                method=method,
                params=params,
                original_norm=original_norm,
                compressed_norm=compressed.norm().item(),
                distortion=distortion,
                relative_error=relative_error,
                sparsity=sparsity,
                unique_values=unique_vals,
                estimated_ratio=ratio,
                duration=0.0,
            )
        except Exception:
            return None

    def _compress(self, w: torch.Tensor, method: str,
                  params: Dict[str, float]) -> torch.Tensor:
        """Apply compression method to weight tensor."""
        if method == "ternary":
            threshold = params.get("threshold", 0.7)
            std = w.std()
            t = threshold * std
            result = torch.zeros_like(w)
            result[w > t] = std
            result[w < -t] = -std
            return result

        elif method == "sparse":
            keep = params.get("keep_ratio", 0.3)
            flat = w.flatten()
            k = max(1, int(keep * flat.numel()))
            _, indices = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            return (flat * mask).reshape(w.shape)

        elif method == "wavelet":
            keep = params.get("keep_ratio", 0.3)
            flat = w.flatten()
            n = flat.numel()
            padded = flat if n % 2 == 0 else torch.cat([flat, torch.zeros(1)])
            freq = torch.fft.rfft(padded)
            k = max(1, int(keep * freq.numel()))
            _, indices = torch.topk(freq.abs(), k)
            mask = torch.zeros_like(freq)
            mask[indices] = 1.0
            return torch.fft.irfft(freq * mask, n=padded.numel())[:n].reshape(w.shape)

        elif method == "lowrank":
            if w.dim() >= 2:
                keep = params.get("keep_ratio", 0.3)
                W2d = w.reshape(w.shape[0], -1)
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
                r = max(1, int(keep * min(W2d.shape)))
                return (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]).reshape(w.shape)
            return w

        elif method == "binary":
            median = w.median()
            std = w.std()
            return torch.where(w > median, std, -std)

        return w

    def get_best_result(self, layer_name: str = None) -> Optional[ExperimentResult]:
        """Get best experiment result, optionally for a specific layer."""
        relevant = self.results
        if layer_name:
            relevant = [r for r in relevant if r.layer_name == layer_name]
        if not relevant:
            return None

        # Best = highest ratio with low distortion
        return max(
            relevant,
            key=lambda r: r.estimated_ratio / max(r.distortion + 0.01, 0.01),
        )

    def get_method_ranking(self) -> Dict[str, float]:
        """Rank methods by average performance."""
        method_scores = {}
        method_counts = {}
        for r in self.results:
            if r.success:
                score = r.estimated_ratio / max(r.distortion + 0.01, 0.01)
                method_scores[r.method] = method_scores.get(r.method, 0) + score
                method_counts[r.method] = method_counts.get(r.method, 0) + 1

        return {
            m: method_scores[m] / method_counts[m]
            for m in method_scores
        }
