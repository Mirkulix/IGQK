"""
Architect Agent - Designs optimal compression architectures.

The Architect Agent thinks at a higher level:
- Which layers should be compressed together?
- What is the optimal compression order?
- Should some layers be left uncompressed?
- Can layers be fused or restructured for better compression?
- What is the optimal resource allocation across layers?

It creates compression blueprints that the Experimenter executes.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base import Agent, AgentAction


@dataclass
class LayerBlueprint:
    """Compression blueprint for a single layer."""
    layer_name: str
    method: str
    priority: int  # Compression order (1 = first)
    skip: bool = False  # True = don't compress
    params: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class CompressionBlueprint:
    """Full compression blueprint for a model."""
    model_name: str
    layers: List[LayerBlueprint]
    estimated_ratio: float = 1.0
    estimated_quality: float = 1.0
    strategy: str = ""  # "uniform", "adaptive", "progressive", "critical_path"


class ArchitectAgent(Agent):
    """
    Architecture agent that designs compression strategies at model level.

    Thinks globally about the model to create optimal compression blueprints.
    """

    AGENT_TYPE = "architect"

    # Layers that should never be compressed (heuristics)
    PROTECTED_PATTERNS = ["embedding", "norm", "ln", "layernorm", "batchnorm"]

    def __init__(self, name: str = "", verbose: bool = False):
        super().__init__(name=name or "architect", verbose=verbose)
        self.memory.add_goal("Design optimal compression blueprints", priority=8)
        self.blueprints: List[CompressionBlueprint] = []

    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model architecture."""
        model = context.get("model")
        if model is None:
            return {"has_model": False}

        observations = {
            "has_model": True,
            "architecture": [],
            "total_params": 0,
            "layer_importance": {},
        }

        with torch.no_grad():
            for name, param in model.named_parameters():
                n = param.numel()
                observations["total_params"] += n

                is_protected = any(
                    p in name.lower() for p in self.PROTECTED_PATTERNS
                )

                layer_info = {
                    "name": name,
                    "shape": list(param.shape),
                    "params": n,
                    "fraction": 0.0,  # Will compute after
                    "protected": is_protected,
                    "dim": param.dim(),
                }

                # Importance estimate: larger layers with more variance = more important
                std = param.data.float().std().item()
                layer_info["importance"] = n * std
                observations["architecture"].append(layer_info)

        # Compute fractions
        total = observations["total_params"]
        for layer in observations["architecture"]:
            layer["fraction"] = layer["params"] / max(total, 1)

        # Normalize importance
        max_imp = max((l["importance"] for l in observations["architecture"]), default=1)
        for layer in observations["architecture"]:
            layer["importance"] /= max(max_imp, 1e-8)

        return observations

    def plan(self, observations: Dict[str, Any],
             context: Dict[str, Any]) -> List[str]:
        """Plan architecture analysis."""
        if not observations.get("has_model"):
            return []

        plan = ["create_blueprint"]

        # If we have research findings, use them
        research_findings = []
        for msg in self.memory.messages_received[-20:]:
            if msg.msg_type == "knowledge":
                research_findings.append(msg.content)

        if research_findings:
            plan.insert(0, "integrate_research")

        return plan

    def act(self, plan: List[str], context: Dict[str, Any]) -> List[AgentAction]:
        """Execute architecture design."""
        actions = []

        for task in plan:
            start = time.time()

            if task == "integrate_research":
                result = self._integrate_research()
            elif task == "create_blueprint":
                result = self._create_blueprint(context)
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

            # Share blueprint
            if task == "create_blueprint" and result:
                blueprint = result
                self.send_message(
                    receiver="",
                    msg_type="knowledge",
                    content={
                        "blueprint": {
                            "strategy": blueprint.strategy,
                            "num_layers": len(blueprint.layers),
                            "estimated_ratio": blueprint.estimated_ratio,
                            "estimated_quality": blueprint.estimated_quality,
                            "layers": [
                                {
                                    "name": l.layer_name,
                                    "method": l.method,
                                    "skip": l.skip,
                                    "priority": l.priority,
                                }
                                for l in blueprint.layers
                            ],
                        }
                    },
                    priority=8,
                )

            self.log(f"{task}: {'complete' if action.success else 'no data'}")

        return actions

    def reflect(self, actions: List[AgentAction], context: Dict[str, Any]):
        """Learn from blueprint designs."""
        super().reflect(actions, context)

    def _integrate_research(self) -> Dict[str, Any]:
        """Integrate research findings into architectural knowledge."""
        findings = {}
        for msg in self.memory.messages_received[-20:]:
            if msg.msg_type == "knowledge":
                for key, val in msg.content.items():
                    if isinstance(val, dict):
                        layer = val.get("layer", "")
                        action = val.get("action", "")
                        if layer and action:
                            findings[layer] = action

        for layer, action in findings.items():
            self.memory.believe(
                f"research:{layer}",
                action,
                confidence=0.7,
                source="research_agent",
            )

        return {"integrated": len(findings)}

    def _create_blueprint(self, context: Dict[str, Any]) -> Optional[CompressionBlueprint]:
        """Create a compression blueprint for the model."""
        model = context.get("model")
        if model is None:
            return None

        observations = self.perceive(context)
        layers = observations["architecture"]

        # Strategy selection based on model characteristics
        strategy = self._select_strategy(layers)

        blueprint_layers = []
        priority = 1

        for layer_info in layers:
            name = layer_info["name"]
            is_protected = layer_info["protected"]
            importance = layer_info["importance"]
            fraction = layer_info["fraction"]

            # Check research findings
            research_method = self.memory.recall(f"research:{name}")

            if is_protected:
                blueprint_layers.append(LayerBlueprint(
                    layer_name=name, method="none", priority=0,
                    skip=True, reason="Protected layer (norm/embedding)",
                ))
                continue

            if layer_info["params"] < 16:
                blueprint_layers.append(LayerBlueprint(
                    layer_name=name, method="none", priority=0,
                    skip=True, reason="Too small to compress",
                ))
                continue

            # Determine method
            if research_method and research_method.startswith("compress_"):
                method = research_method.replace("compress_", "")
            elif strategy == "aggressive":
                method = "ternary"
            elif strategy == "conservative":
                method = "sparse" if importance > 0.5 else "ternary"
            elif strategy == "adaptive":
                if layer_info["dim"] >= 2 and fraction > 0.1:
                    method = "lowrank"
                else:
                    method = "ternary"
            else:
                method = "ternary"

            # Set priority: larger layers first (more impact)
            params = {}
            if method == "ternary":
                params["threshold"] = 0.5 if importance > 0.7 else 0.7
            elif method == "sparse":
                params["keep_ratio"] = 0.4 if importance > 0.7 else 0.2
            elif method == "lowrank":
                params["keep_ratio"] = 0.5 if importance > 0.7 else 0.3

            blueprint_layers.append(LayerBlueprint(
                layer_name=name,
                method=method,
                priority=priority,
                params=params,
                reason=f"Strategy: {strategy}, importance: {importance:.2f}",
            ))
            priority += 1

        # Estimate overall metrics
        compressible = [l for l in blueprint_layers if not l.skip]
        if compressible:
            est_ratio = np.mean([
                16.0 if l.method == "ternary"
                else 1.0 / max(l.params.get("keep_ratio", 0.3), 0.01)
                for l in compressible
            ])
            est_quality = np.mean([
                0.85 if l.method == "ternary"
                else 0.9 if l.method == "sparse"
                else 0.92
                for l in compressible
            ])
        else:
            est_ratio = 1.0
            est_quality = 1.0

        blueprint = CompressionBlueprint(
            model_name=context.get("model_name", "model"),
            layers=blueprint_layers,
            estimated_ratio=est_ratio,
            estimated_quality=est_quality,
            strategy=strategy,
        )
        self.blueprints.append(blueprint)
        return blueprint

    def _select_strategy(self, layers: List[Dict]) -> str:
        """Select compression strategy based on model characteristics."""
        total_params = sum(l["params"] for l in layers)
        num_layers = len(layers)

        # Large model with many layers -> adaptive
        if total_params > 100000 and num_layers > 6:
            return "adaptive"
        # Small model -> aggressive is fine
        elif total_params < 10000:
            return "aggressive"
        # Medium model -> conservative
        else:
            return "conservative"

    def get_latest_blueprint(self) -> Optional[CompressionBlueprint]:
        """Get the most recent blueprint."""
        return self.blueprints[-1] if self.blueprints else None
