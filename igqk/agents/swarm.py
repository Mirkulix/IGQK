"""
Swarm Controller - Orchestrates all IGQK agents into a collective intelligence.

The Swarm Controller:
1. Manages agent lifecycle (create, run, pause, stop)
2. Routes messages between agents
3. Runs multi-step autonomous compression cycles
4. Tracks overall swarm performance
5. Makes meta-decisions about which agents to activate

The key insight: no single agent knows everything, but together
they form a compression intelligence greater than the sum of parts.

    ┌─────────────────────────────────────────────────────┐
    │               Swarm Controller                       │
    │                                                      │
    │  ┌────────────────────────────────────────────┐     │
    │  │           Message Bus                      │     │
    │  └──────┬──────┬──────┬──────┬──────┬────────┘     │
    │         │      │      │      │      │               │
    │    ┌────▼─┐┌───▼──┐┌──▼──┐┌──▼──┐┌──▼───┐          │
    │    │Resear│|Experi│|Optim│|Guard│|Archi │          │
    │    │cher  │|menter│|izer │|ian  │|tect  │          │
    │    └──────┘└──────┘└─────┘└─────┘└──────┘          │
    │                                                      │
    │  ┌────────────────────────────────────────────┐     │
    │  │        Shared Knowledge Base               │     │
    │  └────────────────────────────────────────────┘     │
    └─────────────────────────────────────────────────────┘
"""

import time
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .base import Agent, AgentMessage, AgentAction
from .researcher import ResearchAgent
from .experimenter import ExperimentAgent
from .optimizer import OptimizerAgent
from .guardian import GuardianAgent
from .architect import ArchitectAgent


@dataclass
class SwarmCycleResult:
    """Result of one complete swarm cycle."""
    cycle: int
    duration: float
    agents_active: int
    messages_routed: int
    total_actions: int
    findings: int
    experiments: int
    vetoes: int
    approvals: int


@dataclass
class SwarmCompressionResult:
    """Result of swarm-driven autonomous compression."""
    model: nn.Module
    cycles_run: int
    total_time: float
    blueprint_strategy: str
    layers_compressed: int
    layers_skipped: int
    estimated_ratio: float
    quality_approved: bool
    agent_summary: Dict[str, Any]
    cycle_history: List[SwarmCycleResult]


class SwarmController:
    """
    Multi-agent swarm controller for autonomous compression.

    Orchestrates specialized agents into a collective compression
    intelligence that continuously improves.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Create agent swarm
        self.researcher = ResearchAgent(verbose=verbose)
        self.experimenter = ExperimentAgent(verbose=verbose)
        self.optimizer = OptimizerAgent(verbose=verbose)
        self.guardian = GuardianAgent(verbose=verbose)
        self.architect = ArchitectAgent(verbose=verbose)

        self._agents: Dict[str, Agent] = {
            self.researcher.name: self.researcher,
            self.experimenter.name: self.experimenter,
            self.optimizer.name: self.optimizer,
            self.guardian.name: self.guardian,
            self.architect.name: self.architect,
        }

        self._cycle_count = 0
        self._total_messages = 0
        self._cycle_history: List[SwarmCycleResult] = []

    def add_agent(self, agent: Agent):
        """Add a custom agent to the swarm."""
        self._agents[agent.name] = agent

    def remove_agent(self, name: str):
        """Remove an agent from the swarm."""
        self._agents.pop(name, None)

    def run_cycle(self, context: Dict[str, Any] = None) -> SwarmCycleResult:
        """
        Run one complete swarm cycle.

        Each cycle:
        1. All agents perceive/plan/act/reflect
        2. Messages are routed between agents
        3. Results are aggregated
        """
        context = context or {}
        self._cycle_count += 1
        start = time.time()

        if self.verbose:
            print(f"\n--- Swarm Cycle {self._cycle_count} ---")

        total_actions = 0
        messages_routed = 0

        # Run each agent
        agent_order = [
            self.researcher,    # Discover first
            self.architect,     # Then design
            self.experimenter,  # Then test
            self.optimizer,     # Then optimize
            self.guardian,      # Then validate
        ]

        for agent in agent_order:
            if agent.name not in self._agents:
                continue

            actions = agent.step(context)
            total_actions += len(actions)

            # Route outgoing messages
            outbox = agent.flush_outbox()
            for msg in outbox:
                messages_routed += self._route_message(msg)

        elapsed = time.time() - start

        # Collect stats
        findings = sum(
            1 for a in self.researcher.memory.experiences[-10:]
            if a.result and isinstance(a.result, list)
            for _ in a.result
        )

        result = SwarmCycleResult(
            cycle=self._cycle_count,
            duration=elapsed,
            agents_active=len(self._agents),
            messages_routed=messages_routed,
            total_actions=total_actions,
            findings=findings,
            experiments=len(self.experimenter.results),
            vetoes=self.guardian.veto_count,
            approvals=self.guardian._approvals,
        )
        self._cycle_history.append(result)
        self._total_messages += messages_routed

        if self.verbose:
            print(f"  Cycle {self._cycle_count}: "
                  f"{total_actions} actions, "
                  f"{messages_routed} messages, "
                  f"{elapsed:.2f}s")

        return result

    def _route_message(self, msg: AgentMessage) -> int:
        """Route a message to its recipient(s)."""
        count = 0
        if msg.receiver and msg.receiver in self._agents:
            # Direct message
            self._agents[msg.receiver].receive_message(msg)
            count = 1
        elif msg.receiver:
            # Try partial match
            for name, agent in self._agents.items():
                if msg.receiver in name or msg.receiver == agent.AGENT_TYPE:
                    agent.receive_message(msg)
                    count += 1
        else:
            # Broadcast to all except sender
            for name, agent in self._agents.items():
                if name != msg.sender:
                    agent.receive_message(msg)
                    count += 1
        return count

    def compress(
        self,
        model: nn.Module,
        max_cycles: int = 3,
        quality_threshold: float = 0.15,
    ) -> SwarmCompressionResult:
        """
        Autonomous swarm-driven compression.

        Multiple agents collaborate to find the best compression strategy:
        1. Researcher analyzes the model
        2. Architect creates a blueprint
        3. Experimenter tests strategies
        4. Optimizer refines parameters
        5. Guardian validates quality

        Args:
            model: Model to compress
            max_cycles: Maximum agent cycles to run
            quality_threshold: Maximum acceptable relative error

        Returns:
            SwarmCompressionResult with compressed model and metadata
        """
        start_time = time.time()
        model_copy = copy.deepcopy(model)

        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Swarm Compression starting ({total_params:,} params)")

        # Set guardian policy
        self.guardian.policy.max_relative_error = quality_threshold

        context = {"model": model_copy, "model_name": "target"}

        # Run agent cycles
        for i in range(max_cycles):
            self.run_cycle(context)

        # Get architect's blueprint
        blueprint = self.architect.get_latest_blueprint()
        strategy = blueprint.strategy if blueprint else "heuristic"

        # Apply compression based on collective intelligence
        layers_compressed = 0
        layers_skipped = 0

        if blueprint:
            layers_compressed, layers_skipped = self._apply_blueprint(
                model_copy, blueprint
            )
        else:
            # Fallback: use experiment results
            layers_compressed = self._apply_from_experiments(model_copy)

        # Final quality check
        quality_approved = True
        if blueprint:
            for bp_layer in blueprint.layers:
                if bp_layer.skip:
                    continue
                param_dict = dict(model_copy.named_parameters())
                orig_dict = dict(model.named_parameters())
                if bp_layer.layer_name in param_dict and bp_layer.layer_name in orig_dict:
                    report = self.guardian.validate(
                        bp_layer.layer_name,
                        orig_dict[bp_layer.layer_name].data,
                        param_dict[bp_layer.layer_name].data,
                        bp_layer.method,
                    )
                    if not report.is_acceptable:
                        quality_approved = False

        # Estimate compression
        total_zeros = 0
        total_params = 0
        unique_values = set()
        for _, p in model_copy.named_parameters():
            flat = p.data.flatten()
            total_zeros += (flat == 0).sum().item()
            total_params += flat.numel()
            unique_values.update(flat.unique().tolist()[:50])

        if len(unique_values) <= 4:
            est_ratio = 16.0
        elif total_params > 0 and total_zeros / total_params > 0.5:
            est_ratio = 1.0 / max(1 - total_zeros / total_params, 0.01)
        else:
            est_ratio = 32.0 / max(np.log2(max(len(unique_values), 2)), 1)

        elapsed = time.time() - start_time

        return SwarmCompressionResult(
            model=model_copy,
            cycles_run=max_cycles,
            total_time=elapsed,
            blueprint_strategy=strategy,
            layers_compressed=layers_compressed,
            layers_skipped=layers_skipped,
            estimated_ratio=est_ratio,
            quality_approved=quality_approved,
            agent_summary=self._agent_summary(),
            cycle_history=self._cycle_history[-max_cycles:],
        )

    def _apply_blueprint(
        self, model: nn.Module, blueprint
    ) -> Tuple[int, int]:
        """Apply architect's blueprint to model."""
        compressed = 0
        skipped = 0
        param_dict = dict(model.named_parameters())

        with torch.no_grad():
            for bp_layer in sorted(blueprint.layers, key=lambda l: l.priority):
                if bp_layer.skip or bp_layer.method == "none":
                    skipped += 1
                    continue

                if bp_layer.layer_name not in param_dict:
                    continue

                param = param_dict[bp_layer.layer_name]
                self._compress_layer(param, bp_layer.method, bp_layer.params)
                compressed += 1

        return compressed, skipped

    def _apply_from_experiments(self, model: nn.Module) -> int:
        """Apply compression based on experiment results (fallback)."""
        compressed = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.numel() < 64:
                    continue

                best = self.experimenter.get_best_result(name)
                if best:
                    method = best.method
                    params = best.params
                else:
                    method = "ternary"
                    params = {"threshold": 0.7}

                self._compress_layer(param, method, params)
                compressed += 1

        return compressed

    def _compress_layer(self, param: nn.Parameter, method: str,
                        params: Dict[str, float]):
        """Apply compression to a single parameter."""
        if method == "ternary":
            threshold = params.get("threshold", 0.7)
            std = param.data.std()
            t = threshold * std
            result = torch.zeros_like(param.data)
            result[param.data > t] = std
            result[param.data < -t] = -std
            param.data = result

        elif method == "sparse":
            keep = params.get("keep_ratio", 0.3)
            flat = param.data.flatten()
            k = max(1, int(keep * flat.numel()))
            _, indices = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            param.data = (flat * mask).reshape(param.data.shape)

        elif method == "lowrank":
            if param.data.dim() >= 2:
                keep = params.get("keep_ratio", 0.3)
                W = param.data.float()
                shape = W.shape
                W2d = W.reshape(shape[0], -1)
                U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
                r = max(1, int(keep * min(W2d.shape)))
                param.data = (
                    U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
                ).reshape(shape).to(param.data.dtype)

        elif method == "wavelet":
            keep = params.get("keep_ratio", 0.3)
            flat = param.data.flatten().float()
            n = flat.numel()
            padded = flat if n % 2 == 0 else torch.cat([flat, torch.zeros(1)])
            freq = torch.fft.rfft(padded)
            k = max(1, int(keep * freq.numel()))
            _, indices = torch.topk(freq.abs(), k)
            mask = torch.zeros_like(freq)
            mask[indices] = 1.0
            reconstructed = torch.fft.irfft(freq * mask, n=padded.numel())[:n]
            param.data = reconstructed.reshape(param.data.shape).to(param.data.dtype)

        elif method == "binary":
            median = param.data.median()
            std = param.data.std()
            param.data = torch.where(param.data > median, std, -std)

    def _agent_summary(self) -> Dict[str, Any]:
        """Get summary of all agents."""
        return {
            name: agent.status()
            for name, agent in self._agents.items()
        }

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def total_messages(self) -> int:
        return self._total_messages

    @property
    def agents(self) -> Dict[str, Agent]:
        return self._agents

    def summary(self) -> str:
        """Human-readable swarm summary."""
        lines = [
            "IGQK Agent Swarm",
            "=" * 55,
            f"  Agents: {len(self._agents)}",
            f"  Cycles completed: {self._cycle_count}",
            f"  Messages routed: {self._total_messages}",
            "",
            "  Agent Status:",
        ]

        for name, agent in self._agents.items():
            stats = agent.memory.stats
            lines.append(
                f"    [{agent.AGENT_TYPE:12}] {name}: "
                f"steps={agent.step_count}, "
                f"beliefs={stats['beliefs']}, "
                f"exp={stats['experiences']}, "
                f"success={agent.memory.success_rate():.0%}"
            )

        if self._cycle_history:
            lines.append("")
            lines.append("  Recent Cycles:")
            for cr in self._cycle_history[-5:]:
                lines.append(
                    f"    Cycle {cr.cycle}: "
                    f"{cr.total_actions} actions, "
                    f"{cr.messages_routed} msgs, "
                    f"{cr.duration:.2f}s"
                )

        # Guardian stats
        lines.append("")
        lines.append(
            f"  Quality: {self.guardian._approvals} approved, "
            f"{self.guardian.veto_count} vetoed "
            f"({self.guardian.approval_rate:.0%} rate)"
        )

        # Experiment stats
        if self.experimenter.results:
            ranking = self.experimenter.get_method_ranking()
            if ranking:
                lines.append("")
                lines.append("  Method Rankings:")
                for method, score in sorted(
                    ranking.items(), key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"    {method}: {score:.2f}")

        return "\n".join(lines)


import numpy as np
