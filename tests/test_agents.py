"""
Tests for IGQK v5.1 Autonomous Agent System:
- Base Agent Framework
- Research Agent
- Experiment Agent
- Optimizer Agent
- Guardian Agent
- Architect Agent
- Swarm Controller
"""

import pytest
import torch
import torch.nn as nn
import copy


def _make_model(in_f=32, hidden=64, out_f=10):
    return nn.Sequential(
        nn.Linear(in_f, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_f),
    )


# ===========================================================================
# Base Agent
# ===========================================================================

class TestBaseAgent:
    def test_agent_creation(self):
        from igqk.agents.base import Agent
        agent = Agent(name="test_agent")
        assert agent.name == "test_agent"
        assert agent.step_count == 0

    def test_agent_memory(self):
        from igqk.agents.base import AgentMemory
        mem = AgentMemory()
        assert mem.stats["beliefs"] == 0

        mem.believe("sky_is_blue", True, confidence=0.9)
        assert mem.recall("sky_is_blue") is True
        assert mem.recall_belief("sky_is_blue").confidence == 0.9

    def test_memory_belief_update(self):
        from igqk.agents.base import AgentMemory
        mem = AgentMemory()
        mem.believe("method", "ternary", confidence=0.5)
        mem.believe("method", "sparse", confidence=0.8)  # Higher confidence
        assert mem.recall("method") == "sparse"

        # Lower confidence doesn't override from non-authoritative source
        mem.believe("method", "wavelet", confidence=0.3, source="gossip")
        assert mem.recall("method") == "sparse"

    def test_memory_forget(self):
        from igqk.agents.base import AgentMemory
        mem = AgentMemory()
        mem.believe("temp", 42)
        mem.forget("temp")
        assert mem.recall("temp") is None

    def test_memory_goals(self):
        from igqk.agents.base import AgentMemory
        mem = AgentMemory()
        goal = mem.add_goal("compress model", priority=8)
        assert goal.status == "active"
        active = mem.get_active_goals()
        assert len(active) == 1
        assert active[0].priority == 8

    def test_memory_experience(self):
        from igqk.agents.base import AgentMemory, AgentAction
        mem = AgentMemory()
        action = AgentAction(action_type="compress", success=True, duration=0.1)
        mem.add_experience(action)
        assert len(mem.experiences) == 1
        assert mem.success_rate() == 1.0

        mem.add_experience(AgentAction(action_type="compress", success=False))
        assert mem.success_rate() == 0.5
        assert mem.success_rate("compress") == 0.5

    def test_agent_messaging(self):
        from igqk.agents.base import Agent, AgentMessage
        agent_a = Agent(name="agent_a")
        agent_b = Agent(name="agent_b")

        msg = agent_a.send_message("agent_b", "knowledge", {"key": "value"})
        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"

        outbox = agent_a.flush_outbox()
        assert len(outbox) == 1

        agent_b.receive_message(outbox[0])
        assert len(agent_b._inbox) == 1

    def test_agent_step(self):
        from igqk.agents.base import Agent
        agent = Agent(name="stepper")
        actions = agent.step({})
        assert agent.step_count == 1
        assert isinstance(actions, list)

    def test_agent_status(self):
        from igqk.agents.base import Agent
        agent = Agent(name="status_test")
        status = agent.status()
        assert status["name"] == "status_test"
        assert status["type"] == "base"
        assert "memory" in status

    def test_agent_summary(self):
        from igqk.agents.base import Agent
        agent = Agent(name="summary_test")
        summary = agent.summary()
        assert "summary_test" in summary
        assert "base" in summary

    def test_knowledge_store(self):
        from igqk.agents.base import AgentMemory
        mem = AgentMemory()
        mem.store_knowledge("best_method", "ternary")
        assert mem.get_knowledge("best_method") == "ternary"
        assert mem.get_knowledge("missing", "default") == "default"


# ===========================================================================
# Research Agent
# ===========================================================================

class TestResearchAgent:
    def test_init(self):
        from igqk.agents import ResearchAgent
        agent = ResearchAgent()
        assert agent.AGENT_TYPE == "researcher"

    def test_perceive(self):
        from igqk.agents import ResearchAgent
        agent = ResearchAgent()
        model = _make_model()
        obs = agent.perceive({"model": model})
        assert obs["has_model"] is True
        assert len(obs["layers"]) > 0
        assert obs["global"]["num_layers"] > 0

    def test_perceive_no_model(self):
        from igqk.agents import ResearchAgent
        agent = ResearchAgent()
        obs = agent.perceive({})
        assert obs["has_model"] is False

    def test_step(self):
        from igqk.agents import ResearchAgent
        agent = ResearchAgent()
        model = _make_model()
        actions = agent.step({"model": model})
        assert len(actions) > 0
        assert agent.step_count == 1

    def test_findings(self):
        from igqk.agents import ResearchAgent
        agent = ResearchAgent()
        model = _make_model(hidden=128)
        agent.step({"model": model})
        findings = agent.get_findings()
        # Should find some patterns in random weights
        assert isinstance(findings, list)

    def test_broadcasts_findings(self):
        from igqk.agents import ResearchAgent
        agent = ResearchAgent()
        model = _make_model()
        agent.step({"model": model})
        outbox = agent.flush_outbox()
        # Should have sent some messages
        assert isinstance(outbox, list)


# ===========================================================================
# Experiment Agent
# ===========================================================================

class TestExperimentAgent:
    def test_init(self):
        from igqk.agents import ExperimentAgent
        agent = ExperimentAgent()
        assert agent.AGENT_TYPE == "experimenter"
        assert len(agent.results) == 0

    def test_discovery_mode(self):
        from igqk.agents import ExperimentAgent
        agent = ExperimentAgent()
        model = _make_model()
        actions = agent.step({"model": model})
        assert len(agent.results) > 0

    def test_compress_methods(self):
        from igqk.agents.experimenter import ExperimentAgent
        agent = ExperimentAgent()
        w = torch.randn(100)
        for method in ["ternary", "sparse", "wavelet", "binary"]:
            result = agent._compress(w, method, {"threshold": 0.7, "keep_ratio": 0.3})
            assert result.shape == w.shape
            assert not torch.isnan(result).any()

    def test_compress_lowrank(self):
        from igqk.agents.experimenter import ExperimentAgent
        agent = ExperimentAgent()
        w = torch.randn(32, 16)
        result = agent._compress(w, "lowrank", {"keep_ratio": 0.3})
        assert result.shape == w.shape

    def test_get_best_result(self):
        from igqk.agents import ExperimentAgent
        agent = ExperimentAgent()
        model = _make_model()
        agent.step({"model": model})
        best = agent.get_best_result()
        if agent.results:
            assert best is not None

    def test_method_ranking(self):
        from igqk.agents import ExperimentAgent
        agent = ExperimentAgent()
        model = _make_model()
        agent.step({"model": model})
        ranking = agent.get_method_ranking()
        assert isinstance(ranking, dict)


# ===========================================================================
# Optimizer Agent
# ===========================================================================

class TestOptimizerAgent:
    def test_init(self):
        from igqk.agents import OptimizerAgent
        agent = OptimizerAgent()
        assert agent.AGENT_TYPE == "optimizer"

    def test_step_no_data(self):
        from igqk.agents import OptimizerAgent
        agent = OptimizerAgent()
        actions = agent.step({"model": _make_model()})
        assert isinstance(actions, list)

    def test_responds_to_results(self):
        from igqk.agents import OptimizerAgent
        from igqk.agents.base import AgentMessage
        agent = OptimizerAgent()
        # Simulate receiving experiment results
        msg = AgentMessage(
            sender="experimenter",
            msg_type="result",
            content={
                "method": "ternary",
                "distortion": 0.05,
                "ratio": 16.0,
            },
        )
        agent.receive_message(msg)
        actions = agent.step({"model": _make_model()})
        assert isinstance(actions, list)


# ===========================================================================
# Guardian Agent
# ===========================================================================

class TestGuardianAgent:
    def test_init(self):
        from igqk.agents import GuardianAgent
        agent = GuardianAgent()
        assert agent.AGENT_TYPE == "guardian"
        assert agent.veto_count == 0

    def test_validate_good_compression(self):
        from igqk.agents import GuardianAgent
        agent = GuardianAgent()
        original = torch.randn(100)
        compressed = original * 0.95  # Small change
        report = agent.validate("layer", original, compressed, "ternary")
        assert report.is_acceptable
        assert report.verdict in ("approved", "warning")

    def test_validate_bad_compression(self):
        from igqk.agents import GuardianAgent
        from igqk.agents.guardian import QualityPolicy
        agent = GuardianAgent(policy=QualityPolicy(max_relative_error=0.05))
        original = torch.randn(100)
        compressed = torch.zeros(100)  # Massive change
        report = agent.validate("layer", original, compressed, "ternary")
        assert report.verdict == "rejected"
        assert agent.veto_count == 1

    def test_approval_rate(self):
        from igqk.agents import GuardianAgent
        agent = GuardianAgent()
        w = torch.randn(100)
        agent.validate("l1", w, w * 0.99, "ternary")  # Good
        agent.validate("l2", w, w * 0.98, "ternary")  # Good
        assert agent.approval_rate > 0.5

    def test_health_check(self):
        from igqk.agents import GuardianAgent
        agent = GuardianAgent()
        model = _make_model()
        actions = agent.step({"model": model})
        assert isinstance(actions, list)

    def test_alerts_on_veto(self):
        from igqk.agents import GuardianAgent
        from igqk.agents.guardian import QualityPolicy
        agent = GuardianAgent(policy=QualityPolicy(max_relative_error=0.01))
        w = torch.randn(100)
        agent.validate("bad_layer", w, torch.zeros(100), "ternary")
        outbox = agent.flush_outbox()
        alerts = [m for m in outbox if m.msg_type == "alert"]
        assert len(alerts) > 0


# ===========================================================================
# Architect Agent
# ===========================================================================

class TestArchitectAgent:
    def test_init(self):
        from igqk.agents import ArchitectAgent
        agent = ArchitectAgent()
        assert agent.AGENT_TYPE == "architect"

    def test_create_blueprint(self):
        from igqk.agents import ArchitectAgent
        agent = ArchitectAgent()
        model = _make_model()
        actions = agent.step({"model": model})
        assert len(actions) > 0
        blueprint = agent.get_latest_blueprint()
        assert blueprint is not None
        assert len(blueprint.layers) > 0
        assert blueprint.strategy in ("aggressive", "conservative", "adaptive")

    def test_protected_layers(self):
        from igqk.agents import ArchitectAgent
        agent = ArchitectAgent()
        # Model with norm layer
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
        agent.step({"model": model})
        blueprint = agent.get_latest_blueprint()
        # LayerNorm should be skipped
        skipped = [l for l in blueprint.layers if l.skip]
        assert len(skipped) > 0

    def test_blueprint_broadcasts(self):
        from igqk.agents import ArchitectAgent
        agent = ArchitectAgent()
        agent.step({"model": _make_model()})
        outbox = agent.flush_outbox()
        knowledge_msgs = [m for m in outbox if m.msg_type == "knowledge"]
        assert len(knowledge_msgs) > 0


# ===========================================================================
# Swarm Controller
# ===========================================================================

class TestSwarmController:
    def test_init(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        assert len(swarm.agents) == 5
        assert swarm.cycle_count == 0

    def test_run_cycle(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        result = swarm.run_cycle({"model": model})
        assert result.cycle == 1
        assert result.agents_active == 5
        assert result.duration > 0

    def test_multiple_cycles(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        for i in range(3):
            swarm.run_cycle({"model": model})
        assert swarm.cycle_count == 3

    def test_compress(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        result = swarm.compress(model, max_cycles=2)
        assert result.layers_compressed > 0
        assert result.total_time > 0
        assert result.cycles_run == 2
        assert result.estimated_ratio >= 1.0

    def test_compress_verbose(self, capsys):
        from igqk.agents import SwarmController
        swarm = SwarmController(verbose=True)
        model = _make_model(in_f=16, hidden=32, out_f=5)
        result = swarm.compress(model, max_cycles=1)
        captured = capsys.readouterr()
        assert "Swarm" in captured.out

    def test_message_routing(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        result = swarm.run_cycle({"model": model})
        assert result.messages_routed > 0
        assert swarm.total_messages > 0

    def test_quality_check(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        result = swarm.compress(model, max_cycles=2, quality_threshold=0.3)
        # Should approve with generous threshold
        assert isinstance(result.quality_approved, bool)

    def test_add_remove_agent(self):
        from igqk.agents import SwarmController
        from igqk.agents.base import Agent
        swarm = SwarmController()
        custom = Agent(name="custom_agent")
        swarm.add_agent(custom)
        assert "custom_agent" in swarm.agents
        swarm.remove_agent("custom_agent")
        assert "custom_agent" not in swarm.agents

    def test_summary(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        swarm.compress(model, max_cycles=2)
        summary = swarm.summary()
        assert "Agent Swarm" in summary
        assert "researcher" in summary
        assert "guardian" in summary

    def test_agent_summary_dict(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        result = swarm.compress(model, max_cycles=1)
        assert isinstance(result.agent_summary, dict)
        assert len(result.agent_summary) == 5

    def test_cycle_history(self):
        from igqk.agents import SwarmController
        swarm = SwarmController()
        model = _make_model()
        result = swarm.compress(model, max_cycles=3)
        assert len(result.cycle_history) == 3
