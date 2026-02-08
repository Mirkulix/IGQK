"""IGQK Autonomous Agent System - Multi-Agent Compression Intelligence."""

from .base import Agent, AgentMemory, AgentMessage, AgentAction
from .researcher import ResearchAgent
from .experimenter import ExperimentAgent
from .optimizer import OptimizerAgent
from .guardian import GuardianAgent
from .architect import ArchitectAgent
from .swarm import SwarmController

__all__ = [
    "Agent",
    "AgentMemory",
    "AgentMessage",
    "AgentAction",
    "ResearchAgent",
    "ExperimentAgent",
    "OptimizerAgent",
    "GuardianAgent",
    "ArchitectAgent",
    "SwarmController",
]
