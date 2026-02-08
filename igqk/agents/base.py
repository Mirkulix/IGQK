"""
Base Agent Framework - Foundation for all IGQK autonomous agents.

Each agent has:
- MEMORY: Persistent knowledge about past actions and results
- PERCEPTION: Ability to observe models, weights, compression state
- PLANNING: Goal decomposition and action sequencing
- ACTION: Execute compression operations
- COMMUNICATION: Send/receive messages to/from other agents
- REFLECTION: Learn from successes and failures

This is NOT a wrapper around an LLM. These are deterministic, specialized
agents with domain-specific intelligence for compression tasks.

    ┌─────────────────────────────────────┐
    │              Agent                   │
    │                                      │
    │  ┌──────────┐     ┌──────────┐      │
    │  │ Perceive │ ──> │  Plan    │      │
    │  └──────────┘     └────┬─────┘      │
    │                        │             │
    │                   ┌────▼─────┐       │
    │                   │   Act    │       │
    │                   └────┬─────┘       │
    │                        │             │
    │  ┌──────────┐     ┌───▼──────┐      │
    │  │  Learn   │ <── │ Reflect  │      │
    │  └──────────┘     └──────────┘      │
    │                                      │
    │  ┌──────────────────────────────┐   │
    │  │          Memory              │   │
    │  │  beliefs, experiences, goals │   │
    │  └──────────────────────────────┘   │
    └─────────────────────────────────────┘
"""

import time
import uuid
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    ACTING = "acting"
    REFLECTING = "reflecting"
    WAITING = "waiting"
    ERROR = "error"


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ALERT = "alert"
    KNOWLEDGE = "knowledge"
    TASK = "task"
    RESULT = "result"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    id: str = ""
    sender: str = ""
    receiver: str = ""  # Empty = broadcast
    msg_type: str = "broadcast"
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    priority: int = 0  # Higher = more important
    requires_response: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class AgentAction:
    """An action taken by an agent."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = False
    duration: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class Belief:
    """A belief the agent holds about the world."""
    key: str
    value: Any
    confidence: float = 1.0
    source: str = "observation"
    timestamp: float = 0.0
    expires_at: float = 0.0  # 0 = never expires


@dataclass
class Goal:
    """A goal the agent is trying to achieve."""
    description: str
    priority: int = 5  # 1-10
    status: str = "active"  # active, achieved, failed, abandoned
    progress: float = 0.0  # 0-1
    created_at: float = 0.0
    deadline: float = 0.0  # 0 = no deadline
    sub_goals: List[str] = field(default_factory=list)


class AgentMemory:
    """
    Agent's persistent memory system.

    Stores beliefs, experiences, learned patterns, and goals.
    Supports belief revision when new evidence contradicts old beliefs.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.beliefs: Dict[str, Belief] = {}
        self.experiences: List[AgentAction] = []
        self.goals: List[Goal] = []
        self.messages_received: List[AgentMessage] = []
        self.messages_sent: List[AgentMessage] = []
        self.knowledge: Dict[str, Any] = {}  # Free-form knowledge store

    def believe(self, key: str, value: Any, confidence: float = 1.0,
                source: str = "observation"):
        """Update or create a belief."""
        existing = self.beliefs.get(key)
        if existing and existing.confidence > confidence:
            # Only update if new evidence is more confident
            # or if belief is from a more authoritative source
            if source not in ("observation", "experiment"):
                return
        self.beliefs[key] = Belief(
            key=key, value=value, confidence=confidence,
            source=source, timestamp=time.time(),
        )

    def recall(self, key: str) -> Optional[Any]:
        """Recall a belief by key."""
        belief = self.beliefs.get(key)
        if belief:
            if belief.expires_at > 0 and time.time() > belief.expires_at:
                del self.beliefs[key]
                return None
            return belief.value
        return None

    def recall_belief(self, key: str) -> Optional[Belief]:
        """Recall full belief object."""
        return self.beliefs.get(key)

    def forget(self, key: str):
        """Remove a belief."""
        self.beliefs.pop(key, None)

    def add_experience(self, action: AgentAction):
        """Record an experience."""
        self.experiences.append(action)
        if len(self.experiences) > self.capacity:
            self.experiences = self.experiences[-self.capacity:]

    def add_goal(self, description: str, priority: int = 5) -> Goal:
        """Add a new goal."""
        goal = Goal(description=description, priority=priority,
                    created_at=time.time())
        self.goals.append(goal)
        return goal

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals sorted by priority."""
        return sorted(
            [g for g in self.goals if g.status == "active"],
            key=lambda g: g.priority,
            reverse=True,
        )

    def successful_actions(self, action_type: str = None) -> List[AgentAction]:
        """Get successful past actions."""
        actions = [a for a in self.experiences if a.success]
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        return actions

    def failed_actions(self, action_type: str = None) -> List[AgentAction]:
        """Get failed past actions."""
        actions = [a for a in self.experiences if not a.success]
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        return actions

    def success_rate(self, action_type: str = None) -> float:
        """Get success rate for action type."""
        total = len(self.experiences)
        if action_type:
            relevant = [a for a in self.experiences if a.action_type == action_type]
            total = len(relevant)
            successes = sum(1 for a in relevant if a.success)
        else:
            successes = sum(1 for a in self.experiences if a.success)
        return successes / max(total, 1)

    def store_knowledge(self, key: str, value: Any):
        """Store free-form knowledge."""
        self.knowledge[key] = value

    def get_knowledge(self, key: str, default=None) -> Any:
        """Retrieve stored knowledge."""
        return self.knowledge.get(key, default)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "beliefs": len(self.beliefs),
            "experiences": len(self.experiences),
            "goals": len(self.goals),
            "active_goals": len(self.get_active_goals()),
            "knowledge_items": len(self.knowledge),
            "messages_received": len(self.messages_received),
            "messages_sent": len(self.messages_sent),
        }


class Agent:
    """
    Base autonomous agent for IGQK compression tasks.

    Subclasses implement:
    - perceive(): Observe the current state
    - plan(): Create an action plan
    - act(): Execute actions
    - reflect(): Learn from results
    """

    AGENT_TYPE = "base"

    def __init__(self, name: str = "", verbose: bool = False):
        self.name = name or f"{self.AGENT_TYPE}_{str(uuid.uuid4())[:6]}"
        self.verbose = verbose
        self.memory = AgentMemory()
        self.state = AgentState.IDLE
        self._inbox: List[AgentMessage] = []
        self._outbox: List[AgentMessage] = []
        self._step_count = 0
        self._created_at = time.time()

    def step(self, context: Dict[str, Any] = None) -> List[AgentAction]:
        """
        Execute one agent cycle: perceive -> plan -> act -> reflect.

        Args:
            context: External context (model, weights, etc.)

        Returns:
            List of actions taken this step
        """
        context = context or {}
        self._step_count += 1

        # Process incoming messages
        self._process_inbox()

        # Perceive
        self.state = AgentState.PERCEIVING
        observations = self.perceive(context)

        # Plan
        self.state = AgentState.PLANNING
        plan = self.plan(observations, context)

        # Act
        self.state = AgentState.ACTING
        actions = self.act(plan, context)

        # Reflect
        self.state = AgentState.REFLECTING
        self.reflect(actions, context)

        self.state = AgentState.IDLE
        return actions

    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe the current state of the world.
        Override in subclasses.
        """
        return {}

    def plan(self, observations: Dict[str, Any],
             context: Dict[str, Any]) -> List[str]:
        """
        Create an action plan based on observations and goals.
        Override in subclasses.
        """
        return []

    def act(self, plan: List[str], context: Dict[str, Any]) -> List[AgentAction]:
        """
        Execute the plan.
        Override in subclasses.
        """
        return []

    def reflect(self, actions: List[AgentAction], context: Dict[str, Any]):
        """
        Learn from the results of actions.
        Override in subclasses.
        """
        for action in actions:
            self.memory.add_experience(action)

    def send_message(self, receiver: str, msg_type: str,
                     content: Dict[str, Any], priority: int = 0) -> AgentMessage:
        """Send a message to another agent."""
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            priority=priority,
        )
        self._outbox.append(msg)
        self.memory.messages_sent.append(msg)
        return msg

    def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self._inbox.append(message)
        self.memory.messages_received.append(message)

    def _process_inbox(self):
        """Process received messages."""
        # Sort by priority
        self._inbox.sort(key=lambda m: m.priority, reverse=True)
        for msg in self._inbox:
            self._handle_message(msg)
        self._inbox.clear()

    def _handle_message(self, msg: AgentMessage):
        """Handle a single message. Override for custom handling."""
        if msg.msg_type == "knowledge":
            for key, value in msg.content.items():
                self.memory.believe(key, value, confidence=0.7,
                                    source=f"agent:{msg.sender}")

    def flush_outbox(self) -> List[AgentMessage]:
        """Get and clear all outgoing messages."""
        messages = self._outbox.copy()
        self._outbox.clear()
        return messages

    def log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(f"  [{self.name}] {message}")

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def uptime(self) -> float:
        return time.time() - self._created_at

    def status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "name": self.name,
            "type": self.AGENT_TYPE,
            "state": self.state.value,
            "steps": self._step_count,
            "memory": self.memory.stats,
            "outbox_size": len(self._outbox),
            "inbox_size": len(self._inbox),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        stats = self.memory.stats
        return (
            f"Agent: {self.name} ({self.AGENT_TYPE})\n"
            f"  State: {self.state.value}\n"
            f"  Steps: {self._step_count}\n"
            f"  Beliefs: {stats['beliefs']}\n"
            f"  Experiences: {stats['experiences']}\n"
            f"  Success rate: {self.memory.success_rate():.2%}\n"
            f"  Active goals: {stats['active_goals']}\n"
        )
