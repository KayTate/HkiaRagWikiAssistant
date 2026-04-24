"""Agent state dataclass for the HKIA LangGraph agent."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Shared state passed between all nodes in the agent graph.

    Accumulates retrieved context across iterations and tracks the
    prerequisite chain being built. The visited set enforces cycle
    prevention — any entity already resolved is not re-fetched.
    """

    question: str
    messages: list[Any] = field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    resolved_entities: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prerequisite_chain: list[str] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)
    iteration_count: int = 0
    final_answer: str = ""
    needs_more_retrieval: bool = True
    # Current entity being resolved; None signals semantic-search mode.
    current_entity: str | None = None
    # Correlation id for retrieval logs; populated in route_question.
    trace_id: str = ""
