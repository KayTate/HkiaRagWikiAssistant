"""LangGraph graph definition for the HKIA RAG agent.

Builds and compiles the state graph that drives multi-step prerequisite
chain resolution. MLflow autolog is enabled lazily inside compile_graph
(not at module import) so every agent invocation produces a trace while
still allowing tests to patch mlflow before compilation.
"""

from typing import Any

import mlflow
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    check_complete,
    extract_info,
    handle_iteration_limit,
    retrieve,
    route_question,
    synthesize_answer,
)
from agent.state import AgentState
from config.settings import settings


def _enable_mlflow_autolog() -> None:
    """Enable MLflow LangChain autolog for agent tracing.

    Called lazily at compile time rather than import time so that tests
    can patch mlflow before this runs and avoid side effects during import.
    """
    mlflow.langchain.autolog(
        log_traces=True,
    )


def _route_after_check(state: AgentState) -> str:
    """Determine the next node after check_complete.

    Evaluated as a LangGraph conditional edge function. Priority order:
    1. Iteration limit reached → handle_limit
    2. No more retrieval needed → synthesize
    3. More retrieval needed → retrieve

    Args:
        state: Current agent state after check_complete ran.

    Returns:
        Name of the next node to execute.
    """
    if state.iteration_count >= settings.agent_max_iterations:
        return "handle_limit"
    if not state.needs_more_retrieval:
        return "synthesize"
    return "retrieve"


def build_graph() -> StateGraph:  # type: ignore[type-arg]
    """Construct the HKIA agent state graph without compiling it.

    Separating construction from compilation allows tests to inspect
    graph structure without invoking the full compile step.

    Returns:
        Uncompiled StateGraph instance.
    """
    graph: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    graph.add_node("router", route_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("extract", extract_info)
    graph.add_node("check_complete", check_complete)
    graph.add_node("synthesize", synthesize_answer)
    graph.add_node("handle_limit", handle_iteration_limit)

    graph.add_edge(START, "router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "check_complete")
    graph.add_conditional_edges(
        "check_complete",
        _route_after_check,
        {
            "synthesize": "synthesize",
            "handle_limit": "handle_limit",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("synthesize", END)
    graph.add_edge("handle_limit", END)

    return graph


def compile_graph() -> Any:  # LangGraph compiled graph type is internal/untyped
    """Build and compile the HKIA agent graph, ready for invocation.

    Enables MLflow autologging before compiling so traces are captured
    for every subsequent invocation.

    Returns:
        Compiled LangGraph runnable. Typed as Any because the compiled
        graph type is an internal LangGraph implementation detail that
        varies across versions and is not exposed in the public API.
    """
    _enable_mlflow_autolog()
    return build_graph().compile()
