"""Tests verifying the HKIA agent correctly handles cycles and iteration limits.

All LLM calls, vector store calls, and embedding calls are mocked so these
tests run fully in-process with no external dependencies.
"""

from typing import Any

from agent.state import AgentState

# ---------------------------------------------------------------------------
# Stub data matching the shapes defined in the handoff document
# ---------------------------------------------------------------------------

_QUEST_A_CHUNKS: list[dict[str, Any]] = [
    {
        "text": "Quest A requires Quest B as a prerequisite.",
        "metadata": {"source_title": "Quest A", "chunk_index": 0},
    }
]

_QUEST_B_CHUNKS: list[dict[str, Any]] = [
    {
        "text": "Quest B requires Quest A as a prerequisite.",
        "metadata": {"source_title": "Quest B", "chunk_index": 0},
    }
]


def _stub_get_page_by_title(page_title: str) -> list[dict[str, Any]]:
    """Return fixture chunks for known titles; empty list for unknowns."""
    pages: dict[str, list[dict[str, Any]]] = {
        "Quest A": _QUEST_A_CHUNKS,
        "Quest B": _QUEST_B_CHUNKS,
    }
    return pages.get(page_title, [])


def _stub_embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Return deterministic fixed embeddings without calling a provider."""
    return [[0.1] * 384 for _ in chunks]


def _stub_semantic_search(
    query_embedding: list[float],
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a fixed result set without calling ChromaDB.

    Parameters are ignored — the stub always returns the same fixture data
    so tests are not sensitive to query content.
    """
    del query_embedding, top_k, where
    return [
        {
            "text": (
                "To unlock Ice and Glow you must first complete Straight to Your Heart."
            ),
            "metadata": {"source_title": "Ice and Glow", "chunk_index": 0},
            "distance": 0.1,
        }
    ]


# ---------------------------------------------------------------------------
# LLM response factories
# ---------------------------------------------------------------------------


def _extract_response_circular(entity: str) -> str:
    """Produce an extract response pointing to the other quest in the cycle."""
    if entity == "Quest A":
        return (
            '{"prerequisites": ["Quest B"], "has_unresolved": true, '
            '"next_entity": "Quest B", "is_complete": false}'
        )
    return (
        '{"prerequisites": ["Quest A"], "has_unresolved": true, '
        '"next_entity": "Quest A", "is_complete": false}'
    )


def _extract_response_always_new(call_count: int) -> str:
    """Produce an extract response that always introduces a new unseen prerequisite."""
    new_entity = f"Quest {call_count + 1}"
    return (
        f'{{"prerequisites": ["{new_entity}"], "has_unresolved": true, '
        f'"next_entity": "{new_entity}", "is_complete": false}}'
    )


def _partial_answer_response() -> str:
    """Return a canned partial answer string for synthesis mocks."""
    return "Here is your partial answer based on the information found so far."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_stops_on_circular_prerequisites(mocker: Any) -> None:
    """Agent must terminate without exceeding max_iterations on a circular chain.

    Quest A → Quest B → Quest A forms a cycle. The visited set in the retrieve
    node detects that Quest A is already resolved, sets needs_more_retrieval=False,
    and causes the agent to finalize with whatever partial chain was built.
    """
    mocker.patch(
        "agent.nodes.vs_get_page_by_title",
        side_effect=_stub_get_page_by_title,
    )
    mocker.patch(
        "agent.nodes.embed_chunks",
        side_effect=_stub_embed_chunks,
    )
    mocker.patch(
        "agent.nodes.vs_semantic_search",
        side_effect=_stub_semantic_search,
    )

    extract_call_count: list[int] = [0]

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        """Return circular extract responses or a partial synthesis answer."""
        del user_message, json_mode
        entity = "Quest A" if extract_call_count[0] % 2 == 0 else "Quest B"
        extract_call_count[0] += 1
        if "partial" in system_prompt.lower() or "cut short" in system_prompt.lower():
            return _partial_answer_response()
        if "prerequisites" in system_prompt.lower():
            return _extract_response_circular(entity)
        return _partial_answer_response()

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)
    mocker.patch("agent.graph.mlflow")

    from agent.graph import compile_graph

    graph = compile_graph()

    initial_state = AgentState(
        question="What do I need to complete Quest A?",
        current_entity="Quest A",
    )

    # LangGraph invoke() returns a dict when state is a dataclass
    raw = graph.invoke(initial_state)
    result = AgentState(**raw)

    assert result.iteration_count <= 10, (
        f"Expected iteration_count <= 10 (max_iterations), got {result.iteration_count}"
    )
    assert result.final_answer != "", (
        "Agent must produce a final_answer even when a circular chain is detected"
    )
    assert "Quest A" in result.visited, "Quest A must be in visited after resolution"
    assert "Quest B" in result.visited, "Quest B must be in visited after resolution"


def test_agent_respects_max_iterations(mocker: Any) -> None:
    """Agent must stop at exactly agent_max_iterations and invoke handle_limit.

    A mock that always returns a new unseen prerequisite would loop forever
    without the iteration guard. This test verifies the guard fires and
    handle_iteration_limit populates final_answer.
    """
    mocker.patch(
        "agent.nodes.vs_get_page_by_title",
        return_value=[
            {
                "text": "This quest requires another quest.",
                "metadata": {"source_title": "Quest 0", "chunk_index": 0},
            }
        ],
    )
    mocker.patch(
        "agent.nodes.embed_chunks",
        side_effect=_stub_embed_chunks,
    )
    mocker.patch(
        "agent.nodes.vs_semantic_search",
        side_effect=_stub_semantic_search,
    )

    call_count: list[int] = [0]

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        """Return always-new prerequisite responses until synthesis is requested."""
        del user_message, json_mode
        call_count[0] += 1
        if "partial" in system_prompt.lower() or "cut short" in system_prompt.lower():
            return "Partial answer: the agent reached its iteration limit."
        if "prerequisites" in system_prompt.lower():
            return _extract_response_always_new(call_count[0])
        return "Synthesized answer."

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)
    mocker.patch("agent.graph.mlflow")

    from agent.graph import compile_graph
    from config.settings import settings

    graph = compile_graph()

    initial_state = AgentState(
        question="What do I need to complete Quest 0?",
        current_entity="Quest 0",
    )

    # LangGraph invoke() returns a dict when state is a dataclass
    raw = graph.invoke(initial_state)
    result = AgentState(**raw)

    assert result.iteration_count == settings.agent_max_iterations, (
        f"Expected iteration_count == {settings.agent_max_iterations}, "
        f"got {result.iteration_count}"
    )
    assert result.final_answer != "", (
        "handle_iteration_limit must populate final_answer"
    )


def test_agent_continues_on_persistent_parse_failure(mocker: Any, caplog: Any) -> None:
    """Agent must keep retrieving when extract JSON fails to parse.

    On persistent JSON decode failure, extract_info falls back to
    needs_more_retrieval=True instead of silently marking the question
    complete. The loop continues until agent_max_iterations, at which
    point handle_iteration_limit produces a partial answer.
    """
    import logging

    mocker.patch(
        "agent.nodes.vs_get_page_by_title",
        return_value=[
            {
                "text": "Quest 0 details here.",
                "metadata": {"source_title": "Quest 0", "chunk_index": 0},
            }
        ],
    )
    mocker.patch(
        "agent.nodes.embed_chunks",
        side_effect=_stub_embed_chunks,
    )
    mocker.patch(
        "agent.nodes.vs_semantic_search",
        side_effect=_stub_semantic_search,
    )

    extract_call_count: list[int] = [0]

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        """Return unparseable text for extract prompts; canned answer for partial."""
        del user_message, json_mode
        if "partial" in system_prompt.lower() or "cut short" in system_prompt.lower():
            return _partial_answer_response()
        if "prerequisites" in system_prompt.lower():
            extract_call_count[0] += 1
            return "Here is the answer: I think Quest A requires Quest B."
        return _partial_answer_response()

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)
    mocker.patch("agent.graph.mlflow")

    from agent.graph import compile_graph
    from config.settings import settings

    graph = compile_graph()

    initial_state = AgentState(
        question="What do I need to complete Quest 0?",
        current_entity="Quest 0",
    )

    with caplog.at_level(logging.ERROR, logger="agent.nodes"):
        raw = graph.invoke(initial_state)
    result = AgentState(**raw)

    assert result.iteration_count == settings.agent_max_iterations, (
        f"Expected iteration_count == {settings.agent_max_iterations}, "
        f"got {result.iteration_count}"
    )
    assert result.final_answer != "", (
        "Partial answer must be produced when retries are exhausted"
    )
    # 10 iterations × 3 attempts per iteration = 30 max extract calls
    assert extract_call_count[0] <= settings.agent_max_iterations * 3, (
        f"Extract should be capped at max_iterations*3, got {extract_call_count[0]}"
    )
    assert any(record.levelno == logging.ERROR for record in caplog.records), (
        "At least one error-level log must be emitted when parse retries exhaust"
    )
