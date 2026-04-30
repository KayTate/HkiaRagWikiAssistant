"""Tests for the structured retrieval logger instrumentation.

Each test attaches a StringIO handler to the 'retrieval' logger,
runs the compiled agent graph with stubbed vector-store / LLM
dependencies, and asserts on the JSONL events captured.
"""

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from agent.state import AgentState

# ---------------------------------------------------------------------------
# Stubs (mirrored from tests/test_agent_cycle_detection.py)
# ---------------------------------------------------------------------------

_QUEST_A_CHUNKS: list[dict[str, Any]] = [
    {
        "text": "Quest A details.",
        "metadata": {"source_title": "Quest A", "chunk_index": 0},
    }
]


def _stub_embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [[0.1] * 384 for _ in chunks]


def _stub_semantic_search(
    query_embedding: list[float],
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del query_embedding, top_k, where
    return [
        {
            "text": "Semantic match text.",
            "metadata": {"source_title": "Some Page", "chunk_index": 2},
            "distance": 0.42,
        }
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _collect_after(
    retrieval_logger: logging.Logger, stream: io.StringIO
) -> list[dict[str, Any]]:
    """Parse all JSONL lines currently in the test stream."""
    for h in retrieval_logger.handlers:
        h.flush()
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


@pytest.fixture
def retrieval_stream() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """Redirect the 'retrieval' logger to an in-memory StringIO for the test.

    Replaces whatever handlers (file rotation etc.) are on the logger
    with a single StringIO handler so assertions can parse the JSONL
    output without touching disk. Restores prior state on teardown.
    """
    retrieval_logger = logging.getLogger("retrieval")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    saved_handlers = retrieval_logger.handlers[:]
    saved_level = retrieval_logger.level
    saved_propagate = retrieval_logger.propagate

    retrieval_logger.handlers = [handler]
    retrieval_logger.setLevel(logging.INFO)
    retrieval_logger.propagate = False

    try:
        yield retrieval_logger, stream
    finally:
        retrieval_logger.handlers = saved_handlers
        retrieval_logger.setLevel(saved_level)
        retrieval_logger.propagate = saved_propagate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _configure_common_mocks(mocker: Any) -> None:
    mocker.patch(
        "agent.retrieval.vs_get_page_by_title",
        side_effect=lambda title: _QUEST_A_CHUNKS if title == "Quest A" else [],
    )
    mocker.patch("agent.retrieval.embed_chunks", side_effect=_stub_embed_chunks)
    mocker.patch(
        "agent.retrieval.vs_semantic_search", side_effect=_stub_semantic_search
    )
    mocker.patch("agent.graph.mlflow")


def test_retrieve_logs_query_and_chunks(
    mocker: Any, retrieval_stream: tuple[logging.Logger, io.StringIO]
) -> None:
    """The retrieve node must emit a retrieve event carrying the query and chunks."""
    _configure_common_mocks(mocker)

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        del user_message, json_mode
        if "cut short" in system_prompt.lower() or "partial" in system_prompt.lower():
            return "Partial."
        if "analyzing wiki content" in system_prompt.lower():
            return (
                '{"prerequisites": [], "has_unresolved": false, '
                '"next_entity": null, "is_complete": true, "key_facts": []}'
            )
        return "Answer."

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)

    from agent.graph import compile_graph

    graph = compile_graph()
    raw = graph.invoke(
        AgentState(question="Tell me about Quest A", current_entity="Quest A")
    )
    AgentState(**raw)

    retrieval_logger, stream = retrieval_stream
    events = _collect_after(retrieval_logger, stream)

    retrieve_events = [e for e in events if e["event"] == "retrieve"]
    assert len(retrieve_events) >= 1, "Expected at least one retrieve event"

    first = retrieve_events[0]
    assert first["entity"] == "Quest A"
    assert first["query_text"] == "Quest A"
    assert first["top_k"] == 5
    assert len(first["chunks"]) == 1
    chunk = first["chunks"][0]
    assert chunk["id"] == "Quest A::0"
    assert chunk["source_title"] == "Quest A"
    assert chunk["text"] == "Quest A details."


def test_llm_call_logs_prompt_and_response(
    mocker: Any, retrieval_stream: tuple[logging.Logger, io.StringIO]
) -> None:
    """Every _call_llm invocation must surface prompt + response in a log event."""
    _configure_common_mocks(mocker)

    canned_extract = (
        '{"prerequisites": [], "has_unresolved": false, '
        '"next_entity": null, "is_complete": true, "key_facts": ["fact"]}'
    )
    canned_synth = "Synthesised answer."

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        del user_message, json_mode
        if "analyzing wiki content" in system_prompt.lower():
            return canned_extract
        return canned_synth

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)

    from agent.graph import compile_graph

    graph = compile_graph()
    graph.invoke(AgentState(question="Tell me about Quest A", current_entity="Quest A"))

    retrieval_logger, stream = retrieval_stream
    events = _collect_after(retrieval_logger, stream)

    llm_events = [e for e in events if e["event"] == "llm_call"]
    assert len(llm_events) >= 2, (
        f"Expected extract + synthesize calls, got {len(llm_events)}"
    )

    extract_event = next(e for e in llm_events if e["node"] == "extract_info")
    assert "prerequisites" in extract_event["system_prompt"].lower()
    assert "Question: Tell me about Quest A" in extract_event["user_message"]
    assert extract_event["response"] == canned_extract
    assert extract_event["json_mode"] is True
    assert isinstance(extract_event["latency_ms"], (int, float))

    synth_event = next(e for e in llm_events if e["node"] == "synthesize_answer")
    assert synth_event["response"] == canned_synth
    assert synth_event["json_mode"] is False


def test_trace_id_threads_across_events(
    mocker: Any, retrieval_stream: tuple[logging.Logger, io.StringIO]
) -> None:
    """Every event emitted during one graph run must share a single trace_id."""
    _configure_common_mocks(mocker)

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        del user_message, json_mode
        if "analyzing wiki content" in system_prompt.lower():
            return (
                '{"prerequisites": [], "has_unresolved": false, '
                '"next_entity": null, "is_complete": true, "key_facts": []}'
            )
        return "Answer."

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)

    from agent.graph import compile_graph

    graph = compile_graph()
    graph.invoke(AgentState(question="Tell me about Quest A", current_entity="Quest A"))

    retrieval_logger, stream = retrieval_stream
    events = _collect_after(retrieval_logger, stream)

    assert len(events) >= 4, (
        "Expected at minimum: query_received, retrieve, extract_decision, "
        f"llm_call(s), synthesize. Got {len(events)}."
    )

    trace_ids = {e["trace_id"] for e in events}
    assert len(trace_ids) == 1, f"All events must share one trace_id, saw: {trace_ids}"
    assert next(iter(trace_ids)) != "", "trace_id must be populated"

    event_names = [e["event"] for e in events]
    assert event_names[0] == "query_received"
    assert "retrieve" in event_names
    assert "extract_decision" in event_names
    assert "llm_call" in event_names
    assert "synthesize" in event_names


def test_disabled_flag_suppresses_events(
    mocker: Any, retrieval_stream: tuple[logging.Logger, io.StringIO]
) -> None:
    """retrieval_log_enabled=False must silence every retrieval log call."""
    _configure_common_mocks(mocker)
    mocker.patch("agent.nodes.settings.retrieval_log_enabled", False)

    def mock_call_llm(
        system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        del user_message, json_mode
        if "analyzing wiki content" in system_prompt.lower():
            return (
                '{"prerequisites": [], "has_unresolved": false, '
                '"next_entity": null, "is_complete": true, "key_facts": []}'
            )
        return "Answer."

    mocker.patch("agent.nodes._call_llm", side_effect=mock_call_llm)

    from agent.graph import compile_graph

    graph = compile_graph()
    graph.invoke(AgentState(question="Tell me about Quest A", current_entity="Quest A"))

    retrieval_logger, stream = retrieval_stream
    events = _collect_after(retrieval_logger, stream)
    assert events == [], f"Expected no events when disabled, got {events}"
