"""Node functions for the LangGraph agent graph.

Each function accepts and returns an AgentState. Nodes are pure functions
from the graph's perspective — side effects are limited to LLM calls and
vector store reads.
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agent.extraction import (
    _extract_entity_from_question,
    _strip_markdown_fences,
)
from agent.llm import _call_llm
from agent.prompts import (
    _EXTRACT_SYSTEM_PROMPT,
    _PARTIAL_SYSTEM_PROMPT,
    _SYNTHESIZE_SYSTEM_PROMPT,
)
from agent.retrieval import (
    _fetch_entity_chunks,
    _semantic_search_for_question,
)
from agent.state import AgentState
from config.settings import settings

logger = logging.getLogger(__name__)
_retrieval_logger = logging.getLogger("retrieval")


def _log_event(event: str, state: AgentState, **fields: Any) -> None:
    """Emit one JSON-encoded retrieval observability event.

    Gated by settings.retrieval_log_enabled. Each event carries a
    millisecond-precision UTC timestamp, the per-question trace_id, and
    the current iteration count so events from a single run can be
    reassembled with `jq` or grep.

    Args:
        event: Short event name — one of query_received, retrieve,
            llm_call, extract_decision, synthesize.
        state: Current agent state, used for trace_id and iter.
        **fields: Event-specific payload fields.
    """
    if not settings.retrieval_log_enabled:
        return
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "event": event,
        "trace_id": state.trace_id,
        "iter": state.iteration_count,
        **fields,
    }
    _retrieval_logger.info(json.dumps(payload, default=str, ensure_ascii=False))


def _serialize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Flatten a retrieved chunk into a JSON-serialisable log record."""
    metadata = dict(chunk.get("metadata") or {})
    source_title = metadata.get("source_title", "")
    chunk_index = metadata.get("chunk_index", 0)
    return {
        "id": f"{source_title}::{chunk_index}",
        "source_title": source_title,
        "chunk_index": chunk_index,
        "distance": chunk.get("distance"),
        "text": chunk.get("text", ""),
        "metadata": metadata,
    }


def _call_llm_and_log(
    system_prompt: str,
    user_message: str,
    *,
    json_mode: bool,
    node: str,
    state: AgentState,
) -> str:
    """Invoke _call_llm and emit an llm_call retrieval event.

    Wraps the existing provider-dispatching _call_llm with timing plus a
    structured log of the full system prompt, user message, and response.
    The public _call_llm signature is intentionally left untouched so
    existing test mocks continue to work.
    """
    start = time.perf_counter()
    try:
        response = _call_llm(system_prompt, user_message, json_mode=json_mode)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_event(
            "llm_call",
            state,
            node=node,
            system_prompt=system_prompt,
            user_message=user_message,
            json_mode=json_mode,
            response=None,
            error=repr(exc),
            latency_ms=latency_ms,
        )
        raise
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    _log_event(
        "llm_call",
        state,
        node=node,
        system_prompt=system_prompt,
        user_message=user_message,
        json_mode=json_mode,
        response=response,
        latency_ms=latency_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    """Render a list of retrieved chunks into a plain text block for the LLM.

    Args:
        chunks: Dicts with at least a 'text' key.

    Returns:
        Newline-separated text block.
    """
    return "\n\n".join(c.get("text", "") for c in chunks if c.get("text"))


def _build_context_block(state: AgentState) -> str:
    """Assemble all retrieved context into a single block for synthesis.

    Args:
        state: Current agent state containing all accumulated chunks.

    Returns:
        Formatted string with entity headers and their associated chunks.
    """
    lines: list[str] = []
    for entity, chunks in state.resolved_entities.items():
        lines.append(f"=== {entity} ===")
        lines.append(_format_chunks(chunks))
    if state.retrieved_context:
        lines.append("=== Additional Context ===")
        lines.append(_format_chunks(state.retrieved_context))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def route_question(state: AgentState) -> AgentState:
    """Extract an entity from the question for targeted page lookup.

    Always attempts entity extraction so the retrieve node can do an
    exact page lookup first, regardless of question type. Falls back
    to semantic search only when no entity can be identified.

    Args:
        state: Initial agent state with only question populated.

    Returns:
        Updated state with current_entity set if a named entity was
        detected.
    """
    if not state.trace_id:
        state.trace_id = uuid.uuid4().hex
    entity = _extract_entity_from_question(state.question)
    state.current_entity = entity
    _log_event(
        "query_received",
        state,
        question=state.question,
        current_entity=entity,
    )
    return state


def retrieve(state: AgentState) -> AgentState:
    """Fetch chunks for the current entity or run semantic search.

    Skips entities already in visited to prevent infinite loops on circular
    prerequisite chains.

    iteration_count is incremented unconditionally at the top of this
    function — before the visited-skip check — on purpose. The cap is a
    hard ceiling on graph walks, not a measure of "useful work". When
    extract repeatedly fails to parse JSON, _handle_parse_failure flips
    needs_more_retrieval back to True after every retrieve, so the
    visited-skip path alone cannot break the loop; only the iteration
    cap can. Counting visited-skips against the cap costs at most a
    handful of fast iterations in pathological cases, which is the
    correct tradeoff for guaranteed termination.

    Args:
        state: Agent state with current_entity and visited populated.

    Returns:
        Updated state with new chunks appended to retrieved_context and
        resolved_entities, and current_entity added to visited.
    """
    state.iteration_count += 1

    entity = state.current_entity
    if entity is not None and entity in state.visited:
        state.needs_more_retrieval = False
        _log_event(
            "retrieve",
            state,
            entity=entity,
            query_text=entity,
            top_k=settings.retrieval_top_k,
            chunks=[],
            skipped_reason="already_visited",
        )
        return state

    if entity is not None:
        chunks = _fetch_entity_chunks(entity, state.question)
        state.resolved_entities[entity] = chunks
        state.retrieved_context.extend(chunks)
        state.visited.add(entity)
        query_text = entity
    else:
        chunks = _semantic_search_for_question(state.question)
        state.retrieved_context.extend(chunks)
        query_text = state.question

    _log_event(
        "retrieve",
        state,
        entity=entity,
        query_text=query_text,
        top_k=settings.retrieval_top_k,
        chunks=[_serialize_chunk(c) for c in chunks],
    )

    return state


def extract_info(state: AgentState) -> AgentState:
    """Extract relevant facts and identify entities needing further lookup.

    Sends the retrieved content to the LLM with the original question
    and parses the structured JSON response to decide whether more
    retrieval is needed and what entity to look up next.

    Args:
        state: Agent state after the latest retrieve step.

    Returns:
        Updated state with prerequisite_chain extended and next entity
        queued if further retrieval is needed.
    """
    entity = state.current_entity or "the topic"
    recent_chunks = state.resolved_entities.get(entity, state.retrieved_context[-5:])
    context_text = _format_chunks(recent_chunks)

    user_message = (
        f"Question: {state.question}\n\n"
        f"Currently looking at: {entity}\n\n"
        f"Wiki content:\n{context_text}"
    )

    data = _extract_with_retry(user_message, state)
    if data is None:
        updated = _handle_parse_failure(state)
        _log_event(
            "extract_decision",
            updated,
            entity=entity,
            next_entity=None,
            prerequisite_chain=list(updated.prerequisite_chain),
            needs_more_retrieval=updated.needs_more_retrieval,
            parse_failed=True,
        )
        return updated
    updated = _apply_extract_result(data, state)
    _log_event(
        "extract_decision",
        updated,
        entity=entity,
        next_entity=data.get("next_entity"),
        is_complete=bool(data.get("is_complete", False)),
        prerequisite_chain=list(updated.prerequisite_chain),
        needs_more_retrieval=updated.needs_more_retrieval,
        key_facts=data.get("key_facts", []),
    )
    return updated


def _extract_with_retry(
    user_message: str, state: AgentState, max_attempts: int = 3
) -> dict[str, Any] | None:
    """Call the extract LLM with retries on JSON decode failure.

    First attempt uses the unmodified user_message. On decode failure,
    subsequent attempts include the previous malformed response and an
    explicit nudge to return strict JSON. Capped at max_attempts total
    LLM calls.

    Cost note: this retry budget compounds with agent_max_iterations —
    in the worst case (every extract call returns malformed JSON for
    every iteration) the agent makes max_attempts * agent_max_iterations
    extract calls per question, plus the synthesize call. With the
    defaults of 3 and 10 that is up to 30 extract calls. This is an
    intentional tradeoff: we prefer to keep retrying and produce a
    correct, grounded answer over short-circuiting to save tokens. If
    cost becomes a concern, lower max_attempts here before lowering
    agent_max_iterations — losing iterations sacrifices multi-hop
    reasoning, while losing JSON-retry attempts only hurts robustness
    against a flaky LLM.

    Args:
        user_message: The initial user content for the extract prompt.
        state: Agent state, forwarded so each LLM call can be correlated
            to the enclosing trace in the retrieval log.
        max_attempts: Total number of LLM calls allowed (default 3 =
            1 initial + 2 retries).

    Returns:
        Parsed dict on success, or None if all attempts fail.
    """
    current_message = user_message
    for attempt in range(1, max_attempts + 1):
        response_text = _call_llm_and_log(
            _EXTRACT_SYSTEM_PROMPT,
            current_message,
            json_mode=True,
            node="extract_info",
            state=state,
        )
        cleaned = _strip_markdown_fences(response_text)
        try:
            data: dict[str, Any] = json.loads(cleaned)
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "extract_info: JSON decode failed on attempt %d/%d. Response: %s",
                attempt,
                max_attempts,
                response_text[:200],
            )
            current_message = (
                f"{user_message}\n\n"
                f"Your previous response was:\n{response_text}\n\n"
                "Your previous response was not valid JSON. Respond with "
                "ONLY a valid JSON object matching the schema. No markdown, "
                "no prose."
            )

    logger.error(
        "extract_info: JSON decode failed after %d attempts; giving up.",
        max_attempts,
    )
    return None


def _apply_extract_result(data: dict[str, Any], state: AgentState) -> AgentState:
    """Update agent state from a successfully parsed extract response.

    Args:
        data: Parsed JSON dict from the extract LLM.
        state: Current agent state to update.

    Returns:
        Updated agent state.
    """
    prerequisites: list[str] = data.get("prerequisites", [])
    for prereq in prerequisites:
        if prereq not in state.prerequisite_chain:
            state.prerequisite_chain.append(prereq)

    is_complete: bool = bool(data.get("is_complete", False))
    next_entity: str | None = data.get("next_entity")

    if is_complete or next_entity is None:
        state.needs_more_retrieval = False
    elif next_entity in state.visited:
        # Cycle detected — all remaining entities already resolved.
        state.needs_more_retrieval = False
    else:
        state.current_entity = next_entity
        state.needs_more_retrieval = True

    return state


def _handle_parse_failure(state: AgentState) -> AgentState:
    """Fallback when all JSON parse retries are exhausted.

    Prefer continuing retrieval over returning an incomplete answer.
    Setting needs_more_retrieval=True without changing current_entity
    means the next retrieve will hit the visited-skip path, which by
    itself cannot terminate the loop because this function will run
    again on the next iteration and flip the flag back. Termination
    relies on retrieve incrementing iteration_count unconditionally
    (including on visited-skips) so that _route_after_check's
    iteration-limit branch eventually fires. See the matching note on
    retrieve and _route_after_check.

    Args:
        state: Current agent state.

    Returns:
        State with needs_more_retrieval set to True.
    """
    logger.error(
        "extract_info: falling back to needs_more_retrieval=True after "
        "exhausting JSON parse retries."
    )
    state.needs_more_retrieval = True
    return state


def check_complete(state: AgentState) -> AgentState:
    """Evaluate whether the agent should continue, synthesize, or hit the limit.

    This node does not modify state — it exists as a named checkpoint so
    the conditional edge routing logic has a clean place to branch.

    Args:
        state: Current agent state.

    Returns:
        State unchanged; routing is handled by graph conditional edges.
    """
    return state


def synthesize_answer(state: AgentState) -> AgentState:
    """Compose the final answer from all accumulated context and the prerequisite chain.

    Args:
        state: Agent state with all retrieved context and prerequisite chain.

    Returns:
        Updated state with final_answer populated.
    """
    context_block = _build_context_block(state)
    chain_text = (
        " → ".join(state.prerequisite_chain) if state.prerequisite_chain else "None"
    )

    user_message = (
        f"Question: {state.question}\n\n"
        f"Prerequisite chain discovered: {chain_text}\n\n"
        f"Wiki content:\n{context_block}"
    )

    state.final_answer = _call_llm_and_log(
        _SYNTHESIZE_SYSTEM_PROMPT,
        user_message,
        json_mode=False,
        node="synthesize_answer",
        state=state,
    )
    _log_event(
        "synthesize",
        state,
        kind="full",
        final_answer=state.final_answer,
        prerequisite_chain=list(state.prerequisite_chain),
    )
    return state


def handle_iteration_limit(state: AgentState) -> AgentState:
    """Return the best available partial answer when max iterations is reached.

    Called when the agent has iterated agent_max_iterations times without
    reaching a definitive answer. Produces a partial answer with an explicit
    note that the chain may be incomplete.

    Args:
        state: Agent state at the iteration limit.

    Returns:
        Updated state with final_answer populated with a partial result.
    """
    context_block = _build_context_block(state)
    chain_text = (
        " → ".join(state.prerequisite_chain) if state.prerequisite_chain else "None"
    )

    user_message = (
        f"Question: {state.question}\n\n"
        f"Prerequisite chain found so far (may be incomplete): {chain_text}\n\n"
        f"Wiki content retrieved:\n{context_block}"
    )

    state.final_answer = _call_llm_and_log(
        _PARTIAL_SYSTEM_PROMPT,
        user_message,
        json_mode=False,
        node="handle_iteration_limit",
        state=state,
    )
    _log_event(
        "synthesize",
        state,
        kind="partial",
        final_answer=state.final_answer,
        prerequisite_chain=list(state.prerequisite_chain),
    )
    return state
