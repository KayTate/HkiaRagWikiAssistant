"""Node functions for the LangGraph agent graph.

Each function accepts and returns an AgentState. Nodes are pure functions
from the graph's perspective — side effects are limited to LLM calls and
vector store reads.
"""

import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.state import AgentState
from config.settings import settings
from ingestion.embedder import embed_chunks
from vectorstore.client import (
    get_page_by_title as vs_get_page_by_title,
)
from vectorstore.client import (
    semantic_search as vs_semantic_search,
)

logger = logging.getLogger(__name__)
_retrieval_logger = logging.getLogger("retrieval")

_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAILING_DESCRIPTOR_RE = re.compile(
    r"\s+(quest\s+series|quest|recipe|item|character|location|"
    r"companion|ability|page|series)s?$",
    re.IGNORECASE,
)
_TITLE_VARIANT_SUFFIXES: tuple[str, ...] = (
    " (quest series)",
    " (quest)",
    " (character)",
    " (item)",
    " (location)",
    " (ability)",
    " (companion ability)",
)


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
# LLM helpers
# ---------------------------------------------------------------------------


def _call_llm(system_prompt: str, user_message: str, json_mode: bool = False) -> str:
    """Send a chat request to the configured LLM provider.

    Dispatches to Ollama, OpenAI, or Anthropic based on settings.llm_provider.
    Returns the assistant's reply as a plain string.

    Args:
        system_prompt: Instruction context for the LLM.
        user_message: The user-facing message or question.
        json_mode: When True and the provider supports it, request a
            strict JSON object response. Currently only wired up for
            OpenAI; silently ignored for Ollama and Anthropic.

    Returns:
        The LLM's text response.

    Raises:
        RuntimeError: If the LLM provider returns an unexpected response or
            if an unsupported provider is configured.
    """
    if settings.llm_provider == "ollama":
        # json_mode accepted but not wired up for Ollama yet.
        return _call_ollama(system_prompt, user_message)
    if settings.llm_provider == "openai":
        return _call_openai(system_prompt, user_message, json_mode=json_mode)
    if settings.llm_provider == "anthropic":
        # json_mode accepted but not wired up for Anthropic yet.
        return _call_anthropic(system_prompt, user_message)
    raise RuntimeError(
        f"Unsupported llm_provider '{settings.llm_provider}'. "
        "Expected one of: ollama, openai, anthropic."
    )


def _call_ollama(system_prompt: str, user_message: str) -> str:
    """Call the local Ollama chat API.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.

    Returns:
        The model's reply text.

    Raises:
        RuntimeError: On HTTP error or unexpected response shape.
    """
    url = "http://localhost:11434/api/chat"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["message"]["content"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Ollama response missing expected fields for model "
            f"'{settings.llm_model}': {exc}"
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP error {exc.response.status_code} calling '{url}': {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed calling '{url}': {exc}") from exc


def _call_openai(system_prompt: str, user_message: str, json_mode: bool = False) -> str:
    """Call the OpenAI chat completions API with retry on rate limits.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.
        json_mode: When True, request a strict JSON object response via
            the OpenAI response_format parameter.

    Returns:
        The model's reply text.

    Raises:
        RuntimeError: On HTTP error, missing API key, or unexpected response.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured but llm_provider is 'openai'."
        )
    return _call_openai_with_retry(system_prompt, user_message, json_mode=json_mode)


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_openai_with_retry(
    system_prompt: str, user_message: str, json_mode: bool = False
) -> str:
    """Send a chat request to OpenAI with automatic retry on 429.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.
        json_mode: When True, include response_format={"type":
            "json_object"} in the request so OpenAI guarantees a valid
            JSON object response.

    Returns:
        The model's reply text.

    Raises:
        requests.HTTPError: On rate limit after retries exhausted.
        RuntimeError: On unexpected response shape.
    """
    url = "https://api.openai.com/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        if response.status_code == 429:
            logger.warning("OpenAI rate limit hit, will retry with backoff")
            response.raise_for_status()
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"OpenAI response missing expected fields for model "
            f"'{settings.llm_model}': {exc}"
        ) from exc
    except requests.HTTPError:
        raise
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenAI request failed calling '{url}': {exc}") from exc


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """Call the Anthropic Messages API.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.

    Returns:
        The model's reply text.

    Raises:
        RuntimeError: On HTTP error, missing API key, or unexpected response.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured but llm_provider is 'anthropic'."
        )
    url = "https://api.anthropic.com/v1/messages"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    try:
        response = requests.post(
            url,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["content"][0]["text"])
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Anthropic response missing expected fields for model "
            f"'{settings.llm_model}': {exc}"
        ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Anthropic HTTP error {exc.response.status_code} calling '{url}': {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Anthropic request failed calling '{url}': {exc}") from exc


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


def _normalize_entity(raw: str) -> str:
    """Strip leading articles and trailing descriptors from an extracted entity.

    Iterates until stable because an entity can contain both (e.g.
    'the Wild Mountain Time quest series' → 'Wild Mountain Time').

    Args:
        raw: Raw captured entity text from the regex extractor.

    Returns:
        Cleaned entity name. May be empty string if input was only
        articles and descriptors.
    """
    entity = raw.strip().strip("\"'")
    prev: str | None = None
    while prev != entity:
        prev = entity
        entity = _LEADING_ARTICLE_RE.sub("", entity).strip()
        entity = _TRAILING_DESCRIPTOR_RE.sub("", entity).strip()
    return entity


def _extract_entity_from_question(question: str) -> str | None:
    """Extract the most likely game entity name from a question.

    Uses simple heuristics: looks for quoted names first, then text
    following action verbs like 'craft', 'unlock', 'find', 'get', etc.
    All captures are normalized to strip articles and descriptors.
    Returns None if no entity can be confidently identified.

    Args:
        question: The original user question string.

    Returns:
        Entity name string or None if extraction fails.
    """
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        cleaned = _normalize_entity(str(quoted[0]))
        return cleaned or None

    patterns = [
        r"craft\s+(?:a\s+|an\s+)?(.+?)(?:\?|$|\.|\,)",
        r"unlock\s+(.+?)(?:\?|$|\.|\,)",
        r"complete\s+(.+?)(?:\?|$|\.|\,)",
        r"find\s+(.+?)(?:\?|$|\.|\,)",
        r"get\s+(?:to\s+)?(.+?)(?:\?|$|\.|\,)",
        r"reach\s+(.+?)(?:\?|$|\.|\,)",
        r"where\s+is\s+(.+?)(?:\?|$|\.|\,)",
        r"who\s+is\s+(.+?)(?:\?|$|\.|\,)",
        r"about\s+(.+?)(?:\?|$|\.|\,)",
        r"does\s+(.+?)\s+like",
        r"gifts?\s+(?:does\s+|for\s+)(.+?)(?:\?|$|\.|\,|\s+like)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            entity = _normalize_entity(match.group(1))
            if entity and entity.lower() not in {"i", "you", "it", "this"}:
                return entity

    return None


def retrieve(state: AgentState) -> AgentState:
    """Fetch chunks for the current entity or run semantic search.

    Increments iteration_count before any retrieval so the limit check
    in check_complete fires correctly even if retrieval raises. Skips
    entities already in visited to prevent infinite loops on circular
    prerequisite chains.

    Args:
        state: Agent state with current_entity and visited populated.

    Returns:
        Updated state with new chunks appended to retrieved_context and
        resolved_entities, and current_entity added to visited.
    """
    # Increment before retrieval so the limit check is accurate.
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


def _title_candidates(entity: str) -> list[str]:
    """Return plausible wiki titles for an extracted entity.

    Tries the entity as-is, with a leading 'The ' prefix (for pages where
    the article is part of the canonical title — e.g. 'The Mystery Tree'),
    and each form combined with common disambiguation suffixes.
    Ordered most-likely-match first.

    Args:
        entity: Normalized entity name.

    Returns:
        List of candidate wiki page titles to try in order.
        Empty list if entity is empty.
    """
    cleaned = entity.strip()
    if not cleaned:
        return []

    bases = [cleaned]
    if not cleaned.lower().startswith("the "):
        bases.append(f"The {cleaned}")

    candidates: list[str] = []
    for base in bases:
        candidates.append(base)
        candidates.extend(base + suffix for suffix in _TITLE_VARIANT_SUFFIXES)

    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _resolve_title_via_opensearch(entity: str) -> str | None:
    """Resolve an entity name to a canonical wiki title via MediaWiki opensearch.

    Thin wrapper around the API client that isolates the agent from API
    import concerns. Returns None on any failure — the caller treats
    this as 'no resolution' and continues to semantic search.

    Args:
        entity: Normalized entity name.

    Returns:
        Canonical page title string, or None.
    """
    from ingestion.api_client import opensearch_title

    try:
        return opensearch_title(entity)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        logger.warning("opensearch lookup failed for '%s': %s", entity, exc)
        return None


def _fetch_entity_chunks(
    entity: str, question: str | None = None
) -> list[dict[str, Any]]:
    """Retrieve chunks for the named entity via exact page lookup with fallbacks.

    Resolution order:
    1. Try the entity and common disambiguation suffixes as exact titles.
    2. Try 'The {entity}' and its suffix variants (for pages where 'The'
       is part of the canonical title).
    3. Call the MediaWiki opensearch API to resolve free-text to a
       canonical page title, then look that up.
    4. Fall back to semantic search using the full question as the query.

    Follows wiki redirect pages automatically at any stage.

    Args:
        entity: Extracted entity name (already normalized).
        question: The original user question; used for semantic fallback.

    Returns:
        List of chunk dicts. May be empty if every fallback fails.
    """
    for candidate in _title_candidates(entity):
        chunks = vs_get_page_by_title(candidate)
        if not chunks:
            continue

        redirect_target = _extract_redirect_target(chunks)
        if redirect_target is not None:
            logger.info("'%s' redirects to '%s'; following", candidate, redirect_target)
            redirected = vs_get_page_by_title(redirect_target)
            if redirected:
                return redirected

        if candidate != entity:
            logger.info("Resolved '%s' to wiki page '%s'", entity, candidate)
        return chunks

    resolved_title = _resolve_title_via_opensearch(entity)
    if resolved_title is not None:
        logger.info(
            "opensearch resolved '%s' to canonical title '%s'",
            entity,
            resolved_title,
        )
        chunks = vs_get_page_by_title(resolved_title)
        if chunks:
            redirect_target = _extract_redirect_target(chunks)
            if redirect_target is not None:
                logger.info(
                    "'%s' redirects to '%s'; following",
                    resolved_title,
                    redirect_target,
                )
                redirected = vs_get_page_by_title(redirect_target)
                if redirected:
                    return redirected
            return chunks

    logger.info(
        "No title match for '%s' after all fallbacks; using semantic search",
        entity,
    )
    search_query = question if question else entity
    embeddings = embed_chunks([search_query])
    return vs_semantic_search(
        query_embedding=embeddings[0], top_k=settings.retrieval_top_k
    )


def _extract_redirect_target(chunks: list[dict[str, Any]]) -> str | None:
    """Detect if chunks represent a wiki redirect and extract the target.

    A redirect page has a single chunk whose text starts with 'REDIRECT'
    followed by the target page title.

    Args:
        chunks: Chunks retrieved for a page.

    Returns:
        The redirect target title, or None if not a redirect.
    """
    if len(chunks) != 1:
        return None
    text = str(chunks[0].get("text", "")).strip()
    if text.upper().startswith("REDIRECT"):
        target = text[len("REDIRECT") :].strip()
        if target:
            return target
    return None


def _semantic_search_for_question(question: str) -> list[dict[str, Any]]:
    """Run semantic search using the full question as the query.

    Args:
        question: The user's original question.

    Returns:
        List of chunk dicts from the vector store.
    """
    embeddings = embed_chunks([question])
    return vs_semantic_search(
        query_embedding=embeddings[0], top_k=settings.retrieval_top_k
    )


_EXTRACT_SYSTEM_PROMPT = (
    "You are analyzing wiki content for Hello Kitty Island Adventure. "
    "Given text chunks from the wiki and a player's question, determine:\n"
    "1. Does the retrieved content fully answer the question?\n"
    "2. Are there related entities (quests, characters, items, locations) "
    "mentioned that need to be looked up for a complete answer?\n"
    "3. Are there prerequisites or dependencies that need resolving?\n\n"
    "Output format (STRICT):\n"
    "Respond with ONLY a valid JSON object. DO NOT include markdown code "
    "fences (no ```json, no ```), commentary, explanations, or any prose "
    "before or after the JSON. The entire response must be parseable by "
    "json.loads. Every field listed below is REQUIRED on every response:\n"
    "{\n"
    '  "prerequisites": ["Entity Name 1", "Entity Name 2"],\n'
    '  "has_unresolved": true,\n'
    '  "next_entity": "Entity Name 1",\n'
    '  "is_complete": false,\n'
    '  "key_facts": ["fact 1 from the content", "fact 2"]\n'
    "}\n\n"
    "Example of a CORRECTLY formatted response:\n"
    '{"prerequisites": ["Straight to Your Heart"], '
    '"has_unresolved": true, '
    '"next_entity": "Straight to Your Heart", '
    '"is_complete": false, '
    '"key_facts": ["Ice and Glow unlocks the Frozen Falls area"]}\n\n'
    "Example of an INCORRECT response (do NOT do this):\n"
    "Here is the analysis:\n"
    "```json\n"
    '{"prerequisites": ["Straight to Your Heart"], ...}\n'
    "```\n"
    "This is wrong because it includes prose and markdown fences.\n\n"
    "Rules:\n"
    "- Set is_complete to true ONLY if the content contains enough "
    "specific detail to fully answer the question.\n"
    "- If the content mentions another entity that would help answer "
    "the question, set next_entity to that entity name. Otherwise use "
    "null.\n"
    "- Put specific facts extracted from the content in key_facts "
    "(e.g. recipe ingredients, gift preferences, location names).\n"
    "- prerequisites should list any quests/tasks that must be done first.\n"
    "- If the content is a redirect, set next_entity to the redirect target."
)


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


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from an LLM response.

    LLMs often wrap JSON output in ```json ... ``` fences despite being
    told not to. This strips them so json.loads can parse the content.

    Args:
        text: Raw LLM response that may contain markdown fences.

    Returns:
        The text with leading/trailing fences removed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _extract_with_retry(
    user_message: str, state: AgentState, max_attempts: int = 3
) -> dict[str, Any] | None:
    """Call the extract LLM with retries on JSON decode failure.

    First attempt uses the unmodified user_message. On decode failure,
    subsequent attempts include the previous malformed response and an
    explicit nudge to return strict JSON. Capped at max_attempts total
    LLM calls.

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

    Prefer continuing retrieval over returning an incomplete answer. The
    agent_max_iterations cap (default 10) prevents runaway looping.

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


_SYNTHESIZE_SYSTEM_PROMPT = (
    "You are a helpful assistant for Hello Kitty Island Adventure. "
    "Answer the player's question using ONLY the wiki content provided below. "
    "Include specific details from the wiki: exact item names, quantities, "
    "recipe ingredients, character names, location names, and quest names. "
    "If the wiki content contains the answer, state it directly with the "
    "specific details. Do not give generic gaming advice. "
    "If the wiki content does not contain enough information to answer, "
    "say so clearly rather than guessing. "
    "Be specific about the order of steps if prerequisites are involved."
)


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


_PARTIAL_SYSTEM_PROMPT = (
    "You are a helpful assistant for Hello Kitty Island Adventure. "
    "The research was cut short before all information could be gathered. "
    "Based on the wiki content found so far, provide the best available "
    "answer using ONLY specific details from the content. "
    "Do not give generic advice. Clearly note that the answer may be "
    "incomplete."
)


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
