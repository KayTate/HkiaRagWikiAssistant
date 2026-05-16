"""Entity → chunks resolution for the HKIA agent. Resolution order is title
variants, then opensearch, then semantic search — wrapped in a single
RETRIEVER MLflow span so the trace records exactly what the LLM sees."""

import functools
import logging
from typing import Any

import mlflow
from mlflow.entities import Document, SpanType

from config.settings import settings
from ingestion.embedder import embed_chunks
from ingestion.state_db import get_all_redirects
from vectorstore.client import (
    get_page_by_title as vs_get_page_by_title,
)
from vectorstore.client import (
    semantic_search as vs_semantic_search,
)

logger = logging.getLogger(__name__)

_TITLE_VARIANT_SUFFIXES: tuple[str, ...] = (
    " (quest series)",
    " (quest)",
    " (character)",
    " (item)",
    " (location)",
    " (ability)",
    " (companion ability)",
)


def _chunks_to_documents(chunks: list[dict[str, Any]]) -> list[Document]:
    """Convert chunk dicts to MLflow Documents at the RETRIEVER span boundary."""
    return [
        Document(
            page_content=str(c.get("text", "")),
            metadata=dict(c.get("metadata") or {}),
        )
        for c in chunks
    ]


def _title_candidates(entity: str) -> list[str]:
    """Return plausible wiki titles for an entity, most-likely-match first."""
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


@functools.cache
def _load_redirects() -> dict[str, str]:
    """Load the full redirect side-table once per process.

    ``functools.cache`` memoises the dict for the agent process lifetime
    so the SQLite read happens at most once. Tests reset via the autouse
    fixture in tests/test_entity_resolution.py, which calls
    ``_load_redirects.cache_clear()``. ~5k entries × ~50 chars at the
    current wiki size is well under the cost of paying SQLite latency
    on every agent request.
    """
    return get_all_redirects()


def _resolve_via_redirect(title: str) -> str:
    """Map a wiki title through the redirect side-table, if present.

    Titles not in the map pass through unchanged so every call site can
    wrap a ``vs_get_page_by_title`` argument without a None-check.
    """
    return _load_redirects().get(title, title)


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

    Wrapped in a single RETRIEVER span so the trace records exactly the
    chunks the LLM will see — redirect-follow and semantic-search
    fallbacks are implementation details of one logical retrieval and
    must not produce sibling retriever spans (the RetrievalGroundedness
    scorer would otherwise score against chunks the LLM never received).

    Args:
        entity: Extracted entity name (already normalized).
        question: The original user question; used for semantic fallback.

    Returns:
        List of chunk dicts. May be empty if every fallback fails.
    """
    with mlflow.start_span(
        name="fetch_entity_chunks", span_type=SpanType.RETRIEVER
    ) as span:
        span.set_inputs({"entity": entity, "question": question})
        chunks = _resolve_entity_chunks(entity, question)
        span.set_outputs(_chunks_to_documents(chunks))
        return chunks


def _resolve_entity_chunks(
    entity: str, question: str | None = None
) -> list[dict[str, Any]]:
    """Inner resolution logic for ``_fetch_entity_chunks`` (untraced).

    Separated so the public entry point can wrap the full resolution
    flow in one outer RETRIEVER span without nesting smaller spans for
    each fallback path. See ``_fetch_entity_chunks`` for the resolution
    order and behaviour.

    Args:
        entity: Extracted entity name (already normalized).
        question: The original user question; used for semantic fallback.

    Returns:
        List of chunk dicts. May be empty if every fallback fails.
    """
    for candidate in _title_candidates(entity):
        target = _resolve_via_redirect(candidate)
        chunks = vs_get_page_by_title(target)
        if not chunks:
            continue
        if target != candidate:
            logger.info(
                "'%s' redirects to '%s' (via table); using target chunks",
                candidate,
                target,
            )
        elif candidate != entity:
            logger.info("Resolved '%s' to wiki page '%s'", entity, candidate)
        return chunks

    resolved_title = _resolve_title_via_opensearch(entity)
    if resolved_title is not None:
        logger.info(
            "opensearch resolved '%s' to canonical title '%s'",
            entity,
            resolved_title,
        )
        target = _resolve_via_redirect(resolved_title)
        if target != resolved_title:
            logger.info(
                "'%s' redirects to '%s' (via table); using target chunks",
                resolved_title,
                target,
            )
        chunks = vs_get_page_by_title(target)
        if chunks:
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


def _semantic_search_for_question(question: str) -> list[dict[str, Any]]:
    """Run semantic search using the full question as the query.

    Wrapped in a RETRIEVER span so the chunks reach
    MLflow's RetrievalGroundedness scorer in the canonical document
    shape. Span output mirrors the function's return value, converted
    via ``_chunks_to_documents`` at the boundary.

    Args:
        question: The user's original question.

    Returns:
        List of chunk dicts from the vector store.
    """
    with mlflow.start_span(
        name="semantic_search_for_question", span_type=SpanType.RETRIEVER
    ) as span:
        span.set_inputs({"question": question})
        embeddings = embed_chunks([question])
        chunks = vs_semantic_search(
            query_embedding=embeddings[0], top_k=settings.retrieval_top_k
        )
        span.set_outputs(_chunks_to_documents(chunks))
        return chunks
