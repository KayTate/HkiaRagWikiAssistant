"""Entity → chunks resolution for the HKIA agent.

Given a normalized entity name (or, as a fallback, a free-text user
question), produces the list of vector-store chunks that the
LLM-driven nodes will reason over. Resolution order is title variants
first, then opensearch, then semantic search — wrapped in a single
RETRIEVER MLflow span so the trace records exactly the chunks the LLM
will see.

Constants and functions keep their leading-underscore prefix because
they are package-internal — consumed by ``agent.nodes``'s ``retrieve``
and ``_fetch_entity_chunks`` callers, and not part of any public API.

Tests that previously patched ``agent.nodes.vs_get_page_by_title``,
``agent.nodes.vs_semantic_search``, ``agent.nodes.embed_chunks``, or
``agent.nodes._resolve_title_via_opensearch`` must now patch the
``agent.retrieval`` binding — Python rebinds names per module, and
those names live here after this extraction.
"""

import logging
import re
from typing import Any

import mlflow
from mlflow.entities import Document, SpanType

from config.settings import settings
from ingestion.embedder import embed_chunks
from vectorstore.client import (
    get_page_by_title as vs_get_page_by_title,
)
from vectorstore.client import (
    semantic_search as vs_semantic_search,
)

logger = logging.getLogger(__name__)

# Matches the canonical post-strip_code form of a MediaWiki redirect page:
# wikitext "#REDIRECT [[Target]]" → "REDIRECT Target" after stripping markup.
# The \b after REDIRECT rules out false positives like REDIRECTING /
# REDIRECTED / REDIRECTLY, and [^\n]+ caps the captured title at the first
# newline so trailing parsed content does not get slurped in.
_REDIRECT_RE = re.compile(r"^\s*REDIRECT\b\s+([^\n]+)", re.IGNORECASE)
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
    """Convert agent chunk dicts to MLflow Document entities for span outputs.

    The agent uses ``{"text": ..., "metadata": ...}`` internally, but
    MLflow's RetrievalGroundedness scorer reads RETRIEVER spans by
    looking for documents with ``page_content``. This helper bridges the
    two formats at the span boundary so traces are scorable without
    changing the chunk shape that downstream nodes consume.

    Args:
        chunks: Retrieved chunk dicts as produced by vectorstore.client.

    Returns:
        List of Document entities with page_content sourced from each
        chunk's 'text' field and metadata copied through unchanged.
    """
    return [
        Document(
            page_content=str(c.get("text", "")),
            metadata=dict(c.get("metadata") or {}),
        )
        for c in chunks
    ]


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


def _extract_redirect_target(chunks: list[dict[str, Any]]) -> str | None:
    """Detect if chunks represent a wiki redirect and extract the target.

    A redirect page has a single chunk whose text matches the canonical
    post-strip_code form ``REDIRECT <target>`` (with whitespace between
    the keyword and the title). Match is case-insensitive but requires a
    word boundary after REDIRECT so prose pages starting with words like
    "Redirecting" or "Redirected" are not misclassified as redirects.

    Args:
        chunks: Chunks retrieved for a page.

    Returns:
        The redirect target title, or None if not a redirect.
    """
    if len(chunks) != 1:
        return None
    text = str(chunks[0].get("text", "")).strip()
    match = _REDIRECT_RE.match(text)
    if match is None:
        return None
    target = match.group(1).strip()
    return target or None


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
