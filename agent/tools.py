"""Retrieval tools exposed to the HKIA LangGraph agent.

Each tool wraps one or more vector store operations. The semantic_search
tool embeds the query before calling the vector store so callers never
deal with raw embeddings.
"""

import logging
from typing import Any

from langchain_core.tools import tool

from config.settings import settings
from ingestion.embedder import embed_chunks
from vectorstore.client import (
    get_page_by_title as vs_get_page_by_title,
)
from vectorstore.client import (
    semantic_search as vs_semantic_search,
)

logger = logging.getLogger(__name__)


@tool
def semantic_search(
    query: str, category_filter: str | None = None
) -> list[dict[str, Any]]:
    """Search the HKIA wiki by semantic similarity.

    Use for broad questions where the exact page title is not known.
    Returns the most relevant chunks across all wiki pages. Optionally
    filter by category (e.g. 'Quests', 'Characters', 'Items').

    Args:
        query: Natural language query to search for.
        category_filter: Optional ChromaDB metadata category filter.

    Returns:
        List of result dicts with 'text', 'metadata', and 'distance' keys.
    """
    embeddings = embed_chunks([query])
    query_embedding = embeddings[0]

    where: dict[str, Any] | None = None
    if category_filter is not None:
        where = {"category": {"$eq": category_filter}}

    return vs_semantic_search(
        query_embedding=query_embedding,
        top_k=settings.retrieval_top_k,
        where=where,
    )


@tool
def get_page(page_title: str) -> list[dict[str, Any]]:
    """Retrieve all content chunks for a specific wiki page by exact title.

    Preferred over semantic_search when traversing prerequisite chains
    because it guarantees full page coverage. Falls back to semantic_search
    if the exact title is not found in the vector store.

    Args:
        page_title: Exact name of the quest, character, item, or location.

    Returns:
        List of chunk dicts with 'text' and 'metadata' keys.
    """
    results = vs_get_page_by_title(page_title)
    if results:
        return results

    logger.info(
        "Exact page '%s' not found; falling back to semantic_search", page_title
    )
    embeddings = embed_chunks([page_title])
    query_embedding = embeddings[0]
    return vs_semantic_search(
        query_embedding=query_embedding,
        top_k=settings.retrieval_top_k,
    )
