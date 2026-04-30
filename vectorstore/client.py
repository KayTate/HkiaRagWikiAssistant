"""ChromaDB client interface for the HKIA RAG vector store."""

import logging
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config.settings import settings
from vectorstore.schema import ChunkMetadata

logger = logging.getLogger(__name__)


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when the ChromaDB collection was built with a different embedding model.

    Operator intervention is required: the collection cannot be auto-repaired
    because mixing vectors from different models corrupts search results. See
    the error message for the three-step remediation procedure.
    """


_chroma_client: ClientAPI | None = None


def _get_client() -> ClientAPI:
    """Return the shared ChromaDB persistent client, creating it on first call.

    Uses a module-level singleton to avoid re-opening the persistent store
    on every call.
    """
    global _chroma_client
    if _chroma_client is None:
        logger.info(
            "Initializing ChromaDB client with persist directory '%s'",
            settings.chroma_persist_dir,
        )
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def reset_client() -> None:
    """Reset the ChromaDB client singleton. Intended for test teardown.

    Prevents cross-test contamination when tests patch ``chroma_persist_dir``
    to different paths. No-op if the client was never initialized.
    """
    global _chroma_client
    _chroma_client = None


def get_or_create_collection(name: str) -> Collection:
    """Return the named ChromaDB collection, creating it if absent.

    Args:
        name: Collection name. Typically settings.chroma_collection_name.

    Returns:
        The existing or newly created ChromaDB collection object.
    """
    client = _get_client()
    return client.get_or_create_collection(name=name)


def _current_embedding_model() -> str:
    """Return the formatted embedding model identifier from current settings."""
    return f"{settings.embedding_model}:{settings.embedding_model_version}"


def verify_collection_embedding_model() -> None:
    """Verify that the ChromaDB collection uses the current embedding model.

    Samples up to 10 chunks from the collection (the first N rows in
    insertion order, not a random sample). If any sampled chunk was
    embedded with a different model than the one currently configured,
    raises EmbeddingModelMismatchError with remediation steps.

    Raises:
        EmbeddingModelMismatchError: If the collection contains chunks from
            a different embedding model. Operator intervention is required
            to resolve this — it cannot be auto-repaired.
    """
    logger.info("Verifying ChromaDB collection embedding model")
    collection = get_or_create_collection(settings.chroma_collection_name)
    count = collection.count()
    if count == 0:
        return

    sample_size = min(10, count)
    results = collection.get(limit=sample_size, include=["metadatas"])
    metadatas: list[Any] = results.get("metadatas") or []

    current_model = _current_embedding_model()
    for meta in metadatas:
        if meta is None:
            continue
        stored_model = meta.get("embedding_model", "")
        if stored_model != current_model:
            raise EmbeddingModelMismatchError(
                f"ChromaDB collection '{settings.chroma_collection_name}' contains "
                f"chunks embedded with '{stored_model}', but the current embedding "
                f"model is '{current_model}'. Automatic repair is not possible. "
                f"To resolve:\n"
                f"  1. Update settings.chroma_collection_name to a new version "
                f"(e.g. hkia_v2).\n"
                f"  2. Run full ingestion to build the new collection.\n"
                f"  3. Update config to point the application at the new collection."
            )


def upsert_chunks(
    page_title: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[ChunkMetadata],
) -> None:
    """Insert or replace all chunks for a page in the ChromaDB collection.

    IDs are derived from page_title and chunk_index, so re-running this
    for the same page overwrites the previous chunks without duplicates
    as long as delete_chunks_by_source is called first.

    Args:
        page_title: The wiki page title, used for ID namespacing.
        chunks: List of text chunks to store.
        embeddings: Pre-computed embedding vectors, one per chunk.
        metadatas: ChunkMetadata instances, one per chunk.
    """
    collection = get_or_create_collection(settings.chroma_collection_name)
    ids = [f"{page_title}::{meta.chunk_index}" for meta in metadatas]
    meta_dicts = [meta.model_dump() for meta in metadatas]
    # chromadb's upsert type expects numpy arrays but also accepts list[Sequence[float]]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,  # type: ignore[arg-type]
        metadatas=meta_dicts,  # type: ignore[arg-type]
    )


def delete_chunks_by_source(page_title: str) -> None:
    """Delete all chunks associated with a specific wiki page.

    Safe to call even if the page has no chunks — this is a no-op in
    that case. Used before re-ingesting a page to prevent duplicate chunks
    from accumulating across retries.

    Args:
        page_title: The wiki page title to delete chunks for.
    """
    collection = get_or_create_collection(settings.chroma_collection_name)
    # chromadb Where type is complex; cast to suppress the mypy mismatch
    where: Any = {"source_title": {"$eq": page_title}}
    collection.delete(where=where)


def semantic_search(
    query_embedding: list[float],
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search the vector store by semantic similarity.

    Args:
        query_embedding: The embedded query vector to search against.
        top_k: Maximum number of results to return.
        where: Optional metadata filter dict. If None, no filter is applied.

    Returns:
        List of result dicts with 'text', 'metadata', and 'distance' keys,
        ordered by ascending distance.

    Raises:
        RuntimeError: If the ChromaDB collection cannot be queried.
    """
    collection = get_or_create_collection(settings.chroma_collection_name)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    documents: list[list[str]] = results.get("documents") or [[]]
    metadatas_raw: list[list[Any]] = results.get("metadatas") or [[]]
    distances: list[list[float]] = results.get("distances") or [[]]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            documents[0], metadatas_raw[0], distances[0], strict=True
        )
    ]


def get_page_by_title(page_title: str) -> list[dict[str, Any]]:
    """Retrieve all chunks for a specific wiki page by exact title match.

    Returns chunks ordered by chunk_index ascending. Never raises on a
    missing page — returns an empty list instead.

    Args:
        page_title: The exact wiki page title to retrieve chunks for.

    Returns:
        List of dicts with 'text' and 'metadata' keys, ordered by
        chunk_index. Empty list if the page is not found.
    """
    collection = get_or_create_collection(settings.chroma_collection_name)
    where: Any = {"source_title": {"$eq": page_title}}
    results = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )
    documents: list[str] = results.get("documents") or []
    metadatas_raw: list[Any] = results.get("metadatas") or []

    pairs = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(documents, metadatas_raw, strict=True)
    ]
    pairs.sort(key=lambda p: p["metadata"].get("chunk_index", 0))
    return pairs
