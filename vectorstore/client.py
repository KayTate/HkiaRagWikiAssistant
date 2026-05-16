"""ChromaDB client interface for the HKIA RAG vector store."""

import logging
import random
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config.settings import settings
from vectorstore.schema import ChunkMetadata

logger = logging.getLogger(__name__)

_MODEL_VERIFY_SAMPLE_SIZE = 10

# One drift entry: (field_name, stored_value, current_value). Both value
# slots are typed ``object`` rather than ``Any`` because the four fields
# carry mixed primitive types (str / int) and ``object`` keeps the alias
# narrower than ``Any`` without leaking ``Any`` further down the API.
_DriftEntry = tuple[str, object, object]


class CollectionConfigMismatchError(RuntimeError):
    """Raised when ChromaDB chunks disagree with current ingestion config.

    Covers drift in any of the fields whose value must be identical across
    every chunk in a collection: the embedding model and the chunking
    parameters (strategy, size, overlap). Operator intervention is
    required — mixing chunks built with different settings corrupts both
    retrieval (different vector spaces) and ranking (different chunk
    granularity). See the error message for the three-step remediation
    procedure.
    """


_chroma_client: ClientAPI | None = None


def _get_client() -> ClientAPI:
    """Return the shared ChromaDB persistent client, creating it on first call."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(
            "Initializing ChromaDB client with persist directory '%s'",
            settings.chroma_persist_dir,
        )
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def reset_client() -> None:
    """Reset the ChromaDB client singleton. Intended for test teardown."""
    global _chroma_client
    _chroma_client = None


def get_or_create_collection(name: str) -> Collection:
    """Return the named ChromaDB collection, creating it if absent."""
    client = _get_client()
    return client.get_or_create_collection(name=name)


def _current_embedding_model() -> str:
    """Return ``"{model}:{version}"`` from current settings."""
    return f"{settings.embedding_model}:{settings.embedding_model_version}"


def verify_collection_consistency() -> None:
    """Verify that ChromaDB chunks match the current ingestion config.

    Draws a uniform random sample of up to ``_MODEL_VERIFY_SAMPLE_SIZE``
    chunks and compares each chunk's stored metadata against the current
    settings for: embedding model, chunking strategy, chunk size, chunk
    overlap. Any mismatch raises ``CollectionConfigMismatchError`` listing
    every field that drifted with both stored and current values.

    Random sampling matters: a partial re-ingest could leave the
    collection with stale chunks at one end and current chunks at the
    other. Sampling the first N rows in insertion order would
    systematically miss drift on one side of that boundary.

    Raises:
        CollectionConfigMismatchError: If a sampled chunk's stored
            embedding-model or chunking parameters disagree with current
            settings. The collection cannot be auto-repaired — chunks
            built with different settings corrupt retrieval and ranking.
    """
    logger.info("Verifying ChromaDB collection consistency")
    collection = get_or_create_collection(settings.chroma_collection_name)
    count = collection.count()
    if count == 0:
        return

    all_ids: list[str] = collection.get(include=[]).get("ids") or []
    if not all_ids:
        return
    sample_size = min(_MODEL_VERIFY_SAMPLE_SIZE, len(all_ids))
    sampled_ids = random.sample(all_ids, sample_size)

    results = collection.get(ids=sampled_ids, include=["metadatas"])
    metadatas: list[Any] = results.get("metadatas") or []

    current_model = _current_embedding_model()
    for meta in metadatas:
        if meta is None:
            continue
        drift = _detect_drift(meta, current_model)
        if drift:
            raise CollectionConfigMismatchError(_format_drift_error(drift))


def _detect_drift(
    meta: dict[str, Any], current_model: str
) -> list[_DriftEntry]:
    """Return drift entries ``(field, stored, current)`` for a single chunk.

    Compares the four fields that must be identical across every chunk in
    a healthy collection: embedding_model, chunking_strategy, chunk_size,
    chunk_overlap. A chunk with no value for one of these fields is
    treated as drifted (legacy chunks written before the field existed
    cannot be reconciled with current settings either).
    """
    checks: list[_DriftEntry] = [
        ("embedding_model", meta.get("embedding_model"), current_model),
        (
            "chunking_strategy",
            meta.get("chunking_strategy"),
            settings.chunking_strategy,
        ),
        ("chunk_size", meta.get("chunk_size"), settings.chunk_size),
        ("chunk_overlap", meta.get("chunk_overlap"), settings.chunk_overlap),
    ]
    return [
        (field, stored, current)
        for field, stored, current in checks
        if stored != current
    ]


def _format_drift_error(drift: list[_DriftEntry]) -> str:
    """Compose a remediation-rich error message from one chunk's drift entries."""
    field_lines = "\n".join(
        f"  - {field}: stored={stored!r}, current={current!r}"
        for field, stored, current in drift
    )
    return (
        f"ChromaDB collection '{settings.chroma_collection_name}' contains "
        f"chunks built with different settings than the current configuration. "
        f"Drifted fields:\n"
        f"{field_lines}\n"
        f"Automatic repair is not possible. To resolve:\n"
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

    Raises:
        ValueError: If chunks, embeddings, and metadatas have different
            lengths. ChromaDB would otherwise raise a less helpful
            error mid-write, leaving the collection in an
            indeterminate state. Failing fast at the boundary protects
            the invariant that every chunk has exactly one embedding
            and one metadata record.
    """
    if not (len(chunks) == len(embeddings) == len(metadatas)):
        raise ValueError(
            f"upsert_chunks length mismatch for page '{page_title}': "
            f"{len(chunks)} chunks, {len(embeddings)} embeddings, "
            f"{len(metadatas)} metadatas. All three must agree."
        )
    collection = get_or_create_collection(settings.chroma_collection_name)
    ids = [f"{page_title}::{meta.chunk_index}" for meta in metadatas]
    meta_dicts = [meta.model_dump() for meta in metadatas]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,  # type: ignore[arg-type]
        metadatas=meta_dicts,  # type: ignore[arg-type]
    )


def delete_chunks_by_source(page_title: str) -> None:
    """Delete all chunks associated with a specific wiki page (no-op if absent)."""
    collection = get_or_create_collection(settings.chroma_collection_name)
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
    # ChromaDB may return None for legacy unschemed rows.
    pairs.sort(key=lambda p: (p["metadata"] or {}).get("chunk_index", 0))
    return pairs
