"""Tests for the embedding model version guard.

Verifies that verify_collection_embedding_model correctly raises
EmbeddingModelMismatchError when the ChromaDB collection contains
chunks from a different model, and passes silently when models match.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from vectorstore.client import (
    EmbeddingModelMismatchError,
    verify_collection_embedding_model,
)


def _make_collection(chunks: list[tuple[str, str]]) -> MagicMock:
    """Build a MagicMock ChromaDB collection from (id, embedding_model) pairs.

    Mirrors the two-call pattern used by verify_collection_embedding_model:

    1. ``collection.get(include=[])`` returns every ID (no metadata).
    2. ``collection.get(ids=[...], include=["metadatas"])`` returns
       metadata for only the requested IDs.

    A third branch handles a hypothetical ``limit=N`` regression — if the
    implementation reverts to insertion-order sampling, this branch
    returns the first N chunks so tests can detect that the random-sample
    code path was bypassed.

    Args:
        chunks: List of (chunk_id, embedding_model) pairs in insertion
            order.

    Returns:
        MagicMock that responds to ``count``, ``get`` with realistic
        results derived from the input chunks.
    """
    by_id = dict(chunks)
    all_ids = [chunk_id for chunk_id, _ in chunks]

    collection = MagicMock()
    collection.count.return_value = len(chunks)

    def fake_get(**kwargs: Any) -> dict[str, Any]:
        ids = kwargs.get("ids")
        if ids is not None:
            return {
                "ids": list(ids),
                "documents": None,
                "metadatas": [{"embedding_model": by_id[i]} for i in ids],
            }
        limit = kwargs.get("limit")
        if limit is not None:
            # Insertion-order slice — only used to detect a regression
            # to the pre-random-sampling behaviour.
            items = chunks[:limit]
            return {
                "ids": [c[0] for c in items],
                "documents": None,
                "metadatas": [{"embedding_model": c[1]} for c in items],
            }
        # IDs-only fetch (include=[]). Metadata key is unused by callers
        # in this branch; returning None matches Chroma's actual shape.
        return {"ids": list(all_ids), "documents": None, "metadatas": None}

    collection.get.side_effect = fake_get
    return collection


def test_guard_raises_on_model_mismatch(mocker: Any) -> None:
    """Raise EmbeddingModelMismatchError when stored model differs from settings.

    The guard samples chunks from ChromaDB and detects that the stored
    embedding_model is "nomic-embed-text:v1.5" while current settings
    specify "text-embedding-3-small:3".
    """
    mocker.patch("config.settings.settings.embedding_model", "text-embedding-3-small")
    mocker.patch("config.settings.settings.embedding_model_version", "3")

    fake_collection = _make_collection([("page::0", "nomic-embed-text:v1.5")])
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    with pytest.raises(EmbeddingModelMismatchError) as exc_info:
        verify_collection_embedding_model()

    error_message = str(exc_info.value)
    assert "nomic-embed-text:v1.5" in error_message, (
        "Error message should identify the stored model"
    )
    assert "text-embedding-3-small:3" in error_message, (
        "Error message should identify the current model"
    )
    assert "hkia_v2" in error_message or "new version" in error_message, (
        "Error message should mention creating a new collection version"
    )


def test_guard_passes_on_model_match(mocker: Any) -> None:
    """No exception is raised when the stored model matches current settings.

    The guard samples chunks and finds that the stored embedding_model
    matches the currently configured model exactly.
    """
    mocker.patch("config.settings.settings.embedding_model", "nomic-embed-text")
    mocker.patch("config.settings.settings.embedding_model_version", "v1.5")

    fake_collection = _make_collection([("page::0", "nomic-embed-text:v1.5")])
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    verify_collection_embedding_model()


def test_guard_detects_drift_in_non_leading_chunks(mocker: Any) -> None:
    """Stale chunks past the first N must still be detected.

    Stages a collection of 50 chunks where the first 40 use the current
    model and the last 10 use a stale model. If the implementation
    regressed to insertion-order sampling (limit=10 from the start), it
    would only see current-model chunks and the guard would silently
    pass — letting the agent serve corrupted search results.

    To make the random-sampling path deterministic, ``random.sample`` is
    patched to return the trailing 10 IDs. The patched function also
    records the population it was called with so we can assert that the
    implementation handed it the full ID list (not a pre-sliced prefix).
    """
    mocker.patch("config.settings.settings.embedding_model", "text-embedding-3-small")
    mocker.patch("config.settings.settings.embedding_model_version", "3")

    chunks: list[tuple[str, str]] = [
        (f"page::{i}", "text-embedding-3-small:3") for i in range(40)
    ] + [
        (f"page::{i}", "nomic-embed-text:v1.5") for i in range(40, 50)
    ]
    fake_collection = _make_collection(chunks)
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    sample_calls: list[tuple[list[Any], int]] = []

    def deterministic_sample(population: Any, k: int) -> list[Any]:
        materialised = list(population)
        sample_calls.append((materialised, k))
        return materialised[-k:]

    mocker.patch("vectorstore.client.random.sample", side_effect=deterministic_sample)

    with pytest.raises(EmbeddingModelMismatchError):
        verify_collection_embedding_model()

    assert len(sample_calls) == 1, "random.sample must be called exactly once"
    population, k = sample_calls[0]
    assert len(population) == 50, (
        f"random.sample must be called with the full ID list (50), "
        f"got {len(population)} — implementation may have regressed to "
        f"insertion-order sampling"
    )
    assert k == 10, f"sample size should be 10, got {k}"


def test_guard_no_op_on_empty_collection(mocker: Any) -> None:
    """An empty collection short-circuits before any sampling happens.

    No metadata fetch, no random.sample, no error. Guarantees a fresh
    collection (count == 0) does not raise spuriously on the first ingest.
    """
    fake_collection = _make_collection([])
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )
    sample_spy = mocker.patch("vectorstore.client.random.sample")

    verify_collection_embedding_model()

    sample_spy.assert_not_called()
