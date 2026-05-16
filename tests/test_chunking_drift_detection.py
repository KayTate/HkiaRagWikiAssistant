"""Tests for chunking-parameter drift in verify_collection_consistency.

Embedding-model drift is covered in tests/test_embedding_version_guard.py.
This file pins the chunking half of the same guard: changing
``chunking_strategy``, ``chunk_size``, or ``chunk_overlap`` and pointing at
an existing collection must raise rather than silently mixing chunks built
with different settings. Each parameter has its own test so a regression
that misses one is attributed to the right field at the failure site.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from vectorstore.client import (
    CollectionConfigMismatchError,
    verify_collection_consistency,
)


def _make_collection_with_meta(
    chunks: list[tuple[str, dict[str, Any]]],
) -> MagicMock:
    """Build a MagicMock ChromaDB collection from (id, metadata) pairs.

    Mirrors the two-call shape used by verify_collection_consistency:
    ``collection.get(include=[])`` returns every ID, then
    ``collection.get(ids=[...], include=["metadatas"])`` returns metadata
    for only the requested IDs.

    Args:
        chunks: List of ``(chunk_id, metadata_dict)`` pairs in insertion
            order. Each metadata dict must include every field the guard
            inspects (``embedding_model``, ``chunking_strategy``,
            ``chunk_size``, ``chunk_overlap``) so the only drift comes
            from the field the caller deliberately mismatched.

    Returns:
        MagicMock that responds to ``count`` and ``get`` with results
        derived from the input chunks.
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
                "metadatas": [by_id[i] for i in ids],
            }
        return {"ids": list(all_ids), "documents": None, "metadatas": None}

    collection.get.side_effect = fake_get
    return collection


def _pin_current_settings(mocker: Any) -> None:
    """Pin settings to a single baseline so each test changes one field at a time."""
    mocker.patch("config.settings.settings.embedding_model", "nomic-embed-text")
    mocker.patch("config.settings.settings.embedding_model_version", "v1.5")
    mocker.patch("config.settings.settings.chunking_strategy", "recursive")
    mocker.patch("config.settings.settings.chunk_size", 512)
    mocker.patch("config.settings.settings.chunk_overlap", 64)


def test_drift_detects_chunking_strategy_mismatch(mocker: Any) -> None:
    """Stored ``chunking_strategy='section'`` vs current ``'recursive'`` must raise.

    Section chunks and recursive chunks have different granularity, so
    mixing them in the same collection produces non-uniform retrieval
    quality that's hard to attribute. The guard catches the mismatch
    before any new chunks land in the collection.
    """
    _pin_current_settings(mocker)
    stored_meta = {
        "embedding_model": "nomic-embed-text:v1.5",
        "chunking_strategy": "section",
        "chunk_size": 512,
        "chunk_overlap": 64,
    }
    fake_collection = _make_collection_with_meta([("page::0", stored_meta)])
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    with pytest.raises(CollectionConfigMismatchError) as exc_info:
        verify_collection_consistency()

    message = str(exc_info.value)
    assert "chunking_strategy" in message
    assert "section" in message
    assert "recursive" in message


def test_drift_detects_chunk_size_mismatch(mocker: Any) -> None:
    """Stored ``chunk_size=1024`` vs current ``512`` must raise.

    Chunk size changes the unit of retrieval; mixing sizes within a
    collection means top-k results span both granularities, defeating
    any size-tuning experiment.
    """
    _pin_current_settings(mocker)
    stored_meta = {
        "embedding_model": "nomic-embed-text:v1.5",
        "chunking_strategy": "recursive",
        "chunk_size": 1024,
        "chunk_overlap": 64,
    }
    fake_collection = _make_collection_with_meta([("page::0", stored_meta)])
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    with pytest.raises(CollectionConfigMismatchError) as exc_info:
        verify_collection_consistency()

    message = str(exc_info.value)
    assert "chunk_size" in message
    assert "1024" in message
    assert "512" in message


def test_drift_detects_chunk_overlap_mismatch(mocker: Any) -> None:
    """Stored ``chunk_overlap=128`` vs current ``64`` must raise.

    Overlap is the most subtle of the chunking params — drift here
    silently changes how much context bleeds between adjacent chunks
    and is invisible in retrieval scores. Pinning catches it.
    """
    _pin_current_settings(mocker)
    stored_meta = {
        "embedding_model": "nomic-embed-text:v1.5",
        "chunking_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 128,
    }
    fake_collection = _make_collection_with_meta([("page::0", stored_meta)])
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    with pytest.raises(CollectionConfigMismatchError) as exc_info:
        verify_collection_consistency()

    message = str(exc_info.value)
    assert "chunk_overlap" in message
    assert "128" in message
    assert "64" in message
