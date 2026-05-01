"""Tests for vectorstore schema validators and client-level guards.

Locks in three small hardening changes that protect the boundary
between agent/ingestion code and ChromaDB:

- ``ChunkMetadata`` rejects nonsensical numeric values at construction
  rather than letting them propagate into stored metadata where they
  would silently corrupt downstream queries.
- ``upsert_chunks`` fails fast on a length mismatch among chunks /
  embeddings / metadatas — ChromaDB would otherwise raise a less
  helpful error mid-write.
- ``get_page_by_title`` survives a row whose metadata column is
  ``None`` (legacy data from before metadata was schema-enforced).
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from vectorstore.client import get_page_by_title, upsert_chunks
from vectorstore.schema import ChunkMetadata


def _valid_metadata(**overrides: Any) -> ChunkMetadata:
    """Build a ChunkMetadata with all required fields populated.

    Helpers here let each constraint test override exactly the field
    under test so a regression in an unrelated field cannot mask a
    real validation failure.
    """
    base: dict[str, Any] = {
        "source_title": "Test Page",
        "source_url": "https://example.org/wiki/Test_Page",
        "section": "",
        "category": "",
        "chunk_index": 0,
        "revision_id": 1,
        "ingested_at": "2026-01-01T00:00:00Z",
        "embedding_model": "nomic-embed-text:v1.5",
        "chunking_strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
    }
    base.update(overrides)
    return ChunkMetadata(**base)


# ---------------------------------------------------------------------------
# ChunkMetadata constraints
# ---------------------------------------------------------------------------


def test_chunk_metadata_accepts_valid_values() -> None:
    """The valid-baseline must construct cleanly so failure tests are meaningful."""
    meta = _valid_metadata()
    assert meta.chunk_size == 512
    assert meta.chunk_overlap == 64
    assert meta.chunk_index == 0


def test_chunk_metadata_rejects_zero_chunk_size() -> None:
    """chunk_size must be strictly positive — a zero-size chunk is meaningless."""
    with pytest.raises(ValidationError):
        _valid_metadata(chunk_size=0)


def test_chunk_metadata_rejects_negative_chunk_size() -> None:
    """Negative chunk_size always indicates a programming error."""
    with pytest.raises(ValidationError):
        _valid_metadata(chunk_size=-1)


def test_chunk_metadata_rejects_negative_chunk_index() -> None:
    """chunk_index is a 0-based position; negative values are nonsensical."""
    with pytest.raises(ValidationError):
        _valid_metadata(chunk_index=-1)


def test_chunk_metadata_accepts_zero_chunk_overlap() -> None:
    """Zero overlap is the normal config when chunks must not share content."""
    meta = _valid_metadata(chunk_overlap=0)
    assert meta.chunk_overlap == 0


def test_chunk_metadata_rejects_negative_chunk_overlap() -> None:
    """Negative overlap is meaningless and would corrupt offset math."""
    with pytest.raises(ValidationError):
        _valid_metadata(chunk_overlap=-1)


def test_chunk_metadata_allows_revision_id_minus_one_sentinel() -> None:
    """revision_id == -1 is the sentinel for stale-row resets.

    ingestion.pipeline._repair_stale_sqlite_rows uses -1 when an
    embedding-model change forces a re-fetch. The schema must permit
    it; tightening to ge=0 would break the startup sync check.
    """
    meta = _valid_metadata(revision_id=-1)
    assert meta.revision_id == -1


def test_chunk_metadata_rejects_revision_id_below_sentinel() -> None:
    """Anything below -1 is not a meaningful sentinel or a real revision."""
    with pytest.raises(ValidationError):
        _valid_metadata(revision_id=-2)


# ---------------------------------------------------------------------------
# upsert_chunks length parity
# ---------------------------------------------------------------------------


def test_upsert_chunks_raises_on_length_mismatch_chunks_vs_embeddings(
    mocker: Any,
) -> None:
    """Differing chunk and embedding counts must surface immediately.

    The motivating concern: ChromaDB would raise mid-write with a
    less informative error, leaving the collection in an indeterminate
    state. Failing fast at the boundary means callers learn the
    invariant violation in their own stack frame.
    """
    mocker.patch(
        "vectorstore.client.get_or_create_collection", return_value=MagicMock()
    )
    metadatas = [_valid_metadata(chunk_index=0)]

    with pytest.raises(ValueError, match="length mismatch"):
        upsert_chunks(
            page_title="Test",
            chunks=["one", "two"],
            embeddings=[[0.0]],
            metadatas=metadatas,
        )


def test_upsert_chunks_raises_on_length_mismatch_embeddings_vs_metadatas(
    mocker: Any,
) -> None:
    """Differing embedding and metadata counts must also surface."""
    mocker.patch(
        "vectorstore.client.get_or_create_collection", return_value=MagicMock()
    )
    metadatas = [
        _valid_metadata(chunk_index=0),
        _valid_metadata(chunk_index=1),
    ]

    with pytest.raises(ValueError, match="length mismatch"):
        upsert_chunks(
            page_title="Test",
            chunks=["one", "two"],
            embeddings=[[0.0]],
            metadatas=metadatas,
        )


def test_upsert_chunks_proceeds_when_lengths_agree(mocker: Any) -> None:
    """Aligned lengths must reach the ChromaDB upsert call.

    Pairs with the parity tests above: without this, a regression
    that always raises ValueError would still pass those.
    """
    fake_collection = MagicMock()
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    upsert_chunks(
        page_title="Test",
        chunks=["one", "two"],
        embeddings=[[0.0], [0.0]],
        metadatas=[
            _valid_metadata(chunk_index=0),
            _valid_metadata(chunk_index=1),
        ],
    )

    fake_collection.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# get_page_by_title — None metadata at the ChromaDB boundary
# ---------------------------------------------------------------------------


def test_get_page_by_title_handles_none_metadata_in_sort(mocker: Any) -> None:
    """A row with None metadata must not crash the sort key.

    Real-world cause: legacy chunks ingested before the schema was
    enforced. The sort key used to be ``p["metadata"].get(...)`` which
    AttributeErrors on None. The defensive ``(meta or {}).get(...)``
    coerces gracefully — a single legacy row should not crash the
    page-load path for an entire page.
    """
    fake_collection = MagicMock()
    fake_collection.get.return_value = {
        "documents": ["second", "first", "third"],
        "metadatas": [
            {"chunk_index": 1, "source_title": "Test"},
            None,  # legacy row, no metadata
            {"chunk_index": 2, "source_title": "Test"},
        ],
    }
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    pairs = get_page_by_title("Test")

    # The sort must succeed and place the None-metadata row first
    # (its effective chunk_index is 0 via the defensive default).
    assert len(pairs) == 3
    assert pairs[0]["metadata"] is None
    assert pairs[1]["metadata"]["chunk_index"] == 1
    assert pairs[2]["metadata"]["chunk_index"] == 2
