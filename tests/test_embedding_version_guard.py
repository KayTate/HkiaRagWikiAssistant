"""Tests for the embedding model version guard.

Verifies that verify_collection_embedding_model correctly raises
EmbeddingModelMismatchError when the ChromaDB collection contains
chunks from a different model, and passes silently when models match.
"""

from typing import Any
from unittest.mock import MagicMock

from vectorstore.client import (
    EmbeddingModelMismatchError,
    verify_collection_embedding_model,
)


def _make_collection_with_model(embedding_model: str) -> MagicMock:
    """Build a MagicMock ChromaDB collection that returns chunks with the given model.

    Args:
        embedding_model: The embedding_model value to embed in returned metadata.

    Returns:
        A MagicMock that mimics the subset of ChromaDB collection API used
        by verify_collection_embedding_model.
    """
    collection = MagicMock()
    collection.count.return_value = 5
    collection.get.return_value = {
        "ids": ["page::0"],
        "documents": ["some chunk text"],
        "metadatas": [{"embedding_model": embedding_model}],
    }
    return collection


def test_guard_raises_on_model_mismatch(mocker: Any) -> None:
    """Raise EmbeddingModelMismatchError when stored model differs from settings.

    The guard samples chunks from ChromaDB and detects that the stored
    embedding_model is "nomic-embed-text:v1.5" while current settings
    specify "text-embedding-3-small:3".
    """
    mocker.patch("config.settings.settings.embedding_model", "text-embedding-3-small")
    mocker.patch("config.settings.settings.embedding_model_version", "3")

    fake_collection = _make_collection_with_model("nomic-embed-text:v1.5")
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    import pytest

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

    fake_collection = _make_collection_with_model("nomic-embed-text:v1.5")
    mocker.patch(
        "vectorstore.client.get_or_create_collection",
        return_value=fake_collection,
    )

    verify_collection_embedding_model()
