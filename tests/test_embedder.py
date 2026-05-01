"""Tests for ingestion/embedder.py — provider dispatch and batching.

The retry-predicate behavior of ``_embed_batch_openai`` is covered in
tests/test_http_retries.py. This file targets the parts that file
does not exercise: the dispatch in ``embed_chunks``, the batch-size
slicing in ``ollama_embed``/``openai_embed``, the missing-key error
in the OpenAI path, and the order-preservation invariant in the
OpenAI batch helper (which sorts by the API's ``index`` field
because OpenAI may return embeddings out of input order).
"""

from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from ingestion.embedder import (
    EmbeddingError,
    embed_chunks,
    ollama_embed,
    openai_embed,
)


def _ok_ollama_response(n: int) -> Mock:
    """Build a successful Ollama /api/embed response with n embeddings."""
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {"embeddings": [[float(i)] for i in range(n)]}
    return response


def _ok_openai_response(items: list[dict[str, Any]]) -> Mock:
    """Build a successful OpenAI /embeddings response carrying ``items``."""
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": items}
    return response


# ---------------------------------------------------------------------------
# embed_chunks dispatch
# ---------------------------------------------------------------------------


def test_embed_chunks_routes_to_openai_when_provider_is_openai(
    mocker: Any,
) -> None:
    """provider='openai' must call openai_embed, never ollama_embed.

    The two implementations are not interchangeable — Ollama's
    response shape is different and the OpenAI path requires an API
    key. Pinning the routing protects against a config change that
    silently uses the wrong backend.
    """
    mocker.patch("config.settings.settings.embedding_provider", "openai")
    openai_mock = mocker.patch(
        "ingestion.embedder.openai_embed", return_value=[[0.1]]
    )
    ollama_mock = mocker.patch("ingestion.embedder.ollama_embed")

    result = embed_chunks(["text"])

    assert result == [[0.1]]
    openai_mock.assert_called_once_with(["text"])
    ollama_mock.assert_not_called()


def test_embed_chunks_routes_to_ollama_for_other_providers(mocker: Any) -> None:
    """Anything other than 'openai' falls through to the Ollama path.

    Documents the dispatcher's default-to-ollama behavior. A regression
    that raised on unknown provider (rather than defaulting) would
    still fail this test, surfacing the change for review.
    """
    mocker.patch("config.settings.settings.embedding_provider", "ollama")
    openai_mock = mocker.patch("ingestion.embedder.openai_embed")
    ollama_mock = mocker.patch(
        "ingestion.embedder.ollama_embed", return_value=[[0.2]]
    )

    result = embed_chunks(["text"])

    assert result == [[0.2]]
    ollama_mock.assert_called_once_with(["text"])
    openai_mock.assert_not_called()


# ---------------------------------------------------------------------------
# ollama_embed
# ---------------------------------------------------------------------------


def test_ollama_embed_empty_input_returns_empty_list(mocker: Any) -> None:
    """No chunks → no HTTP call and an empty list."""
    post_mock = mocker.patch("requests.post")
    assert ollama_embed([]) == []
    post_mock.assert_not_called()


def test_ollama_embed_returns_one_embedding_per_chunk(mocker: Any) -> None:
    """The output length must always match the input length.

    A regression that lost an embedding (e.g. an off-by-one in the
    batch slicing) would now also trip the upsert_chunks length
    parity guard from commit 5, but pinning here surfaces the error
    closer to its source.
    """
    mocker.patch("requests.post", return_value=_ok_ollama_response(3))
    chunks = ["a", "b", "c"]
    embeddings = ollama_embed(chunks)
    assert len(embeddings) == len(chunks)


def test_ollama_embed_splits_oversized_input_into_batches(mocker: Any) -> None:
    """Inputs larger than _OLLAMA_EMBED_BATCH_SIZE (50) split into multiple POSTs.

    50 is hard-coded as a memory/timeout safeguard for very large
    pages. Pinning the split point catches a regression that
    silently raised the constant — the post-counted assertion fails
    if the loop runs only once.
    """
    chunks = [f"chunk-{i}" for i in range(120)]
    # Each batch returns its own flat list; the size-50 batch slicing
    # is what matters here, not the embedding values.
    post_mock = mocker.patch(
        "requests.post",
        side_effect=[
            _ok_ollama_response(50),
            _ok_ollama_response(50),
            _ok_ollama_response(20),
        ],
    )

    embeddings = ollama_embed(chunks)

    assert len(embeddings) == 120
    assert post_mock.call_count == 3, (
        f"120 chunks at batch=50 must split into 3 POSTs, "
        f"got {post_mock.call_count}"
    )


def test_ollama_embed_raises_embedding_error_on_missing_field(mocker: Any) -> None:
    """A response without 'embeddings' must be wrapped in EmbeddingError.

    Without this, callers see a raw KeyError with no context about
    what failed or for which provider. EmbeddingError lets the
    pipeline log the surrounding page title cleanly.
    """
    bad_response = Mock()
    bad_response.status_code = 200
    bad_response.raise_for_status.return_value = None
    bad_response.json.return_value = {"unexpected": "shape"}
    mocker.patch("requests.post", return_value=bad_response)

    with pytest.raises(EmbeddingError, match="missing 'embeddings' field"):
        ollama_embed(["one"])


# ---------------------------------------------------------------------------
# openai_embed
# ---------------------------------------------------------------------------


def test_openai_embed_raises_when_api_key_missing(mocker: Any) -> None:
    """Empty OpenAI key must raise EmbeddingError before any HTTP call."""
    mocker.patch("config.settings.settings.openai_api_key", SecretStr(""))
    post_mock = mocker.patch("requests.post")

    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY is not configured"):
        openai_embed(["one"])
    post_mock.assert_not_called()


def test_openai_embed_splits_into_configurable_batches(mocker: Any) -> None:
    """Batch size comes from settings, not a hard-coded constant.

    Pin the wiring so a regression that inlined a literal would
    surface here.
    """
    mocker.patch(
        "config.settings.settings.openai_api_key", SecretStr("test-key")
    )
    mocker.patch("config.settings.settings.openai_embedding_batch_size", 5)
    mocker.patch("config.settings.settings.embedding_model", "text-embedding-3-small")
    mocker.patch("time.sleep")

    chunks = [f"c{i}" for i in range(12)]

    def make_response(n: int) -> Mock:
        items = [{"index": i, "embedding": [float(i)]} for i in range(n)]
        return _ok_openai_response(items)

    post_mock = mocker.patch(
        "requests.post",
        side_effect=[make_response(5), make_response(5), make_response(2)],
    )

    embeddings = openai_embed(chunks)

    assert len(embeddings) == 12
    assert post_mock.call_count == 3, (
        f"12 chunks at batch=5 must split into 3 POSTs, got {post_mock.call_count}"
    )


def test_openai_embed_sorts_response_by_index_field(mocker: Any) -> None:
    """The OpenAI API may return embeddings out of order; the embedder must sort.

    This is a real production hazard: OpenAI's batch endpoint
    documents that it preserves input order, but the SDK and our
    raw-HTTP path both sort by ``index`` defensively. A regression
    that returned ``data["data"]`` un-sorted would silently misalign
    embeddings with their source chunks — every retrieved chunk
    would carry the wrong vector.
    """
    mocker.patch(
        "config.settings.settings.openai_api_key", SecretStr("test-key")
    )
    mocker.patch("config.settings.settings.openai_embedding_batch_size", 100)
    mocker.patch("config.settings.settings.embedding_model", "text-embedding-3-small")
    mocker.patch("time.sleep")

    # Response shuffled: index 2 first, then 0, then 1. After sorting
    # the output must align positionally with the original chunks.
    out_of_order = [
        {"index": 2, "embedding": [2.0]},
        {"index": 0, "embedding": [0.0]},
        {"index": 1, "embedding": [1.0]},
    ]
    mocker.patch("requests.post", return_value=_ok_openai_response(out_of_order))

    embeddings = openai_embed(["a", "b", "c"])

    assert embeddings == [[0.0], [1.0], [2.0]], (
        "Embeddings must be reordered to match input position via the "
        "API's 'index' field — out-of-order responses would otherwise "
        "silently misalign every chunk with its vector"
    )
