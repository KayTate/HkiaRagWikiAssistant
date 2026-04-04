"""Embedding providers for the HKIA ingestion pipeline.

Abstracts over Ollama (single-item) and OpenAI (batched) embedding APIs.
Both providers retry on transient failures via tenacity.
"""

import logging

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails after all retries are exhausted."""


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of text chunks using the configured embedding provider.

    Dispatches to the Ollama or OpenAI implementation based on
    settings.embedding_provider. The output list is always the same length
    as the input, with one embedding vector per chunk, in the same order.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of embedding vectors, one per input chunk.

    Raises:
        EmbeddingError: If the provider fails after retries are exhausted.
    """
    if settings.embedding_provider == "openai":
        return openai_embed(chunks)
    return ollama_embed(chunks)


_OLLAMA_EMBED_BATCH_SIZE = 50


def ollama_embed(chunks: list[str]) -> list[list[float]]:
    """Embed chunks using Ollama's batch /api/embed endpoint.

    Splits chunks into sub-batches to avoid memory and timeout issues
    on very large pages. Each batch is sent as a single request
    containing all chunk texts.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of embedding vectors in the same order as the input.

    Raises:
        EmbeddingError: If any batch fails to embed after retries.
    """
    all_embeddings: list[list[float]] = []
    for start in range(0, len(chunks), _OLLAMA_EMBED_BATCH_SIZE):
        batch = chunks[start : start + _OLLAMA_EMBED_BATCH_SIZE]
        batch_embeddings = _embed_batch_ollama(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch_ollama(chunks: list[str]) -> list[list[float]]:
    """Send a batched embedding request to the Ollama /api/embed endpoint.

    Args:
        chunks: Batch of text strings to embed (max 50).

    Returns:
        Embedding vectors in the same order as the input batch.

    Raises:
        EmbeddingError: If the response structure is unexpected.
    """
    url = "http://localhost:11434/api/embed"
    try:
        response = requests.post(
            url,
            json={"model": settings.embedding_model, "input": chunks},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings: list[list[float]] = data["embeddings"]
        return embeddings
    except (KeyError, ValueError) as exc:
        raise EmbeddingError(
            f"Ollama batch embedding response missing 'embeddings' field: {exc}"
        ) from exc
    except requests.RequestException as exc:
        logger.warning(
            "Ollama batch embedding request failed, will retry: %s", exc
        )
        raise


def openai_embed(chunks: list[str]) -> list[list[float]]:
    """Embed chunks in batches using the OpenAI embeddings API.

    Splits the input into batches of settings.openai_embedding_batch_size
    to stay within API limits. Retries on HTTP 429 (rate limit) with
    exponential backoff.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of embedding vectors in the same order as the input.

    Raises:
        EmbeddingError: If the API fails after retries are exhausted.
    """
    if not settings.openai_api_key:
        raise EmbeddingError(
            "OPENAI_API_KEY is not configured but embedding_provider is 'openai'."
        )

    batch_size = settings.openai_embedding_batch_size
    all_embeddings: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_embeddings = _embed_batch_openai(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch_openai(batch: list[str]) -> list[list[float]]:
    """Send a single batched embedding request to the OpenAI API.

    Retries automatically on HTTP 429 via tenacity. Other HTTP errors
    are raised immediately after the first failure.

    Args:
        batch: Batch of text strings to embed.

    Returns:
        Embedding vectors in the same order as the input batch.

    Raises:
        EmbeddingError: If the response structure is unexpected.
        requests.HTTPError: On non-transient HTTP errors.
    """
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={"input": batch, "model": settings.embedding_model},
            timeout=120,
        )
        if response.status_code == 429:
            logger.warning("OpenAI rate limit hit, will retry with backoff")
            response.raise_for_status()
        response.raise_for_status()
        # response.json() returns Any; API response shape is external/untyped
        data = response.json()
        sorted_items = sorted(data["data"], key=lambda item: int(item["index"]))
        return [[float(x) for x in item["embedding"]] for item in sorted_items]
    except (KeyError, ValueError) as exc:
        raise EmbeddingError(
            f"OpenAI embedding response has unexpected structure: {exc}"
        ) from exc
    except requests.HTTPError:
        raise
    except requests.RequestException as exc:
        raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc
