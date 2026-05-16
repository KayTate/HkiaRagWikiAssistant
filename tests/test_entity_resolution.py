"""Tests for entity extraction, title variant resolution, and opensearch fallback.

All external calls (ChromaDB, MediaWiki API, embeddings) are mocked so these
tests run fully in-process with no external dependencies.
"""

from typing import Any

import pytest
import requests

from agent.extraction import _normalize_entity
from agent.retrieval import (
    _fetch_entity_chunks,
    _load_redirects,
    _resolve_via_redirect,
    _title_candidates,
)


@pytest.fixture(autouse=True)
def _reset_redirects_cache() -> None:
    """Clear the process-lifetime redirect cache so tests are order-independent."""
    _load_redirects.cache_clear()


def test_normalize_entity_strips_article_and_descriptor() -> None:
    """Leading articles and trailing descriptors must be stripped."""
    assert _normalize_entity("the Wild Mountain Time quest") == "Wild Mountain Time"
    assert _normalize_entity("an Ice and Glow quest series") == "Ice and Glow"
    assert _normalize_entity("Mystery Tree") == "Mystery Tree"
    assert _normalize_entity("") == ""


def test_title_candidates_includes_the_prefix_variants() -> None:
    """Candidates must include both bare and 'The '-prefixed forms plus suffixes."""
    candidates = _title_candidates("Mystery Tree")
    assert "Mystery Tree" in candidates
    assert "The Mystery Tree" in candidates
    assert "The Mystery Tree (quest series)" in candidates

    # Already-prefixed entity should not double up.
    the_prefixed = _title_candidates("The Mystery Tree")
    assert "The The Mystery Tree" not in the_prefixed
    assert "The Mystery Tree" in the_prefixed


def test_fetch_entity_chunks_resolves_via_suffix(mocker: Any) -> None:
    """Disambiguation-suffix variants resolve before falling through to opensearch."""
    target_chunks: list[dict[str, Any]] = [
        {
            "text": "Ice and Glow quest series content.",
            "metadata": {
                "source_title": "Ice and Glow (quest series)",
                "chunk_index": 0,
            },
        }
    ]

    def fake_get_page(title: str) -> list[dict[str, Any]]:
        if title == "Ice and Glow (quest series)":
            return target_chunks
        return []

    mocker.patch("agent.retrieval.vs_get_page_by_title", side_effect=fake_get_page)
    opensearch_mock = mocker.patch(
        "agent.retrieval._resolve_title_via_opensearch", return_value=None
    )
    embed_mock = mocker.patch("agent.retrieval.embed_chunks")
    search_mock = mocker.patch("agent.retrieval.vs_semantic_search")

    result = _fetch_entity_chunks("Ice and Glow", "question text")

    assert result == target_chunks
    opensearch_mock.assert_not_called()
    embed_mock.assert_not_called()
    search_mock.assert_not_called()


def test_fetch_entity_chunks_resolves_via_the_prefix(mocker: Any) -> None:
    """'The '-prefixed variants with a suffix should resolve."""
    target_chunks: list[dict[str, Any]] = [
        {
            "text": "Mystery Tree quest series content.",
            "metadata": {
                "source_title": "The Mystery Tree (quest series)",
                "chunk_index": 0,
            },
        }
    ]

    def fake_get_page(title: str) -> list[dict[str, Any]]:
        if title == "The Mystery Tree (quest series)":
            return target_chunks
        return []

    mocker.patch("agent.retrieval.vs_get_page_by_title", side_effect=fake_get_page)
    mocker.patch("agent.retrieval._resolve_title_via_opensearch", return_value=None)
    mocker.patch("agent.retrieval.embed_chunks")
    mocker.patch("agent.retrieval.vs_semantic_search")

    result = _fetch_entity_chunks("Mystery Tree", "How do I unlock The Mystery Tree?")

    assert result == target_chunks


def test_fetch_entity_chunks_uses_opensearch_when_variants_fail(mocker: Any) -> None:
    """When every title variant misses, opensearch resolution should win."""
    opensearch_chunks: list[dict[str, Any]] = [
        {
            "text": "Apple Orchard location content.",
            "metadata": {"source_title": "Apple Orchard", "chunk_index": 0},
        }
    ]

    def fake_get_page(title: str) -> list[dict[str, Any]]:
        if title == "Apple Orchard":
            return opensearch_chunks
        return []

    mocker.patch("agent.retrieval.vs_get_page_by_title", side_effect=fake_get_page)
    mocker.patch(
        "agent.retrieval._resolve_title_via_opensearch",
        return_value="Apple Orchard",
    )
    embed_mock = mocker.patch("agent.retrieval.embed_chunks")
    search_mock = mocker.patch("agent.retrieval.vs_semantic_search")

    result = _fetch_entity_chunks("apple tree", "where is the apple tree?")

    assert result == opensearch_chunks
    embed_mock.assert_not_called()
    search_mock.assert_not_called()


def test_fetch_entity_chunks_semantic_search_uses_full_question(mocker: Any) -> None:
    """Semantic fallback must embed the full question, not the bare entity."""
    mocker.patch("agent.retrieval.vs_get_page_by_title", return_value=[])
    mocker.patch("agent.retrieval._resolve_title_via_opensearch", return_value=None)
    embed_mock = mocker.patch(
        "agent.retrieval.embed_chunks", return_value=[[0.1] * 384]
    )
    search_mock = mocker.patch(
        "agent.retrieval.vs_semantic_search",
        return_value=[{"text": "fallback", "metadata": {}}],
    )

    question = "How do I unlock the Mystery Tree quest?"
    result = _fetch_entity_chunks("Mystery Tree", question)

    embed_mock.assert_called_once_with([question])
    search_mock.assert_called_once()
    assert result == [{"text": "fallback", "metadata": {}}]


def test_fetch_entity_chunks_follows_redirect_table_for_direct_title(
    mocker: Any,
) -> None:
    """A title-variant hit must be transformed via the redirect side-table.

    Redirect pages are skipped at ingest time and replaced with a
    source→target mapping in SQLite. Agent retrieval must consult that
    mapping before each ``vs_get_page_by_title`` call so a query for the
    source title still resolves to the target page's chunks. Without
    this lookup the agent would return an empty result for every page
    that used to be a redirect.
    """
    final_chunks: list[dict[str, Any]] = [
        {
            "text": "Apple Orchard is a location.",
            "metadata": {"source_title": "Apple Orchard", "chunk_index": 0},
        }
    ]

    def fake_get_page(title: str) -> list[dict[str, Any]]:
        if title == "Apple Orchard":
            return final_chunks
        return []

    mocker.patch(
        "agent.retrieval.get_all_redirects",
        return_value={"Apple Tree": "Apple Orchard"},
    )
    get_page_mock = mocker.patch(
        "agent.retrieval.vs_get_page_by_title", side_effect=fake_get_page
    )
    opensearch_mock = mocker.patch(
        "agent.retrieval._resolve_title_via_opensearch", return_value=None
    )
    embed_mock = mocker.patch("agent.retrieval.embed_chunks")
    search_mock = mocker.patch("agent.retrieval.vs_semantic_search")

    result = _fetch_entity_chunks("Apple Tree", "where do apples grow?")

    assert result == final_chunks
    # The side-table lookup must have happened on the candidate, NOT the
    # raw entity, and the resulting fetch must use the target title.
    get_page_mock.assert_any_call("Apple Orchard")
    # Redirect-table resolution wins before opensearch or semantic search.
    opensearch_mock.assert_not_called()
    embed_mock.assert_not_called()
    search_mock.assert_not_called()


def test_fetch_entity_chunks_follows_redirect_table_after_opensearch(
    mocker: Any,
) -> None:
    """Opensearch results must also flow through the redirect side-table.

    Even when MediaWiki opensearch resolves a free-text query to a wiki
    title, that title may itself be a redirect source. The post-
    opensearch lookup must run through ``_resolve_via_redirect`` so the
    chunks fetched belong to the final canonical page, not the redirect
    source (whose chunks no longer exist after re-ingest).
    """
    final_chunks: list[dict[str, Any]] = [
        {
            "text": "Woodblock is a crafting material.",
            "metadata": {"source_title": "Woodblock", "chunk_index": 0},
        }
    ]

    def fake_get_page(title: str) -> list[dict[str, Any]]:
        if title == "Woodblock":
            return final_chunks
        return []

    mocker.patch(
        "agent.retrieval.get_all_redirects",
        return_value={"Wooden Block": "Woodblock"},
    )
    mocker.patch("agent.retrieval.vs_get_page_by_title", side_effect=fake_get_page)
    mocker.patch(
        "agent.retrieval._resolve_title_via_opensearch",
        return_value="Wooden Block",
    )
    mocker.patch("agent.retrieval.embed_chunks")
    mocker.patch("agent.retrieval.vs_semantic_search")

    result = _fetch_entity_chunks("wood block", "what is a wood block?")

    assert result == final_chunks


def test_resolve_via_redirect_caches_map_after_first_load(mocker: Any) -> None:
    """The redirects map must be loaded from SQLite at most once per process.

    The cache keeps every agent request from re-querying the same ~5k
    rows. A regression that reloaded on every call would re-introduce
    per-query SQLite latency on the hot path.
    """
    get_all_redirects_mock = mocker.patch(
        "agent.retrieval.get_all_redirects",
        return_value={"Apple Tree": "Apple Orchard"},
    )

    _resolve_via_redirect("Apple Tree")
    _resolve_via_redirect("Apple Tree")
    _resolve_via_redirect("Something Else")

    assert get_all_redirects_mock.call_count == 1


def test_opensearch_failure_does_not_raise(mocker: Any) -> None:
    """Raising from opensearch_title must not propagate; semantic fallback runs."""
    mocker.patch("agent.retrieval.vs_get_page_by_title", return_value=[])
    mocker.patch(
        "ingestion.api_client.opensearch_title",
        side_effect=requests.ConnectionError("boom"),
    )
    embed_mock = mocker.patch(
        "agent.retrieval.embed_chunks", return_value=[[0.1] * 384]
    )
    search_mock = mocker.patch(
        "agent.retrieval.vs_semantic_search",
        return_value=[{"text": "fallback", "metadata": {}}],
    )

    result = _fetch_entity_chunks("anything", "question")

    assert result == [{"text": "fallback", "metadata": {}}]
    embed_mock.assert_called_once()
    search_mock.assert_called_once()


