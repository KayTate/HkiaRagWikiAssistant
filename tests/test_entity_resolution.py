"""Tests for entity extraction, title variant resolution, and opensearch fallback.

All external calls (ChromaDB, MediaWiki API, embeddings) are mocked so these
tests run fully in-process with no external dependencies.
"""

from typing import Any

import requests

from agent.nodes import (
    _extract_redirect_target,
    _fetch_entity_chunks,
    _normalize_entity,
    _title_candidates,
)


def _redirect_chunks(text: str) -> list[dict[str, Any]]:
    """Build a single-chunk fixture for _extract_redirect_target tests."""
    return [{"text": text, "metadata": {"source_title": "Source", "chunk_index": 0}}]


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

    mocker.patch("agent.nodes.vs_get_page_by_title", side_effect=fake_get_page)
    opensearch_mock = mocker.patch(
        "agent.nodes._resolve_title_via_opensearch", return_value=None
    )
    embed_mock = mocker.patch("agent.nodes.embed_chunks")
    search_mock = mocker.patch("agent.nodes.vs_semantic_search")

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

    mocker.patch("agent.nodes.vs_get_page_by_title", side_effect=fake_get_page)
    mocker.patch("agent.nodes._resolve_title_via_opensearch", return_value=None)
    mocker.patch("agent.nodes.embed_chunks")
    mocker.patch("agent.nodes.vs_semantic_search")

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

    mocker.patch("agent.nodes.vs_get_page_by_title", side_effect=fake_get_page)
    mocker.patch(
        "agent.nodes._resolve_title_via_opensearch",
        return_value="Apple Orchard",
    )
    embed_mock = mocker.patch("agent.nodes.embed_chunks")
    search_mock = mocker.patch("agent.nodes.vs_semantic_search")

    result = _fetch_entity_chunks("apple tree", "where is the apple tree?")

    assert result == opensearch_chunks
    embed_mock.assert_not_called()
    search_mock.assert_not_called()


def test_fetch_entity_chunks_semantic_search_uses_full_question(mocker: Any) -> None:
    """Semantic fallback must embed the full question, not the bare entity."""
    mocker.patch("agent.nodes.vs_get_page_by_title", return_value=[])
    mocker.patch("agent.nodes._resolve_title_via_opensearch", return_value=None)
    embed_mock = mocker.patch("agent.nodes.embed_chunks", return_value=[[0.1] * 384])
    search_mock = mocker.patch(
        "agent.nodes.vs_semantic_search",
        return_value=[{"text": "fallback", "metadata": {}}],
    )

    question = "How do I unlock the Mystery Tree quest?"
    result = _fetch_entity_chunks("Mystery Tree", question)

    embed_mock.assert_called_once_with([question])
    search_mock.assert_called_once()
    assert result == [{"text": "fallback", "metadata": {}}]


def test_fetch_entity_chunks_follows_redirect_from_opensearch(mocker: Any) -> None:
    """Redirect chunks returned after opensearch resolution must be followed."""
    redirect_chunks: list[dict[str, Any]] = [
        {
            "text": "REDIRECT Woodblock",
            "metadata": {"source_title": "Wooden Block", "chunk_index": 0},
        }
    ]
    final_chunks: list[dict[str, Any]] = [
        {
            "text": "Woodblock is a crafting material.",
            "metadata": {"source_title": "Woodblock", "chunk_index": 0},
        }
    ]

    def fake_get_page(title: str) -> list[dict[str, Any]]:
        if title == "Wooden Block":
            return redirect_chunks
        if title == "Woodblock":
            return final_chunks
        return []

    mocker.patch("agent.nodes.vs_get_page_by_title", side_effect=fake_get_page)
    mocker.patch(
        "agent.nodes._resolve_title_via_opensearch",
        return_value="Wooden Block",
    )
    mocker.patch("agent.nodes.embed_chunks")
    mocker.patch("agent.nodes.vs_semantic_search")

    result = _fetch_entity_chunks("wood block", "what is a wood block?")

    assert result == final_chunks


def test_opensearch_failure_does_not_raise(mocker: Any) -> None:
    """Raising from opensearch_title must not propagate; semantic fallback runs."""
    mocker.patch("agent.nodes.vs_get_page_by_title", return_value=[])
    mocker.patch(
        "ingestion.api_client.opensearch_title",
        side_effect=requests.ConnectionError("boom"),
    )
    embed_mock = mocker.patch("agent.nodes.embed_chunks", return_value=[[0.1] * 384])
    search_mock = mocker.patch(
        "agent.nodes.vs_semantic_search",
        return_value=[{"text": "fallback", "metadata": {}}],
    )

    result = _fetch_entity_chunks("anything", "question")

    assert result == [{"text": "fallback", "metadata": {}}]
    embed_mock.assert_called_once()
    search_mock.assert_called_once()


def test_extract_redirect_target_canonical_format() -> None:
    """The canonical 'REDIRECT Target' form must resolve to the target title."""
    assert _extract_redirect_target(_redirect_chunks("REDIRECT Woodblock")) == (
        "Woodblock"
    )


def test_extract_redirect_target_is_case_insensitive() -> None:
    """Lower- and mixed-case REDIRECT keywords must still be detected."""
    assert _extract_redirect_target(_redirect_chunks("redirect Woodblock")) == (
        "Woodblock"
    )
    assert _extract_redirect_target(_redirect_chunks("Redirect Woodblock")) == (
        "Woodblock"
    )


def test_extract_redirect_target_rejects_redirecting_prefix() -> None:
    """Prose starting with 'REDIRECTING' must not be parsed as a redirect.

    The old implementation used startswith('REDIRECT') which matched any
    word beginning with those eight letters and then sliced everything
    past them as the 'target'. The word boundary in the regex closes
    that hole.
    """
    chunks = _redirect_chunks("REDIRECTING players to the next quest hub.")
    assert _extract_redirect_target(chunks) is None


def test_extract_redirect_target_rejects_redirected_prefix() -> None:
    """Same false-positive class — 'REDIRECTED from ...' must not match."""
    chunks = _redirect_chunks("REDIRECTED from an old name.")
    assert _extract_redirect_target(chunks) is None


def test_extract_redirect_target_requires_whitespace_after_keyword() -> None:
    """REDIRECT must be followed by whitespace, not punctuation or letters.

    Guards against a chunk like 'REDIRECTOR' (no boundary, all letters)
    or 'REDIRECT.Target' (boundary, but no whitespace before the title).
    """
    assert _extract_redirect_target(_redirect_chunks("REDIRECTOR")) is None
    assert _extract_redirect_target(_redirect_chunks("REDIRECT.Target")) is None


def test_extract_redirect_target_stops_at_first_newline() -> None:
    """Trailing content after a newline must not be slurped into the target."""
    chunks = _redirect_chunks("REDIRECT Woodblock\nResidual category text.")
    assert _extract_redirect_target(chunks) == "Woodblock"


def test_extract_redirect_target_returns_none_for_non_redirect() -> None:
    """A normal page chunk must not be mistaken for a redirect."""
    chunks = _redirect_chunks("Woodblock is a crafting material.")
    assert _extract_redirect_target(chunks) is None


def test_extract_redirect_target_returns_none_for_multi_chunk_pages() -> None:
    """Redirect pages should always parse to one chunk; reject anything else.

    A multi-chunk result implies the upstream parser/chunker produced
    something other than a canonical redirect, and we'd rather skip the
    redirect-follow path than guess at which chunk holds the target.
    """
    chunks = [
        {"text": "REDIRECT Woodblock", "metadata": {"chunk_index": 0}},
        {"text": "Some other content.", "metadata": {"chunk_index": 1}},
    ]
    assert _extract_redirect_target(chunks) is None
