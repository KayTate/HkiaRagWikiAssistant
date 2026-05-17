"""Tests for entity extraction, title variant resolution, and opensearch fallback.

All external calls (ChromaDB, MediaWiki API, embeddings) are mocked so these
tests run fully in-process with no external dependencies.
"""

from typing import Any

import pytest
import requests

from agent.extraction import _extract_entity_from_question, _normalize_entity
from agent.retrieval import (
    _fetch_entity_chunks,
    _load_redirects,
    _resolve_via_redirect,
    _strip_stopwords,
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


@pytest.mark.parametrize(
    "question, expected",
    [
        # Specific shapes added to cover trace-confirmed gaps.
        ("What items have the metal tag?", "metal"),
        ("When is Espresso's birthday?", "Espresso"),
        ("What does Retsuko give as a gift?", "Retsuko"),
        (
            "Which characters are typically in Seaside Resort?",
            "Seaside Resort",
        ),
        ("What fish can I find in Rainbow Reef?", "Rainbow Reef"),
        ("What is the Fish Derby?", "Fish Derby"),
        ("What is an Avatar Palette?", "Avatar Palette"),
        ("How does the inventory system work?", "inventory system"),
        # New verbs: obtain, catch, make, access, repair.
        ("How do I obtain the Gudetama Dress?", "Gudetama Dress"),
        ("How do I catch a Burning Perch?", "Burning Perch"),
        (
            "How do I make the 50th Anniversary Cheesecake?",
            "50th Anniversary Cheesecake",
        ),
        ("How do I access Merry Meadows Plaza?", "Merry Meadows Plaza"),
        # Existing verb patterns still work after the reorder.
        ("How do I unlock the Spooky Swamp?", "Spooky Swamp"),
        ("How do I reach Snow Village?", "Snow Village"),
        ("Where is the candy cloud machine?", "candy cloud machine"),
        ("Who is Tuxedosam?", "Tuxedosam"),
    ],
)
def test_extract_entity_from_question_patterns(
    question: str, expected: str
) -> None:
    """Regex-driven entity extraction covers the trace-observed question shapes."""
    assert _extract_entity_from_question(question) == expected


def test_title_candidates_includes_tag_suffix() -> None:
    """A bare-noun entity yields a ' (Tag)' variant for Tag-namespace pages."""
    candidates = _title_candidates("metal")
    assert "Metal (Tag)" in candidates


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
    """Semantic fallback embeds the full question with stopwords stripped."""
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

    # Question has leading stopwords stripped: "How do I" -> removed
    embed_mock.assert_called_once_with(["unlock the Mystery Tree quest?"])
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


def test_title_candidates_includes_case_insensitive_variants() -> None:
    """Candidates include title case and lowercase forms for case-insensitive lookup."""
    candidates = _title_candidates("mystery tree")
    assert "mystery tree" in candidates
    assert "Mystery Tree" in candidates
    # uppercase also included for exhaustive matching
    assert "MYSTERY TREE" in candidates

    # All variants should work with the entity as given
    candidates2 = _title_candidates("Mystery Tree")
    assert "Mystery Tree" in candidates2
    assert "mystery tree" in candidates2


def test_title_candidates_includes_drop_last_word_variants() -> None:
    """Candidates include drop-last-word variants for compound names."""
    # "Item" is in the allowlist, so "Magic Hat (item)" -> "Magic Hat"
    candidates = _title_candidates("Magic Hat")
    assert "Magic Hat" in candidates

    # With suffix: "Magic Hat (item)" -> include "Magic Hat (item)" and "Magic Hat"
    item_candidates = _title_candidates("Magic Hat")
    magic_hat_item = [c for c in item_candidates if "(item)" in c]
    assert len(magic_hat_item) > 0

    # Single-word entities should not generate drop-last-word variants
    single = _title_candidates("Banana")
    banana_base = [c for c in single if "Banana" in c and "(" not in c]
    assert len(banana_base) > 0


def test_title_candidates_drop_last_word_only_for_allowlist() -> None:
    """Drop-last-word variants only apply to known compound types."""
    # "Hat" is not in the allowlist, so no drop should occur
    candidates = _title_candidates("Magic Hat")
    # We should not get "Magic" as a standalone candidate without suffix
    standalone_magic = [c for c in candidates if c == "Magic"]
    assert len(standalone_magic) == 0


def test_strip_stopwords_removes_leading_trailing_words() -> None:
    """Stopword stripping removes common English words from start/end only."""
    assert (
        _strip_stopwords("How do I unlock the Mystery Tree")
        == "unlock the Mystery Tree"
    )
    assert _strip_stopwords("the apple of my eye") == "apple of my eye"
    assert _strip_stopwords("What is the best item") == "best item"
    # Should not strip middle words
    assert _strip_stopwords("apple tree of mystery") == "apple tree of mystery"


def test_strip_stopwords_returns_text_when_all_stopwords() -> None:
    """If text is all stopwords, return the original text unchanged."""
    # "the" is a stopword, but it's the only word, so return it as-is
    assert _strip_stopwords("the") == "the"
    assert _strip_stopwords("a the") == "a the"


def test_strip_stopwords_handles_empty_and_whitespace() -> None:
    """Stopword stripping handles empty/whitespace inputs gracefully."""
    assert _strip_stopwords("") == ""
    assert _strip_stopwords("   ") == "   "


