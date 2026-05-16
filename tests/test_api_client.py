"""Tests for ingestion/api_client.py — pagination and response parsing.

The retry-decorator behavior is covered in tests/test_http_retries.py
(401 doesn't retry, 429 does, ReadTimeout does). This file covers the
wrappers that build on ``_get_with_retry``: pagination of
``get_all_page_titles`` and ``get_all_pages_with_revision_ids``,
response shape handling in ``get_pages_wikitext_batch``,
``opensearch_title``, and ``get_page_revision_id``.

Most tests mock ``_get_with_retry`` directly so the wrapper logic is
exercised without touching tenacity. ``opensearch_title`` defines its
own retry-decorated closure, so its tests mock ``requests.get`` and
return 200 responses to keep retries dormant.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from ingestion.api_client import (
    WikiAPIError,
    get_all_page_titles,
    get_all_pages_with_revision_ids,
    get_cargo_items,
    get_page_revision_id,
    get_pages_wikitext_batch,
    opensearch_title,
)


def _ok_response(json_data: dict[str, Any]) -> Mock:
    """Build a 200 response Mock that returns ``json_data``."""
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


# ---------------------------------------------------------------------------
# get_all_page_titles
# ---------------------------------------------------------------------------


def test_get_all_page_titles_paginates_until_no_continue_token(
    mocker: Any,
) -> None:
    """Multiple pages must be concatenated and pagination must stop cleanly.

    The pagination loop reads ``data['continue']['apcontinue']`` and
    breaks when it's missing. Pinning the multi-page concat catches a
    regression that overwrites instead of extends; pinning the stop
    condition catches an infinite loop on a malformed continue block.
    """
    pages = [
        _ok_response(
            {
                "query": {"allpages": [{"title": "Alpha"}, {"title": "Beta"}]},
                "continue": {"apcontinue": "Beta"},
            }
        ),
        _ok_response(
            {"query": {"allpages": [{"title": "Gamma"}]}, "continue": {}}
        ),
    ]
    get_with_retry = mocker.patch(
        "ingestion.api_client._get_with_retry",
        side_effect=[r.json() for r in pages],
    )

    titles = get_all_page_titles()

    assert titles == ["Alpha", "Beta", "Gamma"]
    assert get_with_retry.call_count == 2, (
        "Pagination must follow the continue token exactly once after "
        "the first call, then stop"
    )


def test_get_all_page_titles_handles_missing_continue_block(
    mocker: Any,
) -> None:
    """A response with no 'continue' key at all must end pagination.

    Defensive read against the MediaWiki edge case where 'continue'
    exists but lacks 'apcontinue' — the code uses a chained ``.get``
    that returns None for missing keys; pinning catches a refactor
    back to subscript access (which would KeyError).
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={"query": {"allpages": [{"title": "Solo"}]}},
    )
    titles = get_all_page_titles()
    assert titles == ["Solo"]


# ---------------------------------------------------------------------------
# get_all_pages_with_revision_ids
# ---------------------------------------------------------------------------


def test_get_all_pages_with_revision_ids_parses_generator_response(
    mocker: Any,
) -> None:
    """Generator-style response must surface ``revision_id`` as an int.

    The MediaWiki ``generator=allpages`` API returns pages keyed by
    page-id (a string), each with a ``revisions`` list whose first
    item has ``revid``. A regression that read ``revid`` as the page
    key would silently use stable-but-wrong IDs in SQLite.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={
            "query": {
                "pages": {
                    "1": {"title": "Alpha", "revisions": [{"revid": 100}]},
                    "2": {"title": "Beta", "revisions": [{"revid": 200}]},
                }
            }
        },
    )
    results = get_all_pages_with_revision_ids()

    by_title = {r["title"]: r["revision_id"] for r in results}
    assert by_title == {"Alpha": 100, "Beta": 200}
    assert all(isinstance(r["revision_id"], int) for r in results), (
        "revision_id must be coerced to int — SQLite revision-compare "
        "logic relies on integer equality, not string"
    )


def test_get_all_pages_with_revision_ids_skips_pages_with_no_revisions(
    mocker: Any,
) -> None:
    """A page without a revisions list must be omitted, not crash.

    The wiki occasionally returns placeholder pages with no
    revisions (deleted/protected); skipping is the documented
    behavior. Pinning catches a regression that would IndexError on
    ``revisions[0]``.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={
            "query": {
                "pages": {
                    "1": {"title": "Alpha", "revisions": [{"revid": 100}]},
                    "2": {"title": "NoRevisions", "revisions": []},
                    "3": {"title": "MissingRevisions"},
                }
            }
        },
    )
    results = get_all_pages_with_revision_ids()
    titles = [r["title"] for r in results]

    assert titles == ["Alpha"]


# ---------------------------------------------------------------------------
# get_pages_wikitext_batch
# ---------------------------------------------------------------------------


def test_get_pages_wikitext_batch_parses_star_field(mocker: Any) -> None:
    """The ``*`` key in revisions[0] holds the wikitext content.

    Pinning this exact key path because it's a MediaWiki quirk that
    a future migration to a different action might lose.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={
            "query": {
                "pages": {
                    "1": {
                        "title": "Alpha",
                        "revisions": [{"*": "Wikitext for Alpha."}],
                    },
                    "2": {
                        "title": "Beta",
                        "revisions": [{"*": "Wikitext for Beta."}],
                    },
                }
            }
        },
    )
    result = get_pages_wikitext_batch(["Alpha", "Beta"])

    assert result == {
        "Alpha": "Wikitext for Alpha.",
        "Beta": "Wikitext for Beta.",
    }


def test_get_pages_wikitext_batch_omits_pages_without_content(mocker: Any) -> None:
    """A page whose first revision lacks ``*`` must be silently omitted.

    The pipeline downstream uses ``wikitext_map.get(title)`` and treats
    missing as "skip with warning"; this test pins that the API client
    omits rather than emitting an empty string.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={
            "query": {
                "pages": {
                    "1": {"title": "HasContent", "revisions": [{"*": "Body"}]},
                    "2": {"title": "MissingStar", "revisions": [{}]},
                    "3": {"title": "NoRevisions", "revisions": []},
                }
            }
        },
    )
    result = get_pages_wikitext_batch(
        ["HasContent", "MissingStar", "NoRevisions"]
    )

    assert result == {"HasContent": "Body"}
    assert "MissingStar" not in result
    assert "NoRevisions" not in result


# ---------------------------------------------------------------------------
# get_cargo_items
# ---------------------------------------------------------------------------


def test_get_cargo_items_unwraps_title_rows(mocker: Any) -> None:
    """Cargo wraps every row as ``{"title": {...}}``; the helper must unwrap.

    Callers want a plain ``list[dict]`` of field values. A regression
    that returned the raw cargoquery list would force every caller to
    re-implement the unwrap and silently drop ``.get('title')``-less
    rows.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={
            "cargoquery": [
                {"title": {"name": "Ingot"}},
                {"title": {"name": "Microphone"}},
            ]
        },
    )
    rows = get_cargo_items(
        tables="TagItemList",
        fields="name",
        where='tags HOLDS "Metal"',
    )
    assert rows == [{"name": "Ingot"}, {"name": "Microphone"}]


def test_get_cargo_items_raises_on_malformed_response(mocker: Any) -> None:
    """A response missing the ``cargoquery`` key must surface as WikiAPIError.

    Cargo returns the empty list (not a missing key) when there are no
    matches, so a missing key indicates an actual API problem — surface
    it so the parser's per-template error handler can log + drop the
    template body cleanly.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={},
    )
    with pytest.raises(WikiAPIError, match="cargoquery"):
        get_cargo_items(
            tables="TagItemList",
            fields="name",
            where='tags HOLDS "Metal"',
        )


# ---------------------------------------------------------------------------
# opensearch_title
# ---------------------------------------------------------------------------


def test_opensearch_title_returns_first_match(mocker: Any) -> None:
    """The opensearch endpoint returns ``[query, [titles], [descs], [urls]]``.

    The function must extract ``data[1][0]`` as the canonical title.
    """
    mocker.patch("time.sleep")
    mocker.patch(
        "requests.get",
        return_value=_ok_response(
            ["wood block", ["Woodblock", "Wood Block (item)"], [], []]
        ),
    )
    assert opensearch_title("wood block") == "Woodblock"


def test_opensearch_title_returns_none_for_empty_results(mocker: Any) -> None:
    """An opensearch result with no titles must return None, not crash."""
    mocker.patch("time.sleep")
    mocker.patch(
        "requests.get",
        return_value=_ok_response(["unmatchable", [], [], []]),
    )
    assert opensearch_title("unmatchable") is None


def test_opensearch_title_returns_none_for_unexpected_shape(mocker: Any) -> None:
    """A non-list response must surface as None, not raise.

    The agent's hot path swallows opensearch errors via this None
    return; a refactor that raised on bad shapes would propagate
    the failure into the agent loop and abort a question.
    """
    mocker.patch("time.sleep")
    mocker.patch(
        "requests.get",
        return_value=_ok_response({"oops": "not the array MediaWiki promises"}),
    )
    assert opensearch_title("query") is None


# ---------------------------------------------------------------------------
# get_page_revision_id
# ---------------------------------------------------------------------------


def test_get_page_revision_id_extracts_integer_revid(mocker: Any) -> None:
    """A standard single-page response must yield the int revid."""
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={
            "query": {"pages": {"42": {"revisions": [{"revid": 9876}]}}}
        },
    )
    assert get_page_revision_id("Some Page") == 9876


def test_get_page_revision_id_raises_wiki_api_error_on_unexpected_shape(
    mocker: Any,
) -> None:
    """Missing keys must surface as WikiAPIError with the page title.

    A bare KeyError from this helper would have no context — the
    pipeline log would say ``KeyError: 'revisions'`` with no clue
    which page tripped it. Wrapping in WikiAPIError keeps the title
    in the message.
    """
    mocker.patch(
        "ingestion.api_client._get_with_retry",
        return_value={"query": {"pages": {"42": {}}}},  # no revisions key
    )
    with pytest.raises(WikiAPIError, match="Some Page"):
        get_page_revision_id("Some Page")
