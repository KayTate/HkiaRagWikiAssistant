"""MediaWiki API client for the HKIA wiki.

All requests sleep settings.wiki_request_delay_seconds between calls to
respect the wiki's rate limits. Transient HTTP errors are retried via
tenacity with exponential backoff.
"""

import logging
import time

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class WikiAPIError(RuntimeError):
    """Raised when the MediaWiki API returns an unrecoverable error."""


def _is_transient_http_error(exc: BaseException) -> bool:
    """Return True if the exception represents a transient HTTP error worth retrying."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _TRANSIENT_STATUS_CODES
    return False


def _should_retry_request(exc: BaseException) -> bool:
    """Tenacity retry predicate: transient HTTP responses or read timeouts.

    Combines the status-code filter with ReadTimeout, which is always
    worth retrying. Replaces the previous
    ``retry_if_exception_type((requests.HTTPError, requests.ReadTimeout))``
    so non-transient HTTP errors (401/403/404) skip the backoff budget
    and propagate immediately.
    """
    if isinstance(exc, requests.ReadTimeout):
        return True
    return _is_transient_http_error(exc)


def _get_with_retry(
    params: dict,  # type: ignore[type-arg]
    timeout: int = 30,
) -> dict:  # type: ignore[type-arg]
    """Execute a GET request against the wiki API with retry on transient errors.

    Sleeps after every attempt to respect the configured rate limit, even
    on failure, so retries do not hammer the API without backoff.

    Args:
        params: Query parameters for the MediaWiki API call.
        timeout: Request timeout in seconds. Use a higher value for
            batch content requests that return large payloads.

    Returns:
        Parsed JSON response body as a dict.

    Raises:
        WikiAPIError: If the request fails after all retry attempts.
    """

    # Generous retry budget on purpose: this function is the workhorse
    # of the ingestion pipeline, where a long-running batch job over
    # thousands of pages should tolerate transient wiki flakes rather
    # than abort and lose progress. Worst-case wait per call is roughly
    # 5 + 10 + 20 + 40 + 60 + 60 = 195 seconds of backoff across 7
    # attempts. That is intentional for ingestion but would be far too
    # patient for the agent's read path — opensearch_title below uses a
    # much tighter budget for that reason.
    @retry(
        retry=retry_if_exception(_should_retry_request),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(7),
        reraise=True,
    )
    def _do_request() -> dict:  # type: ignore[type-arg]
        time.sleep(settings.wiki_request_delay_seconds)
        response = requests.get(
            settings.wiki_api_url,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    try:
        return _do_request()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise WikiAPIError(
            f"MediaWiki API request failed after retries: params={params}, "
            f"status={status}"
        ) from exc
    except requests.ReadTimeout as exc:
        raise WikiAPIError(
            f"MediaWiki API request timed out after retries: params={params}"
        ) from exc
    except requests.RequestException as exc:
        raise WikiAPIError(
            f"MediaWiki API request failed: params={params}, error={exc}"
        ) from exc


def get_all_page_titles() -> list[str]:
    """Fetch every page title from the wiki via the allpages API list.

    Paginates using the 'apcontinue' token until all pages are retrieved.
    Returns titles in the order the API provides them (typically alphabetical).

    Returns:
        Flat list of all wiki page titles.

    Raises:
        WikiAPIError: If any paginated request fails after retries.
    """
    titles: list[str] = []
    params: dict = {  # type: ignore[type-arg]
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "format": "json",
    }

    while True:
        data = _get_with_retry(params)
        pages = data.get("query", {}).get("allpages", [])
        titles.extend(page["title"] for page in pages)

        # Read the continuation token defensively. MediaWiki may return a
        # 'continue' block without 'apcontinue' when generators are
        # combined with other modules; treat a missing token as "no more
        # pages" rather than KeyError-ing mid-pagination.
        next_token = data.get("continue", {}).get("apcontinue")
        if next_token is None:
            break
        params["apcontinue"] = next_token

    logger.info("Fetched %d page titles from wiki", len(titles))
    return titles


def get_page_wikitext(page_title: str) -> str:
    """Fetch the raw wikitext for a single page.

    Args:
        page_title: The exact wiki page title to fetch.

    Returns:
        Raw wikitext string for the page.

    Raises:
        WikiAPIError: If the API call fails or returns an error response.
    """
    data = _get_with_retry(
        {
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "format": "json",
        }
    )
    try:
        wikitext = data["parse"]["wikitext"]["*"]
        return str(wikitext)
    except KeyError as exc:
        raise WikiAPIError(
            f"Unexpected response structure fetching wikitext for '{page_title}': "
            f"missing key {exc}"
        ) from exc


def get_all_pages_with_revision_ids() -> list[dict[str, str | int]]:
    """Fetch all page titles with their current revision IDs in a single pass.

    Uses the generator=allpages query to return both title and revision ID
    in each paginated response, avoiding the need for per-page revision
    lookups. Each batch returns up to 50 pages.

    Returns:
        List of dicts, each with keys 'title' (str) and 'revision_id' (int).

    Raises:
        WikiAPIError: If any paginated request fails after retries.
    """
    logger.info("Fetching all page titles with revision IDs from wiki")
    results: list[dict[str, str | int]] = []
    params: dict[str, str] = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": "50",
        "prop": "revisions",
        "rvprop": "ids",
        "format": "json",
    }

    while True:
        data = _get_with_retry(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            revisions = page.get("revisions", [])
            if title and revisions:
                results.append(
                    {
                        "title": title,
                        "revision_id": int(revisions[0]["revid"]),
                    }
                )

        # Defensive read — see get_all_page_titles above for rationale.
        next_token = data.get("continue", {}).get("gapcontinue")
        if next_token is None:
            break
        params["gapcontinue"] = next_token

    logger.info("Fetched %d pages with revision IDs from wiki", len(results))
    return results


def get_pages_wikitext_batch(titles: list[str]) -> dict[str, str]:
    """Fetch raw wikitext for multiple pages in a single API call.

    The MediaWiki API accepts up to 50 pipe-separated titles per request.
    Callers should batch titles into groups of 50 before calling this
    function.

    Args:
        titles: List of page titles to fetch (max 50 per MediaWiki limit).

    Returns:
        Dict mapping page title to its raw wikitext content. Pages that
        have no revisions or are missing are silently omitted.

    Raises:
        WikiAPIError: If the API call fails after retries.
    """
    joined = "|".join(titles)
    data = _get_with_retry(
        {
            "action": "query",
            "titles": joined,
            "prop": "revisions",
            "rvprop": "content",
            "format": "json",
        },
        timeout=120,
    )

    result: dict[str, str] = {}
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        title = page.get("title", "")
        revisions = page.get("revisions", [])
        if title and revisions and "*" in revisions[0]:
            result[title] = str(revisions[0]["*"])

    return result


def opensearch_title(query: str, limit: int = 3) -> str | None:
    """Resolve a free-text query to a canonical wiki page title.

    Calls the MediaWiki opensearch endpoint, which performs prefix and
    fuzzy matching against page titles. Used by the agent when exact
    title variants all fail, before falling back to semantic search.

    Restricts to namespace 0 (main articles) to exclude Talk, User, etc.

    Args:
        query: Free-text entity name, e.g. 'Mystery Tree' or 'apple orchard'.
        limit: Max results to request from the API (we only use the first).

    Returns:
        The best-match canonical page title, or None if no match or on error.
        Errors are logged and swallowed — this is a best-effort fallback
        and must not raise into the agent's retrieval path.
    """

    # Tight retry budget on purpose: opensearch is a best-effort fallback
    # called from the agent's hot path. A single user question must not
    # stall for minutes waiting on a flapping API — we'd rather fall
    # through to semantic search quickly. Worst-case wait here is roughly
    # 2 + 4 = 6 seconds of backoff plus request timeouts. The ingestion
    # path uses _get_with_retry above, which is tuned much more
    # generously because batch jobs can afford to wait.
    @retry(
        retry=retry_if_exception(_should_retry_request),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _do_request() -> list:  # type: ignore[type-arg]
        time.sleep(settings.wiki_request_delay_seconds)
        response = requests.get(
            settings.wiki_api_url,
            params={
                "action": "opensearch",
                "search": query,
                "limit": str(limit),
                "namespace": "0",
                "format": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    try:
        data = _do_request()
        if not isinstance(data, list) or len(data) < 2:
            logger.warning(
                "opensearch returned unexpected shape for query '%s': %r",
                query,
                data,
            )
            return None
        titles = data[1]
        if not isinstance(titles, list) or not titles:
            return None
        return str(titles[0])
    except Exception as exc:  # noqa: BLE001 — best-effort fallback
        logger.warning("opensearch lookup failed for query '%s': %s", query, exc)
        return None


def get_page_revision_id(page_title: str) -> int:
    """Fetch the current revision ID for a single page.

    Args:
        page_title: The exact wiki page title to query.

    Returns:
        The integer revision ID of the current revision.

    Raises:
        WikiAPIError: If the API call fails or the page has no revisions.
    """
    data = _get_with_retry(
        {
            "action": "query",
            "titles": page_title,
            "prop": "revisions",
            "rvprop": "ids",
            "format": "json",
        }
    )
    try:
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return int(page["revisions"][0]["revid"])
    except (KeyError, IndexError, StopIteration) as exc:
        raise WikiAPIError(
            f"Could not extract revision ID for '{page_title}': {exc}"
        ) from exc
