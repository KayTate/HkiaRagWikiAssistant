"""Shared HTTP retry predicates used across the agent and ingestion clients.
``is_transient_http_error`` filters by HTTP status only; ``should_retry_request``
extends it with ``ReadTimeout`` for the wiki API client."""

import requests

# 429 is rate-limiting; 5xx are server-side transient errors.
# Auth/permission/not-found errors (401, 403, 404) are excluded on
# purpose — they will not resolve themselves during the backoff window
# and would just burn the retry budget.
TRANSIENT_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}


def is_transient_http_error(exc: BaseException) -> bool:
    """Tenacity retry predicate: True only for transient HTTP responses.

    Replaces the previous ``retry_if_exception_type(requests.HTTPError)``
    used by the cloud LLM and embedding clients, which retried every
    HTTP error indiscriminately, including permanent failures like 401
    Unauthorized. Those would burn five attempts × up to 60s of
    exponential backoff before re-raising the same error, with no
    chance of recovery.

    Args:
        exc: Any exception raised inside the retried call.

    Returns:
        True iff ``exc`` is a ``requests.HTTPError`` whose response
        status is in ``TRANSIENT_STATUS_CODES``. False for non-HTTP
        exceptions and for HTTP errors with no attached response.
    """
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in TRANSIENT_STATUS_CODES
    return False


def should_retry_request(exc: BaseException) -> bool:
    """Tenacity retry predicate: transient HTTP responses or read timeouts.

    Combines the status-code filter with ``ReadTimeout``, which is
    always worth retrying. Used by the wiki API client because batch
    fetches pull large payloads where a slow response is almost
    always transient. The cloud LLM and embedding clients use
    ``is_transient_http_error`` instead — their decorators were not
    configured to handle ReadTimeout before, and widening their retry
    behavior is intentionally out of scope for the dedup that
    introduced this module.

    Args:
        exc: Any exception raised inside the retried call.

    Returns:
        True for ``ReadTimeout`` or transient HTTP errors; False
        otherwise.
    """
    if isinstance(exc, requests.ReadTimeout):
        return True
    return is_transient_http_error(exc)
