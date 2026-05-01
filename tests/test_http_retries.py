"""Tests for the HTTP retry predicates and their effect on retry behavior.

Covers the change from ``retry_if_exception_type(requests.HTTPError)`` —
which retried any HTTP error indiscriminately — to a status-code filter
that skips the backoff budget for non-transient codes (401/403/404).
The predicates were originally duplicated across three modules; they
now live in ``common.http`` and the predicate-contract tests assert
against the canonical functions there. The end-to-end behavior tests
(further down) still exercise each call site so they catch a wiring
regression even though there is only one predicate to break.
"""

from typing import Any
from unittest.mock import Mock

import pytest
import requests
from pydantic import SecretStr

from common.http import is_transient_http_error, should_retry_request


class _FakeResponse:
    """Minimal stand-in for requests.Response — only status_code is needed."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _make_http_error(status_code: int) -> requests.HTTPError:
    """Build an HTTPError carrying a response with the given status code."""
    err = requests.HTTPError(f"HTTP {status_code}")
    err.response = _FakeResponse(status_code)
    return err


def _fake_response(status_code: int, json_data: dict[str, Any] | None = None) -> Mock:
    """Build a Mock response that behaves like requests.Response.

    For 4xx/5xx, ``raise_for_status`` raises an HTTPError carrying the
    given status; for 2xx, it returns None and ``json()`` returns the
    supplied payload. Lifts the boilerplate that every behavior test
    below would otherwise repeat.
    """
    response = Mock()
    response.status_code = status_code
    if status_code >= 400:
        response.raise_for_status.side_effect = _make_http_error(status_code)
    else:
        response.raise_for_status.return_value = None
        response.json.return_value = json_data or {}
    return response


# ---------------------------------------------------------------------------
# Predicate contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_predicate_retries_transient_status_codes(status_code: int) -> None:
    """Every transient status code must be classified as retryable.

    Parametrised so a regression that drops one of these codes from the
    transient set fails for that code specifically, instead of the whole
    test going green because the most common one (429) still works.
    Both predicates must agree on transient-status responses;
    ``should_retry_request`` is a strict superset of
    ``is_transient_http_error`` and must not subtract any code.
    """
    err = _make_http_error(status_code)
    assert is_transient_http_error(err) is True
    assert should_retry_request(err) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_predicate_skips_non_transient_4xx(status_code: int) -> None:
    """4xx auth/permission/not-found responses must not be retried.

    This is the headline behavior change: previously the decorators
    used ``retry_if_exception_type(requests.HTTPError)`` and any 4xx
    burned five attempts of exponential backoff before giving up.
    """
    err = _make_http_error(status_code)
    assert is_transient_http_error(err) is False
    assert should_retry_request(err) is False


def test_predicate_skips_non_http_exceptions() -> None:
    """Generic exceptions must not match the predicate."""
    assert is_transient_http_error(ValueError("boom")) is False
    assert should_retry_request(ValueError("boom")) is False


def test_predicate_skips_http_error_without_response() -> None:
    """HTTPError without a response object cannot be classified — skip retry.

    requests can construct an HTTPError with response=None when the
    failure happens before a response is received. The predicate must
    not crash on this and must not retry.
    """
    err = requests.HTTPError("no response attached")
    assert is_transient_http_error(err) is False
    assert should_retry_request(err) is False


def test_should_retry_request_extends_with_read_timeout() -> None:
    """The combined predicate adds ReadTimeout to the status filter.

    Wiki batch fetches pull large payloads; a ReadTimeout on those is
    almost always transient and worth retrying. The base predicate
    ``is_transient_http_error`` does NOT include ReadTimeout — the
    cloud LLM and embedding clients use it directly because their
    decorators were not configured to handle ReadTimeout before the
    dedup that introduced ``common.http`` and we kept that contract.
    """
    assert should_retry_request(requests.ReadTimeout()) is True
    assert is_transient_http_error(requests.ReadTimeout()) is False


# ---------------------------------------------------------------------------
# End-to-end behavior: 401 must not retry, 429 must
# ---------------------------------------------------------------------------


def _patch_openai_settings_and_sleep(mocker: Any) -> None:
    """Set OpenAI credentials and silence tenacity's backoff sleeps.

    The api-key mock must be a ``SecretStr`` because the production
    call sites do ``settings.openai_api_key.get_secret_value()``;
    patching with a plain string would AttributeError at runtime and
    a regression of the SecretStr migration would surface here first.
    """
    mocker.patch("config.settings.settings.openai_api_key", SecretStr("test-key"))
    mocker.patch("config.settings.settings.llm_model", "gpt-4o-mini")
    # Tenacity uses time.sleep between attempts. Without this patch the
    # 429-then-200 test below would wait 2+4=6 real seconds.
    mocker.patch("time.sleep")


def _patch_anthropic_settings_and_sleep(mocker: Any) -> None:
    """Set Anthropic credentials and silence tenacity's backoff sleeps."""
    mocker.patch(
        "config.settings.settings.anthropic_api_key", SecretStr("test-key")
    )
    mocker.patch("config.settings.settings.llm_model", "claude-haiku-4-5")
    mocker.patch("time.sleep")


def test_openai_call_does_not_retry_on_401(mocker: Any) -> None:
    """A 401 response must trigger exactly one POST, not five attempts."""
    _patch_openai_settings_and_sleep(mocker)
    post_mock = mocker.patch("requests.post", return_value=_fake_response(401))

    from agent.llm import _call_openai_with_retry

    with pytest.raises(requests.HTTPError):
        _call_openai_with_retry("sys", "user")

    assert post_mock.call_count == 1, (
        f"401 must not be retried (predicate filters it out); got "
        f"{post_mock.call_count} attempts. Regression of the "
        f"retry_if_exception_type(HTTPError) bug."
    )


def test_openai_call_retries_on_429_until_success(mocker: Any) -> None:
    """A 429 response must trigger retries; success on attempt 3 returns OK.

    The complement to the 401 test: proves the predicate still permits
    the retry path it was always meant to cover.
    """
    _patch_openai_settings_and_sleep(mocker)
    post_mock = mocker.patch(
        "requests.post",
        side_effect=[
            _fake_response(429),
            _fake_response(429),
            _fake_response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ],
    )

    from agent.llm import _call_openai_with_retry

    result = _call_openai_with_retry("sys", "user")

    assert result == "ok"
    assert post_mock.call_count == 3


def test_openai_call_does_not_retry_on_404(mocker: Any) -> None:
    """A 404 (e.g. unknown model name) must not be retried.

    Distinct from 401 because 404 is more user-error-shaped: the user
    typo'd the model name in settings. Five attempts of 60s backoff
    before surfacing the typo would be a terrible debugging experience.
    """
    _patch_openai_settings_and_sleep(mocker)
    post_mock = mocker.patch("requests.post", return_value=_fake_response(404))

    from agent.llm import _call_openai_with_retry

    with pytest.raises(requests.HTTPError):
        _call_openai_with_retry("sys", "user")

    assert post_mock.call_count == 1


# ---------------------------------------------------------------------------
# Anthropic — symmetric with OpenAI to guard against single-provider
# regressions (e.g. someone "fixing" only the OpenAI decorator).
# ---------------------------------------------------------------------------


def test_anthropic_call_does_not_retry_on_401(mocker: Any) -> None:
    """A 401 response from Anthropic must trigger exactly one POST."""
    _patch_anthropic_settings_and_sleep(mocker)
    post_mock = mocker.patch("requests.post", return_value=_fake_response(401))

    from agent.llm import _call_anthropic_with_retry

    with pytest.raises(requests.HTTPError):
        _call_anthropic_with_retry("sys", "user")

    assert post_mock.call_count == 1


def test_anthropic_call_retries_on_429_until_success(mocker: Any) -> None:
    """A 429 from Anthropic must trigger retries; success on attempt 3."""
    _patch_anthropic_settings_and_sleep(mocker)
    post_mock = mocker.patch(
        "requests.post",
        side_effect=[
            _fake_response(429),
            _fake_response(429),
            _fake_response(200, {"content": [{"text": "ok"}]}),
        ],
    )

    from agent.llm import _call_anthropic_with_retry

    result = _call_anthropic_with_retry("sys", "user")

    assert result == "ok"
    assert post_mock.call_count == 3


# ---------------------------------------------------------------------------
# api_client — locks in the wiring of should_retry_request into the
# tenacity decorator. The predicate-only tests above don't catch a
# regression where someone reverts the decorator argument.
# ---------------------------------------------------------------------------


def test_api_client_does_not_retry_on_401(mocker: Any) -> None:
    """A 401 from the wiki API must trigger exactly one GET, not seven attempts.

    _get_with_retry has the most generous retry budget in the codebase
    (7 attempts × up to 120s backoff) because it powers the ingestion
    workhorse. Burning that budget on a permanent error like 401 used
    to mean roughly 195s of backoff before surfacing the failure.
    """
    mocker.patch("time.sleep")
    get_mock = mocker.patch("requests.get", return_value=_fake_response(401))

    from ingestion.api_client import WikiAPIError, _get_with_retry

    with pytest.raises(WikiAPIError):
        _get_with_retry({"action": "query"})

    assert get_mock.call_count == 1


def test_api_client_retries_on_429_until_success(mocker: Any) -> None:
    """A 429 from the wiki API must trigger retries until success."""
    mocker.patch("time.sleep")
    get_mock = mocker.patch(
        "requests.get",
        side_effect=[
            _fake_response(429),
            _fake_response(429),
            _fake_response(200, {"query": {"pages": {}}}),
        ],
    )

    from ingestion.api_client import _get_with_retry

    result = _get_with_retry({"action": "query"})
    assert result == {"query": {"pages": {}}}
    assert get_mock.call_count == 3


def test_api_client_retries_on_read_timeout(mocker: Any) -> None:
    """ReadTimeout must trigger a retry — distinct from the status-code path.

    The combined predicate ``should_retry_request`` adds ReadTimeout
    to the transient-status filter. Without that branch, every wiki
    batch that hit a slow response would surface as a hard failure on
    the first try. This test fails if someone wires the decorator to
    ``is_transient_http_error`` instead of ``should_retry_request``.
    """
    mocker.patch("time.sleep")
    get_mock = mocker.patch(
        "requests.get",
        side_effect=[
            requests.ReadTimeout("slow batch"),
            _fake_response(200, {"query": {"pages": {}}}),
        ],
    )

    from ingestion.api_client import _get_with_retry

    result = _get_with_retry({"action": "query"})
    assert result == {"query": {"pages": {}}}
    assert get_mock.call_count == 2


# ---------------------------------------------------------------------------
# Ollama timeout — proves the call site reads from settings rather than
# the previously hard-coded 300s. Without this test, a refactor that
# inlined the constant again would silently revert the fix.
# ---------------------------------------------------------------------------


def test_ollama_call_uses_configurable_timeout(mocker: Any) -> None:
    """_call_ollama_with_retry must pass settings.ollama_request_timeout_seconds."""
    mocker.patch("config.settings.settings.ollama_request_timeout_seconds", 99)
    mocker.patch("config.settings.settings.llm_model", "llama3")
    mocker.patch("time.sleep")
    post_mock = mocker.patch(
        "requests.post",
        return_value=_fake_response(200, {"message": {"content": "hi"}}),
    )

    from agent.llm import _call_ollama_with_retry

    _call_ollama_with_retry("sys", "user")

    assert post_mock.call_args.kwargs["timeout"] == 99, (
        "Ollama timeout must read from settings.ollama_request_timeout_seconds"
    )
