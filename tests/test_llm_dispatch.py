"""Tests for agent/llm.py provider dispatch and missing-key errors.

The retry-predicate and per-provider HTTP behavior are covered in
tests/test_http_retries.py. This file targets the parts of agent.llm
that file does not exercise: the four-branch dispatcher in
``_call_llm`` and the configuration-error paths in ``_call_openai``
and ``_call_anthropic``.
"""

from typing import Any

import pytest
from pydantic import SecretStr


def test_call_llm_dispatches_to_ollama(mocker: Any) -> None:
    """provider='ollama' must route to the Ollama implementation, not others.

    The dispatcher's three providers each take different argument
    shapes (json_mode flows to OpenAI, ignored elsewhere). A
    misrouted call would silently lose json_mode and produce
    unparseable extract responses in the agent loop.
    """
    mocker.patch("config.settings.settings.llm_provider", "ollama")
    ollama_mock = mocker.patch("agent.llm._call_ollama", return_value="from-ollama")
    openai_mock = mocker.patch("agent.llm._call_openai")
    anthropic_mock = mocker.patch("agent.llm._call_anthropic")

    from agent.llm import _call_llm

    result = _call_llm("sys", "user", json_mode=True)

    assert result == "from-ollama"
    ollama_mock.assert_called_once_with("sys", "user")
    openai_mock.assert_not_called()
    anthropic_mock.assert_not_called()


def test_call_llm_dispatches_to_openai_with_json_mode(mocker: Any) -> None:
    """provider='openai' must forward json_mode — it is the only consumer.

    Pinning the keyword forwarding here protects the JSON-extract
    contract: if json_mode is dropped en route to _call_openai, the
    extract LLM may return prose-wrapped JSON and trip the parse
    retry loop. A regression that swallowed json_mode would silently
    increase agent iterations and cost.
    """
    mocker.patch("config.settings.settings.llm_provider", "openai")
    ollama_mock = mocker.patch("agent.llm._call_ollama")
    openai_mock = mocker.patch("agent.llm._call_openai", return_value="from-openai")
    anthropic_mock = mocker.patch("agent.llm._call_anthropic")

    from agent.llm import _call_llm

    result = _call_llm("sys", "user", json_mode=True)

    assert result == "from-openai"
    openai_mock.assert_called_once_with("sys", "user", json_mode=True)
    ollama_mock.assert_not_called()
    anthropic_mock.assert_not_called()


def test_call_llm_dispatches_to_anthropic(mocker: Any) -> None:
    """provider='anthropic' must route to the Anthropic implementation."""
    mocker.patch("config.settings.settings.llm_provider", "anthropic")
    ollama_mock = mocker.patch("agent.llm._call_ollama")
    openai_mock = mocker.patch("agent.llm._call_openai")
    anthropic_mock = mocker.patch(
        "agent.llm._call_anthropic", return_value="from-anthropic"
    )

    from agent.llm import _call_llm

    result = _call_llm("sys", "user", json_mode=True)

    assert result == "from-anthropic"
    anthropic_mock.assert_called_once_with("sys", "user")
    ollama_mock.assert_not_called()
    openai_mock.assert_not_called()


def test_call_llm_raises_for_unsupported_provider(mocker: Any) -> None:
    """An unrecognised provider must raise RuntimeError with a clear message.

    The pydantic Literal in Settings prevents this at load time for
    valid configs, but tests and runtime patches can still slip a
    bad value through. The error must name the offending provider so
    operators don't have to introspect ``settings.llm_provider``.
    """
    mocker.patch("config.settings.settings.llm_provider", "gemini")

    from agent.llm import _call_llm

    with pytest.raises(RuntimeError, match="Unsupported llm_provider 'gemini'"):
        _call_llm("sys", "user")


def test_call_openai_raises_when_api_key_missing(mocker: Any) -> None:
    """An empty OpenAI key must surface as a RuntimeError before any HTTP call.

    Without this guard, the request would fire with an empty Bearer
    token and OpenAI would return 401 — which now (post-commit-2)
    propagates immediately as HTTPError. Either path errors, but the
    pre-flight RuntimeError is the actionable one because it names
    the missing setting.
    """
    mocker.patch("config.settings.settings.openai_api_key", SecretStr(""))
    post_mock = mocker.patch("requests.post")

    from agent.llm import _call_openai

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured"):
        _call_openai("sys", "user")
    post_mock.assert_not_called(), (
        "The missing-key check must short-circuit before any HTTP call"
    )


def test_call_anthropic_raises_when_api_key_missing(mocker: Any) -> None:
    """An empty Anthropic key must surface as a RuntimeError, not a 401."""
    mocker.patch("config.settings.settings.anthropic_api_key", SecretStr(""))
    post_mock = mocker.patch("requests.post")

    from agent.llm import _call_anthropic

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not configured"):
        _call_anthropic("sys", "user")
    post_mock.assert_not_called()
