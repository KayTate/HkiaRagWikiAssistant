"""LLM provider clients for the HKIA agent.

Dispatches chat completions to the configured provider (Ollama, OpenAI,
or Anthropic) and wraps each call in a tenacity retry decorator tuned
to the provider's failure characteristics. The dispatcher (``_call_llm``)
is the single entry point used by ``agent.nodes`` — node code should not
talk to a specific provider directly.

The constants and functions keep their leading-underscore prefix because
they are package-internal: consumed only by ``agent.nodes`` (via
``_call_llm_and_log``) and not part of any public API. Tests patch
``agent.nodes._call_llm`` rather than ``agent.llm._call_llm`` because
nodes.py imports the function and that is the binding the production
call sites resolve.
"""

import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)


def _call_llm(system_prompt: str, user_message: str, json_mode: bool = False) -> str:
    """Send a chat request to the configured LLM provider.

    Dispatches to Ollama, OpenAI, or Anthropic based on settings.llm_provider.
    Returns the assistant's reply as a plain string.

    Args:
        system_prompt: Instruction context for the LLM.
        user_message: The user-facing message or question.
        json_mode: When True and the provider supports it, request a
            strict JSON object response. Currently only wired up for
            OpenAI; silently ignored for Ollama and Anthropic.

    Returns:
        The LLM's text response.

    Raises:
        RuntimeError: If the LLM provider returns an unexpected response or
            if an unsupported provider is configured.
    """
    if settings.llm_provider == "ollama":
        # json_mode accepted but not wired up for Ollama yet.
        return _call_ollama(system_prompt, user_message)
    if settings.llm_provider == "openai":
        return _call_openai(system_prompt, user_message, json_mode=json_mode)
    if settings.llm_provider == "anthropic":
        # json_mode accepted but not wired up for Anthropic yet.
        return _call_anthropic(system_prompt, user_message)
    raise RuntimeError(
        f"Unsupported llm_provider '{settings.llm_provider}'. "
        "Expected one of: ollama, openai, anthropic."
    )


def _call_ollama(system_prompt: str, user_message: str) -> str:
    """Call the local Ollama chat API with automatic retry on transient errors.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.

    Returns:
        The model's reply text.

    Raises:
        requests.RequestException: On network/HTTP failure after retries
            are exhausted.
        RuntimeError: On unexpected response shape.
    """
    return _call_ollama_with_retry(system_prompt, user_message)


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_ollama_with_retry(system_prompt: str, user_message: str) -> str:
    """Send a chat request to Ollama with retry on connection/timeout errors.

    Uses a tighter backoff than the cloud providers because Ollama runs
    locally — long waits don't help recover a hung or absent server, and
    fast feedback is preferable.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.

    Returns:
        The model's reply text.

    Raises:
        requests.RequestException: On network failure after retries exhausted.
        RuntimeError: On unexpected response shape.
    """
    url = "http://localhost:11434/api/chat"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["message"]["content"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Ollama response missing expected fields for model "
            f"'{settings.llm_model}': {exc}"
        ) from exc
    except requests.RequestException as exc:
        logger.warning("Ollama request failed, will retry: %s", exc)
        raise


def _call_openai(system_prompt: str, user_message: str, json_mode: bool = False) -> str:
    """Call the OpenAI chat completions API with retry on rate limits.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.
        json_mode: When True, request a strict JSON object response via
            the OpenAI response_format parameter.

    Returns:
        The model's reply text.

    Raises:
        RuntimeError: On HTTP error, missing API key, or unexpected response.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured but llm_provider is 'openai'."
        )
    return _call_openai_with_retry(system_prompt, user_message, json_mode=json_mode)


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_openai_with_retry(
    system_prompt: str, user_message: str, json_mode: bool = False
) -> str:
    """Send a chat request to OpenAI with automatic retry on 429.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.
        json_mode: When True, include response_format={"type":
            "json_object"} in the request so OpenAI guarantees a valid
            JSON object response.

    Returns:
        The model's reply text.

    Raises:
        requests.HTTPError: On rate limit after retries exhausted.
        RuntimeError: On unexpected response shape.
    """
    url = "https://api.openai.com/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        if response.status_code == 429:
            logger.warning("OpenAI rate limit hit, will retry with backoff")
            response.raise_for_status()
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"OpenAI response missing expected fields for model "
            f"'{settings.llm_model}': {exc}"
        ) from exc
    except requests.HTTPError:
        raise
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenAI request failed calling '{url}': {exc}") from exc


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """Call the Anthropic Messages API with automatic retry on rate limits.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.

    Returns:
        The model's reply text.

    Raises:
        RuntimeError: On missing API key, unexpected response, or non-HTTP
            network failure.
        requests.HTTPError: On HTTP error after retries are exhausted.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured but llm_provider is 'anthropic'."
        )
    return _call_anthropic_with_retry(system_prompt, user_message)


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_anthropic_with_retry(system_prompt: str, user_message: str) -> str:
    """Send a chat request to Anthropic with retry on 429 / transient HTTP errors.

    Mirrors the OpenAI retry shape (5 attempts, 2-60s exponential backoff)
    so all cloud LLM providers behave consistently under rate-limit pressure.

    Args:
        system_prompt: Instruction context.
        user_message: User turn content.

    Returns:
        The model's reply text.

    Raises:
        requests.HTTPError: On rate limit / HTTP error after retries exhausted.
        RuntimeError: On unexpected response shape.
    """
    url = "https://api.anthropic.com/v1/messages"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    try:
        response = requests.post(
            url,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if response.status_code == 429:
            logger.warning("Anthropic rate limit hit, will retry with backoff")
            response.raise_for_status()
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return str(data["content"][0]["text"])
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Anthropic response missing expected fields for model "
            f"'{settings.llm_model}': {exc}"
        ) from exc
    except requests.HTTPError:
        raise
    except requests.RequestException as exc:
        raise RuntimeError(f"Anthropic request failed calling '{url}': {exc}") from exc
