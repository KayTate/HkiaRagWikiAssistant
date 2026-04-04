"""Synthetic Q&A pair generation for the HKIA evaluation dataset.

Generates question/answer pairs from wiki chunks using an LLM, then
formats them for inclusion in the golden eval dataset.
"""

import json
import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are generating evaluation data for a RAG system about the game \
Hello Kitty Island Adventure. Given a wiki chunk and a question category, \
generate question/answer pairs that a player might ask.

Rules:
- Every question must be answerable solely from the provided chunk. \
Do not use outside knowledge.
- Every answer must be a complete standalone response.
- Questions must be specific, not vague. Bad: "What is crafting?" \
Good: "What materials do I need to craft a Wooden Bench?"
- Do not generate questions about information not present in the chunk.
- Generate fewer pairs for short or sparse chunks (1 pair if the chunk \
contains only one distinct fact).
- Return only valid JSON. No preamble, no markdown fences."""

USER_PROMPT_TEMPLATE = """Chunk:
{chunk_text}

Question category: {question_type}

Generate {n} question/answer pairs. Return as JSON array:
[{{"question": "...", "answer": "...", "question_type": "..."}}]"""

SHORT_CHUNK_TOKEN_THRESHOLD = 200
SHORT_CHUNK_PAIR_COUNT = 1
DEFAULT_PAIR_COUNT = 2

REQUIRED_PAIR_KEYS = {"question", "answer", "question_type"}


def _estimate_token_count(text: str) -> int:
    """Estimate token count by whitespace-splitting the text.

    A rough approximation sufficient for deciding whether a chunk is
    short enough to warrant fewer generated pairs.

    Args:
        text: The text to estimate token count for.

    Returns:
        Number of whitespace-delimited tokens in the text.
    """
    return len(text.split())


def _resolve_pair_count(chunk_text: str, n_pairs: int | None) -> int:
    """Determine how many pairs to generate for a given chunk.

    Defaults to 1 for short chunks (under the token threshold) and 2
    for longer ones, unless the caller explicitly overrides.

    Args:
        chunk_text: The chunk text being evaluated for length.
        n_pairs: Caller-supplied override; if not None, used as-is.

    Returns:
        The number of pairs to request from the LLM.
    """
    if n_pairs is not None:
        return n_pairs
    if _estimate_token_count(chunk_text) <= SHORT_CHUNK_TOKEN_THRESHOLD:
        return SHORT_CHUNK_PAIR_COUNT
    return DEFAULT_PAIR_COUNT


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present.

    The LLM sometimes wraps JSON in ```json ... ``` despite instructions
    not to. This strips those fences before attempting JSON parsing.

    Args:
        text: Raw LLM output.

    Returns:
        The text with markdown code fences removed.
    """
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)


def _validate_pairs(
    raw_pairs: object, chunk: dict[str, object]
) -> list[dict[str, object]]:
    """Validate that parsed pairs have the required fields.

    Args:
        raw_pairs: The parsed JSON object (expected to be a list).
        chunk: The source chunk, used for logging context.

    Returns:
        List of valid pair dicts. Malformed pairs are dropped with a
        warning rather than raising, so partial results are preserved.
    """
    if not isinstance(raw_pairs, list):
        metadata = chunk.get("metadata", {})
        source_title = (
            metadata.get("source_title", "unknown")
            if isinstance(metadata, dict)
            else "unknown"
        )
        logger.warning(
            "Expected a JSON array from LLM for chunk %r, got %s",
            source_title,
            type(raw_pairs).__name__,
        )
        return []

    validated: list[dict[str, object]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict pair: %r", item)
            continue
        missing = REQUIRED_PAIR_KEYS - item.keys()
        if missing:
            logger.warning("Skipping pair missing keys %s: %r", missing, item)
            continue
        validated.append(item)
    return validated


def _call_llm(prompt_user: str) -> str:
    """Send a prompt to the configured LLM and return the raw text response.

    Uses the ollama provider by default. Only ollama is supported because
    the eval pipeline is designed to run locally without paid API access.

    Args:
        prompt_user: The user-turn content to send.

    Returns:
        Raw text from the LLM response.

    Raises:
        RuntimeError: If the LLM provider is not 'ollama' or the call fails.
    """
    if settings.llm_provider != "ollama":
        raise RuntimeError(
            f"Synthetic generation only supports 'ollama' provider, "
            f"got '{settings.llm_provider}'. Set LLM_PROVIDER=ollama."
        )

    try:
        import ollama  # type: ignore[import-not-found]  # ollama may not be installed
    except ImportError as err:
        raise RuntimeError(
            "ollama package is required for synthetic generation. "
            "Install it with: uv add ollama"
        ) from err

    response = ollama.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ],
    )
    return str(response["message"]["content"])


def generate_for_chunk(
    chunk: dict[str, object],
    question_type: str,
    n_pairs: int | None = None,
) -> list[dict[str, object]]:
    """Generate Q&A pairs from a single wiki chunk using the LLM.

    The number of pairs defaults to 2 for chunks over 200 tokens and 1
    for shorter chunks, unless overridden by n_pairs.

    Args:
        chunk: A dict with at minimum a 'text' key containing the chunk
            content. Additional metadata keys are used for logging.
        question_type: The category label to pass to the LLM (e.g.
            'prerequisite', 'crafting', 'character').
        n_pairs: Override for the number of pairs to generate. If None,
            the count is chosen automatically based on chunk length.

    Returns:
        List of validated Q&A pair dicts with 'question', 'answer', and
        'question_type' keys. Returns an empty list if the LLM response
        cannot be parsed as valid JSON or yields no valid pairs.
    """
    chunk_text = str(chunk.get("text", ""))
    count = _resolve_pair_count(chunk_text, n_pairs)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        chunk_text=chunk_text,
        question_type=question_type,
        n=count,
    )

    try:
        raw_response = _call_llm(user_prompt)
    except RuntimeError:
        logger.exception("LLM call failed for chunk; returning empty list")
        return []

    cleaned = _strip_markdown_fences(raw_response)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        metadata = chunk.get("metadata", {})
        source_title = (
            metadata.get("source_title", "unknown")
            if isinstance(metadata, dict)
            else "unknown"
        )
        logger.warning(
            "Failed to parse LLM response as JSON for chunk %r: %r",
            source_title,
            cleaned[:200],
        )
        return []

    return _validate_pairs(parsed, chunk)
