"""Synthetic Q&A pair generation for the HKIA evaluation dataset.

Generates question/answer pairs from wiki chunks using an LLM, then
formats them for inclusion in the golden eval dataset.
"""

import json
import logging

from agent.extraction import strip_markdown_fences
from agent.llm import _call_llm

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
- Return only valid JSON. No preamble, no markdown fences.

Return a JSON array of objects matching this exact shape:
[
  {
    "inputs": {"question": "<question text>"},
    "expected_response": "<answer text>",
    "metadata": {"question_type": "<category>"}
  }
]"""

USER_PROMPT_TEMPLATE = """Chunk:
{chunk_text}

Question category: {question_type}

Generate {n} question/answer pairs in the JSON array format described, \
setting "question_type" in metadata to "{question_type}"."""

SHORT_CHUNK_TOKEN_THRESHOLD = 200
SHORT_CHUNK_PAIR_COUNT = 1
DEFAULT_PAIR_COUNT = 2

REQUIRED_PAIR_KEYS = {"inputs", "expected_response", "metadata"}


def _estimate_token_count(text: str) -> int:
    """Rough token-count estimate via whitespace split."""
    return len(text.split())


def _resolve_pair_count(chunk_text: str, n_pairs: int | None) -> int:
    """Pick how many pairs to generate (caller override > short-chunk default)."""
    if n_pairs is not None:
        return n_pairs
    if _estimate_token_count(chunk_text) <= SHORT_CHUNK_TOKEN_THRESHOLD:
        return SHORT_CHUNK_PAIR_COUNT
    return DEFAULT_PAIR_COUNT


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

        inputs = item["inputs"]
        if not isinstance(inputs, dict) or not isinstance(
            inputs.get("question"), str
        ):
            logger.warning("Skipping pair with malformed 'inputs': %r", item)
            continue
        if not isinstance(item["expected_response"], str):
            logger.warning(
                "Skipping pair whose 'expected_response' is not a str: %r", item
            )
            continue
        metadata = item["metadata"]
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("question_type"), str
        ):
            logger.warning("Skipping pair with malformed 'metadata': %r", item)
            continue

        validated.append(item)
    return validated


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
        raw_response = _call_llm(SYSTEM_PROMPT, user_prompt)
    except RuntimeError:
        logger.exception("LLM call failed for chunk; returning empty list")
        return []

    cleaned = strip_markdown_fences(raw_response)

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
