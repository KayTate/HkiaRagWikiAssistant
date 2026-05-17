"""Pure text-extraction helpers for the HKIA agent: question → entity, LLM
response → fence-stripped string. Leaf utilities only — no AgentState, no
logging, no LLM calls."""

import re

_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAILING_DESCRIPTOR_RE = re.compile(
    r"\s+(quest\s+series|quest|recipe|item|character|location|"
    r"companion|ability|page|series)s?$",
    re.IGNORECASE,
)


def _normalize_entity(raw: str) -> str:
    """Strip leading articles and trailing descriptors; iterates until stable."""
    entity = raw.strip().strip("\"'")
    prev: str | None = None
    while prev != entity:
        prev = entity
        entity = _LEADING_ARTICLE_RE.sub("", entity).strip()
        entity = _TRAILING_DESCRIPTOR_RE.sub("", entity).strip()
    return entity


def _extract_entity_from_question(question: str) -> str | None:
    """Extract the most likely game entity name from a question.

    Uses simple heuristics: looks for quoted names first, then text
    following action verbs like 'craft', 'unlock', 'find', 'get', etc.
    All captures are normalized to strip articles and descriptors.
    Returns None if no entity can be confidently identified.

    Args:
        question: The original user question string.

    Returns:
        Entity name string or None if extraction fails.
    """
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        cleaned = _normalize_entity(str(quoted[0]))
        return cleaned or None

    patterns = [
        # Specific shapes go before generic verb patterns so they win the
        # first-match-wins race. "What items have the X tag?" pulls X for
        # a Tag-namespace lookup (paired with the " (Tag)" suffix in
        # agent.retrieval._TITLE_VARIANT_SUFFIXES).
        r"what\s+items?\s+have\s+(?:the\s+)?(.+?)\s+tag",
        r"when\s+is\s+(.+?)['’]s\s+birthday",
        r"what\s+does\s+(.+?)\s+(?:give|do|say)",
        r"which\s+(?:characters?|residents?|visitors?|fish|items?|quests?)"
        r"\s+(?:are|live|reside|can\s+i\s+find)\s+(?:typically\s+)?"
        r"(?:in|at|on)\s+(.+?)(?:\?|$|\.|\,)",
        r"what\s+(?:fish|items?|characters?|quests?)\s+can\s+i\s+find\s+in"
        r"\s+(.+?)(?:\?|$|\.|\,)",
        r"what\s+is\s+(?:an?|the)\s+(.+?)(?:\?|$|\.|\,)",
        r"how\s+does\s+(.+?)\s+work",
        # Existing verb-driven patterns.
        r"craft\s+(?:a\s+|an\s+)?(.+?)(?:\?|$|\.|\,)",
        r"unlock\s+(.+?)(?:\?|$|\.|\,)",
        r"complete\s+(.+?)(?:\?|$|\.|\,)",
        r"find\s+(.+?)(?:\?|$|\.|\,)",
        r"get\s+(?:to\s+)?(.+?)(?:\?|$|\.|\,)",
        r"reach\s+(.+?)(?:\?|$|\.|\,)",
        # New verbs surfaced by trace analysis: obtain, catch, make,
        # access, repair. "the boardwalk" / "a Burning Perch" article is
        # absorbed so _normalize_entity sees a bare entity.
        r"(?:obtain|catch|make|access|repair)\s+(?:a\s+|an\s+|the\s+)?"
        r"(.+?)(?:\?|$|\.|\,)",
        r"where\s+is\s+(.+?)(?:\?|$|\.|\,)",
        r"who\s+is\s+(.+?)(?:\?|$|\.|\,)",
        r"about\s+(.+?)(?:\?|$|\.|\,)",
        r"does\s+(.+?)\s+like",
        r"gifts?\s+(?:does\s+|for\s+)(.+?)(?:\?|$|\.|\,|\s+like)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            entity = _normalize_entity(match.group(1))
            if entity and entity.lower() not in {"i", "you", "it", "this"}:
                return entity

    return None


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from an LLM response.

    LLMs often wrap JSON output in ```json ... ``` fences despite being
    told not to. This strips them so json.loads can parse the content.
    Handles arbitrary language tags (```json, ```python, etc.) by
    discarding everything up to the first newline after the opening
    fence — a stricter regex like ``^```(?:json)?`` would leave the
    payload prefixed with the unrecognised language label.

    Args:
        text: Raw LLM response that may contain markdown fences.

    Returns:
        The text with leading/trailing fences removed.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()
