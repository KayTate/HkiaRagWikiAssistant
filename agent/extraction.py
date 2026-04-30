"""Pure text-extraction helpers for the HKIA agent.

Functions in this module convert text in (a user question, an LLM
response) to text out (an entity name, a fence-stripped string). They
are deliberately scoped to leaf utilities — no AgentState, no logging,
no LLM calls — so they can be tested in isolation and shared between
node-level orchestration in ``agent.nodes`` and any future caller.

Stateful extraction orchestration (the JSON-parse retry loop, the
state-mutating apply/handle helpers) lives in ``agent.nodes`` because
it depends on AgentState and the node-level observability wrappers.

Constants and functions keep their leading-underscore prefix because
they are package-internal — consumed only by ``agent.nodes`` and not
part of any public API.
"""

import re

_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_TRAILING_DESCRIPTOR_RE = re.compile(
    r"\s+(quest\s+series|quest|recipe|item|character|location|"
    r"companion|ability|page|series)s?$",
    re.IGNORECASE,
)


def _normalize_entity(raw: str) -> str:
    """Strip leading articles and trailing descriptors from an extracted entity.

    Iterates until stable because an entity can contain both (e.g.
    'the Wild Mountain Time quest series' → 'Wild Mountain Time').

    Args:
        raw: Raw captured entity text from the regex extractor.

    Returns:
        Cleaned entity name. May be empty string if input was only
        articles and descriptors.
    """
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
        r"craft\s+(?:a\s+|an\s+)?(.+?)(?:\?|$|\.|\,)",
        r"unlock\s+(.+?)(?:\?|$|\.|\,)",
        r"complete\s+(.+?)(?:\?|$|\.|\,)",
        r"find\s+(.+?)(?:\?|$|\.|\,)",
        r"get\s+(?:to\s+)?(.+?)(?:\?|$|\.|\,)",
        r"reach\s+(.+?)(?:\?|$|\.|\,)",
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


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from an LLM response.

    LLMs often wrap JSON output in ```json ... ``` fences despite being
    told not to. This strips them so json.loads can parse the content.

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
