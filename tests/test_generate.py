"""Tests for eval/generate.py — synthetic Q&A pair generation helpers.

The pure helpers (token estimation, pair-count resolution, schema
validation) are exercised here. The end-to-end ``generate_for_chunk``
pipeline depends on a live Ollama process and is left for integration
testing; the local LLM-dispatch path inside this module is small
enough that the helper-level tests guard against the most likely
regressions.
"""

from typing import Any, cast

from eval.generate import (
    _estimate_token_count,
    _resolve_pair_count,
    _validate_pairs,
    generate_for_chunk,
)


def _question_of(pair: dict[str, object]) -> str:
    """Extract the question text from a validated pair for assertions.

    ``_validate_pairs`` returns ``list[dict[str, object]]``, so the
    nested ``inputs`` dict is opaque to mypy. Narrowing here keeps
    individual test bodies free of repeated ``cast`` boilerplate.
    """
    return cast(dict[str, str], pair["inputs"])["question"]

# ---------------------------------------------------------------------------
# _estimate_token_count
# ---------------------------------------------------------------------------


def test_estimate_token_count_uses_whitespace_split() -> None:
    """Whitespace count is the documented "rough approximation" contract.

    Pinning the heuristic catches a refactor that swapped to a real
    tokenizer (e.g. tiktoken) without updating the pair-count
    threshold — the new semantics could halve or double the
    short-chunk classification.
    """
    assert _estimate_token_count("one two three") == 3
    assert _estimate_token_count("a   b\tc\nd") == 4
    assert _estimate_token_count("") == 0


# ---------------------------------------------------------------------------
# _resolve_pair_count
# ---------------------------------------------------------------------------


def test_resolve_pair_count_short_chunk_returns_one() -> None:
    """Chunks at or below the threshold (200 tokens) get one pair.

    Cost-control invariant: short chunks rarely have enough material
    for two distinct questions. Generating two would either repeat
    information or hallucinate.
    """
    short_text = " ".join(["word"] * 50)
    assert _resolve_pair_count(short_text, n_pairs=None) == 1


def test_resolve_pair_count_long_chunk_returns_default_two() -> None:
    """Chunks above the threshold default to two pairs."""
    long_text = " ".join(["word"] * 500)
    assert _resolve_pair_count(long_text, n_pairs=None) == 2


def test_resolve_pair_count_caller_override_wins() -> None:
    """An explicit n_pairs override must bypass the heuristic.

    Documents the dataset-author escape hatch — a particular chunk
    can be tagged for more (or fewer) pairs than the heuristic
    would pick.
    """
    short_text = "tiny"
    long_text = " ".join(["word"] * 500)
    assert _resolve_pair_count(short_text, n_pairs=5) == 5
    assert _resolve_pair_count(long_text, n_pairs=1) == 1


def test_resolve_pair_count_threshold_is_inclusive() -> None:
    """Exactly 200 tokens counts as "short" (≤ threshold), not "long".

    Pin the boundary — a regression that flipped ≤ to < would
    silently bump 200-token chunks into the two-pair path and inflate
    eval-set generation cost.
    """
    boundary_text = " ".join(["word"] * 200)
    assert _resolve_pair_count(boundary_text, n_pairs=None) == 1


# ---------------------------------------------------------------------------
# _validate_pairs
# ---------------------------------------------------------------------------


def _chunk_for_logging() -> dict[str, object]:
    """Minimal chunk dict that satisfies the logger's source_title lookup."""
    return {"metadata": {"source_title": "Test"}}


def test_validate_pairs_accepts_well_formed_list() -> None:
    """All required keys present → all pairs survive."""
    raw = [
        {
            "inputs": {"question": "Q1"},
            "expected_response": "A1",
            "metadata": {"question_type": "crafting"},
        },
        {
            "inputs": {"question": "Q2"},
            "expected_response": "A2",
            "metadata": {"question_type": "friendship"},
        },
    ]
    assert _validate_pairs(raw, _chunk_for_logging()) == raw


def test_validate_pairs_rejects_non_list_input() -> None:
    """A dict where a list was expected returns [] with a warning logged.

    Documented behavior: parse-failure produces partial results
    rather than raising. Pinning here protects the upstream's
    "preserve what you can" contract.
    """
    result = _validate_pairs({"not": "a list"}, _chunk_for_logging())
    assert result == []


def test_validate_pairs_drops_non_dict_items() -> None:
    """A list with one valid dict and one string keeps only the dict.

    The function logs and skips bad items rather than rejecting the
    whole batch. This is a deliberate robustness choice — partial
    results are more useful than zero results when the LLM mostly
    behaved.
    """
    raw = [
        {
            "inputs": {"question": "Q1"},
            "expected_response": "A1",
            "metadata": {"question_type": "crafting"},
        },
        "not a dict",
    ]
    result = _validate_pairs(raw, _chunk_for_logging())
    assert len(result) == 1
    assert _question_of(result[0]) == "Q1"


def test_validate_pairs_drops_items_missing_required_keys() -> None:
    """A dict missing one of the top-level keys is dropped silently.

    Required keys: inputs, expected_response, metadata. A regression
    that relaxed the validation (e.g. allowing inputs alone) would let
    malformed entries flow into the golden dataset.
    """
    raw = [
        {
            "inputs": {"question": "Q1"},
            "expected_response": "A1",
            "metadata": {"question_type": "crafting"},
        },
        {"inputs": {"question": "Q2"}, "expected_response": "A2"},  # missing metadata
        {"inputs": {"question": "Q3"}},  # missing expected_response + metadata
    ]
    result = _validate_pairs(raw, _chunk_for_logging())
    assert [_question_of(p) for p in result] == ["Q1"]


def test_validate_pairs_drops_items_with_malformed_inner_shape() -> None:
    """Top-level keys alone aren't enough — inner shape must match too.

    Guards against an LLM that hits the outer keys but fills them with
    the wrong types (e.g. ``inputs`` as a string, or ``metadata``
    without ``question_type``).
    """
    raw = [
        {
            "inputs": "Q1",  # should be a dict with 'question'
            "expected_response": "A1",
            "metadata": {"question_type": "crafting"},
        },
        {
            "inputs": {"question": "Q2"},
            "expected_response": 42,  # should be a str
            "metadata": {"question_type": "crafting"},
        },
        {
            "inputs": {"question": "Q3"},
            "expected_response": "A3",
            "metadata": {"source": "synthetic"},  # missing question_type
        },
        {
            "inputs": {"question": "Q4"},
            "expected_response": "A4",
            "metadata": {"question_type": "crafting"},
        },
    ]
    result = _validate_pairs(raw, _chunk_for_logging())
    assert [_question_of(p) for p in result] == ["Q4"]


# ---------------------------------------------------------------------------
# generate_for_chunk — end-to-end with mocked LLM
# ---------------------------------------------------------------------------


def test_generate_for_chunk_returns_validated_pairs(mocker: Any) -> None:
    """A successful LLM call must flow through validation to the caller.

    Confirms the happy-path wiring: prompt formatting → LLM call →
    fence stripping → JSON parse → validation. A regression in any
    of those steps would fail this assertion.
    """
    mocker.patch(
        "eval.generate._call_llm",
        return_value=(
            '[{"inputs": {"question": "Q1"}, '
            '"expected_response": "A1", '
            '"metadata": {"question_type": "crafting"}}]'
        ),
    )
    chunk = {
        "text": "Wooden Bench requires 5 wood and 2 stone.",
        "metadata": {"source_title": "Wooden Bench"},
    }

    result = generate_for_chunk(chunk, question_type="crafting", n_pairs=1)

    assert len(result) == 1
    assert _question_of(result[0]) == "Q1"
    assert result[0]["expected_response"] == "A1"


def test_generate_for_chunk_handles_markdown_fenced_response(mocker: Any) -> None:
    """LLM responses wrapped in ``` json ... ``` must be stripped before parsing.

    This is the load-bearing dependency on
    ``agent.extraction.strip_markdown_fences`` introduced in commit 4.
    Without the fence-stripping, json.loads raises and
    generate_for_chunk returns []. Pinning here guards the
    cross-package import.
    """
    mocker.patch(
        "eval.generate._call_llm",
        return_value=(
            "```json\n"
            '[{"inputs": {"question": "Q1"}, '
            '"expected_response": "A1", '
            '"metadata": {"question_type": "crafting"}}]\n'
            "```"
        ),
    )
    chunk = {"text": "content", "metadata": {"source_title": "X"}}

    result = generate_for_chunk(chunk, question_type="crafting", n_pairs=1)

    assert len(result) == 1
    assert _question_of(result[0]) == "Q1"


def test_generate_for_chunk_returns_empty_list_on_parse_failure(
    mocker: Any,
) -> None:
    """Unparseable JSON returns [] rather than raising into the caller.

    The dataset-generation script processes many chunks; a single
    bad LLM response should not abort the run. Documented behavior.
    """
    mocker.patch(
        "eval.generate._call_llm",
        return_value="totally not valid json at all",
    )
    chunk = {"text": "content", "metadata": {"source_title": "X"}}

    assert generate_for_chunk(chunk, question_type="crafting", n_pairs=1) == []


def test_generate_for_chunk_returns_empty_list_on_llm_runtime_error(
    mocker: Any,
) -> None:
    """A RuntimeError from the LLM dispatcher must be caught, not propagated.

    Same rationale: one chunk's failure must not abort a
    multi-chunk generation run. The function logs and continues.
    """
    mocker.patch(
        "eval.generate._call_llm",
        side_effect=RuntimeError("ollama unreachable"),
    )
    chunk = {"text": "content", "metadata": {"source_title": "X"}}

    assert generate_for_chunk(chunk, question_type="crafting", n_pairs=1) == []


