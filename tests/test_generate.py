"""Tests for eval/generate.py — synthetic Q&A pair generation helpers.

The pure helpers (token estimation, pair-count resolution, schema
validation) are exercised here. The end-to-end ``generate_for_chunk``
pipeline depends on a live Ollama process and is left for integration
testing; the local LLM-dispatch path inside this module is small
enough that the helper-level tests guard against the most likely
regressions.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from eval.generate import (
    _estimate_token_count,
    _resolve_pair_count,
    _validate_pairs,
    generate_for_chunk,
)

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
        {"question": "Q1", "answer": "A1", "question_type": "crafting"},
        {"question": "Q2", "answer": "A2", "question_type": "friendship"},
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
        {"question": "Q1", "answer": "A1", "question_type": "crafting"},
        "not a dict",
    ]
    result = _validate_pairs(raw, _chunk_for_logging())
    assert len(result) == 1
    assert result[0]["question"] == "Q1"


def test_validate_pairs_drops_items_missing_required_keys() -> None:
    """A dict missing one of the three required keys is dropped silently.

    Required keys: question, answer, question_type. A regression that
    relaxed the validation (e.g. allowing question alone) would let
    malformed entries flow into the golden dataset.
    """
    raw = [
        {"question": "Q1", "answer": "A1", "question_type": "crafting"},
        {"question": "Q2", "answer": "A2"},  # missing question_type
        {"question": "Q3"},  # missing answer + question_type
    ]
    result = _validate_pairs(raw, _chunk_for_logging())
    questions = [p["question"] for p in result]
    assert questions == ["Q1"]


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
            '[{"question": "Q1", "answer": "A1", "question_type": "crafting"}]'
        ),
    )
    chunk = {
        "text": "Wooden Bench requires 5 wood and 2 stone.",
        "metadata": {"source_title": "Wooden Bench"},
    }

    result = generate_for_chunk(chunk, question_type="crafting", n_pairs=1)

    assert len(result) == 1
    assert result[0]["question"] == "Q1"


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
            '[{"question": "Q1", "answer": "A1", "question_type": "crafting"}]\n'
            "```"
        ),
    )
    chunk = {"text": "content", "metadata": {"source_title": "X"}}

    result = generate_for_chunk(chunk, question_type="crafting", n_pairs=1)

    assert len(result) == 1
    assert result[0]["question"] == "Q1"


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


# ---------------------------------------------------------------------------
# _call_llm — provider gating
# ---------------------------------------------------------------------------


def test_call_llm_raises_for_non_ollama_provider(mocker: Any) -> None:
    """eval/generate's local _call_llm only supports the Ollama backend.

    The synthetic-generation pipeline runs locally without paid API
    access; routing to OpenAI or Anthropic here would silently
    incur cloud costs. The guard surfaces the misconfiguration.
    """
    from eval.generate import _call_llm

    mocker.patch("config.settings.settings.llm_provider", "openai")

    with pytest.raises(RuntimeError, match="only supports 'ollama' provider"):
        _call_llm("user prompt")


def test_call_llm_invokes_ollama_chat_with_settings_model(mocker: Any) -> None:
    """The local _call_llm must pass settings.llm_model through to ollama.chat.

    Without this, a default model (or no model at all) would silently
    be used — invisible to the operator until results regressed.
    """
    from eval.generate import _call_llm

    mocker.patch("config.settings.settings.llm_provider", "ollama")
    mocker.patch("config.settings.settings.llm_model", "llama3")

    fake_ollama = MagicMock()
    fake_ollama.chat.return_value = {"message": {"content": "response text"}}
    mocker.patch.dict("sys.modules", {"ollama": fake_ollama})

    result = _call_llm("user prompt")

    assert result == "response text"
    fake_ollama.chat.assert_called_once()
    call_kwargs = fake_ollama.chat.call_args.kwargs
    assert call_kwargs["model"] == "llama3"
