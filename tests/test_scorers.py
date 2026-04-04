"""Tests for eval/scorers.py."""

from eval.scorers import exact_match, token_overlap


def test_exact_match_returns_one_for_identical_strings() -> None:
    """exact_match returns 1.0 when strings are identical after normalisation."""
    assert exact_match("Hello World", "Hello World") == 1.0


def test_exact_match_is_case_insensitive() -> None:
    """exact_match treats strings as equal regardless of case."""
    assert exact_match("Hello World", "hello world") == 1.0


def test_exact_match_strips_whitespace() -> None:
    """exact_match ignores leading/trailing whitespace."""
    assert exact_match("  answer  ", "answer") == 1.0


def test_exact_match_returns_zero_for_different_strings() -> None:
    """exact_match returns 0.0 for non-matching strings."""
    assert exact_match("Ice and Glow", "totally different") == 0.0


def test_token_overlap_returns_one_for_identical_text() -> None:
    """token_overlap returns 1.0 when both strings share identical tokens."""
    assert token_overlap("the quick brown fox", "the quick brown fox") == 1.0


def test_token_overlap_returns_zero_for_disjoint_tokens() -> None:
    """token_overlap returns 0.0 when the token sets have no intersection."""
    assert token_overlap("apple orange", "banana pear") == 0.0


def test_token_overlap_returns_partial_score_for_overlap() -> None:
    """token_overlap returns a fractional score for partial intersection."""
    score = token_overlap("the quick brown fox", "the slow brown dog")
    # intersection: {the, brown} = 2, union: {the, quick, brown, fox, slow, dog} = 6
    assert abs(score - 2 / 6) < 1e-9


def test_token_overlap_handles_both_empty_strings() -> None:
    """token_overlap returns 0.0 when both inputs are empty."""
    assert token_overlap("", "") == 0.0
