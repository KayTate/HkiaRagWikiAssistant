"""Simple RAGAS-inspired scorers for the HKIA evaluation pipeline.

These scorers operate locally without external API calls, making them
suitable as fallback scorers when mlflow.genai built-in scorers are
unavailable or when running in offline environments.
"""


def exact_match(expected: str, actual: str) -> float:
    """Return 1.0 if the strings match after case-insensitive stripping.

    Useful as a strict correctness check for factual questions where
    the expected answer is short and well-defined.

    Args:
        expected: The ground-truth answer string.
        actual: The model-generated answer string.

    Returns:
        1.0 if the normalized strings are equal, 0.0 otherwise.
    """
    return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0


def token_overlap(expected: str, actual: str) -> float:
    """Return the Jaccard similarity of the word token sets.

    Useful as a soft faithfulness proxy when exact match is too strict.
    Tokenizes by whitespace and lowercases before comparison, so word
    order and punctuation attached to words will affect the score.

    Args:
        expected: The ground-truth answer string.
        actual: The model-generated answer string.

    Returns:
        Jaccard similarity in [0.0, 1.0]. Returns 0.0 if both strings
        are empty (empty intersection and empty union).
    """
    expected_tokens = set(expected.strip().lower().split())
    actual_tokens = set(actual.strip().lower().split())

    union = expected_tokens | actual_tokens
    if not union:
        return 0.0

    intersection = expected_tokens & actual_tokens
    return len(intersection) / len(union)
