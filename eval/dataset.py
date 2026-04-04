"""Dataset loading and validation for the HKIA evaluation pipeline."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_INPUT_KEYS = {"question"}
REQUIRED_TOP_LEVEL_KEYS = {"inputs", "expected_response", "metadata"}


def _validate_entry(entry: object, idx: int) -> dict[str, object]:
    """Validate a single dataset entry against the required schema.

    Ensures the entry is a dict with the required top-level keys, that
    'inputs' contains 'question', and that 'expected_response' and
    'metadata' are the correct types.

    Args:
        entry: The raw parsed object from the JSON file.
        idx: The zero-based index of this entry, used in error messages.

    Returns:
        The validated entry cast to a dict.

    Raises:
        ValueError: If any required key is missing or has the wrong type.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Entry {idx} must be a dict, got {type(entry).__name__}")

    missing = REQUIRED_TOP_LEVEL_KEYS - entry.keys()
    if missing:
        raise ValueError(f"Entry {idx} missing required keys: {missing}")

    inputs = entry["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError(
            f"Entry {idx} 'inputs' must be a dict, got {type(inputs).__name__}"
        )

    missing_input_keys = REQUIRED_INPUT_KEYS - inputs.keys()
    if missing_input_keys:
        raise ValueError(
            f"Entry {idx} 'inputs' missing required keys: {missing_input_keys}"
        )

    if not isinstance(entry["expected_response"], str):
        raise ValueError(
            f"Entry {idx} 'expected_response' must be a str, "
            f"got {type(entry['expected_response']).__name__}"
        )

    if not isinstance(entry["metadata"], dict):
        raise ValueError(
            f"Entry {idx} 'metadata' must be a dict, "
            f"got {type(entry['metadata']).__name__}"
        )

    return entry


def load_dataset(path: str) -> list[dict[str, object]]:
    """Load and validate the evaluation dataset from a JSON file.

    Expects a JSON array where each element has 'inputs' (dict with
    'question'), 'expected_response' (str), and 'metadata' (dict).

    Args:
        path: Filesystem path to the JSON dataset file.

    Returns:
        List of validated dataset entries.

    Raises:
        FileNotFoundError: If the file at path does not exist.
        ValueError: If the file contains invalid JSON or any entry fails
            schema validation.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in dataset file {path}: {err}") from err

    if not isinstance(raw, list):
        raise ValueError(
            f"Dataset file must contain a JSON array, got {type(raw).__name__}"
        )

    validated = [_validate_entry(entry, idx) for idx, entry in enumerate(raw)]
    logger.info("Loaded %d entries from %s", len(validated), path)
    return validated
