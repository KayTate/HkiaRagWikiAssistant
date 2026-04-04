"""Tests for eval/dataset.py."""

import json
import pathlib

import pytest

from eval.dataset import load_dataset


def _write_dataset(tmp_path: pathlib.Path, data: object) -> str:
    """Write data as JSON to a temp file and return its path."""
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_dataset_returns_valid_entries(tmp_path: pathlib.Path) -> None:
    """load_dataset returns all entries when the file is well-formed."""
    data = [
        {
            "inputs": {"question": "What is Ice and Glow?"},
            "expected_response": "Ice and Glow is a quest.",
            "metadata": {"source": "golden", "question_type": "prerequisite"},
        },
        {
            "inputs": {"question": "How do I craft a bench?"},
            "expected_response": "Use wood and stone.",
            "metadata": {"source": "golden", "question_type": "crafting"},
        },
    ]
    path = _write_dataset(tmp_path, data)
    result = load_dataset(path)
    assert len(result) == 2
    assert result[0]["inputs"] == {"question": "What is Ice and Glow?"}


def test_load_dataset_raises_for_missing_file() -> None:
    """load_dataset raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset("/nonexistent/path/golden_set.json")


def test_load_dataset_raises_for_invalid_json(tmp_path: pathlib.Path) -> None:
    """load_dataset raises ValueError when the file contains invalid JSON."""
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_dataset(str(path))


def test_load_dataset_raises_for_non_array(tmp_path: pathlib.Path) -> None:
    """load_dataset raises ValueError when the top-level JSON value is not a list."""
    path = _write_dataset(tmp_path, {"oops": "not a list"})
    with pytest.raises(ValueError, match="JSON array"):
        load_dataset(str(path))


def test_load_dataset_raises_for_missing_top_level_key(tmp_path: pathlib.Path) -> None:
    """load_dataset raises ValueError when a required top-level key is absent."""
    data = [{"inputs": {"question": "q"}, "metadata": {}}]  # missing expected_response
    path = _write_dataset(tmp_path, data)
    with pytest.raises(ValueError, match="missing required keys"):
        load_dataset(str(path))


def test_load_dataset_raises_for_missing_question_key(tmp_path: pathlib.Path) -> None:
    """load_dataset raises ValueError when inputs dict has no 'question' key."""
    data = [
        {
            "inputs": {"not_question": "x"},
            "expected_response": "answer",
            "metadata": {},
        }
    ]
    path = _write_dataset(tmp_path, data)
    with pytest.raises(ValueError, match="missing required keys"):
        load_dataset(str(path))


def test_load_dataset_raises_for_non_string_expected_response(
    tmp_path: pathlib.Path,
) -> None:
    """load_dataset raises ValueError when expected_response is not a string."""
    data = [
        {
            "inputs": {"question": "q"},
            "expected_response": 42,
            "metadata": {},
        }
    ]
    path = _write_dataset(tmp_path, data)
    with pytest.raises(ValueError, match="'expected_response' must be a str"):
        load_dataset(str(path))
