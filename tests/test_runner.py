"""Tests for eval/runner.py.

The runner uses mlflow.genai.evaluate() with LLM judge scorers against
a compiled agent graph. These tests patch the agent, judges, and
evaluate call so the runner can be exercised without network access.
"""

import json
import pathlib
from typing import Any

import mlflow
import pandas as pd

from eval.runner import (
    SCORER_METRIC_NAMES,
    _load_and_transform_dataset,
    run_experiment,
)

BASELINE_PARAMS: dict[str, object] = {
    "chunking_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_model": "nomic-embed-text",
    "llm_model": "gpt-4o",
    "retrieval_top_k": 5,
}

FIXTURE_DATASET = [
    {
        "inputs": {"question": "What do I need to complete to unlock Ice and Glow?"},
        "expected_response": (
            "To unlock Ice and Glow, you must first complete "
            "Straight to Your Heart and Absence Makes the Heart quest series."
        ),
        "metadata": {"source": "golden", "question_type": "prerequisite"},
    },
    {
        "inputs": {"question": "What gifts does Keroppi like?"},
        "expected_response": (
            "Keroppi likes frogs, nature items, and pond-related gifts."
        ),
        "metadata": {"source": "golden", "question_type": "friendship"},
    },
    {
        "inputs": {"question": "How do I craft a Wooden Bench?"},
        "expected_response": "You need 5 wood and 2 stone to craft a Wooden Bench.",
        "metadata": {"source": "golden", "question_type": "crafting"},
    },
]


def _write_fixture_dataset(tmp_path: pathlib.Path) -> str:
    """Write the fixture dataset to a temporary file and return its path."""
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps(FIXTURE_DATASET), encoding="utf-8")
    return str(path)


def _stub_eval_results(dataset_size: int) -> Any:
    """Return an object shaped like mlflow.genai.evaluate()'s return value.

    The runner reads ``tables["eval_results"]`` (a DataFrame) to compute
    per-question-type metric breakdowns, so the stub populates the same
    columns the judge scorers would.
    """
    df = pd.DataFrame({metric: [0.5] * dataset_size for metric in SCORER_METRIC_NAMES})

    class _StubResults:
        tables = {"eval_results": df}

    return _StubResults()


def _patch_runner_externals(mocker: Any, dataset_size: int) -> None:
    """Mock the agent graph, judges, and the evaluate call.

    Keeps run_experiment self-contained so tests don't hit Ollama,
    OpenAI, or the real ChromaDB.
    """
    mocker.patch(
        "eval.runner._build_predict_fn",
        return_value=lambda inputs: f"stub: {inputs['question']}",
    )
    mocker.patch("eval.runner._build_scorers", return_value=[])
    mocker.patch(
        "mlflow.genai.evaluate",
        return_value=_stub_eval_results(dataset_size),
    )


def test_load_and_transform_dataset_moves_expected_response(
    tmp_path: pathlib.Path,
) -> None:
    """_load_and_transform_dataset nests expected_response under expectations."""
    dataset_path = _write_fixture_dataset(tmp_path)

    transformed = _load_and_transform_dataset(dataset_path)

    assert len(transformed) == len(FIXTURE_DATASET)
    for original, result in zip(FIXTURE_DATASET, transformed, strict=True):
        assert result["inputs"] == original["inputs"]
        assert (
            result["expectations"]["expected_response"] == original["expected_response"]
        )
        assert result["metadata"] == original["metadata"]


def test_run_experiment_returns_mlflow_run_object(
    tmp_path: pathlib.Path, mocker: Any
) -> None:
    """run_experiment returns an mlflow.entities.Run instance."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    dataset_path = _write_fixture_dataset(tmp_path)
    _patch_runner_externals(mocker, dataset_size=len(FIXTURE_DATASET))

    result = run_experiment(
        experiment_name="test_hkia_eval_run_type",
        run_name="test_run_type",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
    )

    assert isinstance(result, mlflow.entities.Run)


def test_run_experiment_logs_params_to_mlflow(
    tmp_path: pathlib.Path, mocker: Any
) -> None:
    """run_experiment logs all params so they appear in the MLflow run data."""
    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    dataset_path = _write_fixture_dataset(tmp_path)
    _patch_runner_externals(mocker, dataset_size=len(FIXTURE_DATASET))

    returned_run = run_experiment(
        experiment_name="test_hkia_eval_params",
        run_name="test_params",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    logged_params = client.get_run(returned_run.info.run_id).data.params

    assert logged_params.get("chunking_strategy") == "recursive"
    assert logged_params.get("retrieval_top_k") == "5"
    assert logged_params.get("embedding_model") == "nomic-embed-text"


def test_run_experiment_logs_per_question_type_metrics(
    tmp_path: pathlib.Path, mocker: Any
) -> None:
    """run_experiment logs per-question-type breakdowns from judge scores."""
    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    dataset_path = _write_fixture_dataset(tmp_path)
    _patch_runner_externals(mocker, dataset_size=len(FIXTURE_DATASET))

    returned_run = run_experiment(
        experiment_name="test_hkia_eval_metrics",
        run_name="test_metrics",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    logged_metrics = client.get_run(returned_run.info.run_id).data.metrics

    # Each scorer metric should be broken down by every question type
    # present in the fixture dataset.
    present_types = {entry["metadata"]["question_type"] for entry in FIXTURE_DATASET}
    for metric in SCORER_METRIC_NAMES:
        tag = metric.replace("/", "_")
        for qtype in present_types:
            assert f"{tag}.{qtype}" in logged_metrics, (
                f"Expected per-question-type metric '{tag}.{qtype}' in MLflow run"
            )
