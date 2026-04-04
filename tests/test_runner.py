"""Tests for eval/runner.py — acceptance criterion: eval runner completes a
full experiment run with baseline parameters and logs results to MLflow."""

import json
import pathlib

import mlflow

from eval.runner import run_experiment

BASELINE_PARAMS = {
    "chunking_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_model": "nomic-embed-text:v1.5",
    "llm_model": "llama3",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "version": "test_v1",
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
        "metadata": {"source": "golden", "question_type": "character"},
    },
    {
        "inputs": {"question": "How do I craft a Wooden Bench?"},
        "expected_response": "You need 5 wood and 2 stone to craft a Wooden Bench.",
        "metadata": {"source": "golden", "question_type": "crafting"},
    },
]


def _write_fixture_dataset(tmp_path: pathlib.Path) -> str:
    """Write the fixture dataset to a temporary file and return the path."""
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps(FIXTURE_DATASET), encoding="utf-8")
    return str(path)


def _stub_predict_fn(inputs: dict[str, object]) -> str:
    """Deterministic stub predict function that echoes part of the question."""
    question = str(inputs.get("question", ""))
    return f"Stub answer for: {question}"


def test_run_experiment_completes_without_error(tmp_path: pathlib.Path) -> None:
    """run_experiment completes with a stub predict_fn and no exceptions are raised."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    dataset_path = _write_fixture_dataset(tmp_path)

    returned_run = run_experiment(
        experiment_name="test_hkia_eval",
        run_name="test_baseline",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
        predict_fn=_stub_predict_fn,
    )

    assert returned_run is not None


def test_run_experiment_returns_mlflow_run_object(tmp_path: pathlib.Path) -> None:
    """run_experiment returns an mlflow.entities.Run instance."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    dataset_path = _write_fixture_dataset(tmp_path)

    result = run_experiment(
        experiment_name="test_hkia_eval_run_type",
        run_name="test_run_type",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
        predict_fn=_stub_predict_fn,
    )

    assert isinstance(result, mlflow.entities.Run)


def test_run_experiment_logs_params_to_mlflow(tmp_path: pathlib.Path) -> None:
    """run_experiment logs all params so they appear in the MLflow run data."""
    tracking_uri = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    dataset_path = _write_fixture_dataset(tmp_path)

    returned_run = run_experiment(
        experiment_name="test_hkia_eval_params",
        run_name="test_params",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
        predict_fn=_stub_predict_fn,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run_data = client.get_run(returned_run.info.run_id)
    logged_params = run_data.data.params

    assert logged_params.get("chunking_strategy") == "recursive"
    assert logged_params.get("top_k") == "5"
    assert logged_params.get("embedding_model") == "nomic-embed-text:v1.5"


def test_run_experiment_logs_metrics_to_mlflow(tmp_path: pathlib.Path) -> None:
    """run_experiment logs exact_match and token_overlap metrics to the run."""
    tracking_uri = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    dataset_path = _write_fixture_dataset(tmp_path)

    returned_run = run_experiment(
        experiment_name="test_hkia_eval_metrics",
        run_name="test_metrics",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
        predict_fn=_stub_predict_fn,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run_data = client.get_run(returned_run.info.run_id)
    logged_metrics = run_data.data.metrics

    assert "exact_match" in logged_metrics
    assert "token_overlap" in logged_metrics


def test_run_experiment_handles_predict_fn_exception(tmp_path: pathlib.Path) -> None:
    """run_experiment continues and records empty response when predict_fn raises."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    dataset_path = _write_fixture_dataset(tmp_path)

    def _failing_predict(inputs: dict[str, object]) -> str:
        raise RuntimeError("Simulated failure")

    returned_run = run_experiment(
        experiment_name="test_hkia_eval_failure",
        run_name="test_failure_handling",
        params=BASELINE_PARAMS,
        dataset_path=dataset_path,
        predict_fn=_failing_predict,
    )

    assert isinstance(returned_run, mlflow.entities.Run)
