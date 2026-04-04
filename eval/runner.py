"""MLflow experiment runner for the HKIA RAG evaluation pipeline.

Deviation from TDD Section 7.3: mlflow.genai.evaluate with built-in
scorers (answer_correctness, faithfulness, context_recall,
context_precision) requires a Databricks-hosted LLM judge endpoint that
is not available in a local environment. Instead, we implement a manual
evaluation loop using the local scorers in eval/scorers.py
(exact_match and token_overlap). Per the TDD's Component 5 fallback
guidance, this is acceptable and documented here.
"""

import logging
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.entities
import pandas as pd

from eval.dataset import load_dataset
from eval.scorers import exact_match, token_overlap

logger = logging.getLogger(__name__)

QUESTION_TYPES = ["prerequisite", "crafting", "character", "location", "general"]
METRICS = ["exact_match", "token_overlap"]


def _compute_row_scores(
    expected: str,
    actual: str,
) -> dict[str, float]:
    """Compute all local scorer metrics for a single prediction.

    Args:
        expected: Ground-truth answer from the dataset.
        actual: Model-generated answer.

    Returns:
        Dict mapping metric name to float score in [0.0, 1.0].
    """
    return {
        "exact_match": exact_match(expected, actual),
        "token_overlap": token_overlap(expected, actual),
    }


def _extract_question_type(entry: dict[str, object]) -> str:
    """Extract the question_type string from a dataset entry's metadata.

    Args:
        entry: A validated dataset entry dict.

    Returns:
        The question_type string, defaulting to 'general' if absent.
    """
    metadata = entry.get("metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("question_type", "general"))
    return "general"


def _extract_question(inputs: object) -> str:
    """Extract the question string from an inputs value.

    Args:
        inputs: The value at entry['inputs']; expected to be a dict.

    Returns:
        The question string, or empty string if inputs is not a dict.
    """
    if isinstance(inputs, dict):
        return str(inputs.get("question", ""))
    return ""


def _run_predictions(
    dataset: list[dict[str, object]],
    predict_fn: Callable[[dict[str, object]], str],
) -> pd.DataFrame:
    """Run predict_fn over every row in the dataset and collect scores.

    Args:
        dataset: Validated list of dataset entries.
        predict_fn: Callable that accepts an 'inputs' dict and returns
            a string answer.

    Returns:
        DataFrame with columns: question, expected_response,
        actual_response, question_type, exact_match, token_overlap.
    """
    rows: list[dict[str, Any]] = []
    for entry in dataset:
        inputs = entry["inputs"]
        expected = str(entry["expected_response"])
        question_type = _extract_question_type(entry)

        try:
            # inputs is typed object; predict_fn requires dict — safe because
            # load_dataset validates inputs is always a dict before returning.
            actual = predict_fn(inputs)  # type: ignore[arg-type]
        except Exception:
            logger.exception(
                "predict_fn raised for question %r; recording empty response",
                _extract_question(inputs),
            )
            actual = ""

        scores = _compute_row_scores(expected, actual)
        rows.append(
            {
                "question": _extract_question(inputs),
                "expected_response": expected,
                "actual_response": actual,
                "question_type": question_type,
                **scores,
            }
        )
    return pd.DataFrame(rows)


def _log_aggregate_metrics(results: pd.DataFrame) -> None:
    """Log overall and per-question-type aggregate metrics to the active MLflow run.

    Args:
        results: DataFrame produced by _run_predictions.
    """
    for metric in METRICS:
        if metric in results.columns:
            mlflow.log_metric(metric, float(results[metric].mean()))

    for question_type in QUESTION_TYPES:
        subset = results[results["question_type"] == question_type]
        if subset.empty:
            continue
        for metric in METRICS:
            if metric in subset.columns:
                # Use "." separator — MLflow's file store treats "/" as a path
                # separator, which conflicts with the top-level metric file.
                mlflow.log_metric(
                    f"{metric}.{question_type}",
                    float(subset[metric].mean()),
                )


def _log_dataset_artifact(
    dataset: list[dict[str, object]],
    dataset_path: str,
    version: str,
) -> None:
    """Log the evaluation dataset as an MLflow artifact dict.

    Args:
        dataset: The loaded dataset entries.
        dataset_path: Original path, included in artifact metadata.
        version: Version label used in the artifact key.
    """
    artifact = {
        "path": dataset_path,
        "version": version,
        "num_entries": len(dataset),
    }
    mlflow.log_dict(artifact, f"eval_dataset_{version}.json")


def run_experiment(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    dataset_path: str,
    predict_fn: Callable[[dict[str, object]], str] | None = None,
) -> mlflow.entities.Run:
    """Run a single MLflow evaluation experiment with specified parameters.

    Loads the dataset, runs predict_fn over all rows, scores each
    prediction with exact_match and token_overlap, logs all params and
    metrics to MLflow, and returns the completed Run.

    The predict_fn default invokes the RAG agent pipeline configured in
    settings. Pass a custom callable for testing or offline use.

    Args:
        experiment_name: MLflow experiment name. Created if it does not
            already exist.
        run_name: Display name for this specific run.
        params: Hyperparameter dict logged to MLflow (e.g.
            chunking_strategy, embedding_model, top_k).
        dataset_path: Path to the golden JSON dataset file.
        predict_fn: Callable that accepts an inputs dict (with at least
            a 'question' key) and returns a string answer. If None, a
            default RAG pipeline caller is constructed from settings.

    Returns:
        The completed mlflow.entities.Run object.
    """
    # The tracking URI must be configured by the caller before invoking
    # run_experiment (either via mlflow.set_tracking_uri or MLFLOW_TRACKING_URI
    # env var). Setting it here would override any test-specific URI the caller
    # has configured. At application startup, set it once from settings.
    mlflow.set_experiment(experiment_name)

    dataset = load_dataset(dataset_path)
    version = str(params.get("version", "v1"))

    if predict_fn is None:
        predict_fn = _build_default_predict_fn()

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(params)
        _log_dataset_artifact(dataset, dataset_path, version)

        results = _run_predictions(dataset, predict_fn)
        _log_aggregate_metrics(results)

        logger.info(
            "Experiment '%s' run '%s' complete. %d rows evaluated.",
            experiment_name,
            run_name,
            len(results),
        )

    # start_run context manager returns ActiveRun whose __exit__ returns the Run.
    # We retrieve the completed run via the client to get the typed Run object.
    client = mlflow.tracking.MlflowClient()
    return client.get_run(active_run.info.run_id)


def _build_default_predict_fn() -> Callable[[dict[str, object]], str]:
    """Build a predict function that calls the RAG agent pipeline.

    The agent graph import is deferred so the runner module can be
    imported and tested without a fully wired agent graph in place.

    Returns:
        A callable that accepts an inputs dict and returns the agent's
        final answer string.
    """

    def _predict(inputs: dict[str, object]) -> str:
        """Call the RAG agent graph for a single question.

        Args:
            inputs: Dict containing at minimum a 'question' key.

        Returns:
            The agent's final_answer string.
        """
        from agent.graph import build_graph  # lazy import — agent may not exist yet
        from agent.state import AgentState

        question = str(inputs.get("question", ""))
        graph = build_graph()
        state = AgentState(question=question)
        # LangGraph CompiledGraph exposes invoke() but mypy sees StateGraph.
        result: AgentState = graph.invoke(state)  # type: ignore[attr-defined]
        return result.final_answer

    return _predict
