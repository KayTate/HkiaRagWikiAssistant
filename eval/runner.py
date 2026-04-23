"""MLflow experiment runner for the HKIA RAG evaluation pipeline.

Uses mlflow.genai.evaluate() with built-in LLM judge scorers
(Correctness, RelevanceToQuery, Summarization, RetrievalGroundedness)
powered by OpenAI gpt-4o-mini as the judge model.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.entities
import pandas as pd

from agent.graph import compile_graph
from agent.state import AgentState

logger = logging.getLogger(__name__)

# Rate-limit guard for concurrent LLM calls during evaluation.
# Keeps both the agent (predict_fn) and judge (scorers) from hitting 429s.
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")

JUDGE_MODEL = "openai:/gpt-4o-mini"

QUESTION_TYPES = [
    "prerequisite",
    "crafting",
    "friendship",
    "mechanic",
    "location",
    "general",
]

# Scorer metric names as they appear in the evaluate() results table.
SCORER_METRIC_NAMES = [
    "correctness/score",
    "relevance_to_query/score",
    "summarization/score",
    "retrieval_groundedness/score",
]


def _load_and_transform_dataset(
    dataset_path: str,
) -> list[dict[str, Any]]:
    """Load the golden set JSON and transform it for mlflow.genai.evaluate().

    Moves ``expected_response`` into ``expectations.expected_response``
    so the Correctness scorer can find ground truth.  Metadata is
    preserved at the top level (evaluate() ignores unrecognised keys).

    Args:
        dataset_path: Filesystem path to the golden JSON file.

    Returns:
        List of dicts in the format expected by evaluate().
    """
    raw_text = Path(dataset_path).read_text(encoding="utf-8")
    raw: list[dict[str, Any]] = json.loads(raw_text)

    transformed: list[dict[str, Any]] = []
    for entry in raw:
        transformed.append(
            {
                "inputs": entry["inputs"],
                "expectations": {
                    "expected_response": entry["expected_response"],
                },
                "metadata": entry.get("metadata", {}),
            }
        )
    return transformed


def _build_predict_fn() -> Any:
    """Compile the agent graph and return a predict function.

    MLflow LangChain autolog is enabled inside ``compile_graph()``,
    so traces are produced automatically for each invocation.

    Returns:
        A callable ``(inputs) -> str`` suitable for evaluate().
    """
    graph = compile_graph()

    def predict_fn(inputs: dict[str, str]) -> str:
        """Invoke the agent and return the final answer."""
        raw = graph.invoke({"question": inputs["question"]})
        state = AgentState(**raw)
        return state.final_answer

    return predict_fn


def _build_scorers() -> list[Any]:
    """Instantiate the four LLM judge scorers.

    Returns:
        List of scorer instances configured with the judge model.
    """
    from mlflow.genai.scorers import (
        Correctness,
        RelevanceToQuery,
        RetrievalGroundedness,
        Summarization,
    )

    return [
        Correctness(model=JUDGE_MODEL),
        RelevanceToQuery(model=JUDGE_MODEL),
        Summarization(model=JUDGE_MODEL),
        # RetrievalGroundedness requires MLflow traces with at least one
        # span whose span_type is RETRIEVER.  If the agent does not
        # produce such spans (e.g. because it doesn't use a LangChain
        # retriever natively), this scorer may silently return no results.
        RetrievalGroundedness(model=JUDGE_MODEL),
    ]


def _log_per_question_type_metrics(
    eval_results: Any,
    dataset: list[dict[str, Any]],
) -> None:
    """Compute and log per-question-type metric breakdowns.

    Groups rows by ``question_type`` from the original metadata and
    logs aggregated (mean) metrics with the ``.`` separator convention
    (e.g. ``correctness/score.prerequisite``).

    Args:
        eval_results: The object returned by ``mlflow.genai.evaluate()``.
        dataset: The transformed dataset list (with metadata).
    """
    results_df: pd.DataFrame = eval_results.tables["eval_results"]

    # Attach question_type from the original dataset to each result row.
    question_types = [
        entry.get("metadata", {}).get("question_type", "general") for entry in dataset
    ]
    if len(question_types) != len(results_df):
        logger.warning(
            "Dataset length (%d) != results length (%d); "
            "skipping per-question-type metrics.",
            len(question_types),
            len(results_df),
        )
        return

    results_df = results_df.copy()
    results_df["question_type"] = question_types

    for question_type in QUESTION_TYPES:
        subset = results_df[results_df["question_type"] == question_type]
        if subset.empty:
            continue
        for metric in SCORER_METRIC_NAMES:
            if metric not in subset.columns:
                continue
            col = pd.to_numeric(subset[metric], errors="coerce")
            mean_val = col.mean()
            if pd.isna(mean_val):
                continue
            # Use "." not "/" between metric and question type —
            # MLflow's file store treats "/" as a path separator.
            tag = metric.replace("/", "_")
            mlflow.log_metric(
                f"{tag}.{question_type}",
                float(mean_val),
            )


def run_experiment(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    dataset_path: str,
) -> mlflow.entities.Run:
    """Run an MLflow evaluation experiment using LLM judge scorers.

    Loads and transforms the golden set, invokes the RAG agent for
    each question, and scores responses with four LLM judge scorers
    via ``mlflow.genai.evaluate()``.

    Args:
        experiment_name: MLflow experiment name (created if absent).
        run_name: Display name for this run.
        params: Hyperparameters dict logged to MLflow.
        dataset_path: Path to the golden JSON dataset file.

    Returns:
        The completed mlflow.entities.Run object.
    """
    mlflow.set_experiment(experiment_name)

    dataset = _load_and_transform_dataset(dataset_path)
    predict_fn = _build_predict_fn()
    scorers = _build_scorers()

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(params)

        # Log dataset as an MLflow input artifact.
        mlflow_dataset = mlflow.data.from_pandas(
            pd.DataFrame(dataset),
            name="golden_set",
        )
        mlflow.log_input(mlflow_dataset, context="eval")

        eval_results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=predict_fn,
            scorers=scorers,
        )

        _log_per_question_type_metrics(eval_results, dataset)

        logger.info(
            "Experiment '%s' run '%s' complete. %d rows evaluated.",
            experiment_name,
            run_name,
            len(dataset),
        )

    client = mlflow.tracking.MlflowClient()
    return client.get_run(active_run.info.run_id)
