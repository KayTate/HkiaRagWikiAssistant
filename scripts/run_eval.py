"""Entry point for MLflow evaluation experiment runs.

Usage:
    python scripts/run_eval.py --experiment hkia_baseline --run baseline
    python scripts/run_eval.py --experiment hkia_baseline --run v2 \
        --dataset data/eval/golden_set.json

Params logged to MLflow are auto-populated from the active settings
(chunking, embedding, LLM, retrieval knobs) so every run captures the
full config snapshot.
"""

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_eval.py` to import the project's top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging  # noqa: E402
from config.settings import settings  # noqa: E402
from eval.runner import run_experiment  # noqa: E402


def _params_from_settings() -> dict[str, object]:
    """Collect the settings that influence eval results into an MLflow params dict.

    Returns:
        Mapping of param name to value — logged verbatim to MLflow so
        runs can be compared by config.
    """
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_model_version": settings.embedding_model_version,
        "chunking_strategy": settings.chunking_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_similarity_threshold": settings.retrieval_similarity_threshold,
        "agent_max_iterations": settings.agent_max_iterations,
        "chroma_collection_name": settings.chroma_collection_name,
    }


def main() -> None:
    """Parse CLI args and dispatch to run_experiment."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run an MLflow evaluation experiment against the golden set."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name (created if it doesn't exist).",
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Display name for this run.",
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/golden_set.json",
        help="Path to the golden JSON dataset (default: data/eval/golden_set.json).",
    )
    args = parser.parse_args()

    run = run_experiment(
        experiment_name=args.experiment,
        run_name=args.run,
        params=_params_from_settings(),
        dataset_path=args.dataset,
    )
    print(f"Run complete: {run.info.run_id}")


if __name__ == "__main__":
    main()
