"""Run every MLflow eval experiment in the planned E1–E4 matrix.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --only e2_openai_embed
    python scripts/run_experiments.py --start-from e4_top_k_3
    python scripts/run_experiments.py --dry-run

Loops over the runs defined in ``RUNS`` below and invokes
``python scripts/run_eval.py --experiment <name> --run <run_name>`` per
run, with per-process env vars overriding the parameter under test.

Each scientific experiment (E1–E4) gets its own MLflow experiment with
the baseline re-logged inside it, so each experiment is self-contained
in the MLflow UI.

Stops on the first run failure to avoid burning OpenAI judge tokens on a
misconfigured sweep. Re-run with ``--start-from <name>`` to resume.

E4 covers only the ``top_k`` axis. The original plan included
``similarity_threshold`` variants too, but Chroma's underlying squared
L2 distance metric would require a conversion layer to compare against
a cosine-style threshold, and that overhead isn't worth the experiment.
The parameter has been dropped from the eval matrix.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.errors import NotFoundError

# Allow `python scripts/run_experiments.py` to import top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_eval.py"

# Collection names must match what scripts/run_ablation_ingestion.py built.
BASELINE_COLLECTION = "hkia_nomic-embed-text_recursive_v4"
SECTION_COLLECTION = "hkia_nomic-embed-text_section_v1"
OPENAI_EMBED_COLLECTION = "hkia_text-embedding-3-small_recursive_v1"

# E3 Anthropic variant — Sonnet 4.6 is the current Sonnet tier and the
# reasoning peer of gpt-4o (the E3 OpenAI variant). Bump when a newer
# Sonnet ships and the experiment is worth re-running.
CLAUDE_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class EvalRun:
    """One eval run — an MLflow experiment+run pair with parameter overrides.

    Defaults match the project baseline (recursive chunking,
    nomic-embed-text, gpt-4o-mini, top_k=5). Each variant overrides only
    the field(s) under test for its experiment.
    """

    experiment: str
    name: str
    description: str
    chroma_collection_name: str = BASELINE_COLLECTION
    chunking_strategy: Literal["recursive", "section"] = "recursive"
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_model_version: str = "v1.5"
    llm_provider: Literal["ollama", "openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    retrieval_top_k: int = 5


RUNS: list[EvalRun] = [
    # E1 — Chunking strategy
    EvalRun(
        experiment="hkia_e1_chunking",
        name="e1_baseline_recursive",
        description="E1 baseline: recursive chunking, 512/64",
    ),
    EvalRun(
        experiment="hkia_e1_chunking",
        name="e1_section",
        description="E1 variant: section chunking",
        chroma_collection_name=SECTION_COLLECTION,
        chunking_strategy="section",
    ),
    # E2 — Embedding model
    EvalRun(
        experiment="hkia_e2_embedding",
        name="e2_baseline_nomic",
        description="E2 baseline: nomic-embed-text (Ollama)",
    ),
    EvalRun(
        experiment="hkia_e2_embedding",
        name="e2_openai_embed",
        description="E2 variant: text-embedding-3-small (OpenAI)",
        chroma_collection_name=OPENAI_EMBED_COLLECTION,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_model_version="3",
    ),
    # E3 — Agent LLM
    EvalRun(
        experiment="hkia_e3_llm",
        name="e3_baseline_gpt4o_mini",
        description="E3 baseline: gpt-4o-mini",
    ),
    EvalRun(
        experiment="hkia_e3_llm",
        name="e3_gpt4o",
        description="E3 variant: gpt-4o",
        llm_model="gpt-4o",
    ),
    EvalRun(
        experiment="hkia_e3_llm",
        name="e3_claude_sonnet",
        description=f"E3 variant: {CLAUDE_MODEL} (Anthropic)",
        llm_provider="anthropic",
        llm_model=CLAUDE_MODEL,
    ),
    # E4 — Retrieval top_k
    EvalRun(
        experiment="hkia_e4_retrieval",
        name="e4_baseline_topk5",
        description="E4 baseline: top_k=5",
    ),
    EvalRun(
        experiment="hkia_e4_retrieval",
        name="e4_top_k_3",
        description="E4 variant: top_k=3",
        retrieval_top_k=3,
    ),
    EvalRun(
        experiment="hkia_e4_retrieval",
        name="e4_top_k_10",
        description="E4 variant: top_k=10",
        retrieval_top_k=10,
    ),
]

Status = Literal["succeeded", "failed"]


def _collection_has_data(collection_name: str) -> bool:
    """Return True if the named Chroma collection exists and has chunks.

    Clears the SharedSystemClient cache before returning so the parent's
    SQLite handle is released before any subprocess opens its own client.
    """
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(name=collection_name)
        has_data = col.count() > 0
    except NotFoundError:
        has_data = False
    finally:
        SharedSystemClient.clear_system_cache()
    return has_data


def _preflight(runs: list[EvalRun]) -> None:
    """Validate API keys and collection existence before any eval starts.

    Exits non-zero with an actionable message on the first failed check.
    """
    openai_env = os.environ.get("OPENAI_API_KEY")
    openai_dotenv = settings.openai_api_key.get_secret_value()
    if not (openai_env or openai_dotenv):
        sys.exit(
            "OPENAI_API_KEY is unset but required: MLflow's GenAI judges "
            "are OpenAI-only regardless of the agent's LLM provider."
        )

    needs_anthropic = any(r.llm_provider == "anthropic" for r in runs)
    if needs_anthropic:
        anthropic_env = os.environ.get("ANTHROPIC_API_KEY")
        anthropic_dotenv = settings.anthropic_api_key.get_secret_value()
        if not (anthropic_env or anthropic_dotenv):
            offenders = [r.name for r in runs if r.llm_provider == "anthropic"]
            sys.exit(
                f"ANTHROPIC_API_KEY is unset but required for run(s): "
                f"{', '.join(offenders)}."
            )

    collections = {r.chroma_collection_name for r in runs}
    missing = sorted(c for c in collections if not _collection_has_data(c))
    if missing:
        sys.exit(
            f"Collection(s) missing or empty in {settings.chroma_persist_dir}: "
            f"{', '.join(missing)}. Run "
            f"`python scripts/run_ablation_ingestion.py` first."
        )


def _run_eval(run: EvalRun) -> Status:
    """Invoke ``scripts/run_eval.py`` for a single eval run.

    Streams the child's stdout/stderr live. Returns ``"succeeded"`` on
    exit code 0, ``"failed"`` otherwise.
    """
    env = os.environ.copy()
    env.update(
        {
            "CHROMA_COLLECTION_NAME": run.chroma_collection_name,
            "CHUNKING_STRATEGY": run.chunking_strategy,
            "EMBEDDING_PROVIDER": run.embedding_provider,
            "EMBEDDING_MODEL": run.embedding_model,
            "EMBEDDING_MODEL_VERSION": run.embedding_model_version,
            "LLM_PROVIDER": run.llm_provider,
            "LLM_MODEL": run.llm_model,
            "RETRIEVAL_TOP_K": str(run.retrieval_top_k),
        }
    )
    cmd = [
        sys.executable,
        str(RUN_EVAL_SCRIPT),
        "--experiment",
        run.experiment,
        "--run",
        run.name,
    ]
    print(f"\n=== Running '{run.name}' in experiment '{run.experiment}' ===")
    print(f"    {run.description}")
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=False)
    return "succeeded" if result.returncode == 0 else "failed"


def _select_runs(only: str | None, start_from: str | None) -> list[EvalRun]:
    """Filter ``RUNS`` by ``--only`` or ``--start-from``."""
    if only is not None:
        matches = [r for r in RUNS if r.name == only]
        if not matches:
            valid = ", ".join(r.name for r in RUNS)
            sys.exit(f"Unknown run '{only}'. Valid names: {valid}.")
        return matches
    if start_from is not None:
        names = [r.name for r in RUNS]
        if start_from not in names:
            valid = ", ".join(names)
            sys.exit(f"Unknown run '{start_from}'. Valid names: {valid}.")
        idx = names.index(start_from)
        return RUNS[idx:]
    return RUNS


def _print_dry_run(runs: list[EvalRun]) -> None:
    """Print the planned runs and the config knobs each one sets."""
    print(f"Would run {len(runs)} eval(s):\n")
    for r in runs:
        print(f"  {r.name}")
        print(f"    description: {r.description}")
        print(f"    experiment : {r.experiment}")
        print(f"    collection : {r.chroma_collection_name}")
        print(f"    chunking   : {r.chunking_strategy}")
        print(f"    embedding  : {r.embedding_provider}/{r.embedding_model}")
        print(f"    llm        : {r.llm_provider}/{r.llm_model}")
        print(f"    retrieval  : top_k={r.retrieval_top_k}")


def _print_summary(results: list[tuple[str, Status, str]]) -> None:
    """Print a one-line-per-run summary table."""
    if not results:
        print("\n=== No runs executed ===")
        return
    name_w = max(len(name) for name, _, _ in results)
    status_w = max(len(status) for _, status, _ in results)
    print("\n=== Experiment run summary ===")
    for name, status, experiment in results:
        print(f"  {name:<{name_w}}  {status:<{status_w}}  {experiment}")


def main() -> None:
    """Parse args, run preflight, then loop over eval runs.

    Deliberately does not call ``setup_logging()`` — the parent never
    logs (all output is ``print()``), and holding a RotatingFileHandler
    on ``logs/hkia.log`` blocks the child's log rotation on Windows.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run every MLflow eval experiment in the planned E1–E4 matrix."
        )
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run a single eval by name (e.g. e2_openai_embed).",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Resume from the named run, running it and all subsequent runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without executing them or hitting any API.",
    )
    args = parser.parse_args()

    if args.only and args.start_from:
        sys.exit("--only and --start-from are mutually exclusive.")

    runs = _select_runs(args.only, args.start_from)

    if args.dry_run:
        _print_dry_run(runs)
        return

    _preflight(runs)

    print(
        f"Planned: {len(runs)} eval run(s). Each iterates the golden set with "
        f"4 OpenAI judges — expect minutes per run and judge-token cost."
    )

    results: list[tuple[str, Status, str]] = []
    for run in runs:
        status = _run_eval(run)
        results.append((run.name, status, run.experiment))
        if status == "failed":
            print(
                f"\nRun '{run.name}' failed. Stopping sweep — fix the failure "
                f"and re-run with `--start-from {run.name}` to resume."
            )
            break

    _print_summary(results)
    if any(status == "failed" for _, status, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
