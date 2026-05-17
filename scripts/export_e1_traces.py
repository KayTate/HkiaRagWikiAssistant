"""Export MLflow traces for hkia_e1_chunking runs as CSV.

One CSV per run, written to data/eval/traces/. Targets the e1 experiment
(hkia_e1_chunking) runs. Missing runs are warned about but do not abort
the export of the remaining runs.
"""

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402

EXPERIMENT_NAME = "hkia_e1_chunking"
RUN_NAMES = ["e1_baseline_recursive", "e1_section"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "traces"


def _resolve_tracking_uri() -> str:
    """Pick the tracking URI: env var, then sqlite:///mlflow.db, then settings."""
    import os

    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    sqlite_db = PROJECT_ROOT / "mlflow.db"
    if sqlite_db.exists():
        return f"sqlite:///{sqlite_db.as_posix()}"
    return settings.mlflow_tracking_uri


def _slugify(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def main() -> None:
    tracking_uri = _resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(
            f"Experiment '{EXPERIMENT_NAME}' not found at {tracking_uri}."
        )
    exp_id = experiment.experiment_id
    print(f"Experiment '{EXPERIMENT_NAME}' id={exp_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_traces = 0
    for run_name in RUN_NAMES:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            max_results=1,
        )
        if not runs:
            print(f"  WARN: no run named '{run_name}' found — skipping")
            continue
        run_id = runs[0].info.run_id

        traces_df = mlflow.search_traces(
            locations=[exp_id],
            run_id=run_id,
            return_type="pandas",
        )

        out_path = OUTPUT_DIR / f"{_slugify(run_name)}.csv"
        traces_df.to_csv(out_path, index=False)
        n = len(traces_df)
        total_traces += n
        print(
            f"  {run_name!r} (run_id={run_id}) -> "
            f"{out_path.relative_to(PROJECT_ROOT)} ({n} traces)"
        )

    print(f"Done. {total_traces} traces exported across {len(RUN_NAMES)} runs.")


if __name__ == "__main__":
    main()
