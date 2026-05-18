"""Export MLflow traces for an experiment (or a single run) as CSV.

One CSV per run, written to ``data/eval/traces/<slug>.csv``. Pass
``--run`` to export a specific run; omit it to dump every run in the
experiment. Missing runs are warned about but do not abort the
remaining exports.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "traces"


def _resolve_tracking_uri() -> str:
    """Pick the tracking URI: env var, then sqlite:///mlflow.db, then settings."""
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    sqlite_db = PROJECT_ROOT / "mlflow.db"
    if sqlite_db.exists():
        return f"sqlite:///{sqlite_db.as_posix()}"
    return settings.mlflow_tracking_uri


def _slugify(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def _export_one_run(
    client: MlflowClient,
    exp_id: str,
    run_name: str,
    output_dir: Path,
) -> int:
    """Export traces for a single run name. Returns trace count (0 if missing)."""
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=1,
    )
    if not runs:
        print(f"  WARN: no run named '{run_name}' found — skipping")
        return 0
    run_id = runs[0].info.run_id

    traces_df = mlflow.search_traces(
        locations=[exp_id],
        run_id=run_id,
        return_type="pandas",
    )
    out_path = output_dir / f"{_slugify(run_name)}.csv"
    traces_df.to_csv(out_path, index=False)
    n = len(traces_df)
    print(
        f"  {run_name!r} (run_id={run_id}) -> "
        f"{out_path.relative_to(PROJECT_ROOT)} ({n} traces)"
    )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export MLflow traces for an experiment or a specific run "
            "to CSV files under data/eval/traces/."
        )
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name (e.g. 'hkia_baseline').",
    )
    parser.add_argument(
        "--run",
        default=None,
        help=(
            "Specific run name to export (matches tags.mlflow.runName). "
            "If omitted, every run in the experiment is exported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    tracking_uri = _resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(
            f"Experiment '{args.experiment}' not found at {tracking_uri}. "
            "Check MLFLOW_TRACKING_URI or the mlflow.db location."
        )
    exp_id = experiment.experiment_id
    print(f"Experiment '{args.experiment}' id={exp_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        run_names = [args.run]
    else:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["attributes.start_time DESC"],
        )
        # An experiment can have multiple runs sharing a runName (re-runs
        # overwrite nothing in MLflow). Keep only the most recent of each
        # name so we don't repeatedly overwrite the same CSV.
        seen: set[str] = set()
        run_names = []
        for r in runs:
            name = r.data.tags.get("mlflow.runName", r.info.run_id)
            if name in seen:
                continue
            seen.add(name)
            run_names.append(name)
        if not run_names:
            raise SystemExit(
                f"No runs found in experiment '{args.experiment}'."
            )
        print(f"Exporting {len(run_names)} run(s): {run_names}")

    total = sum(
        _export_one_run(client, exp_id, name, args.output_dir)
        for name in run_names
    )
    print(f"Done. {total} traces exported across {len(run_names)} run(s).")


if __name__ == "__main__":
    main()
