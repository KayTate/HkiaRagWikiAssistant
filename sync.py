"""Entry point for wiki ingestion sync operations.

Usage:
    python sync.py --mode full           # full re-ingest of all pages
    python sync.py --mode incremental    # default; only changed/new pages
    python sync.py --mode status         # print ingestion state summary
    python sync.py --mode replay --snapshot snapshots/<file>.parquet
"""

import argparse
from pathlib import Path

from config.logging_config import setup_logging
from ingestion.pipeline import (
    run_full_ingestion,
    run_incremental_ingestion,
    run_ingestion_from_snapshot,
)
from ingestion.state_db import get_status_summary

setup_logging()

parser = argparse.ArgumentParser(
    description="Sync wiki content into the HKIA RAG vector store."
)
parser.add_argument(
    "--mode",
    choices=["full", "incremental", "status", "replay"],
    default="incremental",
    help=(
        "Ingestion mode: full, incremental (default), status report, or "
        "replay (from a captured snapshot)."
    ),
)
parser.add_argument(
    "--snapshot",
    type=Path,
    default=None,
    help="Path to a Parquet snapshot file. Required when --mode replay.",
)
args = parser.parse_args()

if args.mode == "full":
    run_full_ingestion()
elif args.mode == "incremental":
    run_incremental_ingestion()
elif args.mode == "status":
    print(get_status_summary())
elif args.mode == "replay":
    if args.snapshot is None:
        parser.error("--mode replay requires --snapshot <path>")
    run_ingestion_from_snapshot(args.snapshot)
