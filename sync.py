"""Entry point for wiki ingestion sync operations.

Usage:
    python sync.py --mode full        # full re-ingest of all pages
    python sync.py --mode incremental # default; only changed/new pages
    python sync.py --mode status      # print ingestion state summary
"""

import argparse

from config.logging_config import setup_logging
from ingestion.pipeline import run_full_ingestion, run_incremental_ingestion
from ingestion.state_db import get_status_summary

setup_logging()

parser = argparse.ArgumentParser(
    description="Sync wiki content into the HKIA RAG vector store."
)
parser.add_argument(
    "--mode",
    choices=["full", "incremental", "status"],
    default="incremental",
    help="Ingestion mode: full, incremental (default), or status report.",
)
args = parser.parse_args()

if args.mode == "full":
    run_full_ingestion()
elif args.mode == "incremental":
    run_incremental_ingestion()
elif args.mode == "status":
    print(get_status_summary())
