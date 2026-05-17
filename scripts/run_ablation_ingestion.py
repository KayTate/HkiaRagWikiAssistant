"""Build every ChromaDB collection required for the planned E1–E4 experiments.

Usage:
    python scripts/run_ablation_ingestion.py
    python scripts/run_ablation_ingestion.py --snapshot snapshots/2026-05-12.parquet
    python scripts/run_ablation_ingestion.py --only e1_section
    python scripts/run_ablation_ingestion.py --force

Loops over the variant matrix defined in ``VARIANTS`` below and invokes
``python sync.py --mode replay --snapshot <path>`` once per variant with
per-process env vars set to the variant's chunking + embedding config.
Each variant gets its own ChromaDB collection and SQLite state DB so the
three collections coexist on disk and can be evaluated sequentially with
no further ingestion.

Variants only cover knobs that change ingestion output (chunking strategy,
embedding model). E3 (LLM) and E4 (top_k, similarity threshold) are
query-time only, so they reuse the baseline collection.

Existing collections that already contain chunks are skipped, making
re-runs idempotent. Pass ``--force`` to re-ingest regardless.
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

# Allow `python scripts/run_ablation_ingestion.py` to import top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = REPO_ROOT / "snapshots" / "2026-05-12.parquet"
SYNC_SCRIPT = REPO_ROOT / "sync.py"


@dataclass(frozen=True)
class Variant:
    """One ablation variant — a config snapshot the script ingests under."""

    name: str
    collection_name: str
    state_db_path: str
    chunking_strategy: Literal["recursive", "section"]
    chunk_size: int
    chunk_overlap: int
    embedding_provider: Literal["ollama", "openai"]
    embedding_model: str
    embedding_model_version: str


# Collection naming follows the documented convention
# ``hkia_{embedding_model}_{chunking_strategy}_v{n}``. The current live
# baseline is v3, so the rebuilt baseline below is v4. The other two
# variants are first builds of their (model, strategy) pair, so v1.
VARIANTS: list[Variant] = [
    Variant(
        name="baseline",
        collection_name="hkia_nomic-embed-text_recursive_v4",
        state_db_path="./data/state_recursive_v4.db",
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        embedding_model_version="v1.5",
    ),
    Variant(
        name="e1_section",
        collection_name="hkia_nomic-embed-text_section_v1",
        state_db_path="./data/state_section_v1.db",
        chunking_strategy="section",
        chunk_size=512,
        chunk_overlap=64,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        embedding_model_version="v1.5",
    ),
    Variant(
        name="e2_openai",
        collection_name="hkia_text-embedding-3-small_recursive_v1",
        state_db_path="./data/state_openai_v1.db",
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_model_version="3",
    ),
]

Status = Literal["skipped", "succeeded", "failed"]


def _collection_has_data(collection_name: str) -> bool:
    """Return True if the named Chroma collection exists and contains chunks.

    Used by the skip-if-exists check. The SharedSystemClient cache holds a
    strong reference to PersistentClient instances past function return,
    so we explicitly clear it before returning — otherwise the parent's
    SQLite handle stays open and can conflict with the child subprocess's
    writer on Windows.
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


def _run_variant(variant: Variant, snapshot_path: Path) -> Status:
    """Invoke ``sync.py --mode replay`` for a single variant.

    Streams ``sync.py``'s stdout/stderr directly so ingestion progress is
    visible live. Returns ``"succeeded"`` on exit code 0, ``"failed"``
    otherwise — the caller decides whether to stop the sweep.
    """
    env = os.environ.copy()
    env.update(
        {
            "CHROMA_COLLECTION_NAME": variant.collection_name,
            "STATE_DB_PATH": variant.state_db_path,
            "CHUNKING_STRATEGY": variant.chunking_strategy,
            "CHUNK_SIZE": str(variant.chunk_size),
            "CHUNK_OVERLAP": str(variant.chunk_overlap),
            "EMBEDDING_PROVIDER": variant.embedding_provider,
            "EMBEDDING_MODEL": variant.embedding_model,
            "EMBEDDING_MODEL_VERSION": variant.embedding_model_version,
        }
    )
    cmd = [
        sys.executable,
        str(SYNC_SCRIPT),
        "--mode",
        "replay",
        "--snapshot",
        str(snapshot_path),
    ]
    print(f"\n=== Ingesting variant '{variant.name}' → {variant.collection_name} ===")
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=False)
    return "succeeded" if result.returncode == 0 else "failed"


def _preflight(variants: list[Variant], snapshot_path: Path) -> None:
    """Validate snapshot existence and required credentials before ingesting.

    Exits with code 2 if any check fails — done before any subprocess
    invocation so a missing key doesn't surface three variants in.
    """
    if not snapshot_path.exists():
        sys.exit(
            f"Snapshot file not found: {snapshot_path}. Capture one with "
            f"`python scripts/snapshot_wiki.py --output {snapshot_path}`."
        )
    needs_openai = any(v.embedding_provider == "openai" for v in variants)
    if needs_openai:
        env_key = os.environ.get("OPENAI_API_KEY")
        dotenv_key = settings.openai_api_key.get_secret_value()
        if not (env_key or dotenv_key):
            offenders = [v.name for v in variants if v.embedding_provider == "openai"]
            sys.exit(
                f"OPENAI_API_KEY is unset but required for variant(s): "
                f"{', '.join(offenders)}. Export it or set it in .env."
            )


def _print_summary(results: list[tuple[str, Status, str]]) -> None:
    """Print a one-line-per-variant end-of-run summary table."""
    name_w = max(len(name) for name, _, _ in results) if results else 0
    status_w = max(len(status) for _, status, _ in results) if results else 0
    print("\n=== Ablation ingestion summary ===")
    for name, status, collection in results:
        print(f"  {name:<{name_w}}  {status:<{status_w}}  {collection}")


def _select_variants(only: str | None) -> list[Variant]:
    """Filter ``VARIANTS`` by ``--only`` name, or return all if unset."""
    if only is None:
        return VARIANTS
    matches = [v for v in VARIANTS if v.name == only]
    if not matches:
        valid = ", ".join(v.name for v in VARIANTS)
        sys.exit(f"Unknown variant '{only}'. Valid names: {valid}.")
    return matches


def main() -> None:
    """Parse args, run preflight checks, then loop over variants.

    Deliberately does not call ``setup_logging()`` — the parent never logs
    (all output is ``print()``), and holding a ``RotatingFileHandler`` open
    on ``logs/hkia.log`` blocks the child subprocess's rotation on Windows
    (parent's write handle prevents ``os.rename``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ingest every ChromaDB collection required for the planned "
            "E1–E4 experiments from a Parquet snapshot."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help=f"Parquet snapshot path (default: {DEFAULT_SNAPSHOT}).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Run a single variant by name "
            f"({', '.join(v.name for v in VARIANTS)})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the target collection already has chunks.",
    )
    args = parser.parse_args()

    variants = _select_variants(args.only)
    _preflight(variants, args.snapshot)

    results: list[tuple[str, Status, str]] = []
    for variant in variants:
        if not args.force and _collection_has_data(variant.collection_name):
            print(
                f"\n=== Skipping '{variant.name}' — collection "
                f"'{variant.collection_name}' already has chunks. "
                f"Pass --force to re-ingest. ==="
            )
            results.append((variant.name, "skipped", variant.collection_name))
            continue

        status = _run_variant(variant, args.snapshot)
        results.append((variant.name, status, variant.collection_name))
        if status == "failed":
            print(
                f"\nVariant '{variant.name}' failed. Stopping sweep — "
                f"fix the failure and re-run (completed variants will be skipped)."
            )
            break

    _print_summary(results)
    if any(status == "failed" for _, status, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
