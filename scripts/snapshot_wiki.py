"""Capture a full wiki snapshot to a Parquet file for replay ingestion.

Usage:
    python scripts/snapshot_wiki.py --output snapshots/2026-05-12.parquet
    python scripts/snapshot_wiki.py --output snapshots/smoke.parquet --limit 5

The script fetches every page's current revision ID, then streams wikitext
in batches into the output Parquet file. A snapshot file can be replayed
with `python sync.py --mode replay --snapshot <path>` to ingest the frozen
corpus under different chunking/embedding settings without re-hitting the
wiki.

Exits non-zero if any requested page returned empty wikitext, since that
typically indicates a deleted/protected page or a partial wiki response
that would silently shrink the snapshot.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow `python scripts/snapshot_wiki.py` to import top-level project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging  # noqa: E402
from ingestion import api_client, snapshot  # noqa: E402

logger = logging.getLogger(__name__)


def _fetch_rows(
    title_to_revision: dict[str, int],
) -> tuple[list[snapshot.SnapshotRow], list[str]]:
    """Fetch wikitext for every title and return rows alongside missing titles.

    Fetches wikitext in batches of ``api_client._WIKITEXT_BATCH_SIZE``.
    Titles whose wikitext is missing from a batch response are collected
    into the second return value so the caller can drive the script's
    exit code without inspecting log output.

    Args:
        title_to_revision: Mapping built from the allpages query, in the
            order pages should be written to the snapshot.

    Returns:
        A pair ``(rows, missing_titles)``. ``rows`` is the list of
        ``SnapshotRow`` dicts for pages that returned content;
        ``missing_titles`` is the list of titles whose wikitext was not
        returned.
    """
    rows: list[snapshot.SnapshotRow] = []
    missing_titles: list[str] = []
    titles = list(title_to_revision.keys())
    batch_size = api_client._WIKITEXT_BATCH_SIZE
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        wikitext_map = api_client.get_pages_wikitext_batch(batch)
        fetched_at = datetime.now(UTC).isoformat()
        for title in batch:
            wikitext = wikitext_map.get(title)
            if wikitext is None:
                missing_titles.append(title)
                continue
            rows.append(
                snapshot.SnapshotRow(
                    page_title=title,
                    revision_id=title_to_revision[title],
                    wikitext=wikitext,
                    fetched_at=fetched_at,
                )
            )
    return rows, missing_titles


def main() -> None:
    """Parse CLI args, fetch wiki content, and write it to a Parquet snapshot."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Capture a wiki snapshot to a Parquet file for replay."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Parquet file path (e.g. snapshots/2026-05-12.parquet).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test cap: only snapshot the first N pages.",
    )
    args = parser.parse_args()

    pages = api_client.get_all_pages_with_revision_ids()
    if args.limit is not None:
        pages = pages[: args.limit]
    title_to_revision: dict[str, int] = {
        str(page["title"]): int(page["revision_id"]) for page in pages
    }
    logger.info("Capturing snapshot of %d pages to %s", len(pages), args.output)

    rows, missing_titles = _fetch_rows(title_to_revision)
    written = snapshot.write_snapshot(args.output, rows)

    logger.info("Snapshot complete: %d pages written to %s", written, args.output)
    if missing_titles:
        logger.error(
            "Snapshot missing wikitext for %d page(s): %s",
            len(missing_titles),
            ", ".join(missing_titles[:10])
            + (" ..." if len(missing_titles) > 10 else ""),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
