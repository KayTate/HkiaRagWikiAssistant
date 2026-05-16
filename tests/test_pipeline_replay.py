"""Tests for the snapshot-replay ingestion mode.

``run_ingestion_from_snapshot`` is the only ingest path that must not
touch the live wiki — the snapshot is the single source of wikitext.
Asserting this here pins the boundary so a future refactor that
sneaks an ``api_client`` call back into ``_ingest_batch`` (e.g.
template expansion that resolves remotely) fails loudly instead of
silently re-introducing wiki-drift noise into ablation experiments.
"""

from pathlib import Path
from typing import Any

from ingestion.pipeline import run_ingestion_from_snapshot
from ingestion.snapshot import SnapshotRow, write_snapshot
from ingestion.state_db import get_page

_API_CLIENT_FUNCTIONS = (
    "get_all_pages_with_revision_ids",
    "get_pages_wikitext_batch",
    "get_all_page_titles",
    "get_page_wikitext",
    "get_page_revision_id",
    "get_cargo_items",
    "opensearch_title",
)


def _build_snapshot(path: Path) -> list[SnapshotRow]:
    """Write a 3-page synthetic snapshot to ``path`` and return its rows.

    Wikitext is deliberately plain prose with no templates so the parser
    has no excuse to reach for ``api_client`` for template expansion.
    """
    rows: list[SnapshotRow] = [
        SnapshotRow(
            page_title="Alpha",
            revision_id=10,
            wikitext="Page about Alpha.",
            fetched_at="2026-05-12T10:00:00+00:00",
        ),
        SnapshotRow(
            page_title="Beta",
            revision_id=20,
            wikitext="Page about Beta.",
            fetched_at="2026-05-12T10:00:01+00:00",
        ),
        SnapshotRow(
            page_title="Gamma",
            revision_id=30,
            wikitext="Page about Gamma.",
            fetched_at="2026-05-12T10:00:02+00:00",
        ),
    ]
    write_snapshot(path, iter(rows))
    return rows


def _pin_settings(mocker: Any, tmp_path: Path) -> None:
    """Point settings at tmp_path-backed state DB / Chroma dir and pin chunking."""
    mocker.patch(
        "config.settings.settings.state_db_path",
        str(tmp_path / "state.db"),
    )
    mocker.patch(
        "config.settings.settings.chroma_persist_dir",
        str(tmp_path / "chroma"),
    )
    mocker.patch("config.settings.settings.embedding_model", "nomic-embed-text")
    mocker.patch("config.settings.settings.embedding_model_version", "v1.5")
    mocker.patch("config.settings.settings.chunking_strategy", "recursive")
    mocker.patch("config.settings.settings.chunk_size", 512)
    mocker.patch("config.settings.settings.chunk_overlap", 64)


def test_replay_ingests_snapshot_without_calling_api_client(
    mocker: Any, tmp_path: Path
) -> None:
    """End-to-end replay: snapshot in, complete rows + chunk upserts out, no wiki I/O.

    Pins three invariants of the replay path:

    1. Every ``api_client`` function is untouched — the snapshot is the
       only source of wikitext.
    2. Each snapshot row ends up in SQLite with ``status='complete'`` and
       the snapshot's revision_id, so a later incremental run sees the
       same revision IDs the snapshot was captured against.
    3. ``upsert_chunks`` is called exactly once per page — no duplicates,
       no skipped pages.
    """
    _pin_settings(mocker, tmp_path)

    mocker.patch(
        "vectorstore.client.verify_collection_consistency",
        return_value=None,
    )
    upsert_spy = mocker.patch(
        "vectorstore.client.upsert_chunks",
        return_value=None,
    )
    mocker.patch(
        "vectorstore.client.delete_chunks_by_source",
        return_value=None,
    )
    mocker.patch(
        "ingestion.embedder.embed_chunks",
        side_effect=lambda chunks: [[0.1] * 4 for _ in chunks],
    )

    api_spies = {
        name: mocker.patch(f"ingestion.api_client.{name}", autospec=True)
        for name in _API_CLIENT_FUNCTIONS
    }

    snapshot_path = tmp_path / "snap.parquet"
    rows = _build_snapshot(snapshot_path)

    run_ingestion_from_snapshot(snapshot_path)

    # api_client must remain untouched — the replay path is the snapshot, nothing else.
    for name, spy in api_spies.items():
        assert spy.call_count == 0, (
            f"replay must not call api_client.{name}, called {spy.call_count} time(s)"
        )

    # SQLite rows must reflect snapshot titles, revision IDs, and 'complete' status.
    for row in rows:
        page = get_page(row["page_title"])
        assert page is not None, f"missing state row for {row['page_title']!r}"
        assert page["status"] == "complete", (
            f"{row['page_title']!r} should be complete, got {page['status']!r}"
        )
        assert page["revision_id"] == row["revision_id"], (
            f"{row['page_title']!r} revision drifted: stored "
            f"{page['revision_id']}, snapshot {row['revision_id']}"
        )

    # One upsert_chunks call per page — no page silently dropped, none double-written.
    assert upsert_spy.call_count == len(rows), (
        f"upsert_chunks must be called once per page; "
        f"called {upsert_spy.call_count} time(s) for {len(rows)} pages"
    )
