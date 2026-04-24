"""SQLite-backed state management for the ingestion pipeline.

Tracks per-page ingestion status, revision IDs, and embedding model
metadata so the pipeline can resume after failures and detect when pages
need re-ingestion due to a model change.
"""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS page_ingestion_state (
    page_title      TEXT PRIMARY KEY,
    revision_id     INTEGER NOT NULL,
    status          TEXT NOT NULL
                    CHECK(status IN ('pending', 'in_progress', 'complete')),
    embedding_model TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    """Open a connection to the SQLite state database, creating it if absent.

    Ensures the parent directory and schema exist before returning.
    """
    db_path = Path(settings.state_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection that commits on success, rolls back
    on exception, and always closes.

    Wraps the existing ``with conn:`` protocol so read/write semantics are
    unchanged. The outer ``finally`` guarantees ``conn.close()`` runs even
    on Windows where file locks would otherwise block tempdir cleanup in
    tests.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def upsert_page(
    page_title: str,
    revision_id: int,
    status: str,
    embedding_model: str,
) -> None:
    """Insert or update a page record with the given status and model.

    Safe to call multiple times — subsequent calls overwrite the existing
    row for that page_title.

    Args:
        page_title: The wiki page title (primary key).
        revision_id: The MediaWiki revision ID at time of upsert.
        status: One of 'pending', 'in_progress', or 'complete'.
        embedding_model: Formatted model identifier, e.g. "nomic-embed-text:v1.5".
    """
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO page_ingestion_state
                (page_title, revision_id, status, embedding_model, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(page_title) DO UPDATE SET
                revision_id = excluded.revision_id,
                status = excluded.status,
                embedding_model = excluded.embedding_model,
                updated_at = excluded.updated_at
            """,
            (page_title, revision_id, status, embedding_model, _now_iso()),
        )


def get_page(page_title: str) -> dict | None:  # type: ignore[type-arg]
    """Fetch the state record for a single page.

    Args:
        page_title: The wiki page title to look up.

    Returns:
        A dict with keys matching the table columns, or None if not found.
    """
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM page_ingestion_state WHERE page_title = ?",
            (page_title,),
        ).fetchone()
    return dict(row) if row else None


def get_pages_by_status(status: str) -> list[dict]:  # type: ignore[type-arg]
    """Return all pages with the given status.

    Args:
        status: One of 'pending', 'in_progress', or 'complete'.

    Returns:
        List of row dicts ordered by page_title.
    """
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM page_ingestion_state WHERE status = ? ORDER BY page_title",
            (status,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_pages_with_stale_embedding_model(current_model: str) -> list[dict]:  # type: ignore[type-arg]
    """Return all pages whose stored embedding model differs from the current one.

    Used by the startup sync check to identify pages that were ingested with
    a different model and therefore need re-ingestion.

    Args:
        current_model: The formatted model identifier currently configured,
            e.g. "nomic-embed-text:v1.5".

    Returns:
        List of row dicts for pages where embedding_model != current_model.
    """
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM page_ingestion_state
            WHERE embedding_model != ?
            ORDER BY page_title
            """,
            (current_model,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_complete(page_title: str) -> None:
    """Record successful processing of a page in the state database.

    Sets status to 'complete' and records the current timestamp. Safe to
    call multiple times — subsequent calls are no-ops if the page is
    already complete.

    Args:
        page_title: The wiki page title to mark complete.
    """
    with _connection() as conn:
        conn.execute(
            """
            UPDATE page_ingestion_state
            SET status = 'complete', updated_at = ?
            WHERE page_title = ?
            """,
            (_now_iso(), page_title),
        )


def mark_pending(page_title: str, revision_id: int) -> None:
    """Reset a page to pending status, updating the revision ID.

    Used when a page is detected as new or updated during incremental
    ingestion, or when an embedding model change requires re-ingestion.

    Args:
        page_title: The wiki page title to reset.
        revision_id: The new revision ID from the MediaWiki API.
    """
    with _connection() as conn:
        conn.execute(
            """
            UPDATE page_ingestion_state
            SET status = 'pending', revision_id = ?, updated_at = ?
            WHERE page_title = ?
            """,
            (revision_id, _now_iso(), page_title),
        )


def get_status_summary() -> str:
    """Return a human-readable count of pages grouped by ingestion status.

    Useful for the 'status' mode of sync.py to give operators a quick
    overview of pipeline progress without querying the database directly.

    Returns:
        A formatted string like "pending: 10, in_progress: 2, complete: 4000".
    """
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) as count
            FROM page_ingestion_state
            GROUP BY status
            """
        ).fetchall()
    counts = {row["status"]: row["count"] for row in rows}
    parts = [
        f"pending: {counts.get('pending', 0)}",
        f"in_progress: {counts.get('in_progress', 0)}",
        f"complete: {counts.get('complete', 0)}",
    ]
    return ", ".join(parts)
