"""Unit tests for the bulk SQLite helpers in ingestion/state_db.py.

Covers ``get_pages`` and ``upsert_pages`` plus the connection-count
guarantees that justify their existence — a regression that loops
through ``upsert_page`` per row instead of using the bulk path would
silently restore the per-row-connection cost these helpers exist to
eliminate.
"""

import pathlib
from typing import Any

import pytest

from ingestion import state_db


@pytest.fixture
def isolated_db(mocker: Any, tmp_path: pathlib.Path) -> str:
    """Point state_db at a per-test SQLite file under tmp_path.

    Returns the path so tests can re-open the file directly if they
    need to inspect raw rows.
    """
    db_path = str(tmp_path / "state.db")
    mocker.patch("config.settings.settings.state_db_path", db_path)
    return db_path


# ---------------------------------------------------------------------------
# get_pages
# ---------------------------------------------------------------------------


def test_get_pages_empty_input_returns_empty_dict(
    isolated_db: str, mocker: Any
) -> None:
    """An empty input list must short-circuit before opening a connection.

    Locks in the cheap-call contract: callers can pass an empty list
    without paying the SQLite connection setup cost. A naive "always
    run a SELECT" implementation would still open a connection.
    """
    connect_spy = mocker.spy(state_db, "_connect")

    result = state_db.get_pages([])

    assert result == {}
    assert connect_spy.call_count == 0, (
        "Empty get_pages must not open a SQLite connection"
    )


def test_get_pages_returns_dict_keyed_by_title(isolated_db: str) -> None:
    """Present titles are returned in a dict keyed by page_title."""
    state_db.upsert_pages(
        [
            {
                "page_title": "Alpha",
                "revision_id": 1,
                "status": "complete",
                "embedding_model": "m:v1",
            },
            {
                "page_title": "Beta",
                "revision_id": 2,
                "status": "pending",
                "embedding_model": "m:v1",
            },
        ]
    )

    result = state_db.get_pages(["Alpha", "Beta"])

    assert set(result.keys()) == {"Alpha", "Beta"}
    assert result["Alpha"]["revision_id"] == 1
    assert result["Beta"]["status"] == "pending"


def test_get_pages_omits_titles_not_in_db(isolated_db: str) -> None:
    """Missing titles must be absent from the result, not None-valued.

    Callers use ``existing_by_title.get(title)`` and compare against
    None. A regression that returned ``{title: None}`` for missing
    rows would silently break that check.
    """
    state_db.upsert_pages(
        [
            {
                "page_title": "Alpha",
                "revision_id": 1,
                "status": "complete",
                "embedding_model": "m:v1",
            },
        ]
    )

    result = state_db.get_pages(["Alpha", "DoesNotExist", "AlsoMissing"])

    assert list(result.keys()) == ["Alpha"]
    assert "DoesNotExist" not in result
    assert "AlsoMissing" not in result


def test_get_pages_uses_single_connection(
    isolated_db: str, mocker: Any
) -> None:
    """Locks in the perf claim: one SELECT, one connection, regardless of N.

    The whole point of get_pages over a get_page loop is that 1000
    titles open one SQLite connection, not 1000. If a future refactor
    iterates titles internally and calls _connect per chunk, this test
    fails for the right reason.
    """
    state_db.upsert_pages(
        [
            {
                "page_title": f"P{i}",
                "revision_id": i,
                "status": "complete",
                "embedding_model": "m:v1",
            }
            for i in range(50)
        ]
    )

    connect_spy = mocker.spy(state_db, "_connect")

    result = state_db.get_pages([f"P{i}" for i in range(50)])

    assert len(result) == 50
    assert connect_spy.call_count == 1, (
        f"get_pages must open exactly one connection; got "
        f"{connect_spy.call_count}. Regression of the bulk-read fix."
    )


# ---------------------------------------------------------------------------
# upsert_pages
# ---------------------------------------------------------------------------


def test_upsert_pages_empty_input_is_noop(
    isolated_db: str, mocker: Any
) -> None:
    """Empty input must short-circuit before opening a connection.

    Same rationale as get_pages: callers building up a row list
    conditionally should be able to pass [] without paying for SQLite
    setup.
    """
    connect_spy = mocker.spy(state_db, "_connect")

    state_db.upsert_pages([])

    assert connect_spy.call_count == 0


def test_upsert_pages_inserts_new_rows(isolated_db: str) -> None:
    """Bulk insert produces rows with the expected columns populated."""
    state_db.upsert_pages(
        [
            {
                "page_title": "Alpha",
                "revision_id": 1,
                "status": "pending",
                "embedding_model": "m:v1",
            },
            {
                "page_title": "Beta",
                "revision_id": 2,
                "status": "complete",
                "embedding_model": "m:v1",
            },
        ]
    )

    alpha = state_db.get_page("Alpha")
    beta = state_db.get_page("Beta")

    assert alpha is not None and alpha["revision_id"] == 1
    assert alpha["status"] == "pending"
    assert beta is not None and beta["status"] == "complete"


def test_upsert_pages_updates_existing_rows_on_conflict(
    isolated_db: str,
) -> None:
    """ON CONFLICT semantics must match the single-row upsert_page.

    Inserting then re-inserting the same page_title with different
    field values must overwrite, not duplicate or fail. This is the
    contract that makes the mark-pending paths idempotent on re-run.
    """
    state_db.upsert_pages(
        [
            {
                "page_title": "Alpha",
                "revision_id": 1,
                "status": "complete",
                "embedding_model": "m:v1",
            }
        ]
    )

    state_db.upsert_pages(
        [
            {
                "page_title": "Alpha",
                "revision_id": 99,
                "status": "pending",
                "embedding_model": "m:v2",
            }
        ]
    )

    alpha = state_db.get_page("Alpha")
    assert alpha is not None
    assert alpha["revision_id"] == 99
    assert alpha["status"] == "pending"
    assert alpha["embedding_model"] == "m:v2"


def test_upsert_pages_shares_one_timestamp_across_batch(
    isolated_db: str,
) -> None:
    """All rows in a single bulk call share the same ``updated_at``.

    Documented behavior: the batch is one logical operation, so all
    rows get the same timestamp rather than each row sampling
    ``datetime.now`` separately. Locking this in catches a regression
    that would compute now() inside the executemany loop.
    """
    state_db.upsert_pages(
        [
            {
                "page_title": f"P{i}",
                "revision_id": i,
                "status": "pending",
                "embedding_model": "m:v1",
            }
            for i in range(5)
        ]
    )

    fetched = state_db.get_pages([f"P{i}" for i in range(5)])
    timestamps = {row["updated_at"] for row in fetched.values()}
    assert len(timestamps) == 1, (
        f"All bulk-upserted rows must share one updated_at timestamp; "
        f"got {len(timestamps)} distinct values: {timestamps}"
    )


def test_upsert_pages_uses_single_connection(
    isolated_db: str, mocker: Any
) -> None:
    """Locks in the perf claim: one connection, one transaction, N rows.

    The motivating fix: a wiki with 5000 pages used to mean 5000
    SQLite connections during the mark-pending phase. After this
    helper, it's one. If a future refactor reverts to a per-row loop
    inside upsert_pages, this test catches it.
    """
    rows: list[state_db.PageStateRow] = [
        {
            "page_title": f"P{i}",
            "revision_id": i,
            "status": "pending",
            "embedding_model": "m:v1",
        }
        for i in range(50)
    ]

    connect_spy = mocker.spy(state_db, "_connect")

    state_db.upsert_pages(rows)

    assert connect_spy.call_count == 1, (
        f"upsert_pages must open exactly one connection for any batch "
        f"size; got {connect_spy.call_count}. Regression of the "
        f"bulk-write fix."
    )
    # Sanity: rows actually landed.
    assert len(state_db.get_pages([r["page_title"] for r in rows])) == 50
