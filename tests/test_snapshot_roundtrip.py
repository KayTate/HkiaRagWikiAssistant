"""Tests for ingestion/snapshot.py — write/load roundtrip and no-pandas guarantee.

The snapshot module is the on-disk format for the replay-ingestion mode.
A roundtrip mismatch silently feeds wrong wikitext or revision IDs into
downstream chunking, so the equality assertion here is the only guard
against a Parquet schema/encoding drift.

The no-pandas check defends the streaming I/O contract: ``load_snapshot``
must use ``ParquetFile.iter_batches`` and ``RecordBatch.to_pydict``, not
``.to_pandas()``. Pandas is a heavy import and converting the full table
to a DataFrame would defeat the batched-streaming design.
"""

import inspect
from pathlib import Path

import ingestion.snapshot as snapshot_module
from ingestion.snapshot import SnapshotRow, load_snapshot, write_snapshot


def test_snapshot_write_load_roundtrip_preserves_rows_and_order(
    tmp_path: Path,
) -> None:
    """Rows written via ``write_snapshot`` must load back identically.

    Pins field values, ordering, and the Parquet int64 round-trip on
    revision_id (which is an ``int`` in SnapshotRow and must come back
    as an ``int``, not ``np.int64`` or any other surrogate that would
    fail equality comparison downstream). Also pins the documented
    return value of ``write_snapshot`` (the number of rows written) so
    callers that report it cannot silently diverge from the file's
    actual row count.
    """
    rows: list[SnapshotRow] = [
        SnapshotRow(
            page_title="Alpha",
            revision_id=100,
            wikitext="== Alpha ==\nFirst page.",
            fetched_at="2026-05-12T10:00:00+00:00",
        ),
        SnapshotRow(
            page_title="Beta",
            revision_id=200,
            wikitext="== Beta ==\nSecond page.",
            fetched_at="2026-05-12T10:00:01+00:00",
        ),
        SnapshotRow(
            page_title="Gamma",
            revision_id=300,
            wikitext="",
            fetched_at="2026-05-12T10:00:02+00:00",
        ),
    ]

    path = tmp_path / "test.parquet"
    written = write_snapshot(path, iter(rows))
    loaded = list(load_snapshot(path))

    assert written == len(rows)
    assert loaded == rows


def test_snapshot_module_does_not_reference_pandas() -> None:
    """``ingestion.snapshot`` must rely on pyarrow alone, never pandas.

    Pandas is a heavy import and ``.to_pandas()`` materializes the full
    record batch into a DataFrame, which would defeat the streaming
    design. Pinning the source-level absence catches any future refactor
    that introduces ``import pandas`` or a ``.to_pandas()`` call — the
    two routes by which pandas would re-enter the streaming path. The
    check matches code patterns rather than the bare word so docstrings
    or prose comments mentioning pandas do not trip it.
    """
    source = inspect.getsource(snapshot_module)
    assert "import pandas" not in source, (
        "ingestion.snapshot must not import pandas — streaming roundtrip "
        "relies on pyarrow record batches only"
    )
    assert ".to_pandas(" not in source, (
        "ingestion.snapshot must not call .to_pandas() — that materializes "
        "the full record batch into a DataFrame and defeats streaming"
    )
