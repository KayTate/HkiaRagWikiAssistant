"""Parquet-backed wiki snapshot capture and replay.

A snapshot is a single Parquet file holding every page's title, current
revision_id, raw wikitext, and the UTC timestamp at which it was fetched.
Replay-mode ingestion reads from a snapshot instead of hitting the live
wiki, which makes ablation experiments (different chunking strategies,
embedding models, etc.) reproducible against a frozen corpus.

Streaming I/O — both write and load operate on iterables and never hold
the full corpus in memory.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

_BATCH_SIZE = 500


class SnapshotRow(TypedDict):
    """One row of a snapshot Parquet file."""

    page_title: str
    revision_id: int
    wikitext: str
    fetched_at: str


SNAPSHOT_SCHEMA = pa.schema(
    [
        ("page_title", pa.string()),
        ("revision_id", pa.int64()),
        ("wikitext", pa.string()),
        ("fetched_at", pa.string()),
    ]
)


def write_snapshot(path: Path, rows: Iterable[SnapshotRow]) -> int:
    """Write rows to a Parquet snapshot file, buffering in fixed-size batches.

    Streams rows through a ``ParquetWriter`` so the full corpus never has
    to fit in memory. Rows are accumulated into batches of ``_BATCH_SIZE``
    before each ``write_table`` call.

    Args:
        path: Destination Parquet path. Overwritten if it already exists.
        rows: Iterable of ``SnapshotRow`` dicts. May be a generator.

    Returns:
        The total number of rows written to the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[SnapshotRow] = []
    written = 0
    with pq.ParquetWriter(path, SNAPSHOT_SCHEMA) as writer:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= _BATCH_SIZE:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=SNAPSHOT_SCHEMA)
                )
                written += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=SNAPSHOT_SCHEMA)
            )
            written += len(buffer)
    return written


def load_snapshot(path: Path) -> Iterator[SnapshotRow]:
    """Yield rows from a Parquet snapshot one at a time.

    Reads the file in batches via PyArrow's record-batch iterator — no
    pandas, no full-file materialization. Output row order matches write
    order because Parquet preserves insertion order within row groups.

    Args:
        path: Snapshot file written by ``write_snapshot``.

    Yields:
        One ``SnapshotRow`` per page.
    """
    with pq.ParquetFile(path) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=_BATCH_SIZE):
            columns = batch.to_pydict()
            titles = columns["page_title"]
            revision_ids = columns["revision_id"]
            wikitexts = columns["wikitext"]
            fetched_ats = columns["fetched_at"]
            for title, revision_id, wikitext, fetched_at in zip(
                titles, revision_ids, wikitexts, fetched_ats, strict=True
            ):
                yield SnapshotRow(
                    page_title=title,
                    revision_id=revision_id,
                    wikitext=wikitext,
                    fetched_at=fetched_at,
                )
