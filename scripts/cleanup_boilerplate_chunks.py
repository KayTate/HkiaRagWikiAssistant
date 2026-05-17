"""Delete chunks whose text matches the ingestion-chunker boilerplate allowlist.

One-time cleanup for collections written before chunker.py started
filtering header-only chunks at write time. Targets the collection
named by ``settings.chroma_collection_name``.

Dry-run by default — prints what would be deleted. Pass ``--apply`` to
actually delete.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from ingestion.chunker import _is_boilerplate  # noqa: E402
from vectorstore.client import get_or_create_collection  # noqa: E402

_DELETE_BATCH_SIZE = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete matching chunks. Without this flag, only prints a summary.",
    )
    args = parser.parse_args()

    collection_name = settings.chroma_collection_name
    col = get_or_create_collection(collection_name)
    total = col.count()
    print(f"Collection {collection_name!r}: {total} chunks")
    if total == 0:
        return

    result = col.get(include=["documents"])
    ids: list[str] = list(result.get("ids") or [])
    docs: list[str | None] = list(result.get("documents") or [])

    matching_ids: list[str] = []
    text_counts: Counter[str] = Counter()
    for cid, doc in zip(ids, docs, strict=True):
        if doc and _is_boilerplate(doc):
            matching_ids.append(cid)
            text_counts[doc.strip()] += 1

    print(f"Boilerplate chunks found: {len(matching_ids)}")
    for text, n in text_counts.most_common():
        preview = text.replace("\n", "\\n")
        print(f"  {n:5d}x {preview!r}")

    if not matching_ids:
        return

    if not args.apply:
        print("\nDRY RUN — pass --apply to delete.")
        return

    for start in range(0, len(matching_ids), _DELETE_BATCH_SIZE):
        batch = matching_ids[start : start + _DELETE_BATCH_SIZE]
        col.delete(ids=batch)
    print(f"Deleted {len(matching_ids)} chunks from {collection_name!r}.")


if __name__ == "__main__":
    main()
