"""Top-level orchestrator for the HKIA wiki ingestion pipeline.

Supports full re-ingestion of all pages and incremental ingestion of
only new or changed pages. Both modes share the same per-page loop to
ensure consistent behavior.
"""

import logging
from datetime import UTC, datetime

from config.settings import settings
from ingestion import api_client, chunker, embedder, parser, state_db
from vectorstore import client as vectorstore_client
from vectorstore.schema import ChunkMetadata

logger = logging.getLogger(__name__)


def _current_embedding_model() -> str:
    """Return ``"{model}:{version}"`` from current settings."""
    return f"{settings.embedding_model}:{settings.embedding_model_version}"


def run_startup_sync_check() -> None:
    """Verify embedding model consistency across ChromaDB and SQLite before ingestion.

    Step 1 samples up to 10 ChromaDB chunks to detect collection-level model
    drift. If found, raises EmbeddingModelMismatchError — operator must
    create a new collection and re-ingest.

    Step 2 finds SQLite rows with a stale embedding_model and silently
    resets them to 'pending' so they are re-ingested on the next run.
    This auto-repairs drift that occurs when settings change but the
    collection has not yet been re-ingested.

    Raises:
        EmbeddingModelMismatchError: If ChromaDB chunks from a different
            model are detected. See error message for remediation steps.
    """
    logger.info("Running startup sync check for embedding model consistency")
    vectorstore_client.verify_collection_embedding_model()
    _repair_stale_sqlite_rows()


def _repair_stale_sqlite_rows() -> None:
    """Reset SQLite rows with a stale embedding model back to pending.

    Each stale row is updated to pending with a revision_id of -1 as a
    sentinel, triggering full re-fetch and re-embedding on the next run.
    """
    logger.info("Checking for SQLite rows with stale embedding model")
    current_model = _current_embedding_model()
    stale_pages = state_db.get_pages_with_stale_embedding_model(current_model)
    for page in stale_pages:
        logger.info(
            "Resetting '%s' to pending — stored model '%s' != current '%s'",
            page["page_title"],
            page["embedding_model"],
            current_model,
        )
        state_db.upsert_page(
            page_title=page["page_title"],
            revision_id=page["revision_id"],
            status="pending",
            embedding_model=current_model,
        )


def run_full_ingestion() -> None:
    """Re-ingest every wiki page, regardless of current status.

    Uses a single batch API call to fetch all titles and revision IDs
    together, then marks each as pending before running the per-page
    loop. Existing chunks are replaced during each page's ingestion
    via the delete-before-insert pattern.

    Raises:
        EmbeddingModelMismatchError: If the ChromaDB collection model
            does not match current settings.
    """
    run_startup_sync_check()
    pages = api_client.get_all_pages_with_revision_ids()
    _mark_all_pages_pending(pages)
    _process_pending_pages()


def run_incremental_ingestion() -> None:
    """Ingest only new or changed wiki pages.

    Uses a single batch API call to fetch all titles and revision IDs,
    then compares against stored state to identify pages that are new
    or have been updated. Complete, unchanged pages are skipped.

    Raises:
        EmbeddingModelMismatchError: If the ChromaDB collection model
            does not match current settings.
    """
    run_startup_sync_check()
    pages = api_client.get_all_pages_with_revision_ids()
    _mark_changed_pages_pending(pages)
    _process_pending_pages()


def _mark_all_pages_pending(
    pages: list[dict[str, str | int]],
) -> None:
    """Mark every page as pending using pre-fetched revision IDs."""
    current_model = _current_embedding_model()
    rows: list[state_db.PageStateRow] = [
        {
            "page_title": str(page["title"]),
            "revision_id": int(page["revision_id"]),
            "status": "pending",
            "embedding_model": current_model,
        }
        for page in pages
    ]
    state_db.upsert_pages(rows)


def _mark_changed_pages_pending(
    pages: list[dict[str, str | int]],
) -> None:
    """Mark new or revision-changed pages as pending; leave unchanged ones alone."""
    current_model = _current_embedding_model()
    titles = [str(p["title"]) for p in pages]
    existing_by_title = state_db.get_pages(titles)
    rows: list[state_db.PageStateRow] = []
    for page in pages:
        title = str(page["title"])
        revision_id = int(page["revision_id"])
        existing_row = existing_by_title.get(title)
        if existing_row is None or existing_row["revision_id"] != revision_id:
            rows.append(
                {
                    "page_title": title,
                    "revision_id": revision_id,
                    "status": "pending",
                    "embedding_model": current_model,
                }
            )
    state_db.upsert_pages(rows)


_WIKITEXT_BATCH_SIZE = 20


def _process_pending_pages() -> None:
    """Run the per-page ingestion loop for all pages with status='pending'.

    Fetches wikitext in batches of _WIKITEXT_BATCH_SIZE pages via the batch
    API to minimize the number of wiki requests. Per-page failures are
    caught so the rest of the run still processes — the failing page
    keeps its 'in_progress' status and is retried on the next run.
    Batch-level errors (wikitext fetch failures, embedding-model
    mismatch) propagate, since those signal a wiki- or
    configuration-wide problem where continuing would just produce more
    of the same error.
    """
    pending = state_db.get_pages_by_status("pending")
    logger.info("Processing %d pending pages", len(pending))

    succeeded = 0
    failed = 0

    for start in range(0, len(pending), _WIKITEXT_BATCH_SIZE):
        batch = pending[start : start + _WIKITEXT_BATCH_SIZE]
        titles = [p["page_title"] for p in batch]
        wikitext_map = api_client.get_pages_wikitext_batch(titles)

        for page in batch:
            title = page["page_title"]
            wikitext = wikitext_map.get(title)
            if wikitext is None:
                logger.warning("No wikitext returned for '%s', skipping", title)
                failed += 1
                continue
            try:
                _ingest_page(title, page["revision_id"], wikitext)
                succeeded += 1
            except Exception:
                failed += 1

    logger.info(
        "Ingestion complete: %d pages succeeded, %d failed", succeeded, failed
    )


def _ingest_page(page_title: str, revision_id: int, wikitext: str) -> None:
    """Execute the full ingestion sequence for a single wiki page.

    Status transitions: pending → in_progress → complete on success.
    On any exception, status remains in_progress. The ChromaDB delete
    in step 6 ensures no duplicate chunks accumulate across retries.

    Args:
        page_title: The wiki page title to ingest.
        revision_id: The MediaWiki revision ID for this ingestion.
        wikitext: Pre-fetched raw wikitext content for this page.
    """
    current_model = _current_embedding_model()
    state_db.upsert_page(
        page_title=page_title,
        revision_id=revision_id,
        status="in_progress",
        embedding_model=current_model,
    )
    try:
        plain_text = parser.parse_wikitext(wikitext)
        sections = parser.extract_sections(wikitext)
        chunks = chunker.chunk_text(
            text=plain_text,
            strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            sections=sections,
        )
        if not chunks:
            logger.warning("No chunks produced for '%s', skipping", page_title)
            _finalize_page(page_title, revision_id, current_model)
            return

        embeddings = embedder.embed_chunks(chunks)
        metadatas = _build_metadatas(page_title, revision_id, chunks, current_model)
        _write_to_stores(
            page_title, revision_id, chunks, embeddings, metadatas, current_model
        )
    except Exception:
        logger.exception(
            "Failed to ingest page '%s', status remains in_progress", page_title
        )
        raise


def _build_metadatas(
    page_title: str,
    revision_id: int,
    chunks: list[str],
    current_model: str,
) -> list[ChunkMetadata]:
    """Construct one ChunkMetadata per chunk of a page, in input order."""
    source_url = f"{settings.wiki_base_url}/wiki/{page_title.replace(' ', '_')}"
    ingested_at = datetime.now(UTC).isoformat()
    return [
        ChunkMetadata(
            source_title=page_title,
            source_url=source_url,
            section="",
            category="",
            chunk_index=idx,
            revision_id=revision_id,
            ingested_at=ingested_at,
            embedding_model=current_model,
            chunking_strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for idx, _ in enumerate(chunks)
    ]


def _write_to_stores(
    page_title: str,
    revision_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[ChunkMetadata],
    current_model: str,
) -> None:
    """Replace ChromaDB chunks and update SQLite status to complete."""
    vectorstore_client.delete_chunks_by_source(page_title)
    if chunks:
        vectorstore_client.upsert_chunks(page_title, chunks, embeddings, metadatas)
    state_db.upsert_page(
        page_title=page_title,
        revision_id=revision_id,
        status="complete",
        embedding_model=current_model,
    )
    logger.info("Ingested '%s' (%d chunks)", page_title, len(chunks))


def _finalize_page(
    page_title: str,
    revision_id: int,
    current_model: str,
) -> None:
    """Empty-chunk edge case: clean up stale ChromaDB entries, mark complete."""
    vectorstore_client.delete_chunks_by_source(page_title)
    state_db.upsert_page(
        page_title=page_title,
        revision_id=revision_id,
        status="complete",
        embedding_model=current_model,
    )
