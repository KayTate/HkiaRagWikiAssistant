"""Idempotency tests for the HKIA ingestion pipeline.

Verifies that re-running ingestion after partial or complete success
produces no duplicate chunks and that status transitions are correct.
All external dependencies are mocked — no real network or DB calls are made.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ingestion.embedder import EmbeddingError


def _make_fake_collection() -> MagicMock:
    """Build a MagicMock that simulates a minimal ChromaDB collection.

    Tracks upserted and deleted documents internally so tests can assert
    on final document counts without real ChromaDB.
    """
    stored: dict[str, str] = {}
    collection = MagicMock()

    def fake_upsert(
        ids: list[str],
        documents: list[str],
        embeddings: list[Any],
        metadatas: list[Any],
    ) -> None:
        for doc_id, doc in zip(ids, documents, strict=True):
            stored[doc_id] = doc

    def fake_delete(where: dict[str, Any]) -> None:
        source_title = where.get("source_title", {}).get("$eq", "")
        to_remove = [k for k in stored if k.startswith(f"{source_title}::")]
        for key in to_remove:
            del stored[key]

    def fake_get(**kwargs: Any) -> dict[str, Any]:
        limit = kwargs.get("limit", 10)
        items = list(stored.items())[:limit]
        return {
            "ids": [i[0] for i in items],
            "documents": [i[1] for i in items],
            "metadatas": [{"embedding_model": "nomic-embed-text:v1.5"} for _ in items],
        }

    def fake_count() -> int:
        return len(stored)

    def fake_query(**kwargs: Any) -> dict[str, Any]:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    collection.upsert.side_effect = fake_upsert
    collection.delete.side_effect = fake_delete
    collection.get.side_effect = fake_get
    collection.count.side_effect = fake_count
    collection.query.side_effect = fake_query
    collection._stored = stored
    return collection


def test_rerun_after_failure_produces_no_duplicates(mocker: Any) -> None:
    """Verify that re-running ingestion after a partial failure produces no duplicates.

    Scenario: 3 pages; embedder succeeds for pages 1 and 2, raises on page 3.
    After fixing the embedder, re-running ingestion should result in exactly
    one set of chunks per page and all pages marked complete.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "state.db")
        chroma_path = str(Path(tmpdir) / "chroma")

        mocker.patch("config.settings.settings.state_db_path", db_path)
        mocker.patch("config.settings.settings.chroma_persist_dir", chroma_path)
        mocker.patch("config.settings.settings.embedding_model", "nomic-embed-text")
        mocker.patch("config.settings.settings.embedding_model_version", "v1.5")
        mocker.patch("config.settings.settings.chunking_strategy", "recursive")
        mocker.patch("config.settings.settings.chunk_size", 512)
        mocker.patch("config.settings.settings.chunk_overlap", 64)

        fake_collection = _make_fake_collection()

        mocker.patch(
            "vectorstore.client.get_or_create_collection",
            return_value=fake_collection,
        )

        # Pages named so they sort alphabetically as Alpha, Beta, Zeta.
        # SQLite returns them in alphabetical order, so Alpha is call 1,
        # Beta is call 2, Zeta is call 3 — the intended failure point.
        fake_wikitext = "Some content about the game."
        mocker.patch(
            "ingestion.api_client.get_all_pages_with_revision_ids",
            return_value=[
                {"title": "Alpha_Page", "revision_id": 42},
                {"title": "Beta_Page", "revision_id": 42},
                {"title": "Zeta_Page", "revision_id": 42},
            ],
        )
        mocker.patch(
            "ingestion.api_client.get_pages_wikitext_batch",
            return_value={
                "Alpha_Page": fake_wikitext,
                "Beta_Page": fake_wikitext,
                "Zeta_Page": fake_wikitext,
            },
        )

        fail_on_third = {"count": 0}

        def embed_side_effect(chunks: list[str]) -> list[list[float]]:
            fail_on_third["count"] += 1
            if fail_on_third["count"] == 3:
                raise EmbeddingError("Simulated provider failure on page 3")
            return [[0.1] * 4 for _ in chunks]

        mocker.patch(
            "ingestion.embedder.embed_chunks",
            side_effect=embed_side_effect,
        )

        from ingestion.pipeline import run_full_ingestion
        from ingestion.state_db import get_page, get_pages_by_status

        with pytest.raises(EmbeddingError):
            run_full_ingestion()

        complete_pages = get_pages_by_status("complete")
        in_progress_pages = get_pages_by_status("in_progress")
        assert len(complete_pages) == 2, (
            f"Expected 2 complete pages after partial failure, "
            f"got {len(complete_pages)}"
        )
        assert len(in_progress_pages) == 1, (
            f"Expected 1 in_progress page, got {len(in_progress_pages)}"
        )
        assert in_progress_pages[0]["page_title"] == "Zeta_Page"

        chunks_after_first_run = dict(fake_collection._stored)

        mocker.patch(
            "ingestion.embedder.embed_chunks",
            return_value=[[0.1] * 4, [0.2] * 4],
        )

        from ingestion import state_db

        state_db.upsert_page(
            page_title="Zeta_Page",
            revision_id=42,
            status="pending",
            embedding_model="nomic-embed-text:v1.5",
        )

        from ingestion.pipeline import _process_pending_pages

        _process_pending_pages()

        all_complete = get_pages_by_status("complete")
        assert len(all_complete) == 3, (
            f"Expected all 3 pages complete after re-run, got {len(all_complete)}"
        )

        zeta_record = get_page("Zeta_Page")
        assert zeta_record is not None
        assert zeta_record["status"] == "complete"

        alpha_ids = [k for k in fake_collection._stored if k.startswith("Alpha_Page::")]
        beta_ids = [k for k in fake_collection._stored if k.startswith("Beta_Page::")]
        zeta_ids = [k for k in fake_collection._stored if k.startswith("Zeta_Page::")]

        assert len(alpha_ids) >= 1, "Alpha_Page should have at least one chunk"
        assert len(beta_ids) >= 1, "Beta_Page should have at least one chunk"
        assert len(zeta_ids) >= 1, "Zeta_Page should have at least one chunk"

        for doc_id in alpha_ids:
            assert doc_id in chunks_after_first_run, (
                "Alpha_Page chunks changed unexpectedly — possible duplicate insertion"
            )
        for doc_id in beta_ids:
            assert doc_id in chunks_after_first_run, (
                "Beta_Page chunks changed unexpectedly — possible duplicate insertion"
            )


def test_status_transitions_correctly(mocker: Any) -> None:
    """Verify correct status transitions across success, failure, and skip scenarios.

    - A page moves pending -> in_progress -> complete on success.
    - A page stays in_progress when embedding raises.
    - A complete page with the same revision ID is skipped on re-run.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "state.db")
        chroma_path = str(Path(tmpdir) / "chroma")

        mocker.patch("config.settings.settings.state_db_path", db_path)
        mocker.patch("config.settings.settings.chroma_persist_dir", chroma_path)
        mocker.patch("config.settings.settings.embedding_model", "nomic-embed-text")
        mocker.patch("config.settings.settings.embedding_model_version", "v1.5")
        mocker.patch("config.settings.settings.chunking_strategy", "recursive")
        mocker.patch("config.settings.settings.chunk_size", 512)
        mocker.patch("config.settings.settings.chunk_overlap", 64)

        fake_collection = _make_fake_collection()
        mocker.patch(
            "vectorstore.client.get_or_create_collection",
            return_value=fake_collection,
        )

        # "Alpha_Success" < "Zeta_Fail" alphabetically, so Alpha is processed
        # first (call 1 → succeeds) and Zeta second (call 2 → raises).
        mocker.patch(
            "ingestion.api_client.get_all_pages_with_revision_ids",
            return_value=[
                {"title": "Alpha_Success", "revision_id": 1},
                {"title": "Zeta_Fail", "revision_id": 1},
            ],
        )
        mocker.patch(
            "ingestion.api_client.get_pages_wikitext_batch",
            return_value={
                "Alpha_Success": "Some wiki content.",
                "Zeta_Fail": "Some wiki content.",
            },
        )

        call_count = {"n": 0}

        def embed_side_effect(chunks: list[str]) -> list[list[float]]:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise EmbeddingError("Simulated failure for Zeta_Fail")
            return [[0.1] * 4 for _ in chunks]

        mocker.patch(
            "ingestion.embedder.embed_chunks",
            side_effect=embed_side_effect,
        )

        from ingestion.pipeline import run_full_ingestion
        from ingestion.state_db import get_page

        with pytest.raises(EmbeddingError):
            run_full_ingestion()

        success_state = get_page("Alpha_Success")
        fail_state = get_page("Zeta_Fail")

        assert success_state is not None
        assert success_state["status"] == "complete", (
            f"Alpha_Success should be complete, got {success_state['status']}"
        )

        assert fail_state is not None
        assert fail_state["status"] == "in_progress", (
            f"Zeta_Fail should be in_progress after failure, got {fail_state['status']}"
        )

        mocker.patch(
            "ingestion.api_client.get_all_pages_with_revision_ids",
            return_value=[
                {"title": "Alpha_Success", "revision_id": 1},
                {"title": "Zeta_Fail", "revision_id": 1},
            ],
        )
        embed_spy = mocker.patch(
            "ingestion.embedder.embed_chunks",
            return_value=[[0.1] * 4],
        )

        from ingestion import state_db

        state_db.upsert_page(
            page_title="Zeta_Fail",
            revision_id=1,
            status="pending",
            embedding_model="nomic-embed-text:v1.5",
        )

        from ingestion.pipeline import run_incremental_ingestion

        run_incremental_ingestion()

        assert embed_spy.call_count == 1, (
            f"Embedder should only be called once (for Zeta_Fail), "
            f"called {embed_spy.call_count} times — "
            f"Alpha_Success may have been re-processed"
        )

        final_success = get_page("Alpha_Success")
        final_fail = get_page("Zeta_Fail")

        assert final_success is not None
        assert final_success["status"] == "complete"
        assert final_fail is not None
        assert final_fail["status"] == "complete"
