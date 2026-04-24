"""Shared pytest fixtures for the HKIA test suite."""

from collections.abc import Iterator

import pytest

from vectorstore.client import reset_client


@pytest.fixture(autouse=True)
def _reset_chroma_singleton() -> Iterator[None]:
    """Reset the ChromaDB client singleton around every test.

    Keeps tests isolated from one another when ``chroma_persist_dir`` is
    patched to a tempdir. Runs before and after each test so a leak from a
    previous test cannot poison the next one.
    """
    reset_client()
    yield
    reset_client()
