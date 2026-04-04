"""Pydantic schema for ChromaDB chunk documents."""

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk document stored in ChromaDB.

    The embedding_model field uses the format "{model_name}:{version}",
    for example "nomic-embed-text:v1.5". This value is also mirrored in
    the SQLite page_ingestion_state table to allow cross-store consistency
    checks and automatic re-ingestion when the model changes.
    """

    source_title: str
    source_url: str
    section: str  # empty string if unsectioned, never None
    category: str
    chunk_index: int
    revision_id: int
    ingested_at: str  # ISO 8601
    embedding_model: str  # format: "{model_name}:{version}"
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
