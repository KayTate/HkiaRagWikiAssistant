"""Pydantic schema for ChromaDB chunk documents."""

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk document stored in ChromaDB.

    The embedding_model field uses the format "{model_name}:{version}",
    for example "nomic-embed-text:v1.5". This value is also mirrored in
    the SQLite page_ingestion_state table to allow cross-store consistency
    checks and automatic re-ingestion when the model changes.

    Numeric fields carry minimum-value constraints because a negative
    chunk_size or chunk_overlap is always a programming error — better
    to fail at construction than to write garbage metadata that
    downstream queries silently propagate. ``revision_id`` allows -1 to
    accommodate the sentinel used by the startup sync check when
    resetting a stale row to pending (see ingestion.pipeline).
    """

    source_title: str
    source_url: str
    section: str  # empty string if unsectioned, never None
    category: str
    chunk_index: int = Field(ge=0)
    revision_id: int = Field(ge=-1)  # -1 sentinel during stale-row reset
    ingested_at: str  # ISO 8601
    embedding_model: str  # format: "{model_name}:{version}"
    chunking_strategy: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
