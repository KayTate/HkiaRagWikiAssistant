"""Text chunking strategies for the HKIA ingestion pipeline.

Provides two strategies: recursive character splitting (default) and
section-aware splitting that preserves section headings as context.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_RECURSIVE_SEPARATORS = ["\n\n", "\n", " ", ""]


def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    sections: list[dict] | None = None,  # type: ignore[type-arg]
) -> list[str]:
    """Split text into chunks using the specified strategy.

    Dispatches to recursive_chunk or section_chunk depending on the
    strategy argument. The 'section' strategy requires sections to be
    provided; if they are absent, falls back to 'recursive'.

    Args:
        text: Full plain-text content of the page (used by 'recursive').
        strategy: Either 'recursive' or 'section'.
        chunk_size: Target maximum character count per chunk.
        overlap: Number of characters to overlap between consecutive chunks.
        sections: Pre-extracted sections for the 'section' strategy.
            Required when strategy == 'section'.

    Returns:
        List of text chunks in document order.
    """
    if strategy == "section" and sections:
        return section_chunk(sections, chunk_size, overlap)
    return recursive_chunk(text, chunk_size, overlap)


def recursive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split plain text using LangChain's RecursiveCharacterTextSplitter.

    Tries to split on paragraph boundaries first, then line boundaries,
    then word boundaries, then characters. This preserves semantic units
    wherever possible.

    Args:
        text: Plain text to split.
        chunk_size: Target maximum character count per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks. Empty input returns an empty list.
    """
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        separators=_RECURSIVE_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)


def section_chunk(
    sections: list[dict],  # type: ignore[type-arg]
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split a page's sections into chunks, preserving section heading context.

    Each section is chunked independently. The section heading is prepended
    to every chunk produced from that section so the heading context is not
    lost when the chunk is retrieved in isolation. Sections whose content
    exceeds chunk_size are recursively split.

    Args:
        sections: List of dicts with 'heading' and 'content' keys, as
            returned by ingestion.parser.extract_sections.
        chunk_size: Target maximum character count per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunks in document order, each prefixed with its section
        heading when the heading is non-empty.
    """
    all_chunks: list[str] = []
    for section in sections:
        heading = section.get("heading", "")
        content = section.get("content", "")
        if not content.strip():
            continue
        sub_chunks = recursive_chunk(content, chunk_size, overlap)
        for sub_chunk in sub_chunks:
            if heading:
                all_chunks.append(f"{heading}\n{sub_chunk}")
            else:
                all_chunks.append(sub_chunk)
    return all_chunks
