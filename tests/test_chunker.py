"""Tests for ingestion/chunker.py — recursive and section-aware splitting.

The chunker decides how a wiki page is sliced before embedding, so a
regression here directly affects retrieval quality. These tests pin
the dispatch contract, the empty-input contract, and the
section-strategy guarantee that headings are prepended to every chunk
they produced.
"""

from typing import Any

from ingestion.chunker import chunk_text, recursive_chunk, section_chunk

# ---------------------------------------------------------------------------
# chunk_text dispatch
# ---------------------------------------------------------------------------


def test_chunk_text_recursive_strategy_uses_recursive_chunker() -> None:
    """strategy='recursive' must produce non-empty chunks for non-empty text."""
    text = "First paragraph here.\n\nSecond paragraph follows."
    chunks = chunk_text(
        text=text, strategy="recursive", chunk_size=50, overlap=10
    )
    assert len(chunks) >= 1
    # Every chunk must contain real content, not be empty placeholders.
    assert all(c.strip() for c in chunks)


def test_chunk_text_section_strategy_with_sections_prepends_heading() -> None:
    """strategy='section' with sections must surface headings in every chunk.

    The whole point of section-strategy chunking is to preserve the
    heading as retrieval context. A regression that drops the heading
    on chunk N>0 would silently weaken retrieval for long sections.
    """
    sections: list[dict[str, Any]] = [
        {"heading": "Recipe", "content": "Apple, Wood, Stone"},
    ]
    chunks = chunk_text(
        text="ignored when section strategy + sections present",
        strategy="section",
        chunk_size=512,
        overlap=64,
        sections=sections,
    )
    assert chunks
    assert all(c.startswith("Recipe\n") for c in chunks), (
        "Every section-strategy chunk must be prefixed with its heading"
    )


def test_chunk_text_section_strategy_without_sections_falls_back() -> None:
    """strategy='section' with sections=None must fall through to recursive.

    Documented behavior in the dispatcher's docstring. Without this
    fallback, callers passing the config 'section' but no sections
    (e.g. an empty page that produced none) would get an empty list
    and silently lose chunks.
    """
    chunks = chunk_text(
        text="A paragraph that should chunk.",
        strategy="section",
        chunk_size=100,
        overlap=10,
        sections=None,
    )
    assert chunks
    # Fallback path must NOT prepend a heading prefix.
    assert "\n" not in chunks[0] or not chunks[0].startswith("Recipe\n")


def test_chunk_text_section_strategy_with_empty_sections_falls_back() -> None:
    """An empty sections list is also "no sections" — fallback applies.

    Pin this branch so a refactor that switches the truthiness check
    to an ``is not None`` test (which would treat ``[]`` as "has
    sections" and produce no chunks) trips the test.
    """
    chunks = chunk_text(
        text="Some content here.",
        strategy="section",
        chunk_size=100,
        overlap=10,
        sections=[],
    )
    assert chunks


# ---------------------------------------------------------------------------
# recursive_chunk
# ---------------------------------------------------------------------------


def test_recursive_chunk_empty_text_returns_empty_list() -> None:
    """Empty input must short-circuit before invoking the splitter."""
    assert recursive_chunk("", chunk_size=512, overlap=64) == []


def test_recursive_chunk_whitespace_only_returns_empty_list() -> None:
    """Whitespace-only input is effectively empty.

    Without this guard, the splitter could produce a single chunk of
    whitespace that downstream embedding would waste tokens on.
    """
    assert recursive_chunk("   \n\n   ", chunk_size=512, overlap=64) == []


def test_recursive_chunk_short_text_fits_in_single_chunk() -> None:
    """Text smaller than chunk_size must produce exactly one chunk.

    Off-by-one in the splitter could either produce zero chunks
    (loss of content) or split a 10-char string into two — both
    silent regressions for short pages.
    """
    text = "Short content."
    chunks = recursive_chunk(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].strip() == text


def test_recursive_chunk_long_text_splits_into_multiple_chunks() -> None:
    """Text larger than chunk_size must produce more than one chunk.

    Lower bound: any oversize text → ≥2 chunks. The exact count
    depends on the langchain splitter's separators, so we don't pin
    a specific number — only that splitting actually happened.
    """
    text = "First paragraph.\n\n" + "Body sentence. " * 100
    chunks = recursive_chunk(text, chunk_size=100, overlap=10)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# section_chunk
# ---------------------------------------------------------------------------


def test_section_chunk_emits_one_chunk_per_short_section() -> None:
    """Each short section must produce exactly one heading-prefixed chunk."""
    sections: list[dict[str, Any]] = [
        {"heading": "Recipe", "content": "Apple, Wood"},
        {"heading": "Notes", "content": "Found in orchards"},
    ]
    chunks = section_chunk(sections, chunk_size=512, overlap=64)

    assert len(chunks) == 2
    assert chunks[0] == "Recipe\nApple, Wood"
    assert chunks[1] == "Notes\nFound in orchards"


def test_section_chunk_skips_sections_with_empty_content() -> None:
    """A section whose body is whitespace must contribute zero chunks.

    Without this filter, an empty section would emit just its heading
    line as a "chunk" — a low-signal artifact that pollutes retrieval.
    """
    sections: list[dict[str, Any]] = [
        {"heading": "Empty", "content": "   "},
        {"heading": "Filled", "content": "Some words here."},
    ]
    chunks = section_chunk(sections, chunk_size=512, overlap=64)

    assert len(chunks) == 1
    assert chunks[0].startswith("Filled\n")


def test_section_chunk_omits_heading_prefix_when_heading_empty() -> None:
    """An intro section (heading='') must not get a leading newline.

    The intro section uses '' as its heading. Prefixing with
    ``f"{heading}\\n{chunk}"`` when heading is empty would yield
    ``"\\n<text>"`` — a leading newline that the embedder treats as
    meaningful whitespace.
    """
    sections: list[dict[str, Any]] = [
        {"heading": "", "content": "Intro paragraph content."},
    ]
    chunks = section_chunk(sections, chunk_size=512, overlap=64)

    assert len(chunks) == 1
    assert not chunks[0].startswith("\n")
    assert chunks[0] == "Intro paragraph content."


# ---------------------------------------------------------------------------
# boilerplate-chunk filter
# ---------------------------------------------------------------------------


def test_chunk_text_recursive_drops_header_only_chunks() -> None:
    """A chunk that resolves to a known section-header gets filtered out.

    The recursive splitter routinely isolates wiki section headers
    (``"Quest Information"``, ``"Past Events"``) into their own chunks
    when the section body sits in a separate split window. Those chunks
    carry no answerable content but, because their text is identical
    across many pages, share embeddings and crowd top-k results — so
    chunk_text drops them.
    """
    text = "Quest Information\n\nBody paragraph with real information."
    chunks = chunk_text(
        text=text, strategy="recursive", chunk_size=20, overlap=0
    )
    assert "Quest Information" not in chunks
    assert any("Body paragraph" in c for c in chunks)


def test_chunk_text_section_drops_header_only_chunks() -> None:
    """The same allowlist applies when section_chunk emits a bare header.

    Section-strategy with an empty heading and boilerplate body produces
    a chunk equal to the boilerplate text — exactly what the recursive
    path produces. Both paths share the post-chunking filter.
    """
    sections: list[dict[str, Any]] = [
        {"heading": "", "content": "Quest Information"},
        {"heading": "", "content": "Real prose follows here."},
    ]
    chunks = chunk_text(
        text="",
        strategy="section",
        chunk_size=512,
        overlap=64,
        sections=sections,
    )
    assert "Quest Information" not in chunks
    assert "Real prose follows here." in chunks


def test_chunk_text_does_not_drop_boilerplate_inside_larger_chunk() -> None:
    """The filter is exact-stripped-text match — substring matches are kept.

    A real chunk that happens to contain the literal "Quest Information"
    inside a longer body must survive: only chunks whose entire content
    *is* the section header are artifacts.
    """
    text = (
        "Quest Information: The player must speak to Hello Kitty before "
        "continuing on the way."
    )
    chunks = chunk_text(
        text=text, strategy="recursive", chunk_size=512, overlap=0
    )
    assert chunks
    assert text in chunks[0]


def test_chunk_text_drops_boilerplate_with_trailing_whitespace() -> None:
    """Stripping happens before the allowlist check.

    The splitter occasionally hands us chunks with trailing newlines —
    pin that a literal ``"Quest Information\\n"`` still matches and gets
    filtered, so the filter is robust to splitter whitespace quirks.
    """
    # Force a known boilerplate chunk to reach chunk_text via the
    # section path (more deterministic than coaxing the recursive
    # splitter into producing trailing whitespace).
    sections: list[dict[str, Any]] = [
        {"heading": "", "content": "Quest Information   \n"},
    ]
    chunks = chunk_text(
        text="",
        strategy="section",
        chunk_size=512,
        overlap=64,
        sections=sections,
    )
    assert chunks == []


def test_section_chunk_recursively_splits_oversized_sections() -> None:
    """A section larger than chunk_size must produce multiple chunks.

    Each sub-chunk must still carry the heading prefix — a regression
    that only prepended the heading to the first sub-chunk would
    silently weaken retrieval for any long section.
    """
    long_content = "Sentence. " * 200
    sections: list[dict[str, Any]] = [
        {"heading": "Long Section", "content": long_content},
    ]
    chunks = section_chunk(sections, chunk_size=100, overlap=10)

    assert len(chunks) >= 2
    assert all(c.startswith("Long Section\n") for c in chunks), (
        "Every sub-chunk of an oversized section must keep the heading prefix"
    )
