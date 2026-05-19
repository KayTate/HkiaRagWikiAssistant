"""Generate a synthetic evaluation dataset by fetching pages and prompting the LLM.

Usage:
    python scripts/generate_eval_dataset.py \\
        --pages "Wooden Bench,Marigold Candle" \\
        --question-type crafting \\
        --output data/eval/synthetic_set.json

    python scripts/generate_eval_dataset.py \\
        --pages-file data/eval/synthetic_pages.txt \\
        --question-type general \\
        --output data/eval/synthetic_set.json

Pulls source content directly from the parser (``parse_wikitext`` →
``extract_sections``) rather than from the ChromaDB collection, so the
resulting dataset is independent of the chunking strategy used for
ingestion. Each section of each page is sent to ``generate_for_chunk``;
its output (golden-set-shaped pairs) is enriched with ``source`` and
``source_title`` in metadata and written to the output JSON file.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

# Allow `python scripts/generate_eval_dataset.py` to import top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging  # noqa: E402
from eval.generate import generate_for_chunk  # noqa: E402
from ingestion.api_client import WikiAPIError, get_page_wikitext  # noqa: E402
from ingestion.parser import (  # noqa: E402
    detect_redirect_target,
    extract_sections,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_SECTION_TOKENS = 20


def _read_pages_arg(pages: str | None, pages_file: str | None) -> list[str]:
    """Parse the page-title input from either --pages or --pages-file.

    Args:
        pages: Comma-separated page titles passed inline.
        pages_file: Path to a file with one page title per line.

    Returns:
        Ordered, de-duplicated list of page titles.

    Raises:
        ValueError: If neither input is provided, both are provided, or
            the resolved list is empty.
        FileNotFoundError: If --pages-file points at a missing file.
    """
    if pages and pages_file:
        raise ValueError("Pass --pages or --pages-file, not both.")
    if not pages and not pages_file:
        raise ValueError("One of --pages or --pages-file is required.")

    raw_titles: list[str]
    if pages_file:
        file_path = Path(pages_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Pages file not found: {pages_file}")
        raw_titles = file_path.read_text(encoding="utf-8").splitlines()
    else:
        assert pages is not None
        raw_titles = pages.split(",")

    seen: set[str] = set()
    ordered: list[str] = []
    for title in raw_titles:
        stripped = title.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)

    if not ordered:
        raise ValueError("No page titles found in the provided input.")
    return ordered


def _section_to_chunk(
    page_title: str,
    section: dict[str, object],
) -> dict[str, object]:
    """Convert a parsed section into the chunk shape ``generate_for_chunk`` expects.

    The heading is prepended to the content so the LLM sees the
    section's topic — important for sections whose body is a bare list.

    Args:
        page_title: The wiki page this section came from.
        section: A dict produced by ``parser.extract_sections``, with
            ``heading`` and ``content`` keys.

    Returns:
        A dict with ``text`` and ``metadata`` keys, ready to pass to
        ``generate_for_chunk``.
    """
    heading = str(section.get("heading", "")).strip()
    content = str(section.get("content", "")).strip()
    text = f"{heading}\n\n{content}" if heading else content
    return {
        "text": text,
        "metadata": {
            "source_title": page_title,
            "heading": heading,
        },
    }


def _enrich_pair(
    pair: dict[str, object],
    page_title: str,
    heading: str,
) -> dict[str, object]:
    """Add provenance fields to a pair's metadata before writing.

    The validated pair already has a ``metadata`` dict with
    ``question_type``; this merges in ``source``, ``source_title``, and
    ``heading`` so downstream eval runs can filter or trace each
    entry back to its origin section.
    """
    metadata: dict[str, object] = dict(cast(dict[str, object], pair["metadata"]))
    metadata.setdefault("source", "synthetic")
    metadata["source_title"] = page_title
    if heading:
        metadata["heading"] = heading
    return {
        "inputs": pair["inputs"],
        "expected_response": pair["expected_response"],
        "metadata": metadata,
    }


def _generate_for_page(
    page_title: str,
    question_type: str,
    n_pairs: int | None,
    min_section_tokens: int,
) -> list[dict[str, object]]:
    """Fetch a page, parse its sections, and generate pairs for each.

    Args:
        page_title: The wiki page title to fetch.
        question_type: Category label forwarded to the LLM.
        n_pairs: Optional override for the per-section pair count.
        min_section_tokens: Sections with fewer whitespace-split tokens
            than this are skipped — they rarely yield distinct questions.

    Returns:
        List of golden-set-shaped pairs (possibly empty) with
        provenance metadata attached.
    """
    try:
        wikitext = get_page_wikitext(page_title)
    except WikiAPIError:
        logger.exception("Failed to fetch wikitext for %r; skipping page", page_title)
        return []

    redirect_target = detect_redirect_target(wikitext)
    if redirect_target is not None:
        logger.info(
            "Skipping redirect %r → %r (pass the target title directly)",
            page_title,
            redirect_target,
        )
        return []

    sections = extract_sections(wikitext)
    if not sections:
        logger.warning("No sections extracted from %r; skipping", page_title)
        return []

    pairs: list[dict[str, object]] = []
    for section in sections:
        content = str(section.get("content", "")).strip()
        if len(content.split()) < min_section_tokens:
            logger.debug(
                "Skipping short section %r/%r (<%d tokens)",
                page_title,
                section.get("heading", ""),
                min_section_tokens,
            )
            continue

        chunk = _section_to_chunk(page_title, section)
        section_pairs = generate_for_chunk(chunk, question_type, n_pairs)
        heading = str(section.get("heading", "")).strip()
        for pair in section_pairs:
            pairs.append(_enrich_pair(pair, page_title, heading))

    logger.info("Generated %d pairs from %r", len(pairs), page_title)
    return pairs


def main() -> None:
    """Parse CLI args, drive generation across pages, and write the dataset."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic Q&A dataset from wiki pages, in the golden-set "
            "schema consumed by scripts/run_eval.py."
        ),
    )
    parser.add_argument(
        "--pages",
        help="Comma-separated wiki page titles (e.g. 'Wooden Bench,Marigold Candle').",
    )
    parser.add_argument(
        "--pages-file",
        help="Path to a text file with one wiki page title per line.",
    )
    parser.add_argument(
        "--question-type",
        required=True,
        help="Question category label passed to the LLM (e.g. 'crafting').",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file (overwrites if it exists).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help=(
            "Override the per-section pair count. If omitted, generate.py "
            "picks 1 for short sections and 2 for long ones."
        ),
    )
    parser.add_argument(
        "--min-section-tokens",
        type=int,
        default=DEFAULT_MIN_SECTION_TOKENS,
        help=(
            "Skip sections whose stripped content has fewer whitespace-split "
            f"tokens than this (default {DEFAULT_MIN_SECTION_TOKENS})."
        ),
    )
    args = parser.parse_args()

    try:
        titles = _read_pages_arg(args.pages, args.pages_file)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    logger.info(
        "Generating synthetic dataset from %d page(s) → %s", len(titles), args.output
    )

    dataset: list[dict[str, object]] = []
    for title in titles:
        dataset.extend(
            _generate_for_page(
                page_title=title,
                question_type=args.question_type,
                n_pairs=args.n_pairs,
                min_section_tokens=args.min_section_tokens,
            )
        )

    if not dataset:
        logger.error("No pairs were generated; refusing to write an empty dataset.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d entries to %s", len(dataset), output_path)
    print(f"Wrote {len(dataset)} entries to {output_path}")


if __name__ == "__main__":
    main()
