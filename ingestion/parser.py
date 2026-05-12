"""Wikitext parsing utilities for the HKIA ingestion pipeline.

Uses mwparserfromhell to strip MediaWiki markup and extract section
structure from raw wikitext. Common wiki templates (Icon/Link,
Friendship, Rarity, Tag, etc.) are expanded to plain text before
stripping so that item names, quantities, and game details are
preserved in the output.
"""

import logging
import re
from typing import Any

import mwparserfromhell
from mwparserfromhell.nodes import Template
from mwparserfromhell.wikicode import Wikicode

logger = logging.getLogger(__name__)


def _expand_templates(parsed: Wikicode) -> None:
    """Replace known wiki templates with plain-text equivalents in-place."""
    for template in parsed.filter_templates(recursive=True):
        name = template.name.strip_code().strip().lower()
        replacement = _template_to_text(name, template)
        if replacement is not None:
            try:
                parsed.replace(template, replacement)
            except ValueError as exc:
                # mwparserfromhell raises ValueError when the template
                # was already replaced by an outer-template pass — this
                # is expected and benign. Logging at DEBUG (not WARN)
                # keeps the signal out of normal logs while still
                # giving an operator a way to surface it during
                # template-debugging sessions.
                logger.debug(
                    "Template '%s' already replaced; skipping: %s", name, exc
                )


def _template_to_text(name: str, template: Template) -> str | None:
    """Convert a single template to plain text if it is a known type.

    Args:
        name: Lowercase, stripped template name.
        template: The mwparserfromhell Template node.

    Returns:
        Plain text replacement string, or None to leave the template
        for strip_code() to handle.
    """
    if name in ("icon/link", "iconlink", "icon"):
        return _expand_icon_link(template)
    if name == "friendship":
        return _expand_friendship(template)
    if name == "rarity":
        return _expand_simple_param(template)
    if name == "tag":
        return _expand_simple_param(template)
    if name == "item description":
        return _expand_item_description(template)
    if name == "infobox item":
        return _expand_infobox(template)
    return None


def _expand_icon_link(template: Template) -> str:
    """Expand ``{{Icon/Link|Name|text=Display}}`` to the display name."""
    if template.has("text"):
        return str(template.get("text").value).strip()
    if template.params:
        return str(template.params[0].value).strip()
    return ""


def _expand_friendship(template: Template) -> str:
    """Expand ``{{Friendship|Character|Level}}`` to readable text."""
    params = [str(p.value).strip() for p in template.params]
    if len(params) >= 2:
        return f"friendship level {params[1]} with {params[0]}"
    if params:
        return f"friendship with {params[0]}"
    return ""


def _expand_simple_param(template: Template) -> str:
    """Expand single-value templates like ``{{Rarity|Rare}}`` to the value."""
    if template.params:
        return str(template.params[0].value).strip()
    return ""


def _expand_item_description(template: Template) -> str:
    """Expand ``{{Item Description|col=N|...}}`` to the last positional value."""
    for param in reversed(template.params):
        if not param.showkey:
            return str(param.value).strip()
    return ""


def _expand_infobox(template: Template) -> str:
    """Expand {{Infobox Item|...}} to key-value lines.

    Extracts the most useful fields from infobox templates and formats
    them as readable text. Template values that themselves contain
    templates (like Icon/Link) are recursively expanded.

    Args:
        template: An Infobox Item template node.

    Returns:
        Multi-line plain text summary of the infobox fields.
    """
    lines: list[str] = []
    for param in template.params:
        key = str(param.name).strip().lower()
        if key in ("image", "caption", "alt"):
            continue
        value_parsed = mwparserfromhell.parse(str(param.value))
        _expand_templates(value_parsed)
        value_text = value_parsed.strip_code().strip()
        if value_text:
            display_key = key.replace("_", " ").title()
            lines.append(f"{display_key}: {value_text}")
    return "\n".join(lines)


def _clean_html_fragments(text: str) -> str:
    """Replace HTML leftovers from wiki table markup with commas/spaces."""
    text = re.sub(r"<p>", ", ", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"\s*,\s*,\s*", ", ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Non-whitespace sentinel survives strip_code(collapse=True); a raw \n does not.
_CELL_BREAK = "@@HKIA_CELL@@"


def _strip_with_cell_breaks(parsed: Wikicode) -> str:
    """Strip wikicode but preserve <td>/<th> cell boundaries as newlines.

    Without this, adjacent table cells collapse into one string and
    downstream LLMs cannot tell where one row's data ends and the next
    begins (e.g. companion abilities glue to the wrong character).
    """
    for tag in parsed.filter_tags(recursive=True):
        if tag.tag in ("td", "th"):
            tag.contents.insert(0, _CELL_BREAK)
    text = parsed.strip_code(normalize=True, collapse=True)
    return text.replace(_CELL_BREAK, "\n")


def _drop_file_wikilinks(parsed: Wikicode) -> None:
    """Drop [[File:...]]/[[Image:...]] links so thumbnail debris stays out of chunks."""
    for link in parsed.filter_wikilinks():
        title = str(link.title).strip().lower()
        if title.startswith(("file:", "image:")):
            try:
                parsed.remove(link)
            except ValueError as exc:
                # Mirrors the benign-already-removed case in _expand_templates.
                logger.debug(
                    "Wikilink '%s' already removed; skipping: %s", link.title, exc
                )


def parse_wikitext(wikitext: str) -> str:
    """Convert raw wikitext to clean plain text for embedding."""
    parsed = mwparserfromhell.parse(wikitext)
    _expand_templates(parsed)
    _drop_file_wikilinks(parsed)
    text = _strip_with_cell_breaks(parsed)
    text = _clean_html_fragments(text)
    return text.strip()


def extract_sections(wikitext: str) -> list[dict[str, Any]]:
    """Parse wikitext into ordered ``{'heading', 'content'}`` section dicts.

    'heading' is empty string for the introductory section before the
    first heading.
    """
    parsed = mwparserfromhell.parse(wikitext)
    _expand_templates(parsed)
    _drop_file_wikilinks(parsed)
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_parts: list[str] = []

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.Heading):
            _flush_section(sections, current_heading, current_parts)
            current_heading = node.title.strip_code().strip()
            current_parts = []
        else:
            node_parsed = mwparserfromhell.parse(str(node))
            text = _strip_with_cell_breaks(node_parsed)
            text = _clean_html_fragments(text)
            if text.strip():
                current_parts.append(text)

    _flush_section(sections, current_heading, current_parts)
    return sections


def _flush_section(
    sections: list[dict[str, Any]],
    heading: str,
    parts: list[str],
) -> None:
    """Append the completed section to the list if it has any content."""
    content = " ".join(parts).strip()
    if content:
        sections.append({"heading": heading, "content": content})
