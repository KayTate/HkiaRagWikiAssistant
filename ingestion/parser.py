"""Wikitext parsing utilities for the HKIA ingestion pipeline.

Uses mwparserfromhell to strip MediaWiki markup and extract section
structure from raw wikitext. Common wiki templates (Icon/Link,
Friendship, Rarity, Tag, etc.) are expanded to plain text before
stripping so that item names, quantities, and game details are
preserved in the output.
"""

import logging
import re

import mwparserfromhell
from mwparserfromhell.nodes import Template
from mwparserfromhell.wikicode import Wikicode

logger = logging.getLogger(__name__)


def _expand_templates(parsed: Wikicode) -> None:
    """Replace known wiki templates with their plain-text equivalents in-place.

    Handles the most common templates used on the HKIA wiki so that item
    names, recipe ingredients, and game metadata survive strip_code().
    Unknown templates are left untouched for strip_code() to remove.

    Args:
        parsed: A mwparserfromhell Wikicode object to modify in place.
    """
    for template in parsed.filter_templates(recursive=True):
        name = template.name.strip_code().strip().lower()
        replacement = _template_to_text(name, template)
        if replacement is not None:
            try:
                parsed.replace(template, replacement)
            except ValueError:
                pass


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
    """Expand {{Icon/Link|Name|text=Display}} to the display name.

    Args:
        template: An Icon/Link template node.

    Returns:
        The item/entity name as plain text.
    """
    if template.has("text"):
        return str(template.get("text").value).strip()
    if template.params:
        return str(template.params[0].value).strip()
    return ""


def _expand_friendship(template: Template) -> str:
    """Expand {{Friendship|Character|Level}} to readable text.

    Args:
        template: A Friendship template node.

    Returns:
        e.g. 'friendship level 4 with Keroppi'
    """
    params = [str(p.value).strip() for p in template.params]
    if len(params) >= 2:
        return f"friendship level {params[1]} with {params[0]}"
    if params:
        return f"friendship with {params[0]}"
    return ""


def _expand_simple_param(template: Template) -> str:
    """Expand templates that just wrap a single value like {{Rarity|Rare}}.

    Args:
        template: A single-param template node.

    Returns:
        The parameter value as plain text.
    """
    if template.params:
        return str(template.params[0].value).strip()
    return ""


def _expand_item_description(template: Template) -> str:
    """Expand {{Item Description|col=N|Description text}}.

    The description is usually the last positional parameter.

    Args:
        template: An Item Description template node.

    Returns:
        The description text.
    """
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
    """Replace common HTML leftovers from wiki table markup.

    Wiki tables with <p> tags between recipe ingredients leave fragments
    after strip_code(). This converts them to readable separators.

    Args:
        text: Text that may contain <p> tags or other HTML fragments.

    Returns:
        Cleaned text with HTML replaced by commas or spaces.
    """
    text = re.sub(r"<p>", ", ", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"\s*,\s*,\s*", ", ", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def parse_wikitext(wikitext: str) -> str:
    """Convert raw wikitext to clean plain text for embedding.

    Expands known templates to preserve item names and game details,
    then strips remaining markup via mwparserfromhell. Cleans up any
    leftover HTML fragments from wiki tables.

    Args:
        wikitext: Raw wikitext string as returned by the MediaWiki API.

    Returns:
        Clean plain text with meaningful content preserved.
    """
    parsed = mwparserfromhell.parse(wikitext)
    _expand_templates(parsed)
    text = parsed.strip_code(normalize=True, collapse=True)
    text = _clean_html_fragments(text)
    return text.strip()


def extract_sections(wikitext: str) -> list[dict]:  # type: ignore[type-arg]
    """Parse wikitext into a list of heading-and-content section dicts.

    Expands known templates before extracting sections so that section
    content retains item names and game details.

    Args:
        wikitext: Raw wikitext string as returned by the MediaWiki API.

    Returns:
        List of dicts with keys 'heading' (str) and 'content' (str), in
        document order. 'heading' is empty string for the introductory
        section before the first heading.
    """
    parsed = mwparserfromhell.parse(wikitext)
    _expand_templates(parsed)
    sections: list[dict] = []  # type: ignore[type-arg]
    current_heading = ""
    current_parts: list[str] = []

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.Heading):
            _flush_section(sections, current_heading, current_parts)
            current_heading = node.title.strip_code().strip()
            current_parts = []
        else:
            text = mwparserfromhell.parse(str(node)).strip_code(
                normalize=True, collapse=True
            )
            text = _clean_html_fragments(text)
            if text.strip():
                current_parts.append(text)

    _flush_section(sections, current_heading, current_parts)
    return sections


def _flush_section(
    sections: list[dict],  # type: ignore[type-arg]
    heading: str,
    parts: list[str],
) -> None:
    """Append a completed section to the sections list if it has content.

    Args:
        sections: Accumulator list to append to.
        heading: The section heading, or empty string for the intro section.
        parts: Text fragments collected for this section.
    """
    content = " ".join(parts).strip()
    if content:
        sections.append({"heading": heading, "content": content})
