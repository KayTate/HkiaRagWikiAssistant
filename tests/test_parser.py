"""Tests for ingestion/parser.py — wikitext template expansion + sectioning.

These tests pin specific behaviors of the wiki-template handlers so a
silent change to one expander cannot corrupt every chunk that flows
through ingestion. Each template kind gets at least one assertion that
would fail if the handler returned the wrong field, the wrong join
character, or the empty string.
"""

from ingestion.parser import extract_sections, parse_wikitext

# ---------------------------------------------------------------------------
# parse_wikitext
# ---------------------------------------------------------------------------


def test_parse_wikitext_returns_empty_string_for_empty_input() -> None:
    """Empty input must round-trip to an empty string, not raise."""
    assert parse_wikitext("") == ""


def test_parse_wikitext_strips_plain_wiki_links() -> None:
    """[[Page|display]] must reduce to the display text via strip_code."""
    out = parse_wikitext("Visit [[Apple Orchard|the orchard]] for apples.")
    assert "the orchard" in out
    assert "[[" not in out and "]]" not in out


def test_parse_wikitext_expands_icon_link_default_arg() -> None:
    """{{Icon/Link|Name}} must surface the first positional arg as plain text.

    Without the expander, mwparserfromhell.strip_code drops the entire
    template — losing the item name from a recipe sentence and silently
    making chunks less informative.
    """
    out = parse_wikitext("Recipe: {{Icon/Link|Apple}} and {{Icon/Link|Wood}}.")
    assert "Apple" in out
    assert "Wood" in out


def test_parse_wikitext_expands_icon_link_text_override() -> None:
    """{{Icon/Link|Name|text=Display}} must use the ``text=`` value.

    A regression to "always use first positional" would corrupt links
    whose displayed name differs from the canonical item name.
    """
    out = parse_wikitext("Found {{Icon/Link|Apple|text=Apple Slice}} today.")
    assert "Apple Slice" in out
    assert "Apple Slice today" in out


def test_parse_wikitext_expands_friendship_two_arg_form() -> None:
    """{{Friendship|Character|Level}} must produce the readable two-arg phrasing.

    Pinning the exact phrase guards against a refactor that flips the
    arg order (level-then-character) — an easy-to-miss off-by-one in
    the f-string.
    """
    out = parse_wikitext("Requires {{Friendship|Keroppi|4}}.")
    assert "friendship level 4 with Keroppi" in out


def test_parse_wikitext_expands_rarity_template() -> None:
    """{{Rarity|Rare}} must expand to its single positional value."""
    out = parse_wikitext("This is a {{Rarity|Rare}} item.")
    assert "Rare" in out


def test_parse_wikitext_expands_tag_template() -> None:
    """{{Tag|...}} also uses the simple-param expander; pin its behavior too.

    Documents that ``Tag`` and ``Rarity`` share an expander; a future
    refactor that splits them must preserve the current output.
    """
    out = parse_wikitext("{{Tag|Crafting}}")
    assert "Crafting" in out


def test_parse_wikitext_expands_infobox_to_key_value_lines() -> None:
    """Infobox Item must surface readable Title-cased keys with values.

    The expander deliberately skips ``image``/``caption``/``alt`` to
    avoid emitting binary asset references; the assertion below would
    fail if that allowlist regressed and started leaking those fields.
    """
    wikitext = (
        "{{Infobox Item\n"
        "|name=Wooden Bench\n"
        "|rarity=Common\n"
        "|image=Wooden_Bench.png\n"
        "}}"
    )
    out = parse_wikitext(wikitext)
    assert "Name: Wooden Bench" in out
    assert "Rarity: Common" in out
    # image was filtered intentionally — must not appear.
    assert "Wooden_Bench.png" not in out


def test_parse_wikitext_recursively_expands_templates_inside_infobox() -> None:
    """Infobox values containing nested templates must themselves expand.

    A regression that stops at the first level would leave raw
    ``{{Icon/Link|...}}`` syntax inside the rendered key:value line.
    """
    wikitext = "{{Infobox Item|name={{Icon/Link|Wooden Bench|text=Wooden Bench}}}}"
    out = parse_wikitext(wikitext)
    assert "Name: Wooden Bench" in out
    assert "{{" not in out


def test_parse_wikitext_cleans_paragraph_html_fragments() -> None:
    """Wiki tables leave <p> fragments; they must become readable separators.

    Without _clean_html_fragments, recipe-ingredient lines render as
    ``Apple<p>Wood<p>Stone`` instead of ``Apple, Wood, Stone``.
    """
    out = parse_wikitext("Apple<p>Wood<p>Stone")
    assert "<p>" not in out
    assert "Apple" in out and "Wood" in out and "Stone" in out


def test_parse_wikitext_strips_unknown_template_silently() -> None:
    """A template with no registered expander must be removed by strip_code.

    Pin this so a future addition that returns ``str(template)`` as a
    fallback (which would leak raw wikitext into chunks) trips the test.
    """
    out = parse_wikitext("Before {{UnknownTemplate|foo|bar}} after.")
    assert "{{" not in out
    assert "UnknownTemplate" not in out
    assert "Before" in out
    assert "after" in out


# ---------------------------------------------------------------------------
# extract_sections
# ---------------------------------------------------------------------------


def test_extract_sections_returns_empty_list_for_empty_input() -> None:
    """No content → no sections."""
    assert extract_sections("") == []


def test_extract_sections_returns_one_intro_section_for_unsectioned_text() -> None:
    """Wikitext without headings must produce a single empty-heading section.

    Documents that the parser treats pre-heading content as the intro
    section, not as "no sections". Important because chunker's
    section-strategy code branches on the heading being empty.
    """
    sections = extract_sections("Just some text with no headings.")
    assert len(sections) == 1
    assert sections[0]["heading"] == ""
    assert "Just some text" in sections[0]["content"]


def test_extract_sections_splits_on_headings() -> None:
    """Each heading must produce a separate section with its own content.

    Two assertions: the count, and the heading-content pairing. A
    regression that drops the first character of each heading (a
    common off-by-one when slicing past ``==`` markers) would fail
    on the heading text.
    """
    wikitext = (
        "Intro paragraph.\n"
        "==Recipe==\n"
        "Apple, Wood\n"
        "==Notes==\n"
        "Found in the orchard.\n"
    )
    sections = extract_sections(wikitext)

    headings = [s["heading"] for s in sections]
    assert headings == ["", "Recipe", "Notes"]
    by_heading = {s["heading"]: s["content"] for s in sections}
    assert "Apple" in by_heading["Recipe"]
    assert "orchard" in by_heading["Notes"]


def test_extract_sections_drops_empty_sections() -> None:
    """A heading with no body must not produce an entry.

    Without this filter, downstream chunker would emit a chunk that is
    just the heading prefix with empty content — wasted retrieval space.
    """
    wikitext = "==Empty==\n\n==Has Content==\nReal text here.\n"
    sections = extract_sections(wikitext)

    headings = [s["heading"] for s in sections]
    assert "Empty" not in headings
    assert "Has Content" in headings
