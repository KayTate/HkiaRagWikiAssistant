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


def test_parse_wikitext_separates_wiki_table_cells() -> None:
    """Wikitable {| ... |} cells must be split, not concatenated end-to-end.

    strip_code drops |/||/!/|- delimiters without inserting whitespace.
    Without the cell-break sentinel, every table on the wiki produces
    one continuous string of glued-together cell contents.
    """
    wikitext = "{| class=\"wikitable\"\n| A || B\n|-\n| C || D\n|}"
    out = parse_wikitext(wikitext)
    assert "AB" not in out
    assert "CD" not in out
    for cell in ("A", "B", "C", "D"):
        assert cell in out


def test_parse_wikitext_separates_html_table_cells() -> None:
    """HTML <table>/<td> cells must be split too — same root cause as wiki tables.

    mwparserfromhell parses both syntaxes into td/th Tag nodes, so the
    same fix covers both. Pin this to prevent a regression that special-
    cases only one syntax flavor.
    """
    out = parse_wikitext("<table><tr><td>A</td><td>B</td></tr></table>")
    assert "AB" not in out
    assert "A" in out and "B" in out


def test_parse_wikitext_does_not_glue_companion_ability_to_next_character() -> None:
    """Repro for the eval failure on "When do I unlock the Slow & Steady ability?".

    Without the cell-break fix, "Slow & Steady" (last bullet of
    Pompompurin's cell) concatenates with "Retsuko" (first text of the
    next cell), causing the LLM to attribute the ability to the wrong
    character at the wrong friendship level.
    """
    wikitext = (
        "{| class=\"wikitable\"\n"
        "|-\n"
        "|Pompompurin (5)\n"
        "*Pudding Pants\n"
        "*Slow & Steady\n"
        "|Retsuko (16)\n"
        "*Hot Stuff\n"
        "*Adrenaline\n"
        "|}"
    )
    out = parse_wikitext(wikitext)
    assert "SteadyRetsuko" not in out
    assert "Slow & Steady" in out
    assert "Retsuko" in out


def test_parse_wikitext_does_not_glue_friendship_xp_to_next_row() -> None:
    """Repro for the eval failure on Kuromi friendship level 15 rewards.

    Without the cell-break fix, level-14's XP value (1512) runs straight
    into level-15's row number (15), producing "...15121515..." in the
    chunk. The LLM then mis-assigns level-15 rewards.
    """
    wikitext = (
        "{| class=\"wikitable\"\n"
        "|-\n"
        "|14\n"
        "|Recipe unlock\n"
        "|1512\n"
        "|-\n"
        "|15\n"
        "|Cauldron blueprints unlocked\n"
        "|1967\n"
        "|}"
    )
    out = parse_wikitext(wikitext)
    assert "151215" not in out
    assert "Cauldron blueprints unlocked" in out


def test_parse_wikitext_drops_file_thumbnail_debris() -> None:
    """[[File:Foo.png|45x45px]] markup must not leak its size/alt args into chunks.

    Friendship-reward tables are full of icon thumbnails; without
    filtering them, every chunk gets sprayed with "45x45px"-style debris
    that crowds out the actual game data.
    """
    out = parse_wikitext("[[File:Foo.png|45x45px]] Apple")
    assert "45x45px" not in out
    assert "Foo.png" not in out
    assert "Apple" in out


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


def test_extract_sections_separates_table_cells_within_section() -> None:
    """Cell-break fix must reach extract_sections too, not just parse_wikitext.

    Both flow into the chunker; if only parse_wikitext got fixed,
    section-strategy chunks would still glue cells together and the
    friendship-rewards-table bug would resurface for that strategy.
    """
    wikitext = "==Recipe==\n{| class=\"wikitable\"\n| Apple || Wood\n|}"
    sections = extract_sections(wikitext)

    by_heading = {s["heading"]: s["content"] for s in sections}
    assert "Recipe" in by_heading
    content = by_heading["Recipe"]
    assert "AppleWood" not in content
    assert "Apple" in content
    assert "Wood" in content
