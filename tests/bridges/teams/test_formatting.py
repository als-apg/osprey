"""Golden / code-span / idempotence tests for ``markdown_to_teams``.

Each converter contributes its cases to ``GOLDEN`` (a list of
``(name, markdown_in, teams_out)`` tuples). Properties asserted over the whole
corpus for free:

  * every ``markdown_to_teams(markdown_in) == teams_out`` (golden correctness);
  * ``markdown_to_teams(teams_out) == teams_out`` and
    ``markdown_to_teams(markdown_to_teams(x)) == markdown_to_teams(x)`` (idempotence).

Unlike the Google Chat transform there is no non-fixpoint family: Teams bold is
``**x**``, which no pass can re-read as an italic ``*x*``, so every case above is a
fixpoint and the property is asserted over the whole corpus with no exemptions.

Pure Python, no network — the module under test imports nothing but ``re``.
"""

import pytest

from osprey.bridges.core import MENTION_PLACEHOLDER_RE
from osprey.bridges.teams.formatting import markdown_to_teams, render_mentions

# (name, markdown_in, expected_teams_out). Grows as converters land.
GOLDEN: list[tuple[str, str, str]] = [
    ("plain", "just some plain text", "just some plain text"),
    ("empty", "", ""),
    # code — masked first, restored verbatim
    ("inline_code_verbatim", "use `**kwargs` here", "use `**kwargs` here"),
    (
        "fence_lang_kept",
        "```python\nprint('**hi**')\n```",
        "```python\nprint('**hi**')\n```",
    ),
    ("fence_no_lang", "```\nplain\n```", "```\nplain\n```"),
    (
        "heading_in_fence_literal",
        "```\n#### not a heading\n```",
        "```\n#### not a heading\n```",
    ),
    (
        "table_in_fence_untouched",
        "```\n| a | b |\n| --- | --- |\n| c | d |\n```",
        "```\n| a | b |\n| --- | --- |\n| c | d |\n```",
    ),
    # headings -> a bold line
    ("heading_h1", "# Title", "**Title**"),
    ("heading_h3", "### Heading", "**Heading**"),
    ("heading_closing_hashes", "## Title ##", "**Title**"),
    ("hash_without_space_is_not_a_heading", "#tag stays", "#tag stays"),
    # bullets and blockquotes pass through — Teams renders all of these itself
    ("bullet_dash", "- first\n- second", "- first\n- second"),
    ("bullet_star", "* first\n* second", "* first\n* second"),
    ("bullet_plus", "+ item", "+ item"),
    ("bullet_nested_indent", "  - nested", "  - nested"),
    ("blockquote", "> quoted line", "> quoted line"),
    ("blockquote_nested", ">> deeper", ">> deeper"),
    # tables -> labeled lines, one per data row
    (
        "table_clean_2col",
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |",
        "**Name**: Alice, **Age**: 30",
    ),
    (
        "table_two_data_rows",
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        "**Name**: Alice, **Age**: 30\n**Name**: Bob, **Age**: 25",
    ),
    (
        "table_empty_cell_keeps_its_label",
        "| Name | Age |\n| --- | --- |\n| Alice |  |",
        "**Name**: Alice, **Age**:",
    ),
    (
        "table_embedded_pipe_raw_fallback",
        "| a | b |\n| --- | --- |\n| x \\| y | z |",
        "| a | b |\n| --- | --- |\n| x \\| y | z |",
    ),
    (
        "table_ragged_row_raw_fallback",
        "| a | b |\n| --- | --- |\n| onlyone |",
        "| a | b |\n| --- | --- |\n| onlyone |",
    ),
    (
        "pipes_without_a_separator_are_not_a_table",
        "| just | pipes |\n| no | separator |",
        "| just | pipes |\n| no | separator |",
    ),
    # math — the one inline construct Teams cannot render
    ("math_inline", "the value $\\epsilon_x$ is small", "the value \\epsilon_x is small"),
    ("math_display", "$$E = mc^2$$", "E = mc^2"),
    ("math_paren", "\\(a + b\\)", "a + b"),
    ("math_bracket", "\\[x^2\\]", "x^2"),
    ("math_currency_untouched", "it costs $5 and $10", "it costs $5 and $10"),
    ("math_currency_in_code", "`$5` stays", "`$5` stays"),
    # emphasis — Teams reads Markdown emphasis directly, so it is left alone
    ("bold_star", "**bold**", "**bold**"),
    ("bold_underscore", "__bold__", "__bold__"),
    ("italic_star", "*italic*", "*italic*"),
    ("italic_underscore", "_italic_", "_italic_"),
    ("bold_italic_triple", "***x***", "***x***"),
    ("bold_and_italic_mixed", "**b** and *i*", "**b** and *i*"),
    ("literal_double_star_kept", "2**3 = 8", "2**3 = 8"),
    ("snake_case_untouched", "the run_id value", "the run_id value"),
    ("stars_in_code_untouched", "`a ** b` and `*c*`", "`a ** b` and `*c*`"),
    # links — Teams renders Markdown links, so every spelling passes through
    ("link_inline", "see [docs](http://x) now", "see [docs](http://x) now"),
    ("link_reference", "see [docs][d]\n[d]: http://x", "see [docs][d]\n[d]: http://x"),
    ("link_bare_url_untouched", "visit http://x now", "visit http://x now"),
    ("link_url_underscore_untouched", "see http://x/a_b_c page", "see http://x/a_b_c page"),
    ("link_image_untouched", "![alt](http://x)", "![alt](http://x)"),
    ("link_in_code_untouched", "`[a](b)` literal", "`[a](b)` literal"),
]


@pytest.mark.parametrize("_name,md,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_every_golden_case_converts_to_its_teams_target(_name, md, expected):
    assert markdown_to_teams(md) == expected


@pytest.mark.parametrize("_name,md,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_every_golden_case_is_a_fixpoint(_name, md, expected):
    once = markdown_to_teams(md)
    assert markdown_to_teams(once) == once
    # The Teams target form is itself a fixpoint.
    assert markdown_to_teams(expected) == expected


# --- the four constructs the Teams target table is written for -------------------


def test_a_two_column_table_becomes_one_labeled_line_per_row():
    md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"
    assert markdown_to_teams(md) == "**Name**: Alice, **Age**: 30\n**Name**: Bob, **Age**: 25"
    assert "|" not in markdown_to_teams(md)  # no pipe syntax reaches Teams


def test_a_heading_at_any_level_becomes_a_bold_line():
    for n in range(1, 7):
        assert markdown_to_teams("#" * n + " Title") == "**Title**"


def test_a_fenced_block_survives_untouched_including_its_language_tag():
    md = "```bash\n# a comment\necho **not bold**\n| a | b |\n```"
    assert markdown_to_teams(md) == md


def test_inline_code_and_links_are_preserved_verbatim():
    md = "call `func(**a, _b_)` then read [the docs](http://x/a_b_c#frag)"
    assert markdown_to_teams(md) == md


# --- code masking / restoration --------------------------------------------------


def test_text_with_no_code_at_all_round_trips_unchanged():
    assert markdown_to_teams("no code at all") == "no code at all"


def test_markdown_significant_characters_inside_inline_code_are_preserved():
    md = "call `func(**a, _b_)` and `x | y` and `# not a heading` and `$HOME`"
    assert markdown_to_teams(md) == md


def test_an_unterminated_fence_is_left_verbatim():
    # No closing ``` -> not a fence match -> emitted unchanged (malformed -> raw).
    md = "```python\nprint('hi')\nno closing fence"
    assert markdown_to_teams(md) == md


def test_multiple_code_spans_are_restored_in_order():
    md = "first `one` then `two` then ```\nthree\n``` done"
    assert markdown_to_teams(md) == md


def test_empty_input_returns_empty():
    assert markdown_to_teams("") == ""


def test_a_heading_inside_a_fence_is_never_bolded():
    md = "```\n### still a comment\n```\n\n### but this one converts"
    assert markdown_to_teams(md) == "```\n### still a comment\n```\n\n**but this one converts**"


# --- tables ----------------------------------------------------------------------


def test_the_text_surrounding_a_table_is_preserved():
    md = "before\n\n| H | V |\n| --- | --- |\n| k | 1 |\n\nafter"
    out = markdown_to_teams(md)
    assert out == "before\n\n**H**: k, **V**: 1\n\nafter"


def test_a_table_cell_with_no_header_emits_the_value_alone():
    md = "| Name |  |\n| --- | --- |\n| Alice | extra |"
    assert markdown_to_teams(md) == "**Name**: Alice, extra"


def test_a_header_only_table_falls_back_to_the_raw_block():
    # A header and separator with no data rows has nothing to label.
    md = "| a | b |\n| --- | --- |"
    assert markdown_to_teams(md) == md


def test_inline_code_inside_a_table_cell_survives_the_row_rewrite():
    md = "| Key | Value |\n| --- | --- |\n| `x` | `a | b` |"
    # The masked code span carries the pipe, so the row still splits into two cells.
    assert markdown_to_teams(md) == "**Key**: `x`, **Value**: `a | b`"


# --- math ------------------------------------------------------------------------


def test_a_lone_dollar_sign_is_never_stripped():
    assert markdown_to_teams("price is $5") == "price is $5"
    assert markdown_to_teams("a $ b") == "a $ b"


def test_display_math_is_matched_before_inline_math():
    # ``$$…$$`` is matched as a display block, not two empty inline spans.
    assert markdown_to_teams("$$a$$") == "a"


def test_multiple_inline_math_spans_are_each_stripped():
    assert markdown_to_teams("$x$ and $y$") == "x and y"


# --- emphasis and links pass through ---------------------------------------------


def test_bold_is_left_in_its_markdown_spelling():
    # Teams renders ``**x**``; rewriting it to a single star would make it italic.
    assert markdown_to_teams("**word**") == "**word**"


def test_a_heading_and_an_inline_bold_are_both_bold_in_one_call():
    assert markdown_to_teams("# Title\n\n**strong** and *soft*") == (
        "**Title**\n\n**strong** and *soft*"
    )


def test_a_reference_definition_line_is_kept():
    # Teams resolves reference links itself, so the definition must survive.
    assert markdown_to_teams("body\n[d]: http://x") == "body\n[d]: http://x"


# --- render_mentions ----------------------------------------------------------

ROSTER = {"29:111": "Alice", "29:222": "Carol", "29:333": None}


def render(text, roster=ROSTER, enabled=True):
    return render_mentions(text, roster, enabled=enabled, placeholder=MENTION_PLACEHOLDER_RE)


def carol(label="Carol", ident="29:222"):
    return {
        "type": "mention",
        "text": f"<at>{label}</at>",
        "mentioned": {"id": ident, "name": label},
    }


def test_a_roster_mention_renders_as_an_at_tag_with_its_entity():
    assert render("Carol, see above: <@29:222>") == (
        "Carol, see above: <at>Carol</at>",
        [carol()],
    )


def test_the_entity_text_matches_the_tag_in_the_text_exactly():
    text, entities = render("<@29:111> and <@29:222>")
    assert [entity["text"] for entity in entities] == ["<at>Alice</at>", "<at>Carol</at>"]
    assert all(entity["text"] in text for entity in entities)
    assert text == "<at>Alice</at> and <at>Carol</at>"


def test_a_mention_outside_the_roster_is_plain_text():
    assert render("cc <@29:999>") == ("cc @29:999", [])


def test_a_member_without_a_name_is_plain_text():
    assert render("cc <@29:333>") == ("cc @29:333", [])


def test_mentions_off_renders_every_mention_plain():
    assert render("<@29:111> and <@29:222>", enabled=False) == ("@Alice and @Carol", [])


def test_a_name_is_made_safe_for_the_tag():
    text, entities = render("hi <@29:9>", {"29:9": "A<b>c\nd"})
    assert text == "hi <at>A b c d</at>"
    assert entities == [carol("A b c d", "29:9")]


def test_two_mentions_of_one_member_carry_one_entity_each():
    text, entities = render("<@29:222> then <@29:222>")
    assert text == "<at>Carol</at> then <at>Carol</at>"
    assert entities == [carol(), carol()]


def test_text_without_a_placeholder_is_unchanged_with_no_entities():
    text = "nothing to see: <at>Carol</at> @Carol 29:222"
    assert render(text) == (text, [])


@pytest.mark.parametrize(
    "markdown",
    [
        "tell <@29:1GcS4E_yB-oS> now",
        "**tell <@29:1GcS4E_yB-oS>**",
        "- item <@29:1GcS4E_yB-oS>",
        "# head <@29:1GcS4E_yB-oS>",
        "| v |\n|---|\n| <@29:1GcS4E_yB-oS> |",
    ],
)
def test_the_placeholder_survives_markdown_to_teams(markdown):
    assert "<@29:1GcS4E_yB-oS>" in markdown_to_teams(markdown)
