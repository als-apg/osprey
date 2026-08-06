"""Golden / code-span / idempotence tests for ``markdown_to_chat``.

Each converter contributes its cases to ``GOLDEN`` (a list of
``(name, markdown_in, chat_out, fixpoint)`` tuples). Properties asserted over the
whole corpus for free:

  * every ``markdown_to_chat(markdown_in) == chat_out`` (golden correctness);
  * for ``fixpoint`` cases: ``markdown_to_chat(chat_out) == chat_out`` and
    ``markdown_to_chat(markdown_to_chat(x)) == markdown_to_chat(x)`` (idempotence).

``fixpoint`` is ``False`` only for the bold family (bold, heading-as-bold,
bold+italic). Chat overloads single ``*`` for bold, so a produced Chat bold ``*x*``
is byte-identical to a Markdown italic ``*x*`` — re-feeding it re-reads it as italic.
This is an inherent limit of Chat's syntax, not a transform defect; the transform is
applied exactly once in the real delivery path. Encoding it here keeps the
idempotence guarantee honest rather than asserting an impossible property.

Pure Python, no network — the module under test imports nothing but ``re``.
"""

import pytest

from osprey.bridges.google_chat.formatting import markdown_to_chat

# (name, markdown_in, expected_chat_out, fixpoint). Grows as converters land.
GOLDEN: list[tuple[str, str, str, bool]] = [
    ("plain", "just some plain text", "just some plain text", True),
    ("empty", "", "", True),
    ("inline_code_verbatim", "use `**kwargs` here", "use `**kwargs` here", True),
    (
        "fence_lang_stripped",
        "```python\nprint('**hi**')\n```",
        "```\nprint('**hi**')\n```",
        True,
    ),
    ("fence_no_lang", "```\nplain\n```", "```\nplain\n```", True),
    # headings + bullets
    ("heading_h3", "### Heading", "*Heading*", False),
    ("heading_h1", "# Title", "*Title*", False),
    ("bullet_star", "* first\n* second", "- first\n- second", True),
    ("bullet_plus", "+ item", "- item", True),
    ("bullet_dash_kept", "- already", "- already", True),
    (
        "heading_in_fence_literal",
        "```\n#### not a heading\n```",
        "```\n#### not a heading\n```",
        True,
    ),
    # tables — bold headers make these non-fixpoint
    (
        "table_clean_2col",
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |",
        "*Name:* Alice, *Age:* 30",
        False,
    ),
    (
        "table_embedded_pipe_raw_fallback",
        "| a | b |\n| --- | --- |\n| x \\| y | z |",
        "| a | b |\n| --- | --- |\n| x \\| y | z |",
        True,
    ),
    (
        "table_in_fence_untouched",
        "```\n| a | b |\n| c | d |\n```",
        "```\n| a | b |\n| c | d |\n```",
        True,
    ),
    # math
    ("math_inline", "the value $\\epsilon_x$ is small", "the value \\epsilon_x is small", True),
    ("math_display", "$$E = mc^2$$", "E = mc^2", True),
    ("math_paren", "\\(a + b\\)", "a + b", True),
    ("math_bracket", "\\[x^2\\]", "x^2", True),
    ("math_currency_untouched", "it costs $5 and $10", "it costs $5 and $10", True),
    ("math_currency_in_code", "`$5` stays", "`$5` stays", True),
    # emphasis
    ("bold_star", "**bold**", "*bold*", False),
    ("bold_underscore", "__bold__", "*bold*", False),
    ("italic_star", "*italic*", "_italic_", True),
    ("bold_and_italic_mixed", "**b** and *i*", "*b* and _i_", False),
    ("bold_italic_triple", "***x***", "*_x_*", False),
    ("literal_double_star_kept", "2**3 = 8", "2**3 = 8", True),
    ("snake_case_untouched", "the run_id value", "the run_id value", True),
    ("underscore_italic_passthrough", "_already_", "_already_", True),
    ("url_underscore_untouched", "see http://x/a_b_c page", "see http://x/a_b_c page", True),
    ("stars_in_code_untouched", "`a ** b` and `*c*`", "`a ** b` and `*c*`", True),
    # links — target locked to `text: url` (bare url when text == url)
    ("link_inline", "see [docs](http://x) now", "see docs: http://x now", True),
    ("link_text_equals_url", "[http://x](http://x)", "http://x", True),
    ("link_reference", "see [docs][d]\n[d]: http://x", "see docs: http://x", True),
    ("link_bare_url_untouched", "visit http://x now", "visit http://x now", True),
    ("link_image_untouched", "![alt](http://x)", "![alt](http://x)", True),
    ("link_in_code_untouched", "`[a](b)` literal", "`[a](b)` literal", True),
]


@pytest.mark.parametrize("name,md,expected,_fp", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_every_golden_case_converts_to_its_chat_target(name, md, expected, _fp):
    assert markdown_to_chat(md) == expected


@pytest.mark.parametrize(
    "name,md,expected",
    [(g[0], g[1], g[2]) for g in GOLDEN if g[3]],
    ids=[g[0] for g in GOLDEN if g[3]],
)
def test_a_fixpoint_case_is_unchanged_by_a_second_conversion(name, md, expected):
    once = markdown_to_chat(md)
    assert markdown_to_chat(once) == once
    # The Chat target form is itself a fixpoint.
    assert markdown_to_chat(expected) == expected


def test_the_bold_family_is_correct_on_the_one_application_the_pipeline_uses():
    """Bold-family cases convert correctly on the one application the pipeline uses.

    Documents the known single-``*`` re-read: a second application is NOT asserted
    equal (Chat bold ``*x*`` == Markdown italic ``*x*``).
    """
    for name, md, expected, fp in GOLDEN:
        if not fp:
            assert markdown_to_chat(md) == expected, name


# --- code masking / restoration -------------------------------------------------


def test_text_with_no_code_at_all_round_trips_unchanged():
    assert markdown_to_chat("no code at all") == "no code at all"


def test_markdown_significant_characters_inside_inline_code_are_preserved():
    md = "call `func(**a, _b_)` and `x | y` and `# not a heading`"
    # Every markdown-significant char inside inline code survives verbatim.
    assert markdown_to_chat(md) == md


def test_a_fenced_block_drops_its_language_tag_and_keeps_its_body_verbatim():
    md = "```bash\n# a comment\necho **not bold**\n```"
    out = markdown_to_chat(md)
    assert out == "```\n# a comment\necho **not bold**\n```"
    assert "bash" not in out.split("\n", 1)[0]  # opening fence carries no lang


def test_pipes_and_hashes_inside_a_fenced_block_survive():
    md = "```\n| a | b |\n#### heading-looking\n```"
    assert markdown_to_chat(md) == md


def test_an_unterminated_fence_is_left_verbatim():
    # No closing ``` -> not a fence match -> emitted unchanged (malformed -> raw).
    md = "```python\nprint('hi')\nno closing fence"
    assert markdown_to_chat(md) == md


def test_multiple_code_spans_are_restored_in_order():
    md = "first `one` then `two` then ```\nthree\n``` done"
    out = markdown_to_chat(md)
    assert "`one`" in out and "`two`" in out and "```\nthree\n```" in out


def test_empty_input_returns_empty():
    assert markdown_to_chat("") == ""


# --- headings + bullets ---------------------------------------------------------


def test_a_heading_at_any_level_loses_all_of_its_hashes():
    for n in range(1, 7):
        assert markdown_to_chat("#" * n + " Title") == "*Title*"


def test_a_headings_closing_hashes_are_dropped():
    assert markdown_to_chat("## Title ##") == "*Title*"


def test_bold_at_the_start_of_a_line_is_not_a_bullet():
    # ``**bold**`` (no space after the marker) must not be read as a list item.
    out = markdown_to_chat("**bold** line")
    assert not out.startswith("- ")


def test_an_indented_bullet_marker_is_normalized_and_keeps_its_indent():
    assert markdown_to_chat("  * nested") == "  - nested"


def test_a_hash_without_a_following_space_is_not_a_heading():
    # ``#tag`` (no space) is not an ATX heading.
    assert markdown_to_chat("#tag stays") == "#tag stays"


# --- tables ---------------------------------------------------------------------


def test_each_table_row_becomes_its_own_labeled_line():
    md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"
    assert markdown_to_chat(md) == "*Name:* Alice, *Age:* 30\n*Name:* Bob, *Age:* 25"


def test_a_table_with_a_ragged_row_falls_back_to_the_raw_block():
    md = "| a | b |\n| --- | --- |\n| onlyone |"
    assert markdown_to_chat(md) == md  # column count mismatch -> raw block


def test_the_text_surrounding_a_table_is_preserved():
    md = "before\n\n| H | V |\n| --- | --- |\n| k | 1 |\n\nafter"
    out = markdown_to_chat(md)
    assert out.startswith("before\n\n")
    assert out.endswith("\n\nafter")
    assert "*H:* k" in out
    assert "|" not in out.split("after")[0].split("before")[1]  # no pipes left in the table region


def test_pipes_without_a_separator_row_are_not_a_table():
    md = "| just | pipes |\n| no | separator |"
    assert markdown_to_chat(md) == md


# --- math -----------------------------------------------------------------------


def test_a_lone_dollar_sign_is_never_stripped():
    assert markdown_to_chat("price is $5") == "price is $5"
    assert markdown_to_chat("a $ b") == "a $ b"


def test_a_dollar_sign_inside_inline_code_is_untouched():
    assert markdown_to_chat("use `$HOME` here") == "use `$HOME` here"


def test_display_math_is_matched_before_inline_math():
    # ``$$…$$`` is matched as a display block, not two empty inline spans.
    assert markdown_to_chat("$$a$$") == "a"


def test_multiple_inline_math_spans_are_each_stripped():
    assert markdown_to_chat("$x$ and $y$") == "x and y"


# --- emphasis -------------------------------------------------------------------


def test_the_stars_that_bold_conversion_produces_never_become_italic():
    # The single stars produced by bold conversion must NOT become italic.
    assert markdown_to_chat("**word**") == "*word*"


def test_italic_never_eats_bold_output_in_the_same_pass():
    assert markdown_to_chat("a **b** c *d* e") == "a *b* c _d_ e"


def test_bold_containing_inner_italic_degrades_rather_than_corrupts():
    # Inner emphasis inside bold is protected (Chat can't nest) — no corruption.
    out = markdown_to_chat("**a *b* c**")
    assert out == "*a *b* c*"


def test_a_heading_and_an_inline_bold_are_both_protected_in_one_call():
    # Heading-bold and inline-bold both survive the italic pass in one call.
    assert markdown_to_chat("# Title\n\n**strong** and *soft*") == "*Title*\n\n*strong* and _soft_"


def test_arithmetic_double_stars_are_untouched():
    assert markdown_to_chat("compute 2**3 and 4**2") == "compute 2**3 and 4**2"


def test_stars_with_spaces_around_them_are_not_italic():
    assert markdown_to_chat("a * b * c") == "a * b * c"


# --- links ----------------------------------------------------------------------


def test_an_unresolved_reference_link_is_left_as_is():
    # A reference use with no matching definition is not rewritten.
    assert markdown_to_chat("[docs][missing]") == "[docs][missing]"


def test_a_collapsed_reference_link_uses_its_label_as_the_reference():
    assert markdown_to_chat("[Osprey][]\n[osprey]: http://x") == "Osprey: http://x"


def test_a_reference_definition_line_is_dropped():
    out = markdown_to_chat("body text\n[d]: http://x")
    assert "http://x" not in out  # unused definition line removed
    assert out == "body text"


def test_a_bare_url_is_never_bracketed():
    # Chat auto-links bare URLs; wrapping would defeat it.
    assert markdown_to_chat("http://x/y") == "http://x/y"
