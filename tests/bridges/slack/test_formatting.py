"""Tests for the Slack mrkdwn formatter (:func:`markdown_to_mrkdwn`)."""

from __future__ import annotations

import pytest

from osprey.bridges.slack.formatting import (
    MENTION_PLACEHOLDER_RE,
    markdown_to_mrkdwn,
)

# (name, markdown in, mrkdwn out)
GOLDEN: list[tuple[str, str, str]] = [
    ("plain", "no formatting here", "no formatting here"),
    ("empty", "", ""),
    ("bold-star", "a **b** c", "a *b* c"),
    ("bold-underscore", "a __b__ c", "a *b* c"),
    ("italic-underscore", "a _i_ c", "a _i_ c"),
    ("bold-italic-star", "***x***", "*_x_*"),
    ("bold-italic-underscore", "___x___", "*_x_*"),
    ("strike", "a ~~s~~ c", "a ~s~ c"),
    ("link", "see [docs](https://ex.com/a_b)", "see <https://ex.com/a_b|docs>"),
    ("link-label-is-url", "[https://ex.com](https://ex.com)", "<https://ex.com>"),
    ("link-empty-label", "[](https://ex.com)", "<https://ex.com>"),
    ("mailto-link", "[mail](mailto:a@b.org)", "<mailto:a@b.org|mail>"),
    ("link-label-escaped", "[a < b & c](https://ex.com)", "<https://ex.com|a &lt; b &amp; c>"),
    ("link-url-ampersand", "[q](https://ex.com/?a=1&b=2)", "<https://ex.com/?a=1&amp;b=2|q>"),
    ("relative-link-left-raw", "[t](docs/x.md)", "[t](docs/x.md)"),
    ("heading", "# Title", "*Title*"),
    ("heading-h3-closing-hashes", "### Sub ###", "*Sub*"),
    ("heading-with-bold", "## **Bold** title", "*Bold title*"),
    ("bullet-star", "* item", "- item"),
    ("bullet-plus", "+ item", "- item"),
    (
        "table",
        "| Name | Value |\n|---|---|\n| a | 1 |\n| b | 2 |",
        "*Name*: a, *Value*: 1\n*Name*: b, *Value*: 2",
    ),
    ("math-inline", "energy $E=mc^2$ here", "energy E=mc^2 here"),
    ("math-display", "$$x+y$$", "x+y"),
    ("math-bracket", "\\[a\\] and \\(b\\)", "a and b"),
    ("currency-untouched", "costs $5 and $10", "costs $5 and $10"),
    ("escape-amp", "R&D", "R&amp;D"),
    ("escape-angles", "a < b > c", "a &lt; b &gt; c"),
    ("existing-entity-kept", "R&amp;D &lt;x&gt;", "R&amp;D &lt;x&gt;"),
    ("broadcast-escaped", "hey <!channel>", "hey &lt;!channel&gt;"),
    ("html-tag-escaped", "<b>x</b>", "&lt;b&gt;x&lt;/b&gt;"),
    ("mention-kept", "thanks <@U1>", "thanks <@U1>"),
    ("mention-w-kept", "cc <@W0ABC>!", "cc <@W0ABC>!"),
    ("autolink", "<https://ex.com>", "<https://ex.com>"),
    ("inline-code", "run `a**b** <x> & _y_`", "run `a**b** <x> & _y_`"),
    ("fence-lang-dropped", "```python\nx = a < b & **c**\n```", "```\nx = a < b & **c**\n```"),
    ("snake-case", "use snake_case_name", "use snake_case_name"),
    ("arithmetic-stars", "2**3 and 4**2", "2**3 and 4**2"),
    ("blockquote", "> quoted", "&gt; quoted"),
]


@pytest.mark.parametrize("_name,md,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_every_golden_case_converts_to_its_mrkdwn_target(_name, md, expected):
    assert markdown_to_mrkdwn(md) == expected


@pytest.mark.parametrize("_name,md,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_a_second_conversion_changes_nothing(_name, md, expected):
    once = markdown_to_mrkdwn(md)
    assert markdown_to_mrkdwn(once) == once
    assert markdown_to_mrkdwn(expected) == expected


def test_a_single_star_span_is_kept_as_slack_bold_so_the_transform_stays_idempotent():
    # Markdown ``*x*`` and mrkdwn bold ``*x*`` are the same bytes; rewriting it to
    # ``_x_`` would turn every produced bold into italic on a second pass.
    assert markdown_to_mrkdwn("an *emph* word") == "an *emph* word"


def test_code_content_is_never_escaped_or_rewritten():
    md = "```\n<!channel> & **x** [t](https://u) <@U1>\n```\nand `~~s~~ $x$`"
    assert markdown_to_mrkdwn(md) == md


def test_an_unterminated_fence_falls_back_to_converted_raw_text():
    out = markdown_to_mrkdwn("```\na **b** & c")
    assert out == "```\na *b* &amp; c"
    assert markdown_to_mrkdwn(out) == out


def test_a_ragged_table_is_emitted_verbatim_apart_from_escaping():
    md = "| a | b |\n|---|---|\n| 1 |"
    assert markdown_to_mrkdwn(md) == md


def test_a_mention_inside_bold_survives_unchanged():
    assert markdown_to_mrkdwn("**ask <@U1> now**") == "*ask <@U1> now*"


def test_mention_placeholders_survive_beside_escaped_text():
    out = markdown_to_mrkdwn("<@U1> & <@U2> <!here>")
    assert out == "<@U1> &amp; <@U2> &lt;!here&gt;"
    assert MENTION_PLACEHOLDER_RE.findall(out) == ["U1", "U2"]


def test_a_non_url_angle_form_is_escaped_rather_than_kept_as_a_link():
    assert markdown_to_mrkdwn("<#C123|general>") == "&lt;#C123|general&gt;"


def test_the_local_mention_pattern_matches_the_core_one():
    from osprey.bridges.core import MENTION_PLACEHOLDER_RE as core_re

    assert MENTION_PLACEHOLDER_RE.pattern == core_re.pattern


@pytest.mark.parametrize(
    "junk",
    ["\x00CODE0\x00", "\x00BOLD9\x00", "```", "|", "|---|", "[", "](", "<", "&", "***", "$$"],
)
def test_malformed_or_hostile_input_never_raises(junk):
    out = markdown_to_mrkdwn(junk)
    assert isinstance(out, str)
    assert markdown_to_mrkdwn(out) == out


def test_the_input_is_not_mutated_and_the_function_is_pure():
    md = "a **b**"
    assert markdown_to_mrkdwn(md) == markdown_to_mrkdwn(md)
    assert md == "a **b**"


@pytest.mark.parametrize(
    "md",
    [
        "\n~~~~~~~>**!:[|",
        "&\\(:*~:**<@U1>|***-&amp;**",
        "\\(b```b&:\\(\\) \\)",
        "$$<@U1>!$&amp;$",
        "*\\( $(:\\)!",
        "| a | b |\n|---|---|\n$!|$:#$$",
        "# ~~[~~",
    ],
)
def test_marker_runs_and_chained_math_delimiters_settle_in_one_pass(md):
    once = markdown_to_mrkdwn(md)
    assert markdown_to_mrkdwn(once) == once
