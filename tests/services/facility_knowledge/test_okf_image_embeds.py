"""Tests for the embed scan: which images a concept body embeds, and where.

The scan reports what the panel's Markdown renderer would draw as an image, with
the embed's character span in the body and the resolver's outcome for its target.
"""

from __future__ import annotations

import pytest

from osprey.services.facility_knowledge.okf.assets import (
    EMBED_OUTCOMES,
    ImageEmbed,
    scan_image_embeds,
)


def _only(body: str, source_dir: str = "devices") -> ImageEmbed:
    embeds = scan_image_embeds(body, source_dir)
    assert len(embeds) == 1, embeds
    return embeds[0]


def _spanned(body: str, embed: ImageEmbed) -> str:
    start, end = embed.span
    return body[start:end]


class TestSyntaxes:
    def test_inline_image(self) -> None:
        body = "Intro.\n\n![Beam position monitor](figures/bpm.png)\n"
        embed = _only(body)
        assert embed.target == "figures/bpm.png"
        assert embed.alt == "Beam position monitor"
        assert embed.outcome == "bundle"
        assert embed.path == "devices/figures/bpm.png"
        assert _spanned(body, embed) == "![Beam position monitor](figures/bpm.png)"

    def test_inline_image_with_title_and_angle_brackets(self) -> None:
        body = '![a](<figures/my plot.png> "The plot")'
        embed = _only(body)
        assert embed.target == "figures/my plot.png"
        assert embed.path == "devices/figures/my plot.png"
        assert _spanned(body, embed) == body

    def test_inline_image_with_single_quoted_title(self) -> None:
        embed = _only("![a](x.png 'title')")
        assert embed.target == "x.png"

    def test_full_reference_image(self) -> None:
        body = 'See ![The BPM][bpm] here.\n\n[bpm]: figures/bpm.png "BPM"\n'
        embed = _only(body)
        assert embed.target == "figures/bpm.png"
        assert embed.alt == "The BPM"
        assert embed.path == "devices/figures/bpm.png"
        assert _spanned(body, embed) == "![The BPM][bpm]"

    def test_reference_label_matches_case_and_whitespace_insensitively(self) -> None:
        body = "![x][My  Label]\n\n[my label]: a.png\n"
        assert _only(body).target == "a.png"

    def test_collapsed_reference_image(self) -> None:
        body = "![bpm][]\n\n[BPM]: figures/bpm.png\n"
        embed = _only(body)
        assert embed.target == "figures/bpm.png"
        assert _spanned(body, embed) == "![bpm][]"

    def test_shortcut_reference_image(self) -> None:
        body = "![bpm] after\n\n[bpm]: <figures/b p.png>\n"
        embed = _only(body)
        assert embed.target == "figures/b p.png"
        assert _spanned(body, embed) == "![bpm]"

    def test_first_definition_wins(self) -> None:
        body = "![x][r]\n\n[r]: first.png\n[r]: second.png\n"
        assert _only(body).target == "first.png"

    def test_raw_html_img(self) -> None:
        body = 'Text <IMG alt="The BPM" SRC="figures/bpm.png" width=200> more'
        embed = _only(body)
        assert embed.target == "figures/bpm.png"
        assert embed.alt == "The BPM"
        assert embed.path == "devices/figures/bpm.png"
        assert _spanned(body, embed) == '<IMG alt="The BPM" SRC="figures/bpm.png" width=200>'

    def test_raw_html_img_unquoted_and_self_closing(self) -> None:
        body = "<img src=a.png />"
        embed = _only(body)
        assert embed.target == "a.png"
        assert _spanned(body, embed) == body

    def test_raw_html_img_decodes_entities(self) -> None:
        assert _only('<img src="a&amp;b.png" alt="x &lt; y">').target == "a&b.png"
        assert _only('<img src="a.png" alt="x &lt; y">').alt == "x < y"

    def test_raw_html_img_without_src_is_not_reported(self) -> None:
        assert scan_image_embeds('<img alt="nothing">', "") == []

    def test_not_an_img_tag(self) -> None:
        assert scan_image_embeds('<imgx src="a.png">', "") == []

    def test_root_source_dir(self) -> None:
        assert _only("![a](figures/a.png)", "").path == "figures/a.png"

    def test_root_relative_target(self) -> None:
        assert _only("![a](/figures/a.png)").path == "figures/a.png"


class TestAltText:
    def test_empty_inline_alt(self) -> None:
        embed = _only("![](figures/bpm.png)")
        assert embed.alt == ""
        assert embed.outcome == "bundle"

    def test_whitespace_alt_is_empty(self) -> None:
        assert _only("![  ](a.png)").alt == ""

    def test_html_img_without_alt(self) -> None:
        assert _only('<img src="a.png">').alt == ""

    def test_escaped_alt(self) -> None:
        assert _only(r"![a \] b](x.png)").alt == "a ] b"


class TestMissingReference:
    def test_full_reference_without_definition_is_not_an_image(self) -> None:
        assert scan_image_embeds("![alt][nowhere]\n", "") == []

    def test_shortcut_without_definition_is_not_an_image(self) -> None:
        assert scan_image_embeds("![alt] text\n", "") == []

    def test_collapsed_without_definition_is_not_an_image(self) -> None:
        assert scan_image_embeds("![alt][]\n", "") == []


class TestNotRendered:
    def test_backtick_fence(self) -> None:
        body = "Before\n\n```markdown\n![a](a.png)\n<img src=b.png>\n```\n\n![c](c.png)\n"
        assert [e.target for e in scan_image_embeds(body, "")] == ["c.png"]

    def test_tilde_fence_needs_a_long_enough_closer(self) -> None:
        body = "~~~~\n![a](a.png)\n~~~\n![b](b.png)\n~~~~\n![c](c.png)\n"
        assert [e.target for e in scan_image_embeds(body, "")] == ["c.png"]

    def test_unclosed_fence_runs_to_the_end(self) -> None:
        assert scan_image_embeds("```\n![a](a.png)\n", "") == []

    def test_fence_interrupts_a_paragraph(self) -> None:
        body = "Text\n```\n![a](a.png)\n```\n"
        assert scan_image_embeds(body, "") == []

    def test_indented_code_block(self) -> None:
        body = "Text\n\n    ![a](a.png)\n\t<img src=b.png>\n\n![c](c.png)\n"
        assert [e.target for e in scan_image_embeds(body, "")] == ["c.png"]

    def test_indented_line_continuing_a_paragraph_is_rendered(self) -> None:
        body = "Text\n    ![a](a.png)\n"
        assert [e.target for e in scan_image_embeds(body, "")] == ["a.png"]

    def test_list_item_continuation_is_rendered(self) -> None:
        body = "- item\n\n  ![a](a.png)\n"
        assert [e.target for e in scan_image_embeds(body, "")] == ["a.png"]

    def test_inline_code_span(self) -> None:
        body = "Use `![a](a.png)` or ``<img src=b.png>`` but ![c](c.png)"
        assert [e.target for e in scan_image_embeds(body, "")] == ["c.png"]

    def test_backslash_escaped(self) -> None:
        assert scan_image_embeds(r"\![a](a.png)", "") == []

    def test_escaped_backslash_leaves_the_image(self) -> None:
        body = r"\\![a](a.png)"
        assert [e.target for e in scan_image_embeds(body, "")] == ["a.png"]

    def test_inline_html_comment(self) -> None:
        body = "Text <!-- ![a](a.png) <img src=b.png> --> ![c](c.png)"
        assert [e.target for e in scan_image_embeds(body, "")] == ["c.png"]

    def test_block_html_comment_across_blank_lines(self) -> None:
        body = "<!--\n![a](a.png)\n\n<img src=b.png>\n-->\n![c](c.png)\n"
        assert [e.target for e in scan_image_embeds(body, "")] == ["c.png"]

    def test_reference_definition_inside_code_is_not_a_definition(self) -> None:
        body = "![x][r]\n\n```\n[r]: a.png\n```\n"
        assert scan_image_embeds(body, "") == []

    def test_image_does_not_cross_a_blank_line(self) -> None:
        assert scan_image_embeds("![a\n\nb](a.png)", "") == []


class TestOutcomes:
    @pytest.mark.parametrize(
        "target",
        [
            "https://example.org/a.png",
            "data:image/png;base64,AAAA",
            "//cdn.example.org/a.png",
        ],
    )
    def test_external_targets(self, target: str) -> None:
        for body in (f"![x]({target})", f'<img src="{target}">', f"![x][r]\n\n[r]: {target}\n"):
            embed = _only(body)
            assert embed.target == target
            assert embed.outcome == "external"
            assert embed.path is None

    def test_target_leaving_the_root_is_outside_bundle(self) -> None:
        embed = _only("![x](../../secret.png)")
        assert embed.outcome == "outside_bundle"
        assert embed.path is None
        assert embed.target == "../../secret.png"

    def test_target_inside_root_after_parent_step(self) -> None:
        embed = _only("![x](../figures/a.png)")
        assert embed.outcome == "bundle"
        assert embed.path == "figures/a.png"

    def test_outcomes_are_the_documented_set(self) -> None:
        assert frozenset({"bundle", "external", "outside_bundle"}) == EMBED_OUTCOMES


class TestLinkedImage:
    def test_only_the_inner_image_is_reported(self) -> None:
        body = "[![alt](a.png)](page.md)"
        embed = _only(body, "")
        assert embed.target == "a.png"
        assert embed.alt == "alt"
        assert _spanned(body, embed) == "![alt](a.png)"
        start, end = embed.span
        assert body[:start] + body[end:] == "[](page.md)"

    def test_ordinary_link_is_not_an_image(self) -> None:
        assert scan_image_embeds("[text](a.png)", "") == []


class TestOrder:
    def test_results_follow_source_order(self) -> None:
        body = (
            '<img src="one.png">\n'
            "![two][r] and ![three](three.png)\n\n"
            "[![four](four.png)](x.md)\n\n"
            "[r]: two.png\n"
        )
        embeds = scan_image_embeds(body, "")
        assert [e.target for e in embeds] == ["one.png", "two.png", "three.png", "four.png"]
        starts = [e.span[0] for e in embeds]
        assert starts == sorted(starts)
        for embed in embeds:
            assert embed.target in _spanned(body, embed) or embed.target == "two.png"

    def test_empty_body(self) -> None:
        assert scan_image_embeds("", "") == []

    def test_spans_use_offsets_into_crlf_body(self) -> None:
        body = "Line\r\n\r\n![a](a.png)\r\n"
        embed = _only(body, "")
        assert _spanned(body, embed) == "![a](a.png)"
