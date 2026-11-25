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

    @pytest.mark.parametrize(
        ("body", "target"),
        [
            pytest.param(
                "![x][My  Label]\n\n[my label]: a.png\n",
                "a.png",
                id="reference-label-matches-case-and-whitespace-insensitively",
            ),
            pytest.param(
                "![x][r]\n\n[r]: first.png\n[r]: second.png\n",
                "first.png",
                id="first-definition-wins",
            ),
        ],
    )
    def test_reference_definition_lookup(self, body: str, target: str) -> None:
        assert _only(body).target == target

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
    @pytest.mark.parametrize(
        ("body", "expected_targets"),
        [
            pytest.param(
                "Before\n\n```markdown\n![a](a.png)\n<img src=b.png>\n```\n\n![c](c.png)\n",
                ["c.png"],
                id="backtick-fence",
            ),
            pytest.param(
                "~~~~\n![a](a.png)\n~~~\n![b](b.png)\n~~~~\n![c](c.png)\n",
                ["c.png"],
                id="tilde-fence-needs-a-long-enough-closer",
            ),
            pytest.param("```\n![a](a.png)\n", [], id="unclosed-fence-runs-to-the-end"),
            pytest.param("Text\n```\n![a](a.png)\n```\n", [], id="fence-interrupts-a-paragraph"),
            pytest.param(
                "Text\n\n    ![a](a.png)\n\t<img src=b.png>\n\n![c](c.png)\n",
                ["c.png"],
                id="indented-code-block",
            ),
            pytest.param(
                "Text\n    ![a](a.png)\n",
                ["a.png"],
                id="indented-line-continuing-a-paragraph-is-rendered",
            ),
            pytest.param(
                "- item\n\n  ![a](a.png)\n", ["a.png"], id="list-item-continuation-is-rendered"
            ),
            pytest.param(
                "Use `![a](a.png)` or ``<img src=b.png>`` but ![c](c.png)",
                ["c.png"],
                id="inline-code-span",
            ),
            pytest.param(r"\![a](a.png)", [], id="backslash-escaped"),
            pytest.param(r"\\![a](a.png)", ["a.png"], id="escaped-backslash-leaves-the-image"),
            pytest.param(
                "Text <!-- ![a](a.png) <img src=b.png> --> ![c](c.png)",
                ["c.png"],
                id="inline-html-comment",
            ),
            pytest.param(
                "<!--\n![a](a.png)\n\n<img src=b.png>\n-->\n![c](c.png)\n",
                ["c.png"],
                id="block-html-comment-across-blank-lines",
            ),
            pytest.param(
                "![x][r]\n\n```\n[r]: a.png\n```\n",
                [],
                id="reference-definition-inside-code-is-not-a-definition",
            ),
            pytest.param("![a\n\nb](a.png)", [], id="image-does-not-cross-a-blank-line"),
        ],
    )
    def test_only_rendered_images_are_reported(
        self, body: str, expected_targets: list[str]
    ) -> None:
        assert [e.target for e in scan_image_embeds(body, "")] == expected_targets


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
