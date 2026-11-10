"""Tests for the stdlib-only approval-prompt renderer ``osprey_prompt_render``.

The module is a helper imported by the hooks rather than invoked as a script, so
it is tested by direct import through the ``hook_module`` seam.

The contract under test: a template may only substitute ``input.*`` and
``state.*`` values, optionally with ``| default("..")``; every other construct
refuses the template. Every substituted value and every display line is escaped
for terminal control characters and for HTML, and output, display lines and the
number of display lines are all capped. The input digest is one fixed algorithm,
pinned here by a fixed vector.
"""

from __future__ import annotations

import ast
import hashlib
import sys

import pytest

from tests.hooks.conftest import HOOKS_DIR


@pytest.fixture
def render(hook_module):
    return hook_module("osprey_prompt_render")


# ---------------------------------------------------------------------------
# substitution
# ---------------------------------------------------------------------------


def test_plain_substitution_from_input_and_state(render):
    out = render.render_prompt(
        "Set {{ input.channel }} to {{ state.target }}",
        input={"channel": "SR:BPM1"},
        state={"target": 1.5},
    )
    assert out == "Set SR:BPM1 to 1.5"


def test_nested_dict_walk(render):
    out = render.render_prompt("{{state.plan.name}}", input={}, state={"plan": {"name": "p1"}})
    assert out == "p1"


def test_default_applies_to_a_missing_value(render):
    out = render.render_prompt('{{ state.note | default("none") }}', input={}, state={})
    assert out == "none"


def test_default_applies_to_a_null_value(render):
    out = render.render_prompt("{{ state.note | default('n/a') }}", input={}, state={"note": None})
    assert out == "n/a"


def test_default_applies_to_a_null_intermediate_value(render):
    out = render.render_prompt(
        '{{ state.plan.name | default("-") }}', input={}, state={"plan": None}
    )
    assert out == "-"


def test_null_intermediate_without_default_is_refused(render):
    with pytest.raises(render.PromptTemplateError, match="no value"):
        render.render_prompt("{{ state.plan.name }}", input={}, state={"plan": None})


def test_default_is_ignored_when_the_value_is_present(render):
    out = render.render_prompt('{{ input.x | default("d") }}', input={"x": "v"}, state={})
    assert out == "v"


def test_missing_value_without_default_is_refused(render):
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt("{{ input.absent }}", input={}, state={})


def test_text_without_placeholders_passes_through(render):
    assert render.render_prompt("Apply the plan?", input={}, state={}) == "Apply the plan?"


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "{% if x %}y{% endif %}",
        "{# comment #}",
        "{{ input.x | upper }}",
        "{{ input.x | default(input.y) }}",
        "{{ input.x() }}",
        "{{ input.items[0] }}",
        "{{ config.x }}",
        "{{ input }}",
        "{{ input.x + 1 }}",
        "{{ input.__class__ }}",
        "{{ input.x",
        "{{ }}",
    ],
)
def test_constructs_beyond_substitution_and_default_are_refused(render, template):
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt(template, input={"x": "1", "y": "2", "items": ["a"]}, state={})


@pytest.mark.parametrize(
    "template",
    ["a }} {{ input.x }}", "a {{ b {{ input.x }}", "{{ input.x }} }} {{ input.x }}"],
)
def test_stray_braces_before_a_placeholder_are_refused(render, template):
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt(template, input={"x": "1"}, state={})


def test_attribute_walk_into_a_non_dict_is_refused(render):
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt("{{ input.x.y }}", input={"x": "scalar"}, state={})


@pytest.mark.parametrize("leaf", [{"a": 1}, ["a"]])
def test_a_container_leaf_is_refused(render, leaf):
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt("{{ input.x }}", input={"x": leaf}, state={})


def test_the_error_is_a_value_error(render):
    assert issubclass(render.PromptTemplateError, ValueError)


# ---------------------------------------------------------------------------
# escaping
# ---------------------------------------------------------------------------


def test_control_escape_matches_the_label_semantics(render):
    raw = "a\nb\rc\x1b[31md\x7fe\x85f g h\x00"
    assert render.escape_control(raw) == (
        "a\\x0ab\\x0dc\\x1b[31md\\x7fe\\x85f\\x2028g\\x2029h\\x00"
    )


def test_control_escape_keeps_printable_unicode(render):
    assert render.escape_control("Δ 1.5 µA") == "Δ 1.5 µA"


@pytest.mark.parametrize(
    ("raw", "token"),
    [
        ("\u202e", "\\x202e"),  # RIGHT-TO-LEFT OVERRIDE (Cf)
        ("\u200b", "\\x200b"),  # ZERO WIDTH SPACE (Cf)
        ("\x85", "\\x85"),  # NEXT LINE (Cc, C1)
        ("\x9b", "\\x9b"),  # CONTROL SEQUENCE INTRODUCER (Cc, C1)
        ("\ue000", "\\xe000"),  # private use (Co)
        ("\U000e0001", "\\xe0001"),  # LANGUAGE TAG (Cf)
    ],
    ids=["rlo", "zwsp", "nel", "csi", "private-use", "tag"],
)
def test_control_escape_covers_every_invisible_category(render, raw, token):
    """A character that renders as nothing, or reorders what follows, is shown by codepoint."""
    assert render.escape_control(f"a{raw}b") == f"a{token}b"


def test_a_bidi_override_in_a_value_cannot_reorder_the_prompt(render):
    out = render.render_prompt(
        "set {{ input.ch }} to {{ input.v }}",
        input={"ch": "SR:\u202eBPM1\u200b", "v": "1\x85forged"},
        state={},
    )
    assert out == "set SR:\\x202eBPM1\\x200b to 1\\x85forged"


def test_html_escape(render):
    assert render.escape_html("<b>&\"'</b>") == "&lt;b&gt;&amp;&quot;&#x27;&lt;/b&gt;"


def test_substituted_values_are_escaped_for_terminal_and_html(render):
    out = render.render_prompt(
        "name: {{ input.name }}",
        input={"name": "x\nValidation status: PASSED <script>"},
        state={},
    )
    assert "\n" not in out
    assert out == "name: x\\x0aValidation status: PASSED &lt;script&gt;"


def test_default_literal_is_escaped_too(render):
    out = render.render_prompt('{{ input.x | default("<none>") }}', input={}, state={})
    assert out == "&lt;none&gt;"


def test_substituted_value_cannot_inject_a_placeholder(render):
    out = render.render_prompt(
        "{{ input.a }}", input={"a": "{{ state.secret }}"}, state={"secret": "s"}
    )
    assert out == "{{ state.secret }}"


# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------


def test_each_value_is_capped_on_its_own(render):
    out = render.render_prompt("{{ input.x }}", input={"x": "a" * 100_000}, state={})
    assert len(out) <= render.MAX_VALUE_CHARS
    assert out.endswith(render.TRUNCATION_MARK)


def test_the_value_cap_sits_well_below_the_output_cap(render):
    assert render.MAX_VALUE_CHARS * 4 <= render.MAX_OUTPUT_CHARS


def test_an_oversized_value_never_pushes_later_template_text_out(render):
    """The literal text after a huge value, and every later placeholder, still shows."""
    out = render.render_prompt(
        "write {{ input.x }} to {{ input.ch }} -- irreversible",
        input={"x": "9" * 100_000, "ch": "SR:BPM1"},
        state={},
    )
    assert out.startswith("write 999")
    assert out.endswith(f"{render.TRUNCATION_MARK} to SR:BPM1 -- irreversible")


def test_the_value_cap_never_splits_an_escape(render):
    out = render.render_prompt(
        "{{ input.x }}", input={"x": "\u202e" * render.MAX_VALUE_CHARS}, state={}
    )
    body = out[: -len(render.TRUNCATION_MARK)]
    assert out.endswith(render.TRUNCATION_MARK)
    assert body == "\\x202e" * (len(body) // len("\\x202e"))


def test_a_prompt_over_the_output_cap_is_refused_not_truncated(render):
    template = "{{ input.x }}" * (render.MAX_OUTPUT_CHARS // render.MAX_VALUE_CHARS + 1)
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt(template, input={"x": "a" * 100_000}, state={})


def test_long_template_text_is_refused_not_truncated(render):
    with pytest.raises(render.PromptTemplateError):
        render.render_prompt("t" * (render.MAX_OUTPUT_CHARS + 1), input={}, state={})


def test_display_lines_are_escaped(render):
    lines = render.render_display_lines(["HCM1: +0.1 A\nforged", "<i>x</i>"])
    assert lines == ["HCM1: +0.1 A\\x0aforged", "&lt;i&gt;x&lt;/i&gt;"]


def test_each_display_line_is_capped(render):
    (line,) = render.render_display_lines(["b" * 10_000])
    assert len(line) <= render.MAX_DISPLAY_LINE_CHARS
    assert line.endswith(render.TRUNCATION_MARK)


def test_a_display_line_cap_never_splits_an_escape(render):
    (line,) = render.render_display_lines(["<" * 10_000])
    body = line[: -len(render.TRUNCATION_MARK)]
    assert len(line) <= render.MAX_DISPLAY_LINE_CHARS
    assert body == "&lt;" * (len(body) // len("&lt;"))


def test_display_lines_are_capped_by_count(render):
    n = render.MAX_DISPLAY_LINES + 5
    lines = render.render_display_lines([f"line {i}" for i in range(n)])
    assert len(lines) == render.MAX_DISPLAY_LINES + 1
    assert lines[: render.MAX_DISPLAY_LINES] == [
        f"line {i}" for i in range(render.MAX_DISPLAY_LINES)
    ]
    assert "5 more" in lines[-1]


def test_display_lines_past_the_count_cap_are_never_rendered(render):
    rendered: list[int] = []

    class Line:
        def __init__(self, i: int) -> None:
            self.i = i

        def __str__(self) -> str:
            rendered.append(self.i)
            return f"line {self.i}"

    n = render.MAX_DISPLAY_LINES + 1000
    lines = render.render_display_lines([Line(i) for i in range(n)])
    assert len(lines) == render.MAX_DISPLAY_LINES + 1
    assert "1000 more" in lines[-1]
    assert rendered == list(range(render.MAX_DISPLAY_LINES))


def test_display_lines_accept_scalars(render):
    assert render.render_display_lines([1, 2.5]) == ["1", "2.5"]


@pytest.mark.parametrize("bad", ["one line", {"a": 1}, None, [["nested"]], [{"a": 1}]])
def test_display_lines_must_be_a_list_of_scalars(render, bad):
    with pytest.raises(render.PromptTemplateError):
        render.render_display_lines(bad)


# ---------------------------------------------------------------------------
# input digest
# ---------------------------------------------------------------------------


def test_canonical_json_is_sorted_and_compact(render):
    assert render.canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"\\u00e9"],"b":1}'


def test_input_digest_fixed_vector(render):
    tool_input = {"setpoints": {"HCM1": 0.1, "VCM2": -0.25}, "mode": "apply", "ignored": 7}
    digest = render.input_digest(tool_input, ["mode", "setpoints"])
    assert digest == "719d383b60815520c01df060d04bc0a2ff5aa0b0c3dc140934f39cf0d8fa89b0"
    expected = hashlib.sha256(b'{"mode":"apply","setpoints":{"HCM1":0.1,"VCM2":-0.25}}').hexdigest()
    assert digest == expected


def test_input_digest_ignores_key_order_of_over(render):
    tool_input = {"a": 1, "b": 2}
    assert render.input_digest(tool_input, ["a", "b"]) == render.input_digest(
        tool_input, ["b", "a"]
    )


def test_input_digest_missing_key_raises(render):
    with pytest.raises(KeyError):
        render.input_digest({"a": 1}, ["a", "b"])


# ---------------------------------------------------------------------------
# shipping shape
# ---------------------------------------------------------------------------


def test_module_is_stdlib_only():
    tree = ast.parse((HOOKS_DIR / "osprey_prompt_render.py").read_text())
    heads = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            heads.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            heads.add((node.module or "").split(".")[0])
    assert heads <= set(sys.stdlib_module_names) | {"__future__"}
