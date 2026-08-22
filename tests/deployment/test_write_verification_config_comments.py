"""What the generated ``write_verification`` comments must say — and must not.

Those comments are the only place an operator reads about how a write gets
confirmed, and they were wrong: they claimed a per-channel verification entry
applied "only on single-channel tool writes" and that
``osprey.runtime.write_channel`` "discards" a failed verification. Both are
false — batch writes resolve per channel like single writes, and the Python path
raises. A comment that lies about a safety knob is worse than no comment, so the
corrected sentences are pinned here and the retired ones are pinned as absent.

Both shipped templates carry the block, so both are rendered through the real
deployment render path (``TemplateManager``), not read as raw text: a sentence
that only survives in the source but is cut by a Jinja branch would pass a
plain file grep and still never reach a generated project.
"""

from __future__ import annotations

import re

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager

# Context sufficient to render every template below; extra keys are harmless and
# missing keys leave the `{% if enable_* %}` blocks off. Mirrors
# tests/mcp_server/test_channel_read_config_keys.py.
_CTX = {
    "project_name": "demo",
    "facility_name": "Demo",
    "default_provider": "anthropic",
    "default_model": "haiku",
    "current_python_env": "/usr/bin/python3",
    "selected_web_panels": [],
    "builtin_panels": [],
    "channel_finder_mode": "in_context",
    "default_pipeline": "in_context",
    "enable_in_context": False,
    "enable_hierarchical": False,
    "enable_middle_layer": False,
    "system": {"timezone": "UTC"},
    "osprey_labels": {"project_name": "demo", "project_root": "/tmp/demo"},
}

_PROJECT = "project/config.yml.j2"
_CONTROL = "apps/control_assistant/config.yml.j2"
#: Every shipped template that configures write verification.
_TEMPLATES = [_PROJECT, _CONTROL]

#: The corrected asymmetry, as one sentence pair. Comments wrap across lines, so
#: the rendered text is normalised before this is looked for.
CORRECTED_SENTENCE = (
    "A failed verification is a hard stop on the Python path: "
    "osprey.runtime.write_channel routes through write_channel_checked and "
    "raises ChannelWriteFailedError. The channel_write MCP tool does not raise; "
    "it reports the outcome in write_state and verification.verified."
)

#: Claims the code never made. Each described the pre-#465 behaviour wrongly.
RETIRED_CLAIMS = ["only on single-channel tool writes", "discards it"]

#: The four resolution layers, in order, in the wording the generated agent
#: rules use — the config comments and the rules describe one mechanism.
LAYER_PHRASES = [
    '"verification" entry in channel_limits.json',
    '"defaults" block',
    "connector config",
    '"callback"',
]


def _render(path: str) -> str:
    return TemplateManager().jinja_env.get_template(path).render(**_CTX)


def _comment_text(rendered: str) -> str:
    """All ``#`` comment text from a rendered config, as one normalised string.

    Comments wrap wherever the line length says, so an assertion on a sentence
    has to be made against text with the ``#`` markers and line breaks taken
    out; otherwise every reflow would be a test failure.
    """
    parts = []
    for line in rendered.splitlines():
        marker = line.find("#")
        if marker != -1:
            parts.append(line[marker + 1 :])
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


@pytest.mark.unit
@pytest.mark.parametrize("template", _TEMPLATES)
def test_corrected_hard_stop_sentence_is_present(template):
    """Both templates state that Python raises and the MCP tool reports."""
    text = _comment_text(_render(template))
    assert CORRECTED_SENTENCE in text, (
        f"{template} no longer documents the Python-raises / tool-reports "
        "asymmetry. An operator reading these comments would expect a failed "
        "verification to be silently tolerated on both paths."
    )


@pytest.mark.unit
@pytest.mark.parametrize("template", _TEMPLATES)
@pytest.mark.parametrize("claim", RETIRED_CLAIMS)
def test_retired_claims_are_gone(template, claim):
    """The two false sentences must not come back."""
    text = _comment_text(_render(template))
    assert claim not in text, (
        f"{template} still claims {claim!r}. Per-channel verification applies to "
        "batch writes too, and osprey.runtime.write_channel raises rather than "
        "discarding a failed verification."
    )


@pytest.mark.unit
@pytest.mark.parametrize("template", _TEMPLATES)
def test_four_layer_precedence_is_documented_in_order(template):
    """The comments name all four resolution layers, most specific first."""
    text = _comment_text(_render(template))
    positions = []
    for phrase in LAYER_PHRASES:
        index = text.find(phrase)
        assert index != -1, f"{template} does not name the layer {phrase!r}"
        positions.append(index)
    assert positions == sorted(positions), (
        f"{template} lists the verification layers out of order; the comment has "
        "to read most-specific-first or it describes a different precedence."
    )


@pytest.mark.unit
@pytest.mark.parametrize("template", _TEMPLATES)
def test_config_layer_keys_are_named_on_one_line(template):
    """Both keys are named, each whole on a single line.

    Config-key evidence is collected with single-line regexes, so a key mention
    split across two comment lines reads as no mention at all.
    """
    rendered = _render(template)
    for key in (
        "control_system.write_verification.default_level",
        "control_system.write_verification.default_tolerance_percent",
    ):
        assert any(key in line for line in rendered.splitlines()), (
            f"{template} does not name {key} on a single line"
        )


@pytest.mark.unit
@pytest.mark.parametrize("template", _TEMPLATES)
def test_keys_and_defaults_are_unchanged(template):
    """Comments changed; the configuration did not."""
    config = yaml.safe_load(_render(template))
    block = config["control_system"]["write_verification"]
    assert block["default_level"] == "callback"
    assert block["default_tolerance_percent"] == 0.1
