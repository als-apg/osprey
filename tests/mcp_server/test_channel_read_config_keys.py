"""The two channel_read sizing knobs: code defaults and rendered-config presence.

``channel_read`` decides inline-vs-artifact from two numbers, and both are
deployment-tunable:

* ``control_system.read_inline_max_elements`` (default 2000) — the per-value
  element budget, times four for one call's aggregate budget;
* ``control_system.channel_read_artifact_retention`` (default 20) — the
  per-channel rolling window of unpinned read artifacts, 0 = keep everything.

Two halves here. The first pins the accessors: defaults when config says
nothing, configured values honoured, and unusable values degrading to the
default rather than taking down a read the hardware already answered. The second
pins that every shipped ``config.yml.j2`` actually renders those keys — NESTED
under ``control_system:``, because a dotted ``control_system.read_inline_max_elements:``
line in a rendered config.yml is inert: it loads as a top-level key with a dot in
its name and the tool would silently keep the default.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.mcp_server.control_system.tools import channel_read

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

CONFIG_TARGET = "osprey.utils.config.get_config_value"

INLINE_KEY = "control_system.read_inline_max_elements"
RETENTION_KEY = "control_system.channel_read_artifact_retention"


def _fake_config(values: dict):
    """A get_config_value side_effect answering *values*, else the caller default."""

    def _side_effect(path, default=None, config_path=None):
        return values.get(path, default)

    return _side_effect


def _boom(path, default=None, config_path=None):
    """Config that cannot be resolved at all (no config.yml, broken builder)."""
    raise RuntimeError("no configuration loaded")


# Context sufficient to render every app + project template (extra keys are
# harmless; missing keys render empty and the `{% if enable_* %}` pipeline
# blocks stay off). Mirrors tests/templates/test_telemetry_optin_templates.py.
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
_HELLO = "apps/hello_world/config.yml.j2"
_CONTROL = "apps/control_assistant/config.yml.j2"
_ALL = [_PROJECT, _HELLO, _CONTROL]
#: Templates that ship the EPICS connector block, and so document PVA routing.
_EPICS_TEMPLATES = [_PROJECT, _CONTROL]


def _render(path: str) -> str:
    return TemplateManager().jinja_env.get_template(path).render(**_CTX)


def _source(path: str) -> str:
    return (TemplateManager().template_root / path).read_text()


def _cfg(path: str) -> dict:
    return yaml.safe_load(_render(path))


# ---------------------------------------------------------------------------
# accessors — defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_documented_defaults_are_the_module_constants():
    """The defaults the templates document are the ones the code falls back to."""
    assert channel_read.DEFAULT_READ_INLINE_MAX_ELEMENTS == 2000
    assert channel_read.DEFAULT_CHANNEL_READ_ARTIFACT_RETENTION == 20
    assert channel_read.AGGREGATE_BUDGET_FACTOR == 4


@pytest.mark.unit
def test_accessors_return_defaults_when_keys_absent():
    """An unconfigured deployment gets the documented defaults."""
    with patch(CONFIG_TARGET, _fake_config({})):
        assert channel_read.get_read_inline_max_elements() == 2000
        assert channel_read.get_channel_read_artifact_retention() == 20
        assert channel_read.get_read_aggregate_max_elements() == 8000


@pytest.mark.unit
def test_accessors_return_defaults_when_config_unavailable():
    """No config.yml at all degrades to the defaults, it does not raise.

    These accessors run on the read path; a missing config must never turn a
    successful hardware read into an error.
    """
    with patch(CONFIG_TARGET, _boom):
        assert channel_read.get_read_inline_max_elements() == 2000
        assert channel_read.get_channel_read_artifact_retention() == 20
        assert channel_read.get_read_aggregate_max_elements() == 8000


# ---------------------------------------------------------------------------
# accessors — configured values
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_accessors_honor_configured_values():
    """config.yml overrides both knobs."""
    with patch(CONFIG_TARGET, _fake_config({INLINE_KEY: 500, RETENTION_KEY: 3})):
        assert channel_read.get_read_inline_max_elements() == 500
        assert channel_read.get_channel_read_artifact_retention() == 3


@pytest.mark.unit
def test_aggregate_budget_follows_the_configured_threshold():
    """The per-call budget is 4x whatever the per-value threshold is set to."""
    with patch(CONFIG_TARGET, _fake_config({INLINE_KEY: 500})):
        assert channel_read.get_read_aggregate_max_elements() == 2000


@pytest.mark.unit
def test_retention_zero_is_honored_not_treated_as_unset():
    """0 means "keep everything" — it must not fall back to the default of 20."""
    with patch(CONFIG_TARGET, _fake_config({RETENTION_KEY: 0})):
        assert channel_read.get_channel_read_artifact_retention() == 0


@pytest.mark.unit
def test_string_values_are_coerced():
    """YAML/env round-trips can hand these through as strings."""
    with patch(CONFIG_TARGET, _fake_config({INLINE_KEY: "128", RETENTION_KEY: "5"})):
        assert channel_read.get_read_inline_max_elements() == 128
        assert channel_read.get_channel_read_artifact_retention() == 5


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["not-a-number", None, -1, [2000], True, False])
def test_unusable_values_degrade_to_defaults(bad):
    """A mistyped knob is ignored, not fatal, and not silently negative."""
    with patch(CONFIG_TARGET, _fake_config({INLINE_KEY: bad, RETENTION_KEY: bad})):
        assert channel_read.get_read_inline_max_elements() == 2000
        assert channel_read.get_channel_read_artifact_retention() == 20


# ---------------------------------------------------------------------------
# rendered templates
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("template", _ALL)
def test_templates_render_both_keys_nested_under_control_system(template):
    """Every shipped config template documents both knobs at their defaults."""
    cfg = _cfg(template)
    control_system = cfg.get("control_system")
    assert isinstance(control_system, dict), f"{template}: no control_system block"

    assert control_system.get("read_inline_max_elements") == 2000, (
        f"{template}: read_inline_max_elements missing or not at its documented default"
    )
    assert control_system.get("channel_read_artifact_retention") == 20, (
        f"{template}: channel_read_artifact_retention missing or not at its default"
    )


@pytest.mark.unit
@pytest.mark.parametrize("template", _ALL)
def test_keys_are_nested_never_dotted(template):
    """A dotted key in a rendered config.yml is INERT — guard against that spelling.

    ``control_system.read_inline_max_elements: 2000`` at the top level loads as a
    key whose *name* contains dots; nothing reads it, and the tool keeps its
    default while the config file appears to say otherwise.
    """
    cfg = _cfg(template)
    dotted = [key for key in cfg if isinstance(key, str) and "." in key]
    assert not dotted, f"{template}: dotted top-level keys are inert: {dotted}"


@pytest.mark.unit
@pytest.mark.parametrize("template", _ALL)
def test_default_comment_cites_the_inline_budget_precedent(template):
    """The comment explains where 2000 comes from and names the 4x call budget."""
    text = _render(template)
    assert "read_inline_max_elements" in text
    assert "4x" in text, f"{template}: aggregate budget not documented"
    # The provenance half of the guard: 2000 must be traced to the shared
    # plotting/figure inline budget, not stated as a bare magic number.
    assert "DEFAULT_MAX_POINTS" in text or "inline budget" in text, (
        f"{template}: 2000 default lacks its inline-budget provenance comment"
    )


# ---------------------------------------------------------------------------
# PVA connector keys
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("template", _EPICS_TEMPLATES)
def test_epics_templates_document_pva_routing(template):
    """The EPICS-carrying templates document pva_channels and pva_gateway."""
    text = _render(template)
    assert "pva_channels" in text, f"{template}: pva_channels not documented"
    assert "pva_gateway" in text, f"{template}: pva_gateway not documented"
    assert "EPICS_PVA_AUTO_ADDR_LIST" in text, (
        f"{template}: gateway containment note (AUTO_ADDR_LIST forced to NO) missing"
    )


@pytest.mark.unit
@pytest.mark.parametrize("template", _EPICS_TEMPLATES)
def test_pva_routing_is_off_by_default(template):
    """Both PVA keys ship commented out, so a scaffolded project stays pure CA.

    A live ``pva_gateway`` would force ``EPICS_PVA_AUTO_ADDR_LIST=NO`` against a
    placeholder address, and a live ``pva_channels`` would pull in the p4p import
    at connect time. Absent is the safe default; the comments are the documentation.
    """
    epics = (_cfg(template).get("control_system", {}).get("connector", {}).get("epics", {})) or {}
    assert "pva_channels" not in epics, f"{template}: pva_channels must ship commented out"
    assert "pva_gateway" not in epics, f"{template}: pva_gateway must ship commented out"


@pytest.mark.unit
def test_hello_world_points_at_the_epics_connector_for_pva():
    """The mock-only preset names where the PVA keys live instead of faking a block."""
    text = _source(_HELLO)
    assert "pva_channels" in text and "pva_gateway" in text
    assert "control_system.connector.epics" in text
