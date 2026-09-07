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
pins that every preset carrying a control system actually writes those keys.

The presets are where they live now: the framework template renders no
``control_system`` block at all, so a deployment's sizing knobs come from its
profile's ``config:`` block, which the preset writes down. Presets spell config
keys dotted; the build expands them, and the guard is that they expand — a
dotted key that survived into the rendered config.yml would be INERT, loading as
a top-level key whose *name* contains dots, so the tool would keep its default
while the config file appeared to say otherwise.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import osprey.profiles
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.mcp_server.control_system.tools import channel_read
from osprey.services.bluesky_bridge.figure import DEFAULT_MAX_POINTS

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


#: The bundled presets that ship a control system, and so size channel_read.
#: `ariel-standalone` and `channel-finder-standalone` carry no control_system
#: block at all, so there is nothing for these knobs to size.
_PRESETS = ["hello-world", "control-assistant"]

#: The preset that ships the EPICS connector block, and so documents PVA routing.
_EPICS_PRESETS = ["control-assistant"]


def _source(preset: str) -> str:
    return (Path(osprey.profiles.__file__).parent / "presets" / f"{preset}.yml").read_text(
        encoding="utf-8"
    )


def _cfg(preset: str) -> dict:
    """The preset's resolved ``config:`` block, dotted keys folded in."""
    profile, _profile_dir = resolve_build_profile(None, preset)
    return _expand_dotted(profile.config)


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
# preset config
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("preset", _PRESETS)
def test_presets_carry_both_keys_nested_under_control_system(preset):
    """Every control-system preset writes both knobs at their documented defaults."""
    cfg = _cfg(preset)
    control_system = cfg.get("control_system")
    assert isinstance(control_system, dict), f"{preset}: no control_system block"

    assert control_system.get("read_inline_max_elements") == 2000, (
        f"{preset}: read_inline_max_elements missing or not at its documented default"
    )
    assert control_system.get("channel_read_artifact_retention") == 20, (
        f"{preset}: channel_read_artifact_retention missing or not at its default"
    )


@pytest.mark.unit
@pytest.mark.parametrize("preset", _PRESETS)
def test_keys_are_nested_never_dotted(preset):
    """A dotted key that survives into config.yml is INERT — guard the expansion.

    ``control_system.read_inline_max_elements: 2000`` at the top level loads as a
    key whose *name* contains dots; nothing reads it, and the tool keeps its
    default while the config file appears to say otherwise. Presets author these
    dotted on purpose, so the guard is that every one of them expands.
    """
    cfg = _cfg(preset)
    dotted = [key for key in cfg if isinstance(key, str) and "." in key]
    assert not dotted, f"{preset}: dotted top-level keys are inert: {dotted}"


@pytest.mark.unit
@pytest.mark.parametrize("preset", _PRESETS)
def test_default_comment_documents_the_aggregate_call_budget(preset):
    """The preset comment names the 4x per-call budget beside the per-value one."""
    text = _source(preset)
    assert "read_inline_max_elements" in text
    assert "4x" in text, f"{preset}: aggregate budget not documented"


@pytest.mark.unit
def test_the_default_is_the_repo_wide_inline_budget():
    """2000 is traced to the shared plotting budget, not stated as a magic number.

    The provenance used to be a comment in each app template's config block.
    Those templates are gone, so it is pinned where the number actually lives:
    the accessor default IS the Bluesky figure builder's point budget, and the
    module comment says so.
    """
    assert channel_read.DEFAULT_READ_INLINE_MAX_ELEMENTS == DEFAULT_MAX_POINTS
    source = Path(channel_read.__file__).read_text(encoding="utf-8")
    assert "DEFAULT_MAX_POINTS" in source, (
        "the 2000 default lacks its inline-budget provenance comment"
    )


# ---------------------------------------------------------------------------
# PVA connector keys
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("preset", _EPICS_PRESETS)
def test_epics_presets_document_pva_routing(preset):
    """The EPICS-carrying presets document pva_channels and pva_gateway."""
    text = _source(preset)
    assert "pva_channels" in text, f"{preset}: pva_channels not documented"
    assert "pva_gateway" in text, f"{preset}: pva_gateway not documented"
    assert "EPICS_PVA_" in text, f"{preset}: gateway containment note missing"


@pytest.mark.unit
@pytest.mark.parametrize("preset", _EPICS_PRESETS)
def test_pva_routing_is_off_by_default(preset):
    """Both PVA keys ship commented out, so a scaffolded project stays pure CA.

    A live ``pva_gateway`` would point the EPICS_PVA_* routing at a placeholder
    address, and a live ``pva_channels`` would pull in the p4p import at connect
    time. Absent is the safe default; the comments are the documentation.
    """
    epics = (_cfg(preset).get("control_system", {}).get("connector", {}).get("epics", {})) or {}
    assert "pva_channels" not in epics, f"{preset}: pva_channels must ship commented out"
    assert "pva_gateway" not in epics, f"{preset}: pva_gateway must ship commented out"
