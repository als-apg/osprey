"""Both control-system targets must be described where the operator edits them.

The target switch never edits config: whatever a session can switch to has to be
present in the rendered ``build/config.yml`` already, which means it has to be
present in the preset the deployment materialized. The framework template
renders no connector at all — a connector is a declarative fact about a
facility, not a derived one — so the Control Assistant preset is the single
source these facts live in, and its own file is where an operator reads them.

That makes these preset facts load-bearing rather than cosmetic:

* neither gateway of the ``virtual_accelerator`` block writes a live ``port`` —
  the connector default-fills from ``services.virtual_accelerator.port``, and a
  written port would state the same fact twice (see
  tests/connectors/test_va_gateway_port_fill.py);
* every switchable target carries a ``probe_channel`` disposition, since a
  target without one is ineligible;
* ``control_system.target_switch`` carries the drain/probe tuning defaults;
* the operator acknowledgment for the live machine ships COMMENTED. It is the
  operator's own gateway hostname, and the shipped example is a
  real-hostname-shaped string on purpose so no code can ever string-test a value
  for "still the default".

The epics block is pinned unchanged here as well: adding a switch must not
quietly perturb the production connector configuration the preset already
shipped.

The connector blocks' own shape — which of them coexist, and which the preset
selects — is pinned in tests/templates/test_preset_va_block.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import osprey.templates
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_presets import _presets_dir
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.port_layout import DEFAULT_PORT_BASE, default_port

TEMPLATE_ROOT = Path(osprey.templates.__file__).parent
PRESET = "control-assistant"

#: The VA gateway shape both templates render: localhost over CA name-server
#: (TCP) mode, and NO port — that is derived, not written out.
PROBE_PROVEN_GATEWAY_SHAPE = {
    "address": "localhost",
    "use_name_server": True,
}

#: The Control Assistant preset's epics connector — the timeout and nothing
#: else. The gateways ship commented out: authoring them is the go-live edit
#: (same constant as tests/templates/test_preset_va_block.py pins).
CONTROL_ASSISTANT_SHIPPED_EPICS_BLOCK = {"timeout": 5.0}


def _preset_text() -> str:
    """The preset file's own source, comments and all.

    Half of what this file pins is prose an operator reads beside the key it
    explains, so the assertions below read the file rather than a parse of it.
    """
    return (_presets_dir() / f"{PRESET}.yml").read_text(encoding="utf-8")


def _control_system() -> dict[str, Any]:
    """The preset's resolved ``control_system`` block, dotted keys folded in."""
    profile, _profile_dir = resolve_build_profile(None, PRESET)
    return _expand_dotted(profile.config)["control_system"]


# ── No live gateway port on the preset's VA block ───────────────────────────


def test_va_gateways_never_render_a_live_port():
    va = _control_system()["connector"]["virtual_accelerator"]
    for role in ("read_only", "write_access"):
        assert "port" not in va["gateways"][role], (
            f"{role} port must stay derived from services.virtual_accelerator.port"
        )


#: The port the shipped build-profile example gives the live stand-in VA — the
#: first index of the layout's VA band at the default base.
_STANDIN_PORT = default_port("va_standin", base=DEFAULT_PORT_BASE)

#: The commented line an operator types over to acknowledge the live machine.
#: Preset keys are flat and dotted, so the comment carries the whole path.
_ACKNOWLEDGMENT_EXAMPLE = "# control_system.target_switch.live_gateway_acknowledged:"


def test_va_gateway_port_override_stays_documented():
    """The one case that needs a written port must still reach the operator.

    A gateway port is derived from ``services.virtual_accelerator.port``, so the
    only reason to write one is to reach a VA this deployment does not run. The
    preset says so in the prose above the gateway keys; without it an operator
    reading a block with no port has nothing telling them a port is even
    settable.
    """
    assert "reach a VA this deployment does not run" in _preset_text(), (
        "the override a project needs to reach a VA it does not deploy must stay documented"
    )


def test_no_gateway_port_example_can_steer_an_operator_onto_the_standin():
    """No commented port example ships, so none can name the wrong port.

    The shipped build-profile puts the live stand-in VA on the first port of
    the layout's VA band via ``virtual_accelerator.live_standin``. An example
    an operator uncomments verbatim would point the primary VA's gateways at
    the stand-in, so the preset documents the override in prose and ships no
    example value at all.
    """
    gateway_ports = [
        line
        for line in _preset_text().splitlines()
        if "connector.virtual_accelerator.gateways" in line and line.strip().endswith(".port:")
    ]
    assert not gateway_ports, f"a gateway port example ships after all: {gateway_ports}"
    assert f"# port: {_STANDIN_PORT}" not in _preset_text(), (
        f"{_STANDIN_PORT} is where the shipped build-profile puts the live "
        "stand-in VA (virtual_accelerator.live_standin); no example may steer "
        "an operator onto it"
    )


# ── probe_channel on every switchable target ────────────────────────────────


def test_va_target_ships_a_probe_channel():
    va = _control_system()["connector"]["virtual_accelerator"]
    assert va.get("probe_channel"), "a target with no probe_channel is ineligible to switch to"


def test_control_assistant_probe_channel_is_served_by_its_own_machine_model():
    """The preset's probe channel must exist in the model its VA actually serves.

    The Control Assistant VA is seeded from the ``machine.json`` its bundle
    packages (the same file its ``simulation_file`` names), so a probe
    channel that is not a channel of that model would fail every switch to va on
    a stock preset build — the one deployment where the framework CAN know the
    answer and therefore must get it right.
    """
    model = json.loads(
        (TEMPLATE_ROOT / "apps/control_assistant/data/simulation/machine.json").read_text()
    )
    va = _control_system()["connector"]["virtual_accelerator"]
    assert va["probe_channel"] in model["channels"]


def test_live_target_documents_probe_channel_without_guessing_one():
    """Facility-specific, so it ships commented — unset is the fail-closed side.

    A shipped placeholder would make the live target look eligible while naming
    a channel no facility serves; leaving it unset makes the target simply
    un-switchable-to until an operator names a real channel.

    The VA block does ship a placeholder, and the asymmetry is deliberate: the
    VA is not hardware, and probing a channel it does not serve fails the switch
    at run time and leaves the session where it was — a placeholder there costs
    a failed switch. The same placeholder on epics would advertise the LIVE
    machine as switch-ready on the strength of a channel name nobody checked,
    which is the direction that must never fail open.
    """
    epics = _control_system()["connector"]["epics"]
    assert "probe_channel" not in epics, "no guessed live probe channel"
    assert "# control_system.connector.epics.probe_channel:" in _preset_text(), (
        "the key must stay documented"
    )


# ── target_switch tuning block ──────────────────────────────────────────────


def test_target_switch_tuning_defaults():
    target_switch = _control_system()["target_switch"]
    assert target_switch["drain_timeout_s"] == 5
    assert target_switch["probe_interval_s"] == 30


def test_live_gateway_acknowledgment_ships_only_as_a_comment():
    """Rendered-and-set would BE the acknowledgment, which is the whole point.

    The operator has to type their own gateway's hostname; a rendered default
    would acknowledge the live machine on their behalf.
    """
    assert _ACKNOWLEDGMENT_EXAMPLE in _preset_text(), "key undocumented"
    target_switch = _control_system()["target_switch"]
    assert "live_gateway_acknowledged" not in target_switch, (
        "the acknowledgment must not ship pre-granted"
    )


def test_acknowledgment_prose_sits_directly_above_its_own_key():
    """The explanation must be attached to the key it explains, and to no other.

    No build writes ``live_gateway_acknowledged``: the live stand-in is a third
    control target of its own (``standin``), and ``live`` still means the
    machine the facility authored, so the key stays the operator's to fill in.
    That makes the prose above it the only explanation they get. The preset
    states every key as a flat dotted line, so a comment belongs to whatever
    key follows it — prose separated from its key by a blank line, or by
    another key, is re-attached to something else the moment anything is
    written between them.
    """
    lines = _preset_text().splitlines()
    example = next(
        (i for i, line in enumerate(lines) if line.strip().startswith(_ACKNOWLEDGMENT_EXAMPLE)),
        None,
    )
    assert example is not None, "the commented example is gone"

    # Walk up from the key: the run of comment lines immediately above it is its
    # explanation, and that run has to carry the acknowledgment's own prose.
    prose_lines = []
    i = example - 1
    while i >= 0 and lines[i].strip().startswith("#"):
        prose_lines.insert(0, lines[i].strip())
        i -= 1
    prose = "\n".join(prose_lines)
    assert "acknowledgment for the live machine" in prose, (
        "the acknowledgment prose is gone, reworded, or no longer sits "
        f"directly above its key; found: {prose!r}"
    )
    assert "may not switch TO the live target" in prose, (
        "the prose must still say what stays refused while the key is unset"
    )


def test_acknowledgment_example_is_a_real_hostname_shape():
    """No code may string-test the example, so it must not look like a sentinel."""
    line = next(
        stripped
        for stripped in (raw.strip() for raw in _preset_text().splitlines())
        if stripped.startswith(_ACKNOWLEDGMENT_EXAMPLE)
    )
    value = line.split(":", 1)[1].strip()
    assert value, "empty example"
    assert "." in value and " " not in value, f"{value!r} is not hostname-shaped"
    assert "<" not in value and ">" not in value, f"{value!r} is a sentinel"


# ── The production connector configuration is what the preset promises ──────


def test_control_assistant_epics_block_ships_no_gateway_values():
    epics = _control_system()["connector"]["epics"]
    assert epics == CONTROL_ASSISTANT_SHIPPED_EPICS_BLOCK
