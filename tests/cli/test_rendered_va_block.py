"""Both control-system targets must be rendered, and describable, at build time.

The target switch never edits config: whatever a session can switch to has to be
present in the rendered ``build/config.yml`` already. That makes these template
facts load-bearing rather than cosmetic:

* the generic project template renders a ``virtual_accelerator`` connector block
  (previously only the Control Assistant app template had one), so a stock
  project that deploys the VA service has a target to switch to at all. Its
  ``probe_channel`` is a placeholder the project still has to name, because the
  channels a VA serves come from that project's own machine model; the Control
  Assistant preset ships a machine model, so IT is the one that needs no hand
  editing;
* neither template writes a live gateway ``port`` for that block — the connector
  default-fills from ``services.virtual_accelerator.port``, and a rendered port
  would state the same fact twice (see tests/connectors/test_va_gateway_port_fill.py);
* every switchable target carries a ``probe_channel`` disposition, since a target
  without one is ineligible;
* ``control_system.target_switch`` carries the drain/probe tuning defaults;
* the operator acknowledgment for the live machine ships COMMENTED. It is the
  operator's own gateway hostname, and the shipped example is a
  real-hostname-shaped string on purpose so no code can ever string-test a value
  for "still the default".

The two epics blocks are pinned unchanged here as well: adding a switch must not
quietly perturb the production connector configuration either template already
shipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

import osprey.templates
from osprey.cli.templates.manager import TemplateManager

TEMPLATE_ROOT = Path(osprey.templates.__file__).parent
PROJECT_TEMPLATE = "project/config.yml.j2"
CONTROL_ASSISTANT_TEMPLATE = "apps/control_assistant/config.yml.j2"

#: The VA gateway shape both templates render: localhost over CA name-server
#: (TCP) mode, and NO port — that is derived, not written out.
PROBE_PROVEN_GATEWAY_SHAPE = {
    "address": "localhost",
    "use_name_server": True,
}

#: The generic project template's epics connector as committed before the
#: switch work — placeholder gateway, both CA ports, broadcast discovery off.
PROJECT_ORIGINAL_EPICS_BLOCK = {
    "timeout": 5.0,
    "gateways": {
        "read_only": {
            "address": "your-gateway.example.com",
            "port": 5064,
            "use_name_server": False,
        },
        "write_access": {
            "address": "your-gateway.example.com",
            "port": 5084,
            "use_name_server": False,
        },
    },
}

#: The Control Assistant template's epics connector — the untouched ALS
#: production configuration (same constant as
#: tests/templates/test_preset_va_block.py pins).
CONTROL_ASSISTANT_ORIGINAL_EPICS_BLOCK = {
    "timeout": 5.0,
    "gateways": {
        "read_only": {
            "address": "cagw-alsdmz.als.lbl.gov",
            "port": 5064,
            "use_name_server": False,
        },
        "write_access": {
            "address": "cagw-alsdmz.als.lbl.gov",
            "port": 5084,
            "use_name_server": False,
        },
    },
}


def _render_project_template() -> str:
    """Render the generic project template with a representative context.

    The project template takes build-time variables, so it needs the explicit
    context dict (and ChainableUndefined) that
    tests/connectors/test_va_gateway_port_fill.py established.
    """
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        undefined=ChainableUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(PROJECT_TEMPLATE).render(
        project_name="demo",
        facility_name="Demo Facility",
        default_provider="anthropic",
        default_model="claude-haiku-4-5-20251001",
        channel_finder_mode="in_context",
        default_pipeline="in_context",
        enable_in_context=True,
        enable_hierarchical=False,
        enable_middle_layer=False,
        channel_finder_tools=[],
        project_root="/tmp/demo",
    )


def _render_control_assistant_template() -> str:
    """The app template renders with no context at all."""
    manager = TemplateManager()
    return manager.jinja_env.get_template(CONTROL_ASSISTANT_TEMPLATE).render()


def _control_system(rendered: str) -> dict[str, Any]:
    return yaml.safe_load(rendered)["control_system"]


def _both_templates() -> dict[str, str]:
    return {
        PROJECT_TEMPLATE: _render_project_template(),
        CONTROL_ASSISTANT_TEMPLATE: _render_control_assistant_template(),
    }


# ── The generic project template gains the virtual_accelerator target ───────


def test_project_template_renders_virtual_accelerator_block():
    connector = _control_system(_render_project_template())["connector"]
    assert "virtual_accelerator" in connector, (
        "a stock project cannot switch to a target its config never renders"
    )
    assert "mock" in connector and "epics" in connector


def test_project_template_va_gateways_use_probe_proven_shape():
    va = _control_system(_render_project_template())["connector"]["virtual_accelerator"]
    assert va["gateways"]["read_only"] == PROBE_PROVEN_GATEWAY_SHAPE
    assert va["gateways"]["write_access"] == PROBE_PROVEN_GATEWAY_SHAPE


def test_project_template_va_omits_simulation_file():
    """The generic template ships no machine model, so it must not name one.

    The Control Assistant template carries
    ``connector.virtual_accelerator.simulation_file`` because it ships
    ``data/simulation/machine.json`` beside it. A generic project ships no such
    file, and pointing at one that does not exist would be a rendered claim the
    project cannot honour.
    """
    va = _control_system(_render_project_template())["connector"]["virtual_accelerator"]
    assert "simulation_file" not in va


# ── No live gateway port on either template's VA block ──────────────────────


def test_va_gateways_never_render_a_live_port():
    for name, rendered in _both_templates().items():
        va = _control_system(rendered)["connector"]["virtual_accelerator"]
        for role in ("read_only", "write_access"):
            assert "port" not in va["gateways"][role], (
                f"{name}: {role} port must stay derived from services.virtual_accelerator.port"
            )


def test_va_gateway_port_override_ships_as_a_commented_example():
    for name, rendered in _both_templates().items():
        assert "# port: 5074" in rendered, (
            f"{name}: the override a project needs to reach a VA it does not "
            "deploy must stay documented"
        )


# ── probe_channel on every switchable target ────────────────────────────────


def test_va_target_ships_a_probe_channel():
    for name, rendered in _both_templates().items():
        va = _control_system(rendered)["connector"]["virtual_accelerator"]
        assert va.get("probe_channel"), (
            f"{name}: a target with no probe_channel is ineligible to switch to"
        )


def test_control_assistant_probe_channel_is_served_by_its_own_machine_model():
    """The preset's probe channel must exist in the model its VA actually serves.

    The Control Assistant VA is seeded from the ``machine.json`` shipped beside
    the template (the same file its ``simulation_file`` names), so a probe
    channel that is not a channel of that model would fail every switch to va on
    a stock preset build — the one deployment where the framework CAN know the
    answer and therefore must get it right.
    """
    model = json.loads(
        (TEMPLATE_ROOT / "apps/control_assistant/data/simulation/machine.json").read_text()
    )
    va = _control_system(_render_control_assistant_template())["connector"]["virtual_accelerator"]
    assert va["probe_channel"] in model["channels"]


def test_live_target_documents_probe_channel_without_guessing_one():
    """Facility-specific, so it ships commented — unset is the fail-closed side.

    A shipped placeholder would make the live target look eligible while naming
    a channel no facility serves; leaving it unset makes the target simply
    un-switchable-to until an operator names a real channel.

    The generic template's VA block does ship a placeholder, and the asymmetry
    is deliberate: the VA is not hardware, and probing a channel it does not
    serve fails the switch at run time and leaves the session where it was — a
    placeholder there costs a failed switch. The same placeholder on epics would
    advertise the LIVE machine as switch-ready on the strength of a channel name
    nobody checked, which is the direction that must never fail open.
    """
    for name, rendered in _both_templates().items():
        epics = _control_system(rendered)["connector"]["epics"]
        assert "probe_channel" not in epics, f"{name}: no guessed live probe channel"
        assert "# probe_channel:" in rendered, f"{name}: the key must stay documented"


# ── target_switch tuning block ──────────────────────────────────────────────


def test_target_switch_tuning_defaults():
    for name, rendered in _both_templates().items():
        target_switch = _control_system(rendered)["target_switch"]
        assert target_switch["drain_timeout_s"] == 5, name
        assert target_switch["probe_interval_s"] == 30, name


def test_live_gateway_acknowledgment_ships_only_as_a_comment():
    """Rendered-and-set would BE the acknowledgment, which is the whole point.

    The operator has to type their own gateway's hostname; a rendered default
    would acknowledge the live machine on their behalf.
    """
    for name, rendered in _both_templates().items():
        assert "# live_gateway_acknowledged:" in rendered, f"{name}: key undocumented"
        target_switch = _control_system(rendered)["target_switch"]
        assert "live_gateway_acknowledged" not in target_switch, (
            f"{name}: the acknowledgment must not ship pre-granted"
        )


def test_acknowledgment_example_is_a_real_hostname_shape():
    """No code may string-test the example, so it must not look like a sentinel."""
    for name, rendered in _both_templates().items():
        line = next(
            stripped
            for stripped in (raw.strip() for raw in rendered.splitlines())
            if stripped.startswith("# live_gateway_acknowledged:")
        )
        value = line.split(":", 1)[1].strip()
        assert value, f"{name}: empty example"
        assert "." in value and " " not in value, f"{name}: {value!r} is not hostname-shaped"
        assert "<" not in value and ">" not in value, f"{name}: {value!r} is a sentinel"


# ── The production connector configuration is untouched ─────────────────────


def test_project_template_epics_block_is_unchanged():
    epics = _control_system(_render_project_template())["connector"]["epics"]
    assert epics == PROJECT_ORIGINAL_EPICS_BLOCK


def test_control_assistant_epics_block_is_unchanged():
    epics = _control_system(_render_control_assistant_template())["connector"]["epics"]
    assert epics == CONTROL_ASSISTANT_ORIGINAL_EPICS_BLOCK
