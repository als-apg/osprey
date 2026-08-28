"""The VA connector's gateway port is derived from the VA service it talks to.

Rendering ``control_system.connector.virtual_accelerator.gateways.*.port`` as a
hardcoded ``5064`` while the deployed soft-IOC's port comes from
``services.virtual_accelerator.port`` would state the same fact in two places,
with no validation between them — and documenting the trap ("MUST stay 5064 ...
changing this here would desync the connector from the deployed container") is
not a fix.

The template leaves the port out: with it unset the connector default-fills
from ``services.virtual_accelerator.port`` at config-load time, so moving the
deployed VA's port moves the connector with it.  An explicit gateway ``port``
still wins verbatim — that is how a project reaches a VA it does not deploy,
and what keeps a legacy rendered config that carries the key working.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.templates
from osprey.connectors.control_system import va_connector
from osprey.connectors.control_system.va_connector import (
    DEFAULT_VA_PORT,
    VirtualAcceleratorConnector,
    fill_gateway_ports,
)

TEMPLATE_ROOT = Path(osprey.templates.__file__).parent
CONTROL_ASSISTANT_TEMPLATE = "apps/control_assistant/config.yml.j2"
PRESET_PATH = "profiles/presets/control-assistant.yml"


def _render_template(relative_path: str) -> str:
    """Render a shipped app config template with a representative context."""
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        undefined=ChainableUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(relative_path).render(
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


def _rendered_va_block() -> dict[str, Any]:
    config = yaml.safe_load(_render_template(CONTROL_ASSISTANT_TEMPLATE))
    return config["control_system"]["connector"]["virtual_accelerator"]


@pytest.fixture
def deployed_va_port(monkeypatch):
    """Pin what ``services.virtual_accelerator.port`` resolves to.

    ``resolve_va_gateway_port`` imports ``get_config_value`` from
    ``osprey.utils.config`` at call time, so patching it on that module is
    what the connector actually sees.
    """
    from osprey.utils import config as config_module

    def _set(port: Any | None) -> None:
        def fake_get_config_value(path: str, default: Any = None, config_path: str | None = None):
            assert path == "services.virtual_accelerator.port", path
            return default if port is None else port

        monkeypatch.setattr(config_module, "get_config_value", fake_get_config_value)

    return _set


# ---------------------------------------------------------------------------
# The shipped template and preset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_renders_no_hardcoded_va_gateway_port() -> None:
    """No gateway writes the port out — there is nothing to drift."""
    gateways = _rendered_va_block()["gateways"]

    assert set(gateways) == {"read_only", "write_access"}
    for name, gateway in gateways.items():
        assert "port" not in gateway, (
            f"the {name} gateway still renders a hardcoded port ({gateway.get('port')!r}) — "
            "it is derived from services.virtual_accelerator.port"
        )
        assert gateway["address"] == "localhost"
        assert gateway["use_name_server"] is True


@pytest.mark.unit
def test_template_keeps_a_commented_port_override_example() -> None:
    """A project reaching a VA it does not deploy needs to see how."""
    text = _render_template(CONTROL_ASSISTANT_TEMPLATE)

    assert "# port: 5094" in text, (
        "the commented gateway port override example is gone — a split "
        "host/container setup has no way to see that `port:` still works"
    )
    assert "services.virtual_accelerator.port" in text


@pytest.mark.unit
def test_preset_does_not_warn_the_port_must_not_change() -> None:
    """The preset must not warn about a hardcode it does not have."""
    preset_path = Path(osprey.__file__).parent / PRESET_PATH
    text = preset_path.read_text(encoding="utf-8")
    preset = yaml.safe_load(text)

    assert preset["virtual_accelerator"]["port"] == DEFAULT_VA_PORT
    assert "MUST stay 5064" not in text, (
        "the preset still tells the operator not to change a knob that is now live"
    )


# ---------------------------------------------------------------------------
# An unset gateway port follows services.virtual_accelerator.port
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unset_gateway_port_follows_the_deployed_va_port(deployed_va_port) -> None:
    """The whole point: moving the VA service's port moves the connector."""
    deployed_va_port(5074)

    filled = fill_gateway_ports(
        {
            "gateways": {
                "read_only": {"address": "localhost", "use_name_server": True},
                "write_access": {"address": "localhost", "use_name_server": True},
            }
        }
    )

    assert filled["gateways"]["read_only"]["port"] == 5074
    assert filled["gateways"]["write_access"]["port"] == 5074


@pytest.mark.unit
def test_rendered_template_follows_a_post_render_port_edit(deployed_va_port) -> None:
    """The as-shipped config, fed the edited service port, follows it."""
    deployed_va_port(15064)

    filled = fill_gateway_ports(_rendered_va_block())

    assert filled["gateways"]["read_only"]["port"] == 15064
    assert filled["gateways"]["write_access"]["port"] == 15064
    # Everything else about the gateway survives the fill.
    assert filled["gateways"]["read_only"]["use_name_server"] is True
    assert filled["simulation_file"] == "data/simulation/machine.json"


@pytest.mark.unit
def test_defaults_to_5064_when_the_service_port_is_unset(deployed_va_port) -> None:
    """With neither the gateway nor the service naming a port, 5064 stands."""
    deployed_va_port(None)

    filled = fill_gateway_ports({"gateways": {"read_only": {"address": "localhost"}}})

    assert filled["gateways"]["read_only"]["port"] == DEFAULT_VA_PORT


@pytest.mark.unit
def test_unreadable_config_falls_back_to_5064(monkeypatch) -> None:
    """Outside a project context the connector falls back to the default port."""
    from osprey.utils import config as config_module

    def exploding_get_config_value(path: str, default: Any = None, config_path: str = None):
        raise FileNotFoundError("no config.yml here")

    monkeypatch.setattr(config_module, "get_config_value", exploding_get_config_value)

    filled = fill_gateway_ports({"gateways": {"read_only": {"address": "localhost"}}})

    assert filled["gateways"]["read_only"]["port"] == DEFAULT_VA_PORT


# ---------------------------------------------------------------------------
# An explicit gateway port wins
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_explicit_gateway_port_wins(deployed_va_port) -> None:
    """A project pointed at a VA it does not deploy keeps its own port."""
    deployed_va_port(15064)

    filled = fill_gateway_ports(
        {
            "gateways": {
                "read_only": {"address": "va.example.org", "port": 5074},
                "write_access": {"address": "va.example.org", "port": 5084},
            }
        }
    )

    assert filled["gateways"]["read_only"]["port"] == 5074
    assert filled["gateways"]["write_access"]["port"] == 5084


@pytest.mark.unit
def test_explicit_and_derived_ports_mix_per_gateway(deployed_va_port) -> None:
    """Filling is per gateway — one explicit port does not pin the other."""
    deployed_va_port(15064)

    filled = fill_gateway_ports(
        {
            "gateways": {
                "read_only": {"address": "localhost"},
                "write_access": {"address": "localhost", "port": 5084},
            }
        }
    )

    assert filled["gateways"]["read_only"]["port"] == 15064
    assert filled["gateways"]["write_access"]["port"] == 5084


@pytest.mark.unit
def test_fully_explicit_config_never_reads_the_config_file(monkeypatch) -> None:
    """Nothing to fill means nothing to resolve — and the dict comes back as-is."""
    from osprey.utils import config as config_module

    def unexpected_read(*args, **kwargs):
        raise AssertionError("resolved the service port for a fully explicit config")

    monkeypatch.setattr(config_module, "get_config_value", unexpected_read)

    config = {"gateways": {"read_only": {"address": "localhost", "port": 5074}}}

    assert fill_gateway_ports(config) is config


# ---------------------------------------------------------------------------
# Shape tolerance and non-mutation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_callers_config_is_not_mutated(deployed_va_port) -> None:
    """The factory hands over the loaded config block; the fill must not edit it."""
    deployed_va_port(15064)
    config = {"gateways": {"read_only": {"address": "localhost"}}}

    fill_gateway_ports(config)

    assert config == {"gateways": {"read_only": {"address": "localhost"}}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"gateways": {}},
        {"timeout": 5.0},
        {"gateways": None},
    ],
    ids=["no-gateways-key", "empty-gateways", "timeout-only", "null-gateways"],
)
def test_a_config_with_no_gateways_passes_through(config: dict[str, Any], monkeypatch) -> None:
    """A gateway-less config is the EPICS connector's problem, not the fill's."""
    from osprey.utils import config as config_module

    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda *a, **k: pytest.fail("resolved a port for a gateway-less config"),
    )

    assert fill_gateway_ports(config) is config


# ---------------------------------------------------------------------------
# The fill happens on the connector's config-load path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connect_fills_the_port_before_epics_sees_it(deployed_va_port, monkeypatch) -> None:
    """The derived port reaches EPICSConnector.connect, which configures CA.

    The real ``EPICSConnector.connect`` is stubbed: it imports pyepics and
    sets up a CA context against a soft-IOC that is not running here.
    """
    deployed_va_port(15064)
    captured: dict[str, Any] = {}

    async def fake_epics_connect(self, config):
        captured["config"] = config

    monkeypatch.setattr(va_connector.EPICSConnector, "connect", fake_epics_connect, raising=True)

    connector = VirtualAcceleratorConnector()
    await connector.connect({"timeout": 5.0, "gateways": {"read_only": {"address": "localhost"}}})

    assert captured["config"]["gateways"]["read_only"]["port"] == 15064
    assert captured["config"]["timeout"] == 5.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plain_epics_connector_does_not_follow_the_va_service_port(monkeypatch) -> None:
    """A production EPICS gateway is external infrastructure — it must not move.

    Guards the seam: the fill lives on the VA subclass, so an ``epics`` config
    that omits a port keeps the EPICS connector's own default even when the
    project also deploys a VA on some other port.
    """
    from osprey.connectors.control_system.epics_connector import EPICSConnector
    from osprey.utils import config as config_module

    def fake_get_config_value(path: str, default: Any = None, config_path: str | None = None):
        if path == "services.virtual_accelerator.port":
            return 15064
        return default

    monkeypatch.setattr(config_module, "get_config_value", fake_get_config_value)
    for var in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CA_NAME_SERVERS"):
        monkeypatch.delenv(var, raising=False)

    connector = EPICSConnector()
    await connector.connect({"gateways": {"read_only": {"address": "gw.example.org"}}})

    # 5064 is the EPICS connector's own generic default, not the VA's port.
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064", (
        "the plain EPICS connector followed services.virtual_accelerator.port"
    )
