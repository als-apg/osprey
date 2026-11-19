"""The VA connector's gateway port is derived from the VA service it talks to.

Rendering ``control_system.connector.virtual_accelerator.gateways.*.port`` as a
hardcoded ``5064`` while the deployed soft-IOC's port comes from
``services.virtual_accelerator.port`` would state the same fact in two places,
with no validation between them — and documenting the trap ("MUST stay 5064 ...
changing this here would desync the connector from the deployed container") is
not a fix.

The preset leaves the port out: with it unset the connector default-fills
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

import osprey.profiles
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey_connectors.config_sections import GatewayRow
from osprey_connectors.control_system import ConnectorContext
from osprey_connectors.control_system import _va_connector as va_connector
from osprey_connectors.control_system._epics_connector import EpicsSettings
from osprey_connectors.control_system._limits_validator import LimitsValidator
from osprey_connectors.control_system._va_connector import (
    DEFAULT_VA_PORT,
    VirtualAcceleratorConnector,
    fill_gateway_ports,
)
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import config_tree, serve_config
from osprey_connectors_tests._epics_fakes import (  # noqa: F401 - fixtures, used by name
    clean_epics_env,
    fake_aioca,
)

DEMO_PRESET = "demo"
PRESET_PATH = "profiles/presets/demo.yml"


def _rows(block: dict[str, Any]) -> dict[str, GatewayRow]:
    """*block*'s gateways, read as the virtual accelerator's settings read them."""
    return dict(
        EpicsSettings.from_block(block, key="control_system.connector.virtual_accelerator").gateways
    )


def _filled(block: dict[str, Any]) -> dict[str, GatewayRow]:
    """*block*'s gateway rows with their unset ports filled."""
    filled = fill_gateway_ports(_rows(block))
    assert all(isinstance(row, GatewayRow) for row in filled.values())
    return {role: row for role, row in filled.items() if isinstance(row, GatewayRow)}


def _preset_source(preset: str) -> str:
    return (Path(osprey.profiles.__file__).parent / "presets" / f"{preset}.yml").read_text(
        encoding="utf-8"
    )


def _rendered_va_block() -> dict[str, Any]:
    """The preset's resolved virtual_accelerator connector block."""
    profile, _profile_dir = resolve_build_profile(None, DEMO_PRESET)
    config = _expand_dotted(profile.config)
    return config["control_system"]["connector"]["virtual_accelerator"]


@pytest.fixture
def deployed_va_port():
    """Serve a config whose ``services.virtual_accelerator.port`` is the given port.

    ``None`` serves a config that names no port.
    """

    def _set(port: Any | None) -> None:
        values = {} if port is None else {"services.virtual_accelerator.port": port}
        serve_config(config_of(config_tree(values), environ={}))

    return _set


def _no_config_read(*_args: Any, **_kwargs: Any) -> None:
    """Stands in for ``current_config``: a test that reads no config fails if it does."""
    pytest.fail("resolved the service port, reading the config, where nothing needed it")


# ---------------------------------------------------------------------------
# The shipped preset
# ---------------------------------------------------------------------------


def test_preset_writes_no_hardcoded_va_gateway_port() -> None:
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


def test_preset_documents_the_port_override() -> None:
    """A project reaching a VA it does not deploy needs to see how."""
    text = _preset_source(DEMO_PRESET)

    assert "services.virtual_accelerator.port" in text, (
        "the preset no longer names where the gateway port comes from"
    )
    assert "Set a port on a" in text, (
        "the gateway port override is undocumented — a split host/container "
        "setup has no way to see that `port:` still works"
    )


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


def test_unset_gateway_port_follows_the_deployed_va_port(deployed_va_port) -> None:
    """The whole point: moving the VA service's port moves the connector."""
    deployed_va_port(5074)

    filled = _filled(
        {
            "gateways": {
                "read_only": {"address": "localhost", "use_name_server": True},
                "write_access": {"address": "localhost", "use_name_server": True},
            }
        }
    )

    assert filled["read_only"].port == 5074
    assert filled["write_access"].port == 5074


def test_rendered_preset_follows_a_post_render_port_edit(deployed_va_port) -> None:
    """The as-shipped config, fed the edited service port, follows it."""
    deployed_va_port(15064)

    filled = _filled(_rendered_va_block())

    assert filled["read_only"].port == 15064
    assert filled["write_access"].port == 15064
    # Everything else about the gateway survives the fill.
    assert filled["read_only"].use_name_server is True


def test_defaults_to_5064_when_the_service_port_is_unset(deployed_va_port) -> None:
    """With neither the gateway nor the service naming a port, 5064 stands."""
    deployed_va_port(None)

    filled = _filled({"gateways": {"read_only": {"address": "localhost"}}})

    assert filled["read_only"].port == DEFAULT_VA_PORT


def test_unreadable_config_falls_back_to_5064(monkeypatch, tmp_path) -> None:
    """Outside a project context the connector falls back to the default port."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "missing.yml"))

    filled = _filled({"gateways": {"read_only": {"address": "localhost"}}})

    assert filled["read_only"].port == DEFAULT_VA_PORT


# ---------------------------------------------------------------------------
# An explicit gateway port wins
# ---------------------------------------------------------------------------


def test_explicit_gateway_port_wins(deployed_va_port) -> None:
    """A project pointed at a VA it does not deploy keeps its own port."""
    deployed_va_port(15064)

    filled = _filled(
        {
            "gateways": {
                "read_only": {"address": "va.example.org", "port": 5074},
                "write_access": {"address": "va.example.org", "port": 5084},
            }
        }
    )

    assert filled["read_only"].port == 5074
    assert filled["write_access"].port == 5084


def test_explicit_and_derived_ports_mix_per_gateway(deployed_va_port) -> None:
    """Filling is per gateway — one explicit port does not pin the other."""
    deployed_va_port(15064)

    filled = _filled(
        {
            "gateways": {
                "read_only": {"address": "localhost"},
                "write_access": {"address": "localhost", "port": 5084},
            }
        }
    )

    assert filled["read_only"].port == 15064
    assert filled["write_access"].port == 5084


def test_fully_explicit_config_never_reads_the_config_file(monkeypatch) -> None:
    """Nothing to fill means nothing to resolve — and the dict comes back as-is."""
    monkeypatch.setattr("osprey_connectors.config.current_config", _no_config_read)

    rows = _rows({"gateways": {"read_only": {"address": "localhost", "port": 5074}}})

    assert fill_gateway_ports(rows) == rows


# ---------------------------------------------------------------------------
# Shape tolerance and non-mutation
# ---------------------------------------------------------------------------


def test_the_callers_rows_are_not_mutated(deployed_va_port) -> None:
    """The settings hand over their rows; the fill returns new ones."""
    deployed_va_port(15064)
    rows = _rows({"gateways": {"read_only": {"address": "localhost"}}})

    fill_gateway_ports(rows)

    assert rows == {"read_only": GatewayRow(address="localhost")}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"gateways": {}},
        {"timeout_s": 5.0},
        {"gateways": None},
    ],
    ids=["no-gateways-key", "empty-gateways", "timeout-only", "null-gateways"],
)
def test_a_config_with_no_gateways_passes_through(config: dict[str, Any], monkeypatch) -> None:
    """A gateway-less config is the EPICS connector's problem, not the fill's."""
    monkeypatch.setattr("osprey_connectors.config.current_config", _no_config_read)

    assert fill_gateway_ports(_rows(config)) == {}


# ---------------------------------------------------------------------------
# The fill happens on the connector's config-load path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_fills_the_port_before_epics_sees_it(deployed_va_port, monkeypatch) -> None:
    """The derived port reaches EPICSConnector._open, which configures CA.

    The real ``EPICSConnector._open`` is stubbed: it imports aioca and
    sets up a CA context against a soft-IOC that is not running here.
    """
    deployed_va_port(15064)
    captured: dict[str, Any] = {}

    async def fake_epics_open(_self, config):
        captured["config"] = config

    monkeypatch.setattr(va_connector.EPICSConnector, "_open", fake_epics_open, raising=True)
    # connect() builds the limits validator before _open; this test is about the
    # port, so the validator reads no config here.
    monkeypatch.setattr(LimitsValidator, "from_config", classmethod(lambda cls, **_kw: None))

    connector = VirtualAcceleratorConnector()
    await connector.connect(
        ConnectorContext(
            settings={"timeout_s": 5.0, "gateways": {"read_only": {"address": "localhost"}}}
        )
    )

    assert captured["config"].gateways["read_only"].port == 15064
    assert captured["config"].timeout_s == 5.0


@pytest.mark.asyncio
@pytest.mark.usefixtures("clean_epics_env", "fake_aioca")
async def test_plain_epics_connector_does_not_follow_the_va_service_port() -> None:
    """A production EPICS gateway is external infrastructure — it must not move.

    Guards the seam: the fill lives on the VA subclass, so an ``epics`` config
    that omits a port keeps the EPICS connector's own default even when the
    project also deploys a VA on some other port.
    """
    from osprey_connectors.control_system._epics_connector import EPICSConnector

    serve_config(config_of(config_tree({"services.virtual_accelerator.port": 15064}), environ={}))

    connector = EPICSConnector()
    await connector.connect(
        ConnectorContext(settings={"gateways": {"read_only": {"address": "gw.example.org"}}})
    )

    # 5064 is the EPICS connector's own generic default, not the VA's port.
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064", (
        "the plain EPICS connector followed services.virtual_accelerator.port"
    )
