"""Each connector builds its limits validator for its own connector type.

A deployment may relax ``allow_unlisted_channels`` for its simulator while its
live machine refuses unlisted channels. The block that decides is
``control_system.connector.<type>.limits_checking``, so the connector has to ask
about the type it *is* rather than about the deployment as a whole — otherwise
the simulator's relaxation would leak onto the live machine, or the live
machine's refusal onto the simulator.

The type a connector is comes from the factory, which stamps ``_connector_type``
between construction and ``connect()``. A connector built outside the factory
carries no stamp, and an unstamped connector reads the deployment-wide block —
the same parity rule the per-type write posture follows
(``tests/connectors/test_writes_enabled.py``), so a deployment that wrote no
per-type block behaves exactly as it did before they existed.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.types import DOOCS, EPICS, LIVE_STANDIN, VIRTUAL_ACCELERATOR
from osprey.errors import ChannelLimitsViolationError

DEPLOYMENT_WIDE_ALLOW_KEY = "control_system.limits_checking.allow_unlisted_channels"


def _type_allow_key(connector_type: str) -> str:
    """The per-type spelling of the unlisted-channel key for *connector_type*."""
    return f"control_system.connector.{connector_type}.limits_checking.allow_unlisted_channels"


def _limits_db(tmp_path: Path) -> Path:
    """A one-channel limits database, so a validator loads rather than failsafes."""
    db_file = tmp_path / "limits.json"
    db_file.write_text(json.dumps({"FOO": {"min_value": 0.0, "max_value": 10.0}}))
    return db_file


def _permissive_simulators_section() -> dict[str, Any]:
    """Deployment-wide strict; the simulator, stand-in and DOOCS blocks relaxed.

    The configuration the feature exists for. ``epics`` deliberately carries no
    ``limits_checking`` block: the live machine inherits the deployment-wide
    refusal, and every other type states its own relaxation.
    """
    return {
        "type": EPICS,
        "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
        "connector": {
            EPICS: {"gateway_address": "live.example"},
            VIRTUAL_ACCELERATOR: {
                "limits_checking": {"enabled": True, "allow_unlisted_channels": True}
            },
            LIVE_STANDIN: {"limits_checking": {"enabled": True, "allow_unlisted_channels": True}},
            DOOCS: {"limits_checking": {"enabled": True, "allow_unlisted_channels": True}},
        },
    }


def _patch_config(monkeypatch, section: dict[str, Any], db_path: Path) -> None:
    """Serve one ``control_system:`` section plus the limits database path.

    The posture is read from the whole section (the resolvers walk it) while the
    database path stays deployment-wide and dotted — one file per deployment, so
    a per-type block changes policy and never the data.
    """

    def fake_get_config_value(key: str, default: Any = None) -> Any:
        if key == "control_system":
            return section
        if key == "control_system.limits_checking.database_path":
            return str(db_path)
        if key == "control_system.writes_enabled":
            return section.get("writes_enabled", default)
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)
    monkeypatch.setattr("osprey.utils.config.default_config_path", lambda: None)


def _posture(connector) -> tuple[Any, Any]:
    """The unlisted-channel policy the connector's validator was built with."""
    policy = connector._limits_validator.policy
    return policy["allow_unlisted_channels"], policy["allow_unlisted_key"]


# ---------------------------------------------------------------------------
# Mock connector
# ---------------------------------------------------------------------------


async def _connected_mock(monkeypatch, section, db_path, connector_type):
    from osprey.connectors.control_system.mock_connector import MockConnector

    _patch_config(monkeypatch, section, db_path)
    connector = MockConnector()
    connector._connector_type = connector_type
    await connector.connect({"response_delay_ms": 0})
    return connector


class TestMockConnectorPosture:
    """``MockConnector.connect()`` asks about the type the factory stamped."""

    @pytest.mark.asyncio
    async def test_stamped_type_reads_its_own_block(self, monkeypatch, tmp_path):
        connector = await _connected_mock(
            monkeypatch,
            _permissive_simulators_section(),
            _limits_db(tmp_path),
            VIRTUAL_ACCELERATOR,
        )

        assert _posture(connector) == (True, _type_allow_key(VIRTUAL_ACCELERATOR))

    @pytest.mark.asyncio
    async def test_unstamped_connector_reads_the_deployment_wide_block(self, monkeypatch, tmp_path):
        """Parity: a connector built outside the factory behaves as it always has."""
        connector = await _connected_mock(
            monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path), None
        )

        assert _posture(connector) == (False, DEPLOYMENT_WIDE_ALLOW_KEY)

    @pytest.mark.asyncio
    async def test_type_without_a_block_inherits_deployment_wide(self, monkeypatch, tmp_path):
        """``epics`` states no limits block, so the deployment-wide one answers."""
        connector = await _connected_mock(
            monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path), EPICS
        )

        assert _posture(connector) == (False, DEPLOYMENT_WIDE_ALLOW_KEY)

    @pytest.mark.asyncio
    async def test_refusal_names_the_key_that_answered(self, monkeypatch, tmp_path):
        """The live machine's refusal quotes its own key, not the simulator's.

        The end the posture exists for: an operator reading the refusal is sent
        to the line they can actually edit.
        """
        connector = await _connected_mock(
            monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path), EPICS
        )

        with pytest.raises(ChannelLimitsViolationError) as excinfo:
            connector._limits_validator.validate("NOT:IN:DB", 1.0)

        assert DEPLOYMENT_WIDE_ALLOW_KEY in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_simulator_allows_the_unlisted_channel_the_live_machine_refuses(
        self, monkeypatch, tmp_path
    ):
        """One deployment, two answers for the same channel — the point of the feature."""
        connector = await _connected_mock(
            monkeypatch,
            _permissive_simulators_section(),
            _limits_db(tmp_path),
            VIRTUAL_ACCELERATOR,
        )

        connector._limits_validator.validate("NOT:IN:DB", 1.0)  # allowed, no raise


# ---------------------------------------------------------------------------
# EPICS connector (and the VA connector that inherits its connect())
# ---------------------------------------------------------------------------

_EPICS_VARS = [
    "EPICS_CA_ADDR_LIST",
    "EPICS_CA_SERVER_PORT",
    "EPICS_CA_NAME_SERVERS",
    "EPICS_CA_AUTO_ADDR_LIST",
]


@pytest.fixture
def clean_epics_env(monkeypatch):
    """Snapshot EPICS_* env vars so connect()'s direct os.environ writes are restored."""
    for var in _EPICS_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in _EPICS_VARS:
        os.environ.pop(var, None)


def _gateways() -> dict[str, Any]:
    return {"read_only": {"address": "ro.example.com", "port": 5064}}


class TestEPICSConnectorPosture:
    """The EPICS connector, and everything that inherits its ``connect()``."""

    @pytest.mark.asyncio
    async def test_stand_in_reads_its_own_block(self, monkeypatch, tmp_path, clean_epics_env):
        """``live_standin`` is a type of its own, served by this same connector."""
        from osprey.connectors.control_system.epics_connector import EPICSConnector

        _patch_config(monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path))
        connector = EPICSConnector()
        connector._connector_type = LIVE_STANDIN
        await connector.connect({"gateways": _gateways()})

        assert _posture(connector) == (True, _type_allow_key(LIVE_STANDIN))

    @pytest.mark.asyncio
    async def test_live_type_inherits_deployment_wide(self, monkeypatch, tmp_path, clean_epics_env):
        from osprey.connectors.control_system.epics_connector import EPICSConnector

        _patch_config(monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path))
        connector = EPICSConnector()
        connector._connector_type = EPICS
        await connector.connect({"gateways": _gateways()})

        assert _posture(connector) == (False, DEPLOYMENT_WIDE_ALLOW_KEY)

    @pytest.mark.asyncio
    async def test_unstamped_connector_reads_the_deployment_wide_block(
        self, monkeypatch, tmp_path, clean_epics_env
    ):
        from osprey.connectors.control_system.epics_connector import EPICSConnector

        _patch_config(monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path))
        connector = EPICSConnector()
        await connector.connect({"gateways": _gateways()})

        assert _posture(connector) == (False, DEPLOYMENT_WIDE_ALLOW_KEY)

    @pytest.mark.asyncio
    async def test_va_connector_inherits_the_wiring_through_super_connect(
        self, monkeypatch, tmp_path, clean_epics_env
    ):
        """``VirtualAcceleratorConnector`` builds no validator of its own."""
        from osprey.connectors.control_system.va_connector import VirtualAcceleratorConnector

        _patch_config(monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path))
        connector = VirtualAcceleratorConnector()
        connector._connector_type = VIRTUAL_ACCELERATOR
        await connector.connect({"gateways": _gateways()})

        assert _posture(connector) == (True, _type_allow_key(VIRTUAL_ACCELERATOR))


# ---------------------------------------------------------------------------
# DOOCS connector
# ---------------------------------------------------------------------------


def _mock_doocs4py() -> MagicMock:
    """A doocs4py stand-in, so ``connect()`` needs no DOOCS installation."""
    module = MagicMock()
    module.__version__ = "2.0.0"
    module.names.return_value = [("FACILITY", "XFEL")]
    return module


async def _connected_doocs(monkeypatch, section, db_path, connector_type):
    _patch_config(monkeypatch, section, db_path)
    with patch.dict(sys.modules, {"doocs4py": _mock_doocs4py()}):
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector

        connector = DOOCSConnector()
        connector._connector_type = connector_type
        await connector.connect({})
    return connector


class TestDOOCSConnectorPosture:
    @pytest.mark.asyncio
    async def test_stamped_type_reads_its_own_block(self, monkeypatch, tmp_path):
        connector = await _connected_doocs(
            monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path), DOOCS
        )

        assert _posture(connector) == (True, _type_allow_key(DOOCS))

    @pytest.mark.asyncio
    async def test_unstamped_connector_reads_the_deployment_wide_block(self, monkeypatch, tmp_path):
        connector = await _connected_doocs(
            monkeypatch, _permissive_simulators_section(), _limits_db(tmp_path), None
        )

        assert _posture(connector) == (False, DEPLOYMENT_WIDE_ALLOW_KEY)
