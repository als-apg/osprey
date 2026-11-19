"""Each connector builds its limits validator for its own connector type.

A deployment may run its simulator ``mode: optional`` while its live machine
runs ``mode: exclusive``. The block that decides is
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
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from osprey_connectors.connector_types import DOOCS, EPICS, LIVE_STANDIN, VIRTUAL_ACCELERATOR
from osprey_connectors.control_system import ConnectorContext
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import config_tree, serve_config
from tests.facility.served_tree import in_process_config, served_tree

DEPLOYMENT_WIDE_MODE_KEY = "control_system.limits_checking.mode"


def _type_mode_key(connector_type: str) -> str:
    """The per-type spelling of the mode key for *connector_type*."""
    return f"control_system.connector.{connector_type}.limits_checking.mode"


def _limits_db(tmp_path: Path) -> Path:
    """A one-channel limits database, so a validator loads rather than failsafes."""
    db_file = tmp_path / "limits.json"
    db_file.write_text(json.dumps({"FOO": {"writable": True, "min_value": 0.0, "max_value": 10.0}}))
    return db_file


def _permissive_simulators_section() -> dict[str, Any]:
    """Deployment-wide strict; the simulator, stand-in and DOOCS blocks relaxed.

    The configuration the feature exists for. ``epics`` deliberately carries no
    ``limits_checking`` block: the live machine inherits the deployment-wide
    refusal, and every other type states its own relaxation.
    """
    return {
        "type": EPICS,
        "limits_checking": {"enabled": True, "mode": "exclusive"},
        "connector": {
            EPICS: {"gateway_address": "live.example"},
            VIRTUAL_ACCELERATOR: {"limits_checking": {"enabled": True, "mode": "optional"}},
            LIVE_STANDIN: {"limits_checking": {"enabled": True, "mode": "optional"}},
            DOOCS: {"limits_checking": {"enabled": True, "mode": "optional"}},
        },
    }


def _patch_config(section: dict[str, Any], db_path: Path) -> None:
    """Serve one ``control_system:`` section with the limits database path set in it.

    The posture is read from the whole section (the resolvers walk it) while the
    database path is deployment-wide — one file per deployment, so a per-type
    block changes policy and never the data.
    """
    tree = config_tree(
        {"control_system.limits_checking.database_path": str(db_path)},
        base={"control_system": section},
    )
    serve_config(config_of(tree, environ={}))


def _posture(connector) -> tuple[Any, Any]:
    """The limits mode the connector's validator was built with."""
    policy = connector._limits_validator.policy
    return policy["mode"], policy["mode_key"]


# ---------------------------------------------------------------------------
# Mock connector
# ---------------------------------------------------------------------------


async def _connected_mock(tmp_path, section, db_path, connector_type):
    from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector

    _patch_config(section, db_path)
    connector = VAInProcessConnector()
    await connector.connect(
        ConnectorContext(
            connector_type=connector_type,
            settings=in_process_config(served_tree(tmp_path / "served"), response_delay_ms=0),
        )
    )
    return connector


class TestVAInProcessConnectorPosture:
    """``VAInProcessConnector.connect()`` asks about the type the factory stamped."""

    @pytest.mark.asyncio
    async def test_stamped_type_reads_its_own_block(self, tmp_path):
        connector = await _connected_mock(
            tmp_path,
            _permissive_simulators_section(),
            _limits_db(tmp_path),
            VIRTUAL_ACCELERATOR,
        )

        assert _posture(connector) == ("optional", _type_mode_key(VIRTUAL_ACCELERATOR))

    @pytest.mark.asyncio
    async def test_unstamped_connector_reads_the_deployment_wide_block(self, tmp_path):
        """Parity: a connector built outside the factory behaves as it always has."""
        connector = await _connected_mock(
            tmp_path, _permissive_simulators_section(), _limits_db(tmp_path), None
        )

        assert _posture(connector) == ("exclusive", DEPLOYMENT_WIDE_MODE_KEY)

    @pytest.mark.asyncio
    async def test_type_without_a_block_inherits_deployment_wide(self, tmp_path):
        """``epics`` states no limits block, so the deployment-wide one answers."""
        connector = await _connected_mock(
            tmp_path, _permissive_simulators_section(), _limits_db(tmp_path), EPICS
        )

        assert _posture(connector) == ("exclusive", DEPLOYMENT_WIDE_MODE_KEY)

    @pytest.mark.asyncio
    async def test_refusal_names_the_key_that_answered(self, tmp_path):
        """The live machine's refusal quotes its own key, not the simulator's.

        The end the posture exists for: an operator reading the refusal is sent
        to the line they can actually edit.
        """
        connector = await _connected_mock(
            tmp_path, _permissive_simulators_section(), _limits_db(tmp_path), EPICS
        )

        with pytest.raises(ChannelLimitsViolationError) as excinfo:
            connector._limits_validator.validate("NOT:IN:DB", 1.0)

        assert DEPLOYMENT_WIDE_MODE_KEY in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_the_simulator_allows_an_unlisted_channel(self, tmp_path):
        """The simulator's half of the same deployment; the live refusal is the test above."""
        connector = await _connected_mock(
            tmp_path,
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
    """Start with no EPICS_* variables; the autouse ``restore_environ`` undoes connect()'s writes."""
    for var in _EPICS_VARS:
        monkeypatch.delenv(var, raising=False)


def _gateways() -> dict[str, Any]:
    return {"read_only": {"address": "ro.example.com", "port": 5064}}


class TestEPICSConnectorPosture:
    """The EPICS connector, and everything that inherits its ``connect()``."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_stand_in_reads_its_own_block(self, tmp_path):
        """``live_standin`` is a type of its own, served by this same connector."""
        from osprey_connectors.control_system._epics_connector import EPICSConnector

        _patch_config(_permissive_simulators_section(), _limits_db(tmp_path))
        connector = EPICSConnector()
        await connector.connect(
            ConnectorContext(connector_type=LIVE_STANDIN, settings={"gateways": _gateways()})
        )

        assert _posture(connector) == ("optional", _type_mode_key(LIVE_STANDIN))

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_live_type_inherits_deployment_wide(self, tmp_path):
        from osprey_connectors.control_system._epics_connector import EPICSConnector

        _patch_config(_permissive_simulators_section(), _limits_db(tmp_path))
        connector = EPICSConnector()
        await connector.connect(
            ConnectorContext(connector_type=EPICS, settings={"gateways": _gateways()})
        )

        assert _posture(connector) == ("exclusive", DEPLOYMENT_WIDE_MODE_KEY)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_unstamped_connector_reads_the_deployment_wide_block(self, tmp_path):
        from osprey_connectors.control_system._epics_connector import EPICSConnector

        _patch_config(_permissive_simulators_section(), _limits_db(tmp_path))
        connector = EPICSConnector()
        await connector.connect(ConnectorContext(settings={"gateways": _gateways()}))

        assert _posture(connector) == ("exclusive", DEPLOYMENT_WIDE_MODE_KEY)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("clean_epics_env")
    async def test_va_connector_inherits_the_wiring_through_super_connect(self, tmp_path):
        """``VirtualAcceleratorConnector`` builds no validator of its own."""
        from osprey_connectors.control_system._va_connector import VirtualAcceleratorConnector

        _patch_config(_permissive_simulators_section(), _limits_db(tmp_path))
        connector = VirtualAcceleratorConnector()
        await connector.connect(
            ConnectorContext(connector_type=VIRTUAL_ACCELERATOR, settings={"gateways": _gateways()})
        )

        assert _posture(connector) == ("optional", _type_mode_key(VIRTUAL_ACCELERATOR))


# ---------------------------------------------------------------------------
# DOOCS connector
# ---------------------------------------------------------------------------


def _mock_doocs4py() -> MagicMock:
    """A doocs4py stand-in, so ``connect()`` needs no DOOCS installation."""
    module = MagicMock()
    module.__version__ = "2.0.0"
    module.names.return_value = [("FACILITY", "XFEL")]
    return module


async def _connected_doocs(section, db_path, connector_type):
    _patch_config(section, db_path)
    with patch.dict(sys.modules, {"doocs4py": _mock_doocs4py()}):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        connector = DOOCSConnector()
        await connector.connect(ConnectorContext(connector_type=connector_type))
    return connector


class TestDOOCSConnectorPosture:
    @pytest.mark.asyncio
    async def test_stamped_type_reads_its_own_block(self, tmp_path):
        connector = await _connected_doocs(
            _permissive_simulators_section(), _limits_db(tmp_path), DOOCS
        )

        assert _posture(connector) == ("optional", _type_mode_key(DOOCS))

    @pytest.mark.asyncio
    async def test_unstamped_connector_reads_the_deployment_wide_block(self, tmp_path):
        connector = await _connected_doocs(
            _permissive_simulators_section(), _limits_db(tmp_path), None
        )

        assert _posture(connector) == ("exclusive", DEPLOYMENT_WIDE_MODE_KEY)
