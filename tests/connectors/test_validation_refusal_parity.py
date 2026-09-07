"""A validation error that is not a limits violation refuses the write.

``LimitsValidator.validate()`` answers a write in one of two ways: it raises a
:class:`ChannelLimitsViolationError`, which is the limits contract speaking and
propagates to the caller, or it raises something else — which means the check
could not be MADE. An unmade check is not permission to write.

The EPICS connector already fails closed there
(``tests/connectors/test_write_fail_closed.py``). This file holds the other
three shipped connectors to the same word, through the same seams the parity
suite uses: the mock's store, DOOCS' ``doocs4py.set``, TANGO's
``DeviceProxy.write_attribute``. Each one must answer REFUSED and must not
have touched its client.
"""

import sys
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.mock_connector import MockConnector

_DOOCS_LIMITS_PATCH = "osprey.connectors.control_system.doocs_connector.LimitsValidator.from_config"
_DOOCS_TZ_PATCH = "osprey.connectors.control_system.doocs_connector.get_facility_timezone"
_TANGO_LIMITS_PATCH = "osprey.connectors.control_system.tango_connector.LimitsValidator.from_config"
_TANGO_TZ_PATCH = "osprey.connectors.control_system.tango_connector.get_facility_timezone"

VALUE_SENT = 5.0


def _writes_enabled(key, default=None):
    """Enable writes and answer every other config lookup with its default."""
    if key == "control_system.writes_enabled":
        return True
    return default


def _broken_validator() -> MagicMock:
    """A validator that cannot answer — it raises, and not a limits violation."""
    validator = MagicMock()
    validator.validate = MagicMock(side_effect=RuntimeError("validator is broken"))
    return validator


class WriteRun:
    """One connector's answer, and the client call it must not have made."""

    def __init__(self, result, client_call: MagicMock):
        self.result = result
        self.client_call = client_call


async def _run_mock(monkeypatch) -> WriteRun:
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)
    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})
    connector._limits_validator = _broken_validator()

    put = MagicMock()
    monkeypatch.setattr(connector, "_put", put)

    result = await connector.write_channel("TEST:CHANNEL:SP", VALUE_SENT)
    await connector.disconnect()

    return WriteRun(result=result, client_call=put)


async def _run_doocs(monkeypatch) -> WriteRun:
    doocs4py = MagicMock()
    doocs4py.__version__ = "2.0.0"
    doocs4py.names.return_value = [("FACILITY", "XFEL")]

    with (
        patch.dict(sys.modules, {"doocs4py": doocs4py}),
        patch(_DOOCS_LIMITS_PATCH, return_value=None),
        patch(_DOOCS_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})
        conn._limits_validator = _broken_validator()
        result = await conn.write_channel("FAC/DEV/LOC/PROP", VALUE_SENT)
        await conn.disconnect()

    return WriteRun(result=result, client_call=doocs4py.set)


async def _run_tango(monkeypatch) -> WriteRun:
    proxy = MagicMock()
    tango = MagicMock()
    tango.__version__ = "10.0.0"
    tango.DeviceProxy.return_value = proxy
    database = MagicMock()
    database.get_info.return_value = "TANGO Database sys/database/2"
    tango.Database.return_value = database

    with (
        patch.dict(sys.modules, {"tango": tango}),
        patch(_TANGO_LIMITS_PATCH, return_value=None),
        patch(_TANGO_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})
        conn._limits_validator = _broken_validator()
        result = await conn.write_channel("sr/power_supply/ps01/Current", VALUE_SENT)
        await conn.disconnect()

    return WriteRun(result=result, client_call=proxy.write_attribute)


_DRIVERS = {"mock": _run_mock, "doocs": _run_doocs, "tango": _run_tango}


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_name", list(_DRIVERS), ids=list(_DRIVERS))
async def test_a_validation_error_refuses_the_write(connector_name, monkeypatch):
    run = await _DRIVERS[connector_name](monkeypatch)

    assert run.result.outcome is WriteOutcome.REFUSED
    assert run.result.refusal_reason == "VALIDATION_ERROR"
    assert "validator is broken" in run.result.error_message


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_name", list(_DRIVERS), ids=list(_DRIVERS))
async def test_a_validation_error_sends_nothing_to_the_control_system(connector_name, monkeypatch):
    run = await _DRIVERS[connector_name](monkeypatch)

    run.client_call.assert_not_called()
