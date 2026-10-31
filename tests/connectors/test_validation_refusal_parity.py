"""A validation error that is not a limits violation refuses the write.

``LimitsValidator.validate()`` answers a write in one of two ways: it raises a
:class:`ChannelLimitsViolationError`, which is the limits contract speaking and
propagates to the caller, or it raises something else — which means the check
could not be MADE. An unmade check is not permission to write.

The EPICS connector already fails closed there
(``packages/osprey-connectors/tests/osprey_connectors_tests/test_write_fail_closed.py``). This file holds the other
three shipped connectors to the same word, through the same seams the parity
suite uses: the mock's store, DOOCS' ``doocs4py.set``, TANGO's
``DeviceProxy.write_attribute``. Each one must answer REFUSED and must not
have touched its client.
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests.connectors._driver_fakes import driven_through
from tests.facility.served_tree import in_process_config, served_tree

VALUE_SENT = 5.0


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it.

    With :func:`writes_enabled_config` serving a write tier that allows, that is
    everything the reference monitor's verdict asks before a write reaches the
    transport; a test that disarms serves a tier that does not.
    """
    return arm_switch(monkeypatch, tmp_path / "armed-data")


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


async def _run_mock(monkeypatch, tmp_path) -> WriteRun:
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    view = served_tree(tmp_path, ["TEST:CHANNEL:SP"])
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view, response_delay_ms=0))
    connector._limits_validator = _broken_validator()

    put = MagicMock()
    monkeypatch.setattr(connector, "_put", put)

    result = await connector.write_channel("TEST:CHANNEL:SP", VALUE_SENT)
    await connector.disconnect()

    return WriteRun(result=result, client_call=put)


async def _run_doocs(_monkeypatch, _tmp_path) -> WriteRun:
    doocs4py = MagicMock()
    doocs4py.__version__ = "2.0.0"
    doocs4py.names.return_value = [("FACILITY", "XFEL")]

    with (
        driven_through("doocs4py", doocs4py),
        patch("osprey_connectors.config.get_config_value", side_effect=writes_enabled_config),
    ):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})
        conn._limits_validator = _broken_validator()
        result = await conn.write_channel("FAC/DEV/LOC/PROP", VALUE_SENT)
        await conn.disconnect()

    return WriteRun(result=result, client_call=doocs4py.set)


async def _run_tango(_monkeypatch, _tmp_path) -> WriteRun:
    proxy = MagicMock()
    tango = MagicMock()
    tango.__version__ = "10.0.0"
    tango.DeviceProxy.return_value = proxy
    database = MagicMock()
    database.get_info.return_value = "TANGO Database sys/database/2"
    tango.Database.return_value = database

    with (
        driven_through("tango", tango),
        patch("osprey_connectors.config.get_config_value", side_effect=writes_enabled_config),
    ):
        from osprey_connectors.control_system._tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})
        conn._limits_validator = _broken_validator()
        result = await conn.write_channel("sr/power_supply/ps01/Current", VALUE_SENT)
        await conn.disconnect()

    return WriteRun(result=result, client_call=proxy.write_attribute)


_DRIVERS = {"in_process": _run_mock, "doocs": _run_doocs, "tango": _run_tango}


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_name", list(_DRIVERS), ids=list(_DRIVERS))
async def test_a_validation_error_refuses_the_write_and_sends_nothing(
    connector_name, monkeypatch, tmp_path
):
    run = await _DRIVERS[connector_name](monkeypatch, tmp_path)

    assert run.result.outcome is WriteOutcome.REFUSED
    assert run.result.refusal_reason == "VALIDATION_ERROR"
    assert "validator is broken" in run.result.error_message
    run.client_call.assert_not_called()
