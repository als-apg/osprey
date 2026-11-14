"""A batch means the same on the simulator and on TANGO as on Channel Access.

Every write is checked before any is sent; one refusal sends nothing and
answers every channel. The EPICS connector and the base class are held to it
in ``packages/osprey-connectors/tests/osprey_connectors_tests/test_batch_write.py``;
this file holds the connectors that refuse on their own before the limits.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._write_fakes import arm_switch, writes_enabled_config
from tests.facility.served_tree import in_process_config, served_tree

A, B, STEPPED = "SIM:A:SP", "SIM:B:SP", "SIM:STEPPED:SP"


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it."""
    return arm_switch(monkeypatch, tmp_path / "armed-data")


def _validator() -> LimitsValidator:
    limits = {
        name: ChannelLimitsConfig(channel_address=name, writable=True, max_value=10.0)
        for name in (A, B)
    }
    limits[STEPPED] = ChannelLimitsConfig(
        channel_address=STEPPED, writable=True, max_value=10.0, max_step=3.0
    )
    return LimitsValidator(limits, {"mode": "exclusive"}, {})


@pytest.fixture
async def simulator(monkeypatch, tmp_path):
    monkeypatch.setattr("osprey_connectors.config.get_config_value", writes_enabled_config)
    connector = VAInProcessConnector()
    await connector.connect(
        in_process_config(served_tree(tmp_path, [A, B, STEPPED]), response_delay_ms=0)
    )
    connector._composite.set({A: 0.0, B: 0.0, STEPPED: 1.0})
    connector._limits_validator = _validator()
    yield connector
    await connector.disconnect()


def _held(connector, *names):
    return [connector.current_value_reader()(name) for name in names]


def _verdicts(results):
    return [(r.outcome, r.refusal_reason) for r in results]


class TestSimulatorBatch:
    async def test_a_batch_that_passes_is_put_and_confirmed(self, simulator):
        results = await simulator.write_multiple_channels([(A, 1.0), (B, 2.0), (STEPPED, 3.0)])

        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 3
        assert _held(simulator, A, B, STEPPED) == [1.0, 2.0, 3.0]

    async def test_one_limits_refusal_puts_nothing(self, simulator):
        results = await simulator.write_multiple_channels([(A, 1.0), (B, 99.0)])

        assert _verdicts(results) == [
            (WriteOutcome.REFUSED, "BATCH_REFUSED"),
            (WriteOutcome.REFUSED, "LIMITS"),
        ]
        assert B in results[0].error_message
        assert _held(simulator, A, B) == [0.0, 0.0]

    async def test_a_step_too_large_puts_nothing(self, simulator):
        results = await simulator.write_multiple_channels([(A, 1.0), (STEPPED, 9.0)])

        assert _verdicts(results) == [
            (WriteOutcome.REFUSED, "BATCH_REFUSED"),
            (WriteOutcome.REFUSED, "LIMITS"),
        ]
        assert "MAX_STEP_EXCEEDED" in results[1].error_message
        assert _held(simulator, A, STEPPED) == [0.0, 1.0]

    async def test_an_address_the_view_does_not_serve_puts_nothing(self, simulator):
        results = await simulator.write_multiple_channels([(A, 1.0), ("SIM:NOT:SERVED", 1.0)])

        assert _verdicts(results) == [
            (WriteOutcome.REFUSED, "BATCH_REFUSED"),
            (WriteOutcome.REFUSED, "CHANNEL_NOT_FOUND"),
        ]
        assert _held(simulator, A) == [0.0]

    async def test_the_step_is_read_without_a_worker_thread(self, simulator, monkeypatch):
        """The simulator holds its values in process: checking a batch offloads nothing."""
        offloaded = []

        async def no_thread(function, *args, **kwargs):
            offloaded.append(function)
            return function(*args, **kwargs)

        monkeypatch.setattr("asyncio.to_thread", no_thread)

        await simulator._check_batch([(A, 1.0), (STEPPED, 2.0)])

        assert offloaded == []


class TestTangoBatch:
    async def test_an_address_that_names_no_attribute_refuses_the_batch(self):
        proxy = MagicMock()
        tango = MagicMock()
        tango.__version__ = "10.0.0"
        tango.DeviceProxy.return_value = proxy
        tango.Database.return_value.get_info.return_value = "TANGO Database sys/database/2"

        with (
            patch.dict(sys.modules, {"tango": tango}),
            patch(
                "osprey_connectors.control_system._tango_connector.LimitsValidator.from_config",
                return_value=None,
            ),
            patch("osprey_connectors.config.get_config_value", side_effect=writes_enabled_config),
        ):
            from osprey_connectors.control_system._tango_connector import TangoConnector

            connector = TangoConnector()
            await connector.connect({})
            results = await connector.write_multiple_channels(
                [("sys/tg_test/1/ampli", 1.0), ("not-an-address", 2.0)]
            )
            await connector.disconnect()

        assert _verdicts(results) == [
            (WriteOutcome.REFUSED, "BATCH_REFUSED"),
            (WriteOutcome.REFUSED, "VALIDATION_ERROR"),
        ]
        proxy.write_attribute.assert_not_called()
