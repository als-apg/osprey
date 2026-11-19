"""The base class's write pipeline: one for every connector, and what it changed.

``write_channel``, ``write_multiple_channels`` and ``write_channel_checked`` are
final; a connector writes through ``_send`` and the protected hooks. These tests
pin what the one pipeline decides that the four hand-written copies decided
differently, or not at all:

- a connector whose own ``_open`` loads no limits validator is checked all
  the same, against the one ``connect()`` built from the loaded config for its
  type, and a validator that cannot be loaded fails the connect;
- a DOOCS or TANGO limits check that outlives its bound refuses the write with
  nothing sent, since the check and the set are separate steps;
- a single TANGO write to an address that names no device and attribute is
  refused, as a batch refuses it, rather than raising ``ValueError``;
- a single write the limits refuse is a ``LIMITS`` result carrying its
  violation, as a batch's is, rather than a raised violation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from osprey_connectors.control_system._base import (
    ChannelValue,
    ConnectorContext,
    ControlSystemConnector,
    SendReport,
    WriteOutcome,
)
from osprey_connectors.control_system._limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.refusals import RefusalReason
from osprey_connectors_tests._stall import Stall
from osprey_connectors_tests._write_fakes import (
    ARMED_TARGET,
    arm_switch,
    armed_context,
    serve_writes_enabled,
)

_FROM_CONFIG = "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config"


@pytest.fixture(autouse=True)
def _armed(monkeypatch, tmp_path):
    """A write tier that allows and the simulator's switch on: writes reach the pipeline."""
    arm_switch(monkeypatch, tmp_path / "armed-data")
    serve_writes_enabled()


def _validator(**channels: dict[str, Any]) -> LimitsValidator:
    limits = {
        name: ChannelLimitsConfig(address=name, writable=True, **bounds)
        for name, bounds in channels.items()
    }
    return LimitsValidator(limits, {"mode": "exclusive"}, {})


class _Sender(ControlSystemConnector):
    """A connector whose ``_open`` loads no validator: it records each send and echoes it back."""

    def __init__(self, report: SendReport | None = None) -> None:
        self.sent: list[tuple[str, Any]] = []
        self._report = report or SendReport()

    async def _open(self, settings: Any) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, address: str, *, timeout_s: float | None = None):  # noqa: ARG002 - the read signature
        return ChannelValue(
            address=address, value=dict(self.sent)[address], sampled_at=datetime.now(UTC)
        )

    async def _send(self, address: str, value: Any, *, timeout_s: float, wait: bool) -> SendReport:  # noqa: ARG002 - the send hook's signature
        self.sent.append((address, value))
        return self._report

    async def subscribe(self, address, callback, *, on_connection=None):
        raise NotImplementedError

    async def unsubscribe(self, subscription_id):
        raise NotImplementedError


# -- the write methods are final ---------------------------------------------


@pytest.mark.parametrize(
    "name", ["write_channel", "write_multiple_channels", "write_channel_checked"]
)
def test_a_subclass_defining_a_write_method_is_refused(name):
    async def method(self, *args, **kwargs):  # noqa: ARG001 - stands in for any override
        raise AssertionError("never reached")

    with pytest.raises(TypeError, match=name):
        type("_Overrider", (_Sender,), {name: method})


# -- a connector that loads no validator of its own is checked ---------------


async def test_a_connector_with_no_validator_is_checked_against_the_loaded_config():
    """``connect()`` loads the validator once; the write is judged by it, not by a reload."""
    connector = _Sender()
    loaded = _validator(**{"SR:SP": {"max_value": 10.0}})

    with patch(_FROM_CONFIG, return_value=loaded) as from_config:
        await connector.connect(armed_context(connector_type="epics"))
        result = await connector.write_channel("SR:SP", 99.0)

    from_config.assert_called_once_with(connector_type="epics")
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason is RefusalReason.LIMITS
    assert result.limits_violation is not None
    assert result.limits_violation.kind == "MAX_EXCEEDED"
    assert connector.sent == []


async def test_a_batch_with_no_validator_is_checked_against_the_loaded_config():
    connector = _Sender()
    loaded = _validator(**{"A:SP": {"max_value": 10.0}, "B:SP": {"max_value": 10.0}})

    with patch(_FROM_CONFIG, return_value=loaded):
        await connector.connect(armed_context())
    results = await connector.write_multiple_channels([("A:SP", 1.0), ("B:SP", 99.0)])

    assert [r.refusal_reason for r in results] == [
        RefusalReason.BATCH_REFUSED,
        RefusalReason.LIMITS,
    ]
    assert connector.sent == []


async def test_a_validator_that_cannot_be_loaded_fails_the_connect(monkeypatch):
    """The connect raises before it stores anything: the connector names no
    machine, so the write it is then asked for is refused with nothing sent."""
    monkeypatch.setattr(ControlSystemConnector, "_context", ConnectorContext())
    connector = _Sender()

    with (
        patch(_FROM_CONFIG, side_effect=ValueError("limits block unreadable")),
        pytest.raises(ValueError, match="limits block unreadable"),
    ):
        await connector.connect(
            ConnectorContext(connector_type="epics", control_target=ARMED_TARGET)
        )

    assert connector.context == ConnectorContext()
    result = await connector.write_channel("SR:SP", 1.0)
    assert result.outcome is WriteOutcome.REFUSED
    assert connector.sent == []


async def test_a_validator_hook_that_raises_refuses_the_write():
    class _Unloadable(_Sender):
        def _validator(self) -> LimitsValidator | None:
            raise ValueError("limits block unreadable")

    connector = _Unloadable()

    result = await connector.write_channel("SR:SP", 1.0)

    assert result.refusal_reason is RefusalReason.VALIDATION_ERROR
    assert "limits block unreadable" in result.error_message
    assert connector.sent == []


async def test_limits_checking_off_writes_and_confirms():
    """Limits off: ``connect()`` keeps no validator, and no write loads one."""
    connector = _Sender()

    with patch(_FROM_CONFIG, return_value=None) as from_config:
        await connector.connect(armed_context())
        result = await connector.write_channel("SR:SP", 1.0)

    from_config.assert_called_once()
    assert result.outcome is WriteOutcome.CONFIRMED
    assert connector.sent == [("SR:SP", 1.0)]


# -- a single limits refusal is a result -------------------------------------


async def test_a_single_write_the_limits_refuse_is_a_limits_result():
    connector = _Sender()
    connector._limits_validator = _validator(**{"SR:SP": {"min_value": 0.0, "max_value": 10.0}})

    result = await connector.write_channel("SR:SP", 99.0)

    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason is RefusalReason.LIMITS
    assert result.error_message.startswith("Write to 'SR:SP' refused by the limits (MAX_EXCEEDED)")
    assert (result.limits_violation.min_value, result.limits_violation.max_value) == (0.0, 10.0)
    assert connector.sent == []


# -- the base compares with what the wire took -------------------------------


async def test_a_falsy_value_as_sent_is_what_the_read_back_is_compared_with():
    connector = _Sender(SendReport(sent=0))
    connector._limits_validator = None

    async def held_zero(address, sent, *, timeout_s):  # noqa: ARG001 - the read-back hook's signature
        return ChannelValue(address=address, value=0, sampled_at=datetime.now(UTC))

    connector._confirming_read = held_zero  # type: ignore[method-assign]
    result = await connector.write_channel("SR:SP", "0")

    assert result.outcome is WriteOutcome.CONFIRMED


# -- DOOCS and TANGO: a check that outlives its bound sends nothing ----------


@contextmanager
def _driven_through(driver: str, fake: Any) -> Iterator[None]:
    """Serve *fake* as the *driver* module, with no validator loaded at connect."""
    with patch.dict(sys.modules, {driver: fake}), patch(_FROM_CONFIG, return_value=None):
        yield


def _reading(value: Any) -> MagicMock:
    """A mock DOOCS ``EqData`` or TANGO ``DeviceAttribute`` holding *value*."""
    reading = MagicMock()
    reading.get_data.return_value = value
    reading.timestamp.get_seconds_and_microseconds_since_epoch.return_value = (1_700_000_000, 0)
    reading.value = value
    reading.time.tv_sec = 1_700_000_000
    reading.time.tv_usec = 0
    reading.quality = None
    reading.type = None
    return reading


def _stalled_validator(stall: Stall) -> MagicMock:
    validator = MagicMock()
    validator.validate = MagicMock(side_effect=stall)
    validator.resolve_confirm.return_value = True
    validator.measures_step.return_value = False
    return validator


async def test_a_doocs_check_that_outlives_its_bound_is_refused_with_nothing_sent():
    doocs = MagicMock()
    doocs.__version__ = "2.0.0"
    doocs.names.return_value = [("FACILITY", "XFEL")]
    doocs.get.return_value = _reading(1.0)
    with Stall() as check, _driven_through("doocs4py", doocs):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        connector = DOOCSConnector()
        await connector.connect(armed_context(settings={"timeout_s": 0.2}))
        connector._limits_validator = _stalled_validator(check)
        result = await connector.write_channel("FAC/DEV/LOC/PROP", 1.0)
        answered_first = not check.returned

    assert answered_first
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason is RefusalReason.VALIDATION_ERROR
    assert "within 0.2s" in result.error_message
    doocs.set.assert_not_called()


async def test_a_tango_check_that_outlives_its_bound_is_refused_with_nothing_sent():
    proxy = MagicMock()
    proxy.read_attribute.return_value = _reading(1.0)
    tango = MagicMock()
    tango.__version__ = "10.0.0"
    tango.DeviceProxy.return_value = proxy
    tango.Database.return_value.get_info.return_value = "TANGO Database sys/database/2"
    with Stall() as check, _driven_through("tango", tango):
        from osprey_connectors.control_system._tango_connector import TangoConnector

        connector = TangoConnector()
        await connector.connect(armed_context(settings={"timeout_s": 0.2}))
        connector._limits_validator = _stalled_validator(check)
        result = await connector.write_channel("sr/power_supply/ps01/Current", 1.0)
        answered_first = not check.returned

    assert answered_first
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason is RefusalReason.VALIDATION_ERROR
    assert "within 0.2s" in result.error_message
    proxy.write_attribute.assert_not_called()


# -- TANGO: an unparsable address is refused, single or batched --------------


async def test_a_single_tango_write_to_an_unparsable_address_is_refused():
    tango = MagicMock()
    tango.__version__ = "10.0.0"
    tango.Database.return_value.get_info.return_value = "TANGO Database sys/database/2"
    with _driven_through("tango", tango):
        from osprey_connectors.control_system._tango_connector import TangoConnector

        connector = TangoConnector()
        await connector.connect(armed_context())
        result = await connector.write_channel("not-a-tango-address", 1.0)

    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason is RefusalReason.VALIDATION_ERROR
    assert "not-a-tango-address" in result.error_message
    tango.DeviceProxy.assert_not_called()
