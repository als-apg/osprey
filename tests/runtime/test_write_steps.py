"""The steps every write path shares: ``pin_verdict`` and ``checked_write``.

``pin_verdict`` is a pure comparison, so it is driven with plain records.
``checked_write`` is driven over a minimal in-file connector that records the
order of the net, the attempt observer and the connector call. The last class
pins ``ControlSession.moved()`` on top of ``pin_verdict``: a session opened
without a record has not moved while there is still none to read, even when the
record file is there but unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from osprey.runtime.write_steps import PinVerdict, checked_write, pin_verdict
from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)

# ---------------------------------------------------------------------------
# pin_verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Record:
    target: str
    generation: int


class TestPinVerdict:
    def test_the_bound_target_and_generation_match(self):
        assert pin_verdict(_Record("va", 3), ("va", 3)) is PinVerdict.MATCH

    @pytest.mark.parametrize(("target", "generation"), [("va", 4), ("live", 3), ("live", 4)])
    def test_another_target_or_generation_moved(self, target, generation):
        assert pin_verdict(_Record(target, generation), ("va", 3)) is PinVerdict.MOVED

    @pytest.mark.parametrize("key", [("va", 3), ("va", None), None])
    def test_no_record_is_unavailable_whatever_the_key(self, key):
        assert pin_verdict(None, key) is PinVerdict.UNAVAILABLE

    def test_a_record_appearing_for_a_recordless_binding_moved(self):
        assert pin_verdict(_Record("va", 0), None) is PinVerdict.MOVED

    def test_an_unknown_stamped_generation_never_matches(self):
        assert pin_verdict(_Record("va", 3), ("va", None)) is PinVerdict.MOVED

    def test_the_verdicts_are_the_contract_words(self):
        assert {verdict.value for verdict in PinVerdict} == {"match", "moved", "unavailable"}


# ---------------------------------------------------------------------------
# checked_write
# ---------------------------------------------------------------------------


def _result(outcome=WriteOutcome.CONFIRMED, **overrides) -> ChannelWriteResult:
    fields: dict[str, Any] = {
        "channel_address": "TEST:PV",
        "value_written": 42.0,
        "outcome": outcome,
    }
    if outcome in (WriteOutcome.FAILED, WriteOutcome.UNCONFIRMED):
        # A result that did not land says why; the base class refuses one that does not.
        fields["error_message"] = f"the write came back {outcome.value}"
    fields.update(overrides)
    return ChannelWriteResult(**fields)


class _Connector(ControlSystemConnector):
    """Returns a canned result (or raises a canned error) and logs every step."""

    def __init__(self, events: list, result=None, exc: Exception | None = None):
        self.events = events
        self.result = result if result is not None else _result()
        self.exc = exc
        self.calls: list[tuple[str, Any, dict]] = []

    @property
    def _writes_enabled(self) -> bool:
        return True

    def current_value_reader(self):
        return self._reader

    @staticmethod
    def _reader(address):
        return 0.0

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if confirm is not None:
            kwargs["confirm"] = confirm
        self.events.append(("connector", channel_address))
        self.calls.append((channel_address, value, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result

    async def connect(self, config: dict[str, Any]) -> None: ...
    async def disconnect(self) -> None: ...
    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def subscribe(self, channel_address, callback, *, on_connection=None) -> str:
        raise NotImplementedError

    async def unsubscribe(self, subscription_id: str) -> None: ...
    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        raise NotImplementedError

    async def validate_channel(self, channel_address: str) -> bool:
        raise NotImplementedError


def _violation(address="TEST:PV", value=999.0) -> ChannelLimitsViolationError:
    return ChannelLimitsViolationError(
        channel_address=address,
        value=value,
        violation_type="MAX_EXCEEDED",
        violation_reason="above the channel's maximum",
        max_value=100.0,
    )


class _Recorder:
    """A net and an attempt observer writing to one shared event list."""

    def __init__(self, events: list, refuse: bool = False):
        self.events = events
        self.refuse = refuse
        self.read_current = None

    def net(self, address, value, *, read_current=None):
        self.events.append(("net", address))
        self.read_current = read_current
        if self.refuse:
            raise _violation(address, value)

    def on_attempt(self, address):
        self.events.append(("attempt", address))


async def _write(connector, recorder, **kwargs):
    return await checked_write(
        connector,
        "TEST:PV",
        42.0,
        net=recorder.net if recorder else None,
        on_attempt=recorder.on_attempt if recorder else None,
        **kwargs,
    )


class TestCheckedWriteOrder:
    async def test_net_then_attempt_then_connector(self):
        events: list = []
        connector = _Connector(events)
        recorder = _Recorder(events)

        result = await _write(connector, recorder)

        assert result.outcome == WriteOutcome.CONFIRMED
        assert events == [("net", "TEST:PV"), ("attempt", "TEST:PV"), ("connector", "TEST:PV")]

    async def test_the_net_gets_the_connectors_own_reader(self):
        events: list = []
        connector = _Connector(events)
        recorder = _Recorder(events)

        await _write(connector, recorder)

        assert recorder.read_current is connector._reader

    async def test_a_net_refusal_is_raw_and_nothing_is_observed_or_sent(self):
        events: list = []
        connector = _Connector(events)
        recorder = _Recorder(events, refuse=True)

        with pytest.raises(ChannelLimitsViolationError) as excinfo:
            await _write(connector, recorder)

        assert excinfo.value.attempted_value == 42.0
        assert events == [("net", "TEST:PV")]
        assert connector.calls == []

    async def test_no_net_and_no_observer_writes_directly(self):
        events: list = []
        connector = _Connector(events)

        result = await _write(connector, None)

        assert result.outcome == WriteOutcome.CONFIRMED
        assert events == [("connector", "TEST:PV")]

    async def test_the_attempt_is_observed_even_when_the_write_then_fails(self):
        events: list = []
        connector = _Connector(events, result=_result(WriteOutcome.MISMATCH))
        recorder = _Recorder(events)

        with pytest.raises(ChannelWriteFailedError):
            await _write(connector, recorder)

        assert ("attempt", "TEST:PV") in events


class TestCheckedWriteArguments:
    async def test_omitted_confirm_and_timeout_are_not_forwarded(self):
        connector = _Connector([])

        await _write(connector, None)

        assert connector.calls == [("TEST:PV", 42.0, {})]

    async def test_confirm_and_timeout_are_forwarded(self):
        connector = _Connector([], result=_result(WriteOutcome.UNREQUESTED))

        result = await _write(connector, None, confirm=False, timeout=2.5)

        assert result.outcome == WriteOutcome.UNREQUESTED
        assert connector.calls == [("TEST:PV", 42.0, {"confirm": False, "timeout": 2.5})]

    async def test_the_call_goes_through_the_checked_door(self, monkeypatch):
        connector = _Connector([])
        doors: list[str] = []
        checked = _Connector.write_channel_checked

        async def spy(self, *args, **kwargs):
            doors.append("write_channel_checked")
            return await checked(self, *args, **kwargs)

        monkeypatch.setattr(_Connector, "write_channel_checked", spy)

        await _write(connector, None, timeout=1.0)

        assert doors == ["write_channel_checked"]
        assert connector.calls == [("TEST:PV", 42.0, {"timeout": 1.0})]


class TestCheckedWriteDenialContract:
    @pytest.mark.parametrize(
        "outcome", [WriteOutcome.FAILED, WriteOutcome.MISMATCH, WriteOutcome.UNCONFIRMED]
    )
    async def test_sent_but_unverified_raises_failed(self, outcome):
        connector = _Connector([], result=_result(outcome))

        with pytest.raises(ChannelWriteFailedError) as excinfo:
            await _write(connector, None)

        assert excinfo.value.outcome == outcome

    async def test_a_refusal_raises_blocked_with_its_reason(self):
        connector = _Connector(
            [],
            result=_result(
                WriteOutcome.REFUSED,
                refusal_reason="WRITES_DISABLED",
                error_message="writes are disabled",
            ),
        )

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await _write(connector, None)

        assert excinfo.value.reason == "WRITES_DISABLED"

    async def test_transport_errors_propagate_unchanged(self):
        connector = _Connector([], exc=TimeoutError("no answer"))

        with pytest.raises(TimeoutError):
            await _write(connector, None)

    async def test_connector_limits_become_blocked_with_the_violation_as_cause(self):
        violation = _violation()
        connector = _Connector([], exc=violation)

        with pytest.raises(ChannelWriteBlockedError) as excinfo:
            await _write(connector, None)

        assert excinfo.value.reason == "LIMITS"
        assert excinfo.value.__cause__ is violation
