"""What ``write_multiple_channels`` promises: all checked first, then all sent.

Every write of a batch passes every check a single write passes before any of
them is sent. One refusal sends nothing, and every channel gets a verdict: the
refused ones their own, the others ``BATCH_REFUSED`` naming a refused one. Past
the checks every write is put and verified on its own, so one failure leaves the
others' verdicts as they are. A batch names a channel once, and the write
verdict is asked again right before the send. The base class's sequential batch
and the EPICS connector's concurrent one are held to the same words.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey_connectors import posture_store
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.errors import ChannelUnreachableError, RefusalReason
from osprey_connectors_tests._epics_fakes import (
    FakeAioca,
    FakeChannel,
    ca_connector,
    ca_reading,
    patch_writes_enabled,
    writes_enabled,  # noqa: F401 - fixture, used by name
)
from osprey_connectors_tests._fake_connector import FakeConnector

ECA_DISCONN = 192
ECA_NOWTACCESS = 376

pytestmark = pytest.mark.usefixtures("writes_enabled")


def _validator(**channels: dict[str, Any]) -> LimitsValidator:
    limits = {
        name: ChannelLimitsConfig(channel_address=name, writable=True, **bounds)
        for name, bounds in channels.items()
    }
    return LimitsValidator(limits, {"mode": "exclusive"}, {})


def _outcomes(results) -> list[tuple[str, WriteOutcome, RefusalReason | None]]:
    return [(r.channel_address, r.outcome, r.refusal_reason) for r in results]


# ---------------------------------------------------------------------------
# The base class's batch, on the in-memory connector
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake():
    connector = FakeConnector()
    await connector.connect({})
    connector._limits_validator = _validator(
        A={"max_value": 10.0}, B={"max_value": 10.0}, C={"max_value": 10.0, "max_step": 3.0}
    )
    return connector


class TestBaseBatch:
    async def test_a_batch_that_passes_writes_and_confirms_every_channel(self, fake):
        results = await fake.write_multiple_channels([("A", 1.0), ("B", 2.0), ("C", 3.0)])

        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 3
        assert fake._state == {"A": 1.0, "B": 2.0, "C": 3.0}

    async def test_one_limits_refusal_sends_nothing_and_answers_every_channel(self, fake):
        results = await fake.write_multiple_channels([("A", 1.0), ("B", 99.0), ("C", 3.0)])

        assert _outcomes(results) == [
            ("A", WriteOutcome.REFUSED, RefusalReason.BATCH_REFUSED),
            ("B", WriteOutcome.REFUSED, RefusalReason.LIMITS),
            ("C", WriteOutcome.REFUSED, RefusalReason.BATCH_REFUSED),
        ]
        assert "'B'" in results[0].error_message and "LIMITS" in results[0].error_message
        assert "99" in results[1].error_message
        assert not any(fake._state.values())

    async def test_a_refusal_after_writes_that_pass_sends_none_of_them(self, fake):
        """The sequential batch used to send A and B before C's refusal was raised."""
        await fake.write_multiple_channels([("A", 1.0), ("B", 2.0), ("C", 9.0)])

        assert fake._state.get("A", 0.0) == 0.0
        assert fake._state.get("B", 0.0) == 0.0

    async def test_every_refused_channel_carries_its_own_reason(self, fake):
        results = await fake.write_multiple_channels([("A", 99.0), ("NOT:LISTED", 1.0)])

        assert [r.refusal_reason for r in results] == [RefusalReason.LIMITS, RefusalReason.LIMITS]
        assert "NOT:LISTED" in results[1].error_message

    async def test_the_step_is_measured_against_the_value_the_channel_holds(self, fake):
        fake._state["C"] = 5.0

        refused = await fake.write_multiple_channels([("A", 1.0), ("C", 1.0)])
        allowed = await fake.write_multiple_channels([("A", 1.0), ("C", 7.0)])

        assert refused[1].refusal_reason is RefusalReason.LIMITS
        assert "MAX_STEP_EXCEEDED" in refused[1].error_message
        assert [r.outcome for r in allowed] == [WriteOutcome.CONFIRMED] * 2

    async def test_a_batch_that_names_a_channel_twice_is_refused_whole(self, fake):
        """The second write's step could only be judged on a first put that may never land."""
        results = await fake.write_multiple_channels([("C", 3.0), ("A", 1.0), ("C", 6.0)])

        assert _outcomes(results) == [
            ("C", WriteOutcome.REFUSED, RefusalReason.VALIDATION_ERROR),
            ("A", WriteOutcome.REFUSED, RefusalReason.BATCH_REFUSED),
            ("C", WriteOutcome.REFUSED, RefusalReason.VALIDATION_ERROR),
        ]
        assert "more than once" in results[0].error_message
        assert fake._state == {}

    async def test_a_repeated_channel_is_refused_without_a_confirming_policy(self, fake):
        results = await fake.write_multiple_channels([("A", 1.0), ("A", 1.0)], confirm=False)

        assert [r.refusal_reason for r in results] == [RefusalReason.VALIDATION_ERROR] * 2
        assert fake._state == {}

    async def test_a_check_that_could_not_be_made_refuses_the_batch(self, fake):
        def broken(*_args, **_kwargs):
            raise RuntimeError("limits database unreadable")

        fake._limits_validator.validate = broken
        results = await fake.write_multiple_channels([("A", 1.0), ("B", 2.0)])

        assert [r.refusal_reason for r in results] == [RefusalReason.VALIDATION_ERROR] * 2
        assert fake._state == {}

    async def test_a_failure_past_the_checks_leaves_the_other_verdicts(self, fake):
        fake.drop("B")
        results = await fake.write_multiple_channels([("A", 1.0), ("B", 2.0), ("C", 3.0)])

        assert [r.outcome for r in results] == [
            WriteOutcome.CONFIRMED,
            WriteOutcome.UNCONFIRMED,
            WriteOutcome.CONFIRMED,
        ]
        assert "B" in results[1].error_message
        assert fake._state["A"] == 1.0 and fake._state["C"] == 3.0

    async def test_a_batch_is_refused_whole_while_writes_are_off(self, fake, monkeypatch):
        patch_writes_enabled(monkeypatch, False)
        read: list[str] = []
        monkeypatch.setattr(fake, "current_value_reader", lambda: read.append)

        results = await fake.write_multiple_channels([("A", 1.0), ("C", 2.0)])

        assert [r.refusal_reason for r in results] == [RefusalReason.WRITES_DISABLED] * 2
        assert read == []
        assert fake._state == {}

    async def test_writes_turned_off_mid_batch_refuse_every_write_not_yet_sent(
        self, fake, monkeypatch
    ):
        self._after_the_first_write(
            fake, monkeypatch, lambda: patch_writes_enabled(monkeypatch, False)
        )

        results = await fake.write_multiple_channels([("A", 1.0), ("B", 2.0), ("C", 3.0)])

        assert _outcomes(results) == [
            ("A", WriteOutcome.CONFIRMED, None),
            ("B", WriteOutcome.REFUSED, RefusalReason.WRITES_DISABLED),
            ("C", WriteOutcome.REFUSED, RefusalReason.WRITES_DISABLED),
        ]
        # C's entry is the step read's, not a write.
        assert fake._state == {"A": 1.0, "C": 0.0}

    async def test_a_session_held_mid_batch_refuses_every_write_not_yet_sent(
        self, fake, monkeypatch
    ):
        held = posture_store.WriteVerdict(
            allowed=False, reason="session_held", target=None, kind=None
        )
        self._after_the_first_write(
            fake,
            monkeypatch,
            lambda: monkeypatch.setattr(posture_store, "session_hold_refusal", lambda: held),
        )

        results = await fake.write_multiple_channels([("A", 1.0), ("B", 2.0)])

        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED, WriteOutcome.REFUSED]
        assert results[1].verdict_reason == "session_held"
        assert fake._state == {"A": 1.0}

    @staticmethod
    def _after_the_first_write(fake, monkeypatch, happen) -> None:
        """Have ``happen`` run once the batch's first write has been sent."""
        send = type(fake)._unguarded_write_channel
        sent: list[str] = []

        async def sending(self, channel_address, value, **kwargs):
            result = await send(self, channel_address, value, **kwargs)
            if not sent:
                happen()
            sent.append(channel_address)
            return result

        monkeypatch.setattr(type(fake), "_unguarded_write_channel", sending)

    async def test_an_explicit_confirm_reaches_every_channel(self, fake):
        results = await fake.write_multiple_channels([("A", 1.0), ("B", 2.0)], confirm=False)

        assert [r.outcome for r in results] == [WriteOutcome.UNREQUESTED] * 2

    async def test_an_empty_batch_answers_nothing(self, fake):
        assert await fake.write_multiple_channels([]) == []


# ---------------------------------------------------------------------------
# The EPICS connector's batch: checked without a thread, put together
# ---------------------------------------------------------------------------


class _GatedAioca(FakeAioca):
    """A fake aioca whose puts wait at a gate, counting how many wait at once."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Event()
        self.waiting = 0
        self.peak = 0

    async def caput(self, name, value, **kwargs):
        self.waiting += 1
        self.peak = max(self.peak, self.waiting)
        try:
            await self.gate.wait()
        finally:
            self.waiting -= 1
        return await super().caput(name, value, **kwargs)


def _channels(**values: float) -> dict[str, FakeChannel]:
    return {name: FakeChannel(ca_reading(value)) for name, value in values.items()}


def _epics(aioca: FakeAioca, **limits: dict[str, Any]):
    connector = ca_connector(aioca=aioca, limits_validator=_validator(**limits))

    async def no_thread(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a batch runs no write-validation thread")

    connector._run_write_phase = no_thread  # type: ignore[method-assign]
    return connector


def _gets(aioca: FakeAioca) -> list[str]:
    return [name for call, name, _ in aioca.calls if call == "caget"]


class TestEpicsBatch:
    async def test_every_put_of_a_batch_is_in_flight_at_once(self):
        names = [f"SR:C{n:03d}:SP" for n in range(166)]
        aioca = _GatedAioca(default=FakeChannel(ca_reading(1.0)))
        connector = _epics(aioca, **{name: {"max_value": 10.0} for name in names})

        batch = asyncio.ensure_future(
            connector.write_multiple_channels([(name, 1.0) for name in names], confirm=False)
        )
        for _ in range(2000):
            if aioca.waiting == len(names):
                break
            await asyncio.sleep(0)
        peak = aioca.waiting
        aioca.gate.set()
        results = await batch

        assert peak == len(names)
        assert [r.outcome for r in results] == [WriteOutcome.UNREQUESTED] * len(names)
        assert [r.channel_address for r in results] == names

    async def test_a_batch_is_confirmed_channel_by_channel(self):
        aioca = FakeAioca(_channels(A=1.0, B=5.0))
        connector = _epics(aioca, A={"max_value": 10.0}, B={"max_value": 10.0})

        results = await connector.write_multiple_channels([("A", 1.0), ("B", 2.0)])

        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED, WriteOutcome.MISMATCH]
        assert results[1].observed_value == 5.0

    async def test_one_limits_refusal_puts_nothing(self):
        aioca = FakeAioca(_channels(A=1.0, B=1.0, C=1.0))
        connector = _epics(
            aioca, A={"max_value": 10.0}, B={"max_value": 10.0}, C={"max_value": 10.0}
        )

        results = await connector.write_multiple_channels([("A", 1.0), ("B", 99.0), ("C", 1.0)])

        assert _outcomes(results) == [
            ("A", WriteOutcome.REFUSED, RefusalReason.BATCH_REFUSED),
            ("B", WriteOutcome.REFUSED, RefusalReason.LIMITS),
            ("C", WriteOutcome.REFUSED, RefusalReason.BATCH_REFUSED),
        ]
        assert aioca.puts == []

    async def test_only_the_stepped_channels_are_read_before_the_puts(self):
        aioca = FakeAioca(_channels(A=1.0, B=1.0, C=1.0))
        connector = _epics(
            aioca,
            A={"max_value": 10.0},
            B={"max_value": 10.0, "max_step": 2.0},
            C={"max_value": 10.0, "max_step": 2.0},
        )

        results = await connector.write_multiple_channels(
            [("A", 9.0), ("B", 2.0), ("C", 3.0)], confirm=False
        )

        assert [r.outcome for r in results] == [WriteOutcome.UNREQUESTED] * 3
        assert sorted(_gets(aioca)) == ["B", "C"]

    async def test_a_step_too_large_puts_nothing(self):
        aioca = FakeAioca(_channels(A=1.0, B=1.0))
        connector = _epics(aioca, A={"max_value": 10.0}, B={"max_value": 10.0, "max_step": 2.0})

        results = await connector.write_multiple_channels([("A", 9.0), ("B", 9.0)])

        assert results[1].refusal_reason is RefusalReason.LIMITS
        assert "MAX_STEP_EXCEEDED" in results[1].error_message
        assert aioca.puts == []

    async def test_a_stepped_channel_that_cannot_be_read_refuses_the_batch(self):
        aioca = FakeAioca({"A": FakeChannel(ca_reading(1.0)), "B": FakeChannel(connected=False)})
        connector = _epics(aioca, A={"max_value": 10.0}, B={"max_value": 10.0, "max_step": 2.0})

        results = await connector.write_multiple_channels([("A", 2.0), ("B", 2.0)])

        assert results[1].refusal_reason is RefusalReason.LIMITS
        assert "STEP_CHECK_FAILED" in results[1].error_message
        assert aioca.puts == []

    async def test_a_pva_routed_address_refuses_the_batch(self):
        aioca = FakeAioca(_channels(A=1.0))
        connector = _epics(aioca, A={"max_value": 10.0})
        connector._pva_channel_globs = ["IMG:*"]

        results = await connector.write_multiple_channels([("A", 1.0), ("IMG:ARRAY", 1.0)])

        assert _outcomes(results) == [
            ("A", WriteOutcome.REFUSED, RefusalReason.BATCH_REFUSED),
            ("IMG:ARRAY", WriteOutcome.REFUSED, RefusalReason.VALIDATION_ERROR),
        ]
        assert aioca.calls == []

    async def test_spellings_of_one_record_are_one_channel_named_twice(self):
        aioca = FakeAioca(_channels(A=0.0, B=0.0))
        connector = _epics(aioca, A={"max_value": 10.0, "max_step": 3.0}, B={"max_value": 10.0})

        results = await connector.write_multiple_channels(
            [("A", 3.0), ("B", 1.0), ("ca://A", -1.0)]
        )

        assert [r.refusal_reason for r in results] == [
            RefusalReason.VALIDATION_ERROR,
            RefusalReason.BATCH_REFUSED,
            RefusalReason.VALIDATION_ERROR,
        ]
        assert aioca.calls == []

    async def test_writes_turned_off_while_the_batch_is_checked_put_nothing(self, monkeypatch):
        """The verdict is asked again after the step read, right before the puts start."""

        class _SwitchedOffOnRead(FakeAioca):
            async def caget(self, name, **kwargs):
                patch_writes_enabled(monkeypatch, False)
                return await super().caget(name, **kwargs)

        aioca = _SwitchedOffOnRead(_channels(A=1.0, B=1.0))
        connector = _epics(aioca, A={"max_value": 10.0}, B={"max_value": 10.0, "max_step": 2.0})

        results = await connector.write_multiple_channels([("A", 2.0), ("B", 2.0)])

        assert _outcomes(results) == [
            ("A", WriteOutcome.REFUSED, RefusalReason.WRITES_DISABLED),
            ("B", WriteOutcome.REFUSED, RefusalReason.WRITES_DISABLED),
        ]
        assert _gets(aioca) == ["B"]
        assert aioca.puts == []

    async def test_a_put_the_control_system_refuses_leaves_the_others(self):
        aioca = FakeAioca(
            {
                "A": FakeChannel(ca_reading(1.0)),
                "B": FakeChannel(ca_reading(1.0), put=ECA_NOWTACCESS),
            }
        )
        connector = _epics(aioca, A={"max_value": 10.0}, B={"max_value": 10.0})

        results = await connector.write_multiple_channels([("A", 1.0), ("B", 1.0)])

        assert _outcomes(results) == [
            ("A", WriteOutcome.CONFIRMED, None),
            ("B", WriteOutcome.REFUSED, RefusalReason.CONTROL_SYSTEM_REFUSED),
        ]

    async def test_a_put_with_an_unknown_end_is_unconfirmed_and_leaves_the_others(self):
        aioca = FakeAioca(
            {"A": FakeChannel(ca_reading(1.0)), "B": FakeChannel(ca_reading(1.0), put=ECA_DISCONN)}
        )
        connector = _epics(aioca, A={"max_value": 10.0}, B={"max_value": 10.0})

        results = await connector.write_multiple_channels([("A", 1.0), ("B", 1.0)])

        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED, WriteOutcome.UNCONFIRMED]
        assert "B" in results[1].error_message

    async def test_a_single_write_still_raises_its_unknown_end(self):
        aioca = FakeAioca({"B": FakeChannel(ca_reading(1.0), put=ECA_DISCONN)})
        connector = ca_connector(aioca=aioca, limits_validator=_validator(B={"max_value": 10.0}))

        with pytest.raises(ChannelUnreachableError):
            await connector.write_channel("B", 1.0)

    async def test_a_batch_on_the_read_gateway_is_refused_switching(self):
        aioca = FakeAioca(_channels(A=1.0))
        connector = ca_connector(
            aioca=aioca, limits_validator=_validator(A={"max_value": 10.0}), write_gateway=True
        )

        results = await connector.write_multiple_channels([("A", 1.0), ("A", 2.0)])

        assert [r.refusal_reason for r in results] == [RefusalReason.WRITES_DISABLED] * 2
        assert [r.verdict_reason for r in results] == ["switching"] * 2
        assert aioca.calls == []


def test_a_refused_batch_raises_for_the_refused_write_not_one_it_held_back():
    from osprey_connectors.control_system import ChannelWriteResult, raise_for_write_results
    from osprey_connectors.errors import ChannelWriteBlockedError

    def refused(address, reason):
        return ChannelWriteResult(
            channel_address=address,
            value_written=1.0,
            outcome=WriteOutcome.REFUSED,
            refusal_reason=reason,
            error_message=f"{address}: {reason}",
        )

    results = [refused("A", "BATCH_REFUSED"), refused("B", "LIMITS"), refused("C", "BATCH_REFUSED")]

    with pytest.raises(ChannelWriteBlockedError) as info:
        raise_for_write_results(results)

    assert (info.value.channel_address, info.value.reason) == ("B", "LIMITS")
