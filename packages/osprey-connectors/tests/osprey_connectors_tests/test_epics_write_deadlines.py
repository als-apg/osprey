"""Every phase of an EPICS write is bounded, and so is the write as a whole.

A write runs three phases, each with one deadline that covers its connection
and its calls: the ``max_step`` read of the current value (S, the connector's
``step_read_timeout_s``), the put (T, its ``timeout_s``: connect, field-type
lookup and put) and the confirming read (T: connect, get, and the control
fields only for an enumeration). A session that promises an answer within
``S + 2T`` relies on this.

The fake client below models the parts of aioca that decide timing, with no
IOC and no waiting: every call is bounded the way aioca's
``maybe_throw``/``ca_timeout`` bounds it -- one deadline over the connection
and the call together, given either as a relative float or as an absolute
``(deadline,)`` tuple -- and a call that runs out answers
``CANothing(name, ECA_TIMEOUT)``, as under ``throw=False``. Time is a
:class:`VirtualClock` that the connector reads too: a call that fits before
its deadline moves the clock on by what it took, and one that does not moves
it to the deadline. So every duration below is exact, and a busy runner cannot
change one. A channel connects once (``connect_s`` after its first use) and
stays connected until a test drops it.

Two tests are about concurrency rather than time. Their writes meet at a
:class:`_Rendezvous` that opens only when all of them are in flight at once, so
writes that queue behind each other are caught by what they did, not by how
long it took.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import types
from typing import Any
from unittest.mock import MagicMock

import aioca
import pytest

from osprey_connectors.control_system import _epics_connector
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.control_system._ca_loop import CaLoop
from osprey_connectors.control_system._epics_connector import (
    WRITE_THREAD_PREFIX,
    EPICSConnector,
    _PutFault,
    _WriteExecutor,
)
from osprey_connectors.control_system.write_door import door_is_open
from osprey_connectors_tests._epics_fakes import (
    DBR_DOUBLE,
    DBR_ENUM,
    ECA_TIMEOUT,
    InlineCaLoop,
    ca_connector,
    ca_reading,
)

S = 0.3  # step-read budget
T = 0.4  # put and confirming-read budget
BOUND = S + 2 * T

#: aioca's own default when a call is given no timeout.
AIOCA_DEFAULT_TIMEOUT = 5.0
#: A call or connection that never ends.
NEVER = math.inf
#: Real seconds a concurrency test waits for its writes to meet before it
#: gives up. Only a failing run ever spends them.
GIVE_UP_S = 10.0


class VirtualClock:
    """The time the connector and the fake aioca both read; only a call that takes time moves it.

    It stands in for the connector module's ``time``, so the connector's
    deadlines (``time.time() + T``) and its connection bookkeeping
    (``time.monotonic()``) run on it. One scale serves both, so a deadline is
    compared with the same number elapsed time is measured on.
    """

    def __init__(self, start: float = 1_000.0):
        self._now = start
        self._lock = threading.Lock()

    def time(self) -> float:
        return self._now

    monotonic = time

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def advance_to(self, when: float) -> None:
        """Move the clock on to ``when``; never back."""
        with self._lock:
            self._now = max(self._now, when)


class _Rendezvous:
    """Holds each call that arrives until ``count`` of them are in flight together.

    ``most`` is the largest number ever held at once, so calls that queue
    behind one another leave it below ``count``. The wait gives up after
    :data:`GIVE_UP_S` of real time and lets every held call go, so a failing
    run reports that rather than hanging; a passing run never waits for it.
    Every arrival must come on one event loop: the CA loop's.
    """

    def __init__(self, count: int):
        self.count = count
        self.most = 0
        self._held = 0
        self._all_in = asyncio.Event()

    async def arrive(self) -> None:
        self._held += 1
        self.most = max(self.most, self._held)
        if self._held >= self.count:
            self._all_in.set()
        try:
            await asyncio.wait_for(self._all_in.wait(), GIVE_UP_S)
        except TimeoutError:
            self._all_in.set()
        finally:
            self._held -= 1


class _Record:
    """One channel's record and its timing knobs, shared by every call on it.

    ``connect_s``: time to connect, counted from the first call that needs the
    channel. ``info_s``: time for ``cainfo``. ``get_s``: time for a TIME-form
    (or plain) get. ``ctrl_s``: time for a CTRL-form get. ``put_s``: time until
    the put callback arrives (only a waiting put waits for it). ``after_put``
    runs once the put has ended, however it ended. ``read_gate`` and
    ``put_gate`` hold a connected get or put at a :class:`_Rendezvous`.
    """

    def __init__(
        self,
        value: Any = 1.0,
        *,
        labels: tuple[str, ...] | None = None,
        connect_s: float = 0.0,
        info_s: float = 0.0,
        get_s: float = 0.0,
        ctrl_s: float = 0.0,
        put_s: float = 0.0,
        read_gate: _Rendezvous | None = None,
        put_gate: _Rendezvous | None = None,
    ):
        self.held = value
        self.labels = labels
        self.connect_s = connect_s
        self.info_s = info_s
        self.get_s = get_s
        self.ctrl_s = ctrl_s
        self.put_s = put_s
        self.read_gate = read_gate
        self.put_gate = put_gate
        self.after_put: Any = None
        self.down = False
        self.clock: VirtualClock  # set by the fake that serves it
        self._connected_at: float | None = None
        self.puts: list[Any] = []
        self.put_threads: list[str] = []
        self.door_seen: list[bool] = []
        self.ctrl_fetches = 0

    @property
    def datatype(self) -> int:
        return DBR_ENUM if self.labels is not None else DBR_DOUBLE

    def drop(self) -> None:
        """The channel goes down and stays down."""
        self.down = True

    def reconnect_after(self, seconds: float) -> None:
        """The channel drops and is connected again ``seconds`` from now."""
        self.down = False
        self._connected_at = self.clock.monotonic() + seconds

    def until_connected(self) -> float:
        """Seconds from now until the channel is connected."""
        if self.down:
            return NEVER
        now = self.clock.monotonic()
        if self._connected_at is None:
            self._connected_at = now + self.connect_s
        return max(0.0, self._connected_at - now)

    def reading(self, **attrs: Any) -> Any:
        return ca_reading(self.held, datatype=self.datatype, **attrs)


class TimingAioca(types.ModuleType):
    """A stand-in ``aioca`` whose calls take virtual time and keep their deadlines as aioca does."""

    CANothing = aioca.CANothing
    FORMAT_RAW = aioca.FORMAT_RAW
    FORMAT_TIME = aioca.FORMAT_TIME
    FORMAT_CTRL = aioca.FORMAT_CTRL
    DBR_CHAR_STR = aioca.DBR_CHAR_STR

    def __init__(self, clock: VirtualClock, **records: _Record):
        super().__init__("aioca")
        self.clock = clock
        self.records = {name.replace("_", ":"): record for name, record in records.items()}
        for record in self.records.values():
            record.clock = clock
        self.calls: list[tuple[str, str, Any]] = []

    def _deadline(self, timeout: Any) -> float:
        """aioca's ``ca_timeout``: a relative float, an absolute tuple, or its default."""
        if timeout is None:
            timeout = AIOCA_DEFAULT_TIMEOUT
        if isinstance(timeout, tuple):
            return timeout[0]
        return self.clock.time() + timeout

    def _spend(self, deadline: float, seconds: float) -> bool:
        """Spend ``seconds`` of the call, or answer ``False`` at the deadline if they do not fit."""
        finish = self.clock.monotonic() + seconds
        if finish > deadline:
            self.clock.advance_to(deadline)
            return False
        self.clock.advance_to(finish)
        return True

    @staticmethod
    def _timed_out(name: str) -> Any:
        return aioca.CANothing(name, ECA_TIMEOUT)

    async def connect(self, name, *, timeout=None, throw=True, **_kw):
        assert throw is False
        self.calls.append(("connect", name, timeout))
        record = self.records[name]
        if not self._spend(self._deadline(timeout), record.until_connected()):
            return self._timed_out(name)
        return aioca.CANothing(name)

    async def caget(self, name, *, format=aioca.FORMAT_RAW, timeout=None, throw=True, **_kw):
        assert throw is False
        self.calls.append(("caget", name, format))
        record = self.records[name]
        deadline = self._deadline(timeout)
        if not self._spend(deadline, record.until_connected()):
            return self._timed_out(name)
        if record.read_gate is not None:
            await record.read_gate.arrive()
        if format == aioca.FORMAT_CTRL:
            record.ctrl_fetches += 1
            if not self._spend(deadline, record.ctrl_s):
                return self._timed_out(name)
            return record.reading(units="mA", precision=3, enums=record.labels)
        if not self._spend(deadline, record.get_s):
            return self._timed_out(name)
        return record.reading()

    async def cainfo(self, name, *, timeout=None, throw=True, **_kw):
        assert throw is False
        self.calls.append(("cainfo", name, None))
        record = self.records[name]
        deadline = self._deadline(timeout)
        if not self._spend(deadline, record.until_connected()):
            return self._timed_out(name)
        if not self._spend(deadline, record.info_s):
            return self._timed_out(name)
        return types.SimpleNamespace(name=name, datatype=record.datatype, count=1, write=True)

    async def caput(self, name, value, *, wait=False, timeout=None, throw=True, **_kw):
        assert throw is False
        self.calls.append(("caput", name, value))
        record = self.records[name]
        deadline = self._deadline(timeout)
        if not self._spend(deadline, record.until_connected()):
            return self._timed_out(name)
        if record.put_gate is not None:
            await record.put_gate.arrive()
        record.puts.append(value)
        record.put_threads.append(threading.current_thread().name)
        record.door_seen.append(door_is_open())
        # The IOC maps an enum label to its state itself.
        if record.labels is not None and isinstance(value, str) and value in record.labels:
            record.held = record.labels.index(value)
        else:
            record.held = value
        try:
            if wait and not self._spend(deadline, record.put_s):
                return self._timed_out(name)
        finally:
            if record.after_put is not None:
                record.after_put()
        return aioca.CANothing(name)


class _SlowHopLoop(InlineCaLoop):
    """A busy CA loop: each coroutine starts ``hop`` seconds after it is handed over."""

    def __init__(self, clock: VirtualClock, hop: float):
        super().__init__()
        self.clock = clock
        self.hop = hop

    async def run(self, coro):
        self.clock.advance(self.hop)
        return await coro


def _step_checking_validator(seen_threads: list[str] | None = None):
    """A limits validator with a max_step rule: it reads the current value and fails closed."""
    validator = MagicMock()

    def validate(address, _value, read_current=None):
        if seen_threads is not None:
            seen_threads.append(threading.current_thread().name)
        if read_current(address) is None:
            raise RuntimeError(f"current value of '{address}' unavailable")

    validator.validate.side_effect = validate
    return validator


def _connector(fake: TimingAioca, *, validator=None, loop=None) -> EPICSConnector:
    connector = ca_connector(aioca=fake, limits_validator=validator, timeout=T)
    connector._step_read_timeout = S
    if loop is not None:
        connector._ca = loop
    return connector


@pytest.fixture(autouse=True)
def _writes_enabled(monkeypatch):
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda self: True))


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> VirtualClock:
    """The connector reads the virtual clock, never the wall clock."""
    clock = VirtualClock()
    monkeypatch.setattr(_epics_connector, "time", clock)
    return clock


@pytest.fixture
def ca_loop():
    """A CA loop on a thread of its own, as in production."""
    loop = CaLoop()
    yield loop
    loop.stop()


async def _timed_write(clock, connector, address, value, **kw):
    start = clock.monotonic()
    result = await connector.write_channel(address, value, **kw)
    return result, clock.monotonic() - start


class TestStepRead:
    def test_a_slow_connection_and_a_slow_get_share_one_budget(self, clock):
        """Connection and get each fit in S; together they do not, and the read ends at S."""
        connector = _connector(TimingAioca(clock, SR_CH=_Record(connect_s=0.2, get_s=0.2)))

        start = clock.monotonic()
        current = connector.current_value_reader()("SR:CH")
        elapsed = clock.monotonic() - start

        assert current is None
        assert elapsed == pytest.approx(S)

    @pytest.mark.asyncio
    async def test_a_step_read_that_never_connects_refuses_at_s(self, clock):
        record = _Record(connect_s=NEVER)
        connector = _connector(
            TimingAioca(clock, SR_CH=record), validator=_step_checking_validator()
        )

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.REFUSED
        assert record.puts == []
        assert elapsed == pytest.approx(S)


class TestPut:
    @pytest.mark.asyncio
    async def test_a_put_whose_callback_never_comes_is_unconfirmed_at_t(self, clock):
        record = _Record(put_s=NEVER)
        connector = _connector(TimingAioca(clock, SR_CH=record))

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert record.puts == [2.0]
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_the_connection_is_charged_to_the_put_deadline(self, clock):
        """A connection that takes most of T leaves the callback only the rest."""
        record = _Record(connect_s=0.3, put_s=0.3)
        connector = _connector(TimingAioca(clock, SR_CH=record))

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert record.puts == [2.0]
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_connect_field_lookup_and_put_share_one_deadline(self, clock):
        """Each step fits in T on its own; together they do not, and the put phase ends at T."""
        record = _Record(connect_s=0.15, info_s=0.15, put_s=0.15)
        connector = _connector(TimingAioca(clock, SR_CH=record))

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert record.puts == [2.0]
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_a_field_lookup_that_never_answers_fails_the_put_unsent_at_t(self, clock):
        record = _Record(info_s=NEVER)
        connector = _connector(TimingAioca(clock, SR_CH=record))

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.FAILED
        assert record.puts == []
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_the_put_deadline_is_set_before_the_hop_to_the_ca_loop(self, clock):
        """Time spent waiting for a busy CA loop is charged to the put: past T, nothing is sent."""
        record = _Record()
        hop = 1.5 * T
        connector = _connector(TimingAioca(clock, SR_CH=record), loop=_SlowHopLoop(clock, hop))

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=False)

        assert result.outcome is WriteOutcome.FAILED
        assert record.puts == []
        assert elapsed == pytest.approx(hop)

    @pytest.mark.asyncio
    async def test_a_put_after_its_deadline_is_not_sent(self, clock):
        record = _Record()
        fake = TimingAioca(clock, SR_CH=record)
        connector = _connector(fake)

        outcome = await connector._put_ca("SR:CH", 2.0, wait=True, deadline=(clock.time() - 0.01,))

        assert isinstance(outcome, _PutFault)
        assert outcome.verdict == "not_sent"
        assert record.puts == []
        assert not [call for call in fake.calls if call[0] == "caput"]

    @pytest.mark.asyncio
    async def test_the_door_is_open_where_the_put_runs(self, clock, ca_loop):
        record = _Record()
        connector = _connector(TimingAioca(clock, SR_CH=record), loop=ca_loop)

        result = await connector.write_channel("SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert record.door_seen == [True]
        assert record.put_threads == [ca_loop._thread.name]
        assert door_is_open() is False


class TestConfirmingRead:
    @pytest.mark.asyncio
    async def test_a_get_that_never_answers_is_unconfirmed_at_t(self, clock):
        record = _Record()
        connector = _connector(TimingAioca(clock, SR_CH=record))

        def stall():
            record.get_s = NEVER  # the put lands; the read that follows stalls

        record.after_put = stall

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert record.puts == [2.0]
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_connect_and_get_share_the_confirming_read_deadline(self, clock):
        """A reconnect and a get each fit in T; together they do not, and the read ends at T."""
        record = _Record()
        connector = _connector(TimingAioca(clock, SR_CH=record))

        def reconnect_slowly():
            record.reconnect_after(0.25)
            record.get_s = 0.25

        record.after_put = reconnect_slowly

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_stalled_control_fields_cost_the_read_nothing_beyond_t(self, clock):
        """An enum's labels never arrive; the reading stands and the write is still confirmed."""
        record = _Record(0, labels=("OFF", "ON"), ctrl_s=NEVER)
        connector = _connector(TimingAioca(clock, SR_VALVE=record))

        result, elapsed = await _timed_write(clock, connector, "SR:VALVE", 1, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 1
        assert record.ctrl_fetches == 1
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_the_control_fetch_shares_the_get_deadline(self, clock):
        """Get and label fetch each fit in T; together they do not, and the read ends at T."""
        record = _Record(0, labels=("OFF", "ON"), get_s=0.35, ctrl_s=0.35)
        connector = _connector(TimingAioca(clock, SR_VALVE=record))

        result, elapsed = await _timed_write(clock, connector, "SR:VALVE", 1, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert record.ctrl_fetches == 1
        assert elapsed == pytest.approx(T)

    @pytest.mark.asyncio
    async def test_an_enum_label_is_confirmed_with_the_fetched_labels(self, clock):
        record = _Record(0, labels=("OFF", "ON"))
        connector = _connector(TimingAioca(clock, SR_VALVE=record))

        result = await connector.write_channel("SR:VALVE", "ON", confirm=True)

        assert record.puts == ["ON"]
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 1
        assert record.ctrl_fetches == 1

    @pytest.mark.asyncio
    async def test_a_non_enum_channel_fetches_no_control_fields(self, clock):
        record = _Record()
        fake = TimingAioca(clock, SR_CH=record)
        connector = _connector(fake)

        result = await connector.write_channel("SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.alarm_severity == 0
        assert record.ctrl_fetches == 0
        assert ("caget", "SR:CH", aioca.FORMAT_CTRL) not in fake.calls


class TestConnectionEvidence:
    @pytest.mark.asyncio
    async def test_a_confirming_read_records_its_connection_for_stale_detection(self, clock):
        connector = _connector(TimingAioca(clock, SR_CH=_Record()))

        await connector._confirming_read("SR:CH", T)

        assert "SR:CH" in connector._one_shot_connected
        assert "SR:CH" not in connector._connect_failures

    @pytest.mark.asyncio
    async def test_a_confirming_read_that_cannot_connect_records_the_failure(self, clock):
        record = _Record()
        connector = _connector(TimingAioca(clock, SR_CH=record))
        record.after_put = record.drop  # the put lands, then the channel goes down

        result, elapsed = await _timed_write(clock, connector, "SR:CH", 2.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert "SR:CH" in connector._connect_failures
        assert elapsed == pytest.approx(T)


class TestWholeWrite:
    @pytest.mark.asyncio
    async def test_every_phase_slow_still_ends_within_s_plus_2t(self, clock, ca_loop):
        """Step read and put each succeed late in their budgets, then the read stalls."""
        record = _Record(0, labels=("OFF", "ON"), get_s=0.25, put_s=0.3)
        connector = _connector(
            TimingAioca(clock, SR_VALVE=record), validator=_step_checking_validator(), loop=ca_loop
        )

        def stall_after_put():
            record.get_s = NEVER
            record.ctrl_s = NEVER

        record.after_put = stall_after_put

        result, elapsed = await _timed_write(clock, connector, "SR:VALVE", "ON", confirm=True)

        assert record.puts == ["ON"]
        assert result.outcome is WriteOutcome.UNCONFIRMED
        # The step read's get, the put's callback, then the whole confirming read.
        assert elapsed == pytest.approx(0.25 + 0.3 + T)
        assert elapsed <= BOUND

    @pytest.mark.asyncio
    async def test_a_saturated_default_executor_does_not_delay_a_write(self, clock):
        """Work that fills the loop's default pool cannot queue a write's validation.

        The pool's one worker is held until the write is over, so a validation
        queued there would never run and the write would not end.
        """
        loop = asyncio.get_running_loop()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        release = threading.Event()
        blockers = [loop.run_in_executor(None, release.wait, GIVE_UP_S) for _ in range(3)]
        try:
            record = _Record()
            seen: list[str] = []
            connector = _connector(
                TimingAioca(clock, SR_CH=record), validator=_step_checking_validator(seen)
            )

            result = await asyncio.wait_for(
                connector.write_channel("SR:CH", 2.0, confirm=True), GIVE_UP_S
            )

            assert result.outcome is WriteOutcome.CONFIRMED
            assert len(seen) == 1
            assert seen[0].startswith(WRITE_THREAD_PREFIX)
        finally:
            release.set()
            await asyncio.gather(*blockers)

    @pytest.mark.asyncio
    async def test_concurrent_stalled_writes_do_not_queue_behind_each_other(self, clock, ca_loop):
        """Stalled writes to distinct channels have their reads, then puts, in flight together."""
        count = 16
        reads, puts = _Rendezvous(count), _Rendezvous(count)
        records = {
            f"SR_CH{i}": _Record(put_s=NEVER, read_gate=reads, put_gate=puts) for i in range(count)
        }
        connector = _connector(
            TimingAioca(clock, **records), validator=_step_checking_validator(), loop=ca_loop
        )
        # The step read's real-time backstop on the CA loop must outlast the rendezvous.
        connector._step_read_timeout = 3 * GIVE_UP_S

        results = await asyncio.gather(
            *(connector.write_channel(f"SR:CH{i}", 2.0, confirm=True) for i in range(count))
        )

        assert reads.most == count
        assert puts.most == count
        assert all(result.outcome is WriteOutcome.UNCONFIRMED for result in results)
        assert all(record.puts == [2.0] for record in records.values())

    def test_the_write_executor_refuses_a_call_past_its_thread_cap(self):
        executor = _WriteExecutor(max_threads=1)
        release = threading.Event()
        first = executor.submit(release.wait, 10.0)
        try:
            with pytest.raises(RuntimeError, match="cap is 1"):
                executor.submit(lambda: "never runs")
        finally:
            release.set()
        assert first.result(timeout=5) is True
        executor.shutdown()

        assert executor.submit(lambda: "runs").result(timeout=5) == "runs"
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_disconnect_retires_the_write_executor_and_a_later_write_builds_a_new_one(
        self, clock
    ):
        connector = _connector(
            TimingAioca(clock, SR_CH=_Record()), validator=_step_checking_validator()
        )

        await connector.write_channel("SR:CH", 2.0, confirm=True)
        first = connector._write_executor
        assert first is not None
        await connector.disconnect()
        assert connector._write_executor is None

        connector._connected = True
        result = await connector.write_channel("SR:CH", 3.0, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector._write_executor is not None
        assert connector._write_executor is not first
