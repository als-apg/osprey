"""Unit tests for osprey.dispatch.sources.epics_ca.EpicsCaSource.

Exercises edge detection, cool-down suppression, first-read suppression, and
multi-PV independence without a live IOC. The source watches through the
control-system connector, which is replaced here by a minimal fake that
delivers each update on the serving loop, as the real one does; the last
tests run the real EPICS connector over a fake aioca.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.dispatch.sources import epics_ca
from osprey.dispatch.sources.epics_ca import EpicsCaSource, _PvWatcher
from osprey.dispatch.trigger_config import TriggerConfig
from osprey_connectors.control_system.base import ChannelValue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trigger(
    name: str = "test-trigger",
    pv: str = "SIM:PV",
    threshold: float = 1.0,
    edge: str = "rising",
    cool_down_sec: float = 0.0,
) -> TriggerConfig:
    """Build a TriggerConfig with EPICS CA source_config."""
    return TriggerConfig(
        name=name,
        source="epics_ca",
        action={"prompt": "test prompt"},
        source_config={
            "pv": pv,
            "threshold": threshold,
            "edge": edge,
            "cool_down_sec": cool_down_sec,
        },
    )


class _FakePV:
    """One subscription on :class:`_FakeConnector`; ``fire()`` delivers an update."""

    def __init__(self, pvname: str, callback: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.pvname = pvname
        self._callback = callback
        self._loop = loop
        self.closed = False

    def fire(self, value: Any) -> None:
        """Deliver ``value`` as the connector does: a ChannelValue, on the serving loop."""
        if not self.closed:
            update = ChannelValue(value=value, timestamp=datetime.now(UTC))
            self._loop.call_soon_threadsafe(self._callback, update)


class _FakeConnector:
    """The connector surface the source uses: subscribe, unsubscribe, disconnect.

    A PV named in ``stalled`` never finishes subscribing, and its cancelled
    subscribe is recorded in ``cancelled``.
    """

    def __init__(self, stalled: tuple[str, ...] = ()) -> None:
        self.pvs: dict[str, _FakePV] = {}
        self.stalled = stalled
        self.cancelled: list[str] = []
        self.disconnected = False

    async def subscribe(self, pvname: str, callback: Any) -> str:
        if pvname in self.stalled:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(pvname)
                raise
        sub_id = f"{pvname}_{len(self.pvs)}"
        self.pvs[sub_id] = _FakePV(pvname, callback, asyncio.get_running_loop())
        return sub_id

    async def unsubscribe(self, sub_id: str) -> None:
        self.pvs[sub_id].closed = True

    async def disconnect(self) -> None:
        self.disconnected = True

    @property
    def names(self) -> list[str]:
        return [pv.pvname for pv in self.pvs.values()]


@contextmanager
def _connecting(connector: Any = None):
    """Patch the source's connector build; yield (the mock build, the connector it returns)."""
    connector = connector if connector is not None else _FakeConnector()
    with patch.object(epics_ca, "_connect", AsyncMock(return_value=connector)) as build:
        yield build, connector


# ---------------------------------------------------------------------------
# _PvWatcher unit tests (synchronous, no real asyncio loop)
# ---------------------------------------------------------------------------


class TestPvWatcherEdgeDetection:
    """_detect_edge covers rising, falling, and both directions."""

    def _make_watcher(self, edge: str, threshold: float = 1.0) -> _PvWatcher:
        loop = MagicMock()
        return _PvWatcher(
            trigger=_trigger(threshold=threshold, edge=edge),
            fire_callback=AsyncMock(),
            loop=loop,
        )

    def test_rising_edge_detected(self) -> None:
        w = self._make_watcher("rising", threshold=1.0)
        assert w._detect_edge(0.5, 1.0) is True  # exactly at threshold
        assert w._detect_edge(0.0, 2.0) is True  # crosses above

    def test_rising_edge_not_detected_when_falling(self) -> None:
        w = self._make_watcher("rising", threshold=1.0)
        assert w._detect_edge(1.5, 0.5) is False  # falling through threshold
        assert w._detect_edge(0.5, 0.8) is False  # below threshold both sides

    def test_falling_edge_detected(self) -> None:
        w = self._make_watcher("falling", threshold=1.0)
        assert w._detect_edge(1.5, 0.5) is True  # crosses below
        assert w._detect_edge(2.0, 1.0) is True  # exactly at threshold

    def test_falling_edge_not_detected_when_rising(self) -> None:
        w = self._make_watcher("falling", threshold=1.0)
        assert w._detect_edge(0.5, 1.5) is False
        assert w._detect_edge(1.5, 2.0) is False  # above threshold both sides

    def test_both_edge_detects_rising_and_falling(self) -> None:
        w = self._make_watcher("both", threshold=1.0)
        assert w._detect_edge(0.5, 1.0) is True  # rising
        assert w._detect_edge(1.5, 0.5) is True  # falling

    def test_both_edge_no_fire_when_same_side(self) -> None:
        w = self._make_watcher("both", threshold=1.0)
        assert w._detect_edge(0.5, 0.8) is False
        assert w._detect_edge(1.5, 2.0) is False


# ---------------------------------------------------------------------------
# Integration-style tests: _PvWatcher with a fake connector and a real asyncio
# loop running in a background thread.
#
# This mirrors production: CA updates arrive on the CA loop's thread and the
# connector hands each one to the dispatcher's asyncio loop, where the watcher
# runs. Running a genuine loop (rather than hand-stepping one) keeps that
# threadsafe hand-off deterministic regardless of platform, Python version, or
# scheduling timing.
# ---------------------------------------------------------------------------


@contextmanager
def _running_loop():
    """Yield an asyncio loop that is running in a dedicated background thread."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _serve() -> None:
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_serve, name="epics-ca-test-loop", daemon=True)
    thread.start()
    if not ready.wait(5.0):  # pragma: no cover - safety net
        raise RuntimeError("test event loop failed to start")
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5.0)
        loop.close()


class _FireRecorder:
    """Async fire-callback stand-in that records each dispatch thread-safely.

    The dispatch runs on the loop thread; the test asserts from the main thread.
    ``wait_until`` polls the thread-safe counter so positive assertions block
    only as long as needed, and ``settle`` gives any spurious dispatch a chance
    to land before a negative assertion.
    """

    def __init__(self, return_value: str | None = "dispatch-id") -> None:
        self._return_value = return_value
        self._lock = threading.Lock()
        self._calls: list[tuple[Any, dict[str, Any]]] = []

    async def __call__(self, trigger: Any, payload: dict[str, Any]) -> str | None:
        with self._lock:
            self._calls.append((trigger, payload))
        return self._return_value

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._calls)

    @property
    def payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            return [payload for _, payload in self._calls]

    def wait_until(self, count: int, timeout: float = 2.0) -> bool:
        """Block until at least ``count`` dispatches have run (or timeout)."""
        deadline = time.monotonic() + timeout
        while self.call_count < count and time.monotonic() < deadline:
            time.sleep(0.005)
        return self.call_count >= count

    @staticmethod
    def settle(grace: float = 0.15) -> None:
        """Let any in-flight dispatch land before a 'did not fire' assertion."""
        time.sleep(grace)


def _arm_watcher(
    trigger: TriggerConfig,
    fire_cb: Any,
    loop: asyncio.AbstractEventLoop,
) -> tuple[_PvWatcher, _FakePV]:
    """Arm a _PvWatcher on ``loop`` through a fake connector; return it and its PV."""
    connector = _FakeConnector()
    watcher = _PvWatcher(trigger=trigger, fire_callback=fire_cb, loop=loop)
    asyncio.run_coroutine_threadsafe(watcher.start(connector), loop).result(5.0)
    (fake_pv,) = connector.pvs.values()
    return watcher, fake_pv


class TestFirstReadSuppression:
    """No fire on the initial connect-time PV value."""

    def test_no_fire_on_first_value(self) -> None:
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:SETPOINT", threshold=1.0, edge="rising")
        with _running_loop() as loop:
            _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
            # First value: suppressed even though it crosses the threshold.
            fake_pv.fire(2.0)
            rec.settle()
        assert rec.call_count == 0

    def test_fire_on_second_crossing(self) -> None:
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:SETPOINT", threshold=1.0, edge="rising", cool_down_sec=0.0)
        with _running_loop() as loop:
            _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
            fake_pv.fire(0.0)  # first read (suppressed)
            fake_pv.fire(2.0)  # rising crossing — should fire
            assert rec.wait_until(1)
        assert rec.call_count == 1
        payload = rec.payloads[0]
        assert payload["source"] == "epics_ca"
        assert payload["pv"] == "SIM:SETPOINT"
        assert payload["value"] == 2.0
        assert payload["previous_value"] == 0.0
        assert isinstance(payload["timestamp"], datetime)
        assert payload["timestamp"].utcoffset() == timedelta(0)


class TestCoolDown:
    """Repeat fires within cool_down_sec are suppressed."""

    def test_second_fire_suppressed_within_cool_down(self) -> None:
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:BEAM", threshold=1.0, edge="rising", cool_down_sec=9999.0)
        with _running_loop() as loop:
            _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
            fake_pv.fire(0.0)  # first read (suppressed)
            fake_pv.fire(2.0)  # crossing — fires
            assert rec.wait_until(1)

            # Fall back below then rise again within the cool-down window.
            fake_pv.fire(0.0)  # falling
            fake_pv.fire(2.0)  # rising again — suppressed by cool_down
            rec.settle()
        assert rec.call_count == 1  # second fire suppressed

    def test_fire_allowed_after_cool_down_expires(self) -> None:
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:BEAM", threshold=1.0, edge="rising", cool_down_sec=0.001)
        with _running_loop() as loop:
            _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
            fake_pv.fire(0.0)  # first read (suppressed)
            fake_pv.fire(2.0)  # crossing — fires
            assert rec.wait_until(1)

            time.sleep(0.01)  # wait longer than cool_down_sec

            fake_pv.fire(0.0)  # fall below
            fake_pv.fire(2.0)  # rising again — cool-down expired, should fire
            assert rec.wait_until(2)
        assert rec.call_count == 2

    def test_first_fire_not_suppressed_when_monotonic_below_cooldown(self) -> None:
        """A large cool_down must not suppress the FIRST fire when the monotonic
        clock reads below cool_down_sec (e.g. a freshly booted host, where
        ``time.monotonic()`` can be smaller than the configured cool-down)."""
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:BEAM", threshold=1.0, edge="rising", cool_down_sec=9999.0)
        with patch("osprey.dispatch.sources.epics_ca.time.monotonic", return_value=5.0):
            with _running_loop() as loop:
                _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
                fake_pv.fire(0.0)  # first read (suppressed)
                fake_pv.fire(2.0)  # crossing — must fire despite monotonic(5) < cool_down
                assert rec.wait_until(1)
        assert rec.call_count == 1


class TestNonNumericValue:
    """Non-numeric PV values are logged and ignored, not raised on the CA thread."""

    def test_non_numeric_value_does_not_fire_or_raise(self) -> None:
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:ENUM", threshold=1.0, edge="rising", cool_down_sec=0.0)
        with _running_loop() as loop:
            _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
            fake_pv.fire(0.0)  # numeric first read (suppressed)
            fake_pv.fire("Enabled")  # non-numeric — must be ignored, not raise
            rec.settle()
            assert rec.call_count == 0

            # A subsequent numeric crossing still fires (watcher survived the bad value).
            fake_pv.fire(2.0)
            assert rec.wait_until(1)
        assert rec.call_count == 1

    def test_numeric_string_is_coerced(self) -> None:
        rec = _FireRecorder()
        trigger = _trigger(pv="SIM:STR", threshold=1.0, edge="rising", cool_down_sec=0.0)
        with _running_loop() as loop:
            _watcher, fake_pv = _arm_watcher(trigger, rec, loop)
            fake_pv.fire("0.0")  # numeric string first read (suppressed)
            fake_pv.fire("2.0")  # numeric string crossing — coerced and fires
            assert rec.wait_until(1)
        assert rec.call_count == 1
        payload = rec.payloads[0]
        assert payload["value"] == 2.0
        assert payload["previous_value"] == 0.0


# ---------------------------------------------------------------------------
# Multi-PV independence
# ---------------------------------------------------------------------------


class TestMultiPvIndependence:
    """Each PV watcher has its own state; one PV does not affect another."""

    def test_cool_down_is_per_pv(self) -> None:
        """Triggering PV-A within its cool-down must not affect PV-B."""
        rec_a = _FireRecorder("id-a")
        rec_b = _FireRecorder("id-b")

        trigger_a = _trigger(
            name="trigger-a", pv="SIM:PV:A", threshold=1.0, edge="rising", cool_down_sec=9999.0
        )
        trigger_b = _trigger(
            name="trigger-b", pv="SIM:PV:B", threshold=1.0, edge="rising", cool_down_sec=0.0
        )

        with _running_loop() as loop:
            _watcher_a, pv_a = _arm_watcher(trigger_a, rec_a, loop)
            _watcher_b, pv_b = _arm_watcher(trigger_b, rec_b, loop)

            pv_a.fire(0.0)  # first read A
            pv_b.fire(0.0)  # first read B

            pv_a.fire(2.0)  # A fires (1st)
            pv_b.fire(2.0)  # B fires (1st)
            assert rec_a.wait_until(1)
            assert rec_b.wait_until(1)

            # A is now in cool-down; B is not (cool_down=0).
            pv_a.fire(0.0)
            pv_a.fire(2.0)  # A: suppressed
            pv_b.fire(0.0)
            pv_b.fire(2.0)  # B: fires (2nd)
            assert rec_b.wait_until(2)
            rec_a.settle()
        assert rec_a.call_count == 1  # A still suppressed
        assert rec_b.call_count == 2  # B fired again

    def test_first_read_suppression_is_per_pv(self) -> None:
        """Each PV suppresses only its own first read."""
        rec_a = _FireRecorder("id-a")
        rec_b = _FireRecorder("id-b")

        trigger_a = _trigger(name="trigger-a", pv="SIM:PV:A")
        trigger_b = _trigger(name="trigger-b", pv="SIM:PV:B")

        with _running_loop() as loop:
            _watcher_a, pv_a = _arm_watcher(trigger_a, rec_a, loop)
            _watcher_b, pv_b = _arm_watcher(trigger_b, rec_b, loop)

            # B fires its first read (suppressed); A has not yet had one.
            pv_b.fire(0.0)
            pv_b.fire(2.0)  # B crosses the threshold on its 2nd value → fires.
            assert rec_b.wait_until(1)

            # A still on its first read (suppressed).
            pv_a.fire(2.0)
            rec_a.settle()
            assert rec_a.call_count == 0

            # A's 2nd value fires.
            pv_a.fire(0.0)  # fall below
            pv_a.fire(2.0)  # rise → fires
            assert rec_a.wait_until(1)
        assert rec_a.call_count == 1


# ---------------------------------------------------------------------------
# EpicsCaSource integration: start() + stop()
# ---------------------------------------------------------------------------


class TestEpicsCaSourceLifecycle:
    """EpicsCaSource.start arms watchers through one connector; stop() releases them."""

    @pytest.mark.asyncio
    async def test_start_arms_monitors_per_trigger(self) -> None:
        source = EpicsCaSource()
        triggers = [
            _trigger(name="t1", pv="SIM:PV:1"),
            _trigger(name="t2", pv="SIM:PV:2"),
        ]

        with _connecting() as (build, connector):
            await source.start(triggers, AsyncMock(return_value=None))

        build.assert_awaited_once_with()  # one connector for every trigger
        assert len(source._watchers) == 2
        assert sorted(connector.names) == ["SIM:PV:1", "SIM:PV:2"]

    @pytest.mark.asyncio
    async def test_start_skips_trigger_without_pv(self) -> None:
        source = EpicsCaSource()
        bad_trigger = TriggerConfig(
            name="no-pv",
            source="epics_ca",
            action={"prompt": "test"},
            source_config={},  # no 'pv' key
        )

        with _connecting() as (build, _connector):
            await source.start([bad_trigger], AsyncMock(return_value=None))

        build.assert_not_called()  # nothing to watch, so no connector either
        assert len(source._watchers) == 0

    @pytest.mark.asyncio
    async def test_start_skips_trigger_with_unknown_edge(self, caplog) -> None:
        """`edge: up` used to arm a both-edges monitor; it is now refused."""
        source = EpicsCaSource()

        with (
            caplog.at_level("WARNING", logger="osprey.dispatch.sources.epics_ca"),
            _connecting() as (build, _connector),
        ):
            await source.start([_trigger(name="typo", edge="up")], AsyncMock(return_value=None))

        build.assert_not_called()
        assert len(source._watchers) == 0
        assert "typo" in caplog.text
        assert "'up'" in caplog.text

    @pytest.mark.asyncio
    async def test_sibling_triggers_stay_armed(self) -> None:
        """One mistyped edge takes down that trigger, not the dispatcher's set."""
        source = EpicsCaSource()
        triggers = [
            _trigger(name="typo", pv="SIM:PV:1", edge="up"),
            _trigger(name="good", pv="SIM:PV:2", edge="falling"),
        ]

        with _connecting() as (_build, connector):
            await source.start(triggers, AsyncMock(return_value=None))

        assert len(source._watchers) == 1
        assert connector.names == ["SIM:PV:2"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("edge", ["rising", "falling", "both"])
    async def test_every_valid_edge_arms(self, edge) -> None:
        """The three edges _detect_edge implements are all accepted."""
        source = EpicsCaSource()

        with _connecting():
            await source.start([_trigger(edge=edge)], AsyncMock(return_value=None))

        assert len(source._watchers) == 1

    @pytest.mark.asyncio
    async def test_absent_edge_defaults_to_rising(self) -> None:
        """A trigger that never mentions `edge` keeps the documented default."""
        source = EpicsCaSource()
        trigger = TriggerConfig(
            name="no-edge",
            source="epics_ca",
            action={"prompt": "test"},
            source_config={"pv": "SIM:PV"},
        )

        with _connecting():
            await source.start([trigger], AsyncMock(return_value=None))

        assert len(source._watchers) == 1
        assert source._watchers[0]._edge == "rising"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("threshold", "high"),
            ("threshold", True),
            ("threshold", float("nan")),
            ("threshold", float("inf")),
            ("cool_down_sec", "soon"),
            ("cool_down_sec", -1),
        ],
    )
    async def test_start_skips_a_trigger_with_an_unusable_number(self, caplog, key, value) -> None:
        """A threshold or cool-down the watcher cannot compare against is not armed."""
        source = EpicsCaSource()
        trigger = TriggerConfig(
            name="bad-number",
            source="epics_ca",
            action={"prompt": "test"},
            source_config={"pv": "SIM:PV", key: value},
        )

        with _connecting() as (build, _connector):
            await source.start([trigger], AsyncMock(return_value=None))

        build.assert_not_called()
        assert len(source._watchers) == 0
        assert "bad-number" in caplog.text
        assert f"'{key}'" in caplog.text
        assert repr(value) in caplog.text

    @pytest.mark.asyncio
    async def test_a_bad_number_leaves_its_siblings_armed(self) -> None:
        """One unusable threshold takes down that trigger, not the dispatcher's set."""
        source = EpicsCaSource()
        triggers = [
            TriggerConfig(
                name="bad",
                source="epics_ca",
                action={"prompt": "test"},
                source_config={"pv": "SIM:PV:1", "threshold": "high"},
            ),
            _trigger(name="good", pv="SIM:PV:2"),
        ]

        with _connecting() as (_build, connector):
            await source.start(triggers, AsyncMock(return_value=None))

        assert len(source._watchers) == 1
        assert connector.names == ["SIM:PV:2"]

    @pytest.mark.asyncio
    async def test_a_blank_number_takes_its_default(self) -> None:
        """A blank threshold or cool-down (YAML null) means the documented default."""
        source = EpicsCaSource()
        trigger = TriggerConfig(
            name="blank",
            source="epics_ca",
            action={"prompt": "test"},
            source_config={"pv": "SIM:PV", "threshold": None, "cool_down_sec": None},
        )

        with _connecting():
            await source.start([trigger], AsyncMock(return_value=None))

        assert len(source._watchers) == 1
        assert source._watchers[0]._threshold == 0.0
        assert source._watchers[0]._cool_down == 60.0

    @pytest.mark.asyncio
    async def test_an_exponent_yaml_reads_as_a_string_is_still_a_number(self) -> None:
        """YAML reads ``1e-3`` as a string; it still arms as the number it spells."""
        source = EpicsCaSource()
        trigger = TriggerConfig(
            name="exponent",
            source="epics_ca",
            action={"prompt": "test"},
            source_config={"pv": "SIM:PV", "threshold": "1e-3"},
        )

        with _connecting():
            await source.start([trigger], AsyncMock(return_value=None))

        assert len(source._watchers) == 1
        assert source._watchers[0]._threshold == 0.001

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_and_disconnects(self) -> None:
        source = EpicsCaSource()
        triggers = [_trigger(name="t1", pv="SIM:PV:1")]

        with _connecting() as (_build, connector):
            await source.start(triggers, AsyncMock(return_value=None))

        assert len(source._watchers) == 1
        await source.stop()
        assert len(source._watchers) == 0
        assert [pv.closed for pv in connector.pvs.values()] == [True]
        assert connector.disconnected is True

    @pytest.mark.asyncio
    async def test_a_fire_runs_on_the_serving_loop(self) -> None:
        source = EpicsCaSource()
        fired = asyncio.Event()
        loops: list[asyncio.AbstractEventLoop] = []

        async def fire(_trigger: Any, _payload: dict[str, Any]) -> None:
            loops.append(asyncio.get_running_loop())
            fired.set()

        with _connecting() as (_build, connector):
            await source.start([_trigger(threshold=1.0)], fire)
        (pv,) = connector.pvs.values()
        pv.fire(0.0)
        pv.fire(2.0)
        await asyncio.wait_for(fired.wait(), 2.0)

        assert loops == [asyncio.get_running_loop()]

    def test_source_type_classvar(self) -> None:
        assert EpicsCaSource.source_type == "epics_ca"


class TestOpenTimeout:
    @pytest.mark.asyncio
    async def test_start_skips_a_trigger_whose_monitor_did_not_open_in_time(
        self, caplog, monkeypatch
    ) -> None:
        """The subscribe is cancelled, which is what has the connector close it behind."""
        monkeypatch.setattr(epics_ca, "_OPEN_TIMEOUT_S", 0.05)
        source = EpicsCaSource()

        with (
            caplog.at_level("WARNING", logger="osprey.dispatch.sources.epics_ca"),
            _connecting(_FakeConnector(stalled=("SIM:SLOW",))) as (_build, connector),
        ):
            await source.start(
                [_trigger(name="slow", pv="SIM:SLOW"), _trigger(name="ok", pv="SIM:OK")],
                AsyncMock(),
            )

        assert [w._pv_name for w in source._watchers] == ["SIM:OK"]
        assert connector.cancelled == ["SIM:SLOW"]
        assert "slow" in caplog.text

    @pytest.mark.asyncio
    async def test_a_connect_that_does_not_finish_arms_nothing(self, caplog, monkeypatch) -> None:
        monkeypatch.setattr(epics_ca, "_OPEN_TIMEOUT_S", 0.05)

        async def stalled() -> None:
            await asyncio.Event().wait()

        source = EpicsCaSource()
        with (
            caplog.at_level("WARNING", logger="osprey.dispatch.sources.epics_ca"),
            patch.object(epics_ca, "_connect", stalled),
        ):
            await source.start([_trigger()], AsyncMock())

        assert source._watchers == []
        assert "did not connect" in caplog.text
        await source.stop()  # nothing to release, and no error for it


# ---------------------------------------------------------------------------
# The connector the source builds, and the real EPICS connector under it
# ---------------------------------------------------------------------------


class TestConnect:
    """The source watches through an ``epics`` connector with no gateway of its own."""

    @pytest.mark.asyncio
    async def test_the_factory_builds_an_epics_connector_with_no_settings(self) -> None:
        create = AsyncMock(return_value=MagicMock())
        with patch(
            "osprey_connectors.factory.ConnectorFactory.create_control_system_connector", create
        ):
            connector = await epics_ca._connect()

        create.assert_awaited_once_with({"type": "epics"})
        assert connector is create.return_value

    @pytest.mark.asyncio
    async def test_the_dispatchers_ca_environment_is_left_as_it_stands(self, monkeypatch) -> None:
        """``dispatch.env`` hands the gateway over; a gateway block would rewrite it."""
        from osprey_connectors.control_system.epics_connector import EPICSConnector
        from tests.connectors._epics_fakes import install_fake_aioca

        install_fake_aioca(monkeypatch)
        monkeypatch.setenv("EPICS_CA_ADDR_LIST", "gateway.example.org")
        monkeypatch.setenv("EPICS_CA_AUTO_ADDR_LIST", "NO")

        connector = await epics_ca._connect()

        assert isinstance(connector, EPICSConnector)
        assert connector._connector_type == "epics"
        assert os.environ["EPICS_CA_ADDR_LIST"] == "gateway.example.org"
        assert os.environ["EPICS_CA_AUTO_ADDR_LIST"] == "NO"


class TestThroughTheEpicsConnector:
    """The real connector over a fake aioca: what a trigger sees across a repoint."""

    @pytest.mark.asyncio
    async def test_a_trigger_stays_armed_across_a_repoint(self, monkeypatch) -> None:
        from tests.connectors._epics_fakes import ca_reading, install_fake_aioca

        fake = install_fake_aioca(monkeypatch)
        connector = await epics_ca._connect()
        fired: list[tuple[float, float]] = []

        async def fire(_trigger: Any, payload: dict[str, Any]) -> None:
            fired.append((payload["previous_value"], payload["value"]))

        source = EpicsCaSource()
        with _connecting(connector):
            await source.start([_trigger(pv="SIM:PV", threshold=1.0)], fire)

        fake.monitors[-1].callback(ca_reading(0.0))  # first read, suppressed
        connector._ca.repoint()  # aioca closes every monitor; the connector re-opens
        reopened = fake.monitors[-1]
        assert reopened is not fake.monitors[0]
        reopened.callback(ca_reading(2.0))
        for _ in range(10):
            await asyncio.sleep(0)

        assert fired == [(0.0, 2.0)]
        await source.stop()
        assert reopened.closed is True
