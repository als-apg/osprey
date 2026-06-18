"""Unit tests for osprey.dispatch.sources.epics_ca.EpicsCaSource.

Exercises edge detection, cool-down suppression, first-read suppression, and
multi-PV independence without a live IOC.  pyepics is replaced by a minimal
fake that injects callbacks synchronously.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.dispatch.sources.epics_ca import EpicsCaSource, _PvWatcher
from osprey.dispatch.trigger_config import TriggerConfig

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
    """Minimal pyepics.PV stand-in; stores callback and exposes fire()."""

    def __init__(self, pvname: str, callback=None, **kw: Any) -> None:
        self.pvname = pvname
        self._callback = callback

    def fire(self, value: Any) -> None:
        """Synchronously invoke the callback as if a CA monitor event arrived."""
        if self._callback is not None:
            self._callback(pvname=self.pvname, value=value)

    def clear_callbacks(self) -> None:
        self._callback = None

    def disconnect(self) -> None:
        pass


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
# Integration-style tests: _PvWatcher with a fake CA monitor and a real asyncio
# loop running in a background thread.
#
# This mirrors production: pyepics delivers CA callbacks on its own thread while
# the dispatcher's asyncio loop runs elsewhere, and ``_PvWatcher`` bridges the
# two via ``run_coroutine_threadsafe``.  Running a genuine loop (rather than
# hand-stepping one) keeps the threadsafe hand-off deterministic regardless of
# platform, Python version, or scheduling timing.
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
    """Arm a _PvWatcher with a _FakePV and return both."""
    fake_pv = None

    def _fake_epics_pv(pvname, callback=None, **kw):
        nonlocal fake_pv
        fake_pv = _FakePV(pvname, callback=callback)
        return fake_pv

    with patch("osprey.dispatch.sources.epics_ca.epics.PV", side_effect=_fake_epics_pv):
        watcher = _PvWatcher(trigger=trigger, fire_callback=fire_cb, loop=loop)
        watcher.start()

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
    """EpicsCaSource.start arms watchers; stop() releases them."""

    @pytest.mark.asyncio
    async def test_start_arms_monitors_per_trigger(self) -> None:
        source = EpicsCaSource()
        triggers = [
            _trigger(name="t1", pv="SIM:PV:1"),
            _trigger(name="t2", pv="SIM:PV:2"),
        ]
        fire_cb = AsyncMock(return_value=None)

        created = []

        def _fake_pv(pvname, callback=None, **kw):
            pv = _FakePV(pvname, callback)
            created.append(pv)
            return pv

        with patch("osprey.dispatch.sources.epics_ca.epics.PV", side_effect=_fake_pv):
            await source.start(triggers, fire_cb)

        assert len(source._watchers) == 2
        assert len(created) == 2
        pv_names = {pv.pvname for pv in created}
        assert pv_names == {"SIM:PV:1", "SIM:PV:2"}

    @pytest.mark.asyncio
    async def test_start_skips_trigger_without_pv(self) -> None:
        source = EpicsCaSource()
        bad_trigger = TriggerConfig(
            name="no-pv",
            source="epics_ca",
            action={"prompt": "test"},
            source_config={},  # no 'pv' key
        )
        fire_cb = AsyncMock(return_value=None)

        with patch("osprey.dispatch.sources.epics_ca.epics.PV") as mock_pv:
            await source.start([bad_trigger], fire_cb)

        mock_pv.assert_not_called()
        assert len(source._watchers) == 0

    @pytest.mark.asyncio
    async def test_stop_clears_watchers(self) -> None:
        source = EpicsCaSource()
        triggers = [_trigger(name="t1", pv="SIM:PV:1")]
        fire_cb = AsyncMock(return_value=None)

        with patch("osprey.dispatch.sources.epics_ca.epics.PV", side_effect=_FakePV):
            await source.start(triggers, fire_cb)

        assert len(source._watchers) == 1
        await source.stop()
        assert len(source._watchers) == 0

    def test_source_type_classvar(self) -> None:
        assert EpicsCaSource.source_type == "epics_ca"
