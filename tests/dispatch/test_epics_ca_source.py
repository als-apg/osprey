"""Unit tests for osprey.dispatch.sources.epics_ca.EpicsCaSource.

Exercises edge detection, cool-down suppression, first-read suppression, and
multi-PV independence without a live IOC.  pyepics is replaced by a minimal
fake that injects callbacks synchronously.
"""

from __future__ import annotations

import asyncio
import time
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
        assert w._detect_edge(0.5, 1.0) is True   # exactly at threshold
        assert w._detect_edge(0.0, 2.0) is True   # crosses above

    def test_rising_edge_not_detected_when_falling(self) -> None:
        w = self._make_watcher("rising", threshold=1.0)
        assert w._detect_edge(1.5, 0.5) is False  # falling through threshold
        assert w._detect_edge(0.5, 0.8) is False  # below threshold both sides

    def test_falling_edge_detected(self) -> None:
        w = self._make_watcher("falling", threshold=1.0)
        assert w._detect_edge(1.5, 0.5) is True   # crosses below
        assert w._detect_edge(2.0, 1.0) is True   # exactly at threshold

    def test_falling_edge_not_detected_when_rising(self) -> None:
        w = self._make_watcher("falling", threshold=1.0)
        assert w._detect_edge(0.5, 1.5) is False
        assert w._detect_edge(1.5, 2.0) is False  # above threshold both sides

    def test_both_edge_detects_rising_and_falling(self) -> None:
        w = self._make_watcher("both", threshold=1.0)
        assert w._detect_edge(0.5, 1.0) is True   # rising
        assert w._detect_edge(1.5, 0.5) is True   # falling

    def test_both_edge_no_fire_when_same_side(self) -> None:
        w = self._make_watcher("both", threshold=1.0)
        assert w._detect_edge(0.5, 0.8) is False
        assert w._detect_edge(1.5, 2.0) is False


# ---------------------------------------------------------------------------
# Integration-style tests: _PvWatcher with fake CA + real asyncio loop
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_loop():
    """Fresh asyncio event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _run(coro, loop: asyncio.AbstractEventLoop):
    return loop.run_until_complete(coro)


def _arm_watcher(
    trigger: TriggerConfig,
    fire_cb: AsyncMock,
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

    def test_no_fire_on_first_value(self, event_loop) -> None:
        fire_cb = AsyncMock(return_value="dispatch-id-1")
        trigger = _trigger(pv="SIM:SETPOINT", threshold=1.0, edge="rising")
        watcher, fake_pv = _arm_watcher(trigger, fire_cb, event_loop)

        # First value: should suppress regardless of whether it crosses threshold.
        fake_pv.fire(2.0)
        _run(asyncio.sleep(0), event_loop)  # drain any queued coroutines

        fire_cb.assert_not_called()

    def test_fire_on_second_crossing(self, event_loop) -> None:
        fire_cb = AsyncMock(return_value="dispatch-id-2")
        trigger = _trigger(pv="SIM:SETPOINT", threshold=1.0, edge="rising", cool_down_sec=0.0)
        watcher, fake_pv = _arm_watcher(trigger, fire_cb, event_loop)

        fake_pv.fire(0.0)   # first read (suppressed)
        fake_pv.fire(2.0)   # rising crossing — should fire

        # run_coroutine_threadsafe schedules on the loop; run a step to execute it.
        _run(asyncio.sleep(0), event_loop)

        fire_cb.assert_called_once()
        payload = fire_cb.call_args[0][1]
        assert payload["source"] == "epics_ca"
        assert payload["pv"] == "SIM:SETPOINT"
        assert payload["value"] == 2.0
        assert payload["previous_value"] == 0.0


class TestCoolDown:
    """Repeat fires within cool_down_sec are suppressed."""

    def test_second_fire_suppressed_within_cool_down(self, event_loop) -> None:
        fire_cb = AsyncMock(return_value="dispatch-id")
        trigger = _trigger(
            pv="SIM:BEAM", threshold=1.0, edge="rising", cool_down_sec=9999.0
        )
        watcher, fake_pv = _arm_watcher(trigger, fire_cb, event_loop)

        fake_pv.fire(0.0)   # first read (suppressed)
        fake_pv.fire(2.0)   # crossing — fires
        _run(asyncio.sleep(0), event_loop)
        assert fire_cb.call_count == 1

        # Simulate falling back below then rising again within the cool-down window.
        fake_pv.fire(0.0)   # falling
        fake_pv.fire(2.0)   # rising again — suppressed by cool_down
        _run(asyncio.sleep(0), event_loop)
        assert fire_cb.call_count == 1  # still 1, second fire suppressed

    def test_fire_allowed_after_cool_down_expires(self, event_loop) -> None:
        fire_cb = AsyncMock(return_value="dispatch-id")
        trigger = _trigger(
            pv="SIM:BEAM", threshold=1.0, edge="rising", cool_down_sec=0.001
        )
        watcher, fake_pv = _arm_watcher(trigger, fire_cb, event_loop)

        fake_pv.fire(0.0)   # first read (suppressed)
        fake_pv.fire(2.0)   # crossing — fires
        _run(asyncio.sleep(0), event_loop)
        assert fire_cb.call_count == 1

        time.sleep(0.01)  # wait longer than cool_down_sec

        fake_pv.fire(0.0)   # fall below
        fake_pv.fire(2.0)   # rising again — cool-down expired, should fire
        _run(asyncio.sleep(0), event_loop)
        assert fire_cb.call_count == 2


# ---------------------------------------------------------------------------
# Multi-PV independence
# ---------------------------------------------------------------------------


class TestMultiPvIndependence:
    """Each PV watcher has its own state; one PV does not affect another."""

    def test_cool_down_is_per_pv(self, event_loop) -> None:
        """Triggering PV-A within its cool-down must not affect PV-B."""
        fire_a = AsyncMock(return_value="id-a")
        fire_b = AsyncMock(return_value="id-b")

        trigger_a = _trigger(
            name="trigger-a", pv="SIM:PV:A", threshold=1.0, edge="rising", cool_down_sec=9999.0
        )
        trigger_b = _trigger(
            name="trigger-b", pv="SIM:PV:B", threshold=1.0, edge="rising", cool_down_sec=0.0
        )

        watcher_a, pv_a = _arm_watcher(trigger_a, fire_a, event_loop)
        watcher_b, pv_b = _arm_watcher(trigger_b, fire_b, event_loop)

        pv_a.fire(0.0)   # first read A
        pv_b.fire(0.0)   # first read B

        pv_a.fire(2.0)   # A fires (1st)
        pv_b.fire(2.0)   # B fires (1st)
        _run(asyncio.sleep(0), event_loop)
        assert fire_a.call_count == 1
        assert fire_b.call_count == 1

        # A is now in cool-down; B is not (cool_down=0).
        pv_a.fire(0.0)
        pv_a.fire(2.0)   # A: suppressed
        pv_b.fire(0.0)
        pv_b.fire(2.0)   # B: fires (2nd)
        _run(asyncio.sleep(0), event_loop)
        assert fire_a.call_count == 1  # A still suppressed
        assert fire_b.call_count == 2  # B fired again

    def test_first_read_suppression_is_per_pv(self, event_loop) -> None:
        """Each PV suppresses only its own first read."""
        fire_a = AsyncMock(return_value="id-a")
        fire_b = AsyncMock(return_value="id-b")

        trigger_a = _trigger(name="trigger-a", pv="SIM:PV:A")
        trigger_b = _trigger(name="trigger-b", pv="SIM:PV:B")

        watcher_a, pv_a = _arm_watcher(trigger_a, fire_a, event_loop)
        watcher_b, pv_b = _arm_watcher(trigger_b, fire_b, event_loop)

        # B fires its first read (suppressed); A has not yet had one.
        pv_b.fire(0.0)

        # B crosses the threshold on its 2nd value → fires.
        pv_b.fire(2.0)
        _run(asyncio.sleep(0), event_loop)
        assert fire_b.call_count == 1

        # A still on its first read (suppressed).
        pv_a.fire(2.0)
        _run(asyncio.sleep(0), event_loop)
        assert fire_a.call_count == 0  # A's first read suppressed

        # A's 2nd value fires.
        pv_a.fire(0.0)   # fall below
        pv_a.fire(2.0)   # rise → fires
        _run(asyncio.sleep(0), event_loop)
        assert fire_a.call_count == 1


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
