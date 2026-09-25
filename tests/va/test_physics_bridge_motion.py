"""A served monitor moves the way the machine file says it moves.

The model owns where the beam is; the machine file owns how a monitor's
reading moves around that position -- a slow wander (``texture``) and white
noise (``noise``/``noise_abs``). The archived history of the same monitor is
synthesized from that same declaration, so the live reading has to carry it
too, or a trend across the boundary between the seeded past and the recorded
present changes character at the seam.

What this module pins: the motion follows the engine's own series at each
tick's time, rides on top of an orbit change a write causes, sits before the
readout faults (so a stand-in's offsets apply on top of it), leaves a monitor
the machine file gives no motion exactly as it was, and re-serves readings
without re-solving the orbit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from osprey.services.virtual_accelerator.bindings import BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
from osprey.simulation.engine import SimulationEngine

#: Two tick instants, far enough apart on a one-hour wander for the texture to
#: have visibly moved between them.
_T1 = 1_764_000_000.0
_T2 = _T1 + 900.0

#: A texture large against the solver's repeatability and small against any
#: orbit, in the unit the monitor publishes.
_WANDER = {"kind": "wander", "amplitude": 3.0e-5, "period_s": 3600.0}
_NOISE_ABS = 1.0e-6

_STEP = 1.0
_OFFSET = 0.25
_MOVED = 1.0e-12


class FakeRecord:
    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def document() -> BindingsDocument:
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def channels() -> list[dict]:
    return build_manifest()["channels"]


def _readings(document: BindingsDocument) -> list[str]:
    return [binding.setpoint_address for binding in document.bindings if binding.kind == "monitor"]


def _first_monitor(document: BindingsDocument) -> tuple[str, str]:
    """The first monitor reading of the document and the element it sits at."""
    monitor = next(binding for binding in document.bindings if binding.kind == "monitor")
    return monitor.setpoint_address, str(monitor.element)


def _engine(tmp_path: Path, entries: dict[str, dict]) -> SimulationEngine:
    """An engine whose machine file declares exactly ``entries``."""
    path = tmp_path / "machine.json"
    channels = {
        address: {"value": 0.0, "units": "m", **entry} for address, entry in entries.items()
    }
    path.write_text(json.dumps({"name": "motion", "channels": channels}))
    return SimulationEngine.from_file(path, state_dir=tmp_path / "state")


def _served(
    channels: list[dict],
    document: BindingsDocument,
    engine: SimulationEngine | None,
    clock: Clock,
    *,
    bpm_errors: dict[str, dict[str, float]] | None = None,
) -> tuple[PhysicsBridge, PyATRingModel, dict[str, FakeRecord]]:
    model = PyATRingModel(PACKAGE_PATHS.data_root, channels, bpm_errors=bpm_errors)
    bridge = PhysicsBridge(model, motion=engine, clock=clock)
    records = {address: FakeRecord() for address in _readings(document)}
    bridge.bind(records)
    return bridge, model, records


def _motion(engine: SimulationEngine, address: str, t: float) -> float:
    """The engine's own synthesized deviation from the 0.0 baseline at ``t``."""
    (sample,) = engine.synthesize_series(address, [t])
    return float(sample)


class TestTheReadingFollowsTheMachineFilesMotion:
    def test_two_ticks_land_on_the_engines_own_series(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        address, _element = _first_monitor(document)
        engine = _engine(tmp_path, {address: {"texture": _WANDER}})
        clock = Clock(_T1)
        bridge, _model, records = _served(channels, document, engine, clock)
        truth = bridge.bpm_positions()[address]

        served = []
        for t in (_T1, _T2):
            clock.now = t
            bridge.tick()
            served.append(records[address].value)
            assert records[address].value == pytest.approx(
                truth + _motion(engine, address, t), rel=0, abs=1e-15
            )

        assert abs(served[1] - served[0]) > 1e-7

    def test_bind_already_serves_the_motion(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        """The boot push is the first sample; it is not left unmoved until a tick."""
        address, _element = _first_monitor(document)
        engine = _engine(tmp_path, {address: {"texture": _WANDER}})
        bridge, _model, records = _served(channels, document, engine, Clock(_T1))
        assert records[address].value == pytest.approx(
            bridge.bpm_positions()[address] + _motion(engine, address, _T1), rel=0, abs=1e-15
        )

    def test_the_truth_the_bridge_publishes_carries_no_motion(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        address, _element = _first_monitor(document)
        engine = _engine(tmp_path, {address: {"texture": _WANDER}})
        moving, model, _ = _served(channels, document, engine, Clock(_T1))
        assert moving.bpm_positions() == dict(model.get(_readings(document)))


class TestAWriteStillMovesTheOrbitUnderneath:
    def test_the_orbit_change_shows_with_the_motion_on_top(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        addresses = _readings(document)
        engine = _engine(tmp_path, {address: {"texture": _WANDER} for address in addresses})
        clock = Clock(_T1)
        bridge, _model, records = _served(channels, document, engine, clock)
        before = bridge.bpm_positions()

        actuator = next(binding for binding in document.bindings if binding.kind == "kick")
        clock.now = _T2
        bridge.on_setpoint(actuator.setpoint_address, _STEP)
        after = bridge.bpm_positions()

        moved = [a for a in addresses if abs(after[a] - before[a]) > _MOVED]
        assert moved
        for address in moved:
            assert records[address].value == pytest.approx(
                after[address] + _motion(engine, address, _T2), rel=0, abs=1e-15
            )
            assert records[address].value != pytest.approx(after[address], rel=0, abs=1e-12)

    def test_a_tick_after_a_write_keeps_the_new_orbit(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        addresses = _readings(document)
        engine = _engine(tmp_path, {address: {"texture": _WANDER} for address in addresses})
        clock = Clock(_T1)
        bridge, _model, records = _served(channels, document, engine, clock)
        actuator = next(binding for binding in document.bindings if binding.kind == "kick")
        bridge.on_setpoint(actuator.setpoint_address, _STEP)
        after = bridge.bpm_positions()

        clock.now = _T2
        bridge.tick()
        for address in addresses:
            assert records[address].value == pytest.approx(
                after[address] + _motion(engine, address, _T2), rel=0, abs=1e-15
            )

    def test_a_tick_does_not_solve_the_orbit_again(
        self,
        tmp_path: Path,
        channels: list[dict],
        document: BindingsDocument,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        address, _element = _first_monitor(document)
        engine = _engine(tmp_path, {address: {"texture": _WANDER}})
        clock = Clock(_T1)
        bridge, model, _records = _served(channels, document, engine, clock)
        writes: list[dict] = []
        reads: list[list[str]] = []
        real_get = model.get
        monkeypatch.setattr(model, "set", lambda values: writes.append(values))
        monkeypatch.setattr(
            model, "get", lambda names: reads.append(list(names)) or real_get(names)
        )

        for k in range(3):
            clock.now = _T1 + 10.0 * k
            bridge.tick()

        assert writes == []
        # Only the readout faults are read back; the monitor truth is not.
        assert not any(set(names) & set(_readings(document)) for names in reads)


class TestNoiseIsOnEveryTick:
    def test_absolute_noise_has_its_declared_width_and_a_quiet_channel_stays_exact(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        noisy, quiet = _readings(document)[:2]
        engine = _engine(tmp_path, {noisy: {"noise_abs": _NOISE_ABS}, quiet: {}})
        clock = Clock(_T1)
        bridge, _model, records = _served(channels, document, engine, clock)
        truth = bridge.bpm_positions()

        noisy_values, quiet_values = [], []
        for k in range(500):
            clock.now = _T1 + 10.0 * k
            bridge.tick()
            noisy_values.append(records[noisy].value)
            quiet_values.append(records[quiet].value)

        deviation = np.asarray(noisy_values) - truth[noisy]
        assert deviation.std() == pytest.approx(_NOISE_ABS, rel=0.15)
        assert abs(deviation.mean()) < 0.2 * _NOISE_ABS
        assert quiet_values == [truth[quiet]] * 500


class TestAMonitorWithNoMotionIsUnchanged:
    def test_no_machine_entry_serves_the_truth_on_every_tick(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        moving, still = _readings(document)[:2]
        engine = _engine(tmp_path, {moving: {"texture": _WANDER}})
        clock = Clock(_T1)
        bridge, _model, records = _served(channels, document, engine, clock)
        for t in (_T1, _T2):
            clock.now = t
            bridge.tick()
            assert records[still].value == bridge.bpm_positions()[still]

    def test_without_an_engine_every_reading_is_what_it_was(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        clock = Clock(_T1)
        bridge, _model, records = _served(channels, document, None, clock)
        assert not bridge.moves
        clock.now = _T2
        bridge.tick()
        assert {a: r.value for a, r in records.items()} == bridge.bpm_positions()

    def test_an_engine_declaring_no_motion_for_any_monitor_moves_nothing(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        engine = _engine(tmp_path, {address: {} for address in _readings(document)})
        bridge, _model, records = _served(channels, document, engine, Clock(_T1))
        assert not bridge.moves
        assert {a: r.value for a, r in records.items()} == bridge.bpm_positions()


class TestReadoutFaultsApplyOnTopOfTheMotion:
    def test_a_standin_offset_is_subtracted_from_the_moving_reading(
        self, tmp_path: Path, channels: list[dict], document: BindingsDocument
    ) -> None:
        address, element = _first_monitor(document)
        axis = PyATRingModel(PACKAGE_PATHS.data_root, channels).supported_variables[address].axis
        engine = _engine(tmp_path, {address: {"texture": _WANDER, "noise_abs": _NOISE_ABS}})
        clock = Clock(_T1)
        bridge, _model, records = _served(
            channels, document, engine, clock, bpm_errors={element: {f"offset_{axis}": _OFFSET}}
        )
        for t in (_T1, _T2):
            clock.now = t
            bridge.tick()
            assert records[address].value == pytest.approx(
                bridge.bpm_positions()[address] + _motion(engine, address, t) - _OFFSET,
                rel=0,
                abs=1e-15,
            )
