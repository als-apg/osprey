"""The pyat engine's shot handling on a turn-by-turn ring (decisions Arm, CF-10, M4, CF-6).

A client write to an ``arm`` record arms its monitor; ``set`` never does. A
delivery whose ``shot`` input rose against the previous delivery, outside a
baseline, is a shot: with a monitor armed the ring tracks once and the armed
monitors' arrays and the turns tracked are emitted; the first ``advance``
after that drain drops armed. An unarmed shot is consumed. Nothing raises
out of a time-contract call, and ``reset`` returns everything to its start.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pytest

from osprey.facility.records import FacilityRecords
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines import pyat_turns
from osprey.simulation.engines.pyat_turn_state import ShotStatus, turn_state_of

MODEL = "SR"
INJECTION = "INJ"
TURNS = 20
TRIGGER = 5
SAMPLES = TRIGGER + TURNS

ARM_1 = "SR:DIAG:BPM:01:TBT:ARM"
ARMED_1 = "SR:DIAG:BPM:01:TBT:ARMED"
X_1 = "SR:DIAG:BPM:01:TBT:X"
Y_1 = "SR:DIAG:BPM:01:TBT:Y"
SUM_1 = "SR:DIAG:BPM:01:TBT:SUM"
ARM_2 = "SR:DIAG:BPM:02:TBT:ARM"
ARMED_2 = "SR:DIAG:BPM:02:TBT:ARMED"
X_2 = "SR:DIAG:BPM:02:TBT:X"
TURNS_TRACKED = "SR:DIAG:TBT:TURNS"
TURNS_SURVIVED = "SR:DIAG:TBT:SURVIVED"
MONITORS_REACHED = "SR:DIAG:TBT:REACHED"
SHOT_STATUS = "SR:DIAG:TBT:STATUS"
#: The ring's read-outs of the last shot.
OUTCOME = {TURNS_TRACKED, TURNS_SURVIVED, MONITORS_REACHED, SHOT_STATUS}

SHOT = "INJ:SHOT:COUNTER"
CHARGE = "INJ:CHARGE"
STORED = "SR:DCCT:CURRENT"

ARRAYS_1 = {X_1, Y_1, SUM_1}
SHOT_NAMES_1 = {X_1, Y_1, SUM_1} | OUTCOME


@pytest.fixture(scope="module")
def demo(built_demo: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(FacilityRecords(facility), MODEL),
        "deck": built_demo.facility_dir / model["deck"],
        "settings": model.get("settings"),
    }


def _settings(demo: dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(demo["settings"]) or {}
    settings.setdefault("pyat", {})["turn_by_turn"] = {
        "injection_element": INJECTION,
        "turns": TURNS,
        "trigger_sample": TRIGGER,
        "particles": 2,
    }
    return settings


def _record(signal: str, address: str, *, element: str | None = None, write: bool = False):
    record: dict[str, Any] = {
        "id": f"{MODEL}/{address}",
        "address": address,
        "engine": {"signal": signal},
        "direction": "write" if write else "read",
    }
    if element is not None:
        record["element"] = element
    if write:
        record["default"] = 0.0
    return record


def _turn_records() -> list[dict[str, Any]]:
    return [
        _record("arm", ARM_1, element="BPM01", write=True),
        _record("armed", ARMED_1, element="BPM01"),
        _record("turn_x", X_1, element="BPM01"),
        _record("turn_y", Y_1, element="BPM01"),
        _record("turn_sum", SUM_1, element="BPM01"),
        _record("arm", ARM_2, element="BPM02", write=True),
        _record("armed", ARMED_2, element="BPM02"),
        _record("turn_x", X_2, element="BPM02"),
        _record("turns_tracked", TURNS_TRACKED),
        _record("turns_survived", TURNS_SURVIVED),
        _record("monitors_reached", MONITORS_REACHED),
        _record("shot_status", SHOT_STATUS),
    ]


@pytest.fixture
def ring(demo: dict[str, Any]) -> Any:
    """A fresh turn-by-turn ring, advanced once as the composite does after a build."""
    model = engine.build(
        MODEL, [*demo["wiring"], *_turn_records()], demo["deck"], _settings(demo), {}
    )
    engine.advance(model, 0.0)
    return model


@pytest.fixture
def plain_ring(demo: dict[str, Any]) -> Any:
    """The same ring with no turn-by-turn block."""
    return engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], {})


def _deliver(
    model: Any,
    shot: float,
    *,
    charge: float = 1.0,
    stored: float = 0.0,
    t_s: float = 1.0,
    baseline: bool = False,
) -> None:
    pairs = [
        ({"address": CHARGE, "engine": {"signal": "charge"}}, charge),
        ({"address": SHOT, "engine": {"signal": "shot"}}, shot),
        ({"address": STORED, "engine": {"signal": "stored_current"}}, stored),
    ]
    engine.take_inputs(model, pairs, t_s, baseline=baseline)
    engine.advance(model, t_s)


def _arm(model: Any, address: str, value: float = 1.0, t_s: float = 0.5) -> list[str]:
    model.set({address: value})
    engine.client_write(model, {address: value}, t_s)
    return engine.emitted(model)


def _value(model: Any, name: str) -> Any:
    return model.get([name])[name]


def _baselined(model: Any, shot: float = 10.0, **kwargs: Any) -> None:
    _deliver(model, shot, baseline=True, t_s=0.1, **kwargs)


class TestArming:
    def test_a_client_write_arms_and_emits_armed(self, ring):
        assert _arm(ring, ARM_1) == [ARMED_1]
        assert _value(ring, ARMED_1) == 1.0
        assert _value(ring, ARMED_2) == 0.0

    def test_set_alone_never_arms(self, ring):
        ring.set({ARM_1: 1.0})
        engine.advance(ring, 0.5)
        assert _value(ring, ARMED_1) == 0.0
        assert engine.emitted(ring) == []

    def test_arming_an_armed_monitor_changes_nothing(self, ring):
        _arm(ring, ARM_1)
        assert _arm(ring, ARM_1) == []
        assert _value(ring, ARMED_1) == 1.0

    def test_a_zero_write_disarms(self, ring):
        _arm(ring, ARM_1)
        assert _arm(ring, ARM_1, 0.0) == [ARMED_1]
        assert _value(ring, ARMED_1) == 0.0

    def test_a_write_to_another_record_arms_nothing(self, ring):
        engine.client_write(ring, {X_1: 1.0}, 0.5)
        assert engine.emitted(ring) == []
        assert _value(ring, ARMED_1) == 0.0


class TestShots:
    def test_an_armed_shot_tracks_and_emits_arrays_and_turns(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0, charge=2.0)
        assert set(engine.emitted(ring)) == SHOT_NAMES_1
        assert _value(ring, TURNS_TRACKED) == TURNS
        total = np.asarray(_value(ring, SUM_1))
        assert total.shape == (SAMPLES,)
        assert np.all(np.isfinite(total))
        assert total[TRIGGER] > total[0]
        assert np.all(np.asarray(_value(ring, X_2)) == 0.0)

    def test_an_unarmed_shot_is_consumed(self, ring):
        _baselined(ring)
        _deliver(ring, 11.0)
        assert engine.emitted(ring) == []
        assert _value(ring, TURNS_TRACKED) == 0.0
        _arm(ring, ARM_1)
        _deliver(ring, 11.0, t_s=2.0)
        assert engine.emitted(ring) == []
        assert _value(ring, TURNS_TRACKED) == 0.0

    def test_a_partial_arm_changes_only_the_armed_monitor(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0)
        names = set(engine.emitted(ring))
        assert X_2 not in names and ARMED_2 not in names
        assert np.all(np.asarray(_value(ring, X_2)) == 0.0)
        assert _value(ring, ARMED_2) == 0.0
        engine.advance(ring, 1.5)
        assert engine.emitted(ring) == [ARMED_1]

    def test_both_monitors_armed_both_change(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _arm(ring, ARM_2)
        _deliver(ring, 11.0)
        assert set(engine.emitted(ring)) == SHOT_NAMES_1 | {X_2}
        assert np.any(np.asarray(_value(ring, X_2)) != 0.0)
        engine.advance(ring, 1.5)
        assert set(engine.emitted(ring)) == {ARMED_1, ARMED_2}

    def test_a_baseline_delivery_is_no_shot(self, ring):
        _arm(ring, ARM_1)
        _deliver(ring, 10.0, baseline=True)
        _deliver(ring, 20.0, baseline=True, t_s=2.0)
        assert engine.emitted(ring) == []
        assert _value(ring, TURNS_TRACKED) == 0.0
        assert _value(ring, ARMED_1) == 1.0

    def test_the_first_delivery_is_no_shot_even_unmarked(self, ring):
        _arm(ring, ARM_1)
        _deliver(ring, 10.0)
        assert engine.emitted(ring) == []

    def test_a_lower_counter_is_no_shot(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 3.0)
        assert engine.emitted(ring) == []
        assert _value(ring, ARMED_1) == 1.0
        _deliver(ring, 4.0, t_s=2.0)
        assert set(engine.emitted(ring)) == SHOT_NAMES_1

    def test_only_the_shot_input_starts_tracking(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 10.0, charge=5.0, stored=3.0)
        assert engine.emitted(ring) == []
        assert _value(ring, TURNS_TRACKED) == 0.0

    def test_a_counter_jump_tracks_once(self, ring, monkeypatch):
        calls = []
        real = pyat_turns.track_shot

        def counting(*args, **kwargs):
            calls.append(kwargs.get("shot"))
            return real(*args, **kwargs)

        monkeypatch.setattr(pyat_turns, "track_shot", counting)
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 15.0)
        engine.advance(ring, 1.1)
        assert calls == [15]

    def test_advancing_again_without_a_delivery_tracks_nothing_new(self, ring, monkeypatch):
        _baselined(ring)
        _arm(ring, ARM_1)
        _arm(ring, ARM_2)
        _deliver(ring, 11.0)
        engine.emitted(ring)
        calls = []
        monkeypatch.setattr(pyat_turns, "track_shot", lambda *a, **k: calls.append(1))
        engine.advance(ring, 2.0)
        engine.advance(ring, 3.0)
        assert calls == []

    def test_the_stored_current_is_the_one_held_before_the_shot(self, ring):
        _baselined(ring, stored=2.0)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0, stored=9.0)
        assert _value(ring, SUM_1)[0] == pytest.approx(2.0)

    def test_the_same_shot_reads_the_same_arrays(self, demo, ring):
        other = engine.build(
            MODEL, [*demo["wiring"], *_turn_records()], demo["deck"], _settings(demo), {}
        )
        engine.advance(other, 0.0)
        for model in (ring, other):
            _baselined(model)
            _arm(model, ARM_1)
            _deliver(model, 11.0)
        for name in (X_1, Y_1, SUM_1):
            np.testing.assert_array_equal(_value(ring, name), _value(other, name))


OFFSET_X = "SR:INJ:OFFSET:X"
OFFSET_PY = "SR:INJ:OFFSET:PY"
MISMATCH_X = "SR:INJ:MISMATCH:X"
MISMATCH_Y = "SR:INJ:MISMATCH:Y"
STEERER = "SR:MAG:HCM:01:CURRENT:SP"
POSITION_X = "SR:DIAG:BPM:01:POSITION:X"


def _beam_records() -> list[dict[str, Any]]:
    return [
        _record("offset_x", OFFSET_X, write=True),
        _record("offset_py", OFFSET_PY, write=True),
        _record("mismatch_x", MISMATCH_X, write=True),
        _record("mismatch_y", MISMATCH_Y, write=True),
    ]


def _beam_ring(demo: dict[str, Any], active: dict[str, Any] | None = None, **block: Any) -> Any:
    settings = _settings(demo)
    settings["pyat"]["turn_by_turn"].update(block)
    records = [*demo["wiring"], *_turn_records(), *_beam_records()]
    # The facility build states each record's default: a mismatch starts at the block's.
    stated = block.get("mismatch", {})
    for record in records:
        if record["address"] in (MISMATCH_X, MISMATCH_Y):
            record["default"] = stated.get("x" if record["address"] == MISMATCH_X else "y", 1.0)
    model = engine.build(MODEL, records, demo["deck"], settings, active or {})
    engine.advance(model, 0.0)
    return model


def _first_shot(model: Any) -> tuple[np.ndarray, np.ndarray]:
    _baselined(model)
    _arm(model, ARM_1)
    _deliver(model, 11.0)
    return np.asarray(_value(model, X_1)), np.asarray(_value(model, Y_1))


class TestTheInjectedBeam:
    def test_an_offset_record_launches_as_the_same_launch_does(self, demo):
        stated = _beam_ring(demo, launch={"x": 1.0e-3, "py": -2.0e-4})
        written = _beam_ring(demo)
        written.set({OFFSET_X: 1.0e-3, OFFSET_PY: -2.0e-4})
        for one, other in zip(_first_shot(stated), _first_shot(written), strict=True):
            np.testing.assert_array_equal(one, other)

    def test_an_offset_adds_to_the_blocks_launch(self, demo):
        stated = _beam_ring(demo, launch={"x": 1.0e-3})
        both = _beam_ring(demo, launch={"x": 4.0e-4})
        both.set({OFFSET_X: 6.0e-4})
        np.testing.assert_allclose(_first_shot(stated)[0], _first_shot(both)[0], atol=1e-15)

    def test_an_offset_moves_the_first_turn(self, demo):
        plain, moved = _beam_ring(demo), _beam_ring(demo)
        moved.set({OFFSET_X: 1.0e-3})
        x, x_moved = _first_shot(plain)[0], _first_shot(moved)[0]
        assert abs(x_moved[TRIGGER] - x[TRIGGER]) > 1.0e-4

    def test_the_active_scenario_states_an_offset(self, demo):
        written = _beam_ring(demo)
        written.set({OFFSET_X: 1.0e-3})
        seeded = _beam_ring(demo, active={OFFSET_X: 1.0e-3})
        np.testing.assert_array_equal(_first_shot(written)[0], _first_shot(seeded)[0])

    def test_a_ring_without_beam_records_launches_the_block(self, demo, ring):
        with_records = _beam_ring(demo)
        np.testing.assert_array_equal(_first_shot(ring)[0], _first_shot(with_records)[0])

    def test_the_shot_takes_the_blocks_optics_and_the_records_mismatch(self, demo, monkeypatch):
        twiss = {"beta": [7.0, 3.0], "alpha": [0.4, -0.2]}
        model = _beam_ring(demo, emittance_x=2.0e-8, twiss=twiss, mismatch={"x": 1.5}, particles=4)
        assert _value(model, MISMATCH_X) == 1.5
        model.set({MISMATCH_Y: 0.5})
        seen: dict[str, Any] = {}
        declared = pyat_turns.declared_bunch

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return declared(*args, **kwargs)

        monkeypatch.setattr(pyat_turns, "declared_bunch", spy)
        _first_shot(model)
        assert tuple(seen["mismatch"]) == (1.5, 0.5)
        assert tuple(seen["beta"]) == (7.0, 3.0)
        assert tuple(seen["alpha"]) == (0.4, -0.2)
        assert seen["dispersion"] is None

    def test_a_stated_beam_is_tracked_on_a_ring_without_a_closed_orbit(self, demo):
        """A corrector far off leaves no periodic solution; the stated optics need none."""
        stated = {"beta": [7.0, 3.0], "alpha": [0.4, -0.2], "dispersion": [0.0, 0.0, 0.0, 0.0]}
        block = {"emittance_x": 2.0e-8, "particles": 4, "aperture": {"x": 0.03, "y": 0.03}}
        matched, arriving = _beam_ring(demo, **block), _beam_ring(demo, twiss=stated, **block)
        for model in (matched, arriving):
            model.set({STEERER: 3000.0})
            assert np.isnan(_value(model, POSITION_X))
        x, y = _first_shot(arriving)
        assert _value(arriving, TURNS_TRACKED) == TURNS
        assert np.asarray(_value(arriving, SUM_1))[TRIGGER] > 0.5
        assert np.isfinite(x[TRIGGER]) and np.isfinite(y[TRIGGER])
        assert x[TRIGGER] != 0.0
        # The ring's own optics are not there to match to: that shot fails.
        _first_shot(matched)
        assert _value(matched, TURNS_TRACKED) == 0
        assert np.all(np.asarray(_value(matched, SUM_1)) == 0.0)

    def test_a_mismatch_written_as_zero_fails_the_shot_not_the_ring(self, demo):
        model = _beam_ring(demo, emittance_x=2.0e-8, particles=4)
        model.set({MISMATCH_X: 0.0})
        _first_shot(model)
        assert _value(model, TURNS_TRACKED) == 0
        assert _value(model, ARMED_1) == 1.0  # the drop follows the drain, as for any shot

    def test_the_jitter_of_the_block_reaches_the_shot_in_six_coordinates(self, demo):
        quiet = _beam_ring(demo, jitter={"x": 0, "y": 0})
        kicked = _beam_ring(demo, jitter={"x": 0, "y": 0, "px": 1.0e-4})
        assert np.abs(_first_shot(quiet)[0] - _first_shot(kicked)[0]).max() > 1.0e-5


def _outcome(model: Any) -> dict[str, float]:
    return {name: float(_value(model, name)) for name in sorted(OUTCOME)}


class TestTheShotsOutcome:
    """What the ring reads of its last shot: how it ended, and how far the beam got."""

    def test_before_a_shot_every_read_out_is_zero(self, ring):
        assert set(_outcome(ring).values()) == {0.0}
        assert ShotStatus.NONE == 0

    def test_a_tracked_shot_reads_tracked_and_how_far_the_beam_got(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _arm(ring, ARM_2)
        _deliver(ring, 11.0)
        assert _outcome(ring) == {
            TURNS_TRACKED: TURNS,
            TURNS_SURVIVED: TURNS,
            MONITORS_REACHED: 2.0,
            SHOT_STATUS: ShotStatus.TRACKED,
        }

    def test_the_monitors_reached_are_counted_among_those_that_read_the_shot(self, ring):
        _baselined(ring)
        _arm(ring, ARM_2)
        _deliver(ring, 11.0)
        assert _value(ring, MONITORS_REACHED) == 1.0

    def test_a_beam_lost_on_the_first_turn_is_tracked_and_survives_no_turn(self, demo, caplog):
        model = _beam_ring(demo, launch={"x": 0.03}, aperture={"x": 0.01, "y": 0.01})
        _baselined(model)
        _arm(model, ARM_1)
        _arm(model, ARM_2)
        with caplog.at_level(logging.WARNING):
            _deliver(model, 11.0)
        assert _value(model, SHOT_STATUS) == ShotStatus.TRACKED
        assert _value(model, TURNS_TRACKED) == TURNS
        assert _value(model, TURNS_SURVIVED) == 0.0
        assert _value(model, MONITORS_REACHED) < 2.0
        assert not caplog.records

    def test_a_beam_cut_after_the_first_monitor_reaches_that_one_alone(self, demo):
        cut = {"class": "Quadrupole", "shape": "rectangular", "x": [-0.01, 0.0005], "y": 0.01}
        model = _beam_ring(
            demo, launch={"x": 2.0e-3}, aperture={"x": 0.05, "y": 0.05}, apertures=[cut]
        )
        _baselined(model)
        _arm(model, ARM_1)
        _arm(model, ARM_2)
        _deliver(model, 11.0)
        assert _value(model, SHOT_STATUS) == ShotStatus.TRACKED
        assert _value(model, TURNS_SURVIVED) == 0.0
        assert _value(model, MONITORS_REACHED) == 1.0
        total = np.asarray(_value(model, SUM_1))
        assert total[TRIGGER] > 0.5 and total[TRIGGER + 1] == 0.0

    def test_a_shot_that_cannot_be_set_up_says_so_and_logs_why(self, demo, caplog):
        """With no arriving optics stated, the bunch is matched to the ring's, which are gone."""
        model = _beam_ring(demo, emittance_x=2.0e-8, particles=4, aperture={"x": 0.03, "y": 0.03})
        model.set({STEERER: 3000.0})
        with caplog.at_level(logging.WARNING):
            _first_shot(model)
        assert _outcome(model) == {
            TURNS_TRACKED: 0.0,
            TURNS_SURVIVED: 0.0,
            MONITORS_REACHED: 0.0,
            SHOT_STATUS: ShotStatus.NOT_SET_UP,
        }
        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert "no periodic optics" in record.getMessage()
        assert f"{MODEL}: shot 11" in record.getMessage()
        assert not record.exc_info

    def test_a_mismatch_written_as_zero_is_a_shot_not_set_up(self, demo):
        model = _beam_ring(demo, emittance_x=2.0e-8, particles=4)
        model.set({MISMATCH_X: 0.0})
        _first_shot(model)
        assert _value(model, SHOT_STATUS) == ShotStatus.NOT_SET_UP

    @pytest.mark.parametrize("broken", ["track_shot", "monitor_turn_arrays"])
    def test_an_unexpected_error_says_so_and_logs_its_traceback(
        self, ring, monkeypatch, caplog, broken
    ):
        from osprey.simulation.engines import pyat_turn_readout

        def raising(*args, **kwargs):
            raise RuntimeError("the code broke")

        module = pyat_turns if broken == "track_shot" else pyat_turn_readout
        monkeypatch.setattr(module, broken, raising)
        _baselined(ring)
        _arm(ring, ARM_1)
        with caplog.at_level(logging.WARNING):
            _deliver(ring, 11.0)
        assert _outcome(ring) == {
            TURNS_TRACKED: 0.0,
            TURNS_SURVIVED: 0.0,
            MONITORS_REACHED: 0.0,
            SHOT_STATUS: ShotStatus.ERROR,
        }
        assert set(engine.emitted(ring)) == SHOT_NAMES_1
        (record,) = caplog.records
        assert record.levelno == logging.ERROR
        assert f"{MODEL}: shot 11" in record.getMessage()
        assert record.exc_info and record.exc_info[0] is RuntimeError

    def test_a_tracked_shot_after_a_failed_one_reads_tracked_again(self, ring, monkeypatch):
        with monkeypatch.context() as patch:
            patch.setattr(pyat_turns, "track_shot", lambda *a, **k: 1 / 0)
            _baselined(ring)
            _arm(ring, ARM_1)
            _deliver(ring, 11.0)
        assert _value(ring, SHOT_STATUS) == ShotStatus.ERROR
        engine.emitted(ring)
        engine.advance(ring, 1.5)
        _arm(ring, ARM_1, t_s=1.6)
        _deliver(ring, 12.0, t_s=2.0)
        assert _value(ring, SHOT_STATUS) == ShotStatus.TRACKED
        assert _value(ring, TURNS_SURVIVED) == TURNS

    def test_reset_returns_every_read_out_to_zero(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0)
        ring.reset()
        assert set(_outcome(ring).values()) == {0.0}


KICK = "SR:INJ:KICKER:SP"
KICKER = "HCM01"


def _kick_ring(demo: dict[str, Any], active: dict[str, Any] | None = None, **words: Any) -> Any:
    record = _record("pulsed_kick", KICK, element=KICKER, write=True)
    record["engine"].update({"plane": "x", "scale": 1.0e-4, **words})
    records = [*demo["wiring"], *_turn_records(), record]
    model = engine.build(MODEL, records, demo["deck"], _settings(demo), active or {})
    engine.advance(model, 0.0)
    return model


def _seen_kicks(model: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: dict[str, Any] = {}
    track = pyat_turns.track_shot

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return track(*args, **kwargs)

    monkeypatch.setattr(pyat_turns, "track_shot", spy)
    _first_shot(model)
    return list(seen["pulsed"])


class TestAPulsedKick:
    def test_the_shot_takes_the_written_kick_through_scale_and_pulse(self, demo, monkeypatch):
        model = _kick_ring(demo, pulse=[1.0, 0.5])
        model.set({KICK: 2.0})
        (kick,) = _seen_kicks(model, monkeypatch)
        assert kick.element == model.simulator.unique_element_index(KICKER)
        assert (kick.plane, kick.pulse) == (0, (1.0, 0.5))
        assert kick.angle == pytest.approx(2.0e-4)

    def test_scale_and_pulse_have_defaults(self, demo, monkeypatch):
        record = _record("pulsed_kick", KICK, element=KICKER, write=True)
        record["engine"]["plane"] = "y"
        records = [*demo["wiring"], *_turn_records(), record]
        model = engine.build(MODEL, records, demo["deck"], _settings(demo), {})
        engine.advance(model, 0.0)
        model.set({KICK: 3.0e-5})
        (kick,) = _seen_kicks(model, monkeypatch)
        assert (kick.plane, kick.angle, kick.pulse) == (1, 3.0e-5, (1.0,))

    def test_a_kick_at_zero_changes_no_shot(self, demo, ring):
        np.testing.assert_array_equal(_first_shot(ring)[0], _first_shot(_kick_ring(demo))[0])

    def test_a_one_turn_pulse_moves_the_first_turns_and_not_the_lattice(self, demo):
        plain, kicked = _kick_ring(demo), _kick_ring(demo)
        index = kicked.simulator.unique_element_index(KICKER)
        before = np.array(kicked.lattice[index].KickAngle, dtype=float)
        kicked.set({KICK: 1.0})
        x, x_kicked = _first_shot(plain)[0], _first_shot(kicked)[0]
        assert np.abs(x_kicked[TRIGGER:] - x[TRIGGER:]).max() > 1.0e-5
        np.testing.assert_array_equal(kicked.lattice[index].KickAngle, before)

    def test_the_active_scenario_states_the_kick(self, demo):
        written = _kick_ring(demo)
        written.set({KICK: 1.0})
        seeded = _kick_ring(demo, active={KICK: 1.0})
        np.testing.assert_array_equal(_first_shot(written)[0], _first_shot(seeded)[0])


class TestHandshake:
    def test_armed_drops_only_after_a_drain(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0)
        engine.advance(ring, 1.2)
        assert _value(ring, ARMED_1) == 1.0
        assert set(engine.emitted(ring)) == SHOT_NAMES_1
        assert _value(ring, ARMED_1) == 1.0
        engine.advance(ring, 1.3)
        assert _value(ring, ARMED_1) == 0.0
        assert engine.emitted(ring) == [ARMED_1]

    def test_a_shot_before_the_drain_overwrites_nothing(self, ring, monkeypatch):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0)
        calls = []
        monkeypatch.setattr(pyat_turns, "track_shot", lambda *a, **k: calls.append(1))
        _deliver(ring, 12.0, t_s=1.2)
        assert calls == []

    def test_the_drop_applies_before_a_new_shot_is_looked_for(self, ring, monkeypatch):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0)
        engine.emitted(ring)
        calls = []
        monkeypatch.setattr(pyat_turns, "track_shot", lambda *a, **k: calls.append(1))
        _deliver(ring, 12.0, t_s=2.0)
        assert calls == []
        assert engine.emitted(ring) == [ARMED_1]
        assert _value(ring, ARMED_1) == 0.0

    def test_next_deadline_is_due_while_the_handshake_is_pending(self, ring):
        assert engine.next_deadline(ring) is None
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0, t_s=4.0)
        assert engine.next_deadline(ring) == 4.0
        engine.emitted(ring)
        assert engine.next_deadline(ring) == 4.0
        engine.advance(ring, 5.0)
        engine.emitted(ring)
        assert engine.next_deadline(ring) is None

    def test_zero_charge_completes_the_handshake(self, ring):
        _baselined(ring, stored=1.5)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0, charge=0.0, stored=1.5)
        assert set(engine.emitted(ring)) == SHOT_NAMES_1
        total = np.asarray(_value(ring, SUM_1))
        assert np.all(np.isfinite(total))
        assert total[TRIGGER] == pytest.approx(total[0])
        engine.advance(ring, 2.0)
        assert engine.emitted(ring) == [ARMED_1]

    def test_a_raising_tracking_completes_the_handshake(self, ring, monkeypatch):
        _baselined(ring)
        _arm(ring, ARM_1)
        _deliver(ring, 11.0)
        assert _value(ring, TURNS_TRACKED) == TURNS
        engine.emitted(ring)
        engine.advance(ring, 1.5)
        engine.emitted(ring)
        _arm(ring, ARM_1, t_s=1.6)

        def broken(*args, **kwargs):
            raise RuntimeError("the tracking broke")

        monkeypatch.setattr(pyat_turns, "track_shot", broken)
        _deliver(ring, 12.0, t_s=2.0)
        assert set(engine.emitted(ring)) == SHOT_NAMES_1
        assert _value(ring, TURNS_TRACKED) == 0.0
        for name in ARRAYS_1:
            assert np.all(np.asarray(_value(ring, name)) == 0.0)
        engine.advance(ring, 2.5)
        assert engine.emitted(ring) == [ARMED_1]
        assert _value(ring, ARMED_1) == 0.0


class TestReset:
    def test_reset_restores_start_values(self, ring):
        _baselined(ring)
        _arm(ring, ARM_1)
        _arm(ring, ARM_2)
        _deliver(ring, 11.0)
        ring.reset()
        engine.advance(ring, 10.0)
        assert engine.emitted(ring) == []
        assert engine.next_deadline(ring) is None
        assert _value(ring, TURNS_TRACKED) == 0.0
        assert _value(ring, ARMED_1) == 0.0
        assert _value(ring, ARMED_2) == 0.0
        assert _value(ring, ARM_1) == 0.0
        for name in (*ARRAYS_1, X_2):
            assert np.all(np.asarray(_value(ring, name)) == 0.0)
        state = turn_state_of(ring)
        assert state.inputs is None and state.previous_inputs is None
        assert not state.emitted_pending

    def test_after_reset_the_first_delivery_is_no_shot(self, ring):
        _baselined(ring)
        ring.reset()
        engine.advance(ring, 10.0)
        _arm(ring, ARM_1, t_s=10.5)
        _deliver(ring, 50.0, t_s=11.0)
        assert engine.emitted(ring) == []


class TestAnotherRing:
    def test_a_ring_without_turn_by_turn_has_no_deadline_and_emits_nothing(self, plain_ring):
        engine.advance(plain_ring, 0.0)
        engine.take_inputs(
            plain_ring,
            [({"address": SHOT, "engine": {"signal": "shot"}}, 5.0)],
            1.0,
            baseline=False,
        )
        engine.advance(plain_ring, 1.0)
        engine.client_write(plain_ring, {"anything": 1.0}, 1.0)
        assert engine.next_deadline(plain_ring) is None
        assert engine.emitted(plain_ring) == []


#: BPM01's orbit readings, published in millimetres over a deck in metres.
class TestElementApertures:
    """An ``apertures`` item is written on the deck a shot is tracked through."""

    @staticmethod
    def _sum(demo: dict[str, Any], apertures: list[dict[str, Any]]) -> np.ndarray:
        settings = _settings(demo)
        settings["pyat"]["turn_by_turn"].update(
            launch={"x": 2.0e-3}, aperture={"x": 0.05, "y": 0.05}, apertures=apertures
        )
        model = engine.build(MODEL, [*demo["wiring"], *_turn_records()], demo["deck"], settings, {})
        engine.advance(model, 0.0)
        _baselined(model)
        _arm(model, ARM_1)
        _deliver(model, 11.0)
        return np.asarray(_value(model, SUM_1))[TRIGGER:]

    def test_the_sum_falls_where_an_elements_aperture_cuts_the_beam(self, demo):
        kept = self._sum(demo, [])
        cut = self._sum(
            demo,
            [{"class": "Quadrupole", "shape": "rectangular", "x": [-0.01, 0.001], "y": 0.01}],
        )
        assert np.all(kept > 0.5)
        # The first monitor reads turn 1 before the beam meets a quadrupole.
        assert np.all(cut[1:] < 0.5 * kept[1:])

    def test_a_later_item_wins_on_the_elements_both_select(self, demo):
        tight = {"class": "Quadrupole", "shape": "rectangular", "x": [-0.01, 0.001], "y": 0.01}
        wide = {"class": "Quadrupole", "x": 0.05, "y": 0.05}
        np.testing.assert_array_equal(self._sum(demo, [tight, wide]), self._sum(demo, []))


POSITION_X_1 = "SR:DIAG:BPM:01:POSITION:X"
POSITION_Y_1 = "SR:DIAG:BPM:01:POSITION:Y"
MM = 1.0e-3


class TestReadoutFaultsOnTurnArrays:
    """A fault on a monitor's orbit reading acts on that monitor's turn arrays alike."""

    @staticmethod
    def _shot(demo: dict[str, Any], faults: dict[str, float], *, shot: float = 11.0):
        settings = _settings(demo)
        settings["pyat"]["turn_by_turn"]["launch"] = {"x": 1.0e-3, "y": -5.0e-4}
        model = engine.build(
            MODEL, [*demo["wiring"], *_turn_records()], demo["deck"], settings, faults
        )
        engine.advance(model, 0.0)
        _baselined(model)
        _arm(model, ARM_1)
        _deliver(model, shot)
        x, y, total = (np.asarray(_value(model, name)) for name in (X_1, Y_1, SUM_1))
        return model, x, y, total

    def test_an_offset_is_subtracted_from_every_turn_in_the_arrays_unit(self, demo):
        _, x, y, total = self._shot(demo, {})
        _, x_off, y_off, total_off = self._shot(
            demo, {f"{POSITION_X_1}/offset": 0.5, f"{POSITION_Y_1}/offset": -0.2}
        )
        np.testing.assert_allclose(x_off[TRIGGER:], x[TRIGGER:] - 0.5 * MM, atol=1e-15)
        np.testing.assert_allclose(y_off[TRIGGER:], y[TRIGGER:] + 0.2 * MM, atol=1e-15)
        np.testing.assert_array_equal(total_off, total)

    def test_the_orbit_reading_and_the_turn_array_carry_the_one_offset(self, demo):
        fault = {f"{POSITION_X_1}/offset": 0.5}
        clean, x, _, _ = self._shot(demo, {})
        faulted, x_off, _, _ = self._shot(demo, fault)
        truth = {POSITION_X_1: 0.25, POSITION_Y_1: -0.15}
        orbit_shift = (
            engine.readout(faulted, truth, 1_000)[POSITION_X_1]
            - engine.readout(clean, truth, 1_000)[POSITION_X_1]
        )
        assert orbit_shift == pytest.approx(-0.5)
        np.testing.assert_allclose(x_off[TRIGGER:] - x[TRIGGER:], orbit_shift * MM, atol=1e-15)

    def test_the_chain_is_roll_then_offset_then_polarity_then_gain(self, demo):
        from osprey.simulation.engines.pyat_faults import rolled

        _, x, y, _ = self._shot(demo, {})
        _, x_read, y_read, _ = self._shot(
            demo,
            {
                f"{POSITION_X_1}/roll": 0.05,
                f"{POSITION_X_1}/offset": 0.5,
                f"{POSITION_X_1}/gain": 1.05,
                f"{POSITION_Y_1}/offset": -0.2,
                f"{POSITION_Y_1}/gain": 0.9,
                f"{POSITION_Y_1}/polarity": -1.0,
            },
        )
        turned_x, turned_y = rolled(x[TRIGGER:], y[TRIGGER:], 0.05)
        np.testing.assert_allclose(x_read[TRIGGER:], 1.05 * (turned_x - 0.5 * MM), atol=1e-15)
        np.testing.assert_allclose(y_read[TRIGGER:], -0.9 * (turned_y + 0.2 * MM), atol=1e-15)

    def test_a_noise_fault_adds_its_width_keyed_on_the_shot(self, demo):
        fault = {f"{POSITION_X_1}/noise": 0.3}
        _, x, y, _ = self._shot(demo, {})
        _, x_noisy, y_noisy, _ = self._shot(demo, fault)
        _, x_again, _, _ = self._shot(demo, fault)
        _, x_next, _, _ = self._shot(demo, fault, shot=12.0)
        _, x_clean_next, _, _ = self._shot(demo, {}, shot=12.0)
        added = (x_noisy - x)[TRIGGER:]
        assert 0.5 * 0.3 * MM < added.std() < 1.5 * 0.3 * MM
        np.testing.assert_array_equal(x_again, x_noisy)
        np.testing.assert_array_equal(y_noisy, y)
        assert not np.allclose((x_next - x_clean_next)[TRIGGER:], added)

    def test_where_no_beam_is_seen_the_arrays_read_the_floor_noise_alone(self, demo):
        _, x, y, _ = self._shot(demo, {})
        _, x_off, y_off, _ = self._shot(
            demo, {f"{POSITION_X_1}/offset": 0.5, f"{POSITION_X_1}/roll": 0.05}
        )
        assert np.any(x[:TRIGGER] != 0.0)
        np.testing.assert_array_equal(x_off[:TRIGGER], x[:TRIGGER])
        np.testing.assert_array_equal(y_off[:TRIGGER], y[:TRIGGER])

    def test_a_fault_on_one_monitor_leaves_the_next_monitors_arrays_alone(self, demo):
        def second(faults):
            settings = _settings(demo)
            model = engine.build(
                MODEL, [*demo["wiring"], *_turn_records()], demo["deck"], settings, faults
            )
            engine.advance(model, 0.0)
            _baselined(model)
            _arm(model, ARM_2)
            _deliver(model, 11.0)
            return np.asarray(_value(model, X_2))

        np.testing.assert_array_equal(second({f"{POSITION_X_1}/offset": 0.5}), second({}))
