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
from typing import Any

import numpy as np
import pytest

from osprey.facility.records import FacilityRecords
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines import pyat_turns
from osprey.simulation.engines.pyat_turn_state import turn_state_of

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

SHOT = "INJ:SHOT:COUNTER"
CHARGE = "INJ:CHARGE"
STORED = "SR:DCCT:CURRENT"

ARRAYS_1 = {X_1, Y_1, SUM_1}
SHOT_NAMES_1 = {X_1, Y_1, SUM_1, TURNS_TRACKED}


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
