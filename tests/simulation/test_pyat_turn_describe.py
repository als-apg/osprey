"""The pyat engine's turn-by-turn vocabulary: settings block, descriptions, inputs, start values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import at
import pytest

from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines import pyat_turn_settings as turn
from osprey_facility.errors import FacilityBuildError

TURNS = 20
TRIGGER = 4
SAMPLES = TRIGGER + TURNS


def _save(tmp_path: Path, elements: list[Any], name: str = "deck") -> Path:
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    path = tmp_path / f"{name}.json"
    at.save_lattice(lattice, str(path))
    return path


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    """An injection marker INJ, a quadrupole Q, monitors BPM1 and BPM2 (and a repeated D)."""
    return _save(
        tmp_path,
        [
            at.Marker("INJ"),
            at.Drift("D", 1.0),
            at.Quadrupole("Q", 0.2, 1.1),
            at.Monitor("BPM1"),
            at.Drift("D", 1.0),
            at.Monitor("BPM2"),
        ],
    )


def _settings(**block: Any) -> dict[str, Any]:
    stated = {"injection_element": "INJ", "turns": TURNS, "trigger_sample": TRIGGER}
    stated.update(block)
    return {"pyat": {turn.BLOCK_KEY: stated}}


def _record(signal: str, address: str, *, element: str | None = None, write: bool = False):
    record: dict[str, Any] = {
        "id": f"SR/{address}",
        "address": address,
        "engine": {"signal": signal},
    }
    if element is not None:
        record["element"] = element
    if write:
        record["direction"] = "write"
    return record


def _kick(address: str, *, element: str | None = "Q", write: bool = True, **words: Any):
    record = _record("pulsed_kick", address, element=element, write=write)
    record["engine"].update(words)
    return record


def kind_of(record: dict[str, Any]) -> str:
    return str(record["engine"]["signal"])


# --- The settings block ---------------------------------------------------------------


class TestIsTurnByTurn:
    def test_no_block_is_todays_ring(self) -> None:
        assert turn.is_turn_by_turn({"pyat": {"solve": "periodic"}}) is False
        assert turn.is_turn_by_turn(None) is False
        assert turn.is_turn_by_turn({}) is False

    def test_block_makes_a_turn_by_turn_ring(self) -> None:
        assert turn.is_turn_by_turn(_settings()) is True

    def test_importable_without_pyat_names(self) -> None:
        # The helper reads raw settings only; it takes no deck.
        assert turn.is_turn_by_turn({"pyat": {turn.BLOCK_KEY: {"injection_element": "X"}}})


class TestPrepareBlock:
    def test_defaults(self, deck: Path) -> None:
        prepared = engine.prepare(deck, {"pyat": {turn.BLOCK_KEY: {"injection_element": "INJ"}}})
        block = prepared.turn_by_turn
        assert isinstance(block, turn.TurnSettings)
        assert block.injection_element == "INJ"
        assert block.injection_index == 0
        assert block.turns == turn.DEFAULT_TURNS
        assert block.trigger_sample == turn.DEFAULT_TRIGGER_SAMPLE
        assert block.samples == turn.DEFAULT_TRIGGER_SAMPLE + turn.DEFAULT_TURNS
        assert block.particles == turn.DEFAULT_PARTICLES
        assert block.launch == (0.0,) * 6
        assert block.jitter == turn.DEFAULT_JITTER == (5e-5, 0.0, 3e-5, 0.0, 0.0, 0.0)
        assert block.twiss == turn.InjectedTwiss(None, None, None)
        assert block.mismatch == (1.0, 1.0)
        assert block.aperture is None
        assert block.readout.tone_period == turn.DEFAULT_TONE_PERIOD

    def test_todays_ring_has_no_block(self, deck: Path) -> None:
        assert engine.prepare(deck, {}).turn_by_turn is None

    def test_normalises_every_key(self, deck: Path) -> None:
        settings = _settings(
            injection_element="Q",
            particles=8,
            launch={"x": 1e-3, "py": -2e-4, "delta": 1e-3},
            emittance_x=2e-9,
            emittance_y=1e-11,
            sigma_delta=1e-3,
            sigma_z=3e-3,
            jitter={"x": 1e-5, "y": 0, "px": 2e-6, "delta": 1e-4, "ct": 3e-4},
            twiss={"beta": [7.0, 3.0], "alpha": [0.4, -0.2], "dispersion": [0.1, 0, 0, 0]},
            mismatch={"x": 1.5},
            aperture={"x": 3e-3, "y": 5e-3, "shape": "rectangular"},
            sum_per_charge=2,
            sum_per_stored=0.5,
            sum_floor=0.01,
            position_noise=1e-5,
            sum_noise=1e-3,
            floor_noise=1e-3,
            tone_period=7,
            tone_amplitude=0.5,
        )
        block = engine.prepare(deck, settings).turn_by_turn
        assert block.injection_index == 2
        assert block.samples == SAMPLES
        assert block.particles == 8
        assert block.launch == (1e-3, 0.0, 0.0, -2e-4, 1e-3, 0.0)
        assert (block.emittance_x, block.emittance_y) == (2e-9, 1e-11)
        assert (block.sigma_delta, block.sigma_z) == (1e-3, 3e-3)
        assert block.jitter == (1e-5, 2e-6, 0.0, 0.0, 1e-4, 3e-4)
        assert block.twiss == turn.InjectedTwiss(
            beta=(7.0, 3.0), alpha=(0.4, -0.2), dispersion=(0.1, 0.0, 0.0, 0.0)
        )
        assert block.mismatch == (1.5, 1.0)
        assert block.aperture == turn.Aperture("rectangular", 3e-3, 5e-3)
        readout = block.readout
        assert readout.sum_per_charge == 2.0 and isinstance(readout.sum_per_charge, float)
        assert readout.sum_per_stored == 0.5
        assert readout.sum_floor == 0.01
        assert (readout.position_noise, readout.sum_noise, readout.floor_noise) == (
            1e-5,
            1e-3,
            1e-3,
        )
        assert (readout.tone_period, readout.tone_amplitude) == (7, 0.5)

    def test_pyat_unknown_key_remedy_names_the_block(self, deck: Path) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(deck, {"pyat": {"bogus": 1}})
        assert turn.BLOCK_KEY in caught.value.remedy

    @pytest.mark.parametrize(
        ("block", "detail"),
        [
            ("yes", f"settings key pyat.{turn.BLOCK_KEY} is not a map"),
            ({"injection_element": "INJ", "turnz": 3}, "carries unknown keys turnz"),
            ({"turns": 3}, "lacks injection_element"),
            ({"injection_element": ""}, "injection_element is ''"),
            ({"injection_element": "INJ", "turns": 0}, ".turns is 0"),
            ({"injection_element": "INJ", "turns": 2.5}, ".turns is 2.5"),
            ({"injection_element": "INJ", "turns": True}, ".turns is True"),
            ({"injection_element": "INJ", "trigger_sample": -1}, ".trigger_sample is -1"),
            ({"injection_element": "INJ", "particles": 0}, ".particles is 0"),
            ({"injection_element": "INJ", "launch": [0, 0]}, ".launch is not a map"),
            ({"injection_element": "INJ", "launch": {"z": 1}}, "launch carries unknown keys z"),
            ({"injection_element": "INJ", "launch": {"x": "a"}}, ".launch.x is 'a'"),
            ({"injection_element": "INJ", "emittance_x": -1e-9}, ".emittance_x is -1e-09"),
            ({"injection_element": "INJ", "sigma_z": float("nan")}, ".sigma_z is nan"),
            ({"injection_element": "INJ", "jitter": {"x": -1}}, ".jitter.x is -1"),
            ({"injection_element": "INJ", "jitter": {"q": 1}}, "jitter carries unknown keys q"),
            ({"injection_element": "INJ", "jitter": {"delta": -1e-4}}, ".jitter.delta is -0.0001"),
            ({"injection_element": "INJ", "mismatch": {"x": 0}}, ".mismatch is {'x': 0}"),
            ({"injection_element": "INJ", "mismatch": {"x": -2}}, ".mismatch.x is -2"),
            ({"injection_element": "INJ", "mismatch": {"z": 2}}, "mismatch carries unknown keys z"),
            ({"injection_element": "INJ", "mismatch": 1.5}, ".mismatch is not a map"),
            ({"injection_element": "INJ", "twiss": {"beta": [1.0]}}, ".twiss.beta is [1.0]"),
            ({"injection_element": "INJ", "twiss": {"beta": [0, 1]}}, ".twiss.beta is [0, 1]"),
            ({"injection_element": "INJ", "twiss": {"beta": 4.0}}, ".twiss.beta is 4.0"),
            (
                {"injection_element": "INJ", "twiss": {"alpha": [0, "a"]}},
                ".twiss.alpha is 'a'",
            ),
            (
                {"injection_element": "INJ", "twiss": {"dispersion": [0, 0]}},
                ".twiss.dispersion is [0, 0]",
            ),
            ({"injection_element": "INJ", "twiss": {"gamma": [1, 1]}}, "twiss carries unknown"),
            ({"injection_element": "INJ", "aperture": {"x": 1e-3}}, ".aperture lacks y"),
            ({"injection_element": "INJ", "aperture": {"x": 0, "y": 1}}, ".aperture.x is 0"),
            (
                {"injection_element": "INJ", "aperture": {"x": 1, "y": 1, "shape": "round"}},
                ".aperture.shape is 'round'",
            ),
            ({"injection_element": "INJ", "sum_floor": -0.1}, ".sum_floor is -0.1"),
            ({"injection_element": "INJ", "position_noise": "x"}, ".position_noise is 'x'"),
            ({"injection_element": "INJ", "tone_period": 0}, ".tone_period is 0"),
            ({"injection_element": "INJ", "tone_amplitude": -1}, ".tone_amplitude is -1"),
        ],
        ids=lambda value: str(value)[:40],
    )
    def test_refusals(self, deck: Path, block: Any, detail: str) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(deck, {"pyat": {turn.BLOCK_KEY: block}}, model="SR")
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "SR"
        assert caught.value.record_kind == "model"
        assert detail in caught.value.detail

    def test_an_aperture_may_state_a_rectangles_limits(self, deck: Path) -> None:
        stated = {"shape": "rectangular", "x": [-0.03, 0.006], "y": 0.01}
        block = engine.prepare(deck, _settings(aperture=stated)).turn_by_turn
        assert block.aperture == turn.Aperture("rectangular", (-0.03, 0.006), 0.01)
        assert block.aperture.keywords == {"rectangular": ((-0.03, 0.006), 0.01)}

    def test_no_apertures_list_is_no_element_aperture(self, deck: Path) -> None:
        assert engine.prepare(deck, _settings()).turn_by_turn.apertures == ()

    def test_each_apertures_item_selects_its_elements_in_the_deck(self, deck: Path) -> None:
        # Deck: INJ, D [0, 1], Q [1, 1.2], BPM1, D [1.2, 2.2], BPM2.
        items = [
            {"class": "Drift", "x": 0.02, "y": 0.01},
            {"element": "BPM1", "shape": "rectangular", "x": [-0.03, 0.006], "y": 0.02},
            {"s": [1.0, 1.5], "x": 0.004, "y": 0.003},
        ]
        block = engine.prepare(deck, _settings(apertures=items)).turn_by_turn
        assert block.apertures == (
            turn.ElementAperture(turn.Aperture("elliptical", 0.02, 0.01), (1, 4)),
            turn.ElementAperture(turn.Aperture("rectangular", (-0.03, 0.006), 0.02), (3,)),
            turn.ElementAperture(turn.Aperture("elliptical", 0.004, 0.003), (2,)),
        )

    def test_a_class_selects_its_subclasses_with_a_length(self, deck: Path) -> None:
        items = [{"class": "Element", "x": 0.02, "y": 0.01}]
        block = engine.prepare(deck, _settings(apertures=items)).turn_by_turn
        assert block.apertures[0].indexes == (1, 2, 4)

    @pytest.mark.parametrize(
        ("items", "detail"),
        [
            ({"class": "Drift"}, ".apertures is not a list"),
            (["Drift"], ".apertures[0] is not a map"),
            ([{"x": 1e-3, "y": 1e-3}], ".apertures[0] names none of class, element, s"),
            (
                [{"class": "Drift", "element": "D", "x": 1e-3, "y": 1e-3}],
                ".apertures[0] names class and element",
            ),
            ([{"class": "Drift", "x": 1e-3, "y": 1e-3, "r": 1}], "carries unknown keys r"),
            ([{"class": "Drift", "x": 1e-3}], ".apertures[0] lacks y"),
            ([{"class": "Drift", "x": 0, "y": 1e-3}], ".apertures[0].x is 0"),
            ([{"class": "Drift", "x": [-1e-3, 2e-3], "y": 1e-3}], "limits on an ellipse"),
            (
                [{"class": "Drift", "shape": "rectangular", "x": [2e-3, 1e-3], "y": 1e-3}],
                ".apertures[0].x is [0.002, 0.001]",
            ),
            (
                [{"class": "Drift", "shape": "rectangular", "x": [1e-3], "y": 1e-3}],
                ".apertures[0].x is [0.001]",
            ),
            ([{"class": "Drift", "shape": "round", "x": 1, "y": 1}], ".shape is 'round'"),
            ([{"class": 3, "x": 1e-3, "y": 1e-3}], ".apertures[0].class is 3"),
            ([{"class": "Wedge", "x": 1e-3, "y": 1e-3}], "class Wedge is no element class"),
            ([{"class": "Sextupole", "x": 1e-3, "y": 1e-3}], "selects no element"),
            ([{"class": "Marker", "x": 1e-3, "y": 1e-3}], "selects no element"),
            ([{"element": "NOPE", "x": 1e-3, "y": 1e-3}], "selects no element"),
            ([{"s": [1.0], "x": 1e-3, "y": 1e-3}], ".apertures[0].s is [1.0]"),
            ([{"s": [1.5, 1.0], "x": 1e-3, "y": 1e-3}], ".apertures[0].s is [1.5, 1.0]"),
            ([{"s": [-1.0, 1.0], "x": 1e-3, "y": 1e-3}], ".apertures[0].s is [-1.0, 1.0]"),
            ([{"s": [0.0, 9.0], "x": 1e-3, "y": 1e-3}], "ends beyond the deck"),
            ([{"s": [0.6, 0.9], "x": 1e-3, "y": 1e-3}], "selects no element"),
        ],
        ids=lambda value: str(value)[:48],
    )
    def test_apertures_refusals(self, deck: Path, items: Any, detail: str) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(deck, _settings(apertures=items), model="SR")
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "SR"
        assert detail in caught.value.detail

    @pytest.mark.parametrize(("element", "count"), [("NOPE", 0), ("D", 2)])
    def test_injection_element_held_once(self, deck: Path, element: str, count: int) -> None:
        with pytest.raises(engine.ElementStop) as caught:
            engine.prepare(deck, _settings(injection_element=element), model="SR")
        assert caught.value.element == element
        assert caught.value.count == count
        assert caught.value.record_id == "SR"
        assert f"pyat.{turn.BLOCK_KEY}.injection_element" in caught.value.detail


# --- Descriptions ---------------------------------------------------------------------


class TestDescribe:
    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            (_record("arm", "B1:ARM", element="BPM1", write=True), ("setpoint", "pass", "arm")),
            (_record("armed", "B1:ARMED", element="BPM1"), ("output", "event", "armed")),
            (_record("turn_x", "B1:TBT:X", element="BPM1"), ("output", "event", "turn_x")),
            (_record("turn_y", "B1:TBT:Y", element="BPM1"), ("output", "event", "turn_y")),
            (_record("turn_sum", "B1:TBT:S", element="BPM1"), ("output", "event", "turn_sum")),
            (_record("turns_tracked", "SR:TURNS"), ("output", "event", "turns_tracked")),
            (_record("turns_survived", "SR:SURVIVED"), ("output", "event", "turns_survived")),
            (_record("monitors_reached", "SR:REACHED"), ("output", "event", "monitors_reached")),
            (_record("shot_status", "SR:STATUS"), ("output", "event", "shot_status")),
            (_record("rf_switch", "SR:RF", write=True), ("setpoint", "pass", "rf_switch")),
            (_record("offset_x", "SR:INJ:DX", write=True), ("setpoint", "pass", "offset_x")),
            (_record("offset_ct", "SR:INJ:DCT", write=True), ("setpoint", "pass", "offset_ct")),
            (_record("mismatch_y", "SR:INJ:MY", write=True), ("setpoint", "pass", "mismatch_y")),
            (_record("pilot_tone", "SR:TONE", write=True), ("setpoint", "pass", "pilot_tone")),
            (_record("stored", "SR:BUSY", write=True), ("setpoint", "pass", "stored")),
            (
                _record("stored", "B1:MODE", element="BPM1", write=True),
                ("setpoint", "pass", "stored"),
            ),
        ],
        ids=lambda value: value["address"] if isinstance(value, dict) else "",
    )
    def test_signal(self, record: dict[str, Any], expected: tuple[str, str, str]) -> None:
        described = engine.describe(record)
        marks = {"marks"} if kind_of(record) == "arm" else set()
        assert set(described) == {"engine_role", "plane", "refresh", "kind"} | marks
        role, refresh, kind = expected
        assert described["engine_role"] == role
        assert described["plane"] is None
        assert described["refresh"] == refresh
        assert described["kind"] == kind

    def test_arm_alone_carries_the_command_mark(self) -> None:
        arm = engine.describe(_record("arm", "B1:ARM", element="BPM1", write=True))
        assert arm["marks"] == ["command"]
        for signal in ("rf_switch", "pilot_tone", "stored"):
            assert "marks" not in engine.describe(_record(signal, "SR:SET", write=True))

    def test_every_signal_is_described(self) -> None:
        assert set(turn.SIGNALS) == {
            "arm",
            "armed",
            "turn_x",
            "turn_y",
            "turn_sum",
            "turns_tracked",
            "turns_survived",
            "monitors_reached",
            "shot_status",
            "rf_switch",
            "pilot_tone",
            "stored",
            *(f"offset_{key}" for key in turn.LAUNCH_KEYS),
            "mismatch_x",
            "mismatch_y",
            "pulsed_kick",
        }
        assert turn.BEAM_SIGNALS <= turn.SETTING_SIGNALS & turn.RING_SIGNALS
        assert set(turn.SHOT_SIGNALS) <= turn.RING_SIGNALS - turn.SETTING_SIGNALS

    @pytest.mark.parametrize(
        ("record", "text"),
        [
            (_record("fire", "X:F", write=True), "unknown signal 'fire'"),
            (
                _record("arm", "B1:ARM", element="BPM1"),
                "is a setting; wire it with direction write",
            ),
            (_record("rf_switch", "SR:RF"), "is a setting; wire it with direction write"),
            (_record("armed", "B1:ARMED", element="BPM1", write=True), "is read only"),
            (_record("turn_x", "B1:TBT:X", element="BPM1", write=True), "is read only"),
            (_record("turns_tracked", "SR:TURNS", write=True), "is read only"),
            (_record("arm", "B1:ARM", write=True), "names no monitor element"),
            (_record("armed", "B1:ARMED"), "names no monitor element"),
            (_record("turn_sum", "B1:TBT:S"), "names no monitor element"),
            (_record("turns_tracked", "SR:TURNS", element="BPM1"), "names an element"),
            (_record("turns_survived", "SR:SURVIVED", element="BPM1"), "names an element"),
            (_record("monitors_reached", "SR:REACHED", element="BPM1"), "names an element"),
            (_record("shot_status", "SR:STATUS", element="BPM1"), "names an element"),
            (_record("shot_status", "SR:STATUS", write=True), "is read only"),
            (_record("rf_switch", "SR:RF", element="Q", write=True), "names an element"),
            (_record("pilot_tone", "SR:TONE", element="Q", write=True), "names an element"),
            (_record("offset_px", "SR:INJ:DPX", element="Q", write=True), "names an element"),
            (_record("mismatch_x", "SR:INJ:MX"), "is a setting; wire it with direction write"),
            (_record("offset_z", "SR:INJ:DZ", write=True), "unknown signal 'offset_z'"),
        ],
        ids=lambda value: value["address"] if isinstance(value, dict) else str(value)[:30],
    )
    def test_refusals(self, record: dict[str, Any], text: str) -> None:
        with pytest.raises(ValueError, match=record["address"]) as caught:
            engine.describe(record)
        assert text in str(caught.value)

    @pytest.mark.parametrize(
        ("signal", "element", "write"),
        [
            ("turn_x", "BPM1", False),
            ("turn_sum", "BPM1", False),
            ("armed", "BPM1", False),
            ("turns_tracked", None, False),
            ("stored", None, True),
            ("offset_x", None, True),
        ],
    )
    def test_a_calibration_is_refused_on_every_signal(
        self, signal: str, element: str | None, write: bool
    ) -> None:
        record = _record(signal, "SR:TBT:RECORD", element=element, write=write)
        engine.describe(record)
        record["calibration"] = {"curve": {"linear": {"gain": 1000.0, "offset": 0.0}}}
        with pytest.raises(ValueError, match="SR:TBT:RECORD") as caught:
            engine.describe(record)
        assert f"signal {signal} states a calibration" in str(caught.value)
        assert "remove the calibration" in str(caught.value)
        assert ("fault `gain`" in str(caught.value)) == (signal == "turn_sum")

    def test_signal_beside_other_engine_words_is_refused(self) -> None:
        record = _record("arm", "B1:ARM", element="BPM1", write=True)
        record["engine"]["attribute"] = "KickAngle"
        with pytest.raises(ValueError, match="B1:ARM") as caught:
            engine.describe(record)
        assert "attribute" in str(caught.value)

    def test_a_pulsed_kick_is_a_setting_of_its_element(self) -> None:
        described = engine.describe(_kick("K:KICK", plane="x", scale=2.0e-4, pulse=[1.0, 0.5]))
        assert described["engine_role"] == "setpoint"
        assert described["kind"] == "pulsed_kick"
        assert described["plane"] is None  # no corrector: the closed orbit never sees it
        assert engine.describe(_kick("K:KICK", plane="y"))["refresh"] == "pass"

    @pytest.mark.parametrize(
        ("record", "text"),
        [
            (_kick("K:NONE", element=None, plane="x"), "names no element"),
            (_kick("K:READ", write=False, plane="x"), "is a setting; wire it with direction write"),
            (_kick("K:PLANE"), "states no plane; use x or y"),
            (_kick("K:Z", plane="z"), "plane 'z'; use x or y"),
            (_kick("K:SCALE", plane="x", scale="big"), "scale 'big' is not a number"),
            (_kick("K:NAN", plane="x", scale=float("nan")), "is not a number"),
            (_kick("K:EMPTY", plane="x", pulse=[]), "pulse [] is not a list of numbers"),
            (_kick("K:WORD", plane="x", pulse=[1.0, "half"]), "is not a list of numbers"),
            (_kick("K:ONE", plane="x", pulse=1.0), "is not a list of numbers"),
            (_kick("K:MORE", plane="x", attribute="KickAngle"), "beside attribute"),
        ],
        ids=lambda value: value["address"] if isinstance(value, dict) else str(value)[:30],
    )
    def test_pulsed_kick_refusals(self, record: dict[str, Any], text: str) -> None:
        with pytest.raises(ValueError, match=record["address"]) as caught:
            engine.describe(record)
        assert text in str(caught.value)

    def test_a_kicks_words_are_refused_on_another_signal(self) -> None:
        record = _record("pilot_tone", "SR:TONE", write=True)
        record["engine"]["plane"] = "x"
        with pytest.raises(ValueError, match="beside plane"):
            engine.describe(record)

    def test_slices_are_no_monitor_element(self) -> None:
        record = _record("turn_x", "B1:TBT:X")
        record["slices"] = [{"element": "BPM1", "weight": 1.0}]
        with pytest.raises(ValueError, match="names no monitor element"):
            engine.describe(record)

    def test_todays_records_are_unchanged(self) -> None:
        monitor = {"id": "m", "address": "B:X", "element": "BPM1", "engine": {"axis": "x"}}
        assert engine.describe(monitor)["engine_role"] == "monitor"
        assert engine.describe(monitor)["plane"] == "x"


# --- Inputs ---------------------------------------------------------------------------


class TestDescribeInput:
    @pytest.mark.parametrize("signal", ["shot", "charge", "stored_current"])
    def test_accepted(self, signal: str) -> None:
        described = engine.describe_input({"address": "INJ:SHOT", "engine": {"signal": signal}})
        assert described == {"signal": signal}

    @pytest.mark.parametrize(
        ("block", "text"),
        [
            (None, "is not a map"),
            ("shot", "is not a map"),
            ({}, "states no signal"),
            ({"signal": "fire"}, "unknown signal 'fire'"),
            ({"signal": "shot", "gain": 2}, "carries unknown keys gain"),
            ({"signal": "arm"}, "unknown signal 'arm'"),
        ],
        ids=lambda value: str(value)[:30],
    )
    def test_refused(self, block: Any, text: str) -> None:
        with pytest.raises(ValueError, match="INJ:SHOT") as caught:
            engine.describe_input({"address": "INJ:SHOT", "engine": block})
        assert text in str(caught.value)


class TestInputSet:
    def _inputs(self, *signals: str) -> list[dict[str, Any]]:
        return [
            {"address": f"P:{index}", "engine": {"signal": signal}}
            for index, signal in enumerate(signals)
        ]

    def test_complete_set(self) -> None:
        found = turn.input_signals(self._inputs("charge", "shot", "stored_current"))
        assert found == {"charge": "P:0", "shot": "P:1", "stored_current": "P:2"}

    def test_no_inputs_is_allowed(self) -> None:
        assert turn.input_signals([]) == {}

    def test_shot_and_charge_suffice(self) -> None:
        assert set(turn.input_signals(self._inputs("shot", "charge"))) == {"shot", "charge"}

    @pytest.mark.parametrize(
        ("signals", "text"),
        [
            (("shot", "charge", "shot"), "signal shot is declared twice: P:0, P:2"),
            (("shot",), "lack charge"),
            (("stored_current",), "lack charge, shot"),
        ],
    )
    def test_refused(self, signals: tuple[str, ...], text: str) -> None:
        with pytest.raises(ValueError) as caught:
            turn.input_signals(self._inputs(*signals))
        assert text in str(caught.value)


# --- Start values ---------------------------------------------------------------------


def _turn_wiring() -> list[dict[str, Any]]:
    return [
        _record("arm", "B1:ARM", element="BPM1", write=True),
        _record("armed", "B1:ARMED", element="BPM1"),
        _record("turn_x", "B1:TBT:X", element="BPM1"),
        _record("turn_y", "B1:TBT:Y", element="BPM1"),
        _record("turn_sum", "B1:TBT:S", element="BPM1"),
        _record("turns_tracked", "SR:TURNS"),
        _record("rf_switch", "SR:RF", write=True),
        _record("pilot_tone", "SR:TONE", write=True),
        _record("stored", "SR:BUSY", write=True),
    ]


ARRAYS = ("B1:TBT:X", "B1:TBT:Y", "B1:TBT:S")


def _types() -> dict[str, str]:
    return dict.fromkeys(ARRAYS, "waveform")


def _shapes(length: int = SAMPLES) -> dict[str, list[int]]:
    return {address: [length] for address in ARRAYS}


class TestStartValues:
    def test_an_offset_starts_at_zero_and_a_mismatch_at_the_blocks(self, deck: Path) -> None:
        records = [
            _record("offset_x", "SR:INJ:DX", write=True),
            _record("offset_delta", "SR:INJ:DE", write=True),
            _record("mismatch_x", "SR:INJ:MX", write=True),
            _record("mismatch_y", "SR:INJ:MY", write=True),
        ]
        values = engine.start_values(
            deck,
            records,
            _settings(launch={"x": 1e-3}, mismatch={"y": 1.5}),
            readbacks={},
            value_types={},
            shapes={},
        )
        assert values == {
            "SR:INJ:DE": 0.0,
            "SR:INJ:DX": 0.0,
            "SR:INJ:MX": 1.0,
            "SR:INJ:MY": 1.5,
        }

    def test_every_record_has_a_start_value(self, deck: Path) -> None:
        orbit = {"id": "SR/B1:X", "address": "B1:X", "element": "BPM1", "engine": {"axis": "x"}}
        values = engine.start_values(
            deck,
            [*_turn_wiring(), orbit],
            _settings(),
            readbacks={"B1:X": None},
            value_types=_types(),
            shapes=_shapes(),
        )
        assert list(values) == sorted(values)
        for address in ARRAYS:
            assert values[address] == [0.0] * SAMPLES
        assert values["B1:ARM"] == 0.0
        assert values["B1:ARMED"] == 0.0
        assert values["SR:TURNS"] == 0.0
        assert values["SR:RF"] == 1.0
        assert values["SR:TONE"] == 0.0
        assert values["SR:BUSY"] == 0.0
        assert values["B1:X"] == 0.0

    def test_default_length(self, deck: Path) -> None:
        settings = {"pyat": {turn.BLOCK_KEY: {"injection_element": "INJ"}}}
        length = turn.DEFAULT_TRIGGER_SAMPLE + turn.DEFAULT_TURNS
        values = engine.start_values(
            deck,
            [_record("turn_x", "B1:TBT:X", element="BPM1")],
            settings,
            value_types={"B1:TBT:X": "waveform"},
            shapes={"B1:TBT:X": [length]},
        )
        assert values == {"B1:TBT:X": [0.0] * length}

    @pytest.mark.parametrize("shape", [[SAMPLES - 1], [SAMPLES, 1], []])
    def test_declared_length_must_match(self, deck: Path, shape: list[int]) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                deck,
                _turn_wiring(),
                _settings(),
                value_types=_types(),
                shapes={**_shapes(), "B1:TBT:Y": shape},
            )
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "SR/B1:TBT:Y"
        assert caught.value.record_kind == "wiring"
        assert f"shape {shape}" in caught.value.detail
        assert f"[{SAMPLES}]" in caught.value.remedy

    def test_array_must_be_a_waveform(self, deck: Path) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                deck,
                [_record("turn_sum", "B1:TBT:S", element="BPM1")],
                _settings(),
            )
        assert caught.value.record_id == "SR/B1:TBT:S"
        assert "B1:TBT:S is float; a turn array is a waveform" in caught.value.detail

    def test_scalar_must_be_float(self, deck: Path) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                deck,
                [_record("armed", "B1:ARMED", element="BPM1")],
                _settings(),
                value_types={"B1:ARMED": "waveform"},
            )
        assert caught.value.record_id == "SR/B1:ARMED"
        assert "pyat drives float channels only" in caught.value.detail

    @pytest.mark.parametrize("index", range(9))
    def test_refused_on_a_ring_without_the_block(self, deck: Path, index: int) -> None:
        record = _turn_wiring()[index]
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                deck, [record], {"pyat": {}}, value_types=_types(), shapes=_shapes()
            )
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == record["id"]
        assert caught.value.record_kind == "wiring"
        assert f"settings state no pyat.{turn.BLOCK_KEY}" in caught.value.detail

    @pytest.mark.parametrize(("element", "count"), [("NOPE", 0), ("D", 2)])
    def test_monitor_element_held_once(self, deck: Path, element: str, count: int) -> None:
        with pytest.raises(engine.ElementStop) as caught:
            engine.start_values(deck, [_record("armed", "X:ARMED", element=element)], _settings())
        assert caught.value.record_id == "SR/X:ARMED"
        assert caught.value.count == count

    def test_element_must_be_a_monitor(self, deck: Path) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                deck, [_record("arm", "Q:ARM", element="Q", write=True)], _settings()
            )
        assert caught.value.record_id == "SR/Q:ARM"
        assert "element Q is not a monitor" in caught.value.detail

    def test_stored_may_name_any_element(self, deck: Path) -> None:
        values = engine.start_values(
            deck, [_record("stored", "Q:MODE", element="Q", write=True)], _settings()
        )
        assert values == {"Q:MODE": 0.0}

    def test_a_pulsed_kick_starts_at_zero(self, deck: Path) -> None:
        values = engine.start_values(deck, [_kick("K:KICK", plane="x")], _settings())
        assert values == {"K:KICK": 0.0}

    @pytest.mark.parametrize(("element", "count"), [("NOPE", 0), ("D", 2)])
    def test_a_pulsed_kicks_element_is_held_once(
        self, deck: Path, element: str, count: int
    ) -> None:
        with pytest.raises(engine.ElementStop) as caught:
            engine.start_values(deck, [_kick("K:KICK", element=element, plane="x")], _settings())
        assert caught.value.count == count

    def test_a_pulsed_kicks_element_must_apply_a_kick_angle(self, deck: Path) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(deck, [_kick("K:KICK", element="INJ", plane="x")], _settings())
        assert caught.value.record_id == "SR/K:KICK"
        assert "element INJ applies no kick angle" in caught.value.detail

    def test_block_refusal_reaches_start_values(self, deck: Path) -> None:
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                deck,
                [_record("armed", "B1:ARMED", element="BPM1")],
                _settings(turns=0),
            )
        assert ".turns is 0" in caught.value.detail

    def test_todays_ring_is_unchanged(self, deck: Path) -> None:
        orbit = {"id": "SR/B1:X", "address": "B1:X", "element": "BPM1", "engine": {"axis": "x"}}
        assert engine.start_values(deck, [orbit], {}, readbacks={"B1:X": None}) == {"B1:X": 0.0}
