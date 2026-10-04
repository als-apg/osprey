"""The pyat engine's deck-only plug-in contract: locate, prepare, start_values, plane."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import ENTRY_POINT_GROUP  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

GAIN = 0.37
OFFSET = -0.012
K_A = 1.234
K_B = -0.456
SETTING = {"attribute": "PolynomB", "index": 1}


@dataclass
class LinearCurve:
    gain: float
    offset: float


@dataclass
class TableCurve:
    grid: list[float]
    values: list[float]


@dataclass
class Curve:
    linear: LinearCurve | None = None
    table: TableCurve | None = None


@dataclass
class Calibration:
    curve: Curve
    inverse: Curve | None = None


@dataclass
class Slice:
    element: str
    weight: float | None = None
    device: str | None = None


@dataclass
class Wiring:
    id: str
    address: str
    element: str | None = None
    slices: list[Slice] | None = None
    engine: Any = None
    calibration: Calibration | None = None


def _save(tmp_path: Path, elements: list[Any], name: str = "deck") -> Path:
    lattice = at.Lattice(elements, energy=3e9, particle="electron", periodicity=1)
    path = tmp_path / f"{name}.json"
    at.save_lattice(lattice, str(path))
    return path


@pytest.fixture
def ab_deck(tmp_path: Path) -> Path:
    """Elements A at s 1.0-1.2 and B at s 3.0-3.2, one monitor at the end."""
    return _save(
        tmp_path,
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("A", 0.2, K_A),
            at.Drift("D1", 1.8),
            at.Quadrupole("B", 0.2, K_B),
            at.Drift("D2", 0.5),
            at.Monitor("BPM1"),
        ],
    )


def _linear(gain: float = GAIN, offset: float = OFFSET) -> Calibration:
    return Calibration(curve=Curve(linear=LinearCurve(gain, offset)))


def _slice_record(slices: list[Slice], calibration: Calibration | None = None) -> Wiring:
    return Wiring(
        id="LINE/LINE:Q:SP",
        address="LINE:Q:SP",
        slices=slices,
        engine={"attribute": "PolynomB", "index": 1},
        calibration=calibration if calibration is not None else _linear(),
    )


class TestEntryPoint:
    def test_group_loads_the_module(self):
        loaded = entry_points(group=ENTRY_POINT_GROUP)["pyat"].load()
        assert loaded is importlib.import_module("osprey.simulation.engines.pyat")

    def test_import_leaves_lume_unloaded(self):
        code = (
            "import sys\n"
            "import osprey.simulation.engines.pyat\n"
            "loaded = [m for m in ('lume', 'lume_pyat.model') if m in sys.modules]\n"
            "assert not loaded, loaded\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr


class TestCalibrationTypes:
    def test_bindings_reuse_the_engine_types(self):
        from osprey.services.virtual_accelerator import bindings
        from osprey.simulation.engines import calibration

        assert bindings.Linear is calibration.Linear
        assert bindings.Table is calibration.Table
        assert bindings.Calibration is calibration.Calibration


class TestLocate:
    def test_entrance_and_length(self, ab_deck: Path):
        assert engine.locate(ab_deck, "A") == pytest.approx((1.0, 0.2), abs=1e-12)
        assert engine.locate(ab_deck, "B") == pytest.approx((3.0, 0.2), abs=1e-12)

    def test_unknown_element(self, ab_deck: Path):
        with pytest.raises(FacilityBuildError) as caught:
            engine.locate(ab_deck, "Q9", model="LINE")
        assert caught.value.kind == "engine-invalid"
        assert "Q9" in caught.value.detail
        assert caught.value.record_id == "LINE"

    def test_repeated_element(self, tmp_path: Path):
        deck = _save(tmp_path, [at.Quadrupole("Q", 0.1, 1.0), at.Quadrupole("Q", 0.1, 1.0)])
        with pytest.raises(FacilityBuildError) as caught:
            engine.locate(deck, "Q")
        assert caught.value.kind == "engine-invalid"
        assert "Q" in caught.value.detail


class TestSyntheticSliceDeck:
    """The two-slice A/B record: device span and inverted default."""

    def test_span_and_default(self, ab_deck: Path):
        record = _slice_record([Slice("A", 2.0), Slice("B", 1.0)])
        spans = [engine.locate(ab_deck, s.element) for s in record.slices]
        start = min(entrance for entrance, _ in spans)
        end = max(entrance + length for entrance, length in spans)
        assert start == pytest.approx(1.0, abs=1e-12)
        assert end - start == pytest.approx(2.2, abs=1e-12)

        values = engine.start_values(ab_deck, [record], {})
        assert values["LINE:Q:SP"] == pytest.approx((K_A / 2 - OFFSET) / GAIN, abs=1e-12)

    def test_first_slice_is_the_one_inverted(self, ab_deck: Path):
        record = _slice_record([Slice("B", 1.0), Slice("A", 2.0)])
        values = engine.start_values(ab_deck, [record], {})
        assert values["LINE:Q:SP"] == pytest.approx((K_B - OFFSET) / GAIN, abs=1e-12)

    def test_absent_weight_is_one(self, ab_deck: Path):
        record = _slice_record([Slice("A")])
        values = engine.start_values(ab_deck, [record], {})
        assert values["LINE:Q:SP"] == pytest.approx((K_A - OFFSET) / GAIN, abs=1e-12)


class TestStartValues:
    def test_inverse_curve_wins(self, ab_deck: Path):
        inverse = Curve(table=TableCurve(grid=[0.0, 2.0], values=[10.0, 30.0]))
        calibration = Calibration(curve=Curve(linear=LinearCurve(GAIN, OFFSET)), inverse=inverse)
        record = Wiring(
            "LINE/Q:SP",
            "Q:SP",
            element="A",
            engine={"attribute": "PolynomB", "index": 1},
            calibration=calibration,
        )
        values = engine.start_values(ab_deck, [record], {})
        assert values["Q:SP"] == pytest.approx(10.0 + 10.0 * K_A, abs=1e-12)

    def test_no_calibration_serves_the_physics_value(self, ab_deck: Path):
        record = Wiring(
            "LINE/Q:SP", "Q:SP", element="B", engine={"attribute": "PolynomB", "index": 1}
        )
        assert engine.start_values(ab_deck, [record], {}) == {"Q:SP": pytest.approx(K_B)}

    def test_table_without_inverse(self, ab_deck: Path):
        calibration = Calibration(curve=Curve(table=TableCurve([0.0, 1.0], [0.0, 2.0])))
        record = _slice_record([Slice("A")], calibration)
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(ab_deck, [record], {})
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "LINE/LINE:Q:SP"

    def test_zero_gain(self, ab_deck: Path):
        record = _slice_record([Slice("A")], _linear(gain=0.0))
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(ab_deck, [record], {})
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "LINE/LINE:Q:SP"
        assert "gain" in caught.value.detail

    def test_readback_through_its_pair_else_zero(self, ab_deck: Path):
        setpoint = _slice_record([Slice("A", 2.0)])
        paired = Wiring(
            "LINE/LINE:Q:RB", "LINE:Q:RB", element="A", engine={"attribute": "PolynomB", "index": 1}
        )
        lonely = Wiring("LINE/BPM1:X", "BPM1:X", element="BPM1", engine={"axis": "x"})
        values = engine.start_values(
            ab_deck,
            [setpoint, paired, lonely],
            {},
            readbacks={"LINE:Q:RB": "LINE:Q:SP", "BPM1:X": None},
        )
        assert values["LINE:Q:RB"] == values["LINE:Q:SP"]
        assert values["BPM1:X"] == 0.0
        assert list(values) == sorted(values)

    def test_a_non_float_channel_stops_before_the_deck_is_read(self, tmp_path: Path):
        float_record = Wiring("LINE/Q:SP", "Q:SP", element="A", engine=dict(SETTING))
        int_record = Wiring("LINE/N:RB", "N:RB", element="A", engine=dict(SETTING))
        enum_record = Wiring("LINE/E:RB", "E:RB", element="A", engine=dict(SETTING))
        with pytest.raises(FacilityBuildError) as caught:
            engine.start_values(
                tmp_path / "absent.json",
                [float_record, int_record, enum_record],
                {},
                value_types={"Q:SP": "float", "N:RB": "int", "E:RB": "enum"},
            )
        assert (caught.value.kind, caught.value.record_id, caught.value.record_kind) == (
            "engine-invalid",
            "LINE/N:RB",
            "wiring",
        )
        assert caught.value.detail == "N:RB is int; pyat drives float channels only"
        assert caught.value.remedy == "wire a float channel, or leave the channel unwired"

    def test_float_channels_build(self, ab_deck: Path):
        record = Wiring("LINE/Q:SP", "Q:SP", element="A", engine=dict(SETTING))
        values = engine.start_values(ab_deck, [record], {}, value_types={"Q:SP": "float"})
        assert values == {"Q:SP": pytest.approx(K_A)}

    def test_mapping_records(self, ab_deck: Path):
        record = {
            "id": "LINE/Q:SP",
            "address": "Q:SP",
            "slices": [{"element": "A", "weight": 2.0}],
            "engine": {"attribute": "PolynomB", "index": 1},
            "calibration": {"curve": {"linear": {"gain": GAIN, "offset": OFFSET}}},
        }
        values = engine.start_values(ab_deck, [record], {})
        assert values["Q:SP"] == pytest.approx((K_A / 2 - OFFSET) / GAIN, abs=1e-12)


def _line_deck(tmp_path: Path) -> Path:
    return _save(
        tmp_path,
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("QF", 0.3, 1.2),
            at.Drift("D1", 1.0),
            at.Monitor("BPM1"),
            at.Quadrupole("QD", 0.3, -1.1),
            at.Drift("D2", 1.0),
            at.Monitor("BPM2"),
        ],
        name="line",
    )


TWISS = {"beta": [5.0, 3.0], "alpha": [0.1, -0.2], "dispersion": [0.0, 0.0, 0.0, 0.0]}


class TestPrepare:
    def test_closed_orbit_four_matches_six(self, tmp_path: Path):
        deck = _line_deck(tmp_path)
        four = engine.prepare(
            deck,
            {
                "pyat": {
                    "solve": "single_pass",
                    "twiss_in": {**TWISS, "closed_orbit": [1e-3, 2e-4, -5e-4, 1e-4]},
                }
            },
        )
        six = engine.prepare(
            deck,
            {
                "pyat": {
                    "solve": "single_pass",
                    "twiss_in": {
                        **TWISS,
                        "closed_orbit": np.array([1e-3, 2e-4, -5e-4, 1e-4, 0.0, 0.0]),
                    },
                }
            },
        )
        assert four.twiss_in["closed_orbit"].shape == (6,)
        lattice = at.load_lattice(str(deck))
        monitors = [i for i, e in enumerate(lattice) if isinstance(e, at.Monitor)]
        optics4 = at.get_optics(lattice, refpts=monitors, twiss_in=four.twiss_in)[2]
        optics6 = at.get_optics(lattice, refpts=monitors, twiss_in=six.twiss_in)[2]
        for key in ("beta", "alpha", "closed_orbit", "dispersion"):
            np.testing.assert_allclose(optics4[key], optics6[key], rtol=0, atol=1e-12)
        track4 = at.lattice_track(
            lattice, four.twiss_in["closed_orbit"].reshape(6, 1), refpts=monitors
        )[0]
        track6 = at.lattice_track(
            lattice, six.twiss_in["closed_orbit"].reshape(6, 1), refpts=monitors
        )[0]
        np.testing.assert_allclose(track4, track6, rtol=0, atol=1e-12)

    def test_defaults(self, tmp_path: Path):
        prepared = engine.prepare(_line_deck(tmp_path), None)
        assert prepared.solve == "periodic"
        assert prepared.twiss_in is None
        assert prepared.rest_mass_gev == pytest.approx(0.51099895069e-3, rel=1e-12)

    def test_length_is_the_last_exit(self, ab_deck: Path):
        prepared = engine.prepare(ab_deck, None)
        lattice = at.load_lattice(str(ab_deck))
        last = lattice.get_s_pos(len(lattice) - 1)[0] + lattice[-1].Length
        assert prepared.length_m == pytest.approx(3.7, abs=1e-12)
        assert prepared.length_m == pytest.approx(float(last), abs=1e-12)

    def test_length_is_as_written_under_periodicity(self, tmp_path: Path):
        lattice = at.Lattice(
            [at.Drift("D0", 1.0), at.Quadrupole("Q", 0.5, 1.0), at.Drift("D1", 1.5)],
            energy=3e9,
            particle="electron",
            periodicity=4,
        )
        deck = tmp_path / "cell.json"
        at.save_lattice(lattice, str(deck))
        assert at.load_lattice(str(deck)).circumference == pytest.approx(12.0)
        assert engine.prepare(deck, None).length_m == pytest.approx(3.0, abs=1e-12)

    def test_rest_mass_override(self, tmp_path: Path):
        prepared = engine.prepare(_line_deck(tmp_path), {"pyat": {"rest_mass_gev": 0.938272}})
        assert prepared.rest_mass_gev == 0.938272

    def test_single_pass_needs_twiss_in(self, tmp_path: Path):
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(_line_deck(tmp_path), {"pyat": {"solve": "single_pass"}}, model="LINE")
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "LINE"
        assert "twiss_in" in caught.value.detail

    @pytest.mark.parametrize(
        ("key", "value"),
        [("beta", [1.0, 2.0, 3.0]), ("dispersion", [0.0, 0.0]), ("closed_orbit", [0.0] * 5)],
    )
    def test_wrong_length_twiss_in(self, tmp_path: Path, key: str, value: list[float]):
        twiss = {**TWISS, key: value}
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(
                _line_deck(tmp_path),
                {"pyat": {"solve": "single_pass", "twiss_in": twiss}},
                model="LINE",
            )
        assert caught.value.kind == "engine-invalid"
        assert caught.value.record_id == "LINE"
        assert key in caught.value.detail

    def test_unknown_solve(self, tmp_path: Path):
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(_line_deck(tmp_path), {"pyat": {"solve": "sideways"}})
        assert caught.value.kind == "engine-invalid"
        assert "solve" in caught.value.detail

    def test_repeated_monitor_names(self, tmp_path: Path):
        deck = _save(
            tmp_path,
            [at.Monitor("BPM1"), at.Drift("D", 1.0), at.Monitor("BPM1"), at.Monitor("BPM2")],
        )
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(deck, None)
        assert caught.value.kind == "engine-invalid"
        assert "BPM1" in caught.value.detail
        assert "BPM2" not in caught.value.detail

    def test_periodic_cavity_without_longitudinal_motion(self, tmp_path: Path):
        cavity = at.RFCavity("RFC", 0.0, 1e6, 5e8, 300, 3e9)
        cavity.PassMethod = "IdentityPass"
        deck = _save(tmp_path, [at.Drift("D", 1.0), cavity])
        with pytest.raises(FacilityBuildError) as caught:
            engine.prepare(deck, {"pyat": {"solve": "periodic"}})
        assert caught.value.kind == "engine-invalid"
        assert "RFC" in caught.value.detail

    def test_single_pass_accepts_a_cavity_without_longitudinal_motion(self, tmp_path: Path):
        cavity = at.RFCavity("RFC", 0.0, 1e6, 5e8, 300, 3e9)
        cavity.PassMethod = "IdentityPass"
        deck = _save(tmp_path, [at.Drift("D", 1.0), cavity])
        prepared = engine.prepare(deck, {"pyat": {"solve": "single_pass", "twiss_in": TWISS}})
        assert prepared.solve == "single_pass"
        np.testing.assert_array_equal(prepared.twiss_in["closed_orbit"], np.zeros(6))


class TestPlane:
    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            ({"attribute": "KickAngle", "index": 0}, "x"),
            ({"attribute": "KickAngle", "index": 1}, "y"),
            ({"attribute": "PolynomB", "index": 0}, "x"),
            ({"attribute": "PolynomA", "index": 0}, "y"),
            ({"attribute": "PolynomB", "index": 1}, None),
            ({"attribute": "Frequency"}, None),
            ({"axis": "x"}, None),
        ],
    )
    def test_plane(self, block: dict[str, Any], expected: str | None):
        assert engine.plane(Wiring("M/A", "A", element="E", engine=block)) == expected
