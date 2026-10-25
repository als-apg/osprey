"""The pyat engine's ``condition``: the degraded state a built model is in, and what it voids.

A periodic model that states an ``injection`` keeps a write that leaves its
lattice with no closed orbit. ``condition`` then answers the text
``no closed orbit`` and the names whose values that state voids -- the orbit
readings, the tune, beta and chromaticity readbacks -- and ``None`` at every
other time. These run against a small purpose-built periodic lattice whose
first quadrupole, driven hard enough, loses the closed orbit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.simulation.engines import EnginePlugin  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    BETA_AT_MONITORS,
    CHROMATICITY,
    ORBIT_AT_MONITORS,
    TUNES,
)

MODEL = "SR"
N_CELLS = 8
MARKER = "INJ"

QUAD = "QF:01:SP"
QUAD_READBACK = "QF:01:RB"
CORRECTOR = "HCOR:01:SP"
TUNE_X = "TUNE:X"
TUNE_Y = "TUNE:Y"
TUNE_BOTH = "TUNE:BOTH"
CHROMATICITY_X = "CHROM:X"
CHROMATICITY_BOTH = "CHROM:BOTH"
SERIES_X = "BPM:01:TBT:X"
#: The quadrupole current the lattice is built at, and one it cannot close at.
QUAD_NOMINAL = 100.0
QUAD_UNSTABLE = 1.0e4

SETTINGS = {"pyat": {"injection": {"marker": MARKER}}}
TWISS_IN = {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]}

#: The one text the engine answers.
TEXT = "no closed orbit"

MONITORS = [f"BPM:{cell:02d}:{axis}" for cell in range(1, N_CELLS + 1) for axis in ("X", "Y")]
#: The optics arrays every periodic model declares beside its channels.
OPTICS = [TUNES, CHROMATICITY, BETA_AT_MONITORS, ORBIT_AT_MONITORS]
#: The wired readbacks of the tunes and the chromaticity, one plane or both.
OPTICS_READBACKS = [TUNE_X, TUNE_Y, TUNE_BOTH, CHROMATICITY_X, CHROMATICITY_BOTH]


def lattice() -> Any:
    """A 4D FODO lattice: a marker, then a monitor and a corrector per cell."""
    elements: list[Any] = [at.Marker(MARKER)]
    for cell in range(1, N_CELLS + 1):
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QF_{cell:02d}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(f"HCOR_{cell:02d}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BEND_{cell:02d}", 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(f"BPM_{cell:02d}"),
            at.Drift("DRIFT", 0.8),
            at.Quadrupole(f"QD_{cell:02d}", 0.3, -1.0),
        ]
    built = at.Lattice(elements, name="TEST", energy=3.0e9, periodicity=1)
    built.disable_6d()
    return built


def _linear(gain: float) -> dict[str, Any]:
    return {"curve": {"linear": {"gain": gain, "offset": 0.0}}, "energy_scaling": "none"}


def _optics(address: str, attribute: str, axis: str | None) -> dict[str, Any]:
    """A record naming no element: one plane of an optics output, or the whole of it."""
    if axis is None:
        return {
            "address": address,
            "direction": "read",
            "engine": {"attribute": attribute},
            "value_type": "waveform",
            "shape": [2],
        }
    return {
        "address": address,
        "direction": "read",
        "engine": {"attribute": attribute, "axis": axis},
    }


def setpoints() -> list[dict[str, Any]]:
    """The first cell's corrector and quadrupole, and the quadrupole's readback."""
    return [
        {
            "address": CORRECTOR,
            "element": "HCOR_01",
            "direction": "write",
            "engine": {"attribute": "KickAngle", "index": 0},
            "unit": "A",
            "default": 0.0,
            "calibration": _linear(1.0e-5),
        },
        {
            "address": QUAD,
            "element": "QF_01",
            "direction": "write",
            "engine": {"attribute": "PolynomB", "index": 1},
            "unit": "A",
            "default": QUAD_NOMINAL,
            "calibration": _linear(1.0e-2),
        },
        {
            "address": QUAD_READBACK,
            "element": "QF_01",
            "direction": "read",
            "engine": {"attribute": "PolynomB", "index": 1},
            "unit": "A",
            "default": QUAD_NOMINAL,
            "calibration": _linear(1.0e-2),
        },
    ]


def monitors() -> list[dict[str, Any]]:
    return [
        {
            "address": f"BPM:{cell:02d}:{axis.upper()}",
            "element": f"BPM_{cell:02d}",
            "direction": "read",
            "engine": {"axis": axis},
            "unit": "mm",
            "calibration": _linear(1.0e-3),
        }
        for cell in range(1, N_CELLS + 1)
        for axis in ("x", "y")
    ]


def wiring() -> list[dict[str, Any]]:
    """Every monitor plane, the setpoints, each optics readback and one turn series."""
    return [
        *monitors(),
        *setpoints(),
        _optics(TUNE_X, "tune", "x"),
        _optics(TUNE_Y, "tune", "y"),
        _optics(TUNE_BOTH, "tune", None),
        _optics(CHROMATICITY_X, "chromaticity", "x"),
        _optics(CHROMATICITY_BOTH, "chromaticity", None),
        {
            "address": SERIES_X,
            "element": "BPM_01",
            "direction": "read",
            "engine": {"axis": "x", "series": "turn"},
            "value_type": "waveform",
            "shape": [32],
        },
    ]


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    path = tmp_path / "periodic.json"
    at.save_lattice(lattice(), str(path))
    return path


@pytest.fixture
def built(deck: Path) -> Any:
    return engine.build(MODEL, wiring(), deck, SETTINGS)


@pytest.fixture
def degraded(built: Any) -> Any:
    built.set({QUAD: QUAD_UNSTABLE})
    assert built.no_closed_orbit
    return built


class TestContract:
    def test_the_engine_states_the_name_and_its_texts(self):
        assert "condition" in engine.__all__
        assert "CONDITIONS" in engine.__all__
        assert callable(engine.condition)

    def test_the_closed_list_of_texts_is_the_one_text(self):
        assert engine.CONDITIONS == (TEXT,)

    def test_the_optional_name_table_documents_both_and_every_text(self):
        table = EnginePlugin.__doc__ or ""

        assert "``condition(model)``" in table
        assert "``CONDITIONS``" in table
        for text in engine.CONDITIONS:
            assert f"``{text}``" in engine.condition.__doc__

    def test_the_table_rules_stay_aligned(self):
        rules = [
            line
            for line in (EnginePlugin.__doc__ or "").splitlines()
            if line.strip().startswith("==")
        ]

        assert len(rules) == 3
        assert len(set(rules)) == 1


class TestFlagDown:
    def test_a_model_with_a_closed_orbit_has_no_condition(self, built: Any):
        assert not built.no_closed_orbit
        assert engine.condition(built) is None

    def test_a_periodic_model_without_an_injection_has_none(self, deck: Path):
        strict = engine.build(MODEL, [*monitors(), *setpoints()], deck, None)

        assert engine.condition(strict) is None

    def test_a_single_pass_model_has_none(self, deck: Path):
        settings = {"pyat": {"solve": "single_pass", "twiss_in": TWISS_IN}}
        line = engine.build(MODEL, [*monitors(), *setpoints()], deck, settings)

        assert engine.condition(line) is None

    def test_a_model_that_states_no_flag_has_none(self):
        assert engine.condition(object()) is None


class TestFlagUp:
    def test_the_answer_is_the_text_and_a_frozen_set_of_names(self, degraded: Any):
        answer = engine.condition(degraded)

        assert answer is not None
        text, names = answer
        assert text == TEXT
        assert text in engine.CONDITIONS
        assert isinstance(names, frozenset)

    def test_the_names_are_the_orbit_tune_beta_and_chromaticity_readbacks(self, degraded: Any):
        _, names = engine.condition(degraded)

        assert names == frozenset([*MONITORS, *OPTICS_READBACKS, *OPTICS])

    def test_every_listed_name_reads_nan(self, degraded: Any):
        _, names = engine.condition(degraded)

        for name, value in degraded.get(sorted(names)).items():
            assert np.isnan(np.asarray(value, dtype=np.float64)).all(), name

    def test_no_setpoint_and_no_readback_of_one_is_listed(self, degraded: Any):
        _, names = engine.condition(degraded)

        assert names.isdisjoint({QUAD, CORRECTOR, QUAD_READBACK})
        assert degraded.get([QUAD_READBACK])[QUAD_READBACK] == pytest.approx(QUAD_UNSTABLE)

    def test_no_fault_is_listed(self, degraded: Any):
        _, names = engine.condition(degraded)
        faults = set(degraded.derived_names) - set(OPTICS)

        assert faults
        assert names.isdisjoint(faults)

    def test_no_turn_series_is_listed(self, degraded: Any):
        _, names = engine.condition(degraded)

        assert SERIES_X not in names

    def test_a_model_built_under_the_flag_answers_at_once(self, deck: Path):
        booted = engine.build(MODEL, wiring(), deck, SETTINGS, {QUAD: QUAD_UNSTABLE})

        text, names = engine.condition(booted)
        assert text == TEXT
        assert names == frozenset([*MONITORS, *OPTICS_READBACKS, *OPTICS])


class TestFlagRead:
    def test_the_name_set_is_computed_once_for_a_model(self, degraded: Any):
        _, first = engine.condition(degraded)
        degraded.set({CORRECTOR: 1.0})
        _, second = engine.condition(degraded)

        assert degraded.no_closed_orbit
        assert second is first

    def test_the_set_survives_a_recovery_and_a_second_loss(self, degraded: Any):
        _, first = engine.condition(degraded)
        degraded.set({QUAD: QUAD_NOMINAL})
        assert engine.condition(degraded) is None

        degraded.set({QUAD: QUAD_UNSTABLE})

        assert engine.condition(degraded)[1] is first

    def test_a_write_that_restores_the_closed_orbit_ends_the_condition(self, degraded: Any):
        degraded.set({QUAD: QUAD_NOMINAL})

        assert engine.condition(degraded) is None

    def test_reset_ends_the_condition(self, degraded: Any):
        degraded.reset()

        assert engine.condition(degraded) is None

    def test_each_model_has_the_names_of_its_own_wiring(self, deck: Path, degraded: Any):
        bare = engine.build(MODEL, [*monitors()[:2], *setpoints()], deck, SETTINGS)
        bare.set({QUAD: QUAD_UNSTABLE})

        assert engine.condition(bare)[1] == frozenset([*MONITORS[:2], *OPTICS])
        assert engine.condition(degraded)[1] == frozenset([*MONITORS, *OPTICS_READBACKS, *OPTICS])

    def test_asking_does_not_keep_a_model_alive(self, deck: Path):
        import gc
        import weakref

        short_lived = engine.build(MODEL, wiring(), deck, SETTINGS, {QUAD: QUAD_UNSTABLE})
        assert engine.condition(short_lived) is not None
        reference = weakref.ref(short_lived)

        del short_lived
        gc.collect()

        assert reference() is None
