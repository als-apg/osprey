"""The example's ``first-turn-errors`` scenario: a seeded errored ring that loses the first turn.

The scenario misaligns every quadrupole and sextupole in both planes and puts
an elliptical aperture on every element with length. With the shipped seed the
injected bunch strikes the chamber partway round turn 1; without the scenario
every BPM of the ring sees the whole first turn, on the trajectory a direct
pyAT track of the clean deck gives. Both cases read the built demo through the
composite, the way a served VA reads it, with the BPM electronics' noise and
tone switched off; a BPM the beam never reached reads on axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

at = pytest.importorskip("at")

import numpy as np  # noqa: E402

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey_connectors.control_system._va_in_process_connector import (  # noqa: E402
    simulation_state_dir,
)
from osprey_connectors.simulation.composite import Composite  # noqa: E402
from tests._builds import BuiltProject  # noqa: E402
from tests.simulation._sr_deck import load_sr_deck  # noqa: E402

SCENARIO = "first-turn-errors"
SCENARIO_FILE = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/facilities/example/scenarios"
    / f"{SCENARIO}.yaml"
)
#: The example's machine, where each sum record declares its calibration gain.
MODELS_FILE = (
    Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example/models.yaml"
)
MODEL = "SR"
#: The seed the scenario ships with.
SHIPPED_SEED = 11
#: The BPM (1-based, in beam order from the injection point) where the shipped
#: seed's first turn is lost.
LOSS_BPM = 30
#: The latest BPM a loss on the first turn may land at: half of the ring.
LATEST_LOSS_BPM = 36
BPMS = 72
SIGMA = 200e-6
APERTURE_X = 0.015
APERTURE_Y = 0.008
#: The sum reading below which a BPM sees no beam.
SUM_THRESHOLD = 0.5
#: The index of turn 1 in a turn-by-turn buffer.
TURN_1 = engine.PRE_TRIGGER_SAMPLES
ARM = "SR:DIAG:BPM:01:TBT:ARM"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
INJECTION_MARKER = "INJ"
#: The relative agreement of the served first turn with a direct track.
RTOL = 1e-12
T0 = 1_760_000_000.0


def _declared_sum_gains() -> dict[str, float]:
    """Each sum record's declared calibration gain, ``1/k_b`` (hardware to physics), by address.

    The example declares each monitor's charge-to-sum factor ``k_b`` in the
    machine, so a raw sum reads ``k_b`` times the charge share; times its gain it
    is the share again.
    """
    models = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    return {
        record["address"]: float(record["calibration"]["curve"]["linear"]["gain"])
        for model in models
        for record in model.get("wiring") or ()
        if record["address"].endswith(":TBT:SUM")
    }


def _tbt(bpm: int, signal: str) -> str:
    return f"SR:DIAG:BPM:{bpm:02d}:TBT:{signal}"


@pytest.fixture
def demo_view(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    """The built demo's simulator view, staged into its own render."""
    prefix = "data/simulator/"
    view = tmp_path / "demo" / "data" / "simulator"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = view / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return view


def _composite(view: Path, *active: str) -> Composite:
    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text("".join(f"{name}\n" for name in active))
    composite = Composite(view, state_dir=state, clock=lambda: T0, model_log=False)
    _electronics_off(composite)
    return composite


def _electronics_off(composite: Composite) -> None:
    """Switch every BPM's electronics noise and tone off, so a reading is the tracked position."""
    composite.get([_tbt(1, "X")])
    model = composite._children[MODEL].model
    for bpm in range(1, BPMS + 1):
        element = model.simulator.element(f"BPM{bpm:02d}")
        element.readout_noise_scale = 0.0
        element.readout_tone_amplitude = 0.0


def _first_turn(composite: Composite) -> dict[str, np.ndarray]:
    """Arm, take shot 1, and read turn 1 at every BPM: ``x``, ``y`` and ``sum``.

    ``sum`` is in physics units: each raw sum times its declared gain.
    """
    composite.set({ARM: 1})
    composite.set({REQUEST: 1})
    names = [_tbt(bpm, signal) for bpm in range(1, BPMS + 1) for signal in ("X", "Y", "SUM")]
    read = composite.get(names)
    turn = {
        signal.lower(): np.array(
            [np.asarray(read[_tbt(bpm, signal)])[TURN_1] for bpm in range(1, BPMS + 1)]
        )
        for signal in ("X", "Y", "SUM")
    }
    gains = _declared_sum_gains()
    turn["sum"] = turn["sum"] * np.array([gains[_tbt(bpm, "SUM")] for bpm in range(1, BPMS + 1)])
    return turn


def _clean_track() -> tuple[np.ndarray, np.ndarray]:
    """A direct pyAT track of the clean deck rotated to the injection point: turn-1 x, y per BPM."""
    ring = load_sr_deck()
    rotated = ring.rotate(int(ring.get_uint32_index(INJECTION_MARKER)[0]))
    refpts = np.array(
        [rotated.get_uint32_index(f"BPM{bpm:02d}")[0] for bpm in range(1, BPMS + 1)],
        dtype=np.uint32,
    )
    tracked, _, _ = at.lattice_track(
        rotated, np.zeros((6, 1)), nturns=1, refpts=refpts, losses=True
    )
    return tracked[0, 0, :, 0], tracked[2, 0, :, 0]


def test_the_scenario_carries_physics_only() -> None:
    document: dict[str, Any] = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))

    assert set(document) == {"description", "faults"}
    assert document["faults"] == {
        MODEL: {
            "misalignment_seed": SHIPPED_SEED,
            "quadrupole_dx_sigma": SIGMA,
            "quadrupole_dy_sigma": SIGMA,
            "sextupole_dx_sigma": SIGMA,
            "sextupole_dy_sigma": SIGMA,
            "aperture_elliptical_x": APERTURE_X,
            "aperture_elliptical_y": APERTURE_Y,
        }
    }


@pytest.mark.slow
def test_the_shipped_seed_loses_the_first_turn_at_its_bpm(demo_view: Path) -> None:
    turn = _first_turn(_composite(demo_view, SCENARIO))

    seen = turn["sum"] > SUM_THRESHOLD
    first_dark = int(np.argmin(seen)) + 1 if not seen.all() else None
    assert first_dark == LOSS_BPM, (
        f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: the first turn is lost at BPM "
        f"{first_dark}, pinned at BPM {LOSS_BPM}"
    )
    assert LOSS_BPM <= LATEST_LOSS_BPM
    assert seen[: LOSS_BPM - 1].all()
    assert not seen[LOSS_BPM - 1 :].any()
    for plane in ("x", "y"):
        assert np.isfinite(turn[plane]).all()
        assert (turn[plane][LOSS_BPM - 1 :] == 0.0).all()


@pytest.mark.slow
def test_without_the_scenario_every_bpm_sees_the_clean_first_turn(demo_view: Path) -> None:
    turn = _first_turn(_composite(demo_view))
    clean_x, clean_y = _clean_track()

    assert (turn["sum"] > SUM_THRESHOLD).all()
    for plane, clean in (("x", clean_x), ("y", clean_y)):
        assert np.isfinite(turn[plane]).all()
        assert np.max(np.abs(turn[plane])) <= np.max(np.abs(clean)) * (1 + RTOL)
        np.testing.assert_allclose(turn[plane], clean, rtol=RTOL, atol=1e-15)
