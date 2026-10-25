"""A served shot equals pySC's injection capture of the same ring.

After an arm and a request increment, the turn-1 sample of every BPM's
turn-by-turn x and y equals what pySC's ``capture_injection`` reads on the
same ring with every readout correction switched off, both on the clean
example ring and with the ``first-turn-errors`` scenario losing the beam inside
turn 1. The VA's BPM electronics (their noise and tone) are switched off too;
a BPM the beam never reached reads on axis at the sum's floor, where pySC
reads NaN.

pySC cannot load an errored ring from a file, so its ring is built from the
clean deck rotated to the injection marker and then given, element by element,
the VA lattice's misalignment transforms, apertures and corrector kicks. Its
pass methods must match the VA's before anything is compared.

The served buffers are also held against a direct pyAT track of the VA's own
lattice over the turns the shot tracks: x and y equal where the beam was seen
and on axis where it was not; and the sum 1 where it was seen and at the floor
where it was not, each divided by its record's declared calibration gain. Past the tracked turns every sample holds the no-beam reading.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

at = pytest.importorskip("at")
pytest.importorskip("pySC")

import numpy as np  # noqa: E402

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_faults import SUM_FLOOR  # noqa: E402
from osprey_connectors.control_system._va_in_process_connector import (  # noqa: E402
    simulation_state_dir,
)
from osprey_connectors.simulation.composite import Composite  # noqa: E402
from tests._builds import BuiltProject  # noqa: E402
from tests.simulation._sr_deck import load_sr_deck  # noqa: E402

pytestmark = pytest.mark.slow

SCENARIO = "first-turn-errors"
MODEL = "SR"
BPMS = 72
#: The example's machine, where each sum record declares its calibration gain.
MODELS_FILE = (
    Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example/models.yaml"
)
INJECTION_MARKER = "INJ"
#: The index of turn 1 in a turn-by-turn buffer.
TURN_1 = engine.PRE_TRIGGER_SAMPLES
ARM = "SR:DIAG:BPM:01:TBT:ARM"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
#: The turns a shot tracks from the trigger sample; later samples carry no beam.
TURNS_TRACKED = "SR:DIAG:TBT:TURNS:RB"
#: The relative agreement of the served shot with pySC and with a direct track.
RTOL = 1e-12
ATOL = 1e-15
#: The element attributes the VA lattice carries beyond the clean deck that
#: change a track: misalignment transforms, apertures and corrector kicks.
COPIED = ("T1", "T2", "R1", "R2", "EApertures", "KickAngle")
T0 = 1_760_000_000.0


def _tbt(bpm: int, signal: str) -> str:
    return f"SR:DIAG:BPM:{bpm:02d}:TBT:{signal}"


def _sum_gains() -> np.ndarray:
    """Each BPM's declared sum calibration gain, ``1/k_b``, as a ``(BPMS, 1)`` column.

    The example declares each monitor's charge-to-sum factor ``k_b`` in the
    machine: a served sum is its physics sum divided by this gain.
    """
    models = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
    gains = {
        record["address"]: float(record["calibration"]["curve"]["linear"]["gain"])
        for model in models
        for record in model.get("wiring") or ()
        if record["address"].endswith(":TBT:SUM")
    }
    return np.array([[gains[_tbt(bpm, "SUM")]] for bpm in range(1, BPMS + 1)])


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


def _shot(composite: Composite) -> dict[str, np.ndarray]:
    """Arm, increment the request, and read every BPM's buffers: ``(BPMS, samples)`` per signal."""
    composite.set({ARM: 1})
    composite.set({REQUEST: 1})
    names = [_tbt(bpm, signal) for bpm in range(1, BPMS + 1) for signal in ("X", "Y", "SUM")]
    read = composite.get(names)
    return {
        signal.lower(): np.array(
            [np.asarray(read[_tbt(bpm, signal)], dtype=np.float64) for bpm in range(1, BPMS + 1)]
        )
        for signal in ("X", "Y", "SUM")
    }


def _bpm_indices(ring: at.Lattice) -> list[int]:
    return [int(ring.get_uint32_index(f"BPM{bpm:02d}")[0]) for bpm in range(1, BPMS + 1)]


def _pysc_capture(va_lattice: at.Lattice, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """pySC's turn-1 injection capture on the VA lattice, read with no readout corrections."""
    from pySC.core.bpm_system import BPMSystem
    from pySC.core.lattice import ATLattice
    from pySC.core.simulated_commissioning import SimulatedCommissioning

    deck = load_sr_deck()
    start = int(deck.get_uint32_index(INJECTION_MARKER)[0])
    assert start == int(va_lattice.get_uint32_index(INJECTION_MARKER)[0])
    path = tmp_path / "clean_rotated_ring.json"
    at.save_lattice(copy.deepcopy(deck.rotate(start)), str(path))

    indices = _bpm_indices(deck.rotate(start))
    zeros = np.zeros(BPMS)
    sc = SimulatedCommissioning(
        lattice=ATLattice(lattice_file=str(path)),
        bpm_system=BPMSystem(
            indices=indices,
            names=[f"BPM{bpm:02d}" for bpm in range(1, BPMS + 1)],
            calibration_errors_x=zeros,
            calibration_errors_y=zeros,
            noise_tbt_x=zeros,
            noise_tbt_y=zeros,
            noise_co_x=zeros,
            noise_co_y=zeros,
            rolls=zeros,
        ),
    )

    ring = sc.lattice.ring
    count = len(va_lattice)
    assert len(ring) == count
    methods = [ring[k].PassMethod for k in range(count)]
    va_methods = [va_lattice[(k + start) % count].PassMethod for k in range(count)]
    assert methods == va_methods, "pySC's ring tracks with different pass methods than the VA"

    for k in range(count):
        source, target = va_lattice[(k + start) % count], ring[k]
        assert source.FamName == target.FamName
        for attribute in COPIED:
            if hasattr(source, attribute):
                setattr(target, attribute, np.array(getattr(source, attribute), dtype=np.float64))
            elif hasattr(target, attribute):
                delattr(target, attribute)

    x, y = sc.bpm_system.capture_injection(n_turns=1, bba=False, subtract_reference=False)
    assert x.shape == y.shape == (BPMS, 1)
    return x[:, 0], y[:, 0]


def _direct_track(va_lattice: at.Lattice, turns: int) -> tuple[np.ndarray, np.ndarray]:
    """A direct pyAT track of the VA lattice from its marker: x, y as ``(BPMS, turns)``."""
    start = int(va_lattice.get_uint32_index(INJECTION_MARKER)[0])
    rotated = va_lattice.rotate(start)
    refpts = np.array(_bpm_indices(rotated), dtype=np.uint32)
    tracked, _, _ = at.lattice_track(
        rotated, np.zeros((6, 1)), nturns=turns, refpts=refpts, losses=True
    )
    return tracked[0, 0, :, :], tracked[2, 0, :, :]


@pytest.mark.parametrize(
    ("active", "lost"),
    [pytest.param((), False, id="clean"), pytest.param((SCENARIO,), True, id=SCENARIO)],
)
def test_the_served_shot_equals_the_pysc_capture(
    demo_view: Path, tmp_path: Path, active: tuple[str, ...], lost: bool
) -> None:
    composite = _composite(demo_view, *active)
    shot = _shot(composite)
    va_lattice = composite._children[MODEL].model.lattice

    pysc_x, pysc_y = _pysc_capture(va_lattice, tmp_path)

    served_x, served_y = shot["x"][:, TURN_1], shot["y"][:, TURN_1]
    dark = np.isnan(pysc_x)
    np.testing.assert_array_equal(np.isnan(pysc_y), dark, err_msg=f"{active}: pySC's NaN")
    assert dark.any() == lost, f"{active}: pySC loses the first turn: {dark.any()}"
    assert np.isfinite(served_x).all() and np.isfinite(served_y).all()
    np.testing.assert_array_equal(
        shot["sum"][:, TURN_1],
        np.where(dark, SUM_FLOOR, 1.0) / _sum_gains()[:, 0],
        err_msg=f"{active}: sum",
    )
    for plane, served, pysc in (("x", served_x, pysc_x), ("y", served_y, pysc_y)):
        np.testing.assert_array_equal(served[dark], 0.0, err_msg=f"{active}: dark {plane}")
        np.testing.assert_allclose(
            served[~dark], pysc[~dark], rtol=RTOL, atol=ATOL, err_msg=f"{active}: {plane}"
        )


@pytest.mark.parametrize("active", [(), (SCENARIO,)], ids=["clean", SCENARIO])
def test_the_served_buffers_equal_a_direct_track(demo_view: Path, active: tuple[str, ...]) -> None:
    composite = _composite(demo_view, *active)
    shot = _shot(composite)
    turns = int(composite.get([TURNS_TRACKED])[TURNS_TRACKED])
    end = TURN_1 + turns
    assert 0 < turns and end <= shot["x"].shape[1]
    track_x, track_y = _direct_track(composite._children[MODEL].model.lattice, turns)

    for signal, no_beam in (("x", 0.0), ("y", 0.0), ("sum", SUM_FLOOR / _sum_gains())):
        past = shot[signal][:, end:]
        np.testing.assert_array_equal(
            past, np.broadcast_to(no_beam, past.shape), err_msg=f"past: {signal}"
        )
    served_x = shot["x"][:, TURN_1:end]
    served_y = shot["y"][:, TURN_1:end]
    seen = ~np.isnan(track_x)

    if active:
        assert not seen.all(), f"{active}: the track loses no sample"
    for plane, served, track in (("x", served_x, track_x), ("y", served_y, track_y)):
        np.testing.assert_allclose(
            served[seen], track[seen], rtol=RTOL, atol=ATOL, err_msg=f"{active}: {plane}"
        )
        np.testing.assert_array_equal(served[~seen], 0.0, err_msg=f"{active}: dark {plane}")
    np.testing.assert_array_equal(
        shot["sum"][:, TURN_1:end],
        np.where(seen, 1.0, SUM_FLOOR) / _sum_gains(),
        err_msg=f"{active}: sum",
    )
