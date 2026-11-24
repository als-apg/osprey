"""A further address of a wiring record is served as the record's own, end to end.

Each case builds a facility tree whose wiring states ``also``, writes its
simulator view and serves it from a :class:`Composite` on a fixed clock. A
monitor reading with a noise seed and a magnet setpoint are each under a
second address on the small pyAT ring; the turn-by-turn fixture carries one
monitor's arm setpoint and its turn array under a second address each.

The further address reads what the record's address reads at the same
instant, after a setpoint change too; a write to it is a write to the
record's address; a waveform record does the same; and a facility that
states no ``also`` reads as it did before.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from lume.exceptions import ReadOnlyError

from osprey.apps.sdk.testing import model
from osprey.facility.views.outputs import render_facility_outputs
from osprey.facility.views.simulator import write_simulator_view
from osprey.services.virtual_accelerator.serving.runner_config import (
    pass_readback,
    setting_records,
)
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.servers import model_server
from osprey_connectors.simulation.view import SCENARIOS_FILE, SimulatorView
from osprey_facility.build import build_facility
from tests._builds import FRAMEWORK_CHECKS
from tests.connectors._view_records import activate
from tests.facility._pyaml_trees import measured_tree, view_inputs
from tests.facility._turn_by_turn_fixture import (
    ARMED,
    ARMS,
    FIXTURE,
    MONITORS,
    REQUEST,
    RING,
    TURN_X,
)
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

T0 = 1_760_000_000.0

#: A reading and a setpoint the measured tree wires, and a second address for each.
READING, READING_ALSO = "BPM1:X", "RACK3:UNIT1:X"
SETPOINT, SETPOINT_ALSO = "QD:SP", "RACK3:UNIT2:SP"

#: The readout noise seeded on the reading, in its unit.
NOISE = 0.05

#: The turn-by-turn monitor under second addresses, and those addresses.
MONITOR = MONITORS[0]
ARM_ALSO = "RACK3:UNIT1:ARM"
TURN_X_ALSO = "RACK3:UNIT1:TBT:X"


def _tree(*, also: bool = True) -> dict[str, Any]:
    """The measured tree with the two second addresses, wired under ``also`` or not at all.

    Unwired, the second setpoint address states its role; under ``also`` it is its record's.
    """
    tree = measured_tree()
    tree["records/channels.yaml"] += [
        {"id": READING_ALSO, "on": {"device": "SR/BPM1"}},
        {"id": SETPOINT_ALSO, "on": {"device": "SR/QD"}, **({} if also else {"role": "setpoint"})},
    ]
    tree["limits.yaml"] = {
        "records": [
            {"address": address, "writable": True, "min_value": -200.0, "max_value": 200.0}
            for address in (SETPOINT, SETPOINT_ALSO)
        ]
    }
    tree["seeds.yaml"] = {READING: {"noise": {"absolute": NOISE}}}
    if also:
        for record in model(tree, "SR")["wiring"]:
            if record["address"] == READING:
                record["also"] = [READING_ALSO]
            if record["address"] == SETPOINT:
                record["also"] = [SETPOINT_ALSO]
    return tree


def _view(root: Path, tree: dict[str, Any]) -> Path:
    view = root / "render" / "data" / "simulator"
    write_simulator_view(view, view_inputs(root, tree))
    return view


class _Clock:
    def __init__(self) -> None:
        self.t_s = T0

    def __call__(self) -> float:
        return self.t_s


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _view(tmp_path_factory.mktemp("wiring_also"), _tree())


def _composite(view: Path, clock: Any = None, **options: Any) -> Composite:
    return Composite(view, clock=clock or (lambda: T0), model_log=False, **options)


# -- reads ----------------------------------------------------------------------


def test_a_further_address_reads_as_the_records_address_after_a_setpoint_change(
    view: Path,
) -> None:
    clock = _Clock()
    composite = _composite(view, clock)
    before = composite.get([READING, READING_ALSO])
    assert before[READING_ALSO] == before[READING]

    clock.t_s += 1.0
    composite.set({SETPOINT: -90.0})
    after = composite.get([READING, READING_ALSO])

    assert after[READING_ALSO] == after[READING]
    assert abs(after[READING] - before[READING]) > 1e-6
    # Asked alone, and second, it is still the same draw at the same instant.
    assert composite.get(READING_ALSO) == after[READING]
    assert composite.get(READING) == after[READING]


def test_the_readout_noise_is_one_draw_for_both_addresses(view: Path) -> None:
    clock = _Clock()
    composite = _composite(view, clock)
    readings = []
    for _ in range(8):
        clock.t_s += 1.0
        values = composite.get([READING_ALSO, READING])
        assert values[READING_ALSO] == values[READING]
        readings.append(values[READING])

    assert np.std(readings) > NOISE / 10


def test_the_held_and_the_averaged_reading_agree_too(view: Path) -> None:
    composite = _composite(view)

    times = np.array([T0, T0 + 1.0, T0 + 2.0])

    held = composite.held([READING_ALSO, READING])
    own = composite.readings({READING: 0.25, "BPM1:Y": 0.0}, times)
    further = composite.readings({READING_ALSO: 0.25, "BPM1:Y": 0.0}, times)

    assert held[READING_ALSO] == held[READING]
    assert sorted(further) == ["BPM1:Y", READING_ALSO]
    assert np.array_equal(further[READING_ALSO], own[READING])
    assert np.ptp(own[READING]) > 0.0


def test_a_further_address_stands_in_its_records_readout_group(view: Path) -> None:
    composite = _composite(view)

    assert composite.readout_group(READING) == (READING, "BPM1:Y")
    assert composite.readout_group(READING_ALSO) == (READING_ALSO, "BPM1:Y")


def test_a_channel_fault_on_the_record_shows_under_both_addresses(tmp_path: Path) -> None:
    tree = _tree()
    tree["scenarios/stuck.yaml"] = {"name": "stuck", "channel_faults": {READING_ALSO: "frozen"}}
    clock = _Clock()
    composite = _composite(_view(tmp_path, tree), clock, state_dir=tmp_path / "state")
    live = composite.get(READING)

    activate(tmp_path / "state", "stuck")
    frozen = composite.get([READING, READING_ALSO])
    clock.t_s += 1.0
    composite.set({SETPOINT: -90.0})

    assert frozen[READING_ALSO] == frozen[READING]
    assert composite.get([READING, READING_ALSO]) == frozen
    activate(tmp_path / "state")
    assert composite.get(READING_ALSO) == composite.get(READING) != live


def test_a_stopped_server_stops_every_address_of_its_records(tmp_path: Path) -> None:
    view = _view(tmp_path, _tree())
    scenarios = json.loads((view / SCENARIOS_FILE).read_text(encoding="utf-8"))
    scenarios["scenarios"].append({"name": "outage", "stopped_servers": [model_server("SR")]})
    (view / SCENARIOS_FILE).write_text(json.dumps(scenarios), encoding="utf-8")
    composite = _composite(view, state_dir=tmp_path / "state")
    assert {READING, READING_ALSO} <= composite.moving()

    activate(tmp_path / "state", "outage")

    assert {READING, READING_ALSO, SETPOINT, SETPOINT_ALSO} <= composite.stopped()
    assert composite.moving().isdisjoint({READING, READING_ALSO})
    activate(tmp_path / "state")
    assert composite.stopped() == frozenset()
    assert composite.get(READING_ALSO) == composite.get(READING)


# -- writes ---------------------------------------------------------------------


def test_a_write_to_a_further_address_writes_the_record(view: Path) -> None:
    composite = _composite(view)
    before = composite.get(READING)

    composite.set({SETPOINT_ALSO: -90.0})

    assert composite.get([SETPOINT, SETPOINT_ALSO]) == {SETPOINT: -90.0, SETPOINT_ALSO: -90.0}
    assert abs(composite.get(READING) - before) > 1e-6
    composite.set({SETPOINT: -95.0})
    assert composite.get(SETPOINT_ALSO) == -95.0


def test_a_write_to_a_further_address_of_a_reading_is_refused(view: Path) -> None:
    composite = _composite(view)

    with pytest.raises(ReadOnlyError):
        composite.set({READING_ALSO: 1.0})


def test_a_write_pass_reads_a_setting_back_under_each_address(view: Path) -> None:
    records = setting_records(SimulatorView.open(view))

    assert records[SETPOINT_ALSO] == records[SETPOINT]
    assert {SETPOINT, SETPOINT_ALSO, READING, READING_ALSO} <= records[SETPOINT]
    assert pass_readback(records, [SETPOINT_ALSO]) == records[SETPOINT]


# -- a waveform record ----------------------------------------------------------


def _turn_by_turn_view(root: Path) -> Path:
    """The turn-by-turn fixture, one monitor's arm and turn array under second addresses."""
    facility_dir = root / "data" / "facility"
    shutil.copytree(FIXTURE, facility_dir)
    channels = facility_dir / "records" / "channels.yaml"
    listed = yaml.safe_load(channels.read_text(encoding="utf-8"))
    listed += [{"id": ARM_ALSO}, {"id": TURN_X_ALSO}]
    channels.write_text(yaml.safe_dump(listed), encoding="utf-8")
    models = facility_dir / "models.yaml"
    document = yaml.safe_load(models.read_text(encoding="utf-8"))
    ring = next(entry for entry in document if entry["name"] == RING)
    for record in ring["wiring"]:
        if record["address"] == ARMS[MONITOR]:
            record["also"] = [ARM_ALSO]
        if record["address"] == TURN_X[MONITOR]:
            record["also"] = [TURN_X_ALSO]
    models.write_text(yaml.safe_dump(document), encoding="utf-8")
    render_dir = root / "build"
    render_dir.mkdir(parents=True)
    built = build_facility(
        facility_dir, project_name="turn-by-turn-also", extra_checks=FRAMEWORK_CHECKS
    )
    render_facility_outputs(render_dir, built, IN_PROCESS_RENDER_CONFIG, facility_dir)
    return render_dir / "data" / "simulator"


def test_a_waveform_record_reads_the_same_under_its_further_address(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(_turn_by_turn_view(tmp_path), clock, evolve=True)
    composite.emitted()
    quiet = np.asarray(composite.get(TURN_X_ALSO), dtype=float)

    # Armed through the further address, then one shot.
    clock.t_s += 1.0
    composite.set({ARM_ALSO: 1.0})
    armed = composite.emitted()
    assert composite.get([ARMS[MONITOR], ARMED[MONITOR]]) == {
        ARMS[MONITOR]: 1.0,
        ARMED[MONITOR]: 1.0,
    }
    clock.t_s += 1.0
    composite.set({REQUEST: 1.0})
    composite.emitted()
    clock.t_s += 0.1
    composite.advance()
    shot = composite.emitted()

    arrays = composite.get([TURN_X[MONITOR], TURN_X_ALSO])
    first = np.asarray(arrays[TURN_X[MONITOR]], dtype=float)
    assert ARMED[MONITOR] in armed
    assert {TURN_X[MONITOR], TURN_X_ALSO} <= shot
    assert first.shape == quiet.shape
    assert np.array_equal(np.asarray(arrays[TURN_X_ALSO], dtype=float), first)
    assert not np.array_equal(first, quiet)
    # Only the monitor armed took the shot.
    assert np.array_equal(np.asarray(composite.get(TURN_X[MONITORS[1]]), dtype=float), quiet)


# -- a facility that states no ``also`` -------------------------------------------


def test_a_facility_stating_no_also_reads_as_before(view: Path, tmp_path: Path) -> None:
    """Without ``also`` the second addresses are texture channels and the ring reads the same."""
    plain = _composite(_view(tmp_path, _tree(also=False)))
    stated = _composite(view)
    names = [READING, "BPM1:Y", SETPOINT, "QF:SP"]

    assert plain.get(names) == stated.get(names)
    plain.set({SETPOINT: -90.0})
    stated.set({SETPOINT: -90.0})
    assert plain.get(names) == stated.get(names)
    assert plain.get(READING_ALSO) != plain.get(READING)
    assert plain.readout_group(READING_ALSO) == (READING_ALSO,)
    assert plain.emitted() == stated.emitted() == frozenset()
