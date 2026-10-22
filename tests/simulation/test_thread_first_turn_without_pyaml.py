"""Thread the errored example ring to a full first turn the way a service would, without pyAML.

With the ``first-turn-errors`` scenario active the shipped seed loses the first
turn partway round the storage ring. A service that threads the beam needs no
middle layer to do it: everything it uses is in the build and on the wire. It
reads the trajectory response matrix and its sidecar from the pyAML render, the
channels to arm and fire, each monitor's sum reading and the trigger sample from
the declaration ``trajectory_slots`` in the facility facts, and
talks to the ring through the in-process connector only: arm every monitor,
increment the injection request, read turn 1 from the turn-by-turn buffers,
solve for a corrector change with plain numpy, and write the correctors in
hardware units, until every monitor sees beam.

The loop runs in a fresh interpreter, so that what it imports is observed on its
own: once anything in the pytest process has imported ``pyaml``, an in-process
look at ``sys.modules`` proves nothing. The correction here is test code; the
home of the correction math is pyAML.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("at")

if TYPE_CHECKING:
    from tests._builds import BuiltProject

pytestmark = pytest.mark.slow

CHECKOUT = Path(__file__).resolve().parents[2]
SCENARIO = "first-turn-errors"
MODEL = "SR"
#: The seed the scenario ships with.
SHIPPED_SEED = 11
#: The monitors of the ring.
BPMS = 72
#: The correction steps the shipped seed takes until every monitor sees beam.
STEPS = 3
#: The BPMs the uncorrected first turn reaches: every one before the loss at BPM 30.
FIRST_SHOT_SEEN = 29
#: The steps after which the loop gives up.
MAX_STEPS = 30
#: The render files the service reads, relative to the render root.
SIMULATOR_PREFIX = "data/simulator/"
TRAJECTORY_RM = f"data/pyaml/{MODEL}/trajectory_rm.json"
TRAJECTORY_RM_INDEX = f"data/pyaml/{MODEL}/trajectory_rm.index.json"
FACTS = "data/facility_facts.json"
#: Top-level packages the loop must not import.
FORBIDDEN = ("pyaml",)

#: The service-shaped loop, run as a script with the render root as its argument.
THREAD = r"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

import osprey
import osprey_connectors.config
from osprey_connectors.control_system.base import WriteOutcome
from osprey_connectors.control_system.va_in_process_connector import (
    VAInProcessConnector,
    simulation_state_dir,
)
from osprey_connectors.simulation import composite as composite_module

ROOT = Path(sys.argv[1])
MODEL, SCENARIO, MAX_STEPS, FORBIDDEN = json.loads(sys.argv[2])
VIEW = ROOT / "data" / "simulator"
SUM_THRESHOLD = 0.5
#: Singular values kept, relative to the largest.
RCOND = 1e-3


def writes_enabled(key, default=None):
    return True if key == "control_system.writes_enabled" else default


composite_module.default_config_path = lambda: None
osprey_connectors.config.get_config_value = writes_enabled


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


async def write(connector, address, value):
    result = await connector.write_channel(address, value)
    if result.outcome is not WriteOutcome.CONFIRMED:
        raise RuntimeError(f"write {address}={value!r}: {result.error_message}")


async def turn_one(connector, address, trigger):
    return float(np.asarray((await connector.read_channel(address)).value)[trigger])


async def main():
    rm = load(f"data/pyaml/{MODEL}/trajectory_rm.json")
    index = load(f"data/pyaml/{MODEL}/trajectory_rm.index.json")
    slots = load("data/facility_facts.json")["trajectory_slots"][MODEL]
    matrix = np.asarray(rm["matrix"], dtype=np.float64)
    rows = rm["observable_names"]
    columns = rm["variable_names"]
    assert matrix.shape == (len(rows), len(columns)), matrix.shape
    assert len(index["bpm_index"]) == len(rows)
    assert len(index["hcorr_index"]) + len(index["vcorr_index"]) == len(columns)
    # A row's monitor sees beam when its device's sum signal does.
    devices = slots["devices"]
    device_of = {}
    for device, row in devices.items():
        device_of[row["x_row"]] = device
        device_of[row["y_row"]] = device
    sums = [devices[device_of[row]]["sum_row"] for row in rows]
    assert all(sums), sums
    trigger = slots["trigger_sample"]
    assert isinstance(trigger, int), trigger
    # The columns are in lattice order within each plane, as are the rows.
    for key in ("hcorr_index", "vcorr_index"):
        assert index[key] == sorted(index[key]), key
    half = len(rows) // 2
    for places in (index["bpm_index"][:half], index["bpm_index"][half:]):
        assert all(a < b for a, b in zip(places, places[1:])), places

    state = simulation_state_dir(VIEW)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text(f"{SCENARIO}\n", encoding="utf-8")
    connector = VAInProcessConnector()
    await connector.connect({"simulator_view": str(VIEW), "response_delay_ms": 0})
    seen_per_shot = []
    try:
        request = int((await connector.read_channel(slots["injection_request"])).value)
        steps = 0
        while True:
            for device in devices.values():
                await write(connector, device["arm"], 1)
            request += 1
            await write(connector, slots["injection_request"], request)
            reading = np.array([await turn_one(connector, row, trigger) for row in rows])
            seen = np.array([await turn_one(connector, name, trigger) > SUM_THRESHOLD for name in sums])
            monitors_seen = len({s for s, ok in zip(sums, seen) if ok})
            seen_per_shot.append(monitors_seen)
            if seen.all() or steps >= MAX_STEPS:
                break
            used = seen & np.isfinite(reading)
            delta, *_ = np.linalg.lstsq(matrix[used], -reading[used], rcond=RCOND)
            for column, change in zip(columns, delta):
                current = float((await connector.read_channel(column)).value)
                await write(connector, column, current + float(change))
            steps += 1
    finally:
        await connector.disconnect()
    bad = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN)
    print(json.dumps({
        "steps": steps,
        "seen_per_shot": seen_per_shot,
        "forbidden": bad,
        "osprey": osprey.__file__,
    }))


asyncio.run(main())
"""


@pytest.fixture
def render(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    """The built example's simulator view and the files the service reads, staged fresh."""
    root = tmp_path / "example"
    wanted = {TRAJECTORY_RM, TRAJECTORY_RM_INDEX, FACTS}
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(SIMULATOR_PREFIX) or name in wanted:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            wanted.discard(name)
    assert not wanted, f"the build rendered no {', '.join(sorted(wanted))}"
    return root


def _thread(root: Path) -> dict[str, Any]:
    pythonpath = os.pathsep.join(
        p for p in (str(CHECKOUT / "src"), os.environ.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            THREAD,
            str(root),
            json.dumps([MODEL, SCENARIO, MAX_STEPS, list(FORBIDDEN)]),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env=dict(os.environ, PYTHONPATH=pythonpath),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outcome: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return outcome


def test_the_shipped_seed_threads_to_a_full_first_turn_without_pyaml(render: Path) -> None:
    outcome = _thread(render)

    assert outcome["forbidden"] == [], (
        f"threading the first turn imported {', '.join(outcome['forbidden'])}"
    )
    assert outcome["osprey"].startswith(str(CHECKOUT / "src") + os.sep), outcome["osprey"]
    seen = outcome["seen_per_shot"]
    assert seen[0] == FIRST_SHOT_SEEN, (
        f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: the uncorrected first turn reaches "
        f"{seen[0]} BPMs, pinned at {FIRST_SHOT_SEEN}"
    )
    assert seen[-1] == BPMS, (
        f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: after {outcome['steps']} "
        f"correction steps {seen[-1]} of {BPMS} BPMs see the first turn (per shot: {seen})"
    )
    assert outcome["steps"] == STEPS, (
        f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: the first turn threads in "
        f"{outcome['steps']} correction steps, pinned at {STEPS} (BPMs seen per shot: {seen})"
    )
