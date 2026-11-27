"""The in-process simulator's served nominals against the per-address nominal golden, and its served noise against the seeds.

``tests/facility/golden/nominal_in_process.json`` holds, for every scalar demo
address, the value the in-process simulator holds with no scenario active: a
number as held, a bool or enum as its label. Its ``_reproduce`` command
captures it (:func:`capture`) from the demo build's simulator view through a
composite; these tests build that composite from the shared build and hold
every float to the golden to 1e-12, a beam position a physics model computes
to the solver's precision, and every bool to the golden's label. A waveform
has no scalar nominal, so the golden holds none.

The channels a lattice physics model computes also read that model: each
``SR`` beam position monitor's horizontal position reads the deck's closed
orbit, the wired RF cavity's frequency setpoint and readback read the deck's RF
frequency, each ``BTS`` beam position monitor's position reads the single-pass
trajectory of the ``BTS`` deck, and each ``BR`` beam position monitor's
position reads the closed orbit of the ``BR`` deck. The ideal decks' monitors
read within a stated bound of zero.

The served noise is the seed's: a float readback's read at an instant adds its
seed's ``noise`` times the channel's keyed normal draw for the instant, on top
of its drift.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests._builds import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests/facility/golden/nominal_in_process.json"

#: A served value's tolerance against the golden.
TOLERANCE = 1e-12

#: A physics-computed beam position's tolerance against the golden, in millimetres:
#: solver round-off is platform-dependent (FMA and BLAS paths differ between macOS arm64
#: and Linux x86_64).
ORBIT_TOLERANCE = 1e-9

#: The labels of a bool channel.
TRUE, FALSE = "TRUE", "FALSE"

#: The lattice physics model that serves the ``SR`` channels.
PHYSICS_MODEL = "SR"

#: The single-pass physics model that serves the ``BTS`` beam position monitors.
BTS_MODEL = "BTS"

#: The periodic physics model that serves the ``BR`` beam position monitors.
BR_MODEL = "BR"

#: The storage ring's beam lifetime, which the top-off current decays with,
#: and its operation mode.
LIFETIME = "SR:DIAG:DCCT:01:LIFETIME:RB"
MODE = "SR:OPS:MODE:RB"

#: The operation mode's first option and the label it serves at nominal.
MODE_FIRST = "User operation"
MODE_LABEL = "Machine studies"

#: The storage ring's stored current, a top-off sawtooth.
SR_DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"

#: The served waveforms: a trace has no scalar nominal, so the golden holds none.
WAVEFORMS = frozenset({"ES1:DET:MCA:01:SPECTRUM:WAVEFORM", "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"})

#: An ``SR`` beam position monitor's horizontal position.
SR_BPM_X = re.compile(r"SR:DIAG:BPM:\d+:POSITION:X")

#: A ``BTS`` beam position monitor's position, in either plane.
BTS_BPM = re.compile(r"BTS:DIAG:BPM:\d+:POSITION:[XY]")

#: A ``BR`` beam position monitor's position, in either plane.
BR_BPM = re.compile(r"BR:DIAG:BPM:\d+:POSITION:[XY]")

#: The wired RF cavity's frequency pair, served from the deck's RF frequency.
WIRED_CAVITY = ("SR:RF:CAVITY:01:FREQUENCY:RB", "SR:RF:CAVITY:01:FREQUENCY:SP")

#: The deck's RF frequency, in MHz, to the precision a cavity reading is held to.
DECK_RF_MHZ = 500.417
DECK_RF_TOLERANCE_MHZ = 1e-3

#: The largest closed-orbit offset a monitor of the ideal deck reads, in millimetres.
CLOSED_ORBIT_BOUND_MM = 0.1

#: The largest trajectory offset a ``BTS`` monitor of the ideal deck reads, in millimetres.
TRAJECTORY_BOUND_MM = 1e-4

#: The instants the served noise is sampled at, in epoch seconds.
INSTANTS = 1.7e9 + 0.137 * np.arange(16)


@cache
def golden() -> dict[str, dict[str, Any]]:
    """Every row of the golden file: ``{address: {nominal}}``."""
    channels: dict[str, dict[str, Any]] = json.loads(GOLDEN.read_text(encoding="utf-8"))["channels"]
    return channels


def close(actual: float, expected: float, tolerance: float = TOLERANCE) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


class Served:
    """The composite the in-process simulator builds from the demo simulator view."""

    def __init__(self, view: Path) -> None:
        from osprey_connectors.simulation.composite import Composite

        self.composite = Composite(view, model_log=False)
        self.channels: dict[str, dict[str, Any]] = {
            channel["address"]: channel
            for channel in json.loads((view / "variables.json").read_text(encoding="utf-8"))[
                "channels"
            ]
        }
        self.seeds: dict[str, dict[str, Any]] = json.loads(
            (view / "seeds.json").read_text(encoding="utf-8")
        )["seeds"]
        self.held: dict[str, Any] = self.composite.held(sorted(self.channels))

    def value_type(self, address: str) -> str:
        return str(self.channels[address].get("value_type") or "float")

    def is_float_readback(self, address: str) -> bool:
        return (
            self.value_type(address) == "float" and self.channels[address].get("role") != "setpoint"
        )


@pytest.fixture(scope="module")
def served(built_demo: BuiltProject) -> Served:
    pytest.importorskip("at")
    return Served(built_demo.build_dir / "data" / "simulator")


def capture(built: BuiltProject) -> dict[str, Any]:
    """The golden document for ``built``: every scalar address's held value, sorted."""
    from tests.facility.test_demo_census import REPRODUCE

    served = Served(built.build_dir / "data" / "simulator")
    addresses = sorted(a for a in served.channels if served.value_type(a) != "waveform")
    header = {
        "substrate": "in_process",
        "render": "the example facility's simulator view, built by tests._builds.build_demo",
        "active_scenarios": [],
        "addresses": {"count": len(addresses), "excluded": "every waveform"},
        "nominal": (
            "the composite's held value, without texture or noise; a bool or enum as its label"
        ),
    }
    return {
        "_reproduce": REPRODUCE,
        "header": {"command": REPRODUCE, **header},
        "channels": {address: {"nominal": served.held[address]} for address in addresses},
    }


def sr_monitors() -> set[str]:
    """The ``SR`` beam position monitors' horizontal positions, served from the deck."""
    return {address for address in golden() if SR_BPM_X.fullmatch(address)}


def bts_monitors() -> set[str]:
    """The ``BTS`` beam position monitor positions, served from the ``BTS`` deck."""
    return {address for address in golden() if BTS_BPM.fullmatch(address)}


def br_monitors() -> set[str]:
    """The ``BR`` beam position monitor positions, served from the ``BR`` deck."""
    return {address for address in golden() if BR_BPM.fullmatch(address)}


def test_the_golden_holds_every_served_channel_but_the_waveforms(served: Served) -> None:
    unheld = set(served.channels) - set(golden())
    assert unheld == WAVEFORMS
    assert {a for a in served.channels if served.value_type(a) == "waveform"} == WAVEFORMS
    assert set(golden()) <= set(served.channels)


def test_the_lifetime_reads_the_topoff_seed_and_the_mode_its_label(served: Served) -> None:
    from osprey_connectors.simulation import topoff

    assert served.value_type(LIFETIME) == "float"
    assert float(served.held[LIFETIME]) == served.seeds[SR_DCCT]["topoff"]["lifetime_h"]
    assert topoff.held_value(served.seeds[LIFETIME], served.seeds.get) == float(
        served.held[LIFETIME]
    )
    assert served.value_type(MODE) == "enum"
    assert served.channels[MODE]["options"][0] == MODE_FIRST
    assert served.held[MODE] == MODE_LABEL
    assert served.composite.get([MODE])[MODE] == MODE_LABEL
    assert topoff.topoff_record(served.seeds[SR_DCCT]) is not None


def test_float_nominals_equal_the_golden(served: Served) -> None:
    floats = [a for a in golden() if served.value_type(a) == "float"]
    orbit = sr_monitors() | bts_monitors() | br_monitors()
    moved = {
        a
        for a in floats
        if not close(
            float(served.held[a]),
            golden()[a]["nominal"],
            ORBIT_TOLERANCE if a in orbit else TOLERANCE,
        )
    }
    assert len(floats) == 2672
    assert sorted(moved) == []


def test_the_physics_channels_read_the_physics_model(served: Served) -> None:
    owners = {served.channels[a]["owner"] for a in sr_monitors() | set(WIRED_CAVITY)}
    assert owners == {PHYSICS_MODEL}
    assert {served.channels[a]["owner"] for a in bts_monitors()} == {BTS_MODEL}
    assert {served.channels[a]["owner"] for a in br_monitors()} == {BR_MODEL}
    assert len(br_monitors()) == 40
    for address in sorted(br_monitors()):
        assert abs(float(served.held[address])) < TRAJECTORY_BOUND_MM, address
    assert len(sr_monitors()) == 72
    for address in sorted(sr_monitors()):
        assert abs(float(served.held[address])) < CLOSED_ORBIT_BOUND_MM, address
    assert len(bts_monitors()) == 12
    for address in sorted(bts_monitors()):
        assert abs(float(served.held[address])) < TRAJECTORY_BOUND_MM, address
    for address in WIRED_CAVITY:
        assert abs(float(served.held[address]) - DECK_RF_MHZ) < DECK_RF_TOLERANCE_MHZ, address


def test_bool_labels_equal_the_golden(served: Served) -> None:
    bools = [a for a in golden() if served.value_type(a) == "bool"]
    assert len(bools) == 2096
    read = served.composite.get(bools)
    for address in bools:
        label = served.held[address]
        assert label in (TRUE, FALSE), address
        assert label == golden()[address]["nominal"], address
        assert read[address] == label, address


def test_a_read_adds_the_seed_noise_times_the_channel_s_keyed_draw(served: Served) -> None:
    from osprey_connectors.simulation import series

    counters_ms = np.rint(INSTANTS * 1000.0).astype(np.int64)
    checked = 0
    for address in sorted(golden()):
        seed = served.seeds.get(address, {})
        sigma = (seed.get("noise") or {}).get("absolute")
        if not served.is_float_readback(address) or not sigma:
            continue
        if "clamp" in seed or "linear" in seed:
            continue
        group = served.composite.readout_group(address)
        levels = {name: float(served.held[name]) for name in group}
        reading = served.composite.readings(levels, INSTANTS)[address] - levels[address]
        key = series.channel_key_bytes(address)
        drift = seed.get("drift")
        if drift:
            reading = reading - series.wander(
                key, INSTANTS, float(drift["amplitude"]), float(drift["period_s"])
            )
        expected = sigma * series.keyed_normals(key, counters_ms)
        np.testing.assert_allclose(reading, expected, rtol=1e-9, atol=TOLERANCE, err_msg=address)
        checked += 1
    assert checked == 786
