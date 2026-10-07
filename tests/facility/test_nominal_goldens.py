"""The mock's served nominals against the per-address nominal golden, and its served noise against the seeds.

``tests/facility/golden/nominal_mock.json`` holds, for every demo address, the
nominal the mock serves with no scenario active. The mock serves the
control-assistant build's simulator view through a composite, so these tests
build that composite from the shared build and hold it to the golden to 1e-12
on every address, except the declared re-baselines:

* every channel the lattice physics model computes and the golden holds a
  different value for: each ``SR`` beam position monitor's horizontal
  position, which reads the deck's closed orbit, and the wired RF cavity's
  frequency setpoint and readback, which read the deck's RF frequency;
* every bool channel, which the golden holds as a number: a bool reads its
  label, ``TRUE`` where the golden's nominal is non-zero, with no noise.

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

from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests/facility/golden/nominal_mock.json"

#: A served value's tolerance against the golden.
TOLERANCE = 1e-12

#: The labels of a bool channel.
TRUE, FALSE = "TRUE", "FALSE"

#: The lattice physics model that serves the ``SR`` channels.
PHYSICS_MODEL = "SR"

#: The deck machine's instruments the golden's capture predates.
ADDITIONS = frozenset({"SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y", "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"})

#: An ``SR`` beam position monitor's horizontal position.
SR_BPM_X = re.compile(r"SR:DIAG:BPM:\d+:POSITION:X")

#: The wired RF cavity's frequency pair, served from the deck's RF frequency.
WIRED_CAVITY = ("SR:RF:CAVITY:01:FREQUENCY:RB", "SR:RF:CAVITY:01:FREQUENCY:SP")

#: The deck's RF frequency, in MHz, to the precision a cavity reading is held to.
DECK_RF_MHZ = 500.417
DECK_RF_TOLERANCE_MHZ = 1e-3

#: The largest closed-orbit offset a monitor of the ideal deck reads, in metres.
CLOSED_ORBIT_BOUND_M = 1e-4

#: The instants the served noise is sampled at, in epoch seconds.
INSTANTS = 1.7e9 + 0.137 * np.arange(16)


@cache
def golden() -> dict[str, dict[str, float]]:
    """The golden's channels: ``{address: {nominal}}``."""
    channels: dict[str, dict[str, float]] = json.loads(GOLDEN.read_text(encoding="utf-8"))[
        "channels"
    ]
    return channels


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


class Served:
    """The composite a mock builds from the control-assistant simulator view."""

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
def served(built_control_assistant: BuiltProject) -> Served:
    pytest.importorskip("at")
    return Served(built_control_assistant.build_dir / "data" / "simulator")


def rebaselined() -> set[str]:
    """The physics channels the golden holds another value for."""
    return {address for address in golden() if SR_BPM_X.fullmatch(address)} | set(WIRED_CAVITY)


def test_the_golden_holds_every_served_channel_but_the_deck_instruments(served: Served) -> None:
    assert set(served.channels) - set(golden()) == ADDITIONS
    assert set(golden()) <= set(served.channels)


def test_float_nominals_equal_the_golden_but_the_declared_rebaselines(served: Served) -> None:
    floats = [a for a in golden() if served.value_type(a) == "float"]
    moved = {a for a in floats if not close(float(served.held[a]), golden()[a]["nominal"])}
    assert len(floats) == 1662
    assert sorted(moved) == sorted(rebaselined())


def test_the_rebaselined_channels_read_the_physics_model(served: Served) -> None:
    owners = {served.channels[a]["owner"] for a in rebaselined()}
    assert owners == {PHYSICS_MODEL}
    monitors = sorted(a for a in rebaselined() if SR_BPM_X.fullmatch(a))
    assert len(monitors) == 72
    for address in monitors:
        assert abs(float(served.held[address])) < CLOSED_ORBIT_BOUND_M, address
    for address in WIRED_CAVITY:
        assert abs(float(served.held[address]) - DECK_RF_MHZ) < DECK_RF_TOLERANCE_MHZ, address


def test_bool_labels_follow_the_golden(served: Served) -> None:
    bools = [a for a in golden() if served.value_type(a) == "bool"]
    assert len(bools) == 1246
    read = served.composite.get(bools)
    for address in bools:
        label = served.held[address]
        assert label in (TRUE, FALSE), address
        assert (label == TRUE) == (golden()[address]["nominal"] != 0.0), address
        assert read[address] == label, address


def test_a_read_adds_the_seed_noise_times_the_channel_s_keyed_draw(served: Served) -> None:
    from osprey_connectors.simulation import series

    counters_ms = np.rint(INSTANTS * 1000.0).astype(np.int64)
    checked = 0
    for address in sorted(golden()):
        seed = served.seeds.get(address, {})
        if not served.is_float_readback(address) or not seed.get("noise"):
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
        expected = seed["noise"] * series.keyed_normals(key, counters_ms)
        np.testing.assert_allclose(reading, expected, rtol=1e-9, atol=TOLERANCE, err_msg=address)
        checked += 1
    assert checked == 225
