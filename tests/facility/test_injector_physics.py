"""The injector's physics and beam gating, served in-process from the built example facility.

One composite is built from the control-assistant view with a clock the tests
step one simulator tick at a time. Every noise level is measured from that same
composite over a quiet window, never taken from the seeds, so a threshold of
"three sigma" means three sigma of what an operator would see.

The checks are the proposal's success criteria in process:

- a BTS horizontal corrector step worth 1 mrad through its wiring calibration
  moves a downstream BTS BPM's X by more than three sigma of its noise;
- BTS TV3's paddle in lights its lamp and drops the sum of every BTS BPM after
  it in beam order by at least 90 %, leaves the sums before it alone, and keeps
  every position finite with no invalid severity;
- the gun's high voltage off drops every BPM sum and transmission proxy of the
  injector, and no float readback reads NaN or exactly 0;
- an LTB corrector change moves the LTB BPMs after it on the next tick.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from tests.facility.conftest import BuiltProject

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITY_DEMO = REPO_ROOT / "scripts/facility_demo"

#: The instant the stepped clock starts at, in epoch seconds.
START_S = 1.8e9

#: Ticks in the quiet window each noise sigma is measured over.
QUIET_TICKS = 64

#: The corrector kick each step is worth, in radians.
KICK_RAD = 1.0e-3

#: How many of its noise sigmas a reading must move to count as moved.
SIGMAS = 3.0

#: The largest fraction of its quiet level a gated beam reading may keep.
GATED_FRACTION = 0.1

#: The sections whose beam the gun feeds, in beam order.
INJECTOR = ("GTL", "LN", "LTB", "BR", "BTS")

GUN_HV_SP = "GTL:GUN:EGUN:01:HV_ON:SP"
TV3 = "BTS:DIAG:TV:03"
BTS_HCM = "BTS:MAG:HCM:01:CURRENT:SP"
LTB_HCM = "LTB:MAG:HCM:01:CURRENT:SP"


def _module(name: str) -> ModuleType:
    """Import a ``scripts/facility_demo`` module, which imports its siblings by bare name."""
    if str(FACILITY_DEMO) not in sys.path:
        sys.path.insert(0, str(FACILITY_DEMO))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, FACILITY_DEMO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def rows() -> list[Any]:
    """The element table's rows, one per address, in beam order."""
    elements = _module("_elements")
    seen: dict[str, Any] = {}
    for row in elements.load_elements():
        seen.setdefault(row.demo_name, row)
    order = {section: i for i, section in enumerate(elements.SECTIONS)}
    return sorted(seen.values(), key=lambda row: order[row.section])


def position(address: str) -> int:
    return [row.demo_name for row in rows()].index(address)


def by_role(*roles: str, section: str | None = None) -> list[str]:
    """The addresses of ``roles`` in beam order, in ``section`` when given."""
    return [
        row.demo_name
        for row in rows()
        if row.role in roles and (section is None or row.section == section)
    ]


def after(address: str, addresses: list[str]) -> list[str]:
    return [a for a in addresses if position(a) > position(address)]


def before(address: str, addresses: list[str]) -> list[str]:
    return [a for a in addresses if position(a) < position(address)]


class Served:
    """The in-process composite of the control-assistant view, on a clock the test steps."""

    def __init__(self, build_dir: Path) -> None:
        from osprey_connectors.simulation import DEFAULT_TICK_S
        from osprey_connectors.simulation.composite import Composite

        view = build_dir / "data" / "simulator"
        self.tick_s = DEFAULT_TICK_S
        self.now = START_S
        self.composite = Composite(view, model_log=False, clock=lambda: self.now)
        self.channels: dict[str, dict[str, Any]] = {
            channel["address"]: channel
            for channel in json.loads((view / "variables.json").read_text(encoding="utf-8"))[
                "channels"
            ]
        }
        self.wiring: dict[str, dict[str, Any]] = {
            record["address"]: record
            for model in yaml.safe_load(
                (build_dir / "data" / "facility" / "models.yaml").read_text(encoding="utf-8")
            )
            for record in model.get("wiring") or ()
        }

    def tick(self) -> None:
        self.now += self.tick_s

    def read(self, addresses: list[str]) -> dict[str, Any]:
        """The served values one tick later."""
        self.tick()
        return self.composite.get(sorted(addresses))

    def quiet(self, addresses: list[str]) -> tuple[dict[str, float], dict[str, float]]:
        """Each reading's mean and noise sigma over a quiet window of ticks."""
        samples: dict[str, list[float]] = {a: [] for a in addresses}
        for _ in range(QUIET_TICKS):
            for address, value in self.read(addresses).items():
                samples[address].append(float(value))
        means = {a: float(np.mean(v)) for a, v in samples.items()}
        sigmas = {a: float(np.std(v, ddof=1)) for a, v in samples.items()}
        return means, sigmas

    def one_mrad(self, setpoint: str) -> float:
        """The current ``setpoint`` reads after a 1 mrad kick, by its wiring calibration."""
        from osprey.simulation.engines.calibration import (
            curve_from_record,
            to_hardware,
            to_physics,
        )

        curve = curve_from_record(self.wiring[setpoint]["calibration"]["curve"])
        current = float(self.composite.held([setpoint])[setpoint])
        return to_hardware(curve, None, to_physics(curve, current) + KICK_RAD)

    def float_readbacks(self, sections: tuple[str, ...]) -> list[str]:
        return sorted(
            address
            for address, channel in self.channels.items()
            if channel["value_type"] == "float"
            and channel["role"] == "readback"
            and address.split(":", 1)[0] in sections
        )


@pytest.fixture(scope="module")
def served(built_control_assistant: BuiltProject) -> Served:
    pytest.importorskip("at")
    return Served(built_control_assistant.build_dir)


def _moved(
    served: Served, setpoint: str, monitors: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Each monitor's shift on the tick after a 1 mrad step of ``setpoint``, and its sigma."""
    means, sigmas = served.quiet(monitors)
    nominal = served.composite.held([setpoint])[setpoint]
    served.composite.set({setpoint: served.one_mrad(setpoint)})
    try:
        stepped = served.read(monitors)
    finally:
        served.composite.set({setpoint: nominal})
    return {a: abs(float(stepped[a]) - means[a]) for a in monitors}, sigmas


# --- the beam order the checks rely on -------------------------------------------


def test_the_table_places_tv3_between_bts_bpm2_and_bpm3():
    sums = by_role("bpm_sum", section="BTS")
    paddle = f"{TV3}:PADDLE:RB"

    assert before(paddle, sums) == ["BTS:DIAG:BPM:01:SUM:RB", "BTS:DIAG:BPM:02:SUM:RB"]
    assert after(paddle, sums) == [f"BTS:DIAG:BPM:0{n}:SUM:RB" for n in range(3, 7)]


def test_the_stepped_correctors_are_wired_and_have_monitors_after_them(served: Served):
    for setpoint, section in ((BTS_HCM, "BTS"), (LTB_HCM, "LTB")):
        assert setpoint in served.wiring
        assert after(setpoint, by_role("bpm_x", section=section)), setpoint


# --- physics ---------------------------------------------------------------------


def test_a_one_mrad_bts_hcm_step_moves_a_downstream_bts_bpm_x_by_three_sigma(served: Served):
    monitors = after(BTS_HCM, by_role("bpm_x", section="BTS"))

    shifts, sigmas = _moved(served, BTS_HCM, monitors)

    assert all(sigma > 0.0 for sigma in sigmas.values()), sigmas
    assert any(shifts[a] > SIGMAS * sigmas[a] for a in monitors), (shifts, sigmas)


def test_an_ltb_corrector_change_moves_the_downstream_ltb_bpms_on_the_next_tick(served: Served):
    monitors = after(LTB_HCM, by_role("bpm_x", section="LTB"))

    shifts, sigmas = _moved(served, LTB_HCM, monitors)

    assert monitors
    assert [a for a in monitors if not shifts[a] > SIGMAS * sigmas[a]] == [], (shifts, sigmas)


# --- gating ----------------------------------------------------------------------


def test_tv3_paddle_in_lights_its_lamp_and_drops_every_bts_sum_after_it(served: Served):
    sums = by_role("bpm_sum", section="BTS")
    paddle = f"{TV3}:PADDLE:RB"
    downstream, upstream = after(paddle, sums), before(paddle, sums)
    valid = after(paddle, by_role("bpm_valid", section="BTS"))
    positions = by_role("bpm_x", "bpm_y", section="BTS")
    lamp = f"{TV3}:LAMP:RB"
    means, sigmas = served.quiet(sums)

    served.composite.set({f"{TV3}:PADDLE:SP": "TRUE"})
    try:
        read = served.read([*sums, *valid, *positions, lamp, paddle])
        severity = served.composite.output_severity(sorted(positions))
    finally:
        served.composite.set({f"{TV3}:PADDLE:SP": "FALSE"})

    assert read[paddle] == "TRUE"
    assert read[lamp] == "TRUE"
    assert {f"BTS:DIAG:BPM:0{n}:SUM:RB" for n in (4, 5, 6)} <= set(downstream)
    for address in downstream:
        assert 0.0 < read[address] <= GATED_FRACTION * means[address], address
    for address in upstream:
        assert abs(read[address] - means[address]) <= 5 * sigmas[address] + 1e-12, address
    assert {address: read[address] for address in valid} == dict.fromkeys(valid, "FALSE")
    assert [a for a in positions if not math.isfinite(read[a])] == []
    assert severity == {}


def test_gun_hv_off_drops_every_sum_and_transmission_proxy_and_reads_no_nan_or_zero(
    served: Served,
):
    beam = [a for a in by_role("bpm_sum", "ict_charge") if a.split(":", 1)[0] in INJECTOR]
    readbacks = served.float_readbacks(INJECTOR)
    means, _ = served.quiet(beam)

    served.composite.set({GUN_HV_SP: "FALSE"})
    try:
        read = served.read(readbacks)
    finally:
        served.composite.set({GUN_HV_SP: "TRUE"})

    assert by_role("ict_charge") and by_role("bpm_sum")
    assert [a for a in beam if not read[a] <= GATED_FRACTION * means[a]] == []
    assert [a for a in readbacks if not math.isfinite(read[a]) or read[a] == 0.0] == []
