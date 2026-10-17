"""The injector's gate chain, as the generator writes it and the simulator serves it.

The generator owns one ordered chain, GTL -> LN -> LTB -> BR -> BTS -> SR
injection, in the element table's beam order, and writes it as explicit
``gate`` seeds. A screen's paddle readback ``TRUE`` gates every BPM sum, every
transmission/charge proxy and every BPM ``STATUS:VALID`` after it in beam order,
and lights its own lamp; the gun's high voltage or heater on-status ``FALSE``
gates every one of those readings; any paddle in sets the "all paddles out"
count to 1. A BPM's positions carry no gate, and no storage ring channel does.

The committed ``seeds.yaml`` holds exactly what the generator writes. In the
nominal state of the built example facility no gate is active, and inserting
any one paddle gates what the chain says it gates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from osprey.facility.sources import read_yaml
from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITY_DEMO = REPO_ROOT / "scripts/facility_demo"
SEEDS = REPO_ROOT / "src/osprey/templates/facilities/example/seeds.yaml"

GUN_HV = "GTL:GUN:EGUN:01:HV_ON:RB"
GUN_HEATER = "GTL:GUN:EGUN:01:HEATER_ON:RB"
PADDLE_COUNT = "BTS:DIAG:PADDLES:01:COUNT:INSERTED"

#: The roles of the readings the beam gates, and what each reads when gated.
BEAM_ROLES = ("bpm_sum", "ict_charge", "bpm_valid")


def _module(name: str) -> ModuleType:
    """Import a ``scripts/facility_demo`` module, which imports its siblings by bare name."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, FACILITY_DEMO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the paddle parametrize below reads the loaded table, and decorators are evaluated at import; scripts/ is not a package, so the
    # module is registered for its dataclasses and its siblings are found on a
    # sys.path entry held only while it executes.
    sys.modules[name] = module
    added = str(FACILITY_DEMO) not in sys.path
    if added:
        sys.path.insert(0, str(FACILITY_DEMO))
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(str(FACILITY_DEMO))
    return module


@cache
def seeds() -> dict[str, dict[str, Any]]:
    """The committed seeds by address."""
    return dict(read_yaml(SEEDS.read_text(encoding="utf-8")))


@cache
def gated() -> dict[str, dict[str, Any]]:
    """The committed ``gate`` seeds by address."""
    return {address: seed["gate"] for address, seed in seeds().items() if "gate" in seed}


@cache
def rows() -> list[Any]:
    """The element table's rows, one per address, in beam order."""
    elements = _module("_elements")
    seen: dict[str, Any] = {}
    for row in elements.load_elements():
        seen.setdefault(row.demo_name, row)
    order = {section: i for i, section in enumerate(elements.SECTIONS)}
    return sorted(seen.values(), key=lambda row: order[row.section])


def by_role(*roles: str) -> list[str]:
    return [row.demo_name for row in rows() if row.role in roles]


def paddles() -> list[str]:
    return by_role("tv_paddle_rb")


def position(address: str) -> int:
    return [row.demo_name for row in rows()].index(address)


# --- the written chain -----------------------------------------------------------


def test_the_committed_gates_are_exactly_what_the_generator_writes():
    assert gated() == _module("_injector").gates()


def test_gtl_tv1_gates_bpm2_sum_and_not_bpm1_sum():
    screen = "GTL:DIAG:TV:01:PADDLE:RB"

    assert gated()["GTL:DIAG:BPM:02:SUM:RB"]["inputs"][screen] == "TRUE"
    assert screen not in gated()["GTL:DIAG:BPM:01:SUM:RB"]["inputs"]


def test_no_storage_ring_channel_and_no_bpm_position_carries_a_gate():
    positions = set(by_role("bpm_x", "bpm_y"))

    assert positions
    assert not [a for a in gated() if a.startswith("SR:")]
    assert not positions & set(gated())


def test_every_beam_reading_is_gated_by_the_gun_and_every_paddle_upstream_of_it():
    for address in by_role(*BEAM_ROLES):
        upstream = {p: "TRUE" for p in paddles() if position(p) < position(address)}
        expected = {GUN_HEATER: "FALSE", GUN_HV: "FALSE", **upstream}

        assert gated()[address]["inputs"] == expected, address


def test_a_gated_beam_reading_drops_by_a_factor_and_never_to_zero():
    for address in by_role("bpm_sum", "ict_charge"):
        factor = gated()[address]["gated"]

        assert 0.0 < factor <= 0.1, address
    for address in by_role("bpm_valid"):
        assert gated()[address]["gated"] == "FALSE", address


def test_each_lamp_is_lit_by_its_own_paddle_alone():
    lamps = by_role("tv_lamp_rb")

    assert len(lamps) == len(paddles()) == 21
    for lamp in lamps:
        assert gated()[lamp] == {
            "inputs": {lamp.removesuffix("LAMP:RB") + "PADDLE:RB": "TRUE"},
            "gated": "TRUE",
        }


def test_any_paddle_sets_all_paddles_out_to_one():
    gate = gated()[PADDLE_COUNT]

    assert seeds()[PADDLE_COUNT]["nominal"] == 0
    assert gate == {"inputs": dict.fromkeys(sorted(paddles()), "TRUE"), "gated": 1}


def test_only_the_chain_carries_gates():
    chain = set(by_role(*BEAM_ROLES, "tv_lamp_rb", "paddle_count"))

    assert set(gated()) == chain


# --- the served chain ------------------------------------------------------------


class Served:
    """The composite the in-process simulator builds from the control-assistant view."""

    def __init__(self, view: Path) -> None:
        from osprey_connectors.simulation.composite import Composite

        self.composite = Composite(view, model_log=False)
        self.seeds: dict[str, dict[str, Any]] = json.loads(
            (view / "seeds.json").read_text(encoding="utf-8")
        )["seeds"]

    def held(self, addresses: list[str]) -> dict[str, Any]:
        return self.composite.held(sorted(addresses))

    def nominal(self, address: str) -> Any:
        return self.seeds[address]["nominal"]


@pytest.fixture(scope="module")
def served(built_control_assistant: BuiltProject) -> Served:
    pytest.importorskip("at")
    return Served(built_control_assistant.build_dir / "data" / "simulator")


def _ungated(served: Served, held: dict[str, Any]) -> list[str]:
    """The addresses of ``held`` that do not read their seed nominal."""
    off = []
    for address, value in held.items():
        nominal = served.nominal(address)
        if isinstance(nominal, float):
            if value != pytest.approx(nominal):
                off.append(address)
        elif value != nominal:
            off.append(address)
    return off


def test_in_the_nominal_state_no_gate_is_active(served: Served):
    targets = sorted(gated())

    assert all("gate" in served.seeds[a] for a in targets)
    assert _ungated(served, served.held(targets)) == []


@pytest.mark.parametrize("paddle", paddles())
def test_any_paddle_in_sets_all_paddles_out_and_lights_its_lamp(served: Served, paddle: str):
    lamp = paddle.removesuffix("PADDLE:RB") + "LAMP:RB"
    setpoint = paddle.removesuffix("RB") + "SP"

    served.composite.set({setpoint: "TRUE"})
    try:
        held = served.held([PADDLE_COUNT, lamp])

        assert held[PADDLE_COUNT] == 1
        assert held[lamp] == "TRUE"
    finally:
        served.composite.set({setpoint: "FALSE"})
    assert _ungated(served, served.held(sorted(gated()))) == []


def test_gtl_tv1_in_drops_bpm2_sum_and_leaves_bpm1_sum(served: Served):
    bpm1, bpm2 = "GTL:DIAG:BPM:01:SUM:RB", "GTL:DIAG:BPM:02:SUM:RB"

    served.composite.set({"GTL:DIAG:TV:01:PADDLE:SP": "TRUE"})
    try:
        held = served.held([bpm1, bpm2])
    finally:
        served.composite.set({"GTL:DIAG:TV:01:PADDLE:SP": "FALSE"})

    assert held[bpm1] == pytest.approx(served.nominal(bpm1))
    assert 0.0 < held[bpm2] <= 0.1 * served.nominal(bpm2)
