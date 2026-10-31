"""The committed facility trees the presets show.

The bundled example facility is the tree the control-assistant preset shows.
Hello-world's facility is hand-authored and holds
the channels its tutorial names.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from osprey.facility.build import build_facility
from osprey.facility.sources import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FACILITIES = REPO_ROOT / "src/osprey/templates/facilities"
CONTROL_ASSISTANT = FACILITIES / "example"
HELLO_WORLD = FACILITIES / "hello_world"

#: The addresses the hello-world tutorial names.
HELLO_WORLD_ADDRESSES = [
    "SR:BEAM:CURRENT",
    "SR:MAG:CORR:01:CURRENT:SP",
    "SR:MAG:QD:01:CURRENT:SP",
    "SR:MAG:QF:01:CURRENT:RB",
    "SR:MAG:QF:01:CURRENT:SP",
]


@cache
def built_control_assistant_facility() -> dict[str, Any]:
    """The control-assistant tree's facility file."""
    document: dict[str, Any] = build_facility(CONTROL_ASSISTANT, project_name="ca")
    return document


def test_paired_readbacks_start_at_their_setpoint_value() -> None:
    document = built_control_assistant_facility()
    channels = {channel["id"]: channel for channel in document["channels"]}
    setpoint_of = {c["pair"]: c["id"] for c in channels.values() if "pair" in c}
    models = [model for model in document["models"] if "deck" in model]
    assert [model["name"] for model in models] == ["BTS", "LINE", "LTB", "SR"]
    wiring = {record["address"]: record for model in models for record in model["wiring"]}
    paired = [address for address in wiring if address in setpoint_of]
    for address in paired:
        assert wiring[address]["default"] == wiring[setpoint_of[address]]["default"], address
    unpaired = [
        address
        for address, record in wiring.items()
        if record["direction"] == "read" and address not in setpoint_of
    ]
    # SR 148 (144 BPM positions, tune and chromaticity), LINE 8, LTB 12, BTS 12.
    assert len(unpaired) == 180
    assert sum(":BPM:" in address for address in unpaired) == 176
    assert all(wiring[address]["default"] == 0.0 for address in unpaired)


def test_hello_world_tree_builds_with_the_tutorial_channels() -> None:
    document = build_facility(HELLO_WORLD, project_name="hello")
    assert [channel["id"] for channel in document["channels"]] == HELLO_WORLD_ADDRESSES
    setpoints = [c["id"] for c in document["channels"] if c.get("role") == "setpoint"]
    assert setpoints == [a for a in HELLO_WORLD_ADDRESSES if a.endswith(":SP")]


def test_hello_world_limits_hold_the_three_tutorial_records() -> None:
    limits = read_yaml((HELLO_WORLD / "limits.yaml").read_text(encoding="utf-8"))
    assert list(limits) == ["records"]
    assert limits["records"] == [
        {
            "address": "SR:MAG:QF:01:CURRENT:SP",
            "min_value": 0.0,
            "max_value": 300.0,
            "writable": True,
        },
        {
            "address": "SR:MAG:QD:01:CURRENT:SP",
            "min_value": 0.0,
            "max_value": 250.0,
            "writable": True,
        },
        {"address": "SR:BEAM:CURRENT", "writable": False},
    ]
