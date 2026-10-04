"""The presets' committed ``data/facility/`` trees.

The control-assistant preset and the two standalone presets commit one demo
tree each. The three committed trees are byte-equal except the
standalones' ``identity.yaml``, which adds the facility name, and the files
only the control-assistant preset carries. Hello-world's tree is hand-authored
and holds the channels its tutorial names.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import build_facility
from osprey.facility.sources import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
APPS = REPO_ROOT / "src/osprey/templates/apps"
CONTROL_ASSISTANT = APPS / "control_assistant/data/facility"
STANDALONES = {
    "ariel_standalone": APPS / "ariel_standalone/data/facility",
    "channel_finder_standalone": APPS / "channel_finder_standalone/data/facility",
}
HELLO_WORLD = APPS / "hello_world/data/facility"
VA_BINDINGS = APPS / "control_assistant/data/simulation/va_bindings.json"
CF_STANDALONE_ADDRESSES = REPO_ROOT / "tests/facility/golden/cf_standalone_addresses.json"

#: The directories only the control-assistant tree carries: its hand-authored
#: knowledge pages, the measurement file and the scenarios.
OMITTED = ("knowledge/", "measurement/", "scenarios/")

#: The addresses the hello-world tutorial names.
HELLO_WORLD_ADDRESSES = [
    "SR:BEAM:CURRENT",
    "SR:MAG:CORR:01:CURRENT:SP",
    "SR:MAG:QD:01:CURRENT:SP",
    "SR:MAG:QF:01:CURRENT:RB",
    "SR:MAG:QF:01:CURRENT:SP",
]


def tree(root: Path) -> dict[str, bytes]:
    """Relative path -> bytes of every file under ``root``."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _omitted(rel: str) -> bool:
    return rel.startswith(OMITTED)


@cache
def built_control_assistant_facility() -> dict[str, Any]:
    """The control-assistant tree's facility file."""
    document: dict[str, Any] = build_facility(CONTROL_ASSISTANT, project_name="ca")
    return document


@pytest.mark.parametrize("name", sorted(STANDALONES))
def test_standalone_tree_equals_the_control_assistant_tree_but_identity(name: str) -> None:
    standalone = tree(STANDALONES[name])
    control_assistant = tree(CONTROL_ASSISTANT)
    assert any(_omitted(rel) for rel in control_assistant)
    expected = {
        rel: data
        for rel, data in control_assistant.items()
        if rel != "identity.yaml" and not _omitted(rel)
    }
    assert {rel: data for rel, data in standalone.items() if rel != "identity.yaml"} == expected
    assert len(expected) == 8


@pytest.mark.parametrize("name", sorted(STANDALONES))
def test_standalone_identity_adds_only_the_facility_name(name: str) -> None:
    ca_lines = (CONTROL_ASSISTANT / "identity.yaml").read_text(encoding="utf-8").splitlines()
    lines = (STANDALONES[name] / "identity.yaml").read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line not in ca_lines] == ["name: Example Research Facility"]
    assert [line for line in lines if line in ca_lines] == ca_lines
    assert "name" not in read_yaml((CONTROL_ASSISTANT / "identity.yaml").read_text())


def test_channel_finder_standalone_keeps_every_address_it_served() -> None:
    served = json.loads(CF_STANDALONE_ADDRESSES.read_text(encoding="utf-8"))["addresses"]
    channels = read_yaml(
        (STANDALONES["channel_finder_standalone"] / "records/channels.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(served) <= {channel["id"] for channel in channels}


def test_paired_readbacks_start_at_their_setpoint_value() -> None:
    document = built_control_assistant_facility()
    channels = {channel["id"]: channel for channel in document["channels"]}
    setpoint_of = {c["pair"]: c["id"] for c in channels.values() if "pair" in c}
    (model,) = [model for model in document["models"] if "deck" in model]
    wiring = {record["address"]: record for record in model["wiring"]}
    paired = [address for address in wiring if address in setpoint_of]
    bound = [b for b in json.loads(VA_BINDINGS.read_text())["bindings"] if b["readback_address"]]
    assert {b["readback_address"] for b in bound} <= set(paired)
    for address in paired:
        assert wiring[address]["default"] == wiring[setpoint_of[address]]["default"], address
    unpaired = [
        address
        for address, record in wiring.items()
        if record["direction"] == "read" and address not in setpoint_of
    ]
    assert len(unpaired) == 148
    assert sum(":BPM:" in address for address in unpaired) == 144
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
