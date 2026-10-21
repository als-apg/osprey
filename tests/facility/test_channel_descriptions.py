"""A channel's description: stated by a source, else composed by the build.

A channel that states no description but names a ``signal`` is described as
``<owner> <signal words> (<unit>)``, where the owner is its device's label (else
its id), the devices it is an endpoint of, or the place it is on; the build
records the composed description in ``provenance.defaults``. A stated or fixed
description is kept, and a channel with no signal gets none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osprey.facility.build import build_facility
from osprey.facility.combine import FIXES_HEADER
from tests.facility._synthetic_trees import write_tree


def _channel(
    tmp_path: Path, channel: dict[str, Any], fixes: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The built ``channel``, imported when ``fixes`` correct it, else authored."""
    tree: dict[str, Any] = {
        "records/places.yaml": [{"id": "M"}],
        "records/devices.yaml": [
            {"id": "M/Q1", "class": "Quadrupole", "label": "Q 1"},
            {"id": "M/Q2", "class": "Quadrupole"},
        ],
    }
    if fixes is None:
        tree["records/channels.yaml"] = [channel]
    else:
        tree["imported/mml/channels.yaml"] = [channel]
        tree["fixes.yaml"] = {"schema": FIXES_HEADER, "fixes": fixes}
    document = build_facility(write_tree(tmp_path / "facility", tree), project_name="demo")
    (built,) = [c for c in document["channels"] if c["id"] == channel["id"]]
    return built


def test_a_stated_description_is_kept(tmp_path: Path) -> None:
    channel = _channel(
        tmp_path,
        {
            "id": "Q1:SP",
            "role": "setpoint",
            "on": {"device": "M/Q1"},
            "signal": "current_setpoint",
            "description": "the stated one",
        },
    )

    assert channel["description"] == "the stated one"
    assert "description" not in channel["provenance"]["defaults"]


def test_a_signal_channel_on_a_labelled_device_is_described_by_label_signal_and_unit(
    tmp_path: Path,
) -> None:
    channel = _channel(
        tmp_path,
        {
            "id": "Q1:SP",
            "role": "setpoint",
            "on": {"device": "M/Q1"},
            "signal": "current_setpoint",
            "unit": "A",
        },
    )

    assert channel["description"] == "Q 1 current setpoint (A)"
    assert "description" in channel["provenance"]["defaults"]


def test_an_unlabelled_device_is_named_by_its_id(tmp_path: Path) -> None:
    channel = _channel(
        tmp_path, {"id": "Q2:RB", "on": {"device": "M/Q2"}, "signal": "current_readback"}
    )

    assert channel["description"] == "M/Q2 current readback"


def test_a_shared_endpoint_names_every_device(tmp_path: Path) -> None:
    channel = _channel(
        tmp_path,
        {"id": "BUS:RB", "endpoint_of": ["M/Q2", "M/Q1"], "signal": "current_readback"},
    )

    assert channel["description"] == "Q 1, M/Q2 current readback"


def test_a_channel_on_a_place_names_the_place(tmp_path: Path) -> None:
    channel = _channel(
        tmp_path,
        {"id": "M:TEMP", "on": {"place": "M"}, "signal": "temperature_readback", "unit": "C"},
    )

    assert channel["description"] == "M temperature readback (C)"


def test_a_channel_with_no_signal_gets_no_description(tmp_path: Path) -> None:
    channel = _channel(tmp_path, {"id": "Q1:X", "on": {"device": "M/Q1"}, "unit": "A"})

    assert "description" not in channel
    assert "description" not in channel["provenance"]["defaults"]


def test_a_fix_set_description_wins(tmp_path: Path) -> None:
    fix = {
        "op": "set",
        "kind": "channel",
        "id": "Q1:SP",
        "fields": {"description": "the fixed one"},
        "why": "The source states none.",
    }
    channel = _channel(
        tmp_path,
        {"id": "Q1:SP", "role": "setpoint", "on": {"device": "M/Q1"}, "signal": "current_setpoint"},
        fixes=[fix],
    )

    assert channel["description"] == "the fixed one"
    assert "description" not in channel["provenance"]["defaults"]
