"""The smallest facility trees through ``osprey build``.

A project whose ``data/facility/`` holds only two authored channel addresses
builds a facility file with those two channels and nothing else, each a
readback by default. A project whose ``data/facility/`` is empty builds one
with no records at all: only the built-in ``texture`` model, no classes, and an
identity folded from the project name. Under a profile with a mock control
system and a ``bluesky:`` block the render carries the Bluesky devices view and
stages no copy of it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import yaml
from click.testing import Result

from osprey.facility import TEXTURE

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

Build = Callable[..., tuple["BuiltProject", Result]]

#: Two authored addresses: no tags, no role, no device and no place.
TWO_CHANNELS = {"records/channels.yaml": [{"id": "LAB:TEMP:01"}, {"id": "LAB:TEMP:02"}]}

#: The profile the minimal trees are also built under: a mock control system, a
#: hierarchical channel finder and a ``bluesky:`` block.
CRITERION_7_PROFILE: dict[str, Any] = {
    "bluesky": {},
    "channel_finder_mode": "hierarchical",
    "config": {"control_system.type": "mock"},
}


def test_two_authored_addresses_build_two_readbacks(build_project: Build) -> None:
    project, result = build_project(TWO_CHANNELS)

    assert result.exit_code == 0, result.output
    facility = project.facility
    assert (len(facility["channels"]), facility["devices"], facility["places"]) == (2, [], [])
    assert [(c["id"], c["role"]) for c in facility["channels"]] == [
        ("LAB:TEMP:01", "readback"),
        ("LAB:TEMP:02", "readback"),
    ]
    for channel in facility["channels"]:
        assert "tags" not in channel
        assert "role" in channel["provenance"]["defaults"]


def test_zero_sources_build_the_texture_model_alone(build_project: Build) -> None:
    project, result = build_project({}, name="min-lab.v2")

    assert result.exit_code == 0, result.output
    facility = project.facility
    assert facility["models"] == [{"name": TEXTURE, "engine": TEXTURE}]
    assert facility["classes"] == []
    assert facility["identity"] == {"code": "min_lab_v2", "name": "min-lab.v2"}
    assert [facility[kind] for kind in ("places", "devices", "channels", "groups")] == [[]] * 4


def test_a_missing_facility_directory_is_zero_sources(build_project: Build) -> None:
    empty, empty_result = build_project({}, name="first")
    missing, missing_result = build_project(None, name="second")

    assert (empty_result.exit_code, missing_result.exit_code) == (0, 0)
    assert missing.facility["identity"] == {"code": "second", "name": "second"}
    assert {**missing.facility, "identity": None} == {**empty.facility, "identity": None}


def test_two_authored_addresses_have_an_empty_limits_view(build_project: Build) -> None:
    project, result = build_project(TWO_CHANNELS)

    assert result.exit_code == 0, result.output
    limits = json.loads((project.build_dir / "data" / "channel_limits.json").read_bytes())
    assert limits == {"_version": "4.0"}


def _bluesky_view(project: BuiltProject) -> dict[str, Any]:
    from osprey.services.bluesky_bridge.devices._specs_from_file import validate_device_document

    text = (project.build_dir / "data" / "bluesky_devices.yml").read_text(encoding="utf-8")
    assert text.split("\n", 1)[0] == "schema: osprey.facility.bluesky_devices/1"
    document = yaml.safe_load(text)
    assert validate_device_document(document) == []
    return document


def _staged(project: BuiltProject) -> bool:
    return (project.build_dir / "services" / "bluesky" / "bluesky_devices.yml").exists()


def test_two_authored_addresses_are_two_read_only_bluesky_signals(build_project: Build) -> None:
    project, result = build_project(TWO_CHANNELS, profile=CRITERION_7_PROFILE)

    assert result.exit_code == 0, result.output
    document = _bluesky_view(project)
    assert document["settables"] == []
    assert document["readables"] == [
        {"name": "LAB:TEMP:01", "pv": "LAB:TEMP:01"},
        {"name": "LAB:TEMP:02", "pv": "LAB:TEMP:02"},
    ]
    assert not _staged(project), "a mock control system stages no device file"


def test_zero_sources_write_a_bluesky_view_with_no_device(build_project: Build) -> None:
    project, result = build_project({}, profile=CRITERION_7_PROFILE)

    assert result.exit_code == 0, result.output
    document = _bluesky_view(project)
    assert (document["settables"], document["readables"]) == ([], [])
    assert not _staged(project), "a mock control system stages no device file"
