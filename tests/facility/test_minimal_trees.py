"""The smallest facility trees through ``osprey build``.

A project whose ``data/facility/`` holds only two authored channel addresses
builds a facility file with those two channels and nothing else, each a
readback by default; the same two addresses imported as a header-less channel
list build the same channels. A project whose ``data/facility/`` is empty builds one
with no records at all: only the built-in ``texture`` model, no classes, and an
identity folded from the project name. Both render a simulator view that serves
texture alone, with no status address. The mock connector serving a minimal
tree serves exactly the facility file's addresses, spelled as authored, and
refuses any other. Under a profile with a mock control
system and a ``bluesky:`` block the render carries the Bluesky devices view and
stages no copy of it. The empty tree under a profile that selects the
channel-finder agent with ``channel_finder_mode: in_context`` has no channel
for the in_context index, so the build stops.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.facility import TEXTURE

if TYPE_CHECKING:
    from pathlib import Path

    from tests.facility.conftest import BuiltProject

Build = Callable[..., tuple["BuiltProject", Result]]

#: Two authored addresses: no tags, no role, no device and no place.
TWO_CHANNELS = {"records/channels.yaml": [{"id": "LAB:TEMP:01"}, {"id": "LAB:TEMP:02"}]}

#: The simulator view's files, every one written whatever the tree holds.
SIMULATOR_FILES = (
    "addresses.json",
    "scenarios.json",
    "seeds.json",
    "served_models.json",
    "variables.json",
)

#: A mock control system, the profile the served minimal trees are built under.
MOCK_PROFILE: dict[str, Any] = {
    "config": {
        "control_system.type": "virtual_accelerator",
        "control_system.connector.virtual_accelerator.serving": "in_process",
    }
}

#: Two separator conventions in one namespace: setpoints paired with their
#: readbacks by ``pair``, each seeded with a nominal value.
SR_SETPOINT, SR_READBACK = "SR04U___GDS1PS_AC00", "SR04U___GDS1PS_AM00"
BTS_SETPOINT, BTS_READBACK = "BTS:HCM1:AC", "BTS:HCM1:AM"
MIXED_NAMESPACE: dict[str, Any] = {
    "records/channels.yaml": [
        {
            "id": SR_SETPOINT,
            "role": "setpoint",
            "pair": SR_READBACK,
            "simulation": {"nominal": 12.5},
        },
        {"id": SR_READBACK},
        {
            "id": BTS_SETPOINT,
            "role": "setpoint",
            "pair": BTS_READBACK,
            "simulation": {"nominal": 0.25},
        },
        {"id": BTS_READBACK},
    ],
    "limits.yaml": {
        "records": [
            {"address": SR_SETPOINT, "min_value": 0.0, "max_value": 20.0},
            {"address": BTS_SETPOINT, "min_value": -1.0, "max_value": 1.0},
        ]
    },
}

#: The profile the minimal trees are also built under: a mock control system, a
#: hierarchical channel finder and a ``bluesky:`` block.
CRITERION_7_PROFILE: dict[str, Any] = {
    "bluesky": {},
    "channel_finder_mode": "hierarchical",
    "config": {
        "control_system.type": "virtual_accelerator",
        "control_system.connector.virtual_accelerator.serving": "in_process",
    },
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


def _import_list(tmp_path: Path, text: str) -> tuple[BuiltProject, Result]:
    """Init a repo with no ``data/facility/``, import a channel list into it and build."""
    from osprey.cli.main import cli
    from tests._builds import init_project, run_build
    from tests.facility.conftest import BuiltProject

    repo = init_project(tmp_path, "hello-world", "demo")
    shutil.rmtree(repo / "data" / "facility")
    listing = tmp_path / "two.csv"
    listing.write_text(text, encoding="utf-8")
    imported = CliRunner().invoke(
        cli, ["facility", "import", "list", str(listing), "--repo", str(repo)]
    )
    assert imported.exit_code == 0, imported.output
    return BuiltProject(repo), run_build(repo)


def test_two_listed_addresses_build_the_same_two_readbacks(
    tmp_path: Path, build_project: Build
) -> None:
    authored, authored_result = build_project(TWO_CHANNELS)
    listed, result = _import_list(tmp_path / "listed", "LAB:TEMP:01\nLAB:TEMP:02\n")

    assert (authored_result.exit_code, result.exit_code) == (0, 0), result.output
    assert sorted(p.name for p in (listed.facility_dir / "imported" / "list").iterdir()) == [
        "channels.yaml"
    ]
    facility = listed.facility
    assert (facility["devices"], facility["places"]) == ([], [])
    for channel in facility["channels"]:
        assert channel["provenance"]["sources"] == [
            {"layer": "list", "file": "imported/list/channels.yaml", "fields": []}
        ]
    unsourced = [
        {**channel, "provenance": {**channel["provenance"], "sources": []}}
        for channel in facility["channels"]
    ]
    assert unsourced == [
        {**channel, "provenance": {**channel["provenance"], "sources": []}}
        for channel in authored.facility["channels"]
    ]


def test_zero_sources_build_the_texture_model_alone(build_project: Build) -> None:
    project, result = build_project({}, name="min-lab.v2")

    assert result.exit_code == 0, result.output
    facility = project.facility
    assert facility["models"] == [{"name": TEXTURE, "engine": TEXTURE}]
    assert facility["classes"] == []
    # The zero-source identity folds the profile's project_name, which init
    # proposed from the folder name in compose's spelling (the dot dropped).
    assert facility["identity"] == {"code": "min_labv2", "name": "min-labv2"}
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


def test_zero_sources_under_in_context_stop_with_view_unsupported(build_project: Build) -> None:
    in_context = {
        **CRITERION_7_PROFILE,
        "agents": ["channel-finder"],
        "channel_finder_mode": "in_context",
    }

    _project, result = build_project({}, profile=in_context)

    assert result.exit_code == 1, result.output
    assert result.stderr == (
        "facility: view-unsupported: path channel_finder.pipeline_mode — selects in_context "
        "and the facility has no channel; fix: add a channel, or select another "
        "channel_finder_mode\n"
    )


def _simulator_view(project: BuiltProject) -> dict[str, Any]:
    view = project.build_dir / "data" / "simulator"
    assert sorted(path.name for path in view.iterdir()) == list(SIMULATOR_FILES)
    return {name: json.loads((view / name).read_bytes()) for name in SIMULATOR_FILES}


def test_two_authored_addresses_simulate_texture_alone(build_project: Build) -> None:
    project, result = build_project(TWO_CHANNELS, profile=MOCK_PROFILE)

    assert result.exit_code == 0, result.output
    view = _simulator_view(project)
    assert view["served_models.json"]["models"] == [TEXTURE]
    assert view["addresses.json"] == {
        "schema": "osprey.facility.addresses/1",
        "channels": ["LAB:TEMP:01", "LAB:TEMP:02"],
        "status": [],
    }
    assert {c["owner"] for c in view["variables.json"]["channels"]} == {TEXTURE}


def test_zero_sources_write_the_same_view_set_with_no_channel(build_project: Build) -> None:
    project, result = build_project({}, profile=MOCK_PROFILE)

    assert result.exit_code == 0, result.output
    view = _simulator_view(project)
    assert view["served_models.json"]["models"] == [TEXTURE]
    assert (view["addresses.json"]["channels"], view["addresses.json"]["status"]) == ([], [])
    assert view["variables.json"]["channels"] == []
    assert view["seeds.json"]["seeds"] == {}
    assert view["scenarios.json"]["scenarios"] == [
        {
            "name": "still",
            "description": "Every reading serves without drift, couplings or noise.",
            "still": "all",
        }
    ]


def _writes_enabled(key: str, default: Any = None) -> Any:
    """A config lookup with writes enabled and every other key at its default."""
    return True if key == "control_system.writes_enabled" else default


async def _served(project: BuiltProject) -> Any:
    from osprey_connectors.control_system.va_in_process_connector import VAInProcessConnector
    from tests.facility.served_tree import in_process_config

    connector = VAInProcessConnector()
    view = project.build_dir / "data" / "simulator"
    await connector.connect(in_process_config(view, response_delay_ms=0))
    return connector


async def test_the_in_process_simulator_refuses_an_address_outside_the_facility_file(
    build_project: Build,
) -> None:
    project, result = build_project(TWO_CHANNELS, profile=MOCK_PROFILE)
    assert result.exit_code == 0, result.output

    connector = await _served(project)
    try:
        assert await connector.validate_channel("LAB:TEMP:01") is True
        assert await connector.validate_channel("LAB:TEMP:03") is False
        with pytest.raises(ValueError) as refusal:
            await connector.read_channel("LAB:TEMP:03")
    finally:
        await connector.disconnect()

    assert str(refusal.value) == "LAB:TEMP:03 is not in build/facility.json"


async def test_als_shaped_namespace_round_trip(
    build_project: Build, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, result = build_project(MIXED_NAMESPACE, profile=MOCK_PROFILE)
    assert result.exit_code == 0, result.output

    addresses = [BTS_SETPOINT, BTS_READBACK, SR_SETPOINT, SR_READBACK]
    pairs = {SR_SETPOINT: SR_READBACK, BTS_SETPOINT: BTS_READBACK}
    facility = project.facility
    assert {c["id"] for c in facility["channels"]} == set(addresses)
    assert {c["id"]: c["pair"] for c in facility["channels"] if c["id"] in pairs} == pairs

    view = _simulator_view(project)
    assert view["addresses.json"]["schema"] == "osprey.facility.addresses/1"
    assert view["addresses.json"]["status"] == []
    assert view["addresses.json"]["channels"] == addresses
    assert {c["address"]: c["pair"] for c in view["variables.json"]["channels"]} == {
        SR_SETPOINT: SR_READBACK,
        SR_READBACK: None,
        BTS_SETPOINT: BTS_READBACK,
        BTS_READBACK: None,
    }
    assert {
        c["address"]: (c["writable"], c["value_range"])
        for c in view["variables.json"]["channels"]
        if c["address"] in pairs
    } == {SR_SETPOINT: (True, [0.0, 20.0]), BTS_SETPOINT: (True, [-1.0, 1.0])}

    from osprey_connectors.control_system.base import WriteOutcome

    collapsed = "SR04U_GDS1PS_AC00"
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)
    connector = await _served(project)
    try:
        for address in addresses:
            assert await connector.validate_channel(address) is True
        assert await connector.validate_channel(collapsed) is False
        with pytest.raises(ValueError) as refusal:
            await connector.read_channel(collapsed)

        seeded = [await connector.read_channel(a) for a in (SR_READBACK, BTS_READBACK)]
        assert [reading.value for reading in seeded] == [12.5, 0.25]
        assert [reading.metadata.alarm_severity for reading in seeded] == [None, None]

        sr_write = await connector.write_channel(SR_SETPOINT, 13.0)
        sr_echo = await connector.read_channel(SR_READBACK)
        bts_write = await connector.write_channel(BTS_SETPOINT, 0.5)
        bts_echo = await connector.read_channel(BTS_READBACK)
    finally:
        await connector.disconnect()

    assert str(refusal.value) == f"{collapsed} is not in build/facility.json"
    assert (sr_write.outcome, sr_echo.value) == (WriteOutcome.CONFIRMED, 13.0)
    assert (bts_write.outcome, bts_echo.value) == (WriteOutcome.CONFIRMED, 0.5)
