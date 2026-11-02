"""The build derives the channels a served simulator proves itself by from the facility file.

Each proof relates the rendered value to the facility file a build wrote: the
probe is the first readback by address the served simulator holds, the
freshness check the first of those whose seed states ``noise``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_archiver import FRESHNESS_CONFIG_KEY, VAArchiverConfig
from osprey.facility import DEFAULT_ROLE, TEXTURE
from osprey.facility.records import ChannelSelection, FacilityRecords
from osprey.facility.served import (
    NO_READBACK_WARNING,
    SERVED_PROBE_KEYS,
    resolve_served,
    served_channel_fill,
)
from osprey_connectors.types import LIVE_STANDIN, VIRTUAL_ACCELERATOR
from tests._builds import BuiltProject, init_project, run_build

VA_PROBE_KEY = SERVED_PROBE_KEYS[0]


def _channel(address: str, **slots: Any) -> dict[str, Any]:
    return {"id": address, **slots}


def _noisy(address: str, **slots: Any) -> dict[str, Any]:
    return _channel(address, simulation={"nominal": 1.0, "noise": 0.1}, **slots)


#: A render whose connector table carries the simulator.
RENDERED: dict[str, Any] = {
    "control_system": {"connector": {VIRTUAL_ACCELERATOR: {"timeout_s": 5}}}
}


def _fill(
    document: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    va_archiver: VAArchiverConfig | None = None,
) -> dict[str, Any]:
    return served_channel_fill(FacilityRecords(document), RENDERED, profile or {}, va_archiver)


def _is_readback(channel: Mapping[str, Any]) -> bool:
    return channel.get("role", DEFAULT_ROLE) == "readback"


def _scope(records: FacilityRecords, served: list[str]) -> list[Mapping[str, Any]]:
    """The served simulator's channels: its models' selections and every channel in none."""
    selected = {
        str(model["name"]): {
            str(channel["id"])
            for channel in records.select_channels(ChannelSelection(model=str(model["name"])))
        }
        for model in records.of_kind("model")
    }
    physics = set(served) & set(selected) - {TEXTURE}
    if not physics:
        return list(records.of_kind("channel"))
    return [
        channel
        for channel in records.of_kind("channel")
        if any(str(channel["id"]) in selected[name] for name in physics)
        or not any(str(channel["id"]) in held for held in selected.values())
    ]


@pytest.fixture(scope="module")
def built(built_control_assistant: BuiltProject) -> dict[str, Any]:
    root = built_control_assistant.repo / "build"
    config = yaml.safe_load((root / "config.yml").read_text(encoding="utf-8"))
    records = FacilityRecords(built_control_assistant.facility)
    served = resolve_served(config, records)
    return {
        "config": config,
        "records": records,
        "readbacks": [c for c in _scope(records, served) if _is_readback(c)],
    }


def test_the_probe_is_the_first_readback_the_served_simulator_holds(built: dict) -> None:
    connector = built["config"]["control_system"]["connector"]
    probe = connector[VIRTUAL_ACCELERATOR]["probe_channel"]
    assert built["records"].record("channel", probe) is not None
    assert probe == min(str(channel["id"]) for channel in built["readbacks"])


def test_the_stand_in_proves_the_channel_the_simulator_proves(built: dict) -> None:
    connector = built["config"]["control_system"]["connector"]
    assert (
        connector[LIVE_STANDIN]["probe_channel"] == connector[VIRTUAL_ACCELERATOR]["probe_channel"]
    )


def test_the_freshness_check_watches_the_first_noisy_readback(built: dict) -> None:
    (check,) = built["config"]["health"]["categories"]["archiver"]["checks"]
    channel = built["records"].require("channel", check["channel"])
    assert channel["simulation"]["noise"] is not None
    noisy = [c for c in built["readbacks"] if (c.get("simulation") or {}).get("noise") is not None]
    assert check["channel"] == min(str(c["id"]) for c in noisy)


def test_a_quiet_facility_gets_a_probe_and_no_health_check() -> None:
    document = {"channels": [_channel("B:RB"), _channel("A:RB")]}
    fill = _fill(document, va_archiver=VAArchiverConfig())
    assert fill[VA_PROBE_KEY] == "A:RB"
    assert FRESHNESS_CONFIG_KEY not in fill


def test_a_facility_with_no_readback_derives_nothing_and_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = {"channels": [_channel("A:SP", role="setpoint", simulation={"noise": 0.1})]}
    assert _fill(document, va_archiver=VAArchiverConfig()) == {}
    assert capsys.readouterr().err == f"{NO_READBACK_WARNING}\n"


def test_the_warning_prints_once_per_build(capsys: pytest.CaptureFixture[str]) -> None:
    records = FacilityRecords({"channels": []})
    reported: set[str] = set()
    for _render in range(3):
        served_channel_fill(records, RENDERED, {}, None, reported=reported)
    assert capsys.readouterr().err == f"{NO_READBACK_WARNING}\n"


def test_a_render_with_no_simulator_block_gains_no_simulator_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = FacilityRecords({"channels": []})
    assert served_channel_fill(records, {}, {}, None) == {}
    assert (
        served_channel_fill(FacilityRecords({"channels": [_channel("A:RB")]}), {}, {}, None) == {}
    )
    assert capsys.readouterr().err == ""


def test_a_stated_probe_is_not_derived_over() -> None:
    document = {"channels": [_channel("A:RB"), _channel("B:RB")]}
    assert VA_PROBE_KEY not in _fill(document, {VA_PROBE_KEY: "B:RB"})


def test_a_stated_freshness_channel_is_kept() -> None:
    document = {"channels": [_noisy("A:RB"), _noisy("B:RB")]}
    archiver = replace(VAArchiverConfig(), freshness_channel="B:RB")
    assert FRESHNESS_CONFIG_KEY not in _fill(document, va_archiver=archiver)


def test_a_profile_that_writes_the_category_owns_it() -> None:
    document = {"channels": [_noisy("A:RB")]}
    profile = {"health.categories.archiver.checks": []}
    assert FRESHNESS_CONFIG_KEY not in _fill(document, profile, VAArchiverConfig())


def test_no_archive_block_gets_no_health_check() -> None:
    assert FRESHNESS_CONFIG_KEY not in _fill({"channels": [_noisy("A:RB")]})


def test_a_device_assigned_channel_of_an_unserved_model_is_out_of_scope() -> None:
    document = {
        "models": [
            {"name": "ON", "wiring": [{"address": "Z:RB"}]},
            {"name": "OFF", "wiring": []},
        ],
        "devices": [{"id": "OFF/D", "model": "OFF"}],
        "channels": [_channel("A:RB", on={"device": "OFF/D"}), _channel("Z:RB")],
    }
    fill = _fill(document, {"simulation.models": ["ON"]})
    assert fill[VA_PROBE_KEY] == "Z:RB"


def test_a_channel_in_no_model_is_served_by_the_texture() -> None:
    document = {
        "models": [{"name": "ON", "wiring": [{"address": "Z:RB"}]}],
        "channels": [_channel("Z:RB"), _channel("A:RB")],
    }
    assert _fill(document, {"simulation.models": ["ON"]})[VA_PROBE_KEY] == "A:RB"


def test_with_no_physics_served_every_channel_is_in_scope() -> None:
    document = {
        "models": [{"name": "OFF", "wiring": [{"address": "A:RB"}]}],
        "channels": [_channel("A:RB"), _channel("B:RB")],
    }
    assert _fill(document, {"simulation.models": []})[VA_PROBE_KEY] == "A:RB"


def test_the_profile_states_the_served_models_over_the_render() -> None:
    document = {
        "models": [
            {"name": "ON", "wiring": [{"address": "Z:RB"}]},
            {"name": "OFF", "wiring": [{"address": "A:RB"}]},
        ],
        "channels": [_channel("A:RB"), _channel("Z:RB")],
    }
    rendered = {**RENDERED, "simulation": {"models": ["OFF"]}}
    fill = served_channel_fill(
        FacilityRecords(document), rendered, {"simulation.models": ["ON"]}, None
    )
    assert fill[VA_PROBE_KEY] == "Z:RB"


def test_hello_world_derives_its_probe_and_no_health_check(tmp_path: Path) -> None:
    repo = init_project(tmp_path, "hello-world", "demo")
    result = run_build(repo)
    assert result.exit_code == 0, result.output

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    probe = config["control_system"]["connector"][VIRTUAL_ACCELERATOR]["probe_channel"]
    facility = json.loads((repo / "build" / "facility.json").read_text(encoding="utf-8"))
    readbacks = [str(c["id"]) for c in facility["channels"] if _is_readback(c)]
    assert probe == min(readbacks)
    assert "archiver" not in ((config.get("health") or {}).get("categories") or {})
