"""The app contract's surface matches its record for ``APP_CONTRACT``.

``surface()`` reads from the live code the names an application relies on;
``tests/apps/contract/surface.json`` holds them per contract number, written by
``scripts/record_app_contract.py``. A recorded name that is gone, or a manifest
path whose type changed, breaks apps written for this number; a name not yet
recorded is unprotected until it is.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.apps.contract import APP_CONTRACT, SURFACE_SOURCES, surface
from tests._by_path import load_module
from tests._fresh_import import modules_added_by_import

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD = REPO_ROOT / "tests" / "apps" / "contract" / "surface.json"
SCRIPT = REPO_ROOT / "scripts" / "record_app_contract.py"

RAISE = "raise `APP_CONTRACT`, add the history row, re-record"
RERECORD = "re-record; no bump needed"


@pytest.fixture(scope="module")
def recorder():
    return load_module("record_app_contract", SCRIPT)


@pytest.fixture(scope="module")
def live() -> dict[str, Any]:
    return surface()


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    return json.loads(RECORD.read_text(encoding="utf-8"))[str(APP_CONTRACT)]


def drift(recorder, live: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """Every difference between the live surface and the record, with what to do."""
    lines = [f"{line}: {RAISE}" for line in recorder.breaks(live, recorded)]
    for section, names in live.items():
        known = recorder.entries(recorded.get(section, []))
        for name in recorder.entries(names):
            if name not in known:
                lines.append(f"{section}: {name} is not recorded: {RERECORD}")
    return lines


def test_the_live_surface_is_the_record(recorder, live, recorded):
    found = drift(recorder, live, recorded)
    assert not found, "\n".join(
        [f"the app contract surface differs from contract {APP_CONTRACT}'s record:", *found]
        + ["re-record with: uv run python scripts/record_app_contract.py"]
    )


def test_every_section_has_its_sources(live):
    assert set(live) == set(SURFACE_SOURCES)
    for section, paths in SURFACE_SOURCES.items():
        for path in paths:
            assert (REPO_ROOT / path).is_file(), f"{section}: {path}"


def test_a_nested_manifest_path_is_recorded_with_its_type(recorded):
    assert recorded["manifest"]["panel.launch_token"] == "boolean"
    assert recorded["manifest"]["panel.read_timeout_s"] == "integer|null"
    assert recorded["manifest"]["containers.*.port"] == "integer|null"
    assert recorded["manifest"]["tools.approval[].require.one_of[][]"] == "string"


def test_both_arms_of_a_vendor_entry_are_recorded(recorded):
    for key in ("url", "package", "version", "file", "into", "sha256"):
        assert f"vendor[].{key}" in recorded["manifest"], key


def test_agent_frontmatter_is_recorded_by_alias(recorded):
    assert "disallowedTools" in recorded["agent_frontmatter"]
    assert "maxTurns" in recorded["agent_frontmatter"]
    assert "disallowed_tools" not in recorded["agent_frontmatter"]


@pytest.mark.parametrize(
    "section",
    [
        "sdk",
        "sdk_testing",
        "kinds",
        "agent_frontmatter",
        "page_ready",
        "primitives",
        "panel_messages",
        "headers",
        "environment",
    ],
)
def test_a_removed_name_asks_for_a_new_number(recorder, live, section):
    mutated = copy.deepcopy(live)
    gone = mutated[section].pop(0)
    found = drift(recorder, mutated, live)
    assert found == [f"{section}: {gone} is gone: {RAISE}"]


def test_the_panel_sections_hold_what_a_panel_uses(recorded):
    assert "markPageReady" in recorded["page_ready"]
    assert "osprey-btn" in recorded["primitives"]
    assert "osprey-btn.ghost" in recorded["primitives"]
    assert "osprey-theme-change" in recorded["panel_messages"]


def test_the_process_sections_hold_what_a_process_reads(recorded):
    assert recorded["headers"] == sorted(
        ["X-Osprey-Owner", "X-Osprey-Session", "X-Osprey-Caller", "X-Launch-Token"]
    )
    assert "OSPREY_APP_NAME" in recorded["environment"]
    assert "OSPREY_SESSION_ID" in recorded["environment"]
    assert "OSPREY_TERMINAL_SECRET_*" in recorded["environment"]


def test_the_environment_section_holds_only_promised_osprey_names(live):
    # The launch-token variable and compose interpolation are not handed over.
    assert all(name.startswith("OSPREY_") for name in live["environment"])
    assert "OSPREY_WORKER_IMAGE" not in live["environment"]
    assert "CONFIG_FILE" not in live["environment"]


def test_the_panel_sections_ignore_comments_and_foreign_names(live):
    # Prose in comments and non-OSPREY message types are not contract.
    assert "attestFailure" not in live["page_ready"]
    assert "canopy-host-header-action" not in live["panel_messages"]
    assert all(name.startswith("osprey-") for name in live["primitives"])
    assert all(name.startswith("osprey-") for name in live["panel_messages"])


def test_a_removed_primitive_modifier_asks_for_a_new_number(recorder, live):
    mutated = copy.deepcopy(live)
    mutated["primitives"].remove("osprey-btn.ghost")
    found = drift(recorder, mutated, live)
    assert found == [f"primitives: osprey-btn.ghost is gone: {RAISE}"]


def test_a_removed_manifest_path_asks_for_a_new_number(recorder, live):
    mutated = copy.deepcopy(live)
    del mutated["manifest"]["panel.launch_token"]
    found = drift(recorder, mutated, live)
    assert found == [f"manifest: panel.launch_token is gone: {RAISE}"]


def test_a_changed_manifest_type_asks_for_a_new_number(recorder, live):
    mutated = copy.deepcopy(live)
    mutated["manifest"]["panel.launch_token"] = "boolean|null"
    found = drift(recorder, mutated, live)
    assert found == [f"manifest: panel.launch_token changed type boolean -> boolean|null: {RAISE}"]


def test_an_added_name_asks_only_for_a_re_record(recorder, live):
    mutated = copy.deepcopy(live)
    mutated["sdk"].append("brand_new_helper")
    mutated["manifest"]["panel.brand_new_key"] = "string"
    found = drift(recorder, mutated, live)
    assert sorted(found) == [
        f"manifest: panel.brand_new_key is not recorded: {RERECORD}",
        f"sdk: brand_new_helper is not recorded: {RERECORD}",
    ]
    assert all(RAISE not in line for line in found)


def test_the_script_refuses_to_drop_a_name_from_an_existing_number(recorder, live):
    existing = {"1": copy.deepcopy(live)}
    smaller = copy.deepcopy(live)
    smaller["kinds"].pop()
    with pytest.raises(recorder.RecordRefused, match="Raise APP_CONTRACT"):
        recorder.record(existing, 1, smaller)
    retyped = copy.deepcopy(live)
    retyped["manifest"]["contract"] = "string"
    with pytest.raises(recorder.RecordRefused, match="changed type"):
        recorder.record(existing, 1, retyped)


def test_the_script_main_refuses_and_leaves_the_record_untouched(recorder, live, tmp_path):
    record = tmp_path / "surface.json"
    padded = copy.deepcopy(live)
    padded["sdk"].append("a_name_the_code_no_longer_has")
    text = json.dumps({str(APP_CONTRACT): padded}, indent=2) + "\n"
    record.write_text(text, encoding="utf-8")
    assert recorder.main(["--record", str(record)]) == 1
    assert record.read_text(encoding="utf-8") == text


def test_the_script_adds_names_and_keeps_other_numbers(recorder, live):
    older = {"sdk": ["only_in_zero"]}
    smaller = copy.deepcopy(live)
    smaller["sdk"].pop()
    updated = recorder.record({"0": older, "1": smaller}, 1, live)
    assert list(updated) == ["0", "1"]
    assert updated["0"] == older
    assert updated["1"] == live
    assert recorder.record({}, 2, live) == {"2": live}


def test_a_fresh_manifest_import_loads_neither_the_sdk_nor_the_registry():
    added = modules_added_by_import("osprey.apps.manifest")
    assert "osprey.apps.manifest" in added
    heavy = sorted(
        name
        for name in added
        for root in ("osprey.apps.sdk", "osprey.registry")
        if name == root or name.startswith(root + ".")
    )
    assert not heavy, heavy
