"""fixes.yaml over a copy of the demo's facility tree.

A fix targets a record some imported layer states, so each case plants a small
``imported/csv/`` layer beside the demo's authored records: it restates one
channel's description and adds one channel of its own. Over that tree a ``set``
settles the description, an ``add`` brings in a device, and a ``drop`` removes
the layer's channel; each leaves its entry in the record's provenance, or, for
the dropped channel, in the build's dropped records. The order of fixes.yaml
does not change a byte of the facility file.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES, FacilityDocument
from osprey.facility.combine import FIXES_HEADER
from osprey.facility.render import facility_bytes
from osprey.facility.validate import need, run_stages

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_FACILITY = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/facility"

TUNE_X = "SR:DIAG:TUNE:X"
SPARE = "SR:DIAG:SPARE:01"

#: The imported layer: one restated demo channel and one channel of its own.
CSV_CHANNELS = [
    {"id": TUNE_X, "description": "Horizontal tune"},
    {"id": SPARE, "description": "Spare diagnostic input"},
]

SET = {
    "op": "set",
    "kind": "channel",
    "id": TUNE_X,
    "fields": {"description": "Storage ring horizontal betatron tune"},
    "was": {
        "description": {
            "authored": "Storage ring betatron tune horizontal",
            "csv": "Horizontal tune",
        }
    },
    "why": "The two sources word the tune differently.",
}
ADD = {
    "op": "add",
    "kind": "device",
    "id": "SR/SPAREBPM",
    "record": {"class": "BeamPositionMonitor", "names": ["spare BPM"], "place": "SR"},
    "why": "Installed after the export.",
}
DROP = {"op": "drop", "kind": "channel", "id": SPARE, "why": "Not connected."}


def _build(root: Path, fixes: list[dict[str, Any]]) -> tuple[FacilityDocument, dict[Any, Any]]:
    """Copy the demo tree to ``root``, plant the layer and fixes, and build it."""
    shutil.copytree(DEMO_FACILITY, root)
    layer = root / "imported" / "csv"
    layer.mkdir(parents=True)
    (layer / "channels.yaml").write_text(yaml.safe_dump(CSV_CHANNELS), encoding="utf-8")
    document = {"schema": FIXES_HEADER, "fixes": fixes}
    (root / "fixes.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    report = run_stages(root, project_name="demo", later=LATER_STAGES)
    assert [error.format_message() for error in report.errors] == []
    return need(report.validated.document), need(report.validated.combined).dropped


@pytest.fixture(scope="module")
def fixed(tmp_path_factory: pytest.TempPathFactory) -> tuple[FacilityDocument, dict[Any, Any]]:
    """The demo built with the set, the add and the drop, in that order."""
    return _build(tmp_path_factory.mktemp("fixed") / "facility", [SET, ADD, DROP])


def _record(document: FacilityDocument, plural: str, rid: str) -> dict[str, Any]:
    [record] = [r for r in document[plural] if r["id"] == rid]
    return record


def test_set_settles_the_description(fixed: tuple[FacilityDocument, dict[Any, Any]]) -> None:
    document, _dropped = fixed
    channel = _record(document, "channels", TUNE_X)

    assert channel["description"] == SET["fields"]["description"]
    assert channel["provenance"]["fixes"] == [{"op": "set", "why": SET["why"]}]
    assert [(s["layer"], s["file"]) for s in channel["provenance"]["sources"]] == [
        ("authored", "records/channels.yaml"),
        ("csv", "imported/csv/channels.yaml"),
    ]


def test_add_brings_in_the_device(fixed: tuple[FacilityDocument, dict[Any, Any]]) -> None:
    document, _dropped = fixed
    device = _record(document, "devices", "SR/SPAREBPM")

    assert {k: device[k] for k in ADD["record"]} == ADD["record"]
    assert device["provenance"]["sources"] == []
    assert device["provenance"]["fixes"] == [{"op": "add", "why": ADD["why"]}]


def test_drop_removes_the_channel(fixed: tuple[FacilityDocument, dict[Any, Any]]) -> None:
    document, dropped = fixed

    assert SPARE not in {channel["id"] for channel in document["channels"]}
    assert dropped == {("channel", SPARE): DROP}


def test_a_permuted_fixes_yaml_gives_a_byte_equal_facility_file(
    fixed: tuple[FacilityDocument, dict[Any, Any]], tmp_path: Path
) -> None:
    permuted, _dropped = _build(tmp_path / "facility", [DROP, ADD, SET])

    assert facility_bytes(permuted) == facility_bytes(fixed[0])
