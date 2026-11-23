"""The vocabulary tables the facility document carries at ``/4``.

``signals`` is the effective signal-role table: the vocabulary's roles and the
facility's own, one row per role, sorted by name, each with only the fields its
source states. ``relation_types`` is the vocabulary's relation-type rows as they
stand, sorted by name. Both are in ``build_facility()``'s document, not only in
the rendered file.

That the facts view and the channel-finder role descriptions read ``signals``
as the ``/3`` build gave them needs the framework, and stays in the root
``tests/facility/test_facility_document_tables.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_facility.validate import FACILITY_HEADER, vocabulary
from osprey_facility_tests._harness import build

DECLARED = {
    "name": "fixture_role",
    "aliases": ["fixture signal"],
    "description": "A signal the fixture facility adds beside the vocabulary.",
}

#: The fields a relation-type row carries.
RELATION_TYPE_KEYS = {"name", "description", "subject", "object", "symmetric", "single_per"}


def _plain_tree() -> dict[str, Any]:
    """One quadrupole with a setpoint/readback pair and one BPM; no model."""
    return {
        "records/devices.yaml": [
            {"id": "SR/Q1", "class": "Quadrupole"},
            {"id": "SR/BPM1", "class": "BeamPositionMonitor"},
        ],
        "records/channels.yaml": [
            {"id": "Q1:SP", "role": "setpoint", "pair": "Q1:RB", "on": {"device": "SR/Q1"}},
            {"id": "Q1:RB", "on": {"device": "SR/Q1"}},
            {"id": "BPM1:X", "on": {"device": "SR/BPM1"}},
        ],
    }


def _write_tree(root: Path, tree: dict[str, Any]) -> Path:
    for rel, data in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def _built(tmp_path: Path, signals: list[dict[str, Any]] | None) -> dict[str, Any]:
    tree = _plain_tree()
    if signals is not None:
        tree["signals.yaml"] = signals
        for channel in tree["records/channels.yaml"]:
            if channel["id"] == "Q1:RB":
                channel["signal"] = signals[0]["name"]
    return build(_write_tree(tmp_path / "facility", tree), project_name="p")


def test_the_document_carries_the_current_header(tmp_path: Path) -> None:
    assert FACILITY_HEADER == "osprey.facility.facility/4"
    assert _built(tmp_path, None)["schema"] == FACILITY_HEADER


@pytest.mark.parametrize("declared", [[], [DECLARED]], ids=["vocabulary-only", "declared"])
def test_signals_is_the_effective_role_table(tmp_path: Path, declared: list[dict]) -> None:
    signals = _built(tmp_path, declared or None)["signals"]

    expected = sorted([*vocabulary()["signal_roles"], *declared], key=lambda row: row["name"])
    assert signals == expected
    names = [row["name"] for row in signals]
    assert names == sorted(set(names))


def test_a_signal_row_omits_the_fields_its_role_lacks(tmp_path: Path) -> None:
    signals = _built(tmp_path, [{"name": "fixture_role"}])["signals"]

    assert {"name": "fixture_role"} in signals
    assert all(value is not None for row in signals for value in row.values())
    assert not any("unit" in row for row in signals)


def test_relation_types_are_the_vocabulary_s_rows_by_name(tmp_path: Path) -> None:
    rows = _built(tmp_path, None)["relation_types"]

    assert rows == sorted(vocabulary()["relations"], key=lambda row: row["name"])
    assert rows
    for row in rows:
        assert set(row) == RELATION_TYPE_KEYS
        assert isinstance(row["subject"], list) and isinstance(row["object"], list)
        assert set(row["subject"]) | set(row["object"]) <= {"device", "place", "group"}


def test_the_document_s_tables_are_copies_not_the_vocabulary(tmp_path: Path) -> None:
    document = _built(tmp_path, None)
    document["signals"][0]["name"] = "changed"
    document["relation_types"][0]["subject"].append("changed")

    assert all(row["name"] != "changed" for row in vocabulary()["signal_roles"])
    assert all("changed" not in row["subject"] for row in vocabulary()["relations"])
