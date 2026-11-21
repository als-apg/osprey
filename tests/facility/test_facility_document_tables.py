"""The vocabulary tables the facility document carries at ``/4``.

``signals`` is the effective signal-role table: the vocabulary's roles and the
facility's own, one row per role, sorted by name, each with only the fields its
source states. ``relation_types`` is the vocabulary's relation-type rows as they
stand, sorted by name. Both are in ``build_facility()``'s document, not only in
the rendered file. The facts view and the channel-finder role descriptions read
``signals`` as it stands and give what the ``/3`` build gave
(``golden/demo_role_views.json``, captured from that build).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.build import build_facility
from osprey.facility.records import FacilityRecords
from osprey.facility.validate import FACILITY_HEADER, vocabulary
from tests.facility._synthetic_trees import plain_tree, write_tree

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

GOLDEN = Path(__file__).resolve().parent / "golden" / "demo_role_views.json"

DECLARED = {
    "name": "fixture_role",
    "aliases": ["fixture signal"],
    "description": "A signal the fixture facility adds beside the vocabulary.",
}

#: The fields a relation-type row carries.
RELATION_TYPE_KEYS = {"name", "description", "subject", "object", "symmetric", "single_per"}


def _built(tmp_path: Path, signals: list[dict[str, Any]] | None) -> dict[str, Any]:
    tree = plain_tree()
    if signals is not None:
        tree["signals.yaml"] = signals
        for channel in tree["records/channels.yaml"]:
            if channel["id"] == "Q1:RB":
                channel["signal"] = signals[0]["name"]
    return build_facility(write_tree(tmp_path / "facility", tree), project_name="p")


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


def _without_configuration_digests(facts: dict[str, Any]) -> dict[str, Any]:
    """``facts`` with each measurement model's configuration digest taken out.

    A pyAML configuration carries numbers pyAT computes, such as ``alphac``,
    whose last digits differ between platforms, so its digest cannot be frozen.
    """
    for model in facts["measurement_models"].values():
        model.pop("sha256")
    return facts


def test_the_demo_role_views_are_those_of_the_base_build(built_demo: BuiltProject) -> None:
    from osprey.facility.views.channel_finder import _signal_descriptions

    reference = json.loads(GOLDEN.read_text(encoding="utf-8"))
    facts = json.loads((built_demo.build_dir / "data" / "facility_facts.json").read_bytes())
    descriptions = _signal_descriptions(FacilityRecords(built_demo.facility))

    for model in facts["measurement_models"].values():
        written = (built_demo.build_dir / model["path"]).read_bytes()
        assert model["sha256"] == hashlib.sha256(written).hexdigest()
    assert _without_configuration_digests(facts) == _without_configuration_digests(
        reference["facts"]
    )
    assert descriptions == reference["channel_finder_signal_descriptions"]
