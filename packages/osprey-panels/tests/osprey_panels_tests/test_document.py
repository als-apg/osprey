"""The engine's facility-document reader: the header it reads, the tables it checks."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from osprey_panels import document as document_module
from osprey_panels.document import (
    END_KINDS,
    FACILITY_FILE,
    READS,
    FacilityDocument,
    FacilityDocumentError,
    end_identity,
)

HEADER = "osprey.facility.facility/4"
REBUILD = "-- rebuild with osprey build"


def _document(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": HEADER,
        "signals": [{"name": "current", "title": "Current"}],
        "relation_types": [{"name": "located_in"}],
        "places": [{"id": "sr"}],
        "devices": [{"id": "q1", "class": "quadrupole"}],
        "channels": [{"id": "q1:I"}],
        "groups": [{"id": "quads", "kind": "family"}],
        "models": [{"name": "lattice"}],
    }
    for key, value in changes.items():
        if value is _DROP:
            document.pop(key)
        else:
            document[key] = value
    return document


_DROP = object()


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_contract_constants() -> None:
    assert READS == frozenset({HEADER})
    assert FACILITY_FILE == "facility.json"
    assert END_KINDS == ("device", "place", "group")
    assert issubclass(FacilityDocumentError, ValueError)


def test_stdlib_only_and_no_description_import() -> None:
    tree = ast.parse(Path(document_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    for module in modules:
        root = module.split(".")[0]
        if root in ("osprey", "osprey_panels"):
            assert module.startswith("osprey_panels"), module
        else:
            assert root in {"__future__", "json", "collections", "pathlib", "types", "typing"}, (
                module
            )


def test_reads_a_current_document() -> None:
    read = FacilityDocument(_document())
    assert read.document["schema"] == HEADER
    assert [row["name"] for row in read.signal_rows()] == ["current"]
    assert [row["name"] for row in read.relation_types()] == ["located_in"]
    assert [row["id"] for row in read.of_kind("device")] == ["q1"]
    assert [row["name"] for row in read.of_kind("model")] == ["lattice"]


def test_document_is_read_only() -> None:
    read = FacilityDocument(_document())
    with pytest.raises(TypeError):
        read.document["schema"] = "x"  # type: ignore[index]


@pytest.mark.parametrize(
    ("header", "found"),
    [
        ("osprey.facility.facility/3", "osprey.facility.facility/3"),
        ("osprey.facility.facility/5", "osprey.facility.facility/5"),
        ("osprey.facility.facility/4 ", "'osprey.facility.facility/4 '"),
    ],
)
def test_other_headers_are_refused(header: str, found: str) -> None:
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument(_document(schema=header), source="x/facility.json")
    message = str(caught.value)
    assert message.startswith(f"x/facility.json is {found}; ")
    assert f"this reader reads {HEADER} {REBUILD}" in message


def test_header_less_document_is_refused() -> None:
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument(_document(schema=_DROP), source="f.json")
    message = str(caught.value)
    assert message.startswith("f.json is ")
    assert message.endswith(f"this reader reads {HEADER} {REBUILD}")


def test_non_string_header_is_refused() -> None:
    with pytest.raises(FacilityDocumentError, match="rebuild with osprey build"):
        FacilityDocument(_document(schema=4))


def test_not_json_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / FACILITY_FILE, "{not json")
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument.from_file(path)
    message = str(caught.value)
    assert message.startswith(str(path))
    assert "not JSON" in message
    assert message.endswith(REBUILD)


@pytest.mark.parametrize("text", ["[]", '"facility"', "4", "null"])
def test_non_object_is_refused(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path / FACILITY_FILE, text)
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument.from_file(path)
    message = str(caught.value)
    assert "not a JSON object" in message
    assert message.endswith(REBUILD)


def test_unreadable_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / FACILITY_FILE
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(FacilityDocumentError, match="rebuild with osprey build"):
        FacilityDocument.from_file(path)


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert FacilityDocument.from_file(tmp_path / FACILITY_FILE) is None


def test_from_file_reads_and_names_the_path(tmp_path: Path) -> None:
    path = _write(tmp_path / FACILITY_FILE, json.dumps(_document()))
    read = FacilityDocument.from_file(path)
    assert read is not None
    assert [row["id"] for row in read.of_kind("place")] == ["sr"]
    bad = _write(tmp_path / "old.json", json.dumps(_document(schema="osprey.facility.facility/3")))
    with pytest.raises(FacilityDocumentError, match=f"^{bad} is osprey.facility.facility/3"):
        FacilityDocument.from_file(bad)


def test_from_file_runs_the_mapping_checks(tmp_path: Path) -> None:
    path = _write(tmp_path / FACILITY_FILE, json.dumps(_document(signals=_DROP)))
    with pytest.raises(FacilityDocumentError, match="signals"):
        FacilityDocument.from_file(path)


@pytest.mark.parametrize("table", ["signals", "relation_types"])
def test_required_table_missing_is_refused(table: str) -> None:
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument(_document(**{table: _DROP}), source="f.json")
    message = str(caught.value)
    assert message.startswith("f.json ")
    assert table in message
    assert message.endswith(REBUILD)


@pytest.mark.parametrize(
    "table", ["signals", "relation_types", "places", "devices", "channels", "groups"]
)
@pytest.mark.parametrize("value", [{}, "rows", 3, None])
def test_non_list_table_is_refused(table: str, value: Any) -> None:
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument(_document(**{table: value}), source="f.json")
    assert table in str(caught.value)
    assert str(caught.value).endswith(REBUILD)


@pytest.mark.parametrize(
    "table", ["signals", "relation_types", "places", "devices", "channels", "groups"]
)
@pytest.mark.parametrize("row", ["q1", 3, None, ["id", "q1"]])
def test_non_mapping_row_is_refused(table: str, row: Any) -> None:
    rows = list(_document()[table]) + [row]
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument(_document(**{table: rows}), source="f.json")
    assert table in str(caught.value)
    assert str(caught.value).endswith(REBUILD)


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("signals", "name"),
        ("relation_types", "name"),
        ("places", "id"),
        ("devices", "id"),
        ("channels", "id"),
        ("groups", "id"),
    ],
)
@pytest.mark.parametrize("value", [_DROP, None, 7])
def test_row_without_string_key_is_refused(table: str, key: str, value: Any) -> None:
    row = {"other": "x"} if value is _DROP else {key: value}
    rows = list(_document()[table]) + [row]
    with pytest.raises(FacilityDocumentError) as caught:
        FacilityDocument(_document(**{table: rows}), source="f.json")
    message = str(caught.value)
    assert table in message
    assert key in message


@pytest.mark.parametrize("table", ["places", "devices", "channels", "groups", "models"])
def test_absent_record_table_reads_as_empty(table: str) -> None:
    read = FacilityDocument(_document(**{table: _DROP}))
    kind = {"places": "place", "devices": "device", "channels": "channel"}.get(
        table, table.rstrip("s")
    )
    assert read.of_kind(kind) == ()


def test_unknown_keys_are_ignored() -> None:
    document = _document(
        future_table=[1, 2, "not a row"],
        provenance={"anything": True},
        devices=[{"id": "q1", "class": "quadrupole", "new_field": [1, {"x": 2}]}],
        signals=[{"name": "current", "unheard_of": 1}],
    )
    read = FacilityDocument(document)
    (device,) = read.of_kind("device")
    assert device["new_field"] == [1, {"x": 2}]
    assert device.get("missing") is None
    assert read.signal_rows()[0]["unheard_of"] == 1
    assert read.document["future_table"] == [1, 2, "not a row"]


def test_duplicate_ids_resolve_first_wins_in_document_order() -> None:
    read = FacilityDocument(
        _document(
            devices=[
                {"id": "b", "n": 1},
                {"id": "a", "n": 2},
                {"id": "b", "n": 3},
                {"id": "c", "n": 4},
                {"id": "a", "n": 5},
            ]
        )
    )
    assert [(row["id"], row["n"]) for row in read.of_kind("device")] == [
        ("b", 1),
        ("a", 2),
        ("c", 4),
    ]


def test_models_key_by_name() -> None:
    read = FacilityDocument(
        _document(
            models=[
                {"name": "ring", "n": 1},
                {"id": "ring", "name": "other", "n": 2},
                {"name": "ring", "n": 3},
            ]
        )
    )
    assert [(row["name"], row["n"]) for row in read.of_kind("model")] == [
        ("ring", 1),
        ("other", 2),
    ]


def test_of_kind_rows_are_read_only() -> None:
    read = FacilityDocument(_document())
    (device,) = read.of_kind("device")
    with pytest.raises(TypeError):
        device["id"] = "z"  # type: ignore[index]


def test_unknown_kind_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        FacilityDocument(_document()).of_kind("signal")


def test_mapping_constructor_does_not_copy_rows_in() -> None:
    document = _document()
    read = FacilityDocument(document)
    assert read.document == document


@pytest.mark.parametrize(
    ("end", "identity"),
    [
        ({"device": "q1"}, ("device", "q1")),
        ({"place": "sr", "port": "a"}, ("place", "sr")),
        ({"group": 3}, ("group", "3")),
        ({"place": "sr", "device": "q1"}, ("device", "q1")),
        ({"port": "a"}, None),
        ({"device": None}, None),
        ({}, None),
    ],
)
def test_end_identity(end: dict[str, Any], identity: tuple[str, str] | None) -> None:
    assert end_identity(end) == identity
