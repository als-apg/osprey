"""The relations import end to end: Appendix A moved into tables builds as authored.

The Appendix A fixture is split: the records it reuses (``SR``, the sectors,
the five reused devices) stay authored, and everything
Appendix A adds -- its places, its devices (as stubs with class columns), its
group, the names of ``SR/QF05-PS`` and the eighteen relation claims -- moves
into the relations, places, members and names tables of one ``cables`` layer.
The facility then validates with the same claims and the same standings as the
authored fixture, and with the warnings the import leaves (``place-unplaced``
for the uncited top-level ``HALL``; ``relation-stub`` only for stubs of class
``Unidentified``).

Two layers that share places but no unknown device end and no group write the
same files whichever is imported first. That is all order independence covers:
by p5-r2 a device stub and by p5-r3 a group are created only when no other
source has the record, so a layer naming the same unknown device or group as
an earlier import writes something else once that import is in place.

A 10,000-row table imports in under 10 s and the facility then validates in
under 30 s.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES
from osprey.facility.layers.relations import import_relations
from osprey.facility.validate import StageReport, run_stages
from tests.facility._appendix_a import POLICY, appendix_a_tree, build_appendix_a
from tests.facility._synthetic_trees import write_tree

LAYER = "cables"

REUSED_PLACES = {"SR", "SR/SECT3", "SR/SECT4"}
REUSED_DEVICES = {"SR/QF05", "SR/BPM13", "SR/VALVE03", "SR/VALVE04", "SR/ION-PUMP13"}

RELATION_COLUMNS = (
    "type",
    "subject",
    "object",
    "subject_kind",
    "object_kind",
    "subject_port",
    "object_port",
    "object_unit",
    "attrs",
    "basis",
    "rule",
    "subject_class",
    "object_class",
    "document",
    "version",
    "state",
    "locator",
)

PLACES_TABLE = "place,within\nHALL,\nG2,SR/SECT3\nROW3,HALL\nRK07,HALL/ROW3\nRK08,HALL/ROW3\n"

MEMBERS_TABLE = "group,member\nSR/VS03,SR/ION-PUMP13\nSR/VS03,SR/BPM13\nSR/VS03,SR/QF05\n"

NAMES_TABLE = (
    "thing,name,scheme,basis,rule\n"
    "SR/QF05-PS,PS-Q-0301,cable-schedule,,\n"
    "SR/QF05-PS,QF05 supply,drawings,inferred,label-match\n"
)


# -- the split ---------------------------------------------------------------


def _base_tree(policy: bool = False) -> dict[str, Any]:
    """Appendix A with only the reused records authored."""
    tree = appendix_a_tree()
    tree["records/places.yaml"] = [
        p for p in tree["records/places.yaml"] if p["id"] in REUSED_PLACES
    ]
    tree["records/devices.yaml"] = [
        d for d in tree["records/devices.yaml"] if d["id"] in REUSED_DEVICES
    ]
    del tree["records/groups.yaml"]
    del tree["records/relations.yaml"]
    if policy:
        tree["policy.yaml"] = POLICY
    return tree


def _added_classes() -> dict[str, str]:
    return {
        d["id"]: d["class"]
        for d in appendix_a_tree()["records/devices.yaml"]
        if d["id"] not in REUSED_DEVICES
    }


def _end_cells(end: dict[str, Any], side: str, classes: dict[str, str]) -> dict[str, str]:
    (kind, name), *_ = ((k, v) for k, v in end.items() if k in ("device", "place", "group"))
    cells = {side: name}
    if kind != "device":
        cells[f"{side}_kind"] = kind
    if "port" in end:
        cells[f"{side}_port"] = str(end["port"])
    if "unit" in end:
        cells[f"{side}_unit"] = str(end["unit"])
    if kind == "device" and name in classes:
        cells[f"{side}_class"] = classes[name]
    return cells


def _relations_table(classes: dict[str, str]) -> str:
    """The eighteen Appendix A claims as relations rows."""
    lines = [",".join(RELATION_COLUMNS)]
    for claim in appendix_a_tree()["records/relations.yaml"]:
        cells: dict[str, str] = {"type": claim["type"]}
        cells.update(_end_cells(claim["subject"], "subject", classes))
        cells.update(_end_cells(claim["object"], "object", classes))
        if "attributes" in claim:
            cells["attrs"] = ";".join(f"{k}={v}" for k, v in claim["attributes"].items())
        for key in ("basis", "rule"):
            if key in claim:
                cells[key] = claim[key]
        cells.update(claim.get("source", {}))
        lines.append(",".join(cells.get(column, "") for column in RELATION_COLUMNS))
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _import_appendix(
    tmp_path: Path,
    *,
    policy: bool = False,
    classes: dict[str, str] | None = None,
    stub_unknown: bool = False,
) -> Path:
    facility = write_tree(tmp_path / "facility", _base_tree(policy))
    tables = tmp_path / "in"
    import_relations(
        _write(
            tables / "cables.csv",
            _relations_table(_added_classes() if classes is None else classes),
        ),
        facility,
        LAYER,
        names=_write(tables / "names.csv", NAMES_TABLE),
        places=_write(tables / "places.csv", PLACES_TABLE),
        members=_write(tables / "members.csv", MEMBERS_TABLE),
        stub_unknown=stub_unknown,
    )
    return facility


def _build(facility: Path) -> StageReport:
    report = run_stages(facility, project_name="p", later=LATER_STAGES)
    assert report.failed is None, [e.format_message() for e in report.errors]
    return report


def _lines(report: StageReport, kind: str) -> list[str]:
    prefix = f"facility: {kind}:"
    return [w.line for w in report.validated.warnings if w.line.startswith(prefix)]


def _end_key(end: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (k, str(v)) for k, v in end.items() if k in ("device", "place", "group", "port", "unit")
        )
    )


def _claim_key(claim: dict[str, Any]) -> tuple[Any, ...]:
    source = claim.get("source") or {}
    return (
        claim["type"],
        _end_key(claim["subject"]),
        _end_key(claim["object"]),
        source.get("document") if isinstance(source, dict) else None,
    )


def _standings(document: dict[str, Any]) -> dict[tuple[Any, ...], str]:
    return {_claim_key(c): c["standing"] for c in document["relations"]}


def _by_id(document: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in document.get(kind) or []}


# -- the round trip ----------------------------------------------------------


class TestAppendixRoundTrip:
    @pytest.mark.parametrize("policy", [False, True], ids=["no_policy", "located_in_precedence"])
    def test_claims_and_standings_match_the_authored_fixture(
        self, tmp_path: Path, policy: bool
    ) -> None:
        authored = build_appendix_a(
            tmp_path / "authored", "located_in_precedence" if policy else "no_policy"
        )
        report = _build(_import_appendix(tmp_path / "imported", policy=policy))
        document = report.validated.document
        assert len(document["relations"]) == 18
        assert all(str(c["id"]).startswith(f"{LAYER}/") for c in document["relations"])
        assert _standings(document) == _standings(authored)

    def test_rack_claims_stand_as_the_design_says(self, tmp_path: Path) -> None:
        rack_a = (
            "located_in",
            (("device", "SR/QF05-PS"),),
            (("place", "HALL/ROW3/RK07"), ("unit", "12")),
            "RACK-ROW3",
        )
        rack_b = (
            "located_in",
            (("device", "SR/QF05-PS"),),
            (("place", "HALL/ROW3/RK08"),),
            "CS-RING",
        )
        plain = _standings(_build(_import_appendix(tmp_path / "a")).validated.document)
        ranked = _standings(
            _build(_import_appendix(tmp_path / "b", policy=True)).validated.document
        )
        assert (plain[rack_a], plain[rack_b]) == ("contested", "contested")
        assert (ranked[rack_a], ranked[rack_b]) == ("accepted", "superseded")

    def test_records_match_the_authored_fixture(self, tmp_path: Path) -> None:
        authored = build_appendix_a(tmp_path / "authored")
        document = _build(_import_appendix(tmp_path / "imported")).validated.document
        assert set(_by_id(document, "places")) == set(_by_id(authored, "places"))
        devices, expected = _by_id(document, "devices"), _by_id(authored, "devices")
        assert set(devices) == set(expected)
        assert {i: d["class"] for i, d in devices.items()} == {
            i: d["class"] for i, d in expected.items()
        }
        names = {(n["name"], n["scheme"]) for n in devices["SR/QF05-PS"]["scheme_names"]}
        assert names == {("PS-Q-0301", "cable-schedule"), ("QF05 supply", "drawings")}
        groups = _by_id(document, "groups")
        assert sorted(groups["SR/VS03"]["members"]) == sorted(
            _by_id(authored, "groups")["SR/VS03"]["members"]
        )

    def test_the_layer_holds_what_moved(self, tmp_path: Path) -> None:
        layer = _import_appendix(tmp_path) / "imported" / LAYER
        load = lambda name: yaml.safe_load((layer / name).read_text(encoding="utf-8"))  # noqa: E731
        assert {d["id"] for d in load("devices.yaml")} == set(_added_classes())
        assert {p["id"] for p in load("places.yaml")} == {
            "SR/SECT3/G2",
            "HALL",
            "HALL/ROW3",
            "HALL/ROW3/RK07",
            "HALL/ROW3/RK08",
        }
        assert load("groups.yaml") == [
            {"id": "SR/VS03", "members": ["SR/BPM13", "SR/ION-PUMP13", "SR/QF05"]}
        ]
        assert len(load("relations.yaml")) == 18

    def test_warnings(self, tmp_path: Path) -> None:
        report = _build(_import_appendix(tmp_path))
        assert _lines(report, "place-unplaced") == [
            f"facility: place-unplaced: place HALL — relations layer {LAYER} creates it "
            "top-level with no parent and no citation; fix: give it a parent or a "
            "citation in the layer's places table"
        ]
        assert _lines(report, "place-parents-disagree") == []
        assert _lines(report, "relation-stub") == []

    def test_a_class_less_stub_is_reported(self, tmp_path: Path) -> None:
        classes = {k: v for k, v in _added_classes().items() if k != "CTRL/SW-3A"}
        report = _build(_import_appendix(tmp_path, classes=classes, stub_unknown=True))
        assert _by_id(report.validated.document, "devices")["CTRL/SW-3A"]["class"] == "Unidentified"
        assert _lines(report, "relation-stub") == [
            f"facility: relation-stub: layer {LAYER} — 1 device of class Unidentified "
            "stands only for claim ends; fix: import the devices the claims name, or "
            "name known devices"
        ]


# -- two layers, both orders -------------------------------------------------


# The shared places carry no citation: one place cited differently by two
# layers is a layer-conflict stop, not an order question.
_TWO_LAYERS = {
    "racks": (
        "type,subject,object,object_kind,object_unit,document\n"
        "located_in,SR/BPM13,HALL/ROW3/RK07,place,22,RACK-ROW3\n"
        "located_in,SR/VALVE03,HALL/ROW3/RK08,place,,RACK-ROW3\n",
        "place,within\nHALL,\nROW3,HALL\nRK07,HALL/ROW3\nRK08,HALL/ROW3\n",
    ),
    "schedule": (
        "type,subject,object,subject_port,object_kind,attrs,document\n"
        "cabled_to,SR/QF05,HALL/ROW3/RK07,T1,place,cable=W-03-0117,CS-RING\n"
        "cabled_to,SR/BPM13,SR/SECT3/G2,A,place,cable=W-03-0240,CS-RING\n",
        "place,within\nHALL,\nROW3,HALL\nRK07,HALL/ROW3\nG2,SR/SECT3\n",
    ),
}


def _layer_files(facility: Path, layer: str) -> dict[str, bytes]:
    root = facility / "imported" / layer
    return {p.name: p.read_bytes() for p in sorted(root.iterdir())}


class TestTwoLayersEitherOrder:
    def test_each_layer_writes_the_same_files_in_either_order(self, tmp_path: Path) -> None:
        tables: dict[str, tuple[Path, Path]] = {
            name: (
                _write(tmp_path / "in" / name / "relations.csv", relations),
                _write(tmp_path / "in" / name / "places.csv", places),
            )
            for name, (relations, places) in _TWO_LAYERS.items()
        }
        facilities = {}
        for order in (("racks", "schedule"), ("schedule", "racks")):
            facility = write_tree(tmp_path / "-".join(order), _base_tree())
            for name in order:
                table, places = tables[name]
                import_relations(table, facility, name, places=places)
            facilities[order] = facility
        first, second = facilities.values()
        for name in _TWO_LAYERS:
            assert _layer_files(first, name) == _layer_files(second, name)
        reports = [_build(f) for f in facilities.values()]
        assert reports[0].validated.document == reports[1].validated.document
        assert _lines(reports[0], "place-parents-disagree") == []
        assert _lines(reports[0], "place-unplaced") == [
            "facility: place-unplaced: place HALL — relations layers racks, schedule "
            "create it top-level with no parent and no citation; fix: give it a parent "
            "or a citation in the layer's places table"
        ]


# -- scale -------------------------------------------------------------------


ROWS = 10_000


class TestScale:
    def test_ten_thousand_rows_import_and_validate_in_time(self, tmp_path: Path) -> None:
        facility = write_tree(tmp_path / "facility", _base_tree())
        lines = ["type,subject,object,subject_class,object_class,document,version"]
        lines.extend(
            f"powered_by,GEN/MAG-{i:05d},GEN/PS-{i // 10:04d},Quadrupole,PowerSupply,PS-LIST,C"
            for i in range(ROWS)
        )
        table = _write(tmp_path / "in" / "generated.csv", "\n".join(lines) + "\n")

        start = time.perf_counter()
        import_relations(table, facility, "generated")
        imported = time.perf_counter() - start

        start = time.perf_counter()
        report = run_stages(facility, project_name="p", later=LATER_STAGES)
        validated = time.perf_counter() - start

        assert report.failed is None, [e.format_message() for e in report.errors]
        assert len(report.validated.document["relations"]) == ROWS
        assert imported < 10, f"import took {imported:.1f} s"
        assert validated < 30, f"validation took {validated:.1f} s"
