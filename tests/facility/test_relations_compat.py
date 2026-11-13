"""Relations, documents, policy and keyed sets leave every machine-facing view's bytes alone.

A synthetic facility already holding a ``PowerSupply``, a ``Host`` and an
``Unidentified`` device is built twice: as it is, and with relation claims,
``documents.yaml``, ``policy.yaml`` and ``scheme_names``/``citations`` on a
device and a place added. Each build is rendered once per channel-finder
pipeline mode. The simulator view's files, everything under
``data/channel_finder/`` and the limits view's file are byte-identical between
the two renders, except a DuckDB database, which DuckDB does not write
byte-reproducibly even for one document twice and which stamps its rows with
the time of import: it is compared row by row, every table of every schema,
every column but a timestamp; ``facility.json`` and the facts view legitimately differ and
are not compared. The base facility, which has no relation, document or policy
file, also builds and renders byte-identically to the same tree built with the
``relations`` stage removed from the build's later stages, every rendered file
and ``facility.json`` included, and raises no warning: its authored
``Unidentified`` device is no stub. The new-class devices are ordinary devices of the build and
of the facts view, and a facility that declares its own ``PowerSupply`` class
still builds.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import LATER_STAGES, build_facility, build_facility_with_warnings
from osprey.facility.render import facility_bytes, render_facility_outputs
from osprey.facility.validate import run_stages
from osprey.facility.views.facts import FACTS_FILE
from tests.facility._synthetic_trees import deck_tree, write_tree
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

pytestmark = [pytest.mark.slow]

MODES = ("in_context", "hierarchical", "middle_layer")

#: The devices of the classes the core vocabulary gained, by id.
NEW_CLASS_DEVICES = {"SR/PS1": "PowerSupply", "SR/HOST1": "Host", "SR/U1": "Unidentified"}

_CITATION = {"document": "spec-1", "version": "B", "locator": "p. 4"}


def _base_tree() -> dict[str, Any]:
    tree = deck_tree()
    tree["records/devices.yaml"] += [
        {"id": rid, "class": cls} for rid, cls in NEW_CLASS_DEVICES.items()
    ]
    return tree


def _with_relations(tree: dict[str, Any]) -> dict[str, Any]:
    tree = copy.deepcopy(tree)
    [place] = [p for p in tree["records/places.yaml"] if p["id"] == "SR"]
    place["scheme_names"] = [{"name": "Storage Ring", "scheme": "controls"}]
    place["citations"] = [dict(_CITATION)]
    [device] = [d for d in tree["records/devices.yaml"] if d["id"] == "SR/QF"]
    device["scheme_names"] = [
        {"name": "QF", "scheme": "lattice", "basis": "stated", "source": dict(_CITATION)},
        {"name": "SR01C___QF", "scheme": "controls"},
    ]
    device["citations"] = [dict(_CITATION), {"document": "dwg-1"}]
    tree["documents.yaml"] = [
        {"id": "spec-1", "title": "Magnet list", "kind": "spec"},
        {"id": "dwg-1", "title": "Rack layout", "kind": "drawing"},
    ]
    tree["policy.yaml"] = {
        "schema": "osprey.facility.policy/1",
        "relations": {"precedence": {"default": ["spec", "drawing"]}},
    }
    tree["records/relations.yaml"] = [
        {
            "id": "qf-power",
            "type": "powered_by",
            "subject": {"device": "SR/QF"},
            "object": {"device": "SR/PS1", "port": "OUT1"},
            "basis": "stated",
            "source": dict(_CITATION),
        },
        {
            "id": "ps1-in-sr",
            "type": "located_in",
            "subject": {"device": "SR/PS1"},
            "object": {"place": "SR"},
        },
        {
            "id": "host1-in-sr",
            "type": "located_in",
            "subject": {"device": "SR/HOST1"},
            "object": {"place": "SR"},
        },
    ]
    return tree


def _build(root: Path, tree: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    facility_dir = write_tree(root / "facility", tree)
    return build_facility(facility_dir, project_name="compat"), facility_dir


def _render(root: Path, document: dict[str, Any], facility_dir: Path, mode: str) -> Path:
    config = copy.deepcopy(dict(IN_PROCESS_RENDER_CONFIG))
    config["channel_finder"] = {"pipeline_mode": mode}
    render_dir = root / f"render-{mode}"
    render_dir.mkdir(parents=True)
    render_facility_outputs(render_dir, document, config, facility_dir, omitted_reported=set())
    return render_dir


def _database_rows(path: Path) -> bytes:
    """Every table's rows, sorted, in every schema of a DuckDB file, as bytes.

    A timestamp column is left out: it holds the time of import.
    """
    import duckdb

    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = connection.execute(
            "SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1, 2"
        ).fetchall()
        dump = {}
        for schema, table in tables:
            columns = [
                f'"{name}"'
                for name, kind in connection.execute(
                    "SELECT column_name, data_type FROM information_schema.columns"
                    " WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                    [schema, table],
                ).fetchall()
                if not kind.startswith("TIMESTAMP")
            ]
            query = f'SELECT {", ".join(columns)} FROM "{schema}"."{table}"'
            dump[f"{schema}.{table}"] = sorted(map(repr, connection.execute(query).fetchall()))
    finally:
        connection.close()
    return json.dumps(dump, sort_keys=True).encode()


def _every_file(render_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(render_dir).as_posix(): (
            _database_rows(path) if path.suffix == ".duckdb" else path.read_bytes()
        )
        for path in sorted(render_dir.rglob("*"))
        if path.is_file()
    }


def _compared(render_dir: Path) -> dict[str, bytes]:
    data = render_dir / "data"
    paths = [
        *(data / "simulator").rglob("*"),
        *(data / "channel_finder").rglob("*"),
        data / "channel_limits.json",
    ]
    return {
        path.relative_to(render_dir).as_posix(): (
            _database_rows(path) if path.suffix == ".duckdb" else path.read_bytes()
        )
        for path in sorted(paths)
        if path.is_file()
    }


@pytest.fixture(scope="module")
def builds(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Path, dict[str, Any], Path]]:
    out = {}
    for name, tree in (("base", _base_tree()), ("full", _with_relations(_base_tree()))):
        root = tmp_path_factory.mktemp(name)
        document, facility_dir = _build(root, tree)
        out[name] = (root, document, facility_dir)
    return out


def test_the_full_build_carries_what_the_base_build_lacks(builds: dict[str, Any]) -> None:
    _, base, _ = builds["base"]
    _, full, _ = builds["full"]
    assert "relations" not in base and "documents" not in base
    assert [r["id"] for r in full["relations"]] == ["host1-in-sr", "ps1-in-sr", "qf-power"]
    assert [d["id"] for d in full["documents"]] == ["dwg-1", "spec-1"]
    assert "policy" not in full
    [device] = [d for d in full["devices"] if d["id"] == "SR/QF"]
    assert len(device["scheme_names"]) == 2 and len(device["citations"]) == 2
    [place] = [p for p in full["places"] if p["id"] == "SR"]
    assert place["scheme_names"] == [{"name": "Storage Ring", "scheme": "controls"}]
    assert base["schema"] == full["schema"]


@pytest.mark.parametrize("mode", MODES)
def test_machine_views_are_byte_identical(builds: dict[str, Any], mode: str) -> None:
    rendered = {}
    for name in ("base", "full"):
        root, document, facility_dir = builds[name]
        rendered[name] = _compared(_render(root, document, facility_dir, mode))

    base, full = rendered["base"], rendered["full"]
    assert any(path.startswith("data/simulator/") for path in base)
    assert any(path.startswith("data/channel_finder/") for path in base)
    assert "data/channel_limits.json" in base
    assert sorted(base) == sorted(full)
    assert [path for path in sorted(base) if base[path] != full[path]] == []


@pytest.fixture(scope="module")
def without_relations_stage(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any], Path]:
    """The base tree built with the ``relations`` stage left out: the reference."""
    root = tmp_path_factory.mktemp("no-relations-stage")
    facility_dir = write_tree(root / "facility", _base_tree())
    later = [stage for stage in LATER_STAGES if stage[0] != "relations"]
    report = run_stages(facility_dir, project_name="compat", later=later)
    assert report.failed is None, report.errors
    assert report.validated.document is not None
    return root, report.validated.document, facility_dir


def test_no_relation_files_build_byte_identical_and_warn_nothing(
    builds: dict[str, Any], without_relations_stage: tuple[Path, dict[str, Any], Path]
) -> None:
    _, base, facility_dir = builds["base"]
    _, reference, _ = without_relations_stage
    assert facility_bytes(base) == facility_bytes(reference)

    document, warnings = build_facility_with_warnings(facility_dir, project_name="compat")
    assert [warning.line for warning in warnings] == []
    assert facility_bytes(document) == facility_bytes(reference)
    [unidentified] = [d for d in document["devices"] if d["id"] == "SR/U1"]
    assert unidentified["class"] == "Unidentified"


@pytest.mark.parametrize("mode", MODES)
def test_no_relation_files_render_byte_identical(
    builds: dict[str, Any],
    without_relations_stage: tuple[Path, dict[str, Any], Path],
    tmp_path: Path,
    mode: str,
) -> None:
    _, base, facility_dir = builds["base"]
    _, reference, reference_dir = without_relations_stage
    with_stage = _every_file(_render(tmp_path / "with", base, facility_dir, mode))
    without = _every_file(_render(tmp_path / "without", reference, reference_dir, mode))

    assert "facility.json" in with_stage
    assert any(path.startswith("data/channel_finder/") for path in with_stage)
    assert sorted(with_stage) == sorted(without)
    assert [path for path in sorted(with_stage) if with_stage[path] != without[path]] == []


@pytest.mark.parametrize("name", ["base", "full"])
def test_new_class_devices_are_ordinary_devices(
    builds: dict[str, Any], name: str, tmp_path: Path
) -> None:
    _, document, facility_dir = builds[name]
    classes = {d["id"]: d["class"] for d in document["devices"]}
    assert {rid: classes.get(rid) for rid in NEW_CLASS_DEVICES} == NEW_CLASS_DEVICES

    render_dir = _render(tmp_path, document, facility_dir, "in_context")
    facts = json.loads((render_dir / "data" / FACTS_FILE).read_bytes())
    for cls in NEW_CLASS_DEVICES.values():
        assert facts["device_classes"][cls]["count"] == 1, cls
        assert facts["unplaced_devices"][cls] == 1, cls


def test_a_facility_declaring_its_own_power_supply_builds(tmp_path: Path) -> None:
    tree = _base_tree()
    tree["classes.yaml"] = [
        {"class": "PowerSupply", "parent": "Magnet", "description": "A magnet supply."}
    ]
    document, _ = _build(tmp_path, tree)
    [ps] = [d for d in document["devices"] if d["id"] == "SR/PS1"]
    assert ps["class"] == "PowerSupply"
    assert [c["class"] for c in document["classes"]] == ["PowerSupply"]
