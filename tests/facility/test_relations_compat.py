"""Relations, documents, policy and keyed sets change machine-facing views only by projected places.

A synthetic facility already holding a ``PowerSupply``, a ``Host`` and an
``Unidentified`` device is built three times: as it is; with relation claims,
``documents.yaml``, ``policy.yaml`` and ``scheme_names``/``citations`` on a
device and a place added; and as it is with the places the added
``located_in`` claims project (``SR`` for the supply and the host) authored on
the two devices instead. Each build is rendered once per channel-finder
pipeline mode. The simulator view's files, everything under
``data/channel_finder/`` and the limits view's file are byte-identical between
the build with relations and the build with the projected places authored, so
the relations change those views only by the places they project, except a
DuckDB database, which DuckDB does not write
byte-reproducibly even for one document twice and which stamps its rows with
the time of import: it is compared row by row, every table of every schema,
every column but a timestamp; ``facility.json`` and the facts view legitimately differ and
are not compared. The base facility, which has no relation, document or policy
file, also builds and renders byte-identically to the same tree built with the
``relations`` stage removed from the build's later stages, every rendered file
and ``facility.json`` included, and raises no warning: its authored
``Unidentified`` device is no stub. The new-class devices are ordinary devices of the build and
of the facts view, placed only where a claim projects a place, and a facility that declares its own
``PowerSupply`` class still builds.

The second half of the module runs the same questions on the packaged example facility, built as
shipped, as ``unrelated`` (shipped without ``documents.yaml`` and ``records/relations.yaml``, each
device a claim projects a place for given that place as authored) and as ``pre`` (the example
without the records Appendix A adds). Every render is configured so that the pyAML, Bluesky and
benchmark views are written, and runs in every channel-finder mode. Shipped against unrelated:
everything under ``data/`` except the graph view is identical, the facts included. Shipped against
pre: the simulator, limits, pyAML, Bluesky, benchmark, in-context and hierarchical files are
identical; the middle-layer index differs by the ``-`` and ``HALL`` systems and the ``VS03``
family only (its DuckDB database is derived from the same index and is not compared); the facts
only gain what Appendix A adds. Each view of ``views.VIEWS`` is compared or listed in
``EXCLUDED_VIEWS`` with its reason; ``facility.json`` is excluded because it carries the claims
and documents themselves, and the graph view because it writes them out as triples.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.bluesky_bridge_connection import LANE_KEYS
from osprey.facility.build import LATER_STAGES, build_facility, build_facility_with_warnings
from osprey.facility.render import facility_bytes, render_facility_outputs
from osprey.facility.validate import run_stages
from osprey.facility.views import VIEWS
from osprey.facility.views.facts import FACTS_FILE
from tests._preset_data import packaged_facility_dir
from tests.facility._appendix_a import example_without_appendix_a
from tests.facility._synthetic_trees import deck_tree, write_tree
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

pytestmark = [pytest.mark.slow]

MODES = ("in_context", "hierarchical", "middle_layer")

#: The devices of the classes the core vocabulary gained, by id.
NEW_CLASS_DEVICES = {"SR/PS1": "PowerSupply", "SR/HOST1": "Host", "SR/U1": "Unidentified"}

#: The places the ``located_in`` claims of :func:`_with_relations` project, by device id.
PROJECTED = {"SR/PS1": "SR", "SR/HOST1": "SR"}

#: The benchmark query file a render names so that the benchmarks view is written.
BENCHMARK_DATASET = "data/benchmarks/queries.json"

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


def _with_projected_places_authored(tree: dict[str, Any]) -> dict[str, Any]:
    tree = copy.deepcopy(tree)
    for device in tree["records/devices.yaml"]:
        if device["id"] in PROJECTED:
            device["place"] = PROJECTED[device["id"]]
    return tree


def _build(root: Path, tree: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    facility_dir = write_tree(root / "facility", tree)
    return build_facility(facility_dir, project_name="compat"), facility_dir


def _render(
    root: Path,
    document: dict[str, Any],
    facility_dir: Path,
    mode: str,
    *,
    all_views: bool = False,
) -> Path:
    config = copy.deepcopy(dict(IN_PROCESS_RENDER_CONFIG))
    config["channel_finder"] = {"pipeline_mode": mode}
    if all_views:
        config["channel_finder"]["benchmark"] = {"dataset_path": BENCHMARK_DATASET}
        config["services"] = {LANE_KEYS[0]: {}}
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
    for name, tree in (
        ("base", _base_tree()),
        ("full", _with_relations(_base_tree())),
        ("placed", _with_projected_places_authored(_base_tree())),
    ):
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


def test_the_located_in_claims_project_their_places(builds: dict[str, Any]) -> None:
    for name, expected in (("base", {}), ("full", PROJECTED)):
        _, document, _ = builds[name]
        devices = {d["id"]: d for d in document["devices"]}
        placed = {
            rid: (devices[rid].get("place"), devices[rid]["provenance"].get("place_from"))
            for rid in NEW_CLASS_DEVICES
            if "place" in devices[rid]
        }
        assert placed == {rid: (place, "relation") for rid, place in expected.items()}, name


@pytest.mark.parametrize("mode", MODES)
def test_machine_views_differ_only_by_projected_places(builds: dict[str, Any], mode: str) -> None:
    rendered = {}
    for name in ("full", "placed"):
        root, document, facility_dir = builds[name]
        rendered[name] = _compared(_render(root, document, facility_dir, mode))

    placed, full = rendered["placed"], rendered["full"]
    assert any(path.startswith("data/simulator/") for path in placed)
    assert any(path.startswith("data/channel_finder/") for path in placed)
    assert "data/channel_limits.json" in placed
    assert sorted(placed) == sorted(full)
    assert [path for path in sorted(placed) if placed[path] != full[path]] == []


@pytest.fixture(scope="module")
def without_relations_stage(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any], Path]:
    """The base tree built with the ``relations`` stage left out: the reference."""
    root = tmp_path_factory.mktemp("no-relations-stage")
    facility_dir = write_tree(root / "facility", _base_tree())
    later = [stage for stage in LATER_STAGES if stage[0] != "relations"]
    # A renamed stage would leave nothing filtered and make the comparison vacuous.
    assert [stage[0] for stage in LATER_STAGES if stage not in later] == ["relations"]
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
    for rid, cls in NEW_CLASS_DEVICES.items():
        assert facts["device_classes"][cls]["count"] == 1, cls
        projected = name == "full" and rid in PROJECTED
        assert facts["unplaced_devices"].get(cls, 0) == (0 if projected else 1), cls


def test_a_facility_declaring_its_own_power_supply_builds(tmp_path: Path) -> None:
    tree = _base_tree()
    tree["classes.yaml"] = [
        {"class": "PowerSupply", "parent": "Magnet", "description": "A magnet supply."}
    ]
    document, _ = _build(tmp_path, tree)
    [ps] = [d for d in document["devices"] if d["id"] == "SR/PS1"]
    assert ps["class"] == "PowerSupply"
    assert [c["class"] for c in document["classes"]] == ["PowerSupply"]


# ---------------------------------------------------------------------------
# The packaged example facility
# ---------------------------------------------------------------------------

#: Every view of ``views.VIEWS`` this module does not compare, with the reason.
EXCLUDED_VIEWS = {
    "graph": "it writes the relation claims out as triples, so claims change it by design",
}

#: The views compared, each written by a render configured as ``_render(all_views=True)``.
COMPARED_VIEWS = frozenset(
    {
        "simulator",
        "limits",
        "pyaml",
        "facts",
        "bluesky",
        "in_context",
        "hierarchical",
        "middle_layer",
        "benchmarks",
    }
)

#: Files directly under ``data/`` that views with path ``.`` write.
_ROOT_FILES = {
    "limits": ("channel_limits.json",),
    "facts": ("facility_facts.json", "facility_facts.md"),
    "bluesky": ("bluesky_devices.yml",),
}

_MIDDLE_LAYER = "data/channel_finder/middle_layer"


def test_every_view_is_compared_or_excluded_with_a_reason() -> None:
    names = [view.name for view in VIEWS]
    assert sorted(names) == sorted(COMPARED_VIEWS | set(EXCLUDED_VIEWS))
    assert not COMPARED_VIEWS & set(EXCLUDED_VIEWS)
    assert all(EXCLUDED_VIEWS.values())


def _view_files(files: dict[str, bytes]) -> dict[str, bytes]:
    """The files of a render under ``data/``, less the excluded views' directories."""
    excluded = tuple(
        f"data/{view.path}/" for view in VIEWS if view.name in EXCLUDED_VIEWS and view.path != "."
    )
    return {
        path: content
        for path, content in files.items()
        if path.startswith("data/") and not path.startswith(excluded)
    }


def _unrelated(source: Path, document: dict[str, Any], root: Path) -> Path:
    """The shipped example minus its documents, policy and claims, projected places authored."""
    shutil.copytree(source, root)
    (root / "documents.yaml").unlink()
    (root / "policy.yaml").unlink()
    (root / "records" / "relations.yaml").unlink()
    projected = {
        d["id"]: d["place"]
        for d in document["devices"]
        if d.get("provenance", {}).get("place_from") == "relation"
    }
    assert projected, "the example projects no place: the comparison would be vacuous"
    path = root / "records" / "devices.yaml"
    devices = yaml.safe_load(path.read_text(encoding="utf-8"))
    for device in devices:
        if device["id"] in projected:
            device["place"] = projected[device["id"]]
    path.write_text(yaml.safe_dump(devices, sort_keys=False), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def example_renders(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, str], dict[str, bytes]]:
    """Every file of every render of the shipped, unrelated and pre example, by variant and mode."""
    root = tmp_path_factory.mktemp("example")
    shipped_dir = root / "shipped" / "facility"
    shutil.copytree(packaged_facility_dir(), shipped_dir)
    shipped = build_facility(shipped_dir, project_name="compat")
    assert shipped["relations"] and shipped["documents"]
    unrelated_dir = _unrelated(shipped_dir, shipped, root / "unrelated" / "facility")
    pre_dir = example_without_appendix_a(root / "pre" / "facility")
    variants = {}
    for name, facility_dir in (
        ("shipped", shipped_dir),
        ("unrelated", unrelated_dir),
        ("pre", pre_dir),
    ):
        variants[name] = (build_facility(facility_dir, project_name="compat"), facility_dir)
    assert "relations" not in variants["unrelated"][0]
    assert "relations" not in variants["pre"][0]

    renders = {}
    for name, (document, facility_dir) in variants.items():
        for mode in MODES:
            render_dir = _render(root / name / mode, document, facility_dir, mode, all_views=True)
            renders[name, mode] = _every_file(render_dir)
    return renders


def _differing(left: dict[str, bytes], right: dict[str, bytes]) -> list[str]:
    return [p for p in sorted(set(left) | set(right)) if left.get(p) != right.get(p)]


@pytest.mark.parametrize("mode", MODES)
def test_every_configured_view_is_written(
    example_renders: dict[Any, dict[str, bytes]], mode: str
) -> None:
    files = _view_files(example_renders["shipped", mode])
    for prefix in ("simulator", "pyaml", "benchmarks", "channel_finder"):
        assert any(path.startswith(f"data/{prefix}/") for path in files), prefix
    for names in _ROOT_FILES.values():
        for name in names:
            assert f"data/{name}" in files, name


@pytest.mark.parametrize("mode", MODES)
def test_example_claims_change_no_machine_view(
    example_renders: dict[Any, dict[str, bytes]], mode: str
) -> None:
    shipped = _view_files(example_renders["shipped", mode])
    unrelated = _view_files(example_renders["unrelated", mode])
    assert _differing(shipped, unrelated) == []


@pytest.mark.parametrize("mode", MODES)
def test_example_additions_leave_the_other_views_unmoved(
    example_renders: dict[Any, dict[str, bytes]], mode: str
) -> None:
    shipped = _view_files(example_renders["shipped", mode])
    pre = _view_files(example_renders["pre", mode])
    differing = _differing(shipped, pre)
    # The devices the example adds hold no channel, so no channel-finder index moves.
    assert set(differing) == {f"data/{name}" for name in _ROOT_FILES["facts"]}, differing


def test_middle_layer_gains_nothing_from_the_example_additions(
    example_renders: dict[Any, dict[str, bytes]],
) -> None:
    # The racks, supplies and hosts hold no channel, and SR/VS03 is a section.
    shipped = json.loads(example_renders["shipped", "middle_layer"][f"{_MIDDLE_LAYER}.json"])
    pre = json.loads(example_renders["pre", "middle_layer"][f"{_MIDDLE_LAYER}.json"])
    assert shipped == pre


def _additions(pre: Any, shipped: Any, path: str = "") -> dict[str, Any]:
    """What *shipped* adds to *pre*, by path; anything removed or changed otherwise fails."""
    if isinstance(pre, dict) and isinstance(shipped, dict):
        assert set(pre) <= set(shipped), f"{path}: removed {sorted(set(pre) - set(shipped))}"
        added = {}
        for key, value in shipped.items():
            if key not in pre:
                added[f"{path}/{key}"] = value
            else:
                added.update(_additions(pre[key], value, f"{path}/{key}"))
        return added
    if isinstance(pre, list) and isinstance(shipped, list):
        missing = [item for item in pre if item not in shipped]
        assert not missing, f"{path}: removed {missing}"
        grown = [item for item in shipped if item not in pre]
        return {path: grown} if grown else {}
    if isinstance(pre, int) and isinstance(shipped, int) and not isinstance(pre, bool):
        assert shipped >= pre, f"{path}: {pre} fell to {shipped}"
        return {path: shipped - pre} if shipped > pre else {}
    assert pre == shipped, f"{path}: {pre!r} became {shipped!r}"
    return {}


def test_facts_gain_only_the_example_additions(
    example_renders: dict[Any, dict[str, bytes]],
) -> None:
    def facts(variant: str) -> dict[str, Any]:
        return json.loads(example_renders[variant, "in_context"][f"data/{FACTS_FILE}"])

    added = _additions(facts("pre"), facts("shipped"))

    assert {key.split("/")[1] for key in added} <= {
        "device_classes",
        "places",
        "place_levels",
        "unplaced_devices",
    }
    new_classes = {
        "Circuit",
        "ControlApplication",
        "ElectronicsChassis",
        "Host",
        "NetworkSwitch",
        "PowerSupply",
        "TimingChannel",
        "UtilityLoop",
    }
    gained = {key.split("/")[2] for key in added if key.startswith("/device_classes/")}
    assert new_classes <= gained
    assert added["/device_classes/Quadrupole/count"] == 1
    assert {"HALL", "SR/SECT3/G2"} <= {place["id"] for place in added["/places"]}
    assert "/place_levels" in added
    # The power supply is placed by the rack its accepted claim names.
    assert new_classes - {"PowerSupply"} <= {
        k.split("/")[2] for k in added if k.startswith("/unplaced_devices/")
    }
    # The VS03 group is the only group the example adds.
    groups = {key: value for key, value in added.items() if key.endswith("/groups")}
    assert groups
    assert all(value == ["SR/VS03"] for value in groups.values())
