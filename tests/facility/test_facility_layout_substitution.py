"""Moving facility sources by editing only the layout declaration: the framework side.

A fresh interpreter rebinds the facility tree's layout registry to a layout
where the vocabulary, provenance and simulation sources sit in their own
directories and the benchmark queries sit under ``benchmarks/channel_finder``,
before any consumer is imported. It then imports the framework's consumers of
that layout and reports what they computed, read and wrote on trees in
``tmp_path``. Nothing in the repository moves.

Expected: every framework reader of a moved source follows it; a message that
names a moved source names its new path; what a render writes keeps its own
names; and the deployment's content directories are the same three.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

#: Each moved source and where it sits in the moved layout.
_MOVED: dict[str, str] = {
    "classes": "vocabulary/classes.yaml",
    "signals": "vocabulary/signals.yaml",
    "documents": "provenance/documents.yaml",
    "policy": "provenance/policy.yaml",
    "fixes": "provenance/fixes.yaml",
    "models": "simulation/models.yaml",
    "seeds": "simulation/seeds.yaml",
    "decks": "simulation/decks",
    "measurement": "simulation/measurement",
    "scenarios": "simulation/scenarios",
    "benchmark_queries": "benchmarks/channel_finder",
}

#: The first bytes of a PNG picture, enough for a logbook attachment.
_PNG = "\x89PNG\r\n\x1a\n" + "\0" * 16

_SCENARIO = {
    "description": "A quiet day.",
    "overrides": {"SR:BPM1:X": 0.0},
    "logbook": [
        {
            "entry_id": "E-1",
            "when": {"days_ago": 1, "time": "03:20:00"},
            "author": "A. Author",
            "title": "Quiet",
            "text": "Nothing happened.",
            "attachments": [{"path": "plots/trend.png"}],
        }
    ],
}

#: A facility tree in the moved layout, each moved file also left at its old place.
_FACILITY_TREE: dict[str, Any] = {
    "identity.yaml": {"code": "demo"},
    "records/places.yaml": [{"id": "SR"}],
    "records/channels.yaml": [{"id": "SR:BPM1:X", "on": {"place": "SR"}}],
    "simulation/models.yaml": [{"name": "ring", "engine": "pyat"}],
    "simulation/measurement/ring.yaml": {},
    "simulation/scenarios/quiet.yaml": _SCENARIO,
    "simulation/scenarios/quiet/plots/trend.png": _PNG + "moved",
    "simulation/decks/ring.json": '{"deck": "moved"}\n',
    "benchmarks/channel_finder/in_context_queries.json": '["moved in context"]\n',
    "benchmarks/channel_finder/tree_queries.json": '["moved tree"]\n',
    # The old places, which no reader may take.
    "scenarios/quiet/plots/trend.png": _PNG + "old",
    "decks/ring.json": '{"deck": "old"}\n',
    "benchmarks/cross_paradigm/queries/in_context_queries.json": '["old in context"]\n',
    "benchmarks/cross_paradigm/queries/tree_queries.json": '["old tree"]\n',
}

#: A facility tree in the moved layout whose benchmarks hold no tree query set.
_NO_TREE_QUERIES: dict[str, Any] = {
    "identity.yaml": {"code": "demo"},
    "benchmarks/channel_finder/in_context_queries.json": "[]\n",
    "benchmarks/cross_paradigm/queries/tree_queries.json": "[]\n",
}

#: A profile's data tree whose benchmarks hold more than the query sources.
_DATA_TREE_SHARED: dict[str, Any] = {
    "facility/identity.yaml": {"code": "demo"},
    "facility/knowledge/index.md": "# Notes\n",
    "facility/benchmarks/channel_finder/tree_queries.json": "[]\n",
    "facility/benchmarks/channel_finder/in_context_queries.json": "[]\n",
    "facility/benchmarks/README.md": "Benchmarks.\n",
    "facility/benchmarks/cross_paradigm/notes.md": "Kept.\n",
    "other/keep.txt": "Kept.\n",
}

#: A profile's data tree whose benchmarks hold the query sources and nothing else.
_DATA_TREE_ALONE: dict[str, Any] = {
    "facility/identity.yaml": {"code": "demo"},
    "facility/benchmarks/channel_finder/tree_queries.json": "[]\n",
}

#: A profile's data tree whose facility holds only the query sources.
_DATA_TREE_ONLY_QUERIES: dict[str, Any] = {
    "facility/benchmarks/channel_finder/tree_queries.json": "[]\n",
    "other/keep.txt": "Kept.\n",
}

#: Runs in a fresh interpreter: rebinds the registry, then imports the consumers.
_PROBE = r"""
import json
import sys
from pathlib import Path

consumers_before = sorted(
    name for name in sys.modules
    if (name.startswith(("osprey.", "osprey_facility.")) or name == "osprey")
    and name != "osprey_facility.source_tree"
)
from osprey_facility import source_tree

source_tree._FACILITY_SLOTS = source_tree._with_paths(json.loads(sys.argv[1]))

from osprey.cli import facility_cmd
from osprey.cli.templates.scaffolding import copy_template_data
from osprey.facility import settings
from osprey.facility.views import ViewInputs
from osprey.facility.views import benchmarks
from osprey.facility.views.simulator import DECKS_DIR, SCENARIOS_DIR, write_simulator_view
from osprey.registry.scenario_servers import scenario_server_errors
from osprey.simulation import model_request
from osprey.simulation.engines import injection
from osprey_facility import FACILITY_FILE
from osprey_facility.build import build_facility
from osprey_facility.errors import FacilityBuildError
from osprey_facility.records import FacilityRecords
from osprey_facility.roots import DEPLOYMENT_CONTENT_DIRS

facility, no_tree, out = (Path(arg) for arg in sys.argv[2:5])
data_trees = {name: Path(arg) for name, arg in zip(("shared", "alone", "only"), sys.argv[5:8])}


def files_under(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def dirs_under(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir())


# --- the benchmarks view ---------------------------------------------------------------
config = {"channel_finder": {"benchmark": {"dataset_path": "data/benchmarks/queries.json"}}}
written = {}
for mode in ("in_context", "hierarchical"):
    config["channel_finder"]["pipeline_mode"] = mode
    inputs = ViewInputs(
        records=FacilityRecords({"channels": []}),
        rendered_config=config,
        facility_dir=facility,
        served=[],
    )
    render = out / "bench" / mode
    (path,) = benchmarks.write_benchmarks_view(render / "data", inputs)
    written[mode] = [path.relative_to(render).as_posix(), path.read_text(encoding="utf-8")]
wanted = benchmarks.benchmarks_wanted(inputs)
missing_inputs = ViewInputs(
    records=FacilityRecords({"channels": []}),
    rendered_config=config,
    facility_dir=no_tree,
    served=[],
)
try:
    benchmarks.write_benchmarks_view(out / "bench" / "missing" / "data", missing_inputs)
    refusal = None
except FacilityBuildError as exc:
    refusal = exc.format_message()
unwanted = benchmarks.benchmarks_wanted(
    ViewInputs(
        records=FacilityRecords({"channels": []}),
        rendered_config=config,
        facility_dir=out / "empty-facility",
        served=[],
    )
)

# --- the scaffolding copy of a profile's data tree -------------------------------------
copied = {}
for name, data_root in data_trees.items():
    project = out / "project" / name
    copy_template_data(
        out / "templates", project, "pkg", "bundle", {}, data_root=data_root
    )
    copied[name] = {
        "files": files_under(project / "data"),
        "dirs": dirs_under(project / "data"),
    }

# --- the simulator view ----------------------------------------------------------------
document = build_facility(facility, project_name="demo", extra_checks=())
for model in document.get("models", []):
    if model["name"] == "ring":
        model["deck"] = source_tree.path("decks") + "/ring.json"
view_root = out / "render" / "data" / "simulator"
written_view = write_simulator_view(
    view_root,
    ViewInputs(
        records=FacilityRecords(document),
        rendered_config={"control_system": {"type": "virtual_accelerator"}},
        facility_dir=facility,
        served=[],
    ),
)
copies = {
    path.relative_to(view_root).as_posix(): path.read_bytes().decode("latin-1")
    for path in written_view
    if path.parent != view_root
}

# --- scenario sessions, the corrector step, the injection engine -----------------------
sessions_doc = {
    "channels": [{"id": "SR:DCCT"}],
    "scenarios": [
        {
            "name": "warm",
            "sessions": [
                {
                    "slug": "overnight",
                    "title": "Overnight",
                    "when": {"days_ago": 1, "time": "08:00"},
                    "parts": [
                        {
                            "title": "Current",
                            "turns": [
                                {
                                    "prompt": "How was the current?",
                                    "at_offset_s": 0,
                                    "channels": ["SR:DCCT"],
                                    "steps": [
                                        {
                                            "tool": {"server": "nowhere", "name": "read"},
                                            "input": {},
                                            "result": "ok",
                                        },
                                        {"answer": "Done."},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ],
}
(server_error,) = scenario_server_errors(sessions_doc, {"controls"})

render_root = out / "render"
(render_root / FACILITY_FILE).write_text(json.dumps(document), encoding="utf-8")
try:
    model_request._corrector_delta(render_root, "ring")
    step_refusal = None
except model_request.ModelRequestRefused as exc:
    step_refusal = [exc.message, exc.suggestion]

print(json.dumps({
    "consumers_before": consumers_before,
    "benchmarks": {
        "QUERY_SOURCES": benchmarks.QUERY_SOURCES,
        "names": [benchmarks.query_source_name(m) for m in ("in_context", "graph")],
        "written": written,
        "wanted": list(wanted),
        "unwanted": list(unwanted),
        "refusal": refusal,
    },
    "copied": copied,
    "writes": {verb: [list(pair) for pair in pairs] for verb, pairs in facility_cmd._writes().items()},
    "simulator": {
        "DECKS_DIR": DECKS_DIR,
        "SCENARIOS_DIR": SCENARIOS_DIR,
        "copies": copies,
    },
    "scenario_servers": {
        "files": list(server_error.sources),
        "detail": server_error.detail,
    },
    "step_refusal": step_refusal,
    "injection_sources": list(injection._SOURCES),
    "DEPLOYMENT_CONTENT_DIRS": list(DEPLOYMENT_CONTENT_DIRS),
    "VOCABULARY_SOURCE": settings.VOCABULARY_SOURCE,
}))
"""


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_bytes(data.encode("latin-1"))
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def _run_probe(base: Path, moved: dict[str, str]) -> dict[str, Any]:
    trees = [
        _write(base / name, files)
        for name, files in (
            ("facility", _FACILITY_TREE),
            ("no-tree", _NO_TREE_QUERIES),
        )
    ]
    out = base / "out"
    (out / "templates").mkdir(parents=True)
    data_trees = [
        _write(base / name / "data", files)
        for name, files in (
            ("data-shared", _DATA_TREE_SHARED),
            ("data-alone", _DATA_TREE_ALONE),
            ("data-only", _DATA_TREE_ONLY_QUERIES),
        )
    ]
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(moved), *map(str, [*trees, out, *data_trees])],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """What the framework's consumers computed, read and wrote under the moved layout."""
    return _run_probe(tmp_path_factory.mktemp("substitution"), _MOVED)


def test_the_registry_is_rebound_before_any_consumer_is_imported(probe: dict[str, Any]) -> None:
    assert probe["consumers_before"] == []


# --- the benchmarks view ---------------------------------------------------------------


def test_the_query_sources_directory_follows(probe: dict[str, Any]) -> None:
    assert probe["benchmarks"]["QUERY_SOURCES"] == "benchmarks/channel_finder"
    assert probe["benchmarks"]["names"] == ["in_context_queries.json", "tree_queries.json"]


def test_the_view_reads_each_query_file_from_the_new_directory(probe: dict[str, Any]) -> None:
    assert probe["benchmarks"]["written"] == {
        "in_context": ["data/benchmarks/queries.json", '["moved in context"]\n'],
        "hierarchical": ["data/benchmarks/queries.json", '["moved tree"]\n'],
    }


def test_the_view_is_wanted_by_the_new_directory_alone(probe: dict[str, Any]) -> None:
    assert probe["benchmarks"]["wanted"] == [True, "channel_finder.benchmark.dataset_path"]
    assert probe["benchmarks"]["unwanted"] == [False, "data/facility/benchmarks/channel_finder"]


def test_a_missing_query_file_is_named_at_its_new_path(probe: dict[str, Any]) -> None:
    refusal = probe["benchmarks"]["refusal"]
    assert refusal is not None
    assert "the facility ships no data/facility/benchmarks/channel_finder/tree_queries.json" in (
        refusal
    )
    assert "add tree_queries.json to data/facility/benchmarks/channel_finder/" in refusal
    assert "cross_paradigm" not in refusal


# --- the scaffolding copy --------------------------------------------------------------


def test_the_copy_skips_the_query_directory_and_nothing_beside_it(probe: dict[str, Any]) -> None:
    assert probe["copied"]["shared"]["files"] == [
        "facility/benchmarks/README.md",
        "facility/benchmarks/cross_paradigm/notes.md",
        "facility/identity.yaml",
        "facility/knowledge/index.md",
        "other/keep.txt",
    ]


def test_the_copy_skips_benchmarks_holding_only_the_query_directory(
    probe: dict[str, Any],
) -> None:
    assert probe["copied"]["alone"]["files"] == ["facility/identity.yaml"]
    assert probe["copied"]["alone"]["dirs"] == ["facility"]


def test_the_copy_keeps_a_facility_holding_only_benchmarks(probe: dict[str, Any]) -> None:
    assert probe["copied"]["only"]["files"] == ["other/keep.txt"]
    assert probe["copied"]["only"]["dirs"] == ["facility", "other"]


# --- the import verbs ------------------------------------------------------------------


def test_the_import_writes_table_follows(probe: dict[str, Any]) -> None:
    assert probe["writes"] == {
        "mml": [
            ["records/channels.yaml", "channel"],
            ["records/devices.yaml", "device"],
            ["records/groups.yaml", "group"],
            ["simulation/models.yaml", "model"],
            ["simulation/decks", "deck"],
            ["simulation/seeds.yaml", "seed"],
            ["limits.yaml", "limit"],
            ["identity.yaml", "identity"],
            ["simulation/measurement", "measurement"],
        ],
        "list": [["records/channels.yaml", "channel"], ["records/devices.yaml", "device"]],
        "relations": [],
    }


# --- the simulator view ----------------------------------------------------------------


def test_the_simulator_view_keeps_its_output_names(probe: dict[str, Any]) -> None:
    assert probe["simulator"]["DECKS_DIR"] == "decks"
    assert probe["simulator"]["SCENARIOS_DIR"] == "scenarios"


def test_the_simulator_view_copies_from_the_moved_sources(probe: dict[str, Any]) -> None:
    assert probe["simulator"]["copies"] == {
        "scenarios/quiet/plots/trend.png": _PNG + "moved",
        "decks/ring.json": '{"deck": "moved"}\n',
    }


# --- scenario sessions, the corrector step, the injection engine -----------------------


def test_a_scenario_session_error_names_the_new_path(probe: dict[str, Any]) -> None:
    servers = probe["scenario_servers"]
    assert servers["files"] == ["simulation/scenarios/warm.yaml"]
    assert servers["detail"].startswith("simulation/scenarios/warm.yaml `sessions[0]")


def test_a_missing_corrector_step_names_the_new_path(probe: dict[str, Any]) -> None:
    message, suggestion = probe["step_refusal"]
    assert message.startswith("simulation/measurement/ring.yaml states no ")
    assert suggestion.endswith(" (radians) to simulation/measurement/ring.yaml and rebuild.")


def test_the_injection_engine_names_the_new_models_file(probe: dict[str, Any]) -> None:
    assert probe["injection_sources"] == ["simulation/models.yaml"]


# --- what does not move ----------------------------------------------------------------


def test_the_deployment_content_directories_are_unchanged(probe: dict[str, Any]) -> None:
    assert probe["DEPLOYMENT_CONTENT_DIRS"] == ["knowledge", "ariel", "benchmarks"]


def test_the_vocabulary_render_path_keeps_its_render_prefix(probe: dict[str, Any]) -> None:
    assert probe["VOCABULARY_SOURCE"] == "data/facility/ariel/vocabulary.yml"


# --- the scaffolding copy, query sources one level down ---------------------------------

#: A profile's data tree whose query sources sit directly in a top-level
#: facility directory, beside entries that share a part of its name or path.
_DATA_TREE_ONE_SEGMENT: dict[str, Any] = {
    "facility/identity.yaml": {"code": "demo"},
    "facility/queries/tree_queries.json": "[]\n",
    "facility/queries/in_context_queries.json": "[]\n",
    "facility/queries_notes/keep.md": "Kept.\n",
    "facility/benchmarks/cross_paradigm/queries/tree_queries.json": "[]\n",
    "facility/knowledge/index.md": "# Notes\n",
    "other/queries/keep.txt": "Kept.\n",
}

#: Runs in a fresh interpreter: rebinds the registry, then copies one data tree.
_COPY_PROBE = r"""
import json
import sys
from pathlib import Path

from osprey_facility import source_tree

source_tree._FACILITY_SLOTS = source_tree._with_paths(json.loads(sys.argv[1]))

from osprey.cli.templates.scaffolding import copy_template_data

data_root, out = Path(sys.argv[2]), Path(sys.argv[3])
(out / "templates").mkdir(parents=True)
project = out / "project"
copy_template_data(out / "templates", project, "pkg", "bundle", {}, data_root=data_root)
data = project / "data"
print(json.dumps({
    "files": sorted(p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()),
    "dirs": sorted(p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_dir()),
}))
"""


@pytest.fixture(scope="module")
def one_segment_copy(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The copy of a data tree whose query sources sit in the top-level ``queries``."""
    base = tmp_path_factory.mktemp("one-segment")
    data_root = _write(base / "data", _DATA_TREE_ONE_SEGMENT)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            _COPY_PROBE,
            json.dumps({"benchmark_queries": "queries"}),
            str(data_root),
            str(base / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_copy_skips_a_top_level_query_directory_and_nothing_beside_it(
    one_segment_copy: dict[str, Any],
) -> None:
    assert one_segment_copy["files"] == [
        "facility/benchmarks/cross_paradigm/queries/tree_queries.json",
        "facility/identity.yaml",
        "facility/knowledge/index.md",
        "facility/queries_notes/keep.md",
        "other/queries/keep.txt",
    ]
    assert "facility/queries" not in one_segment_copy["dirs"]
