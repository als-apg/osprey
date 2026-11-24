"""Moving facility sources by editing only the layout declaration.

A fresh interpreter rebinds the declaration's registry to a layout where the
vocabulary, provenance and simulation sources sit in their own directories and
the benchmark queries sit under ``benchmarks/channel_finder``, before any
consumer is imported. It then imports the reader, the validators and the
layers and reports what they computed and read from trees in ``tmp_path``.
Nothing in the repository moves.

Expected: every table that names a moved source follows it; the reader loads
each source from its new path and records that path as provenance; a
validator's error names the new path; imported layers stay flat; and the same
files left at their old root locations are not read.
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

_CHANNELS = [
    {"id": "SR:BPM1:X", "on": {"place": "SR"}},
    {"id": "TL:BPM1:X", "on": {"place": "TL"}, "description": "Transfer line BPM."},
]
_PLACES = [{"id": "SR"}, {"id": "TL"}]
_CLASSES = [{"class": "FastKicker", "parent": "Magnet", "description": "A fast kicker."}]
_SIGNALS = [{"name": "kick_strength", "description": "The kick a kicker applies."}]
_DOCUMENTS = [{"id": "dwg-1", "title": "Rack layout", "kind": "drawing"}]
_POLICY = {"relations": {"precedence": {"default": ["drawing"]}}}
_SEEDS = {"SR:BPM1:X": {"nominal": 0.0}}
_AUTHORED_MODELS = [{"name": "ring", "engine": "pyat"}]
_LAYER_MODELS = [{"name": "transfer", "engine": "pyat"}]
_FIXES = {
    "schema": "osprey.facility.fixes/1",
    "fixes": [
        {
            "op": "set",
            "kind": "channel",
            "id": "TL:BPM1:X",
            "fields": {"description": "Transfer line BPM, horizontal."},
            "was": {"description": {"mml": "Transfer line BPM."}},
            "why": "The export names no plane.",
        }
    ],
}
_SCENARIO = {"description": "A quiet day.", "overrides": {"SR:BPM1:X": 0.0}}

#: One source tree in the moved layout, one imported layer included.
_MOVED_TREE: dict[str, Any] = {
    "identity.yaml": {"code": "demo"},
    "records/places.yaml": _PLACES,
    "records/channels.yaml": _CHANNELS[:1],
    "vocabulary/classes.yaml": _CLASSES,
    "vocabulary/signals.yaml": _SIGNALS,
    "provenance/documents.yaml": _DOCUMENTS,
    "provenance/policy.yaml": _POLICY,
    "provenance/fixes.yaml": _FIXES,
    "simulation/models.yaml": _AUTHORED_MODELS,
    "simulation/seeds.yaml": _SEEDS,
    "simulation/scenarios/quiet.yaml": _SCENARIO,
    "simulation/measurement/ring.yaml": {},
    "simulation/decks/ring.json": "{}\n",
    "imported/mml/channels.yaml": _CHANNELS[1:],
    "imported/mml/models.yaml": _LAYER_MODELS,
    "imported/mml/decks/transfer.json": "{}\n",
}

#: The same sources at their root locations, beside one that sits where it moved to.
_OLD_TREE: dict[str, Any] = {
    "identity.yaml": {"code": "demo"},
    "records/places.yaml": _PLACES,
    "records/channels.yaml": _CHANNELS[:1],
    "classes.yaml": _CLASSES,
    "signals.yaml": _SIGNALS,
    "documents.yaml": _DOCUMENTS,
    "policy.yaml": _POLICY,
    "fixes.yaml": _FIXES,
    "models.yaml": _AUTHORED_MODELS,
    "seeds.yaml": _SEEDS,
    "scenarios/quiet.yaml": _SCENARIO,
    "measurement/ring.yaml": {},
}

#: A seed for an address no channel has: a validator stop on the seeds file.
_BAD_SEED_TREE: dict[str, Any] = {
    "records/places.yaml": _PLACES,
    "records/channels.yaml": _CHANNELS[:1],
    "simulation/seeds.yaml": {"SR:NOWHERE": {"nominal": 0.0}},
}

#: Runs in a fresh interpreter: rebinds the registry, then imports the consumers.
_PROBE = r"""
import json
import sys
from pathlib import Path

consumers_before = sorted(
    name for name in sys.modules
    if name.startswith("osprey_facility.") and name != "osprey_facility.source_tree"
)
from osprey_facility import source_tree

source_tree._FACILITY_SLOTS = source_tree._with_paths(json.loads(sys.argv[1]))

from osprey_facility import combine, compute, knowledge_links, layers, sources, validate, wiring
from osprey_facility.build import build_facility
from osprey_facility.layers import list as list_layer
from osprey_facility.layers import relations as relations_layer
from osprey_facility.layers.mml import decks as mml_decks
from osprey_facility.layers.mml import mapping as mml_mapping
from osprey_facility.layers.mml import seed as mml_seed

moved, old, bad_seed = (Path(arg) for arg in sys.argv[2:5])


def errors(found):
    return [
        {"sources": list(e.sources), "record_id": e.record_id, "detail": e.detail}
        for e in found
    ]


def loaded(root):
    result = sources.load_sources(root)
    got = result.sources
    return {
        "errors": errors(result.errors),
        "identity": got.identity,
        "classes": got.classes,
        "signals": got.signals,
        "documents": got.documents,
        "policy": got.policy,
        "fixes": got.fixes,
        "seeds": got.seeds,
        "scenarios": [s["name"] for s in got.scenarios],
        "measurement": sorted(got.measurement),
        "records": [[r.kind, r.id, r.layer, r.file] for r in got.records],
    }


def provenance(document):
    return {
        plural: {
            str(record.get("id", record.get("name"))): sorted(
                {(s["layer"], s["file"]) for s in record["provenance"]["sources"]}
            )
            for record in document.get(plural, [])
            if "provenance" in record
        }
        for plural in ("channels", "models")
    }


document = build_facility(moved, project_name="demo", extra_checks=())
report = validate.run_stages(bad_seed, project_name="demo", extra_checks=())

print(json.dumps({
    "consumers_before": consumers_before,
    "tables": {
        "sources._TOP_FILES": sorted(sources._TOP_FILES),
        "sources.RECORD_FILES": sources.RECORD_FILES,
        "validate._TOP_SLOT_FILES": validate._TOP_SLOT_FILES,
        "combine.FIXES_FILE": combine.FIXES_FILE,
        "compute._MODELS_FILE": compute._MODELS_FILE,
        "compute._LIMITS_FILE": compute._LIMITS_FILE,
        "wiring._MODELS_FILE": wiring._MODELS_FILE,
        "knowledge_links.KNOWLEDGE_DIR": knowledge_links.KNOWLEDGE_DIR,
        "seed.LIMITS_FILE": mml_seed.LIMITS_FILE,
        "seed.READOUT_FILE": mml_seed.READOUT_FILE,
        "seed.SEEDS_FILE": mml_seed.SEEDS_FILE,
        "seed.CLASSES_FILE": mml_seed.CLASSES_FILE,
        "seed.IDENTITY_FILE": mml_seed.IDENTITY_FILE,
        "seed.MEASUREMENT_DIR": mml_seed.MEASUREMENT_DIR,
        "layers.LAYER_ROOT": layers.LAYER_ROOT,
        "mapping.LAYER_DIR": mml_mapping.LAYER_DIR,
        "mapping.MAPPING_FILE": mml_mapping.MAPPING_FILE,
        "decks.DECKS_DIR": mml_decks.DECKS_DIR,
        "list.LAYER_DIR": list_layer.LAYER_DIR,
        "relations.RECORD_FILES": list(relations_layer.RECORD_FILES),
        "layer models entry": source_tree.layer_entry("mml", "models"),
        "decks slot": source_tree.path("decks"),
        "in-context queries": source_tree.path("in_context_queries"),
        "tree queries": source_tree.path("tree_queries"),
    },
    "moved": loaded(moved),
    "provenance": provenance(document),
    "fixed": {
        record["id"]: [record.get("description"), record["provenance"]["fixes"]]
        for record in document["channels"]
    },
    "old": loaded(old),
    "bad_seed": {"failed": report.failed, "errors": errors(report.errors)},
}, default=sorted))
"""


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """What the consumers computed and read under the moved layout."""
    base = tmp_path_factory.mktemp("substitution")
    trees = [
        _write(base / name, files)
        for name, files in (("moved", _MOVED_TREE), ("old", _OLD_TREE), ("bad", _BAD_SEED_TREE))
    ]
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(_MOVED), *map(str, trees)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_registry_is_rebound_before_any_consumer_is_imported(probe: dict[str, Any]) -> None:
    assert probe["consumers_before"] == []


# --- tables ----------------------------------------------------------------------------


def test_the_root_file_table_keeps_only_the_files_left_at_the_root(
    probe: dict[str, Any],
) -> None:
    assert probe["tables"]["sources._TOP_FILES"] == ["identity.yaml", "limits.yaml"]


def test_the_document_slot_files_follow(probe: dict[str, Any]) -> None:
    assert probe["tables"]["validate._TOP_SLOT_FILES"] == {
        "limits": "limits.yaml",
        "classes": "vocabulary/classes.yaml",
        "signals": "vocabulary/signals.yaml",
        "identity": "identity.yaml",
        "documents": "provenance/documents.yaml",
    }


def test_the_validation_constants_follow(probe: dict[str, Any]) -> None:
    tables = probe["tables"]
    assert tables["combine.FIXES_FILE"] == "provenance/fixes.yaml"
    assert tables["compute._MODELS_FILE"] == "simulation/models.yaml"
    assert tables["wiring._MODELS_FILE"] == "simulation/models.yaml"
    assert tables["compute._LIMITS_FILE"] == "limits.yaml"
    assert tables["knowledge_links.KNOWLEDGE_DIR"] == "knowledge"


#: The importer's seed paths, each a module constant of ``layers.mml.seed``.
_SEED_KEYS = (
    "LIMITS_FILE",
    "READOUT_FILE",
    "SEEDS_FILE",
    "CLASSES_FILE",
    "IDENTITY_FILE",
    "MEASUREMENT_DIR",
)


def test_the_importer_seed_paths_follow(probe: dict[str, Any]) -> None:
    tables = probe["tables"]
    assert {key: tables[f"seed.{key}"] for key in _SEED_KEYS} == {
        "LIMITS_FILE": "limits.yaml",
        "READOUT_FILE": "simulation/scenarios/readout.yaml",
        "SEEDS_FILE": "simulation/seeds.yaml",
        "CLASSES_FILE": "vocabulary/classes.yaml",
        "IDENTITY_FILE": "identity.yaml",
        "MEASUREMENT_DIR": "simulation/measurement",
    }


def test_the_layer_constants_stay_flat(probe: dict[str, Any]) -> None:
    tables = probe["tables"]
    assert tables["layers.LAYER_ROOT"] == "imported"
    assert tables["mapping.LAYER_DIR"] == "imported/mml"
    assert tables["mapping.MAPPING_FILE"] == "imported/mml/mapping.yaml"
    assert tables["decks.DECKS_DIR"] == "imported/mml/decks"
    assert tables["list.LAYER_DIR"] == "imported/list"
    assert tables["layer models entry"] == "imported/mml/models.yaml"
    assert tables["sources.RECORD_FILES"] == {
        "places.yaml": "place",
        "devices.yaml": "device",
        "channels.yaml": "channel",
        "groups.yaml": "group",
        "relations.yaml": "relation",
    }
    assert tables["relations.RECORD_FILES"] == [
        "relations.yaml",
        "devices.yaml",
        "places.yaml",
        "groups.yaml",
    ]


def test_the_moved_directories_and_their_files_follow(probe: dict[str, Any]) -> None:
    tables = probe["tables"]
    assert tables["decks slot"] == "simulation/decks"
    assert tables["in-context queries"] == "benchmarks/channel_finder/in_context_queries.json"
    assert tables["tree queries"] == "benchmarks/channel_finder/tree_queries.json"


# --- the reader on the moved tree ------------------------------------------------------


def test_every_source_loads_from_its_new_path(probe: dict[str, Any]) -> None:
    moved = probe["moved"]
    assert moved["errors"] == []
    assert moved["identity"] == {"code": "demo"}
    assert moved["classes"] == _CLASSES
    assert moved["signals"] == _SIGNALS
    assert moved["documents"] == _DOCUMENTS
    assert moved["policy"] == _POLICY
    assert moved["fixes"] == _FIXES
    assert moved["seeds"] == _SEEDS
    assert moved["scenarios"] == ["quiet"]
    assert moved["measurement"] == ["ring"]


def test_each_record_names_the_file_it_was_read_from(probe: dict[str, Any]) -> None:
    assert sorted(probe["moved"]["records"]) == [
        ["channel", "SR:BPM1:X", "authored", "records/channels.yaml"],
        ["channel", "TL:BPM1:X", "mml", "imported/mml/channels.yaml"],
        ["model", "ring", "authored", "simulation/models.yaml"],
        ["model", "transfer", "mml", "imported/mml/models.yaml"],
        ["place", "SR", "authored", "records/places.yaml"],
        ["place", "TL", "authored", "records/places.yaml"],
    ]


def test_the_built_file_carries_the_new_paths_as_provenance(probe: dict[str, Any]) -> None:
    built = probe["provenance"]
    assert built["models"] == {
        "ring": [
            ["authored", "simulation/measurement/ring.yaml"],
            ["authored", "simulation/models.yaml"],
        ],
        "transfer": [["mml", "imported/mml/models.yaml"]],
    }
    assert built["channels"] == {
        "SR:BPM1:X": [
            ["authored", "records/channels.yaml"],
            ["authored", "simulation/seeds.yaml"],
        ],
        "TL:BPM1:X": [["mml", "imported/mml/channels.yaml"]],
    }


def test_the_fixes_file_is_applied_from_its_new_path(probe: dict[str, Any]) -> None:
    fix = _FIXES["fixes"][0]
    assert probe["fixed"]["TL:BPM1:X"] == [
        fix["fields"]["description"],
        [{"op": "set", "why": fix["why"]}],
    ]


def test_a_validator_error_names_the_new_path(probe: dict[str, Any]) -> None:
    bad = probe["bad_seed"]
    assert bad["failed"] is not None
    assert [e["sources"] for e in bad["errors"] if e["record_id"] == "SR:NOWHERE"] == [
        ["simulation/seeds.yaml"]
    ]


# --- the old root locations ------------------------------------------------------------


def test_sources_left_at_their_old_locations_are_not_read(probe: dict[str, Any]) -> None:
    old = probe["old"]
    assert old["classes"] == []
    assert old["signals"] == []
    assert old["documents"] == []
    assert old["policy"] is None
    assert old["fixes"] is None
    assert old["seeds"] == {}
    assert old["scenarios"] == []
    assert old["measurement"] == []
    assert [r for r in old["records"] if r[0] == "model"] == []


def test_old_root_files_are_refused_as_strays(probe: dict[str, Any]) -> None:
    strays = sorted(
        e["sources"][0]
        for e in probe["old"]["errors"]
        if e["detail"] == "is not a facility source file"
    )
    assert strays == [
        "classes.yaml",
        "documents.yaml",
        "fixes.yaml",
        "models.yaml",
        "policy.yaml",
        "seeds.yaml",
        "signals.yaml",
    ]
