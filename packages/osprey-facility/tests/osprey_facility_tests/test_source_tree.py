"""The facility tree's layout declaration: its slots, accessors and registry.

The package's tables are computed from these functions; the derivation tests
rebuild every pinned table value from ``source_tree`` alone, so the declaration
holds everything a table needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from osprey_facility import source_tree
from osprey_facility.source_tree import (
    FILE,
    display_path,
    keyed_dir,
    keyed_file,
    layer_dir,
    layer_entry,
    layer_root,
    path,
    slot,
    slots,
)
from osprey_facility_tests import test_source_tree_tables as pinned

#: A layout with nested sources: a single file, a parent of record files, a
#: keyed directory and the layer root each under a new directory.
_MOVED = {
    "limits": "machine/limits.yaml",
    "records": "machine/records",
    "measurement": "machine/measurement",
    "scenarios": "sim/scenarios",
    "imported": "machine/imported",
    "knowledge": "notes/knowledge",
}


@pytest.fixture
def moved(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The registry rebound to :data:`_MOVED` for one test."""
    monkeypatch.setattr(source_tree, "_FACILITY_SLOTS", source_tree._with_paths(_MOVED))
    yield


# --- the registry ---------------------------------------------------------------------


def test_names_and_resolved_paths_are_unique() -> None:
    names = [row.name for row in slots()]
    paths = [path(name) for name in names]
    assert len(set(names)) == len(names)
    assert len(set(paths)) == len(paths)


def test_every_path_is_relative_posix() -> None:
    for row in slots():
        resolved = path(row.name)
        assert not resolved.startswith("/") and not resolved.endswith("/"), resolved
        assert "\\" not in resolved and "//" not in resolved, resolved
        assert ".." not in resolved.split("/"), resolved


def test_every_slot_the_tree_names_is_declared() -> None:
    assert [row.name for row in slots()] == [
        "identity",
        "classes",
        "signals",
        "models",
        "limits",
        "seeds",
        "fixes",
        "documents",
        "policy",
        "records",
        "places",
        "devices",
        "channels",
        "groups",
        "relations",
        "imported",
        "scenarios",
        "measurement",
        "decks",
        "knowledge",
        "ariel",
        "ariel_vocabulary",
        "benchmark_queries",
        "in_context_queries",
        "tree_queries",
    ]


def test_the_row_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        slot("limits").path = "elsewhere.yaml"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"limits": "/abs/limits.yaml"}, "not relative POSIX"),
        ({"limits": "records/"}, "not relative POSIX"),
        ({"limits": "a//limits.yaml"}, "not relative POSIX"),
        ({"limits": "../limits.yaml"}, "not relative POSIX"),
        ({"limits": "a\\limits.yaml"}, "not relative POSIX"),
        ({"limits": ""}, "not relative POSIX"),
        ({"limits": "seeds.yaml"}, "share a path"),
        ({"knowledge": "records/places.yaml"}, "share a path"),
        ({"knowledge": "limits.yaml/knowledge"}, "inside file slot 'limits'"),
        ({"records": "limits.yaml/records"}, "inside file slot 'limits'"),
    ],
)
def test_a_substitution_that_breaks_the_layout_is_refused(
    change: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        source_tree._with_paths(change)


def test_a_substitution_of_an_unknown_slot_is_refused() -> None:
    with pytest.raises(KeyError, match="no facility slot"):
        source_tree._with_paths({"nowhere": "nowhere.yaml"})


def test_a_substitution_leaves_the_bound_registry_alone() -> None:
    before = source_tree._FACILITY_SLOTS
    source_tree._with_paths(_MOVED)
    assert source_tree._FACILITY_SLOTS is before
    assert path("limits") == "limits.yaml"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (source_tree._FacilitySlot("limits", "x.yaml", FILE), "declared twice"),
        (source_tree._FacilitySlot("odd", "odd", "folder"), "unknown kind"),
        (source_tree._FacilitySlot("odd", "odd.yaml", FILE, "nowhere"), "undeclared"),
        (source_tree._FacilitySlot("odd", "odd.yaml", FILE, "limits"), "file slot"),
        (source_tree._FacilitySlot("odd", "odd.yaml", FILE, record_kind="odd"), "record slot"),
        (source_tree._FacilitySlot("odd", "odd.yaml", FILE, layer_name="decks"), "layer name"),
    ],
)
def test_an_invalid_row_is_refused(row: source_tree._FacilitySlot, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source_tree._check((*source_tree._FACILITY_SLOTS, row))


# --- the accessors return the declared strings ----------------------------------------


def test_the_root_single_files() -> None:
    assert path("identity") == "identity.yaml"
    assert path("policy") == "policy.yaml"


def test_a_record_file_sits_in_records() -> None:
    assert path("records") == "records"
    assert path("channels") == "records/channels.yaml"
    assert slot("channels").record_kind == "channel"


def test_the_keyed_directories() -> None:
    assert keyed_file("measurement", "SR") == "measurement/SR.yaml"
    assert keyed_file("scenarios", "readout") == "scenarios/readout.yaml"
    assert keyed_dir("scenarios", "bump") == "scenarios/bump"


def test_the_keyed_directories_glob() -> None:
    assert keyed_file("scenarios", "*") == "scenarios/*.yaml"


def test_the_layer_directories_and_entries() -> None:
    assert path("imported") == "imported"
    assert layer_root() == "imported"
    assert layer_dir("mml") == "imported/mml"
    assert layer_entry("mml", "models") == "imported/mml/models.yaml"
    assert layer_entry("mml", "decks") == "imported/mml/decks"
    assert layer_entry("list", "channels") == "imported/list/channels.yaml"


def test_the_content_directories_and_their_files() -> None:
    assert path("decks") == "decks"
    assert path("knowledge") == "knowledge"
    assert source_tree._inside("knowledge", "index.md") == "knowledge/index.md"
    assert path("ariel") == "ariel"
    assert path("ariel_vocabulary") == "ariel/vocabulary.yml"
    assert path("benchmark_queries") == "benchmarks/cross_paradigm/queries"
    assert path("in_context_queries") == (
        "benchmarks/cross_paradigm/queries/in_context_queries.json"
    )
    assert path("tree_queries") == "benchmarks/cross_paradigm/queries/tree_queries.json"
    assert slot("tree_queries").path == "tree_queries.json"
    assert slot("in_context_queries").path == "in_context_queries.json"


def test_the_display_path() -> None:
    assert source_tree._DISPLAY_PREFIX == "data/facility"
    assert display_path(path("seeds")) == "data/facility/seeds.yaml"
    assert display_path(path("ariel_vocabulary")) == "data/facility/ariel/vocabulary.yml"


def test_the_slot_kinds() -> None:
    assert slot("limits").kind == FILE
    assert slot("records").kind == source_tree._DIRECTORY
    assert slot("measurement").kind == source_tree._KEYED_DIRECTORY
    assert slot("scenarios").kind == source_tree._KEYED_DIRECTORY


def test_an_unknown_slot_is_refused() -> None:
    with pytest.raises(KeyError, match="no facility slot"):
        path("nowhere")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: keyed_file("limits", "x"), "not a keyed directory"),
        (lambda: keyed_dir("records", "x"), "not a keyed directory"),
        (lambda: source_tree._inside("limits", "x"), "is a file"),
        (lambda: source_tree._inside("knowledge"), "names nothing"),
        (lambda: layer_entry("mml", "limits"), "carries no 'limits'"),
    ],
)
def test_an_accessor_refuses_the_wrong_slot(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


# --- every accessor follows a rebound registry -----------------------------------------


@pytest.mark.usefixtures("moved")
def test_every_accessor_follows_a_rebound_registry() -> None:
    assert slots() is source_tree._FACILITY_SLOTS
    assert slot("limits").path == "machine/limits.yaml"
    assert path("limits") == "machine/limits.yaml"
    assert keyed_file("measurement", "SR") == "machine/measurement/SR.yaml"
    assert keyed_dir("scenarios", "bump") == "sim/scenarios/bump"
    assert source_tree._inside("knowledge", "index.md") == "notes/knowledge/index.md"
    assert layer_root() == "machine/imported"
    assert layer_dir("mml") == "machine/imported/mml"
    assert layer_entry("mml", "decks") == "machine/imported/mml/decks"
    assert display_path(path("limits")) == "data/facility/machine/limits.yaml"


@pytest.mark.usefixtures("moved")
def test_a_moved_parent_moves_its_children() -> None:
    assert path("places") == "machine/records/places.yaml"
    assert path("relations") == "machine/records/relations.yaml"


@pytest.mark.usefixtures("moved")
def test_a_layer_entry_never_follows_the_authored_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_tree,
        "_FACILITY_SLOTS",
        source_tree._with_paths({"models": "machine/models.yaml", "decks": "machine/decks"}),
    )
    assert path("models") == "machine/models.yaml"
    assert layer_entry("mml", "models") == "machine/imported/mml/models.yaml"
    assert layer_entry("mml", "decks") == "machine/imported/mml/decks"


# --- the derivation rules reproduce every pinned table ----------------------------------


def _top_files() -> frozenset[str]:
    return frozenset(
        path(row.name) for row in slots() if row.kind == FILE and "/" not in path(row.name)
    )


def _record_files() -> tuple[tuple[str, str], ...]:
    return tuple(
        (row.layer_name, row.record_kind)  # type: ignore[misc]
        for row in slots()
        if row.record_kind is not None
    )


def _content_dirs() -> tuple[str, ...]:
    return tuple(dict.fromkeys(path(row.name).split("/")[0] for row in slots() if row.content))


def test_the_root_single_files_derive() -> None:
    assert _top_files() == pinned.TOP_FILES


def test_the_record_files_derive_in_order() -> None:
    assert _record_files() == pinned.RECORD_FILES


def test_the_document_slot_files_derive_in_order() -> None:
    validator_slots = ("limits", "classes", "signals", "identity", "documents")
    assert tuple((name, path(name)) for name in validator_slots) == pinned.TOP_SLOT_FILES


def test_the_fixes_file_derives() -> None:
    assert path("fixes") == pinned.FIXES_FILE


def test_the_knowledge_directory_derives() -> None:
    assert path("knowledge") == pinned.KNOWLEDGE_DIR


def test_the_layer_root_derives() -> None:
    assert path("imported") == pinned.LAYER_ROOT
    assert layer_root() == pinned.LAYER_ROOT


def test_the_mml_layer_paths_derive() -> None:
    (own_file,) = pinned.LAYER_OWN_FILES
    assert layer_dir("mml") == pinned.MML_LAYER_DIR
    assert layer_dir("mml", own_file) == pinned.MML_MAPPING_FILE
    assert layer_entry("mml", "decks") == pinned.MML_DECKS_DIR


def test_the_mml_seed_once_files_derive() -> None:
    seeded = (
        ("LIMITS_FILE", path("limits")),
        ("READOUT_FILE", keyed_file("scenarios", "readout")),
        ("SEEDS_FILE", path("seeds")),
        ("CLASSES_FILE", path("classes")),
        ("IDENTITY_FILE", path("identity")),
        ("MEASUREMENT_DIR", path("measurement")),
    )
    assert seeded == pinned.MML_SEED_FILES


def test_the_list_layer_directory_derives() -> None:
    assert layer_dir("list") == pinned.LIST_LAYER_DIR


def test_the_relations_layer_record_files_derive_in_order() -> None:
    importer_slots = ("relations", "devices", "places", "groups")
    derived = tuple(slot(name).layer_name for name in importer_slots)
    assert derived == pinned.RELATIONS_RECORD_FILES
    assert set(derived) <= {name for name, _ in pinned.RECORD_FILES}


def test_the_deployment_content_directories_derive_in_order() -> None:
    assert _content_dirs() == pinned.DEPLOYMENT_CONTENT_DIRS


def test_the_files_a_layer_keeps_for_itself_are_no_slot() -> None:
    entries = {row.layer_name for row in slots()} | {path(row.name) for row in slots()}
    assert not (pinned.LAYER_OWN_FILES | {pinned.RELATIONS_MARK}) & entries
