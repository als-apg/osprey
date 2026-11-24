"""The facility tree's tables, pinned as literals.

Every table of the package that names an entry of the facility tree is held
here to its exact value, order included wherever a consumer can observe it.
A table computed from a declaration must keep producing these values; a
change to one is a visible, reviewed edit of this file.

The expected values are module constants so other tests can import them.
"""

from __future__ import annotations

from osprey_facility import combine, knowledge_links, layers, roots, sources, validate
from osprey_facility.layers import list as list_layer
from osprey_facility.layers import relations as relations_layer
from osprey_facility.layers.mml import decks as mml_decks
from osprey_facility.layers.mml import importer as mml_importer
from osprey_facility.layers.mml import mapping as mml_mapping
from osprey_facility.layers.mml import seed as mml_seed

TOP_FILES: frozenset[str] = frozenset(
    {
        "identity.yaml",
        "classes.yaml",
        "signals.yaml",
        "models.yaml",
        "limits.yaml",
        "seeds.yaml",
        "fixes.yaml",
        "documents.yaml",
        "policy.yaml",
    }
)

RECORD_FILES: tuple[tuple[str, str], ...] = (
    ("places.yaml", "place"),
    ("devices.yaml", "device"),
    ("channels.yaml", "channel"),
    ("groups.yaml", "group"),
    ("relations.yaml", "relation"),
)

LAYER_OWN_FILES: frozenset[str] = frozenset({"mapping.yaml"})

RELATIONS_MARK = "relations.csv"

TOP_SLOT_FILES: tuple[tuple[str, str], ...] = (
    ("limits", "limits.yaml"),
    ("classes", "classes.yaml"),
    ("signals", "signals.yaml"),
    ("identity", "identity.yaml"),
    ("documents", "documents.yaml"),
)

FIXES_FILE = "fixes.yaml"

KNOWLEDGE_DIR = "knowledge"

LAYER_ROOT = "imported"

MML_LAYER_DIR = "imported/mml"

MML_MAPPING_FILE = "imported/mml/mapping.yaml"

MML_DECKS_DIR = "imported/mml/decks"

MML_SEED_FILES: tuple[tuple[str, str], ...] = (
    ("LIMITS_FILE", "limits.yaml"),
    ("READOUT_FILE", "scenarios/readout.yaml"),
    ("SEEDS_FILE", "seeds.yaml"),
    ("CLASSES_FILE", "classes.yaml"),
    ("IDENTITY_FILE", "identity.yaml"),
    ("MEASUREMENT_DIR", "measurement"),
)

LIST_LAYER_DIR = "imported/list"

RELATIONS_RECORD_FILES: tuple[str, ...] = (
    "relations.yaml",
    "devices.yaml",
    "places.yaml",
    "groups.yaml",
)

RELATIONS_TABLE_FILES: tuple[tuple[str, str], ...] = (
    ("relations", "relations.csv"),
    ("names", "names.csv"),
    ("places", "places.csv"),
    ("members", "members.csv"),
)

DEPLOYMENT_CONTENT_DIRS: tuple[str, ...] = ("knowledge", "ariel", "benchmarks")


def test_the_root_single_files() -> None:
    assert sources._TOP_FILES == TOP_FILES


def test_the_record_files_and_their_kinds_in_order() -> None:
    assert tuple(sources.RECORD_FILES.items()) == RECORD_FILES


def test_the_files_a_layer_keeps_for_itself() -> None:
    assert sources.LAYER_OWN_FILES == LAYER_OWN_FILES


def test_the_relations_layer_mark() -> None:
    assert sources.RELATIONS_MARK == RELATIONS_MARK


def test_the_document_slot_files_in_order() -> None:
    assert tuple(validate._TOP_SLOT_FILES.items()) == TOP_SLOT_FILES


def test_the_fixes_file() -> None:
    assert combine.FIXES_FILE == FIXES_FILE


def test_the_knowledge_directory() -> None:
    assert knowledge_links.KNOWLEDGE_DIR == KNOWLEDGE_DIR


def test_the_layer_root() -> None:
    assert layers.LAYER_ROOT == LAYER_ROOT


def test_the_mml_layer_directory_everywhere_it_is_exported() -> None:
    assert mml_mapping.LAYER_DIR == MML_LAYER_DIR
    assert mml_importer.LAYER_DIR == MML_LAYER_DIR


def test_the_mml_mapping_file() -> None:
    assert mml_mapping.MAPPING_FILE == MML_MAPPING_FILE


def test_the_mml_decks_directory() -> None:
    assert mml_decks.DECKS_DIR == MML_DECKS_DIR


def test_the_mml_seed_once_files() -> None:
    assert tuple((name, getattr(mml_seed, name)) for name, _ in MML_SEED_FILES) == MML_SEED_FILES


def test_the_mml_seed_once_files_stay_exported() -> None:
    assert {name for name, _ in MML_SEED_FILES} <= set(mml_seed.__all__)


def test_the_list_layer_directory() -> None:
    assert list_layer.LAYER_DIR == LIST_LAYER_DIR


def test_the_relations_layer_record_files_in_order() -> None:
    assert relations_layer.RECORD_FILES == RELATIONS_RECORD_FILES


def test_the_relations_layer_table_files_in_order() -> None:
    assert tuple(relations_layer.TABLE_FILES.items()) == RELATIONS_TABLE_FILES


def test_the_deployment_content_directories_in_order() -> None:
    assert roots.DEPLOYMENT_CONTENT_DIRS == DEPLOYMENT_CONTENT_DIRS
