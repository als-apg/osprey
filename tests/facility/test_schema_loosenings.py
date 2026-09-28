"""The facility schema modules and the NARAD seeds they start from.

The seeds are vendored byte for byte, so their digests are pinned here and in
the seeds README; a copy that drifted from the recorded upstream commit fails
before anything is built on it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = REPO_ROOT / "scripts" / "facility_schema" / "seeds"
SCHEMA_DIR = REPO_ROOT / "src" / "osprey" / "facility" / "schema"

#: The upstream commit the seeds were copied from.
SEED_COMMIT = "02f64d5"

#: sha256 of each vendored seed at :data:`SEED_COMMIT`.
SEED_DIGESTS = {
    "canonical_ingest.yaml": "2efa863505615c555449d575343534463eda7ed23fe296760eeeed78776e9a91",
    "facility_bindings.yaml": "252208465d48ef582b031fc61be2371754eab6dec10582c933e42b54e53c6849",
    "concept_vocabulary.yaml": "48b1dc6cf7b96721706e9b0bdfc286a6224e59f6fa123d2275ffda86dee14b64",
    "shared_semantics.yaml": "f1836e33402f1ffb63d7518ef966374beb2581780cfbf7f3a3b57710717f6629",
}

#: The top-level slots of the facility file, in order.
FACILITY_SLOTS = [
    "schema",
    "identity",
    "classes",
    "places",
    "devices",
    "channels",
    "groups",
    "models",
    "limits",
    "scenarios",
]

#: The multivalued slots that are sets: compared and written sorted by their
#: string form. Every other multivalued slot keeps its source order.
SET_VALUED = {
    ("Channel", "tags"),
    ("Channel", "former_addresses"),
    ("Channel", "endpoint_of"),
    ("Group", "members"),
    ("Device", "groups"),
    ("Measurement", "kinds"),
}

#: Words that name one facility, matched case-sensitively. Each is spelled with
#: a character class so this module does not name the facility itself.
FACILITY_WORDS = (
    r"\bA[L]S\b",
    r"\bl[b]l\b",
    r"\bLBN[L]\b",
    r"\bSPEA[R]\d*\b",
    r"\bNSL[S]\b",
    r"\bUIT[F]\b",
    r"\bJLa[b]\b",
    r"\bCEBA[F]\b",
    r"\bSLA[C]\b",
)

#: Accelerator words the core module never uses, matched case-insensitively.
#: The measurement block's pyAML words (bpm, hcor, vcor, quad, sext, tune,
#: chromaticity, rf and the step keys) are that outside format's own and exempt.
ACCELERATOR_WORDS = (
    r"\baccelerators?\b",
    r"\bbeam(line)?s?\b",
    r"\bmagnets?\b",
    r"\bquadrupoles?\b",
    r"\bsextupoles?\b",
    r"\bdipoles?\b",
    r"\bcorrectors?\b",
    r"\bsynchrotrons?\b",
    r"\bboosters?\b",
    r"\blinacs?\b",
    r"\bstorage\b",
    r"\belectrons?\b",
)

#: The word `ring` in any identifier shape; `string` is not a hit.
RING_TOKEN = r"(?i)\bring\b|Ring(?=[A-Z_])|_ring\b|\bring_"


def _load(name: str) -> dict:
    return yaml.safe_load((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def core() -> dict:
    return _load("core.yaml")


@pytest.fixture(scope="module")
def vocabulary() -> dict:
    return _load("vocabulary.yaml")


def _attrs(core: dict, cls: str) -> dict:
    return core["classes"][cls]["attributes"]


# --- the vendored seeds -------------------------------------------------------


@pytest.mark.parametrize("seed", sorted(SEED_DIGESTS))
def test_vendored_seed_matches_its_pinned_digest(seed: str) -> None:
    digest = hashlib.sha256((SEEDS_DIR / seed).read_bytes()).hexdigest()
    assert digest == SEED_DIGESTS[seed]


def test_seeds_dir_holds_exactly_the_four_seeds_and_the_readme() -> None:
    names = sorted(path.name for path in SEEDS_DIR.iterdir())
    assert names == sorted([*SEED_DIGESTS, "README.md"])


def test_readme_records_the_commit_and_every_digest() -> None:
    readme = (SEEDS_DIR / "README.md").read_text(encoding="utf-8")
    assert f"commit `{SEED_COMMIT}`" in readme
    for seed, digest in SEED_DIGESTS.items():
        assert f"| `{seed}` | `{digest}` |" in readme


# --- the schema modules -------------------------------------------------------


def test_module_names(core: dict, vocabulary: dict) -> None:
    assert core["name"] == "osprey.facility.core"
    assert vocabulary["name"] == "osprey.facility.vocabulary"
    assert "vocabulary" in core["imports"]


def test_schema_view_resolves_core_with_its_vocabulary() -> None:
    from linkml_runtime.utils.schemaview import SchemaView

    view = SchemaView(str(SCHEMA_DIR / "core.yaml"))
    assert "Facility" in view.all_classes()
    assert "BeamPositionMonitor" in view.all_classes()
    assert "signal_role_enum" in view.all_enums()


def test_facility_is_the_one_tree_root_with_exactly_the_ten_slots(core: dict) -> None:
    roots = [name for name, cls in core["classes"].items() if cls.get("tree_root")]
    assert roots == ["Facility"]
    assert list(_attrs(core, "Facility")) == FACILITY_SLOTS


def test_channel_declares_an_optional_signal(core: dict) -> None:
    signal = _attrs(core, "Channel")["signal"]
    assert not signal.get("required")
    assert not signal.get("multivalued")
    assert signal.get("range", core["default_range"]) == "string"


def test_group_declares_an_optional_signals_map_of_strings(core: dict) -> None:
    signals = _attrs(core, "Group")["signals"]
    assert not signals.get("required")
    assert signals.get("multivalued") and signals.get("inlined")
    assert not signals.get("inlined_as_list")
    entry = _attrs(core, signals["range"])
    keys = [name for name, slot in entry.items() if slot and slot.get("key")]
    values = [name for name in entry if name not in keys]
    assert len(keys) == 1 and len(values) == 1
    value = entry[values[0]]
    assert value.get("required")
    assert value.get("range", core["default_range"]) == "string"


def test_device_attributes_is_a_free_map(core: dict) -> None:
    assert _attrs(core, "Device")["attributes"]["range"] == "Any"
    assert core["classes"]["Any"]["class_uri"] == "linkml:Any"


def test_scenario_record_shape(core: dict) -> None:
    scenario = _attrs(core, "Scenario")
    assert list(scenario) == ["name", "overrides", "faults", "archiver", "logbook"]
    faults = scenario["faults"]
    assert faults["range"] == "Any" and not faults.get("multivalued")
    for part in ("<model>", "<address or engine variable>", "`stuck`", "{<fault field>: <value>}"):
        assert part in faults["description"]


def test_measurement_block(core: dict) -> None:
    measurement = _attrs(core, "Measurement")
    assert list(measurement)[:3] == ["kinds", "groups", "instruments"]
    kinds = measurement["kinds"]
    assert kinds.get("multivalued") and not kinds.get("list_elements_ordered")
    assert set(_attrs(core, "MeasurementGroups")) == {"bpm", "hcor", "vcor", "quad", "sext"}
    assert set(_attrs(core, "MeasurementInstruments")) == {"tune", "chromaticity", "rf"}
    assert not any(slot and slot.get("required") for slot in measurement.values())


def test_wiring_drives_element_or_slices_with_an_engine_block(core: dict) -> None:
    wiring = _attrs(core, "Wiring")
    for slot in ("id", "address", "element", "slices", "engine", "calibration"):
        assert slot in wiring
    for computed in ("direction", "unit", "default", "value_range"):
        assert not wiring[computed].get("required")
    assert set(_attrs(core, "Slice")) == {"element", "weight", "device"}
    assert set(_attrs(core, "Calibration")) == {"curve", "inverse", "energy_scaling"}


def test_limits_are_records_only(core: dict) -> None:
    assert set(_attrs(core, "Limits")) == {"records"}
    assert "LimitDefaults" not in core["classes"]
    record = _attrs(core, "LimitRecord")
    assert set(record) - {"address"} == {
        "min_value",
        "max_value",
        "max_step",
        "writable",
        "confirm",
    }


def test_every_record_kind_carries_provenance(core: dict) -> None:
    for cls in ("Place", "Device", "Channel", "Group", "Model", "Wiring"):
        assert _attrs(core, cls)["provenance"]["range"] == "Provenance"
    assert set(_attrs(core, "Provenance")) == {"sources", "fixes", "defaults", "place_from"}


def test_every_multivalued_slot_but_the_sets_keeps_source_order(core: dict) -> None:
    ordered, unordered = set(), set()
    for cls_name, cls in core["classes"].items():
        for slot_name, slot in (cls.get("attributes") or {}).items():
            if slot and slot.get("multivalued"):
                target = ordered if slot.get("list_elements_ordered") else unordered
                target.add((cls_name, slot_name))
    assert unordered == SET_VALUED
    for named in ("slices", "names", "options", "shape", "clamp"):
        assert any(slot == named for _, slot in ordered), named


@pytest.mark.parametrize(
    ("pattern", "flags"),
    [(word, 0) for word in FACILITY_WORDS]
    + [(word, re.IGNORECASE) for word in ACCELERATOR_WORDS]
    + [(RING_TOKEN, 0)],
)
def test_core_names_no_facility_or_accelerator_word(pattern: str, flags: int) -> None:
    text = (SCHEMA_DIR / "core.yaml").read_text(encoding="utf-8")
    hits = [line for line in text.splitlines() if re.search(pattern, line, flags)]
    assert hits == []


def test_the_word_checks_would_catch_a_hit() -> None:
    assert re.search(RING_TOKEN, "  lattice_ring: x")
    assert re.search(RING_TOKEN, "RingModel")
    assert not re.search(RING_TOKEN, "  default_range: string")
    assert re.search(ACCELERATOR_WORDS[1], "a Beam line", re.IGNORECASE)
    assert re.search(FACILITY_WORDS[0], "the " + "A" + "LS demo")


def test_vocabulary_keeps_the_seed_class_tree_and_signal_roles(vocabulary: dict) -> None:
    seed = yaml.safe_load((SEEDS_DIR / "shared_semantics.yaml").read_text(encoding="utf-8"))
    wrappers = {"SemanticDataset", "SemanticDeviceRecord"}
    seed_tree = {
        name: (cls.get("is_a"), bool(cls.get("abstract")))
        for name, cls in seed["classes"].items()
        if name not in wrappers
    }
    tree = {
        name: (cls.get("is_a"), bool(cls.get("abstract")))
        for name, cls in vocabulary["classes"].items()
    }
    assert tree == seed_tree
    seed_roles = seed["enums"]["semantic_signal_enum"]["permissible_values"]
    roles = vocabulary["enums"]["signal_role_enum"]["permissible_values"]
    assert list(roles) == list(seed_roles)


def test_vocabulary_carries_aliases_and_property_names(vocabulary: dict) -> None:
    classes = vocabulary["classes"]
    assert "BPM" in classes["BeamPositionMonitor"]["aliases"]
    roles = vocabulary["enums"]["signal_role_enum"]["permissible_values"]
    assert roles["current_setpoint"]["aliases"]
    properties = vocabulary["enums"]["property_name_enum"]["permissible_values"]
    assert {"betax", "betay", "etax", "s_position", "length"} <= set(properties)
    assert all(value["aliases"] for value in properties.values())


# --- the loosenings table -----------------------------------------------------

_SCRIPT = REPO_ROOT / "scripts" / "facility_schema" / "loosenings.py"
_spec = importlib.util.spec_from_file_location("facility_schema_loosenings", _SCRIPT)
assert _spec and _spec.loader
loosenings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loosenings)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _load("loosenings.yaml")["rows"]


def _required_seed_slots() -> set[tuple[str, str]]:
    """Every required slot of the seeds, read here without the script."""
    pairs = set()
    for seed in SEED_DIGESTS:
        schema = yaml.safe_load((SEEDS_DIR / seed).read_text(encoding="utf-8"))
        for name, slot in (schema.get("slots") or {}).items():
            if isinstance(slot, dict) and slot.get("required") is True:
                pairs.add((seed, name))
    return pairs


def test_committed_table_is_what_the_script_writes() -> None:
    committed = (SCHEMA_DIR / "loosenings.yaml").read_text(encoding="utf-8")
    assert committed == loosenings.render()


def test_every_required_seed_slot_has_exactly_one_row(rows: list[dict]) -> None:
    keys = [(row["seed"], row["slot"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert set(keys) == _required_seed_slots()
    assert len(keys) == 40
    assert sum(seed == "canonical_ingest.yaml" for seed, _ in keys) == 22


def test_every_fate_is_one_of_the_four_shapes(rows: list[dict]) -> None:
    shape = re.compile(r"^(dropped|optional|header|renamed:[A-Za-z_][A-Za-z0-9_]*)$")
    assert [row for row in rows if not shape.match(row["fate"])] == []
    assert all(row["why"] for row in rows)


def test_every_renamed_slot_exists_in_core(rows: list[dict], core: dict) -> None:
    slots = {name for cls in core["classes"].values() for name in (cls.get("attributes") or {})}
    renamed = {row["fate"].split(":", 1)[1] for row in rows if row["fate"].startswith("renamed:")}
    assert renamed - slots == set()


def test_headline_loosenings(rows: list[dict]) -> None:
    fate = {(row["seed"], row["slot"]): row["fate"] for row in rows}
    assert fate["canonical_ingest.yaml", "beamline_sections"] == "optional"
    assert fate["canonical_ingest.yaml", "devices"] == "optional"
    assert fate["canonical_ingest.yaml", "raw_type"] == "dropped"
    assert fate["canonical_ingest.yaml", "source_section_id"] == "optional"
    assert fate["canonical_ingest.yaml", "unit"] == "optional"
    assert fate["facility_bindings.yaml", "control_system"] == "dropped"
    assert fate["facility_bindings.yaml", "binding_id"] == "renamed:id"
    assert fate["facility_bindings.yaml", "canonical_device_id"] == "renamed:on"
    for seed in ("canonical_ingest.yaml", "facility_bindings.yaml", "shared_semantics.yaml"):
        assert fate[seed, "facility"] == "header"


def test_script_refuses_a_required_slot_it_does_not_decide(tmp_path: Path) -> None:
    for seed in SEED_DIGESTS:
        (tmp_path / seed).write_bytes((SEEDS_DIR / seed).read_bytes())
    extra = tmp_path / "canonical_ingest.yaml"
    extra.write_text(
        extra.read_text(encoding="utf-8") + "\n  new_slot:\n    required: true\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="undecided required slot: canonical_ingest.yaml new_slot"):
        loosenings.render(tmp_path)


def test_script_reads_only_the_vendored_seeds() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "narad-uitf" not in source
    assert loosenings.SEEDS_DIR == SEEDS_DIR
    assert set(loosenings.SEED_FILES) == set(SEED_DIGESTS)
