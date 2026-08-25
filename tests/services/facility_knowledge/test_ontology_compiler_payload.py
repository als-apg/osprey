"""Tests for the ontology compiler's schema-to-payload translation.

Two things are pinned here.

The first is the **golden equivalence**: compiling the shipped
``demo_ontology.yaml`` must reproduce, class for class and family for family,
the table :func:`load_demo_ontology` reads from the committed
``demo_ontology.json``.  That is the whole premise of the migration — the YAML
is the source the JSON is generated from, so the day the two disagree is the
day the compiled artifact stopped meaning what its schema says.  Comparing the
parsed *tables* rather than the raw JSON keeps the assertion about content:
formatting and key order belong to ``render.py``.

The second is the **rejection surface**.  A LinkML schema can express far more
than an ontology table can hold, and expansion of a CURIE fails soft — an
undeclared prefix comes back unexpanded rather than raising — so every check in
``payload.py`` has to be proven to fire rather than assumed to.  Each rejection
test therefore asserts the offending element is *named* in the message: an
operator fixing the schema needs to know which class, which family token, not
merely that something is wrong.

Fixture schemas are built by mutating a decoded copy of the shipped YAML in
``tmp_path``.  Deriving them from the real schema rather than hand-writing
minimal ones means each fixture differs from a valid schema in exactly the one
way under test, so a passing rejection test cannot be passing for an unrelated
reason.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.services.facility_knowledge.ontology_compiler.errors import OntologyCompileError
from osprey.services.facility_knowledge.ontology_compiler.loader import load_schema
from osprey.services.facility_knowledge.ontology_compiler.payload import schema_to_payload
from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    OntologyMapError,
    load_demo_ontology,
    parse_ontology,
)

#: The authored schema the shipped ``demo_ontology.json`` is compiled from.
SCHEMA_PATH = Path(
    str(files("osprey.services.facility_knowledge.ttl_generator").joinpath("demo_ontology.yaml"))
)


def _variant(tmp_path: Path, mutate: Callable[[dict[str, Any]], None], name: str) -> Path:
    """Write a copy of the shipped schema with one deliberate defect applied.

    The shipped file is decoded, mutated in memory and re-dumped into
    *tmp_path*; it is never modified in place.

    Args:
        tmp_path: Per-test temporary directory.
        mutate: Applied to the decoded schema to introduce the defect.
        name: Stem for the written file, so a failing test names its fixture.

    Returns:
        Path to the written variant schema.
    """
    document = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    mutate(document)
    target = tmp_path / f"{name}.yaml"
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


def _compile_variant(tmp_path: Path, mutate: Callable[[dict[str, Any]], None], name: str) -> Any:
    """Load a defective variant of the shipped schema and translate it."""
    source = _variant(tmp_path, mutate, name)
    return schema_to_payload(load_schema(source), source)


@pytest.fixture(scope="module")
def shipped_payload() -> dict[str, object]:
    """The payload compiled from the shipped schema, built once for the module."""
    return schema_to_payload(load_schema(SCHEMA_PATH), SCHEMA_PATH)


class TestShippedSchema:
    """The shipped YAML must compile to the table the shipped JSON already holds."""

    def test_payload_parses_into_the_shipped_table(
        self, shipped_payload: dict[str, object]
    ) -> None:
        """Compiled payload and committed artifact describe the same ontology.

        ``source_path`` is excluded deliberately: it records where a table was
        read from, and the two tables come from different files by design.
        """
        compiled = parse_ontology(shipped_payload, source_path=SCHEMA_PATH)
        shipped = load_demo_ontology()

        assert dict(compiled.family_to_class) == dict(shipped.family_to_class)
        assert sorted(compiled.classes) == sorted(shipped.classes)
        for name, expected in shipped.classes.items():
            actual = compiled.classes[name]
            assert (actual.name, actual.parent, actual.alt_labels) == (
                expected.name,
                expected.parent,
                expected.alt_labels,
            )

    def test_root_is_the_sole_parentless_class(self, shipped_payload: dict[str, object]) -> None:
        """One class has no ``is_a``, and the payload names it as ``root``."""
        assert shipped_payload["root"] == "AcceleratorDevice"

    def test_alt_labels_are_sorted_and_deduplicated(
        self, shipped_payload: dict[str, object]
    ) -> None:
        """``altLabels`` is a sorted, duplicate-free list, whatever the author wrote."""
        classes = shipped_payload["classes"]
        assert isinstance(classes, dict)
        for name, entry in classes.items():
            labels = entry["altLabels"]
            assert labels == sorted(set(labels)), name

    def test_hyphenated_family_token_survives_verbatim(
        self, shipped_payload: dict[str, object]
    ) -> None:
        """FAMILY tokens keep the channel database's own spelling.

        ``ION-PUMP`` is the one token whose spelling a schema language could
        plausibly mangle — LinkML permissible values are commonly written as
        identifiers — so it is worth pinning on its own.
        """
        families = shipped_payload["family_to_class"]
        assert isinstance(families, dict)
        assert families["ION-PUMP"] == "Pump"

    def test_translation_writes_nothing(self, tmp_path: Path) -> None:
        """The stage is pure: compiling touches no file next to the schema."""
        source = _variant(tmp_path, lambda document: None, "unchanged")
        before = sorted(path.name for path in tmp_path.iterdir())
        schema_to_payload(load_schema(source), source)
        assert sorted(path.name for path in tmp_path.iterdir()) == before


class TestClassRejections:
    """Every class-level rule fires, and names the class that broke it."""

    def test_missing_class_uri_is_rejected(self, tmp_path: Path) -> None:
        """A class with no ``class_uri`` has no IRI the emitted graph could use."""

        def mutate(document: dict[str, Any]) -> None:
            del document["classes"]["Dipole"]["class_uri"]

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "missing_class_uri")
        message = str(excinfo.value)
        assert "Dipole" in message
        assert "class_uri" in message

    def test_undeclared_prefix_is_rejected(self, tmp_path: Path) -> None:
        """An unknown prefix must fail loudly.

        ``expand_curie`` returns an unexpandable CURIE unchanged instead of
        raising, so this case is only caught by comparing the expansion to the
        IRI the class should have had.  Without that comparison the misspelled
        prefix would reach the artifact as a valid-looking class.
        """

        def mutate(document: dict[str, Any]) -> None:
            document["classes"]["Dipole"]["class_uri"] = "narad_semm:Dipole"

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "undeclared_prefix")
        message = str(excinfo.value)
        assert "Dipole" in message
        assert "narad_semm:Dipole" in message

    def test_class_uri_outside_narad_sem_is_rejected(self, tmp_path: Path) -> None:
        """A CURIE that expands into another namespace is not this vocabulary."""

        def mutate(document: dict[str, Any]) -> None:
            document["classes"]["Dipole"]["class_uri"] = "linkml:Dipole"

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "foreign_namespace")
        message = str(excinfo.value)
        assert "Dipole" in message
        assert "https://w3id.org/linkml/Dipole" in message

    def test_class_uri_with_mismatched_local_name_is_rejected(self, tmp_path: Path) -> None:
        """Right namespace, wrong local name still breaks the table's IRI contract."""

        def mutate(document: dict[str, Any]) -> None:
            document["classes"]["Dipole"]["class_uri"] = "narad_sem:BendingMagnet"

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "mismatched_local_name")
        assert "Dipole" in str(excinfo.value)

    def test_mixins_are_rejected(self, tmp_path: Path) -> None:
        """``mixins`` expresses inheritance the single-parent table cannot carry."""

        def mutate(document: dict[str, Any]) -> None:
            document["classes"]["Dipole"]["mixins"] = ["Magnet"]

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "with_mixins")
        message = str(excinfo.value)
        assert "Dipole" in message
        assert "mixins" in message

    def test_attributes_are_rejected(self, tmp_path: Path) -> None:
        """Any other populated construct is refused, not silently dropped."""

        def mutate(document: dict[str, Any]) -> None:
            document["classes"]["Dipole"]["attributes"] = {"field_strength": {"range": "string"}}

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "with_attributes")
        message = str(excinfo.value)
        assert "Dipole" in message
        assert "attributes" in message


class TestFamilyRejections:
    """Every enum-level rule fires, and names the enum or family token at fault."""

    def test_missing_device_family_enum_lists_the_enums_present(self, tmp_path: Path) -> None:
        """The message must say what the schema *does* declare.

        A schema whose family enum is misnamed looks correct at a glance; the
        list of enums present is what turns the error into a diagnosis.
        """

        def mutate(document: dict[str, Any]) -> None:
            document["enums"]["DeviceFamilies"] = document["enums"].pop("DeviceFamily")

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "misnamed_enum")
        message = str(excinfo.value)
        assert "DeviceFamily" in message
        assert "DeviceFamilies" in message

    def test_empty_permissible_values_are_rejected(self, tmp_path: Path) -> None:
        """An enum with no values maps no families at all."""

        def mutate(document: dict[str, Any]) -> None:
            document["enums"]["DeviceFamily"]["permissible_values"] = {}

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "empty_enum")
        assert "DeviceFamily" in str(excinfo.value)

    def test_value_without_meaning_is_rejected(self, tmp_path: Path) -> None:
        """A FAMILY token with no ``meaning`` maps to nothing."""

        def mutate(document: dict[str, Any]) -> None:
            document["enums"]["DeviceFamily"]["permissible_values"]["BPM"] = {
                "description": "A beam position monitor."
            }

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "meaningless_value")
        message = str(excinfo.value)
        assert "BPM" in message
        assert "meaning" in message

    def test_meaning_outside_narad_sem_is_rejected(self, tmp_path: Path) -> None:
        """A family may only map to a class in the vocabulary this table describes."""

        def mutate(document: dict[str, Any]) -> None:
            document["enums"]["DeviceFamily"]["permissible_values"]["BPM"] = {
                "meaning": "linkml:BeamPositionMonitor"
            }

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "foreign_meaning")
        message = str(excinfo.value)
        assert "BPM" in message
        assert "https://w3id.org/linkml/BeamPositionMonitor" in message

    def test_meaning_with_undeclared_prefix_is_rejected(self, tmp_path: Path) -> None:
        """The soft-failing expansion is caught for values, as it is for classes."""

        def mutate(document: dict[str, Any]) -> None:
            document["enums"]["DeviceFamily"]["permissible_values"]["BPM"] = {
                "meaning": "narad_semm:BeamPositionMonitor"
            }

        with pytest.raises(OntologyCompileError) as excinfo:
            _compile_variant(tmp_path, mutate, "unprefixed_meaning")
        assert "BPM" in str(excinfo.value)


class TestRootDerivation:
    """``root`` is emitted only when the schema settles the question by itself."""

    def test_root_is_omitted_when_two_classes_are_parentless(self, tmp_path: Path) -> None:
        """An ambiguous hierarchy is handed to ``parse_ontology`` to report.

        The compiler could complain here, but ``parse_ontology`` already names
        every candidate it found — the message an author actually needs — and
        it is the same message a hand-written JSON table produces for the same
        mistake.  Emitting no ``root`` is what routes the complaint there.
        """

        def mutate(document: dict[str, Any]) -> None:
            del document["classes"]["Magnet"]["is_a"]

        payload = _compile_variant(tmp_path, mutate, "two_roots")
        assert "root" not in payload

        with pytest.raises(OntologyMapError) as excinfo:
            parse_ontology(payload, source_path=SCHEMA_PATH)
        message = str(excinfo.value)
        assert "AcceleratorDevice" in message
        assert "Magnet" in message
