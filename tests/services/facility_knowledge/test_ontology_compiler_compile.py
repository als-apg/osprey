"""Tests for the ontology compiler's drive-the-stages-and-validate step.

Three things are pinned here, and none of them is a rule ``compile.py`` owns —
which is the point.  This stage writes no checks of its own; it composes three
functions that already have them, so what has to be proven is the *composition*.

**The shipped schema compiles to today's table.**  ``compile_schema`` on the
authored YAML must produce, family for family and class for class, the map
:func:`load_demo_ontology` reads from the committed JSON.  The payload stage
already proves its translation reproduces that table; what this adds is that
running the translation through ``parse_ontology`` — the runtime's own reader —
leaves it intact.  An artifact rendered from this result therefore loads.

**Delegation order is observable.**  Which of the two exception types a caller
sees is the contract, not an accident: a schema the loader rejects must never
reach ``parse_ontology``, so it must surface as :class:`OntologyCompileError`.
If the stages were reordered, or a failure wrapped one level too far out, an
authoring mistake in the YAML would be reported as a malformed *table* — a
message pointing at the generated artifact for a defect in the source.  Both a
missing file and a misspelled slot are tested, because they fail through
different paths inside the loader.

**The translation of a table complaint keeps the original whole.**  Removing
``is_a`` from ``Magnet`` leaves valid LinkML describing an ontology with two
roots.  The re-raise must keep the type, prepend the preamble that names the
LinkML spellings, and chain the original — so the operator gets the vocabulary
they are reading in front of the complaint, verbatim, and a debugger still has
the untranslated exception.

Fixture schemas are mutated copies of the shipped YAML in ``tmp_path``, as in
``test_ontology_compiler_payload.py``: each differs from a valid schema in
exactly the one way under test, so a rejection test cannot pass for an
unrelated reason.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.services.facility_knowledge.ontology_compiler.compile import (
    LINKML_PREAMBLE,
    CompiledOntology,
    compile_schema,
)
from osprey.services.facility_knowledge.ontology_compiler.errors import OntologyCompileError
from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    OntologyMapError,
    load_demo_ontology,
)

#: The authored schema the shipped ``demo_ontology.json`` is compiled from.
SCHEMA_PATH = Path(
    str(files("osprey.services.facility_knowledge.ttl_generator").joinpath("demo_ontology.yaml"))
)

#: The sentence ``parse_ontology`` emits for a table without exactly one root.
#: Pinned verbatim so this suite fails if the re-raise ever paraphrases the
#: original instead of carrying it through.
PARENTLESS_COMPLAINT = "must declare exactly one parentless class"


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


@pytest.fixture(scope="module")
def shipped() -> CompiledOntology:
    """The shipped schema compiled once for the whole module."""
    return compile_schema(SCHEMA_PATH)


class TestShippedSchema:
    """Compiling the authored schema reproduces the committed table."""

    def test_family_map_matches_the_shipped_artifact(self, shipped: CompiledOntology) -> None:
        """Every FAMILY token maps to the class the committed JSON maps it to."""
        assert dict(shipped.table.family_to_class) == dict(load_demo_ontology().family_to_class)

    def test_classes_match_the_shipped_artifact(self, shipped: CompiledOntology) -> None:
        """Class for class, the hierarchy and its labels are unchanged.

        Compared field-wise rather than by object equality because
        :class:`OntologyMap` carries the file it was read from, and these two
        tables come from different files by design.
        """
        expected_classes = load_demo_ontology().classes

        assert sorted(shipped.table.classes) == sorted(expected_classes)
        for name, expected in expected_classes.items():
            actual = shipped.table.classes[name]
            assert (actual.name, actual.parent, actual.alt_labels) == (
                expected.name,
                expected.parent,
                expected.alt_labels,
            )

    def test_result_carries_the_source_and_the_renderable_payload(
        self, shipped: CompiledOntology
    ) -> None:
        """The result holds both forms a caller needs, plus the file they came from."""
        assert shipped.source == SCHEMA_PATH
        assert shipped.payload["root"] == "AcceleratorDevice"
        assert sorted(shipped.payload) == ["classes", "family_to_class", "root"]

    def test_result_is_frozen(self, shipped: CompiledOntology) -> None:
        """A validated result cannot be edited between validation and rendering."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            shipped.source = Path("elsewhere.yaml")  # type: ignore[misc]


class TestLoaderFailuresKeepTheirType:
    """A schema fault stops at the loader; it is never reported as a table fault.

    ``OntologyMapError`` subclasses ``ValueError`` and so does
    ``OntologyCompileError``, so each test asserts the *absence* of the wrong
    type explicitly — catching a bare ``ValueError`` would pass either way.
    """

    def test_missing_file_is_a_compile_error(self, tmp_path: Path) -> None:
        """A path that does not exist fails in the loader, naming the file."""
        missing = tmp_path / "absent_ontology.yaml"

        with pytest.raises(OntologyCompileError) as excinfo:
            compile_schema(missing)
        assert not isinstance(excinfo.value, OntologyMapError)
        assert str(excinfo.value).startswith("absent_ontology.yaml: ")

    def test_misspelled_slot_is_a_compile_error(self, tmp_path: Path) -> None:
        """LinkML's own rejection of an unknown slot surfaces unchanged.

        This is the failure the delegation order exists for: the defect is a
        typo in the YAML, and reporting it against the compiled table would
        point the operator at a file they never wrote.
        """

        def mutate(document: dict[str, Any]) -> None:
            document["classes"]["Dipole"]["aliasees"] = ["bend"]

        source = _variant(tmp_path, mutate, "misspelled_slot")

        with pytest.raises(OntologyCompileError) as excinfo:
            compile_schema(source)
        assert not isinstance(excinfo.value, OntologyMapError)
        assert "aliasees" in str(excinfo.value)


class TestTableFailuresAreTranslated:
    """A valid schema describing an invalid table keeps its type and gains a preamble."""

    @staticmethod
    def _drop_magnet_parent(document: dict[str, Any]) -> None:
        """Remove ``Magnet``'s ``is_a``, leaving the schema with two roots."""
        del document["classes"]["Magnet"]["is_a"]

    def test_second_root_raises_an_ontology_map_error(self, tmp_path: Path) -> None:
        """The type is unchanged, so a caller's existing ``except`` still fires."""
        source = _variant(tmp_path, self._drop_magnet_parent, "magnet_without_parent")

        with pytest.raises(OntologyMapError):
            compile_schema(source)

    def test_message_translates_into_linkml_spelling(self, tmp_path: Path) -> None:
        """The preamble leads and names ``is_a``; the original follows verbatim."""
        source = _variant(tmp_path, self._drop_magnet_parent, "magnet_without_parent")

        with pytest.raises(OntologyMapError) as excinfo:
            compile_schema(source)
        message = str(excinfo.value)
        assert message.startswith(LINKML_PREAMBLE)
        assert "`is_a`" in message
        assert PARENTLESS_COMPLAINT in message
        assert "'Magnet'" in message

    def test_original_error_is_chained(self, tmp_path: Path) -> None:
        """The untranslated exception stays reachable as ``__cause__``."""
        source = _variant(tmp_path, self._drop_magnet_parent, "magnet_without_parent")

        with pytest.raises(OntologyMapError) as excinfo:
            compile_schema(source)
        cause = excinfo.value.__cause__
        assert isinstance(cause, OntologyMapError)
        assert str(cause) == str(excinfo.value).removeprefix(LINKML_PREAMBLE)
        assert not str(cause).startswith(LINKML_PREAMBLE)
