"""The regen-consistency guard between the LinkML schema and its compiled table.

``demo_ontology.json`` is generated from ``demo_ontology.yaml`` and *also*
committed, so the two can silently part ways: someone edits the schema and
forgets to recompile, or edits the artifact by hand despite its ``_generated``
header.  Neither file complains — both stay valid, both keep loading — and the
graph keeps being built from a table that no longer says what its source says.

This module is the gate that notices.  It asserts four things about the shipped
pair, from four angles, so a failure points at *which* kind of drift happened:

1. **Recompiling changes nothing.**  ``check_artifact`` renders the schema in
   memory and compares it to the committed bytes; an empty result is the whole
   claim that the artifact is up to date.
2. **The runtime reads what the compiler wrote.**  The table
   ``compile_schema`` validates and the table ``load_demo_ontology`` reads back
   off disk must agree field for field, which closes the loop that (1) only
   checks at the level of text.
3. **The check has teeth.**  A schema mutated in exactly one place must be
   reported against the committed artifact, naming the class and the label.  A
   guard that passes on a clean tree and also passes on a dirty one guards
   nothing.
4. **The artifact declares its provenance.**  The committed ``_generated``
   value must be the header the renderer stamps, naming ``demo_ontology.yaml``.

**If this file fails, do not edit the JSON.**  Re-run the compiler::

    uv run osprey knowledge compile-ontology \\
        src/osprey/services/facility_knowledge/ttl_generator/demo_ontology.yaml \\
        src/osprey/services/facility_knowledge/ttl_generator/demo_ontology.json

then re-read the resulting diff: it is the drift, stated in the compiler's own
terms.  Only a failure of test (3) means something else — that the check itself
stopped detecting a difference it used to detect.

Every test here is **read-only on the package resources**.  The schema and the
artifact are opened, never written; mutated copies live under ``tmp_path``, and
``load_demo_ontology``'s cache is never cleared.  Both properties matter under
``pytest -n``: the shipped files are shared by every worker, so a test that
wrote one — or invalidated a cache another worker was mid-read on — would make
its neighbours fail instead of itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from osprey.services.facility_knowledge.ontology_compiler.check import check_artifact
from osprey.services.facility_knowledge.ontology_compiler.compile import compile_schema
from osprey.services.facility_knowledge.ontology_compiler.render import GENERATED_HEADER
from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    DEMO_ONTOLOGY_FILENAME,
    load_demo_ontology,
)

#: Package both halves of the shipped pair live in.
_PACKAGE = "osprey.services.facility_knowledge.ttl_generator"

#: Name of the authored schema.  Also the name the ``_generated`` header
#: carries, since the renderer stamps :attr:`~pathlib.Path.name` only.
SCHEMA_FILENAME = "demo_ontology.yaml"

#: The authored LinkML schema, resolved the way the runtime resolves its own
#: package data rather than by walking up from ``__file__``.
SCHEMA_PATH = Path(str(files(_PACKAGE).joinpath(SCHEMA_FILENAME)))

#: The committed artifact compiled from :data:`SCHEMA_PATH`.
ARTIFACT_PATH = Path(str(files(_PACKAGE).joinpath(DEMO_ONTOLOGY_FILENAME)))

#: The class and synonym test (3) renames.  Any class with a synonym would do;
#: this one is picked because it carries several, so dropping one leaves the
#: class itself present and only the label set differing.
DRIFT_CLASS = "AcceleratingCavity"
DRIFT_OLD_LABEL = "rf cavity"
DRIFT_NEW_LABEL = "rf resonator"


def _schema_copy(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Write a mutated copy of the shipped schema under *tmp_path*.

    The copy keeps the shipped file *name*, so the ``_generated`` header a
    compile of it renders is identical to the committed one and the only
    differences a check reports are the ones *mutate* introduced.  The shipped
    schema is read and never written.

    Args:
        tmp_path: Per-test temporary directory.
        mutate: Applied to the parsed schema document in memory.

    Returns:
        Path to the written copy.
    """
    document = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    mutate(document)
    target = tmp_path / SCHEMA_FILENAME
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


class TestShippedPairIsInSync:
    """The committed artifact is what the committed schema compiles to."""

    def test_recompiling_the_schema_reproduces_the_artifact(self) -> None:
        """``--check`` on the shipped pair reports nothing at all.

        This is the assertion CI depends on, and the one that fails when
        someone edits the schema without re-running the compiler.
        """
        assert check_artifact(SCHEMA_PATH, ARTIFACT_PATH) == ()

    def test_compiled_table_matches_the_table_the_runtime_loads(self) -> None:
        """The compiler's table and the runtime's table agree field for field.

        ``source_path`` is deliberately excluded from the comparison: the two
        tables are loaded from *different files by design* — the compiler's
        from ``demo_ontology.yaml``, the runtime's from ``demo_ontology.json``
        — so that field must differ for the pipeline to be working at all.
        Every other field is the ontology itself, and those must match.

        Comparing the parsed tables rather than the bytes is what makes this
        test say something the byte check above does not: it proves the JSON
        the runtime reads back *parses into* the ontology the schema declares,
        not merely that its text is what the renderer would emit.
        """
        compiled = compile_schema(SCHEMA_PATH).table
        loaded = load_demo_ontology()

        assert dict(compiled.family_to_class) == dict(loaded.family_to_class)
        assert dict(compiled.classes) == dict(loaded.classes)
        assert compiled.source_path != loaded.source_path

    def test_artifact_declares_the_schema_it_was_generated_from(self) -> None:
        """The committed ``_generated`` header names ``demo_ontology.yaml``.

        The header is what tells an operator who opens the artifact not to edit
        it, and which file to edit instead, so it is pinned independently of
        the byte check — a header naming the wrong source would be wrong advice
        even in a tree where everything else is in sync.
        """
        document = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

        assert document["_generated"] == GENERATED_HEADER.format(name=SCHEMA_FILENAME)


class TestTheGuardDetectsDrift:
    """A schema that moved away from the artifact is reported, by element."""

    def test_renamed_synonym_is_reported_against_the_committed_artifact(
        self, tmp_path: Path
    ) -> None:
        """Renaming one alias names that class and both labels, and nothing else.

        The copy carries the shipped file name, so no ``_generated`` difference
        can stand in for the finding this test is looking for: every line comes
        from the renamed label alone.
        """

        def rename_alias(document: dict[str, Any]) -> None:
            aliases = document["classes"][DRIFT_CLASS]["aliases"]
            aliases[aliases.index(DRIFT_OLD_LABEL)] = DRIFT_NEW_LABEL

        drifted = _schema_copy(tmp_path, rename_alias)

        lines = check_artifact(drifted, ARTIFACT_PATH)

        assert lines != ()
        element_lines = [line for line in lines if line.startswith("class ")]
        assert element_lines == [
            f"class {DRIFT_CLASS}: altLabel {DRIFT_NEW_LABEL!r} missing in OUTPUT.",
            f"class {DRIFT_CLASS}: altLabel {DRIFT_OLD_LABEL!r} in OUTPUT but not in the schema.",
        ]
        assert lines[-1].startswith("Re-run `osprey knowledge compile-ontology")

    def test_unmutated_copy_of_the_schema_still_matches(self, tmp_path: Path) -> None:
        """A copy round-tripped through YAML but not mutated reports nothing.

        Without this, the drift test above could be passing on the round-trip
        — key order, block style, folded scalars — rather than on the rename,
        and would keep passing if the rename stopped mattering.
        """
        unchanged = _schema_copy(tmp_path, lambda document: None)

        assert check_artifact(unchanged, ARTIFACT_PATH) == ()


class TestTheGuardWritesNothing:
    """The shipped package resources are read, never modified."""

    def test_checking_and_compiling_leave_the_shipped_pair_untouched(self, tmp_path: Path) -> None:
        """Every entry point this module uses is read-only on the package data.

        The guard runs in CI and in pre-commit hooks and in a ``-n`` worker
        pool, all places where quietly rewriting a shared file would turn a red
        gate green or break a neighbouring test instead of this one.
        """
        before = (ARTIFACT_PATH.read_bytes(), SCHEMA_PATH.read_bytes())

        check_artifact(SCHEMA_PATH, ARTIFACT_PATH)
        compile_schema(SCHEMA_PATH)
        check_artifact(_schema_copy(tmp_path, lambda document: None), ARTIFACT_PATH)

        assert (ARTIFACT_PATH.read_bytes(), SCHEMA_PATH.read_bytes()) == before
