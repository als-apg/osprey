"""Tests for ``osprey knowledge compile-ontology``.

Covers:
- ``compile-ontology --help`` reads as operator documentation, names both
  positionals, and names ``build-ttl --ontology`` as the verb to run next.
- A run against the shipped demo schema writes a table the runtime's own
  ``load_ontology`` reads back, and reports the counts it wrote.
- ``--check`` agrees with a table this verb just wrote, and writes nothing.
- ``--check`` against a table one synonym short fails, names the class and the
  synonym, and carries the command that regenerates it.
- ``--check`` against a path holding nothing says so in its own words, rather
  than reporting a difference it never managed to compute.
- ``--check`` against bytes that are not UTF-8 at all says so, rather than
  escaping as a decoding error nobody catches.
- ``--check`` reads a read-only artifact -- the CI and pre-commit case the flag
  exists for -- rather than refusing it for not being writable, and reports an
  I/O failure as a failure to *read*, since it never writes.
- A successful compile replaces OUTPUT and leaves no temporary file behind.
- A schema with a misspelled slot is one legible line and a non-zero exit.
- A schema that is valid LinkML but describes an ontology that does not stand
  up is reported in the schema's own vocabulary, not the table's.
- Every failure above exits non-zero without a traceback.
- An absent ``linkml_runtime`` names the extra to install rather than failing
  inside the compiler.

The shipped ``demo_ontology.yaml`` is the source throughout, read-only: it is
the schema the project actually authors, so a test that compiles it is testing
the real thing rather than a fixture that resembles it. Every file written goes
to ``tmp_path`` -- in particular the shipped ``demo_ontology.json`` is never the
OUTPUT of a test run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from click.testing import CliRunner

from osprey.cli.knowledge_cmd import knowledge

#: The schema shipped with OSPREY, and the one this verb exists to compile.
DEMO_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "services"
    / "facility_knowledge"
    / "ttl_generator"
    / "demo_ontology.yaml"
)

#: What the demo machine's vocabulary holds. Asserted rather than derived so a
#: schema edit that silently drops half the table is visible here too.
DEMO_CLASS_COUNT = 19
DEMO_FAMILY_COUNT = 19


def _flat(result: Any) -> str:
    """Rich wraps at the console width, so compare on whitespace-collapsed text."""
    return " ".join(result.output.split())


def _compile(source: Path, output: Path, *extra: str) -> Any:
    """Invoke the verb and return the click result."""
    return CliRunner().invoke(knowledge, ["compile-ontology", str(source), str(output), *extra])


@pytest.fixture
def compiled_table(tmp_path: Path) -> Path:
    """A freshly compiled table for the shipped demo schema, inside tmp_path."""
    output = tmp_path / "demo_ontology.json"
    result = _compile(DEMO_SCHEMA, output)
    assert result.exit_code == 0, result.output
    return output


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_compile_ontology_help_documents_both_paths_and_the_next_verb() -> None:
    """The listing shows the verb, and its own help says what it reads, writes and precedes."""
    runner = CliRunner()

    listing = runner.invoke(knowledge, ["--help"])
    assert listing.exit_code == 0, listing.output
    assert "compile-ontology" in listing.output

    detail = runner.invoke(knowledge, ["compile-ontology", "--help"])
    assert detail.exit_code == 0, detail.output
    flat = _flat(detail)
    assert "SOURCE is the LinkML schema to compile" in flat
    assert "OUTPUT is the JSON table to write" in flat
    assert "--check" in flat
    # The verb this one stands upstream of, named with the flag that reads the
    # file it just wrote.
    assert "osprey knowledge build-ttl" in flat
    assert "--ontology" in flat
    # Written for operators, not for the framework's authors.
    assert "Claude Code" not in detail.output


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_compile_ontology_writes_a_table_the_runtime_reads(tmp_path: Path) -> None:
    """The compiled artifact loads through the same function build-ttl uses."""
    from osprey.services.facility_knowledge.ttl_generator.ontology_map import load_ontology

    output = tmp_path / "demo_ontology.json"

    result = _compile(DEMO_SCHEMA, output)

    assert result.exit_code == 0, result.output
    assert output.is_file()

    table = load_ontology(output)
    assert len(table.classes) == DEMO_CLASS_COUNT
    assert len(table.family_to_class) == DEMO_FAMILY_COUNT

    flat = _flat(result)
    assert f"{DEMO_CLASS_COUNT} classes, {DEMO_FAMILY_COUNT} families" in flat
    # The name only: rich wraps a long tmp_path across lines.
    assert output.name in flat


def test_compile_ontology_marks_the_artifact_generated(compiled_table: Path) -> None:
    """The written table says it is generated and names the schema it came from."""
    document = json.loads(compiled_table.read_text(encoding="utf-8"))
    assert "compile-ontology" in document["_generated"]
    assert DEMO_SCHEMA.name in document["_generated"]


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def test_check_agrees_with_a_table_this_verb_just_wrote(compiled_table: Path) -> None:
    """A fresh compile is by definition up to date, and --check says so."""
    before = compiled_table.read_bytes()

    result = _compile(DEMO_SCHEMA, compiled_table, "--check")

    assert result.exit_code == 0, result.output
    assert "up to date" in _flat(result)
    # --check never writes.
    assert compiled_table.read_bytes() == before


def test_check_names_the_class_and_synonym_that_drifted(compiled_table: Path) -> None:
    """A table one altLabel short fails, naming the element rather than 'files differ'."""
    document = json.loads(compiled_table.read_text(encoding="utf-8"))
    labels = document["classes"]["AcceleratingCavity"]["altLabels"]
    assert "cavity" in labels, labels
    labels.remove("cavity")
    mutated = compiled_table.parent / "mutated.json"
    mutated.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    before = mutated.read_bytes()

    result = _compile(DEMO_SCHEMA, mutated, "--check")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert "AcceleratingCavity" in flat
    assert "cavity" in flat
    # The report ends with its own remedy, so a CI log needs nothing added.
    assert "Re-run `osprey knowledge compile-ontology" in flat
    # A failing check repairs nothing.
    assert mutated.read_bytes() == before


def test_check_without_an_artifact_says_there_is_nothing_to_check(tmp_path: Path) -> None:
    """A missing OUTPUT is its own message, not a diff against a file that is not there."""
    missing = tmp_path / "never-written.json"

    result = _compile(DEMO_SCHEMA, missing, "--check")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert missing.name in flat
    assert "compile-ontology" in flat
    assert not missing.exists()


def test_check_against_bytes_that_are_not_text_says_so(tmp_path: Path) -> None:
    """OUTPUT that is not UTF-8 is a named failure, not an escaping decode error."""
    output = tmp_path / "not-text.json"
    output.write_bytes(b"\xff\xfe{\x00")
    before = output.read_bytes()

    result = _compile(DEMO_SCHEMA, output, "--check")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert output.name in flat
    assert "UTF-8" in flat
    # The remedy travels with the complaint, as it does for a missing artifact.
    assert "compile-ontology" in flat
    # A failing check still writes nothing.
    assert output.read_bytes() == before


def test_check_reads_an_artifact_that_is_read_only(compiled_table: Path) -> None:
    """--check only reads, so a checked-out read-only artifact is checkable.

    This is the case the flag exists for: a CI checkout or a pre-commit hook
    where the tree is not writable. A verb that refused it for not being
    writable would fail on exactly the runs that need it most.
    """
    compiled_table.chmod(0o444)
    if os.access(compiled_table, os.W_OK):  # pragma: no cover - root, or a permissionless FS
        pytest.skip("the chmod did not take effect (running as root?)")

    result = _compile(DEMO_SCHEMA, compiled_table, "--check")

    assert result.exit_code == 0, result.output
    assert "up to date" in _flat(result)


def test_check_that_cannot_read_output_says_read_not_write(compiled_table: Path) -> None:
    """An I/O failure under --check is reported as a failure to read.

    The flag writes nothing, so "Cannot write OUTPUT" would name an operation
    the run never attempts and send the reader looking for a permissions
    problem that is not there. The error is raised from inside ``check_artifact``
    because that is where the read happens; click rejects an OUTPUT it cannot
    even stat long before then.
    """
    with mock.patch(
        "osprey.services.facility_knowledge.ontology_compiler.check_artifact",
        side_effect=OSError("Input/output error"),
    ):
        result = _compile(DEMO_SCHEMA, compiled_table, "--check")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert "Cannot read" in flat
    assert "Input/output error" in flat
    assert "Cannot write" not in flat


def test_compile_leaves_no_temporary_file_beside_the_output(tmp_path: Path) -> None:
    """The write goes through a sibling temporary file, and it does not survive the run."""
    output = tmp_path / "demo_ontology.json"

    result = _compile(DEMO_SCHEMA, output)

    assert result.exit_code == 0, result.output
    assert sorted(child.name for child in tmp_path.iterdir()) == [output.name]


def test_compile_over_an_existing_table_replaces_it_whole(compiled_table: Path) -> None:
    """A second compile of the same schema leaves the same bytes and no debris."""
    before = compiled_table.read_bytes()

    result = _compile(DEMO_SCHEMA, compiled_table)

    assert result.exit_code == 0, result.output
    assert compiled_table.read_bytes() == before
    assert sorted(child.name for child in compiled_table.parent.iterdir()) == [compiled_table.name]


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_compile_ontology_rejects_a_misspelled_slot_in_one_line(tmp_path: Path) -> None:
    """A schema LinkML cannot load is a legible line, not a TypeError traceback."""
    schema = tmp_path / "typo.yaml"
    schema.write_text(
        "id: https://narad.example.org/schema/shared_semantics/typo\n"
        "name: typo\n"
        "prefixes:\n"
        "  narad_sem: https://narad.example.org/schema/shared_semantics/\n"
        "  linkml: https://w3id.org/linkml/\n"
        "default_prefix: narad_sem\n"
        "default_range: string\n"
        "imports:\n"
        "  - linkml:types\n"
        "classes:\n"
        "  AcceleratorDevice:\n"
        "    class_uri: narad_sem:AcceleratorDevice\n"
        "    descriptoin: Root of the tiny hierarchy.\n"
        "enums:\n"
        "  DeviceFamily:\n"
        "    permissible_values:\n"
        "      DEV:\n"
        "        meaning: narad_sem:AcceleratorDevice\n",
        encoding="utf-8",
    )
    output = tmp_path / "typo.json"

    result = _compile(schema, output)

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert schema.name in flat
    assert "descriptoin" in flat
    # Nothing is written when the schema does not compile.
    assert not output.exists()


def test_compile_ontology_rejects_an_ontology_that_does_not_stand_up(tmp_path: Path) -> None:
    """Valid LinkML describing an impossible table is explained in the schema's words.

    The shipped schema with one ``is_a`` line removed is still a schema LinkML
    loads happily -- the fault is that it now declares two parentless classes,
    which only the table's own validation can see. The message therefore has to
    reach back to ``is_a``, since that is the line the reader has to edit.
    """
    schema = tmp_path / "two-roots.yaml"
    text = DEMO_SCHEMA.read_text(encoding="utf-8")
    rooted = "  Magnet:\n    class_uri: narad_sem:Magnet\n    is_a: AcceleratorDevice\n"
    assert text.count(rooted) == 1, "the demo schema no longer parents Magnet as expected"
    schema.write_text(
        text.replace(rooted, "  Magnet:\n    class_uri: narad_sem:Magnet\n"), encoding="utf-8"
    )
    output = tmp_path / "two-roots.json"

    result = _compile(schema, output)

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert "does not stand up" in flat
    # The compiler translates the table's vocabulary back into the schema's.
    assert "is_a" in flat
    assert "Magnet" in flat
    # A schema that does not validate writes nothing at all.
    assert not output.exists()


def test_compile_ontology_without_linkml_names_the_extra(tmp_path: Path) -> None:
    """An absent linkml_runtime asks for the extra rather than failing inside the compiler."""
    output = tmp_path / "no-linkml.json"

    with mock.patch.dict(sys.modules, {"linkml_runtime": None}):
        result = _compile(DEMO_SCHEMA, output)

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert "knowledge" in flat
    assert "osprey-framework[knowledge]" in flat
    assert not output.exists()
