"""Tests for the ontology compiler's artifact check.

``check_artifact`` is what stands between a schema and the committed table
drifting apart without anyone noticing, so what these tests pin is not the
prose of any one line but the two properties an operator and CI depend on.

**Agreement is byte-level, disagreement is element-level.**  A rendered copy of
the shipped schema must come back clean, and a copy mutated in exactly one
place must come back with a line naming *that* element — the class whose parent
moved, the class and the label that vanished, the family that was retargeted.
A diff that only said "files differ" would send an operator to ``git diff``;
one that named the wrong element would send them somewhere worse.  Each fixture
therefore differs from a clean render in exactly one field, re-dumped with the
renderer's own formatting so no whitespace difference can produce the line the
test is looking for.

**The check never writes and never repairs.**  ``--check`` runs in CI and in
pre-commit hooks, where silently rewriting the artifact would turn a failing
gate into a passing one.  The bytes and mtime of OUTPUT are compared across the
call, and an unreadable OUTPUT surfaces as :class:`OSError` rather than as a
diff line, because "the file is missing" is not a disagreement about content.

Every non-empty result ends with the line naming the verb to re-run: the check
tells an operator what to do about what it found, in every path including the
one where OUTPUT is not JSON at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from osprey.services.facility_knowledge.ontology_compiler.check import check_artifact
from osprey.services.facility_knowledge.ontology_compiler.compile import compile_schema
from osprey.services.facility_knowledge.ontology_compiler.render import render_json

#: The authored schema the shipped ``demo_ontology.json`` is compiled from.
SCHEMA_PATH = Path(
    str(files("osprey.services.facility_knowledge.ttl_generator").joinpath("demo_ontology.yaml"))
)


def _rendered() -> str:
    """Return the artifact text a clean compile of the shipped schema produces."""
    return render_json(compile_schema(SCHEMA_PATH).payload, SCHEMA_PATH)


def _write(path: Path, text: str) -> Path:
    """Write *text* to *path* and return the path."""
    path.write_text(text, encoding="utf-8")
    return path


def _mutated(tmp_path: Path, mutate: Callable[[dict[str, Any]], None], name: str) -> Path:
    """Write an OUTPUT that differs from a clean render in exactly one field.

    The clean render is decoded, mutated in memory and re-dumped with the
    renderer's own formatting, so the only difference from a matching artifact
    is the one *mutate* introduces — never key order or whitespace.

    Args:
        tmp_path: Per-test temporary directory.
        mutate: Applied to the decoded document to introduce the difference.
        name: Stem for the written file, so a failing test names its fixture.

    Returns:
        Path to the written OUTPUT.
    """
    document = json.loads(_rendered())
    mutate(document)
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _write(tmp_path / f"{name}.json", text)


@pytest.fixture
def clean_output(tmp_path: Path) -> Path:
    """An OUTPUT that matches the shipped schema byte for byte."""
    return _write(tmp_path / "demo_ontology.json", _rendered())


def test_matching_artifact_reports_no_differences(clean_output: Path) -> None:
    """A freshly rendered OUTPUT is reported as up to date."""
    assert check_artifact(SCHEMA_PATH, clean_output) == ()


def test_mutation_helper_reproduces_the_renderer_bytes(tmp_path: Path) -> None:
    """An unmutated ``_mutated`` OUTPUT is byte-identical to a clean render.

    The helper re-dumps with the renderer's formatting by hand; this pins that
    promise, so every drift assertion below fails only for the field it mutates.
    """
    identity = _mutated(tmp_path, lambda document: None, "identity")

    assert identity.read_text(encoding="utf-8") == _rendered()
    assert check_artifact(SCHEMA_PATH, identity) == ()


def test_removed_alt_label_names_the_class_and_the_label(tmp_path: Path) -> None:
    """Dropping one synonym names both the class it belongs to and the label."""

    def drop_label(document: dict[str, Any]) -> None:
        entry = document["classes"]["BeamPositionMonitor"]
        entry["altLabels"] = [label for label in entry["altLabels"] if label != "bpm"]

    output = _mutated(tmp_path, drop_label, "dropped_label")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any("BeamPositionMonitor" in line and "bpm" in line for line in lines), lines


def test_retargeted_family_names_the_family_and_both_classes(tmp_path: Path) -> None:
    """Pointing a family at another class names the family and both targets."""

    def retarget(document: dict[str, Any]) -> None:
        document["family_to_class"]["BPM"] = "AcceleratorDevice"

    output = _mutated(tmp_path, retarget, "retargeted_family")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any(
        "BPM" in line and "AcceleratorDevice" in line and "BeamPositionMonitor" in line
        for line in lines
    ), lines


def test_changed_parent_names_the_class(tmp_path: Path) -> None:
    """Reparenting a class names that class, not merely "classes differ"."""

    def reparent(document: dict[str, Any]) -> None:
        document["classes"]["BeamPositionMonitor"]["parent"] = "AcceleratorDevice"

    output = _mutated(tmp_path, reparent, "reparented")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any("BeamPositionMonitor" in line and "parent" in line for line in lines), lines


def test_added_family_is_reported(tmp_path: Path) -> None:
    """A family present only in OUTPUT is named as such."""

    def add_family(document: dict[str, Any]) -> None:
        document["family_to_class"]["INVENTED"] = "AcceleratorDevice"

    output = _mutated(tmp_path, add_family, "added_family")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any("INVENTED" in line for line in lines), lines


def test_removed_class_is_reported(tmp_path: Path) -> None:
    """A class missing from OUTPUT is named, alongside the families it breaks."""

    def drop_class(document: dict[str, Any]) -> None:
        del document["classes"]["BeamPositionMonitor"]

    output = _mutated(tmp_path, drop_class, "dropped_class")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any("BeamPositionMonitor" in line for line in lines), lines


def test_replaced_header_mentions_the_generated_key(tmp_path: Path) -> None:
    """An edited provenance header is reported against ``_generated``."""

    def replace_header(document: dict[str, Any]) -> None:
        document["_generated"] = "hand-written, honest"

    output = _mutated(tmp_path, replace_header, "replaced_header")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any("_generated" in line for line in lines), lines


def test_output_that_is_not_json_says_so(tmp_path: Path) -> None:
    """Undecodable OUTPUT gets one honest line, not a parser traceback."""
    output = _write(tmp_path / "broken.json", "{ this is not json ]\n")

    lines = check_artifact(SCHEMA_PATH, output)

    assert any("OUTPUT is not valid JSON" in line for line in lines), lines


def test_formatting_only_drift_is_reported_as_formatting(tmp_path: Path) -> None:
    """Same table, different bytes: the check says the formatting drifted."""
    document = json.loads(_rendered())
    output = _write(tmp_path / "reformatted.json", json.dumps(document, indent=4) + "\n")

    lines = check_artifact(SCHEMA_PATH, output)

    assert lines, "reformatted OUTPUT must not be reported as up to date"
    assert any("format" in line.lower() for line in lines), lines


@pytest.mark.parametrize(
    "mutate,name",
    [
        (
            lambda doc: doc["classes"]["BeamPositionMonitor"].__setitem__("parent", "Magnet"),
            "parent",
        ),
        (lambda doc: doc["family_to_class"].__setitem__("BPM", "Magnet"), "family"),
        (lambda doc: doc.__setitem__("_generated", "edited"), "header"),
    ],
)
def test_every_report_ends_with_the_rerun_line(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], name: str
) -> None:
    """However OUTPUT differs, the last line says which verb to re-run."""
    output = _mutated(tmp_path, mutate, name)

    lines = check_artifact(SCHEMA_PATH, output)

    assert lines
    assert "osprey knowledge compile-ontology" in lines[-1]
    assert str(SCHEMA_PATH) in lines[-1]
    assert str(output) in lines[-1]


def test_undecodable_output_still_ends_with_the_rerun_line(tmp_path: Path) -> None:
    """The not-JSON path is a report like any other, and ends the same way."""
    output = _write(tmp_path / "broken.json", "nonsense\n")

    lines = check_artifact(SCHEMA_PATH, output)

    assert "osprey knowledge compile-ontology" in lines[-1]


def test_missing_output_raises_oserror(tmp_path: Path) -> None:
    """An OUTPUT that cannot be read is an error, not a difference."""
    with pytest.raises(OSError):
        check_artifact(SCHEMA_PATH, tmp_path / "absent.json")


def test_check_never_writes_to_output(tmp_path: Path) -> None:
    """A stale OUTPUT is left exactly as it was found — bytes and mtime."""

    def reparent(document: dict[str, Any]) -> None:
        document["classes"]["BeamPositionMonitor"]["parent"] = "Magnet"

    output = _mutated(tmp_path, reparent, "untouched")
    before_bytes = output.read_bytes()
    before_mtime = output.stat().st_mtime_ns

    assert check_artifact(SCHEMA_PATH, output)

    assert output.read_bytes() == before_bytes
    assert output.stat().st_mtime_ns == before_mtime


def test_check_never_writes_when_the_artifact_matches(clean_output: Path) -> None:
    """The matching path does not rewrite OUTPUT either."""
    before_bytes = clean_output.read_bytes()
    before_mtime = clean_output.stat().st_mtime_ns

    assert check_artifact(SCHEMA_PATH, clean_output) == ()

    assert clean_output.read_bytes() == before_bytes
    assert clean_output.stat().st_mtime_ns == before_mtime
