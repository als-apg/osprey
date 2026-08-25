"""Proving the committed artifact still matches the schema it came from.

``demo_ontology.json`` is generated, but it is also *committed*, which means it
can drift: someone edits the YAML and forgets to recompile, or edits the JSON
directly despite the header telling them not to.  Nothing about either file
announces the drift — both stay valid, both keep loading — so this module makes
it announceable.  :func:`check_artifact` recompiles the schema in memory,
renders it with the very function that would have written the file, and
compares the text to what is on disk.  ``osprey knowledge compile-ontology
--check`` runs that in CI; the artifact is then as trustworthy as any other
tested thing in the repository.

Two decisions shape everything below.

**The verdict is byte-level; the explanation is element-level.**  Equality has
to be on bytes, because the point is that recompiling would leave ``git diff``
silent, and a comparison that normalised whitespace or key order would pass a
file that recompilation still rewrites.  But bytes make a terrible explanation:
"file differs" sends an operator to a diff tool to work out *what* differs.  So
once the texts disagree, both sides are decoded and compared as tables, and the
report names elements — this class's parent, that class's missing synonym, this
family's new target.  When the tables turn out to agree and only the formatting
does not, that is itself the finding, and it is reported as such rather than
left as a diff with nothing in it.

**A report is advice, so it ends with the action.**  Every non-empty result's
last line names the verb that fixes it, with the paths the caller passed, so a
failing CI log carries its own remedy.  That holds even for OUTPUT that is not
JSON at all: the check has nothing to say about *which* element is wrong, but
"recompile it" is still the right next move.

This module never writes: not the artifact, not a backup, not a temporary file.
A check that repaired what it found would turn a red gate green without anyone
choosing that, and the same code runs in pre-commit hooks where a surprise
rewrite is worse than a failure.  An unreadable OUTPUT is therefore an
:class:`OSError` that propagates, not a diff line — a missing file is not a
disagreement about content, and the caller decides whether it means "not
generated yet" or "wrong path".

Like :mod:`.render`, this module is **pure standard library**: no
``linkml_runtime`` here, and none at import time anywhere in the package — the
toolchain loads only inside :func:`~.compile.compile_schema`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .compile import compile_schema
from .render import render_json

#: Reported when OUTPUT cannot be decoded at all.  There is no element-level
#: story to tell in that case, so the check says the one true thing it knows
#: and lets the re-run line carry the remedy.
NOT_JSON_LINE = "OUTPUT is not valid JSON."

#: Reported when both sides decode to the same table yet the bytes differ —
#: key order, indentation, a missing trailing newline.  The artifact loads
#: correctly today and would still be rewritten by a recompile, which is
#: exactly the drift ``--check`` exists to catch.
FORMATTING_LINE = (
    "OUTPUT holds the same ontology as the schema but is formatted differently "
    "(key order, indentation or trailing newline)."
)


def check_artifact(source: Path, output: Path) -> tuple[str, ...]:
    """Report how a committed artifact differs from its schema.

    Compiles *source* in memory, renders it exactly as the writing path would,
    and compares that text with the bytes of *output*.  Nothing is written.

    Args:
        source: The authored ``.yaml`` schema, treated as the truth.
        output: The committed artifact to check against it.  Read, never
            modified.

    Returns:
        An empty tuple when *output* already holds the rendered text byte for
        byte.  Otherwise one line per difference — each naming the class,
        family or key involved — followed by a final line naming the command to
        re-run.  The lines are ordered so they can be printed as they come.

    Raises:
        OntologyCompileError: *source* could not be read, or uses something the
            ontology table cannot represent.
        OntologyMapError: *source* is valid LinkML but describes a table that
            does not validate.
        OSError: *output* could not be read.  Propagated unchanged, because a
            missing or unreadable artifact is a different problem from a stale
            one and only the caller knows which it expected.
    """
    expected = render_json(compile_schema(source).payload, source)
    found = output.read_text(encoding="utf-8")
    if expected == found:
        return ()
    return (*_describe(expected, found), _rerun_line(source, output))


def _rerun_line(source: Path, output: Path) -> str:
    """Return the closing line naming the command that fixes the report.

    The paths are reproduced as the caller passed them — relative if the caller
    was relative — so the line can be pasted into the same shell.

    Args:
        source: Schema path, as given.
        output: Artifact path, as given.

    Returns:
        A single sentence naming the verb and both paths.
    """
    return f"Re-run `osprey knowledge compile-ontology {source} {output}` to regenerate OUTPUT."


def _describe(expected: str, found: str) -> tuple[str, ...]:
    """Explain, element by element, how the artifact text disagrees.

    Args:
        expected: Text a clean compile of the schema renders.
        found: Text read from OUTPUT.  Assumed to differ from *expected*.

    Returns:
        Zero or more lines, without the closing re-run line.  Never empty in
        practice: when no element-level difference is found, the mismatch is
        formatting, and that is reported too.
    """
    try:
        found_doc = json.loads(found)
    except ValueError:
        return (NOT_JSON_LINE,)
    if not isinstance(found_doc, dict):
        return ("OUTPUT is valid JSON but not a JSON object.",)
    expected_doc = json.loads(expected)

    lines = [
        *_header_lines(expected_doc, found_doc),
        *_root_lines(expected_doc, found_doc),
        *_class_lines(_section(expected_doc, "classes"), _section(found_doc, "classes")),
        *_family_lines(
            _section(expected_doc, "family_to_class"), _section(found_doc, "family_to_class")
        ),
    ]
    if lines:
        return tuple(lines)
    if found_doc == expected_doc:
        return (FORMATTING_LINE,)
    return (_unexpected_keys_line(expected_doc, found_doc),)


def _section(document: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return one top-level block of a decoded artifact as a mapping.

    A block of the wrong shape — or an absent one — is treated as empty, so its
    every entry is reported as missing rather than raising while building the
    report.

    Args:
        document: Decoded artifact.
        key: Top-level key to read.

    Returns:
        The block, or an empty mapping when it is absent or not an object.
    """
    value = document.get(key)
    return value if isinstance(value, dict) else {}


def _header_lines(expected: Mapping[str, object], found: Mapping[str, object]) -> tuple[str, ...]:
    """Compare the ``_generated`` provenance header.

    A missing header means the artifact does not declare itself generated —
    typically a hand-written table sitting where a compiled one belongs — and
    reads as a mismatch against the header the renderer stamps.

    Args:
        expected: Decoded clean render.
        found: Decoded OUTPUT.

    Returns:
        At most one line.
    """
    want = expected.get("_generated")
    have = found.get("_generated")
    if want == have:
        return ()
    if "_generated" not in found:
        return (f"_generated: missing in OUTPUT, expected {want!r}.",)
    return (f"_generated: {have!r} in OUTPUT, expected {want!r}.",)


def _root_lines(expected: Mapping[str, object], found: Mapping[str, object]) -> tuple[str, ...]:
    """Compare the declared root class.

    Args:
        expected: Decoded clean render.
        found: Decoded OUTPUT.

    Returns:
        At most one line.
    """
    want = expected.get("root")
    have = found.get("root")
    if want == have:
        return ()
    return (f"root: {have!r} in OUTPUT, expected {want!r}.",)


def _class_lines(expected: Mapping[str, object], found: Mapping[str, object]) -> tuple[str, ...]:
    """Compare the class hierarchy, one class at a time.

    Classes are visited in sorted order so a report is stable across runs and
    two runs of CI on the same drift produce the same log.

    Args:
        expected: ``classes`` block of the clean render.
        found: ``classes`` block of OUTPUT.

    Returns:
        One or more lines per differing class: its absence, its extra presence,
        its parent, and each synonym that only one side declares.
    """
    lines: list[str] = []
    for name in sorted(set(expected) | set(found)):
        if name not in found:
            lines.append(f"class {name}: missing in OUTPUT.")
            continue
        if name not in expected:
            lines.append(f"class {name}: in OUTPUT but not in the schema.")
            continue
        lines.extend(_class_entry_lines(name, expected[name], found[name]))
    return tuple(lines)


def _class_entry_lines(name: str, expected: object, found: object) -> tuple[str, ...]:
    """Compare one class's parent and synonyms.

    Args:
        name: Class name, named in every line so a report reads standalone.
        expected: The class's entry in the clean render.
        found: The class's entry in OUTPUT, of unverified shape.

    Returns:
        Zero or more lines for this class.
    """
    if not isinstance(expected, dict) or not isinstance(found, dict):
        if expected == found:
            return ()
        return (f"class {name}: entry in OUTPUT is not a JSON object.",)

    lines: list[str] = []
    if expected.get("parent") != found.get("parent"):
        lines.append(
            f"class {name}: parent {found.get('parent')!r} in OUTPUT, "
            f"expected {expected.get('parent')!r}."
        )
    lines.extend(_alt_label_lines(name, expected.get("altLabels"), found.get("altLabels")))
    return tuple(lines)


def _alt_label_lines(name: str, expected: object, found: object) -> tuple[str, ...]:
    """Compare one class's ``altLabels`` as a set, then as a sequence.

    The set difference is what matters to a reader — a synonym gained or lost
    changes what the graph answers — so each missing or extra label gets its
    own line naming both the class and the label.  Order is compared only after
    the sets agree, because the renderer sorts them and a reordered list still
    means a recompile would rewrite the file.

    Args:
        name: Class the labels belong to.
        expected: ``altLabels`` from the clean render.
        found: ``altLabels`` from OUTPUT, of unverified shape.

    Returns:
        Zero or more lines for this class's labels.
    """
    if not isinstance(expected, list) or not isinstance(found, list):
        if expected == found:
            return ()
        return (f"class {name}: altLabels in OUTPUT is not a JSON array.",)

    lines: list[str] = []
    want = {str(label) for label in expected}
    have = {str(label) for label in found}
    for label in sorted(want - have):
        lines.append(f"class {name}: altLabel {label!r} missing in OUTPUT.")
    for label in sorted(have - want):
        lines.append(f"class {name}: altLabel {label!r} in OUTPUT but not in the schema.")
    if not lines and expected != found:
        lines.append(f"class {name}: altLabels are in a different order in OUTPUT.")
    return tuple(lines)


def _family_lines(expected: Mapping[str, object], found: Mapping[str, object]) -> tuple[str, ...]:
    """Compare the FAMILY-to-class table, one family at a time.

    Args:
        expected: ``family_to_class`` block of the clean render.
        found: ``family_to_class`` block of OUTPUT.

    Returns:
        One line per family that is absent, extra, or pointed somewhere else.
        A retargeting names both classes, since which one is wrong is the whole
        question.
    """
    lines: list[str] = []
    for family in sorted(set(expected) | set(found)):
        if family not in found:
            lines.append(
                f"family {family}: missing in OUTPUT, expected class {expected[family]!r}."
            )
        elif family not in expected:
            lines.append(f"family {family}: in OUTPUT but not in the schema.")
        elif expected[family] != found[family]:
            lines.append(
                f"family {family}: class {found[family]!r} in OUTPUT, "
                f"expected {expected[family]!r}."
            )
    return tuple(lines)


def _unexpected_keys_line(expected: Mapping[str, object], found: Mapping[str, object]) -> str:
    """Describe a difference that lives outside the blocks this module reads.

    The only way to reach this is an artifact carrying data the ontology table
    does not define — a leftover ``_comment`` block, a key from a future format
    — since every block ``parse_ontology`` reads is compared above.  Naming the
    keys is more useful than restating that the files differ.

    Args:
        expected: Decoded clean render.
        found: Decoded OUTPUT.

    Returns:
        One line naming the top-level keys that only one side carries, or a
        last-resort sentence when even those agree.
    """
    extra = sorted(set(found) - set(expected))
    missing = sorted(set(expected) - set(found))
    if extra or missing:
        parts = []
        if extra:
            parts.append(f"OUTPUT carries unexpected top-level keys: {', '.join(extra)}")
        if missing:
            parts.append(f"OUTPUT is missing top-level keys: {', '.join(missing)}")
        return "; ".join(parts) + "."
    return "OUTPUT differs from the schema outside the class and family tables."
