"""Classify a change between two JSON Schemas of facility.json as additive or breaking.

A facility document's header ``osprey.facility.facility/N`` bumps N on a removed,
renamed or retyped key, or a key whose meaning changed; an added key does not.
:func:`classify` decides which side of that line a schema change falls on, by
structure alone, so a reader pinned to N keeps reading every document written
under N.

The comparison walks the keywords ``type``, ``anyOf``, ``$ref``, ``enum``,
``required``, ``properties``, ``items``, ``additionalProperties``, ``const`` and
``$defs``. It drops ``linkml_meta`` and ignores the annotation keywords
``description``, ``title`` and ``default`` where they are schema keywords, never
where they are the keys of a ``properties`` map, since there they name real
properties.

A top-level ``$schema`` or ``$id`` is the document's identity, not its shape: the
dialect it is written in and the URL it is published at. :func:`classify` skips
both at the root, so adding, changing or dropping either is neither breaking nor
additive; the generator's own test checks that the identity is right. Below the
root, ``$schema`` and ``$id`` compare like any other keyword.

**Additive**: a new optional property, a new ``$defs`` entry, a new enum member.
**Breaking**: a removed or renamed property, a changed ``type``, ``$ref``,
``anyOf`` or ``const``, a removed enum member, a newly required field. Any other
difference is one the classifier cannot vouch for, and is breaking too.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from osprey_facility.schema._generate import without_identity

__all__ = ["Compatibility", "classify"]

#: Keywords that annotate a schema without constraining a document.
_ANNOTATIONS = frozenset({"description", "title", "default", "linkml_meta"})

#: Keywords whose value is a map of names to subschemas.
_SCHEMA_MAPS = frozenset({"properties", "$defs"})

#: Keywords whose value is one subschema (or a boolean schema).
_SUBSCHEMAS = frozenset({"items", "additionalProperties"})


@dataclass(frozen=True)
class Compatibility:
    """The outcome of comparing an older schema with a newer one.

    Attributes:
        breaking: One line per change a reader of the older schema cannot
            absorb, each led by the JSON pointer of the schema it occurs in.
        additions: One line per additive change, in the same form.
    """

    breaking: tuple[str, ...]
    additions: tuple[str, ...]

    @property
    def additive(self) -> bool:
        """Whether the newer schema only extends the older one."""
        return not self.breaking


def classify(old: Mapping[str, Any], new: Mapping[str, Any]) -> Compatibility:
    """Compare two JSON Schema documents by structure.

    Args:
        old: The schema a reader is pinned to (the floor for the current N).
        new: The schema the producer now writes (current).

    Returns:
        The breaking and additive changes from ``old`` to ``new``; no breaking
        change means ``new`` is an additive extension of ``old``.
    """
    breaking: list[str] = []
    additions: list[str] = []
    _compare(without_identity(old), without_identity(new), "#", breaking, additions)
    return Compatibility(tuple(breaking), tuple(additions))


def _compare(old: Any, new: Any, where: str, breaking: list[str], additions: list[str]) -> None:
    """Compare two schemas at ``where``, appending each difference found."""
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        if old != new:
            breaking.append(f"{where}: schema changed from {old!r} to {new!r}")
        return
    keys = (set(old) | set(new)) - _ANNOTATIONS
    for key in sorted(keys):
        at = f"{where}/{key}"
        if key not in new:
            breaking.append(f"{at}: removed")
            continue
        if key not in old:
            if key == "$defs":
                additions.extend(f"{at}/{name}: added" for name in sorted(new[key]))
            else:
                breaking.append(f"{at}: added {new[key]!r}")
            continue
        before, after = old[key], new[key]
        if key in _SCHEMA_MAPS:
            _compare_map(key, before, after, new, at, breaking, additions)
        elif key in _SUBSCHEMAS:
            _compare(before, after, at, breaking, additions)
        elif key == "anyOf":
            _compare_any_of(before, after, at, breaking, additions)
        elif key == "enum":
            _compare_enum(before, after, at, breaking, additions)
        elif key == "required":
            _compare_required(before, after, at, breaking)
        elif before != after:
            breaking.append(f"{at}: changed from {before!r} to {after!r}")


def _compare_map(
    key: str,
    before: Any,
    after: Any,
    schema: Mapping[str, Any],
    at: str,
    breaking: list[str],
    additions: list[str],
) -> None:
    """Compare a ``properties`` or ``$defs`` map, whose keys are names."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        if before != after:
            breaking.append(f"{at}: changed from {before!r} to {after!r}")
        return
    required = schema.get("required") or ()
    for name in sorted(set(before) | set(after)):
        here = f"{at}/{name}"
        if name not in after:
            breaking.append(f"{here}: removed")
        elif name not in before:
            if key == "properties" and name in required:
                breaking.append(f"{here}: added as a required property")
            else:
                additions.append(f"{here}: added")
        else:
            _compare(before[name], after[name], here, breaking, additions)


def _compare_any_of(
    before: Any, after: Any, at: str, breaking: list[str], additions: list[str]
) -> None:
    """Compare two ``anyOf`` lists branch by branch; any other change is breaking."""
    if (
        not isinstance(before, Sequence)
        or not isinstance(after, Sequence)
        or len(before) != len(after)
    ):
        breaking.append(f"{at}: changed from {before!r} to {after!r}")
        return
    for index, (one, other) in enumerate(zip(before, after, strict=True)):
        _compare(one, other, f"{at}/{index}", breaking, additions)


def _compare_enum(
    before: Any, after: Any, at: str, breaking: list[str], additions: list[str]
) -> None:
    """Compare two ``enum`` lists: a removed member breaks, a new one adds."""
    if not isinstance(before, Sequence) or not isinstance(after, Sequence):
        if before != after:
            breaking.append(f"{at}: changed from {before!r} to {after!r}")
        return
    breaking.extend(f"{at}: member {m!r} removed" for m in before if m not in after)
    additions.extend(f"{at}: member {m!r} added" for m in after if m not in before)


def _compare_required(before: Any, after: Any, at: str, breaking: list[str]) -> None:
    """Compare two ``required`` lists: any change in the set is breaking.

    A newly required name breaks documents that omit it; a name no longer
    required breaks readers that rely on it being present.
    """
    old, new = set(before or ()), set(after or ())
    breaking.extend(f"{at}: {name!r} newly required" for name in sorted(new - old))
    breaking.extend(f"{at}: {name!r} no longer required" for name in sorted(old - new))
