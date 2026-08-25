"""Turning a loaded LinkML schema into the ontology table shape the runtime reads.

This is the compiler's translation stage, and it is deliberately narrow.  A
LinkML schema can express far more than an ontology table can hold — slots,
mixins, unions, slot usage, ranges — and a schema that quietly used any of it
would compile into a table that silently dropped the meaning the author wrote
down.  So the translation runs on an allowlist: exactly six authored fields per
class (:data:`_AUTHORED_CLASS_FIELDS`, which :data:`_ALLOWED_CLASS_FIELDS`
widens by the two the loader fills in), and one enum.  Anything else populated
is an authoring error, reported by name.

The mapping itself is small:

============================  ==========================================
LinkML                        ontology table
============================  ==========================================
class with no ``is_a``        ``root``
``is_a``                      ``classes[name]["parent"]``
``aliases``                   ``classes[name]["altLabels"]`` (sorted, deduped)
``DeviceFamily`` value        ``family_to_class[<value>]``
value's ``meaning``           the class that value maps to
============================  ==========================================

Two habits are load-bearing here.  First, everything is read from
``view.schema`` — the *raw authored* definitions — never from the induced views
``SchemaView`` also offers, because those roll parents' slots and aliases down
into children and would turn the allowlist into a formality.  Second, every
CURIE is expanded through :meth:`~linkml_runtime.SchemaView.expand_curie` and
then *compared to the IRI it should have produced*.  ``expand_curie`` fails
soft: given an undeclared prefix it hands the string back unchanged rather than
raising, so a schema that misspelled its prefix would sail through a check that
only asked whether expansion raised.  Comparing against
``NARAD_SEM_NS + name`` catches the undeclared prefix, the wrong namespace and
the mismatched local name in one test.

This module holds no ``linkml_runtime`` import at module scope — not even for
typing — so importing it stays free of ``rdflib``.  It writes nothing and calls
nothing downstream: :func:`schema_to_payload` returns the dict that
:func:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.parse_ontology`
accepts, and validating it against that function is
:mod:`~osprey.services.facility_knowledge.ontology_compiler.compile`'s job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..ttl_generator.ontology_map import NARAD_SEM_NS
from .errors import OntologyCompileError

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from linkml_runtime import SchemaView
    from linkml_runtime.linkml_model.meta import ClassDefinition, PermissibleValue

#: Name of the enum that carries the FAMILY-token-to-class map.  The channel
#: database's FAMILY tokens are its permissible values; each value's ``meaning``
#: is the class devices of that family are typed as.
FAMILY_ENUM = "DeviceFamily"

#: The ``ClassDefinition`` fields an *author* may write.  Anything else —
#: ``mixins``, ``slots``, ``attributes``, ``slot_usage``, ``union_of``,
#: ``abstract`` — would express meaning the ontology table cannot hold, so it is
#: rejected rather than dropped.  This is the list a rejection message quotes,
#: because it is the list the person editing the schema can act on.
_AUTHORED_CLASS_FIELDS = frozenset(
    {
        "name",
        "is_a",
        "aliases",
        "class_uri",
        "description",
        "comments",
    }
)

#: Fields ``linkml_runtime`` fills in while loading.  They are always populated
#: and carry no authoring intent, so they are tolerated but never suggested.
_LOADER_FILLED_CLASS_FIELDS = frozenset({"from_schema", "definition_uri"})

#: Every ``ClassDefinition`` field a schema may leave populated.
_ALLOWED_CLASS_FIELDS = _AUTHORED_CLASS_FIELDS | _LOADER_FILLED_CLASS_FIELDS


def schema_to_payload(view: SchemaView, source: Path) -> dict[str, object]:
    """Translate a loaded LinkML schema into an ontology-table payload.

    The result is the decoded-JSON shape
    :func:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.parse_ontology`
    already accepts: ``family_to_class``, ``classes``, and ``root`` — the last
    only when exactly one class is parentless.  When none or several are, the
    key is left out on purpose, so the complaint comes from ``parse_ontology``,
    which names every candidate it found; duplicating that check here would
    produce a worse message for the same mistake.

    Structural questions about the resulting table — does every parent exist,
    does every family map to a declared class, is the hierarchy acyclic — are
    not asked here.  ``parse_ontology`` asks them, and asking twice would mean
    two places to keep in step.  What this function owns is everything that is
    only visible in the *schema*: which fields the author used, and whether the
    CURIEs resolve where they claim to.

    Args:
        view: The loaded schema, from
            :func:`~osprey.services.facility_knowledge.ontology_compiler.loader.load_schema`.
        source: Path the schema was read from.  Used only for error messages.

    Returns:
        A payload dict with ``family_to_class`` and ``classes``, plus ``root``
        when the schema has exactly one parentless class.

    Raises:
        OntologyCompileError: A class populates a field outside the allowlist;
            a class has no ``class_uri`` or one that does not expand to its own
            name in the ``narad_sem`` namespace; the schema declares no
            :data:`FAMILY_ENUM` enum or declares it with no permissible values;
            a permissible value has no ``meaning``, or a ``meaning`` that
            expands outside ``narad_sem``.  Every message names the class,
            enum or family token at fault.
    """
    payload: dict[str, object] = {
        "family_to_class": _family_map(view, source),
        "classes": _class_table(view, source),
    }
    root = _root_class(view)
    if root is not None:
        payload["root"] = root
    return payload


def _raw_classes(view: SchemaView) -> dict[str, ClassDefinition]:
    """Return the schema's classes exactly as authored, keyed by name.

    ``view.schema.classes`` is the raw block; ``SchemaView``'s induced
    accessors would roll inherited slots and aliases into every child, which
    would defeat both the field allowlist and the verbatim alias lists the
    table is meant to carry.  The ``or {}`` guards a schema with no ``classes``
    block at all — the field is optional, and indexing it unguarded would
    raise a :class:`TypeError` instead of the error ``parse_ontology`` gives
    for an empty table.
    """
    return dict(view.schema.classes or {})


def _class_table(view: SchemaView, source: Path) -> dict[str, object]:
    """Build the ``classes`` block, checking each class as it is translated."""
    table: dict[str, object] = {}
    for name, definition in _raw_classes(view).items():
        _reject_unsupported_fields(name, definition, source)
        _check_class_uri(view, name, definition, source)
        table[str(name)] = {
            "parent": str(definition.is_a) if definition.is_a else None,
            "altLabels": sorted({str(alias) for alias in definition.aliases or ()}),
        }
    return table


def _reject_unsupported_fields(name: str, definition: ClassDefinition, source: Path) -> None:
    """Fail when a class populates a field the ontology table cannot carry.

    Absent LinkML fields normalise to empty containers rather than ``None`` —
    ``aliases`` is ``[]``, ``attributes`` is ``{}`` — so the test is
    truthiness, not ``is not None``.  Testing for ``None`` would flag every
    class in every schema.
    """
    populated = {field for field, value in vars(definition).items() if value}
    unsupported = sorted(populated - _ALLOWED_CLASS_FIELDS)
    if unsupported:
        allowed = ", ".join(sorted(_AUTHORED_CLASS_FIELDS))
        raise OntologyCompileError(
            source,
            f"class {name!r} uses {', '.join(repr(field) for field in unsupported)}, which the "
            f"ontology table cannot represent; a class may declare only {allowed}",
        )


def _check_class_uri(
    view: SchemaView, name: str, definition: ClassDefinition, source: Path
) -> None:
    """Fail unless the class's ``class_uri`` names the class itself in ``narad_sem``."""
    class_uri = getattr(definition, "class_uri", None)
    if not class_uri:
        raise OntologyCompileError(
            source,
            f"class {name!r} has no 'class_uri'; every class must declare "
            f"'class_uri: narad_sem:{name}' so the compiled table and the emitted graph "
            f"agree on its IRI",
        )
    expected = f"{NARAD_SEM_NS}{name}"
    expanded = view.expand_curie(str(class_uri))
    if expanded != expected:
        raise OntologyCompileError(
            source,
            f"class {name!r} declares 'class_uri: {class_uri}', which resolves to "
            f"{expanded!r}; expected {expected!r}. Check that the CURIE's prefix is "
            f"declared in the schema's 'prefixes' block and that its local name matches "
            f"the class name",
        )


def _root_class(view: SchemaView) -> str | None:
    """Return the sole parentless class, or ``None`` when there is not exactly one.

    Returning ``None`` omits ``root`` from the payload, which hands the
    complaint to ``parse_ontology``: it lists every parentless class it found,
    which is the message an author needs, and it is the same message a
    hand-written JSON table produces for the same mistake.
    """
    roots = [str(name) for name, definition in _raw_classes(view).items() if not definition.is_a]
    return roots[0] if len(roots) == 1 else None


def _family_map(view: SchemaView, source: Path) -> dict[str, str]:
    """Build ``family_to_class`` from the :data:`FAMILY_ENUM` enum's values."""
    enums = view.schema.enums or {}
    family_enum = enums.get(FAMILY_ENUM)
    if family_enum is None:
        present = ", ".join(repr(str(name)) for name in sorted(enums)) or "none"
        raise OntologyCompileError(
            source,
            f"schema declares no {FAMILY_ENUM!r} enum; that enum is the family map, one "
            f"permissible value per FAMILY token of the channel database. Enums present: "
            f"{present}",
        )

    values = family_enum.permissible_values or {}
    if not values:
        raise OntologyCompileError(
            source,
            f"enum {FAMILY_ENUM!r} declares no permissible values; it must list one per "
            f"FAMILY token, each with 'meaning: narad_sem:<Class>'",
        )

    families: dict[str, str] = {}
    for token, value in values.items():
        families[str(token)] = _mapped_class(view, str(token), value, source)
    return families


def _mapped_class(view: SchemaView, token: str, value: PermissibleValue, source: Path) -> str:
    """Return the class name one permissible value's ``meaning`` points at."""
    meaning = getattr(value, "meaning", None)
    if not meaning:
        raise OntologyCompileError(
            source,
            f"family {token!r} in enum {FAMILY_ENUM!r} has no 'meaning'; every FAMILY token "
            f"must name the class its devices are typed as, as 'meaning: narad_sem:<Class>'",
        )
    expanded = view.expand_curie(str(meaning))
    local_name = expanded[len(NARAD_SEM_NS) :] if expanded.startswith(NARAD_SEM_NS) else ""
    if not local_name:
        raise OntologyCompileError(
            source,
            f"family {token!r} in enum {FAMILY_ENUM!r} means {str(meaning)!r}, which resolves "
            f"to {expanded!r}; a family must map to a class in the narad_sem namespace "
            f"({NARAD_SEM_NS}). Check that the CURIE's prefix is declared in the schema's "
            f"'prefixes' block",
        )
    return local_name
