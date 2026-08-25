"""FAMILY-to-class table and class hierarchy for the generated knowledge graph.

The channel database says a device belongs to the ``DIPOLE`` family; the graph
store needs it typed as ``narad_sem:Dipole``, and needs to know that a Dipole is
a Magnet and a Magnet is an AcceleratorDevice.  This module holds that table.

The table itself is data, not code: :data:`DEMO_ONTOLOGY_FILENAME` ships next to
this module and :func:`load_demo_ontology` reads it.  A facility with its own
vocabulary points :func:`load_ontology` at its own file instead of editing the
generator.

The demo table copies its class hierarchy edge-for-edge from the NARAD
shared-semantics vocabulary the als-ontology project defines, so a query that
rolls devices up to ``narad_sem:Magnet`` on any NARAD graph rolls the analogous
devices up on the demo graph.  The file's own ``_comment`` block records where
the demo machine has no exact NARAD class and which nearest class stands in.

This module is **pure**: standard library only, no ``rdflib``, no ``neo4j``.  It
reads one JSON file and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

from .model import NARAD_SEM_NS

#: Name of the ontology table shipped alongside this module.
DEMO_ONTOLOGY_FILENAME = "demo_ontology.json"

#: Package the shipped table is read from.
_PACKAGE = __name__.rsplit(".", 1)[0]

#: The class every other class must ultimately descend from.
ROOT_CLASS = "AcceleratorDevice"

_TOP_LEVEL_KEYS = frozenset({"root", "family_to_class", "classes"})


class OntologyMapError(ValueError):
    """Raised when an ontology table is malformed or internally inconsistent."""


class UnknownFamilyError(KeyError):
    """Raised when a FAMILY token has no entry in the ontology table.

    Subclasses :class:`KeyError` so callers can treat it as a lookup miss, but
    ``str(exc)`` is a plain operator-facing sentence naming both the family and
    the table file that would have to gain the entry.
    """

    def __init__(self, family: str, source_path: Path, known: Iterable[str]) -> None:
        super().__init__(family)
        self.family = family
        self.source_path = source_path
        known_families = ", ".join(sorted(known))
        self._message = (
            f"Device family {family!r} has no class in the ontology table {source_path}. "
            f"Add it to that file's 'family_to_class' block. "
            f"Known families: {known_families}"
        )

    def __str__(self) -> str:
        return self._message


@dataclass(frozen=True)
class ClassDef:
    """One class in the device vocabulary.

    Args:
        name: Local name of the class, e.g. ``Dipole``.  The full IRI is this
            name in the ``narad_sem`` namespace — see :attr:`iri`.
        parent: Local name of the ``rdfs:subClassOf`` parent, or ``None`` for
            the root class.
        alt_labels: ``skos:altLabel`` synonyms, in the order they should be
            emitted (the tables sort them alphabetically).
    """

    name: str
    parent: str | None
    alt_labels: tuple[str, ...]

    @property
    def iri(self) -> str:
        """Full IRI of the class in the ``narad_sem`` namespace."""
        return f"{NARAD_SEM_NS}{self.name}"

    @property
    def parent_iri(self) -> str | None:
        """Full IRI of the parent class, or ``None`` for the root class."""
        return None if self.parent is None else f"{NARAD_SEM_NS}{self.parent}"


@dataclass(frozen=True)
class OntologyMap:
    """A FAMILY-to-class table together with the class hierarchy it draws on.

    Args:
        family_to_class: Class name per FAMILY token.
        classes: Every class the table declares, keyed by class name.
        source_path: File the table was read from.  Carried so error messages
            can name the file an operator has to edit.
    """

    family_to_class: Mapping[str, str]
    classes: Mapping[str, ClassDef]
    source_path: Path

    def class_for(self, family: str) -> ClassDef:
        """Return the class a FAMILY token maps to.

        Args:
            family: FAMILY token, e.g. ``DIPOLE``.

        Returns:
            The mapped :class:`ClassDef`.

        Raises:
            UnknownFamilyError: If the table has no entry for *family*.
        """
        try:
            name = self.family_to_class[family]
        except KeyError:
            raise UnknownFamilyError(family, self.source_path, self.family_to_class) from None
        return self.classes[name]

    def ancestors(self, name: str) -> tuple[str, ...]:
        """Return the ancestors of a class, nearest first, ending at the root.

        The class itself is not included, so the root class has no ancestors.

        Args:
            name: Local name of a class in this table.

        Returns:
            Parent, grandparent, ... up to and including :data:`ROOT_CLASS`.

        Raises:
            OntologyMapError: If *name* or any ancestor is not declared, or if
                the chain revisits a class (the table has a cycle).
        """
        chain: list[str] = []
        seen = {name}
        current = self._lookup(name).parent
        while current is not None:
            if current in seen:
                raise OntologyMapError(
                    f"Ontology table {self.source_path} has a subClassOf cycle: "
                    f"{' -> '.join([name, *chain, current])}"
                )
            seen.add(current)
            chain.append(current)
            current = self._lookup(current).parent
        return tuple(chain)

    def used_classes(self, families: Iterable[str]) -> tuple[ClassDef, ...]:
        """Return the classes needed to type *families*, closed under subClassOf.

        Every class a family maps to is included, along with all of its
        ancestors, so the returned set can be emitted on its own and still be a
        complete hierarchy rooted at :data:`ROOT_CLASS`.

        Args:
            families: FAMILY tokens, in any order; duplicates are ignored.

        Returns:
            The closure, sorted by class name.

        Raises:
            UnknownFamilyError: If any family has no entry in the table.
            OntologyMapError: If the hierarchy above a mapped class is broken.
        """
        names: set[str] = set()
        for family in families:
            mapped = self.class_for(family).name
            names.add(mapped)
            names.update(self.ancestors(mapped))
        return tuple(self.classes[name] for name in sorted(names))

    def validate(self) -> None:
        """Check that the table is a well-formed hierarchy.

        Every mapped class and every referenced parent must be declared, exactly
        one class may be parentless, that class must be :data:`ROOT_CLASS`, and
        every class must reach the root without revisiting a class.

        Raises:
            OntologyMapError: On the first problem found.
        """
        for family, name in sorted(self.family_to_class.items()):
            if name not in self.classes:
                raise OntologyMapError(
                    f"Ontology table {self.source_path} maps family {family!r} to class "
                    f"{name!r}, which the 'classes' block does not declare"
                )

        roots = sorted(name for name, klass in self.classes.items() if klass.parent is None)
        if roots != [ROOT_CLASS]:
            raise OntologyMapError(
                f"Ontology table {self.source_path} must declare exactly one parentless "
                f"class, {ROOT_CLASS!r}; found {roots or 'none'}"
            )

        for name in sorted(self.classes):
            klass = self.classes[name]
            if klass.parent is not None and klass.parent not in self.classes:
                raise OntologyMapError(
                    f"Ontology table {self.source_path} declares class {name!r} with parent "
                    f"{klass.parent!r}, which the 'classes' block does not declare"
                )
            # Raises on a cycle; the parentless-class check above already
            # guarantees a terminating chain ends at the root.
            self.ancestors(name)

    def _lookup(self, name: str) -> ClassDef:
        """Return a declared class, or explain which table is missing it."""
        try:
            return self.classes[name]
        except KeyError:
            raise OntologyMapError(
                f"Ontology table {self.source_path} references class {name!r}, "
                f"which its 'classes' block does not declare"
            ) from None


def parse_ontology(payload: Mapping[str, object], source_path: Path) -> OntologyMap:
    """Build an :class:`OntologyMap` from a decoded table, then validate it.

    Args:
        payload: Decoded table.  Keys starting with ``_`` are documentation and
            are ignored; the rest must be ``root``, ``family_to_class`` and
            ``classes``.
        source_path: File the payload came from, for error messages.

    Returns:
        The validated map.

    Raises:
        OntologyMapError: If the payload has the wrong shape, or if the
            resulting hierarchy does not validate.
    """
    unknown = sorted(
        key for key in payload if not key.startswith("_") and key not in _TOP_LEVEL_KEYS
    )
    if unknown:
        raise OntologyMapError(
            f"Ontology table {source_path} has unknown top-level key(s) "
            f"{', '.join(repr(key) for key in unknown)}; expected "
            f"{', '.join(sorted(_TOP_LEVEL_KEYS))}"
        )

    root = payload.get("root", ROOT_CLASS)
    if root != ROOT_CLASS:
        raise OntologyMapError(
            f"Ontology table {source_path} declares root {root!r}; "
            f"the NARAD vocabulary is rooted at {ROOT_CLASS!r}"
        )

    raw_families = payload.get("family_to_class")
    if not isinstance(raw_families, Mapping) or not raw_families:
        raise OntologyMapError(
            f"Ontology table {source_path} needs a non-empty 'family_to_class' mapping"
        )
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, Mapping) or not raw_classes:
        raise OntologyMapError(f"Ontology table {source_path} needs a non-empty 'classes' mapping")

    families: dict[str, str] = {}
    for family, name in raw_families.items():
        if not isinstance(name, str):
            raise OntologyMapError(
                f"Ontology table {source_path} maps family {family!r} to a "
                f"{type(name).__name__}; expected a class name"
            )
        families[family] = name

    classes = {name: _parse_class(name, entry, source_path) for name, entry in raw_classes.items()}

    ontology = OntologyMap(
        family_to_class=MappingProxyType(families),
        classes=MappingProxyType(classes),
        source_path=source_path,
    )
    ontology.validate()
    return ontology


def _parse_class(name: str, entry: object, source_path: Path) -> ClassDef:
    """Build one :class:`ClassDef` from its decoded table entry."""
    if not isinstance(entry, Mapping):
        raise OntologyMapError(
            f"Ontology table {source_path} declares class {name!r} as a "
            f"{type(entry).__name__}; expected an object with 'parent' and 'altLabels'"
        )
    parent = entry.get("parent")
    if parent is not None and not isinstance(parent, str):
        raise OntologyMapError(
            f"Ontology table {source_path} gives class {name!r} a parent of type "
            f"{type(parent).__name__}; expected a class name or null"
        )
    raw_labels = entry.get("altLabels") or ()
    if isinstance(raw_labels, str) or not isinstance(raw_labels, Iterable):
        raise OntologyMapError(
            f"Ontology table {source_path} gives class {name!r} altLabels of type "
            f"{type(raw_labels).__name__}; expected a list of strings"
        )
    labels = tuple(str(label) for label in raw_labels)
    return ClassDef(name=name, parent=parent, alt_labels=labels)


def load_ontology(path: Path) -> OntologyMap:
    """Read and validate an ontology table from disk.

    Args:
        path: Path to a JSON table in the shape of ``demo_ontology.json``.

    Returns:
        The validated map.

    Raises:
        OntologyMapError: If the file is not valid JSON, or the table does not
            validate.
        OSError: If the file cannot be read.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OntologyMapError(f"Ontology table {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise OntologyMapError(
            f"Ontology table {path} must hold a JSON object, not a {type(payload).__name__}"
        )
    return parse_ontology(payload, path)


@lru_cache(maxsize=1)
def load_demo_ontology() -> OntologyMap:
    """Return the ontology table shipped with this package.

    The result is immutable and cached, so repeated calls neither re-read the
    file nor hand out a table one caller could mutate under another.

    Returns:
        The validated demo-machine map.

    Raises:
        OntologyMapError: If the shipped table is malformed — which would be a
            packaging fault, not an operator error.
    """
    resource = files(_PACKAGE).joinpath(DEMO_ONTOLOGY_FILENAME)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return parse_ontology(payload, Path(str(resource)))
