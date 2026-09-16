"""The LinkML ontology schema ``osprey mml emit`` writes, and its compiled table.

:func:`build_ontology_yaml` turns the parsed mapping into a LinkML schema dict
the ``ontology_compiler`` accepts; :func:`compile_to_json` compiles such a
schema file into the JSON ontology table and returns the parsed table.

Rules:

* The root class is the packaged root. Every other packaged class appears only
  when a family, a mapping branch or a new class needs it or one of its
  descendants, so the tree is closed under ``is_a`` and holds nothing unused.
* Every class declares ``class_uri: narad_sem:<name>``; every non-root class
  declares ``is_a``. A class carries only the fields the compiler allows:
  ``class_uri``, ``is_a``, ``aliases`` and ``description``.
* A packaged class's ``aliases`` are its packaged alt labels plus the aliases of
  every family typed as it, sorted and de-duplicated.
* Every mapping branch becomes a class whose ``is_a`` is its ``parent``.
* Each distinct new family ``class`` becomes one class whose ``is_a`` is the
  family's ``branch``, whose ``aliases`` are the sorted union of the sharing
  families' aliases, and whose ``description`` is the first sharing family's by
  raw token.
* The ``DeviceFamily`` enum holds one value per mapped family token, sorted,
  each with ``meaning: narad_sem:<class>``. Families with ``channels: 0`` or no
  ``class`` are left out.
* The schema-level ``description`` carries the emit provenance string.
* Classes are written root first, then by name, so the output is deterministic.

Pure: stdlib plus the mapping schema, the packaged class table and the emit
context. The ontology compiler (and ``linkml_runtime``) is imported only inside
:func:`compile_to_json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.services.mml.emit.context import EmitContext
from osprey.services.mml.mapping.branches import ROOT_CLASS, packaged_classes
from osprey.services.mml.mapping.schema import Family, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey.services.facility_knowledge.ttl_generator.ontology_map import OntologyMap

__all__ = ["FAMILY_ENUM", "NARAD_SEM_PREFIX", "build_ontology_yaml", "compile_to_json"]

#: Namespace the compiled table requires every class IRI to live in.
NARAD_SEM_PREFIX = "https://narad.example.org/schema/shared_semantics/"

#: Enum carrying the family-token-to-class map.
FAMILY_ENUM = "DeviceFamily"

_DEFAULT_NAME = "facility"


def _curie(name: str) -> str:
    return f"narad_sem:{name}"


def _emitted_families(mapping: Mapping) -> list[Family]:
    """Families that type devices, sorted by raw token."""
    return [
        family
        for _, family in sorted(mapping.families.items())
        if family.channels > 0 and family.class_ is not None
    ]


def _parent_of(name: str, mapping: Mapping, packaged: dict) -> str | None:
    if name == ROOT_CLASS:
        return None
    if name in mapping.branches:
        return mapping.branches[name].parent
    if name in packaged:
        return packaged[name].parent
    raise ValueError(
        f"class {name!r} is neither the root, a packaged class nor a branch under 'branches:'"
    )


def _close(names: set[str], mapping: Mapping, packaged: dict) -> set[str]:
    """Return *names* plus every ancestor, walking branches and packaged classes."""
    closed: set[str] = set()
    for start in names:
        chain = [start]
        current: str | None = start
        while current is not None and current not in closed:
            closed.add(current)
            current = _parent_of(current, mapping, packaged)
            if current in chain:
                raise ValueError(f"class hierarchy cycle: {' -> '.join([*chain, current])}")
            if current is not None:
                chain.append(current)
    return closed


def build_ontology_yaml(mapping: Mapping, ctx: EmitContext) -> dict[str, Any]:
    """Build the LinkML schema for a mapping.

    Args:
        mapping: The parsed mapping.
        ctx: Provenance of this emit run.

    Returns:
        A schema dict ready for ``yaml.safe_dump(sort_keys=False)``.

    Raises:
        ValueError: A family's new class has no ``branch``, or a branch or
            parent names a class that is neither packaged nor a mapping branch.
    """
    packaged = packaged_classes()
    families = _emitted_families(mapping)

    family_aliases: dict[str, set[str]] = {}
    new_classes: dict[str, list[Family]] = {}
    for family in families:
        class_name = str(family.class_)
        family_aliases.setdefault(class_name, set()).update(family.aliases)
        if class_name == ROOT_CLASS or class_name in packaged:
            continue
        if family.branch is None:
            raise ValueError(
                f"family {family.raw!r}: new class {class_name!r} needs a 'branch' to sit under"
            )
        new_classes.setdefault(class_name, []).append(family)

    needed: set[str] = {ROOT_CLASS, *mapping.branches}
    for family in families:
        if family.class_ in new_classes:
            needed.add(new_classes[family.class_][0].branch)  # type: ignore[arg-type]
        else:
            needed.add(family.class_)  # type: ignore[arg-type]
    closed = _close(needed, mapping, packaged)

    classes: dict[str, dict[str, Any]] = {}
    for name in [ROOT_CLASS, *sorted((closed | set(new_classes)) - {ROOT_CLASS})]:
        body: dict[str, Any] = {"class_uri": _curie(name)}
        if name in new_classes:
            sharing = new_classes[name]
            body["is_a"] = sharing[0].branch
            aliases = sorted(family_aliases.get(name, ()))
            if aliases:
                body["aliases"] = aliases
            if sharing[0].description:
                body["description"] = sharing[0].description
        elif name in mapping.branches:
            branch = mapping.branches[name]
            body["is_a"] = branch.parent
            if branch.description:
                body["description"] = branch.description
        else:
            parent = _parent_of(name, mapping, packaged)
            if parent is not None:
                body["is_a"] = parent
            labels = set(packaged[name].alt_labels) if name in packaged else set()
            aliases = sorted(labels | family_aliases.get(name, set()))
            if aliases:
                body["aliases"] = aliases
        classes[name] = body

    values = {
        mapping.mapped(family.raw): {"meaning": _curie(family.class_)}  # type: ignore[arg-type]
        for family in families
    }

    name = mapping.facility.token or _DEFAULT_NAME
    schema: dict[str, Any] = {
        "id": f"{NARAD_SEM_PREFIX}{name}",
        "name": name,
    }
    if mapping.facility.title:
        schema["title"] = mapping.facility.title
    schema.update(
        {
            "description": (
                f"Facility ontology generated by osprey mml emit. {ctx.provenance_string}"
            ),
            "prefixes": {"narad_sem": NARAD_SEM_PREFIX, "linkml": "https://w3id.org/linkml/"},
            "default_prefix": "narad_sem",
            "default_range": "string",
            "imports": ["linkml:types"],
            "classes": classes,
            "enums": {FAMILY_ENUM: {"permissible_values": dict(sorted(values.items()))}},
        }
    )
    return schema


def compile_to_json(yaml_path: Path, json_path: Path) -> OntologyMap:
    """Compile a schema file and write its JSON ontology table atomically.

    Args:
        yaml_path: The LinkML schema to compile.
        json_path: Where the table goes. Its directory must already exist.

    Returns:
        The compiled, validated table.

    Raises:
        OntologyCompileError: The schema could not be read or uses something
            the table cannot represent.
        OntologyMapError: The described table does not validate.
        OSError: The JSON file could not be written; it is left untouched.
    """
    from osprey.cli.knowledge_cmd import _replace_file  # noqa: PLC0415
    from osprey.services.facility_knowledge.ontology_compiler import (  # noqa: PLC0415
        compile_schema,
        render_json,
    )

    yaml_path = Path(yaml_path)
    compiled = compile_schema(yaml_path)
    _replace_file(Path(json_path), render_json(compiled.payload, yaml_path))
    return compiled.table
