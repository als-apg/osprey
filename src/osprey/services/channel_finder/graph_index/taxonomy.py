"""Reducing raw ontology rows to the device taxonomy the explorer draws.

Split out of the channel finder's REST layer so the flat search index can
build the same taxonomy without importing FastAPI or any store driver: only
stdlib and typing, per the package docstring's cheap-import contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def class_name(uri: str) -> str:
    """Return the display name of a class URI: its trailing fragment.

    Ontology URIs end in the class name after either a path separator or a
    fragment marker, and which one is used is the corpus author's choice rather
    than something the explorer should care about.

    Args:
        uri: The class URI as the store holds it.

    Returns:
        The text after the last ``/`` or ``#``, or the whole URI when it
        carries neither.
    """
    cut = max(uri.rfind("/"), uri.rfind("#"))
    return uri[cut + 1 :] if cut >= 0 else uri


def prune_device_taxonomy(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce raw ontology rows to the classes worth drawing as a device tree.

    The store's ``:Class`` nodes describe more than devices — signal and binding
    classes live in the same tree — and drawing all of them buries the taxonomy
    an operator came for. A class earns its place by either holding devices
    (its rollup is non-zero) or by being an abstract parent of a class that
    does; anything else is a leaf about something other than devices and is
    dropped.

    Args:
        rows: One mapping per ``:Class`` node, as the graph paradigm's ontology
            query returns them — ``uri``, ``altLabel``, ``parents``, ``rollup``
            and ``direct``.

    Returns:
        The surviving classes, each carrying its ``uri``, derived ``name``,
        ``altLabel`` list, ``parents`` list, ``rollup`` and whether it is
        ``abstract``, sorted by name.
    """
    materialised = list(rows)

    #: A class is abstract-but-wanted when another class declares it a parent.
    #: A row naming itself does not count, so a self-referential SUBCLASSOF edge
    #: cannot keep an otherwise empty class alive.
    parent_uris: set[str] = set()
    for row in materialised:
        own_uri = row.get("uri")
        for parent in row.get("parents") or []:
            if parent is not None and parent != own_uri:
                parent_uris.add(parent)

    kept: list[dict[str, Any]] = []
    for row in materialised:
        uri = row.get("uri")
        if uri is None:
            continue
        rollup = row.get("rollup") or 0
        if rollup == 0 and uri not in parent_uris:
            continue
        kept.append(
            {
                "uri": uri,
                "name": class_name(uri),
                "altLabel": list(row.get("altLabel") or []),
                "parents": list(row.get("parents") or []),
                "rollup": rollup,
                # A class nothing is typed directly as is a grouping rather than
                # a kind of device, whatever its subclasses roll up to it.
                "abstract": (row.get("direct") or 0) == 0,
            }
        )

    kept.sort(key=lambda entry: (entry["name"], entry["uri"]))
    return kept
