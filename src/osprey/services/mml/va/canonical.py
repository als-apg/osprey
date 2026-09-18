"""Canonical ``va.json`` and ``response.json`` writer.

A 2.0 export carries two sibling documents beside its AO and AD: the
virtual-accelerator facts sampled in MATLAB and the orbit response matrix.
``osprey mml import`` keys them by system exactly as it keys AO and AD, so
``data/mml/va.json`` is ``{system: block}`` and ``data/mml/response.json`` is
``{system: block}``, one block per input and never two inputs for one system.

The bytes follow the rules of :mod:`osprey.services.mml.canonical`, whose
:func:`~osprey.services.mml.canonical.write_if_changed` does the writing: the
digests of these files are stamped into emitted artifacts just as the canonical
pair's are, so the bytes must be a pure function of the content and an
unchanged re-import must leave the file alone. ``test_va_canonical`` holds the
two serialisations byte-equal.

A block is stored as the exporter wrote it. Nothing is normalised and nothing
is dropped, the ``_export`` block included: its timestamp records when the
sampling ran, which is not when the AO was written.

Neither file is written when no input carried its document, so importing an
export without the siblings leaves ``data/mml/`` with the files it had.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click

from osprey.services.mml.canonical import write_if_changed

__all__ = [
    "RESPONSE_FILENAME",
    "VA_FILENAME",
    "merge_response_inputs",
    "merge_va_inputs",
    "sibling_system",
    "write_va_canonical",
]

#: File name of the canonical virtual-accelerator document.
VA_FILENAME = "va.json"

#: File name of the canonical response-matrix document.
RESPONSE_FILENAME = "response.json"

#: Provenance key a sibling document carries, holding its ``submachine``.
_EXPORT_KEY = "_export"


def sibling_system(source: Path, document: dict, explicit: str | None) -> str:
    """Resolve the system token of one sibling document.

    Args:
        source: The path the document was read from.
        document: The parsed sibling document.
        explicit: The token the caller already knows -- the ``--system`` token
            given for this input, or the system of the export it was paired
            with -- or ``None`` to take it from the document.

    Returns:
        The system token, stripped.

    Raises:
        click.UsageError: No token was given and the document carries no
            ``_export.submachine``, or the token is not a usable name.
    """
    token = explicit if explicit is not None else _submachine(document)
    if token is None:
        raise click.UsageError(
            f"Cannot tell which system {source} belongs to: it has no "
            f"_export.submachine and no export beside it carries its name; "
            f"pass --system {source.name}=TOKEN for it."
        )
    text = token.strip()
    if not text or text.startswith("_"):
        raise click.UsageError(
            f"Invalid system token {token!r} for {source}: a system token is a "
            "non-empty name that does not start with '_'."
        )
    return text


def merge_va_inputs(inputs: Sequence[tuple[Path, dict, str]]) -> dict[str, Any]:
    """Merge the virtual-accelerator documents into one document keyed by system.

    Args:
        inputs: Each document with the path it was read from and its resolved
            system token, in command-line order.

    Returns:
        ``{system: block}``, each block as the exporter wrote it.

    Raises:
        click.UsageError: Two inputs carry the same system.
    """
    return _merge(inputs, "virtual-accelerator export", "virtual-accelerator exports")


def merge_response_inputs(inputs: Sequence[tuple[Path, dict, str]]) -> dict[str, Any]:
    """Merge the response documents into one document keyed by system.

    Args:
        inputs: Each document with the path it was read from and its resolved
            system token, in command-line order.

    Returns:
        ``{system: block}``, each block as the exporter wrote it.

    Raises:
        click.UsageError: Two inputs carry the same system.
    """
    return _merge(inputs, "response matrix", "response matrices")


def write_va_canonical(va: dict, response: dict, out_dir: Path) -> tuple[Path | None, Path | None]:
    """Write the canonical virtual-accelerator and response documents.

    Args:
        va: The merged ``{system: block}`` of virtual-accelerator facts, or
            ``{}`` when no input carried one.
        response: The merged ``{system: block}`` of response matrices, or
            ``{}`` when no input carried one.
        out_dir: Directory to write into, created (with parents) if missing.

    Returns:
        The paths of ``va.json`` and ``response.json``, in that order, each
        ``None`` when its document was empty and no file was written.

    Raises:
        ValueError: Either document holds a non-finite float. Nothing is written.
        TypeError: Either document holds a value JSON cannot represent.
        OSError: A file could not be written; that file is left as it was.
    """
    out_dir = Path(out_dir)
    texts = {
        name: _dumps(document)
        for name, document in ((VA_FILENAME, va), (RESPONSE_FILENAME, response))
        if document
    }
    written: list[Path | None] = []
    for name in (VA_FILENAME, RESPONSE_FILENAME):
        if name not in texts:
            written.append(None)
            continue
        path = out_dir / name
        write_if_changed(path, texts[name])
        written.append(path)
    return written[0], written[1]


def _merge(inputs: Sequence[tuple[Path, dict, str]], singular: str, plural: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    seen: dict[str, Path] = {}
    for source, document, system in inputs:
        if system in seen:
            raise click.UsageError(
                f"System {system!r} is given two {plural}, {seen[system]} and "
                f"{source}; give each system one {singular}."
            )
        seen[system] = source
        merged[system] = document
    return merged


def _submachine(document: dict) -> str | None:
    export = document.get(_EXPORT_KEY)
    if not isinstance(export, dict):
        return None
    submachine = export.get("submachine")
    return submachine if isinstance(submachine, str) and submachine.strip() else None


def _dumps(document: Any) -> str:
    return (
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
