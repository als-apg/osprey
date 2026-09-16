"""Provenance shared by every artifact ``osprey mml emit`` writes.

Every emitted artifact names the same three facts, so a deployment can always be
traced back to the inputs that produced it:

* ``exporter`` -- the MML exporter version recorded in ``ao.json``'s
  ``_exports`` block, or ``none`` when the export carried no version;
* ``ao_sha256`` -- the sha256 of the canonical ``ao.json``;
* ``mapping_sha256`` -- the sha256 of ``mapping.yaml``.

:class:`EmitContext` renders them once, in each shape a consumer needs: a
single provenance STRING for the channel database (a string, never a mapping,
so the loader's system census skips it), bare header entries for the Turtle
emitter (which adds the ``# `` prefix itself), and OKF front-matter keys.

Pure stdlib plus ``click``: the ``knowledge`` extra is only probed, never
imported at module level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from osprey.services.mml.canonical import sha256_of
from osprey.services.mml.systems import EXPORTS_KEY, IMPORT_ORDER_KEY

__all__ = [
    "NO_EXPORTER",
    "EmitContext",
    "build_context",
    "exporter_version",
    "require_knowledge_extra",
]

#: Exporter version recorded when no export block names one.
NO_EXPORTER = "none"

#: Export-block keys that may carry the exporter version, in precedence order.
_EXPORTER_KEYS = ("exporter", "exporter_version")


@dataclass(frozen=True)
class EmitContext:
    """The provenance facts of one emit run, pre-rendered for each artifact.

    Attributes:
        ao_sha256: Lowercase hex sha256 of ``ao.json``.
        mapping_sha256: Lowercase hex sha256 of ``mapping.yaml``.
        exporter_version: The exporter version, or ``none``.
        provenance_string: ``exporter=<v> ao_sha256=<h> mapping_sha256=<h>``.
        header_lines: The three Turtle header entries, without a ``# `` prefix.
    """

    ao_sha256: str
    mapping_sha256: str
    exporter_version: str
    provenance_string: str
    header_lines: tuple[str, str, str]

    @property
    def front_matter(self) -> dict[str, str]:
        """The three OKF front-matter keys, as a fresh ordered dict."""
        return {
            "exporter": self.exporter_version,
            "ao_sha256": self.ao_sha256,
            "mapping_sha256": self.mapping_sha256,
        }


def build_context(
    ao_path: Path | str, mapping_path: Path | str, ao: Mapping[str, Any]
) -> EmitContext:
    """Hash the inputs and render the provenance of one emit run.

    Args:
        ao_path: The canonical ``ao.json`` the emit reads.
        mapping_path: The ``mapping.yaml`` the emit reads.
        ao: The decoded ``ao.json``; only ``_exports`` and ``_import_order``
            are read.

    Returns:
        The frozen :class:`EmitContext`.

    Raises:
        OSError: Either file could not be read.
    """
    ao_sha256 = sha256_of(Path(ao_path))
    mapping_sha256 = sha256_of(Path(mapping_path))
    exporter = exporter_version(ao)
    header_lines = (
        f"exporter={exporter}",
        f"ao_sha256={ao_sha256}",
        f"mapping_sha256={mapping_sha256}",
    )
    return EmitContext(
        ao_sha256=ao_sha256,
        mapping_sha256=mapping_sha256,
        exporter_version=exporter,
        provenance_string=" ".join(header_lines),
        header_lines=header_lines,
    )


def exporter_version(ao: Mapping[str, Any]) -> str:
    """Return the exporter version of the first imported system that records one.

    Systems are visited in ``_import_order`` (command-line order at import);
    without it, in ``_exports`` order. A system with no export block, or whose
    block names no non-blank version, is skipped.

    Args:
        ao: The decoded ``ao.json``.

    Returns:
        The version string, or :data:`NO_EXPORTER`.
    """
    exports = ao.get(EXPORTS_KEY)
    if not isinstance(exports, Mapping):
        return NO_EXPORTER
    order = ao.get(IMPORT_ORDER_KEY)
    systems = order if isinstance(order, list) else list(exports)
    for system in systems:
        block = exports.get(system) if isinstance(system, str) else None
        if not isinstance(block, Mapping):
            continue
        for key in _EXPORTER_KEYS:
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return NO_EXPORTER


def require_knowledge_extra() -> None:
    """Refuse unless the ``knowledge`` extra (``linkml_runtime``) is importable.

    Called before ``emit`` writes anything, so a missing extra never leaves a
    half-written deployment behind.

    Raises:
        click.ClickException: ``linkml_runtime`` cannot be imported.
    """
    try:
        import linkml_runtime  # noqa: F401, PLC0415
    except ImportError as exc:
        raise click.ClickException(
            f"The 'knowledge' extra is required for emit: {exc}\n"
            "Install it with: pip install 'osprey-framework[knowledge]'"
        ) from exc
