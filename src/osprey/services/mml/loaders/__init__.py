"""Loaders for MML exports.

Every loader decodes one container format (a ``.mat`` file or any
family-keyed JSON) and returns a :class:`LoadedInput`. A loader only decodes:
family bodies in ``ao`` are raw, and :func:`osprey.services.mml.normalize.normalize_family`
is applied to them afterwards.

One ``.mat`` shape carries no families at all: a lattice deck. It reaches the
importer as an input whose ``lattice`` names the file, so the import files the
deck beside the canonical documents instead of merging it into them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["LoadedInput"]


@dataclass(frozen=True)
class LoadedInput:
    """One decoded input file.

    Attributes:
        ao: Accelerator Objects. Keyed by system token when ``system_keyed``,
            otherwise keyed by family name; family bodies are raw.
        ad: Accelerator Data, when the input carries or pairs with one.
        export: The exporter's ``_export`` block, when present.
        system_keyed: True when the top-level keys of ``ao`` are systems.
        source: The path the input was read from.
        lattice: The lattice deck the input *is*, when it carries a ring
            instead of families; ``ao`` is then empty and ``ad`` is ``None``.
    """

    ao: dict
    ad: dict | None
    export: dict | None
    system_keyed: bool
    source: Path
    lattice: Path | None = None
