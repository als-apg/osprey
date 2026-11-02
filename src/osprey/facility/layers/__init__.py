"""Layers: the importers that turn an outside export into facility sources.

Each layer reads one outside format and writes record sources under
``data/facility/imported/<layer>/``; the build merges them with the authored
sources field by field. A seed-once file an import creates opens with
:data:`HEADER`, whichever format wrote it, and is a person's from then on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from osprey.facility.errors import FacilityBuildError

__all__ = ["HEADER", "LayerSourceInvalid"]

#: The first line of every file an import seeds, in every format.
HEADER = "# Seeded once by the facility import; edit this file by hand."

#: The body of the knowledge index an import seeds.


class LayerSourceInvalid(click.ClickException):
    """The source holds records the layer cannot write; nothing was written.

    Attributes:
        errors: One ``source-invalid`` error per problem, in the order the
            layer reports them.
    """

    exit_code = 1

    def __init__(self, errors: list[FacilityBuildError]) -> None:
        self.errors = errors
        super().__init__("\n".join(error.format_message() for error in errors))

    def show(self, file: Any = None) -> None:
        """Write each error's line alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        for error in self.errors:
            error.show(file)


def _seed_knowledge_index(facility_dir: Path) -> Path | None:
    """Write the blank placeholder knowledge index when none exists; never rewrite one.

    The placeholder is empty: the index regenerator neither renders it nor calls
    it stale, and writes over it once the bundle holds a page.

    Args:
        facility_dir: The ``data/facility`` directory.

    Returns:
        The path written, or ``None`` when an index already exists.
    """
    index = facility_dir / "knowledge" / "index.md"
    if index.exists():
        return None
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("", encoding="utf-8")
    return index
