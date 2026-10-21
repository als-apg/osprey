"""Layers: the importers that turn an outside export into facility sources.

Each layer reads one outside format and writes record sources under
``data/facility/imported/<layer>/``; the build merges them with the authored
sources field by field.
"""

from __future__ import annotations

from typing import Any

import click

from osprey.facility.errors import FacilityBuildError

__all__ = ["LayerSourceInvalid"]


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
