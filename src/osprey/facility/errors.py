"""The one error a facility build stops with.

Every stop is a single line on stderr::

    facility: <kind>: <record kind> <id> — <detail>; fix: <remedy>

A schema failure names its location as record kind ``path`` and the dotted path
as the id. The exit code is 1; click keeps 2 for usage errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import click

__all__ = ["KINDS", "FacilityBuildError"]

#: Every error kind, each ``<thing>-<problem>``.
KINDS: tuple[str, ...] = (
    "source-invalid",
    "layer-conflict",
    "layer-duplicate",
    "fix-missing",
    "fix-stale",
    "fix-duplicate",
    "fix-computed",
    "fix-authored",
    "fix-referenced",
    "reference-missing",
    "class-unknown",
    "pair-invalid",
    "value-invalid",
    "seed-invalid",
    "limit-invalid",
    "place-conflict",
    "span-invalid",
    "wiring-conflict",
    "engine-missing",
    "engine-invalid",
    "model-conflict",
)


class FacilityBuildError(click.ClickException):
    """A facility build stop: one kind, one record, one remedy.

    Attributes:
        kind: The error kind, one of ``KINDS``.
        record_id: The id of the offending record, or the dotted path of a
            schema failure.
        sources: The source files the record came from, in source order.
        remedy: What the author changes to clear the stop.
        record_kind: The kind of record the id names (``device``, ``channel``,
            ...), or ``path`` for a schema failure.
        detail: What is wrong with the record.
    """

    exit_code = 1

    def __init__(
        self,
        kind: str,
        record_id: str,
        sources: Sequence[str | Path],
        remedy: str,
        *,
        record_kind: str,
        detail: str,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown facility error kind {kind!r}")
        self.kind = kind
        self.record_id = record_id
        self.sources = tuple(str(source) for source in sources)
        self.remedy = remedy
        self.record_kind = record_kind
        self.detail = detail
        super().__init__(f"facility: {kind}: {record_kind} {record_id} — {detail}; fix: {remedy}")

    def show(self, file: IO[Any] | None = None) -> None:
        """Write the error line alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        if file is None:
            click.echo(self.format_message(), err=True, color=self.show_color)
        else:
            click.echo(self.format_message(), file=file, color=self.show_color)
