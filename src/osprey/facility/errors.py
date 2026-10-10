"""The one error a facility build stops with.

Every stop is a single line on stderr::

    facility: <kind>: <record kind> <id> — <detail>; fix: <remedy>

A schema failure names its location as record kind ``path`` and the dotted path
as the id. The exit code is 1; click keeps 2 for usage errors. A warning shares
the line shape, prints after a clean build, and leaves the exit code at 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import click

__all__ = [
    "KINDS",
    "WARNING_KINDS",
    "FacilityBuildError",
    "FacilityBuildWarning",
    "quoted_slots",
]

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
    "seed-missing",
    "limit-invalid",
    "place-conflict",
    "span-invalid",
    "wiring-conflict",
    "engine-missing",
    "engine-invalid",
    "model-conflict",
    "view-unsupported",
    "profile-invalid",
)

#: Every warning kind, each ``<thing>-<problem>``.
WARNING_KINDS: tuple[str, ...] = ("place-wrapped",)


def quoted_slots(slots: Iterable[str]) -> str:
    """Name slots in an error line: each in backticks, comma-separated.

    Args:
        slots: The slot names, in the order they are named.

    Returns:
        The names, such as ```noise`, `drift``` for ``noise`` and
        ``drift``.
    """
    return ", ".join(f"`{slot}`" for slot in slots)


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


@dataclass(frozen=True)
class FacilityBuildWarning:
    """A fact of a clean facility build its author should read.

    A warning never stops the build and never changes the exit code; its line
    has the shape of a stop's line.

    Attributes:
        kind: The warning kind, one of ``WARNING_KINDS``.
        record_kind: The kind of record the id names (``device``, ...).
        record_id: The id of the record the warning is about.
        detail: What the build did with the record.
        remedy: What the author changes to clear the warning.
    """

    kind: str
    record_kind: str
    record_id: str
    detail: str
    remedy: str

    def __post_init__(self) -> None:
        if self.kind not in WARNING_KINDS:
            raise ValueError(f"unknown facility warning kind {self.kind!r}")

    @property
    def summary(self) -> str:
        """The line without its remedy."""
        return f"facility: {self.kind}: {self.record_kind} {self.record_id} — {self.detail}"

    @property
    def line(self) -> str:
        """The whole line, remedy included."""
        return f"{self.summary}; fix: {self.remedy}"
