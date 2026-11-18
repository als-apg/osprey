"""Roster sources: deployment code that says which IOCs a facility expects, and where.

Public, not yet declared stable. A roster source states, per IOC, its name, host,
class, whether it is expected to run and, optionally, a group. The server fleet view
reads every configured source on each refresh of the live target and merges them
with :func:`merge_rosters`: one row per IOC name, naming every source that stated
it. A source that raises is reported unavailable with the exception's class as the
reason, and the other sources are kept.

``channel_directory.fleet.roster_sources`` lists ``module:attribute`` references, each
to a :class:`RosterSource` or to a zero-argument factory (a class counts) that
returns one; :func:`load_roster_sources` resolves them. A deployment checks its
source with :func:`check_roster_source`, which returns the problems found.

OSPREY ships no built-in source. The documentation's example tree holds one: a
static list read from a YAML file.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast, runtime_checkable

__all__ = [
    "MergedRosterRow",
    "RosterMerge",
    "RosterRow",
    "RosterSource",
    "RosterSourceError",
    "SourceOutcome",
    "check_roster_source",
    "load_roster_source",
    "load_roster_sources",
    "merge_rosters",
]

Expectation = Literal["running", "stopped"]

_EXPECTATIONS = ("running", "stopped", None)


@dataclass(frozen=True)
class RosterRow:
    """One IOC as one roster source states it.

    ``host`` and ``class_name`` may be ``None`` when the source does not know them;
    an IOC with no class has the default class. ``expected`` is ``"running"``,
    ``"stopped"`` or ``None`` (the source states no expectation).
    """

    name: str
    host: str | None
    class_name: str | None
    expected: Expectation | None
    group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"a roster row needs a non-blank name, got {self.name!r}")
        if self.expected not in _EXPECTATIONS or isinstance(self.expected, bool):
            raise ValueError(
                f"roster row {self.name!r}: expected must be 'running', 'stopped' or None, "
                f"got {self.expected!r}"
            )
        for key in ("host", "class_name", "group"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"roster row {self.name!r}: {key} must be a string or None, "
                    f"got {type(value).__name__}"
                )


@runtime_checkable
class RosterSource(Protocol):
    """What a deployment's roster source provides.

    ``read(deadline)`` returns the rows; ``deadline`` is the seconds it may take.
    It raises to report failure. ``isinstance(obj, RosterSource)`` checks that both
    members are present, not their types; :func:`check_roster_source` checks more.
    """

    name: str

    def read(self, deadline: float) -> list[RosterRow]: ...


class RosterSourceError(ValueError):
    """A ``module:attribute`` reference that does not resolve to a roster source."""


@dataclass(frozen=True)
class MergedRosterRow:
    """One IOC after the merge: each field from the first source that states it.

    ``sources`` names every source that stated the IOC, in configured order.
    """

    name: str
    host: str | None
    class_name: str | None
    expected: Expectation | None
    group: str | None
    sources: tuple[str, ...]


@dataclass(frozen=True)
class SourceOutcome:
    """How one source's read went: ``ok``, or ``unavailable`` with a reason."""

    name: str
    outcome: Literal["ok", "unavailable"]
    reason: str | None


@dataclass(frozen=True)
class RosterMerge:
    """The merged rows, sorted by name, and one outcome per source in configured order."""

    rows: tuple[MergedRosterRow, ...] = field(default=())
    sources: tuple[SourceOutcome, ...] = field(default=())


# --- loading --------------------------------------------------------------------------


def load_roster_source(ref: str) -> RosterSource:
    """Resolve one ``module:attribute`` reference to a roster source.

    The attribute is used as is when it is already a source; otherwise it is called
    with no arguments and must return one. Raises :class:`RosterSourceError` naming
    the reference.
    """
    module_path, sep, attribute = ref.partition(":")
    if not sep or not module_path or not attribute:
        raise RosterSourceError(f"roster source {ref!r}: expected a 'module:attribute' reference")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RosterSourceError(
            f"roster source {ref!r}: cannot import {module_path!r} ({type(exc).__name__})"
        ) from exc
    except Exception as exc:
        raise RosterSourceError(
            f"roster source {ref!r}: importing {module_path!r} raised {type(exc).__name__}"
        ) from exc
    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise RosterSourceError(
            f"roster source {ref!r}: module {module_path!r} has no attribute {attribute!r}"
        ) from exc
    if not isinstance(target, type) and isinstance(target, RosterSource):
        return cast(RosterSource, target)
    if callable(target):
        try:
            made = target()
        except Exception as exc:
            raise RosterSourceError(
                f"roster source {ref!r}: the factory raised {type(exc).__name__}"
            ) from exc
        if not isinstance(made, type) and isinstance(made, RosterSource):
            return cast(RosterSource, made)
        raise RosterSourceError(
            f"roster source {ref!r}: the factory returned {type(made).__name__}, "
            "not a roster source"
        )
    raise RosterSourceError(
        f"roster source {ref!r}: {type(target).__name__} is not a roster source or a factory"
    )


def load_roster_sources(refs: Iterable[str]) -> list[RosterSource]:
    """Resolve every reference, keeping the configured order.

    Two sources with the same ``name`` are refused: the merge reports each source
    by its name, so a shared name would hide which of them stated a row or failed.
    """
    sources: list[RosterSource] = []
    first_ref: dict[str, str] = {}
    for ref in refs:
        source = load_roster_source(ref)
        if source.name in first_ref:
            raise RosterSourceError(
                f"roster sources {first_ref[source.name]!r} and {ref!r} share the name "
                f"{source.name!r}; each source needs its own"
            )
        first_ref[source.name] = ref
        sources.append(source)
    return sources


# --- merging --------------------------------------------------------------------------


def _read_rows(source: RosterSource, deadline: float) -> list[RosterRow]:
    rows = source.read(deadline)
    if not isinstance(rows, list):
        raise TypeError(f"read() returned {type(rows).__name__}, not a list")
    for row in rows:
        if not isinstance(row, RosterRow):
            raise TypeError(f"read() returned a {type(row).__name__}, not a RosterRow")
    return rows


def merge_rosters(sources: Sequence[RosterSource], deadline: float) -> RosterMerge:
    """Read every source and merge the rows into one row per IOC name.

    Sources are read in order, each with the same ``deadline``. For each field the
    first source that states it (a value other than ``None``) wins. A source that
    raises an exception, or returns anything but a list of :class:`RosterRow`, is
    reported ``unavailable`` with the exception's class name as the reason; its rows
    are dropped and the other sources are kept. The reason never carries the
    exception's message, which may hold hosts or credentials.
    """
    outcomes: list[SourceOutcome] = []
    merged: dict[str, dict[str, object]] = {}
    for source in sources:
        try:
            rows = _read_rows(source, deadline)
        except Exception as exc:
            outcomes.append(SourceOutcome(source.name, "unavailable", type(exc).__name__))
            continue
        outcomes.append(SourceOutcome(source.name, "ok", None))
        for row in rows:
            entry = merged.setdefault(
                row.name,
                {"host": None, "class_name": None, "expected": None, "group": None, "sources": []},
            )
            for key in ("host", "class_name", "expected", "group"):
                if entry[key] is None:
                    entry[key] = getattr(row, key)
            named: list[str] = entry["sources"]  # type: ignore[assignment]
            if source.name not in named:
                named.append(source.name)
    rows_out = tuple(
        MergedRosterRow(
            name=name,
            host=entry["host"],  # type: ignore[arg-type]
            class_name=entry["class_name"],  # type: ignore[arg-type]
            expected=entry["expected"],  # type: ignore[arg-type]
            group=entry["group"],  # type: ignore[arg-type]
            sources=tuple(entry["sources"]),  # type: ignore[arg-type]
        )
        for name, entry in sorted(merged.items())
    )
    return RosterMerge(rows=rows_out, sources=tuple(outcomes))


# --- conformance ----------------------------------------------------------------------


def check_roster_source(source: RosterSource, deadline: float = 5.0) -> list[str]:
    """Check a source against the contract; return the problems found, empty if none.

    Reads the source once with ``deadline``. Checks: both members present; ``name``
    a non-blank string; ``read`` returns a list of :class:`RosterRow` within the
    deadline without raising; no IOC name appears twice.
    """
    if not isinstance(source, RosterSource):
        return [f"{type(source).__name__} is not a RosterSource: it needs `name` and `read`"]
    problems: list[str] = []
    name = getattr(source, "name", None)
    if not isinstance(name, str) or not name.strip():
        problems.append(f"name must be a non-blank string, got {name!r}")
    started = time.monotonic()
    try:
        rows = source.read(deadline)
    except Exception as exc:
        problems.append(f"read() raised {type(exc).__name__}: {exc}")
        return problems
    elapsed = time.monotonic() - started
    if elapsed > deadline:
        problems.append(f"read() took {elapsed:.2f} s, past its deadline of {deadline:.2f} s")
    if not isinstance(rows, list):
        problems.append(f"read() must return a list, got {type(rows).__name__}")
        return problems
    seen: set[str] = set()
    reported: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, RosterRow):
            problems.append(f"row {index} is a {type(row).__name__}, not a RosterRow")
            continue
        if row.name in seen and row.name not in reported:
            problems.append(f"IOC {row.name!r} appears more than once")
            reported.add(row.name)
        seen.add(row.name)
    return problems
