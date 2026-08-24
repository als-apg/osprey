"""Read/write direction for the demo-machine knowledge graph.

Every ``narad_sem:SemanticSignal`` the generator mints is reached from its
bindings by exactly one predicate — ``narad_p:writesSignal`` or
``narad_p:readsSignal``.  This module decides which, and it decides it once per
``(FAMILY, FIELD, SUBFIELD)`` signal group rather than per channel, because a
group whose members disagree would need two signals to describe one physical
quantity.

There are two sources, in priority order:

1. **The channel limits file** — the preset's enforced writability ground
   truth.  A channel is writable when the defaults-merged entry in
   ``channel_limits.json`` says so, which is the same rule the runtime write
   path applies (see
   :meth:`osprey_connectors.control_system.limits_validator.LimitsValidator._load_limits_database`).
   This is authoritative: the demo machine marks the twelve gate valves'
   ``CONTROL:OPEN`` / ``CONTROL:CLOSE`` channels ``writable: false`` even though
   they read like commands, and the graph must agree with the control system
   that refuses those writes.
2. **The PV grammar** — ``subfield == "SP"`` means write, everything else
   reads.  Used only when no limits file is available at all.  It is
   group-constant by construction, since the subfield is part of the group key.

The two sources agree on the shipped demo data (the 396 writable channels are
exactly the ``:SP`` set); the group-constant check turns that agreement into an
asserted invariant instead of an unstated assumption.

This module is standard-library plus in-repo imports only: no ``rdflib``, no
``neo4j``, and no I/O beyond reading the limits JSON.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# ``osprey.utils.config`` is a compatibility shim whose module object *is*
# ``osprey_connectors.config``; importing from the real module keeps the two
# connector-side helpers below spelled from one place.
from osprey_connectors.config import default_config_path, get_config_value
from osprey_connectors.control_system.limits_validator import LimitsValidator

from .model import GraphModel

#: Direction of a signal group whose channels are written.
DIRECTION_WRITE = "write"

#: Direction of a signal group whose channels are only read.
DIRECTION_READ = "read"

#: Subfield token the PV grammar treats as a setpoint.
WRITE_SUBFIELD = "SP"

#: Config key naming the channel limits database, shared with the write path.
LIMITS_DATABASE_CONFIG_KEY = "control_system.limits_checking.database_path"

#: Config key naming the project root, the last-resort path anchor.
PROJECT_ROOT_CONFIG_KEY = "project_root"

#: Keys in the limits file that name no channel.
_LIMITS_DEFAULTS_KEY = "defaults"

#: How many dissenting addresses a conflict message spells out before eliding.
_MAX_NAMED_DISSENTERS = 10


class DirectionSource(StrEnum):
    """Where a run's read/write directions came from."""

    #: Derived from a channel limits file (authoritative).
    LIMITS = "limits"

    #: Derived from the PV grammar because no limits file was available.
    GRAMMAR = "grammar"


@dataclass(frozen=True)
class DirectionReport:
    """What the generator should tell the operator about its direction source.

    Args:
        source: Which rule produced the directions.
        limits_path: The limits file that was read, or ``None`` for the
            grammar fallback.
        message: The single operator-facing line the CLI prints.
    """

    source: DirectionSource
    limits_path: Path | None
    message: str


class DirectionConflictError(ValueError):
    """Raised when a signal group's channels do not share one direction.

    Args:
        group_key: The ``(FAMILY, FIELD, SUBFIELD)`` group that disagrees.
        by_direction: Member addresses grouped by the direction they resolved
            to.  Must contain more than one direction.

    Attributes:
        group_key: As above.
        by_direction: As above, with each address sequence sorted.
        majority: The direction the largest share of members resolved to.
        dissenting: Addresses that resolved to any other direction, sorted.
    """

    def __init__(
        self,
        group_key: tuple[str, str, str],
        by_direction: Mapping[str, Sequence[str]],
    ) -> None:
        self.group_key = group_key
        self.by_direction = {
            direction: tuple(sorted(addresses)) for direction, addresses in by_direction.items()
        }
        # Ties break on the direction name so the message is deterministic.
        self.majority = max(
            sorted(self.by_direction), key=lambda direction: len(self.by_direction[direction])
        )
        self.dissenting = tuple(
            sorted(
                address
                for direction, addresses in self.by_direction.items()
                if direction != self.majority
                for address in addresses
            )
        )
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        family, field, subfield = self.group_key
        census = ", ".join(
            f"{len(self.by_direction[direction])} {direction}"
            for direction in sorted(self.by_direction)
        )
        return (
            f"Signal group ({family}, {field}, {subfield}) has no constant direction: "
            f"{census}. A (FAMILY, FIELD, SUBFIELD) group mints one SemanticSignal, so "
            f"its channels must all read or all write. Dissenting from '{self.majority}': "
            f"{_format_addresses(self.dissenting)}"
        )


class LimitsFileMissing(FileNotFoundError):
    """Raised when an explicitly requested limits file does not exist."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Channel limits file not found: {path}. The --limits path was given "
            "explicitly, so the generator will not fall back to the PV grammar."
        )


def _format_addresses(addresses: Sequence[str]) -> str:
    """Join *addresses* for a message, eliding past :data:`_MAX_NAMED_DISSENTERS`."""
    named = ", ".join(addresses[:_MAX_NAMED_DISSENTERS])
    remainder = len(addresses) - _MAX_NAMED_DISSENTERS
    if remainder > 0:
        return f"{named} (+{remainder} more)"
    return named


def resolve_limits_path(
    explicit: Path | None,
    *,
    config_lookup: Callable[[str, Any], Any] | None = None,
) -> tuple[Path | None, str]:
    """Find the channel limits file to derive directions from.

    An explicit path wins outright and must exist — falling back to the PV
    grammar after the operator named a file would silently ignore what they
    asked for.  Otherwise the config key is read and resolved exactly the way
    the runtime write path resolves it, so both readers of that key agree on
    which file they mean.

    Args:
        explicit: Path passed on the command line, or ``None``.
        config_lookup: Callable with the signature of
            :func:`osprey_connectors.config.get_config_value`, injectable for
            tests.  Defaults to that function.

    Returns:
        Tuple of (path, message).  The path is ``None`` when no limits file is
        available; the message says what was looked for either way and is meant
        to be folded into the operator-facing report line.

    Raises:
        LimitsFileMissing: If *explicit* is given and names no existing file.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise LimitsFileMissing(path)
        return path, f"limits file given explicitly: {path}"

    lookup = get_config_value if config_lookup is None else config_lookup
    configured = lookup(LIMITS_DATABASE_CONFIG_KEY, None)
    if not configured or not isinstance(configured, str):
        return None, f"{LIMITS_DATABASE_CONFIG_KEY} is not set in the OSPREY config"

    project_root = lookup(PROJECT_ROOT_CONFIG_KEY, None)
    resolved = Path(
        LimitsValidator.resolve_database_path(
            configured,
            project_root if isinstance(project_root, str) else None,
            config_path=default_config_path(),
        )
    )
    if not resolved.is_file():
        return None, f"no limits file found at {resolved} (from {LIMITS_DATABASE_CONFIG_KEY})"
    return resolved, f"limits file from {LIMITS_DATABASE_CONFIG_KEY}: {resolved}"


def load_writable_set(limits_path: Path) -> frozenset[str]:
    """Read the addresses a channel limits file declares writable.

    Writability is defaults-aware: the demo file grants it by *omitting* the
    ``writable`` key so the entry inherits ``defaults.writable: true``.  Reading
    only explicit ``writable: true`` entries would find none at all.

    Args:
        limits_path: Path to a ``channel_limits.json``-shaped file.

    Returns:
        The writable addresses.

    Raises:
        ValueError: If the file is not JSON, or is not a JSON object.
    """
    try:
        raw = json.loads(Path(limits_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Channel limits file {limits_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"Channel limits file {limits_path} must contain a JSON object keyed by "
            f"channel address, got {type(raw).__name__}"
        )

    defaults = raw.get(_LIMITS_DEFAULTS_KEY)
    defaults = defaults if isinstance(defaults, dict) else {}
    writable = {
        address
        for address, entry in raw.items()
        if not address.startswith("_")
        and address != _LIMITS_DEFAULTS_KEY
        and {**defaults, **(entry if isinstance(entry, dict) else {})}.get("writable", True)
    }
    return frozenset(writable)


def assign_directions(
    model: GraphModel,
    limits_path: Path | None,
    *,
    lookup_message: str = "",
) -> tuple[GraphModel, DirectionReport]:
    """Fill in every signal group's direction.

    Args:
        model: The model to annotate; it is not mutated.
        limits_path: Limits file to derive directions from, or ``None`` to use
            the PV grammar.
        lookup_message: What :func:`resolve_limits_path` reported, folded into
            the grammar-fallback message so the operator can see which file was
            looked for.

    Returns:
        Tuple of (annotated model, report).

    Raises:
        DirectionConflictError: If a signal group's members do not share one
            limits-derived direction.
        ValueError: If the limits file is unreadable (see
            :func:`load_writable_set`).
    """
    if limits_path is None:
        directions = {
            group.key: (DIRECTION_WRITE if group.subfield == WRITE_SUBFIELD else DIRECTION_READ)
            for group in model.signal_groups
        }
        grammar = f"direction from PV grammar (subfield == {WRITE_SUBFIELD} → write)"
        message = f"{grammar}; {lookup_message}" if lookup_message else grammar
        return (
            model.with_directions(directions),
            DirectionReport(source=DirectionSource.GRAMMAR, limits_path=None, message=message),
        )

    path = Path(limits_path)
    writable = load_writable_set(path)
    directions: dict[tuple[str, str, str], str] = {}
    for group in model.signal_groups:
        by_direction: dict[str, list[str]] = {}
        for address in group.members:
            direction = DIRECTION_WRITE if address in writable else DIRECTION_READ
            by_direction.setdefault(direction, []).append(address)
        if len(by_direction) > 1:
            raise DirectionConflictError(group.key, by_direction)
        if by_direction:
            directions[group.key] = next(iter(by_direction))

    return (
        model.with_directions(directions),
        DirectionReport(
            source=DirectionSource.LIMITS,
            limits_path=path,
            message=f"direction from channel limits: {path}",
        ),
    )


def resolve_and_assign(
    model: GraphModel,
    explicit_limits: Path | None,
    *,
    config_lookup: Callable[[str, Any], Any] | None = None,
) -> tuple[GraphModel, DirectionReport]:
    """Resolve the limits file and annotate *model* in one call.

    This is the shape the ``osprey knowledge build-ttl`` verb wants: one call
    that yields the annotated model plus the line to print.

    Args:
        model: The model to annotate.
        explicit_limits: ``--limits`` path, or ``None``.
        config_lookup: See :func:`resolve_limits_path`.

    Returns:
        Tuple of (annotated model, report).

    Raises:
        LimitsFileMissing: If *explicit_limits* names no existing file.
        DirectionConflictError: If a signal group's direction is not constant.
    """
    limits_path, lookup_message = resolve_limits_path(explicit_limits, config_lookup=config_lookup)
    return assign_directions(model, limits_path, lookup_message=lookup_message)
