"""Which PV is an IOC's liveness or health PV: one rule for the IOC page and the fleet.

A role (``heartbeat``, ``uptime``, ``engineer``, or a configured class's own) is
found in the channel directory by a **selector**. On a simulator directory a role that
is one of the simulated IOC health signals is selected by the served-signal
property the simulator marks its row with, so a facility's custom suffix never
blanks the simulated fleet; every other role, and every role on any other
directory, is selected by its record-name suffix.

The IOC page runs each selector filtered to one IOC; the fleet refresher runs the
same selectors unfiltered, one paged search per selector. Either way
:func:`match_rows` assigns each returned row its role and IOC, and
:func:`choose` settles several matches for one IOC and role: the shortest name
wins, ties by name, and the choice says it was ambiguous.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from osprey.services.channel_directory._connectors import (
    SERVED_SIGNAL_PROPERTY,
    SERVER_HEALTH_SIGNALS,
)

__all__ = [
    "SIMULATED_SIGNALS",
    "SIMULATOR_DIRECTORY",
    "HealthChoice",
    "HealthMatch",
    "HealthSelector",
    "choose",
    "health_selectors",
    "match_rows",
    "role_of",
]

#: The directory kind of a target whose channel directory is the simulator's view.
SIMULATOR_DIRECTORY = "simulator_view"


def role_of(signal: str) -> str:
    """The IOC health role a simulator health signal serves: its word, lower case."""
    return signal.lower()


#: Each IOC health role a simulator serves, and the signal its rows are marked with.
SIMULATED_SIGNALS = {role_of(signal): signal for signal in SERVER_HEALTH_SIGNALS}


@dataclass(frozen=True)
class HealthSelector:
    """How the directory finds the PVs of one role: exactly one of a signal or a suffix.

    Attributes:
        role: The role the selected PVs serve.
        signal: The served signal a simulator directory marks the role's rows with.
        suffix: The record-name suffix that ends the role's PV names.
    """

    role: str
    signal: str | None = None
    suffix: str | None = None

    def __post_init__(self) -> None:
        if (self.signal is None) == (self.suffix is None):
            raise ValueError(f"selector for {self.role!r} names exactly one of signal and suffix")

    def query(self) -> dict[str, Any]:
        """The search fields that select the role's PVs, beside any IOC filter and page size."""
        if self.signal is not None:
            return {"properties": {SERVED_SIGNAL_PROPERTY: self.signal}}
        return {"name": f"*{self.suffix}"}

    def admits(self, name: str) -> bool:
        """Whether a row named *name*, returned by this selector's search, serves the role.

        A signal selector's search matched the mark exactly; a name glob may match
        more loosely than the suffix, so the suffix must end the name exactly.
        """
        return self.suffix is None or name.endswith(self.suffix)


@dataclass(frozen=True)
class HealthMatch:
    """One row a selector returned: the PV, the role it serves, the IOC serving it."""

    server: str | None
    role: str
    name: str


@dataclass(frozen=True)
class HealthChoice:
    """The PV chosen for one IOC and role, and every name that matched.

    ``ambiguous`` is true when more than one distinct name matched.
    """

    name: str
    ambiguous: bool
    candidates: tuple[str, ...]


def health_selectors(directory_kind: str, roles: Mapping[str, str]) -> tuple[HealthSelector, ...]:
    """The selectors for *roles* (role to configured suffix) on a directory of *directory_kind*.

    On :data:`SIMULATOR_DIRECTORY` a role that is a simulated IOC health signal
    is selected by that signal and its suffix is not used; any other role, and
    every role elsewhere, by its suffix. One selector per role, in *roles* order.
    """
    simulator = directory_kind == SIMULATOR_DIRECTORY
    return tuple(
        HealthSelector(role=role, signal=SIMULATED_SIGNALS[role])
        if simulator and role in SIMULATED_SIGNALS
        else HealthSelector(role=role, suffix=suffix)
        for role, suffix in roles.items()
    )


def match_rows(selector: HealthSelector, rows: Iterable[Mapping[str, Any]]) -> list[HealthMatch]:
    """The rows *selector*'s search returned that serve its role, each with its IOC.

    A row's IOC is its ``server``, ``None`` when the directory names none.
    """
    return [
        HealthMatch(server=row.get("server"), role=selector.role, name=row["name"])
        for row in rows
        if selector.admits(row["name"])
    ]


def choose(matches: Iterable[HealthMatch]) -> dict[tuple[str, str], HealthChoice]:
    """One PV per IOC and role among *matches*: the shortest name, ties by name.

    Matches without an IOC are not chosen; the same name matched twice is one candidate.
    """
    found: dict[tuple[str, str], set[str]] = {}
    for match in matches:
        if match.server is not None:
            found.setdefault((match.server, match.role), set()).add(match.name)
    chosen: dict[tuple[str, str], HealthChoice] = {}
    for key, names in found.items():
        ranked = tuple(sorted(names, key=lambda name: (len(name), name)))
        chosen[key] = HealthChoice(name=ranked[0], ambiguous=len(ranked) > 1, candidates=ranked)
    return chosen
