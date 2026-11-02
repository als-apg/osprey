"""The CANOPY this OSPREY is proven against, and whether it is the one installed.

This module is the only reader of which CANOPY and PyDMConverter are installed
and whether they are the proven commits, and the only formatter of the line
that says so. It imports nothing outside the standard library, so ``--pins``
runs by path with a bare interpreter before any environment exists, and prints
the two pins, one ``name=commit`` per line::

    python3 src/osprey/integrations/canopy/proven.py --pins

The judgement runs as the package inside the environment whose CANOPY it
judges, and prints the line: exit 0 when both are at the pin, 1 when anything
else is installed, 2 when neither is::

    python -m osprey.integrations.canopy

An installed distribution is at a commit only when its ``direct_url.json``
records that commit under ``vcs_info.commit_id``, which a ``git+…@<commit>``
install does. An editable, path, wheel or index install records none, and its
version's ``g<hex>`` label is never read: that label is frozen when the
install is built, and an editable tree moves on without it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

#: The CANOPY and PyDMConverter commits this OSPREY's CANOPY contract
#: (tests/integrations/canopy) passes against, keyed by distribution name.
#: The canopy-contract check runs against exactly these; a payload and the shipped
#: library's report must name them.
PROVEN_COMMITS: dict[str, str] = {
    "canopy": "59cf29f9d783adb1060e495ba5de95dd10b042f7",  # osprey:not-a-port (a sha)
    "PyDMConverter": "69d41977bbe53e456220b2f8ca3cb73057cd21ea",
}

#: The ``STAGED.json`` key that names each distribution's commit.
STAGED_KEYS = {"canopy": "canopy", "PyDMConverter": "pydm"}

#: Why an installed distribution has no commit.
NOT_INSTALLED = "not installed"
EDITABLE = "editable, records no commit"
NO_RECORD = "installed without a recorded commit"

#: The states of a judgement.
PROVEN = "proven"
UNPROVEN = "unproven"
ABSENT = "absent"

#: How much of a commit a line shows where it does not attest the pair.
_SHOWN_HEX = 12

_VCS_LABEL = re.compile(r"g([0-9a-f]+)")
_DIRTY_LABEL = re.compile(r"d\d{8}")


@dataclass(frozen=True)
class Held:
    """What an installed distribution holds: a recorded commit, or why it has none."""

    commit: str | None = None
    reason: str | None = None

    def shown(self) -> str:
        return self.commit[:_SHOWN_HEX] if self.commit else str(self.reason)


@dataclass(frozen=True)
class Reading:
    """One distribution of the proven pair, as installed."""

    distribution: str
    proven: str
    held: Held
    at_pin: bool


@dataclass(frozen=True)
class Judgement:
    """The installed pair against the proven one: one reading each, and a state."""

    readings: tuple[Reading, ...]
    state: str


def installed(distribution: str) -> Held:
    """The commit the installed *distribution* records, or why it records none."""
    try:
        dist = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return Held(reason=NOT_INSTALLED)
    raw = dist.read_text("direct_url.json")
    try:
        record = json.loads(raw) if raw else None
    except ValueError:
        record = None
    if not isinstance(record, dict):
        return Held(reason=NO_RECORD)
    vcs_info = record.get("vcs_info")
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if isinstance(commit, str) and commit:
        return Held(commit=commit)
    dir_info = record.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        return Held(reason=EDITABLE)
    return Held(reason=NO_RECORD)


def version_commit(version: str) -> str | None:
    """The commit a clean ``+g<hex>`` version label names, abbreviated as it is."""
    _, plus, local = version.partition("+")
    if not plus:
        return None
    labels = local.split(".")
    if any(_DIRTY_LABEL.fullmatch(label) for label in labels):
        return None
    match = _VCS_LABEL.fullmatch(labels[0])
    return match.group(1) if match else None


def is_pin(distribution: str, commit: str) -> bool:
    """Whether *commit* is the proven commit of *distribution*, compared whole."""
    return commit == PROVEN_COMMITS[distribution]


def version_names_pin(distribution: str, version: str) -> bool:
    """Whether the version string a CANOPY artefact carries names the pin.

    This reads a version recorded in an artefact, such as a library report's
    generator version, never an install. The version names a commit by a clean
    ``g<hex>`` local label (no ``d<date>`` label for uncommitted changes); it
    names the pin when that hex is a non-empty prefix of it.
    """
    commit = version_commit(version)
    return commit is not None and PROVEN_COMMITS[distribution].startswith(commit)


def judge() -> Judgement:
    """This interpreter's installs of the pair, judged against the pins."""
    readings = []
    for distribution, proven in PROVEN_COMMITS.items():
        held = installed(distribution)
        at_pin = held.commit is not None and held.commit == proven
        readings.append(Reading(distribution, proven, held, at_pin))
    if all(reading.at_pin for reading in readings):
        state = PROVEN
    elif all(reading.held.reason == NOT_INSTALLED for reading in readings):
        state = ABSENT
    else:
        state = UNPROVEN
    return Judgement(tuple(readings), state)


def venue() -> str:
    """The platform this interpreter runs on, as the attestation names it."""
    return f"{platform.system()} {platform.machine()}"


def line(judgement: Judgement, ran: bool | None) -> str:
    """The one line that says what *judgement* means for the contract.

    *ran* is whether the contract's tests passed: ``True`` passed, ``False``
    failed, ``None`` not run (a probe of the installs alone). A line that says
    the tests passed or failed against the pair also names the platform they
    ran on, because a proof holds for a pair on a platform.
    """
    readings = judgement.readings
    if judgement.state == PROVEN:
        pair = ", ".join(f"{r.distribution} {r.proven}" for r in readings)
        if ran is None:
            return f"CANOPY at the proven commits: {pair}"
        verdict = "proven" if ran else "failed"
        return f"CANOPY contract {verdict} against {pair} on {venue()}"
    pins = ", ".join(f"{r.distribution} {r.proven[:_SHOWN_HEX]}" for r in readings)
    if judgement.state == ABSENT:
        found = "not run here: CANOPY is not installed"
    else:
        held = ", ".join(f"{r.distribution} {r.held.shown()}" for r in readings)
        found = f"proves nothing here: {held}"
    return f"CANOPY contract {found}; this OSPREY is proven against {pins}"


def attestation(passed: bool) -> str:
    """The line a run of the contract against the proven pair ends with.

    The caller has judged the installs at the pins; this renders that
    judgement from the pins, so a caller outside the installed environment
    words it the same way.
    """
    readings = tuple(
        Reading(distribution, commit, Held(commit=commit), at_pin=True)
        for distribution, commit in PROVEN_COMMITS.items()
    )
    return line(Judgement(readings, PROVEN), ran=passed)


_EXIT = {PROVEN: 0, UNPROVEN: 1, ABSENT: 2}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.integrations.canopy",
        description="Judge the installed CANOPY against the commits this OSPREY is proven against.",
    )
    parser.add_argument(
        "--pins", action="store_true", help="print the proven commits, one name=commit per line"
    )
    args = parser.parse_args(argv)
    if args.pins:
        for distribution, commit in PROVEN_COMMITS.items():
            sys.stdout.write(f"{distribution}={commit}\n")
        return 0
    judgement = judge()
    sys.stdout.write(line(judgement, ran=None) + "\n")
    return _EXIT[judgement.state]


if __name__ == "__main__":
    sys.exit(main())
