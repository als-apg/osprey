"""Parse explicit EPICS PV lists for the substrate runner out of env vars.

The EPICS substrate branch (``app.py``'s ``_lifespan``) must never import the
virtual-accelerator manifest to discover PVs — the bridge runs in its own
container/venv and cannot import ``osprey.services.virtual_accelerator``. So
the PV list has to come from somewhere the bridge process *can* see: two env
vars, read and parsed here.

The var NAMES are OSPREY's own and may be renamed; every reader reaches them
through the constants below (the deploy writer, the compose passthroughs, and
the substrate e2e all import rather than spell them). What is **stable** is the
per-entry SYNTAX, which is a wire format written by one process and read by
another:

- setpoints: comma-separated ``name=SETPOINT_PV`` or
  ``name=SETPOINT_PV|READBACK_PV`` entries.
- readbacks: comma-separated ``name=READ_PV`` entries.

Example::

    BLUESKY_EPICS_SETPOINTS="sp1=RING:SEXT:01:CURRENT:SP|RING:SEXT:01:CURRENT:RB,sp2=RING:SEXT:02:CURRENT:SP"
    BLUESKY_EPICS_READBACKS="rb1=RING:BPM:01:X:RB,rb2=RING:BPM:02:X:RB"

A pipe (``|``), not a colon, separates the setpoint PV from an optional
readback PV: OSPREY EPICS addresses are themselves colon-delimited
(``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD``), so a colon separator would be
ambiguous against the PV names this is meant to carry. Neither ``|`` nor
``=`` nor ``,`` is a legal EPICS PV name character, so the format is
unambiguous without any escaping.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from .specs import ReadableSpec, SettableSpec

logger = logging.getLogger("osprey.services.bluesky_bridge.devices._specs_from_env")

SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"
"""Env var enabling the EPICS substrate branch; the two PV lists below are read
only when it is set. Defined here, beside them, so the trio has one home."""

SETPOINTS_ENV = "BLUESKY_EPICS_SETPOINTS"
"""Env var carrying the setpoint PV list (see module docstring for format)."""

READBACKS_ENV = "BLUESKY_EPICS_READBACKS"
"""Env var carrying the readback PV list (see module docstring for format)."""


def _split_entries(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_setpoint_specs(raw: str) -> list[SettableSpec]:
    """Parse ``BLUESKY_EPICS_SETPOINTS``-shaped text into ``SettableSpec``\\ s.

    Each entry is ``name=SETPOINT_PV`` or ``name=SETPOINT_PV|READBACK_PV``.
    A malformed entry (no ``=``, empty name, empty setpoint PV, or more than
    one ``|``) is skipped with a warning log rather than raising — one typo
    in a long PV list should not prevent every other device from connecting.
    """
    specs: list[SettableSpec] = []
    for entry in _split_entries(raw):
        name, sep, spec_text = entry.partition("=")
        name = name.strip()
        if not sep or not name:
            logger.warning(
                "%s: skipping malformed setpoint entry %r (expected name=SP_PV)",
                SETPOINTS_ENV,
                entry,
            )
            continue

        pv_parts = spec_text.split("|")
        if len(pv_parts) > 2:
            logger.warning(
                "%s: skipping malformed setpoint entry %r (more than one '|')",
                SETPOINTS_ENV,
                entry,
            )
            continue

        setpoint_pv = pv_parts[0].strip()
        readback_pv = pv_parts[1].strip() if len(pv_parts) == 2 else ""
        if not setpoint_pv:
            logger.warning(
                "%s: skipping setpoint entry %r (empty setpoint PV)", SETPOINTS_ENV, entry
            )
            continue
        if len(pv_parts) == 2 and not readback_pv:
            logger.warning(
                "%s: skipping setpoint entry %r (empty readback PV after '|')",
                SETPOINTS_ENV,
                entry,
            )
            continue

        specs.append(
            SettableSpec(name=name, setpoint_pv=setpoint_pv, readback_pv=readback_pv or None)
        )
    return specs


def parse_readback_specs(raw: str) -> list[ReadableSpec]:
    """Parse ``BLUESKY_EPICS_READBACKS``-shaped text into ``ReadableSpec``\\ s.

    Each entry is ``name=READ_PV``. A malformed entry (no ``=``, empty name,
    or empty PV) is skipped with a warning log — see ``parse_setpoint_specs``.
    """
    specs: list[ReadableSpec] = []
    for entry in _split_entries(raw):
        name, sep, read_pv = entry.partition("=")
        name = name.strip()
        read_pv = read_pv.strip()
        if not sep or not name or not read_pv:
            logger.warning(
                "%s: skipping malformed readback entry %r (expected name=READ_PV)",
                READBACKS_ENV,
                entry,
            )
            continue
        specs.append(ReadableSpec(name=name, read_pv=read_pv))
    return specs


def _drop_duplicate_names(
    setpoints: list[SettableSpec], readbacks: list[ReadableSpec]
) -> tuple[list[SettableSpec], list[ReadableSpec]]:
    """Drop any spec whose device name was already seen (setpoints first, then
    readbacks), warning on each collision.

    Device names become ophyd-async device names *and* event-data column keys;
    two devices sharing a name would make the scanned column ambiguous (see the
    bridge's device-column lookup), so a later entry that reuses an
    already-claimed name is dropped rather than silently shadowing the first.
    """
    seen: set[str] = set()

    def _keep(specs: list) -> list:
        kept = []
        for spec in specs:
            if spec.name in seen:
                logger.warning(
                    "skipping device %r: name already claimed by an earlier "
                    "setpoint/readback entry",
                    spec.name,
                )
                continue
            seen.add(spec.name)
            kept.append(spec)
        return kept

    return _keep(setpoints), _keep(readbacks)


def specs_from_env(env: Mapping[str, str]) -> tuple[list[SettableSpec], list[ReadableSpec]]:
    """Read and parse both PV-list env vars from ``env`` (typically ``os.environ``).

    Returns ``([], [])`` when a var is absent or empty — an empty device set is
    a valid (if useless) configuration, not an error here; the *caller* (which
    knows the substrate is enabled) is responsible for warning that an enabled
    substrate wired nothing. Any device name that collides with an earlier one
    is dropped with a warning (``_drop_duplicate_names``).
    """
    setpoints = parse_setpoint_specs(env.get(SETPOINTS_ENV, ""))
    readbacks = parse_readback_specs(env.get(READBACKS_ENV, ""))
    return _drop_duplicate_names(setpoints, readbacks)
