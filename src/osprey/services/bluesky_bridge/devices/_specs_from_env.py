"""Parse explicit EPICS PV lists for the substrate scanner out of env vars.

The EPICS substrate branch (``app.py``'s ``_lifespan``) must never import the
virtual-accelerator manifest to discover PVs — the bridge runs in its own
container/venv and cannot import ``osprey.services.virtual_accelerator``. So
the PV list has to come from somewhere the bridge process *can* see: two env
vars, read and parsed here.

Format (stable — the deploy compose and the Phase 3 e2e both depend on these
exact names and this exact syntax):

- ``BLUESKY_EPICS_MOTORS``: comma-separated ``name=SETPOINT_PV`` or
  ``name=SETPOINT_PV|READBACK_PV`` entries.
- ``BLUESKY_EPICS_DETECTORS``: comma-separated ``name=READ_PV`` entries.

Example::

    BLUESKY_EPICS_MOTORS="mot1=RING:SEXT:01:CURRENT:SP|RING:SEXT:01:CURRENT:RB,mot2=RING:SEXT:02:CURRENT:SP"
    BLUESKY_EPICS_DETECTORS="det1=RING:BPM:01:X:RB,det2=RING:BPM:02:X:RB"

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

from .epics import EpicsDetectorSpec, EpicsMotorSpec

logger = logging.getLogger("osprey.services.bluesky_bridge.devices._specs_from_env")

MOTORS_ENV = "BLUESKY_EPICS_MOTORS"
"""Env var carrying the motor PV list (see module docstring for format)."""

DETECTORS_ENV = "BLUESKY_EPICS_DETECTORS"
"""Env var carrying the detector PV list (see module docstring for format)."""


def _split_entries(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_motor_specs(raw: str) -> list[EpicsMotorSpec]:
    """Parse ``BLUESKY_EPICS_MOTORS``-shaped text into ``EpicsMotorSpec``\\ s.

    Each entry is ``name=SETPOINT_PV`` or ``name=SETPOINT_PV|READBACK_PV``.
    A malformed entry (no ``=``, empty name, empty setpoint PV, or more than
    one ``|``) is skipped with a warning log rather than raising — one typo
    in a long PV list should not prevent every other device from connecting.
    """
    specs: list[EpicsMotorSpec] = []
    for entry in _split_entries(raw):
        name, sep, spec_text = entry.partition("=")
        name = name.strip()
        if not sep or not name:
            logger.warning(
                "%s: skipping malformed motor entry %r (expected name=SP_PV)", MOTORS_ENV, entry
            )
            continue

        pv_parts = spec_text.split("|")
        if len(pv_parts) > 2:
            logger.warning(
                "%s: skipping malformed motor entry %r (more than one '|')", MOTORS_ENV, entry
            )
            continue

        setpoint_pv = pv_parts[0].strip()
        readback_pv = pv_parts[1].strip() if len(pv_parts) == 2 else ""
        if not setpoint_pv:
            logger.warning("%s: skipping motor entry %r (empty setpoint PV)", MOTORS_ENV, entry)
            continue
        if len(pv_parts) == 2 and not readback_pv:
            logger.warning(
                "%s: skipping motor entry %r (empty readback PV after '|')", MOTORS_ENV, entry
            )
            continue

        specs.append(
            EpicsMotorSpec(name=name, setpoint_pv=setpoint_pv, readback_pv=readback_pv or None)
        )
    return specs


def parse_detector_specs(raw: str) -> list[EpicsDetectorSpec]:
    """Parse ``BLUESKY_EPICS_DETECTORS``-shaped text into ``EpicsDetectorSpec``\\ s.

    Each entry is ``name=READ_PV``. A malformed entry (no ``=``, empty name,
    or empty PV) is skipped with a warning log — see ``parse_motor_specs``.
    """
    specs: list[EpicsDetectorSpec] = []
    for entry in _split_entries(raw):
        name, sep, read_pv = entry.partition("=")
        name = name.strip()
        read_pv = read_pv.strip()
        if not sep or not name or not read_pv:
            logger.warning(
                "%s: skipping malformed detector entry %r (expected name=READ_PV)",
                DETECTORS_ENV,
                entry,
            )
            continue
        specs.append(EpicsDetectorSpec(name=name, read_pv=read_pv))
    return specs


def _drop_duplicate_names(
    motors: list[EpicsMotorSpec], detectors: list[EpicsDetectorSpec]
) -> tuple[list[EpicsMotorSpec], list[EpicsDetectorSpec]]:
    """Drop any spec whose device name was already seen (motors first, then
    detectors), warning on each collision.

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
                    "skipping device %r: name already claimed by an earlier motor/detector entry",
                    spec.name,
                )
                continue
            seen.add(spec.name)
            kept.append(spec)
        return kept

    return _keep(motors), _keep(detectors)


def specs_from_env(env: Mapping[str, str]) -> tuple[list[EpicsMotorSpec], list[EpicsDetectorSpec]]:
    """Read and parse both PV-list env vars from ``env`` (typically ``os.environ``).

    Returns ``([], [])`` when a var is absent or empty — an empty device set is
    a valid (if useless) configuration, not an error here; the *caller* (which
    knows the substrate is enabled) is responsible for warning that an enabled
    substrate wired nothing. Any device name that collides with an earlier one
    is dropped with a warning (``_drop_duplicate_names``).
    """
    motors = parse_motor_specs(env.get(MOTORS_ENV, ""))
    detectors = parse_detector_specs(env.get(DETECTORS_ENV, ""))
    return _drop_duplicate_names(motors, detectors)
