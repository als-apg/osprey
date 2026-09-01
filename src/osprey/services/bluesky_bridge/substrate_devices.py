"""Canonical derivation of the bluesky bridge's EPICS-substrate plan devices.

Single source of truth for turning the facility's channel roster (the
:class:`~osprey.channel_roster.records.ChannelRecord` sequence
:func:`osprey.channel_roster.registered_channels` returns) into the bridge's
device set, in the two-list device-file format the queueserver worker reads
(see ``osprey.services.bluesky_bridge.devices._specs_from_file`` for the schema
and the parser).

Every channel the roster enumerated becomes a device: each write-direction
record a settable, each read-direction record a readable. Nothing here filters
by ring, family, field or address grammar. Which channels exist and which way
they point is the roster's answer, given once per build from the facility's own
knowledge graph or channel-finder database; a second opinion at this seam is
precisely the divergence that had a build report 144 devices for a 2908-channel
machine.

Device name == channel address
------------------------------

Each device is keyed by the address it drives or reads -- a settable by its
setpoint address, a readable by its read address -- not by a synthetic
``corrector_01``/``bpm_01`` label. This is deliberate: channel-finder output
IS the worker namespace. The addresses an agent discovers are exactly the
names a plan may reference, so there is no second, agent-invisible namespace
to translate through and no discovery surface that has to be kept in sync
with this derivation.

The conscious trade-off is in queueserver's device-permission patterns
(``user_group_permissions.yaml``'s ``allowed_devices``/``forbidden_devices``).
``:`` is that mini-language's own component separator and is not escapable,
so an address cannot be written *literally* in a rule. A rule may still
target exactly one address-named device, but only by wildcarding each colon
--- ``:?^SR.MAG.COIL.01.CURRENT.SP$:depth=1`` selects that one device, since
``.`` matches ``:`` --- or by falling back to a catch-all like
``:?.*:depth=5``, which is what this project ships.

Beware the fail-open trap when writing such a rule: a pattern that attempts
a literal (or backslash-escaped) colon address raises inside
``load_allowed_plans_and_devices``, which catches every exception and falls
back to the *unfiltered* device set -- so an operator reaching for a tighter
rule can silently end up with allow-everything.

That is accepted rather than worked around: as ``user_group_permissions.
yaml``'s own header states, the permission layer is not the safety boundary
and must not be mistaken for one. Every write a plan performs still passes
the connector's per-put reference monitor and the bridge's arming + limits
facade, which are the boundary.

Deriving a device set for a live lane is deliberate
---------------------------------------------------

The derived set is the whole machine, and it is staged for a live lane as
readily as for a virtual one. That is a considered position rather than an
oversight: a device in the worker's namespace is a name a plan MAY reference,
never a write that has happened. The gates that decide whether a write lands
sit on the write path -- the connector's per-put reference monitor and the
bridge's arming + limits facade -- and the build refuses to stage a derived set
at all for a lane whose target has writes enabled without an enabled limits
posture. Withholding the machine's own channels from the namespace would add no
gate; it would only make the channels an agent is allowed to read invisible to
it, and push operators back to hand-authored device files that nothing keeps in
step with the facility.

Two consumers share this module (DRY, one derivation):

- ``osprey.deployment.compose_generator`` (``_stage_bluesky_devices``), which
  derives and stages the device file on every render, so the worker starts with
  real channel names, turn-key.
- ``tests/e2e/_orm_stack.py``, which selects the subset of records its plans
  need and hands them here for the document and its atomic write, rather than
  assembling a device file of its own.

There is exactly one producer of the derived document -- ``devices_document``
below -- so the build path and the e2e harness can never drift on what the
worker is handed.

Host/deploy-side only — NOT part of the bridge's own container import
surface. This module consumes :mod:`osprey.channel_roster`, which reads the
build's roster sources off disk during the build window; nothing under
``osprey.services.bluesky_bridge`` that runs *inside* the bridge container
(``app.py``, ``devices/*``) may import this module. It lives alongside the
bridge's device code only because it is conceptually about the bridge's
devices, not because it shares the bridge's runtime import surface. It runs
only from the host-side deploy/CLI process and from tests.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import yaml

from osprey.channel_roster import ChannelRecord, RosterSource

# Key names of the device-file document the worker parses. Imported rather
# than restated so the host-side producer and the container-side consumer can
# never drift on the schema.
from osprey.services.bluesky_bridge.devices._specs_from_file import (
    READABLES_KEY,
    SETTABLES_KEY,
)


def devices_document(records: Sequence[ChannelRecord]) -> dict[str, list[dict[str, str]]]:
    """Build the worker's device document from the roster's ``records``.

    Returns the two-list mapping ``_specs_from_file`` parses: one ``settables``
    entry (``name``/``setpoint``, plus ``readback`` where the roster paired one)
    per write-direction record, and one ``readables`` entry (``name``/``pv``)
    per read-direction record, in roster order. Both keys are always present,
    even when empty, so a caller can see *which* half a facility yielded nothing
    for rather than inferring it from an absent key.

    ``readback`` is emitted only for a settable the roster actually paired: an
    unpaired setpoint gets no ``readback`` key at all, which is how the loader
    is told the device reads its setpoint back. Writing ``readback: null`` would
    mean the same thing to the parser but reads as "unset by mistake", and
    restating the setpoint as its own readback would claim a pairing the roster
    did not make.

    A record whose direction the source could not say (``direction is None``)
    becomes no device. It is not silently demoted to a readable: the honest
    handling of an unknown direction is the build's, which refuses to stage a
    derived file rather than guess which channels are settable.

    Args:
        records: The roster's channel records, e.g.
            ``registered_channels(config).records``.

    Returns:
        The device document, ready for :func:`write_devices_file` or the
        build's ``validate_device_document``.
    """
    settables: list[dict[str, str]] = []
    readables: list[dict[str, str]] = []

    for record in records:
        if record.direction == "write":
            entry = {"name": record.address, "setpoint": record.address}
            if record.readback is not None and record.readback != record.address:
                entry["readback"] = record.readback
            settables.append(entry)
        elif record.direction == "read":
            readables.append({"name": record.address, "pv": record.address})

    return {SETTABLES_KEY: settables, READABLES_KEY: readables}


_FILE_MODE = 0o644
"""Mode the staged device file is written with; see ``write_devices_file``."""

_GENERATED_HEADER = """\
# Generated by OSPREY from {provenance}
# (osprey.services.bluesky_bridge.substrate_devices). Every render rewrites this
# file, so edits here are lost -- author your own device file and point
# `bluesky.devices_file` at it instead.
#
# The channel roster is this facility's one enumeration of which channels exist
# and which way they point. Every write-direction channel is a settable device,
# every read-direction channel a readable one, and a settable names a readback
# only where the roster paired one.
"""
"""Header the staged file carries, filled with the roster source's
``describe()`` so the file names the artifact it was derived from."""


def write_devices_file(
    path: Path, records: Sequence[ChannelRecord], *, source: RosterSource
) -> dict[str, list[dict[str, str]]]:
    """Write the document ``devices_document(records)`` builds to ``path`` as
    YAML, and return it.

    ``source`` is the roster source the records came from; it is named in the
    file's header via :meth:`~osprey.channel_roster.records.RosterSource.describe`,
    so a reader of a staged file can see which corpus or database the device
    set is a projection of. It is passed rather than read off the records
    because an empty roster has to name its provenance too.

    The write is atomic (same-directory temp file + ``os.replace``): the file is
    staged into a build tree that a running deploy may mount, so a reader must
    never observe a half-written device set, and a failed write must leave the
    previous document intact rather than truncated.

    Returns the document so a caller that also wants to report counts or
    validate what it just wrote does not have to re-derive or re-read it.
    """
    path = Path(path)
    document = devices_document(records)
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_GENERATED_HEADER.format(provenance=source.describe()))
            handle.write(body)
        # ``mkstemp`` creates the temp file 0600 and ``os.replace`` carries that
        # mode onto the destination -- unreadable to a container user that is not
        # the host user who rendered it, which is exactly how this file is
        # consumed (bind-mounted ``:ro`` into the queueserver worker).
        os.chmod(tmp_name, _FILE_MODE)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    return document
