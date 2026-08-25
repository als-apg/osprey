"""Canonical derivation of the bluesky bridge's EPICS-substrate plan devices.

Single source of truth for turning a *built project's own*
``data/channel_limits.json`` into the bridge's EPICS-substrate device set, in
the two-list device-file format the queueserver worker reads (see
``osprey.services.bluesky_bridge.devices._specs_from_file`` for the schema and
the parser). Correctors are restricted to the pyat-coupled SR HCM/VCM
``:SP``/``:RB`` partition (a write actually steers the beam via the AT lattice
model); BPMs are the pyat-coupled SR ``DIAG:BPM`` readbacks. Never a hardcoded
preset channel — always derived from the deployed project's own data.

Device name == channel address
------------------------------

Each device is keyed by the address it drives or reads -- a corrector by its
``:SP`` address, a BPM by its read address -- not by a synthetic
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
--- ``:?^SR.MAG.HCM.01.CURRENT.SP$:depth=1`` selects that one device, since
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

Two consumers share this module (DRY, one derivation):

- ``osprey.deployment.compose_generator`` (``_stage_bluesky_devices``), which
  derives and stages the device file for a VA-backed Bluesky stack on every
  render, so the worker starts with real channel names, turn-key.
- ``tests/e2e/_orm_stack.py``, whose ``select_correctors``/``select_bpms``/
  ``write_devices_file`` delegate here instead of re-deriving the same logic.

There is exactly one producer of the derived document -- ``devices_document``
below -- so the build path and the e2e harness can never drift on what the
worker is handed.

Host/deploy-side only — NOT part of the bridge's own container import
surface. This module imports ``osprey.services.virtual_accelerator.manifest``
(``classify_partition``/``PARTITION_PYAT_COUPLED``) -- the
virtual-accelerator/channel-finder coupling the bridge must never take on
directly (the bridge is meant to stay control-system agnostic; the device set
reaches it only as a mounted file). Nothing under
``osprey.services.bluesky_bridge`` that runs *inside* the bridge container
(``app.py``, ``devices/*``) may import this module — it lives alongside the
bridge's device code only because it is conceptually about the bridge's
devices, not because it shares the bridge's runtime import surface. It runs
only from the host-side deploy/CLI process and from tests.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml

# Key names of the device-file document the worker parses. Imported rather
# than restated so the host-side producer and the container-side consumer can
# never drift on the schema.
from osprey.services.bluesky_bridge.devices._specs_from_file import (
    READABLES_KEY,
    SETTABLES_KEY,
)

_T = TypeVar("_T")


def _address_path(address: str) -> dict[str, str] | None:
    """Split a 6-part colon address into its named partition components
    (the dict shape ``classify_partition`` consumes), or ``None`` if it does
    not have exactly six parts."""
    parts = address.split(":")
    if len(parts) != 6:
        return None
    ring, system, family, device, field, subfield = parts
    return {
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
    }


def _usable_keys(limits: dict[str, Any]) -> set[str]:
    """Channel-limit keys that name channels (skips ``_``-prefixed metadata
    entries and the ``defaults`` block)."""
    return {k for k in limits if not k.startswith("_") and k != "defaults"}


def _keyed_by_address(
    items: list[_T], address_of: Callable[[_T], str], count: int | None, unit_label: str
) -> dict[str, _T]:
    """Key ``items`` by ``address_of(item)`` -- the device name IS the channel
    address (see the module docstring). ``count=None`` takes all; an int
    raises ``AssertionError`` when fewer than ``count`` are available, else
    slices to exactly ``count``.

    The "exactly ``count``" promise holds only while the addresses are
    distinct, which both callers guarantee by deriving them from
    ``channel_limits.json`` keys. Colliding addresses would silently return a
    shorter dict, so the invariant is asserted rather than assumed."""
    if count is not None and len(items) < count:
        raise AssertionError(
            f"deployed project's channel_limits.json only yields {len(items)} "
            f"{unit_label}, need {count}"
        )
    take = len(items) if count is None else count
    keyed = {address_of(items[i]): items[i] for i in range(take)}
    if len(keyed) != take:
        raise AssertionError(
            f"duplicate addresses among the selected {unit_label}: "
            f"{take} selected, {len(keyed)} distinct names"
        )
    return keyed


def select_correctors(
    limits: dict[str, Any], count: int | None = None
) -> dict[str, tuple[str, str]]:
    """Derive SR corrector (HCM/VCM) ``:SP``/``:RB`` pairs from ``limits``
    (a parsed ``channel_limits.json``) -- never a hardcoded preset channel.

    Restricted to the pyat-coupled corrector partition (a write actually
    steers the beam via the AT lattice model) rather than any writable
    ``:SP``: a generic sp-echo pair (physics-free) is the wrong device class
    for a plan that sweeps correctors specifically.

    ``count=None`` (the default) returns the FULL available pyat-coupled
    corrector set -- the deploy wants every available device, not a fixed
    slice. When ``count`` is an int, raises ``AssertionError`` if fewer than
    ``count`` pairs are available; returns exactly ``count`` pairs otherwise.

    Returns a dict of ``sp_address -> (sp_address, rb_address)``: the setpoint's
    device name is its own ``:SP`` address, so a plan can reference the
    address the agent discovered (see the module docstring).
    """
    from osprey.services.virtual_accelerator.manifest import (
        PARTITION_PYAT_COUPLED,
        classify_partition,
    )

    keys = _usable_keys(limits)

    pairs: list[tuple[str, str]] = []
    for sp in sorted(k for k in keys if k.endswith(":SP")):
        path = _address_path(sp)
        if path is None:
            continue
        if path["ring"] != "SR" or path["system"] != "MAG" or path["family"] not in ("HCM", "VCM"):
            continue
        if classify_partition(path) != PARTITION_PYAT_COUPLED:
            continue
        rb = sp[:-3] + ":RB"
        if rb in keys:
            pairs.append((sp, rb))

    return _keyed_by_address(pairs, lambda pair: pair[0], count, "SR corrector (HCM/VCM) pairs")


def select_bpms(limits: dict[str, Any], count: int | None = None) -> dict[str, str]:
    """Derive SR BPM readbacks from ``limits`` (a parsed ``channel_limits.json``)
    -- same generic, no-hardcoded-channel convention as ``select_correctors``.

    ``count=None`` (the default) returns the FULL available pyat-coupled BPM
    set. When ``count`` is an int, raises ``AssertionError`` if fewer than
    ``count`` readbacks are available; returns exactly ``count`` otherwise.

    Returns a dict of ``read_address -> read_address``: the readback's device
    name is its own read address, so a plan can reference the address the
    agent discovered (see the module docstring).
    """
    from osprey.services.virtual_accelerator.manifest import (
        PARTITION_PYAT_COUPLED,
        classify_partition,
    )

    keys = _usable_keys(limits)

    addresses: list[str] = []
    for addr in sorted(keys):
        path = _address_path(addr)
        if path is None:
            continue
        if path["ring"] != "SR" or path["system"] != "DIAG" or path["family"] != "BPM":
            continue
        if classify_partition(path) != PARTITION_PYAT_COUPLED:
            continue
        addresses.append(addr)

    return _keyed_by_address(addresses, lambda addr: addr, count, "SR BPM readbacks")


def devices_document(limits: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Build the worker's device document from ``limits`` (a parsed
    ``channel_limits.json``).

    Returns the two-list mapping ``_specs_from_file`` parses: ``settables``
    entries carry ``name``/``setpoint``/``readback`` (one per
    ``select_correctors`` pair), ``readables`` entries carry ``name``/``pv``
    (one per ``select_bpms`` readback). Both keys are always present, even
    when empty, so a caller can see *which* half a project yielded nothing for
    rather than inferring it from an absent key.

    ``readback`` is emitted for every corrector because the selector only ever
    returns complete ``:SP``/``:RB`` pairs; a spec whose readback equalled its
    setpoint would omit the key entirely rather than write ``null``, which the
    loader accepts but which reads as "unset by mistake".
    """
    correctors = select_correctors(limits, count=None)
    bpms = select_bpms(limits, count=None)

    settables: list[dict[str, str]] = []
    for name, (setpoint, readback) in correctors.items():
        entry = {"name": name, "setpoint": setpoint}
        if readback != setpoint:
            entry["readback"] = readback
        settables.append(entry)

    readables = [{"name": name, "pv": read_pv} for name, read_pv in bpms.items()]

    return {SETTABLES_KEY: settables, READABLES_KEY: readables}


_FILE_MODE = 0o644
"""Mode the staged device file is written with; see ``write_devices_file``."""

_GENERATED_HEADER = """\
# Generated by OSPREY from the deployed project's own data/channel_limits.json
# (osprey.services.bluesky_bridge.substrate_devices). Every render rewrites this
# file, so edits here are lost -- author your own device file and point
# `bluesky.devices_file` at it instead.
#
# Which channels are scan devices, and which readback pairs with which setpoint,
# is a projection of the facility's namespace. That same namespace is described
# by the channel-finder database and the knowledge graph; the three are to be
# kept in step, and unifying them is a later piece of work.
"""


def write_devices_file(path: Path, limits: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Write the document ``devices_document(limits)`` builds to ``path`` as
    YAML, and return it.

    The write is atomic (same-directory temp file + ``os.replace``): the file is
    staged into a build tree that a running deploy may mount, so a reader must
    never observe a half-written device set, and a failed write must leave the
    previous document intact rather than truncated.

    Returns the document so a caller that also wants to report counts or
    validate what it just wrote does not have to re-derive or re-read it.
    """
    path = Path(path)
    document = devices_document(limits)
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_GENERATED_HEADER)
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
