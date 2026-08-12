"""Canonical derivation of the bluesky bridge's EPICS-substrate scan devices.

Single source of truth for turning a *built project's own*
``data/channel_limits.json`` into the bridge's EPICS-substrate device set
(``BLUESKY_EPICS_SUBSTRATE`` / ``BLUESKY_EPICS_MOTORS`` / ``_DETECTORS`` — see
``osprey.services.bluesky_bridge.devices._specs_from_env`` for the format
those env vars carry). Correctors are restricted to the pyat-coupled SR
HCM/VCM ``:SP``/``:RB`` partition (a write actually steers the beam via the
AT lattice model); BPMs are the pyat-coupled SR ``DIAG:BPM`` readbacks. Never
a hardcoded preset channel — always derived from the deployed project's own
data.

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

Deploy caveat: ``container_lifecycle._ensure_bluesky_substrate_env`` never
overwrites an already-set value, so a project deployed before this naming
change keeps whatever ``BLUESKY_EPICS_MOTORS``/``_DETECTORS`` its ``.env``
already holds until those lines are removed. Fresh deploys get the
address names automatically.

Two consumers share this module (DRY, one derivation):

- ``osprey.deployment.container_lifecycle`` (``_ensure_bluesky_substrate_env``),
  which auto-configures a VA-backed Bluesky stack's ``.env`` on ``osprey up``
  so the bridge starts in substrate mode with real channel names,
  turn-key.
- ``tests/e2e/_orm_stack.py``, whose ``select_correctors``/``select_bpms``/
  ``write_scan_env`` delegate here instead of re-deriving the same logic.

Host/deploy-side only — NOT part of the bridge's own container import
surface. This module imports ``osprey.services.virtual_accelerator.manifest``
(``classify_partition``/``PARTITION_PYAT_COUPLED``) -- the same
virtual-accelerator/channel-finder coupling ``_specs_from_env``'s module
docstring says the bridge's substrate branch must never take on directly
(the bridge is meant to stay control-system agnostic; the PV list reaches it
only via ``BLUESKY_EPICS_MOTORS``/``_DETECTORS``). Nothing under
``osprey.services.bluesky_bridge`` that runs *inside* the bridge container
(``app.py``, ``devices/*``) may import this module — it lives alongside the
bridge's device code only because it is conceptually about the bridge's
devices, not because it shares the bridge's runtime import surface. It runs
only from the host-side deploy/CLI process and from tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

# Env var names the bridge's own substrate-mode parser reads (see
# osprey.services.bluesky_bridge.devices._specs_from_env for the format).
# Imported here rather than restated so the deploy-time producer and the
# bridge's own consumer can never drift on the var names.
from osprey.services.bluesky_bridge.devices._specs_from_env import DETECTORS_ENV, MOTORS_ENV

SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"
"""Env var that switches the bridge from its demo runner to the EPICS substrate."""

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
    for a scan plan that sweeps correctors specifically.

    ``count=None`` (the default) returns the FULL available pyat-coupled
    corrector set -- the deploy wants every scannable device, not a fixed
    slice. When ``count`` is an int, raises ``AssertionError`` if fewer than
    ``count`` pairs are available; returns exactly ``count`` pairs otherwise.

    Returns a dict of ``sp_address -> (sp_address, rb_address)``: the motor's
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

    Returns a dict of ``read_address -> read_address``: the detector's device
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


def format_motors_env(correctors: dict[str, tuple[str, str]]) -> str:
    """Format ``correctors`` (as returned by ``select_correctors``) as the
    ``BLUESKY_EPICS_MOTORS`` value (see ``_specs_from_env``'s module
    docstring for the exact ``name=SP|RB`` syntax)."""
    return ",".join(f"{name}={sp}|{rb}" for name, (sp, rb) in correctors.items())


def format_detectors_env(bpms: dict[str, str]) -> str:
    """Format ``bpms`` (as returned by ``select_bpms``) as the
    ``BLUESKY_EPICS_DETECTORS`` value (``name=RB`` syntax)."""
    return ",".join(f"{name}={rb}" for name, rb in bpms.items())


def derive_substrate_env(project_dir: Path) -> dict[str, str]:
    """Derive the bridge's EPICS-substrate env from a *built* project's own
    ``data/channel_limits.json``.

    Returns ``{"BLUESKY_EPICS_SUBSTRATE": "1", "BLUESKY_EPICS_MOTORS": "...",
    "BLUESKY_EPICS_DETECTORS": "..."}`` when the project yields at least one
    corrector pair and one BPM readback. Returns ``{}`` -- never raises -- when
    ``channel_limits.json`` is missing, unreadable/malformed, or yields no
    correctors or no BPMs, so a caller on a deploy path can always treat an
    empty result as "skip auto-configuration" rather than a hard failure.
    """
    limits_path = Path(project_dir) / "data" / "channel_limits.json"
    if not limits_path.is_file():
        return {}

    try:
        limits = json.loads(limits_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}

    if not isinstance(limits, dict):
        return {}

    correctors = select_correctors(limits, count=None)
    bpms = select_bpms(limits, count=None)
    if not correctors or not bpms:
        return {}

    return {
        SUBSTRATE_ENV: "1",
        MOTORS_ENV: format_motors_env(correctors),
        DETECTORS_ENV: format_detectors_env(bpms),
    }
