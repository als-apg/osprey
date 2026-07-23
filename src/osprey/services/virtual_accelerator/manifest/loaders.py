"""Loads and normalizes the paradigm channel DBs plus the scenario data sources.

Reuses osprey's own channel_finder database parsers (the same code that
loads these files at runtime) so this generator never re-implements
``_expansion`` range/list parsing. Note the IOC does *not* read the emitted
``channel_manifest.json``: ``entrypoint.py`` regenerates the manifest
in-process via ``build_manifest()``, and the committed JSON serves only as a
drift guard. (An IOC *can* be pointed at a manifest JSON explicitly -- that
is the file-backed channel source below, :func:`load_manifest_file`.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)
from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase
from osprey.services.channel_finder.databases.template import (
    ChannelDatabase as TemplateChannelDatabase,
)

from . import paths


@dataclass(frozen=True)
class HierarchicalChannel:
    """One expanded address plus its decomposed hierarchy path.

    ``path`` maps hierarchy level names (ring/system/family/device/field/
    subfield) to the value selected for this channel, as produced by
    HierarchicalChannelDatabase's tree expansion.
    """

    address: str
    path: dict[str, str]


class ParadigmMismatchError(RuntimeError):
    """Raised when the three paradigm DBs disagree on their expanded address set.

    The whole premise of this generator is that the tutorial's channel-finder
    DBs already define a single namespace in three interchangeable formats.
    A mismatch means that premise is broken and must be fixed upstream in the
    DB source files -- never silently reconciled here.
    """


def load_hierarchical_channels() -> list[HierarchicalChannel]:
    """Expand the tier-3 hierarchical DB into (address, path) pairs."""
    db = HierarchicalChannelDatabase(str(paths.HIERARCHICAL_DB))
    db.load_database()
    return [
        HierarchicalChannel(address=ch["address"], path=ch["path"]) for ch in db.get_all_channels()
    ]


def load_in_context_addresses() -> set[str]:
    """Expand the tier-3 in_context (flat/template) DB into an address set."""
    db = TemplateChannelDatabase(str(paths.IN_CONTEXT_DB))
    db.load_database()
    return {ch["address"] for ch in db.get_all_channels()}


def load_middle_layer_addresses() -> set[str]:
    """Expand the tier-3 middle_layer (MML) DB into an address set."""
    db = MiddleLayerDatabase(str(paths.MIDDLE_LAYER_DB))
    db.load_database()
    return {ch["address"] for ch in db.get_all_channels()}


def load_machine_json_channels(path: Path | None = None) -> dict[str, dict]:
    """Return the scenario-seed machine.json channels keyed by address.

    ``path`` selects which machine.json to read: ``None`` (the default)
    keeps the historical behaviour of reading the bundled control-assistant
    template's copy; a file-backed facility passes its own mounted
    machine.json instead (see ``entrypoint.py``).
    """
    data = json.loads((path or paths.MACHINE_JSON).read_text())
    channels: dict[str, dict] = data["channels"]
    return channels


# --- file-backed channel source ------------------------------------------

# The full per-channel schema build_records() consumes -- identical to the
# in-memory shape build_manifest()["channels"] produces. A file-backed
# manifest must supply every key for every channel; the identity keys
# (ring/system/family/device/field) may be empty strings only if the
# facility accepts the pairing collisions that implies (setpoint/readback
# pairs are matched on exactly those five keys).
MANIFEST_CHANNEL_KEYS = frozenset(
    {
        "address",
        "ring",
        "system",
        "family",
        "device",
        "field",
        "subfield",
        "partition",
        "record_type",
        "noise",
    }
)


class ManifestFileError(RuntimeError):
    """A file-backed channel manifest is missing, unreadable, or malformed.

    Raised eagerly at load time so a misconfigured IOC dies at boot with a
    named cause, never serving a partial channel set.
    """


def load_manifest_file(path: Path) -> list[dict]:
    """Load the channel list from a manifest JSON file.

    This is the file-backed channel source: a facility that does not use the
    built-in generated manifest supplies ``{"channels": [...]}`` where each
    entry carries the exact per-channel schema ``build_manifest()`` produces
    (see ``MANIFEST_CHANNEL_KEYS``). Facility-neutral by construction -- no
    address grammar is imposed beyond the presence of the schema keys, so
    any facility's namespace (three-part addresses included) loads through
    the same call.

    Raises:
        ManifestFileError: if the file is absent, not valid JSON, lacks a
            top-level ``channels`` list, contains a channel missing schema
            keys, or declares the same address twice.
    """
    if not path.is_file():
        raise ManifestFileError(f"channel manifest file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestFileError(f"channel manifest {path} is not valid JSON: {exc}") from exc

    channels = data.get("channels") if isinstance(data, dict) else None
    if not isinstance(channels, list):
        raise ManifestFileError(
            f"channel manifest {path} must be a JSON object with a 'channels' list"
        )

    seen: set[str] = set()
    for index, channel in enumerate(channels):
        if not isinstance(channel, dict):
            raise ManifestFileError(f"channel manifest {path}: channels[{index}] is not an object")
        missing = MANIFEST_CHANNEL_KEYS - channel.keys()
        if missing:
            raise ManifestFileError(
                f"channel manifest {path}: channels[{index}] "
                f"({channel.get('address', '<no address>')!r}) is missing "
                f"key(s): {', '.join(sorted(missing))}"
            )
        address = channel["address"]
        if not address:
            raise ManifestFileError(
                f"channel manifest {path}: channels[{index}] has an empty address"
            )
        if address in seen:
            raise ManifestFileError(f"channel manifest {path}: duplicate address {address!r}")
        seen.add(address)

    return channels


# Matches `"<address>": { "label": ...` entries in machine_state_channels.json.j2
_MACHINE_STATE_KEY_RE = re.compile(r'"([^"]+)":\s*\{\s*"label"')


def load_machine_state_candidate_addresses() -> list[str]:
    """Extract every candidate channel key referenced by the machine-state template.

    ``machine_state_channels.json.j2`` is not valid JSON -- it branches on
    ``default_pipeline`` via Jinja ``{% if %}``/``{% elif %}``/``{% else %}`` --
    and all three of its branches are known to reference addresses that don't
    match the real ``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` namespace
    (tracked by the ``machine-state-canonical`` follow-up task). Rather than
    rendering a single branch, this pulls every candidate key across *all*
    branches so the reconciliation report in the manifest covers the whole
    file regardless of which pipeline mode is active.
    """
    text = paths.MACHINE_STATE_TEMPLATE.read_text()
    return _MACHINE_STATE_KEY_RE.findall(text)
