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

from .paths import PACKAGE_PATHS, ManifestPaths


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
    """Raised when the file-backed paradigm DBs disagree on their address set.

    The whole premise of this generator is that the tutorial's channel-finder
    DBs already define a single namespace in three interchangeable file
    formats: in_context, hierarchical, middle_layer. (The ``graph`` paradigm
    has no tier file and is never read here.) A mismatch means that premise is
    broken and must be fixed upstream in the DB source files -- never silently
    reconciled here.
    """


def load_hierarchical_database(
    paths: ManifestPaths = PACKAGE_PATHS,
) -> tuple[list[HierarchicalChannel], tuple[str, ...]]:
    """Expand ``paths``' hierarchical DB into its channels and its level names.

    The level names are the facility's own: a hierarchical database declares
    what its levels are called, and the manifest's identity keys and every
    partition rule are read from those names. Returned alongside the channels
    so the generator can compare the two without re-opening the file.
    """
    db = HierarchicalChannelDatabase(str(paths.hierarchical_db))
    db.load_database()
    channels = [
        HierarchicalChannel(address=ch["address"], path=ch["path"]) for ch in db.get_all_channels()
    ]
    return channels, tuple(db.hierarchy_levels)


def load_hierarchical_channels(paths: ManifestPaths = PACKAGE_PATHS) -> list[HierarchicalChannel]:
    """Expand ``paths``' hierarchical DB into (address, path) pairs."""
    return load_hierarchical_database(paths)[0]


def load_in_context_addresses(paths: ManifestPaths = PACKAGE_PATHS) -> set[str]:
    """Expand ``paths``' in_context (flat/template) DB into an address set."""
    db = TemplateChannelDatabase(str(paths.in_context_db))
    db.load_database()
    return {ch["address"] for ch in db.get_all_channels()}


#: The list a field states its addresses in, per protocol a record is read
#: under. A field may carry either list, or both.
_PROTOCOL_CHANNEL_KEYS = {"ca": "ChannelNames", "tango": "TangoNames"}


def _listed_addresses(db: MiddleLayerDatabase, channel: dict) -> list[str]:
    """One field's address list, exactly as the database states it.

    Blanks and repeats kept: this list is read for its POSITIONS, and a
    position is a device. The expanded records are the same list with the
    blanks and repeats gone, which is why they no longer say which device
    each surviving address belongs to.
    """
    node = db.data.get(channel["system"], {}).get(channel["family"], {}).get(channel["field"])
    for level in channel["subfield"] or ():
        node = node.get(level) if isinstance(node, dict) else None
    if not isinstance(node, dict):
        return []
    names = node.get(_PROTOCOL_CHANNEL_KEYS.get(channel.get("protocol"), "ChannelNames"), [])
    return [names] if isinstance(names, str) else list(names)


def load_middle_layer_channels(paths: ManifestPaths = PACKAGE_PATHS) -> list[dict]:
    """Expand ``paths``' middle_layer (MML) DB into its channel records.

    Each record carries the signal group the address was read from -- its
    system, family and field path -- plus whatever field metadata the export
    wrote (``MemberOf``, ``HWUnits`` and the rest). The manifest generator
    needs those: on a tree with no hierarchical database they are the only
    statement of which addresses are the read and write halves of one device
    field (see ``build._middle_layer_pairs``).

    Each record also carries ``slot``: the position its field lists it at,
    which is the device the family puts at that position. Two fields of one
    family state the same device where their slots agree, and only there --
    a field that leaves a device blank, or names one address twice, keeps
    addresses whose places in the expanded records have shifted against its
    sibling field's. ``slot`` is ``None`` for an address its own field does
    not list.
    """
    db = MiddleLayerDatabase(str(paths.middle_layer_db))
    db.load_database()
    listings: dict[tuple, list[str]] = {}
    records: list[dict] = []
    for channel in db.get_all_channels():
        key = (
            channel["system"],
            channel["family"],
            channel["field"],
            tuple(channel["subfield"] or ()),
            channel.get("protocol"),
        )
        if key not in listings:
            listings[key] = [name.strip() for name in _listed_addresses(db, channel)]
        listed = listings[key]
        address = channel["address"]
        slot = listed.index(address) if address in listed else None
        records.append(dict(channel, slot=slot))
    return records


def load_middle_layer_addresses(paths: ManifestPaths = PACKAGE_PATHS) -> set[str]:
    """Expand ``paths``' middle_layer (MML) DB into an address set."""
    return {ch["address"] for ch in load_middle_layer_channels(paths)}


def load_machine_json_channels(
    path: Path | None = None, paths: ManifestPaths = PACKAGE_PATHS
) -> dict[str, dict]:
    """Return the scenario-seed machine.json channels keyed by address.

    ``path`` selects which machine.json to read: ``None`` (the default)
    reads the bundled control-assistant template's copy; a file-backed
    facility passes its own mounted
    machine.json instead (see ``entrypoint.py``). ``paths`` supplies the
    fallback for callers that anchor on a data tree rather than a single
    file (the build-time generator).
    """
    data = json.loads((path or paths.machine_json).read_text())
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

    The address text is free; the ``subfield`` VALUE is not. It is a reserved
    vocabulary (``classify.SETPOINT_SUBFIELD`` / ``READBACK_SUBFIELD``):
    ``SP`` marks the writable channel and ``RB`` marks its readback, and a
    channel carrying any other token is neither written nor paired with one.

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


# Matches `"<address>": { "label": ...` entries in machine_state_channels.json
_MACHINE_STATE_KEY_RE = re.compile(r'"([^"]+)":\s*\{\s*"label"')


def load_machine_state_candidate_addresses(paths: ManifestPaths = PACKAGE_PATHS) -> list[str]:
    """Extract every candidate channel key in the machine-state channel list.

    ``machine_state_channels.json`` is a plain JSON object mapping each address
    to a ``{"label": ..., "group": ...}`` entry, alongside underscore-prefixed
    metadata keys (``_comment``, ``_version``). Keying off the ``"label"``
    member picks up exactly the channel entries and skips the metadata without
    an underscore-prefix convention having to be encoded here.

    The caller (``manifest/build.py``) checks each candidate against the
    addresses the VA actually serves and publishes the split under
    ``_metadata.machine_state_reconciliation`` as ``candidates_checked`` /
    ``valid`` / ``invalid``, so an address that drifts out of the
    ``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` namespace shows up in the
    manifest instead of failing silently.
    """
    text = paths.machine_state_channels.read_text()
    return _MACHINE_STATE_KEY_RE.findall(text)
