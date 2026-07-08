"""Loads and normalizes the paradigm channel DBs plus the scenario data sources.

Reuses osprey's own channel_finder database parsers (the same code that
loads these files at runtime) so this generator never re-implements
``_expansion`` range/list parsing. This is a generation-time dependency
only: the emitted ``channel_manifest.json`` has no osprey import
requirement for downstream consumers (the future IOC just reads the JSON).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

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


def load_machine_json_channels() -> dict[str, dict]:
    """Return the scenario-seed machine.json channels keyed by address."""
    data = json.loads(paths.MACHINE_JSON.read_text())
    return data["channels"]


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
