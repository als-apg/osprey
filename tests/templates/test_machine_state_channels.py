"""Content/reconciliation test for ``machine_state_channels.json``.

The file once branched on ``default_pipeline`` into three separate channel
lists (fictional FEL names for ``in_context``, ``SR01C:``-style names for
``middle_layer``, obsolete bracket names ``MAG:DIPOLE[B01]`` for
``hierarchical``) -- none of which exist in any channel-finder DB or the
namespace-union manifest (see ``tests/va/test_manifest.py`` /
``osprey.services.virtual_accelerator.manifest``). It now ships ONE canonical
channel list, independent of pipeline mode, drawn from real
``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` addresses that the manifest
actually contains -- and, having no Jinja left, ships as plain JSON under its
final filename rather than as a ``.j2`` template (see
``test_data_trees_are_not_templates.py``).

This is the CC-4 regression guard: every machine_state channel must be a real
address in the manifest's namespace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.manifest import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANNELS_PATH = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "machine_state_channels.json"
)

# Fictional/broken names from the pre-fix template, pinned so they never
# silently reappear.
FICTIONAL_ADDRESSES = (
    "TerminalVoltageReadBack",
    "GunPressure",
    "CollectorPressure",
    "IP41Pressure",
    "CoronaTriodCurrent",
    "SR01C:BPM1:X",
    "SR01C:BPM1:Y",
    "SR01C:HCM1:Current",
    "SR01C:VCM1:Current",
    "SR:DCCT:Current",
    "SR:RF1:Freq",
    "MAG:DIPOLE[B01]:CURRENT:RB",
    "MAG:QF[QF01]:CURRENT:RB",
    "MAG:QD[QD01]:CURRENT:RB",
    "MAG:HCM[H01]:CURRENT:RB",
    "MAG:VCM[V01]:CURRENT:RB",
    "RF:CAVITY[C1]:VOLTAGE:RB",
)


def _load() -> dict:
    return json.loads(CHANNELS_PATH.read_text())


def _channels(loaded: dict) -> dict:
    """Entries minus the underscore-prefixed metadata keys."""
    return {k: v for k, v in loaded.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def manifest_addresses() -> set[str]:
    manifest = build_manifest()
    return {c["address"] for c in manifest["channels"]}


class TestShipsAsOneCanonicalList:
    def test_is_plain_json_at_its_final_filename(self):
        assert CHANNELS_PATH.exists(), f"{CHANNELS_PATH} is missing"
        assert not CHANNELS_PATH.with_name(CHANNELS_PATH.name + ".j2").exists()
        loaded = _load()
        assert isinstance(loaded, dict)
        assert _channels(loaded), "expected at least one machine_state channel"

    def test_carries_no_jinja_constructs(self):
        """No pipeline-mode branching survives -- the list is mode-independent."""
        text = CHANNELS_PATH.read_text()
        for construct in ("{{", "{%", "{#"):
            assert construct not in text, f"Jinja construct {construct!r} reappeared"
        assert "default_pipeline" not in text
        assert "channel_finder_mode" not in text


class TestManifestConsistency:
    """CC-4 regression guard: every channel must be a real address."""

    def test_every_channel_is_in_the_manifest(self, manifest_addresses):
        channels = _channels(_load())
        for address in channels:
            assert address in manifest_addresses, f"{address!r} not in manifest namespace"

    def test_no_fictional_addresses_remain(self):
        channels = _channels(_load())
        for fictional in FICTIONAL_ADDRESSES:
            assert fictional not in channels, f"fictional address {fictional!r} reappeared"

    def test_addresses_follow_the_real_naming_grammar(self):
        channels = _channels(_load())
        for address in channels:
            parts = address.split(":")
            assert len(parts) == 6, (
                f"{address!r} does not match RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD"
            )
            assert "[" not in address and "]" not in address


class TestChannelShape:
    def test_every_entry_has_label_and_group(self):
        channels = _channels(_load())
        for address, defn in channels.items():
            assert defn.get("label"), f"{address!r} missing a non-empty label"
            assert defn.get("group"), f"{address!r} missing a non-empty group"

    def test_covers_the_representative_categories(self):
        """One representative channel per category named in the task:
        DCCT current, RF cavity voltage, a BPM pair, one corrector RB, and a
        representative vacuum pressure."""
        channels = _channels(_load())
        groups = {defn["group"] for defn in channels.values()}
        assert {"beam", "rf", "orbit", "magnets", "vacuum"} <= groups

        addresses = set(channels)
        assert any("DCCT" in a and a.endswith(":CURRENT:RB") for a in addresses)
        assert any(":RF:CAVITY:" in a and a.endswith(":VOLTAGE:RB") for a in addresses)
        assert any(":DIAG:BPM:" in a and a.endswith(":POSITION:X") for a in addresses)
        assert any(":DIAG:BPM:" in a and a.endswith(":POSITION:Y") for a in addresses)
        assert any(":MAG:" in a and a.endswith(":CURRENT:RB") for a in addresses)
        assert any(":VAC:GAUGE:" in a and a.endswith(":PRESSURE:RB") for a in addresses)
