"""Tests for the benchmark channel helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.benchmarks.generator import (
    collect_middle_layer_pvs,
    expand_hierarchy,
)

#: Frozen copies of the demo's hierarchical and middle-layer databases.
GOLDEN = Path(__file__).resolve().parents[3] / "facility" / "golden" / "cf_index_pre_line"


@pytest.fixture(scope="module")
def tree_data() -> dict:
    """Load the frozen hierarchical database once per module."""
    return json.loads((GOLDEN / "hierarchical.json").read_text())


@pytest.fixture(scope="module")
def all_channels(tree_data: dict) -> list[dict]:
    """Expand all channels once per module."""
    return expand_hierarchy(tree_data)


class TestExpandHierarchy:
    """Tests for expand_hierarchy()."""

    @pytest.mark.usefixtures("tree_data")
    def test_expand_hierarchy(self, all_channels: list[dict]) -> None:
        """Expand template and verify entry structure."""
        # Every entry must have the required keys
        required_keys = {
            "pv",
            "ring",
            "system",
            "family",
            "device",
            "field",
            "subfield",
        }
        for ch in all_channels:
            assert required_keys <= set(ch.keys()), f"Missing keys in {ch}"

        # PV must be a colon-joined 6-part name
        for ch in all_channels:
            parts = ch["pv"].split(":")
            assert len(parts) == 6, f"PV has wrong segments: {ch['pv']}"
            assert parts == [
                ch["ring"],
                ch["system"],
                ch["family"],
                ch["device"],
                ch["field"],
                ch["subfield"],
            ]

    def test_channels_sorted(self, all_channels: list[dict]) -> None:
        """Channels must be sorted by PV name."""
        pvs = [ch["pv"] for ch in all_channels]
        assert pvs == sorted(pvs)

    def test_known_pv_present(self, all_channels: list[dict]) -> None:
        """Spot-check known PVs exist."""
        pvs = {ch["pv"] for ch in all_channels}
        assert "SR:MAG:DIPOLE:01:CURRENT:SP" in pvs
        assert "SR:MAG:DIPOLE:24:STATUS:FAULT" in pvs
        assert "SR:DIAG:BPM:01:POSITION:X" in pvs
        assert "SR:RF:CAVITY:01:VOLTAGE:RB" in pvs
        assert "BR:MAG:DIPOLE:01:CURRENT:SP" in pvs
        assert "BTS:DIAG:BPM:01:POSITION:X" in pvs

    def test_no_metadata_keys(self, all_channels: list[dict]) -> None:
        """No PV segment should start with underscore."""
        for ch in all_channels:
            for key in ("ring", "system", "family", "device", "field", "subfield"):
                assert not ch[key].startswith("_"), f"Metadata leaked into {key}: {ch[key]}"

    def test_ring_distribution(self, all_channels: list[dict]) -> None:
        """Verify channels exist for all three rings."""
        rings = {ch["ring"] for ch in all_channels}
        assert rings == {"SR", "BR", "BTS"}


class TestCollectMiddleLayerPvs:
    """Tests for collect_middle_layer_pvs()."""

    def test_lists_the_hierarchy_addresses(self, all_channels: list[dict]) -> None:
        """The middle-layer database lists exactly the hierarchy's addresses."""
        data = json.loads((GOLDEN / "middle_layer.json").read_text())
        assert collect_middle_layer_pvs(data) == {ch["pv"] for ch in all_channels}
