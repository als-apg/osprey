"""Tests for cross-paradigm benchmark generator."""

from __future__ import annotations

import json

import pytest

from osprey.services.channel_finder.benchmarks.generator import (
    ALIAS_FAMILY_NAMES,
    ALIAS_FIELD_NAMES,
    ALIAS_RING_NAMES,
    ALIAS_SUBFIELD_NAMES,
    FAMILY_NAMES,
    FIELD_NAMES,
    RING_NAMES,
    SUBFIELD_NAMES,
    TEMPLATE_DB_PATH,
    TIER_1,
    TIER_2,
    TIER_3,
    expand_hierarchy,
    filter_channels,
    format_hierarchical,
    format_in_context,
    format_middle_layer,
    generate_alias,
    generate_description,
    load_template,
    validate_queries,
)


@pytest.fixture(scope="module")
def tree_data() -> dict:
    """Load the hierarchical template database once per module."""
    with open(TEMPLATE_DB_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_channels(tree_data: dict) -> list[dict]:
    """Expand all channels once per module."""
    return expand_hierarchy(tree_data)


class TestExpandHierarchy:
    """Tests for expand_hierarchy()."""

    def test_expand_hierarchy(self, tree_data: dict, all_channels: list[dict]) -> None:
        """Expand template and verify total count and entry structure."""
        # Total channel count must match Tier 3 (full DB)
        assert len(all_channels) == TIER_3.target_count

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
        assert "SR:MAG:DIPOLE:B01:CURRENT:SP" in pvs
        assert "SR:MAG:DIPOLE:B24:STATUS:FAULT" in pvs
        assert "SR:DIAG:BPM:BPM01:POSITION:X" in pvs
        assert "SR:RF:CAVITY:C1:VOLTAGE:RB" in pvs
        assert "BR:MAG:DIPOLE:B01:CURRENT:SP" in pvs
        assert "BTS:DIAG:BPM:BPM01:POSITION:X" in pvs

    def test_no_metadata_keys(self, all_channels: list[dict]) -> None:
        """No PV segment should start with underscore."""
        for ch in all_channels:
            for key in ("ring", "system", "family", "device", "field", "subfield"):
                assert not ch[key].startswith("_"), f"Metadata leaked into {key}: {ch[key]}"

    def test_ring_distribution(self, all_channels: list[dict]) -> None:
        """Verify channels exist for all three rings."""
        rings = {ch["ring"] for ch in all_channels}
        assert rings == {"SR", "BR", "BTS"}


class TestGenerateDescription:
    """Tests for generate_description()."""

    def test_generate_description(self) -> None:
        """Test description generation for several PV patterns."""
        cases = [
            (
                {
                    "ring": "SR",
                    "system": "MAG",
                    "family": "DIPOLE",
                    "device": "B01",
                    "field": "CURRENT",
                    "subfield": "SP",
                },
                "Storage ring dipole bending magnet B01 current setpoint",
            ),
            (
                {
                    "ring": "SR",
                    "system": "DIAG",
                    "family": "BPM",
                    "device": "BPM01",
                    "field": "POSITION",
                    "subfield": "X",
                },
                "Storage ring beam position monitor BPM01 position horizontal",
            ),
            (
                {
                    "ring": "BR",
                    "system": "MAG",
                    "family": "QF",
                    "device": "QF01",
                    "field": "STATUS",
                    "subfield": "FAULT",
                },
                "Booster ring focusing quadrupole QF01 status fault",
            ),
            (
                {
                    "ring": "BTS",
                    "system": "MAG",
                    "family": "VCM",
                    "device": "V01",
                    "field": "CURRENT",
                    "subfield": "RB",
                },
                "Booster-to-storage transfer line vertical corrector V01 current readback",
            ),
            (
                {
                    "ring": "SR",
                    "system": "RF",
                    "family": "CAVITY",
                    "device": "C1",
                    "field": "POWER",
                    "subfield": "FWD",
                },
                "Storage ring RF cavity C1 power forward",
            ),
            (
                {
                    "ring": "SR",
                    "system": "VAC",
                    "family": "ION-PUMP",
                    "device": "SR01",
                    "field": "PRESSURE",
                    "subfield": "RB",
                },
                "Storage ring ion pump SR01 pressure readback",
            ),
        ]

        for pv_parts, expected in cases:
            result = generate_description(pv_parts)
            assert result == expected, (
                f"For {pv_parts['ring']}:...:{pv_parts['subfield']}: "
                f"got {result!r}, expected {expected!r}"
            )

    def test_description_starts_uppercase(self) -> None:
        """Every description must start with a capital letter."""
        parts = {
            "ring": "SR",
            "system": "MAG",
            "family": "DIPOLE",
            "device": "B01",
            "field": "CURRENT",
            "subfield": "RB",
        }
        desc = generate_description(parts)
        assert desc[0].isupper()


class TestTierSpecs:
    """Tests for tier specifications and filter_channels()."""

    def test_tier_specs(self, tree_data: dict, all_channels: list[dict]) -> None:
        """Verify each tier produces the expected channel count."""
        tier1 = filter_channels(all_channels, TIER_1)
        assert len(tier1) == TIER_1.target_count, (
            f"Tier 1: expected {TIER_1.target_count}, got {len(tier1)}"
        )

        tier2 = filter_channels(all_channels, TIER_2)
        assert len(tier2) == TIER_2.target_count, (
            f"Tier 2: expected {TIER_2.target_count}, got {len(tier2)}"
        )

        tier3 = filter_channels(all_channels, TIER_3)
        assert len(tier3) == TIER_3.target_count, (
            f"Tier 3: expected {TIER_3.target_count}, got {len(tier3)}"
        )

    def test_tier_ordering(self, all_channels: list[dict]) -> None:
        """Tier 1 < Tier 2 < Tier 3 (strict subset)."""
        t1 = {ch["pv"] for ch in filter_channels(all_channels, TIER_1)}
        t2 = {ch["pv"] for ch in filter_channels(all_channels, TIER_2)}
        t3 = {ch["pv"] for ch in filter_channels(all_channels, TIER_3)}

        assert t1 < t2, "Tier 1 is not a strict subset of Tier 2"
        assert t2 < t3, "Tier 2 is not a strict subset of Tier 3"

    def test_tier1_sr_only(self, all_channels: list[dict]) -> None:
        """Tier 1 must contain only SR channels."""
        tier1 = filter_channels(all_channels, TIER_1)
        rings = {ch["ring"] for ch in tier1}
        assert rings == {"SR"}

    def test_tier2_sr_only(self, all_channels: list[dict]) -> None:
        """Tier 2 must contain only SR channels."""
        tier2 = filter_channels(all_channels, TIER_2)
        rings = {ch["ring"] for ch in tier2}
        assert rings == {"SR"}

    def test_tier3_all_rings(self, all_channels: list[dict]) -> None:
        """Tier 3 must contain all three rings."""
        tier3 = filter_channels(all_channels, TIER_3)
        rings = {ch["ring"] for ch in tier3}
        assert rings == {"SR", "BR", "BTS"}

    def test_tier1_families(self, all_channels: list[dict]) -> None:
        """Tier 1 must only contain the specified families."""
        tier1 = filter_channels(all_channels, TIER_1)
        families = {ch["family"] for ch in tier1}
        assert families == {
            "DIPOLE",
            "QF",
            "HCM",
            "VCM",
            "BPM",
            "DCCT",
            "CAVITY",
        }

    def test_tier1_no_status(self, all_channels: list[dict]) -> None:
        """Tier 1 must not include STATUS fields."""
        tier1 = filter_channels(all_channels, TIER_1)
        subfields = {ch["subfield"] for ch in tier1}
        assert "READY" not in subfields
        assert "ON" not in subfields
        assert "FAULT" not in subfields

    def test_step_ratios(self, all_channels: list[dict]) -> None:
        """Verify approximate step ratios between tiers."""
        n1 = len(filter_channels(all_channels, TIER_1))
        n2 = len(filter_channels(all_channels, TIER_2))
        n3 = len(filter_channels(all_channels, TIER_3))

        ratio_2_1 = n2 / n1
        ratio_3_2 = n3 / n2

        # Tier 2 / Tier 1 ~2.7x (allow 2.0 - 3.5)
        assert 2.0 < ratio_2_1 < 3.5, f"Tier2/Tier1 ratio {ratio_2_1:.2f} out of range"
        # Tier 3 / Tier 2 ~2.2x (allow 1.5 - 3.0)
        assert 1.5 < ratio_3_2 < 3.0, f"Tier3/Tier2 ratio {ratio_3_2:.2f} out of range"


class TestFormatInContext:
    """Tests for format_in_context()."""

    def test_format_in_context(self, tree_data: dict, all_channels: list[dict]) -> None:
        """Verify in-context envelope format structure and channel count."""
        for tier_spec in (TIER_1, TIER_2, TIER_3):
            result = format_in_context(all_channels, tier_spec)

            # Must be envelope dict with _metadata and channels
            assert isinstance(result, dict), (
                f"{tier_spec.name}: expected dict, got {type(result).__name__}"
            )
            assert "_metadata" in result, f"{tier_spec.name}: missing '_metadata' key"
            assert "channels" in result, f"{tier_spec.name}: missing 'channels' key"

            # Metadata checks
            assert result["_metadata"]["tier"] == tier_spec.name
            assert result["_metadata"]["total_channels"] == tier_spec.target_count

            # Each entry must have channel (alias), address (PV), description
            for entry in result["channels"]:
                assert "channel" in entry, f"{tier_spec.name}: missing 'channel' key"
                assert "address" in entry, f"{tier_spec.name}: missing 'address' key"
                assert "description" in entry, f"{tier_spec.name}: missing 'description' key"
                # address is the PV (contains colons)
                assert ":" in entry["address"], (
                    f"{tier_spec.name}: invalid PV address: {entry['address']}"
                )
                # channel is the alias (contains underscores, no colons)
                assert "_" in entry["channel"], (
                    f"{tier_spec.name}: alias missing underscores: {entry['channel']}"
                )

            # Count must match tier target
            assert len(result["channels"]) == tier_spec.target_count, (
                f"{tier_spec.name}: expected {tier_spec.target_count}, "
                f"got {len(result['channels'])}"
            )

    def test_in_context_round_trip(self, all_channels: list[dict], tmp_path) -> None:
        """Round-trip: generate → write → load via ChannelDatabase → lookup alias → get PV."""
        from osprey.services.channel_finder.databases.flat import ChannelDatabase

        ic_data = format_in_context(all_channels, TIER_1)

        db_path = tmp_path / "in_context.json"
        db_path.write_text(json.dumps(ic_data))

        db = ChannelDatabase(str(db_path))
        db.load_database()

        stats = db.get_statistics()
        assert stats["total_channels"] == TIER_1.target_count

        first = ic_data["channels"][0]
        last = ic_data["channels"][-1]

        for entry in [first, last]:
            alias = entry["channel"]
            expected_pv = entry["address"]
            result = db.get_channel(alias)
            assert result is not None, f"Channel not found: {alias}"
            assert result["address"] == expected_pv, (
                f"Address mismatch for {alias}: {result['address']} != {expected_pv}"
            )

    def test_in_context_metadata(self, all_channels: list[dict]) -> None:
        """Metadata block must contain required fields."""
        result = format_in_context(all_channels, TIER_1)
        meta = result["_metadata"]

        assert "version" in meta
        assert "tier" in meta
        assert "total_channels" in meta
        assert "generated_by" in meta

        assert meta["tier"] == "tier1"
        assert meta["total_channels"] == TIER_1.target_count
        assert meta["generated_by"] == "osprey-benchmark-generator"


class TestFormatHierarchical:
    """Tests for format_hierarchical()."""

    def test_format_hierarchical(self, tree_data: dict) -> None:
        """Verify hierarchical format preserves structure and prunes correctly."""
        result = format_hierarchical(tree_data, TIER_1)

        # Must preserve hierarchy section
        assert "hierarchy" in result, "Missing 'hierarchy' section"
        assert "levels" in result["hierarchy"]
        assert "naming_pattern" in result["hierarchy"]

        # Must have tree key
        assert "tree" in result, "Missing 'tree' key"

        # Tier 1 -> only SR ring
        assert {k for k in result["tree"] if not k.startswith("_")} == {"SR"}, (
            "Tier 1 should contain only SR"
        )

        # Tier 1 families: collect all family-level keys
        sr = result["tree"]["SR"]
        families: set[str] = set()
        for sys_name, sys_node in sr.items():
            if sys_name.startswith("_") or not isinstance(sys_node, dict):
                continue
            for fam_name in sys_node:
                if not fam_name.startswith("_"):
                    families.add(fam_name)

        assert families == TIER_1.families, f"Expected families {TIER_1.families}, got {families}"

    def test_expansion_preserved(self, tree_data: dict) -> None:
        """_expansion directives must survive pruning with unchanged ranges."""
        result = format_hierarchical(tree_data, TIER_1)
        sr_tree = result["tree"]["SR"]

        # Find DIPOLE -> DEVICE -> _expansion
        dipole_dev = sr_tree["MAG"]["DIPOLE"]["DEVICE"]
        exp = dipole_dev.get("_expansion")
        assert exp is not None, "Missing _expansion in DIPOLE/DEVICE"
        assert exp["_type"] == "range"
        assert exp["_range"] == [1, 24], f"Range should be [1, 24], got {exp['_range']}"

    def test_channel_count_matches_tier(self, tree_data: dict, all_channels: list[dict]) -> None:
        """Expanding the pruned tree should yield the tier target count."""
        from osprey.services.channel_finder.benchmarks.generator import (
            expand_hierarchy,
        )

        for tier_spec in (TIER_1, TIER_2, TIER_3):
            pruned = format_hierarchical(tree_data, tier_spec)
            expanded = expand_hierarchy(pruned)
            assert len(expanded) == tier_spec.target_count, (
                f"{tier_spec.name}: expanded to {len(expanded)}, expected {tier_spec.target_count}"
            )

    def test_tier3_all_rings(self, tree_data: dict) -> None:
        """Tier 3 (no pruning) should preserve all three rings."""
        result = format_hierarchical(tree_data, TIER_3)
        rings = {k for k in result["tree"] if not k.startswith("_")}
        assert rings == {"SR", "BR", "BTS"}


class TestFormatMiddleLayer:
    """Tests for format_middle_layer()."""

    def test_format_middle_layer(self, tree_data: dict, all_channels: list[dict]) -> None:
        """Verify middle-layer format structure and total channel count."""
        for tier_spec in (TIER_1, TIER_2, TIER_3):
            result = format_middle_layer(all_channels, tier_spec)

            # Top-level keys should be ring names (subset of tier rings)
            top_keys = {k for k in result if not k.startswith("_")}
            assert top_keys <= tier_spec.rings, (
                f"{tier_spec.name}: unexpected top keys {top_keys - tier_spec.rings}"
            )

            # Each ring should have family keys
            for ring_key in top_keys:
                ring_node = result[ring_key]
                family_keys = {k for k in ring_node if not k.startswith("_")}
                assert len(family_keys) > 0, f"{tier_spec.name}: no families under {ring_key}"

            # ChannelNames arrays should exist at leaf level and
            # total count must match tier target
            total = 0
            for ring_key in top_keys:
                for fam_key, fam_node in result[ring_key].items():
                    if fam_key.startswith("_"):
                        continue
                    for field_key, field_node in fam_node.items():
                        if field_key.startswith("_"):
                            continue
                        for sf_key, sf_node in field_node.items():
                            if sf_key.startswith("_"):
                                continue
                            assert "ChannelNames" in sf_node, (
                                f"{tier_spec.name}: missing ChannelNames "
                                f"at {ring_key}/{fam_key}/{field_key}/{sf_key}"
                            )
                            total += len(sf_node["ChannelNames"])

            assert total == tier_spec.target_count, (
                f"{tier_spec.name}: total channels {total}, expected {tier_spec.target_count}"
            )

    def test_leaf_metadata(self, all_channels: list[dict]) -> None:
        """Leaf nodes must have DataType and HWUnits."""
        result = format_middle_layer(all_channels, TIER_1)
        for ring_node in result.values():
            for fam_key, fam_node in ring_node.items():
                if fam_key.startswith("_"):
                    continue
                for field_key, field_node in fam_node.items():
                    if field_key.startswith("_"):
                        continue
                    for sf_key, sf_node in field_node.items():
                        if sf_key.startswith("_"):
                            continue
                        assert "DataType" in sf_node, (
                            f"Missing DataType at {fam_key}/{field_key}/{sf_key}"
                        )
                        assert "HWUnits" in sf_node, (
                            f"Missing HWUnits at {fam_key}/{field_key}/{sf_key}"
                        )


class TestValidateQueries:
    """Tests for validate_queries()."""

    @pytest.fixture
    def mini_benchmark(self, tree_data, all_channels, tmp_path):
        """Create a minimal benchmark DB using the real generator functions."""
        # Use tier 1 (smallest) to generate consistent databases
        for tier_num in (1, 2, 3):
            tier_dir = tmp_path / f"tier{tier_num}"
            tier_dir.mkdir()

            ic_data = format_in_context(all_channels, TIER_1)
            (tier_dir / "in_context.json").write_text(json.dumps(ic_data))

            hier_data = format_hierarchical(tree_data, TIER_1)
            (tier_dir / "hierarchical.json").write_text(json.dumps(hier_data))

            ml_data = format_middle_layer(all_channels, TIER_1)
            (tier_dir / "middle_layer.json").write_text(json.dumps(ml_data))

        # Pick a few PVs that exist in the tier-1 database
        tier1_channels = filter_channels(all_channels, TIER_1)
        sample_pvs = [ch["pv"] for ch in tier1_channels[:3]]

        queries = [
            {"user_query": "find first", "targeted_pv": [sample_pvs[0]]},
            {"user_query": "find others", "targeted_pv": sample_pvs[1:]},
        ]
        queries_path = tmp_path / "queries.json"
        queries_path.write_text(json.dumps(queries))

        return tmp_path, queries_path, sample_pvs

    def test_all_pvs_present(self, mini_benchmark):
        """Happy path: all PVs exist in all databases."""
        db_dir, queries_path, _pvs = mini_benchmark
        result = validate_queries(queries_path, db_dir)
        assert result["valid"] is True
        assert result["missing"] == []
        assert result["missing_databases"] == []
        assert result["total_queries"] == 2

    def test_missing_pv_in_one_format(self, mini_benchmark):
        """A PV missing from one format is detected."""
        db_dir, queries_path, sample_pvs = mini_benchmark
        target_pv = sample_pvs[0]

        # Remove target PV from tier1/in_context (envelope format)
        ic_path = db_dir / "tier1" / "in_context.json"
        ic_data = json.loads(ic_path.read_text())
        ic_data["channels"] = [e for e in ic_data["channels"] if e.get("address") != target_pv]
        ic_data["_metadata"]["total_channels"] = len(ic_data["channels"])
        ic_path.write_text(json.dumps(ic_data))

        result = validate_queries(queries_path, db_dir)
        assert result["valid"] is False
        assert any(
            e["pv"] == target_pv and e["tier"] == 1 and e["format"] == "in_context"
            for e in result["missing"]
        )

    def test_missing_database_file(self, mini_benchmark):
        """Missing database file is reported."""
        db_dir, queries_path, _pvs = mini_benchmark
        (db_dir / "tier2" / "middle_layer.json").unlink()

        result = validate_queries(queries_path, db_dir)
        assert result["valid"] is False
        assert len(result["missing_databases"]) == 1

    def test_empty_queries(self, tmp_path):
        """Empty query list is valid."""
        queries_path = tmp_path / "queries.json"
        queries_path.write_text("[]")
        for t in (1, 2, 3):
            tier_dir = tmp_path / f"tier{t}"
            tier_dir.mkdir()
            (tier_dir / "in_context.json").write_text("[]")
            (tier_dir / "hierarchical.json").write_text('{"tree": {}}')
            (tier_dir / "middle_layer.json").write_text("{}")

        result = validate_queries(queries_path, tmp_path)
        assert result["valid"] is True
        assert result["total_queries"] == 0


class TestGenerateAlias:
    """Tests for alias generation maps and generate_alias()."""

    def test_alias_known_examples(self) -> None:
        """Verify aliases match expected output for known inputs."""
        cases = [
            (
                {
                    "ring": "SR",
                    "system": "MAG",
                    "family": "DIPOLE",
                    "device": "B05",
                    "field": "CURRENT",
                    "subfield": "SP",
                },
                "StorageRing_Dipole_B05_Current_Setpoint",
            ),
            (
                {
                    "ring": "SR",
                    "system": "DIAG",
                    "family": "BPM",
                    "device": "BPM01",
                    "field": "POSITION",
                    "subfield": "X",
                },
                "StorageRing_BPM_BPM01_Position_X",
            ),
            (
                {
                    "ring": "SR",
                    "system": "RF",
                    "family": "CAVITY",
                    "device": "C2",
                    "field": "VOLTAGE",
                    "subfield": "RB",
                },
                "StorageRing_Cavity_C2_Voltage_Readback",
            ),
            (
                {
                    "ring": "BR",
                    "system": "MAG",
                    "family": "DIPOLE",
                    "device": "B01",
                    "field": "CURRENT",
                    "subfield": "SP",
                },
                "BoosterRing_Dipole_B01_Current_Setpoint",
            ),
            (
                {
                    "ring": "BTS",
                    "system": "DIAG",
                    "family": "BPM",
                    "device": "BPM01",
                    "field": "POSITION",
                    "subfield": "Y",
                },
                "BoosterToStorageRing_BPM_BPM01_Position_Y",
            ),
        ]
        for pv_parts, expected in cases:
            result = generate_alias(pv_parts)
            assert result == expected, f"Got {result!r}, expected {expected!r}"

    def test_alias_map_completeness(self) -> None:
        """Every key in verbose maps must have a corresponding alias map entry."""
        for key in FAMILY_NAMES:
            assert key in ALIAS_FAMILY_NAMES, f"ALIAS_FAMILY_NAMES missing key: {key}"
        for key in SUBFIELD_NAMES:
            assert key in ALIAS_SUBFIELD_NAMES, f"ALIAS_SUBFIELD_NAMES missing key: {key}"
        for key in FIELD_NAMES:
            assert key in ALIAS_FIELD_NAMES, f"ALIAS_FIELD_NAMES missing key: {key}"
        for key in RING_NAMES:
            assert key in ALIAS_RING_NAMES, f"ALIAS_RING_NAMES missing key: {key}"

    def test_alias_uniqueness_per_tier(self, all_channels: list[dict]) -> None:
        """All aliases must be unique within each tier's channel set."""
        for tier_spec in (TIER_1, TIER_2, TIER_3):
            filtered = filter_channels(all_channels, tier_spec)
            aliases = [generate_alias(ch) for ch in filtered]
            assert len(aliases) == len(set(aliases)), (
                f"{tier_spec.name}: duplicate aliases found "
                f"({len(aliases)} total, {len(set(aliases))} unique)"
            )

    def test_alias_fallback_unmapped(self) -> None:
        """Unmapped keys should fall back to the raw name."""
        parts = {
            "ring": "UNKNOWN_RING",
            "system": "SYS",
            "family": "UNKNOWN_FAM",
            "device": "D01",
            "field": "UNKNOWN_FIELD",
            "subfield": "UNKNOWN_SF",
        }
        result = generate_alias(parts)
        assert result == "UNKNOWN_RING_UNKNOWN_FAM_D01_UNKNOWN_FIELD_UNKNOWN_SF"

    def test_alias_format(self, all_channels: list[dict]) -> None:
        """All aliases should be underscore-separated with no colons."""
        tier1 = filter_channels(all_channels, TIER_1)
        for ch in tier1:
            alias = generate_alias(ch)
            assert "_" in alias, f"Alias missing underscores: {alias}"
            assert ":" not in alias, f"Alias contains colons: {alias}"
            # Should have exactly 4 underscores (5 parts)
            assert alias.count("_") == 4, f"Alias has wrong number of parts: {alias}"


class TestSetupBlocks:
    """Tests for setup block generation in format_middle_layer()."""

    def test_setup_blocks_present_all_tiers(self, all_channels: list[dict]) -> None:
        """Every family in every tier should have a _setup block."""
        for tier_spec in (TIER_1, TIER_2, TIER_3):
            result = format_middle_layer(all_channels, tier_spec)
            for ring_key in result:
                if ring_key.startswith("_"):
                    continue
                for fam_key, fam_node in result[ring_key].items():
                    if fam_key.startswith("_"):
                        continue
                    assert "_setup" in fam_node, (
                        f"{tier_spec.name}: missing _setup in {ring_key}/{fam_key}"
                    )

    def test_device_list_length_matches_channels(self, all_channels: list[dict]) -> None:
        """DeviceList length must match ChannelNames length at every leaf."""
        result = format_middle_layer(all_channels, TIER_3)
        for ring_key in result:
            if ring_key.startswith("_"):
                continue
            for fam_key, fam_node in result[ring_key].items():
                if fam_key.startswith("_"):
                    continue
                setup = fam_node.get("_setup", {})
                device_list = setup.get("DeviceList", [])
                num_devices = len(device_list)

                # Check every leaf's ChannelNames length
                for field_key, field_node in fam_node.items():
                    if field_key.startswith("_"):
                        continue
                    for sf_key, sf_node in field_node.items():
                        if sf_key.startswith("_"):
                            continue
                        if "ChannelNames" in sf_node:
                            assert len(sf_node["ChannelNames"]) == num_devices, (
                                f"DeviceList ({num_devices}) != ChannelNames "
                                f"({len(sf_node['ChannelNames'])}) at "
                                f"{ring_key}/{fam_key}/{field_key}/{sf_key}"
                            )

    def test_common_names_format(self, all_channels: list[dict]) -> None:
        """CommonNames should have one entry per device with family prefix."""
        result = format_middle_layer(all_channels, TIER_1)
        for ring_key in result:
            if ring_key.startswith("_"):
                continue
            for fam_key, fam_node in result[ring_key].items():
                if fam_key.startswith("_"):
                    continue
                setup = fam_node.get("_setup", {})
                common_names = setup.get("CommonNames", [])
                device_list = setup.get("DeviceList", [])
                assert len(common_names) == len(device_list), (
                    f"CommonNames ({len(common_names)}) != DeviceList ({len(device_list)}) "
                    f"at {ring_key}/{fam_key}"
                )
                # Each common name should be a non-empty string
                for name in common_names:
                    assert isinstance(name, str) and len(name) > 0

    def test_element_list_sequential(self, all_channels: list[dict]) -> None:
        """ElementList should be [1, 2, ..., N]."""
        result = format_middle_layer(all_channels, TIER_1)
        for ring_key in result:
            if ring_key.startswith("_"):
                continue
            for fam_key, fam_node in result[ring_key].items():
                if fam_key.startswith("_"):
                    continue
                setup = fam_node.get("_setup", {})
                element_list = setup.get("ElementList", [])
                expected = list(range(1, len(element_list) + 1))
                assert element_list == expected, (
                    f"ElementList at {ring_key}/{fam_key} not sequential: {element_list}"
                )

    def test_setup_blocks_with_middle_layer_database(
        self, all_channels: list[dict], tmp_path
    ) -> None:
        """Generated DB loads via MiddleLayerDatabase; sector filtering and common names work."""
        from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

        ml_data = format_middle_layer(all_channels, TIER_3)
        db_path = tmp_path / "middle_layer.json"
        db_path.write_text(json.dumps(ml_data))

        db = MiddleLayerDatabase(str(db_path))
        db.load_database()

        # Sector filtering should return a subset
        all_bpm_x = db.list_channel_names("SR", "BPM", "POSITION", "X")
        filtered = db.list_channel_names("SR", "BPM", "POSITION", "X", sectors=[1])
        assert len(filtered) > 0, "Sector filtering returned empty"
        assert len(filtered) < len(all_bpm_x), "Sector filtering didn't reduce results"

        # Common names should be available
        names = db.get_common_names("SR", "BPM")
        assert names is not None, "CommonNames not found for SR/BPM"
        assert len(names) > 0, "CommonNames is empty"


class TestPerTierValidation:
    """Tests for per-tier validation mode of validate_queries()."""

    @pytest.fixture
    def tier_benchmark(self, tree_data, all_channels, tmp_path):
        """Create tier-specific benchmark databases and query files."""
        output_dir = tmp_path / "output"

        for tier_num, tier_spec in [(1, TIER_1), (2, TIER_2), (3, TIER_3)]:
            tier_dir = output_dir / f"tier{tier_num}"
            tier_dir.mkdir(parents=True)

            ic_data = format_in_context(all_channels, tier_spec)
            (tier_dir / "in_context.json").write_text(json.dumps(ic_data))

            hier_data = format_hierarchical(tree_data, tier_spec)
            (tier_dir / "hierarchical.json").write_text(json.dumps(hier_data))

            ml_data = format_middle_layer(all_channels, tier_spec)
            (tier_dir / "middle_layer.json").write_text(json.dumps(ml_data))

        # Create per-tier query files
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()

        t1_channels = filter_channels(all_channels, TIER_1)
        t1_pvs = [ch["pv"] for ch in t1_channels[:3]]
        t1_queries = [{"user_query": "find", "targeted_pv": t1_pvs}]
        (queries_dir / "t1.json").write_text(json.dumps(t1_queries))

        # Tier 3 query with BR channel (doesn't exist in Tier 1)
        t3_channels = filter_channels(all_channels, TIER_3)
        br_pvs = [ch["pv"] for ch in t3_channels if ch["ring"] == "BR"][:2]
        t3_queries = [{"user_query": "find BR", "targeted_pv": br_pvs}]
        (queries_dir / "t3.json").write_text(json.dumps(t3_queries))

        return output_dir, queries_dir, t1_pvs, br_pvs

    def test_per_tier_validation_passes(self, tier_benchmark):
        """Per-tier mode: each tier's queries validated against its own databases."""
        output_dir, queries_dir, _t1_pvs, _br_pvs = tier_benchmark
        result = validate_queries(
            tier_queries={
                1: queries_dir / "t1.json",
                3: queries_dir / "t3.json",
            },
            output_dir=output_dir,
        )
        assert result["valid"] is True
        assert result["missing"] == []

    def test_cross_tier_no_false_failure(self, tier_benchmark):
        """BR channels in Tier 3 queries do NOT fail against Tier 1."""
        output_dir, queries_dir, _t1_pvs, br_pvs = tier_benchmark
        # Validate T3 queries only against T3 databases
        result = validate_queries(
            tier_queries={3: queries_dir / "t3.json"},
            output_dir=output_dir,
        )
        assert result["valid"] is True

    def test_backward_compatible_single_file(self, tier_benchmark):
        """Old-style call still works: validate_queries(queries_path, db_dir)."""
        output_dir, queries_dir, _t1_pvs, _br_pvs = tier_benchmark
        result = validate_queries(queries_dir / "t1.json", output_dir)
        # May have missing since t1 PVs checked against all tier dirs
        # But the call itself should not error
        assert isinstance(result, dict)
        assert "valid" in result
        assert "missing" in result

    def test_missing_output_dir_raises(self):
        """Per-tier mode requires output_dir."""
        from pathlib import Path

        with pytest.raises(ValueError, match="output_dir"):
            validate_queries(tier_queries={1: Path("x.json")})


class TestLoadTemplate:
    """Tests for load_template() convenience function."""

    def test_default_path(self):
        """Default call loads the built-in hierarchical template."""
        tree_data, channels = load_template()
        assert isinstance(tree_data, dict)
        assert isinstance(channels, list)
        assert len(channels) == TIER_3.target_count

    def test_channels_have_required_keys(self):
        """Expanded channels have the expected key set."""
        _, channels = load_template()
        required = {"pv", "ring", "system", "family", "device", "field", "subfield"}
        for ch in channels[:5]:
            assert required.issubset(ch.keys())

    def test_custom_source(self, tmp_path):
        """load_template() accepts a custom source path."""
        # Create a minimal hierarchical template
        mini_template = {
            "SR": {
                "_description": "Test ring",
                "_expansion": {"rings": {"SR": "SR"}},
                "MAG": {
                    "_description": "Magnets",
                    "BPM": {
                        "_description": "BPMs",
                        "_expansion": {
                            "count": 2,
                            "device_prefix": "BPM",
                            "zero_pad": 2,
                        },
                        "POSITION": {
                            "_description": "Position",
                            "X": {"_description": "Horizontal"},
                            "Y": {"_description": "Vertical"},
                        },
                    },
                },
            },
        }
        src = tmp_path / "custom.json"
        src.write_text(json.dumps(mini_template))

        tree_data, channels = load_template(src)
        assert isinstance(tree_data, dict)
        assert "SR" in tree_data
        # Custom template may produce fewer channels
        assert isinstance(channels, list)
