"""Tests for cross-paradigm benchmark generator."""

from __future__ import annotations

import json
import shutil

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
    TEMPLATE_DATA_DIR,
    TEMPLATE_DB_PATH,
    TIER_PARADIGMS,
    expand_hierarchy,
    generate_alias,
    generate_description,
    load_template,
    validate_queries,
)
from osprey.services.channel_finder.tools.generate_from_spec import TIER1_FILTER

# The shipped tier databases: tier 1 is the TIER1_FILTER subset of tier 3, and
# tier 3 is the whole expanded template.
TIERS_ROOT = TEMPLATE_DATA_DIR / "channel_databases" / "tiers"


def _tier1(channels: list[dict]) -> list[dict]:
    """The tier-1 subset of expanded channels, per :data:`TIER1_FILTER`."""
    return [ch for ch in channels if TIER1_FILTER.matches(ch["pv"])]


def _copy_shipped_tiers(dest) -> None:
    """Copy every shipped tier database into ``dest/tier<N>/``."""
    for tier_num, paradigms in TIER_PARADIGMS.items():
        tier_dir = dest / f"tier{tier_num}"
        tier_dir.mkdir(parents=True)
        for paradigm in paradigms:
            shutil.copy(TIERS_ROOT / f"tier{tier_num}" / f"{paradigm}.json", tier_dir)


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
                    "device": "01",
                    "field": "CURRENT",
                    "subfield": "SP",
                },
                "Storage ring dipole bending magnet 01 current setpoint",
            ),
            (
                {
                    "ring": "SR",
                    "system": "DIAG",
                    "family": "BPM",
                    "device": "01",
                    "field": "POSITION",
                    "subfield": "X",
                },
                "Storage ring beam position monitor 01 position horizontal",
            ),
            (
                {
                    "ring": "BR",
                    "system": "MAG",
                    "family": "QF",
                    "device": "01",
                    "field": "STATUS",
                    "subfield": "FAULT",
                },
                "Booster ring focusing quadrupole 01 status fault",
            ),
            (
                {
                    "ring": "BTS",
                    "system": "MAG",
                    "family": "VCM",
                    "device": "01",
                    "field": "CURRENT",
                    "subfield": "RB",
                },
                "Booster-to-storage transfer line vertical corrector 01 current readback",
            ),
            (
                {
                    "ring": "SR",
                    "system": "RF",
                    "family": "CAVITY",
                    "device": "01",
                    "field": "POWER",
                    "subfield": "FWD",
                },
                "Storage ring RF cavity 01 power forward",
            ),
            (
                {
                    "ring": "SR",
                    "system": "VAC",
                    "family": "ION-PUMP",
                    "device": "01",
                    "field": "PRESSURE",
                    "subfield": "RB",
                },
                "Storage ring ion pump 01 pressure readback",
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


class TestValidateQueries:
    """Tests for validate_queries()."""

    @pytest.fixture
    def mini_benchmark(self, all_channels, tmp_path):
        """Copy the shipped tier databases and write queries over tier-1 PVs."""
        _copy_shipped_tiers(tmp_path)

        # Pick a few PVs that exist in the tier-1 database (and so in tier 3)
        tier1_channels = _tier1(all_channels)
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
        (db_dir / "tier3" / "middle_layer.json").unlink()

        result = validate_queries(queries_path, db_dir)
        assert result["valid"] is False
        assert len(result["missing_databases"]) == 1

    def test_empty_queries(self, tmp_path):
        """Empty query list is valid."""
        queries_path = tmp_path / "queries.json"
        queries_path.write_text("[]")
        for t in (1, 3):
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
                    "device": "05",
                    "field": "CURRENT",
                    "subfield": "SP",
                },
                "StorageRing_Dipole_05_Current_Setpoint",
            ),
            (
                {
                    "ring": "SR",
                    "system": "DIAG",
                    "family": "BPM",
                    "device": "01",
                    "field": "POSITION",
                    "subfield": "X",
                },
                "StorageRing_BPM_01_Position_X",
            ),
            (
                {
                    "ring": "SR",
                    "system": "RF",
                    "family": "CAVITY",
                    "device": "02",
                    "field": "VOLTAGE",
                    "subfield": "RB",
                },
                "StorageRing_Cavity_02_Voltage_Readback",
            ),
            (
                {
                    "ring": "BR",
                    "system": "MAG",
                    "family": "DIPOLE",
                    "device": "01",
                    "field": "CURRENT",
                    "subfield": "SP",
                },
                "BoosterRing_Dipole_01_Current_Setpoint",
            ),
            (
                {
                    "ring": "BTS",
                    "system": "DIAG",
                    "family": "BPM",
                    "device": "01",
                    "field": "POSITION",
                    "subfield": "Y",
                },
                "BoosterToStorageRing_BPM_01_Position_Y",
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
        for ch in _tier1(all_channels):
            alias = generate_alias(ch)
            assert "_" in alias, f"Alias missing underscores: {alias}"
            assert ":" not in alias, f"Alias contains colons: {alias}"
            # Should have exactly 4 underscores (5 parts)
            assert alias.count("_") == 4, f"Alias has wrong number of parts: {alias}"


class TestPerTierValidation:
    """Tests for per-tier validation mode of validate_queries()."""

    @pytest.fixture
    def tier_benchmark(self, all_channels, tmp_path):
        """Copy the shipped tier databases and write per-tier query files."""
        output_dir = tmp_path / "output"
        _copy_shipped_tiers(output_dir)

        # Create per-tier query files
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()

        t1_channels = _tier1(all_channels)
        t1_pvs = [ch["pv"] for ch in t1_channels[:3]]
        t1_queries = [{"user_query": "find", "targeted_pv": t1_pvs}]
        (queries_dir / "t1.json").write_text(json.dumps(t1_queries))

        # Tier 3 query with BR channel (doesn't exist in Tier 1)
        br_pvs = [ch["pv"] for ch in all_channels if ch["ring"] == "BR"][:2]
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


class TestMaterializedTierDatabases:
    """Verify the on-disk tier DBs shipped with the control_assistant preset.

    The materialized JSON files under
    src/osprey/templates/.../channel_databases/tiers/{tier1,tier3}/ must not
    drift from the expanded template and TIER1_FILTER — e.g. someone bumps the
    template or the filter but forgets to re-run
    scripts/generate_tier_databases.py for every (tier, paradigm) combination.
    """

    @staticmethod
    def _count_hierarchical(path) -> int:
        """Sum (device_count × subfield_leaf_count) across the tree."""
        raw = json.loads(path.read_text())
        tree = raw.get("tree", raw)
        n = 0

        def expansion_count(exp: dict) -> int:
            if exp["_type"] == "range":
                lo, hi = exp["_range"]
                return hi - lo + 1
            if exp["_type"] == "list":
                return len(exp["_instances"])
            raise ValueError(f"Unknown expansion type: {exp['_type']}")

        def recurse(node, ring=None):
            nonlocal n
            if not isinstance(node, dict):
                return
            for k, v in node.items():
                if k.startswith("_") or not isinstance(v, dict):
                    continue
                if k in ("SR", "BR", "BTS") and ring is None:
                    recurse(v, ring=k)
                    continue
                if "DEVICE" in v and "_expansion" in v.get("DEVICE", {}):
                    dev = v["DEVICE"]
                    ndev = expansion_count(dev["_expansion"])
                    for fk, fv in dev.items():
                        if fk.startswith("_") or not isinstance(fv, dict):
                            continue
                        for sk, sv in fv.items():
                            if sk.startswith("_") or not isinstance(sv, dict):
                                continue
                            n += ndev
                    continue
                recurse(v, ring=ring)

        recurse(tree)
        return n

    @staticmethod
    def _count_in_context(path) -> int:
        """Envelope schema: {_metadata, channels: [...]}."""
        data = json.loads(path.read_text())
        return len(data["channels"])

    @staticmethod
    def _count_middle_layer(path) -> int:
        """ring → family → field → subfield → {ChannelNames: [...]}."""
        data = json.loads(path.read_text())
        n = 0

        def recurse(node):
            nonlocal n
            if not isinstance(node, dict):
                return
            if "ChannelNames" in node and isinstance(node["ChannelNames"], list):
                n += len(node["ChannelNames"])
                return
            for k, v in node.items():
                if k.startswith("_"):
                    continue
                recurse(v)

        recurse(data)
        return n

    _COUNTERS = {
        "hierarchical": _count_hierarchical.__func__,
        "in_context": _count_in_context.__func__,
        "middle_layer": _count_middle_layer.__func__,
    }

    @pytest.mark.parametrize(
        ("tier", "paradigm"),
        [
            # Tier 1 ships only the in_context paradigm; tier 3 ships every
            # tier view. Driven off TIER_PARADIGMS so the ``graph`` exemption
            # (no tier database — the store is seeded from the corpus TTL)
            # stays a subtraction from the registry rather than a list here.
            *((tier, paradigm) for tier in TIER_PARADIGMS for paradigm in TIER_PARADIGMS[tier]),
        ],
        ids=str,
    )
    def test_materialized_db_matches_target_count(
        self, tier: int, paradigm: str, all_channels: list[dict]
    ):
        """Each shipped tier DB file must enumerate exactly the tier's channels.

        Catches: stale regen (one paradigm forgotten), hand-edits, template or
        filter changes without rerunning generate_tier_databases.py.
        """
        expected = len(_tier1(all_channels) if tier == 1 else all_channels)
        path = TIERS_ROOT / f"tier{tier}" / f"{paradigm}.json"
        assert path.exists(), f"Missing materialized DB: {path}"
        counter = self._COUNTERS[paradigm]
        n = counter(path)
        assert n == expected, (
            f"tier{tier}/{paradigm}.json has {n} channels, "
            f"expected {expected}. Re-run "
            f"scripts/generate_tier_databases.py to regenerate."
        )
