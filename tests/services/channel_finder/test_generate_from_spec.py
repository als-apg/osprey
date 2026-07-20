"""Tests for the tier-3 channel-database generator.

Operates entirely on temp copies of the shipped tier-3 JSONs -- the
committed databases under ``templates/apps/control_assistant/data/
channel_databases/tiers/tier3/`` are never modified by this test.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase
from osprey.services.channel_finder.databases.template import ChannelDatabase
from osprey.services.channel_finder.tools.generate_from_spec import (
    TIER1_FILTER,
    TIER3_FILENAMES,
    address_set,
    apply_tier1_filter,
    generate,
    verify_cross_paradigm_identity,
)
from osprey.simulation.channel_schema import CHANNEL_SCHEMA
from osprey.simulation.facility_spec import ALS_U_AR

# Anchored to the repo root via __file__ -- a cwd-relative path here makes the
# whole module's fixtures fail when any earlier test in a full-suite run leaves
# the process cwd changed.
_CHANNEL_DB_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/osprey/templates/apps/control_assistant/data/channel_databases"
)
SHIPPED_TIER3_DIR = _CHANNEL_DB_DIR / "tiers/tier3"
SHIPPED_TIER1_DIR = _CHANNEL_DB_DIR / "tiers/tier1"

_GROWN_COUNTS: dict[str, int] = {fam.name: fam.count for fam in ALS_U_AR.families}
_NEW_FAMILIES = {"QFA": "QF", "SHF": "SF", "SHD": "SD"}


@pytest.fixture()
def tier3_copy(tmp_path: Path) -> Path:
    """Copy the shipped tier-3 JSONs into a temp directory."""
    dest = tmp_path / "tier3"
    dest.mkdir()
    for filename in TIER3_FILENAMES.values():
        shutil.copy2(SHIPPED_TIER3_DIR / filename, dest / filename)
    return dest


class TestCrossParadigmIdentity:
    def test_generated_dbs_share_one_address_set(self, tier3_copy: Path):
        written = generate(tier3_copy)
        # generate() already asserts this internally; re-derive independently
        # here via the same parser-reuse helper for direct test coverage.
        verify_cross_paradigm_identity(
            written["hierarchical"], written["in_context"], written["middle_layer"]
        )

    def test_addresses_actually_match(self, tier3_copy: Path):
        written = generate(tier3_copy)
        hier_addrs = address_set(written["hierarchical"], HierarchicalChannelDatabase)
        ctx_addrs = address_set(written["in_context"], ChannelDatabase)
        ml_addrs = address_set(written["middle_layer"], MiddleLayerDatabase)
        assert hier_addrs == ctx_addrs == ml_addrs
        assert len(hier_addrs) > 0


class TestGrownInventory:
    @pytest.fixture()
    def generated(self, tier3_copy: Path) -> Path:
        generate(tier3_copy)
        return tier3_copy

    @pytest.mark.parametrize("family", sorted(_GROWN_COUNTS))
    def test_family_grows_to_target_count_via_parsers(self, generated: Path, family: str):
        count = _GROWN_COUNTS[family]
        schema = CHANNEL_SCHEMA[family]

        hier = HierarchicalChannelDatabase(str(generated / "hierarchical.json"))
        ctx = ChannelDatabase(str(generated / "in_context.json"))
        ml = MiddleLayerDatabase(str(generated / "middle_layer.json"))

        for db in (hier, ctx, ml):
            addrs = {c["address"] for c in db.get_all_channels()}
            device_ids = {
                addr.split(":")[3]
                for addr in addrs
                if addr.startswith(f"SR:{schema.system}:{family}:")
            }
            assert device_ids == {f"{i:02d}" for i in range(1, count + 1)}, (
                f"{family} in {db.__class__.__name__} did not grow to {count} devices"
            )

    def test_bpm_hcm_vcm_grow_to_72(self, generated: Path):
        for family in ("BPM", "HCM", "VCM"):
            assert _GROWN_COUNTS[family] == 72

    def test_dipole_grows_to_36(self, generated: Path):
        assert _GROWN_COUNTS["DIPOLE"] == 36

    def test_remaining_magnet_families_grow_to_24(self, generated: Path):
        for family in ("QF", "QD", "QFA", "SF", "SD", "SHF", "SHD"):
            assert _GROWN_COUNTS[family] == 24


class TestNewFamiliesStructuralClone:
    @pytest.fixture()
    def generated(self, tier3_copy: Path) -> Path:
        generate(tier3_copy)
        return tier3_copy

    @pytest.mark.parametrize("new_family,analog", sorted(_NEW_FAMILIES.items()))
    def test_new_family_present_with_full_channel_set(
        self, generated: Path, new_family: str, analog: str
    ):
        ml = MiddleLayerDatabase(str(generated / "middle_layer.json"))
        sr = ml.data["SR"]
        assert new_family in sr

        new_schema = CHANNEL_SCHEMA[new_family]
        analog_schema = CHANNEL_SCHEMA[analog]

        # Same field/subfield shape as the structural analog.
        assert set(new_schema.fields) == set(analog_schema.fields)
        for field, subs in new_schema.fields.items():
            assert set(subs) == set(analog_schema.fields[field])

        # STATUS subfields are enum/unitless; CURRENT subfields are double/A
        # -- both for the new family and its analog, matching the shared
        # metadata shape.
        for field in ("CURRENT", "STATUS"):
            for subfield, subschema in new_schema.fields[field].items():
                if field == "STATUS":
                    assert subschema.data_type == "enum"
                    assert subschema.hw_units == ""
                else:
                    assert subschema.data_type == "double"
                    assert subschema.hw_units == "A"
                analog_subschema = analog_schema.fields[field][subfield]
                assert subschema.data_type == analog_subschema.data_type
                assert subschema.hw_units == analog_subschema.hw_units

    def test_new_family_channels_parse_through_all_three_formats(
        self, generated: Path
    ):
        hier = HierarchicalChannelDatabase(str(generated / "hierarchical.json"))
        ctx = ChannelDatabase(str(generated / "in_context.json"))
        ml = MiddleLayerDatabase(str(generated / "middle_layer.json"))

        for db in (hier, ctx, ml):
            addrs = {c["address"] for c in db.get_all_channels()}
            assert "SR:MAG:QFA:01:CURRENT:SP" in addrs
            assert "SR:MAG:SHF:24:STATUS:READY" in addrs
            assert "SR:MAG:SHD:12:STATUS:ON" in addrs


class TestMergePreserve:
    def test_non_sr_ring_addresses_untouched(self, tier3_copy: Path):
        before_ml = json.loads((tier3_copy / "middle_layer.json").read_text())
        before_hier = json.loads((tier3_copy / "hierarchical.json").read_text())
        before_ctx = json.loads((tier3_copy / "in_context.json").read_text())

        # Sample a BR address and an untouched SR system before generation.
        br_dipole_before = before_ml["BR"]["DIPOLE"]
        sr_rf_before = before_hier["tree"]["SR"]["RF"]
        br_addr_entries_before = [
            c for c in before_ctx["channels"] if c["address"] == "BR:MAG:DIPOLE:01:CURRENT:RB"
        ]

        generate(tier3_copy)

        after_ml = json.loads((tier3_copy / "middle_layer.json").read_text())
        after_hier = json.loads((tier3_copy / "hierarchical.json").read_text())
        after_ctx = json.loads((tier3_copy / "in_context.json").read_text())

        assert after_ml["BR"]["DIPOLE"] == br_dipole_before
        assert after_hier["tree"]["SR"]["RF"] == sr_rf_before
        br_addr_entries_after = [
            c for c in after_ctx["channels"] if c["address"] == "BR:MAG:DIPOLE:01:CURRENT:RB"
        ]
        assert br_addr_entries_after == br_addr_entries_before

    def test_existing_sr_device_entry_preserved_verbatim(self, tier3_copy: Path):
        before_ctx = json.loads((tier3_copy / "in_context.json").read_text())
        before_entry = next(
            c for c in before_ctx["channels"] if c["address"] == "SR:MAG:DIPOLE:01:CURRENT:SP"
        )

        generate(tier3_copy)

        after_ctx = json.loads((tier3_copy / "in_context.json").read_text())
        after_entry = next(
            c for c in after_ctx["channels"] if c["address"] == "SR:MAG:DIPOLE:01:CURRENT:SP"
        )
        assert after_entry == before_entry


class TestIdempotent:
    def test_rerunning_generator_is_byte_identical(self, tier3_copy: Path):
        generate(tier3_copy)
        first_pass = {
            filename: (tier3_copy / filename).read_bytes()
            for filename in TIER3_FILENAMES.values()
        }

        generate(tier3_copy)
        second_pass = {
            filename: (tier3_copy / filename).read_bytes()
            for filename in TIER3_FILENAMES.values()
        }

        assert first_pass == second_pass


def _load_channels(path: Path) -> list[dict[str, str]]:
    """Return the ``channels`` list of a committed ``in_context.json``."""
    return json.loads(path.read_text())["channels"]


class TestTier1FilterDeclaration:
    """Unit coverage for the ``TIER1_FILTER`` predicate itself (no I/O)."""

    def test_declared_families_match_the_spec_plus_non_spec_set(self):
        assert TIER1_FILTER.families == {
            "DIPOLE",
            "QF",
            "HCM",
            "VCM",
            "DCCT",
            "BPM",
            "CAVITY",
            "GAUGE",
        }

    def test_ring_and_subfields_are_the_declared_analog_leaves(self):
        assert TIER1_FILTER.rings == {"SR"}
        assert TIER1_FILTER.subfields == {"SP", "RB", "X", "Y"}

    @pytest.mark.parametrize(
        "address",
        [
            "SR:MAG:QF:01:CURRENT:SP",
            "SR:MAG:QF:24:CURRENT:RB",
            "SR:MAG:DIPOLE:01:CURRENT:SP",
            "SR:MAG:HCM:72:CURRENT:RB",
            "SR:MAG:VCM:01:CURRENT:SP",
            "SR:DIAG:BPM:01:POSITION:X",
            "SR:DIAG:BPM:72:POSITION:Y",
            "SR:DIAG:DCCT:01:CURRENT:RB",
            "SR:RF:CAVITY:01:VOLTAGE:SP",
            "SR:VAC:GAUGE:01:PRESSURE:RB",
        ],
    )
    def test_matches_tier1_addresses(self, address: str):
        assert TIER1_FILTER.matches(address)

    @pytest.mark.parametrize(
        "address",
        [
            "BR:MAG:DIPOLE:01:CURRENT:SP",  # non-SR ring
            "SR:MAG:QD:01:CURRENT:SP",  # family not in tier 1
            "SR:MAG:QFA:01:CURRENT:SP",  # spec family excluded from tier 1
            "SR:MAG:QF:01:STATUS:READY",  # STATUS field, not the family's tier-1 field
            "SR:MAG:QF:01:CURRENT:GOLDEN",  # GOLDEN subfield excluded
            "SR:DIAG:BPM:01:OFFSET:X",  # BPM tier-1 field is POSITION only
            "SR:DIAG:BPM:01:GOLDEN:X",  # BPM GOLDEN excluded
            "SR:RF:CAVITY:01:CURRENT:SP",  # wrong field for CAVITY (VOLTAGE only)
            "SR:MAG:QF:01:CURRENT",  # malformed (5 parts)
        ],
    )
    def test_rejects_non_tier1_addresses(self, address: str):
        assert not TIER1_FILTER.matches(address)


class TestTier1FilterReproducesCommittedDb:
    """``apply_tier1_filter(tier3)`` must reproduce the committed tier-1 DB
    entry-for-entry. Counts are always computed from the data, never pinned."""

    def test_filter_reproduces_committed_tier1_entry_for_entry(self):
        tier3_channels = _load_channels(SHIPPED_TIER3_DIR / "in_context.json")
        committed_tier1 = _load_channels(SHIPPED_TIER1_DIR / "in_context.json")

        filtered = apply_tier1_filter(tier3_channels)

        # Count is derived from both sides, never asserted against a literal.
        assert len(filtered) == len(committed_tier1)

        # Entry-for-entry equality, keyed by address (the committed tier-1 file
        # predates this pipeline and carries a different document order; the
        # generator task later rewrites it into filter order).
        filtered_by_addr = {c["address"]: c for c in filtered}
        committed_by_addr = {c["address"]: c for c in committed_tier1}
        assert filtered_by_addr == committed_by_addr

        # No address collisions collapsed the count (keys are unique).
        assert len(filtered_by_addr) == len(filtered)

    def test_filter_selects_entries_verbatim_from_tier3(self):
        tier3_channels = _load_channels(SHIPPED_TIER3_DIR / "in_context.json")
        tier3_by_addr = {c["address"]: c for c in tier3_channels}

        for entry in apply_tier1_filter(tier3_channels):
            # Selected, never rewritten: the object is the tier-3 entry itself.
            assert entry == tier3_by_addr[entry["address"]]

    def test_filter_output_preserves_tier3_order(self):
        tier3_channels = _load_channels(SHIPPED_TIER3_DIR / "in_context.json")
        filtered = apply_tier1_filter(tier3_channels)

        expected_order = [
            c["address"] for c in tier3_channels if TIER1_FILTER.matches(c["address"])
        ]
        assert [c["address"] for c in filtered] == expected_order

    def test_every_committed_tier1_family_is_declared(self):
        committed_tier1 = _load_channels(SHIPPED_TIER1_DIR / "in_context.json")
        families = {c["address"].split(":")[2] for c in committed_tier1}
        assert families == TIER1_FILTER.families


@pytest.fixture()
def tiers_copy(tmp_path: Path) -> Path:
    """Copy the shipped ``tiers/{tier1,tier3}`` layout into a temp directory."""
    dest = tmp_path / "tiers"
    shutil.copytree(SHIPPED_TIER3_DIR, dest / "tier3")
    shutil.copytree(SHIPPED_TIER1_DIR, dest / "tier1")
    return dest


class TestGenerateEmitsTier1:
    """``generate()`` regenerates tier 1 from tier 3 when the sibling DB is
    present, and leaves an already-current tier 1 byte-identical."""

    def test_generate_returns_tier1_path_when_sibling_present(self, tiers_copy: Path):
        written = generate(tiers_copy / "tier3")
        assert "tier1_in_context" in written
        assert written["tier1_in_context"] == tiers_copy / "tier1" / "in_context.json"

    def test_regenerated_tier1_is_byte_identical_to_committed(self, tiers_copy: Path):
        generate(tiers_copy / "tier3")
        regenerated = (tiers_copy / "tier1" / "in_context.json").read_bytes()
        committed = (SHIPPED_TIER1_DIR / "in_context.json").read_bytes()
        assert regenerated == committed

    def test_regenerated_tier1_equals_filter_of_tier3_entry_for_entry(self, tiers_copy: Path):
        generate(tiers_copy / "tier3")
        tier3_channels = _load_channels(tiers_copy / "tier3" / "in_context.json")
        tier1_channels = _load_channels(tiers_copy / "tier1" / "in_context.json")

        filtered = apply_tier1_filter(tier3_channels)
        assert len(tier1_channels) == len(filtered)
        assert {c["address"]: c for c in tier1_channels} == {
            c["address"]: c for c in filtered
        }

    def test_tier1_metadata_total_channels_is_computed(self, tiers_copy: Path):
        generate(tiers_copy / "tier3")
        data = json.loads((tiers_copy / "tier1" / "in_context.json").read_text())
        assert data["_metadata"]["total_channels"] == len(data["channels"])

    def test_generate_is_byte_idempotent_over_both_tiers(self, tiers_copy: Path):
        generate(tiers_copy / "tier3")
        first = {
            p.relative_to(tiers_copy): p.read_bytes()
            for p in tiers_copy.rglob("*.json")
        }
        generate(tiers_copy / "tier3")
        second = {
            p.relative_to(tiers_copy): p.read_bytes()
            for p in tiers_copy.rglob("*.json")
        }
        assert first == second

    def test_generate_skips_tier1_when_sibling_absent(self, tier3_copy: Path):
        # The isolated tier-3 fixture has no sibling tier1/ dir: tier 1 is
        # simply not emitted, and tier-3 generation is unaffected.
        written = generate(tier3_copy)
        assert "tier1_in_context" not in written
        assert not (tier3_copy.parent / "tier1").exists()
