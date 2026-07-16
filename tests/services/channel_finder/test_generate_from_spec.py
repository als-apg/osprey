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
    TIER3_FILENAMES,
    address_set,
    generate,
    verify_cross_paradigm_identity,
)
from osprey.simulation.channel_schema import CHANNEL_SCHEMA
from osprey.simulation.facility_spec import ALS_U_AR

SHIPPED_TIER3_DIR = Path(
    "src/osprey/templates/apps/control_assistant/data/channel_databases/tiers/tier3"
)

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
