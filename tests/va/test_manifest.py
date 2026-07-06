"""Tests for the namespace-union manifest generator.

docker/virtual-accelerator/manifest is importable here as a top-level module
name ("manifest") via the sys.path setup in tests/va/conftest.py -- see that
module's docstring.
"""

from __future__ import annotations

import pytest
from manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    build_manifest,
    derive_record_type,
    loaders,
)

# Measured tier-3 expansion counts (see docker/virtual-accelerator/manifest/paths.py
# for why tier 3 is the build-resolved default for this preset).
EXPECTED_RING_COUNTS = {"SR": 1066, "BR": 90, "BTS": 72}
EXPECTED_TOTAL = 1228
EXPECTED_SETPOINTS = 168


@pytest.fixture(scope="module")
def manifest() -> dict:
    return build_manifest()


class TestParadigmAgreement:
    """The three paradigm DBs must expand to identical address sets at tier 3.

    This is the load-bearing assumption behind treating "the namespace" as a
    single derived thing rather than three separate lists.
    """

    def test_hierarchical_in_context_middle_layer_agree(self):
        hier = {c.address for c in loaders.load_hierarchical_channels()}
        in_context = loaders.load_in_context_addresses()
        middle_layer = loaders.load_middle_layer_addresses()

        assert hier == in_context
        assert hier == middle_layer

    def test_build_manifest_does_not_raise_on_mismatch(self, manifest):
        # build_manifest() raises ParadigmMismatchError internally if the
        # paradigms disagree; reaching this point means they didn't.
        assert manifest["_metadata"]["total_channels"] > 0


class TestRingCounts:
    def test_sr_br_bts_counts_match_measured_values(self, manifest):
        assert manifest["_metadata"]["by_ring"] == EXPECTED_RING_COUNTS

    def test_total_channel_count(self, manifest):
        assert manifest["_metadata"]["total_channels"] == EXPECTED_TOTAL


class TestSetpointCount:
    def test_exactly_168_setpoint_writables(self, manifest):
        assert manifest["_metadata"]["setpoint_count"] == EXPECTED_SETPOINTS

    def test_setpoint_count_matches_actual_channel_tally(self, manifest):
        sp_channels = [c for c in manifest["channels"] if c["subfield"] == "SP"]
        assert len(sp_channels) == EXPECTED_SETPOINTS


class TestPartitionA_PyatCoupled:
    """Partition (a) must contain ONLY SR magnet-current and BPM-position channels."""

    def test_only_sr_magnet_or_bpm_channels(self, manifest):
        pyat = [c for c in manifest["channels"] if c["partition"] == PARTITION_PYAT_COUPLED]
        assert pyat, "expected at least one pyat-coupled channel"

        for c in pyat:
            assert c["ring"] == "SR", c
            is_magnet_current = c["system"] == "MAG" and c["field"] == "CURRENT"
            is_bpm_position = (
                c["system"] == "DIAG" and c["family"] == "BPM" and c["field"] == "POSITION"
            )
            assert is_magnet_current or is_bpm_position, c

    def test_no_br_or_bts_channels_in_pyat_coupled(self, manifest):
        pyat = [c for c in manifest["channels"] if c["partition"] == PARTITION_PYAT_COUPLED]
        rings = {c["ring"] for c in pyat}
        assert rings == {"SR"}

    def test_no_golden_or_status_channels_in_pyat_coupled(self, manifest):
        pyat = [c for c in manifest["channels"] if c["partition"] == PARTITION_PYAT_COUPLED]
        subfields = {c["subfield"] for c in pyat}
        assert "GOLDEN" not in subfields
        assert subfields <= {"SP", "RB", "X", "Y"}


class TestPartitionB_SpEcho:
    def test_all_br_bts_magnet_channels_are_sp_echo(self, manifest):
        br_bts_mag = [
            c for c in manifest["channels"] if c["ring"] in ("BR", "BTS") and c["system"] == "MAG"
        ]
        assert br_bts_mag, "expected BR/BTS magnet channels to exist"
        assert all(c["partition"] == PARTITION_SP_ECHO for c in br_bts_mag)

    def test_sp_echo_never_touches_sr_mag_or_diag(self, manifest):
        sp_echo = [c for c in manifest["channels"] if c["partition"] == PARTITION_SP_ECHO]
        for c in sp_echo:
            if c["ring"] == "SR":
                assert c["system"] in ("RF", "VAC"), c


class TestPartitionC_StaticNoisy:
    def test_golden_channels_are_static_noisy(self, manifest):
        golden = [c for c in manifest["channels"] if c["subfield"] == "GOLDEN"]
        assert golden, "expected GOLDEN reference channels to exist"
        assert all(c["partition"] == PARTITION_STATIC_NOISY for c in golden)

    def test_sr_status_channels_are_static_noisy(self, manifest):
        # BR/BTS MAG status channels are deliberately sp-echo: the spec
        # classifies ALL BR/BTS magnet channels there, with no per-field
        # carve-out. Only SR status channels (outside the RF/VAC sp-echo
        # setpoint/readback fields) are expected to land in static-noisy.
        status = [c for c in manifest["channels"] if c["field"] == "STATUS" and c["ring"] == "SR"]
        assert status, "expected SR STATUS channels to exist"
        assert all(c["partition"] == PARTITION_STATIC_NOISY for c in status)

    def test_partitions_are_exhaustive_and_disjoint(self, manifest):
        valid_partitions = {PARTITION_PYAT_COUPLED, PARTITION_SP_ECHO, PARTITION_STATIC_NOISY}
        for c in manifest["channels"]:
            assert c["partition"] in valid_partitions


class TestRecordTypeDerivation:
    def test_status_fields_are_binary_without_noise(self):
        path = {
            "ring": "SR",
            "system": "MAG",
            "family": "DIPOLE",
            "device": "01",
            "field": "STATUS",
            "subfield": "FAULT",
        }
        record_type, noise = derive_record_type(path)
        assert record_type == RECORD_TYPE_BINARY
        assert noise is False

    def test_current_readback_is_analog_with_noise(self):
        path = {
            "ring": "SR",
            "system": "MAG",
            "family": "DIPOLE",
            "device": "01",
            "field": "CURRENT",
            "subfield": "RB",
        }
        record_type, noise = derive_record_type(path)
        assert record_type == RECORD_TYPE_ANALOG
        assert noise is True

    def test_valve_position_open_closed_is_binary(self):
        path = {
            "ring": "SR",
            "system": "VAC",
            "family": "VALVE",
            "device": "01",
            "field": "POSITION",
            "subfield": "OPEN",
        }
        record_type, noise = derive_record_type(path)
        assert record_type == RECORD_TYPE_BINARY
        assert noise is False

    def test_manifest_channels_only_use_bi_or_ai(self, manifest):
        # The current namespace has no genuinely string-valued channel.
        record_types = {c["record_type"] for c in manifest["channels"]}
        assert record_types == {RECORD_TYPE_BINARY, RECORD_TYPE_ANALOG}

    def test_bi_channels_never_have_noise(self, manifest):
        for c in manifest["channels"]:
            if c["record_type"] == RECORD_TYPE_BINARY:
                assert c["noise"] is False, c


class TestStructuralIntegrity:
    def test_no_duplicate_addresses(self, manifest):
        addresses = [c["address"] for c in manifest["channels"]]
        assert len(addresses) == len(set(addresses))

    def test_addresses_match_naming_grammar(self, manifest):
        for c in manifest["channels"]:
            if not c["ring"]:
                continue  # machine.json-only entries carry no hierarchy path
            expected = ":".join(
                [c["ring"], c["system"], c["family"], c["device"], c["field"], c["subfield"]]
            )
            assert c["address"] == expected

    def test_machine_json_channels_are_all_within_the_manifest(self, manifest):
        machine_json_channels = loaders.load_machine_json_channels()
        manifest_addresses = {c["address"] for c in manifest["channels"]}
        assert set(machine_json_channels) <= manifest_addresses

    def test_machine_json_fully_subsumed_by_db_namespace(self, manifest):
        # Currently machine.json's 78 channels are all already promised by
        # the paradigm DBs -- this pins that fact so a future addition of a
        # genuinely novel machine.json channel is a visible manifest change,
        # not a silent one.
        assert manifest["_metadata"]["machine_json_novel_addresses"] == []


class TestMachineStateReconciliation:
    """machine_state_channels.json.j2's branches are known-broken/fictional
    (a separate task fixes the template against this manifest); the manifest
    still reports what it checked so that follow-up work has ground truth.
    """

    def test_reconciliation_report_present(self, manifest):
        report = manifest["_metadata"]["machine_state_reconciliation"]
        assert report["candidates_checked"] > 0
        assert report["candidates_checked"] == len(report["valid"]) + len(report["invalid"])
