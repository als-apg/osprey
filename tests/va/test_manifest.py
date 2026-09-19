"""Tests for the namespace-union manifest generator."""

from __future__ import annotations

from collections import Counter

import pytest

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.services.virtual_accelerator.bindings import BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    build_manifest,
    derive_record_type,
    loaders,
)
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths

# Measured tier-3 expansion counts (see
# src/osprey/services/virtual_accelerator/manifest/paths.py for why tier 3
# is the build-resolved default for this preset).
#
# The ring tally counts only the channels that carry a ring token. A coupled
# channel is keyed on its binding instead (an empty ring, system, family and
# field -- see TestPartitionA_PyatCoupled), so the 840 pyat-coupled entries
# are outside this tally and inside EXPECTED_TOTAL: 1906 + 90 + 72 + 840.
EXPECTED_RING_COUNTS = {"SR": 1906, "BR": 90, "BTS": 72}
EXPECTED_TOTAL = 2908
EXPECTED_SETPOINTS = 396


@pytest.fixture(scope="module")
def manifest() -> dict:
    return build_manifest()


def _bindings_document() -> BindingsDocument:
    """The bundled tree's bindings, read the way the generator reads them."""
    return load_bindings(PACKAGE_PATHS.va_bindings)


def _claimed_addresses() -> set[str]:
    """Every address the bindings claim: each setpoint, and each readback."""
    claimed: set[str] = set()
    for binding in _bindings_document().bindings:
        claimed.add(binding.setpoint_address)
        if binding.readback_address is not None:
            claimed.add(binding.readback_address)
    return claimed


class TestParadigmAgreement:
    """The three paradigm DBs must expand to identical address sets at tier 3.

    This is the load-bearing assumption behind treating "the namespace" as a
    single derived thing rather than three separate lists.
    """

    def test_graph_is_the_only_paradigm_outside_this_gate(self):
        """The manifest is three-paradigm by exemption, not by oversight.

        ``graph`` ships no tiered database, so there is nothing here to expand
        and compare; its namespace agreement is pinned one step earlier, at the
        corpus, by ``tests/services/facility_knowledge/test_demo_ttl_consistency
        .py::test_demo_ttl_bindings_equal_the_channel_database``. Registering
        any *other* paradigm fails this assertion, which is the point: a
        file-backed paradigm must join the agreement gate above.
        """
        assert set(VALID_CHANNEL_FINDER_MODES) - {"graph"} == {
            "hierarchical",
            "in_context",
            "middle_layer",
        }

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

    def test_the_entries_carry_the_ring_tally_the_metadata_states(self, manifest):
        """The summary is checked against the entries it summarises.

        ``_metadata`` is written by the generator that writes the entries, so
        a tally read back from it alone would agree with itself however the
        entries came out. Counting the entries is the independent half.
        """
        tally = Counter(c["ring"] for c in manifest["channels"] if c["ring"])
        assert dict(tally) == EXPECTED_RING_COUNTS

    def test_the_ringless_entries_are_the_coupled_ones_and_the_two_tile(self, manifest):
        """Every channel is either keyed on a ring token or on its binding.

        A coupled channel is keyed on the binding that claims it, so it leaves
        the identity keys -- the ring among them -- empty. The two groups
        therefore have to partition the manifest exactly, with nothing counted
        twice and nothing outside both.
        """
        ringless = {c["address"] for c in manifest["channels"] if not c["ring"]}
        coupled = {
            c["address"] for c in manifest["channels"] if c["partition"] == PARTITION_PYAT_COUPLED
        }
        assert ringless == coupled
        assert sum(EXPECTED_RING_COUNTS.values()) + len(coupled) == EXPECTED_TOTAL


class TestSetpointCount:
    def test_exactly_396_setpoint_writables(self, manifest):
        assert manifest["_metadata"]["setpoint_count"] == EXPECTED_SETPOINTS

    def test_setpoint_count_matches_actual_channel_tally(self, manifest):
        sp_channels = [c for c in manifest["channels"] if c["subfield"] == "SP"]
        assert len(sp_channels) == EXPECTED_SETPOINTS


class TestPartitionA_PyatCoupled:
    """Partition (a) is the tree's bindings document, address for address.

    What a write steers the beam with is read off ``simulation/va_bindings.json``
    and nothing else -- not the ring a name starts with, not the system token in
    the middle of it. A partition keyed on either would be one facility's naming
    convention standing in for a physics fact, so the checks here ask the
    document rather than the address text.
    """

    def test_the_partition_is_exactly_what_the_bindings_claim(self, manifest):
        pyat = {
            c["address"] for c in manifest["channels"] if c["partition"] == PARTITION_PYAT_COUPLED
        }
        assert pyat, "expected at least one pyat-coupled channel"
        assert pyat == _claimed_addresses()

    def test_every_coupled_channel_carries_the_pair_key_shape(self, manifest):
        """The bindings pair the two halves, so the identity keys stay empty.

        A hierarchy path could disagree with the document about which addresses
        are one device's two halves, so the pair is keyed on the one thing that
        settles it: the binding's own setpoint address, carried in ``device``.
        """
        setpoints = {binding.setpoint_address for binding in _bindings_document().bindings}
        for c in manifest["channels"]:
            if c["partition"] != PARTITION_PYAT_COUPLED:
                continue
            assert (c["ring"], c["system"], c["family"], c["field"]) == ("", "", "", ""), c
            assert c["device"] in setpoints, c

    def test_the_partition_stays_inside_the_system_the_bindings_describe(self):
        """One document describes one accelerator; it claims nothing outside it."""
        document = _bindings_document()
        outside = sorted(
            address
            for address in _claimed_addresses()
            if not address.startswith(f"{document.system}:")
        )
        assert not outside, (
            f"the bindings describe system {document.system!r} but claim addresses "
            f"outside it: {outside[:10]}"
        )

    def test_no_golden_or_status_channels_in_pyat_coupled(self, manifest):
        pyat = [c for c in manifest["channels"] if c["partition"] == PARTITION_PYAT_COUPLED]
        subfields = {c["subfield"] for c in pyat}
        assert "GOLDEN" not in subfields
        assert subfields <= {"SP", "RB", "X", "Y"}


class TestPartitionB_SpEcho:
    def test_br_bts_magnet_channels_echo_exactly_where_they_pair(self, manifest):
        """The transport lines carry no model, so a write there can only echo.

        Which of their channels echo is the tree's own pairing and not their
        family: a setpoint and the readback that reports it are a pair, and a
        status bit is a reading with no setpoint behind it, so it stays in the
        partition the simulation engine drives.
        """
        br_bts_mag = [
            c for c in manifest["channels"] if c["ring"] in ("BR", "BTS") and c["system"] == "MAG"
        ]
        assert br_bts_mag, "expected BR/BTS magnet channels to exist"
        for c in br_bts_mag:
            paired = c["subfield"] in ("SP", "RB")
            expected = PARTITION_SP_ECHO if paired else PARTITION_STATIC_NOISY
            assert c["partition"] == expected, c

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

    def test_status_channels_are_static_noisy(self, manifest):
        """A status bit is a reading with no setpoint behind it, on every ring.

        Nothing writes it, so there is neither a binding to steer it nor a
        setpoint for it to echo, and the ring its address starts with does not
        change that: the partition follows what the tree says drives a channel,
        not the token the name begins with.
        """
        status = [c for c in manifest["channels"] if c["field"] == "STATUS"]
        assert status, "expected STATUS channels to exist"
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
    """machine_state_channels.json's addresses are reconciled against this
    manifest (tests/templates/test_machine_state_channels.py asserts every one
    is real); the manifest reports what it checked so drift stays visible.
    """

    def test_reconciliation_report_present(self, manifest):
        report = manifest["_metadata"]["machine_state_reconciliation"]
        assert report["candidates_checked"] > 0
        assert report["candidates_checked"] == len(report["valid"]) + len(report["invalid"])


def _tree(root, *, bindings: bool, lattice: bool):
    """A data tree carrying the outright-required files, plus what is asked for."""
    (root / "simulation").mkdir()
    (root / "simulation" / "machine.json").write_text("{}")
    (root / "machine_state_channels.json").write_text("{}")
    if lattice:
        (root / "simulation" / "lattice.json").write_text("{}")
    if bindings:
        (root / "simulation" / "va_bindings.json").write_text("{}")
    return ManifestPaths(data_root=root)


class TestManifestPathsLayout:
    """One layout for a data tree, read the same at build time and serve time.

    What the facility carries whatever it simulates sits at the data root;
    what describes the simulated accelerator sits under ``simulation/``.
    """

    def test_paths_resolve_the_simulation_model_under_simulation(self, tmp_path):
        paths = ManifestPaths(data_root=tmp_path)

        assert paths.machine_json == tmp_path / "simulation" / "machine.json"
        assert paths.lattice_json == tmp_path / "simulation" / "lattice.json"
        assert paths.va_bindings == tmp_path / "simulation" / "va_bindings.json"

    def test_paths_resolve_limits_and_machine_state_at_the_data_root(self, tmp_path):
        paths = ManifestPaths(data_root=tmp_path)

        assert paths.channel_limits == tmp_path / "channel_limits.json"
        assert paths.machine_state_channels == tmp_path / "machine_state_channels.json"


class TestManifestPathsSimulationModelRequirement:
    """A tree owes a lattice and its bindings exactly when it serves one."""

    def test_paths_ask_for_no_simulation_model_without_bindings(self, tmp_path):
        paths = _tree(tmp_path, bindings=False, lattice=True)

        assert paths.lattice_json not in paths.required_sources
        assert paths.va_bindings not in paths.required_sources
        assert paths.missing_sources() == []

    def test_paths_ask_for_both_once_the_tree_carries_bindings(self, tmp_path):
        paths = _tree(tmp_path, bindings=True, lattice=True)

        assert paths.required_sources[-2:] == (paths.lattice_json, paths.va_bindings)
        assert paths.missing_sources() == []

    def test_paths_report_bindings_without_their_lattice_as_missing(self, tmp_path):
        paths = _tree(tmp_path, bindings=True, lattice=False)

        assert paths.missing_sources() == [paths.lattice_json]

    def test_paths_keep_channel_limits_out_of_the_required_sources(self, tmp_path):
        paths = _tree(tmp_path, bindings=True, lattice=True)

        assert paths.channel_limits not in paths.required_sources
