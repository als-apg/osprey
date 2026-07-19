"""Consistency tests binding machine.json's lattice-channel population to the
namespace-union manifest (task 4.6), after the one-facility calibration grew
both to the full ALS-U Accumulator Ring inventory declared by
``osprey.simulation.facility_spec.ALS_U_AR``.

The manifest (built by
``osprey.services.virtual_accelerator.manifest.build_manifest()``, and
committed to disk as ``channel_manifest.json``) classifies every namespace
address into partition (a) ``pyat-coupled`` (SR magnet CURRENT SP/RB + SR BPM
POSITION X/Y -- backed by the AT lattice model), (b) ``sp-echo`` (writable but
physics-free), and (c) ``static-noisy`` (everything else). ``machine.json`` is
the scenario-seed data the simulation engine serves in mock mode; every
pyat-coupled address must resolve to a real, calibrated entry there, or a
mock read/write of that channel is a connection-refused fiction.

This module also pins the manifest-file consistency invariant that ends the
stale-manifest split-brain: the committed ``channel_manifest.json`` must
always equal ``build_manifest()``'s live output, so a future lattice/spec
change can't silently drift the two apart.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.simulation.facility_spec import ALS_U_AR
from osprey.simulation.machine import parse_machine

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "src/osprey/services/virtual_accelerator/manifest/channel_manifest.json"
MACHINE_PATH = (
    REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation/machine.json"
)

# machine.json's total channel count is a hand-calibrated fact of the
# scenario-seed file -- unlike the pyat-coupled partition, it has no
# facility-spec source (it also carries hand-authored RF/vacuum/status
# channels the spec doesn't declare at all), so it's pinned as a bare
# literal rather than derived.
EXPECTED_MACHINE_JSON_CHANNEL_COUNT = 1036


@pytest.fixture(scope="module")
def manifest() -> dict:
    return build_manifest()


@pytest.fixture(scope="module")
def manifest_channels(manifest) -> list[dict]:
    return manifest["channels"]


@pytest.fixture(scope="module")
def pyat_coupled_channels(manifest_channels) -> list[dict]:
    return [c for c in manifest_channels if c["partition"] == "pyat-coupled"]


@pytest.fixture(scope="module")
def machine() -> dict:
    return json.loads(MACHINE_PATH.read_text())


@pytest.fixture(scope="module")
def machine_channels(machine) -> dict:
    return machine["channels"]


class TestManifestFileConsistency:
    """The committed channel_manifest.json must never drift from
    build_manifest()'s live output -- this is the test that ends the
    stale-manifest split-brain (the old test file pinned counts, 1228/280,
    that had already gone stale against the grown lattice)."""

    def test_committed_manifest_equals_build_manifest_output(self, manifest):
        committed = json.loads(MANIFEST_PATH.read_text())
        assert committed == manifest


class TestPyatCoupledCountMatchesSpec:
    """The pyat-coupled partition size is fully derived from ALS_U_AR: every
    magnet/corrector family contributes a CURRENT SP + RB pair per device,
    and the BPM family contributes a POSITION X + Y pair per device."""

    def test_total_count_derived_from_facility_spec(self, pyat_coupled_channels):
        mag_and_corrector_devices = sum(
            f.count for f in ALS_U_AR.families if f.kind in ("magnet", "corrector")
        )
        bpm_devices = ALS_U_AR.family("BPM").count
        expected = mag_and_corrector_devices * 2 + bpm_devices * 2
        assert len(pyat_coupled_channels) == expected

    def test_per_family_counts_match_spec_device_counts(self, pyat_coupled_channels):
        by_family = Counter(c["family"] for c in pyat_coupled_channels)
        for fam in ALS_U_AR.families:
            if fam.kind in ("magnet", "corrector"):
                assert by_family[fam.name] == fam.count * 2, fam.name
        assert by_family["BPM"] == ALS_U_AR.family("BPM").count * 2


class TestEveryPyatCoupledAddressHasAMachineJsonEntry:
    """Primary consistency gate: every pyat-coupled address the manifest
    declares must have a machine.json entry for mock reads/writes to serve."""

    def test_no_pyat_coupled_address_missing_from_machine_json(
        self, pyat_coupled_channels, machine_channels
    ):
        missing = [
            c["address"] for c in pyat_coupled_channels if c["address"] not in machine_channels
        ]
        assert not missing, (
            f"{len(missing)} pyat-coupled channels missing from machine.json: {missing[:10]}"
        )


class TestBrBtsSpEchoAddressesStillCovered:
    """The BR/BTS transport-line sp-echo channels (added by an earlier task,
    untouched by this calibration) must remain present -- this file is the
    manifest<->machine.json binding, so it's the right place to keep that
    guarantee even though this task didn't touch that data."""

    def test_no_br_bts_sp_echo_address_missing_from_machine_json(
        self, manifest_channels, machine_channels
    ):
        br_bts_echo = [
            c["address"]
            for c in manifest_channels
            if c["partition"] == "sp-echo" and c["ring"] in ("BR", "BTS")
        ]
        assert br_bts_echo, "expected BR/BTS sp-echo channels to exist"
        missing = [addr for addr in br_bts_echo if addr not in machine_channels]
        assert not missing, (
            f"{len(missing)} BR/BTS sp-echo channels missing from machine.json: {missing[:10]}"
        )


class TestMachineJsonChannelCount:
    def test_machine_json_channel_count(self, machine_channels):
        assert len(machine_channels) == EXPECTED_MACHINE_JSON_CHANNEL_COUNT


class TestNoProvisionalMarkersRemain:
    """The calibration's whole point was to replace provisional placeholder
    values with genuine per-device anchors; this pins that it stuck."""

    def test_zero_provisional_strings_in_machine_json(self):
        text = MACHINE_PATH.read_text()
        assert "provisional" not in text.lower()


class TestSrCorrectorsAreZeroed:
    """SR HCM/VCM CURRENT SP/RB were calibrated to a zeroed baseline (value
    0.0) with a physical current limit (min -12.0 A)."""

    def test_sr_correctors_zeroed_with_current_limit(
        self, pyat_coupled_channels, machine_channels
    ):
        corrector_families = {f.name for f in ALS_U_AR.families if f.kind == "corrector"}
        correctors = [
            c
            for c in pyat_coupled_channels
            if c["ring"] == "SR" and c["family"] in corrector_families and c["field"] == "CURRENT"
        ]
        expected_count = sum(ALS_U_AR.family(name).count for name in corrector_families) * 2
        assert len(correctors) == expected_count
        for c in correctors:
            entry = machine_channels[c["address"]]
            assert entry["value"] == 0.0, c["address"]
            assert entry["min"] == -12.0, c["address"]


class TestSrBpmPositionsAreZeroed:
    """SR BPM POSITION X/Y were calibrated to an ideal (zeroed) closed orbit."""

    def test_sr_bpm_positions_zeroed_with_ideal_orbit_description(
        self, pyat_coupled_channels, machine_channels
    ):
        bpms = [c for c in pyat_coupled_channels if c["ring"] == "SR" and c["family"] == "BPM"]
        expected_count = ALS_U_AR.family("BPM").count * 2
        assert len(bpms) == expected_count
        for c in bpms:
            entry = machine_channels[c["address"]]
            assert entry["value"] == 0.0, c["address"]
            assert "ideal" in entry["description"].lower(), c["address"]


class TestQfaShfShdCarryGenuineAnchors:
    """QFA/SHF/SHD (families added by the one-facility spec growth) must be
    present with genuine, nonzero per-device anchor values -- not a zeroed or
    provisional placeholder."""

    @pytest.mark.parametrize("family_name", ["QFA", "SHF", "SHD"])
    def test_family_present_with_nonzero_current_setpoints(
        self, family_name, pyat_coupled_channels, machine_channels
    ):
        expected_count = ALS_U_AR.family(family_name).count
        setpoints = [
            c
            for c in pyat_coupled_channels
            if c["ring"] == "SR"
            and c["family"] == family_name
            and c["field"] == "CURRENT"
            and c["subfield"] == "SP"
        ]
        assert len(setpoints) == expected_count
        for c in setpoints:
            entry = machine_channels[c["address"]]
            assert entry["value"] != 0.0, c["address"]


class TestMachineJsonParsesAsValidMachineDescription:
    """The file must still validate against the simulation engine's real
    schema loader (osprey.simulation.machine.parse_machine), not just be
    syntactically valid JSON."""

    def test_parses_and_channel_count_matches(self, machine):
        parsed = parse_machine(machine, MACHINE_PATH)
        assert len(parsed.channels) == EXPECTED_MACHINE_JSON_CHANNEL_COUNT
