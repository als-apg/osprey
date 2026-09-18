"""The bundled demo tree holds together: its manifest, its model and its
scenario seed describe one accelerator.

The demo is a facility like any other -- one data tree, resolved through
:class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`,
carrying everything the generator and the service read from it. This module
reads it as such rather than by climbing to repo paths of its own, so a test
here cannot be looking at a different set of files than the build is.

Three things have to agree across that tree:

* **The manifest and the tree it was generated from.** The committed
  ``channel_manifest.json`` equals what ``build_manifest()`` produces from
  this tree today. A manifest that has drifted from its sources serves a
  namespace nothing else in the tree knows about.

* **The manifest and the model.** The ``pyat-coupled`` partition is exactly
  what the tree's ``va_bindings.json`` binds to elements of its
  ``lattice.json`` -- the one rule for every tree, the demo included. Its
  population is therefore derivable from the facility spec the demo's
  generator works from, which is what the per-family counts here check: spec
  to generator to bindings to manifest, end to end.

* **The manifest and the scenario seed.** Every pyat-coupled address resolves
  to a real, calibrated ``machine.json`` entry, because the engine serves
  that file in mock mode and an address missing from it is a
  connection-refused fiction rather than a channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import MANIFEST_OUTPUT, PACKAGE_PATHS
from osprey.simulation.facility_spec import ALS_U_AR
from osprey.simulation.machine import parse_machine

# The demo tree, and the files in it, named the way the build and the service
# name them: one layout, written down once, rather than a second spelling of
# it that can drift from the first without either side noticing.
MANIFEST_PATH = MANIFEST_OUTPUT
MACHINE_PATH = PACKAGE_PATHS.machine_json

# machine.json's total channel count is a hand-calibrated fact of the
# scenario-seed file -- unlike the pyat-coupled partition, it has no
# facility-spec source (it also carries hand-authored RF/vacuum/status
# channels the spec doesn't declare at all), so it's pinned as a bare
# literal rather than derived.
EXPECTED_MACHINE_JSON_CHANNEL_COUNT = 1036


@pytest.fixture(scope="module")
def manifest() -> dict:
    return build_manifest(PACKAGE_PATHS)


@pytest.fixture(scope="module")
def manifest_channels(manifest) -> list[dict]:
    return manifest["channels"]


@pytest.fixture(scope="module")
def pyat_coupled_channels(manifest_channels) -> list[dict]:
    return [c for c in manifest_channels if c["partition"] == "pyat-coupled"]


@pytest.fixture(scope="module")
def bindings():
    """The tree's own bindings, which say which channels the model drives.

    The manifest's coupled entries carry the pair-key shape -- their identity
    keys are empty, because the document rather than a hierarchy path is what
    says two addresses are one device's halves -- so a check that needs to know
    which family or which device an address belongs to asks the document.
    """
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def addresses_by_family(bindings) -> dict[str, list[str]]:
    """Every address each family's bindings claim, setpoints and readbacks."""
    claimed: dict[str, list[str]] = {}
    for binding in bindings.bindings:
        for address in (binding.setpoint_address, binding.readback_address):
            if address is not None:
                claimed.setdefault(binding.family, []).append(address)
    return claimed


@pytest.fixture(scope="module")
def setpoints_by_family(bindings) -> dict[str, list[str]]:
    """Every written address each family's bindings claim."""
    claimed: dict[str, list[str]] = {}
    for binding in bindings.bindings:
        claimed.setdefault(binding.family, []).append(binding.setpoint_address)
    return claimed


@pytest.fixture(scope="module")
def machine() -> dict:
    return json.loads(MACHINE_PATH.read_text())


@pytest.fixture(scope="module")
def machine_channels(machine) -> dict:
    return machine["channels"]


class TestTheDemoTreeCarriesItsOwnModel:
    """The demo tree holds the lattice and bindings its manifest is built from.

    Which channels are backed by physics is read off the bindings, so a tree
    without them has no pyat-coupled partition at all -- not an empty one by
    coincidence, but none by construction. Asking first means the rest of this
    module reports what actually disagrees instead of every count in it
    collapsing to zero at once.
    """

    def test_the_tree_carries_the_lattice_and_the_bindings(self):
        missing = [
            path
            for path in (PACKAGE_PATHS.lattice_json, PACKAGE_PATHS.va_bindings)
            if not path.is_file()
        ]
        assert not missing, (
            "the demo tree does not carry the model its manifest is generated "
            f"against: {[str(p) for p in missing]}. Everything downstream of the "
            "bindings -- the pyat-coupled partition, its census, the physics the "
            "service serves for it -- is absent until the tree does."
        )


class TestManifestFileConsistency:
    """The committed channel_manifest.json equals what this tree generates.

    A manifest committed beside the sources it came from is a cache, and a
    cache that has drifted is worse than none: the service serves the file,
    every other check reads the sources, and the two describe different
    namespaces without either side being obviously wrong."""

    def test_committed_manifest_equals_build_manifest_output(self, manifest):
        committed = json.loads(MANIFEST_PATH.read_text())
        assert committed == manifest


class TestPyatCoupledCountMatchesSpec:
    """The pyat-coupled partition is the spec's inventory, carried through.

    The demo's bindings are generated from ``ALS_U_AR``, and the manifest's
    pyat-coupled partition is generated from those bindings, so the spec's
    device counts have to survive both steps: every magnet and corrector
    family contributes a CURRENT SP + RB pair per device, and the monitor
    family a POSITION X + Y pair per device. Counting them here is what makes
    a device lost anywhere along that chain visible at the end of it."""

    def test_total_count_derived_from_facility_spec(self, pyat_coupled_channels):
        mag_and_corrector_devices = sum(
            f.count for f in ALS_U_AR.families if f.kind in ("magnet", "corrector")
        )
        bpm_devices = ALS_U_AR.family("BPM").count
        expected = mag_and_corrector_devices * 2 + bpm_devices * 2
        assert len(pyat_coupled_channels) == expected

    def test_per_family_counts_match_spec_device_counts(
        self, addresses_by_family, pyat_coupled_channels
    ):
        coupled = {c["address"] for c in pyat_coupled_channels}
        for fam in ALS_U_AR.families:
            claimed = addresses_by_family.get(fam.name, [])
            assert len(claimed) == fam.count * 2, fam.name
            assert set(claimed) <= coupled, fam.name
        assert set(addresses_by_family) == {f.name for f in ALS_U_AR.families}


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
    """The transport lines need scenario entries as much as the ring does.

    Their channels are sp-echo rather than pyat-coupled -- writable, with no
    physics behind them -- which is exactly why they are easy to lose: nothing
    in the model refers to them, so only the seed says what they read."""

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
    """Every entry is a real anchor, not a placeholder standing in for one.

    A placeholder reads like data to every consumer of the seed, so the only
    place it can be caught is here, where it is still spelled as one."""

    def test_zero_provisional_strings_in_machine_json(self):
        text = MACHINE_PATH.read_text()
        assert "provisional" not in text.lower()


class TestSrCorrectorsAreZeroed:
    """SR HCM/VCM CURRENT SP/RB were calibrated to a zeroed baseline (value
    0.0) with a physical current limit (min -12.0 A)."""

    def test_sr_correctors_zeroed_with_current_limit(self, addresses_by_family, machine_channels):
        corrector_families = {f.name for f in ALS_U_AR.families if f.kind == "corrector"}
        correctors = [
            address
            for family in sorted(corrector_families)
            for address in addresses_by_family[family]
        ]
        expected_count = sum(ALS_U_AR.family(name).count for name in corrector_families) * 2
        assert len(correctors) == expected_count
        for address in correctors:
            entry = machine_channels[address]
            assert entry["value"] == 0.0, address
            assert entry["min"] == -12.0, address


class TestSrBpmPositionsAreZeroed:
    """SR BPM POSITION X/Y were calibrated to an ideal (zeroed) closed orbit."""

    def test_sr_bpm_positions_zeroed_with_ideal_orbit_description(
        self, addresses_by_family, machine_channels
    ):
        bpms = addresses_by_family["BPM"]
        expected_count = ALS_U_AR.family("BPM").count * 2
        assert len(bpms) == expected_count
        for address in bpms:
            entry = machine_channels[address]
            assert entry["value"] == 0.0, address
            assert "ideal" in entry["description"].lower(), address


class TestQfaShfShdCarryGenuineAnchors:
    """The families whose anchors are genuinely nonzero carry real values.

    A zeroed setpoint is indistinguishable from an uncalibrated one on any
    family that is legitimately parked at zero, so the check is worth having
    exactly on the families that are not."""

    @pytest.mark.parametrize("family_name", ["QFA", "SHF", "SHD"])
    def test_family_present_with_nonzero_current_setpoints(
        self, family_name, setpoints_by_family, machine_channels
    ):
        expected_count = ALS_U_AR.family(family_name).count
        setpoints = setpoints_by_family[family_name]
        assert len(setpoints) == expected_count
        for address in setpoints:
            entry = machine_channels[address]
            assert entry["value"] != 0.0, address


@dataclass(frozen=True)
class _Subfamily:
    """One calibrated subfamily of machine.json entries.

    Attributes:
        prefix: Address prefix every member shares.
        suffix: Address suffix every member shares.
        count: Exact number of member addresses expected.
        entry: The full expected entry. ``description`` is a template whose
            ``{index}`` placeholder is filled with the member's device index --
            that index is the *only* thing allowed to vary across members.
    """

    prefix: str
    suffix: str
    count: int
    entry: dict[str, Any]


# The 288 SR channels below used to synthesize dead-flat zeros: their noise was
# relative and their baseline is 0.0, so `value * noise` was identically 0. They
# now carry absolute noise plus a wander texture. These dicts are the calibrated
# contract, pinned verbatim so a later hand-edit of machine.json cannot quietly
# desynchronize one device from its subfamily or drift the whole subfamily back
# towards flatness.
#
# The two texture periods differ ON PURPOSE and must not be "harmonised":
#   * BPM (43200 s) needs per-channel DISTINCTNESS -- adjacent BPMs must sit in
#     visibly separate lanes across a 3-hour window, which needs the slow
#     component to dominate within-window motion.
#   * Corrector RB (21600 s) needs visible mA-scale ripple tracking a setpoint of
#     ~0, where distinctness is meaningless and a longer period would only
#     flatten the ripple.
# One shared value would serve one job and break the other.
#
# `noise_abs` and `texture.amplitude` are absolute, in the channel's declared
# units, so the BPM magnitudes read small: those channels are labelled in meters
# (the bridge serves closed orbit in meters), making them 1 um noise on a 30 um
# wander. The corrector readbacks are in amperes and unrelated in scale.
_SUBFAMILIES: dict[str, _Subfamily] = {
    "SR BPM POSITION:X": _Subfamily(
        prefix="SR:DIAG:BPM:",
        suffix=":POSITION:X",
        count=72,
        entry={
            "value": 0.0,
            "noise_abs": 1e-06,
            "texture": {"kind": "wander", "amplitude": 3e-05, "period_s": 43200.0},
            "units": "m",
            "description": (
                "Storage-ring BPM {index} horizontal position readback (pyat-coupled -- "
                "recomputed from the AT lattice model; ideal closed orbit, 0.0 m baseline)"
            ),
        },
    ),
    "SR BPM POSITION:Y": _Subfamily(
        prefix="SR:DIAG:BPM:",
        suffix=":POSITION:Y",
        count=72,
        entry={
            "value": 0.0,
            "noise_abs": 1e-06,
            "texture": {"kind": "wander", "amplitude": 3e-05, "period_s": 43200.0},
            "units": "m",
            "description": (
                "Storage-ring BPM {index} vertical position readback (pyat-coupled -- "
                "recomputed from the AT lattice model; ideal closed orbit, 0.0 m baseline)"
            ),
        },
    ),
    "SR HCM CURRENT:RB": _Subfamily(
        prefix="SR:MAG:HCM:",
        suffix=":CURRENT:RB",
        count=72,
        entry={
            "value": 0.0,
            "noise_abs": 0.001,
            "texture": {"kind": "wander", "amplitude": 0.005, "period_s": 21600.0},
            "units": "A",
            "description": (
                "Storage-ring horizontal corrector {index} current readback (nominal ~0.0 A; "
                "pyat-coupled -- backed by the AT lattice model)"
            ),
            "min": -12.0,
            "max": 12.0,
        },
    ),
    "SR VCM CURRENT:RB": _Subfamily(
        prefix="SR:MAG:VCM:",
        suffix=":CURRENT:RB",
        count=72,
        entry={
            "value": 0.0,
            "noise_abs": 0.001,
            "texture": {"kind": "wander", "amplitude": 0.005, "period_s": 21600.0},
            "units": "A",
            "description": (
                "Storage-ring vertical corrector {index} current readback (nominal ~0.0 A; "
                "pyat-coupled -- backed by the AT lattice model)"
            ),
            "min": -12.0,
            "max": 12.0,
        },
    ),
    "SR HCM CURRENT:SP": _Subfamily(
        prefix="SR:MAG:HCM:",
        suffix=":CURRENT:SP",
        count=72,
        entry={
            "value": 0.0,
            "noise": 0,
            "units": "A",
            "description": "Storage-ring horizontal corrector {index} current setpoint",
            "min": -12.0,
            "max": 12.0,
        },
    ),
    "SR VCM CURRENT:SP": _Subfamily(
        prefix="SR:MAG:VCM:",
        suffix=":CURRENT:SP",
        count=72,
        entry={
            "value": 0.0,
            "noise": 0,
            "units": "A",
            "description": "Storage-ring vertical corrector {index} current setpoint",
            "min": -12.0,
            "max": 12.0,
        },
    ),
}

# BTS corrector setpoints are deliberately NOT part of the uniform-entry sweep
# above: they were excluded from the zero-baseline calibration because each one
# carries a genuine per-device baseline (0.984 .. 1.235 A). Only their bounds,
# noise and units are subfamily-uniform, so only those are pinned.
_BTS_CORRECTOR_SP_PREFIXES = ("BTS:MAG:HCM:", "BTS:MAG:VCM:")
_BTS_CORRECTOR_SP_COUNT = 12
_BTS_CORRECTOR_SP_BOUNDS: dict[str, Any] = {"min": 0.0, "max": 5.0, "noise": 0, "units": "A"}
_BTS_CORRECTOR_SP_KEYS = {"value", "noise", "units", "description", "min", "max"}


def _members(machine_channels: dict, sub: _Subfamily) -> list[str]:
    """Return the sorted addresses belonging to ``sub``."""
    return sorted(
        addr
        for addr in machine_channels
        if addr.startswith(sub.prefix) and addr.endswith(sub.suffix)
    )


def _device_index(address: str) -> str:
    """Return the device-index field of a ``RING:GROUP:FAMILY:NN:...`` address."""
    return address.split(":")[3]


class TestCalibratedSubfamiliesAreUniform:
    """Every member of a calibrated subfamily must be byte-identical to its
    subfamily spec once the device index is substituted in.

    This is a drift guard over the 288 SR channels that were calibrated out of
    dead-flat zeros (absolute noise + wander texture) plus the corrector current
    band. Equality is over the WHOLE entry, so it also catches a stray extra key
    or a silently dropped one."""

    @pytest.mark.parametrize("subfamily_name", sorted(_SUBFAMILIES))
    def test_subfamily_member_count(self, subfamily_name, machine_channels):
        sub = _SUBFAMILIES[subfamily_name]
        assert len(_members(machine_channels, sub)) == sub.count

    @pytest.mark.parametrize("subfamily_name", sorted(_SUBFAMILIES))
    def test_every_member_matches_the_subfamily_spec(self, subfamily_name, machine_channels):
        sub = _SUBFAMILIES[subfamily_name]
        members = _members(machine_channels, sub)
        assert members, subfamily_name
        for address in members:
            expected = dict(sub.entry)
            expected["description"] = expected["description"].format(index=_device_index(address))
            assert machine_channels[address] == expected, address

    def test_bpm_and_corrector_rb_periods_stay_distinct(self):
        """The two wander periods are a deliberate asymmetry, not an oversight."""
        bpm_period = _SUBFAMILIES["SR BPM POSITION:X"].entry["texture"]["period_s"]
        rb_period = _SUBFAMILIES["SR HCM CURRENT:RB"].entry["texture"]["period_s"]
        assert bpm_period == 43200.0
        assert rb_period == 21600.0
        assert bpm_period != rb_period


class TestBtsCorrectorSetpointBoundsAreUniform:
    """BTS corrector setpoints share bounds, noise and units -- but NOT ``value``.

    Each of the 12 carries its own non-zero operational baseline, which is why
    they were excluded from the zero-baseline calibration; pinning ``value``
    here would be wrong."""

    def test_bts_corrector_setpoints_share_bounds_but_not_values(self, machine_channels):
        addresses = sorted(
            addr
            for addr in machine_channels
            if addr.startswith(_BTS_CORRECTOR_SP_PREFIXES) and addr.endswith(":CURRENT:SP")
        )
        assert len(addresses) == _BTS_CORRECTOR_SP_COUNT
        for address in addresses:
            entry = machine_channels[address]
            assert set(entry) == _BTS_CORRECTOR_SP_KEYS, address
            for field, value in _BTS_CORRECTOR_SP_BOUNDS.items():
                assert entry[field] == value, f"{address}.{field}"
            assert entry["value"] != 0.0, address


class TestMachineJsonParsesAsValidMachineDescription:
    """The file must still validate against the simulation engine's real
    schema loader (osprey.simulation.machine.parse_machine), not just be
    syntactically valid JSON."""

    def test_parses_and_channel_count_matches(self, machine):
        parsed = parse_machine(machine, MACHINE_PATH)
        assert len(parsed.channels) == EXPECTED_MACHINE_JSON_CHANNEL_COUNT
