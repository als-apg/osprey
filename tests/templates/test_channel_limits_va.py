"""Consistency test for ``channel_limits.json``.

``channel_limits.json`` is a pure projection of the namespace-union manifest
(``osprey.services.virtual_accelerator.manifest``): it carries exactly one
entry for every address the manifest defines, and the write-safety contract
is a single rule --

    a channel is writable if and only if it is a setpoint (``:SP``).

The manifest's 396 writable ``:SP`` addresses each get a writable entry with
min/max bounds (family-banded from the SR lattice model's device inventory,
``osprey.services.virtual_accelerator.lattice``, and machine.json's simulated
nominal values) plus readback verification. Every other manifest address -- readbacks
(``:RB``, BPM ``:X``/``:Y``), status/fault flags, golden references and slow
telemetry -- gets a read-only entry (``writable: false``) so OSPREY's own
software safety layer refuses a write to it, rather than leaving the block to
the downstream IOC. ``TestValidatorEnforcesTheContract`` proves the shipped
file, loaded by the real LimitsValidator, does exactly that.

Historically the file carried five illustrative example entries -- two using an
obsolete bracket convention (``MAG:HCM[H01]:CURRENT:SP``,
``MAG:QF[QF01]:CURRENT:SP``) and three fictional FEL-tutorial names
(``ElectronGunFilamentCurrentSetPoint``, ``TerminalVoltageSetValue``,
``DIAGNOSTICS:TEMPERATURE:SP``) -- none of which is a real manifest address.
They are pinned in ``STALE_ADDRESSES`` below so they can never reappear. The
generic write-safety e2e that once relied on them now use a dedicated,
decoupled fixture (tests/e2e/claude_code/fixtures/safety_limits.json).

``max_step`` is deliberately NOT set on any entry: LimitsValidator's max_step
check (src/osprey/connectors/control_system/limits_validator.py) reads the
current value via a direct, connector-unaware ``epics.caget`` -- configuring
it on these channels would block every write to them under the mock control
system (the preset's default type), which has no live EPICS server behind its
addresses. min/max bounds plus readback verification already satisfy "a write
outside its limits is rejected" (SC9) without that failure mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.simulation.facility_spec import ALS_U_AR

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMITS_PATH = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_limits.json"
)

# The FR3/3.8 demo write: orbit_response's own convention (see
# src/osprey/services/virtual_accelerator/lattice/response.py and
# tests/va/test_lattice.py's orbit_response("HCM01", 10.0)) is a corrector
# CURRENT:SP of 10.0 A.
DEMO_WRITE_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"
DEMO_WRITE_VALUE = 10.0

# A representative readback that must be write-blocked, exercised by the
# validator-behavior test below.
READBACK_CHANNEL = "SR:MAG:QF:01:CURRENT:RB"

# Stale entries from before the "pure projection of the manifest" rewrite --
# must never reappear (obsolete bracket convention + fictional FEL names).
STALE_ADDRESSES = (
    "MAG:HCM[H01]:CURRENT:SP",
    "MAG:QF[QF01]:CURRENT:SP",
    "ElectronGunFilamentCurrentSetPoint",
    "TerminalVoltageSetValue",
    "DIAGNOSTICS:TEMPERATURE:SP",
)

METADATA_KEYS = {"_comment", "_version", "_description"}
RESERVED_KEYS = METADATA_KEYS | {"defaults"}

# FR9/SC5: the four families whose channel_limits.json bands are physics-
# derived (scripts/va/derive_bands.py) against the real ALS-U AR ring, rather
# than a static family-wide constant -- QF/QD/QFA at the ring's one-turn-
# trace stability edge (with a unipolar-supply floor at 0 A), DIPOLE at a
# fixed +/-0.5% policy window with 2x headroom over its own derived edge.
QUAD_DIPOLE_FAMILIES = ("QF", "QD", "QFA", "DIPOLE")

# scripts/va/derive_bands.py's TRACE_EDGE: the max-plane one-turn |trace|
# QF/QD/QFA sweeps stop at when deriving a band edge -- a safety margin below
# the hard |trace| >= 2.0 instability guard in lattice.solve.solve_orbit.
# DIPOLE's committed band is a policy window strictly inside its own derived
# edge (2x headroom invariant), so it never actually reaches this value in
# practice; QF/QD/QFA sit right at it by construction.
TRACE_MARGIN = 1.8

# derive_bands.py locates each band edge by linearly interpolating *current*
# between the last sub-threshold and first over-threshold sweep sample, then
# commits that interpolated current verbatim. Because trace-vs-current isn't
# itself linear, re-evaluating the trace at the committed edge overshoots 1.8
# by a small residual (measured here: up to ~2.2e-6, at SR:MAG:QFA:12, across
# all 216 QF/QD/QFA/DIPOLE edges). This tolerance is ~40x that observed
# residual while staying ~2000x tighter than the 0.2 margin to the hard 2.0
# instability guard -- it absorbs the interpolation residual without masking
# a real regression.
TRACE_MARGIN_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def limits_db() -> dict:
    return json.loads(LIMITS_PATH.read_text())


@pytest.fixture(scope="module")
def manifest_addresses() -> set[str]:
    manifest = build_manifest()
    return {c["address"] for c in manifest["channels"]}


@pytest.fixture(scope="module")
def manifest_sp_addresses() -> set[str]:
    manifest = build_manifest()
    return {c["address"] for c in manifest["channels"] if c["subfield"] == "SP"}


@pytest.fixture(scope="module")
def manifest_non_sp_addresses() -> set[str]:
    manifest = build_manifest()
    return {c["address"] for c in manifest["channels"] if c["subfield"] != "SP"}


@pytest.fixture(scope="module")
def family_sp_addresses(manifest_sp_addresses) -> dict[str, list[str]]:
    """SR:MAG:{family}: setpoint addresses for each recalibrated family.

    DIPOLE is SR-ring only: BR:MAG:DIPOLE (a different ring/spec) also has
    ``:SP`` entries in channel_limits.json, but neither ALS_U_AR's family
    count nor scripts/va/derive_bands.py's recalibration cover it.
    """
    return {
        family: sorted(a for a in manifest_sp_addresses if a.startswith(f"SR:MAG:{family}:"))
        for family in QUAD_DIPOLE_FAMILIES
    }


class TestDefaultsSemanticsPreserved:
    def test_defaults_block_present(self, limits_db):
        assert "defaults" in limits_db

    def test_defaults_still_writable_with_callback_verification(self, limits_db):
        defaults = limits_db["defaults"]
        assert defaults["writable"] is True
        assert defaults["verification"]["level"] == "callback"


class TestManifestCoverageIsComplete:
    """channel_limits.json is a pure projection of the manifest: every manifest
    address has exactly one entry, and no entry is an orphan."""

    def test_manifest_reports_396_sp_addresses(self, manifest_sp_addresses):
        assert len(manifest_sp_addresses) == 396

    def test_manifest_reports_2512_non_sp_addresses(self, manifest_non_sp_addresses):
        assert len(manifest_non_sp_addresses) == 2512

    def test_every_manifest_address_has_a_limits_entry(self, limits_db, manifest_addresses):
        missing = manifest_addresses - set(limits_db.keys())
        assert not missing, (
            f"manifest addresses missing from channel_limits.json "
            f"({len(missing)}): {sorted(missing)[:10]}"
        )

    def test_every_limits_entry_is_a_real_manifest_address(self, limits_db, manifest_addresses):
        """No orphan entries: every non-reserved key must be a real manifest
        address (the CC-4-style regression guard for this file)."""
        entry_keys = set(limits_db.keys()) - RESERVED_KEYS
        extra = entry_keys - manifest_addresses
        assert not extra, f"channel_limits.json has entries not in the manifest: {sorted(extra)}"


class TestWritableIffSetpoint:
    """The write-safety contract: a channel is writable iff it is a setpoint."""

    def test_every_setpoint_entry_is_writable(self, limits_db, manifest_sp_addresses):
        # SP entries inherit defaults.writable=true; none may declare writable:false.
        readonly = [a for a in manifest_sp_addresses if limits_db[a].get("writable", True) is False]
        assert not readonly, f"setpoint addresses wrongly marked read-only: {sorted(readonly)[:10]}"

    def test_every_non_setpoint_entry_is_read_only(self, limits_db, manifest_non_sp_addresses):
        writable = [
            a
            for a in manifest_non_sp_addresses
            if limits_db.get(a, {}).get("writable", True) is not False
        ]
        assert not writable, (
            f"non-setpoint addresses not marked writable:false "
            f"({len(writable)}): {sorted(writable)[:10]}"
        )


class TestStaleEntriesRemoved:
    def test_no_stale_addresses_remain(self, limits_db):
        for stale in STALE_ADDRESSES:
            assert stale not in limits_db, f"stale entry {stale!r} reappeared"


class TestEntryShape:
    def test_every_setpoint_entry_has_min_max_and_verification(
        self, limits_db, manifest_sp_addresses
    ):
        for address in manifest_sp_addresses:
            entry = limits_db[address]
            assert "min_value" in entry
            assert "max_value" in entry
            assert entry["min_value"] < entry["max_value"]
            assert "verification" in entry
            assert entry["verification"]["level"] in ("none", "callback", "readback")

    def test_no_entry_sets_max_step(self, limits_db, manifest_addresses):
        """See module docstring: max_step is deliberately omitted (mock-mode
        safety -- LimitsValidator's max_step check bypasses connector
        abstraction and would block every mock-mode write to these channels)."""
        for address in manifest_addresses:
            assert "max_step" not in limits_db.get(address, {})


class TestDemoWriteFitsInsideItsOwnLimits:
    """SC9 / FR3-FR6: the FR3/3.8 demo write must fit inside its own limits."""

    def test_demo_channel_is_in_the_database(self, limits_db):
        assert DEMO_WRITE_CHANNEL in limits_db

    def test_demo_write_value_is_within_limits(self, limits_db):
        entry = limits_db[DEMO_WRITE_CHANNEL]
        assert entry["min_value"] < DEMO_WRITE_VALUE < entry["max_value"]

    def test_every_sr_corrector_accepts_the_demo_write_magnitude(
        self, limits_db, manifest_sp_addresses
    ):
        """Every SR HCM/VCM corrector (not just BPM01's) must hold +-10A, since
        orbit-response-e2e (task 3.8) may exercise any of them."""
        correctors = [
            a
            for a in manifest_sp_addresses
            if a.startswith("SR:MAG:HCM:") or a.startswith("SR:MAG:VCM:")
        ]
        assert len(correctors) == 144  # 72 HCM + 72 VCM
        for address in correctors:
            entry = limits_db[address]
            assert entry["min_value"] <= -DEMO_WRITE_VALUE, address
            assert entry["max_value"] >= DEMO_WRITE_VALUE, address


class TestValidatorEnforcesTheContract:
    """The checks above validate JSON shape. This proves the shipped file,
    loaded by the real LimitsValidator under the preset's permissive policy
    (allow_unlisted_channels=true), actually blocks a readback write while
    allowing an in-bounds setpoint write -- i.e. the software safety layer,
    not just the IOC, refuses writes to non-setpoints."""

    @staticmethod
    def _validator():
        from osprey.connectors.control_system.limits_validator import LimitsValidator

        limits_db, raw_db = LimitsValidator._load_limits_database(str(LIMITS_PATH))
        return LimitsValidator(limits_db, {"allow_unlisted_channels": True}, raw_db)

    def test_readback_write_is_blocked_read_only(self):
        from osprey.errors import ChannelLimitsViolationError

        with pytest.raises(ChannelLimitsViolationError) as exc:
            self._validator().validate(READBACK_CHANNEL, 1.0)
        assert exc.value.violation_type == "READ_ONLY_CHANNEL"

    def test_in_bounds_setpoint_write_is_allowed(self):
        # Must not raise: HCM:01 ships writable with a +-12A band.
        self._validator().validate(DEMO_WRITE_CHANNEL, DEMO_WRITE_VALUE)


class TestQuadLimitsArePhysicallyStable:
    """FR9/SC5: channel_limits.json's SR QF/QD/QFA/DIPOLE bands must bracket
    currents the real ALS-U AR ring (lattice.build_ring(), PhysicsBridge)
    actually accepts, with margin -- not just survive the hard |trace| >= 2.0
    instability guard (lattice.solve.solve_orbit), but stay inside the 1.8
    margin scripts/va/derive_bands.py derived them at. Both halves of the
    check below cover every committed band edge (108 addresses x min/max =
    216 edges):

    (a) a fresh PhysicsBridge() accepts a write at the edge without raising
        (the SC2 "no OrbitSolveError/UnknownDeviceError" survival check);
    (b) the edge's one-turn |trace|, computed directly against a bare ring
        (no bridge/BPM readback needed), sits inside the 1.8 committed
        margin, not just below the 2.0 hard guard (the actual SC5 margin
        assert).

    Superseded by this class: the pre-repoint version of this test targeted
    a 24-cell toy ring and asserted a flat 48-address QF+QD-only band
    (300-400A QF, 250-320A QD) around a ~100A toy operating point. The real
    ring has no such flat bands -- every device's min/max is individually
    swept (see derive_bands.py) -- so there is nothing left to assert a
    fixed numeric band against; the address-count and stability checks below
    replace it.
    """

    def test_family_address_counts_match_facility_spec(self, family_sp_addresses):
        # Recalibration covers exactly ALS_U_AR's declared per-family device
        # counts -- the facility spec is the single source of truth for how
        # many QF/QD/QFA/DIPOLE devices exist, not a number hardcoded here.
        for family in QUAD_DIPOLE_FAMILIES:
            assert len(family_sp_addresses[family]) == ALS_U_AR.family(family).count, family

    def test_recalibrated_address_count_is_108(self, family_sp_addresses):
        # Replaces the old toy-ring "48 (QF+QD)" count: the repointed bridge
        # covers four families, all derived from the facility spec --
        # 24 QF + 24 QD + 24 QFA + 36 DIPOLE = 108.
        expected = sum(ALS_U_AR.family(family).count for family in QUAD_DIPOLE_FAMILIES)
        assert expected == 108
        total = sum(len(addresses) for addresses in family_sp_addresses.values())
        assert total == expected

    def test_every_band_edge_survives_a_fresh_bridge_write(self, limits_db, family_sp_addresses):
        """SC5(a): the write-survival half of the margin check, via the real
        bridge (construction + solve + rollback path), not the cheap
        ring-level trace computation the next test uses.

        One ``PhysicsBridge()`` per edge, not a shared/reused bridge, so a
        prior edge's write can never leak state into the next edge's result
        -- each write starts from that bridge's own freshly solved nominal
        state. Measured cost: ~21s for all 216 edges on this machine
        (PhysicsBridge construction ~90ms + one on_setpoint call each),
        comfortably inside the ~2-3 minute budget the task brief flagged as
        the line for falling back to a per-family/representative sample --
        so this covers every edge, not a sample.
        """
        from osprey.services.virtual_accelerator.ioc.physics_bridge import (
            OrbitSolveError,
            PhysicsBridge,
            UnknownDeviceError,
        )

        for family in QUAD_DIPOLE_FAMILIES:
            for address in family_sp_addresses[family]:
                entry = limits_db[address]
                for value in (entry["min_value"], entry["max_value"]):
                    bridge = PhysicsBridge()
                    try:
                        bridge.on_setpoint(address, value)
                    except (OrbitSolveError, UnknownDeviceError) as exc:
                        pytest.fail(f"{address}={value} is not physically stable: {exc}")

    def test_every_band_edge_trace_is_inside_the_committed_margin(
        self, limits_db, family_sp_addresses
    ):
        """SC5(b): the actual margin assert -- |trace| <= 1.8 (+ the
        interpolation tolerance documented at TRACE_MARGIN_TOLERANCE), not
        just the bridge's 2.0 survival threshold the previous test exercises.

        Computed ring-level: one shared ``build_ring()`` + ``StrengthMap``,
        mutating and restoring a single element per edge and reading
        ``at.find_m44`` directly, rather than a fresh PhysicsBridge per edge
        -- this only needs the one-turn matrix, not a full closed-orbit solve
        plus BPM readback, so all 216 edges cost ~1s total here (versus ~21s
        for the bridge-based survival check above).
        """
        import warnings

        import at
        import numpy as np

        from osprey.services.virtual_accelerator.lattice import build_ring
        from osprey.services.virtual_accelerator.lattice.strengths import (
            StrengthMap,
            restore_element,
            snapshot_element,
        )

        ring = build_ring()
        strength_map = StrengthMap(ring)
        index_by_famname = {element.FamName: i for i, element in enumerate(ring)}

        for family in QUAD_DIPOLE_FAMILIES:
            for address in family_sp_addresses[family]:
                device_id = address.split(":")[3]  # "SR:MAG:QF:01:CURRENT:SP" -> "01"
                idx = index_by_famname[f"{family}{device_id}"]
                entry = limits_db[address]
                for value in (entry["min_value"], entry["max_value"]):
                    previous = snapshot_element(ring[idx])
                    strength_map.apply(ring, family, device_id, value)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=at.AtWarning)
                        m44 = at.find_m44(ring)[0]
                    restore_element(ring[idx], previous)

                    assert np.all(np.isfinite(m44)), (
                        f"{address}={value}: non-finite one-turn matrix"
                    )
                    trace_x = float(m44[0, 0] + m44[1, 1])
                    trace_y = float(m44[2, 2] + m44[3, 3])
                    worst_trace = max(abs(trace_x), abs(trace_y))
                    assert worst_trace <= TRACE_MARGIN + TRACE_MARGIN_TOLERANCE, (
                        f"{address}={value}: |trace|={worst_trace:.6f} exceeds the "
                        f"{TRACE_MARGIN} committed margin (+/-{TRACE_MARGIN_TOLERANCE} tol)"
                    )
