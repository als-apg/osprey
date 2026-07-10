"""Consistency test for ``channel_limits.json``.

``channel_limits.json`` is a pure projection of the namespace-union manifest
(``osprey.services.virtual_accelerator.manifest``): it carries exactly one
entry for every address the manifest defines, and the write-safety contract
is a single rule --

    a channel is writable if and only if it is a setpoint (``:SP``).

The manifest's 168 writable ``:SP`` addresses each get a writable entry with
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

    def test_manifest_reports_168_sp_addresses(self, manifest_sp_addresses):
        assert len(manifest_sp_addresses) == 168

    def test_manifest_reports_1060_non_sp_addresses(self, manifest_non_sp_addresses):
        assert len(manifest_non_sp_addresses) == 1060

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
        assert len(correctors) == 40  # 20 HCM + 20 VCM
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
    """FR9: channel_limits.json's SR quad bands must bracket currents PyAT
    actually accepts. The shipped bands (250-320A QD, 300-400A QF) sat well
    outside the ~100A stable operating point and raised OrbitSolveError from
    every quad on first use -- this pins the fix so it can't regress."""

    def test_channel_limits_quad_range_is_physically_stable(self, limits_db, manifest_sp_addresses):
        from osprey.services.virtual_accelerator.ioc.physics_bridge import (
            OrbitSolveError,
            PhysicsBridge,
        )

        quad_addresses = [
            a
            for a in manifest_sp_addresses
            if a.startswith("SR:MAG:QF:") or a.startswith("SR:MAG:QD:")
        ]
        assert len(quad_addresses) == 32  # 16 QF + 16 QD

        for address in quad_addresses:
            entry = limits_db[address]
            for value in (entry["min_value"], entry["max_value"]):
                bridge = PhysicsBridge()
                try:
                    bridge.on_setpoint(address, value)
                except OrbitSolveError as exc:
                    pytest.fail(f"{address}={value} is not physically stable: {exc}")
