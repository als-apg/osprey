"""Consistency test for ``channel_limits.json`` (task 5.2).

Before this task the file had five illustrative example entries -- two using
obsolete bracket-convention addresses (``MAG:HCM[H01]:CURRENT:SP``,
``MAG:QF[QF01]:CURRENT:SP``) and three using fictional FEL-tutorial names
(``ElectronGunFilamentCurrentSetPoint``, ``TerminalVoltageSetValue``,
``DIAGNOSTICS:TEMPERATURE:SP``) -- none of which match any real address in the
namespace-union manifest (docker/virtual-accelerator/manifest). It now carries
one real entry for every one of the manifest's 168 writable ``:SP`` addresses,
with min/max bounds family-banded from the SR lattice model's device inventory
(docker/virtual-accelerator/lattice) and machine.json's simulated nominal
values, plus readback verification.

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
import sys
from pathlib import Path

import pytest

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

# docker/virtual-accelerator/manifest is not an importable dotted package
# (its parent directory name contains a hyphen); see tests/va/test_manifest.py
# for the same import shim.
_MANIFEST_PARENT = REPO_ROOT / "docker" / "virtual-accelerator"
if str(_MANIFEST_PARENT) not in sys.path:
    sys.path.insert(0, str(_MANIFEST_PARENT))

from manifest import build_manifest  # noqa: E402

# The FR3/3.8 demo write: orbit_response's own convention (see
# docker/virtual-accelerator/lattice/response.py and tests/va/test_lattice.py's
# orbit_response("HCM01", 10.0)) is a corrector CURRENT:SP of 10.0 A.
DEMO_WRITE_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"
DEMO_WRITE_VALUE = 10.0

# Stale entries from before this task -- must never reappear.
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
def manifest_sp_addresses() -> set[str]:
    manifest = build_manifest()
    return {c["address"] for c in manifest["channels"] if c["subfield"] == "SP"}


class TestDefaultsSemanticsPreserved:
    def test_defaults_block_present(self, limits_db):
        assert "defaults" in limits_db

    def test_defaults_still_writable_with_callback_verification(self, limits_db):
        defaults = limits_db["defaults"]
        assert defaults["writable"] is True
        assert defaults["verification"]["level"] == "callback"


class TestEveryManifestSpHasAnEntry:
    def test_manifest_reports_168_sp_addresses(self, manifest_sp_addresses):
        assert len(manifest_sp_addresses) == 168

    def test_every_manifest_sp_address_has_a_limits_entry(self, limits_db, manifest_sp_addresses):
        missing = manifest_sp_addresses - set(limits_db.keys())
        assert not missing, f"manifest :SP addresses missing from channel_limits.json: {missing}"

    def test_every_limits_entry_is_a_real_manifest_sp_address(
        self, limits_db, manifest_sp_addresses
    ):
        """No orphan entries: every non-reserved key must be a real :SP address
        (the CC-4-style regression guard for this file)."""
        entry_keys = set(limits_db.keys()) - RESERVED_KEYS
        extra = entry_keys - manifest_sp_addresses
        assert not extra, f"channel_limits.json has entries not in the manifest: {extra}"


class TestStaleEntriesRemoved:
    def test_no_stale_addresses_remain(self, limits_db):
        for stale in STALE_ADDRESSES:
            assert stale not in limits_db, f"stale entry {stale!r} reappeared"


class TestEntryShape:
    def test_every_entry_has_min_max_and_verification(self, limits_db, manifest_sp_addresses):
        for address in manifest_sp_addresses:
            entry = limits_db[address]
            assert "min_value" in entry
            assert "max_value" in entry
            assert entry["min_value"] < entry["max_value"]
            assert "verification" in entry
            assert entry["verification"]["level"] in ("none", "callback", "readback")

    def test_no_entry_sets_max_step(self, limits_db, manifest_sp_addresses):
        """See module docstring: max_step is deliberately omitted (mock-mode
        safety -- LimitsValidator's max_step check bypasses connector
        abstraction and would block every mock-mode write to these channels)."""
        for address in manifest_sp_addresses:
            assert "max_step" not in limits_db[address]


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
            a for a in manifest_sp_addresses if a.startswith("SR:MAG:HCM:") or a.startswith("SR:MAG:VCM:")
        ]
        assert len(correctors) == 40  # 20 HCM + 20 VCM
        for address in correctors:
            entry = limits_db[address]
            assert entry["min_value"] <= -DEMO_WRITE_VALUE, address
            assert entry["max_value"] >= DEMO_WRITE_VALUE, address
