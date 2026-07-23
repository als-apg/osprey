"""Boot-value / channel-limits band invariant test (task 4.7, SC7).

The two data files this repo derives independently -- machine.json's
per-address boot values (task "calibrate-machine-json") and
channel_limits.json's per-address ``[min_value, max_value]`` bands (task
"recalibrate-channel-limits") -- share a single uniform rule that neither
file's own generator checks against the other: every writable ``:SP``
address boots inside its own band. A derivation that excludes its own
device's nominal (e.g. a floored QD band that clips the QD nominal current),
a zeroed corrector outside its declared +-12 A band, or an sp-echo nominal
sitting outside its own band would all violate this without either source
file being internally inconsistent -- this test is the join that catches it.

This test drives the join through the same code the real VA IOC entrypoint
uses (``osprey.services.virtual_accelerator.entrypoint._load_drive_limits()``
and ``...manifest.loaders.load_machine_json_channels()`` -- see
``entrypoint.main()``'s ``build_records(drive_limits=..., boot_values=...)``
call), rather than re-parsing channel_limits.json/machine.json independently,
so a change to either loader's merge semantics (e.g. the ``defaults`` block
merge, or the ``writable``/``:SP``-suffix filter in ``_load_drive_limits``)
is exercised here too.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator.entrypoint import _load_drive_limits
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels


@pytest.fixture(scope="module")
def drive_limits() -> dict[str, tuple[float, float]]:
    """{address: (min_value, max_value)} for every writable :SP address with
    a numeric band -- the entrypoint's own derivation from
    channel_limits.json (entrypoint.py's docstring/main() call this the
    build_records(drive_limits=...) input)."""
    return _load_drive_limits()


@pytest.fixture(scope="module")
def machine_channels() -> dict[str, dict]:
    """{address: entry} for machine.json's scenario-seed channels -- the
    entrypoint's own load_machine_json_channels() call (its _load_boot_values()
    helper reads entry["value"] from exactly this)."""
    return load_machine_json_channels()


@pytest.fixture(scope="module")
def manifest_sp_channels() -> list[dict]:
    manifest = build_manifest()
    return [c for c in manifest["channels"] if c["subfield"] == "SP"]


@pytest.fixture(scope="module")
def sp_addresses(manifest_sp_channels) -> set[str]:
    return {c["address"] for c in manifest_sp_channels}


@pytest.fixture(scope="module")
def sp_channel_by_address(manifest_sp_channels) -> dict[str, dict]:
    return {c["address"]: c for c in manifest_sp_channels}


@pytest.fixture(scope="module")
def joined_addresses(drive_limits, machine_channels, sp_addresses) -> list[str]:
    """The full join: every writable :SP address with BOTH a channel_limits
    band and a machine.json entry carrying a static boot value. Iterated in
    full below (not sampled) -- this is the whole point of the invariant.

    Restricted to entries that carry a "value" key: _load_boot_values()
    (entrypoint.py) skips machine.json entries computed via "expr" rather
    than a stored value for the same reason -- none of them are :SP/:RB
    addresses this map is ever consulted for, so a static "value" absence
    here would mean the manifest/machine.json pairing itself is broken, not
    that the boot value is merely dynamic.
    """
    return sorted(
        addr
        for addr in (set(drive_limits) & set(machine_channels) & sp_addresses)
        if "value" in machine_channels[addr]
    )


class TestJoinIsNonEmptyAndCoversExpectedClasses:
    """The join must not be vacuous, and must cover the two address classes
    the task brief names explicitly."""

    # Derived from osprey.simulation.facility_spec.ALS_U_AR, the same spec
    # tests/templates/test_machine_json_lattice.py's
    # TestPyatCoupledCountMatchesSpec derives its expected counts from: SR
    # quad/dipole magnet families QF(24) + QD(24) + QFA(24) + DIPOLE(36) = 108
    # CURRENT:SP addresses; SR corrector families HCM(72) + VCM(72) = 144
    # CURRENT:SP addresses. Both groups are pyat-coupled (real device with a
    # real calibrated boot value and a real derived band), so every one of
    # them must appear whole in the join.
    EXPECTED_SR_QUAD_DIPOLE_SP_COUNT = 108
    EXPECTED_SR_CORRECTOR_SP_COUNT = 144

    # 396 manifest :SP addresses all get a channel_limits band (every
    # writable :SP entry has min_value/max_value -- see
    # tests/templates/test_channel_limits_va.py's TestEntryShape), but 6 of
    # them (SR:VAC:ION-PUMP:0{1..6}:VOLTAGE:SP, an sp-echo family) have no
    # machine.json entry at all, so they drop out of the join: 396 - 6 = 390.
    EXPECTED_JOIN_SIZE = 390

    def test_join_is_non_empty(self, joined_addresses):
        assert joined_addresses, (
            "expected at least one :SP address with both a machine.json boot "
            "value and a channel_limits band -- got an empty join"
        )

    def test_join_size_matches_the_derived_expectation(self, joined_addresses):
        assert len(joined_addresses) == self.EXPECTED_JOIN_SIZE, (
            f"join size drifted: expected {self.EXPECTED_JOIN_SIZE}, got {len(joined_addresses)}"
        )

    def test_join_covers_every_sr_quad_and_dipole_setpoint(
        self, joined_addresses, sp_channel_by_address
    ):
        quad_dipole = [
            a
            for a in joined_addresses
            if sp_channel_by_address[a]["ring"] == "SR"
            and sp_channel_by_address[a]["family"] in ("QF", "QD", "QFA", "DIPOLE")
        ]
        assert len(quad_dipole) == self.EXPECTED_SR_QUAD_DIPOLE_SP_COUNT, (
            f"expected {self.EXPECTED_SR_QUAD_DIPOLE_SP_COUNT} SR quad/dipole "
            f"CURRENT:SP addresses in the join, got {len(quad_dipole)}"
        )

    def test_join_covers_every_sr_corrector_setpoint(self, joined_addresses, sp_channel_by_address):
        correctors = [
            a
            for a in joined_addresses
            if sp_channel_by_address[a]["ring"] == "SR"
            and sp_channel_by_address[a]["family"] in ("HCM", "VCM")
        ]
        assert len(correctors) == self.EXPECTED_SR_CORRECTOR_SP_COUNT, (
            f"expected {self.EXPECTED_SR_CORRECTOR_SP_COUNT} SR corrector "
            f"CURRENT:SP addresses in the join, got {len(correctors)}"
        )


class TestBootValueFallsInsideItsOwnBand:
    """SC7: for every joined address, min_value <= boot value <= max_value.

    Iterates the full join, not a sample -- a single out-of-band address
    (a derived band excluding its own nominal, a zeroed corrector outside a
    +-12 A band, an sp-echo nominal outside its band) must fail this test.
    """

    def test_every_boot_value_is_within_its_band(
        self, joined_addresses, drive_limits, machine_channels
    ):
        violations = []
        for address in joined_addresses:
            boot_value = machine_channels[address]["value"]
            min_value, max_value = drive_limits[address]
            if not (min_value <= boot_value <= max_value):
                violations.append(
                    f"{address}: boot value {boot_value} outside band [{min_value}, {max_value}]"
                )
        assert not violations, "boot-band invariant violated:\n" + "\n".join(violations)
