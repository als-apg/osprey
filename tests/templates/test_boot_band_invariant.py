"""Every channel a tree boots starts inside its own write band (SC7).

Two files of a served tree are derived independently and never checked against
each other by either generator: the starting value of each channel
(``machine.json``) and the band a write to it must fall inside
(``channel_limits.json``). A band that excludes its own device's nominal -- a
floored magnet band that clips its nominal current, a corrector zeroed outside
its declared window, a seed carried over from an export whose operating
``Range`` no longer covers it -- leaves both files internally consistent and
the machine unable to start. This module is the join that catches it.

It runs on every tree the repo ships:

* the **demo tree**, read through the loaders the virtual-accelerator
  entrypoint itself calls (``_load_drive_limits`` and
  ``load_machine_json_channels``), so a change to either loader's merge
  semantics -- the ``defaults`` block, the writable/setpoint filter -- is
  exercised here too rather than re-parsed independently;
* every **Middle Layer export** the repo commits a 2.0 ``va.json`` beside,
  emitted through the virtual-accelerator lane and read back through those same
  two loaders. This is where the widening rule earns its place: an export whose
  device nominal sits outside the family's exported ``Range`` has its band
  widened to include it, precisely so that this invariant holds.
"""

from __future__ import annotations

import pytest
from tests.templates.test_channel_limits_va import TREE_NAMES, Tree, build_tree

from osprey.services.virtual_accelerator.bindings import load_bindings, setpoints
from osprey.services.virtual_accelerator.entrypoint import _load_drive_limits
from osprey.services.virtual_accelerator.manifest import build_manifest, setpoint_addresses
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS

#: The system the demo tree's bindings describe. The classes below say "SR"
#: in their names, and the document is what makes that true.
DEMO_SYSTEM = "SR"


# ===================================================================
# The demo tree
# ===================================================================


@pytest.fixture(scope="module")
def demo_bindings():
    """The demo tree's bindings document -- what it drives, and of which system."""
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def drive_limits() -> dict[str, tuple[float, float]]:
    """{address: (min_value, max_value)} for every writable setpoint address
    with a numeric band -- the entrypoint's own derivation from
    channel_limits.json (entrypoint.py's docstring/main() call this the
    build_records(drive_limits=...) input)."""
    return _load_drive_limits(
        PACKAGE_PATHS.channel_limits,
        setpoints=setpoint_addresses(build_manifest()["channels"]),
    )


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
def family_by_setpoint(demo_bindings) -> dict[str, str]:
    """{setpoint address: family} for every address the tree's bindings drive.

    A coupled channel's manifest entry carries no family: the bindings pair its
    two halves, so the identity keys stay empty and the pair key holds the
    setpoint address instead. The document is therefore the one place that says
    which family a driven address belongs to.
    """
    return {
        binding.setpoint_address: binding.family
        for binding in demo_bindings.bindings
        if binding.is_writable
    }


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

    # Derived from the demo tree's own bindings document, which says which
    # addresses the virtual accelerator drives and of which family: its SR
    # quad/dipole magnet families QF + QD + QFA + DIPOLE = 108 CURRENT:SP
    # addresses, and its corrector families HCM + VCM = 144. Every one of them
    # is a real device with a real calibrated boot value and a real derived
    # band, so every one must appear whole in the join.
    EXPECTED_SR_QUAD_DIPOLE_SP_COUNT = 108
    EXPECTED_SR_CORRECTOR_SP_COUNT = 144

    # 396 manifest :SP addresses all get a channel_limits band (every
    # writable :SP entry has min_value/max_value -- see
    # tests/templates/test_channel_limits_va.py's TestWritableIffSetpoint),
    # but 6 of them (SR:VAC:ION-PUMP:0{1..6}:VOLTAGE:SP, an sp-echo family)
    # have no machine.json entry at all, so they drop out: 396 - 6 = 390.
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
        self, joined_addresses, family_by_setpoint, demo_bindings
    ):
        assert demo_bindings.system == DEMO_SYSTEM, (
            f"these counts are the storage ring's; the tree's bindings describe "
            f"{demo_bindings.system!r}"
        )
        quad_dipole = [
            a
            for a in joined_addresses
            if family_by_setpoint.get(a) in ("QF", "QD", "QFA", "DIPOLE")
        ]
        assert len(quad_dipole) == self.EXPECTED_SR_QUAD_DIPOLE_SP_COUNT, (
            f"expected {self.EXPECTED_SR_QUAD_DIPOLE_SP_COUNT} SR quad/dipole "
            f"CURRENT:SP addresses in the join, got {len(quad_dipole)}"
        )

    def test_join_covers_every_sr_corrector_setpoint(
        self, joined_addresses, family_by_setpoint, demo_bindings
    ):
        assert demo_bindings.system == DEMO_SYSTEM, (
            f"these counts are the storage ring's; the tree's bindings describe "
            f"{demo_bindings.system!r}"
        )
        correctors = [a for a in joined_addresses if family_by_setpoint.get(a) in ("HCM", "VCM")]
        assert len(correctors) == self.EXPECTED_SR_CORRECTOR_SP_COUNT, (
            f"expected {self.EXPECTED_SR_CORRECTOR_SP_COUNT} SR corrector "
            f"CURRENT:SP addresses in the join, got {len(correctors)}"
        )


class TestBootValueFallsInsideItsOwnBand:
    """SC7: for every joined address, min_value <= boot value <= max_value.

    Iterates the full join, not a sample -- a single out-of-band address
    (a derived band excluding its own nominal, a zeroed corrector outside a
    declared window, an sp-echo nominal outside its band) must fail this test.
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


# ===================================================================
# Every tree emitted from a Middle Layer export
# ===================================================================


@pytest.fixture(scope="module", params=[name for name in TREE_NAMES if name != "demo"])
def emitted(request, tmp_path_factory) -> Tree:
    """One committed 2.0 export, run through the whole emit lane.

    Trees that hold a 1.0 export skip, naming themselves and what would make
    them run; the day one is re-exported with ``mml_export`` 2.0 the whole
    invariant runs over it without this module being edited.
    """
    return build_tree(request.param, tmp_path_factory.mktemp(request.param))


@pytest.fixture(scope="module")
def emitted_join(emitted) -> dict[str, tuple[float, tuple[float, float]]]:
    """{address: (boot value, band)} over the emitted tree's own two files.

    Read through the loaders the served process reads them with: the bands
    through the entrypoint's own derivation, keyed by the addresses the
    bindings declare written, and the seeds through the manifest loader the
    entrypoint hands its mounted machine.json.
    """
    limits = _load_drive_limits(emitted.paths.channel_limits, setpoints=setpoints(emitted.document))
    channels = load_machine_json_channels(emitted.paths.machine_json)
    return {
        address: (channels[address]["value"], limits[address])
        for address in sorted(set(limits) & set(channels))
        if "value" in channels[address]
    }


class TestAnEmittedTreeBootsInsideItsBands:
    def test_the_join_covers_every_banded_channel_the_export_seeds(self, emitted, emitted_join):
        """A driven channel with a band and an exported nominal is in the join;
        an empty join would pass the invariant below while proving nothing."""
        expected = {
            binding.setpoint_address
            for binding in emitted.document.bindings
            if binding.is_writable
            and binding.nominal is not None
            and all(bound is not None for bound in emitted.band(binding.setpoint_address))
        }
        assert expected, f"{emitted.name} bands no driven channel it also seeds"
        assert expected <= set(emitted_join), (
            f"banded, seeded channels missing from the join: "
            f"{sorted(expected - set(emitted_join))[:10]}"
        )

    def test_every_seeded_value_is_within_its_band(self, emitted_join):
        violations = [
            f"{address}: boot value {value} outside band [{low}, {high}]"
            for address, (value, (low, high)) in emitted_join.items()
            if not low <= value <= high
        ]
        assert not violations, "boot-band invariant violated:\n" + "\n".join(violations)

    @pytest.mark.usefixtures("emitted_join")
    def test_a_nominal_outside_the_exported_range_widened_its_band(self, emitted):
        """The reason the widening rule exists: where an export states an
        operating ``Range`` its own device nominal sits outside, the band gives
        way to the nominal rather than the tree refusing to boot. Each such
        address is banded exactly to the nominal on the side that pushed it.

        A tree whose every nominal already sits inside its exported ``Range``
        states no such address and asserts nothing here; the committed
        synthetic export carries one deliberately, so the rule is exercised.
        """
        from tests.templates.test_channel_limits_va import _exported_range

        for binding in emitted.document.bindings:
            if not binding.is_writable or binding.nominal is None:
                continue
            low, high = _exported_range(emitted, binding.family, binding.setpoint_address)
            band_low, band_high = emitted.band(binding.setpoint_address)
            if low is not None and binding.nominal < low:
                assert band_low == binding.nominal, binding.setpoint_address
            if high is not None and binding.nominal > high:
                assert band_high == binding.nominal, binding.setpoint_address
