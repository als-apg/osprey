"""Unit tests for the canonical EPICS-substrate plan-device derivation.

Covers ``osprey.services.bluesky_bridge.substrate_devices`` -- the single
host-side source shared by the build's device-file staging
(``compose_generator._stage_bluesky_devices``) and ``tests/e2e/_orm_stack.py``:
the channel selectors, the derived device document, and the atomic write of
that document to a file the queueserver worker mounts.
"""

from __future__ import annotations

import pytest
import yaml

from osprey.services.bluesky_bridge.substrate_devices import (
    devices_document,
    select_bpms,
    select_correctors,
    write_devices_file,
)

# A synthetic channel_limits.json-shaped dict covering:
#  - two full SR HCM/VCM pyat-coupled corrector SP/RB pairs
#  - one SR corrector SP with NO matching RB (must be excluded)
#  - a non-HCM/VCM SR magnet family (QF; in MAG_FAMILIES but not a corrector)
#  - a BR magnet (sp-echo partition; wrong ring for a corrector)
#  - two SR BPM pyat-coupled X/Y position readbacks
#  - a BPM STATUS field (same family, but classify_partition falls through
#    to static-noisy since the field isn't POSITION)
#  - metadata keys ("_meta", "defaults") that must be ignored entirely
_LIMITS = {
    "SR:MAG:HCM:01:CURRENT:SP": {"min": -10, "max": 10},
    "SR:MAG:HCM:01:CURRENT:RB": {"min": -10, "max": 10},
    "SR:MAG:VCM:02:CURRENT:SP": {"min": -10, "max": 10},
    "SR:MAG:VCM:02:CURRENT:RB": {"min": -10, "max": 10},
    "SR:MAG:HCM:03:CURRENT:SP": {"min": -10, "max": 10},  # no RB counterpart
    "SR:MAG:QF:01:CURRENT:SP": {"min": -10, "max": 10},
    "SR:MAG:QF:01:CURRENT:RB": {"min": -10, "max": 10},
    "BR:MAG:HCM:01:CURRENT:SP": {"min": -10, "max": 10},
    "BR:MAG:HCM:01:CURRENT:RB": {"min": -10, "max": 10},
    "SR:DIAG:BPM:01:POSITION:X": {"min": -5, "max": 5},
    "SR:DIAG:BPM:01:POSITION:Y": {"min": -5, "max": 5},
    "SR:DIAG:BPM:02:POSITION:X": {"min": -5, "max": 5},
    "SR:DIAG:BPM:02:POSITION:Y": {"min": -5, "max": 5},
    "SR:DIAG:BPM:03:STATUS:VALID": {"min": 0, "max": 1},
    "_meta": {"ignored": True},
    "defaults": {"ignored": True},
}


class TestSelectCorrectors:
    def test_full_set_default_returns_all_pyat_coupled_pairs(self) -> None:
        correctors = select_correctors(_LIMITS)
        # Only the two complete SR HCM/VCM SP/RB pairs qualify.
        assert len(correctors) == 2
        pairs = set(correctors.values())
        assert ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:RB") in pairs
        assert ("SR:MAG:VCM:02:CURRENT:SP", "SR:MAG:VCM:02:CURRENT:RB") in pairs

    def test_device_name_is_the_setpoint_address(self) -> None:
        """Device name == channel address: an agent that discovered a ``:SP``
        address via channel-finder can name it directly in a plan, with no
        synthetic ``corrector_NN`` namespace in between."""
        correctors = select_correctors(_LIMITS)
        assert set(correctors) == {
            "SR:MAG:HCM:01:CURRENT:SP",
            "SR:MAG:VCM:02:CURRENT:SP",
        }
        assert all(name == sp for name, (sp, _rb) in correctors.items())

    def test_excludes_sp_without_matching_rb(self) -> None:
        correctors = select_correctors(_LIMITS)
        assert not any(sp == "SR:MAG:HCM:03:CURRENT:SP" for sp, _rb in correctors.values())

    def test_excludes_non_hcm_vcm_family(self) -> None:
        correctors = select_correctors(_LIMITS)
        assert not any(sp.startswith("SR:MAG:QF:") for sp, _rb in correctors.values())

    def test_excludes_non_sr_ring(self) -> None:
        correctors = select_correctors(_LIMITS)
        assert not any(sp.startswith("BR:") for sp, _rb in correctors.values())

    def test_count_none_never_raises_regardless_of_availability(self) -> None:
        assert select_correctors({}, count=None) == {}

    def test_count_int_returns_exact_slice(self) -> None:
        correctors = select_correctors(_LIMITS, count=1)
        # Address-keyed on the sliced path too, and the slice takes the
        # lowest-sorting address rather than an arbitrary one.
        assert correctors == {
            "SR:MAG:HCM:01:CURRENT:SP": ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:HCM:01:CURRENT:RB")
        }

    def test_count_int_raises_when_insufficient(self) -> None:
        with pytest.raises(AssertionError):
            select_correctors(_LIMITS, count=5)

    def test_ignores_metadata_keys(self) -> None:
        limits = {"_meta": {}, "defaults": {}}
        assert select_correctors(limits, count=None) == {}


class TestSelectBpms:
    def test_full_set_default_returns_all_pyat_coupled_readbacks(self) -> None:
        readbacks = select_bpms(_LIMITS)
        assert len(readbacks) == 4
        addresses = set(readbacks.values())
        assert "SR:DIAG:BPM:01:POSITION:X" in addresses
        assert "SR:DIAG:BPM:01:POSITION:Y" in addresses
        assert "SR:DIAG:BPM:02:POSITION:X" in addresses
        assert "SR:DIAG:BPM:02:POSITION:Y" in addresses

    def test_device_name_is_the_read_address(self) -> None:
        """Same convention as the correctors: the detector's name is the
        address it reads, so a discovered BPM address is directly usable."""
        readbacks = select_bpms(_LIMITS)
        assert all(name == address for name, address in readbacks.items())

    def test_excludes_non_position_field(self) -> None:
        readbacks = select_bpms(_LIMITS)
        assert "SR:DIAG:BPM:03:STATUS:VALID" not in set(readbacks.values())

    def test_count_none_never_raises_regardless_of_availability(self) -> None:
        assert select_bpms({}, count=None) == {}

    def test_count_int_returns_exact_slice(self) -> None:
        readbacks = select_bpms(_LIMITS, count=2)
        # Address-keyed on the sliced path too (see the corrector counterpart).
        assert readbacks == {
            "SR:DIAG:BPM:01:POSITION:X": "SR:DIAG:BPM:01:POSITION:X",
            "SR:DIAG:BPM:01:POSITION:Y": "SR:DIAG:BPM:01:POSITION:Y",
        }

    def test_count_int_raises_when_insufficient(self) -> None:
        with pytest.raises(AssertionError):
            select_bpms(_LIMITS, count=10)


class TestDevicesDocument:
    def test_both_top_level_keys_are_always_present(self) -> None:
        """Even a project that yields nothing gets both keys, so a caller can
        see WHICH half came up empty instead of inferring it from a missing key."""
        assert devices_document({}) == {"settables": [], "readables": []}

    def test_settable_entries_carry_name_setpoint_and_readback(self) -> None:
        document = devices_document(_LIMITS)
        assert document["settables"] == [
            {
                "name": "SR:MAG:HCM:01:CURRENT:SP",
                "setpoint": "SR:MAG:HCM:01:CURRENT:SP",
                "readback": "SR:MAG:HCM:01:CURRENT:RB",
            },
            {
                "name": "SR:MAG:VCM:02:CURRENT:SP",
                "setpoint": "SR:MAG:VCM:02:CURRENT:SP",
                "readback": "SR:MAG:VCM:02:CURRENT:RB",
            },
        ]

    def test_readable_entries_carry_name_and_pv(self) -> None:
        document = devices_document(_LIMITS)
        assert document["readables"] == [
            {"name": address, "pv": address}
            for address in (
                "SR:DIAG:BPM:01:POSITION:X",
                "SR:DIAG:BPM:01:POSITION:Y",
                "SR:DIAG:BPM:02:POSITION:X",
                "SR:DIAG:BPM:02:POSITION:Y",
            )
        ]

    def test_never_emits_a_null_readback(self) -> None:
        """``readback: null`` is legal for the loader but reads as "unset by
        mistake"; an entry with no separate readback omits the key instead."""
        document = devices_document(_LIMITS)
        assert all(entry.get("readback") is not None for entry in document["settables"])

    def test_the_build_validator_accepts_what_the_producer_emits(self) -> None:
        """The producer and the build's refusal gate must agree: anything this
        writes has to pass ``validate_device_document`` unchanged."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import (
            validate_device_document,
        )

        assert validate_device_document(devices_document(_LIMITS)) == []

    def test_ignores_metadata_keys(self) -> None:
        assert devices_document({"_meta": {}, "defaults": {}}) == {
            "settables": [],
            "readables": [],
        }


class TestWriteDevicesFile:
    def test_written_yaml_parses_back_to_the_returned_document(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        document = write_devices_file(path, _LIMITS)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == document
        assert document == devices_document(_LIMITS)

    def test_header_marks_the_file_generated_and_names_the_unification(self, tmp_path) -> None:
        """The header has to answer both questions a reader of a staged file
        asks: may I edit this (no), and where does the device set come from."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _LIMITS)

        header = [
            line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")
        ]
        text = "\n".join(header)
        assert "Generated by OSPREY" in text
        assert "channel_limits.json" in text
        assert "channel-finder" in text
        assert "knowledge graph" in text

    def test_leaves_no_temp_file_behind(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _LIMITS)

        assert [entry.name for entry in tmp_path.iterdir()] == ["bluesky_devices.yml"]

    def test_rewrite_replaces_rather_than_appends(self, tmp_path) -> None:
        """Every render rewrites the staged file; a second write must not leave
        two concatenated documents behind."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _LIMITS)
        document = write_devices_file(path, _LIMITS)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == document

    def test_failed_write_leaves_the_previous_document_intact(self, tmp_path, monkeypatch) -> None:
        """Atomicity is the point of the temp file: a deploy may be mounting
        this path, so a failure must not truncate what is already there."""
        import os

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, _LIMITS)
        before = path.read_text(encoding="utf-8")

        def _boom(*args, **kwargs):
            raise OSError("rename failed")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            write_devices_file(path, {})

        assert path.read_text(encoding="utf-8") == before
        assert [entry.name for entry in tmp_path.iterdir()] == ["bluesky_devices.yml"]

    def test_written_file_is_readable_by_a_container_user(self, tmp_path) -> None:
        """The file is bind-mounted ``:ro`` into the worker; ``mkstemp``'s 0600
        would make it unreadable to any uid but the one that rendered it."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _LIMITS)

        assert path.stat().st_mode & 0o044 == 0o044


class TestDeviceFileRoundTrip:
    """The written file, read back by the worker's own parser."""

    # Addresses the env-var channel could not have carried: every name holds
    # colons, and one corrector's device component holds a comma (16 of ALS's
    # BTS quadrupoles really do). ``device`` plays no part in
    # ``classify_partition``, so the comma-bearing channel is selected like any
    # other pyat-coupled corrector.
    _AWKWARD_LIMITS = {
        "SR:MAG:HCM:01,02:CURRENT:SP": {"min": -10, "max": 10},
        "SR:MAG:HCM:01,02:CURRENT:RB": {"min": -10, "max": 10},
        "SR:MAG:VCM:03:CURRENT:SP": {"min": -10, "max": 10},
        "SR:MAG:VCM:03:CURRENT:RB": {"min": -10, "max": 10},
        "SR:DIAG:BPM:01:POSITION:X": {"min": -5, "max": 5},
    }

    def test_colon_and_comma_named_devices_survive_the_worker_parser(self, tmp_path) -> None:
        """Nothing on either side splits a value on any character, so an
        address-named device reaches the worker with its name and PVs intact --
        the whole reason the device file replaced the comma-separated env var.
        """
        from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, self._AWKWARD_LIMITS)

        settables, readables = specs_from_file(path)

        assert [(spec.name, spec.setpoint_pv, spec.readback_pv) for spec in settables] == [
            (
                "SR:MAG:HCM:01,02:CURRENT:SP",
                "SR:MAG:HCM:01,02:CURRENT:SP",
                "SR:MAG:HCM:01,02:CURRENT:RB",
            ),
            (
                "SR:MAG:VCM:03:CURRENT:SP",
                "SR:MAG:VCM:03:CURRENT:SP",
                "SR:MAG:VCM:03:CURRENT:RB",
            ),
        ]
        assert [(spec.name, spec.read_pv) for spec in readables] == [
            ("SR:DIAG:BPM:01:POSITION:X", "SR:DIAG:BPM:01:POSITION:X")
        ]

    def test_full_derivation_round_trips_with_nothing_dropped(self, tmp_path) -> None:
        """No two selected addresses collide, so ``_drop_duplicate_names`` keeps
        every spec: 2 correctors and 4 BPM readbacks in, the same 6 out."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, _LIMITS)

        settables, readables = specs_from_file(path)

        assert {spec.name for spec in settables} == {
            "SR:MAG:HCM:01:CURRENT:SP",
            "SR:MAG:VCM:02:CURRENT:SP",
        }
        assert all(spec.name == spec.setpoint_pv for spec in settables)
        assert all(spec.readback_pv == spec.name[: -len(":SP")] + ":RB" for spec in settables)
        assert {spec.name for spec in readables} == {
            "SR:DIAG:BPM:01:POSITION:X",
            "SR:DIAG:BPM:01:POSITION:Y",
            "SR:DIAG:BPM:02:POSITION:X",
            "SR:DIAG:BPM:02:POSITION:Y",
        }
        assert all(spec.name == spec.read_pv for spec in readables)
