"""Tests for pending-judgment detection.

``pending_judgments`` reads one raw family view and asks the reviewer about
exactly three shapes: a row beyond the family's devices, a device no channel
list reaches, and a PV several devices share. Everything else stays a rule, so
the negatives are pinned as carefully as the positives: blank slots inside a
full-length list, broadcast rows, zero-length lists, zero-channel families and
a PV repeated across different fields ask nothing.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from osprey.services.mml.family import FamilyView
from osprey.services.mml.judgments import (
    PendingJudgments,
    PendingRow,
    SupplyGroup,
    pending_judgments,
)


def _view(body: dict, system: str = "SR", name: str = "DCCT") -> FamilyView:
    return FamilyView(system, name, body)


def _pending(body: dict, system: str = "SR", name: str = "DCCT") -> PendingJudgments:
    return pending_judgments(_view(body, system, name))


def _rows(n_devices: int) -> list[list[int]]:
    return [[1, index + 1] for index in range(n_devices)]


class TestIdentity:
    """The view's own grain is carried, never re-derived."""

    def test_system_family_and_device_count_are_carried(self):
        """system, family and n_devices come from the view."""
        pending = _pending({"DeviceList": _rows(2)}, system="LTB", name="bpm-x")
        assert (pending.system, pending.family, pending.n_devices) == ("LTB", "bpm-x", 2)

    def test_body_is_not_modified(self):
        """Detection only reads the body."""
        body = {
            "DeviceList": _rows(2),
            "Monitor": {"ChannelNames": ["P", "P", "Q"]},
        }
        before = copy.deepcopy(body)
        _pending(body)
        assert body == before

    def test_records_are_frozen(self):
        """The records are frozen dataclasses."""
        pending = _pending({"DeviceList": _rows(1), "Monitor": {"ChannelNames": ["A", "B"]}})
        with pytest.raises(dataclasses.FrozenInstanceError):
            pending.rows_beyond[0].index = 0


class TestRowsBeyondDevices:
    """Non-blank slots at or past ``n_devices`` of an over-long list."""

    def test_three_rows_on_one_device(self):
        """A DCCT-shaped family of 3 rows on 1 device pends its 2 extra rows."""
        pending = _pending(
            {
                "DeviceList": [[1, 1]],
                "Monitor": {"ChannelNames": ["CURRENT", "LIFETIME", "TOTAL"]},
            }
        )
        assert pending.rows_beyond == (
            PendingRow("Monitor", ("ChannelNames",), 1, "LIFETIME"),
            PendingRow("Monitor", ("ChannelNames",), 2, "TOTAL"),
        )
        assert pending.unbound_devices == ()
        assert pending.groups == ()

    def test_identical_strings_are_one_row(self):
        """``[a, b, b]`` on one device pends one row and no supply group."""
        pending = _pending(
            {
                "DeviceList": [[1, 1]],
                "Monitor": {"ChannelNames": ["A", "B", "B"]},
            }
        )
        assert pending.rows_beyond == (PendingRow("Monitor", ("ChannelNames",), 1, "B"),)
        assert pending.groups == ()

    def test_blank_slots_beyond_devices_are_not_rows(self):
        """A blank slot past the devices asks nothing."""
        pending = _pending(
            {
                "DeviceList": [[1, 1]],
                "Monitor": {"ChannelNames": ["A", None, "  ", "B"]},
            }
        )
        assert pending.rows_beyond == (PendingRow("Monitor", ("ChannelNames",), 3, "B"),)

    def test_blank_slots_inside_a_full_length_list_are_not_rows(self):
        """A list exactly as long as the family asks nothing."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["A", None, "C"]},
            }
        )
        assert pending.is_empty

    def test_one_string_on_both_keys_is_one_row(self):
        """A dual-key row carrying one string is one question on both keys."""
        pending = _pending(
            {
                "DeviceList": [[1, 1]],
                "Monitor": {"ChannelNames": ["A", "B"], "TangoNames": ["A", "B"]},
            }
        )
        assert pending.rows_beyond == (
            PendingRow("Monitor", ("ChannelNames", "TangoNames"), 1, "B"),
        )

    def test_two_strings_on_one_row_are_two_rows(self):
        """A dual-key row whose keys differ is one question per string."""
        pending = _pending(
            {
                "DeviceList": [[1, 1]],
                "Monitor": {"ChannelNames": ["A", "B"], "TangoNames": ["A", "T"]},
            }
        )
        assert pending.rows_beyond == (
            PendingRow("Monitor", ("ChannelNames",), 1, "B"),
            PendingRow("Monitor", ("TangoNames",), 1, "T"),
        )

    def test_the_same_string_in_two_fields_is_one_row_per_field(self):
        """Fields are asked separately; the signal keys the question inside one."""
        pending = _pending(
            {
                "DeviceList": [[1, 1]],
                "Monitor": {"ChannelNames": ["A", "B"]},
                "Setpoint": {"ChannelNames": ["A", "B"]},
            }
        )
        assert pending.rows_beyond == (
            PendingRow("Monitor", ("ChannelNames",), 1, "B"),
            PendingRow("Setpoint", ("ChannelNames",), 1, "B"),
        )


class TestUnboundDevices:
    """Ordinals past the longest expanded list."""

    def test_three_devices_two_rows(self):
        """A TUNE-shaped family of 3 devices and 2 rows pends device 3."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["A", "B"]},
            }
        )
        assert pending.unbound_devices == (3,)
        assert pending.rows_beyond == ()
        assert pending.groups == ()

    def test_every_unreached_ordinal_is_pending(self):
        """Two devices past the lists are two questions."""
        pending = _pending(
            {
                "DeviceList": _rows(4),
                "Monitor": {"ChannelNames": ["A", "B"]},
            }
        )
        assert pending.unbound_devices == (3, 4)

    def test_a_broadcast_field_reaches_every_device(self):
        """A broadcast field alongside a 2-row list on 3 devices asks nothing."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["A", "B"]},
                "Setpoint": {"ChannelNames": ["S"]},
            }
        )
        assert pending.is_empty

    def test_a_broadcast_key_of_one_field_reaches_every_device(self):
        """One 2-row and one 1-row list on 3 devices asks nothing."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["A", "B"], "TangoNames": ["T"]},
            }
        )
        assert pending.is_empty

    def test_a_zero_channel_family_asks_nothing(self):
        """A family whose lists are empty or blank has no unbound device."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": []},
                "Setpoint": {"ChannelNames": [None, None]},
            }
        )
        assert pending.is_empty

    def test_a_family_without_fields_asks_nothing(self):
        """Devices with no field at all are not unbound devices."""
        assert _pending({"DeviceList": _rows(3)}).is_empty

    def test_rows_beyond_devices_do_not_leave_devices_unbound(self):
        """A list longer than the family reaches every device."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["A", "B", "C"]},
            }
        )
        assert pending.unbound_devices == ()
        assert pending.rows_beyond == (PendingRow("Monitor", ("ChannelNames",), 2, "C"),)


class TestSharedPVs:
    """One string bound at two or more device indices of one list."""

    def test_two_fields_sharing_an_index_set_are_one_group(self):
        """``[p, p, q]`` and ``[r, r, s]`` on 3 devices pend one group of two PVs."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["P", "P", "Q"]},
                "Setpoint": {"ChannelNames": ["R", "R", "S"]},
            }
        )
        assert pending.groups == (
            SupplyGroup(
                lowest=1,
                ordinals=(1, 2),
                pvs=("P", "R"),
                device_rows=((1, 1), (1, 2)),
                group_only_members=2,
            ),
        )
        assert pending.count == 1

    def test_different_index_sets_are_different_groups(self):
        """Groups are keyed by index set, in lowest-ordinal order."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["P", "P", "Q"]},
                "Setpoint": {"ChannelNames": ["R", "S", "S"]},
            }
        )
        assert [group.ordinals for group in pending.groups] == [(1, 2), (2, 3)]
        assert [group.lowest for group in pending.groups] == [1, 2]
        assert [group.pvs for group in pending.groups] == [("P",), ("S",)]

    def test_a_pv_repeated_across_fields_is_not_a_group(self):
        """One PV in two fields, once each, binds one device twice, not two devices."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["P", "Q"]},
                "Setpoint": {"ChannelNames": ["P", "R"]},
            }
        )
        assert pending.is_empty

    def test_a_broadcast_list_is_not_a_group(self):
        """A 1-row list reaching every device is a rule, not a judgment."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["P"]},
            }
        )
        assert pending.is_empty

    def test_a_repetition_beyond_the_devices_is_not_a_group(self):
        """Only indices below ``n_devices`` are counted."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["A", "P", "P"]},
            }
        )
        assert pending.groups == ()
        assert pending.rows_beyond == (PendingRow("Monitor", ("ChannelNames",), 2, "P"),)

    def test_device_rows_are_none_when_the_family_states_none(self):
        """A family counted by its longest list has no DeviceList rows to print."""
        pending = _pending({"Monitor": {"ChannelNames": ["P", "P", "Q"]}})
        assert pending.groups[0].device_rows == (None, None)


class TestGroupOnlyMembers:
    """Members whose every channel belongs to the group."""

    def test_every_member_is_group_only(self):
        """Two devices bound by the group's PVs alone."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["P", "P"]},
                "Setpoint": {"ChannelNames": ["Q", "Q"]},
            }
        )
        assert pending.groups[0].pvs == ("P", "Q")
        assert pending.groups[0].group_only_members == 2

    def test_a_member_with_its_own_channel_is_not_group_only(self):
        """A device carrying one channel of its own leaves the count at one."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["P", "P"]},
                "Setpoint": {"ChannelNames": ["Q", None]},
            }
        )
        assert pending.groups[0].pvs == ("P",)
        assert pending.groups[0].group_only_members == 1

    def test_no_member_is_group_only(self):
        """Both devices carrying a channel of their own leave the count at zero."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["P", "P"]},
                "Setpoint": {"ChannelNames": ["Q", "R"]},
            }
        )
        assert pending.groups[0].group_only_members == 0

    def test_a_broadcast_channel_counts_as_outside_the_group(self):
        """A broadcast field binds every member to a PV the group does not own."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["P", "P"]},
                "Setpoint": {"ChannelNames": ["S"]},
            }
        )
        assert pending.groups[0].group_only_members == 0


class TestFallbackFamilies:
    """A family without a ``DeviceList`` can pend shared PVs alone."""

    def test_a_repeated_string_pends_a_group(self):
        """One repeated string is the family's only question."""
        pending = _pending({"Monitor": {"ChannelNames": ["P", "P", "Q"]}})
        assert pending.n_devices == 3
        assert pending.rows_beyond == ()
        assert pending.unbound_devices == ()
        assert pending.groups[0].ordinals == (1, 2)

    def test_a_shorter_list_pends_nothing(self):
        """Lists of different lengths ask nothing without a stated device count."""
        pending = _pending(
            {
                "Monitor": {"ChannelNames": ["A", "B", "C"]},
                "Setpoint": {"ChannelNames": ["D", "E"]},
            }
        )
        assert pending.is_empty


class TestCount:
    """The ``--init`` slot count: one per row, one per ordinal, one per supply."""

    def test_an_empty_family(self):
        """Nothing pending is no slot."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["A", "B"]},
            }
        )
        assert pending.is_empty
        assert pending.count == 0

    def test_rows_and_groups_together(self):
        """Two rows and two supply groups are three slots."""
        pending = _pending(
            {
                "DeviceList": _rows(4),
                "Monitor": {"ChannelNames": ["P", "P", "Q", None, "X", "Y"]},
                "Setpoint": {"ChannelNames": ["R", "S", "S"]},
            }
        )
        assert len(pending.rows_beyond) == 2
        assert pending.unbound_devices == ()
        assert [group.ordinals for group in pending.groups] == [(1, 2), (2, 3)]
        assert pending.count == 3
        assert not pending.is_empty

    def test_rows_beyond_and_unbound_devices_never_coexist(self):
        """A list past the devices reaches them all, so no ordinal is left unbound."""
        pending = _pending(
            {
                "DeviceList": _rows(3),
                "Monitor": {"ChannelNames": ["A", "B"]},
                "Setpoint": {"ChannelNames": ["C", "D", "E", "F"]},
            }
        )
        assert pending.rows_beyond == (PendingRow("Setpoint", ("ChannelNames",), 3, "F"),)
        assert pending.unbound_devices == ()
        assert pending.count == 1

    def test_a_group_is_one_slot_however_many_pvs(self):
        """The supply is answered once per family."""
        pending = _pending(
            {
                "DeviceList": _rows(2),
                "Monitor": {"ChannelNames": ["P", "P"]},
                "Setpoint": {"ChannelNames": ["Q", "Q"]},
            }
        )
        assert len(pending.groups[0].pvs) == 2
        assert pending.count == 1
