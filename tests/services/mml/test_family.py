"""Tests for the MML family view.

``FamilyView`` is the one place the family grain is computed: where the family
arrays come from, how many devices a family has, which sub-dicts are fields,
how a 1-row channel list broadcasts, and the raw and binding counts. Every
consumer (census, voter, mapping skeleton, emitters) reads these numbers, so
each rule is pinned here on a normalised body.
"""

from __future__ import annotations

import copy

import pytest

from osprey.services.mml.family import FamilyView, FieldView


def _view(body: dict, system: str = "SR", name: str = "BPM") -> FamilyView:
    return FamilyView(system, name, body)


class TestIdentity:
    """Constructor arguments are kept verbatim."""

    def test_system_raw_name_and_body_are_kept(self):
        """system, raw_name and body are exposed as given."""
        body = {"Monitor": {"ChannelNames": ["A"]}}
        view = FamilyView("RING", "bpm-x", body)
        assert view.system == "RING"
        assert view.raw_name == "bpm-x"
        assert view.body is body

    def test_body_is_not_modified(self):
        """Building a view and reading every slot leaves the body untouched."""
        body = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["A"]},
        }
        before = copy.deepcopy(body)
        view = _view(body)
        view.fields["Monitor"].slots("ChannelNames")
        _ = view.channel_count
        assert body == before


class TestArrays:
    """Family arrays come from the family level, else from setup/_setup."""

    def test_family_level_arrays(self):
        """Arrays at the family level are read from there."""
        view = _view({"DeviceList": [[1, 1]], "Status": [1], "Units": "mm"})
        assert view.arrays == {"DeviceList": [[1, 1]], "Status": [1]}
        assert view.arrays_source == "family"

    def test_setup_arrays(self):
        """Arrays only under setup are read from setup."""
        view = _view({"setup": {"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["a", "b"]}})
        assert view.arrays == {"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["a", "b"]}
        assert view.arrays_source == "setup"

    def test_underscore_setup_arrays(self):
        """Arrays under _setup are read like setup."""
        view = _view({"_setup": {"Position": [1.5]}})
        assert view.arrays == {"Position": [1.5]}
        assert view.arrays_source == "setup"

    def test_family_level_wins_over_setup(self):
        """When a key is at both levels the family-level value wins."""
        view = _view({"DeviceList": [[1, 1]], "setup": {"DeviceList": [[9, 9], [9, 8]]}})
        assert view.arrays["DeviceList"] == [[1, 1]]
        assert view.arrays_source == "family"

    def test_setup_preferred_over_underscore_setup(self):
        """setup is consulted before _setup."""
        view = _view({"setup": {"Status": [1]}, "_setup": {"Status": [0]}})
        assert view.arrays["Status"] == [1]

    def test_no_arrays(self):
        """A family with no arrays anywhere has empty arrays sourced from the family."""
        view = _view({"Monitor": {"ChannelNames": ["A"]}})
        assert view.arrays == {}
        assert view.arrays_source == "family"

    def test_non_dict_setup_is_ignored(self):
        """A setup value that is not a dict contributes nothing."""
        view = _view({"setup": "none"})
        assert view.arrays == {}


class TestDeviceCount:
    """n_devices from DeviceList, else the longest channel list."""

    def test_nx2_device_list(self):
        """An Nx2 DeviceList gives N devices."""
        view = _view({"DeviceList": [[1, 1], [1, 2], [2, 1]], "M": {"ChannelNames": ["A"]}})
        assert view.n_devices == 3
        assert view.n_devices_from_fallback is False

    def test_single_row_nx2_device_list(self):
        """A 1x2 DeviceList gives one device."""
        assert _view({"DeviceList": [[4, 1]]}).n_devices == 1

    def test_flat_pair_device_list(self):
        """A flat [sector, device] pair gives one device."""
        view = _view({"DeviceList": [3, 1], "M": {"ChannelNames": ["A", "B", "C"]}})
        assert view.n_devices == 1
        assert view.n_devices_from_fallback is False

    def test_device_list_from_setup(self):
        """The DeviceList under setup counts like a family-level one."""
        assert _view({"setup": {"DeviceList": [[1, 1], [1, 2]]}}).n_devices == 2

    def test_fallback_longest_channel_list(self):
        """Without a usable DeviceList the longest channel list across fields decides."""
        view = _view(
            {
                "Monitor": {"ChannelNames": ["A", "B"]},
                "Setpoint": {"ChannelNames": ["C"], "TangoNames": ["t/1", "t/2", "t/3"]},
            }
        )
        assert view.n_devices == 3
        assert view.n_devices_from_fallback is True

    def test_empty_device_list_falls_back(self):
        """An empty DeviceList is not an Nx2 list and falls back."""
        view = _view({"DeviceList": [], "M": {"ChannelNames": ["A", "B"]}})
        assert view.n_devices == 2
        assert view.n_devices_from_fallback is True

    def test_ragged_device_list_falls_back(self):
        """A DeviceList with rows that are not pairs falls back."""
        view = _view({"DeviceList": [[1, 1], [1]], "M": {"ChannelNames": ["A"]}})
        assert view.n_devices == 1
        assert view.n_devices_from_fallback is True

    def test_no_fields_no_device_list(self):
        """A family with no DeviceList and no fields has zero devices."""
        view = _view({"Units": "A"})
        assert view.n_devices == 0
        assert view.n_devices_from_fallback is True


class TestFields:
    """A field is a sub-dict carrying a channel key."""

    def test_fields_are_channel_keyed_sub_dicts(self):
        """Only sub-dicts with ChannelNames or TangoNames are fields, in body order."""
        view = _view(
            {
                "Setpoint": {"TangoNames": ["t/a"]},
                "Monitor": {"ChannelNames": ["A"]},
                "Other": {"Units": "A"},
                "setup": {"ChannelNames": ["S"]},
                "_setup": {"ChannelNames": ["S"]},
                "_meta": {"ChannelNames": ["M"]},
                "Units": "A",
            }
        )
        assert list(view.fields) == ["Setpoint", "Monitor"]
        assert all(isinstance(field, FieldView) for field in view.fields.values())
        assert view.fields["Monitor"].name == "Monitor"

    def test_keys_in_channel_key_order(self):
        """FieldView.keys lists the present channel keys in CHANNEL_KEYS order."""
        view = _view({"M": {"TangoNames": ["t"], "ChannelNames": ["c"]}, "N": {"TangoNames": []}})
        assert view.fields["M"].keys == ("ChannelNames", "TangoNames")
        assert view.fields["N"].keys == ("TangoNames",)

    def test_raw_slots_exactly_as_exported(self):
        """raw_slots returns the list as exported, blank slots included."""
        view = _view({"DeviceList": [[1, 1], [1, 2]], "M": {"ChannelNames": ["A", None]}})
        assert view.fields["M"].raw_slots("ChannelNames") == ["A", None]

    def test_raw_slots_returns_a_copy(self):
        """Mutating the returned list does not reach the body."""
        body = {"M": {"ChannelNames": ["A"]}}
        _view(body).fields["M"].raw_slots("ChannelNames").append("X")
        assert body["M"]["ChannelNames"] == ["A"]

    def test_absent_key_raises(self):
        """Asking for a channel key the field does not carry raises KeyError."""
        field = _view({"M": {"ChannelNames": ["A"]}}).fields["M"]
        with pytest.raises(KeyError):
            field.raw_slots("TangoNames")
        with pytest.raises(KeyError):
            field.slots("TangoNames")

    def test_field_description(self):
        """A field description comes from _description, else Description."""
        view = _view(
            {
                "A": {"ChannelNames": ["a"], "_description": "under", "Description": "plain"},
                "B": {"ChannelNames": ["b"], "Description": "plain"},
                "C": {"ChannelNames": ["c"], "Description": ""},
                "D": {"ChannelNames": ["d"]},
            }
        )
        assert view.fields["A"].description == "under"
        assert view.fields["B"].description == "plain"
        assert view.fields["C"].description is None
        assert view.fields["D"].description is None


class TestBroadcast:
    """A 1-row list broadcasts to n_devices; 0-length and partial lists do not."""

    def test_one_row_list_broadcasts(self):
        """A 1-row list with n_devices > 1 expands in slots() and keeps raw_slots()."""
        view = _view({"DeviceList": [[1, 1], [1, 2], [1, 3]], "M": {"ChannelNames": ["PV"]}})
        field = view.fields["M"]
        assert field.broadcast is True
        assert field.raw_slots("ChannelNames") == ["PV"]
        assert field.slots("ChannelNames") == ["PV", "PV", "PV"]

    def test_full_list_does_not_broadcast(self):
        """A list of n_devices slots is returned as-is."""
        view = _view({"DeviceList": [[1, 1], [1, 2]], "M": {"ChannelNames": ["A", "B"]}})
        field = view.fields["M"]
        assert field.broadcast is False
        assert field.slots("ChannelNames") == ["A", "B"]

    def test_single_device_single_slot_is_not_broadcast(self):
        """With one device a 1-row list is simply aligned."""
        field = _view({"DeviceList": [1, 1], "M": {"ChannelNames": ["A"]}}).fields["M"]
        assert field.broadcast is False
        assert field.slots("ChannelNames") == ["A"]

    def test_empty_list_returned_as_is(self):
        """A 0-length list is returned as-is by both accessors."""
        field = _view({"DeviceList": [[1, 1], [1, 2]], "M": {"ChannelNames": []}}).fields["M"]
        assert field.raw_slots("ChannelNames") == []
        assert field.slots("ChannelNames") == []
        assert field.broadcast is False

    def test_partial_list_returned_as_is(self):
        """A list shorter than n_devices (but not 1-row) is returned as-is."""
        view = _view({"DeviceList": [[1, 1], [1, 2], [1, 3]], "M": {"ChannelNames": ["A", "B"]}})
        field = view.fields["M"]
        assert field.slots("ChannelNames") == ["A", "B"]
        assert field.broadcast is False

    def test_dual_key_broadcast_per_key(self):
        """On a dual-key field each key broadcasts on its own shape."""
        view = _view(
            {
                "DeviceList": [[1, 1], [1, 2]],
                "M": {"ChannelNames": ["A", "B"], "TangoNames": ["t/x"]},
            }
        )
        field = view.fields["M"]
        assert field.slots("ChannelNames") == ["A", "B"]
        assert field.slots("TangoNames") == ["t/x", "t/x"]
        assert field.broadcast is True


class TestCounts:
    """raw_slot_count counts raw slots; channel_count counts bindings."""

    def test_counts_across_fields_and_keys(self):
        """Raw slots are counted as exported; bindings are non-blank after expansion."""
        view = _view(
            {
                "DeviceList": [[1, 1], [1, 2], [1, 3]],
                "Monitor": {"ChannelNames": ["A", None, "C"]},
                "Setpoint": {"ChannelNames": ["S"], "TangoNames": ["t/1", "t/2", " "]},
                "Empty": {"ChannelNames": []},
                "Partial": {"TangoNames": ["p/1", "p/2"]},
            }
        )
        # raw: 3 + 1 + 3 + 0 + 2
        assert view.raw_slot_count == 9
        # bindings: 2 + 3 (broadcast) + 2 + 0 + 2
        assert view.channel_count == 9
        assert view.fields["Monitor"].raw_slot_count == 3
        assert view.fields["Monitor"].channel_count == 2
        assert view.fields["Setpoint"].raw_slot_count == 4
        assert view.fields["Setpoint"].channel_count == 5

    def test_broadcast_blank_slot_counts_zero(self):
        """A broadcast blank slot yields no bindings."""
        view = _view({"DeviceList": [[1, 1], [1, 2]], "M": {"ChannelNames": [None]}})
        assert view.raw_slot_count == 1
        assert view.channel_count == 0

    def test_family_without_fields(self):
        """A family with no channel key reports zero counts and no fields."""
        view = _view({"DeviceList": [[1, 1]], "Units": "A"})
        assert view.fields == {}
        assert view.raw_slot_count == 0
        assert view.channel_count == 0


class TestFamilyDescription:
    """Family description with provenance ``imported``."""

    def test_underscore_description(self):
        """_description is carried as imported."""
        assert _view({"_description": "Beam position"}).description == (
            "Beam position",
            "imported",
        )

    def test_mml_description(self):
        """An MML Description is carried as imported."""
        assert _view({"Description": "BPMs"}).description == ("BPMs", "imported")

    def test_underscore_description_first(self):
        """_description is preferred when both are present."""
        view = _view({"_description": "osprey", "Description": "mml"})
        assert view.description == ("osprey", "imported")

    def test_blank_or_missing_description(self):
        """No description, a blank one or a non-string one gives None."""
        assert _view({}).description is None
        assert _view({"Description": "  "}).description is None
        assert _view({"Description": ["x"]}).description is None


class TestDisabledDevices:
    """Disabled devices are the positions where an aligned Status is 0."""

    def test_aligned_status(self):
        """Zero entries of an aligned Status list are disabled, by 0-based position."""
        view = _view({"DeviceList": [[1, 1], [1, 2], [1, 3]], "Status": [1, 0, 0]})
        assert view.disabled_devices == (1, 2)

    def test_status_from_setup(self):
        """Status under setup is used like a family-level one."""
        view = _view({"setup": {"DeviceList": [[1, 1], [1, 2]], "Status": [0, 1]}})
        assert view.disabled_devices == (0,)

    def test_misaligned_status_ignored(self):
        """A Status list whose length differs from n_devices disables nothing."""
        view = _view({"DeviceList": [[1, 1], [1, 2], [1, 3]], "Status": [0, 0]})
        assert view.disabled_devices == ()

    def test_scalar_status_single_device(self):
        """A scalar Status 0 on a one-device family disables device 0."""
        assert _view({"DeviceList": [1, 1], "Status": 0}).disabled_devices == (0,)

    def test_non_numeric_status_slots_ignored(self):
        """None or string slots never count as disabled."""
        view = _view({"DeviceList": [[1, 1], [1, 2], [1, 3]], "Status": [None, "NaN", 0.0]})
        assert view.disabled_devices == (2,)

    def test_no_status(self):
        """Without Status no device is disabled."""
        assert _view({"DeviceList": [[1, 1]]}).disabled_devices == ()


class TestAligned:
    """aligned() returns a family array only when it has one slot per device."""

    def test_aligned_array(self):
        """A list of n_devices entries is aligned."""
        view = _view({"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["a", "b"]})
        assert view.aligned("CommonNames") == ["a", "b"]

    def test_misaligned_or_absent(self):
        """A list of another length, a scalar on many devices, or no array is not aligned."""
        view = _view({"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["a"], "Position": 1.0})
        assert view.aligned("CommonNames") is None
        assert view.aligned("Position") is None
        assert view.aligned("ElementList") is None

    def test_scalar_on_single_device(self):
        """A scalar on a one-device family is aligned as a one-slot list."""
        assert _view({"DeviceList": [1, 1], "Position": 2.5}).aligned("Position") == [2.5]
