"""The protocol a middle-layer channel speaks, and the readers that surface it.

``ChannelNames`` lists Channel Access names (protocol ``ca``) and
``TangoNames`` lists Tango device attributes (protocol ``tango``). The channel
map records which, ``inspect_fields`` labels a field by the key it carries, and
``list_channel_names`` answers either key on request, and narrows either by
place and device ordinal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase


def _write(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "middle_layer.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


DUAL_KEY = {
    "RING": {
        "KICK": {
            "Voltage": {
                "ChannelNames": ["K1:V", " K2:V "],
                "TangoNames": ["dom/kick/1/v", "dom/kick/2/v"],
                "Description": "Kicker voltage",
            },
            "Current": {
                "X": {"ChannelNames": "K:IX", "TangoNames": "dom/kick/ix"},
            },
            "setup": {"DeviceList": [[1, 1], [1, 2]]},
        }
    }
}

TANGO_ONLY = {
    "RING": {
        "KICK": {
            "Voltage": {
                "TangoNames": ["dom/kick/1/v", "dom/kick/2/v"],
                "_description": "Kicker voltage",
            },
            "Current": {"X": {"TangoNames": "dom/kick/ix"}},
            "setup": {"DeviceList": [[1, 1], [1, 2]]},
        }
    }
}


@pytest.fixture
def dual(tmp_path):
    return MiddleLayerDatabase(str(_write(tmp_path, DUAL_KEY)))


@pytest.fixture
def tango(tmp_path):
    return MiddleLayerDatabase(str(_write(tmp_path, TANGO_ONLY)))


class TestChannelMapProtocol:
    """Every membership of a channel-map entry names the protocol of the key that listed it."""

    def test_dual_key_entries_carry_their_protocol(self, dual):
        """CA names are ``ca``, Tango attributes ``tango``, on one field."""
        assert dual.channel_map["K1:V"]["memberships"][0]["protocol"] == "ca"
        assert dual.channel_map["K2:V"]["memberships"][0]["protocol"] == "ca"
        assert dual.channel_map["dom/kick/1/v"]["memberships"][0]["protocol"] == "tango"
        assert dual.channel_map["K:IX"]["memberships"][0]["protocol"] == "ca"
        assert dual.channel_map["dom/kick/ix"]["memberships"][0]["protocol"] == "tango"

    def test_tango_only_entries_are_tango(self, tango):
        """A Tango-only database yields only ``tango`` entries."""
        assert {
            membership["protocol"]
            for entry in tango.channel_map.values()
            for membership in entry["memberships"]
        } == {"tango"}
        assert tango.channel_map["dom/kick/ix"]["memberships"][0]["subfield"] == ["X"]

    def test_a_name_under_both_keys_keeps_the_first_key(self, tmp_path):
        """A string listed under both keys of one field resolves in key order."""
        body = {"R": {"F": {"V": {"TangoNames": ["same"], "ChannelNames": ["same"]}}}}
        db = MiddleLayerDatabase(str(_write(tmp_path, body)))
        assert db.channel_map["same"]["memberships"][0]["protocol"] == "ca"


class TestInspectFieldsLabels:
    """A field is labelled by its first present channel key."""

    def test_family_level_labels_dual_key_as_channel_names(self, dual):
        """``ChannelNames`` precedes ``TangoNames`` in the label."""
        fields = dual.inspect_fields("RING", "KICK")
        assert fields["Voltage"]["type"] == "ChannelNames"
        assert fields["Current"]["type"] == "dict (has subfields)"

    def test_family_level_labels_tango_only_as_tango_names(self, tango):
        """A Tango-only field is labelled ``TangoNames``, not a subfield dict."""
        fields = tango.inspect_fields("RING", "KICK")
        assert fields["Voltage"]["type"] == "TangoNames"
        assert fields["Current"]["type"] == "dict (has subfields)"

    def test_description_falls_back_to_capitalised_key(self, dual, tango):
        """``_description`` first, then ``Description``, then empty."""
        assert dual.inspect_fields("RING", "KICK")["Voltage"]["description"] == "Kicker voltage"
        assert tango.inspect_fields("RING", "KICK")["Voltage"]["description"] == "Kicker voltage"
        assert dual.inspect_fields("RING", "KICK")["Current"]["description"] == ""

    def test_field_level_labels_subfields_by_key(self, dual, tango):
        """Subfields carrying either key are labelled by that key."""
        assert dual.inspect_fields("RING", "KICK", "Current")["X"]["type"] == "ChannelNames"
        assert tango.inspect_fields("RING", "KICK", "Current")["X"]["type"] == "TangoNames"

    def test_field_level_call_reports_no_setup_subfield(self, tmp_path):
        """A ``setup`` block inside a field is not a subfield."""
        body = {
            "R": {
                "F": {
                    "Setpoint": {
                        "X": {"TangoNames": ["r/f/x"]},
                        "setup": {"DeviceList": [[1, 1]]},
                        "Y": {"Units": "A"},
                    }
                }
            }
        }
        db = MiddleLayerDatabase(str(_write(tmp_path, body)))
        fields = db.inspect_fields("R", "F", "Setpoint")
        assert "setup" not in fields
        assert fields["X"]["type"] == "TangoNames"
        assert fields["Y"]["type"] == "dict (subfield)"


class TestListChannelNamesProtocol:
    """``protocol`` selects the key; absent, the first present key wins."""

    def test_default_returns_first_present_key(self, dual, tango):
        """No protocol: ``ChannelNames`` on dual-key, ``TangoNames`` on Tango-only."""
        assert dual.list_channel_names("RING", "KICK", "Voltage") == ["K1:V", "K2:V"]
        assert tango.list_channel_names("RING", "KICK", "Voltage") == [
            "dom/kick/1/v",
            "dom/kick/2/v",
        ]

    def test_tango_protocol_on_dual_key(self, dual):
        """``protocol='tango'`` returns the ``TangoNames`` list."""
        assert dual.list_channel_names("RING", "KICK", "Voltage", protocol="tango") == [
            "dom/kick/1/v",
            "dom/kick/2/v",
        ]
        assert dual.list_channel_names("RING", "KICK", "Current", "X", protocol="tango") == [
            "dom/kick/ix"
        ]

    def test_ca_protocol_on_dual_key(self, dual):
        """``protocol='ca'`` returns the ``ChannelNames`` list."""
        assert dual.list_channel_names("RING", "KICK", "Voltage", protocol="ca") == [
            "K1:V",
            "K2:V",
        ]

    def test_tango_protocol_with_device_filter(self, dual):
        """Device filtering applies to the selected key's list."""
        assert dual.list_channel_names(
            "RING", "KICK", "Voltage", None, None, [2], protocol="tango"
        ) == ["dom/kick/2/v"]

    def test_absent_protocol_names_the_keys_present(self, tango):
        """Asking a Tango-only field for ``ca`` names what it does carry."""
        with pytest.raises(ValueError, match="RING:KICK:Voltage") as excinfo:
            tango.list_channel_names("RING", "KICK", "Voltage", protocol="ca")
        message = str(excinfo.value)
        assert "ChannelNames" in message
        assert "TangoNames" in message

    def test_unknown_protocol_is_refused(self, dual):
        """A protocol outside ``ca``/``tango`` is a ``ValueError``."""
        with pytest.raises(ValueError, match="pva"):
            dual.list_channel_names("RING", "KICK", "Voltage", protocol="pva")

    def test_protocol_is_keyword_only(self, dual):
        """A seventh positional argument does not bind ``protocol``."""
        with pytest.raises(TypeError):
            dual.list_channel_names("RING", "KICK", "Voltage", None, None, None, "tango")


PLACED = {
    "R": {
        "Q": {
            "Current": {"ChannelNames": ["Q1:I", "Q2:I", "Q3:I", "Q4:I"]},
            "setup": {
                "CommonNames": ["Q1", "Q2", "Q3", "Q4"],
                "DeviceList": [[0, 1], [1, 1], [1, 2], [0, 2]],
                "PlaceList": ["R", "R/A", "R/A", "R"],
            },
        },
        "K": {
            "Current": {"ChannelNames": ["K1:I", "K2:I", "K3:I"]},
            "setup": {"DeviceList": [[1, 1], [2, 1], [1, 2]]},
        },
    }
}


@pytest.fixture
def placed(tmp_path):
    return MiddleLayerDatabase(str(_write(tmp_path, PLACED)))


class TestPlaceFilter:
    """``place`` narrows a field to a place and everything below it."""

    def test_a_place_id_selects_its_own_devices(self, placed):
        assert placed.list_channel_names("R", "Q", "Current", place="R/A") == ["Q2:I", "Q3:I"]

    def test_a_place_selects_every_place_below_it(self, placed):
        assert placed.list_channel_names("R", "Q", "Current", place="R") == [
            "Q1:I",
            "Q2:I",
            "Q3:I",
            "Q4:I",
        ]

    def test_a_place_id_is_not_a_text_prefix(self, tmp_path):
        body = json.loads(json.dumps(PLACED))
        body["R"]["Q"]["setup"]["PlaceList"] = ["R/A1", "R/A", "R/A/G", "R/A10"]
        database = MiddleLayerDatabase(str(_write(tmp_path, body)))

        assert database.list_channel_names("R", "Q", "Current", place="R/A") == ["Q2:I", "Q3:I"]

    def test_a_device_ordinal_spans_places(self, placed):
        assert placed.list_channel_names("R", "Q", "Current", devices=[2]) == ["Q3:I", "Q4:I"]

    def test_place_and_ordinal_narrow_together(self, placed):
        assert placed.list_channel_names("R", "Q", "Current", place="R/A", devices=[2]) == ["Q3:I"]

    def test_a_place_no_device_sits_in_is_refused(self, placed):
        with pytest.raises(ValueError, match="place: R/Z"):
            placed.list_channel_names("R", "Q", "Current", place="R/Z")

    def test_a_placed_family_does_not_take_a_bare_index(self, placed):
        with pytest.raises(ValueError, match="place: 1"):
            placed.list_channel_names("R", "Q", "Current", place="1")

    def test_a_family_with_no_place_list_takes_the_bare_index(self, placed):
        assert placed.list_channel_names("R", "K", "Current", place="1") == ["K1:I", "K3:I"]

    def test_a_place_list_of_another_length_is_refused(self, tmp_path):
        body = json.loads(json.dumps(PLACED))
        body["R"]["Q"]["setup"]["PlaceList"] = ["R"]
        database = MiddleLayerDatabase(str(_write(tmp_path, body)))

        with pytest.raises(ValueError, match="PlaceList length"):
            database.list_channel_names("R", "Q", "Current", place="R")


class TestDeviceInfo:
    """``get_device_info`` hands the explorer a family's rows and places."""

    def test_a_placed_family_lists_its_places(self, placed):
        assert placed.get_device_info("R", "Q") == {
            "common_names": ["Q1", "Q2", "Q3", "Q4"],
            "device_list": [[0, 1], [1, 1], [1, 2], [0, 2]],
            "place_list": ["R", "R/A", "R/A", "R"],
            "total_devices": 4,
        }

    def test_a_family_with_no_place_list_has_none(self, placed):
        assert placed.get_device_info("R", "K") == {
            "common_names": None,
            "device_list": [[1, 1], [2, 1], [1, 2]],
            "place_list": None,
            "total_devices": 3,
        }
