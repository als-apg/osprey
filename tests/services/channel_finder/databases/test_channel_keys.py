"""The channel keys and the family shape rule every middle-layer reader shares.

``ChannelNames`` and ``TangoNames`` are equal channel keys: the database
loader, the paradigm sniffer and the family shape rule must all accept either,
and a ``ChannelNames``-only database must read exactly as it always did.
"""

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.middle_layer import (
    CHANNEL_KEYS,
    MiddleLayerDatabase,
)
from osprey.services.channel_finder.tools.preview_database import (
    _looks_like_middle_layer,
    _reaches_channel_names,
    is_family_dict,
)


def _write(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "middle_layer.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


class TestChannelKeys:
    """The constant names both keys, in the order readers prefer them."""

    def test_the_keys_are_channel_names_then_tango_names(self):
        """``ChannelNames`` is consulted first, ``TangoNames`` second."""
        assert CHANNEL_KEYS == ("ChannelNames", "TangoNames")


class TestLoaderReadsEveryChannelKey:
    """The middle-layer database extracts, lists and counts under either key."""

    def test_a_channel_names_database_reads_unchanged(self, tmp_path):
        """Padding stripped, bare string wrapped, subfields walked, as before."""
        path = _write(
            tmp_path,
            {
                "RING": {
                    "QF": {
                        "Monitor": {"ChannelNames": [" QF1:I ", "QF2:I", "  "], "Units": "A"},
                        "Setpoint": {"X": {"ChannelNames": "QF:SP"}},
                        "setup": {"DeviceList": [[1, 1], [1, 2]]},
                    }
                }
            },
        )
        db = MiddleLayerDatabase(str(path))
        assert set(db.channel_map) == {"QF1:I", "QF2:I", "QF:SP"}
        assert db.channel_map["QF1:I"]["Units"] == "A"
        assert db.channel_map["QF:SP"]["subfield"] == ["X"]
        assert db.list_channel_names("RING", "QF", "Monitor") == ["QF1:I", "QF2:I"]
        assert db.list_channel_names("RING", "QF", "Setpoint", "X") == ["QF:SP"]
        assert db.count_family_channels("RING", "QF") == 4

    def test_a_tango_names_field_is_extracted(self, tmp_path):
        """A field keyed only by ``TangoNames`` contributes its channels."""
        path = _write(
            tmp_path,
            {"RING": {"KICK": {"Voltage": {"TangoNames": ["ring/kick/1/v", "ring/kick/2/v"]}}}},
        )
        db = MiddleLayerDatabase(str(path))
        assert set(db.channel_map) == {"ring/kick/1/v", "ring/kick/2/v"}
        assert db.channel_map["ring/kick/1/v"]["field"] == "Voltage"

    def test_a_tango_names_field_is_listed(self, tmp_path):
        """``list_channel_names`` answers a ``TangoNames``-only field."""
        path = _write(tmp_path, {"RING": {"KICK": {"Voltage": {"TangoNames": "ring/kick/1/v"}}}})
        db = MiddleLayerDatabase(str(path))
        assert db.list_channel_names("RING", "KICK", "Voltage") == ["ring/kick/1/v"]

    def test_listing_prefers_channel_names_when_both_are_present(self, tmp_path):
        """The first key in ``CHANNEL_KEYS`` order wins a dual-key field."""
        path = _write(
            tmp_path,
            {"RING": {"KICK": {"Voltage": {"ChannelNames": ["K:V"], "TangoNames": ["k/v"]}}}},
        )
        db = MiddleLayerDatabase(str(path))
        assert db.list_channel_names("RING", "KICK", "Voltage") == ["K:V"]

    def test_a_field_with_no_channel_key_still_refuses(self, tmp_path):
        """The listing error survives for a field that carries neither key."""
        path = _write(
            tmp_path,
            {"RING": {"KICK": {"Voltage": {"ChannelNames": ["K:V"]}, "Empty": {"Units": "V"}}}},
        )
        db = MiddleLayerDatabase(str(path))
        with pytest.raises(ValueError, match="RING:KICK:Empty"):
            db.list_channel_names("RING", "KICK", "Empty")

    def test_tango_names_are_counted(self, tmp_path):
        """The family census counts ``TangoNames`` at field and subfield depth."""
        path = _write(
            tmp_path,
            {
                "RING": {
                    "KICK": {
                        "Voltage": {"TangoNames": ["a", "b"]},
                        "Current": {"X": {"TangoNames": "c"}},
                    }
                }
            },
        )
        db = MiddleLayerDatabase(str(path))
        assert db.count_family_channels("RING", "KICK") == 3


class TestReachesChannelNames:
    """The paradigm sniffer bottoms out in any channel key."""

    @pytest.mark.parametrize("key", CHANNEL_KEYS)
    def test_either_key_is_a_leaf(self, key):
        """A list under either channel key reaches the paradigm's leaf."""
        assert _reaches_channel_names({"FAM": {"Field": {key: ["X"]}}}) is True

    def test_a_tango_only_file_looks_like_middle_layer(self):
        """A file whose every field is ``TangoNames`` classifies as middle layer."""
        assert _looks_like_middle_layer({"RING": {"KICK": {"V": {"TangoNames": ["k/v"]}}}})

    def test_metadata_keys_do_not_carry_tango_names(self):
        """A ``_``-prefixed block never classifies, whichever key it holds."""
        assert _looks_like_middle_layer({"_notes": {"TangoNames": ["X"]}}) is False


class TestIsFamilyDict:
    """A family is a dict with a channel-keyed sub-dict or a ``DeviceList``."""

    @pytest.mark.parametrize("key", CHANNEL_KEYS)
    def test_a_channel_keyed_field_makes_a_family(self, key):
        """One field carrying either channel key is enough."""
        assert is_family_dict("QF", {"Monitor": {key: ["QF:I"]}, "FamilyName": "QF"}) is True

    def test_a_device_list_makes_a_family(self):
        """A family with no channel field is still a family when it lists devices."""
        assert is_family_dict("GIRDER", {"FamilyName": "GIRDER", "DeviceList": [[1, 1]]}) is True

    def test_a_device_list_under_setup_makes_a_family(self):
        """Osprey's own dialect keeps the family arrays in ``setup``."""
        assert is_family_dict("BPM", {"setup": {"DeviceList": [[1, 1]]}}) is True
        assert is_family_dict("BPM", {"_setup": {"DeviceList": [[1, 1]]}}) is True

    def test_underscore_names_never_count(self):
        """``_export`` and ``_description`` are metadata whatever they hold."""
        body = {"Monitor": {"ChannelNames": ["X"]}, "DeviceList": [[1, 1]]}
        assert is_family_dict("_export", body) is False
        assert is_family_dict("_description", body) is False

    def test_setup_is_not_a_field(self):
        """A channel key inside ``setup`` does not make its parent a family."""
        assert is_family_dict("X", {"setup": {"ChannelNames": ["X"]}}) is False
        assert is_family_dict("X", {"_setup": {"TangoNames": ["x"]}}) is False

    def test_underscore_sub_dicts_are_not_fields(self):
        """A channel key under a ``_``-prefixed sub-dict does not count either."""
        assert is_family_dict("X", {"_notes": {"ChannelNames": ["X"]}}) is False

    def test_a_system_dict_is_not_a_family(self):
        """Channel keys two levels down describe a system, not a family."""
        assert is_family_dict("RING", {"QF": {"Monitor": {"ChannelNames": ["QF:I"]}}}) is False

    @pytest.mark.parametrize("value", [None, "QF", 3, ["ChannelNames"], {}])
    def test_non_family_values_are_refused(self, value):
        """Scalars, lists and empty dicts are never families."""
        assert is_family_dict("QF", value) is False

    def test_a_bare_field_is_not_a_family(self):
        """A field dict holds the channel key itself, not in a sub-dict."""
        assert is_family_dict("Monitor", {"ChannelNames": ["X"]}) is False
