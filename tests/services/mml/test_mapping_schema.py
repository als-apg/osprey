"""Structural parsing of a ``mapping.yaml`` document into frozen dataclasses.

``parse_mapping`` checks shape only: required keys, value types, and unknown
keys. Semantic rules (nulls in the null domain, PN_LOCAL, permutations) belong
to the checker, so every test here that passes a ``None`` expects it to parse.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from osprey.services.mml.mapping import (
    Branch,
    Direction,
    Facility,
    Family,
    Field,
    Mapping,
    MappingError,
    System,
    parse_mapping,
)


def _document() -> dict:
    """A complete mapping document with invented names."""
    return {
        "facility": {
            "token": "Zephyr",
            "title": "ZEPHYR",
            "description": "A synthetic ring.",
            "provenance": "derived",
        },
        "systems": {
            "SR": {"name": "SR", "description": "Storage ring.", "provenance": "stated"},
            "BTS": {"name": "BTS", "description": None, "provenance": "derived"},
        },
        "section_order": ["BTS", "SR"],
        "branches": {"Kicker": {"parent": "Magnet", "description": "Fast kickers."}},
        "families": {
            "BPMx": {
                "branch": "Monitor",
                "class": "BPM",
                "aliases": ["BPMx"],
                "description": "Horizontal BPMs.",
                "provenance": "stated",
                "channels": 4,
                "fields": {
                    "Monitor": {"description": "Position.", "provenance": "derived"},
                },
            },
            "bpmx": {
                "rename": "bpmx_b",
                "class": "HCorrector",
                "aliases": ["bpmx"],
                "description": None,
                "provenance": "derived",
                "channels": 2,
                "fields": {},
            },
            "Empty": {
                "aliases": [],
                "description": "No channels.",
                "provenance": "imported",
                "channels": 0,
                "fields": {},
            },
        },
        "directions": {
            "BPMx.Monitor": {"direction": "read", "provenance": "derived", "override": False},
            "bpmx.Setpoint": {"direction": None, "provenance": "stated", "override": True},
        },
    }


def _raises(data: dict, key: str) -> MappingError:
    with pytest.raises(MappingError) as info:
        parse_mapping(data)
    assert info.value.key == key, str(info.value)
    return info.value


class TestHappyPath:
    """A well-formed document becomes the expected dataclasses."""

    def test_returns_a_mapping(self):
        """The top-level result is a Mapping."""
        assert isinstance(parse_mapping(_document()), Mapping)

    def test_facility(self):
        """Facility slots are carried verbatim."""
        m = parse_mapping(_document())
        assert m.facility == Facility("Zephyr", "ZEPHYR", "A synthetic ring.", "derived")

    def test_systems_keep_raw_token_and_order(self):
        """Systems are keyed by raw token in document order and carry it."""
        m = parse_mapping(_document())
        assert list(m.systems) == ["SR", "BTS"]
        assert m.systems["BTS"] == System("BTS", "BTS", None, "derived")

    def test_section_order_is_a_tuple(self):
        """section_order is an ordered, immutable sequence."""
        assert parse_mapping(_document()).section_order == ("BTS", "SR")

    def test_branches(self):
        """Declared branches carry name, parent and description."""
        m = parse_mapping(_document())
        assert m.branches == {"Kicker": Branch("Kicker", "Magnet", "Fast kickers.")}

    def test_branches_are_optional(self):
        """An absent branches block parses as empty."""
        data = _document()
        del data["branches"]
        assert parse_mapping(data).branches == {}

    def test_family_with_new_class(self):
        """A family with branch and class parses every slot."""
        fam = parse_mapping(_document()).families["BPMx"]
        assert fam == Family(
            raw="BPMx",
            rename=None,
            branch="Monitor",
            class_="BPM",
            aliases=("BPMx",),
            description="Horizontal BPMs.",
            provenance="stated",
            channels=4,
            fields={"Monitor": Field("Position.", "derived")},
        )

    def test_packaged_class_without_branch(self):
        """A packaged class may omit branch, which parses as None."""
        fam = parse_mapping(_document()).families["bpmx"]
        assert fam.branch is None
        assert fam.class_ == "HCorrector"
        assert fam.rename == "bpmx_b"

    def test_zero_channel_family_without_branch_or_class(self):
        """A zero-channel family with neither key parses with both None."""
        fam = parse_mapping(_document()).families["Empty"]
        assert fam.branch is None and fam.class_ is None
        assert fam.channels == 0

    def test_explicit_null_branch_class_rename(self):
        """Explicit nulls for the optional family keys mean absent."""
        data = _document()
        data["families"]["Empty"].update(branch=None, **{"class": None}, rename=None)
        fam = parse_mapping(data).families["Empty"]
        assert (fam.rename, fam.branch, fam.class_) == (None, None, None)

    def test_directions(self):
        """Directions are keyed by the raw dotted key and keep every slot."""
        m = parse_mapping(_document())
        assert m.directions["BPMx.Monitor"] == Direction("read", "derived", False)
        assert m.directions["bpmx.Setpoint"] == Direction(None, "stated", True)

    def test_override_defaults_to_false(self):
        """An omitted override means no override."""
        data = _document()
        del data["directions"]["BPMx.Monitor"]["override"]
        assert parse_mapping(data).directions["BPMx.Monitor"].override is False

    def test_nulls_in_null_domain_parse(self):
        """Nulls the checker rejects are structurally valid."""
        data = _document()
        data["facility"].update(token=None, title=None, description=None)
        data["families"]["BPMx"].update(description=None, branch=None, **{"class": None})
        data["families"]["BPMx"]["fields"]["Monitor"]["description"] = None
        m = parse_mapping(data)
        assert m.facility.token is None
        assert m.families["BPMx"].class_ is None

    def test_input_is_not_mutated(self):
        """Parsing leaves the caller's dict untouched."""
        data = _document()
        before = copy.deepcopy(data)
        parse_mapping(data)
        assert data == before

    def test_dataclasses_are_frozen(self):
        """Every value object refuses attribute assignment."""
        m = parse_mapping(_document())
        for obj in (
            m,
            m.facility,
            m.systems["SR"],
            m.families["BPMx"],
            m.directions["BPMx.Monitor"],
            m.branches["Kicker"],
            m.families["BPMx"].fields["Monitor"],
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                obj.description = "x"  # type: ignore[misc]


class TestMapped:
    """Mapping.mapped applies rename."""

    def test_rename_applies(self):
        """A renamed family maps to its rename."""
        assert parse_mapping(_document()).mapped("bpmx") == "bpmx_b"

    def test_no_rename_is_identity(self):
        """A family without rename maps to its raw token."""
        assert parse_mapping(_document()).mapped("BPMx") == "BPMx"

    def test_unknown_family_raises_key_error(self):
        """A family absent from families: is a lookup miss."""
        with pytest.raises(KeyError):
            parse_mapping(_document()).mapped("Nope")


class TestProvenanceSlots:
    """provenance_slots yields every provenance with its key path."""

    def test_every_slot_in_order(self):
        """Facility, systems, families, fields, then directions."""
        slots = list(parse_mapping(_document()).provenance_slots())
        assert slots == [
            ("facility", "derived"),
            ("systems.SR", "stated"),
            ("systems.BTS", "derived"),
            ("families.BPMx", "stated"),
            ("families.BPMx.fields.Monitor", "derived"),
            ("families.bpmx", "derived"),
            ("families.Empty", "imported"),
            ("directions.BPMx.Monitor", "derived"),
            ("directions.bpmx.Setpoint", "stated"),
        ]


class TestMissingKeys:
    """Required keys are named by path when absent."""

    @pytest.mark.parametrize(
        "top", ["facility", "systems", "section_order", "families", "directions"]
    )
    def test_top_level(self, top):
        """Every required top-level block must be present."""
        data = _document()
        del data[top]
        _raises(data, top)

    @pytest.mark.parametrize("slot", ["token", "title", "description", "provenance"])
    def test_facility_slot(self, slot):
        """Facility requires all four slots, even when null."""
        data = _document()
        del data["facility"][slot]
        _raises(data, f"facility.{slot}")

    def test_system_name(self):
        """A system without name is refused at its path."""
        data = _document()
        del data["systems"]["SR"]["name"]
        _raises(data, "systems.SR.name")

    @pytest.mark.parametrize("slot", ["aliases", "description", "provenance", "channels", "fields"])
    def test_family_slot(self, slot):
        """Every non-optional family slot is required."""
        data = _document()
        del data["families"]["BPMx"][slot]
        _raises(data, f"families.BPMx.{slot}")

    def test_field_provenance(self):
        """A field without provenance is refused at its path."""
        data = _document()
        del data["families"]["BPMx"]["fields"]["Monitor"]["provenance"]
        _raises(data, "families.BPMx.fields.Monitor.provenance")

    def test_direction_slot(self):
        """A direction without direction is refused at its path."""
        data = _document()
        del data["directions"]["BPMx.Monitor"]["direction"]
        _raises(data, "directions.BPMx.Monitor.direction")

    def test_branch_parent(self):
        """A branch without parent is refused at its path."""
        data = _document()
        del data["branches"]["Kicker"]["parent"]
        _raises(data, "branches.Kicker.parent")

    def test_message_names_the_key(self):
        """str() leads with the key path an operator has to edit."""
        data = _document()
        del data["systems"]["SR"]["name"]
        err = _raises(data, "systems.SR.name")
        assert str(err).startswith("systems.SR.name: ")


class TestUnknownKeys:
    """Unknown keys are refused with their full path."""

    def test_top_level(self):
        """A stray top-level key is refused."""
        data = _document()
        data["extras"] = {}
        _raises(data, "extras")

    def test_facility(self):
        """A stray facility key is refused."""
        data = _document()
        data["facility"]["prefix"] = "Z"
        _raises(data, "facility.prefix")

    def test_family(self):
        """A misspelt family key is refused, not ignored."""
        data = _document()
        data["families"]["BPMx"]["clas"] = "BPM"
        _raises(data, "families.BPMx.clas")

    def test_class_underscore_is_not_the_yaml_key(self):
        """The dataclass spelling class_ is not accepted in the document."""
        data = _document()
        data["families"]["Empty"]["class_"] = "X"
        _raises(data, "families.Empty.class_")

    def test_field(self):
        """A stray field key is refused."""
        data = _document()
        data["families"]["BPMx"]["fields"]["Monitor"]["direction"] = "read"
        _raises(data, "families.BPMx.fields.Monitor.direction")

    def test_direction(self):
        """A stray direction key is refused."""
        data = _document()
        data["directions"]["BPMx.Monitor"]["vote"] = "read"
        _raises(data, "directions.BPMx.Monitor.vote")

    def test_system_and_branch(self):
        """Stray system and branch keys are refused."""
        data = _document()
        data["systems"]["SR"]["token"] = "x"
        _raises(data, "systems.SR.token")
        data = _document()
        data["branches"]["Kicker"]["is_a"] = "x"
        _raises(data, "branches.Kicker.is_a")


class TestWrongTypes:
    """Values of the wrong type are refused at their path."""

    def test_document_not_a_dict(self):
        """The document itself must be a mapping."""
        with pytest.raises(MappingError):
            parse_mapping(["not", "a", "dict"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("block", ["facility", "systems", "families", "directions"])
    def test_block_not_a_dict(self, block):
        """Mapping blocks must be dicts."""
        data = _document()
        data[block] = []
        _raises(data, block)

    def test_branches_null(self):
        """A present branches block must be a dict."""
        data = _document()
        data["branches"] = None
        _raises(data, "branches")

    def test_section_order_not_a_list(self):
        """section_order must be a list."""
        data = _document()
        data["section_order"] = "SR"
        _raises(data, "section_order")

    def test_section_order_item(self):
        """section_order items must be strings, named by index."""
        data = _document()
        data["section_order"] = ["SR", 3]
        _raises(data, "section_order[1]")

    def test_non_string_system_key(self):
        """A key YAML parsed to a non-string is refused."""
        data = _document()
        data["systems"][1] = data["systems"].pop("SR")
        _raises(data, "systems.1")

    def test_system_entry_not_a_dict(self):
        """A system entry must be a dict."""
        data = _document()
        data["systems"]["SR"] = "SR"
        _raises(data, "systems.SR")

    def test_system_name_null(self):
        """A system name is always a string."""
        data = _document()
        data["systems"]["SR"]["name"] = None
        _raises(data, "systems.SR.name")

    def test_description_not_string(self):
        """A description is a string or null."""
        data = _document()
        data["facility"]["description"] = 5
        _raises(data, "facility.description")

    def test_provenance_not_string(self):
        """A provenance is a string."""
        data = _document()
        data["families"]["BPMx"]["provenance"] = ["stated"]
        _raises(data, "families.BPMx.provenance")

    @pytest.mark.parametrize("value", ["4", 4.0, True, None])
    def test_channels_not_int(self, value):
        """channels is an integer, never a bool, float or string."""
        data = _document()
        data["families"]["BPMx"]["channels"] = value
        _raises(data, "families.BPMx.channels")

    def test_channels_negative(self):
        """A negative channel count is not a count."""
        data = _document()
        data["families"]["BPMx"]["channels"] = -1
        _raises(data, "families.BPMx.channels")

    def test_aliases_not_a_list(self):
        """aliases is a list."""
        data = _document()
        data["families"]["BPMx"]["aliases"] = "BPMx"
        _raises(data, "families.BPMx.aliases")

    def test_alias_item(self):
        """aliases items are strings, named by index."""
        data = _document()
        data["families"]["BPMx"]["aliases"] = ["BPMx", None]
        _raises(data, "families.BPMx.aliases[1]")

    def test_class_not_string(self):
        """class is a string or null."""
        data = _document()
        data["families"]["BPMx"]["class"] = 7
        _raises(data, "families.BPMx.class")

    def test_fields_not_a_dict(self):
        """fields is a dict."""
        data = _document()
        data["families"]["BPMx"]["fields"] = ["Monitor"]
        _raises(data, "families.BPMx.fields")

    def test_direction_value(self):
        """direction is read, write or null."""
        data = _document()
        data["directions"]["BPMx.Monitor"]["direction"] = "both"
        _raises(data, "directions.BPMx.Monitor.direction")

    def test_override_not_bool(self):
        """override is a boolean."""
        data = _document()
        data["directions"]["BPMx.Monitor"]["override"] = "yes"
        _raises(data, "directions.BPMx.Monitor.override")

    @pytest.mark.parametrize("key", ["BPMx", ".Monitor", "BPMx.", "a.b.c"])
    def test_direction_key_shape(self, key):
        """A directions key is exactly <family>.<field>."""
        data = _document()
        data["directions"][key] = data["directions"].pop("BPMx.Monitor")
        _raises(data, f"directions.{key}")

    def test_branch_parent_not_string(self):
        """A branch parent is a string."""
        data = _document()
        data["branches"]["Kicker"]["parent"] = None
        _raises(data, "branches.Kicker.parent")


class TestMappingError:
    """MappingError carries its key and message."""

    def test_attributes_and_str(self):
        """key and message are kept apart; str joins them."""
        err = MappingError("facility.token", "must be a string or null")
        assert err.key == "facility.token"
        assert err.message == "must be a string or null"
        assert str(err) == "facility.token: must be a string or null"

    def test_is_a_value_error(self):
        """Callers treating malformed input as ValueError keep working."""
        assert issubclass(MappingError, ValueError)
