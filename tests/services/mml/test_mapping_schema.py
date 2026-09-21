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
    FamilyJudgments,
    Field,
    FieldAnswer,
    Mapping,
    MappingError,
    OwnerMap,
    System,
    judgment_key,
    parse_mapping,
)
from osprey.services.mml.mapping.schema import (
    UNIT_CLASSES,
    KickAnswer,
    MonitorAnswer,
    OwnerAnswer,
    StrengthAnswer,
    VAFamily,
    VASlot,
    VirtualAccelerator,
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
        "judgments": {
            "DCCT": {
                "rows_beyond_devices": {
                    "Monitor": {
                        "Z:C03-BI{DCCT:1}Lifetime-I": {"field": "Lifetime"},
                        "Z:C03-BI{DCCT:1}I:Total-I": "drop",
                        "Z:C03-BI{DCCT:1}I:Spare-I": None,
                    },
                    "Setpoint": {"Z:C03-BI{DCCT:1}Ref-SP": "device"},
                },
            },
            "TUNE": {"unbound_devices": {3: "drop", 4: "keep", 5: None}},
            "SM1": {"shared_pvs": None},
            "SM2": {"shared_pvs": "keep_all"},
            "SM3": {"shared_pvs": {2: 2, 5: "keep_all"}},
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


class TestJudgments:
    """The judgments block round-trips every accepted answer spelling."""

    def test_the_block_is_optional(self):
        """A mapping without the block parses with no judgments."""
        data = _document()
        del data["judgments"]
        assert parse_mapping(data).judgments == {}

    def test_the_block_is_accepted_anywhere(self):
        """judgments may sit among the top-level keys in any position."""
        data = _document()
        first = {"judgments": data.pop("judgments"), **data}
        assert next(iter(first)) == "judgments"
        assert parse_mapping(first).judgments == parse_mapping(_document()).judgments

    def test_rows_beyond_devices(self):
        """Each row answer keeps its field, its signal and its spelling."""
        fam = parse_mapping(_document()).judgments["DCCT"]
        assert fam == FamilyJudgments(
            rows_beyond={
                "Monitor": {
                    "Z:C03-BI{DCCT:1}Lifetime-I": FieldAnswer("Lifetime"),
                    "Z:C03-BI{DCCT:1}I:Total-I": "drop",
                    "Z:C03-BI{DCCT:1}I:Spare-I": None,
                },
                "Setpoint": {"Z:C03-BI{DCCT:1}Ref-SP": "device"},
            },
            unbound_devices={},
            shared_pvs=None,
            shared_pvs_present=False,
        )

    def test_unbound_devices_are_keyed_by_integer_ordinal(self):
        """YAML integer keys stay integers, and every word round-trips."""
        fam = parse_mapping(_document()).judgments["TUNE"]
        assert fam == FamilyJudgments(unbound_devices={3: "drop", 4: "keep", 5: None})

    def test_shared_pvs_null_slot(self):
        """A null shared_pvs slot is present and undecided."""
        fam = parse_mapping(_document()).judgments["SM1"]
        assert fam.shared_pvs is None
        assert fam.shared_pvs_present is True

    def test_shared_pvs_absent_slot(self):
        """A family with no shared_pvs key has no slot at all."""
        fam = parse_mapping(_document()).judgments["DCCT"]
        assert fam.shared_pvs is None
        assert fam.shared_pvs_present is False

    def test_shared_pvs_keep_all(self):
        """keep_all is carried as the word itself."""
        assert parse_mapping(_document()).judgments["SM2"].shared_pvs == "keep_all"

    def test_shared_pvs_owner_map(self):
        """An owner map keeps integer group keys and both owner spellings."""
        fam = parse_mapping(_document()).judgments["SM3"]
        assert fam.shared_pvs == OwnerMap({2: 2, 5: "keep_all"})
        assert fam.shared_pvs_present is True

    def test_families_keep_document_order(self):
        """The block preserves document order, like every other block."""
        assert list(parse_mapping(_document()).judgments) == ["DCCT", "TUNE", "SM1", "SM2", "SM3"]

    def test_answers_carry_no_provenance(self):
        """Judgment slots add nothing to the provenance walk."""
        data = _document()
        with_block = list(parse_mapping(data).provenance_slots())
        del data["judgments"]
        assert with_block == list(parse_mapping(data).provenance_slots())

    def test_judgments_are_frozen(self):
        """Judgment value objects refuse attribute assignment."""
        m = parse_mapping(_document())
        for obj in (m.judgments["SM1"], m.judgments["SM3"].shared_pvs):
            with pytest.raises(dataclasses.FrozenInstanceError):
                obj.shared_pvs = "keep_all"  # type: ignore[misc]


class TestJudgmentKey:
    """judgment_key renders the document path of one judgment slot."""

    def test_row_beyond_devices(self):
        """A signal goes in square brackets, so its dots stay unambiguous."""
        key = judgment_key("DCCT", "rows_beyond_devices", "Monitor", "SR:C03-BI{DCCT:1}Lifetime-I")
        assert key == "judgments.DCCT.rows_beyond_devices.Monitor[SR:C03-BI{DCCT:1}Lifetime-I]"

    def test_unbound_device(self):
        """An unbound device is named by its ordinal."""
        key = judgment_key("TUNE", "unbound_devices", ordinal=3)
        assert key == "judgments.TUNE.unbound_devices.3"

    def test_shared_pvs_slot_and_group(self):
        """The shared_pvs slot and one of its groups."""
        assert judgment_key("SM1", "shared_pvs") == "judgments.SM1.shared_pvs"
        assert judgment_key("SM1", "shared_pvs", ordinal=2) == "judgments.SM1.shared_pvs.2"

    def test_family_alone(self):
        """A family entry is named without a kind."""
        assert judgment_key("SM1") == "judgments.SM1"


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


class TestMalformedJudgments:
    """Every malformed judgment spelling is refused at its own key."""

    def test_block_not_a_dict(self):
        """The block itself is a mapping of families."""
        data = _document()
        data["judgments"] = ["DCCT"]
        _raises(data, "judgments")

    def test_non_string_family_key(self):
        """A family key YAML parsed to a non-string is refused."""
        data = _document()
        data["judgments"][3] = data["judgments"].pop("SM1")
        _raises(data, "judgments.3")

    def test_family_entry_not_a_dict(self):
        """A family entry holds its kinds, never an answer of its own."""
        data = _document()
        data["judgments"]["SM1"] = "keep_all"
        _raises(data, "judgments.SM1")

    def test_unknown_kind(self):
        """A misspelt kind is refused, not ignored."""
        data = _document()
        data["judgments"]["TUNE"]["unbound_device"] = {3: "drop"}
        _raises(data, "judgments.TUNE.unbound_device")

    def test_rows_beyond_devices_not_a_dict(self):
        """rows_beyond_devices is keyed by field."""
        data = _document()
        data["judgments"]["DCCT"]["rows_beyond_devices"] = ["Monitor"]
        _raises(data, "judgments.DCCT.rows_beyond_devices")

    def test_field_entry_not_a_dict(self):
        """A field entry is keyed by signal."""
        data = _document()
        data["judgments"]["DCCT"]["rows_beyond_devices"]["Setpoint"] = "drop"
        _raises(data, "judgments.DCCT.rows_beyond_devices.Setpoint")

    def test_non_string_signal(self):
        """A signal key YAML parsed to a non-string is refused."""
        data = _document()
        data["judgments"]["DCCT"]["rows_beyond_devices"]["Setpoint"] = {7: "drop"}
        _raises(data, "judgments.DCCT.rows_beyond_devices.Setpoint[7]")

    @pytest.mark.parametrize("answer", ["keep", "keep_all", True, 3, ["drop"]])
    def test_row_answer_vocabulary(self, answer):
        """A row answer is drop, device, a field: entry or null."""
        data = _document()
        rows = data["judgments"]["DCCT"]["rows_beyond_devices"]
        rows["Setpoint"]["Z:C03-BI{DCCT:1}Ref-SP"] = answer
        _raises(data, "judgments.DCCT.rows_beyond_devices.Setpoint[Z:C03-BI{DCCT:1}Ref-SP]")

    def test_field_answer_without_a_name(self):
        """A field: entry names the field it creates."""
        data = _document()
        rows = data["judgments"]["DCCT"]["rows_beyond_devices"]
        rows["Setpoint"]["Z:C03-BI{DCCT:1}Ref-SP"] = {}
        _raises(data, "judgments.DCCT.rows_beyond_devices.Setpoint[Z:C03-BI{DCCT:1}Ref-SP].field")

    def test_field_answer_name_not_a_string(self):
        """A field: name is a string."""
        data = _document()
        rows = data["judgments"]["DCCT"]["rows_beyond_devices"]
        rows["Setpoint"]["Z:C03-BI{DCCT:1}Ref-SP"] = {"field": None}
        _raises(data, "judgments.DCCT.rows_beyond_devices.Setpoint[Z:C03-BI{DCCT:1}Ref-SP].field")

    def test_field_answer_stray_key(self):
        """A field: entry carries nothing else."""
        data = _document()
        rows = data["judgments"]["DCCT"]["rows_beyond_devices"]
        rows["Setpoint"]["Z:C03-BI{DCCT:1}Ref-SP"] = {"field": "Ref", "provenance": "stated"}
        _raises(
            data,
            "judgments.DCCT.rows_beyond_devices.Setpoint[Z:C03-BI{DCCT:1}Ref-SP].provenance",
        )

    def test_unbound_devices_not_a_dict(self):
        """unbound_devices is keyed by ordinal."""
        data = _document()
        data["judgments"]["TUNE"]["unbound_devices"] = [3]
        _raises(data, "judgments.TUNE.unbound_devices")

    @pytest.mark.parametrize("ordinal", ["3", 3.0, True])
    def test_unbound_key_is_an_integer(self, ordinal):
        """An ordinal is a YAML integer, never a bool, float or string."""
        data = _document()
        data["judgments"]["TUNE"]["unbound_devices"] = {ordinal: "drop"}
        _raises(data, f"judgments.TUNE.unbound_devices.{ordinal}")

    @pytest.mark.parametrize("answer", ["device", "keep_all", 1, {"field": "X"}])
    def test_unbound_answer_vocabulary(self, answer):
        """An unbound device is answered drop, keep or null."""
        data = _document()
        data["judgments"]["TUNE"]["unbound_devices"][3] = answer
        _raises(data, "judgments.TUNE.unbound_devices.3")

    @pytest.mark.parametrize("answer", ["keep", "drop", 2, True, ["keep_all"]])
    def test_shared_pvs_vocabulary(self, answer):
        """shared_pvs is keep_all, an owner map or null."""
        data = _document()
        data["judgments"]["SM2"]["shared_pvs"] = answer
        _raises(data, "judgments.SM2.shared_pvs")

    @pytest.mark.parametrize("group", ["2", 2.0, True])
    def test_owner_map_key_is_an_integer(self, group):
        """A group is keyed by the lowest ordinal of its members."""
        data = _document()
        data["judgments"]["SM3"]["shared_pvs"] = {group: 2}
        _raises(data, f"judgments.SM3.shared_pvs.{group}")

    @pytest.mark.parametrize("owner", ["2", "keep", 2.0, True, None, {"field": "X"}])
    def test_owner_map_value(self, owner):
        """An owner is a device ordinal or keep_all."""
        data = _document()
        data["judgments"]["SM3"]["shared_pvs"] = {2: owner}
        _raises(data, "judgments.SM3.shared_pvs.2")

    def test_message_names_the_key(self):
        """str() leads with the bracketed path an operator has to edit."""
        data = _document()
        rows = data["judgments"]["DCCT"]["rows_beyond_devices"]
        rows["Setpoint"]["Z:C03-BI{DCCT:1}Ref-SP"] = "keep"
        err = _raises(data, "judgments.DCCT.rows_beyond_devices.Setpoint[Z:C03-BI{DCCT:1}Ref-SP]")
        assert str(err).startswith(
            "judgments.DCCT.rows_beyond_devices.Setpoint[Z:C03-BI{DCCT:1}Ref-SP]: "
        )


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


def _va_document() -> dict:
    """A document whose virtual-accelerator block spells every accepted shape."""
    data = _document()
    data["virtual_accelerator"] = {
        "system": "SR",
        "families": {
            "QF": {
                "verdict": "couple",
                "kind": "strength",
                "element_field": "PolynomB[1]",
                "calibration": "linear",
                "nominal_source": "getpvmodel",
            },
            "HC": {
                "verdict": "couple",
                "kind": "kick",
                "element_field": "KickAngle[0]",
                "calibration": "table",
                "nominal_source": "synthetic",
                "reason": None,
                "slot": None,
            },
            "BDM": {
                "verdict": "latch",
                "reason": "element BDM (BndMPoleSymplectic4) takes no KickAngle",
            },
            "KickerAmp": {
                "verdict": "latch",
                "reason": "KickerAmp is in no ATType table",
                "slot": {
                    "kind": "attype",
                    "question": "What does KickerAmp drive?",
                    "answer": None,
                },
            },
            "VC": {
                "verdict": "latch",
                "reason": "VC and HC resolve to one KickAngle[0]",
                "slot": {
                    "kind": "shared_field",
                    "question": "Which family owns KickAngle[0]?",
                    "answer": "owner:HC",
                },
            },
            "IDGAP": {
                "verdict": "latch",
                "reason": "the Setpoint field carries a SpecialFunctionSet",
                "slot": {
                    "kind": "escape_hatch",
                    "question": "Is the hook safe to ignore?",
                    "answer": "ignore_hook",
                },
            },
        },
    }
    return data


def _va_answer(kind: str, answer: object) -> object:
    """Parse one slot answer of ``kind`` through a whole document."""
    data = _va_document()
    data["virtual_accelerator"]["families"]["QF"]["slot"] = {
        "kind": kind,
        "question": "Which one?",
        "answer": answer,
    }
    block = parse_mapping(data).virtual_accelerator
    assert block is not None
    slot = block.families["QF"].slot
    assert slot is not None
    return slot.answer


def _va_raises(kind: str, answer: object) -> MappingError:
    """Refuse one slot answer of ``kind``, at the answer's own key."""
    data = _va_document()
    data["virtual_accelerator"]["families"]["QF"]["slot"] = {
        "kind": kind,
        "question": "Which one?",
        "answer": answer,
    }
    return _raises(data, "virtual_accelerator.families.QF.slot.answer")


class TestVirtualAcceleratorBlock:
    """The virtual-accelerator block carries one verdict per family."""

    def test_va_block_is_optional(self):
        """A mapping without the block parses with no block at all."""
        assert parse_mapping(_document()).virtual_accelerator is None

    def test_va_block_is_accepted_anywhere(self):
        """virtual_accelerator may sit among the top-level keys in any position."""
        data = _va_document()
        first = {"virtual_accelerator": data.pop("virtual_accelerator"), **data}
        assert next(iter(first)) == "virtual_accelerator"
        assert (
            parse_mapping(first).virtual_accelerator
            == parse_mapping(_va_document()).virtual_accelerator
        )

    def test_va_system_is_carried(self):
        """The block names the system whose export it describes."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.system == "SR"

    def test_va_system_may_be_undecided(self):
        """A facility with several systems leaves the choice to the reviewer."""
        data = _va_document()
        data["virtual_accelerator"]["system"] = None
        block = parse_mapping(data).virtual_accelerator
        assert block is not None
        assert block.system is None

    def test_va_coupled_family(self):
        """A coupled family carries the element it binds and how it converts."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.families["QF"] == VAFamily(
            verdict="couple",
            kind="strength",
            element_field="PolynomB[1]",
            calibration="linear",
            nominal_source="getpvmodel",
        )

    def test_va_latched_family_carries_its_reason(self):
        """A latched family says why, and binds nothing."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.families["BDM"] == VAFamily(
            verdict="latch",
            reason="element BDM (BndMPoleSymplectic4) takes no KickAngle",
        )

    def test_va_explicit_nulls_read_as_absent_keys(self):
        """Spelling a key null says the same as leaving it out."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.families["HC"].reason is None
        assert block.families["HC"].slot is None

    def test_va_families_keep_document_order(self):
        """The block preserves document order, like every other block."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert list(block.families) == ["QF", "HC", "BDM", "KickerAmp", "VC", "IDGAP"]

    def test_va_families_may_be_empty(self):
        """An export whose every family latched still writes the block."""
        data = _va_document()
        data["virtual_accelerator"]["families"] = {}
        assert parse_mapping(data).virtual_accelerator == VirtualAccelerator(
            system="SR", families={}
        )

    def test_va_block_carries_no_provenance(self):
        """Verdicts and answers add nothing to the provenance walk."""
        data = _va_document()
        with_block = list(parse_mapping(data).provenance_slots())
        del data["virtual_accelerator"]
        assert with_block == list(parse_mapping(data).provenance_slots())

    def test_va_input_is_not_mutated(self):
        """Parsing reads the document; it never edits it."""
        data = _va_document()
        before = copy.deepcopy(data)
        parse_mapping(data)
        assert data == before

    def test_va_dataclasses_are_frozen(self):
        """Every virtual-accelerator value object refuses assignment."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        for obj in (block, block.families["QF"], block.families["VC"].slot):
            with pytest.raises(dataclasses.FrozenInstanceError):
                obj.kind = "rf"  # type: ignore[misc, union-attr]


class TestVASlots:
    """A slot is the one question a rule could not answer, and its answer."""

    def test_va_slot_is_absent_where_the_rules_decided(self):
        """A family the rules settled carries no slot."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.families["BDM"].slot is None

    def test_va_open_slot_keeps_its_question(self):
        """An unanswered slot is present, spelled, and undecided."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.families["KickerAmp"].slot == VASlot(
            kind="attype", question="What does KickerAmp drive?", answer=None
        )

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("latch", "latch"),
            ("energy", "energy"),
            ("rf", "rf"),
            ("strength:PolynomB[1]", StrengthAnswer(attribute="PolynomB", index=1)),
            ("strength:PolynomA[0]", StrengthAnswer(attribute="PolynomA", index=0)),
            ("strength:PolynomB[12]", StrengthAnswer(attribute="PolynomB", index=12)),
            ("kick:0", KickAnswer(plane=0)),
            ("kick:1", KickAnswer(plane=1)),
            ("monitor:x", MonitorAnswer(plane="x")),
            ("monitor:y", MonitorAnswer(plane="y")),
            (None, None),
        ],
    )
    def test_va_attype_answers(self, answer, expected):
        """Every attype spelling parses to the decision it states."""
        assert _va_answer("attype", answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("latch", "latch"),
            ("owner:HC", OwnerAnswer(family="HC")),
            ("owner:BPM.x", OwnerAnswer(family="BPM.x")),
            (None, None),
        ],
    )
    def test_va_shared_field_answers(self, answer, expected):
        """A shared field is owned by one family, or neither couples."""
        assert _va_answer("shared_field", answer) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [("latch", "latch"), ("ignore_hook", "ignore_hook"), (None, None)],
    )
    def test_va_escape_hatch_answers(self, answer, expected):
        """A hook is either ignored or the family latches."""
        assert _va_answer("escape_hatch", answer) == expected

    def test_va_answers_are_typed_per_slot_kind(self):
        """The answered slots of the document carry their parsed values."""
        block = parse_mapping(_va_document()).virtual_accelerator
        assert block is not None
        assert block.families["VC"].slot == VASlot(
            kind="shared_field",
            question="Which family owns KickAngle[0]?",
            answer=OwnerAnswer(family="HC"),
        )
        assert block.families["IDGAP"].slot == VASlot(
            kind="escape_hatch",
            question="Is the hook safe to ignore?",
            answer="ignore_hook",
        )


class TestMalformedVABlock:
    """Every malformed virtual-accelerator spelling is refused at its own key."""

    def test_va_block_not_a_dict(self):
        """The block holds a system and its families."""
        data = _va_document()
        data["virtual_accelerator"] = ["SR"]
        _raises(data, "virtual_accelerator")

    def test_va_unknown_key(self):
        """A misspelt block key is refused, not ignored."""
        data = _va_document()
        data["virtual_accelerator"]["lattice"] = "quokka.mat"
        _raises(data, "virtual_accelerator.lattice")

    @pytest.mark.parametrize("slot", ["system", "families"])
    def test_va_required_block_key(self, slot):
        """Both block keys are written, even when one is null or empty."""
        data = _va_document()
        del data["virtual_accelerator"][slot]
        _raises(data, f"virtual_accelerator.{slot}")

    def test_va_families_not_a_dict(self):
        """families is keyed by raw family token."""
        data = _va_document()
        data["virtual_accelerator"]["families"] = ["QF"]
        _raises(data, "virtual_accelerator.families")

    def test_va_family_entry_not_a_dict(self):
        """A family entry holds its verdict, never the verdict alone."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["QF"] = "couple"
        _raises(data, "virtual_accelerator.families.QF")

    def test_va_non_string_family_key(self):
        """A family key YAML parsed to a non-string is refused."""
        data = _va_document()
        families = data["virtual_accelerator"]["families"]
        families[3] = families.pop("QF")
        _raises(data, "virtual_accelerator.families.3")

    def test_va_family_unknown_key(self):
        """A misspelt family key is refused, not ignored."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["QF"]["element"] = "QF_1_1"
        _raises(data, "virtual_accelerator.families.QF.element")

    def test_va_verdict_is_required(self):
        """Every family states what the rules reached."""
        data = _va_document()
        del data["virtual_accelerator"]["families"]["BDM"]["verdict"]
        _raises(data, "virtual_accelerator.families.BDM.verdict")

    @pytest.mark.parametrize("verdict", ["couples", "open", "COUPLE", None, True, 1])
    def test_va_verdict_vocabulary(self, verdict):
        """A verdict is couple or latch, and nothing else."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["QF"]["verdict"] = verdict
        err = _raises(data, "virtual_accelerator.families.QF.verdict")
        assert "couple" in err.message
        assert "latch" in err.message

    @pytest.mark.parametrize("kind", ["strenght", "bpm", "Strength", 3, ["kick"]])
    def test_va_kind_vocabulary(self, kind):
        """A kind names one of the element kinds the model can drive."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["QF"]["kind"] = kind
        _raises(data, "virtual_accelerator.families.QF.kind")

    @pytest.mark.parametrize("slot", ["element_field", "calibration", "nominal_source", "reason"])
    def test_va_family_strings(self, slot):
        """The descriptive family keys are strings or null."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["QF"][slot] = 7
        _raises(data, f"virtual_accelerator.families.QF.{slot}")

    def test_va_slot_not_a_dict(self):
        """A slot holds its kind, its question and its answer."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["KickerAmp"]["slot"] = "attype"
        _raises(data, "virtual_accelerator.families.KickerAmp.slot")

    def test_va_slot_unknown_key(self):
        """A misspelt slot key is refused, not ignored."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["KickerAmp"]["slot"]["answers"] = "latch"
        _raises(data, "virtual_accelerator.families.KickerAmp.slot.answers")

    @pytest.mark.parametrize("slot", ["kind", "question", "answer"])
    def test_va_slot_required_key(self, slot):
        """A slot the reviewer can answer spells all three keys."""
        data = _va_document()
        del data["virtual_accelerator"]["families"]["KickerAmp"]["slot"][slot]
        _raises(data, f"virtual_accelerator.families.KickerAmp.slot.{slot}")

    @pytest.mark.parametrize("kind", ["at_type", "shared", "escape", None, 2])
    def test_va_slot_kind_vocabulary(self, kind):
        """A slot is one of the three kinds a reviewer is ever asked."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["KickerAmp"]["slot"]["kind"] = kind
        _raises(data, "virtual_accelerator.families.KickerAmp.slot.kind")

    def test_va_slot_question_is_a_string(self):
        """The question is written out, so the card can ask it."""
        data = _va_document()
        data["virtual_accelerator"]["families"]["KickerAmp"]["slot"]["question"] = None
        _raises(data, "virtual_accelerator.families.KickerAmp.slot.question")

    @pytest.mark.parametrize(
        "answer",
        [
            "quadrupole",
            "strength",
            "strength:PolynomC[1]",
            "strength:PolynomB[]",
            "strength:PolynomB[-1]",
            "strength:PolynomB[1] ",
            "kick",
            "kick:2",
            "kick:x",
            "monitor",
            "monitor:z",
            "owner:HC",
            "ignore_hook",
            3,
            ["latch"],
        ],
    )
    def test_va_attype_answer_vocabulary(self, answer):
        """An attype answer outside the vocabulary is refused by name."""
        err = _va_raises("attype", answer)
        assert "strength:<PolynomB|PolynomA>[<i>]" in err.message

    @pytest.mark.parametrize(
        "answer", ["owner:", "owner", "keep_all", "energy", "rf", "ignore_hook", 3]
    )
    def test_va_shared_field_answer_vocabulary(self, answer):
        """A shared-field answer outside the vocabulary is refused by name."""
        err = _va_raises("shared_field", answer)
        assert "owner:<family>" in err.message

    @pytest.mark.parametrize("answer", ["ignore", "ignore_hooks", "owner:HC", "rf", True, 0])
    def test_va_escape_hatch_answer_vocabulary(self, answer):
        """An escape-hatch answer outside the vocabulary is refused by name."""
        err = _va_raises("escape_hatch", answer)
        assert "ignore_hook" in err.message

    def test_va_message_names_the_answer_and_its_key(self):
        """str() leads with the path, and the message quotes the refused word."""
        err = _va_raises("escape_hatch", "ignore")
        assert str(err).startswith("virtual_accelerator.families.QF.slot.answer: ")
        assert "'ignore'" in err.message


class TestVAUnitClasses:
    """The physics-unit vocabulary a kind's units check reads."""

    def test_va_kick_units_are_the_angle_words(self):
        """A kick is spelled in an angle."""
        assert UNIT_CLASSES["kick"] == frozenset(
            {"rad", "radian", "radians", "mrad", "mradian", "urad"}
        )

    def test_va_monitor_units_are_the_length_words(self):
        """A monitor is spelled in a length."""
        assert UNIT_CLASSES["monitor"] == frozenset({"m", "meter", "meters", "metre", "mm"})

    def test_va_strength_has_no_closed_vocabulary(self):
        """A strength is any other non-empty unit word, so it lists none."""
        assert "strength" not in UNIT_CLASSES

    def test_va_unit_classes_are_disjoint(self):
        """No unit word belongs to two kinds, so a match names one kind."""
        assert not UNIT_CLASSES["kick"] & UNIT_CLASSES["monitor"]

    def test_va_unit_words_are_lowercase(self):
        """The words are folded, so a caller compares one spelling."""
        for words in UNIT_CLASSES.values():
            assert all(word == word.lower() and word for word in words)
