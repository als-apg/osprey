"""Tests for the ``mapping.yaml`` skeleton ``map --init`` writes.

``build_skeleton`` turns a merged export, its AD blocks and the direction votes
into the nested mapping document: every semantic slot is pre-filled from the
export where a fact exists (``derived`` or ``imported``) and left ``None`` where
it does not, so the agent edits one stable file. The output must parse with
``parse_mapping`` before anyone has touched it.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from osprey.services.mml.directions import vote_directions
from osprey.services.mml.mapping.branches import is_pn_local
from osprey.services.mml.mapping.schema import parse_mapping
from osprey.services.mml.mapping.skeleton import (
    build_skeleton,
    count_judgment_slots,
    dump_yaml,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"


def _field(*channels, member_of=None, **extra) -> dict:
    body: dict = {"ChannelNames": list(channels)}
    if member_of is not None:
        body["MemberOf"] = member_of
    body.update(extra)
    return body


def _bpm(prefix: str = "SR") -> dict:
    return {
        "DeviceList": [[1, 1], [1, 2]],
        "Monitor": _field(f"{prefix}:BPM1:X", f"{prefix}:BPM2:X", member_of=["BPM", "Monitor"]),
        "Setpoint": _field(f"{prefix}:BPM1:SP", f"{prefix}:BPM2:SP"),
        "Gain": _field(f"{prefix}:BPM1:G", f"{prefix}:BPM2:G"),
    }


def _ao() -> dict:
    return {
        "_import_order": ["SR", "BR"],
        "_exports": {"SR": {}, "BR": {}},
        "BR": {"BPMx": _bpm("BR"), "DCCT": {"DCCT": _field("BR:DCCT")}},
        "SR": {
            "_description": "The storage ring.",
            "BPMx": _bpm("SR"),
            "Empty": {"DeviceList": [], "Monitor": _field()},
            "HCM": {
                "_description": "Horizontal correctors.",
                "DeviceList": [[1, 1], [1, 2]],
                "Setpoint": _field("SR:HCM:SP", Description="Corrector current."),
            },
        },
    }


def _rows(n: int) -> list[list[int]]:
    return [[1, i + 1] for i in range(n)]


def _pending_ao() -> dict:
    """An export whose families pend one judgment of every kind.

    ``BEND`` lives in both systems with different device counts, the shape
    NSLS-II's four LTB bends and sixty storage-ring bends have: it pends rows
    beyond its devices in SR and an unbound device in BR, and must still come
    out as one entry. ``SQ`` pends the same signal in both systems plus one of
    its own, ``SM1`` shares a PV between its two devices and ``HCM`` pends
    nothing.
    """
    return {
        "_import_order": ["SR", "BR"],
        "SR": {
            "BEND": {
                "DeviceList": _rows(2),
                "Monitor": _field("SR:BEND1:I", "SR:BEND2:I", "SR:BEND:Spare-I"),
                "Setpoint": _field("SR:BEND1:SP", "SR:BEND2:SP", "SR:BEND:Extra-SP"),
            },
            "SQ": {"DeviceList": _rows(1), "Monitor": _field("SQ1:I", "SQ:Shared-I")},
            "SM1": {"DeviceList": _rows(2), "Monitor": _field("SR:SM1:I", "SR:SM1:I")},
        },
        "BR": {
            "BEND": {"DeviceList": _rows(3), "Monitor": _field("BR:BEND1:I", "BR:BEND2:I")},
            "SQ": {
                "DeviceList": _rows(1),
                "Monitor": _field("BR:SQ1:I", "SQ:Shared-I", "BR:SQ:Own-I"),
            },
            "HCM": {"DeviceList": _rows(2), "Setpoint": _field("BR:HCM1:SP", "BR:HCM2:SP")},
        },
    }


def _ad() -> dict:
    return {
        "SR": {"Machine": "Test Ring-2", "SubMachine": "StorageRing", "OperationalMode": "User"},
        "BR": {"Machine": "Test Ring-2", "SubMachine": "Booster"},
    }


def _skeleton(ao=None, ad="default") -> dict:
    ao = _ao() if ao is None else ao
    ad = _ad() if ad == "default" else ad
    return build_skeleton(ao, ad, vote_directions(ao))


class TestFacility:
    """The facility block comes from ``AD.Machine``."""

    def test_token_is_machine_folded_to_pn_local(self):
        """A Machine with spaces and dashes folds to a PN_LOCAL token."""
        facility = _skeleton()["facility"]
        assert is_pn_local(facility["token"])
        assert facility["token"] == "Test_Ring_2"
        assert facility["title"] == "Test Ring-2"
        assert facility["provenance"] == "derived"
        assert "Test Ring-2" in facility["description"]

    def test_leading_digit_is_folded(self):
        """A Machine starting with a digit still yields a PN_LOCAL token."""
        facility = _skeleton(ad={"SR": {"Machine": "3GeV"}})["facility"]
        assert is_pn_local(facility["token"])

    def test_no_ad_leaves_token_and_title_null(self):
        """Without AD there is no facility fact to derive from."""
        facility = _skeleton(ad=None)["facility"]
        assert facility["token"] is None
        assert facility["title"] is None
        assert facility["provenance"] == "derived"

    def test_empty_ad_behaves_like_no_ad(self):
        """An empty AD dict yields a null token too."""
        assert _skeleton(ad={})["facility"]["token"] is None


class TestSystems:
    """Systems are keyed by raw token and described from AD or ``_description``."""

    def test_imported_description_wins(self):
        """A system-level ``_description`` is carried through as imported."""
        sr = _skeleton()["systems"]["SR"]
        assert sr == {"name": "SR", "description": "The storage ring.", "provenance": "imported"}

    def test_derived_description_from_ad(self):
        """Without ``_description`` the prose comes from AD as derived."""
        br = _skeleton()["systems"]["BR"]
        assert br["name"] == "BR"
        assert br["provenance"] == "derived"
        assert "Booster" in br["description"]
        assert "Test Ring-2" in br["description"]

    def test_operational_mode_in_derived_prose(self):
        """OperationalMode is part of the derived system prose."""
        ao = _ao()
        del ao["SR"]["_description"]
        sr = _skeleton(ao)["systems"]["SR"]
        assert sr["provenance"] == "derived"
        assert "User" in sr["description"]

    def test_systems_follow_import_order(self):
        """Systems are written in import order, not alphabetically."""
        assert list(_skeleton()["systems"]) == ["SR", "BR"]


class TestSectionOrder:
    """``section_order`` lists mapped system names in import order."""

    def test_import_order_kept(self):
        """Two systems imported in non-alphabetical order keep that order."""
        assert _skeleton()["section_order"] == ["SR", "BR"]

    def test_fallback_is_sorted_keys(self):
        """Without ``_import_order`` the order is the sorted system keys."""
        ao = _ao()
        del ao["_import_order"]
        assert _skeleton(ao)["section_order"] == ["BR", "SR"]

    def test_renamed_system_appears_under_its_name(self):
        """section_order is built from ``name``, so a rename is translated."""
        from osprey.services.mml.mapping import skeleton as module

        systems = {"SR": {"name": "StorageRing"}, "BR": {"name": "BR"}}
        assert module.section_order(_ao(), systems) == ["StorageRing", "BR"]

    def test_is_permutation_of_system_names(self):
        """section_order is always a permutation of systems.*.name."""
        data = _skeleton()
        names = sorted(s["name"] for s in data["systems"].values())
        assert sorted(data["section_order"]) == names


class TestFamilies:
    """Families are keyed by raw token and merged across systems."""

    def test_family_order_is_first_encounter(self):
        """Families appear in system import order, then export order."""
        assert list(_skeleton()["families"]) == ["BPMx", "Empty", "HCM", "DCCT"]

    def test_class_prefilled_and_aliases_raw(self):
        """class is pre-filled with the mapped token; aliases keep the raw one."""
        bpm = _skeleton()["families"]["BPMx"]
        assert bpm["class"] == "BPMx"
        assert bpm["branch"] is None
        assert bpm["aliases"] == ["BPMx"]

    def test_channels_summed_across_systems(self):
        """channels counts bindings from every system carrying the family."""
        assert _skeleton()["families"]["BPMx"]["channels"] == 12

    def test_zero_channel_family_has_no_branch_or_class(self):
        """A channels: 0 family carries neither branch nor class."""
        empty = _skeleton()["families"]["Empty"]
        assert empty["channels"] == 0
        assert "branch" not in empty
        assert "class" not in empty

    def test_rename_never_written(self):
        """rename is omitted, never written as null."""
        for family in _skeleton()["families"].values():
            assert "rename" not in family

    def test_imported_family_description(self):
        """A native family description is carried as imported."""
        hcm = _skeleton()["families"]["HCM"]
        assert hcm["description"] == "Horizontal correctors."
        assert hcm["provenance"] == "imported"

    def test_derived_family_description_from_export_facts(self):
        """Without a native description the prose is derived from counts."""
        bpm = _skeleton()["families"]["BPMx"]
        assert bpm["provenance"] == "derived"
        assert "BPMx" in bpm["description"]
        assert "12" in bpm["description"]

    def test_fields_merged_in_order(self):
        """Fields union across systems in encounter order."""
        fields = _skeleton()["families"]["BPMx"]["fields"]
        assert list(fields) == ["Monitor", "Setpoint", "Gain"]
        assert fields["Monitor"]["provenance"] == "derived"
        assert "Monitor" in fields["Monitor"]["description"]

    def test_imported_field_description(self):
        """A native field description is carried as imported."""
        sp = _skeleton()["families"]["HCM"]["fields"]["Setpoint"]
        assert sp == {"description": "Corrector current.", "provenance": "imported"}

    def test_derived_field_prose_names_no_family_token(self):
        """The field sits under its family, so its prose never repeats the raw token."""
        fields = _skeleton()["families"]["BPMx"]["fields"]
        assert fields["Monitor"]["description"] == "Field Monitor: 4 channels via ChannelNames."
        assert "BPMx" not in fields["Gain"]["description"]

    def test_per_device_unit_list_reaches_the_field_prose(self):
        """A ``HWUnits`` list whose non-blank entries agree states that one unit."""
        ao = {
            "SR": {
                "QM": {
                    "DeviceList": [[1, 1], [1, 2], [1, 3]],
                    "Monitor": _field(
                        "SR:QM1:RB", "SR:QM2:RB", "SR:QM3:RB", HWUnits=["Amps", "Amps", ""]
                    ),
                    "Setpoint": _field(
                        "SR:QM1:SP", "SR:QM2:SP", "SR:QM3:SP", HWUnits=["Amps", "V", "V"]
                    ),
                    "Gain": _field("SR:QM1:G", "SR:QM2:G", "SR:QM3:G", HWUnits=[]),
                }
            }
        }
        fields = _skeleton(ao, ad=None)["families"]["QM"]["fields"]
        assert fields["Monitor"]["description"].endswith(", hardware units Amps.")
        assert "hardware units" not in fields["Setpoint"]["description"]
        assert "hardware units" not in fields["Gain"]["description"]


class TestDirections:
    """Every signal group gets a direction slot filled from the vote."""

    def test_every_vote_key_present(self):
        """Direction keys cover every (raw family, field) the voter sees."""
        ao = _ao()
        data = _skeleton(ao)
        expected = {f"{family}.{field}" for family, field in vote_directions(ao)}
        assert set(data["directions"]) == expected

    def test_decided_and_undecided(self):
        """A decided vote fills the slot; an undecided one leaves it null."""
        directions = _skeleton()["directions"]
        assert directions["BPMx.Monitor"] == {
            "direction": "read",
            "provenance": "derived",
            "override": False,
        }
        assert directions["BPMx.Setpoint"]["direction"] == "write"
        assert directions["BPMx.Gain"]["direction"] is None
        assert directions["BPMx.Gain"]["provenance"] == "derived"
        assert directions["BPMx.Gain"]["override"] is False

    def test_missing_vote_is_null(self):
        """A field with no vote entry still gets a null direction slot."""
        data = build_skeleton(_ao(), _ad(), {})
        assert data["directions"]["BPMx.Monitor"]["direction"] is None


class TestDocument:
    """The whole document parses and dumps stably."""

    def test_omitted_keys_are_absent_not_null(self):
        """An optional top-level block the export has nothing for is omitted."""
        document = _skeleton()
        assert "branches" not in document
        assert "judgments" not in document

    def test_round_trips_through_parse_mapping(self):
        """The skeleton parses with the structural schema."""
        mapping = parse_mapping(_skeleton())
        assert mapping.section_order == ("SR", "BR")
        assert mapping.families["Empty"].class_ is None

    def test_dump_yaml_round_trip_and_order(self):
        """dump_yaml keeps insertion order in block style and reloads equal."""
        data = _skeleton()
        text = dump_yaml(data)
        assert yaml.safe_load(text) == data
        assert "{" not in text
        assert list(yaml.safe_load(text)) == list(data)
        assert text.index("facility:") < text.index("systems:") < text.index("directions:")

    def test_deterministic(self):
        """Two builds of the same input dump byte-identically."""
        assert dump_yaml(_skeleton()) == dump_yaml(_skeleton())

    def test_inputs_not_modified(self):
        """build_skeleton does not modify its inputs."""
        ao, ad = _ao(), _ad()
        before = json.dumps([ao, ad], sort_keys=True)
        build_skeleton(ao, ad, vote_directions(ao))
        assert json.dumps([ao, ad], sort_keys=True) == before

    def test_paired_fixture(self):
        """The paired Quokka fixture yields a parseable skeleton."""
        ao_flat = json.loads((FIXTURES / "paired" / "quokka.ring.ao.json").read_text())
        ad_flat = json.loads((FIXTURES / "paired" / "quokka.ring.ad.json").read_text())
        ao = {
            "_import_order": ["RING"],
            "RING": {k: v for k, v in ao_flat.items() if not k.startswith("_")},
        }
        data = build_skeleton(ao, {"RING": ad_flat}, vote_directions(ao))
        mapping = parse_mapping(yaml.safe_load(dump_yaml(data)))
        assert mapping.facility.token == "Quokka"
        assert mapping.section_order == ("RING",)
        assert mapping.families


class TestJudgments:
    """The judgments block asks once per pending slot, unioned across systems."""

    def test_absent_when_nothing_pends(self):
        """An export pending no judgment carries no block at all."""
        assert "judgments" not in _skeleton()
        assert count_judgment_slots(_skeleton()) == 0

    def test_written_after_directions_and_last(self):
        """The block is the document's last key, after directions."""
        document = _skeleton(_pending_ao(), ad=None)
        assert list(document)[-1] == "judgments"
        text = dump_yaml(document)
        assert text.index("directions:") < text.index("judgments:")

    def test_one_entry_per_family_unioned_across_systems(self):
        """A family in two systems gets one entry carrying both systems' kinds."""
        bend = _skeleton(_pending_ao(), ad=None)["judgments"]["BEND"]
        assert list(bend) == ["rows_beyond_devices", "unbound_devices"]
        assert bend["rows_beyond_devices"] == {
            "Monitor": {"SR:BEND:Spare-I": None},
            "Setpoint": {"SR:BEND:Extra-SP": None},
        }
        assert bend["unbound_devices"] == {3: None}

    def test_a_signal_pending_in_two_systems_is_one_slot(self):
        """Signals are a union: the shared one asks once, each own one asks too."""
        judgments = _skeleton(_pending_ao(), ad=None)["judgments"]
        assert judgments["SQ"] == {
            "rows_beyond_devices": {"Monitor": {"SQ:Shared-I": None, "BR:SQ:Own-I": None}}
        }

    def test_shared_pvs_is_one_null_slot_per_family(self):
        """A family with supply groups asks once, however many groups it has."""
        assert _skeleton(_pending_ao(), ad=None)["judgments"]["SM1"] == {"shared_pvs": None}

    def test_family_pending_nothing_has_no_entry(self):
        """A family whose lists are all aligned is absent from the block."""
        judgments = _skeleton(_pending_ao(), ad=None)["judgments"]
        assert "HCM" not in judgments
        assert list(judgments) == ["BEND", "SQ", "SM1"]

    def test_counts_every_null_slot(self):
        """count_judgment_slots counts rows, ordinals and one per shared family."""
        # BEND two rows and one ordinal, SQ two rows, SM1 one supply.
        assert count_judgment_slots(_skeleton(_pending_ao(), ad=None)) == 6

    def test_round_trips_through_parse_mapping(self):
        """The written block parses, with a real int ordinal key."""
        document = _skeleton(_pending_ao(), ad=None)
        mapping = parse_mapping(yaml.safe_load(dump_yaml(document)))
        bend = mapping.judgments["BEND"]
        assert bend.rows_beyond["Monitor"] == {"SR:BEND:Spare-I": None}
        assert bend.unbound_devices == {3: None}
        assert mapping.judgments["SM1"].shared_pvs_present
        assert mapping.judgments["SM1"].shared_pvs is None

    def test_inputs_not_modified(self):
        """Detecting judgments only reads the export."""
        ao = _pending_ao()
        before = json.dumps(ao, sort_keys=True)
        build_skeleton(ao, None, vote_directions(ao))
        assert json.dumps(ao, sort_keys=True) == before
