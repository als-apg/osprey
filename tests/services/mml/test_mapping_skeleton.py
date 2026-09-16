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
from osprey.services.mml.mapping.skeleton import build_skeleton, dump_yaml

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

    def test_no_branches_key(self):
        """branches is omitted, never written as null."""
        assert "branches" not in _skeleton()

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
