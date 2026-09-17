"""Tests for the MML import census.

``take_census`` reads a merged, normalised ``ao`` (``{system: {family: body}}``
plus ``_``-prefixed bookkeeping keys) and the merged ``ad`` into one frozen
record of everything ``PROFILE.md`` prints: the per-system family and field
grain, the ``MemberOf`` tags, description coverage, the hazards, the
Position/DeviceType coverage, the AD scalars and the import-walk totals.
Inputs here are normalised bodies, so each case shows the exact spelling the
census reads.

Two of those readings answer to a reviewer. The pending judgments are always
detected from the raw export, because the profile is written at import; every
count, hazard and owner follows the reviewer's answers once a mapping is handed
in. The hazard the judgments take over from is the partial channel list, so the
shapes that now pend are pinned as not being hazards any more.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from osprey.services.mml.census import (
    KNOWN_HANDLE_KEYS,
    Census,
    Owner,
    take_census,
)
from osprey.services.mml.mapping.schema import (
    Facility,
    FamilyJudgments,
    FieldAnswer,
    Mapping,
)
from osprey.services.mml.normalize import normalize_family

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"


def _bpm(**extra: object) -> dict:
    body: dict = {
        "FamilyName": "BPM",
        "MemberOf": ["BPM", "Diagnostics"],
        "DeviceList": [[1, 1], [1, 2]],
        "X": {"MemberOf": ["BPM", "Monitor"], "ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]},
    }
    body.update(extra)
    return body


def _system(census: Census, name: str):
    (found,) = (s for s in census.systems if s.name == name)
    return found


def _family(census: Census, system: str, name: str):
    (found,) = (f for f in _system(census, system).families if f.name == name)
    return found


def _dcct() -> dict:
    """One device carrying two channels past it, as NSLS-II exports DCCT."""
    return {
        "FamilyName": "DCCT",
        "DeviceList": [[1, 1]],
        "Monitor": {
            "HWUnits": "mA",
            "ChannelNames": ["SR:DCCT:AveI-I", "SR:DCCT:Lifetime-I", "SR:DCCT:I:Total-I"],
        },
    }


def _tune() -> dict:
    """Three devices with only two of them reached, as NSLS-II exports TUNE."""
    return {
        "FamilyName": "TUNE",
        "DeviceList": [[1, 1], [1, 2], [1, 3]],
        "Monitor": {"HWUnits": "Tune", "ChannelNames": ["SR:TUNE:Vx-I", "SR:TUNE:Vy-I"]},
    }


def _mapping(judgments: dict[str, FamilyJudgments]) -> Mapping:
    return Mapping(
        facility=Facility(token="quokka", title=None, description=None, provenance="human"),
        systems={},
        section_order=(),
        judgments=judgments,
    )


class TestShape:
    """The census is a frozen record and reads without modifying its inputs."""

    def test_census_is_frozen(self):
        """Census and its nested records refuse attribute assignment."""
        census = take_census({"SR": {"BPM": _bpm()}}, None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            census.totals = None  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            census.systems[0].name = "X"  # type: ignore[misc]

    def test_inputs_are_not_modified(self):
        """ao and ad are identical before and after the census."""
        ao = {"SR": {"BPM": _bpm()}, "_import_order": ["SR"]}
        ad = {"SR": {"Energy": 1.9}}
        before = (copy.deepcopy(ao), copy.deepcopy(ad))
        take_census(ao, ad)
        assert (ao, ad) == before

    def test_underscore_keys_are_never_systems_or_families(self):
        """Top-level and system-level ``_`` keys are bookkeeping, not grain."""
        ao = {
            "_exports": {"SR": {"exporter": "mml_export 1.0.0"}},
            "_import_order": ["SR"],
            "SR": {"_description": "Storage ring", "BPM": _bpm()},
        }
        census = take_census(ao, None)
        assert [s.name for s in census.systems] == ["SR"]
        assert [f.name for f in census.systems[0].families] == ["BPM"]

    def test_systems_follow_import_order(self):
        """Systems are listed in ``_import_order``, then any others in ao order."""
        ao = {"BR": {"BPM": _bpm()}, "SR": {"BPM": _bpm()}, "_import_order": ["SR", "BR"]}
        assert [s.name for s in take_census(ao, None).systems] == ["SR", "BR"]

    def test_same_input_gives_equal_census(self):
        """Two censuses of one input compare equal."""
        ao = {"SR": {"BPM": _bpm()}}
        assert take_census(ao, None) == take_census(copy.deepcopy(ao), None)


class TestFamilyGrain:
    """Families, fields, device counts and disabled devices come from FamilyView."""

    def test_fields_and_their_channel_keys(self):
        """Each field lists its channel keys in CHANNEL_KEYS order."""
        body = _bpm(Y={"TangoNames": ["a/b/1", "a/b/2"], "ChannelNames": ["Y1", "Y2"]})
        family = _family(take_census({"SR": {"BPM": body}}, None), "SR", "BPM")
        assert [(f.name, f.keys) for f in family.fields] == [
            ("X", ("ChannelNames",)),
            ("Y", ("ChannelNames", "TangoNames")),
        ]

    def test_device_count_and_fallback(self):
        """A family without DeviceList takes the longest list and is marked fallback."""
        ao = {"SR": {"BPM": _bpm(), "HCM": {"SP": {"ChannelNames": ["A", "B", "C"]}}}}
        census = take_census(ao, None)
        assert _family(census, "SR", "BPM").n_devices == 2
        assert not _family(census, "SR", "BPM").n_devices_from_fallback
        assert _family(census, "SR", "HCM").n_devices == 3
        assert _family(census, "SR", "HCM").n_devices_from_fallback
        assert _system(census, "SR").fallback_families == ("HCM",)

    def test_disabled_devices(self):
        """Status 0 positions are reported per family."""
        family = _family(take_census({"SR": {"BPM": _bpm(Status=[1, 0])}}, None), "SR", "BPM")
        assert family.disabled_devices == (1,)

    def test_setup_sourced_families(self):
        """Families whose arrays came from setup are listed per system."""
        setup = {
            "FamilyName": "Q",
            "setup": {"DeviceList": [[1, 1]]},
            "SP": {"ChannelNames": ["Q"]},
        }
        census = take_census({"SR": {"BPM": _bpm(), "Q": setup}}, None)
        assert _system(census, "SR").setup_families == ("Q",)
        assert _family(census, "SR", "Q").arrays_source == "setup"


class TestMemberOf:
    """Tags are kept verbatim with every owner that carries them."""

    def test_tags_verbatim_with_owners(self):
        """Family-level and field-level tags both count; case is preserved."""
        body = _bpm()
        body["X"]["MemberOf"] = ["BPM", "monitor", None]
        system = _system(take_census({"SR": {"BPM": body}}, None), "SR")
        tags = {t.tag: t.owners for t in system.member_of}
        assert tags == {
            "BPM": (("BPM", None), ("BPM", "X")),
            "Diagnostics": (("BPM", None),),
            "monitor": (("BPM", "X"),),
        }
        assert [t.tag for t in system.member_of] == ["BPM", "Diagnostics", "monitor"]


class TestDescriptions:
    """Families with and without a native description."""

    def test_with_and_without(self):
        """A Description or _description counts; blank does not."""
        ao = {"SR": {"BPM": _bpm(Description="Beam monitors"), "HCM": _bpm(Description="  ")}}
        system = _system(take_census(ao, None), "SR")
        assert system.families_with_descriptions == (("BPM", "Beam monitors"),)
        assert system.families_without_descriptions == ("HCM",)


class TestHandlesAndTypoKeys:
    """Function handles wherever they sit; unknown handle keys are typo keys."""

    def test_known_handle_key_is_not_a_typo(self):
        """A handle under HW2PhysicsFcn is a handle hazard only."""
        body = _bpm()
        body["X"]["HW2PhysicsFcn"] = {"$fn": "amp2k", "file": None}
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [(h.family, h.path, h.function) for h in hazards.function_handles] == [
            ("BPM", ("X", "HW2PhysicsFcn"), "amp2k")
        ]
        assert hazards.typo_keys == ()
        assert "HW2PhysicsFcn" in KNOWN_HANDLE_KEYS

    def test_handle_under_unknown_key_is_also_a_typo(self):
        """The ALS HW2PhysicSDcn record is listed as a handle and as a typo key."""
        body = _bpm()
        body["X"]["HW2PhysicSDcn"] = {"$fn": "amp2k", "file": ""}
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [h.path for h in hazards.function_handles] == [("X", "HW2PhysicSDcn")]
        assert [(t.family, t.path) for t in hazards.typo_keys] == [("BPM", ("X", "HW2PhysicSDcn"))]

    def test_unknown_fcn_key_and_miscased_key_are_typos(self):
        """A *Fcn key outside the known set and a mis-cased MML key are typo keys."""
        body = _bpm()
        body["X"]["Physics2HwFcn"] = "k2amp"
        body["X"]["hwunits"] = "mm"
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert sorted(t.path for t in hazards.typo_keys) == [
            ("X", "Physics2HwFcn"),
            ("X", "hwunits"),
        ]

    def test_integer_one_handle_has_no_function(self):
        """A {$fn: None} handle is still a handle."""
        body = _bpm()
        body["X"]["HW2PhysicsFcn"] = {"$fn": None, "file": None}
        (handle,) = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards.function_handles
        assert handle.function is None


class TestValueHazards:
    """Non-finite ranges and the HWUnits/DataType shapes that emit nothing."""

    def test_non_finite_ranges(self):
        """A Range with an Inf/-Inf/NaN string is listed; a finite one is not."""
        body = _bpm(Y={"ChannelNames": ["Y1", "Y2"], "Range": [-1, 1]})
        body["X"]["Range"] = ["-Inf", "Inf"]
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [(r.family, r.field, r.value) for r in hazards.non_finite_ranges] == [
            ("BPM", "X", ("-Inf", "Inf"))
        ]

    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            ([], "empty"),
            ("", "empty"),
            (None, "empty"),
            (["mm", "mm"], "per-device"),
            (["mm", "mm", "mm"], "non-scalar"),
            ({"a": 1}, "non-scalar"),
            (3, "non-string"),
        ],
    )
    def test_unit_and_datatype_shapes(self, value, kind):
        """HWUnits and DataType that are not a non-empty string are listed by kind."""
        body = _bpm()
        body["X"]["HWUnits"] = value
        body["X"]["DataType"] = value
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [(u.field, u.key, u.kind) for u in hazards.unit_shapes] == [
            ("X", "DataType", kind),
            ("X", "HWUnits", kind),
        ]

    def test_scalar_units_and_absent_keys_are_not_hazards(self):
        """A non-empty string or an absent key is fine."""
        body = _bpm()
        body["X"]["HWUnits"] = "mm"
        assert _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards.unit_shapes == ()


class TestStructuralHazards:
    """Case duplicates, illegal system tokens, dual keys, list shapes, zero channels."""

    def test_case_duplicate_families(self):
        """Families differing only by case in one system are grouped."""
        ao = {"MAIN": {"BPMx": _bpm(), "bpmx": _bpm(), "HCM": _bpm()}}
        hazards = _system(take_census(ao, None), "MAIN").hazards
        assert hazards.case_duplicate_families == (("BPMx", "bpmx"),)

    def test_pn_local_illegal_system_tokens(self):
        """A system token that is not a Turtle local name is listed."""
        census = take_census({"SR": {"BPM": _bpm()}, "LN-A": {"BPM": _bpm()}}, None)
        assert census.illegal_system_tokens == ("LN-A",)

    def test_dual_key_fields(self):
        """A field with both channel keys is listed."""
        body = _bpm()
        body["X"]["TangoNames"] = ["a/1", "a/2"]
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [(d.family, d.field) for d in hazards.dual_key_fields] == [("BPM", "X")]

    def test_empty_partial_and_broadcast_lists(self):
        """0-length, partial and 1-row broadcast lists are each listed per key."""
        body = _bpm(
            DeviceList=[[1, 1], [1, 2], [1, 3]],
            X={"ChannelNames": []},
            Y={"ChannelNames": ["Y1", "Y2"]},
            Z={"ChannelNames": ["Z"]},
            W={"ChannelNames": ["W1", "W2", "W3"]},
        )
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [(e.field, e.key) for e in hazards.empty_channel_lists] == [("X", "ChannelNames")]
        assert [(p.field, p.key, p.length, p.n_devices) for p in hazards.partial_channel_lists] == [
            ("Y", "ChannelNames", 2, 3)
        ]
        assert [(b.field, b.key) for b in hazards.broadcast_rows] == [("Z", "ChannelNames")]

    def test_zero_channel_families(self):
        """A family with a DeviceList but no bindings is listed."""
        ao = {"SR": {"BPM": _bpm(), "BEND": {"DeviceList": [[1, 1]], "Monitor": {"HWUnits": "A"}}}}
        assert _system(take_census(ao, None), "SR").hazards.zero_channel_families == ("BEND",)


class TestNarrowedPartialLists:
    """A short list is a hazard only where the family covers what it misses."""

    def test_a_short_list_another_field_reaches_is_listed(self):
        """Device 3 is bound by W, so X's gap is a shape hazard, not a question."""
        body = _bpm(
            DeviceList=[[1, 1], [1, 2], [1, 3]],
            X={"ChannelNames": ["X1", "X2"]},
            W={"ChannelNames": ["W1", "W2", "W3"]},
        )
        hazards = _system(take_census({"SR": {"BPM": body}}, None), "SR").hazards
        assert [(p.field, p.key, p.length, p.n_devices) for p in hazards.partial_channel_lists] == [
            ("X", "ChannelNames", 2, 3)
        ]

    def test_a_short_list_nothing_reaches_past_is_a_judgment(self):
        """The TUNE shape: device 3 is pending, so no hazard is reported for it."""
        body = _bpm(DeviceList=[[1, 1], [1, 2], [1, 3]], X={"ChannelNames": ["X1", "X2"]})
        system = _system(take_census({"SR": {"BPM": body}}, None), "SR")
        assert system.hazards.partial_channel_lists == ()
        assert [p.unbound_devices for p in system.pending] == [(3,)]

    def test_a_list_longer_than_the_devices_is_a_judgment(self):
        """The DCCT shape: rows past the devices are pending, never a hazard."""
        system = _system(take_census({"SR": {"DCCT": _dcct()}}, None), "SR")
        assert system.hazards.partial_channel_lists == ()
        assert [r.signal for r in system.pending[0].rows_beyond] == [
            "SR:DCCT:Lifetime-I",
            "SR:DCCT:I:Total-I",
        ]


class TestSharedPVs:
    """Every PV bound by two or more owners, with all owners."""

    def test_shared_across_systems_with_all_owners(self):
        """Owners are (system, family, field, index), sorted."""
        ao = {
            "SR": {"BPM": _bpm(Y={"ChannelNames": ["SR:BPM1:X", "Y2"]})},
            "BR": {"BPM": {"DeviceList": [[1, 1]], "X": {"ChannelNames": ["SR:BPM2:X "]}}},
        }
        census = take_census(ao, None)
        shared = {s.pv: s.owners for s in census.shared_pvs}
        assert shared == {
            "SR:BPM1:X": (Owner("SR", "BPM", "X", 0), Owner("SR", "BPM", "Y", 0)),
            "SR:BPM2:X": (Owner("BR", "BPM", "X", 0), Owner("SR", "BPM", "X", 1)),
        }

    def test_broadcast_row_alone_is_not_shared(self):
        """A 1-row broadcast list binds many devices but is one owner."""
        ao = {"SR": {"BPM": _bpm(Z={"ChannelNames": ["SR:Z"]})}}
        assert take_census(ao, None).shared_pvs == ()

    def test_dual_key_same_string_is_one_owner(self):
        """The same PV under both keys at one index is one owner, not shared."""
        body = _bpm()
        body["X"]["TangoNames"] = ["SR:BPM1:X", "t2"]
        assert take_census({"SR": {"BPM": body}}, None).shared_pvs == ()


class TestPendingJudgments:
    """What the raw export asks its reviewer, per family and per system."""

    def test_rows_beyond_devices_and_unbound_devices(self):
        """Each pending family carries its own rows, ordinals and device count."""
        census = take_census({"SR": {"DCCT": _dcct(), "TUNE": _tune()}}, None)
        pending = _system(census, "SR").pending
        assert [(p.family, p.n_devices, p.unbound_devices) for p in pending] == [
            ("DCCT", 1, ()),
            ("TUNE", 3, (3,)),
        ]
        assert [(r.field, r.keys, r.index, r.signal) for r in pending[0].rows_beyond] == [
            ("Monitor", ("ChannelNames",), 1, "SR:DCCT:Lifetime-I"),
            ("Monitor", ("ChannelNames",), 2, "SR:DCCT:I:Total-I"),
        ]

    def test_families_asking_nothing_are_absent(self):
        """A family whose grain is decidable by rule pends no entry at all."""
        census = take_census({"SR": {"BPM": _bpm(), "TUNE": _tune()}}, None)
        assert [p.family for p in _system(census, "SR").pending] == ["TUNE"]

    def test_the_facility_roll_up_follows_system_order(self):
        """Census.pending lists every system's families in import order."""
        ao = {"BR": {"TUNE": _tune()}, "SR": {"DCCT": _dcct()}, "_import_order": ["SR", "BR"]}
        assert [(p.system, p.family) for p in take_census(ao, None).pending] == [
            ("SR", "DCCT"),
            ("BR", "TUNE"),
        ]

    def test_an_answered_judgment_is_still_pending(self):
        """Pending is what the export asked, so a mapping never shortens it."""
        ao = {"SR": {"TUNE": _tune()}}
        mapping = _mapping({"TUNE": FamilyJudgments(unbound_devices={3: "drop"})})
        census = take_census(ao, None, mapping)
        assert [(p.family, p.unbound_devices) for p in census.pending] == [("TUNE", (3,))]


class TestCoverage:
    """Position and DeviceType real vs stand-in, per device."""

    def test_position_stand_ins(self):
        """None, strings and non-finite slots are stand-ins; finite numbers are real."""
        body = _bpm(
            DeviceList=[[1, 1], [1, 2], [1, 3], [1, 4]],
            Position=[1.5, None, "NaN", 3],
            DeviceType=["BPM", None, "", "BPM"],
            X={"ChannelNames": ["A", "B", "C", "D"]},
        )
        family = _family(take_census({"SR": {"BPM": body}}, None), "SR", "BPM")
        assert (family.position_real, family.position_stand_in) == (2, 2)
        assert (family.device_type_real, family.device_type_stand_in) == (2, 2)

    def test_unaligned_arrays_are_all_stand_in(self):
        """A Position list whose length is not n_devices gives no real slot."""
        family = _family(take_census({"SR": {"BPM": _bpm(Position=[1.0])}}, None), "SR", "BPM")
        assert (family.position_real, family.position_stand_in) == (0, 2)
        assert (family.device_type_real, family.device_type_stand_in) == (0, 2)

    def test_scalar_position_on_one_device(self):
        """A finite scalar Position on a one-device family is real."""
        body = {"DeviceList": [[1, 1]], "Position": 4.0, "X": {"ChannelNames": ["A"]}}
        family = _family(take_census({"SR": {"Q": body}}, None), "SR", "Q")
        assert (family.position_real, family.position_stand_in) == (1, 0)

    def test_coverage_totals_sum_to_every_device(self):
        """Coverage totals count every family, zero-channel ones included."""
        ao = {"SR": {"BPM": _bpm(Position=[1.0, 2.0]), "BEND": {"DeviceList": [[1, 1]]}}}
        totals = take_census(ao, None).totals
        assert (totals.position_real, totals.position_stand_in) == (2, 1)
        assert totals.position_real + totals.position_stand_in == totals.devices == 3
        assert totals.device_type_real + totals.device_type_stand_in == totals.devices


class TestAdScalars:
    """Scalars from the merged AD, keyed by system, nested keys dotted."""

    def test_system_keyed_ad(self):
        """Nested dicts flatten to dotted keys; lists and _ keys are skipped."""
        ad = {
            "SR": {
                "_export": {"exporter": "x"},
                "Machine": "Quokka",
                "Energy": 2.4,
                "HarmonicNumber": 150,
                "OpsData": {"LatticeFile": "lat"},
                "Tunes": [0.1, 0.2],
            }
        }
        census = take_census({"SR": {"BPM": _bpm()}}, ad)
        assert _system(census, "SR").ad_scalars == (
            ("Energy", 2.4),
            ("HarmonicNumber", 150),
            ("Machine", "Quokka"),
            ("OpsData.LatticeFile", "lat"),
        )

    def test_no_ad(self):
        """A missing AD yields no scalars."""
        assert _system(take_census({"SR": {"BPM": _bpm()}}, None), "SR").ad_scalars == ()

    def test_flat_ad_on_single_system(self):
        """A flat AD (not keyed by system) belongs to the one system imported."""
        census = take_census({"SR": {"BPM": _bpm()}}, {"Machine": "Q", "OpsData": {"A": 1}})
        assert _system(census, "SR").ad_scalars == (("Machine", "Q"), ("OpsData.A", 1))


class TestTotals:
    """The import-walk totals criterion 5 pins."""

    def test_totals(self):
        """Every total on a two-system input with blanks, broadcast and sharing."""
        ao = {
            "_import_order": ["SR", "BR"],
            "SR": {
                "BPM": _bpm(
                    DeviceList=[[1, 1], [1, 2], [1, 3]],
                    X={"ChannelNames": ["P1", None, "P3"]},
                    Y={"ChannelNames": ["BROAD"], "TangoNames": ["t1", "t2", "t3"]},
                ),
                "BEND": {"setup": {"DeviceList": [[1, 1]]}, "M": {"HWUnits": "A"}},
            },
            "BR": {"BPM": {"X": {"ChannelNames": ["P1", "Q2"]}}},
        }
        totals = take_census(ao, None).totals
        assert dataclasses.asdict(totals) == {
            "fields": 3,
            "raw_slots": 9,
            "raw_non_blank": 8,
            "blank": 1,
            "bindings": 10,
            "broadcast_fields": 1,
            "distinct_pvs": 7,
            "system_families": 3,
            "families": 2,
            "devices": 6,
            "setup_families": 1,
            "fallback_families": 1,
            "position_real": 0,
            "position_stand_in": 6,
            "device_type_real": 0,
            "device_type_stand_in": 6,
        }


class TestJudgedTotals:
    """With a mapping, every count follows the answers the reviewer wrote."""

    def test_a_supply_group_kept_whole_changes_nothing(self):
        """``keep_all`` leaves the export's own grain, so the census is unchanged."""
        body = {"DeviceList": [[1, 1], [1, 2]], "Monitor": {"ChannelNames": ["SR:Q", "SR:Q"]}}
        ao = {"SR": {"QM": body}}
        mapping = _mapping({"QM": FamilyJudgments(shared_pvs="keep_all", shared_pvs_present=True)})
        raw, judged = take_census(ao, None), take_census(ao, None, mapping)
        assert judged.totals == raw.totals
        assert judged.shared_pvs == raw.shared_pvs
        assert judged == raw

    def test_a_dropped_device_leaves_the_device_count(self):
        """TUNE's unreached device 3 goes; nothing was bound to it, so bindings hold."""
        ao = {"SR": {"TUNE": _tune()}}
        mapping = _mapping({"TUNE": FamilyJudgments(unbound_devices={3: "drop"})})
        raw, judged = take_census(ao, None).totals, take_census(ao, None, mapping).totals
        assert (judged.devices, judged.bindings) == (raw.devices - 1, raw.bindings)
        assert _family(take_census(ao, None, mapping), "SR", "TUNE").n_devices == 2

    def test_a_moved_row_becomes_a_field_of_its_own(self):
        """DCCT's lifetime row is a field now, so the same PVs sit in one more field."""
        ao = {"SR": {"DCCT": _dcct()}}
        mapping = _mapping(
            {
                "DCCT": FamilyJudgments(
                    rows_beyond={"Monitor": {"SR:DCCT:Lifetime-I": FieldAnswer("Lifetime")}}
                )
            }
        )
        census = take_census(ao, None, mapping)
        raw = take_census(ao, None).totals
        assert (census.totals.fields, census.totals.bindings, census.totals.distinct_pvs) == (
            raw.fields + 1,
            raw.bindings,
            raw.distinct_pvs,
        )
        assert [f.name for f in _family(census, "SR", "DCCT").fields] == ["Monitor", "Lifetime"]


class TestDialectFixture:
    """The committed system-keyed dialect export, normalised, end to end."""

    def test_dialect_census(self):
        """The fixture's hazards and totals are all found."""
        raw = json.loads((FIXTURES / "dialect" / "export.json").read_text())
        ao = {
            system: {
                name: (normalize_family(body) if isinstance(body, dict) else body)
                for name, body in families.items()
            }
            for system, families in raw.items()
        }
        ao["_import_order"] = ["RING", "BOOST"]
        census = take_census(ao, None)
        ring = _system(census, "RING")

        assert [f.name for f in ring.families] == ["HCM", "QF", "BEND"]
        assert ring.hazards.zero_channel_families == ("BEND",)
        assert [(b.family, b.field) for b in ring.hazards.broadcast_rows] == [("QF", "Setpoint")]
        assert ("QF", ("Setpoint", "HW2PhysicSDcn")) in [
            (t.family, t.path) for t in ring.hazards.typo_keys
        ]
        assert {(h.family, h.path) for h in ring.hazards.function_handles} == {
            ("HCM", ("Setpoint", "HW2PhysicsFcn")),
            ("HCM", ("Setpoint", "Physics2HWFcn")),
            ("QF", ("Setpoint", "HW2PhysicsFcn")),
            ("QF", ("Setpoint", "HW2PhysicSDcn")),
        }
        assert {(r.family, r.field) for r in ring.hazards.non_finite_ranges} == {
            ("HCM", "Setpoint"),
            ("HCM", "Monitor"),
            ("QF", "Setpoint"),
        }
        assert ("QF", "Monitor", "HWUnits", "empty") in [
            (u.family, u.field, u.key, u.kind) for u in ring.hazards.unit_shapes
        ]
        assert {s.pv: s.owners for s in census.shared_pvs} == {
            "RG:HCM2:RB": (Owner("RING", "HCM", "Monitor", 1), Owner("RING", "HCM", "Monitor", 2))
        }
        hcm = _family(census, "RING", "HCM")
        assert hcm.disabled_devices == (2,)
        assert (hcm.position_real, hcm.position_stand_in) == (2, 1)
        assert ring.setup_families == ("HCM", "BEND")
        assert ring.families_with_descriptions == (("QF", "Focusing quadrupoles"),)
        assert census.totals.system_families == 5
        assert census.totals.families == 5
