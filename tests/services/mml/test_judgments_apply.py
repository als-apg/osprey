"""Tests for applying a reviewer's judgment answers to one family body.

``apply_judgments`` reads the answers back onto one family body: a dropped row
leaves its list, a moved row becomes a field of its own, a promoted row becomes
a device the whole family grows a slot for, a dropped device leaves every list
that holds one slot per device, and a kept one gains an empty slot in the lists
that stopped short of it. ``judged_family_views`` is the same half over a whole
system. It holds no rule text -- an answer the semantic checker would refuse is
an internal ``AssertionError`` -- and it never touches an answer another
system's export pends, so the negatives are pinned as carefully as the
positives: a scalar on a family of several devices, a broadcast row and a list
the export already left short all survive the judgment unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from osprey.services.facility_knowledge.ttl_generator.mml_source import (
    bindings_for_family,
    devices_for_family,
)
from osprey.services.mml.family import FamilyView, family_views
from osprey.services.mml.judgments import (
    apply_judgments,
    judged_family_views,
    judged_va_block,
    pending_judgments,
)
from osprey.services.mml.mapping.schema import (
    Facility,
    Family,
    FamilyJudgments,
    Field,
    FieldAnswer,
    Mapping,
    OwnerMap,
    SharedAnswer,
    System,
)
from osprey.services.mml.normalize import normalize_family

FAMILY = "QM"
SYSTEM = "SR"


def _rows(n_devices: int) -> list[list[int]]:
    return [[1, index + 1] for index in range(n_devices)]


def _mapping(
    rows: dict,
    family: str = FAMILY,
    unbound: dict | None = None,
    shared: SharedAnswer | None = None,
) -> Mapping:
    return Mapping(
        facility=Facility(token="quokka", title=None, description=None, provenance="human"),
        systems={},
        section_order=(),
        judgments={
            family: FamilyJudgments(
                rows_beyond=rows,
                unbound_devices=unbound or {},
                shared_pvs=shared,
                shared_pvs_present=shared is not None,
            )
        },
    )


def _apply(body: dict, rows: dict, system: str = SYSTEM) -> dict:
    return apply_judgments(system, FAMILY, body, _mapping(rows))


def _apply_unbound(body: dict, unbound: dict, system: str = SYSTEM) -> dict:
    return apply_judgments(system, FAMILY, body, _mapping({}, unbound=unbound))


def _apply_shared(body: dict, shared: SharedAnswer, system: str = SYSTEM) -> dict:
    return apply_judgments(system, FAMILY, body, _mapping({}, shared=shared))


def _model_mapping(shared: SharedAnswer | None = None) -> Mapping:
    """Return a mapping the corpus builder can resolve, carrying the answers."""
    judged = _mapping({}, shared=shared)
    return Mapping(
        facility=judged.facility,
        systems={SYSTEM: System(raw=SYSTEM, name=SYSTEM, description=None, provenance="stated")},
        section_order=(SYSTEM,),
        families={
            FAMILY: Family(
                raw=FAMILY,
                rename=None,
                branch=None,
                class_="Magnet",
                aliases=(FAMILY,),
                description=f"{FAMILY} family",
                provenance="stated",
                channels=2,
                fields={
                    name: Field(description=f"{name} of {FAMILY}", provenance="stated")
                    for name in ("Monitor", "Setpoint")
                },
            )
        },
        judgments=judged.judgments,
    )


def _distinct_pvs(view: FamilyView) -> set[str]:
    """Return the distinct strings the family binds, as the census counts them."""
    return {
        slot.strip()
        for field in view.fields.values()
        for key in field.keys
        for slot in field.slots(key)
        if isinstance(slot, str) and slot.strip()
    }


def _shared_body() -> dict:
    """Two devices on one supply: one PV per field, bound at both ordinals."""
    return {
        "DeviceList": _rows(2),
        "CommonNames": ["QM1", "QM2"],
        "Monitor": {"ChannelNames": ["p", "p"]},
        "Setpoint": {"ChannelNames": ["q", "q"]},
    }


def _view(body: dict) -> FamilyView:
    return FamilyView(SYSTEM, FAMILY, body)


class TestIdentity:
    """The body handed in is read, never written."""

    def test_input_body_is_not_modified(self):
        """Every answer lands on a copy."""
        body = {
            "DeviceList": [[1, 1]],
            "CommonNames": "QM1",
            "A": {"ChannelNames": ["a1", "a2"]},
            "B": {"ChannelNames": ["b1", "b2"]},
        }
        before = copy.deepcopy(body)
        _apply(body, {"A": {"a2": "drop"}, "B": {"b2": "device"}})
        assert body == before

    def test_family_with_no_answers_is_copied_through(self):
        """A family the mapping says nothing about comes back as it was."""
        body = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["m1", "m2"]}}
        judged = apply_judgments(SYSTEM, FAMILY, body, _mapping({}, family="BPMx"))
        assert judged == body
        assert judged is not body


class TestDropAndDevice:
    """A dropped row leaves; a promoted row becomes a device."""

    def test_drop_beside_device_pads_the_dropped_list(self):
        """The dropped list keeps a slot for the device the other list mints."""
        body = {
            "DeviceList": [[1, 1]],
            "A": {"ChannelNames": ["a1", "a2"]},
            "B": {"ChannelNames": ["b1", "b2"]},
        }
        judged = _apply(body, {"A": {"a2": "drop"}, "B": {"b2": "device"}})
        view = _view(judged)
        assert judged["A"]["ChannelNames"] == ["a1", None]
        assert judged["B"]["ChannelNames"] == ["b1", "b2"]
        assert judged["DeviceList"] == [[1, 1], [1, 2]]
        assert view.n_devices == 2
        assert [field.broadcast for field in view.fields.values()] == [False, False]

    def test_a_drop_between_two_devices_closes_the_gap(self):
        """The rows that survive are numbered by where they land, not where they were."""
        body = {
            "DeviceList": [[1, 1]],
            "A": {"ChannelNames": ["a1", "a2"]},
            "B": {"ChannelNames": ["b1", "b2", "b3"]},
        }
        judged = _apply(body, {"A": {"a2": "device"}, "B": {"b2": "drop", "b3": "device"}})
        assert _view(judged).n_devices == 2
        assert judged["A"]["ChannelNames"] == ["a1", "a2"]
        assert judged["B"]["ChannelNames"] == ["b1", "b3"]

    def test_a_shorter_list_is_padded_to_the_new_device_count(self):
        """A list that spanned the export's devices spans the new ones too."""
        body = {
            "DeviceList": _rows(3),
            "Setpoint": {"ChannelNames": ["s1", "s2", "s3"]},
            "Monitor": {"ChannelNames": ["m1", "m2", "m3", "m4"]},
        }
        judged = _apply(body, {"Monitor": {"m4": "device"}})
        assert judged["Setpoint"]["ChannelNames"] == ["s1", "s2", "s3", None]
        assert judged["Monitor"]["ChannelNames"] == ["m1", "m2", "m3", "m4"]
        assert judged["DeviceList"] == [[1, 1], [1, 2], [1, 3], [1, 4]]
        assert _view(judged).n_devices == 4

    def test_two_promoted_rows_carry_the_family_to_three_devices(self):
        """The longest list sets the count; a per-device list pads with the export's blank."""
        body = {
            "DeviceList": [1, 1],
            "CommonNames": "QM1",
            "A": {"ChannelNames": ["a1", "a2", "a3"]},
            "B": {"ChannelNames": ["b1", "b2"], "HWUnits": ["amp"]},
        }
        judged = _apply(
            body,
            {"A": {"a2": "device", "a3": "device"}, "B": {"b2": "device"}},
        )
        assert _view(judged).n_devices == 3
        assert judged["A"]["ChannelNames"] == ["a1", "a2", "a3"]
        assert judged["B"]["ChannelNames"] == ["b1", "b2", None]
        assert judged["B"]["HWUnits"] == ["amp", "", ""]
        assert judged["CommonNames"] == ["QM1", "", ""]
        assert judged["DeviceList"] == [[1, 1], [1, 2], [1, 3]]

    def test_the_setup_block_grows_with_the_family(self):
        """Arrays read from ``setup`` are the family's arrays, and grow there."""
        body = {
            "setup": {"DeviceList": [[1, 1]], "CommonNames": ["QM1"]},
            "Monitor": {"ChannelNames": ["m1", "m2"]},
        }
        judged = _apply(body, {"Monitor": {"m2": "device"}})
        assert judged["setup"]["DeviceList"] == [[1, 1], [1, 2]]
        assert judged["setup"]["CommonNames"] == ["QM1", ""]
        assert _view(judged).n_devices == 2


class TestUntouched:
    """What the judgment must leave exactly as exported."""

    def test_an_answer_this_system_does_not_pend_is_ignored(self):
        """The same family is judged per system; another system's row is not this one's."""
        body = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["m1", "m2"]}}
        judged = _apply(body, {"Monitor": {"m3": "device"}})
        assert judged == body

    def test_tags_limits_and_tolerances_survive_a_promotion(self):
        """Only the per-device keys are per-device; every other key is carried whole."""
        body = {
            "DeviceList": _rows(2),
            "MemberOf": ["quad", "magnet"],
            "Monitor": {
                "ChannelNames": ["m1", "m2", "m3"],
                "MemberOf": ["quad", "magnet"],
                "Range": [0.0, 1.0],
                "Tolerance": [0.1, 0.2],
            },
        }
        judged = _apply(body, {"Monitor": {"m3": "device"}})
        assert _view(judged).n_devices == 3
        assert judged["MemberOf"] == ["quad", "magnet"]
        assert judged["Monitor"]["MemberOf"] == ["quad", "magnet"]
        assert judged["Monitor"]["Range"] == [0.0, 1.0]
        assert judged["Monitor"]["Tolerance"] == [0.1, 0.2]

    def test_a_scalar_on_several_devices_is_left_alone(self):
        """A scalar ``Position`` states one value for the family, not device 1's."""
        body = {
            "DeviceList": _rows(3),
            "Position": 0,
            "Monitor": {"ChannelNames": ["m1", "m2", "m3", "m4"]},
        }
        judged = _apply(body, {"Monitor": {"m4": "device"}})
        assert judged["Position"] == 0
        assert _view(judged).n_devices == 4

    def test_a_broadcast_row_still_broadcasts(self):
        """A one-row list on a family of several devices reaches the new device too."""
        body = {
            "DeviceList": _rows(2),
            "Setpoint": {"ChannelNames": ["all"]},
            "Monitor": {"ChannelNames": ["m1", "m2", "m3"]},
        }
        judged = _apply(body, {"Monitor": {"m3": "device"}})
        view = _view(judged)
        assert judged["Setpoint"]["ChannelNames"] == ["all"]
        assert view.fields["Setpoint"].broadcast is True
        assert view.fields["Setpoint"].slots("ChannelNames") == ["all", "all", "all"]

    def test_a_one_device_family_stays_one_device(self):
        """A DCCT-shaped flat pair is written as the one row it stands for."""
        body = {
            "DeviceList": [1, 1],
            "Monitor": {"ChannelNames": ["current", "lifetime", "beam"]},
        }
        assert _view(body).n_devices == 1
        judged = _apply(body, {"Monitor": {"lifetime": "drop", "beam": "drop"}})
        assert judged["DeviceList"] == [[1, 1]]
        assert judged["Monitor"]["ChannelNames"] == ["current"]
        assert _view(judged).n_devices == 1


class TestMovedRows:
    """A ``field:`` answer moves the row into a field of its own."""

    def test_the_row_moves_with_the_metadata_its_field_stated(self):
        """A scalar comes as it stands, an agreeing unit list collapses, tags come whole."""
        body = {
            "DeviceList": _rows(2),
            "Monitor": {
                "ChannelNames": ["m1", "m2", "lifetime"],
                "Units": "hardware",
                "HWUnits": ["mA", "mA"],
                "MemberOf": ["dcct"],
                "Tolerance": [0.1, 0.2],
            },
        }
        judged = _apply(body, {"Monitor": {"lifetime": FieldAnswer("Lifetime")}})
        assert judged["Monitor"]["ChannelNames"] == ["m1", "m2"]
        assert judged["Lifetime"] == {
            "ChannelNames": ["lifetime", None],
            "Units": "hardware",
            "HWUnits": "mA",
            "MemberOf": ["dcct"],
            "Tolerance": [0.1, 0.2],
        }
        assert _view(judged).n_devices == 2

    def test_a_unit_list_the_devices_disagree_on_is_left_behind(self):
        """The row's own unit is not knowable, so none is stated."""
        body = {
            "DeviceList": _rows(2),
            "Monitor": {"ChannelNames": ["m1", "m2", "extra"], "HWUnits": ["mA", "A"]},
        }
        judged = _apply(body, {"Monitor": {"extra": FieldAnswer("Extra")}})
        assert judged["Extra"] == {"ChannelNames": ["extra", None]}

    def test_both_keys_of_one_row_reach_the_same_new_field(self):
        """A dual-key row is two answers and one field."""
        body = {
            "DeviceList": [[1, 1]],
            "Monitor": {"ChannelNames": ["m1", "extra"], "TangoNames": ["t1", "t_extra"]},
        }
        judged = _apply(
            body,
            {"Monitor": {"extra": FieldAnswer("Extra"), "t_extra": FieldAnswer("Extra")}},
        )
        assert judged["Monitor"] == {"ChannelNames": ["m1"], "TangoNames": ["t1"]}
        assert judged["Extra"] == {"ChannelNames": ["extra"], "TangoNames": ["t_extra"]}
        assert _view(judged).n_devices == 1


class TestGuards:
    """Answers the semantic checker would have refused never reach here."""

    def test_device_on_a_signal_bound_below_the_devices_is_refused(self):
        """Promoting it would mint a shared PV detection never asked about."""
        body = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["p", "q", "p"]}}
        with pytest.raises(AssertionError):
            _apply(body, {"Monitor": {"p": "device"}})

    def test_a_field_name_the_family_already_carries_is_refused(self):
        """The move would overwrite a field the export states."""
        body = {
            "DeviceList": [[1, 1]],
            "Monitor": {"ChannelNames": ["m1", "extra"]},
            "Spare": {"ChannelNames": ["s1"]},
        }
        with pytest.raises(AssertionError):
            _apply(body, {"Monitor": {"extra": FieldAnswer("Spare")}})

    def test_one_field_name_on_two_different_rows_is_refused(self):
        """Two rows are two fields; one name cannot hold both."""
        body = {"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": ["m1", "x", "y"]}}
        with pytest.raises(AssertionError):
            _apply(body, {"Monitor": {"x": FieldAnswer("New"), "y": FieldAnswer("New")}})


class TestUnboundDevices:
    """A device no channel list reaches is dropped, or kept with empty slots."""

    def test_the_input_body_is_not_modified(self):
        """A drop lands on a copy, like every other answer."""
        body = {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["A", "B"]}}
        before = copy.deepcopy(body)
        _apply_unbound(body, {3: "drop"})
        assert body == before

    def test_a_dropped_device_leaves_every_per_device_list(self):
        """A BEND-shaped family of 4 devices and 3 channels becomes a family of 3."""
        body = {
            "DeviceList": _rows(4),
            "CommonNames": ["B1", "B2", "B3", "B4"],
            "ElementList": [1, 2, 3, 4],
            "Status": [1, 1, 1, 1],
            "Monitor": {"ChannelNames": ["m1", "m2", "m3"]},
            "Fault": {
                "ChannelNames": ["f1", "f2", "f3"],
                "MemberOf": ["BEND", "Magnet", "PlotFamily", "Boolean Monitor"],
            },
        }
        judged = _apply_unbound(body, {4: "drop"})
        assert _view(judged).n_devices == 3
        assert judged["DeviceList"] == [[1, 1], [1, 2], [1, 3]]
        assert judged["CommonNames"] == ["B1", "B2", "B3"]
        assert judged["ElementList"] == [1, 2, 3]
        assert judged["Status"] == [1, 1, 1]
        assert judged["Monitor"]["ChannelNames"] == ["m1", "m2", "m3"]
        assert judged["Fault"]["ChannelNames"] == ["f1", "f2", "f3"]

    def test_a_tag_list_as_long_as_the_devices_is_not_a_per_device_list(self):
        """``MemberOf`` names what the field belongs to, whatever its length."""
        body = {
            "DeviceList": _rows(4),
            "MemberOf": ["BEND", "Magnet", "PlotFamily", "Boolean Monitor"],
            "Fault": {
                "ChannelNames": ["f1", "f2", "f3"],
                "MemberOf": ["BEND", "Magnet", "PlotFamily", "Boolean Monitor"],
            },
        }
        judged = _apply_unbound(body, {4: "drop"})
        assert _view(judged).n_devices == 3
        assert judged["MemberOf"] == ["BEND", "Magnet", "PlotFamily", "Boolean Monitor"]
        assert judged["Fault"]["MemberOf"] == ["BEND", "Magnet", "PlotFamily", "Boolean Monitor"]

    def test_the_same_answer_is_ignored_where_the_family_binds_the_device(self):
        """The same family in another system reaches device 4, so the answer is not its."""
        body = {
            "DeviceList": _rows(4),
            "CommonNames": ["B1", "B2", "B3", "B4"],
            "Monitor": {"ChannelNames": ["m1", "m2", "m3", "m4"]},
        }
        judged = apply_judgments("StorageRing", FAMILY, body, _mapping({}, unbound={4: "drop"}))
        assert judged == body

    def test_a_dropped_device_leaves_a_family_scalar_alone(self):
        """A TUNE-shaped scalar ``Position`` states one value for the family, not device 1's."""
        body = {
            "DeviceList": _rows(3),
            "Position": 0,
            "Status": [1, 1, 1],
            "Monitor": {"ChannelNames": ["A", "B"]},
        }
        judged = _apply_unbound(body, {3: "drop"})
        assert _view(judged).n_devices == 2
        assert judged["Position"] == 0
        assert judged["Status"] == [1, 1]
        assert judged["Monitor"]["ChannelNames"] == ["A", "B"]

    def test_a_kept_device_gains_an_empty_slot(self):
        """The family keeps its devices, and the lists that stopped short reach the last one."""
        body = {
            "DeviceList": _rows(3),
            "Position": 0,
            "Status": [1, 1, 1],
            "Monitor": {"ChannelNames": ["A", "B"], "HWUnits": ["mA", "mA"]},
        }
        judged = _apply_unbound(body, {3: "keep"})
        view = _view(judged)
        assert view.n_devices == 3
        assert judged["DeviceList"] == _rows(3)
        assert judged["Monitor"]["ChannelNames"] == ["A", "B", None]
        assert judged["Monitor"]["HWUnits"] == ["mA", "mA", ""]
        assert judged["Position"] == 0
        assert judged["Status"] == [1, 1, 1]
        assert view.fields["Monitor"].broadcast is False
        assert pending_judgments(view).unbound_devices == ()

    def test_a_drop_and_a_keep_answer_one_family(self):
        """The kept device carries the row the export gave it; the dropped one is gone."""
        body = {
            "DeviceList": _rows(5),
            "CommonNames": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "Monitor": {
                "ChannelNames": ["m1", "m2", "m3"],
                "HWUnits": ["A", "A", "A", "A", "A"],
            },
        }
        judged = _apply_unbound(body, {4: "drop", 5: "keep"})
        assert _view(judged).n_devices == 4
        assert judged["DeviceList"] == [[1, 1], [1, 2], [1, 3], [1, 5]]
        assert judged["CommonNames"] == ["Q1", "Q2", "Q3", "Q5"]
        assert judged["Monitor"]["ChannelNames"] == ["m1", "m2", "m3", None]
        assert judged["Monitor"]["HWUnits"] == ["A", "A", "A", "A"]

    def test_a_broadcast_row_leaves_no_device_unbound(self):
        """A broadcast field reaches every device, so the family pends no ordinal at all."""
        body = {
            "DeviceList": _rows(3),
            "Monitor": {"ChannelNames": ["A", "B"]},
            "Setpoint": {"ChannelNames": ["S"]},
        }
        judged = _apply_unbound(body, {3: "keep"})
        view = _view(judged)
        assert judged == body
        assert view.fields["Setpoint"].broadcast is True
        assert view.fields["Setpoint"].slots("ChannelNames") == ["S", "S", "S"]
        assert view.fields["Monitor"].broadcast is False

    def test_an_undecided_ordinal_stays_pending(self):
        """A ``null`` answer is not an answer, and the device it names is still asked about."""
        body = {
            "DeviceList": _rows(4),
            "CommonNames": ["Q1", "Q2", "Q3", "Q4"],
            "Monitor": {"ChannelNames": ["A", "B"]},
        }
        judged = _apply_unbound(body, {3: None, 4: "drop"})
        view = _view(judged)
        assert view.n_devices == 3
        assert judged["CommonNames"] == ["Q1", "Q2", "Q3"]
        assert judged["Monitor"]["ChannelNames"] == ["A", "B"]
        assert pending_judgments(view).unbound_devices == (3,)

    def test_a_list_the_export_left_short_is_not_filled(self):
        """Only the lists that stop at the last bound device are padded."""
        body = {
            "DeviceList": _rows(5),
            "CommonNames": ["Q1", "Q2", "Q3", "Q4"],
            "Monitor": {"ChannelNames": ["m1", "m2", "m3"]},
        }
        judged = _apply_unbound(body, {4: "keep", 5: "keep"})
        assert _view(judged).n_devices == 5
        assert judged["Monitor"]["ChannelNames"] == ["m1", "m2", "m3", None, None]
        assert judged["CommonNames"] == ["Q1", "Q2", "Q3", "Q4"]

    def test_the_setup_block_loses_the_dropped_device(self):
        """Arrays read from ``setup`` are the family's arrays, and shrink there."""
        body = {
            "setup": {"DeviceList": _rows(3), "CommonNames": ["A1", "A2", "A3"]},
            "Monitor": {"ChannelNames": ["m1", "m2"]},
        }
        judged = _apply_unbound(body, {3: "drop"})
        assert judged["setup"]["DeviceList"] == [[1, 1], [1, 2]]
        assert judged["setup"]["CommonNames"] == ["A1", "A2"]
        assert _view(judged).n_devices == 2


class TestSharedSupply:
    """An owned supply group keeps its PVs on the owning device alone."""

    def test_the_owner_keeps_the_supply_and_the_other_members_lose_it(self):
        """Every field of the group loses its PV on every member but the owner."""
        body = _shared_body()
        before = copy.deepcopy(body)
        judged = _apply_shared(body, OwnerMap(owners={1: 1}))
        assert judged["Monitor"]["ChannelNames"] == ["p", None]
        assert judged["Setpoint"]["ChannelNames"] == ["q", None]
        assert body == before

    def test_the_stranded_member_is_still_a_device_of_the_family(self):
        """The family keeps its devices and its rows; only the bindings go."""
        judged = _view(_apply_shared(_shared_body(), OwnerMap(owners={1: 1})))
        assert judged.n_devices == 2
        assert judged.device_rows == [[1, 1], [1, 2]]
        assert judged.body["CommonNames"] == ["QM1", "QM2"]
        assert [field.slots("ChannelNames")[1] for field in judged.fields.values()] == [
            None,
            None,
        ]

    def test_the_corpus_states_both_devices_and_one_binding_per_field(self):
        """A member bound by nothing is still a device; the owner carries the PVs."""
        mapping = _model_mapping(OwnerMap(owners={1: 1}))
        view = FamilyView(SYSTEM, FAMILY, apply_judgments(SYSTEM, FAMILY, _shared_body(), mapping))
        devices = devices_for_family(view, mapping)
        bindings, _ = bindings_for_family(view, mapping)
        assert [device.device for device in devices] == ["1", "2"]
        assert [binding.full_pv for binding in bindings] == ["p", "q"]
        assert {binding.address.device for binding in bindings} == {"1"}

    def test_the_bindings_drop_by_one_per_field_for_each_member_that_is_not_the_owner(self):
        """The census counts one binding per field, and the same distinct PVs."""
        body = _shared_body()
        raw = _view(body)
        judged = _view(_apply_shared(body, OwnerMap(owners={1: 1})))
        assert raw.channel_count - judged.channel_count == (2 - 1) * 2
        assert judged.channel_count == 2
        assert _distinct_pvs(judged) == _distinct_pvs(raw) == {"p", "q"}

    def test_only_the_field_that_shares_the_supply_loses_a_slot(self):
        """A member's own channels stand beside the supply it no longer owns."""
        body = {
            "DeviceList": _rows(3),
            "Monitor": {"ChannelNames": ["IJ:QM1:RB", "IJ:QM2:RB", "IJ:QM2:RB"]},
            "On": {"ChannelNames": ["IJ:QM1:On", "IJ:QM2:On", "IJ:QM3:On"]},
            "OnControl": {"ChannelNames": ["IJ:QM1:Cmd", "IJ:QM2:Cmd", "IJ:QM3:Cmd"]},
        }
        judged = _apply_shared(body, OwnerMap(owners={2: 2}))
        assert judged["Monitor"]["ChannelNames"] == ["IJ:QM1:RB", "IJ:QM2:RB", None]
        assert judged["On"]["ChannelNames"] == ["IJ:QM1:On", "IJ:QM2:On", "IJ:QM3:On"]
        assert judged["OnControl"]["ChannelNames"] == ["IJ:QM1:Cmd", "IJ:QM2:Cmd", "IJ:QM3:Cmd"]

    def test_a_middle_member_can_own_the_supply(self):
        """The owner is the ordinal the answer names, not the group's lowest."""
        body = {
            "DeviceList": _rows(3),
            "Monitor": {"ChannelNames": ["s", "s", "s"]},
        }
        judged = _apply_shared(body, OwnerMap(owners={1: 2}))
        assert judged["Monitor"]["ChannelNames"] == [None, "s", None]

    @pytest.mark.parametrize(
        "answer",
        ["keep_all", OwnerMap(owners={1: "keep_all"}), None],
        ids=["family", "group", "undecided"],
    )
    def test_an_answer_that_settles_nothing_leaves_the_body_as_exported(self, answer):
        """``keep_all`` and an undecided supply change nothing, key for key."""
        body = _shared_body()
        judged = apply_judgments(SYSTEM, FAMILY, body, _mapping({}, shared=answer))
        assert judged == body
        assert judged is not body

    def test_a_group_this_system_does_not_pend_is_ignored(self):
        """One export's shared supply is another export's device of its own."""
        body = {
            "DeviceList": _rows(2),
            "Monitor": {"ChannelNames": ["p1", "p2"]},
            "Setpoint": {"ChannelNames": ["q1", "q2"]},
        }
        assert pending_judgments(_view(body)).groups == ()
        assert _apply_shared(body, OwnerMap(owners={1: 1})) == body

    def test_a_broadcast_field_carrying_the_supply_keeps_its_row(self):
        """A one-row list broadcasts to every device, so no member owns its slot."""
        body = {
            "DeviceList": _rows(2),
            "Monitor": {"ChannelNames": ["p", "p"]},
            "Setpoint": {"ChannelNames": ["p"]},
        }
        judged = _apply_shared(body, OwnerMap(owners={1: 1}))
        assert judged["Monitor"]["ChannelNames"] == ["p", None]
        assert judged["Setpoint"]["ChannelNames"] == ["p"]
        assert _view(judged).fields["Setpoint"].broadcast

    def test_an_owner_outside_the_group_is_refused(self):
        """An answer the semantic checker would refuse is an internal assertion."""
        with pytest.raises(AssertionError, match="not one of its members"):
            _apply_shared(_shared_body(), OwnerMap(owners={1: 3}))


class TestJudgedFamilyViews:
    """One view per family of a system, the answers already applied."""

    def test_a_system_with_no_answers_reads_as_the_export(self):
        """Without a judgment the judged views state what the raw views state."""
        system_body = {
            FAMILY: {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["m1", "m2"]}},
            "BPMx": {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["b1", "b2", "b3"]}},
            "_bookkeeping": {"note": "skipped"},
        }
        mapping = _mapping({}, family="ZZ")
        raw = list(family_views(SYSTEM, system_body))
        judged = list(judged_family_views(SYSTEM, system_body, mapping))
        assert [view.raw_name for view in judged] == [view.raw_name for view in raw]
        assert [view.n_devices for view in judged] == [view.n_devices for view in raw]
        assert [view.channel_count for view in judged] == [view.channel_count for view in raw]
        assert [sorted(view.fields) for view in judged] == [sorted(view.fields) for view in raw]

    def test_the_views_carry_the_answered_grain(self):
        """The family the reviewer answered is seen as answered, and only that one."""
        system_body = {
            FAMILY: {
                "DeviceList": _rows(3),
                "CommonNames": ["Q1", "Q2", "Q3"],
                "Monitor": {"ChannelNames": ["m1", "m2"]},
            },
            "BPMx": {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["b1", "b2"]}},
        }
        before = copy.deepcopy(system_body)
        mapping = _mapping({}, unbound={3: "drop"})
        views = {view.raw_name: view for view in judged_family_views(SYSTEM, system_body, mapping)}
        assert views[FAMILY].n_devices == 2
        assert views[FAMILY].body["CommonNames"] == ["Q1", "Q2"]
        assert views["BPMx"].n_devices == 2
        assert system_body == before


SYNTHETIC = Path(__file__).resolve().parents[2] / "fixtures" / "mml" / "synthetic"


def _synthetic(suffix: str) -> dict:
    """Return one committed file of the synthetic export."""
    loaded: dict = json.loads((SYNTHETIC / f"quokka.sr.{suffix}.json").read_text())
    return loaded


def _synthetic_family(raw: str, reach: int) -> dict:
    """Return one exported family, its channel lists stopping at ``reach``.

    The committed export binds every device of every family, so a family whose
    last devices are unbound -- the export shape a drop answers -- is made from
    it by cutting the channel lists short of them. Everything else is the
    export's own.
    """
    body = normalize_family(_synthetic("ao")[raw])
    for field_body in body.values():
        if not isinstance(field_body, dict):
            continue
        for key in ("ChannelNames", "TangoNames"):
            if key in field_body:
                field_body[key] = field_body[key][:reach]
    return body


def _value_body() -> dict:
    """A four-device family carrying every per-device number an export writes.

    The shapes are the ones the committed exports spell: limits as one row of
    two per device, a gain, an offset, a roll and a golden value per device, a
    conversion parameter per device, and a parameter cell of one row per
    parameter and one column per device.
    """
    return {
        "DeviceList": _rows(4),
        "CommonNames": ["C1", "C2", "C3", "C4"],
        "Status": [1, 1, 1, 1],
        "AT": {"ATType": "COR", "ATIndex": [[9, 10], [19, 20], [29, 30], [39, "NaN"]]},
        "Monitor": {
            "ChannelNames": ["m1", "m2", "m3"],
            "HWUnits": ["A", "A", "A"],
            "Range": [[-1, 1], [-1, 1], [-1, 1], [-2, 2]],
            "Gain": [1.0, 1.0, 1.0, 0.5],
            "Offset": [0, 0, 0, 0.25],
            "Roll": [0, 0, 0, 0.5],
            "Golden": [0, 0, 0, 1],
            "HW2PhysicsParams": [0.1, 0.2, 0.3, 0.4],
            "Physics2HWParams": [[10, 10, 10, 20], [0, 0, 0, 1]],
        },
    }


class TestPerDeviceValuesAndIndices:
    """A device leaves every list that holds a slot for it, whatever it holds."""

    def test_a_dropped_device_leaves_the_atindex_of_a_sliced_family(self):
        """The kick family's slices go with the device that was sliced."""
        body = _synthetic_family("HC", reach=3)
        mapping = _mapping({}, family="HC", unbound={4: "drop"})
        judged = apply_judgments(SYSTEM, "HC", body, mapping)
        assert judged["DeviceList"] == [[1, 1], [2, 1], [3, 1]]
        assert judged["AT"]["ATIndex"] == [[9, 10], [19, 20], [29, 30]]
        assert judged["AT"]["ATType"] == "HCM"
        assert judged["CommonNames"] == ["qk-hc-1", "qk-hc-2", "qk-hc-3"]

    def test_atindex_range_gain_offset_roll_golden_and_parameters_drop_together(self):
        """One dropped device leaves every per-device number of the family at once."""
        judged = _apply_unbound(_value_body(), {4: "drop"})
        assert _view(judged).n_devices == 3
        assert judged["AT"]["ATIndex"] == [[9, 10], [19, 20], [29, 30]]
        monitor = judged["Monitor"]
        assert monitor["Range"] == [[-1, 1], [-1, 1], [-1, 1]]
        assert monitor["Gain"] == [1.0, 1.0, 1.0]
        assert monitor["Offset"] == [0, 0, 0]
        assert monitor["Roll"] == [0, 0, 0]
        assert monitor["Golden"] == [0, 0, 0]
        assert monitor["HW2PhysicsParams"] == [0.1, 0.2, 0.3]

    def test_the_atindex_and_a_parameter_cell_realign_on_different_axes(self):
        """A cell states its devices along its columns, an index list along its rows."""
        judged = _apply_unbound(_value_body(), {4: "drop"})
        assert judged["AT"]["ATIndex"] == [[9, 10], [19, 20], [29, 30]]
        assert judged["Monitor"]["Physics2HWParams"] == [[10, 10, 10], [0, 0, 0]]

    def test_a_family_wide_range_beside_the_atindex_is_left_alone(self):
        """A low/high pair is one span for the family, and a scalar one value."""
        body = _value_body()
        body["Setpoint"] = {
            "ChannelNames": ["s1", "s2", "s3"],
            "Range": [-1, 1],
            "HW2PhysicsParams": 0.5,
            "Tolerance": 0.1,
        }
        judged = _apply_unbound(body, {4: "drop"})
        assert judged["Setpoint"]["Range"] == [-1, 1]
        assert judged["Setpoint"]["HW2PhysicsParams"] == 0.5
        assert judged["Setpoint"]["Tolerance"] == 0.1

    def test_a_kept_device_leaves_the_atindex_as_the_export_wrote_it(self):
        """A list already spanning the devices carries the kept one's slot already."""
        judged = _apply_unbound(_value_body(), {4: "keep"})
        assert _view(judged).n_devices == 4
        assert judged["AT"]["ATIndex"] == [[9, 10], [19, 20], [29, 30], [39, "NaN"]]
        assert judged["Monitor"]["Gain"] == [1.0, 1.0, 1.0, 0.5]
        assert judged["Monitor"]["Physics2HWParams"] == [[10, 10, 10, 20], [0, 0, 0, 1]]
        assert judged["Monitor"]["ChannelNames"] == ["m1", "m2", "m3", None]

    def test_a_promoted_row_pads_the_atindex_and_the_parameters(self):
        """The devices the family grows reach every list that held a slot per device."""
        body = _value_body()
        body["Monitor"]["ChannelNames"] = ["m1", "m2", "m3", "m4", "m5"]
        body["Monitor"]["HWUnits"] = ["A", "A", "A", "A"]
        judged = _apply(body, {"Monitor": {"m5": "device"}})
        assert _view(judged).n_devices == 5
        assert judged["AT"]["ATIndex"] == [[9, 10], [19, 20], [29, 30], [39, "NaN"], ["NaN", "NaN"]]
        assert judged["Monitor"]["Gain"] == [1.0, 1.0, 1.0, 0.5, "NaN"]
        assert judged["Monitor"]["Range"] == [[-1, 1], [-1, 1], [-1, 1], [-2, 2], ["NaN", "NaN"]]
        assert judged["Monitor"]["Physics2HWParams"] == [
            [10, 10, 10, 20, "NaN"],
            [0, 0, 0, 1, "NaN"],
        ]


class TestJudgedVaBlock:
    """The virtual-accelerator block follows the devices its family was left with."""

    def test_the_va_block_loses_the_dropped_device_from_every_row(self):
        """The kick family's block reads three devices where the export wrote four."""
        va = {SYSTEM: _synthetic("va")}
        before = copy.deepcopy(va)
        mapping = _mapping({}, family="HC", unbound={4: "drop"})
        judged = judged_va_block(SYSTEM, "HC", va, mapping)
        assert judged["device_list"] == [[1, 1], [2, 1], [3, 1]]
        nominal = judged["nominals"]["Setpoint"]
        assert nominal["values"] == [1.5, -0.8, 0.4]
        assert nominal["at_index"] == [[9, 10], [19, 20], [29, 30]]
        assert nominal["at_type"] == "HCM"
        assert judged["Setpoint"]["calibration"]["gain"] == [0.0001, 0.0001, 0.0001]
        assert judged["Setpoint"]["calibration"]["offset"] == [0, 0, 0]
        assert judged["Monitor"]["monitor_inverse"]["gain"] == [10000, 10000, 10000]
        assert judged["Monitor"]["monitor_inverse"]["offset"] == [-4.44089209850063e-16, 0, 0]
        assert va == before

    def test_the_va_block_drops_the_rows_of_a_sampled_conversion(self):
        """A table holds one row of points per device, so a device takes its row."""
        mapping = _mapping({}, family="BEND", unbound={4: "drop"})
        judged = judged_va_block(SYSTEM, "BEND", {SYSTEM: _synthetic("va")}, mapping)
        calibration = judged["Setpoint"]["calibration"]
        assert len(calibration["grid"]) == 3
        assert len(calibration["values"]) == 3
        assert calibration["finite_span"] == [[0, 487.5], [0, 487.5], [0, 487.5]]
        assert all(len(row) == 33 for row in calibration["grid"])

    def test_the_va_block_keeps_the_energy_table_of_the_device_it_names(self):
        """The table describes one device's ramp by the row it names, not a slot each."""
        block = _synthetic("va")["families"]["BEND"]
        mapping = _mapping({}, family="BEND", unbound={4: "drop"})
        judged = judged_va_block(SYSTEM, "BEND", {SYSTEM: _synthetic("va")}, mapping)
        assert judged["energy_table"] == block["energy_table"]

    def test_the_va_block_of_a_family_with_no_answers_is_copied_through(self):
        """A family the reviewer never answered reads as the export wrote it."""
        va = {SYSTEM: _synthetic("va")}
        mapping = _mapping({}, family="HC", unbound={4: "drop"})
        judged = judged_va_block(SYSTEM, "QF", va, mapping)
        assert judged == va[SYSTEM]["families"]["QF"]

    def test_the_va_block_reads_a_document_that_is_one_system_already(self):
        """The document holds one block per system, or is one system's own."""
        mapping = _mapping({}, family="HC", unbound={4: "drop"})
        keyed = judged_va_block(SYSTEM, "HC", {SYSTEM: _synthetic("va")}, mapping)
        bare = judged_va_block(SYSTEM, "HC", _synthetic("va"), mapping)
        assert keyed == bare

    def test_the_va_block_leaves_an_empty_index_and_a_stated_nominal_alone(self):
        """The escape hatch states no index at all, and its nominals are per device."""
        mapping = _mapping({}, family="IDGAP", unbound={2: "drop"})
        judged = judged_va_block(SYSTEM, "IDGAP", {SYSTEM: _synthetic("va")}, mapping)
        assert judged["device_list"] == [[1, 1]]
        assert judged["nominals"]["Setpoint"]["values"] == ["NaN"]
        assert judged["nominals"]["Setpoint"]["at_index"] == []
        assert judged["nominals"]["Setpoint"]["synthetic"] == 1

    def test_the_va_block_of_a_one_device_family_states_a_span_not_two_devices(self):
        """A one-device family writes its device as a flat pair, which is no two slots."""
        mapping = _mapping({}, family="RF", unbound={1: "drop"})
        judged = judged_va_block(SYSTEM, "RF", {SYSTEM: _synthetic("va")}, mapping)
        assert judged["device_list"] == [1, 1]
        assert judged["nominals"]["Setpoint"]["values"] == 516.883548276
        assert judged["Setpoint"]["calibration"]["gain"] == 1000000

    def test_the_va_block_pads_its_rows_out_to_the_judged_device_count(self):
        """A row promoted to a device was no device when the export sampled the rows."""
        mapping = _mapping({}, family="HC", unbound={})
        judged = judged_va_block(SYSTEM, "HC", {SYSTEM: _synthetic("va")}, mapping, devices=5)
        assert judged["device_list"] == [[1, 1], [2, 1], [3, 1], [4, 1], ["NaN", "NaN"]]
        nominal = judged["nominals"]["Setpoint"]
        assert nominal["values"] == [1.5, -0.8, 0.4, 0, "NaN"]
        assert nominal["at_index"] == [[9, 10], [19, 20], [29, 30], [39, "NaN"], ["NaN", "NaN"]]
        assert judged["Setpoint"]["calibration"]["gain"] == [0.0001] * 4 + ["NaN"]

    def test_the_readout_rows_follow_the_devices_the_family_kept(self):
        """What corrects a reading is one number per device, so a drop takes its number.

        Left behind, the numbers would keep the export's device order while the
        device list moved to the judged one, and the third monitor would be
        corrected by the fourth monitor's gain.
        """
        mapping = _mapping({}, family="BPMx", unbound={3: "drop"})
        judged = judged_va_block(SYSTEM, "BPMx", {SYSTEM: _synthetic("va")}, mapping)
        readout = judged["Monitor"]["readout"]

        assert judged["device_list"] == [[1, 1], [2, 1], [4, 1]]
        assert readout["gain"] == [1.02, 0.98, 0.995]
        assert readout["offset"] == [0.12, -0.05, 0]
        assert readout["roll"] == [0.001, -0.002, 0.0005]
        assert readout["crunch"] == [0.002, 0, 0.004]

    def test_a_promoted_row_leaves_the_readout_with_nothing_to_correct_it_by(self):
        """A device the export never sampled has no correction of its own."""
        mapping = _mapping({}, family="BPMx", unbound={})
        judged = judged_va_block(SYSTEM, "BPMx", {SYSTEM: _synthetic("va")}, mapping, devices=5)

        assert judged["Monitor"]["readout"]["gain"] == [1.02, 0.98, 1.01, 0.995, "NaN"]

    def test_a_drop_and_a_pad_leave_the_va_block_at_the_judged_count(self):
        """The rows are realigned first and padded after, so the order is the judged one."""
        mapping = _mapping({}, family="HC", unbound={4: "drop"})
        judged = judged_va_block(SYSTEM, "HC", {SYSTEM: _synthetic("va")}, mapping, devices=4)
        assert judged["device_list"] == [[1, 1], [2, 1], [3, 1], ["NaN", "NaN"]]
        assert judged["nominals"]["Setpoint"]["values"] == [1.5, -0.8, 0.4, "NaN"]

    def test_a_family_the_va_document_has_no_block_for_is_refused(self):
        """A block the document does not hold is a caller's mistake, not an empty one."""
        mapping = _mapping({}, family="HC", unbound={4: "drop"})
        with pytest.raises(KeyError, match="HC"):
            judged_va_block(SYSTEM, "HC", {"LTB": _synthetic("va")}, mapping)
