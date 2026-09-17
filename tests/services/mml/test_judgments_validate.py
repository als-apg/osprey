"""Tests for the one home of the judgment answer rules.

``validate_answers`` reads a filled mapping back against what the raw export
actually pends, and returns one entry per answer the export cannot carry or the
mapping has yet to describe. ``map --check`` renders those entries and the emit
pre-flight refuses on them, so both lanes refuse exactly the same answers.

The answer vocabulary is structural and refused by ``parse_mapping`` long
before this module sees it; those refusals are pinned in
``test_mapping_schema.py``.
"""

from __future__ import annotations

import pytest

from osprey.services.mml.family import FamilyView
from osprey.services.mml.judgments import (
    PendingJudgments,
    pending_judgments,
    validate_answers,
)
from osprey.services.mml.mapping.schema import (
    Facility,
    Family,
    FamilyJudgments,
    Field,
    FieldAnswer,
    Mapping,
    OwnerMap,
)


def _rows(n_devices: int) -> list[list[int]]:
    return [[1, index + 1] for index in range(n_devices)]


def _view(body: dict, system: str = "SR", name: str = "DCCT") -> FamilyView:
    return FamilyView(system, name, body)


def _pending(*views: FamilyView) -> dict[tuple[str, str], PendingJudgments]:
    return {(view.system, view.raw_name): pending_judgments(view) for view in views}


def _mapping(judgments: dict[str, FamilyJudgments], *fields: str) -> Mapping:
    """A mapping carrying ``judgments`` and a ``DCCT`` entry describing ``fields``."""
    return Mapping(
        facility=Facility(token="Zephyr", title="ZEPHYR", description=None, provenance="stated"),
        systems={},
        section_order=(),
        families={
            "DCCT": Family(
                raw="DCCT",
                rename=None,
                branch=None,
                class_=None,
                aliases=(),
                description=None,
                provenance="stated",
                channels=0,
                fields={name: Field(description=None, provenance="stated") for name in fields},
            )
        },
        judgments=judgments,
    )


def _validate(body: dict, judgments: dict[str, FamilyJudgments], *fields: str):
    """Validate ``judgments`` against a one-system export of ``body``."""
    return validate_answers(_pending(_view(body)), _mapping(judgments, *fields))


def _records(findings) -> list[tuple[str, bool]]:
    """Return each finding's key and whether the export cannot carry the answer."""
    return [(key, incompatible) for key, _message, incompatible in findings]


def _rows_beyond(**answers) -> FamilyJudgments:
    """The answers of a ``DCCT`` whose ``Monitor`` carries the rows."""
    return FamilyJudgments(rows_beyond={"Monitor": dict(answers)})


#: A one-device family whose ``Monitor`` carries one row beyond it.
_BODY = {"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": ["A", "B"]}}

#: The document path of that row's answer.
_KEY = "judgments.DCCT.rows_beyond_devices.Monitor[B]"


class TestAnsweredFamilies:
    """A mapping answering exactly what pends is silent."""

    def test_an_answered_row_is_silent(self):
        """The row the export pends is decided, so nothing is reported."""
        assert _validate(_BODY, {"DCCT": _rows_beyond(B="drop")}) == []

    def test_a_family_pending_nothing_needs_no_block(self):
        """An absent block is a problem only when something pends."""
        assert (
            _validate({"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["A", "B"]}}, {}) == []
        )


class TestMissingSlots:
    """A pending judgment the document leaves no slot for is reported at its key."""

    def test_a_missing_row_slot_names_its_system(self):
        """The whole entry: the row's key, the system it pends in, and its class."""
        assert _validate(_BODY, {}) == [(_KEY, "is pending in SR and has no answer", True)]

    def test_a_missing_ordinal_slot_is_reported(self):
        """A device no list reaches is asked about even with the block written."""
        body = {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["A", "B"]}}
        assert _records(_validate(body, {"DCCT": FamilyJudgments()})) == [
            ("judgments.DCCT.unbound_devices.3", True)
        ]

    def test_a_missing_supply_slot_is_reported(self):
        """A family sharing a PV is asked once, at the family's ``shared_pvs`` key."""
        body = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["P", "P"]}}
        assert _records(_validate(body, {})) == [("judgments.DCCT.shared_pvs", True)]

    def test_a_null_answer_is_a_slot(self):
        """An undecided answer fills the slot; the null domain reports it alone."""
        assert _validate(_BODY, {"DCCT": _rows_beyond(B=None)}) == []


class TestKeysTheExportDoesNotHave:
    """An answer naming something no system pends is refused and left out."""

    def test_an_unknown_family(self):
        """A family the export does not carry is reported once, at the family key."""
        judgments = {"DCCT": _rows_beyond(B="drop"), "GHOST": FamilyJudgments()}
        assert _records(_validate(_BODY, judgments)) == [("judgments.GHOST", True)]

    def test_an_unknown_signal(self):
        """A signal no row carries names no judgment."""
        judgments = {"DCCT": _rows_beyond(B="drop", C="drop")}
        assert _records(_validate(_BODY, judgments)) == [
            ("judgments.DCCT.rows_beyond_devices.Monitor[C]", True)
        ]

    def test_an_unknown_field(self):
        """A field the row does not sit in names no judgment."""
        judgments = {
            "DCCT": FamilyJudgments(
                rows_beyond={"Monitor": {"B": "drop"}, "Setpoint": {"B": "drop"}}
            )
        }
        assert _records(_validate(_BODY, judgments)) == [
            ("judgments.DCCT.rows_beyond_devices.Setpoint[B]", True)
        ]

    def test_an_unknown_ordinal(self):
        """A device every list reaches is not unbound."""
        judgments = {
            "DCCT": FamilyJudgments(
                rows_beyond={"Monitor": {"B": "drop"}}, unbound_devices={1: "drop"}
            )
        }
        assert _records(_validate(_BODY, judgments)) == [("judgments.DCCT.unbound_devices.1", True)]


class TestPerSystem:
    """The same family is judged per system, and a slot is reported once."""

    BODIES = (
        {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["A", "B"]}},
        {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["A", "B", "C"]}},
    )

    def _two_systems(self):
        return _pending(_view(self.BODIES[0], "LTB"), _view(self.BODIES[1], "SR"))

    def test_the_message_names_the_system_that_pends(self):
        """Only ``LTB`` leaves device 3 unbound, so the message says ``LTB``."""
        findings = validate_answers(self._two_systems(), _mapping({}))
        assert findings == [
            ("judgments.DCCT.unbound_devices.3", "is pending in LTB and has no answer", True)
        ]

    def test_an_answer_pending_in_one_system_is_accepted(self):
        """The system that does not pend the ordinal does not refuse the answer."""
        judgments = {"DCCT": FamilyJudgments(unbound_devices={3: "drop"})}
        assert validate_answers(self._two_systems(), _mapping(judgments)) == []

    def test_a_slot_missing_in_two_systems_is_reported_once(self):
        """One system-shaped message per key, from the first system that refuses."""
        pending = _pending(_view(self.BODIES[0], "LTB"), _view(self.BODIES[0], "SR"))
        findings = validate_answers(pending, _mapping({}))
        assert findings == [
            ("judgments.DCCT.unbound_devices.3", "is pending in LTB and has no answer", True)
        ]


class TestFieldNames:
    """A ``field:`` answer mints a name the family can carry."""

    def test_a_name_that_is_not_pn_local(self):
        """A name no IRI can carry is refused before it reaches the ontology."""
        judgments = {"DCCT": _rows_beyond(B=FieldAnswer(name="1bad"))}
        assert _records(_validate(_BODY, judgments)) == [(_KEY, True)]

    def test_a_name_the_family_already_carries(self):
        """The new field is a key of the body, so any key of it is taken."""
        judgments = {"DCCT": _rows_beyond(B=FieldAnswer(name="DeviceList"))}
        assert _records(_validate(_BODY, judgments)) == [(_KEY, True)]

    def test_a_name_answered_on_two_rows(self):
        """The collision is reported at the later row, in document order."""
        body = {"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": ["A", "B", "C"]}}
        judgments = {"DCCT": _rows_beyond(B=FieldAnswer(name="Extra"), C=FieldAnswer(name="Extra"))}
        assert _records(_validate(body, judgments, "Extra")) == [
            ("judgments.DCCT.rows_beyond_devices.Monitor[C]", True)
        ]

    def test_a_name_beside_another_kind_on_the_same_row(self):
        """One row answered two ways moves half of itself and cuts the other half."""
        body = {
            "DeviceList": [[1, 1]],
            "Monitor": {"ChannelNames": ["A", "B"], "TangoNames": ["a", "t"]},
        }
        judgments = {"DCCT": _rows_beyond(B=FieldAnswer(name="Extra"), t="drop")}
        assert _records(_validate(body, judgments, "Extra")) == [(_KEY, True)]

    def test_one_name_may_be_shared_by_the_keys_of_one_row(self):
        """The two channel keys of a row are one row, so one new field holds both."""
        body = {
            "DeviceList": [[1, 1]],
            "Monitor": {"ChannelNames": ["A", "B"], "TangoNames": ["a", "t"]},
        }
        judgments = {"DCCT": _rows_beyond(B=FieldAnswer(name="Extra"), t=FieldAnswer(name="Extra"))}
        assert _validate(body, judgments, "Extra") == []


class TestCreatedFields:
    """A ``field:`` answer needs the mapping entry for the field it mints."""

    JUDGMENTS = {"DCCT": _rows_beyond(B=FieldAnswer(name="Extra"))}

    def test_a_missing_entry_is_a_completeness_problem(self):
        """The answer stands; the mapping is the thing that is incomplete."""
        key, message, incompatible = _validate(_BODY, self.JUDGMENTS)[0]
        assert (key, incompatible) == (_KEY, False)
        assert message == (
            "creates the field 'Extra' of DCCT in SR; add families.DCCT.fields.Extra "
            "and directions.DCCT.Extra"
        )

    def test_a_described_field_is_silent(self):
        """With the entry written there is nothing left to ask for."""
        assert _validate(_BODY, self.JUDGMENTS, "Extra") == []

    def test_a_refused_name_is_not_also_asked_for(self):
        """An answer the export refuses mints no field, so it needs no entry."""
        judgments = {"DCCT": _rows_beyond(B=FieldAnswer(name="DeviceList"))}
        assert _records(_validate(_BODY, judgments)) == [(_KEY, True)]


class TestDeviceAnswers:
    """A row promoted to a device must not mint a supply nobody asked about."""

    def test_a_signal_bound_below_the_devices_is_refused(self):
        """Promoting the row would bind one PV to two devices of the family."""
        body = {"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": ["A", "A"]}}
        judgments = {"DCCT": _rows_beyond(A="device")}
        key = "judgments.DCCT.rows_beyond_devices.Monitor[A]"
        assert _records(_validate(body, judgments)) == [(key, True)]

    def test_a_signal_of_its_own_is_accepted(self):
        """A row carrying a PV no device below it holds becomes a device."""
        assert _validate(_BODY, {"DCCT": _rows_beyond(B="device")}) == []


class TestSharedPVs:
    """``keep_all`` is always accepted; an owner map is structural here."""

    SHARED = {
        "DeviceList": _rows(2),
        "Monitor": {"ChannelNames": ["P", "P"]},
        "Setpoint": {"ChannelNames": ["Q", "Q"]},
    }

    def test_keep_all_is_accepted(self):
        """The family keeps its supply on every member."""
        judgments = {"DCCT": FamilyJudgments(shared_pvs="keep_all", shared_pvs_present=True)}
        assert _validate(self.SHARED, judgments) == []

    def test_keep_all_on_a_family_pending_nothing_is_accepted(self):
        """A family answered once keeps its answer when another export shares nothing."""
        judgments = {"DCCT": FamilyJudgments(shared_pvs="keep_all", shared_pvs_present=True)}
        body = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["P", "Q"]}}
        assert _validate(body, judgments) == []

    def test_an_owner_stranding_a_member_is_legal(self):
        """Both members are group-only; owning the supply leaves one bound by nothing."""
        judgments = {
            "DCCT": FamilyJudgments(shared_pvs=OwnerMap(owners={1: 1}), shared_pvs_present=True)
        }
        assert _validate(self.SHARED, judgments) == []


class TestOwnerMaps:
    """An owner map names devices, so every name it carries must be the export's."""

    #: Two devices on one supply: one group, keyed by ordinal 1.
    ONE_GROUP = {
        "DeviceList": _rows(2),
        "Monitor": {"ChannelNames": ["P", "P"]},
        "Setpoint": {"ChannelNames": ["Q", "Q"]},
    }

    #: Four devices on two supplies: groups keyed by ordinals 1 and 3.
    TWO_GROUPS = {
        "DeviceList": _rows(4),
        "Monitor": {"ChannelNames": ["P", "P", "R", "R"]},
    }

    #: Three devices whose two supplies both reach device 2.
    OVERLAPPING = {
        "DeviceList": _rows(3),
        "Monitor": {"ChannelNames": ["P", "P", "X"]},
        "Setpoint": {"ChannelNames": ["Y", "Q", "Q"]},
    }

    #: Three devices whose supply starts at device 2, the wrapped fixture's shape.
    WRAPPED = {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["A", "B", "B"]}}

    @staticmethod
    def _owners(owners: dict) -> dict[str, FamilyJudgments]:
        """A ``DCCT`` answering its supply with ``owners``."""
        return {
            "DCCT": FamilyJudgments(shared_pvs=OwnerMap(owners=owners), shared_pvs_present=True)
        }

    def test_a_key_that_is_no_groups_lowest_ordinal(self):
        """A group is keyed by its lowest ordinal, so ordinal 5 keys nothing."""
        findings = _validate(self.ONE_GROUP, self._owners({1: 1, 5: 5}))
        assert _records(findings) == [("judgments.DCCT.shared_pvs.5", True)]

    def test_a_member_that_is_not_the_lowest_ordinal_keys_nothing(self):
        """Keying the group by its second member leaves its own key unanswered."""
        findings = _validate(self.ONE_GROUP, self._owners({2: 2}))
        assert _records(findings) == [
            ("judgments.DCCT.shared_pvs.2", True),
            ("judgments.DCCT.shared_pvs.1", True),
        ]

    def test_an_owner_outside_its_group(self):
        """Device 3 is not one of the two devices sharing the supply."""
        findings = _validate(self.ONE_GROUP, self._owners({1: 3}))
        assert _records(findings) == [("judgments.DCCT.shared_pvs.1", True)]
        assert "SR" in findings[0][1]

    def test_an_owner_map_missing_a_group(self):
        """The second supply is pending and the map decides nothing for it."""
        findings = _validate(self.TWO_GROUPS, self._owners({1: 1}))
        assert _records(findings) == [("judgments.DCCT.shared_pvs.3", True)]

    def test_a_group_answered_keep_all_is_decided(self):
        """``keep_all`` is an answer per group as much as per family."""
        assert _validate(self.TWO_GROUPS, self._owners({1: "keep_all", 3: 3})) == []

    def test_groups_sharing_a_device(self):
        """Two supplies reaching device 2 leave no key that says which devices it owns."""
        findings = _validate(self.OVERLAPPING, self._owners({1: 1, 2: 2}))
        assert _records(findings) == [("judgments.DCCT.shared_pvs", True)]
        assert "answer `keep_all`" in findings[0][1]

    def test_groups_that_differ_between_systems(self):
        """Ordinal 1 keys two devices in ``SR`` and three in ``BTS``."""
        wider = {"DeviceList": _rows(3), "Monitor": {"ChannelNames": ["P", "P", "P"]}}
        pending = _pending(_view(self.ONE_GROUP), _view(wider, system="BTS"))
        findings = validate_answers(pending, _mapping(self._owners({1: 1})))
        assert _records(findings) == [("judgments.DCCT.shared_pvs", True)]
        assert "answer `keep_all`" in findings[0][1]

    def test_a_collision_refuses_the_whole_answer_once(self):
        """The family's key takes the answer out, so its groups are not also picked over."""
        findings = _validate(self.OVERLAPPING, self._owners({1: 9}))
        assert _records(findings) == [("judgments.DCCT.shared_pvs", True)]

    def test_an_owner_map_on_a_family_sharing_nothing(self):
        """No export of the family shares a PV, so there is no supply to own."""
        body = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["P", "Q"]}}
        findings = _validate(body, self._owners({1: 1}))
        assert _records(findings) == [("judgments.DCCT.shared_pvs", True)]

    def test_a_group_starting_past_the_first_device(self):
        """The wrapped fixture's flip: the supply is keyed 2 and owned by device 2."""
        assert _validate(self.WRAPPED, self._owners({2: 2})) == []

    def test_an_answer_applying_in_one_system_only(self):
        """A supply one export shares is one device's own in another, and the map stands."""
        alone = {"DeviceList": _rows(2), "Monitor": {"ChannelNames": ["P", "Q"]}}
        pending = _pending(_view(self.ONE_GROUP), _view(alone, system="BTS"))
        assert validate_answers(pending, _mapping(self._owners({1: 1}))) == []


class TestMessages:
    """Every message says which system the judgment was found in."""

    @pytest.mark.parametrize(
        "judgments",
        [
            {},
            {"DCCT": _rows_beyond(B=FieldAnswer(name="1bad"))},
            {"DCCT": _rows_beyond(B=FieldAnswer(name="DeviceList"))},
            {"DCCT": _rows_beyond(B=FieldAnswer(name="Extra"))},
            {"DCCT": _rows_beyond(B="drop", C="drop")},
        ],
    )
    def test_a_message_names_a_system_or_says_there_is_none(self, judgments):
        """A judgment found in no system is the one message without a system name."""
        for _key, message, _incompatible in _validate(_BODY, judgments):
            assert "SR" in message or "in any system" in message
