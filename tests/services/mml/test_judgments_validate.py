"""Tests for the one home of the judgment answer rules.

``validate_answers`` reads a filled mapping back against what the raw export
actually pends, and returns one entry per answer the export cannot carry or the
mapping has yet to describe. ``map --check`` renders those entries and the emit
pre-flight refuses on them, so both lanes refuse exactly the same answers.

The answer vocabulary is structural and refused by ``parse_mapping`` long
before this module sees it; those refusals are pinned in
``test_mapping_schema.py``.

The virtual-accelerator slots are the same rules over the other half of the
document: the reviewer's answer is read back against the question the rules
asked and against the deck's own elements, so an answer naming an attribute no
bound element carries is refused where a row answer the export cannot carry is.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from osprey.services.mml.family import FamilyView
from osprey.services.mml.judgments import (
    PendingJudgments,
    VAPending,
    pending_judgments,
    unanswered_slots,
    validate_answers,
)
from osprey.services.mml.mapping.schema import (
    ATTYPE_KIND,
    ESCAPE_HATCH_KIND,
    SHARED_FIELD_KIND,
    Facility,
    Family,
    FamilyJudgments,
    Field,
    FieldAnswer,
    KickAnswer,
    Mapping,
    MonitorAnswer,
    OwnerAnswer,
    OwnerMap,
    StrengthAnswer,
    VAFamily,
    VASlot,
    VirtualAccelerator,
)
from tests.templates.mml_export_contract import VA_FAMILY_KEYS, VA_NOMINAL_KEYS


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


class _VAFixture:
    """The virtual-accelerator facts these tests hand ``validate_answers``.

    A block is spelled as a 2.0 export spells one, so the keys come from the
    frozen contract rather than from a second reading of the exporter.
    """

    FIELD = "Setpoint"
    NOMINAL = ([0.0], "m^-2", "quad", [[7]], [0])


def _element(name: str = "QF_1_1", pass_method: str = "StrMPoleSymplectic4Pass", **attributes):
    """One deck element, carrying only the attributes a write would reach for."""
    return SimpleNamespace(FamName=name, PassMethod=pass_method, **attributes)


def _family_block(at_type: str, at_index) -> dict:
    """One family of a virtual-accelerator export block."""
    values, units, _token, _index, synthetic = _VAFixture.NOMINAL
    nominal = dict(zip(VA_NOMINAL_KEYS, (values, units, at_type, at_index, synthetic), strict=True))
    return {_VAFixture.FIELD: {}, "nominals": {_VAFixture.FIELD: nominal}}


def _block(**families) -> dict:
    """One system's export block, each family given as ``(at_type, at_index)``."""
    return {
        "families": {
            name: _family_block(at_type, at_index) for name, (at_type, at_index) in families.items()
        }
    }


def _slot(kind: str = ATTYPE_KIND, answer=None) -> VASlot:
    return VASlot(kind=kind, question="what does this family drive?", answer=answer)


def _latched(slot: VASlot | None = None) -> VAFamily:
    """A family the rules leave to a reviewer, or settle on a reason of their own."""
    return VAFamily(verdict="latch", reason="a reason", slot=slot)


def _document(families: dict[str, VAFamily], system: str | None = "SR") -> Mapping:
    """A mapping whose virtual-accelerator block carries ``families``."""
    return replace(
        _mapping({}), virtual_accelerator=VirtualAccelerator(system=system, families=families)
    )


def _va(proposed: dict[str, VAFamily], block: dict, ring=(), system: str = "SR") -> VAPending:
    return VAPending(system=system, proposed=proposed, block=block, ring=tuple(ring))


def _va_findings(document: Mapping, va: VAPending):
    """Validate a virtual-accelerator document against what the rules ask."""
    return validate_answers({}, document, va=va)


#: A deck whose seventh element is the quadrupole the block binds.
_RING = tuple(_element(name=f"E{index}") for index in range(1, 7)) + (
    _element(PolynomB=[0.0, 1.2, 0.0]),
)

#: The export block that deck was sampled over.
_BLOCK = _block(QUAD=("quad", [[7]]))

#: The document path of the quadrupole's answer.
_VA_KEY = "virtual_accelerator.families.QUAD.slot.answer"


class TestVASlotsAreRead:
    """A virtual-accelerator answer is read back against the question the rules asked."""

    def test_va_a_family_the_export_does_not_carry(self):
        """The document decides a family no export of the system states."""
        document = _document({"SEXT": _latched(_slot())})
        findings = _va_findings(document, _va({"QUAD": _latched(_slot())}, _BLOCK, _RING))
        assert ("virtual_accelerator.families.SEXT", True) in _records(findings)

    def test_va_a_slot_the_rules_do_not_ask(self):
        """The rules settle the family, so its answer decides nothing."""
        document = _document({"QUAD": _latched(_slot(answer="latch"))})
        findings = _va_findings(document, _va({"QUAD": _latched()}, _BLOCK, _RING))
        assert _records(findings) == [("virtual_accelerator.families.QUAD.slot", True)]

    def test_va_a_pending_slot_the_document_leaves_out(self):
        """A question the document carries no slot for has no answer."""
        document = _document({"QUAD": _latched()})
        findings = _va_findings(document, _va({"QUAD": _latched(_slot())}, _BLOCK, _RING))
        assert _records(findings) == [("virtual_accelerator.families.QUAD.slot", True)]
        assert "is pending in SR and has no answer" in findings[0][1]

    def test_va_a_pending_family_the_document_leaves_out(self):
        """A family the document says nothing about is a question with no answer."""
        findings = _va_findings(_document({}), _va({"QUAD": _latched(_slot())}, _BLOCK, _RING))
        assert _records(findings) == [("virtual_accelerator.families.QUAD", True)]

    def test_va_a_family_no_rule_and_no_document_asks_about(self):
        """A family the rules settle and the document leaves alone is silent."""
        coupled = VAFamily(verdict="couple", kind="strength", element_field="PolynomB[1]")
        document = _document({"QUAD": coupled})
        assert _va_findings(document, _va({"QUAD": coupled}, _BLOCK, _RING)) == []

    def test_va_a_slot_of_another_kind(self):
        """An answer to a different question is no answer to the one asked."""
        document = _document({"QUAD": _latched(_slot(ESCAPE_HATCH_KIND, "ignore_hook"))})
        findings = _va_findings(document, _va({"QUAD": _latched(_slot())}, _BLOCK, _RING))
        assert _records(findings) == [("virtual_accelerator.families.QUAD.slot.kind", True)]
        assert ATTYPE_KIND in findings[0][1] and ESCAPE_HATCH_KIND in findings[0][1]


class TestVAElementAnswers:
    """An answer that binds an element is held to the deck's own elements."""

    def _findings(self, answer, ring=_RING, block=_BLOCK, kind: str = ATTYPE_KIND):
        document = _document({"QUAD": _latched(_slot(kind, answer))})
        return _va_findings(document, _va({"QUAD": _latched(_slot(kind))}, block, ring))

    def test_va_a_strength_the_element_carries(self):
        """``PolynomB[2]`` needs three coefficients, and the element has them."""
        assert self._findings(StrengthAnswer(attribute="PolynomB", index=2)) == []

    def test_va_a_strength_past_the_polynomial(self):
        """A sextupole coefficient on a two-coefficient element is nothing to write."""
        ring = (_element(PolynomB=[0.0, 1.2]),)
        findings = self._findings(
            StrengthAnswer(attribute="PolynomB", index=2), ring=ring, block=_block(QUAD=("quad", 1))
        )
        assert _records(findings) == [(_VA_KEY, True)]
        assert "PolynomB[2]" in findings[0][1]

    def test_va_a_strength_attribute_the_element_has_not_got(self):
        """A skew answer on an element carrying no ``PolynomA`` is refused by name."""
        findings = self._findings(StrengthAnswer(attribute="PolynomA", index=1))
        assert _records(findings) == [(_VA_KEY, True)]
        assert "PolynomA[1]" in findings[0][1]

    def test_va_a_kick_the_element_takes(self):
        """A corrector answer needs a ``KickAngle`` of the answered plane."""
        ring = (_element(KickAngle=[0.0, 0.0]),)
        assert self._findings(KickAnswer(plane=1), ring=ring, block=_block(QUAD=("hcm", 1))) == []

    def test_va_a_kick_with_no_kick_angle(self):
        """The quadrupole element takes no kick, whatever the reviewer answers."""
        findings = self._findings(KickAnswer(plane=0))
        assert _records(findings) == [(_VA_KEY, True)]
        assert "KickAngle[0]" in findings[0][1]

    def test_va_a_kick_plane_past_the_kick_angle(self):
        """A one-plane corrector cannot be driven in the other plane."""
        ring = (_element(KickAngle=[0.0]),)
        findings = self._findings(KickAnswer(plane=1), ring=ring, block=_block(QUAD=("hcm", 1)))
        assert _records(findings) == [(_VA_KEY, True)]

    def test_va_a_monitor_needs_only_an_element(self):
        """A reading binds no attribute of its element, so only the index is checked."""
        ring = (_element(name="BPM_1_1", pass_method="IdentityPass"),)
        assert (
            self._findings(MonitorAnswer(plane="x"), ring=ring, block=_block(QUAD=("bpmx", 1)))
            == []
        )

    def test_va_an_index_past_the_end_of_the_deck(self):
        """An index the deck does not reach is refused before any attribute is read."""
        findings = self._findings(StrengthAnswer(attribute="PolynomB", index=1), ring=_RING[:3])
        assert _records(findings) == [(_VA_KEY, True)]
        assert "past the end of a ring of 3" in findings[0][1]

    def test_va_an_element_answer_on_a_family_binding_nothing(self):
        """A family stating no index binds no element for the answer to write."""
        findings = self._findings(
            StrengthAnswer(attribute="PolynomB", index=1), block=_block(QUAD=("quad", "NaN"))
        )
        assert _records(findings) == [(_VA_KEY, True)]

    def test_va_every_bound_element_is_checked(self):
        """The refusal is the second element's, so the first one passing is not enough."""
        ring = (_element(PolynomB=[0.0, 1.2]), _element(name="QD_1_1", PolynomB=[0.0]))
        findings = self._findings(
            StrengthAnswer(attribute="PolynomB", index=1),
            ring=ring,
            block=_block(QUAD=("quad", [[1], [2]])),
        )
        assert _records(findings) == [(_VA_KEY, True)]
        assert "QD_1_1" in findings[0][1]

    @pytest.mark.parametrize("answer", ["latch", "energy", "rf"])
    def test_va_an_answer_binding_no_element(self, answer):
        """The energy knob, the cavity and a latch are decided without the deck."""
        assert self._findings(answer, ring=(), block=_block(QUAD=("bend", "NaN"))) == []

    @pytest.mark.parametrize("answer", ["latch", "ignore_hook"])
    def test_va_an_escape_hatch_answer(self, answer):
        """Both hook answers are the reviewer's alone; no element decides them."""
        assert self._findings(answer, ring=(), kind=ESCAPE_HATCH_KIND) == []


class TestVAOwnerAnswers:
    """``owner:<family>`` names which family of a collision binds the field."""

    BLOCK = _block(HCM=("hcm", [[3]]), TRIM=("hcor", [[3]]), QUAD=("quad", [[7]]))
    RING = tuple(_element(name=f"E{index}", KickAngle=[0.0, 0.0]) for index in range(1, 8))

    def _findings(self, family: str):
        """Validate ``owner:<family>`` on the ``HCM`` the rules left to a reviewer."""
        slot = _slot(SHARED_FIELD_KIND, OwnerAnswer(family=family))
        document = _document({"HCM": _latched(slot)})
        settled = VAFamily(verdict="couple", kind="kick", element_field="KickAngle[0]")
        proposed = {"HCM": _latched(_slot(SHARED_FIELD_KIND)), "TRIM": settled, "QUAD": settled}
        return _va_findings(document, _va(proposed, self.BLOCK, self.RING))

    def test_va_an_owner_in_the_collision(self):
        """``HCM`` and ``TRIM`` both drive ``KickAngle[0]`` of element 3."""
        assert self._findings("TRIM") == []

    def test_va_an_owner_that_is_the_answering_family(self):
        """A family may answer that it binds the field itself."""
        assert self._findings("HCM") == []

    def test_va_an_owner_outside_the_collision(self):
        """The quadrupole drives another element's field, so it owns nothing here."""
        findings = self._findings("QUAD")
        assert _records(findings) == [("virtual_accelerator.families.HCM.slot.answer", True)]
        assert "QUAD" in findings[0][1]

    def test_va_an_owner_the_export_does_not_carry(self):
        """A named family the export does not state cannot own anything it states."""
        findings = self._findings("SEXT")
        assert _records(findings) == [("virtual_accelerator.families.HCM.slot.answer", True)]
        assert "SEXT" in findings[0][1]


class TestVAUnansweredSlots:
    """A null virtual-accelerator slot decides nothing, exactly like a null judgment."""

    def test_va_a_document_with_no_block(self):
        """A mapping carrying no block has no virtual-accelerator slot to answer."""
        assert unanswered_slots(_mapping({})) == []

    def test_va_a_null_slot_answer(self):
        """The slot the reviewer has yet to answer is reported at its answer key."""
        assert unanswered_slots(_document({"QUAD": _latched(_slot())})) == [
            (_VA_KEY, "must not be null")
        ]

    def test_va_an_answered_slot(self):
        """An answered slot is a decision, not a question."""
        assert unanswered_slots(_document({"QUAD": _latched(_slot(answer="latch"))})) == []

    def test_va_a_family_with_no_slot(self):
        """A family the rules settled asks the reviewer nothing."""
        assert unanswered_slots(_document({"QUAD": _latched()})) == []

    def test_va_a_null_system(self):
        """Which system the block describes is the reviewer's first answer."""
        assert unanswered_slots(_document({}, system=None)) == [
            ("virtual_accelerator.system", "must not be null")
        ]

    def test_va_slots_come_after_the_judgment_slots(self):
        """Both kinds are read off one document, the judgments in document order first."""
        document = replace(
            _document({"QUAD": _latched(_slot())}), judgments={"DCCT": _rows_beyond(B=None)}
        )
        assert [key for key, _message in unanswered_slots(document)] == [_KEY, _VA_KEY]


class TestVAFindingsJoinTheJudgmentFindings:
    """One call validates both halves of one document."""

    def test_va_both_kinds_of_answer_in_one_call(self):
        """A refused row answer and a refused answer of the deck come back together."""
        answer = StrengthAnswer(attribute="PolynomA", index=1)
        document = replace(
            _document({"QUAD": _latched(_slot(answer=answer))}),
            judgments={"DCCT": _rows_beyond(B="drop", Z="drop")},
        )
        findings = validate_answers(
            _pending(_view(_BODY)), document, va=_va({"QUAD": _latched(_slot())}, _BLOCK, _RING)
        )
        assert _records(findings) == [
            ("judgments.DCCT.rows_beyond_devices.Monitor[Z]", True),
            (_VA_KEY, True),
        ]

    def test_va_no_facts_is_no_virtual_accelerator_finding(self):
        """Without the facts, a document's virtual-accelerator half is not judged."""
        assert validate_answers({}, _document({"QUAD": _latched(_slot())})) == []

    def test_va_every_message_names_the_system(self):
        """A virtual-accelerator refusal says which system the question was found in."""
        document = _document({"SEXT": _latched(_slot(answer="latch"))})
        findings = _va_findings(document, _va({"QUAD": _latched(_slot())}, _BLOCK, _RING))
        assert findings and all("SR" in message for _key, message, _flag in findings)

    def test_va_the_fixture_is_spelled_as_the_export_spells_one(self):
        """These blocks are the 2.0 contract's keys, not a second reading of the exporter."""
        family = _BLOCK["families"]["QUAD"]
        assert set(family) <= set(VA_FAMILY_KEYS)
        assert tuple(family["nominals"][_VAFixture.FIELD]) == VA_NOMINAL_KEYS
