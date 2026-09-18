"""Semantic checks on a parsed ``mapping.yaml`` against the export it maps.

``check_mapping`` is the fail-closed gate between a filled mapping and
``emit``. Every test starts from one complete, passing mapping and breaks it in
exactly one place, then asserts the problem names that place's key path.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from osprey.services.mml.directions import vote_directions
from osprey.services.mml.judgments import VAPending
from osprey.services.mml.mapping import ATTYPE_KIND, VAFamily, VASlot, parse_mapping
from osprey.services.mml.mapping import check as check_module
from osprey.services.mml.mapping.branches import ROOT_CLASS
from osprey.services.mml.mapping.check import CheckResult, Problem, VAExport, check_mapping
from tests.templates.mml_export_contract import VA_NOMINAL_KEYS


def _ao() -> dict:
    """A two-system merged export with invented names."""
    return {
        "_import_order": ["SR", "BTS"],
        "_exports": {"SR": {}, "BTS": {}},
        "SR": {
            "_description": "Storage ring.",
            "BPM": {
                "DeviceList": [[1, 1], [1, 2]],
                "Monitor": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"], "MemberOf": ["Monitor"]},
            },
            "HCM": {
                "DeviceList": [[1, 1]],
                "Setpoint": {"ChannelNames": ["SR:HCM:SP"], "MemberOf": ["Setpoint"]},
                "Monitor": {"ChannelNames": ["SR:HCM:RB"], "MemberOf": ["Monitor"]},
            },
            "Empty": {"DeviceList": [[1, 1]]},
        },
        "BTS": {
            "HCM": {
                "DeviceList": [[1, 1]],
                "Setpoint": {"ChannelNames": ["BTS:HCM:SP"], "MemberOf": ["Setpoint"]},
                "RampRate": {"ChannelNames": ["BTS:HCM:RR"]},
            },
        },
    }


def _document() -> dict:
    """A complete mapping of :func:`_ao` that passes every check."""
    return {
        "facility": {
            "token": "Zephyr",
            "title": "ZEPHYR",
            "description": "A synthetic ring.",
            "provenance": "stated",
        },
        "systems": {
            "SR": {"name": "SR", "description": "Storage ring.", "provenance": "imported"},
            "BTS": {"name": "BTS", "description": "Transfer line.", "provenance": "stated"},
        },
        "section_order": ["SR", "BTS"],
        "branches": {
            "PulsedMagnet": {"parent": "Magnet", "description": "Pulsed magnets."},
            "Kicker": {"parent": "PulsedMagnet", "description": "Kickers."},
        },
        "families": {
            "BPM": {
                "class": "BeamPositionMonitor",
                "aliases": ["BPM"],
                "description": "Beam position monitors.",
                "provenance": "stated",
                "channels": 2,
                "fields": {"Monitor": {"description": "Position.", "provenance": "stated"}},
            },
            "HCM": {
                "branch": "Corrector",
                "class": "ZephyrCorrector",
                "aliases": ["HCM"],
                "description": "Horizontal correctors.",
                "provenance": "stated",
                "channels": 4,
                "fields": {
                    "Setpoint": {"description": "Current setpoint.", "provenance": "stated"},
                    "Monitor": {"description": "Current readback.", "provenance": "stated"},
                    "RampRate": {"description": "Ramp rate.", "provenance": "stated"},
                },
            },
            "Empty": {
                "aliases": ["Empty"],
                "description": "A family with no channels.",
                "provenance": "stated",
                "channels": 0,
                "fields": {},
            },
        },
        "directions": {
            "BPM.Monitor": {"direction": "read", "provenance": "stated"},
            "HCM.Setpoint": {"direction": "write", "provenance": "stated"},
            "HCM.Monitor": {"direction": "read", "provenance": "stated"},
            "HCM.RampRate": {"direction": "write", "provenance": "stated"},
        },
    }


def _check(
    doc: dict | None = None,
    ao: dict | None = None,
    *,
    no_derived: bool = False,
    va=None,
):
    doc = _document() if doc is None else doc
    ao = _ao() if ao is None else ao
    return check_mapping(parse_mapping(doc), ao, vote_directions(ao), no_derived=no_derived, va=va)


def _keys(result: CheckResult) -> list[str]:
    return [problem.key for problem in result.problems]


def _mutated(path: str, value) -> dict:
    """Return the passing document with ``path`` set to ``value`` (``...`` deletes)."""
    doc = copy.deepcopy(_document())
    *parents, last = path.split("/")
    node = doc
    for part in parents:
        node = node[part]
    if value is ...:
        del node[last]
    else:
        node[last] = value
    return doc


def _judged_ao() -> dict:
    """:func:`_ao` plus the two families a reviewer has to judge.

    ``DCCT`` sits on one device and carries a second ``Monitor`` row, the shape
    of a row beyond a family's devices. ``BEND`` lives in both systems with four
    devices: ``SR`` binds all four, ``BTS`` reaches three, so ordinal 4 is
    unbound in ``BTS`` alone. ``QF`` binds its two devices to one supply, the
    shape of a shared PV.
    """
    ao = _ao()
    ao["SR"]["DCCT"] = {
        "DeviceList": [[1, 1]],
        "Monitor": {"ChannelNames": ["SR:DCCT:I", "SR:DCCT:Lifetime"], "MemberOf": ["Monitor"]},
    }
    ao["SR"]["BEND"] = {
        "DeviceList": [[1, 1], [1, 2], [1, 3], [1, 4]],
        "Monitor": {"ChannelNames": [f"SR:BEND{n}:I" for n in (1, 2, 3, 4)]},
    }
    ao["BTS"]["BEND"] = {
        "DeviceList": [[1, 1], [1, 2], [1, 3], [1, 4]],
        "Monitor": {"ChannelNames": [f"BTS:BEND{n}:I" for n in (1, 2, 3)]},
    }
    ao["SR"]["QF"] = {
        "DeviceList": [[1, 1], [1, 2]],
        "Monitor": {"ChannelNames": ["SR:QF:I", "SR:QF:I"]},
        "Setpoint": {"ChannelNames": ["SR:QF:SP", "SR:QF:SP"]},
    }
    return ao


def _judged_document(judgments: dict) -> dict:
    """A complete mapping of :func:`_judged_ao` carrying ``judgments``."""
    doc = _document()
    doc["families"]["DCCT"] = {
        "branch": "Instrumentation",
        "class": "ZephyrDCCT",
        "aliases": ["DCCT"],
        "description": "Beam current monitor.",
        "provenance": "stated",
        "channels": 2,
        "fields": {"Monitor": {"description": "Beam current.", "provenance": "stated"}},
    }
    doc["families"]["BEND"] = {
        "branch": "Magnet",
        "class": "ZephyrBend",
        "aliases": ["BEND"],
        "description": "Dipoles.",
        "provenance": "stated",
        "channels": 7,
        "fields": {"Monitor": {"description": "Current readback.", "provenance": "stated"}},
    }
    doc["families"]["QF"] = {
        "branch": "Magnet",
        "class": "ZephyrQuad",
        "aliases": ["QF"],
        "description": "Focusing quadrupoles.",
        "provenance": "stated",
        "channels": 4,
        "fields": {
            "Monitor": {"description": "Current readback.", "provenance": "stated"},
            "Setpoint": {"description": "Current setpoint.", "provenance": "stated"},
        },
    }
    doc["directions"]["DCCT.Monitor"] = {"direction": "read", "provenance": "stated"}
    doc["directions"]["BEND.Monitor"] = {"direction": "read", "provenance": "stated"}
    doc["directions"]["QF.Monitor"] = {"direction": "read", "provenance": "stated"}
    doc["directions"]["QF.Setpoint"] = {"direction": "write", "provenance": "stated"}
    doc["judgments"] = judgments
    return doc


def _context(doc: dict, ao: dict):
    """The context the predicates read, built as ``check_mapping`` builds it."""
    return check_module._Context(parse_mapping(doc), ao, vote_directions(ao))


def _judged(ctx, system: str, family: str):
    """Return one judged family view of ``ctx``."""
    return next(view for view in ctx.judged_views[system] if view.raw_name == family)


#: The signal of the ``DCCT`` row beyond its one device.
_ROW = "SR:DCCT:Lifetime"

#: The document path of that row's answer.
_ROW_KEY = f"judgments.DCCT.rows_beyond_devices.Monitor[{_ROW}]"

#: One answer for every judgment :func:`_judged_ao` pends.
_ANSWERED: dict = {
    "DCCT": {"rows_beyond_devices": {"Monitor": {_ROW: "drop"}}},
    "BEND": {"unbound_devices": {4: "drop"}},
    "QF": {"shared_pvs": "keep_all"},
}


def _answers(**changes) -> dict:
    """Return the answered block with families replaced (``...`` deletes one)."""
    judgments = copy.deepcopy(_ANSWERED)
    for family, block in changes.items():
        if block is ...:
            del judgments[family]
        else:
            judgments[family] = block
    return judgments


class TestPassingMapping:
    """The complete mapping passes and reports its counts."""

    def test_no_problems(self):
        """A fully stated mapping yields no problems."""
        result = _check()
        assert result.problems == []

    def test_no_problems_with_no_derived(self):
        """``no_derived`` adds nothing when nothing is derived."""
        assert _check(no_derived=True).problems == []

    def test_counts_are_zero(self):
        """Every derived count is zero for a stated mapping."""
        result = _check()
        assert result.derived_descriptions == {"facility": 0, "system": 0, "family": 0, "field": 0}
        assert result.derived_directions == 0

    def test_problem_prints_like_mapping_error(self):
        """A problem prints as ``<key>: <message>``."""
        assert str(Problem("facility.token", "must not be null")) == (
            "facility.token: must not be null"
        )

    def test_zero_channel_family_without_class_passes(self):
        """A zero-channel family with no ``branch``/``class`` keys is fine."""
        doc = _document()
        assert "class" not in doc["families"]["Empty"]
        assert _check(doc).problems == []

    def test_branch_equal_to_packaged_parent_passes(self):
        """A packaged class may state its packaged parent as ``branch``."""
        doc = _mutated("families/BPM/branch", "Instrumentation")
        assert _check(doc).problems == []

    def test_undecided_vote_accepts_stated_direction(self):
        """``RampRate`` is undecided, so either stated direction passes."""
        doc = _mutated("directions/HCM.RampRate/direction", "read")
        assert _check(doc).problems == []

    def test_new_class_branch_may_be_root_or_declared(self):
        """A new class may extend the root or a declared branch."""
        doc = _mutated("families/HCM/branch", "Kicker")
        assert _check(doc).problems == []
        doc = _mutated("families/HCM/branch", ROOT_CLASS)
        assert _check(doc).problems == []

    def test_shared_new_class_with_same_branch_passes(self):
        """Two families may share a new class under one branch."""
        doc = _document()
        doc["families"]["BPM"]["class"] = "ZephyrCorrector"
        doc["families"]["BPM"]["branch"] = "Corrector"
        assert _check(doc).problems == []


class TestNullDomain:
    """A ``null`` in any undecided slot is refused at its key."""

    @pytest.mark.parametrize(
        ("path", "key"),
        [
            ("facility/token", "facility.token"),
            ("facility/description", "facility.description"),
            ("systems/BTS/description", "systems.BTS.description"),
            ("families/BPM/description", "families.BPM.description"),
            (
                "families/HCM/fields/Setpoint/description",
                "families.HCM.fields.Setpoint.description",
            ),
            ("families/BPM/class", "families.BPM.class"),
            ("families/HCM/branch", "families.HCM.branch"),
            ("directions/BPM.Monitor/direction", "directions.BPM.Monitor.direction"),
        ],
    )
    def test_null_refused(self, path, key):
        """The null slot is the one reported problem."""
        assert _keys(_check(_mutated(path, None))) == [key]

    def test_packaged_class_needs_no_branch(self):
        """A packaged class with ``branch`` omitted is not a null problem."""
        assert "branch" not in _document()["families"]["BPM"]
        assert _check().problems == []

    def test_zero_channel_family_null_class_passes(self):
        """``channels: 0`` takes ``class``/``branch`` out of the null domain."""
        doc = _mutated("families/Empty/class", None)
        assert _check(doc).problems == []


class TestFacilityToken:
    """The facility token is a non-null PN_LOCAL."""

    def test_null_token_reported_once(self):
        """A null token is one problem, not a null and a PN_LOCAL problem."""
        assert _keys(_check(_mutated("facility/token", None))) == ["facility.token"]

    @pytest.mark.parametrize("token", ["Zephyr Ring", "1Zephyr", "Ze-phyr", ""])
    def test_non_pn_local_refused(self, token):
        """A token that cannot be a Turtle local name is refused."""
        assert _keys(_check(_mutated("facility/token", token))) == ["facility.token"]


class TestUnknownFamilies:
    """``families:`` and ``ao.json`` name the same families."""

    def test_mapping_family_absent_from_ao(self):
        """A ``families:`` key with no ``ao.json`` family is refused."""
        doc = _document()
        doc["families"]["Ghost"] = copy.deepcopy(doc["families"]["Empty"])
        doc["families"]["Ghost"]["aliases"] = ["Ghost"]
        assert _keys(_check(doc)) == ["families.Ghost"]

    def test_ao_family_absent_from_mapping(self):
        """An ``ao.json`` family with no ``families:`` key is refused."""
        doc = _mutated("families/Empty", ...)
        assert _keys(_check(doc)) == ["families.Empty"]

    def test_ao_system_absent_from_mapping(self):
        """An ``ao.json`` system with no ``systems:`` key is refused."""
        ao = _ao()
        ao["LTB"] = {"Empty": {"DeviceList": [[1, 1]]}}
        ao["_import_order"].append("LTB")
        assert _keys(_check(ao=ao)) == ["systems.LTB"]

    def test_mapping_system_absent_from_ao(self):
        """A ``systems:`` key with no ``ao.json`` system is refused."""
        doc = _document()
        doc["systems"]["LTB"] = {"name": "LTB", "description": "x", "provenance": "stated"}
        doc["section_order"].append("LTB")
        assert _keys(_check(doc)) == ["systems.LTB"]


class TestBranchAndClass:
    """A family's class and branch name real, consistent ontology classes."""

    def test_branch_nowhere(self):
        """A branch naming neither a packaged class nor a declared branch is refused."""
        assert _keys(_check(_mutated("families/HCM/branch", "Nowhere"))) == ["families.HCM.branch"]

    @pytest.mark.parametrize("klass", ["Zephyr Corrector", "9Corrector", "Zephyr-Corrector"])
    def test_class_not_pn_local(self, klass):
        """A class that is not PN_LOCAL is refused."""
        assert _keys(_check(_mutated("families/HCM/class", klass))) == ["families.HCM.class"]

    def test_class_is_root(self):
        """The root class cannot type a family."""
        assert _keys(_check(_mutated("families/HCM/class", ROOT_CLASS))) == ["families.HCM.class"]

    def test_packaged_class_with_other_branch(self):
        """A packaged class given a branch other than its packaged parent is refused."""
        assert _keys(_check(_mutated("families/BPM/branch", "Magnet"))) == ["families.BPM.branch"]

    def test_shared_new_class_with_two_branches(self):
        """A new class shared by two families under different branches is refused."""
        doc = _document()
        doc["families"]["BPM"]["class"] = "ZephyrCorrector"
        doc["families"]["BPM"]["branch"] = "Magnet"
        assert _keys(_check(doc)) == ["families.HCM.branch"]

    def test_rename_not_pn_local(self):
        """A ``rename`` that is not PN_LOCAL is refused."""
        assert _keys(_check(_mutated("families/HCM/rename", "H CM"))) == ["families.HCM.rename"]


class TestDirections:
    """Every signal group has a direction key, and every key names a real group."""

    def test_missing_direction_group(self):
        """A channel-bearing field with no ``directions`` key is refused."""
        doc = _mutated("directions/HCM.Monitor", ...)
        assert _keys(_check(doc)) == ["directions.HCM.Monitor"]

    def test_missing_group_reported_once_across_systems(self):
        """A group carried by two systems is reported once."""
        doc = _mutated("directions/HCM.Setpoint", ...)
        assert _keys(_check(doc)) == ["directions.HCM.Setpoint"]

    def test_key_names_unknown_family(self):
        """A ``directions`` key whose family is not in ``families:`` is refused."""
        doc = _document()
        doc["directions"]["Ghost.Monitor"] = {"direction": "read", "provenance": "stated"}
        assert _keys(_check(doc)) == ["directions.Ghost.Monitor"]

    def test_key_names_absent_field(self):
        """A ``directions`` key naming a field absent in ``ao.json`` is refused."""
        doc = _document()
        doc["directions"]["BPM.Setpoint"] = {"direction": "write", "provenance": "stated"}
        assert _keys(_check(doc)) == ["directions.BPM.Setpoint"]

    def test_stated_disagrees_with_decided_vote(self):
        """A stated direction against a decided vote is refused."""
        doc = _mutated("directions/BPM.Monitor/direction", "write")
        assert _keys(_check(doc)) == ["directions.BPM.Monitor"]

    def test_override_accepts_disagreement(self):
        """``override: true`` accepts a stated direction against the vote."""
        doc = _mutated("directions/BPM.Monitor/direction", "write")
        doc["directions"]["BPM.Monitor"]["override"] = True
        assert _check(doc).problems == []

    def test_derived_disagreement_is_not_the_stated_rule(self):
        """Only a ``stated`` direction is compared to the vote."""
        doc = _mutated("directions/BPM.Monitor/direction", "write")
        doc["directions"]["BPM.Monitor"]["provenance"] = "derived"
        assert _check(doc).problems == []


class TestSystemNames:
    """System names are PN_LOCAL and distinct after ``lower()``."""

    @pytest.mark.parametrize("name", ["S R", "1SR", "S-R"])
    def test_non_pn_local(self, name):
        """A system name that is not PN_LOCAL is refused."""
        doc = _mutated("systems/SR/name", name)
        doc["section_order"][0] = name
        assert _keys(_check(doc)) == ["systems.SR.name"]

    def test_case_fold_collision(self):
        """Two system names differing only by case are refused."""
        doc = _mutated("systems/BTS/name", "sr")
        doc["section_order"][1] = "sr"
        assert _keys(_check(doc)) == ["systems.BTS.name"]


class TestFamilyPrefixes:
    """Mapped family tokens stay distinct as signal prefixes."""

    def _case_duplicate(self) -> tuple[dict, dict]:
        ao = _ao()
        ao["BTS"]["Hcm"] = copy.deepcopy(ao["BTS"]["HCM"])
        doc = _document()
        doc["families"]["Hcm"] = copy.deepcopy(doc["families"]["HCM"])
        doc["families"]["Hcm"]["aliases"] = ["Hcm"]
        for field in ("Setpoint", "RampRate"):
            doc["directions"][f"Hcm.{field}"] = copy.deepcopy(doc["directions"][f"HCM.{field}"])
        return doc, ao

    def test_case_fold_collision(self):
        """Two raw families folding to one prefix without ``rename`` are refused."""
        doc, ao = self._case_duplicate()
        assert _keys(_check(doc, ao)) == ["families.Hcm"]

    def test_rename_resolves_collision(self):
        """``rename`` on one of the pair resolves the collision."""
        doc, ao = self._case_duplicate()
        doc["families"]["Hcm"]["rename"] = "HCM2"
        assert _check(doc, ao).problems == []

    def test_rename_case_fold_collision(self):
        """A rename folding onto another mapped token is refused at the later family."""
        doc = _mutated("families/BPM/rename", "hcm")
        assert _keys(_check(doc)) == ["families.HCM"]

    def test_hyphen_folds_to_underscore(self):
        """A raw ``-`` token collides with a mapped ``_`` token."""
        ao = _ao()
        ao["SR"]["H-CM"] = {"DeviceList": [[1, 1]]}
        doc = _mutated("families/HCM/rename", "H_CM")
        doc["families"]["H-CM"] = copy.deepcopy(doc["families"]["Empty"])
        doc["families"]["H-CM"]["aliases"] = ["H-CM"]
        assert _keys(_check(doc, ao)) == ["families.H-CM"]


class TestBranches:
    """Declared branches extend the packaged table without shadowing or looping."""

    @pytest.mark.parametrize("name", ["Magnet", ROOT_CLASS])
    def test_named_like_packaged_class(self, name):
        """A branch named like a packaged class, the root included, is refused."""
        doc = _document()
        doc["branches"][name] = {"parent": "Instrumentation", "description": None}
        assert _keys(_check(doc)) == [f"branches.{name}"]

    def test_undeclared_parent(self):
        """A branch whose parent names nothing known is refused."""
        doc = _mutated("branches/PulsedMagnet/parent", "Nowhere")
        assert _keys(_check(doc)) == ["branches.PulsedMagnet.parent"]

    def test_cycle(self):
        """Branches that form a cycle are refused."""
        doc = _document()
        doc["branches"] = {
            "A": {"parent": "B", "description": None},
            "B": {"parent": "A", "description": None},
        }
        assert _keys(_check(doc)) == ["branches.A.parent", "branches.B.parent"]

    def test_self_cycle(self):
        """A branch that is its own parent is a cycle."""
        doc = _mutated("branches/PulsedMagnet/parent", "PulsedMagnet")
        assert "branches.PulsedMagnet.parent" in _keys(_check(doc))


class TestSectionOrder:
    """``section_order`` is a permutation of the mapped system names."""

    def test_missing_name(self):
        """A system name absent from ``section_order`` is refused."""
        assert _keys(_check(_mutated("section_order", ["SR"]))) == ["section_order"]

    def test_duplicate(self):
        """A repeated entry is refused at its index."""
        assert _keys(_check(_mutated("section_order", ["SR", "BTS", "SR"]))) == ["section_order[2]"]

    def test_unknown(self):
        """An entry that is not a system name is refused at its index."""
        doc = _mutated("section_order", ["SR", "LTB"])
        assert _keys(_check(doc)) == ["section_order[1]", "section_order"]

    def test_raw_token_is_not_the_name(self):
        """Entries are mapped names, not raw system tokens."""
        doc = _mutated("systems/SR/name", "Ring")
        assert _keys(_check(doc)) == ["section_order[0]", "section_order"]


class TestDerived:
    """Derived provenance is counted, and ``no_derived`` refuses it."""

    def _derived(self) -> dict:
        doc = _document()
        doc["facility"]["provenance"] = "derived"
        doc["systems"]["BTS"]["provenance"] = "derived"
        doc["families"]["BPM"]["provenance"] = "derived"
        doc["families"]["HCM"]["provenance"] = "derived"
        doc["families"]["HCM"]["fields"]["RampRate"]["provenance"] = "derived"
        doc["directions"]["HCM.Monitor"]["provenance"] = "derived"
        doc["directions"]["BPM.Monitor"]["provenance"] = "derived"
        return doc

    def test_counts_per_level(self):
        """Derived descriptions are counted per level, directions separately."""
        result = _check(self._derived())
        assert result.derived_descriptions == {"facility": 1, "system": 1, "family": 2, "field": 1}
        assert result.derived_directions == 2

    def test_counts_alone_are_not_problems(self):
        """Without ``no_derived`` derived slots only count."""
        assert _check(self._derived()).problems == []

    def test_imported_is_not_derived(self):
        """``imported`` descriptions do not count as derived."""
        assert _check().derived_descriptions["system"] == 0

    def test_no_derived_refuses_each_slot(self):
        """``no_derived`` reports every derived slot at its key."""
        result = _check(self._derived(), no_derived=True)
        assert _keys(result) == [
            "facility.provenance",
            "systems.BTS.provenance",
            "families.BPM.provenance",
            "families.HCM.provenance",
            "families.HCM.fields.RampRate.provenance",
            "directions.BPM.Monitor.provenance",
            "directions.HCM.Monitor.provenance",
        ]

    def test_result_is_frozen(self):
        """The result record is immutable."""
        result = _check()
        with pytest.raises(AttributeError):
            result.derived_directions = 3  # type: ignore[misc]


class TestJudgmentNullDomain:
    """An undecided judgment answer is refused at its bracketed key."""

    @pytest.mark.parametrize(
        ("judgments", "key"),
        [
            (_answers(DCCT={"rows_beyond_devices": {"Monitor": {_ROW: None}}}), _ROW_KEY),
            (_answers(BEND={"unbound_devices": {4: None}}), "judgments.BEND.unbound_devices.4"),
            (_answers(QF={"shared_pvs": None}), "judgments.QF.shared_pvs"),
        ],
    )
    def test_null_answer_refused(self, judgments, key):
        """The undecided answer is the one reported problem."""
        assert _keys(_check(_judged_document(judgments), _judged_ao())) == [key]

    def test_answered_judgments_pass(self):
        """A family answering every slot it carries is no null problem."""
        assert _check(_judged_document(_answers()), _judged_ao()).problems == []

    def test_absent_shared_slot_is_not_null(self):
        """A family carrying no ``shared_pvs`` key is not an undecided answer."""
        judgments = _answers()
        assert "shared_pvs" not in judgments["DCCT"]
        assert _check(_judged_document(judgments), _judged_ao()).problems == []


class TestJudgedGrain:
    """Every predicate reads the export the answers describe, not the raw one."""

    @staticmethod
    def _field_answer() -> dict:
        return _answers(DCCT={"rows_beyond_devices": {"Monitor": {_ROW: {"field": "Lifetime"}}}})

    def test_created_field_needs_a_directions_entry(self):
        """A ``field:`` answer mints a signal group like any exported field.

        The answer stands, so the mapping is asked for the field entry once and
        for the directions key by the rule that asks for every other field's.
        """
        doc = _judged_document(self._field_answer())
        assert _keys(_check(doc, _judged_ao())) == [_ROW_KEY, "directions.DCCT.Lifetime"]

    def test_created_field_takes_a_stated_direction_without_override(self):
        """The new field has no vote to disagree with, so ``override`` is not needed."""
        doc = _judged_document(self._field_answer())
        doc["families"]["DCCT"]["fields"]["Lifetime"] = {
            "description": "Beam lifetime.",
            "provenance": "stated",
        }
        doc["directions"]["DCCT.Lifetime"] = {"direction": "read", "provenance": "stated"}
        assert _check(doc, _judged_ao()).problems == []

    def test_created_field_reaches_family_fields(self):
        """The judged view carries the moved row as a field of its own."""
        ctx = _context(_judged_document(self._field_answer()), _judged_ao())
        assert list(ctx.family_fields["DCCT"]) == ["Monitor", "Lifetime"]
        view = _judged(ctx, "SR", "DCCT")
        assert view.fields["Lifetime"].raw_slots("ChannelNames") == [_ROW]

    def test_dropped_row_keeps_its_field(self):
        """Dropping a row cuts the row, never the field that carried it."""
        ctx = _context(_judged_document(_answers()), _judged_ao())
        assert list(ctx.family_fields["DCCT"]) == ["Monitor"]
        view = _judged(ctx, "SR", "DCCT")
        assert view.fields["Monitor"].raw_slots("ChannelNames") == ["SR:DCCT:I"]


class TestPerSystemAnswers:
    """An answer applies in the systems that pend it and nowhere else."""

    JUDGMENTS = _answers()

    def test_ordinal_pends_in_one_system_only(self):
        """``BTS`` reaches three of four devices; ``SR`` binds all four."""
        ctx = _context(_judged_document(self.JUDGMENTS), _judged_ao())
        assert ctx.pending[("BTS", "BEND")].unbound_devices == (4,)
        assert ctx.pending[("SR", "BEND")].unbound_devices == ()

    def test_drop_leaves_the_other_system_bound(self):
        """The dropped device leaves ``BTS`` and stays in ``SR``."""
        ctx = _context(_judged_document(self.JUDGMENTS), _judged_ao())
        assert _judged(ctx, "BTS", "BEND").n_devices == 3
        storage_ring = _judged(ctx, "SR", "BEND")
        assert storage_ring.n_devices == 4
        assert storage_ring.fields["Monitor"].raw_slots("ChannelNames")[3] == "SR:BEND4:I"

    def test_drop_is_no_problem(self):
        """A per-system answer breaks no other rule."""
        assert _check(_judged_document(self.JUDGMENTS), _judged_ao()).problems == []


class TestRefusedAnswers:
    """A refused answer becomes a problem, and an incompatible one is left out."""

    @staticmethod
    def _findings(*entries):
        return lambda pending, mapping, va: list(entries)

    def test_findings_become_problems(self, monkeypatch):
        """Every entry is rendered at its own key."""
        monkeypatch.setattr(
            check_module, "_answer_findings", self._findings((_ROW_KEY, "is not pending", True))
        )
        judgments = {"DCCT": {"rows_beyond_devices": {"Monitor": {_ROW: "drop"}}}}
        ctx = _context(_judged_document(judgments), _judged_ao())
        assert ctx.judgment_problems == [Problem(_ROW_KEY, "is not pending")]

    def test_export_incompatible_answer_is_not_applied(self, monkeypatch):
        """The answer leaves ``answers``, so the judged view is the exported one."""
        monkeypatch.setattr(
            check_module, "_answer_findings", self._findings((_ROW_KEY, "is not pending", True))
        )
        judgments = {"DCCT": {"rows_beyond_devices": {"Monitor": {_ROW: {"field": "Lifetime"}}}}}
        ctx = _context(_judged_document(judgments), _judged_ao())
        assert ctx.answers.judgments["DCCT"].rows_beyond["Monitor"] == {}
        assert list(ctx.family_fields["DCCT"]) == ["Monitor"]
        view = _judged(ctx, "SR", "DCCT")
        assert view.fields["Monitor"].raw_slots("ChannelNames") == ["SR:DCCT:I", _ROW]

    def test_completeness_problem_leaves_the_answer_in(self, monkeypatch):
        """An answer the mapping is merely incomplete for still applies."""
        monkeypatch.setattr(
            check_module,
            "_answer_findings",
            self._findings((_ROW_KEY, "add this field entry", False)),
        )
        judgments = {"DCCT": {"rows_beyond_devices": {"Monitor": {_ROW: {"field": "Lifetime"}}}}}
        ctx = _context(_judged_document(judgments), _judged_ao())
        assert ctx.judgment_problems == [Problem(_ROW_KEY, "add this field entry")]
        assert list(ctx.family_fields["DCCT"]) == ["Monitor", "Lifetime"]

    def test_refused_ordinal_stays_bound(self, monkeypatch):
        """An incompatible ``unbound_devices`` answer leaves the device where it is."""
        key = "judgments.BEND.unbound_devices.4"
        monkeypatch.setattr(
            check_module, "_answer_findings", self._findings((key, "names no unbound device", True))
        )
        ctx = _context(_judged_document({"BEND": {"unbound_devices": {4: "drop"}}}), _judged_ao())
        assert ctx.answers.judgments["BEND"].unbound_devices == {}
        assert _judged(ctx, "BTS", "BEND").n_devices == 4

    def test_untouched_mapping_is_reused(self):
        """With nothing refused, ``answers`` is the parsed mapping itself."""
        ctx = _context(_judged_document(_answers()), _judged_ao())
        assert ctx.answers is ctx.mapping


class TestJudgmentRules:
    """Every answer names a judgment the export pends, and mints a name it can carry."""

    @staticmethod
    def _dcct(monitor: dict) -> dict:
        """:func:`_judged_ao` with the ``DCCT`` monitor replaced."""
        ao = _judged_ao()
        ao["SR"]["DCCT"]["Monitor"] = monitor
        return ao

    def test_answer_names_no_pending_row(self):
        """A signal no system carries beyond its devices is refused."""
        judgments = _answers(
            DCCT={"rows_beyond_devices": {"Monitor": {_ROW: "drop", "SR:DCCT:Gone": "drop"}}}
        )
        assert _keys(_check(_judged_document(judgments), _judged_ao())) == [
            "judgments.DCCT.rows_beyond_devices.Monitor[SR:DCCT:Gone]"
        ]

    def test_answer_names_no_unbound_device(self):
        """``BTS`` reaches device 3, so device 3 is nobody's judgment."""
        judgments = _answers(BEND={"unbound_devices": {3: "drop", 4: "drop"}})
        assert _keys(_check(_judged_document(judgments), _judged_ao())) == [
            "judgments.BEND.unbound_devices.3"
        ]

    def test_answer_names_no_family(self):
        """A family the export does not carry is refused once, at the family key."""
        judgments = _answers(GHOST={"unbound_devices": {1: "drop"}})
        assert _keys(_check(_judged_document(judgments), _judged_ao())) == ["judgments.GHOST"]

    def test_pending_row_with_no_slot(self):
        """A pending row the document says nothing about is asked for."""
        assert _keys(_check(_judged_document(_answers(DCCT=...)), _judged_ao())) == [_ROW_KEY]

    def test_pending_supply_with_no_slot(self):
        """The supply is one slot per family, asked for at the family's key."""
        assert _keys(_check(_judged_document(_answers(QF=...)), _judged_ao())) == [
            "judgments.QF.shared_pvs"
        ]

    def test_field_name_the_family_already_carries(self):
        """``field: Monitor`` would overwrite the field the row came from.

        The answer is refused, so it mints no field and no directions key goes
        missing for one.
        """
        judgments = _answers(
            DCCT={"rows_beyond_devices": {"Monitor": {_ROW: {"field": "Monitor"}}}}
        )
        assert _keys(_check(_judged_document(judgments), _judged_ao())) == [_ROW_KEY]

    def test_device_answer_on_a_signal_bound_below(self):
        """Promoting the row would bind one PV to two devices of the family."""
        ao = self._dcct({"ChannelNames": ["SR:DCCT:I", "SR:DCCT:I"]})
        judgments = _answers(DCCT={"rows_beyond_devices": {"Monitor": {"SR:DCCT:I": "device"}}})
        assert _keys(_check(_judged_document(judgments), ao)) == [
            "judgments.DCCT.rows_beyond_devices.Monitor[SR:DCCT:I]"
        ]

    def test_owner_stranding_a_member_is_no_problem(self):
        """Device 2 of ``QF`` keeps nothing of its own, and that is the reviewer's call."""
        doc = _judged_document(_answers(QF={"shared_pvs": {1: 1}}))
        assert _check(doc, _judged_ao()).problems == []

    def test_keep_all_on_a_family_pending_no_supply(self):
        """A supply answer stands where another export of the family shares nothing."""
        judgments = _answers(
            DCCT={
                "rows_beyond_devices": {"Monitor": {_ROW: "drop"}},
                "shared_pvs": "keep_all",
            }
        )
        assert _check(_judged_document(judgments), _judged_ao()).problems == []


class TestOwnerMapRules:
    """An owner map names devices, and the check refuses every name the export lacks."""

    @staticmethod
    def _qf(monitor: list[str], setpoint: list[str]) -> dict:
        """:func:`_judged_ao` with ``QF`` re-shaped over as many devices as it names."""
        ao = _judged_ao()
        ao["SR"]["QF"] = {
            "DeviceList": [[1, index + 1] for index in range(len(monitor))],
            "Monitor": {"ChannelNames": monitor},
            "Setpoint": {"ChannelNames": setpoint},
        }
        return ao

    def _owners(self, owners: dict, ao: dict | None = None):
        """Check :func:`_judged_ao` with ``QF``'s supply answered by ``owners``."""
        doc = _judged_document(_answers(QF={"shared_pvs": owners}))
        return _check(doc, _judged_ao() if ao is None else ao)

    def test_a_key_that_is_no_groups_lowest_ordinal(self):
        """A group is keyed by its lowest ordinal, so ordinal 5 keys nothing."""
        assert _keys(self._owners({1: 1, 5: 5})) == ["judgments.QF.shared_pvs.5"]

    def test_an_owner_outside_its_group(self):
        """``QF`` binds two devices to its supply, and device 3 is not one of them."""
        assert _keys(self._owners({1: 3})) == ["judgments.QF.shared_pvs.1"]

    def test_an_owner_map_missing_a_group(self):
        """The second supply is pending and the map decides nothing for it."""
        ao = self._qf(
            ["SR:QF1:I", "SR:QF1:I", "SR:QF3:I", "SR:QF3:I"],
            ["SR:QF1:SP", "SR:QF1:SP", "SR:QF3:SP", "SR:QF3:SP"],
        )
        assert _keys(self._owners({1: 1}, ao)) == ["judgments.QF.shared_pvs.3"]

    def test_groups_sharing_a_device(self):
        """Two supplies reaching device 2 leave no key that says which devices it owns."""
        ao = self._qf(
            ["SR:QF1:I", "SR:QF1:I", "SR:QF3:I"],
            ["SR:QF1:SP", "SR:QF2:SP", "SR:QF2:SP"],
        )
        result = self._owners({1: 1, 2: 2}, ao)
        assert _keys(result) == ["judgments.QF.shared_pvs"]
        assert "answer `keep_all`" in result.problems[0].message

    def test_groups_that_differ_between_systems(self):
        """Ordinal 1 keys two devices in ``SR`` and three in ``BTS``."""
        ao = _judged_ao()
        ao["BTS"]["QF"] = {
            "DeviceList": [[1, 1], [1, 2], [1, 3]],
            "Monitor": {"ChannelNames": ["BTS:QF:I"] * 3},
            "Setpoint": {"ChannelNames": ["BTS:QF:SP"] * 3},
        }
        result = self._owners({1: 1}, ao)
        assert _keys(result) == ["judgments.QF.shared_pvs"]
        assert "answer `keep_all`" in result.problems[0].message

    def test_an_owner_map_on_a_family_sharing_nothing(self):
        """``DCCT`` shares no PV in any system, so it has no supply to own."""
        judgments = _answers(
            DCCT={
                "rows_beyond_devices": {"Monitor": {_ROW: "drop"}},
                "shared_pvs": {1: 1},
            }
        )
        assert _keys(_check(_judged_document(judgments), _judged_ao())) == [
            "judgments.DCCT.shared_pvs"
        ]

    def test_a_group_starting_past_the_first_device(self):
        """The wrapped fixture's flip: the supply is keyed 2 and owned by device 2."""
        ao = self._qf(
            ["SR:QF1:I", "SR:QF2:I", "SR:QF2:I"],
            ["SR:QF1:SP", "SR:QF2:SP", "SR:QF2:SP"],
        )
        assert self._owners({2: 2}, ao).problems == []


#: The deck the virtual-accelerator fixture was sampled over: one corrector.
_VA_RING = (SimpleNamespace(FamName="HCM1", PassMethod="CorrectorPass", KickAngle=[0.0, 0.0]),)

#: What that export states about ``HCM``: a type the table does not know, bound
#: to the deck's first element, which is why the rules leave it to a reviewer.
_VA_BLOCK = {
    "families": {
        "HCM": {
            "Setpoint": {},
            "nominals": {
                "Setpoint": dict(
                    zip(VA_NOMINAL_KEYS, ([0.0], "rad", "wombat", [[1]], [0]), strict=True)
                )
            },
        }
    }
}

#: The one question the rules ask of that block.
_VA_PROPOSED = {
    "HCM": VAFamily(
        verdict="latch",
        reason="its type is not one the table knows",
        slot=VASlot(kind=ATTYPE_KIND, question="What does HCM drive?", answer=None),
    )
}


def _va_export(*, deck: bool = True, system: str = "SR") -> VAExport:
    """The 2.0 export ``map --check`` holds the block to, deck and all."""
    pending = (
        VAPending(system=system, proposed=_VA_PROPOSED, block=_VA_BLOCK, ring=_VA_RING)
        if deck
        else None
    )
    return VAExport(system=system, pending=pending)


def _va_document(answer: str | None = "kick:0", system: str | None = "SR") -> dict:
    """The passing document plus a virtual-accelerator block answering ``HCM``."""
    doc = _document()
    doc["virtual_accelerator"] = {
        "system": system,
        "families": {
            "HCM": {
                "verdict": "latch",
                "reason": "its type is not one the table knows",
                "slot": {
                    "kind": ATTYPE_KIND,
                    "question": "What does HCM drive?",
                    "answer": answer,
                },
            }
        },
    }
    return doc


class TestVirtualAcceleratorBlock:
    """The VA block is checked only where the tree carries a 2.0 export of it."""

    def test_va_an_answered_block_passes(self):
        """The corrector kicks the plane its element carries, so nothing is wrong."""
        assert _check(_va_document(), va=_va_export()).problems == []

    def test_va_a_one_point_oh_tree_checks_no_block(self):
        """No 2.0 export, no virtual-accelerator rule: the 1.0 mappings pass unchanged."""
        assert _check().problems == []

    def test_va_a_mapping_without_the_block_is_refused(self):
        """The export carries a virtual accelerator the mapping says nothing about."""
        result = _check(va=_va_export())
        assert _keys(result) == ["virtual_accelerator"]
        assert "SR" in result.problems[0].message

    def test_va_a_null_slot_is_refused(self):
        """The reviewer has yet to answer, which ``unanswered_slots`` reads off alone."""
        result = _check(_va_document(answer=None), va=_va_export())
        assert _keys(result) == ["virtual_accelerator.families.HCM.slot.answer"]

    def test_va_a_null_system_is_refused(self):
        """A block naming no system describes none of them."""
        result = _check(_va_document(system=None), va=_va_export())
        assert "virtual_accelerator.system" in _keys(result)

    def test_va_a_block_for_another_system_is_refused(self):
        """The document decides a system this export carries no virtual accelerator for."""
        result = _check(_va_document(system="BTS"), va=_va_export())
        assert _keys(result) == ["virtual_accelerator.system"]
        assert "BTS" in result.problems[0].message and "SR" in result.problems[0].message

    def test_va_an_answer_the_deck_refuses_is_reported(self):
        """The element carries no ``PolynomB``, so it cannot be a strength."""
        result = _check(_va_document(answer="strength:PolynomB[2]"), va=_va_export())
        assert _keys(result) == ["virtual_accelerator.families.HCM.slot.answer"]

    def test_va_an_uncheckable_answer_needs_the_deck(self):
        """Without the deck the answers are unchecked, which is itself the problem."""
        result = _check(_va_document(answer="strength:PolynomB[2]"), va=_va_export(deck=False))
        assert _keys(result) == ["virtual_accelerator"]
        assert "lattice/SR.mat" in result.problems[0].message

    def test_va_facts_reach_the_answer_rules(self):
        """Handing over no facts leaves the answers unjudged; the null slots still read."""
        assert _check(_va_document(answer="strength:PolynomB[2]")).problems == []
