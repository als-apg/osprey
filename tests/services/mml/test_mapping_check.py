"""Semantic checks on a parsed ``mapping.yaml`` against the export it maps.

``check_mapping`` is the fail-closed gate between a filled mapping and
``emit``. Every test starts from one complete, passing mapping and breaks it in
exactly one place, then asserts the problem names that place's key path.
"""

from __future__ import annotations

import copy

import pytest

from osprey.services.mml.directions import vote_directions
from osprey.services.mml.mapping import parse_mapping
from osprey.services.mml.mapping.branches import ROOT_CLASS
from osprey.services.mml.mapping.check import CheckResult, Problem, check_mapping


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


def _check(doc: dict | None = None, ao: dict | None = None, *, no_derived: bool = False):
    doc = _document() if doc is None else doc
    ao = _ao() if ao is None else ao
    return check_mapping(parse_mapping(doc), ao, vote_directions(ao), no_derived=no_derived)


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
