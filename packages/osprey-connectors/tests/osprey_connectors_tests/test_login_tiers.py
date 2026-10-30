"""The owner rung: a login's tier read out of the build-rendered login→tier map.

A shared container reads other logins' records, so its own section's tier says
nothing about the owner of the record it reads. The build renders
``control_system.login_tiers`` — ``{<login>: "never" | "asked" | "allowed"}`` — and
:func:`~osprey_connectors.posture_store.owner_tier_term` reads the owner's entry.
The rung applies when the section carries the map or a control-context tree is
bound; under a tree a missing or unreadable map is ``unknown``, never skipped.
"""

from __future__ import annotations

import json

import pytest

from osprey_connectors import posture_store
from osprey_connectors.posture_store import NO_OWNER, owner_tier_term
from osprey_connectors_tests._harness._write_verdict_cases import LOGIN_TIERS_SHAPE_PATH

SHAPE = LOGIN_TIERS_SHAPE_PATH
TREE_ENV = posture_store.CONTROL_CONTEXT_TREE_ENV_VAR


@pytest.fixture
def no_tree(monkeypatch):
    monkeypatch.delenv(TREE_ENV, raising=False)


@pytest.fixture
def tree(monkeypatch, tmp_path):
    monkeypatch.setenv(TREE_ENV, str(tmp_path / "tree"))


def _section(login_tiers):
    return {"writes": "allowed", "login_tiers": login_tiers}


def test_the_key_names_the_documented_location():
    assert posture_store.LOGIN_TIERS_KEY == "control_system.login_tiers"


@pytest.mark.parametrize(
    "section",
    [{}, {"writes": "allowed"}, {"writes": "never"}, None, "garbage"],
    ids=["empty", "allowed-tier", "never-tier", "none", "not-a-dict"],
)
@pytest.mark.usefixtures("no_tree")
def test_no_map_and_no_tree_skips_the_rung(section):
    assert owner_tier_term(section, "alice") == "skip"


@pytest.mark.usefixtures("no_tree")
def test_no_map_and_no_tree_skips_even_without_an_owner():
    assert owner_tier_term({"writes": "allowed"}, None) == "skip"
    assert owner_tier_term({"writes": "allowed"}, NO_OWNER) == "skip"


@pytest.mark.parametrize(
    "section",
    [{}, {"writes": "allowed"}, None, "garbage"],
    ids=["empty", "allowed-tier", "none", "not-a-dict"],
)
@pytest.mark.usefixtures("tree")
def test_tree_bound_and_no_map_is_unknown(section):
    assert owner_tier_term(section, "alice") == "unknown"


def test_a_blank_tree_bind_is_not_a_bound_tree(monkeypatch):
    monkeypatch.setenv(TREE_ENV, "   ")
    assert owner_tier_term({"writes": "allowed"}, "alice") == "skip"


@pytest.mark.parametrize("bound", [False, True], ids=["host", "tree"])
def test_an_owner_missing_from_the_map_is_unknown(monkeypatch, tmp_path, bound):
    if bound:
        monkeypatch.setenv(TREE_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(TREE_ENV, raising=False)
    assert owner_tier_term(_section({"bob": "allowed"}), "alice") == "unknown"


@pytest.mark.usefixtures("no_tree")
def test_an_empty_map_knows_no_owner():
    assert owner_tier_term(_section({}), "alice") == "unknown"


@pytest.mark.parametrize("bound", [False, True], ids=["host", "tree"])
def test_a_mapped_never_is_never(monkeypatch, tmp_path, bound):
    if bound:
        monkeypatch.setenv(TREE_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(TREE_ENV, raising=False)
    assert owner_tier_term(_section({"alice": "never"}), "alice") == "never"


@pytest.mark.usefixtures("tree")
def test_a_mapped_allowed_is_allowed():
    assert owner_tier_term(_section({"alice": "allowed"}), "alice") == "allowed"


@pytest.mark.parametrize("bound", [False, True], ids=["host", "tree"])
def test_a_mapped_asked_is_asked(monkeypatch, tmp_path, bound):
    if bound:
        monkeypatch.setenv(TREE_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(TREE_ENV, raising=False)
    assert owner_tier_term(_section({"alice": "asked"}), "alice") == "asked"


@pytest.mark.usefixtures("no_tree")
def test_a_map_carrying_all_three_words_is_readable():
    login_tiers = {"alice": "never", "bob": "asked", "carol": "allowed"}
    for login, tier in login_tiers.items():
        assert owner_tier_term(_section(login_tiers), login) == tier


@pytest.mark.usefixtures("no_tree")
def test_the_map_decides_even_over_the_section_tier():
    section = {"writes": "never", "login_tiers": {"alice": "allowed"}}
    assert owner_tier_term(section, "alice") == "allowed"
    section = {"writes": "allowed", "login_tiers": {"alice": "asked"}}
    assert owner_tier_term(section, "alice") == "asked"


@pytest.mark.parametrize(
    "login_tiers",
    [None, "alice=allowed", ["alice"], 1, True],
    ids=["none", "string", "list", "int", "bool"],
)
@pytest.mark.usefixtures("no_tree")
def test_a_map_that_is_not_a_dict_is_unknown(login_tiers):
    assert owner_tier_term(_section(login_tiers), "alice") == "unknown"


@pytest.mark.parametrize(
    "login_tiers",
    [
        {"alice": True},
        {"alice": None},
        {"alice": 1},
        {"alice": ["allowed"]},
        {"alice": {"tier": "allowed"}},
    ],
    ids=["bool", "none", "int", "list", "dict"],
)
@pytest.mark.usefixtures("no_tree")
def test_a_non_string_value_for_the_owner_is_unknown(login_tiers):
    assert owner_tier_term(_section(login_tiers), "alice") == "unknown"


@pytest.mark.parametrize(
    "value",
    ["Allowed", "ALLOWED", " allowed", "Asked", "asked ", "on", "off", "yes", "", "readwrite"],
)
@pytest.mark.usefixtures("no_tree")
def test_a_string_outside_the_three_tiers_is_unknown(value):
    assert owner_tier_term(_section({"alice": value}), "alice") == "unknown"


@pytest.mark.parametrize(
    "login_tiers",
    [
        {"alice": "allowed", "bob": True},
        {"alice": "allowed", "bob": "maybe"},
        {"alice": "allowed", 1: "never"},
        {"alice": "asked", "bob": "on"},
    ],
    ids=["other-value-bool", "other-value-string", "non-string-key", "other-value-on"],
)
@pytest.mark.usefixtures("no_tree")
def test_one_unreadable_entry_makes_the_whole_map_unreadable(login_tiers):
    assert owner_tier_term(_section(login_tiers), "alice") == "unknown"


@pytest.mark.parametrize(
    "owner",
    [None, NO_OWNER, "", 7, " alice", "Alice"],
    ids=["none", "no-owner", "empty", "int", "padded", "case"],
)
@pytest.mark.usefixtures("no_tree")
def test_an_owner_that_is_not_a_mapped_name_is_unknown(owner):
    assert owner_tier_term(_section({"alice": "allowed"}), owner) == "unknown"


@pytest.mark.usefixtures("no_tree")
def test_the_reader_never_raises_on_an_odd_section():
    class Weird(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

        def __contains__(self, key):
            raise RuntimeError("boom")

    assert owner_tier_term(Weird(), "alice") == "unknown"


# -- the shape fixture -------------------------------------------------------


def _shape() -> dict:
    return json.loads(SHAPE.read_text(encoding="utf-8"))


def test_the_shape_fixture_states_location_shape_sources_and_arms():
    shape = _shape()
    assert shape["location"] == posture_store.LOGIN_TIERS_KEY
    assert shape["value_shape"]["values"] == ["allowed", "never"]
    assert shape["sources"]
    assert set(shape["terms"]) == {"skip", "allowed", "never", "unknown"}
    reasons = {arm["reason"] for arm in shape["refusal_arms"]}
    assert reasons == {"owner_unknown", "tier_never"}
    for arm in shape["refusal_arms"]:
        assert {"owner_unknown": "unknown", "tier_never": "never"}[arm["reason"]] == arm["term"]


def test_the_shape_fixture_reasons_are_refusal_reasons():
    from osprey_connectors.refusals import REASONS

    shape = _shape()
    for arm in shape["refusal_arms"]:
        assert arm["reason"] in REASONS
    for lookup in shape["example"]["lookups"]:
        assert lookup["reason"] is None or lookup["reason"] in REASONS


def test_the_shape_fixture_example_validates_against_the_reader(monkeypatch, tmp_path):
    shape = _shape()
    example = shape["example"]
    if example["tree_bound"]:
        monkeypatch.setenv(TREE_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(TREE_ENV, raising=False)
    leaf = shape["location"].split(".", 1)[1]
    assert leaf in example["section"]
    terms = set()
    for lookup in example["lookups"]:
        term = owner_tier_term(example["section"], lookup["owner"])
        assert term == lookup["term"], lookup
        terms.add(term)
    assert terms == {"allowed", "never", "unknown"}
