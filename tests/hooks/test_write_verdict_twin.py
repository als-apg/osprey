"""The hook's stdlib write verdict against the package's, row by row.

``osprey_target_state.write_verdict`` restates
:func:`osprey_connectors.posture_store.write_verdict` for the hooks, which run
outside the osprey venv. Both read one case table
(``osprey_connectors_tests/_harness/fixtures/write_verdict_cases.json``); this module drives its ``twin``
rows through the hook, and pins the restated words — ``REASONS``, every phrase
and ``STORE_PATH_MARKERS`` — key by key against
:mod:`osprey_connectors.refusals`, and the tier and switch words against the
modules that own them. A drift between the two is a write refused for one
reason by the connector and another by the hook, allowed by one of them only,
or prompted by one and not the other.

The position matrix stages its records raw rather than through the table, so
each stored word meets each ceiling on both sides whatever the table holds.

The package is imported here and only here: a test may import what the code
under test may not. The last test checks that the hook module itself still
imports nothing beyond the standard library and its own directory.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader
from osprey_connectors import control_context, posture_store, refusals, session_hold, types
from osprey_connectors_tests import _paths
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_payload,
)
from osprey_connectors_tests._harness._write_verdict_cases import (
    Case,
    case_environment,
    cases_for,
    host_root,
    owner_argument,
    stage,
)
from tests.hooks.conftest import _HOOK_VARS, HOOKS_DIR

__all__ = ["case_environment"]

TWIN_CASES = cases_for("twin")


def test_the_table_has_twin_rows():
    assert TWIN_CASES
    assert all(case.verdict == "write" for case in TWIN_CASES)


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("case", TWIN_CASES, ids=[case.id for case in TWIN_CASES])
def test_twin_verdict_matches_the_row(case: Case, monkeypatch, tmp_path):
    stage(case, monkeypatch, tmp_path)

    verdict = reader.write_verdict(
        case.section,
        case.target,
        ignore_launch_pin=case.ignore_launch_pin,
        **owner_argument(case, no_owner=reader.NO_OWNER),
    )

    expect = case.expect
    assert verdict.allowed is expect["allowed"]
    assert verdict.reason == expect["reason"]
    assert verdict.target == (case.target if isinstance(case.target, str) else None)
    if "target" in expect:
        assert verdict.target == expect["target"]
    if "kind" in expect:
        assert verdict.kind == expect["kind"]
    if "position" in expect:
        assert verdict.position == expect["position"]
    if not expect["allowed"]:
        assert verdict.position is None


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("case", TWIN_CASES, ids=[case.id for case in TWIN_CASES])
def test_twin_and_package_agree_on_kind_and_position(case: Case, monkeypatch, tmp_path):
    """Kind is a word in the phrase and position decides the prompt: both match on every row."""
    stage(case, monkeypatch, tmp_path)
    kwargs = {"ignore_launch_pin": case.ignore_launch_pin}

    twin = reader.write_verdict(
        case.section, case.target, **kwargs, **owner_argument(case, no_owner=reader.NO_OWNER)
    )
    package = posture_store.write_verdict(
        case.section, case.target, **kwargs, **owner_argument(case)
    )

    assert (twin.allowed, twin.reason, twin.target, twin.kind, twin.position) == (
        package.allowed,
        package.reason,
        package.target,
        package.kind,
        package.position,
    )


# -- each stored word against each ceiling ----------------------------------


def _both_verdicts(section, **kwargs):
    twin = reader.write_verdict(section, "live", **kwargs)
    package = posture_store.write_verdict(section, "live", **kwargs)
    return (
        (twin.allowed, twin.reason, twin.position),
        (package.allowed, package.reason, package.position),
    )


def _stage_host_switch(tmp_path, switch):
    """This host's record holding *switch* verbatim: a legacy ``on`` has no writer."""
    payload = control_context.ControlContext(
        target="live", generation=1, switch_epoch=1
    ).to_payload()
    payload["switch"] = switch
    path = control_context._record_path_under(host_root(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_payload(path, payload)
    control_context.invalidate_cache()


def _stage_tree_switch(monkeypatch, tmp_path, switch):
    """A provisioned tree where ``alice``'s record holds *switch* verbatim."""
    stage(
        Case(
            id="matrix",
            applies_to=("package", "twin"),
            verdict="write",
            section=None,
            target="live",
            has_owner=True,
            owner="alice",
            ignore_launch_pin=False,
            env={},
            record=None,
            tree={"bind": "provisioned", "records": {"alice": {"status": "ok", "switch": switch}}},
            expect={"allowed": True, "reason": None},
        ),
        monkeypatch,
        tmp_path,
    )


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize(
    ("tier", "switch", "expected"),
    [
        ("allowed", {"live": "allowed"}, (True, None, "allowed")),
        ("allowed", {"live": "asked"}, (True, None, "asked")),
        ("allowed", {"live": "on"}, (True, None, "asked")),
        # The ceiling caps what is stored and never rewrites it.
        ("asked", {"live": "allowed"}, (True, None, "asked")),
        ("asked", {"live": "asked"}, (True, None, "asked")),
        ("asked", {"live": "on"}, (True, None, "asked")),
        ("never", {"live": "allowed"}, (False, "tier_never", None)),
        ("allowed", {}, (False, "switch_off", None)),
        ("allowed", {"va": "allowed"}, (False, "switch_off", None)),
        ("allowed", {"live": "off"}, (False, "switch_off", None)),
        ("allowed", {"live": "ON"}, (False, "switch_off", None)),
        ("allowed", {"live": True}, (False, "switch_off", None)),
    ],
    ids=[
        "allowed-under-allowed",
        "asked-under-allowed",
        "legacy-on-under-allowed",
        "allowed-capped-by-asked",
        "asked-under-asked",
        "legacy-on-under-asked",
        "allowed-under-never",
        "no-entry",
        "another-machine",
        "off",
        "capital",
        "boolean",
    ],
)
def test_host_position_is_the_stored_word_capped_by_the_tier(tmp_path, tier, switch, expected):
    _stage_host_switch(tmp_path, switch)

    twin, package = _both_verdicts({"type": "epics", "writes": tier})

    assert twin == package == expected


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize(
    ("section_tier", "map_tier", "stored", "expected"),
    [
        # Under a tree the rung is the ceiling, not the section.
        ("asked", "allowed", "allowed", (True, None, "allowed")),
        ("allowed", "asked", "allowed", (True, None, "asked")),
        ("allowed", "asked", "asked", (True, None, "asked")),
        ("allowed", "allowed", "on", (True, None, "asked")),
        ("allowed", "never", "allowed", (False, "tier_never", None)),
        # A map spelling the switch's retired word is unreadable as a whole.
        ("allowed", "on", "allowed", (False, "owner_unknown", None)),
    ],
    ids=[
        "rung-above-section",
        "rung-caps",
        "asked-under-asked-rung",
        "legacy-on",
        "rung-never",
        "rung-retired-word",
    ],
)
def test_tree_position_is_the_stored_word_capped_by_the_rung(
    monkeypatch, tmp_path, section_tier, map_tier, stored, expected
):
    _stage_tree_switch(monkeypatch, tmp_path, {"live": stored})
    section = {"type": "epics", "writes": section_tier, "login_tiers": {"alice": map_tier}}

    twin, package = _both_verdicts(section, owner="alice")

    assert twin == package == expected


_CEILING_PAIRS = [
    ("asked", "allowed"),
    ("allowed", "asked"),
    ("asked", "asked"),
    ("allowed", "allowed"),
]


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize(("section_tier", "map_tier"), _CEILING_PAIRS)
def test_the_routes_ceiling_is_the_verdicts_under_a_tree(
    monkeypatch, tmp_path, section_tier, map_tier
):
    """``write_ceiling`` is the one producer: the chip's ceiling never disagrees with the verdict."""
    from osprey.interfaces.web_terminal.routes import websocket as websocket_routes

    _stage_tree_switch(monkeypatch, tmp_path, {"live": "allowed"})
    section = {"type": "epics", "writes": section_tier, "login_tiers": {"alice": map_tier}}

    verdict = posture_store.write_verdict(section, "live", owner="alice")

    assert verdict.position == map_tier
    assert websocket_routes._login_tier(section, "alice") == verdict.position
    assert posture_store.write_ceiling(section, "alice") == verdict.position


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize(("section_tier", "map_tier"), _CEILING_PAIRS)
def test_the_routes_ceiling_is_the_verdicts_on_a_host(tmp_path, section_tier, map_tier):
    from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
    from osprey_connectors.identity import acting_identity

    _stage_host_switch(tmp_path, {"live": "allowed"})
    login = acting_identity()
    section = {"type": "epics", "writes": section_tier, "login_tiers": {login: map_tier}}

    verdict = posture_store.write_verdict(section, "live", owner=login)

    assert verdict.position == section_tier
    assert websocket_routes._login_tier(section, login) == verdict.position
    assert posture_store.write_ceiling(section, login) == verdict.position


def test_a_ceiling_under_a_tree_with_no_map_is_unknown(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "/nonexistent")
    assert posture_store.write_ceiling({"writes": "allowed"}, "alice") == "unknown"
    assert posture_store.write_ceiling({"writes": "allowed"}, "alice", tree_bound=False) == (
        "allowed"
    )


@pytest.mark.usefixtures("case_environment")
def test_the_launch_pin_refuses_a_stored_allowed_with_no_position(monkeypatch, tmp_path):
    _stage_host_switch(tmp_path, {"live": "allowed"})
    monkeypatch.setenv("OSPREY_LAUNCH_POSTURE", "live=sandbox")
    section = {"type": "epics", "writes": "allowed"}

    assert _both_verdicts(section) == (
        (False, "launch_pinned", None),
        (False, "launch_pinned", None),
    )
    twin, package = _both_verdicts(section, ignore_launch_pin=True)
    assert twin == package == (True, None, "allowed")


# -- the restated tier and switch readers -------------------------------------


_SWITCH_VALUES = [
    {"live": "on"},
    {"live": "asked", "va": "allowed"},
    {"live": "off", "va": "ON", "standin": True},
    {"live": None, "va": " asked", "standin": 1},
    {"future-target": "allowed", 7: "asked"},
    {},
    "on",
    ["live"],
    None,
]


@pytest.mark.parametrize("value", _SWITCH_VALUES, ids=repr)
def test_parse_switch_matches_the_package(value):
    assert reader.parse_switch(value) == control_context._parse_switch(value)


@pytest.mark.parametrize("value", [v for v in _SWITCH_VALUES if isinstance(v, dict)], ids=repr)
@pytest.mark.parametrize("target", ["live", "va", "standin", "future-target"])
def test_switch_position_matches_the_package(value, target):
    record = control_context.ControlContext(target="live", generation=1, switch=value)

    assert reader.switch_position({"switch": value}, target) == control_context.switch_position(
        record, target
    )


def test_switch_position_reads_no_record_as_off_on_both_sides():
    assert reader.switch_position(None, "live") == "off"
    assert control_context.switch_position(None, "live") == "off"
    # The three-field shape of parse_record carries no switch at all.
    assert reader.switch_position({"target": "live"}, "live") == "off"


@pytest.mark.parametrize(
    "value",
    ["never", "asked", "allowed", "on", "ON", "Allowed", " asked", "", True, None, 7, ["asked"]],
    ids=repr,
)
def test_tier_readers_match_the_package(value):
    section = {"writes": value}

    assert reader.tier_writes(section) == types.tier_writes(section)
    assert reader.tier_allows(section) is types.tier_allows(section)


@pytest.mark.parametrize("section", [None, "allowed", [], {}], ids=repr)
def test_tier_readers_match_the_package_on_a_section_with_no_tier(section):
    assert reader.tier_writes(section) == types.tier_writes(section) == "never"
    assert reader.tier_allows(section) is types.tier_allows(section) is False


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize(
    ("login_tiers", "owner"),
    [
        ({"alice": "never"}, "alice"),
        ({"alice": "asked"}, "alice"),
        ({"alice": "allowed"}, "alice"),
        ({"alice": "asked"}, "bob"),
        ({"alice": "asked"}, "ALICE"),
        ({"alice": "on"}, "alice"),
        ({"alice": "asked", "bob": "on"}, "alice"),
        ({"alice": True}, "alice"),
        ("asked", "alice"),
        ({"alice": "asked"}, ""),
        ({"alice": "asked"}, None),
    ],
    ids=repr,
)
def test_owner_tier_term_matches_the_package(login_tiers, owner):
    section = {"login_tiers": login_tiers}

    assert reader.owner_tier_term(section, owner) == posture_store.owner_tier_term(section, owner)


def test_a_dispatch_run_resolves_no_owner_on_both_sides(monkeypatch, tmp_path):
    """The dispatch rung sits below the stamp and above the tree and the account."""
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("OSPREY_CONTROL_CONTEXT_TREE", raising=False)
    monkeypatch.delenv("OSPREY_CONTROL_OWNER", raising=False)
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    posture_store._owner_var.set(posture_store.NO_OWNER)
    assert reader.current_owner() is reader.NO_OWNER
    assert posture_store.current_owner() is posture_store.NO_OWNER

    monkeypatch.setenv("OSPREY_CONTROL_OWNER", "alice")
    assert reader.current_owner() == posture_store.current_owner() == "alice"

    monkeypatch.delenv("OSPREY_CONTROL_OWNER")
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "0")
    assert reader.current_owner() is not reader.NO_OWNER
    assert posture_store.current_owner() is not posture_store.NO_OWNER


def test_the_verdict_never_raises_on_a_hostile_section(monkeypatch):
    class Hostile(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

        def __contains__(self, key):
            raise RuntimeError("boom")

    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    verdict = reader.write_verdict(Hostile(), "live", owner="alice")
    assert verdict.allowed is False
    assert verdict.reason in reader.REASONS


def test_a_verdict_carries_the_five_fields():
    verdict = reader.WriteVerdict(
        allowed=True, reason=None, target="va", kind="simulator", position="asked"
    )
    assert (verdict.allowed, verdict.reason, verdict.target, verdict.kind, verdict.position) == (
        True,
        None,
        "va",
        "simulator",
        "asked",
    )


def test_a_verdict_defaults_to_no_position_as_the_package_does():
    args = {"allowed": False, "reason": "switch_off", "target": "va", "kind": "simulator"}

    assert reader.WriteVerdict(**args).position is None
    assert posture_store.WriteVerdict(**args).position is None


def test_position_counts_in_a_verdicts_equality_and_repr():
    args = {"allowed": True, "reason": None, "target": "va", "kind": "simulator"}
    asked = reader.WriteVerdict(**args, position="asked")

    assert asked == reader.WriteVerdict(**args, position="asked")
    assert asked != reader.WriteVerdict(**args, position="allowed")
    assert "position='asked'" in repr(asked)


def test_the_no_owner_sentinel_has_no_spelling():
    assert str(reader.NO_OWNER) == "<no owner>"
    assert reader.NO_OWNER != "<no owner>"
    assert reader.current_owner(reader.NO_OWNER) is reader.NO_OWNER


# -- the restated words ------------------------------------------------------


def test_reasons_match_in_order():
    assert reader.REASONS == refusals.REASONS


def test_phrase_table_matches_key_by_key():
    assert list(reader._PHRASES) == list(refusals._PHRASES)
    for reason in refusals.REASONS:
        assert reader._PHRASES[reason] == refusals._PHRASES[reason], reason


@pytest.mark.parametrize("kind", [None, *refusals.KIND_WORDS, "unknown"])
@pytest.mark.parametrize("reason", refusals.REASONS)
def test_refusal_phrase_matches(reason, kind):
    assert reader.refusal_phrase(reason, kind) == refusals.refusal_phrase(reason, kind)


def test_an_unknown_reason_raises_on_both_sides():
    with pytest.raises(ValueError, match="unknown refusal reason"):
        refusals.refusal_phrase("not_a_reason")
    with pytest.raises(ValueError, match="unknown refusal reason"):
        reader.refusal_phrase("not_a_reason")


def test_kind_words_match_key_by_key():
    assert list(reader.KIND_WORDS) == list(refusals.KIND_WORDS)
    for kind, words in refusals.KIND_WORDS.items():
        assert reader.KIND_WORDS[kind] == words, kind


def test_store_path_markers_match():
    assert reader.STORE_PATH_MARKERS == refusals.STORE_PATH_MARKERS
    assert reader.STORE_PATH_MARKERS == (reader.STATE_DIR_NAME, reader.RECORD_FILENAME)


def test_restated_store_names_match_the_package():
    assert reader.CONTROL_OWNER_ENV_VAR == posture_store.CONTROL_OWNER_ENV_VAR
    assert reader.DISPATCH_RUN_ENV_VAR == posture_store.DISPATCH_RUN_ENV_VAR
    assert reader.CONTROL_CONTEXT_TREE_ENV_VAR == posture_store.CONTROL_CONTEXT_TREE_ENV_VAR
    assert reader.CONTROL_TREE_MARKER_NAME == posture_store.CONTROL_TREE_MARKER_NAME
    assert reader.LAUNCH_POSTURE_ENV_VAR == posture_store.LAUNCH_POSTURE_ENV_VAR
    assert reader.LAUNCH_POSTURE_ALL_TARGETS == posture_store.LAUNCH_POSTURE_ALL_TARGETS
    assert reader.RECORD_READ_LIMIT == posture_store._RECORD_READ_LIMIT
    assert reader.HOLD_DIR_ENV_VAR == session_hold.HOLD_DIR_ENV_VAR
    assert reader.HOLD_RECORD_NAME == session_hold.RECORD_NAME
    assert reader.HOLD_SCHEMA_VERSION == session_hold.SCHEMA_VERSION
    assert reader.HOLD_RECORD_READ_LIMIT == session_hold._RECORD_READ_LIMIT
    assert reader.HOLD_SESSION_ENV_VAR == posture_store._SESSION_KEY_ENV_VAR
    assert reader._HOLD_SAFE_KEY == session_hold._SAFE_KEY.pattern
    assert reader._HOLD_STR_FIELDS == session_hold._STR_FIELDS


def test_tier_words_match_the_package():
    assert reader.TIER_NEVER == types._TIER_NEVER
    assert reader.TIER_ASKED == types._TIER_ASKED
    assert reader.TIER_ALLOWED == types._TIER_ALLOWED
    # The order is the ceiling's rank: ``min`` on it is the cap.
    assert reader._LOGIN_TIERS == posture_store._LOGIN_TIERS == ("never", "asked", "allowed")


def test_switch_words_match_the_package():
    assert reader.SWITCH_OFF == control_context._SWITCH_OFF
    assert reader.SWITCH_ASKED == control_context._SWITCH_ASKED
    assert reader.SWITCH_ALLOWED == control_context._SWITCH_ALLOWED
    assert reader._SWITCH_LEGACY_ON == control_context._SWITCH_LEGACY_ON


def test_new_names_are_exported():
    for name in (
        "DISPATCH_RUN_ENV_VAR",
        "KIND_WORDS",
        "NO_OWNER",
        "REASONS",
        "STORE_PATH_MARKERS",
        "SWITCH_ALLOWED",
        "SWITCH_ASKED",
        "SWITCH_OFF",
        "TIER_ASKED",
        "WriteVerdict",
        "current_owner",
        "execution_verdict",
        "is_held",
        "launch_pin_term",
        "machine_kind",
        "owner_tier_term",
        "parse_switch",
        "refusal_phrase",
        "switch_position",
        "tier_allows",
        "tier_writes",
        "write_verdict",
    ):
        assert name in reader.__all__, name
        assert hasattr(reader, name), name


def test_the_retired_switch_constant_is_gone():
    assert "SWITCH_ON" not in reader.__all__
    assert not hasattr(reader, "SWITCH_ON")


# -- the session's hold ------------------------------------------------------

HOLD_FIXTURES = Path(_paths.__file__).parent / "fixtures" / "session_hold"
HOLD_CASES = json.loads((HOLD_FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]
HOLD_KEY = "3f2b8c1e-6a0d-4c55-9a1e-0b7f2d4e9c11"  # gitleaks:allow - a test session id
_SECTION = {"type": "epics", "writes": "allowed"}


@pytest.fixture
def no_hold(monkeypatch, tmp_path):
    """No stamped run state and no hold store; the host record under scratch."""
    isolate_verdict_environment(monkeypatch, tmp_path / "data")
    monkeypatch.delenv(reader.HOLD_SESSION_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.HOLD_DIR_ENV_VAR, raising=False)
    yield
    reset_verdict_state()


@pytest.fixture
def store(no_hold, monkeypatch, tmp_path) -> Path:  # noqa: ARG001 - ordered after its isolation
    """This session's record present under a stamped store, nothing open."""
    root = tmp_path / "session_holds"
    monkeypatch.setenv(reader.HOLD_DIR_ENV_VAR, str(root))
    monkeypatch.setenv(reader.HOLD_SESSION_ENV_VAR, HOLD_KEY)
    session_hold.ensure_session(HOLD_KEY)
    return root


def _open_flag() -> None:
    session_hold.open_flag(
        HOLD_KEY,
        flag_id="f1",
        source="watchdog:scope",
        reason="off the plan",
        prompt_ref="record:f1",
        prompt_excerpt="move the corrector",
    )


def _agree(section: object, target: object) -> tuple:
    """Both sides' whole answer, asserted equal; returned for a test to read."""
    twin = reader.execution_verdict(section, target)
    package = posture_store.execution_verdict(section, target)
    assert (twin.allowed, twin.reason, twin.target, twin.kind) == (
        package.allowed,
        package.reason,
        package.target,
        package.kind,
    )
    return twin.allowed, twin.reason


@pytest.mark.usefixtures("no_hold")
@pytest.mark.parametrize("case", sorted(HOLD_CASES))
def test_twin_and_package_agree_over_every_shared_fixture(case, tmp_path, monkeypatch):
    spec = HOLD_CASES[case]
    root = tmp_path / "root"
    shutil.copytree(HOLD_FIXTURES / case / "root", root)
    monkeypatch.setenv(reader.HOLD_DIR_ENV_VAR, str(root))
    monkeypatch.setenv(reader.HOLD_SESSION_ENV_VAR, spec["session_key"])

    assert reader.is_held(spec["session_key"]) is spec["held"]
    assert session_hold.is_held(spec["session_key"]) is spec["held"]
    assert (_agree(_SECTION, "live")[1] == "session_held") is spec["held"]


def _flags_with(**overrides) -> list[dict]:
    flag = {
        "flag_id": "f1",
        "source": "s",
        "reason": "r",
        "prompt_ref": "p",
        "prompt_excerpt": "",
        "opened_at": "t",
        "state": "closed",
        "stop_unconfirmed": False,
        "started": [],
    }
    return [{**flag, **overrides}]


@pytest.mark.parametrize(
    "payload",
    [
        {"flags": _flags_with()},
        {"flags": _flags_with(state="open")},
        {"flags": _flags_with(state="shut")},
        {"flags": _flags_with(source=3)},
        {"flags": _flags_with(stop_unconfirmed="no")},
        {"flags": _flags_with(started=[1])},
        {"flags": "none"},
        {"flags": []},
        {"schema": 2, "flags": []},
        {"session_key": "another-session", "flags": []},
    ],
    ids=[
        "closed",
        "open",
        "bad-state",
        "bad-source",
        "bad-stop",
        "bad-started",
        "flags-not-a-list",
        "empty",
        "other-schema",
        "other-session",
    ],
)
def test_twin_and_package_read_every_record_alike(store, payload):
    record = {"schema": 1, "session_key": HOLD_KEY, **payload}
    (store / HOLD_KEY / reader.HOLD_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    assert reader.is_held(HOLD_KEY) is session_hold.is_held(HOLD_KEY)
    _agree(_SECTION, "live")


@pytest.mark.parametrize(
    "padding, held",
    [(reader.RECORD_READ_LIMIT + 1024, False), (reader.HOLD_RECORD_READ_LIMIT + 1, True)],
    ids=["past-the-posture-limit", "past-the-hold-limit"],
)
def test_the_hold_reader_has_its_own_limit(store, padding, held):
    record = {"schema": 1, "session_key": HOLD_KEY, "flags": [], "pad": "x" * padding}
    (store / HOLD_KEY / reader.HOLD_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    assert reader.is_held(HOLD_KEY) is held
    assert session_hold.is_held(HOLD_KEY) is held


def test_a_link_at_the_record_or_the_directory_is_held_on_both_sides(store, tmp_path):
    record = store / HOLD_KEY / reader.HOLD_RECORD_NAME
    real = tmp_path / "real.json"
    real.write_text(record.read_text(encoding="utf-8"), encoding="utf-8")
    record.unlink()
    record.symlink_to(real)
    assert reader.is_held(HOLD_KEY) is True
    assert session_hold.is_held(HOLD_KEY) is True

    record.unlink()
    record.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    moved = tmp_path / "moved"
    (store / HOLD_KEY).rename(moved)
    (store / HOLD_KEY).symlink_to(moved)
    assert reader.is_held(HOLD_KEY) is True
    assert session_hold.is_held(HOLD_KEY) is True


@pytest.mark.parametrize("key", [" ", "../up", ".hidden", "a/b", "x" * 129, "a\n"])
@pytest.mark.usefixtures("store")
def test_an_unsafe_key_is_held_on_both_sides_and_only_under_a_store(monkeypatch, key):
    monkeypatch.setenv(reader.HOLD_SESSION_ENV_VAR, key)
    assert reader.is_held(key) is session_hold.is_held(key) is True
    assert _agree(_SECTION, "live")[1] == "session_held"

    monkeypatch.delenv(reader.HOLD_DIR_ENV_VAR)
    assert reader.is_held(key) is session_hold.is_held(key) is False


@pytest.mark.usefixtures("store")
def test_held_refuses_session_held_and_a_release_lifts_it():
    assert _agree(_SECTION, "live") == (False, "switch_off")
    _open_flag()
    assert _agree(_SECTION, "live") == (False, "session_held")
    session_hold.answer_flags(HOLD_KEY, ["f1"], answer="continue", operator="op")
    assert _agree(_SECTION, "live") == (False, "switch_off")


@pytest.mark.usefixtures("store")
def test_the_hold_follows_the_read_only_run_and_the_tier(monkeypatch):
    _open_flag()
    assert _agree({"writes": "never"}, "live") == (False, "tier_never")
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    assert _agree(_SECTION, "live") == (False, "readonly_run")


@pytest.mark.usefixtures("store")
def test_the_hold_precedes_the_target_and_keeps_its_name():
    _open_flag()
    assert _agree(_SECTION, "no-such-machine") == (False, "session_held")
    assert reader.execution_verdict(_SECTION, "live").target == "live"


@pytest.mark.usefixtures("store")
def test_a_process_with_no_session_is_never_held(monkeypatch):
    _open_flag()
    monkeypatch.setenv(reader.HOLD_SESSION_ENV_VAR, "")
    assert _agree(_SECTION, "live") == (False, "switch_off")
    monkeypatch.delenv(reader.HOLD_SESSION_ENV_VAR)
    assert _agree(_SECTION, "live") == (False, "switch_off")


@pytest.mark.usefixtures("store")
def test_a_reader_that_raises_is_held(monkeypatch):
    def _boom(*_args):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(reader, "_read_hold_flags", _boom)
    assert reader.is_held(HOLD_KEY) is True
    assert reader.execution_verdict(_SECTION, "live").reason == "session_held"


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("case", TWIN_CASES, ids=[case.id for case in TWIN_CASES])
def test_twin_and_package_agree_on_every_row_while_held(case: Case, monkeypatch, tmp_path):
    stage(case, monkeypatch, tmp_path)
    monkeypatch.setenv(reader.HOLD_DIR_ENV_VAR, str(tmp_path / "session_holds"))
    monkeypatch.setenv(reader.HOLD_SESSION_ENV_VAR, HOLD_KEY)
    session_hold.ensure_session(HOLD_KEY)
    _open_flag()
    kwargs = {"ignore_launch_pin": case.ignore_launch_pin}

    twin = reader.execution_verdict(
        case.section, case.target, **kwargs, **owner_argument(case, no_owner=reader.NO_OWNER)
    )
    package = posture_store.execution_verdict(
        case.section, case.target, **kwargs, **owner_argument(case)
    )

    assert (twin.allowed, twin.reason, twin.target, twin.kind) == (
        package.allowed,
        package.reason,
        package.target,
        package.kind,
    )


# -- the forwarded environment ----------------------------------------------


@pytest.mark.parametrize(
    "name", ["OSPREY_CONTROL_OWNER", "OSPREY_CONTROL_CONTEXT_TREE", "OSPREY_DISPATCH_RUN"]
)
def test_the_owner_ladder_vars_reach_a_hook_subprocess(name):
    assert name in _HOOK_VARS


@pytest.mark.parametrize(
    "name", ["OSPREY_CONTROL_OWNER", "OSPREY_CONTROL_CONTEXT_TREE", "OSPREY_DISPATCH_RUN"]
)
def test_the_owner_ladder_vars_are_blank_by_default(name):
    assert not os.environ.get(name)


# -- stdlib only -------------------------------------------------------------


def _hook_stems() -> set[str]:
    return {path.stem for path in HOOKS_DIR.glob("*.py")}


def test_the_module_source_imports_only_the_stdlib_and_the_hook_directory():
    tree = ast.parse((HOOKS_DIR / "osprey_target_state.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in a hook module"
            roots.add((node.module or "").split(".")[0])
    roots.discard("__future__")
    foreign = {root for root in roots if root not in sys.stdlib_module_names} - _hook_stems()
    assert not foreign


def test_the_module_imports_and_decides_without_site_packages(tmp_path):
    """Outside the venv — no site-packages at all — it imports and still decides.

    ``-S`` drops site-packages, which is where a hook stands at run time: the
    sibling ``osprey_hook_log``'s one optional package import then falls back,
    and everything that loads has to be the standard library or this directory.
    """
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(HOOKS_DIR)!r})\n"
        "before = set(sys.modules)\n"
        "import osprey_target_state\n"
        "added = {name.split('.')[0] for name in set(sys.modules) - before}\n"
        "foreign = sorted(name for name in added if name not in sys.stdlib_module_names)\n"
        "verdict = osprey_target_state.write_verdict(\n"
        "    {'type': 'epics', 'writes': 'allowed'}, 'live', owner='alice'\n"
        ")\n"
        "assert osprey_target_state.execution_verdict(\n"
        "    {'type': 'epics', 'writes': 'allowed'}, 'live', owner='alice'\n"
        ").reason == 'session_held'\n"
        "print(json.dumps([foreign, verdict.reason]))\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OSPREY_AGENT_DATA_ROOT": str(tmp_path),
        # The host's own login, so the owner is not refused as foreign.
        "OSPREY_AUDIT_IDENTITY": "alice",
        # A store with no record for the session: held, decided by the stdlib alone.
        "OSPREY_SESSION_HOLD_DIR": str(tmp_path / "session_holds"),
        "OSPREY_SESSION_ID": "s1",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=120,
    )

    foreign, reason = json.loads(result.stdout)
    assert set(foreign) <= _hook_stems()
    assert reason == "switch_off"
