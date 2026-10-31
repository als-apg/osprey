"""Every ``package`` row of the write-verdict case table, through the package.

The table (``_harness/fixtures/write_verdict_cases.json``) is shared with the
hook's stdlib twin; this module drives its ``package`` rows through
:func:`~osprey_connectors.posture_store.write_verdict` and
:func:`~osprey_connectors.posture_store.backstop_verdict`, and pins the
loader's own checks: a reason outside ``REASONS`` and a reason with no row
both fail the load.
"""

from __future__ import annotations

import copy
import json

import pytest

from osprey_connectors.posture_store import backstop_verdict, write_verdict
from osprey_connectors.refusals import REASONS
from osprey_connectors_tests._harness._write_verdict_cases import (
    CASES,
    CASES_PATH,
    LOGIN_TIERS_SHAPE_SOURCE,
    UNCOVERED_REASONS,
    Case,
    CaseTableError,
    case_environment,
    cases_for,
    expected_position,
    login_tiers_shape,
    owner_argument,
    parse_cases,
    stage,
)

__all__ = ["case_environment"]

PACKAGE_CASES = cases_for("package")


@pytest.mark.usefixtures("case_environment")
@pytest.mark.parametrize("case", PACKAGE_CASES, ids=[case.id for case in PACKAGE_CASES])
def test_package_verdict_matches_the_row(case: Case, monkeypatch, tmp_path):
    stage(case, monkeypatch, tmp_path)

    if case.verdict == "backstop":
        verdict = backstop_verdict(case.section)
        assert verdict.target is None
        assert verdict.kind is None
    else:
        verdict = write_verdict(
            case.section,
            case.target,
            ignore_launch_pin=case.ignore_launch_pin,
            **owner_argument(case),
        )
        assert verdict.target == (case.target if isinstance(case.target, str) else None)

    expect = case.expect
    assert verdict.allowed is expect["allowed"]
    assert verdict.reason == expect["reason"]
    assert verdict.position == expected_position(case)
    if not verdict.allowed:
        assert verdict.reason in REASONS
    if "target" in expect:
        assert verdict.target == expect["target"]
    if "kind" in expect:
        assert verdict.kind == expect["kind"]


# -- the table itself --------------------------------------------------------


def _rows() -> list[dict]:
    rows: list[dict] = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return rows


def test_every_reason_the_verdict_answers_has_a_row():
    covered = {case.expect["reason"] for case in CASES}
    assert covered - {None} == set(REASONS) - UNCOVERED_REASONS
    assert None in covered


def test_the_lock_reasons_have_no_row():
    assert not {case.expect["reason"] for case in CASES} & UNCOVERED_REASONS


def test_every_row_drives_the_package():
    """No row is twin-only: the package is the reference the twin matches."""
    assert {case.id for case in CASES} == {case.id for case in PACKAGE_CASES}


#: Write rows staging an ``asked`` login tier, which the framework's twin does not
#: read yet (it knows ``never | allowed``); the framework floor raise lifts them.
_NOT_YET_TWIN = frozenset(
    {
        "position-allowed-under-asked-ceiling",
        "position-asked-under-asked-ceiling",
        "position-legacy-on-under-asked-ceiling",
        "position-off-under-asked-ceiling",
        "position-host-map-is-no-ceiling",
        "tree-section-asked-map-allowed",
        "tree-section-allowed-map-asked",
        "tree-map-asked-stored-asked",
    }
)


def test_write_rows_drive_the_twin():
    """Every write row is a parity row but the gated ones; the backstop is package-only."""
    for case in CASES:
        if case.verdict == "write":
            assert ("twin" in case.applies_to) != (case.id in _NOT_YET_TWIN), case.id


def test_a_row_with_a_reason_outside_reasons_fails_the_load():
    rows = _rows()
    rows[0] = {**rows[0], "expect": {"allowed": False, "reason": "not_a_reason"}}
    with pytest.raises(CaseTableError, match="not in REASONS"):
        parse_cases(rows)


def test_a_reason_with_no_row_fails_the_load():
    rows = [row for row in _rows() if row["expect"]["reason"] != "launch_pinned"]
    with pytest.raises(CaseTableError, match="launch_pinned"):
        parse_cases(rows)


@pytest.mark.parametrize("reason", sorted(UNCOVERED_REASONS))
def test_a_row_for_a_lock_reason_fails_the_load(reason):
    rows = _rows()
    rows.append(
        {**copy.deepcopy(rows[0]), "id": "lock", "expect": {"allowed": False, "reason": reason}}
    )
    with pytest.raises(CaseTableError, match="never answers"):
        parse_cases(rows)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"applies_to": ["elsewhere"]}, "applies_to"),
        ({"verdict": "maybe"}, "verdict"),
        ({"env": {"PATH": "/"}}, "env may set only"),
        ({"surprise": 1}, "unknown keys"),
        ({"expect": {"allowed": True, "reason": "switch_off"}}, "an allow has no reason"),
        ({"record": {"status": "lost"}}, "record status"),
        ({"tree": {"bind": "somewhere"}}, "tree bind"),
    ],
)
def test_a_malformed_row_fails_the_load(change, message):
    rows = _rows()
    rows[0] = {**rows[0], **change}
    with pytest.raises(CaseTableError, match=message):
        parse_cases(rows)


@pytest.mark.parametrize(
    ("expect", "message"),
    [
        ({"allowed": True, "reason": None}, "states position"),
        ({"allowed": True, "reason": None, "position": None}, "states position"),
        ({"allowed": True, "reason": None, "position": "off"}, "states position"),
        ({"allowed": True, "reason": None, "position": "on"}, "states position"),
        ({"allowed": False, "reason": "switch_off", "position": "asked"}, "only a write allow"),
    ],
)
def test_a_write_row_with_a_wrong_position_fails_the_load(expect, message):
    rows = _rows()
    rows[0] = {**rows[0], "expect": expect}
    with pytest.raises(CaseTableError, match=message):
        parse_cases(rows)


def test_a_backstop_row_with_a_position_fails_the_load():
    rows = _rows()
    index = next(i for i, row in enumerate(rows) if row["verdict"] == "backstop")
    rows[index] = {
        **rows[index],
        "expect": {"allowed": True, "reason": None, "position": "allowed"},
    }
    with pytest.raises(CaseTableError, match="only a write allow"):
        parse_cases(rows)


def test_a_refusal_may_leave_the_position_out():
    (case,) = [case for case in CASES if case.id == "switch-off-no-stamp"]
    assert "position" not in case.expect
    assert expected_position(case) is None


@pytest.mark.parametrize(
    ("case_id", "allowed", "reason", "position"),
    [
        ("position-off", False, "switch_off", None),
        ("position-asked", True, None, "asked"),
        ("position-allowed", True, None, "allowed"),
        ("position-legacy-on", True, None, "asked"),
        ("position-unknown-value", False, "switch_off", None),
        ("position-allowed-under-asked-ceiling", True, None, "asked"),
        ("position-allowed-under-never", False, "tier_never", None),
        ("tree-section-asked-map-allowed", True, None, "allowed"),
        ("tree-section-allowed-map-asked", True, None, "asked"),
        ("dispatch-host-owner-position-off", False, "switch_off", None),
        ("dispatch-host-owner-position-asked", True, None, "asked"),
        ("dispatch-host-owner-position-allowed", True, None, "allowed"),
        ("dispatch-tree-owner-position-off", False, "switch_off", None),
        ("dispatch-tree-owner-position-asked", True, None, "asked"),
        ("dispatch-tree-owner-position-allowed", True, None, "allowed"),
        ("dispatch-host-no-owner", False, "owner_unscheduled", None),
        ("dispatch-tree-no-owner", False, "owner_unscheduled", None),
    ],
)
def test_the_table_states_each_position(case_id, allowed, reason, position):
    """The rows the switch's three positions rest on, spelled out here so none can drift."""
    (case,) = [case for case in CASES if case.id == case_id]
    assert case.expect["allowed"] is allowed
    assert case.expect["reason"] == reason
    assert expected_position(case) == position


def test_the_legacy_word_stays_on_the_rows_about_it():
    """Every other row stores what a writer stores now."""
    legacy = {
        case.id
        for case in CASES
        if case.record is not None and "on" in (case.record.get("switch") or {}).values()
    } | {
        case.id
        for case in CASES
        if case.tree is not None
        for record in case.tree.get("records", {}).values()
        if "on" in (record.get("switch") or {}).values()
    }
    assert legacy == {
        "position-legacy-on",
        "position-legacy-on-under-asked-ceiling",
        "tree-legacy-on-map-allowed",
    }


def test_a_repeated_id_fails_the_load():
    rows = _rows()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(CaseTableError, match="repeated"):
        parse_cases(rows)


def test_the_shape_rows_agree_with_the_shape_fixture():
    """The rows that load login_tiers_shape.json expect what its example states."""
    lookups = {item["owner"]: item["reason"] for item in login_tiers_shape()["example"]["lookups"]}
    shape_rows = [case for case in CASES if case.section_from == LOGIN_TIERS_SHAPE_SOURCE]
    assert {case.owner for case in shape_rows} == set(lookups)
    for case in shape_rows:
        assert case.expect["reason"] == lookups[case.owner], case.id
        assert (case.tree is not None) is login_tiers_shape()["example"]["tree_bound"]
        assert (
            case.section["login_tiers"] == login_tiers_shape()["example"]["section"]["login_tiers"]
        )
