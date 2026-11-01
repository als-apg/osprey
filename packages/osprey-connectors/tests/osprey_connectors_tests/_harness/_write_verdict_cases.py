"""The write-verdict case table: loader, schema check and staging.

``_harness/fixtures/write_verdict_cases.json`` is one list of rows, each naming a
verdict (``write`` or ``backstop``), its inputs — section, target, owner, the
launch-pin flag, the environment, the host record and the control-context tree
— and the expected answer: ``allowed``, ``reason`` and, on a write allow,
``position`` (``asked`` or ``allowed``; :func:`expected_position` hands it to the
surface projections). Every row is JSON only, so the package verdict and the
hook's stdlib twin read the same table; ``applies_to`` names which of the two a
row drives.

A row's ``switch`` is what the file holds: the stored words ``asked`` and
``allowed``, with ``off``, an unknown value or the legacy ``on`` only on the rows
about those spellings.

Staging is environment and files only (never a patched package internal), so a
row stages the same way for either side:

* the host record is written under ``OSPREY_AGENT_DATA_ROOT`` through
  :func:`osprey_connectors_tests._harness._control_context_fixtures.write_control_context`,
  and its ``switch`` laid over that file verbatim;
* a tree is a directory bound through ``OSPREY_CONTROL_CONTEXT_TREE``, with one
  record per owner;
* a controls server's report (``report``: ``server`` ``live`` or ``dead``, and
  the ``targets`` metadata it publishes) is written beside the host record
  through :func:`~osprey_connectors_tests._harness._control_context_fixtures.write_server_report`,
  so a row's ``kind`` comes from the published ``machine_kind`` (C-KIND).

The loader fails on a row whose reason is not a member of
:data:`~osprey_connectors.refusals.REASONS`, and on a member no row expects —
except ``stopped_since`` and ``switching``, which the verdict never answers
(their callers hold the record's lock and test them there).

This is a helper module, not a conftest: test modules import the fixture and
the rows by name. It lives in the connectors suite's harness because both the
package tests and the hook twin test in ``tests/hooks`` drive the same table.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, posture_store
from osprey_connectors.posture_store import NO_OWNER
from osprey_connectors.refusals import REASONS
from osprey_connectors_tests._harness import (
    VERDICT_ENV_VARS,
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
    write_payload,
    write_server_report,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES_PATH = FIXTURES / "write_verdict_cases.json"
LOGIN_TIERS_SHAPE_PATH = FIXTURES / "login_tiers_shape.json"

EXECUTION_MODE_ENV_VAR = "OSPREY_EXECUTION_MODE"

#: The reasons no row expects: the verdict never answers them.
UNCOVERED_REASONS = frozenset({"session_held", "stopped_since", "switching"})

#: Which sides a row can drive.
SIDES = frozenset({"package", "twin"})

#: The variables a row may set. Every one is cleared before each row.
ROW_ENV_VARS = frozenset(
    {
        EXECUTION_MODE_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.DISPATCH_RUN_ENV_VAR,
    }
)
# What clears them is :func:`isolate_verdict_environment`; a row variable it does
# not know would leak from one row into the next.
assert frozenset(VERDICT_ENV_VARS) >= ROW_ENV_VARS

#: The JSON marker for the explicit no-owner sentinel, ``{"marker": "no_owner"}``.
NO_OWNER_MARKER = {"marker": "no_owner"}

#: The section source that reads ``login_tiers_shape.json``'s example section.
LOGIN_TIERS_SHAPE_SOURCE = "login_tiers_shape"

_ROW_KEYS = frozenset(
    {
        "id",
        "about",
        "applies_to",
        "verdict",
        "section",
        "section_from",
        "target",
        "owner",
        "ignore_launch_pin",
        "env",
        "record",
        "tree",
        "report",
        "expect",
    }
)
_REQUIRED_KEYS = frozenset({"id", "applies_to", "verdict", "expect"})
_EXPECT_KEYS = frozenset({"allowed", "reason", "target", "kind", "position"})
_POSITIONS = frozenset({"asked", "allowed"})
_RECORD_STATUSES = frozenset({"ok", "corrupt", "foreign_schema"})
_TREE_RECORD_STATUSES = frozenset({"ok", "corrupt", "foreign_schema", "dir_only", "planted_link"})
_TREE_BINDS = frozenset({"provisioned", "unprovisioned", "relative"})
_KINDS = frozenset({"real", "rehearsal", "simulator"})
_BACKSTOP_ONLY_KEYS = frozenset({"section", "section_from", "env", "record", "tree"})
_REPORT_SERVERS = frozenset({"live", "dead"})

#: A PID no test host runs: a report under it is a dead server's.
DEAD_SERVER_PID = 2_147_483_646


class CaseTableError(ValueError):
    """The case table is malformed or does not cover the reasons."""


@dataclass(frozen=True)
class Case:
    """One row of the case table, as loaded.

    ``owner`` is the raw JSON value; :attr:`has_owner` says whether the row
    passes one at all (an absent key means the verdict asks the owner ladder).
    """

    id: str
    applies_to: frozenset[str]
    verdict: str
    section: Any
    target: Any
    has_owner: bool
    owner: Any
    ignore_launch_pin: bool
    env: Mapping[str, str]
    record: Mapping[str, Any] | None
    tree: Mapping[str, Any] | None
    expect: Mapping[str, Any]
    section_from: str | None = None
    report: Mapping[str, Any] | None = None


# -- loading ----------------------------------------------------------------


def login_tiers_shape() -> dict[str, Any]:
    """The parsed ``login_tiers_shape.json``."""
    shape: dict[str, Any] = json.loads(LOGIN_TIERS_SHAPE_PATH.read_text(encoding="utf-8"))
    return shape


def _resolve_section(row: Mapping[str, Any]) -> Any:
    source = row.get("section_from")
    if source is None:
        return row.get("section")
    if source != LOGIN_TIERS_SHAPE_SOURCE:
        raise CaseTableError(f"{row['id']}: unknown section_from {source!r}")
    # The shape's example section, with the row's own keys laid over it.
    extra = row.get("section", {})
    if not isinstance(extra, dict):
        raise CaseTableError(f"{row['id']}: section over a section_from must be an object")
    return {**login_tiers_shape()["example"]["section"], **extra}


def _check_record(case_id: str, record: Any, statuses: frozenset[str]) -> None:
    if not isinstance(record, dict) or record.get("status") not in statuses:
        raise CaseTableError(f"{case_id}: record status must be one of {sorted(statuses)}")
    unknown = set(record) - {"status", "switch", "posture"}
    if unknown:
        raise CaseTableError(f"{case_id}: unknown record keys {sorted(unknown)}")


def _check_row(row: Any) -> None:
    if not isinstance(row, dict):
        raise CaseTableError(f"row is not an object: {row!r}")
    case_id = row.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise CaseTableError(f"row without a string id: {row!r}")
    missing = _REQUIRED_KEYS - set(row)
    if missing:
        raise CaseTableError(f"{case_id}: missing keys {sorted(missing)}")
    unknown = set(row) - _ROW_KEYS
    if unknown:
        raise CaseTableError(f"{case_id}: unknown keys {sorted(unknown)}")

    applies = row["applies_to"]
    if not isinstance(applies, list) or not applies or not set(applies) <= SIDES:
        raise CaseTableError(f"{case_id}: applies_to must name some of {sorted(SIDES)}")
    if row["verdict"] not in {"write", "backstop"}:
        raise CaseTableError(f"{case_id}: verdict must be write or backstop")
    if row["verdict"] == "backstop":
        extra = set(row) - _REQUIRED_KEYS - _BACKSTOP_ONLY_KEYS - {"about"}
        if extra:
            raise CaseTableError(f"{case_id}: a backstop row takes no {sorted(extra)}")

    env = row.get("env", {})
    if not isinstance(env, dict) or not set(env) <= ROW_ENV_VARS:
        raise CaseTableError(f"{case_id}: env may set only {sorted(ROW_ENV_VARS)}")
    if not all(isinstance(value, str) for value in env.values()):
        raise CaseTableError(f"{case_id}: env values must be strings")
    if not isinstance(row.get("ignore_launch_pin", False), bool):
        raise CaseTableError(f"{case_id}: ignore_launch_pin must be a boolean")

    if row.get("record") is not None:
        _check_record(case_id, row["record"], _RECORD_STATUSES)
    tree = row.get("tree")
    if tree is not None:
        if not isinstance(tree, dict) or tree.get("bind") not in _TREE_BINDS:
            raise CaseTableError(f"{case_id}: tree bind must be one of {sorted(_TREE_BINDS)}")
        if set(tree) - {"bind", "records"}:
            raise CaseTableError(f"{case_id}: unknown tree keys")
        records = tree.get("records", {})
        if not isinstance(records, dict):
            raise CaseTableError(f"{case_id}: tree records must be an object")
        for record in records.values():
            _check_record(case_id, record, _TREE_RECORD_STATUSES)

    report = row.get("report")
    if report is not None:
        if not isinstance(report, dict) or report.get("server") not in _REPORT_SERVERS:
            raise CaseTableError(
                f"{case_id}: report server must be one of {sorted(_REPORT_SERVERS)}"
            )
        if set(report) - {"server", "targets"} or not isinstance(report.get("targets"), dict):
            raise CaseTableError(f"{case_id}: a report takes a server and a targets object")

    expect = row["expect"]
    if not isinstance(expect, dict) or not {"allowed", "reason"} <= set(expect):
        raise CaseTableError(f"{case_id}: expect needs allowed and reason")
    if set(expect) - _EXPECT_KEYS:
        raise CaseTableError(f"{case_id}: unknown expect keys")
    if not isinstance(expect["allowed"], bool):
        raise CaseTableError(f"{case_id}: expect.allowed must be a boolean")
    reason = expect["reason"]
    if expect["allowed"] != (reason is None):
        raise CaseTableError(f"{case_id}: an allow has no reason and a refusal has one")
    if reason is not None and reason not in REASONS:
        raise CaseTableError(f"{case_id}: reason {reason!r} is not in REASONS")
    if reason in UNCOVERED_REASONS:
        raise CaseTableError(f"{case_id}: the verdict never answers {reason!r}")
    if expect.get("kind") is not None and expect["kind"] not in _KINDS:
        raise CaseTableError(f"{case_id}: kind must be one of {sorted(_KINDS)}")
    position = expect.get("position")
    if expect["allowed"] and row["verdict"] == "write":
        if position not in _POSITIONS:
            raise CaseTableError(f"{case_id}: a write allow states position {sorted(_POSITIONS)}")
    elif position is not None:
        raise CaseTableError(f"{case_id}: only a write allow has a position")


def check_coverage(cases: list[Case]) -> None:
    """Fail unless every reason the verdict can answer has a row.

    Raises:
        CaseTableError: a member of ``REASONS`` (outside
            :data:`UNCOVERED_REASONS`) that no row expects, a row expecting a
            reason outside ``REASONS``, a table with no allow, or a repeated id.
    """
    ids = [case.id for case in cases]
    repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if repeated:
        raise CaseTableError(f"repeated row ids: {repeated}")
    seen = {case.expect["reason"] for case in cases}
    stray = sorted(reason for reason in seen - {None} if reason not in REASONS)
    if stray:
        raise CaseTableError(f"reasons not in REASONS: {stray}")
    uncovered = sorted(set(REASONS) - UNCOVERED_REASONS - seen)
    if uncovered:
        raise CaseTableError(f"reasons with no row: {uncovered}")
    if None not in seen:
        raise CaseTableError("no row allows")


def parse_cases(rows: Any) -> list[Case]:
    """Check *rows* (the parsed JSON list) and build the cases."""
    if not isinstance(rows, list) or not rows:
        raise CaseTableError("the case table is not a non-empty list")
    cases = []
    for row in rows:
        _check_row(row)
        cases.append(
            Case(
                id=row["id"],
                applies_to=frozenset(row["applies_to"]),
                verdict=row["verdict"],
                section=_resolve_section(row),
                target=row.get("target"),
                has_owner="owner" in row,
                owner=row.get("owner"),
                ignore_launch_pin=row.get("ignore_launch_pin", False),
                env=dict(row.get("env", {})),
                record=row.get("record"),
                tree=row.get("tree"),
                expect=dict(row["expect"]),
                section_from=row.get("section_from"),
                report=row.get("report"),
            )
        )
    check_coverage(cases)
    return cases


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Load and check the case table at *path*."""
    return parse_cases(json.loads(path.read_text(encoding="utf-8")))


CASES = load_cases()


def cases_for(side: str) -> list[Case]:
    """The rows that drive *side* (``package`` or ``twin``)."""
    return [case for case in CASES if side in case.applies_to]


def owner_argument(case: Case, no_owner: Any = NO_OWNER) -> dict[str, Any]:
    """The ``owner=`` keyword for *case*'s verdict call; empty when it passes none.

    *no_owner* is the sentinel the side under test spells "no owner" with, so
    the twin decodes the marker to its own.
    """
    if not case.has_owner:
        return {}
    if case.owner == NO_OWNER_MARKER:
        return {"owner": no_owner}
    return {"owner": case.owner}


def expected_position(case: Case) -> str | None:
    """The position *case* expects: ``asked`` or ``allowed`` on a write allow, else ``None``.

    A refusal or a backstop row may leave the key out; the loader has already
    refused any other value there.
    """
    position: str | None = case.expect.get("position")
    return position


# -- staging ----------------------------------------------------------------


def host_root(tmp_path: Path) -> Path:
    """The agent-data root :func:`case_environment` points the host at."""
    return tmp_path / "data"


@pytest.fixture
def case_environment(monkeypatch, tmp_path) -> Iterator[None]:
    """No run mode, launch pin, stamped owner or tree; the host root at a scratch dir.

    Clears ``OSPREY_CONTROL_OWNER`` and ``OSPREY_LAUNCH_POSTURE`` too, which
    the root conftest leaves alone.
    """
    isolate_verdict_environment(monkeypatch, host_root(tmp_path))
    yield
    reset_verdict_state()


def _stage_host_record(root: Path, record: Mapping[str, Any]) -> None:
    path = write_control_context(root, owned_by=None)
    if record["status"] == "corrupt":
        write_payload(path, "{not json")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Verbatim over the epoch the writer stamped, so a legacy ``on``, an unknown
    # value or a stored ``off`` reaches the reader as written: the gesture takes
    # none of them.
    switch = dict(record.get("switch") or {})
    payload["switch"] = switch
    payload["posture"] = control_context.posture_mirror(switch)
    if record["status"] == "foreign_schema":
        payload["schema"] = 2
    if "posture" in record:
        # An old narrowing, laid over the mirror verbatim: no writer stores
        # one any more, so a row about a file written before the mirror has
        # to put it there by hand.
        payload["posture"] = record["posture"]
    write_payload(path, payload)


def _tree_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = control_context.ControlContext(
        target="live",
        generation=3,
        switch_epoch=1,
    ).to_payload()
    # Verbatim, so a degraded value — or an old narrowing — reaches the reader
    # as written rather than as the mirror the writer derives.
    payload["switch"] = dict(record.get("switch") or {})
    if "posture" in record:
        payload["posture"] = record["posture"]
    return payload


def _stage_tree(tmp_path: Path, tree: Mapping[str, Any], monkeypatch) -> None:
    bind = tree["bind"]
    if bind == "relative":
        monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "relative/tree")
        return
    base = tmp_path / "shared" / posture_store.STATE_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    if bind == "provisioned":
        (base / posture_store.CONTROL_TREE_MARKER_NAME).write_text(
            "provisioned\n", encoding="utf-8"
        )
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(base))
    for owner, record in tree.get("records", {}).items():
        path = base / owner / control_context.RECORD_FILENAME
        status = record["status"]
        if status == "dir_only":
            path.parent.mkdir(parents=True, exist_ok=True)
        elif status == "corrupt":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json", encoding="utf-8")
        elif status == "planted_link":
            elsewhere = tmp_path / "elsewhere" / control_context.RECORD_FILENAME
            write_payload(elsewhere, _tree_payload({**record, "status": "ok"}))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(elsewhere)
        else:
            payload = _tree_payload(record)
            if status == "foreign_schema":
                payload["schema"] = 2
            write_payload(path, payload)


def stage(case: Case, monkeypatch, tmp_path: Path) -> None:
    """Lay *case*'s environment, host record and tree down under *tmp_path*.

    Run under :func:`case_environment`, which has already cleared everything a
    row may set.
    """
    for name, value in case.env.items():
        monkeypatch.setenv(name, value)
    if case.record is not None:
        _stage_host_record(host_root(tmp_path), case.record)
    if case.tree is not None:
        _stage_tree(tmp_path, case.tree, monkeypatch)
    if case.report is not None:
        pid = os.getpid() if case.report["server"] == "live" else DEAD_SERVER_PID
        write_server_report(host_root(tmp_path), pid, targets=case.report["targets"])
    reset_verdict_state()
