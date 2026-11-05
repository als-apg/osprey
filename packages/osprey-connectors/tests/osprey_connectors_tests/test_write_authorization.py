"""A program's bounded write authorization: :mod:`osprey_connectors.write_authorization`.

The grant and the ledger are two files with two writers. These tests pin the
formats and their fail-closed readers, the verdict's terms row by row (the case
table ``_harness/fixtures/authorization_cases.json``), its composition with the
switch verdict, the step accounting under concurrent appenders in several
processes, the lease and the holder's liveness, self-issue at ``allowed``, the
approval-time overlap check and the stop's revocation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, posture_store, refusals, session_hold
from osprey_connectors import write_authorization as wa
from osprey_connectors.write_authorization import (
    Approver,
    AuthorizationRefused,
    ChannelRange,
    Grant,
    Ledger,
    LedgerStep,
    Revocation,
    auth_overlap,
    authorization_verdict,
    authorized_execution_verdict,
    issue_grant,
    list_grants,
    parse_grant,
    parse_ledger,
    renew_lease,
    revoke_grants,
    self_issue,
    spend_step,
    start_lease,
)
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    pin_identity,
    write_control_context,
    write_payload,
)
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY

SECTION = {"type": "epics", "writes": "allowed"}
NEVER = {"type": "epics", "writes": "never"}
TARGET = "va"
ME = FIXTURE_IDENTITY
OPS = [("A:SP", 5.0)]
CHANNELS = {"A:SP": ChannelRange(0.0, 10.0), "B:SP": ChannelRange(-1.0, 1.0)}

#: Far enough ahead that nothing in a test reaches it.
FAR_FUTURE_NS = time.time_ns() + 10 * 365 * 24 * 3600 * 10**9

#: A PID no test host runs.
ABSENT_PID = 2_147_483_646

CASES_PATH = Path(__file__).parent / "_harness" / "fixtures" / "authorization_cases.json"


# -- the environment ----------------------------------------------------------


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with this login's record (every switch off, an epoch) and no grant."""
    data = tmp_path / "root"
    isolate_verdict_environment(monkeypatch, data)
    for name in (wa.AUTHORIZATION_ENV_VAR, wa.LEDGER_DIR_ENV_VAR, "OSPREY_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    write_control_context(data, switch={})
    return data


def record() -> control_context.ControlContext:
    control_context.invalidate_cache()
    current = control_context.read_record()
    assert current is not None
    return current


def epoch() -> int:
    value = record().switch_epoch
    assert isinstance(value, int)
    return value


def switch(position: str) -> None:
    """Move the host's switch for the target through the person's gesture."""
    if position == "off":
        control_context.set_switch([TARGET], "off")
    else:
        control_context.set_switch([TARGET], position, epoch=epoch())  # type: ignore[arg-type]
    control_context.invalidate_cache()


def grants_dir() -> Path:
    path = wa.authorizations_dir()
    assert path is not None
    return path


def make_grant(**fields: Any) -> Grant:
    now = time.time_ns()
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "owner": ME,
        "holder": ME,
        "target": TARGET,
        "switch_epoch": epoch(),
        "channels": CHANNELS,
        "steps": 3,
        "issued_at_ns": now,
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 60.0,
        "approver": Approver(identity="alice", surface="chip", at_ns=now),
        "label": "run-1",
        "revoked": None,
    }
    values.update(fields)
    return Grant(**values)


def approve(**fields: Any) -> Grant:
    """Issue a grant into this login's directory and start this process's lease."""
    grant = make_grant(**fields)
    issue_grant(grant)
    start_lease(grant.id)
    return grant


def verdict(grant_id: str, operations: Any = OPS, section: Any = SECTION) -> Any:
    return authorization_verdict(section, TARGET, operations, authorization=grant_id)


def ledger_path(grant_id: str) -> Path:
    directory = wa.ledger_dir()
    assert directory is not None
    return directory / f"{grant_id}.ledger.json"


def ledger(grant_id: str) -> Ledger:
    return parse_ledger(ledger_path(grant_id).read_text())


def grant_path(grant_id: str) -> Path:
    return grants_dir() / f"{grant_id}.json"


def at(monkeypatch: pytest.MonkeyPatch, now_ns: int) -> None:
    monkeypatch.setattr(wa, "_now_ns", lambda: now_ns)


# -- the case table -----------------------------------------------------------


def _load_cases() -> list[dict[str, Any]]:
    rows = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    allowed_keys = {
        "id", "about", "switch", "grant", "grant_text_replace", "ledger", "operations",
        "env", "record", "expect",
    }  # fmt: skip
    for row in rows:
        assert set(row) <= allowed_keys, row["id"]
        assert set(row["expect"]) == {"allowed", "reason"}, row["id"]
        assert row["expect"]["allowed"] is (row["expect"]["reason"] is None), row["id"]
        assert row["expect"]["reason"] in (None, *refusals.REASONS), row["id"]
    return rows


CASES = _load_cases()


def test_the_table_covers_every_reason_a_write_can_meet():
    """Every ``auth_*`` reason but the approval-time one has a row, and so does an allow."""
    seen = {row["expect"]["reason"] for row in CASES}
    wanted = {r for r in refusals.REASONS if r.startswith("auth_")} - {"auth_overlap"}
    assert wanted <= seen
    assert None in seen
    assert len({row["id"] for row in CASES}) == len(CASES)


def _resolve(value: Any, stale_epoch: int) -> Any:
    if value == "@past":
        return 1
    if value == "@stale_epoch":
        return stale_epoch
    if isinstance(value, dict):
        return {key: _resolve(item, stale_epoch) for key, item in value.items()}
    return value


def _stage(row: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> str:
    """Lay the row's files and environment down; the id the process names."""
    if row.get("switch", "off") != "off":
        switch(row["switch"])
    grant = make_grant(steps=3)
    if row.get("grant", {}) is not None:
        payload = grant.to_payload()
        payload.update(_resolve(row.get("grant", {}), epoch() - 1))
        text = json.dumps(payload, indent=2, sort_keys=True)
        if "grant_text_replace" in row:
            old, new = row["grant_text_replace"]
            assert old in text, row["id"]
            text = text.replace(old, new)
        grants_dir().mkdir(parents=True, exist_ok=True)
        grant_path(grant.id).write_text(text)
    if row.get("ledger", {}) is not None:
        payload = Ledger(
            authorization=grant.id,
            holder_pid=os.getpid(),
            holder_start=wa.process_start_time(os.getpid()),
            pid_ns=wa.pid_namespace(),
            lease_until_ns=FAR_FUTURE_NS,
        ).to_payload()
        overrides = dict(row.get("ledger", {}))
        steps = overrides.get("steps")
        if isinstance(steps, str) and steps.startswith("@spent:"):
            count = int(steps.split(":")[1])
            overrides["steps"] = [
                {"n": n, "at_ns": n, "channels": ["A:SP"]} for n in range(1, count + 1)
            ]
        payload.update(_resolve(overrides, 0))
        grants_dir().mkdir(parents=True, exist_ok=True)
        ledger_path(grant.id).write_text(json.dumps(payload))
    if row.get("record") == "corrupt":
        path = control_context.record_path()
        assert path is not None
        path.write_text("{not json")
        control_context.invalidate_cache()
    for name, value in row.get("env", {}).items():
        monkeypatch.setenv(name, value)
    return grant.id


def _operations(row: dict[str, Any]) -> Any:
    operations = row.get("operations", [["A:SP", 5.0]])
    if isinstance(operations, dict):
        return operations
    return [tuple(pair) for pair in operations]


@pytest.mark.parametrize("row", CASES, ids=[row["id"] for row in CASES])
def test_the_verdict_matches_the_row(row: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    grant_id = _stage(row, monkeypatch)

    answer = verdict(grant_id, _operations(row))

    assert (answer.allowed, answer.reason) == (row["expect"]["allowed"], row["expect"]["reason"])
    assert answer.authorization == grant_id
    assert answer.position is None
    assert answer.target == TARGET


@pytest.mark.parametrize("row", CASES, ids=[row["id"] for row in CASES])
def test_spending_answers_as_the_verdict_does(row: dict[str, Any], monkeypatch):
    """One order: the writer refuses for the reason the reader names, and spends only an allow."""
    grant_id = _stage(row, monkeypatch)
    before = ledger_path(grant_id).read_bytes() if ledger_path(grant_id).exists() else None

    answer = spend_step(SECTION, TARGET, _operations(row), authorization=grant_id)

    assert (answer.allowed, answer.reason) == (row["expect"]["allowed"], row["expect"]["reason"])
    after = ledger_path(grant_id).read_bytes() if ledger_path(grant_id).exists() else None
    if answer.allowed:
        assert before is not None and after is not None
        assert len(parse_ledger(after).steps) == len(parse_ledger(before).steps) + 1
    elif not row["id"].startswith("self-issued"):
        assert after == before


# -- the formats --------------------------------------------------------------


class TestFormats:
    def test_a_grant_round_trips(self):
        grant = make_grant(revoked=Revocation(at_ns=5, by="bob"))
        assert parse_grant(json.dumps(grant.to_payload())) == grant

    def test_a_ledger_round_trips(self):
        entry = Ledger(
            authorization="a" * 32,
            holder_pid=7,
            holder_start="100",
            pid_ns="pid:[1]",
            lease_until_ns=9,
            steps=(LedgerStep(n=1, at_ns=3, channels=("A:SP", "B:SP")),),
        )
        assert parse_ledger(json.dumps(entry.to_payload())) == entry

    @pytest.mark.parametrize(
        "change",
        [
            {"schema": True},
            {"schema": "1"},
            {"id": "ABC"},
            {"id": "a" * 31},
            {"owner": ""},
            {"owner": " alice"},
            {"holder": 7},
            {"target": None},
            {"switch_epoch": -1},
            {"switch_epoch": 1.5},
            {"channels": {}},
            {"channels": []},
            {"channels": {"": {"min": 0, "max": 1}}},
            {"channels": {"A:SP": {"min": 0}}},
            {"channels": {"A:SP": {"min": 0, "max": 1, "step": 1}}},
            {"channels": {"A:SP": {"min": "0", "max": 1}}},
            {"channels": {"A:SP": {"min": False, "max": 1}}},
            {"channels": {"A:SP": {"min": 2, "max": 1}}},
            {"steps": 0},
            {"steps": 1.0},
            {"steps": True},
            {"issued_at_ns": None},
            {"expires_at_ns": 0},
            {"lease_s": 0},
            {"lease_s": -1},
            {"lease_s": "60"},
            {"approver": {"identity": "a", "surface": "phone", "at_ns": 1}},
            {"approver": {"identity": "a", "surface": "chip"}},
            {"approver": "alice"},
            {"label": None},
            {"revoked": True},
            {"revoked": {"at_ns": 1}},
            {"revoked": {"at_ns": 1, "by": ""}},
            {"extra": 1},
        ],
        ids=repr,
    )
    def test_a_malformed_grant_is_refused(self, change: dict[str, Any]):
        payload = make_grant().to_payload()
        payload.update(change)
        with pytest.raises(ValueError):
            parse_grant(json.dumps(payload))

    @pytest.mark.parametrize("missing", sorted(wa._GRANT_KEYS))
    def test_a_grant_missing_any_field_is_refused(self, missing: str):
        payload = make_grant().to_payload()
        del payload[missing]
        with pytest.raises(ValueError):
            parse_grant(payload)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "[]",
            "null",
            "{",
            '{"schema": 1, "schema": 1}',
        ],
    )
    def test_a_grant_that_is_not_one_object_is_refused(self, text: str):
        with pytest.raises(ValueError):
            parse_grant(text)

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_bound_is_refused(self, constant: str):
        text = json.dumps(make_grant().to_payload()).replace('"max": 10.0', f'"max": {constant}')
        assert constant in text
        with pytest.raises(ValueError):
            parse_grant(text)

    @pytest.mark.parametrize(
        "change",
        [
            {"schema": 2},
            {"authorization": "x"},
            {"holder_pid": 0},
            {"holder_pid": True},
            {"holder_start": 5},
            {"pid_ns": ["pid:[1]"]},
            {"lease_until_ns": -1},
            {"lease_until_ns": None},
            {"steps": {}},
            {"steps": [{"n": 1, "at_ns": 1, "channels": []}]},
            {"steps": [{"n": 1, "at_ns": 1, "channels": [""]}]},
            {"steps": [{"n": 1, "at_ns": 1}]},
            {"steps": [{"n": 1, "at_ns": 1, "channels": ["A"], "extra": 1}]},
            {
                "steps": [
                    {"n": 1, "at_ns": 1, "channels": ["A"]},
                    {"n": 1, "at_ns": 2, "channels": ["A"]},
                ]
            },
            {"extra": 1},
        ],
        ids=repr,
    )
    def test_a_malformed_ledger_is_refused(self, change: dict[str, Any]):
        payload = Ledger(
            authorization="a" * 32, holder_pid=1, holder_start=None, pid_ns=None, lease_until_ns=1
        ).to_payload()
        payload.update(change)
        with pytest.raises(ValueError):
            parse_ledger(json.dumps(payload))

    def test_a_channel_range_holds_finite_reals_only(self):
        bound = ChannelRange(0.0, 1.0)
        assert bound.holds(0) and bound.holds(1.0) and bound.holds(0.5)
        for value in (math.nan, math.inf, -math.inf, True, None, "0.5", 1.0000001, -1e-9):
            assert not bound.holds(value), value

    def test_the_record_is_untouched_by_a_grant(self):
        """Old readers: the record keeps its fields and schema, and the switch verdict is off."""
        path = control_context.record_path()
        assert path is not None
        before = path.read_bytes()
        approve()
        assert path.read_bytes() == before
        assert set(json.loads(before)) == {
            "schema", "owner", "target", "generation", "posture", "last_switch", "switch", "switch_epoch",
        }  # fmt: skip
        assert posture_store.write_verdict(SECTION, TARGET).reason == "switch_off"


# -- fail-closed reads --------------------------------------------------------


def _relocate(source: Path, destination: Path) -> None:
    """Copy *source* (a file or a directory) to *destination* and remove it, to plant a link there."""
    if source.is_dir():
        shutil.copytree(source, destination)
        shutil.rmtree(source)
    else:
        shutil.copyfile(source, destination)
        source.unlink()


def _skip_as_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a file whatever its mode")


class TestFailClosedReads:
    def test_a_grant_that_is_not_utf8(self):
        grant = approve()
        grant_path(grant.id).write_bytes(b"\xff\xfe{}")
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_an_oversized_grant(self):
        grant = approve()
        payload = grant.to_payload()
        payload["label"] = "x" * (wa._GRANT_READ_LIMIT + 1)
        grant_path(grant.id).write_text(json.dumps(payload))
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_link_at_the_grant(self, tmp_path: Path):
        grant = approve()
        elsewhere = tmp_path / "planted.json"
        _relocate(grant_path(grant.id), elsewhere)
        grant_path(grant.id).symlink_to(elsewhere)
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_link_at_the_authorizations_directory(self, tmp_path: Path, monkeypatch):
        grant = approve()
        # The ledger lives apart, so only the grants directory is planted.
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir()
        _relocate(ledger_path(grant.id), ledgers / ledger_path(grant.id).name)
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        assert verdict(grant.id).allowed
        moved = tmp_path / "moved"
        _relocate(grants_dir(), moved)
        grants_dir().symlink_to(moved)
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_file_where_the_authorizations_directory_belongs(self):
        grant = make_grant()
        grants_dir().parent.mkdir(parents=True, exist_ok=True)
        grants_dir().write_text("")
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_directory_at_the_grant(self):
        grant = approve()
        grant_path(grant.id).unlink()
        grant_path(grant.id).mkdir()
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_fifo_at_the_grant_neither_hangs_nor_allows(self):
        grant = approve()
        grant_path(grant.id).unlink()
        os.mkfifo(grant_path(grant.id))
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_grant_this_process_cannot_read(self):
        _skip_as_root()
        grant = approve()
        grant_path(grant.id).chmod(0)
        try:
            assert verdict(grant.id).reason == "auth_unreadable"
        finally:
            grant_path(grant.id).chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_a_ledger_this_process_cannot_read(self):
        _skip_as_root()
        grant = approve()
        ledger_path(grant.id).chmod(0)
        try:
            assert verdict(grant.id).reason == "auth_unreadable"
        finally:
            ledger_path(grant.id).chmod(stat.S_IRUSR | stat.S_IWUSR)

    def test_a_link_at_the_ledger(self, tmp_path: Path):
        grant = approve()
        elsewhere = tmp_path / "planted.ledger.json"
        _relocate(ledger_path(grant.id), elsewhere)
        ledger_path(grant.id).symlink_to(elsewhere)
        assert verdict(grant.id).reason == "auth_unreadable"
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"
        assert len(parse_ledger(elsewhere.read_text()).steps) == 0

    def test_a_ledger_naming_another_authorization(self):
        grant = approve()
        payload = json.loads(ledger_path(grant.id).read_text())
        payload["authorization"] = "b" * 32
        ledger_path(grant.id).write_text(json.dumps(payload))
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_missing_ledger_directory(self, tmp_path: Path, monkeypatch):
        grant = approve()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(tmp_path / "nowhere"))
        assert verdict(grant.id).reason == "auth_unreadable"
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"

    def test_a_ledger_directory_that_is_a_file(self, tmp_path: Path, monkeypatch):
        grant = approve()
        planted = tmp_path / "file"
        planted.write_text("")
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(planted))
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"

    @pytest.mark.parametrize("named", ["", "x", "A" * 32, "../" + "a" * 29, "a" * 33, None, 7])
    def test_an_id_that_names_no_file(self, named: Any):
        assert verdict(named).reason == "auth_unreadable"  # type: ignore[arg-type]
        answer = spend_step(SECTION, TARGET, OPS, authorization=named)  # type: ignore[arg-type]
        assert answer.reason == "auth_unreadable"

    def test_an_approval_planted_in_the_holders_own_directory(self, tmp_path: Path, monkeypatch):
        """The holder can write its ledger directory, so only a self-issued grant counts there."""
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        grant = make_grant()
        issue_grant(grant, directory=ledgers)
        with pytest.raises(AuthorizationRefused) as refused:
            start_lease(grant.id)
        assert refused.value.reason == "auth_unreadable"
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_verdict_that_raises_refuses(self, monkeypatch):
        grant = approve()

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(wa, "_judge", boom)
        assert verdict(grant.id).reason == "auth_unreadable"
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"

    def test_a_ledger_lock_that_is_a_link_refuses(self, tmp_path: Path):
        grant = approve()
        lock = ledger_path(grant.id).with_name(ledger_path(grant.id).name + ".lock")
        lock.unlink(missing_ok=True)
        lock.symlink_to(tmp_path / "elsewhere")
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"
        assert ledger(grant.id).steps == ()


# -- the composition ----------------------------------------------------------


class TestComposition:
    def test_a_process_naming_no_authorization_keeps_the_switch_verdict(self):
        approve()
        answer = authorized_execution_verdict(SECTION, TARGET, OPS, authorization=None)
        assert (answer.reason, answer.authorization) == ("switch_off", None)

    def test_the_switch_off_is_lifted_for_the_listed_channels_only(self):
        grant = approve()
        assert authorized_execution_verdict(SECTION, TARGET, OPS, authorization=grant.id).allowed
        unlisted = authorized_execution_verdict(
            SECTION, TARGET, [("C:SP", 1.0)], authorization=grant.id
        )
        assert unlisted.reason == "auth_channel"

    @pytest.mark.parametrize("position", ["asked", "allowed"])
    def test_a_switch_turned_on_mid_run_does_not_lift_the_bounds(self, position: str):
        grant = approve()
        switch(position)
        assert posture_store.write_verdict(SECTION, TARGET).allowed
        for operations, reason in (
            ([("C:SP", 1.0)], "auth_channel"),
            ([("A:SP", 11.0)], "auth_range"),
        ):
            answer = authorized_execution_verdict(
                SECTION, TARGET, operations, authorization=grant.id
            )
            assert answer.reason == reason

    def test_an_authorization_lifts_no_tier(self):
        grant = approve()
        answer = authorized_execution_verdict(NEVER, TARGET, OPS, authorization=grant.id)
        assert (answer.reason, answer.authorization) == ("tier_never", grant.id)

    def test_an_authorization_lifts_no_read_only_run(self, monkeypatch):
        grant = approve()
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        answer = authorized_execution_verdict(SECTION, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "readonly_run"

    def test_an_authorization_lifts_no_hold(self, tmp_path: Path, monkeypatch):
        grant = approve()
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(tmp_path / "holds"))
        monkeypatch.setenv("OSPREY_SESSION_ID", "s1")
        answer = authorized_execution_verdict(SECTION, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "session_held"

    def test_an_authorization_lifts_no_pin_with_the_switch_on_or_off(self, monkeypatch):
        grant = approve()
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "va=sandbox")
        assert (
            authorized_execution_verdict(SECTION, TARGET, OPS, authorization=grant.id).reason
            == "launch_pinned"
        )
        switch("allowed")
        assert (
            authorized_execution_verdict(SECTION, TARGET, OPS, authorization=grant.id).reason
            == "launch_pinned"
        )

    def test_spending_through_the_composition(self):
        grant = approve()
        answer = authorized_execution_verdict(
            SECTION, TARGET, OPS, authorization=grant.id, spend=True
        )
        assert answer.allowed and answer.authorization == grant.id
        assert len(ledger(grant.id).steps) == 1
        assert authorized_execution_verdict(SECTION, TARGET, OPS, authorization=grant.id).allowed
        assert len(ledger(grant.id).steps) == 1


# -- steps --------------------------------------------------------------------


class TestSteps:
    def test_a_batch_is_one_step(self):
        grant = approve()
        assert spend_step(SECTION, TARGET, [("A:SP", 1.0), ("B:SP", 0.0)], authorization=grant.id)
        (entry,) = ledger(grant.id).steps
        assert (entry.n, entry.channels) == (1, ("A:SP", "B:SP"))

    def test_the_budget_ends_the_run_and_nothing_renews_it(self):
        grant = approve(steps=2)
        answers = [spend_step(SECTION, TARGET, OPS, authorization=grant.id) for _ in range(4)]
        assert [a.allowed for a in answers] == [True, True, False, False]
        assert answers[-1].reason == "auth_exhausted"
        assert [entry.n for entry in ledger(grant.id).steps] == [1, 2]
        assert renew_lease(grant.id)
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_exhausted"

    def test_a_one_shot_batch_is_judged_and_recorded_whole(self):
        grant = approve()
        batch = iter([("A:SP", 1.0), ("B:SP", 0.0)])
        assert spend_step(SECTION, TARGET, batch, authorization=grant.id).allowed
        assert ledger(grant.id).steps[0].channels == ("A:SP", "B:SP")

    def test_a_refused_write_spends_nothing(self):
        grant = approve()
        before = ledger_path(grant.id).read_bytes()
        assert spend_step(SECTION, TARGET, [("A:SP", 99.0)], authorization=grant.id).reason == (
            "auth_range"
        )
        assert ledger_path(grant.id).read_bytes() == before

    def test_the_read_only_verdict_spends_nothing(self):
        grant = approve()
        for _ in range(5):
            assert verdict(grant.id).allowed
        assert ledger(grant.id).steps == ()

    def test_a_step_that_cannot_be_recorded_is_refused(self, monkeypatch):
        grant = approve()
        before = ledger_path(grant.id).read_bytes()

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(wa, "atomic_replace", disk_full)
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"
        assert ledger_path(grant.id).read_bytes() == before

    def test_a_ledger_held_locked_refuses_rather_than_waits_forever(self, monkeypatch):
        grant = approve()
        monkeypatch.setattr(wa, "_LOCK_TIMEOUT_S", 0.05)
        lock = ledger_path(grant.id).with_name(ledger_path(grant.id).name + ".lock")
        import fcntl

        fd = os.open(lock, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            # flock is per open file description, so this process's own second
            # open contends with it like another process would.
            answer = spend_step(SECTION, TARGET, OPS, authorization=grant.id)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        assert answer.reason == "auth_unreadable"
        assert ledger(grant.id).steps == ()


SPENDER = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path
    from osprey_connectors import write_authorization as wa

    section, grant_id, tries, go = json.loads(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    while not go.exists():
        time.sleep(0.001)
    spent = 0
    for _ in range(tries):
        answer = wa.spend_step(section, "va", [("A:SP", 1.0), ("B:SP", 0.5)], authorization=grant_id)
        if answer.allowed:
            spent += 1
        elif answer.reason != "auth_exhausted":
            print("refused", answer.reason, flush=True)
            sys.exit(1)
    print(spent, flush=True)
    """
)


class TestConcurrentAppenders:
    def test_processes_spend_exactly_the_budget(self, tmp_path: Path):
        """Several processes race for the last steps; the lock lets exactly *steps* through."""
        budget, processes, tries = 13, 6, 5
        grant = approve(steps=budget)
        go = tmp_path / "go"
        spawned = [
            subprocess.Popen(
                [sys.executable, "-c", SPENDER, json.dumps(SECTION), grant.id, str(tries), str(go)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
            for _ in range(processes)
        ]
        go.write_text("")
        outputs = [process.communicate(timeout=120) for process in spawned]
        assert all(process.returncode == 0 for process in spawned), outputs
        assert sum(int(out.strip()) for out, _err in outputs) == budget
        steps = ledger(grant.id).steps
        assert [entry.n for entry in steps] == list(range(1, budget + 1))
        assert all(entry.channels == ("A:SP", "B:SP") for entry in steps)

    def test_threads_spend_exactly_the_budget(self):
        from concurrent.futures import ThreadPoolExecutor

        grant = approve(steps=10)
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(
                pool.map(
                    lambda _: spend_step(SECTION, TARGET, OPS, authorization=grant.id), range(24)
                )
            )
        assert sum(answer.allowed for answer in answers) == 10
        assert {a.reason for a in answers if not a.allowed} == {"auth_exhausted"}
        assert len(ledger(grant.id).steps) == 10


# -- the lease ----------------------------------------------------------------


class TestLease:
    def test_start_records_this_process_as_the_holder(self):
        grant = approve()
        entry = ledger(grant.id)
        assert entry.holder_pid == os.getpid()
        assert entry.pid_ns == wa.pid_namespace()
        assert entry.holder_start == wa.process_start_time(os.getpid())
        assert entry.steps == ()

    def test_a_lease_is_started_once(self):
        grant = approve()
        with pytest.raises(AuthorizationRefused) as refused:
            start_lease(grant.id)
        assert refused.value.reason == "auth_lease_lapsed"

    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            ({"holder": "someone-else"}, "auth_holder"),
            ({"revoked": Revocation(at_ns=1, by="bob")}, "auth_revoked"),
            ({"expires_at_ns": 2, "issued_at_ns": 1}, "auth_expired"),
        ],
    )
    def test_a_lease_is_not_started_on_a_grant_out_of_force(self, fields, reason):
        grant = make_grant(**fields)
        issue_grant(grant)
        with pytest.raises(AuthorizationRefused) as refused:
            start_lease(grant.id)
        assert refused.value.reason == reason
        assert not ledger_path(grant.id).exists()

    def test_renewal_outlives_the_first_lease(self, monkeypatch):
        now = time.time_ns()
        at(monkeypatch, now)
        grant = approve(lease_s=10.0)
        at(monkeypatch, now + 9 * 10**9)
        assert renew_lease(grant.id)
        at(monkeypatch, now + 15 * 10**9)
        assert verdict(grant.id).allowed
        assert ledger(grant.id).lease_until_ns == now + 19 * 10**9

    def test_a_lapsed_lease_refuses_and_does_not_come_back(self, monkeypatch):
        now = time.time_ns()
        at(monkeypatch, now)
        grant = approve(lease_s=10.0)
        at(monkeypatch, now + 10 * 10**9)
        assert verdict(grant.id).reason == "auth_lease_lapsed"
        assert not renew_lease(grant.id)
        at(monkeypatch, now + 1)
        # Even a clock that steps back finds the lease unmoved by the failed renewal.
        assert ledger(grant.id).lease_until_ns == now + 10 * 10**9

    def test_only_the_holder_process_renews(self):
        grant = approve()
        payload = json.loads(ledger_path(grant.id).read_text())
        payload["holder_pid"] = ABSENT_PID
        payload["pid_ns"] = "pid:[0]"
        ledger_path(grant.id).write_text(json.dumps(payload))
        before = ledger_path(grant.id).read_bytes()
        assert not renew_lease(grant.id)
        assert ledger_path(grant.id).read_bytes() == before

    def test_a_revoked_grant_is_not_renewed(self):
        grant = approve()
        revoke_grants(by="bob")
        assert not renew_lease(grant.id)

    def test_a_missing_ledger_is_not_renewed(self):
        grant = make_grant()
        issue_grant(grant)
        assert not renew_lease(grant.id)
        assert not ledger_path(grant.id).exists()


HOLDER = textwrap.dedent(
    """
    import signal, sys
    from osprey_connectors import write_authorization as wa

    wa.start_lease(sys.argv[1])
    print("leased", flush=True)
    signal.pause()
    """
)


class TestHolderProcess:
    @pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM], ids=["sigkill", "sigterm"])
    def test_a_dead_holder_lapses_the_lease_before_its_time(self, sig: signal.Signals):
        """Same PID namespace: the kernel's answer ends the lease, not the clock."""
        grant = make_grant(lease_s=3600.0)
        issue_grant(grant)
        process = subprocess.Popen(
            [sys.executable, "-c", HOLDER, grant.id],
            stdout=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        try:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "leased"
            assert verdict(grant.id).allowed
            process.send_signal(sig)
            process.wait(timeout=30)
        finally:
            process.kill()
            process.wait()
        assert ledger(grant.id).lease_until_ns > time.time_ns()
        assert verdict(grant.id).reason == "auth_lease_lapsed"


class TestHolderLiveness:
    """What the lease concludes about its holder, with the kernel's answer forged."""

    def ledger(self, **fields: Any) -> Ledger:
        values: dict[str, Any] = {
            "authorization": "a" * 32,
            "holder_pid": os.getpid(),
            "holder_start": "100",
            "pid_ns": wa.pid_namespace(),
            "lease_until_ns": FAR_FUTURE_NS,
        }
        values.update(fields)
        return Ledger(**values)

    def forge_kernel(self, monkeypatch, answer: tuple[str, str] | None) -> None:
        monkeypatch.setattr(wa, "_kernel_stat", lambda pid: answer)

    def test_the_same_process(self, monkeypatch):
        self.forge_kernel(monkeypatch, ("R", "100"))
        assert wa._lease_live(self.ledger(), time.time_ns())

    def test_a_reused_pid(self, monkeypatch):
        self.forge_kernel(monkeypatch, ("R", "999"))
        assert not wa._lease_live(self.ledger(), time.time_ns())

    def test_a_zombie_holder(self, monkeypatch):
        self.forge_kernel(monkeypatch, ("Z", "100"))
        assert not wa._lease_live(self.ledger(), time.time_ns())

    def test_no_kernel_view_falls_back_on_the_pid_and_the_lease(self, monkeypatch):
        self.forge_kernel(monkeypatch, None)
        now = time.time_ns()
        assert wa._lease_live(self.ledger(), now)
        assert not wa._lease_live(self.ledger(holder_pid=ABSENT_PID), now)
        assert not wa._lease_live(self.ledger(lease_until_ns=now), now)

    def test_a_reused_pid_refuses_the_write(self, monkeypatch):
        grant = approve()
        self.forge_kernel(monkeypatch, ("S", "not-the-holders-start"))
        payload = json.loads(ledger_path(grant.id).read_text())
        payload["holder_start"] = "the-holders-start"
        ledger_path(grant.id).write_text(json.dumps(payload))
        assert verdict(grant.id).reason == "auth_lease_lapsed"

    def test_a_stat_line_with_awkward_parentheses(self, tmp_path: Path, monkeypatch):
        stat_file = tmp_path / "stat"
        rest = " ".join(str(n) for n in range(4, 22))
        stat_file.write_text(f"123 (odd) name)) S {rest} 777 0 0\n")
        monkeypatch.setattr(wa, "_PROC_STAT", str(stat_file))
        assert wa.process_start_time(123) == "777"


class TestPidNamespaces:
    """A holder's PID is looked up only from its own PID namespace, ``None`` matching ``None``."""

    OTHER = "pid:[0]"  # no namespace has inode 0, so this matches no reader on any platform

    @pytest.mark.parametrize(
        ("reader", "holder", "live"),
        [
            ("pid:[1]", "pid:[1]", False),
            ("pid:[1]", "pid:[2]", True),
            ("pid:[1]", None, True),
            (None, "pid:[1]", True),
            (None, None, False),
        ],
    )
    def test_the_comparison(self, monkeypatch, reader, holder, live):
        """The Linux and macOS answers, with both forged, on either platform."""
        monkeypatch.setattr(wa, "pid_namespace", lambda: reader)
        entry = Ledger(
            authorization="a" * 32,
            holder_pid=ABSENT_PID,
            holder_start=None,
            pid_ns=holder,
            lease_until_ns=FAR_FUTURE_NS,
        )
        assert wa._lease_live(entry, time.time_ns()) is live

    def test_another_namespace_is_judged_by_the_lease_time_alone(self, monkeypatch):
        grant = approve()
        payload = json.loads(ledger_path(grant.id).read_text())
        payload.update(holder_pid=ABSENT_PID, pid_ns=self.OTHER)
        ledger_path(grant.id).write_text(json.dumps(payload))
        assert verdict(grant.id).allowed
        at(monkeypatch, payload["lease_until_ns"])
        assert verdict(grant.id).reason == "auth_lease_lapsed"

    def test_the_namespace_is_read_from_the_link(self, monkeypatch):
        def link(path: str) -> str:
            assert path == "/proc/self/ns/pid"
            return "pid:[42]"

        monkeypatch.setattr(os, "readlink", link)
        assert wa.pid_namespace() == "pid:[42]"

    def test_no_link_is_no_namespace(self, monkeypatch):
        def no_link(path: str) -> str:
            raise FileNotFoundError(path)

        monkeypatch.setattr(os, "readlink", no_link)
        assert wa.pid_namespace() is None


# -- stops --------------------------------------------------------------------


class TestStops:
    def test_a_stop_of_every_machine_ends_it_by_the_epoch_alone(self):
        grant = approve()
        control_context.set_switch("all", "off")
        control_context.invalidate_cache()
        assert verdict(grant.id).reason == "auth_revoked"
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_revoked"
        assert not renew_lease(grant.id)

    def test_a_named_off_keeps_the_epoch_and_the_revocation_ends_it(self):
        grant = approve()
        before = epoch()
        switch("asked")
        switch("allowed")
        assert verdict(grant.id).allowed  # turning a machine on is not what a grant stands on
        switch("off")
        assert epoch() == before
        answer = verdict(grant.id)
        assert answer.reason == "auth_revoked"
        assert ME in (answer.detail or "")
        assert revoke_grants(by="alice", targets=[TARGET]) == ()  # marked once, by the stop

    def test_revocation_names_its_machines_and_is_kept(self):
        on_va = approve()
        on_live = approve(target="live", channels={"L:SP": ChannelRange(0, 1)})
        assert revoke_grants(by="alice", targets=["live"]) == (on_live.id,)
        assert verdict(on_va.id).allowed
        assert revoke_grants(by="bob") == (on_va.id,)
        assert revoke_grants(by="carol") == ()
        assert parse_grant(grant_path(on_va.id).read_text()).revoked == Revocation(
            at_ns=parse_grant(grant_path(on_va.id).read_text()).revoked.at_ns,  # type: ignore[union-attr]
            by="bob",
        )

    def test_revocation_leaves_an_unreadable_grant_as_it_is(self):
        grant = approve()
        grant_path(grant.id).write_text("{")
        assert revoke_grants(by="alice") == ()
        assert grant_path(grant.id).read_text() == "{"

    def test_a_lost_record_ends_it(self):
        grant = approve()
        path = control_context.record_path()
        assert path is not None
        path.unlink()
        control_context.invalidate_cache()
        assert verdict(grant.id).reason == "auth_revoked"


# -- self-issue at allowed ----------------------------------------------------


def _self_issue(**fields: Any) -> Grant:
    values: dict[str, Any] = {
        "steps": 3,
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 60.0,
        "label": "run-self",
    }
    values.update(fields)
    return self_issue(SECTION, TARGET, {"A:SP": (0.0, 10.0)}, **values)


class TestSelfIssue:
    def test_at_allowed_the_holder_writes_its_own_bounded_grant(self, tmp_path, monkeypatch):
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        switch("allowed")
        grant = _self_issue()
        assert (ledgers / f"{grant.id}.json").is_file()
        assert not (grants_dir() / f"{grant.id}.json").exists()
        assert grant.approver.surface == "write_switch"
        assert (grant.owner, grant.holder, grant.switch_epoch) == (ME, ME, epoch())
        start_lease(grant.id)
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).allowed
        assert verdict(grant.id, [("A:SP", 11.0)]).reason == "auth_range"

    @pytest.mark.parametrize(
        ("position", "reason"), [("off", "switch_off"), ("asked", "auth_revoked")]
    )
    def test_below_allowed_nothing_is_issued(self, position, reason):
        if position != "off":
            switch(position)
        with pytest.raises(AuthorizationRefused) as refused:
            _self_issue()
        assert refused.value.reason == reason
        assert list_grants(grants_dir()) == ((), ())

    def test_a_pinned_launch_issues_nothing(self, monkeypatch):
        switch("allowed")
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
        with pytest.raises(AuthorizationRefused) as refused:
            _self_issue()
        assert refused.value.reason == "launch_pinned"

    def test_a_stop_during_the_issue_removes_the_grant(self, monkeypatch):
        switch("allowed")
        real = wa.issue_grant

        def stop_after(grant: Grant, *, directory: Path | None = None, peers: Any = ()) -> Path:
            path = real(grant, directory=directory, peers=peers)
            control_context.set_switch("all", "off")
            control_context.invalidate_cache()
            return path

        monkeypatch.setattr(wa, "issue_grant", stop_after)
        with pytest.raises(AuthorizationRefused):
            _self_issue()
        assert list_grants(grants_dir()) == ((), ())

    # On a host the holder's directory is the owner's, so a stop marks the
    # self-issued grant revoked as well; leaving for ``asked`` lapses its lease.
    @pytest.mark.parametrize(
        ("gesture", "after"), [("asked", "auth_lease_lapsed"), ("off", "auth_revoked")]
    )
    def test_leaving_allowed_ends_it_for_good(self, gesture: str, after: str):
        switch("allowed")
        grant = _self_issue()
        start_lease(grant.id)
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).allowed
        switch(gesture)
        refused = authorized_execution_verdict(
            SECTION, TARGET, OPS, authorization=grant.id, spend=True
        )
        assert refused.reason == "auth_revoked"
        switch("allowed")
        again = authorized_execution_verdict(
            SECTION, TARGET, OPS, authorization=grant.id, spend=True
        )
        assert again.reason == after
        assert not renew_lease(grant.id)
        assert len(ledger(grant.id).steps) == 1

    def test_a_stop_of_every_machine_ends_it(self):
        switch("allowed")
        grant = _self_issue()
        start_lease(grant.id)
        control_context.set_switch("all", "off")
        control_context.invalidate_cache()
        assert verdict(grant.id).reason == "auth_revoked"

    def test_malformed_bounds_are_refused_before_anything_is_written(self):
        switch("allowed")
        with pytest.raises(ValueError):
            self_issue(
                SECTION,
                TARGET,
                {"A:SP": (5.0, 1.0)},
                steps=3,
                expires_at_ns=FAR_FUTURE_NS,
                lease_s=60.0,
            )
        with pytest.raises(ValueError):
            _self_issue(steps=0)
        assert list_grants(grants_dir()) == ((), ())


# -- overlap at approval ------------------------------------------------------


class TestOverlap:
    @pytest.mark.parametrize(
        ("fields", "overlaps"),
        [
            ({"holder": "other"}, True),
            ({"holder": "other", "channels": {"Z:SP": ChannelRange(0, 1)}}, False),
            ({}, False),  # the same holder
            ({"holder": "other", "target": "live"}, False),
            ({"holder": "other", "revoked": Revocation(at_ns=1, by="bob")}, False),
            ({"holder": "other", "expires_at_ns": 2, "issued_at_ns": 1}, False),
        ],
        ids=["shared-channel", "disjoint", "same-holder", "other-machine", "revoked", "expired"],
    )
    def test_what_overlaps(self, fields: dict[str, Any], overlaps: bool):
        existing = make_grant(**fields)
        candidate = make_grant(channels={"A:SP": ChannelRange(0, 1)})
        assert (auth_overlap(candidate, [existing]) is existing) is overlaps

    def test_an_overlapping_approval_writes_nothing(self):
        held = make_grant(holder="optimizer")
        issue_grant(held)
        candidate = make_grant(holder="bluesky")
        with pytest.raises(AuthorizationRefused) as refused:
            issue_grant(candidate)
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert not grant_path(candidate.id).exists()

    def test_an_unreadable_grant_beside_it_is_left_out_not_a_veto(self):
        broken = make_grant(holder="optimizer")
        issue_grant(broken)
        grant_path(broken.id).write_text("{")
        candidate = make_grant(holder="bluesky")
        issue_grant(candidate)
        assert grant_path(candidate.id).exists()

    def test_an_id_is_issued_once(self):
        grant = approve()
        with pytest.raises(FileExistsError):
            issue_grant(grant)

    def test_ledgers_and_locks_are_not_grants(self):
        grant = approve()
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).allowed
        assert list_grants(grants_dir()) == ((grant,), ())

    def test_simultaneous_approvals_have_exactly_one_winner(self, tmp_path: Path):
        go = tmp_path / "go"
        directory = grants_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payloads = [make_grant(holder=f"holder-{n}").to_payload() for n in range(6)]
        spawned = [
            subprocess.Popen(
                [sys.executable, "-c", ISSUER, json.dumps(payload), str(directory), str(go)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
            for payload in payloads
        ]
        go.write_text("")
        answers = sorted(process.communicate(timeout=120)[0].strip() for process in spawned)
        assert answers == ["issued"] + ["refused auth_overlap"] * 5
        grants, unreadable = list_grants(directory)
        assert (len(grants), unreadable) == (1, ())


ISSUER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from osprey_connectors import write_authorization as wa

    grant, directory, go = wa.parse_grant(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    while not go.exists():
        time.sleep(0.001)
    try:
        wa.issue_grant(grant, directory=directory)
        print("issued", flush=True)
    except wa.AuthorizationRefused as refused:
        print("refused", refused.reason, flush=True)
    """
)


# -- a container holding the tree ---------------------------------------------


class TestTree:
    """The holder is a container with the tree bound read-only and its own ledger directory."""

    @pytest.fixture
    def tree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        tree = tmp_path / "tree"
        (tree / "alice").mkdir(parents=True)
        (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("")
        payload = control_context.ControlContext(target="live", generation=1).to_payload()
        payload["switch_epoch"] = 1234
        write_payload(tree / "alice" / control_context.RECORD_FILENAME, payload)
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir()
        monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
        monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        pin_identity(monkeypatch, "optimizer")
        return tree

    SECTION = {"type": "epics", "writes": "allowed", "login_tiers": {"alice": "allowed"}}

    def issue(self, tree: Path) -> Grant:
        now = time.time_ns()
        grant = Grant(
            id=uuid.uuid4().hex,
            owner="alice",
            holder="optimizer",
            target=TARGET,
            switch_epoch=1234,
            channels=CHANNELS,
            steps=2,
            issued_at_ns=now,
            expires_at_ns=FAR_FUTURE_NS,
            lease_s=60.0,
            approver=Approver(identity="alice", surface="chip", at_ns=now),
        )
        issue_grant(grant, directory=tree / "alice" / wa.AUTHORIZATIONS_DIR_NAME)
        start_lease(grant.id)
        return grant

    def test_the_owners_grant_lets_the_holder_write_with_the_switch_off(self, tree: Path):
        grant = self.issue(tree)
        assert wa.authorizations_dir("alice") == tree / "alice" / wa.AUTHORIZATIONS_DIR_NAME
        assert posture_store.write_verdict(self.SECTION, TARGET).reason == "switch_off"
        answer = authorized_execution_verdict(
            self.SECTION, TARGET, OPS, authorization=grant.id, spend=True
        )
        assert answer.allowed
        assert len(ledger(grant.id).steps) == 1

    def test_another_identity_under_the_same_owner_gains_nothing(self, tree: Path, monkeypatch):
        grant = self.issue(tree)
        pin_identity(monkeypatch, "bluesky")
        answer = authorized_execution_verdict(self.SECTION, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "auth_holder"

    def test_an_owner_the_tiers_do_not_name_refuses_first(self, tree: Path):
        grant = self.issue(tree)
        section = {"type": "epics", "writes": "allowed", "login_tiers": {"bob": "allowed"}}
        answer = authorized_execution_verdict(section, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "owner_unknown"

    def test_a_planted_link_at_the_owners_grants_refuses(self, tree: Path, tmp_path: Path):
        grant = self.issue(tree)
        directory = tree / "alice" / wa.AUTHORIZATIONS_DIR_NAME
        moved = tmp_path / "moved"
        _relocate(directory, moved)
        directory.symlink_to(moved)
        assert verdict(grant.id, section=self.SECTION).reason == "auth_unreadable"

    def test_no_ledger_directory_under_a_tree_refuses(self, tree: Path, monkeypatch):
        grant = self.issue(tree)
        monkeypatch.delenv(wa.LEDGER_DIR_ENV_VAR)
        assert wa.ledger_dir() is None
        assert verdict(grant.id, section=self.SECTION).reason == "auth_unreadable"
        answer = spend_step(self.SECTION, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "auth_unreadable"

    def test_an_unprovisioned_tree_refuses_as_the_switch_does(self, tree: Path):
        grant = self.issue(tree)
        (tree / posture_store.CONTROL_TREE_MARKER_NAME).unlink()
        assert verdict(grant.id, section=self.SECTION).reason == "record_unavailable"
