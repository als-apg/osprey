"""A program's bounded write authorization: :mod:`osprey_connectors.write_authorization`.

The grant and the ledger are two files with two writers. These tests pin the
formats and their fail-closed readers, the verdict's terms row by row (the case
table ``_harness/fixtures/authorization_cases.json``), its composition with the
switch verdict, the step accounting under concurrent appenders in several
processes, the lease and the holder's liveness, self-issue at ``allowed``, the
approval-time overlap check and the stop's revocation.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, refusals, session_hold, testing, write_switch
from osprey_connectors import write_authorization as wa
from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.errors import AlreadyApprovedError
from osprey_connectors.model import evolve, parse_instant
from osprey_connectors.write_authorization import (
    Approver,
    AuthorizationRefusedError,
    ChannelRange,
    Grant,
    Ledger,
    LedgerStep,
    Revocation,
    WriteRequest,
    _auth_overlap,
    _issue_grant,
    _parse_grant,
    _parse_ledger,
    authorized_execution_verdict,
    list_grants,
    renew_lease,
    revoke_grants,
    self_issue,
    spend_authorized_write,
    start_lease,
)
from osprey_connectors.write_switch import FixedCause
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
CHANNELS = {"A:SP": ChannelRange(min=0.0, max=10.0), "B:SP": ChannelRange(min=-1.0, max=1.0)}

#: Far enough ahead that nothing in a test reaches it.
FAR_FUTURE = datetime.now(UTC) + timedelta(hours=12)

#: An issue time, and the longest expiry after it, for the ceiling cases.
ISSUED = datetime.now(UTC)
DAY = timedelta(seconds=wa.MAX_EXPIRES_S)

#: A fixed instant for the cases that pin a stamp's value or its bytes.
T0 = datetime(2026, 1, 1, tzinfo=UTC)

#: The smallest step an instant takes.
TICK = timedelta(microseconds=1)


def stamp(seconds: float) -> str:
    """*seconds* after :data:`T0` as a file stores it."""
    return (T0 + timedelta(seconds=seconds)).isoformat()


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


def spend(section: Any, target: str, operations: Any, *, authorization: str) -> Any:
    """One write call judged and its step spent, with no switch verdict read first."""
    return wa._spend(
        ControlSystemSection.read(section), target, operations, authorization, None, None
    )


def record() -> control_context.ControlContext:
    testing.invalidate_cache()
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
    testing.invalidate_cache()


def _grants(directory: Path) -> tuple[tuple[Grant, ...], tuple[Path, ...]]:
    """What a listing of *directory* reads: its grants, and the paths it could not read."""
    listing = list_grants(directory)
    return tuple(listed.grant for listed in listing.items), listing.unreadable


def grants_dir() -> Path:
    path = wa.authorizations_dir()
    assert path is not None
    return path


def make_grant(**fields: Any) -> Grant:
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "owner": ME,
        "holder": ME,
        "target": TARGET,
        "switch_epoch": epoch(),
        "channels": CHANNELS,
        "steps": 3,
        "issued_at": now,
        "expires_at": FAR_FUTURE,
        "lease_s": 60.0,
        "approver": Approver(by="alice", surface="chip", approved_at=now),
        "label": "run-1",
        "revoked": None,
    }
    values.update(fields)
    return Grant(**values)


def approve(**fields: Any) -> Grant:
    """Issue a grant into this login's directory and start this process's lease."""
    grant = make_grant(**fields)
    _issue_grant(grant)
    start_lease(grant.id)
    return grant


def verdict(grant_id: str, operations: Any = OPS, section: Any = SECTION) -> Any:
    return wa._verdict(ControlSystemSection.read(section), TARGET, operations, grant_id, None, None)


def ledger_path(grant_id: str) -> Path:
    directory = wa.ledger_dir()
    assert directory is not None
    return directory / f"{grant_id}.ledger.json"


def ledger(grant_id: str) -> Ledger:
    return _parse_ledger(ledger_path(grant_id).read_text())


def grant_path(grant_id: str) -> Path:
    return grants_dir() / f"{grant_id}.json"


def at(monkeypatch: pytest.MonkeyPatch, moment: datetime) -> None:
    monkeypatch.setattr(wa, "_now", lambda: moment)


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
        return (wa._EPOCH + TICK).isoformat()
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
        payload = grant.model_dump(mode="json")
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
            holder_start=wa._process_start_time(os.getpid()),
            pid_ns=wa._pid_namespace(),
            lease_expires_at=FAR_FUTURE,
        ).model_dump(mode="json")
        overrides = dict(row.get("ledger", {}))
        steps = overrides.get("steps")
        if isinstance(steps, str) and steps.startswith("@spent:"):
            count = int(steps.split(":")[1])
            overrides["steps"] = [
                {"n": n, "spent_at": stamp(n), "channels": ["A:SP"]} for n in range(1, count + 1)
            ]
        payload.update(_resolve(overrides, 0))
        grants_dir().mkdir(parents=True, exist_ok=True)
        ledger_path(grant.id).write_text(json.dumps(payload))
    if row.get("record") == "corrupt":
        path = control_context.record_path()
        assert path is not None
        path.write_text("{not json")
        testing.invalidate_cache()
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

    answer = spend(SECTION, TARGET, _operations(row), authorization=grant_id)

    assert (answer.allowed, answer.reason) == (row["expect"]["allowed"], row["expect"]["reason"])
    after = ledger_path(grant_id).read_bytes() if ledger_path(grant_id).exists() else None
    if answer.allowed:
        assert before is not None and after is not None
        assert len(_parse_ledger(after).spent) == len(_parse_ledger(before).spent) + 1
    elif not row["id"].startswith("self-issued"):
        assert after == before


# -- the formats --------------------------------------------------------------


class TestFormats:
    def test_a_grant_round_trips(self):
        grant = make_grant(revoked=Revocation(revoked_at=T0 + timedelta(seconds=5), by="bob"))
        assert _parse_grant(json.dumps(grant.model_dump(mode="json"))) == grant

    @pytest.mark.parametrize(
        ("parse", "text"),
        [
            (
                _parse_grant,
                '{"approver": {"approved_at": "2026-01-01T00:01:40+00:00", "identity": "alice", '
                '"surface": "chip"}, '
                '"channels": {"A:SP": {"max": 10.0, "min": 0.0}, "B:SP": {"max": 1.5, "min": -1.0}}, '
                '"expires_at": "2026-01-01T00:03:20+00:00", "holder": "optimizer", '
                '"id": "' + "ab" * 16 + '", '
                '"issued_at": "2026-01-01T00:01:40+00:00", "label": "run 7", "lease_s": 30.0, '
                '"owner": "alice", '
                '"revoked": {"by": "bob", "revoked_at": "2026-01-01T00:02:30+00:00"}, '
                '"schema": 1, "steps": 4, "switch_epoch": 5, "target": "va"}',
            ),
            (
                _parse_ledger,
                '{"authorization": "' + "ab" * 16 + '", "holder_pid": 42, "holder_start": "917", '
                '"lease_expires_at": "2026-01-01T00:05:00+00:00", "pid_ns": null, "schema": 1, '
                '"steps": [{"channels": ["A:SP"], "n": 1, '
                '"spent_at": "2026-01-01T00:00:10+00:00"}, '
                '{"channels": ["A:SP", "B:SP"], "n": 2, '
                '"spent_at": "2026-01-01T00:00:11.250000+00:00"}]}',
            ),
            (
                wa._parse_request,
                '{"channels": {"A:SP": {"max": 10.0, "min": 0.0}}, "completion": "apply_best", '
                '"expires_at": "2026-01-01T00:03:20+00:00", "holder": "optimizer", '
                '"id": "' + "ab" * 16 + '", '
                '"label": "", "lease_s": 45.5, "owner": "alice", '
                '"requested_at": "2026-01-01T00:01:40+00:00", '
                '"schema": 1, "steps": 41, "target": "va"}',
            ),
        ],
        ids=["grant", "ledger", "request"],
    )
    def test_a_file_reads_and_writes_back_byte_for_byte(self, tmp_path, parse, text):
        """A file as the writer lays it down reads into its record and writes the same bytes."""
        on_disk = (json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n").encode()
        path = tmp_path / "rewritten.json"
        wa._write_json(path, parse(on_disk.decode()).model_dump(mode="json"))
        assert path.read_bytes() == on_disk

    def test_a_ledger_round_trips(self):
        entry = Ledger(
            authorization="a" * 32,
            holder_pid=7,
            holder_start="100",
            pid_ns="pid:[1]",
            lease_expires_at=T0 + timedelta(seconds=9),
            spent=(LedgerStep(n=1, spent_at=T0 + timedelta(seconds=3), channels=("A:SP", "B:SP")),),
        )
        assert _parse_ledger(json.dumps(entry.model_dump(mode="json"))) == entry

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
            {"issued_at": None},
            {"issued_at": 1},
            {"issued_at": "2026-01-01T00:00:00"},
            {"issued_at": "1969-12-31T23:59:59+00:00"},
            {"issued_at_ns": 1},
            {"expires_at": wa._EPOCH.isoformat()},
            {"lease_s": 0},
            {"lease_s": -1},
            {"lease_s": "60"},
            {"approver": {"identity": "a", "surface": "phone", "approved_at": stamp(1)}},
            {"approver": {"identity": "a", "surface": "chip"}},
            {"approver": "alice"},
            {"label": None},
            {"revoked": True},
            {"revoked": {"revoked_at": stamp(1)}},
            {"revoked": {"revoked_at": stamp(1), "by": ""}},
            {"revoked": {"revoked_at": "1969-12-31T23:59:59+00:00", "by": "bob"}},
            {"revoked": {"at_ns": 1, "by": "bob"}},
            {"extra": 1},
        ],
        ids=repr,
    )
    def test_a_malformed_grant_is_refused(self, change: dict[str, Any]):
        payload = make_grant().model_dump(mode="json")
        payload.update(change)
        with pytest.raises(ValueError):
            _parse_grant(json.dumps(payload))

    @pytest.mark.parametrize("missing", sorted(["schema", *Grant.model_fields]))
    def test_a_grant_missing_any_field_is_refused(self, missing: str):
        payload = make_grant().model_dump(mode="json")
        del payload[missing]
        with pytest.raises(ValueError):
            _parse_grant(json.dumps(payload))

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
            _parse_grant(text)

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_bound_is_refused(self, constant: str):
        text = json.dumps(make_grant().model_dump(mode="json")).replace(
            '"max": 10.0', f'"max": {constant}'
        )
        assert constant in text
        with pytest.raises(ValueError):
            _parse_grant(text)

    @pytest.mark.parametrize(
        ("change", "refused"),
        [
            ({}, False),
            ({"steps": wa.MAX_STEPS + 1}, True),
            ({"expires_at": ISSUED + DAY + TICK}, True),
            ({"lease_s": wa._MAX_LEASE_S + 0.5}, True),
        ],
        ids=["at-ceilings", "steps", "expires_at", "lease_s"],
    )
    def test_a_grant_past_any_ceiling_is_read_but_never_written(
        self, change: dict[str, Any], refused: bool
    ):
        """The ceilings are the writer's: a file on disk past one is honoured as before."""
        at_ceilings = make_grant(
            steps=wa.MAX_STEPS,
            issued_at=ISSUED,
            expires_at=ISSUED + DAY,
            lease_s=wa._MAX_LEASE_S,
        )
        grant = evolve(at_ceilings, **change)
        assert _parse_grant(json.dumps(grant.model_dump(mode="json"))) == grant
        if refused:
            with pytest.raises(ValueError, match="ceiling|more than"):
                _issue_grant(grant)
            assert not grant_path(grant.id).exists()
        else:
            _issue_grant(grant)
            assert grant_path(grant.id).exists()

    @pytest.mark.parametrize(
        "change",
        [
            {"schema": 2},
            {"authorization": "x"},
            {"holder_pid": 0},
            {"holder_pid": True},
            {"holder_start": 5},
            {"pid_ns": ["pid:[1]"]},
            {"lease_expires_at": "1969-12-31T23:59:59+00:00"},
            {"lease_expires_at": None},
            {"lease_expires_at": 1},
            {"lease_expires_at": "2026-01-01T00:00:00"},
            {"steps": {}},
            {"steps": [{"n": 1, "spent_at": stamp(1), "channels": []}]},
            {"steps": [{"n": 1, "spent_at": stamp(1), "channels": [""]}]},
            {"steps": [{"n": 1, "spent_at": stamp(1)}]},
            {"steps": [{"n": 1, "spent_at": stamp(1), "channels": ["A"], "extra": 1}]},
            {"steps": [{"n": 1, "spent_at": "1969-12-31T23:59:59+00:00", "channels": ["A"]}]},
            {"steps": [{"n": 1, "at_ns": 1, "channels": ["A"]}]},
            {
                "steps": [
                    {"n": 1, "spent_at": stamp(1), "channels": ["A"]},
                    {"n": 1, "spent_at": stamp(2), "channels": ["A"]},
                ]
            },
            {"extra": 1},
        ],
        ids=repr,
    )
    def test_a_malformed_ledger_is_refused(self, change: dict[str, Any]):
        payload = Ledger(
            authorization="a" * 32,
            holder_pid=1,
            holder_start=None,
            pid_ns=None,
            lease_expires_at=T0,
        ).model_dump(mode="json")
        payload.update(change)
        with pytest.raises(ValueError):
            _parse_ledger(json.dumps(payload))

    @pytest.mark.parametrize(
        "build",
        [
            lambda: ChannelRange(min=2.0, max=1.0),
            lambda: ChannelRange(min=math.nan, max=1.0),
            lambda: ChannelRange.of(0.0, math.inf),
            lambda: ChannelRange.of(-math.inf, 0),
            lambda: make_grant(id="ABC"),
            lambda: make_grant(owner=""),
            lambda: make_grant(owner=" alice"),
            lambda: make_grant(holder="bob "),
            lambda: make_grant(switch_epoch=-1),
            lambda: make_grant(channels={}),
            lambda: make_grant(channels={" ": ChannelRange(min=0.0, max=1.0)}),
            lambda: make_grant(steps=0),
            lambda: make_grant(issued_at=wa._EPOCH - TICK),
            lambda: make_grant(issued_at=T0, expires_at=T0),
            lambda: make_grant(lease_s=0),
            lambda: make_grant(lease_s=math.inf),
            lambda: make_grant(lease_s=True),
            lambda: Approver(by=" alice", surface="chip", approved_at=T0),
            lambda: Approver(by="", surface="chip", approved_at=T0),
            lambda: Revocation(revoked_at=T0, by="bob\n"),
            lambda: Revocation(revoked_at=wa._EPOCH - TICK, by="bob"),
            lambda: LedgerStep(n=0, spent_at=T0, channels=("A:SP",)),
            lambda: LedgerStep(n=1, spent_at=T0, channels=()),
            lambda: LedgerStep(n=1, spent_at=T0, channels=("",)),
            lambda: Ledger(
                authorization="a" * 32,
                holder_pid=0,
                holder_start=None,
                pid_ns=None,
                lease_expires_at=T0,
            ),
            lambda: Ledger(
                authorization="a" * 32,
                holder_pid=None,
                holder_start=None,
                pid_ns=None,
                lease_expires_at=T0,
                spent=(LedgerStep(n=2, spent_at=T0, channels=("A:SP",)),),
            ),
            lambda: WriteRequest(
                id="a" * 32,
                owner=ME,
                holder=ME,
                target=TARGET,
                channels=CHANNELS,
                steps=1,
                requested_at=T0,
                expires_at=T0,
                lease_s=60.0,
            ),
        ],
    )
    def test_a_record_built_here_meets_the_readers_rules(self, build: Any):
        """Every rule a file is read under refuses a record when it is built, in-process too."""
        with pytest.raises(ValueError):
            build()

    def test_a_number_built_here_is_held_as_given_and_read_as_a_float(self):
        """Only how a number is held depends on where it came from, never whether it is valid."""
        grant = make_grant(lease_s=60, channels={"A:SP": ChannelRange(min=0, max=1)})
        written = grant.model_dump(mode="json")
        assert (written["lease_s"], written["channels"]["A:SP"]) == (60, {"min": 0, "max": 1})
        assert type(written["lease_s"]) is int
        read = _parse_grant(json.dumps(written))
        assert type(read.lease_s) is float and type(read.channels["A:SP"].min) is float

    def test_a_range_of_two_bounds_is_held_as_floats(self):
        assert ChannelRange.of(0, 1).model_dump(mode="json") == {"min": 0.0, "max": 1.0}

    def test_a_ledger_and_an_approver_keep_their_file_keys(self):
        entry = Ledger(
            authorization="a" * 32,
            holder_pid=None,
            holder_start=None,
            pid_ns=None,
            lease_expires_at=T0,
            spent=(LedgerStep(n=1, spent_at=T0, channels=("A:SP",)),),
        )
        assert "steps" in entry.model_dump(mode="json") and "spent" not in entry.model_dump()
        approver = Approver(by="alice", surface="chip", approved_at=T0)
        assert set(approver.model_dump()) == {"identity", "surface", "approved_at"}
        assert Approver.model_validate(approver.model_dump(mode="json")) == approver

    def test_a_channel_range_holds_finite_reals_only(self):
        bound = ChannelRange(min=0.0, max=1.0)
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
        assert (
            write_switch.write_verdict(ControlSystemSection.read(SECTION), TARGET).reason
            == "switch_off"
        )


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
        payload = grant.model_dump(mode="json")
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
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"
        assert len(_parse_ledger(elsewhere.read_text()).spent) == 0

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
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"

    def test_a_ledger_directory_that_is_a_file(self, tmp_path: Path, monkeypatch):
        grant = approve()
        planted = tmp_path / "file"
        planted.write_text("")
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(planted))
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"

    @pytest.mark.parametrize("named", ["", "x", "A" * 32, "../" + "a" * 29, "a" * 33, None, 7])
    def test_an_id_that_names_no_file(self, named: Any):
        assert verdict(named).reason == "auth_unreadable"  # type: ignore[arg-type]
        answer = spend(SECTION, TARGET, OPS, authorization=named)  # type: ignore[arg-type]
        assert answer.reason == "auth_unreadable"

    def test_an_approval_planted_in_the_holders_own_directory(self, tmp_path: Path, monkeypatch):
        """The holder can write its ledger directory, so only a self-issued grant counts there."""
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        grant = make_grant()
        _issue_grant(grant, directory=ledgers)
        with pytest.raises(AuthorizationRefusedError) as refused:
            start_lease(grant.id)
        assert refused.value.reason == "auth_unreadable"
        assert verdict(grant.id).reason == "auth_unreadable"

    def test_a_verdict_that_raises_refuses(self, monkeypatch):
        grant = approve()

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(wa, "_judge", boom)
        assert verdict(grant.id).reason == "auth_unreadable"
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"

    def test_a_ledger_lock_that_is_a_link_refuses(self, tmp_path: Path):
        grant = approve()
        lock = ledger_path(grant.id).with_name(ledger_path(grant.id).name + ".lock")
        lock.unlink(missing_ok=True)
        lock.symlink_to(tmp_path / "elsewhere")
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"
        assert ledger(grant.id).spent == ()


# -- the composition ----------------------------------------------------------


class TestComposition:
    @pytest.mark.parametrize(
        "composition",
        [authorized_execution_verdict, spend_authorized_write, wa.authorized_write_verdict],
    )
    def test_every_composition_takes_the_authorization_it_judges(self, composition):
        # A process naming none asks the switch verdict itself; no composition
        # defaults the name, and none reads it from the environment.
        parameter = inspect.signature(composition).parameters["authorization"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    def test_the_switch_off_is_lifted_for_the_listed_channels_only(self):
        grant = approve()
        assert authorized_execution_verdict(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        ).allowed
        unlisted = authorized_execution_verdict(
            ControlSystemSection.read(SECTION), TARGET, [("C:SP", 1.0)], authorization=grant.id
        )
        assert unlisted.reason == "auth_channel"

    @pytest.mark.parametrize("position", ["asked", "allowed"])
    def test_a_switch_turned_on_mid_run_does_not_lift_the_bounds(self, position: str):
        grant = approve()
        switch(position)
        assert write_switch.write_verdict(ControlSystemSection.read(SECTION), TARGET).allowed
        for operations, reason in (
            ([("C:SP", 1.0)], "auth_channel"),
            ([("A:SP", 11.0)], "auth_range"),
        ):
            answer = authorized_execution_verdict(
                ControlSystemSection.read(SECTION), TARGET, operations, authorization=grant.id
            )
            assert answer.reason == reason

    def test_an_authorization_lifts_no_tier(self):
        grant = approve()
        answer = authorized_execution_verdict(
            ControlSystemSection.read(NEVER), TARGET, OPS, authorization=grant.id
        )
        assert (answer.reason, answer.authorization) == ("tier_never", grant.id)

    def test_an_authorization_lifts_no_read_only_run(self, monkeypatch):
        grant = approve()
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        answer = authorized_execution_verdict(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        )
        assert answer.reason == "readonly_run"

    def test_an_authorization_lifts_no_hold(self, tmp_path: Path, monkeypatch):
        grant = approve()
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(tmp_path / "holds"))
        monkeypatch.setenv("OSPREY_SESSION_ID", "s1")
        answer = authorized_execution_verdict(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        )
        assert answer.reason == "session_held"

    def test_an_authorization_lifts_no_pin_with_the_switch_on_or_off(self, monkeypatch):
        grant = approve()
        monkeypatch.setenv(write_switch.LAUNCH_PIN_ENV_VAR, "va=sandbox")
        assert (
            authorized_execution_verdict(
                ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
            ).reason
            == "launch_pinned"
        )
        switch("allowed")
        assert (
            authorized_execution_verdict(
                ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
            ).reason
            == "launch_pinned"
        )

    def test_spending_through_the_composition(self):
        grant = approve()
        answer = spend_authorized_write(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        )
        assert answer.allowed and answer.authorization == grant.id
        assert len(ledger(grant.id).spent) == 1
        assert authorized_execution_verdict(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        ).allowed
        assert len(ledger(grant.id).spent) == 1


# -- steps --------------------------------------------------------------------


class TestSteps:
    def test_a_batch_is_one_step(self):
        grant = approve()
        assert spend(SECTION, TARGET, [("A:SP", 1.0), ("B:SP", 0.0)], authorization=grant.id)
        (entry,) = ledger(grant.id).spent
        assert (entry.n, entry.channels) == (1, ("A:SP", "B:SP"))

    def test_the_budget_ends_the_run_and_nothing_renews_it(self):
        grant = approve(steps=2)
        answers = [spend(SECTION, TARGET, OPS, authorization=grant.id) for _ in range(4)]
        assert [a.allowed for a in answers] == [True, True, False, False]
        assert answers[-1].reason == "auth_exhausted"
        assert [entry.n for entry in ledger(grant.id).spent] == [1, 2]
        assert [entry.n for entry in renew_lease(grant.id).spent] == [1, 2]
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_exhausted"

    def test_a_one_shot_batch_is_judged_and_recorded_whole(self):
        grant = approve()
        batch = iter([("A:SP", 1.0), ("B:SP", 0.0)])
        assert spend(SECTION, TARGET, batch, authorization=grant.id).allowed
        assert ledger(grant.id).spent[0].channels == ("A:SP", "B:SP")

    def test_a_refused_write_spends_nothing(self):
        grant = approve()
        before = ledger_path(grant.id).read_bytes()
        assert spend(SECTION, TARGET, [("A:SP", 99.0)], authorization=grant.id).reason == (
            "auth_range"
        )
        assert ledger_path(grant.id).read_bytes() == before

    def test_the_read_only_verdict_spends_nothing(self):
        grant = approve()
        for _ in range(5):
            assert verdict(grant.id).allowed
        assert ledger(grant.id).spent == ()

    def test_a_step_that_cannot_be_recorded_is_refused(self, monkeypatch):
        grant = approve()
        before = ledger_path(grant.id).read_bytes()

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(wa, "atomic_replace", disk_full)
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_unreadable"
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
            answer = spend(SECTION, TARGET, OPS, authorization=grant.id)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        assert answer.reason == "auth_unreadable"
        assert ledger(grant.id).spent == ()


SPENDER = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path
    from osprey_connectors import write_authorization as wa
    from osprey_connectors.config_sections import ControlSystemSection

    section, grant_id, tries, go = json.loads(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    while not go.exists():
        time.sleep(0.001)
    spent = 0
    for _ in range(tries):
        answer = wa._spend(ControlSystemSection.read(section), "va", [("A:SP", 1.0), ("B:SP", 0.5)], grant_id, None, None)
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
        steps = ledger(grant.id).spent
        assert [entry.n for entry in steps] == list(range(1, budget + 1))
        assert all(entry.channels == ("A:SP", "B:SP") for entry in steps)

    def test_threads_spend_exactly_the_budget(self):
        from concurrent.futures import ThreadPoolExecutor

        grant = approve(steps=10)
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(
                pool.map(lambda _: spend(SECTION, TARGET, OPS, authorization=grant.id), range(24))
            )
        assert sum(answer.allowed for answer in answers) == 10
        assert {a.reason for a in answers if not a.allowed} == {"auth_exhausted"}
        assert len(ledger(grant.id).spent) == 10


# -- the lease ----------------------------------------------------------------


class TestLease:
    def test_start_records_this_process_as_the_holder(self):
        grant = approve()
        entry = ledger(grant.id)
        assert entry.holder_pid == os.getpid()
        assert entry.pid_ns == wa._pid_namespace()
        assert entry.holder_start == wa._process_start_time(os.getpid())
        assert entry.spent == ()

    def test_a_lease_is_started_once(self):
        grant = approve()
        with pytest.raises(AuthorizationRefusedError) as refused:
            start_lease(grant.id)
        assert refused.value.reason == "auth_lease_lapsed"

    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            ({"holder": "someone-else"}, "auth_holder"),
            ({"revoked": Revocation(revoked_at=T0, by="bob")}, "auth_revoked"),
            ({"expires_at": T0 + timedelta(seconds=2), "issued_at": T0}, "auth_expired"),
        ],
    )
    def test_a_lease_is_not_started_on_a_grant_out_of_force(self, fields, reason):
        grant = make_grant(**fields)
        _issue_grant(grant)
        with pytest.raises(AuthorizationRefusedError) as refused:
            start_lease(grant.id)
        assert refused.value.reason == reason
        assert not ledger_path(grant.id).exists()

    def test_renewal_outlives_the_first_lease(self, monkeypatch):
        now = datetime.now(UTC)
        at(monkeypatch, now)
        grant = approve(lease_s=10.0)
        at(monkeypatch, now + timedelta(seconds=9))
        assert renew_lease(grant.id).lease_expires_at == now + timedelta(seconds=19)
        at(monkeypatch, now + timedelta(seconds=15))
        assert verdict(grant.id).allowed
        assert ledger(grant.id).lease_expires_at == now + timedelta(seconds=19)

    def test_a_lapsed_lease_refuses_and_does_not_come_back(self, monkeypatch):
        now = datetime.now(UTC)
        at(monkeypatch, now)
        grant = approve(lease_s=10.0)
        at(monkeypatch, now + timedelta(seconds=10))
        assert verdict(grant.id).reason == "auth_lease_lapsed"
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(grant.id)
        assert unrenewed.value.reason == "auth_lease_lapsed"
        assert unrenewed.value.cause is None
        at(monkeypatch, now + TICK)
        # Even a clock that steps back finds the lease unmoved by the failed renewal.
        assert ledger(grant.id).lease_expires_at == now + timedelta(seconds=10)

    def test_only_the_holder_process_renews(self):
        grant = approve()
        payload = json.loads(ledger_path(grant.id).read_text())
        payload["holder_pid"] = ABSENT_PID
        payload["pid_ns"] = "pid:[0]"
        ledger_path(grant.id).write_text(json.dumps(payload))
        before = ledger_path(grant.id).read_bytes()
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(grant.id)
        assert unrenewed.value.reason == "auth_lease_lapsed"
        assert unrenewed.value.cause == FixedCause(kind="lease_held_elsewhere")
        assert ledger_path(grant.id).read_bytes() == before

    def test_a_revoked_grant_is_not_renewed(self):
        grant = approve()
        revoke_grants(by="bob")
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(grant.id)
        assert unrenewed.value.reason == "auth_revoked"
        assert unrenewed.value.cause is None

    def test_a_missing_ledger_is_not_renewed(self):
        grant = make_grant()
        _issue_grant(grant)
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(grant.id)
        assert unrenewed.value.reason == "auth_lease_lapsed"
        assert unrenewed.value.cause == FixedCause(kind="lease_never_started")
        assert not ledger_path(grant.id).exists()

    def test_an_authorization_with_no_grant_is_not_renewed_and_its_lease_lapses(self):
        approve()
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(uuid.uuid4().hex)
        assert unrenewed.value.reason == "auth_lease_lapsed"
        assert unrenewed.value.cause == FixedCause(kind="no_such_authorization")

    def test_a_renewal_that_fails_on_the_way_lapses_the_lease(self, monkeypatch):
        grant = approve()
        before = ledger_path(grant.id).read_bytes()

        def unwritable(*_args, **_kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(wa, "_write_json", unwritable)
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(grant.id)
        assert unrenewed.value.reason == "auth_lease_lapsed"
        assert unrenewed.value.cause == FixedCause(kind="renewal_failed")
        assert ledger_path(grant.id).read_bytes() == before


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
        grant = make_grant(lease_s=600.0)
        _issue_grant(grant)
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
        assert ledger(grant.id).lease_expires_at > datetime.now(UTC)
        assert verdict(grant.id).reason == "auth_lease_lapsed"


class TestHolderLiveness:
    """What the lease concludes about its holder, with the kernel's answer forged."""

    def ledger(self, **fields: Any) -> Ledger:
        values: dict[str, Any] = {
            "authorization": "a" * 32,
            "holder_pid": os.getpid(),
            "holder_start": "100",
            "pid_ns": wa._pid_namespace(),
            "lease_expires_at": FAR_FUTURE,
        }
        values.update(fields)
        return Ledger(**values)

    def forge_kernel(self, monkeypatch, answer: tuple[str, str] | None) -> None:
        monkeypatch.setattr(wa, "_kernel_stat", lambda pid: answer)

    def test_the_same_process(self, monkeypatch):
        self.forge_kernel(monkeypatch, ("R", "100"))
        assert wa._lease_live(self.ledger(), datetime.now(UTC))

    def test_a_reused_pid(self, monkeypatch):
        self.forge_kernel(monkeypatch, ("R", "999"))
        assert not wa._lease_live(self.ledger(), datetime.now(UTC))

    def test_a_zombie_holder(self, monkeypatch):
        self.forge_kernel(monkeypatch, ("Z", "100"))
        assert not wa._lease_live(self.ledger(), datetime.now(UTC))

    def test_no_kernel_view_falls_back_on_the_pid_and_the_lease(self, monkeypatch):
        self.forge_kernel(monkeypatch, None)
        now = datetime.now(UTC)
        assert wa._lease_live(self.ledger(), now)
        assert not wa._lease_live(self.ledger(holder_pid=ABSENT_PID), now)
        assert not wa._lease_live(self.ledger(lease_expires_at=now), now)

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
        assert wa._process_start_time(123) == "777"


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
        monkeypatch.setattr(wa, "_pid_namespace", lambda: reader)
        entry = Ledger(
            authorization="a" * 32,
            holder_pid=ABSENT_PID,
            holder_start=None,
            pid_ns=holder,
            lease_expires_at=FAR_FUTURE,
        )
        assert wa._lease_live(entry, datetime.now(UTC)) is live

    def test_another_namespace_is_judged_by_the_lease_time_alone(self, monkeypatch):
        grant = approve()
        payload = json.loads(ledger_path(grant.id).read_text())
        payload.update(holder_pid=ABSENT_PID, pid_ns=self.OTHER)
        ledger_path(grant.id).write_text(json.dumps(payload))
        assert verdict(grant.id).allowed
        at(monkeypatch, parse_instant(payload["lease_expires_at"]))
        assert verdict(grant.id).reason == "auth_lease_lapsed"

    def test_the_namespace_is_read_from_the_link(self, monkeypatch):
        def link(path: str) -> str:
            assert path == "/proc/self/ns/pid"
            return "pid:[42]"

        monkeypatch.setattr(os, "readlink", link)
        assert wa._pid_namespace() == "pid:[42]"

    def test_no_link_is_no_namespace(self, monkeypatch):
        def no_link(path: str) -> str:
            raise FileNotFoundError(path)

        monkeypatch.setattr(os, "readlink", no_link)
        assert wa._pid_namespace() is None


# -- stops --------------------------------------------------------------------


class TestStops:
    def test_a_stop_of_every_machine_ends_it_by_the_epoch_alone(self):
        grant = approve()
        control_context.set_switch("all", "off")
        testing.invalidate_cache()
        assert verdict(grant.id).reason == "auth_revoked"
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).reason == "auth_revoked"
        with pytest.raises(AuthorizationRefusedError) as unrenewed:
            renew_lease(grant.id)
        assert unrenewed.value.reason == "auth_revoked"
        assert unrenewed.value.cause is None

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
        on_live = approve(target="live", channels={"L:SP": ChannelRange(min=0, max=1)})
        assert revoke_grants(by="alice", targets=["live"]) == (on_live.id,)
        assert verdict(on_va.id).allowed
        assert revoke_grants(by="bob") == (on_va.id,)
        assert revoke_grants(by="carol") == ()
        assert _parse_grant(grant_path(on_va.id).read_text()).revoked == Revocation(
            revoked_at=_parse_grant(grant_path(on_va.id).read_text()).revoked.revoked_at,  # type: ignore[union-attr]
            by="bob",
        )

    def test_a_revocation_by_a_name_that_is_not_a_login_marks_nothing(self):
        """Refused when the mark is built: before, it wrote a grant no reader could read."""
        grant = approve()
        before = grant_path(grant.id).read_bytes()
        with pytest.raises(ValueError):
            revoke_grants(by=" alice")
        assert grant_path(grant.id).read_bytes() == before
        assert verdict(grant.id).allowed

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
        testing.invalidate_cache()
        assert verdict(grant.id).reason == "auth_revoked"


# -- self-issue at allowed ----------------------------------------------------


def _self_issue(**fields: Any) -> Grant:
    values: dict[str, Any] = {
        "steps": 3,
        "expires_at": FAR_FUTURE,
        "lease_s": 60.0,
        "label": "run-self",
    }
    values.update(fields)
    return self_issue(
        ControlSystemSection.read(SECTION), TARGET, {"A:SP": ChannelRange.of(0.0, 10.0)}, **values
    )


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
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).allowed
        assert verdict(grant.id, [("A:SP", 11.0)]).reason == "auth_range"

    @pytest.mark.parametrize(
        ("position", "reason"), [("off", "switch_off"), ("asked", "auth_revoked")]
    )
    def test_below_allowed_nothing_is_issued(self, position, reason):
        if position != "off":
            switch(position)
        with pytest.raises(AuthorizationRefusedError) as refused:
            _self_issue()
        assert refused.value.reason == reason
        assert _grants(grants_dir()) == ((), ())

    def test_a_pinned_launch_issues_nothing(self, monkeypatch):
        switch("allowed")
        monkeypatch.setenv(write_switch.LAUNCH_PIN_ENV_VAR, "*=sandbox")
        with pytest.raises(AuthorizationRefusedError) as refused:
            _self_issue()
        assert refused.value.reason == "launch_pinned"

    def test_a_stop_during_the_issue_removes_the_grant(self, monkeypatch):
        switch("allowed")
        real = wa._issue_grant

        def stop_after(grant: Grant, *, directory: Path | None = None, peers: Any = ()) -> Path:
            path = real(grant, directory=directory, peers=peers)
            control_context.set_switch("all", "off")
            testing.invalidate_cache()
            return path

        monkeypatch.setattr(wa, "_issue_grant", stop_after)
        with pytest.raises(AuthorizationRefusedError):
            _self_issue()
        assert _grants(grants_dir()) == ((), ())

    # On a host the holder's directory is the owner's, so a stop marks the
    # self-issued grant revoked as well; leaving for ``asked`` lapses its lease.
    @pytest.mark.parametrize(
        ("gesture", "after"), [("asked", "auth_lease_lapsed"), ("off", "auth_revoked")]
    )
    def test_leaving_allowed_ends_it_for_good(self, gesture: str, after: str):
        switch("allowed")
        grant = _self_issue()
        start_lease(grant.id)
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).allowed
        switch(gesture)
        refused = spend_authorized_write(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        )
        assert refused.reason == "auth_revoked"
        switch("allowed")
        again = spend_authorized_write(
            ControlSystemSection.read(SECTION), TARGET, OPS, authorization=grant.id
        )
        assert again.reason == after
        with pytest.raises(AuthorizationRefusedError):
            renew_lease(grant.id)
        assert len(ledger(grant.id).spent) == 1

    def test_a_stop_of_every_machine_ends_it(self):
        switch("allowed")
        grant = _self_issue()
        start_lease(grant.id)
        control_context.set_switch("all", "off")
        testing.invalidate_cache()
        assert verdict(grant.id).reason == "auth_revoked"

    def test_no_ledger_directory_refuses_before_a_malformed_grant(self, tmp_path, monkeypatch):
        """The ledger directory is judged before the grant is built, as before."""
        switch("allowed")
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(tmp_path / "absent"))
        with pytest.raises(AuthorizationRefusedError) as refused:
            _self_issue(steps=0)
        assert refused.value.reason == "auth_unreadable"

    def test_a_range_that_is_not_a_channel_range_is_refused(self):
        switch("allowed")
        with pytest.raises(ValueError):
            self_issue(
                ControlSystemSection.read(SECTION),
                TARGET,
                {"A:SP": (0.0, 1.0)},
                steps=3,
                expires_at=FAR_FUTURE,
                lease_s=60.0,
            )
        assert _grants(grants_dir()) == ((), ())

    def test_malformed_bounds_are_refused_before_anything_is_written(self):
        switch("allowed")
        with pytest.raises(ValueError):
            self_issue(
                ControlSystemSection.read(SECTION),
                TARGET,
                {"A:SP": ChannelRange.of(5.0, 1.0)},
                steps=3,
                expires_at=FAR_FUTURE,
                lease_s=60.0,
            )
        with pytest.raises(ValueError):
            _self_issue(steps=0)
        assert _grants(grants_dir()) == ((), ())


# -- overlap at approval ------------------------------------------------------


class TestOverlap:
    @pytest.mark.parametrize(
        ("fields", "overlaps"),
        [
            ({"holder": "other"}, True),
            ({"holder": "other", "channels": {"Z:SP": ChannelRange(min=0, max=1)}}, False),
            ({}, False),  # the same holder
            ({"holder": "other", "target": "live"}, False),
            ({"holder": "other", "revoked": Revocation(revoked_at=T0, by="bob")}, False),
            (
                {"holder": "other", "expires_at": T0 + timedelta(seconds=2), "issued_at": T0},
                False,
            ),
        ],
        ids=["shared-channel", "disjoint", "same-holder", "other-machine", "revoked", "expired"],
    )
    def test_what_overlaps(self, fields: dict[str, Any], overlaps: bool):
        existing = make_grant(**fields)
        candidate = make_grant(channels={"A:SP": ChannelRange(min=0, max=1)})
        assert (_auth_overlap(candidate, [existing]) is existing) is overlaps

    def test_an_overlapping_approval_writes_nothing(self):
        held = make_grant(holder="optimizer")
        _issue_grant(held)
        candidate = make_grant(holder="bluesky")
        with pytest.raises(AuthorizationRefusedError) as refused:
            _issue_grant(candidate)
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert not grant_path(candidate.id).exists()

    @pytest.mark.parametrize(
        ("leases", "blocks"),
        [
            ((), True),
            (("live",), True),
            (("in-grace",), True),
            (("at-grace-end",), True),
            (("lapsed",), False),
            (("unreadable",), True),
            (("lapsed", "live"), True),
        ],
        ids=[
            "no-ledger-yet",
            "lease-live",
            "within-skew-grace",
            "exactly-at-grace-end",
            "lease-ran-out",
            "unreadable",
            "one-lapsed-one-live",
        ],
    )
    def test_a_grant_whose_lease_lapsed_does_not_block(
        self, tmp_path: Path, leases: tuple[str, ...], blocks: bool
    ):
        """A person-approved grant of a run that finished or crashed stops blocking at its lease.

        Only when every ledger found agrees: one lapsed in a directory searched
        first does not free a grant whose other ledger is live.
        """
        now = datetime.now(UTC)
        existing = make_grant(holder="optimizer")
        candidate = make_grant(holder="bluesky", channels={"A:SP": ChannelRange(min=0, max=1)})
        dirs = [tmp_path / "absent"]
        for n, lease in enumerate(leases):
            directory = tmp_path / f"ledgers-{n}"
            directory.mkdir()
            dirs.append(directory)
            path = directory / f"{existing.id}.ledger.json"
            if lease == "unreadable":
                path.write_text("{")
                continue
            entry = Ledger(
                authorization=existing.id,
                holder_pid=None,
                holder_start=None,
                pid_ns=None,
                lease_expires_at={
                    "live": now + timedelta(seconds=1),
                    "in-grace": now - TICK,
                    # Exactly the grace past its end still blocks; a tick more frees it.
                    "at-grace-end": now - wa._LEASE_SKEW_GRACE,
                    "lapsed": now - wa._LEASE_SKEW_GRACE - TICK,
                }[lease],
            )
            path.write_text(json.dumps(entry.model_dump(mode="json")))
        held = _auth_overlap(candidate, [existing], now=now, ledger_dirs=dirs)
        assert (held is existing) is blocks

    def test_a_dead_holders_grant_blocks_until_its_lease_time_passes(self):
        """The clock alone decides, even where the holder's PID could be asked."""
        held = make_grant(holder="optimizer")
        _issue_grant(held)
        leased = Ledger(
            authorization=held.id,
            holder_pid=ABSENT_PID,
            holder_start=None,
            pid_ns=wa._pid_namespace(),
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )
        ledger_path(held.id).write_text(json.dumps(leased.model_dump(mode="json")))
        candidate = make_grant(holder="bluesky")
        with pytest.raises(AuthorizationRefusedError):
            _issue_grant(candidate)
        lapsed = evolve(leased, lease_expires_at=datetime.now(UTC) - wa._LEASE_SKEW_GRACE - TICK)
        ledger_path(held.id).write_text(json.dumps(lapsed.model_dump(mode="json")))
        _issue_grant(candidate)
        assert grant_path(candidate.id).exists()

    def test_a_grant_on_disk_past_a_ceiling_still_blocks(self):
        """Written by an older version, before the ceilings: honoured, so it still blocks."""
        held = make_grant(holder="optimizer", steps=wa.MAX_STEPS + 1)
        grants_dir().mkdir(parents=True, exist_ok=True)
        grant_path(held.id).write_text(json.dumps(held.model_dump(mode="json")))
        candidate = make_grant(holder="bluesky")
        with pytest.raises(AuthorizationRefusedError) as refused:
            _issue_grant(candidate)
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert not grant_path(candidate.id).exists()

    @pytest.mark.parametrize("where", ["peer", "view"])
    def test_a_lapsed_ledger_seen_only_beside_the_approver_frees_the_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
    ):
        """The run's ledger is in a peer or the view directory, not the grants directory."""
        held = make_grant(holder="optimizer")
        _issue_grant(held)
        assert not ledger_path(held.id).exists()
        elsewhere = tmp_path / "holders-ledgers"
        elsewhere.mkdir()
        leased = Ledger(
            authorization=held.id,
            holder_pid=None,
            holder_start=None,
            pid_ns=None,
            lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        )
        (elsewhere / f"{held.id}.ledger.json").write_text(
            json.dumps(leased.model_dump(mode="json"))
        )
        if where == "view":
            monkeypatch.setenv(wa.LEDGER_VIEW_DIR_ENV_VAR, str(elsewhere))

        def approve_candidate() -> str:
            if where == "peer":
                candidate = make_grant(holder="bluesky")
                _issue_grant(candidate, peers=[elsewhere])
                return candidate.id
            request = wa.WriteRequest(
                id=uuid.uuid4().hex,
                owner=ME,
                holder="bluesky",
                target=TARGET,
                channels={"A:SP": ChannelRange(min=0.0, max=1.0)},
                steps=3,
                requested_at=datetime.now(UTC),
                expires_at=FAR_FUTURE,
                lease_s=60.0,
                label="",
                completion="none",
            )
            return wa.approve_request(request, surface="cli").id

        with pytest.raises(AuthorizationRefusedError) as refused:
            approve_candidate()
        assert refused.value.reason == "auth_overlap"
        lapsed = evolve(leased, lease_expires_at=datetime.now(UTC) - wa._LEASE_SKEW_GRACE - TICK)
        (elsewhere / f"{held.id}.ledger.json").write_text(
            json.dumps(lapsed.model_dump(mode="json"))
        )
        assert grant_path(approve_candidate()).exists()

    def test_an_unreadable_grant_beside_it_is_left_out_not_a_veto(self):
        broken = make_grant(holder="optimizer")
        _issue_grant(broken)
        grant_path(broken.id).write_text("{")
        candidate = make_grant(holder="bluesky")
        _issue_grant(candidate)
        assert grant_path(candidate.id).exists()

    def test_an_id_is_issued_once(self):
        grant = approve()
        with pytest.raises(AlreadyApprovedError) as caught:
            _issue_grant(grant)
        assert caught.value.authorization == grant.id
        assert isinstance(caught.value, FileExistsError)

    def test_ledgers_and_locks_are_not_grants(self):
        grant = approve()
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).allowed
        assert _grants(grants_dir()) == ((grant,), ())

    def test_simultaneous_approvals_have_exactly_one_winner(self, tmp_path: Path):
        go = tmp_path / "go"
        directory = grants_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payloads = [make_grant(holder=f"holder-{n}").model_dump(mode="json") for n in range(6)]
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
        grants, unreadable = _grants(directory)
        assert (len(grants), unreadable) == (1, ())


ISSUER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from osprey_connectors import write_authorization as wa

    grant, directory, go = wa._parse_grant(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    while not go.exists():
        time.sleep(0.001)
    try:
        wa._issue_grant(grant, directory=directory)
        print("issued", flush=True)
    except wa.AuthorizationRefusedError as refused:
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
        (tree / write_switch.CONTROL_TREE_MARKER_NAME).write_text("")
        payload = control_context.ControlContext(target="live", generation=1).model_dump(
            mode="json"
        )
        payload["switch_epoch"] = 1234
        write_payload(tree / "alice" / control_context.RECORD_FILENAME, payload)
        ledgers = tmp_path / "ledgers"
        ledgers.mkdir()
        monkeypatch.setenv(write_switch.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
        monkeypatch.setenv(write_switch.CONTROL_OWNER_ENV_VAR, "alice")
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        pin_identity(monkeypatch, "optimizer")
        return tree

    SECTION = {"type": "epics", "writes": "allowed", "login_tiers": {"alice": "allowed"}}

    def issue(self, tree: Path) -> Grant:
        now = datetime.now(UTC)
        grant = Grant(
            id=uuid.uuid4().hex,
            owner="alice",
            holder="optimizer",
            target=TARGET,
            switch_epoch=1234,
            channels=CHANNELS,
            steps=2,
            issued_at=now,
            expires_at=FAR_FUTURE,
            lease_s=60.0,
            approver=Approver(by="alice", surface="chip", approved_at=now),
        )
        _issue_grant(grant, directory=tree / "alice" / wa.AUTHORIZATIONS_DIR_NAME)
        start_lease(grant.id)
        return grant

    def test_the_owners_grant_lets_the_holder_write_with_the_switch_off(self, tree: Path):
        grant = self.issue(tree)
        assert wa.authorizations_dir("alice") == tree / "alice" / wa.AUTHORIZATIONS_DIR_NAME
        assert (
            write_switch.write_verdict(ControlSystemSection.read(self.SECTION), TARGET).reason
            == "switch_off"
        )
        answer = spend_authorized_write(
            ControlSystemSection.read(self.SECTION), TARGET, OPS, authorization=grant.id
        )
        assert answer.allowed
        assert len(ledger(grant.id).spent) == 1

    def test_another_identity_under_the_same_owner_gains_nothing(self, tree: Path, monkeypatch):
        grant = self.issue(tree)
        pin_identity(monkeypatch, "bluesky")
        answer = authorized_execution_verdict(
            ControlSystemSection.read(self.SECTION), TARGET, OPS, authorization=grant.id
        )
        assert answer.reason == "auth_holder"

    def test_an_owner_the_tiers_do_not_name_refuses_first(self, tree: Path):
        grant = self.issue(tree)
        section = {"type": "epics", "writes": "allowed", "login_tiers": {"bob": "allowed"}}
        answer = authorized_execution_verdict(
            ControlSystemSection.read(section), TARGET, OPS, authorization=grant.id
        )
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
        answer = spend(self.SECTION, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "auth_unreadable"

    def test_an_unprovisioned_tree_refuses_as_the_switch_does(self, tree: Path):
        grant = self.issue(tree)
        (tree / write_switch.CONTROL_TREE_MARKER_NAME).unlink()
        assert verdict(grant.id, section=self.SECTION).reason == "record_unavailable"
