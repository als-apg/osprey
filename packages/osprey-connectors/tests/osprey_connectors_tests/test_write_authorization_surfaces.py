"""What the surfaces of a program's write authorization stand on.

:mod:`osprey_connectors.write_authorization` from the person's side: a stop
through the switch's one writer revokes the grants on the machines it names,
a holder's request has a format read the fail-closed way, an approval issues
the grant under the record lock and checks overlap against the owner's own
grants and the holders' self-issued ones, and a listing prunes what has ended
and shows what the holder's ledger ended.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, posture_store, session_hold
from osprey_connectors import write_authorization as wa
from osprey_connectors.write_authorization import (
    Approver,
    AuthorizationRefused,
    ChannelRange,
    Grant,
    Revocation,
    WriteRequest,
    approve_request,
    approved_grant,
    authorization_verdict,
    grant_in_force,
    issue_grant,
    list_grants,
    list_requests,
    parse_request,
    prune_grants,
    read_ledger,
    read_request,
    self_issue,
    self_issued_directories,
    self_issued_grants,
    spend_step,
    start_lease,
    withdraw_request,
    write_request,
)
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY

SECTION = {"type": "epics", "writes": "allowed"}
TARGET = "va"
ME = FIXTURE_IDENTITY
OPS = [("A:SP", 5.0)]
FAR_FUTURE_NS = time.time_ns() + 10 * 365 * 24 * 3600 * 10**9


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with this login's record (every switch off, an epoch) and no grant."""
    data = tmp_path / "root"
    isolate_verdict_environment(monkeypatch, data)
    for name in (
        wa.AUTHORIZATION_ENV_VAR,
        wa.LEDGER_DIR_ENV_VAR,
        wa.REQUESTS_DIR_ENV_VAR,
        wa.LEDGER_VIEW_DIR_ENV_VAR,
        "OSPREY_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    write_control_context(data, switch={})
    return data


def epoch() -> int:
    control_context.invalidate_cache()
    record = control_context.read_record()
    assert record is not None and isinstance(record.switch_epoch, int)
    return record.switch_epoch


def own_dir() -> Path:
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
        "channels": {"A:SP": ChannelRange(0.0, 10.0)},
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


def ask(**fields: Any) -> WriteRequest:
    values: dict[str, Any] = {
        "steps": 3,
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 60.0,
        "label": "run-1",
        "completion": "apply_best",
    }
    values.update(fields)
    channels = values.pop("channels", {"A:SP": (0.0, 10.0)})
    return write_request(TARGET, channels, **values)


def request_payload(**fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": 1,
        "id": uuid.uuid4().hex,
        "owner": ME,
        "holder": "optimizer",
        "target": TARGET,
        "channels": {"A:SP": {"min": 0.0, "max": 10.0}},
        "steps": 4,
        "requested_at_ns": 1,
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 30.0,
        "label": "run-7",
        "completion": "restore",
    }
    payload.update(fields)
    return payload


# -- a stop revokes -------------------------------------------------------------


class TestStopRevokes:
    def test_a_named_stop_ends_the_next_write_through_the_spend(self):
        grant = make_grant()
        issue_grant(grant)
        start_lease(grant.id)
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).allowed
        control_context.set_switch([TARGET], "off")
        control_context.invalidate_cache()
        answer = spend_step(SECTION, TARGET, OPS, authorization=grant.id)
        assert (answer.allowed, answer.reason) == (False, "auth_revoked")
        assert ME in (answer.detail or "")

    def test_a_named_stop_names_its_machines_only(self):
        on_va = make_grant()
        on_live = make_grant(target="live", channels={"L:SP": ChannelRange(0, 1)})
        issue_grant(on_va)
        issue_grant(on_live)
        control_context.set_switch(["live"], "off", by="alice")
        stored = {g.id: g for g in list_grants(own_dir())[0]}
        assert stored[on_va.id].revoked is None
        revoked = stored[on_live.id].revoked
        assert revoked is not None and revoked.by == "alice"

    def test_all_off_revokes_every_grant_and_moves_the_epoch(self):
        grants = [make_grant(), make_grant(target="live")]
        for grant in grants:
            issue_grant(grant)
        before = epoch()
        control_context.set_switch("all", "off")
        assert epoch() > before
        assert all(g.revoked is not None for g in list_grants(own_dir())[0])

    def test_a_stop_with_no_record_still_marks_the_grants(self):
        grant = make_grant()
        issue_grant(grant)
        path = control_context.record_path()
        assert path is not None
        path.unlink()
        assert control_context.set_switch([TARGET], "off") is None
        assert list_grants(own_dir())[0][0].revoked is not None

    def test_a_widening_revokes_nothing(self):
        grant = make_grant()
        issue_grant(grant)
        control_context.set_switch([TARGET], "asked", epoch=epoch())
        assert list_grants(own_dir())[0][0].revoked is None

    @pytest.mark.parametrize("failure", [RuntimeError("locked"), PermissionError(13, "denied")])
    def test_a_stop_whose_marks_cannot_be_written_ends_every_grant_by_the_epoch(
        self, monkeypatch, failure: Exception
    ):
        on_va = make_grant()
        on_live = make_grant(target="live", channels={"L:SP": ChannelRange(0, 1)})
        issue_grant(on_va)
        issue_grant(on_live)
        control_context.set_switch([TARGET], "asked", epoch=epoch())
        before = epoch()

        def jammed(**_kwargs: Any) -> tuple[str, ...]:
            raise failure

        monkeypatch.setattr(wa, "revoke_grants", jammed)
        stored = control_context.set_switch([TARGET], "off")
        control_context.invalidate_cache()
        record = control_context.read_record()
        assert stored is not None and record is not None
        assert record.switch_epoch == stored.switch_epoch
        assert control_context.switch_position(record, TARGET) == "off"
        assert epoch() > before  # wider than asked: every machine's grants end
        assert not grant_in_force(on_va, record)
        assert not grant_in_force(on_live, record)

    def test_a_lock_is_created_group_writable_whatever_the_umask(self):
        directory = own_dir()
        directory.mkdir(parents=True, exist_ok=True)
        previous = os.umask(0o077)
        try:
            issue_grant(make_grant())
        finally:
            os.umask(previous)
        lock = directory / "authorizations.lock"
        assert lock.stat().st_mode & 0o777 == 0o660

    @pytest.mark.skipif(os.geteuid() == 0, reason="root opens any file for writing")
    def test_a_stop_still_marks_behind_a_lock_this_uid_cannot_write(self):
        # A lock another uid created at 0o640 before the locks were made 0o660:
        # this uid reaches it through the group, read-only. The owner bits stand
        # in for that here (0o440), which the same uid is held to.
        grant = make_grant()
        issue_grant(grant)
        lock = own_dir() / "authorizations.lock"
        lock.chmod(0o440)
        before = epoch()
        control_context.set_switch([TARGET], "off", by="alice")
        revoked = list_grants(own_dir())[0][0].revoked
        assert revoked is not None and revoked.by == "alice"
        assert epoch() == before  # the marks were written; no fallback was needed


# -- the request ----------------------------------------------------------------


class TestRequestFormat:
    def test_round_trip(self):
        payload = request_payload()
        request = parse_request(json.dumps(payload))
        assert request.to_payload() == payload
        assert request.channels["A:SP"] == ChannelRange(0.0, 10.0)
        assert request.completion == "restore"

    @pytest.mark.parametrize(
        "change",
        [
            {"schema": 2},
            {"extra": 1},
            {"completion": "apply_worst"},
            {"steps": 0},
            {"lease_s": 0},
            {"expires_at_ns": 1},
            {"channels": {}},
            {"channels": {"A:SP": {"min": 2, "max": 1}}},
            {"channels": {"A:SP": {"min": 0, "max": "1"}}},
            {"id": "not-an-id"},
            {"holder": " padded"},
            {"label": None},
        ],
        ids=lambda change: next(iter(change)),
    )
    def test_anything_else_is_unreadable(self, change: dict[str, Any]):
        with pytest.raises(ValueError):
            parse_request(request_payload(**change))

    def test_a_non_finite_bound_and_a_repeated_key_are_unreadable(self):
        text = json.dumps(request_payload()).replace("10.0", "NaN")
        with pytest.raises(ValueError):
            parse_request(text)
        with pytest.raises(ValueError):
            parse_request('{"schema": 1, "schema": 1}')
        with pytest.raises(ValueError):
            parse_request(request_payload(channels={"A:SP": {"min": 0, "max": float("inf")}}))

    def test_the_holder_writes_into_its_ledger_directory(self, tmp_path, monkeypatch):
        ledgers = tmp_path / "ledger"
        ledgers.mkdir()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        request = ask()
        path = ledgers / "requests" / f"{request.id}.json"
        assert read_request(path) == request
        assert (request.owner, request.holder) == (ME, ME)
        assert wa.requests_dir() == ledgers / "requests"
        assert list_requests(ledgers / "requests") == ((request,), ())
        assert withdraw_request(request.id)
        assert not withdraw_request(request.id)

    def test_a_terminal_reads_the_bound_directory(self, tmp_path, monkeypatch):
        bound = tmp_path / "bound"
        monkeypatch.setenv(wa.REQUESTS_DIR_ENV_VAR, str(bound))
        assert wa.requests_dir() == bound
        assert list_requests(bound) == ((), ())

    def test_a_listing_reports_what_it_cannot_read_and_skips_other_names(self, tmp_path):
        directory = tmp_path / "requests"
        directory.mkdir()
        good = parse_request(request_payload())
        (directory / f"{good.id}.json").write_text(json.dumps(good.to_payload()))
        broken = directory / f"{uuid.uuid4().hex}.json"
        broken.write_text("{")
        misnamed = directory / f"{uuid.uuid4().hex}.json"
        misnamed.write_text(json.dumps(request_payload()))
        (directory / "notes.txt").write_text("x")
        os.symlink(directory / f"{good.id}.json", directory / f"{uuid.uuid4().hex}.json")
        found, unreadable = list_requests(directory)
        assert found == (good,)
        assert set(unreadable) == {broken, misnamed} | {
            p for p in directory.iterdir() if p.is_symlink()
        }

    def test_a_holder_with_no_owner_cannot_ask(self, monkeypatch):
        monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "/nonexistent")
        with pytest.raises(AuthorizationRefused) as refused:
            ask()
        assert refused.value.reason == "owner_unknown"


# -- approval -------------------------------------------------------------------


class TestApproval:
    def test_the_grant_is_the_request_under_its_id(self):
        request = ask()
        assert approved_grant(request.id) is None
        grant = approve_request(request, surface="cli")
        assert grant.id == request.id
        assert (grant.approver.identity, grant.approver.surface) == (ME, "cli")
        assert grant.switch_epoch == epoch()
        assert (grant.channels, grant.steps, grant.label) == (
            request.channels,
            request.steps,
            request.label,
        )
        assert approved_grant(request.id) == grant
        start_lease(grant.id)
        assert authorization_verdict(SECTION, TARGET, OPS, authorization=grant.id).allowed

    def test_an_approval_is_issued_once(self):
        request = ask()
        approve_request(request, surface="chip")
        with pytest.raises(FileExistsError):
            approve_request(request, surface="chip")

    def test_no_record_yet_is_stamped_with_an_epoch(self):
        path = control_context.record_path()
        assert path is not None
        path.unlink()
        grant = approve_request(ask(), surface="cli")
        assert grant.switch_epoch == epoch()

    def test_an_unreadable_record_refuses(self):
        path = control_context.record_path()
        assert path is not None
        path.write_text("{")
        with pytest.raises(AuthorizationRefused) as refused:
            approve_request(ask(), surface="cli")
        assert refused.value.reason == "record_unavailable"
        assert list_grants(own_dir()) == ((), ())

    def test_an_expired_request_refuses(self, monkeypatch):
        request = ask(expires_at_ns=time.time_ns() + 10**9)
        monkeypatch.setattr(wa, "_now_ns", lambda: request.expires_at_ns)
        with pytest.raises(AuthorizationRefused) as refused:
            approve_request(request, surface="chip")
        assert refused.value.reason == "auth_expired"

    def test_another_owners_request_needs_its_record(self):
        request = parse_request(request_payload(owner="bob"))
        with pytest.raises(AuthorizationRefused) as refused:
            approve_request(request, surface="cli")
        assert refused.value.reason == "owner_foreign"

    def test_a_stop_after_the_approval_revokes_it(self):
        grant = approve_request(ask(), surface="chip")
        control_context.set_switch([TARGET], "off")
        with pytest.raises(AuthorizationRefused) as refused:
            approved_grant(grant.id)
        assert refused.value.reason == "auth_revoked"

    def test_the_record_lock_is_held_around_the_issue(self, monkeypatch):
        held: list[bool] = []
        real = wa.issue_grant

        def probe(grant: Grant, **kwargs: Any) -> Path:
            path = control_context.record_path()
            assert path is not None
            lock = path.with_name(path.name + ".lock")
            import fcntl

            fd = os.open(lock, os.O_RDWR)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held.append(False)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except BlockingIOError:
                    held.append(True)
            finally:
                os.close(fd)
            return real(grant, **kwargs)

        monkeypatch.setattr(wa, "issue_grant", probe)
        approve_request(ask(), surface="cli")
        assert held == [True]


# -- overlap ---------------------------------------------------------------------


def _tree_grant(tree: Path, owner: str, **fields: Any) -> Grant:
    grant = make_grant(owner=owner, **fields)
    issue_grant(grant, directory=tree / owner / "authorizations")
    return grant


class TestOverlap:
    @pytest.fixture
    def tree(self, root: Path) -> Path:
        return root / "control_target"

    def test_another_holders_grant_beside_it_on_the_channel_refuses(self):
        held = make_grant(holder="bluesky")
        issue_grant(held, directory=own_dir())
        with pytest.raises(AuthorizationRefused) as refused:
            approve_request(ask(), surface="chip")
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert [g.id for g in list_grants(own_dir())[0]] == [held.id]

    @pytest.mark.parametrize(
        "fields",
        [
            {"holder": "bluesky", "channels": {"Z:SP": ChannelRange(0, 1)}},
            {"holder": "bluesky", "target": "live"},
            {"holder": "bluesky", "revoked": Revocation(at_ns=1, by="bob")},
            {"holder": ME},
        ],
        ids=["other-channel", "other-machine", "revoked", "same-holder"],
    )
    def test_what_does_not_overlap(self, fields):
        issue_grant(make_grant(**fields), directory=own_dir())
        assert approve_request(ask(), surface="chip").id

    def test_another_owners_grant_is_not_read(self, tree):
        _tree_grant(tree, "bob", holder="bluesky")
        assert approve_request(ask(), surface="chip").id

    def test_an_unreadable_grant_beside_it_is_skipped_not_a_veto(self):
        (own_dir()).mkdir(parents=True, exist_ok=True)
        (own_dir() / f"{uuid.uuid4().hex}.json").write_text("{")
        assert approve_request(ask(), surface="chip").id

    def test_an_unreadable_grant_in_the_ledger_view_is_skipped_not_a_veto(
        self, tmp_path, monkeypatch
    ):
        ledgers = tmp_path / "write_ledger"
        ledgers.mkdir()
        (ledgers / f"{uuid.uuid4().hex}.json").write_text("{")
        monkeypatch.setenv(wa.LEDGER_VIEW_DIR_ENV_VAR, str(ledgers))
        assert approve_request(ask(), surface="chip").id

    def test_a_terminal_sees_the_holders_self_issued_grants_through_its_view(
        self, tmp_path, monkeypatch
    ):
        # A program's grant at allowed, in the holders' ledger directory, which
        # a web terminal binds read-only: an approval there must not overlap it.
        ledgers = tmp_path / "write_ledger"
        held = make_grant(
            holder="tuning-scripts",
            approver=Approver(identity="tuning-scripts", surface="write_switch", at_ns=1),
        )
        issue_grant(held, directory=ledgers)
        monkeypatch.setenv(wa.LEDGER_VIEW_DIR_ENV_VAR, str(ledgers))
        assert ledgers in self_issued_directories()
        with pytest.raises(AuthorizationRefused) as refused:
            approve_request(ask(), surface="chip")
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert list_grants(own_dir()) == ((), ())

    def test_only_a_switch_approval_is_a_self_issued_grant(self, tmp_path):
        ledgers = tmp_path / "write_ledger"
        own = make_grant(approver=Approver(identity=ME, surface="write_switch", at_ns=1))
        planted = make_grant(approver=Approver(identity=ME, surface="chip", at_ns=1))
        issue_grant(own, directory=ledgers)
        issue_grant(planted, directory=ledgers, peers=())
        assert [g.id for g in self_issued_grants(ledgers)[0]] == [own.id]

    def test_self_issue_refuses_another_holders_self_issued_grant(self):
        issue_grant(
            make_grant(
                holder="bluesky",
                approver=Approver(identity="bluesky", surface="write_switch", at_ns=1),
            ),
            directory=own_dir(),
        )
        control_context.set_switch([TARGET], "allowed", epoch=epoch())
        with pytest.raises(AuthorizationRefused) as refused:
            self_issue(
                SECTION, TARGET, {"A:SP": (0.0, 1.0)}, steps=1, expires_at_ns=FAR_FUTURE_NS,
                lease_s=30.0,
            )  # fmt: skip
        assert refused.value.reason == "auth_overlap"


# -- listing ---------------------------------------------------------------------


class TestPruneAndShow:
    def test_prune_removes_only_what_ended_long_enough_ago(self):
        now = time.time_ns()
        hour = 3600 * 10**9
        live = make_grant()
        stopped_long_ago = make_grant(revoked=Revocation(at_ns=now - 2 * hour, by="bob"))
        stopped_now = make_grant(revoked=Revocation(at_ns=now, by="bob"))
        expired_long_ago = make_grant(issued_at_ns=1, expires_at_ns=now - 2 * hour)
        for grant in (live, stopped_long_ago, stopped_now, expired_long_ago):
            issue_grant(grant, directory=own_dir())
        broken = own_dir() / f"{uuid.uuid4().hex}.json"
        broken.write_text("{")
        removed = prune_grants(own_dir(), now_ns=now)
        assert set(removed) == {stopped_long_ago.id, expired_long_ago.id}
        assert {g.id for g in list_grants(own_dir())[0]} == {live.id, stopped_now.id}
        assert broken.exists()
        assert prune_grants(own_dir() / "absent") == ()

    def test_a_recently_stopped_holder_still_hears_auth_revoked(self):
        grant = make_grant()
        issue_grant(grant)
        start_lease(grant.id)
        control_context.set_switch([TARGET], "off")
        prune_grants(own_dir())
        answer = authorization_verdict(SECTION, TARGET, OPS, authorization=grant.id)
        assert answer.reason == "auth_revoked"

    def test_in_force_is_live_under_the_records_epoch(self):
        grant = make_grant()
        control_context.invalidate_cache()
        record = control_context.read_record()
        assert grant_in_force(grant, record)
        assert not grant_in_force(grant, None)
        assert not grant_in_force(grant, record, now_ns=FAR_FUTURE_NS)
        stale = make_grant(switch_epoch=grant.switch_epoch - 1)
        assert not grant_in_force(stale, record)

    def test_a_self_issued_grant_is_in_force_only_at_allowed(self):
        control_context.set_switch([TARGET], "allowed", epoch=epoch())
        grant = make_grant(approver=Approver(identity=ME, surface="write_switch", at_ns=1))
        control_context.invalidate_cache()
        assert grant_in_force(grant, control_context.read_record())
        control_context.set_switch([TARGET], "asked")
        control_context.invalidate_cache()
        assert not grant_in_force(grant, control_context.read_record())

    def test_a_grant_its_ledger_ended_is_not_in_force(self):
        grant = make_grant(steps=1)
        issue_grant(grant)
        control_context.invalidate_cache()
        record = control_context.read_record()
        ledgers = [own_dir()]
        assert wa.grant_standing(grant, record, ledgers) == wa.GrantStanding(grant, True, 0, None)
        start_lease(grant.id)
        assert wa.grant_standing(grant, record, ledgers).in_force
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).allowed
        spent = wa.grant_standing(grant, record, ledgers)
        assert (spent.in_force, spent.used) == (False, 1)
        assert grant_in_force(grant, record)  # the grant alone still stands

    def test_a_grant_whose_lease_lapsed_is_not_in_force(self):
        grant = make_grant()
        issue_grant(grant)
        start_lease(grant.id)
        control_context.invalidate_cache()
        record = control_context.read_record()
        assert wa.grant_standing(grant, record, [own_dir()]).in_force
        later = wa.grant_standing(grant, record, [own_dir()], now_ns=FAR_FUTURE_NS - 1)
        assert later.in_force is False and later.used == 0

    def test_an_unreadable_ledger_counts_nothing(self):
        grant = make_grant()
        issue_grant(grant)
        (own_dir() / f"{grant.id}.ledger.json").write_text("{")
        control_context.invalidate_cache()
        standing = wa.grant_standing(grant, control_context.read_record(), [own_dir()])
        assert standing.used is None and standing.ledger is None

    def test_listed_grants_reads_each_directory_once_and_each_grant_once(self, tmp_path):
        mine = make_grant()
        issue_grant(mine)
        ledgers = tmp_path / "write_ledger"
        own = make_grant(approver=Approver(identity=ME, surface="write_switch", at_ns=1))
        planted = make_grant(approver=Approver(identity=ME, surface="chip", at_ns=1))
        issue_grant(own, directory=ledgers)
        issue_grant(planted, directory=ledgers, peers=())
        broken = ledgers / f"{uuid.uuid4().hex}.json"
        broken.write_text("{")
        found, unreadable = wa.listed_grants([own_dir(), own_dir()], [ledgers, own_dir()])
        assert [(g.id, d) for g, d in found] == [(mine.id, own_dir()), (own.id, ledgers)]
        assert unreadable == [broken]

    def test_read_ledger(self):
        grant = make_grant()
        issue_grant(grant)
        start_lease(grant.id)
        assert spend_step(SECTION, TARGET, OPS, authorization=grant.id).allowed
        ledger = read_ledger(own_dir(), grant.id)
        assert ledger is not None and len(ledger.steps) == 1
        assert read_ledger(own_dir(), uuid.uuid4().hex) is None
        (own_dir() / f"{grant.id}.ledger.json").write_text("{")
        with pytest.raises(ValueError):
            read_ledger(own_dir(), grant.id)
