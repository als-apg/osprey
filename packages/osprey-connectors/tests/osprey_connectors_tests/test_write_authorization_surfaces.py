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
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, session_hold, testing, write_switch
from osprey_connectors import write_authorization as wa
from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.errors import AlreadyApprovedError
from osprey_connectors.model import evolve
from osprey_connectors.write_authorization import (
    Approver,
    AuthorizationRefusedError,
    ChannelRange,
    Grant,
    Revocation,
    WriteRequest,
    _grant_in_force,
    _issue_grant,
    _parse_request,
    _self_issued_grants,
    approve_request,
    approved_grant,
    list_grants,
    list_requests,
    prune_grants,
    read_request,
    self_issue,
    self_issued_directories,
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
FAR_FUTURE = datetime.now(UTC) + timedelta(hours=12)
#: A fixed instant for a stamp whose value does not matter.
T0 = datetime(2026, 1, 1, tzinfo=UTC)
#: The smallest step an instant takes.
TICK = timedelta(microseconds=1)


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


def spend(section: Any, target: str, operations: Any, *, authorization: str) -> Any:
    """One write call judged and its step spent, with no switch verdict read first."""
    return wa._spend(
        ControlSystemSection.read(section), target, operations, authorization, None, None
    )


def epoch() -> int:
    testing.invalidate_cache()
    record = control_context.read_record()
    assert record is not None and isinstance(record.switch_epoch, int)
    return record.switch_epoch


def own_dir() -> Path:
    path = wa.authorizations_dir()
    assert path is not None
    return path


def _grants(directory: Path) -> tuple[tuple[Grant, ...], tuple[Path, ...]]:
    """What a listing of *directory* reads: its grants, and the paths it could not read."""
    listing = list_grants(directory)
    return tuple(listed.grant for listed in listing.items), listing.unreadable


def make_grant(**fields: Any) -> Grant:
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "owner": ME,
        "holder": ME,
        "target": TARGET,
        "switch_epoch": epoch(),
        "channels": {"A:SP": ChannelRange(min=0.0, max=10.0)},
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


def ask(**fields: Any) -> WriteRequest:
    values: dict[str, Any] = {
        "steps": 3,
        "expires_at": FAR_FUTURE,
        "lease_s": 60.0,
        "label": "run-1",
        "completion": "apply_best",
    }
    values.update(fields)
    channels = values.pop("channels", {"A:SP": ChannelRange.of(0.0, 10.0)})
    return write_request(TARGET, channels, **values)


def request_payload(**fields: Any) -> dict[str, Any]:
    """A request file's object; a ``datetime`` in *fields* is written as the file stores it."""
    payload: dict[str, Any] = {
        "schema": 1,
        "id": uuid.uuid4().hex,
        "owner": ME,
        "holder": "optimizer",
        "target": TARGET,
        "channels": {"A:SP": {"min": 0.0, "max": 10.0}},
        "steps": 4,
        "requested_at": datetime.now(UTC).isoformat(),
        "expires_at": FAR_FUTURE.isoformat(),
        "lease_s": 30.0,
        "label": "run-7",
        "completion": "restore",
    }
    payload.update(
        {key: v.isoformat() if isinstance(v, datetime) else v for key, v in fields.items()}
    )
    return payload


# -- a stop revokes -------------------------------------------------------------


class TestStopRevokes:
    def test_a_named_stop_ends_the_next_write_through_the_spend(self):
        grant = make_grant()
        _issue_grant(grant)
        start_lease(grant.id)
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).allowed
        control_context.set_switch([TARGET], "off")
        testing.invalidate_cache()
        answer = spend(SECTION, TARGET, OPS, authorization=grant.id)
        assert (answer.allowed, answer.reason) == (False, "auth_revoked")
        assert ME in (answer.detail or "")

    def test_a_named_stop_names_its_machines_only(self):
        on_va = make_grant()
        on_live = make_grant(target="live", channels={"L:SP": ChannelRange(min=0, max=1)})
        _issue_grant(on_va)
        _issue_grant(on_live)
        control_context.set_switch(["live"], "off", by="alice")
        stored = {g.id: g for g in _grants(own_dir())[0]}
        assert stored[on_va.id].revoked is None
        revoked = stored[on_live.id].revoked
        assert revoked is not None and revoked.by == "alice"

    def test_all_off_revokes_every_grant_and_moves_the_epoch(self):
        grants = [make_grant(), make_grant(target="live")]
        for grant in grants:
            _issue_grant(grant)
        before = epoch()
        control_context.set_switch("all", "off")
        assert epoch() > before
        assert all(g.revoked is not None for g in _grants(own_dir())[0])

    def test_a_stop_with_no_record_still_marks_the_grants(self):
        grant = make_grant()
        _issue_grant(grant)
        path = control_context.record_path()
        assert path is not None
        path.unlink()
        assert control_context.set_switch([TARGET], "off") is None
        assert _grants(own_dir())[0][0].revoked is not None

    def test_a_widening_revokes_nothing(self):
        grant = make_grant()
        _issue_grant(grant)
        control_context.set_switch([TARGET], "asked", epoch=epoch())
        assert _grants(own_dir())[0][0].revoked is None

    @pytest.mark.parametrize("failure", [RuntimeError("locked"), PermissionError(13, "denied")])
    def test_a_stop_whose_marks_cannot_be_written_ends_every_grant_by_the_epoch(
        self, monkeypatch, failure: Exception
    ):
        on_va = make_grant()
        on_live = make_grant(target="live", channels={"L:SP": ChannelRange(min=0, max=1)})
        _issue_grant(on_va)
        _issue_grant(on_live)
        control_context.set_switch([TARGET], "asked", epoch=epoch())
        before = epoch()

        def jammed(**_kwargs: Any) -> tuple[str, ...]:
            raise failure

        monkeypatch.setattr(wa, "revoke_grants", jammed)
        stored = control_context.set_switch([TARGET], "off")
        testing.invalidate_cache()
        record = control_context.read_record()
        assert stored is not None and record is not None
        assert record.switch_epoch == stored.switch_epoch
        assert control_context.switch_position(record, TARGET) == "off"
        assert epoch() > before  # wider than asked: every machine's grants end
        assert not _grant_in_force(on_va, record)
        assert not _grant_in_force(on_live, record)

    def test_a_stop_by_a_name_that_is_not_a_login_ends_every_grant_by_the_epoch(self):
        """No mark can carry the name, so the stop falls back on the epoch: wider, never narrower."""
        on_va = make_grant()
        on_live = make_grant(target="live", channels={"L:SP": ChannelRange(min=0, max=1)})
        _issue_grant(on_va)
        _issue_grant(on_live)
        before = epoch()
        stored = control_context.set_switch([TARGET], "off", by=" alice")
        testing.invalidate_cache()
        record = control_context.read_record()
        assert stored is not None and record is not None
        assert control_context.switch_position(record, TARGET) == "off"
        assert epoch() > before
        assert all(grant.revoked is None for grant in _grants(own_dir())[0])
        assert not _grant_in_force(on_va, record)
        assert not _grant_in_force(on_live, record)
        answer = spend(SECTION, TARGET, OPS, authorization=on_va.id)
        assert (answer.allowed, answer.reason) == (False, "auth_revoked")

    def test_a_listing_reports_a_directory_it_cannot_list(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        listing = list_grants(link)
        assert (listing.items, listing.unreadable) == ((), (link,))

    def test_a_lock_is_created_group_writable_whatever_the_umask(self):
        directory = own_dir()
        directory.mkdir(parents=True, exist_ok=True)
        previous = os.umask(0o077)
        try:
            _issue_grant(make_grant())
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
        _issue_grant(grant)
        lock = own_dir() / "authorizations.lock"
        lock.chmod(0o440)
        before = epoch()
        control_context.set_switch([TARGET], "off", by="alice")
        revoked = _grants(own_dir())[0][0].revoked
        assert revoked is not None and revoked.by == "alice"
        assert epoch() == before  # the marks were written; no fallback was needed


# -- the request ----------------------------------------------------------------


class TestRequestFormat:
    def test_round_trip(self):
        payload = request_payload()
        request = _parse_request(json.dumps(payload))
        assert request.model_dump(mode="json") == payload
        assert request.channels["A:SP"] == ChannelRange(min=0.0, max=10.0)
        assert request.completion == "restore"

    @pytest.mark.parametrize(
        "change",
        [
            {"schema": 2},
            {"extra": 1},
            {"completion": "apply_worst"},
            {"steps": 0},
            {"lease_s": 0},
            {"expires_at": (wa._EPOCH + TICK).isoformat()},
            {"requested_at": "2026-01-01T00:00:00"},
            {"requested_at_ns": 1},
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
            _parse_request(json.dumps(request_payload(**change)))

    @pytest.mark.parametrize(
        "change",
        [
            {"steps": wa.MAX_STEPS + 1},
            {"lease_s": wa._MAX_LEASE_S + 1},
            {"expires_at": datetime.now(UTC) + 2 * timedelta(seconds=wa.MAX_EXPIRES_S)},
        ],
        ids=lambda change: next(iter(change)),
    )
    def test_a_request_past_a_ceiling_is_read_but_never_written(
        self, change: dict[str, Any], tmp_path, monkeypatch
    ):
        """The ceilings are the writer's: a file on disk past one is honoured as before."""
        payload = request_payload(**change)
        assert _parse_request(json.dumps(payload)).model_dump(mode="json") == payload
        ledgers = tmp_path / "ledger"
        ledgers.mkdir()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        with pytest.raises(ValueError, match="ceiling|more than"):
            ask(**change)
        assert not (ledgers / "requests").exists() or not any((ledgers / "requests").iterdir())

    def test_a_non_finite_bound_and_a_repeated_key_are_unreadable(self):
        text = json.dumps(request_payload()).replace("10.0", "NaN")
        with pytest.raises(ValueError):
            _parse_request(text)
        with pytest.raises(ValueError):
            _parse_request('{"schema": 1, "schema": 1}')
        with pytest.raises(ValueError):
            _parse_request(
                json.dumps(request_payload(channels={"A:SP": {"min": 0, "max": float("inf")}}))
            )

    def test_the_holder_writes_into_its_ledger_directory(self, tmp_path, monkeypatch):
        ledgers = tmp_path / "ledger"
        ledgers.mkdir()
        monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(ledgers))
        request = ask()
        path = ledgers / "requests" / f"{request.id}.json"
        assert read_request(path) == request
        assert (request.owner, request.holder) == (ME, ME)
        assert wa.requests_dir() == ledgers / "requests"
        assert list_requests(ledgers / "requests").items == (request,)
        assert withdraw_request(request.id)
        assert not withdraw_request(request.id)

    def test_a_terminal_reads_the_bound_directory(self, tmp_path, monkeypatch):
        bound = tmp_path / "bound"
        monkeypatch.setenv(wa.REQUESTS_DIR_ENV_VAR, str(bound))
        assert wa.requests_dir() == bound
        listing = list_requests(bound)
        assert (listing.items, listing.unreadable) == ((), ())

    def test_a_listing_reports_what_it_cannot_read_and_skips_other_names(self, tmp_path):
        directory = tmp_path / "requests"
        directory.mkdir()
        good = _parse_request(json.dumps(request_payload()))
        (directory / f"{good.id}.json").write_text(json.dumps(good.model_dump(mode="json")))
        broken = directory / f"{uuid.uuid4().hex}.json"
        broken.write_text("{")
        misnamed = directory / f"{uuid.uuid4().hex}.json"
        misnamed.write_text(json.dumps(request_payload()))
        (directory / "notes.txt").write_text("x")
        os.symlink(directory / f"{good.id}.json", directory / f"{uuid.uuid4().hex}.json")
        listing = list_requests(directory)
        assert listing.items == (good,)
        assert set(listing.unreadable) == {broken, misnamed} | {
            p for p in directory.iterdir() if p.is_symlink()
        }

    def test_a_holder_with_no_owner_cannot_ask(self, monkeypatch):
        monkeypatch.setenv(write_switch.CONTROL_CONTEXT_TREE_ENV_VAR, "/nonexistent")
        with pytest.raises(AuthorizationRefusedError) as refused:
            ask()
        assert refused.value.reason == "owner_unknown"


# -- approval -------------------------------------------------------------------


class TestApproval:
    def test_a_request_past_a_ceiling_is_never_approved(self):
        """The approval's write applies the ceilings, so a budget past :data:`MAX_STEPS` refuses."""
        request = evolve(ask(), steps=wa.MAX_STEPS + 1)
        with pytest.raises(ValueError):
            approve_request(request, surface="cli")
        assert approved_grant(request.id) is None

    def test_the_grant_is_the_request_under_its_id(self):
        request = ask()
        assert approved_grant(request.id) is None
        grant = approve_request(request, surface="cli")
        assert grant.id == request.id
        assert (grant.approver.by, grant.approver.surface) == (ME, "cli")
        assert grant.switch_epoch == epoch()
        assert (grant.channels, grant.steps, grant.label) == (
            request.channels,
            request.steps,
            request.label,
        )
        assert approved_grant(request.id) == grant
        start_lease(grant.id)
        assert wa._verdict(
            ControlSystemSection.read(SECTION), TARGET, OPS, grant.id, None, None
        ).allowed

    def test_an_approval_is_issued_once(self):
        request = ask()
        approve_request(request, surface="chip")
        with pytest.raises(AlreadyApprovedError) as caught:
            approve_request(request, surface="chip")
        assert caught.value.authorization == request.id
        assert isinstance(caught.value, FileExistsError)

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
        with pytest.raises(AuthorizationRefusedError) as refused:
            approve_request(ask(), surface="cli")
        assert refused.value.reason == "record_unavailable"
        assert _grants(own_dir()) == ((), ())

    def test_an_expired_request_refuses(self, monkeypatch):
        request = ask(expires_at=datetime.now(UTC) + timedelta(seconds=1))
        monkeypatch.setattr(wa, "_now", lambda: request.expires_at)
        with pytest.raises(AuthorizationRefusedError) as refused:
            approve_request(request, surface="chip")
        assert refused.value.reason == "auth_expired"

    def test_another_owners_request_needs_its_record(self):
        request = _parse_request(json.dumps(request_payload(owner="bob")))
        with pytest.raises(AuthorizationRefusedError) as refused:
            approve_request(request, surface="cli")
        assert refused.value.reason == "owner_foreign"

    def test_a_stop_after_the_approval_revokes_it(self):
        grant = approve_request(ask(), surface="chip")
        control_context.set_switch([TARGET], "off")
        with pytest.raises(AuthorizationRefusedError) as refused:
            approved_grant(grant.id)
        assert refused.value.reason == "auth_revoked"

    def test_the_record_lock_is_held_around_the_issue(self, monkeypatch):
        held: list[bool] = []
        real = wa._issue_grant

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

        monkeypatch.setattr(wa, "_issue_grant", probe)
        approve_request(ask(), surface="cli")
        assert held == [True]


# -- overlap ---------------------------------------------------------------------


def _tree_grant(tree: Path, owner: str, **fields: Any) -> Grant:
    grant = make_grant(owner=owner, **fields)
    _issue_grant(grant, directory=tree / owner / "authorizations")
    return grant


class TestOverlap:
    @pytest.fixture
    def tree(self, root: Path) -> Path:
        return root / "control_target"

    def test_another_holders_grant_beside_it_on_the_channel_refuses(self):
        held = make_grant(holder="bluesky")
        _issue_grant(held, directory=own_dir())
        with pytest.raises(AuthorizationRefusedError) as refused:
            approve_request(ask(), surface="chip")
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert [g.id for g in _grants(own_dir())[0]] == [held.id]

    @pytest.mark.parametrize(
        "fields",
        [
            {"holder": "bluesky", "channels": {"Z:SP": ChannelRange(min=0, max=1)}},
            {"holder": "bluesky", "target": "live"},
            {"holder": "bluesky", "revoked": Revocation(revoked_at=T0, by="bob")},
            {"holder": ME},
        ],
        ids=["other-channel", "other-machine", "revoked", "same-holder"],
    )
    def test_what_does_not_overlap(self, fields):
        _issue_grant(make_grant(**fields), directory=own_dir())
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
            approver=Approver(by="tuning-scripts", surface="write_switch", approved_at=T0),
        )
        _issue_grant(held, directory=ledgers)
        monkeypatch.setenv(wa.LEDGER_VIEW_DIR_ENV_VAR, str(ledgers))
        assert ledgers in self_issued_directories()
        with pytest.raises(AuthorizationRefusedError) as refused:
            approve_request(ask(), surface="chip")
        assert refused.value.reason == "auth_overlap"
        assert held.id in (refused.value.detail or "")
        assert _grants(own_dir()) == ((), ())

    def test_only_a_switch_approval_is_a_self_issued_grant(self, tmp_path):
        ledgers = tmp_path / "write_ledger"
        own = make_grant(approver=Approver(by=ME, surface="write_switch", approved_at=T0))
        planted = make_grant(approver=Approver(by=ME, surface="chip", approved_at=T0))
        _issue_grant(own, directory=ledgers)
        _issue_grant(planted, directory=ledgers, peers=())
        assert [g.id for g in _self_issued_grants(ledgers)[0]] == [own.id]

    def test_self_issue_refuses_another_holders_self_issued_grant(self):
        _issue_grant(
            make_grant(
                holder="bluesky",
                approver=Approver(by="bluesky", surface="write_switch", approved_at=T0),
            ),
            directory=own_dir(),
        )
        control_context.set_switch([TARGET], "allowed", epoch=epoch())
        with pytest.raises(AuthorizationRefusedError) as refused:
            self_issue(
                ControlSystemSection.read(SECTION), TARGET, {"A:SP": ChannelRange.of(0.0, 1.0)}, steps=1, expires_at=FAR_FUTURE,
                lease_s=30.0,
            )  # fmt: skip
        assert refused.value.reason == "auth_overlap"


# -- listing ---------------------------------------------------------------------


class TestPruneAndShow:
    def test_prune_removes_only_what_ended_long_enough_ago(self):
        now = datetime.now(UTC)
        hour = timedelta(hours=1)
        live = make_grant()
        stopped_long_ago = make_grant(revoked=Revocation(revoked_at=now - 2 * hour, by="bob"))
        stopped_now = make_grant(revoked=Revocation(revoked_at=now, by="bob"))
        expired_long_ago = make_grant(issued_at=now - 3 * hour, expires_at=now - 2 * hour)
        for grant in (live, stopped_long_ago, stopped_now, expired_long_ago):
            _issue_grant(grant, directory=own_dir())
        broken = own_dir() / f"{uuid.uuid4().hex}.json"
        broken.write_text("{")
        removed = prune_grants(own_dir(), now=now)
        assert set(removed) == {stopped_long_ago.id, expired_long_ago.id}
        assert {g.id for g in _grants(own_dir())[0]} == {live.id, stopped_now.id}
        assert broken.exists()
        assert prune_grants(own_dir() / "absent") == ()

    def test_a_recently_stopped_holder_still_hears_auth_revoked(self):
        grant = make_grant()
        _issue_grant(grant)
        start_lease(grant.id)
        control_context.set_switch([TARGET], "off")
        prune_grants(own_dir())
        answer = wa._verdict(ControlSystemSection.read(SECTION), TARGET, OPS, grant.id, None, None)
        assert answer.reason == "auth_revoked"

    def test_in_force_is_live_under_the_records_epoch(self):
        grant = make_grant()
        testing.invalidate_cache()
        record = control_context.read_record()
        assert _grant_in_force(grant, record)
        assert not _grant_in_force(grant, None)
        assert not _grant_in_force(grant, record, now=FAR_FUTURE)
        stale = make_grant(switch_epoch=grant.switch_epoch - 1)
        assert not _grant_in_force(stale, record)

    def test_a_self_issued_grant_is_in_force_only_at_allowed(self):
        control_context.set_switch([TARGET], "allowed", epoch=epoch())
        grant = make_grant(approver=Approver(by=ME, surface="write_switch", approved_at=T0))
        testing.invalidate_cache()
        assert _grant_in_force(grant, control_context.read_record())
        control_context.set_switch([TARGET], "asked")
        testing.invalidate_cache()
        assert not _grant_in_force(grant, control_context.read_record())

    def test_a_grant_its_ledger_ended_is_not_in_force(self):
        grant = make_grant(steps=1)
        _issue_grant(grant)
        testing.invalidate_cache()
        record = control_context.read_record()
        ledgers = [own_dir()]
        assert wa.grant_standing(grant, record, ledgers) == wa.GrantStanding(
            grant=grant, in_force=True, steps_spent=0, ledger=None
        )
        start_lease(grant.id)
        assert wa.grant_standing(grant, record, ledgers).in_force
        assert spend(SECTION, TARGET, OPS, authorization=grant.id).allowed
        spent = wa.grant_standing(grant, record, ledgers)
        assert (spent.in_force, spent.steps_spent) == (False, 1)
        assert _grant_in_force(grant, record)  # the grant alone still stands

    def test_a_grant_whose_lease_lapsed_is_not_in_force(self):
        grant = make_grant()
        _issue_grant(grant)
        start_lease(grant.id)
        testing.invalidate_cache()
        record = control_context.read_record()
        assert wa.grant_standing(grant, record, [own_dir()]).in_force
        later = wa.grant_standing(grant, record, [own_dir()], now=FAR_FUTURE - TICK)
        assert later.in_force is False and later.steps_spent == 0

    def test_an_unreadable_ledger_counts_nothing(self):
        grant = make_grant()
        _issue_grant(grant)
        (own_dir() / f"{grant.id}.ledger.json").write_text("{")
        testing.invalidate_cache()
        standing = wa.grant_standing(grant, control_context.read_record(), [own_dir()])
        assert standing.steps_spent is None and standing.ledger is None

    def test_a_listing_reads_each_directory_once_and_each_grant_once(self, tmp_path):
        mine = make_grant()
        _issue_grant(mine)
        ledgers = tmp_path / "write_ledger"
        own = make_grant(approver=Approver(by=ME, surface="write_switch", approved_at=T0))
        planted = make_grant(approver=Approver(by=ME, surface="chip", approved_at=T0))
        _issue_grant(own, directory=ledgers)
        _issue_grant(planted, directory=ledgers, peers=())
        broken = ledgers / f"{uuid.uuid4().hex}.json"
        broken.write_text("{")
        listing = list_grants(own_dir(), own_dir(), self_issued=[ledgers, own_dir()])
        assert [(listed.grant.id, listed.directory) for listed in listing.items] == [
            (mine.id, own_dir()),
            (own.id, ledgers),
        ]
        assert listing.unreadable == (broken,)
