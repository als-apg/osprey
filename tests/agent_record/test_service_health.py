"""The agent-record service's health report and on-demand export.

Pins ``/health`` unchanged (open, ``{status, jobs_ready}`` only);
``/health/sources`` readable by every authenticated login and reporting per
source ``live``, ``stopped`` and ``idle`` as the contract defines them, the
stored refusals of the last 24 hours (a refusal from an earlier run of the
service shown after a restart, an old one aged out), ``spool_errors`` since
start, and the ``export`` block (``lag_days`` 0 on an empty record, counting
unmarked closed days, cleared by an export); and ``POST /jobs/export`` run by
the host and operator logins only, a terminal login refused, a failure kept
as the jobs' ``last_error``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey.agent_record import archive, service
from osprey.agent_record import contract as c
from osprey.agent_record.backends.embedded import EmbeddedBackend
from tests.agent_record import test_service as base
from tests.agent_record.test_service import _auth, _Down, _eventually, _Jobs, _jobs_ready, _serving
from tests.agent_record.test_spool_consumer import _line, _sealed

# The service tests' fixtures, shared under their own names.
config = base.config
env = base.env
backend = base.backend
_fresh_writer = base._fresh_writer

HOST = "builder"  # the host login of the test config
LATER = timedelta(minutes=6)  # past STOPPED_AFTER


class _Clock:
    """Moves the service's health clock forward by *ahead*."""

    def __init__(self, monkeypatch, ahead: timedelta) -> None:
        real = datetime

        class _Later(real):
            @classmethod
            def now(cls, tz=None):
                return real.now(tz) + ahead

        monkeypatch.setattr(service, "datetime", _Later)


@pytest.fixture
def archive_env(env, tmp_path) -> dict[str, str]:
    root = tmp_path / "archive"
    root.mkdir()
    os.chmod(root, 0o700)
    env[service.ARCHIVE_DIR_ENV] = str(root)
    return env


async def _sources(client, login: str = "operator") -> dict:
    response = await client.get("/health/sources", headers=_auth(login))
    assert response.status_code == 200, response.text
    return response.json()


def _forged_run() -> c.SpoolLine:
    """A run line from a terminal: refused, and stored as a refusal."""
    return _line(
        c.KIND_RUN, {"phase": c.RUN_ACCEPT, "requester": "mallory"}, run_id="r1", session_id="s1"
    )


# --- /health stays as it was -----------------------------------------------------------


async def test_health_is_open_and_unchanged(config, env, backend, monkeypatch):
    _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=backend, env=env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "jobs_ready": True}
        assert (await client.get("/health/sources")).status_code == 401
        assert (await client.post(service.EXPORT_PATH)).status_code == 401


# --- who may read and export -----------------------------------------------------------


async def test_a_terminal_login_reads_the_report_and_is_refused_the_export(
    config, archive_env, backend, monkeypatch
):
    jobs = _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=backend, env=archive_env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))
        for login in ("alice", "history", "reader.auditor", "dispatch-worker-1", HOST):
            body = await _sources(client, login)
            assert {"backend", "spool", "transcripts", "export", "jobs_clock"} <= set(body)
        refused = await client.post(service.EXPORT_PATH, headers=_auth("alice"))
        assert refused.status_code == 403
        assert refused.json()["status"] == "refused"
        for login in ("history", "dispatch-worker-1", "teams-bridge"):
            response = await client.post(service.EXPORT_PATH, headers=_auth(login))
            assert response.status_code == 403, login
        assert len(jobs.exports) == 1, "a refused export runs nothing"
        for login in ("operator", HOST):
            response = await client.post(service.EXPORT_PATH, headers=_auth(login))
            assert response.status_code == 200, login
            assert response.json() == {"status": "ok", "exported": []}
    assert len(jobs.exports) == 3


async def test_the_export_runs_on_the_jobs_clock(config, archive_env, backend, monkeypatch):
    archive_env[service.JOBS_CLOCK_ENV] = "2030-01-02"
    jobs = _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=backend, env=archive_env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))
        await client.post(service.EXPORT_PATH, headers=_auth("operator"))
        body = await _sources(client)
    fixed = datetime(2030, 1, 2, tzinfo=UTC)
    assert [now for _r, now, _root in jobs.exports] == [fixed, fixed]
    assert body["jobs_clock"] == fixed.isoformat()


async def test_a_failed_export_answers_why_and_is_kept_as_last_error(
    config, archive_env, backend, monkeypatch
):
    _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=backend, env=archive_env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))

        def down(record, now, root):  # noqa: ARG001 - run_export signature
            raise c.RecordBackendError("the record is down")

        monkeypatch.setattr(archive, "run_export", down)
        response = await client.post(service.EXPORT_PATH, headers=_auth("operator"))
        assert response.status_code == 503
        assert response.json() == {
            "status": "unavailable",
            "reason": "RecordBackendError: the record is down",
        }
        body = await _sources(client)
        assert body["export"]["last_error"] == "RecordBackendError: the record is down"

        def full(record, now, root):  # noqa: ARG001 - run_export signature
            raise OSError("archive full")

        monkeypatch.setattr(archive, "run_export", full)
        response = await client.post(service.EXPORT_PATH, headers=_auth("operator"))
        assert response.status_code == 500
        assert response.json()["status"] == "error"
        assert (await _sources(client))["export"]["last_error"] == "OSError: archive full"


# --- the export block ------------------------------------------------------------------


async def test_lag_days_is_zero_on_an_empty_record(config, archive_env, backend, monkeypatch):
    _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=backend, env=archive_env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))
        body = await _sources(client)
    assert body["export"] == {"last_exported_day": None, "lag_days": 0, "last_error": None}
    assert body["jobs_clock"] is None


async def test_lag_days_counts_unmarked_closed_days_until_an_export(
    config, archive_env, backend, monkeypatch
):
    today = datetime.now(UTC).date()
    backend.append(
        [
            c.Entry(
                key=c.spool_key("alice", f"lag{back}"),
                kind=c.KIND_FEEDBACK,
                ts=datetime.now(UTC) - timedelta(days=back),
                identity="alice",
                body={"n": back},
                inserted_at=datetime.now(UTC) - timedelta(days=back),
            )
            for back in (3, 1, 0)
        ]
    )
    real = archive.run_export
    calls: list[datetime] = []

    def first_skipped(record, now, root):
        calls.append(now)
        return [] if len(calls) == 1 else real(record, now, root)

    monkeypatch.setattr(archive, "run_export", first_skipped)
    app = service.create_app(config=config, backend=backend, env=archive_env)
    async with _serving(app) as client:
        await _eventually(lambda: _jobs_ready(client))
        before = (await _sources(client))["export"]
        assert before == {"last_exported_day": None, "lag_days": 3, "last_error": None}
        response = await client.post(service.EXPORT_PATH, headers=_auth(HOST))
        assert response.status_code == 200
        expected = [(today - timedelta(days=back)).isoformat() for back in (3, 2, 1)]
        assert response.json() == {"status": "ok", "exported": expected}
        after = (await _sources(client))["export"]
    yesterday = (today - timedelta(days=1)).isoformat()
    assert after == {"last_exported_day": yesterday, "lag_days": 0, "last_error": None}


async def test_an_unreachable_record_has_no_lag_and_the_report_still_answers(
    config, archive_env, monkeypatch
):
    _Jobs(monkeypatch)
    app = service.create_app(config=config, backend=_Down(), env=archive_env)
    async with _serving(app) as client:
        body = await _sources(client)
    assert body["backend"]["ok"] is False
    assert body["export"]["lag_days"] is None


# --- refusals --------------------------------------------------------------------------


async def test_a_refusal_from_an_earlier_session_shows_after_a_restart_and_ages_out(
    config, env, tmp_path, monkeypatch
):
    _Jobs(monkeypatch)
    spool = tmp_path / "spool"
    forged = _forged_run()
    _sealed(spool / HOST, forged)
    directory = str(spool / HOST)
    path = tmp_path / "kept.duckdb"

    with EmbeddedBackend(path) as first_backend:
        app = service.create_app(config=config, backend=first_backend, env=env)
        async with _serving(app) as client:

            async def refused():
                body = await _sources(client)
                source = body["spool"]["sources"].get(directory)
                return source if source and source["spool_errors"] else None

            source = await _eventually(refused)
            assert source["refusals"] == 1
            assert source["refused"][0]["entry_id"] == forged.entry_id

    with EmbeddedBackend(path) as second_backend:
        stored = second_backend.get_state(c.refused_state_name(HOST, forged.entry_id))
        assert stored is not None and stored[c.STATE_TIME_FIELD]
        app = service.create_app(config=config, backend=second_backend, env=env)
        async with _serving(app) as client:

            async def listed():
                return (await _sources(client))["spool"]["sources"].get(directory)

            source = await _eventually(listed)
            assert source["refusals"] == 1, "the earlier run's refusal is reported"
            assert source["spool_errors"] == 0, "this run refused nothing yet"
            assert source["refused"] == []

            old = datetime.now(UTC) - timedelta(hours=25)
            second_backend.put_state(
                c.refused_state_name(HOST, forged.entry_id),
                c.refused_state_value(c.KIND_RUN, "terminal", old),
            )
            second_backend.put_state(
                c.refused_state_name(HOST, "e-older"),
                c.refused_state_value(c.KIND_RUN, "terminal", old),
            )
            source = await listed()
            assert source["refusals"] == 0, "a refusal older than 24 hours is aged out"
            body = await _sources(client)
            assert body["spool"]["spool_errors_since"] is not None


async def test_refusals_are_unknown_not_zero_when_the_record_cannot_count(
    config, env, tmp_path, monkeypatch
):
    _Jobs(monkeypatch)
    spool = tmp_path / "spool"
    _sealed(spool / "alice", _line())
    app = service.create_app(config=config, backend=_Down(), env=env)
    async with _serving(app) as client:

        async def listed():
            return (await _sources(client))["spool"]["sources"].get(str(spool / "alice"))

        source = await _eventually(listed)
    assert source["refusals"] is None


# --- live, stopped, idle ---------------------------------------------------------------


def _open_segment(directory: Path, *lines: c.SpoolLine) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / c.SegmentName("h", 7, 1, sealed=False).name
    path.write_bytes(b"".join((line.to_json() + "\n").encode() for line in lines))
    return path


async def test_a_spool_source_is_idle_live_then_stopped(
    config, env, backend, tmp_path, monkeypatch
):
    _Jobs(monkeypatch)
    spool = tmp_path / "spool"
    (spool / "bob").mkdir(parents=True)
    _open_segment(spool / "alice", _line())
    app = service.create_app(config=config, backend=backend, env=env)
    async with _serving(app) as client:

        async def committed():
            sources = (await _sources(client))["spool"]["sources"]
            alice = sources.get(str(spool / "alice"))
            return sources if alice and alice["last_commit"] else None

        sources = await _eventually(committed)
        alice, bob = sources[str(spool / "alice")], sources[str(spool / "bob")]
        assert (alice["live"], alice["stopped"], alice["idle"]) == (True, False, False)
        assert (bob["live"], bob["stopped"], bob["idle"]) == (False, False, True)

        _Clock(monkeypatch, LATER)
        sources = (await _sources(client))["spool"]["sources"]
        alice, bob = sources[str(spool / "alice")], sources[str(spool / "bob")]
        assert alice["live"] and alice["stopped"], "live with no commit within 5 minutes"
        assert bob["idle"] and not bob["stopped"], "a source with no live session is idle"


async def test_a_backlog_with_no_commit_is_stopped(config, env, tmp_path, monkeypatch):
    _Jobs(monkeypatch)
    spool = tmp_path / "spool"
    _sealed(spool / "alice", _line(), _line())
    app = service.create_app(config=config, backend=_Down(), env=env)
    async with _serving(app) as client:

        async def backlog():
            source = (await _sources(client))["spool"]["sources"].get(str(spool / "alice"))
            return source if source and source["backlog_lines"] else None

        source = await _eventually(backlog)
        assert not source["live"] and not source["stopped"], "measured from the service start"
        _Clock(monkeypatch, LATER)
        source = (await _sources(client))["spool"]["sources"][str(spool / "alice")]
        assert source["stopped"] and not source["idle"]


async def test_a_transcript_root_is_live_while_written_and_idle_after(
    config, env, backend, tmp_path, monkeypatch
):
    _Jobs(monkeypatch)
    root = "alice-claude-projects"
    transcript = tmp_path / "transcripts" / root / "-data-workspace" / "sess-1.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_bytes(b'{"partial": ')  # never a complete line, so never committed
    app = service.create_app(config=config, backend=backend, env=env)
    async with _serving(app) as client:

        async def seen():
            row = (await _sources(client))["transcripts"][root]
            return row if row["last_modified"] else None

        row = await _eventually(seen)
        assert (row["live"], row["stopped"], row["idle"]) == (True, False, False)
        assert row["last_seen"] is None

        _Clock(monkeypatch, LATER)
        row = (await _sources(client))["transcripts"][root]
        assert (row["live"], row["stopped"], row["idle"]) == (False, False, True)

        ahead = (datetime.now(UTC) + LATER).timestamp()
        os.utime(transcript, (ahead, ahead))

        async def modified_again():
            row = (await _sources(client))["transcripts"][root]
            return row if row["live"] else None

        row = await _eventually(modified_again)
        assert row["stopped"], "written within 5 minutes, nothing committed for longer"
