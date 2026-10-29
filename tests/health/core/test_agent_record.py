"""Tests for the core ``agent_record`` health category.

The row reads the record's ``GET /health/sources`` report through
:class:`~osprey.agent_record.client.RecordClient` under whatever login the
caller's environment carries. Each condition the operator must be told about —
a stopped source, a refusal, a growing backlog, an export more than one day
behind or failing — is pinned on a report body handed in through the
``client`` seam; idle sources show idle; with no record answering the row says
so as information and points at ``osprey agent-record health``. One test runs a
real record service and renders the row as a terminal login, from the
environment a terminal session carries.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import URL_ENV, ClientAnswer
from osprey.health.core.agent_record import CATEGORY, agent_record
from osprey.health.models import CheckResult, Status
from osprey.interfaces._serving import run_app_server
from osprey_connectors_tests._harness._ports import reserved_port
from tests.agent_record.test_identities import _config

HEALTH_COMMAND = "osprey agent-record health"


class _Client:
    """A record client answering one fixed ``health_sources`` report."""

    def __init__(self, answer: ClientAnswer) -> None:
        self.answer = answer
        self.calls = 0

    def health_sources(self) -> ClientAnswer:
        self.calls += 1
        return self.answer


def _source(identity: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "identity": identity,
        "directory": f"/spool/{identity}",
        "receiver_owned": False,
        "backlog_lines": 0,
        "oldest_backlog_age_s": None,
        "abandoned": [],
        "writer_errors": [],
        "refused": [],
        "spool_errors": 0,
        "refusals": 0,
        "last_seen": "2026-10-26T10:00:00+00:00",
        "last_commit": "2026-10-26T10:00:00+00:00",
        "live": False,
        "stopped": False,
        "idle": True,
    }
    row.update(over)
    return row


def _transcript(identity: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "identity": identity,
        "ok": True,
        "error": "",
        "file_errors": {},
        "last_polled": "2026-10-26T10:00:00+00:00",
        "last_seen": None,
        "last_modified": None,
        "files": 0,
        "appended": 0,
        "duplicates": 0,
        "skipped_retention": 0,
        "live": False,
        "stopped": False,
        "idle": True,
    }
    row.update(over)
    return row


def _report(
    *,
    sources: dict[str, dict[str, Any]] | None = None,
    transcripts: dict[str, dict[str, Any]] | None = None,
    export: dict[str, Any] | None = None,
    backend: dict[str, Any] | None = None,
    backend_error: str = "",
) -> dict[str, Any]:
    sources = sources or {}
    return {
        "backend": {
            "ok": True,
            "backend": "embedded",
            "detail": "",
            "entries": 12,
            "oldest_arrival_day": "2026-10-20",
            "newest_arrival_day": "2026-10-26",
            **(backend or {}),
        },
        "spool": {
            "backend_error": backend_error,
            "backlog_lines": sum(s["backlog_lines"] for s in sources.values()),
            "spool_errors_since": "2026-10-26T09:00:00+00:00",
            "sources": sources,
            "quarantined": [],
            "problems": [],
        },
        "transcripts": transcripts or {},
        "export": {
            "last_exported_day": "2026-10-25",
            "lag_days": 0,
            "last_error": None,
            **(export or {}),
        },
        "jobs_clock": None,
    }


def _rows(report: dict[str, Any]) -> list[CheckResult]:
    client = _Client(ClientAnswer(data=report, complete=True))
    rows = agent_record(None, client=client)()
    assert client.calls == 1
    assert all(r.category == CATEGORY for r in rows)
    return rows


def _row(rows: list[CheckResult], fragment: str) -> CheckResult:
    found = [r for r in rows if fragment in r.name]
    assert len(found) == 1, [r.name for r in rows]
    return found[0]


def _problems(rows: list[CheckResult]) -> list[CheckResult]:
    return [r for r in rows if r.status is not Status.OK]


# --------------------------------------------------------------------------- #
# A healthy record, and idle sources
# --------------------------------------------------------------------------- #


def test_category_name():
    assert CATEGORY == "agent_record"


def test_a_healthy_record_reports_every_part_ok():
    rows = _rows(
        _report(
            sources={"/spool/alice": _source("alice")},
            transcripts={"alice-claude-projects": _transcript("alice")},
        )
    )
    assert _problems(rows) == []
    names = [r.name for r in rows]
    assert "agent_record_backend" in names and "agent_record_export" in names
    assert all(r.probed for r in rows)


def test_idle_sources_show_idle_not_stopped():
    rows = _rows(
        _report(
            sources={"/spool/alice": _source("alice", idle=True)},
            transcripts={"bob-claude-projects": _transcript("bob", idle=True)},
        )
    )
    spool = _row(rows, "alice")
    transcript = _row(rows, "bob")
    for row in (spool, transcript):
        assert row.status is Status.OK
        assert "idle" in row.message
        assert "stopped" not in row.message


def test_a_source_that_is_receiving_is_ok_and_not_idle():
    rows = _rows(
        _report(
            sources={
                "/spool/alice": _source(
                    "alice", live=True, idle=False, backlog_lines=3, oldest_backlog_age_s=1.5
                )
            }
        )
    )
    row = _row(rows, "alice")
    assert row.status is Status.OK
    assert "idle" not in row.message


# --------------------------------------------------------------------------- #
# The conditions the operator is told about, each naming its source
# --------------------------------------------------------------------------- #


def test_a_stopped_source_is_named():
    rows = _rows(
        _report(
            sources={
                "/spool/alice": _source("alice"),
                "/spool/worker-1": _source(
                    "worker-1",
                    live=True,
                    idle=False,
                    stopped=True,
                    backlog_lines=40,
                    oldest_backlog_age_s=900.0,
                ),
            }
        )
    )
    row = _row(rows, "worker-1")
    assert row.status is Status.WARNING
    assert "worker-1" in row.message and "stopped" in row.message
    assert "40" in row.message
    assert _row(rows, "alice").status is Status.OK


def test_a_stopped_transcript_root_is_named():
    rows = _rows(
        _report(transcripts={"bob-claude-projects": _transcript("bob", stopped=True, idle=False)})
    )
    row = _row(rows, "bob")
    assert row.status is Status.WARNING
    assert "bob" in row.message and "stopped" in row.message


def test_a_transcript_root_that_cannot_be_read_is_named():
    rows = _rows(
        _report(
            transcripts={
                "bob-claude-projects": _transcript("bob", ok=False, error="permission denied")
            }
        )
    )
    row = _row(rows, "bob")
    assert row.status is Status.WARNING
    assert "permission denied" in row.message


def test_a_refusal_is_named():
    refused = [{"segment": "s1", "entry_id": "e1", "kind": "audit", "reason": "unknown kind"}]
    rows = _rows(
        _report(
            sources={
                "/spool/alice": _source(
                    "alice", refusals=2, spool_errors=2, refused=refused, idle=True
                )
            }
        )
    )
    row = _row(rows, "alice")
    assert row.status is Status.WARNING
    assert "alice" in row.message and "refused 2" in row.message
    assert "unknown kind" in row.details


def test_a_refusal_seen_this_run_but_not_yet_counted_is_named():
    rows = _rows(_report(sources={"/spool/alice": _source("alice", refusals=0, spool_errors=1)}))
    row = _row(rows, "alice")
    assert row.status is Status.WARNING and "refused 1" in row.message


def test_an_unreadable_refusal_count_is_not_reported_as_zero():
    rows = _rows(_report(sources={"/spool/alice": _source("alice", refusals=None)}))
    row = _row(rows, "alice")
    assert row.status is Status.WARNING and "unknown" in row.message


def test_writer_errors_are_named():
    rows = _rows(_report(sources={"/spool/alice": _source("alice", writer_errors=["disk full"])}))
    row = _row(rows, "alice")
    assert row.status is Status.WARNING and "disk full" in row.details


def test_a_growing_backlog_is_named():
    rows = _rows(
        _report(
            sources={
                "/spool/worker-1": _source(
                    "worker-1",
                    live=True,
                    idle=False,
                    backlog_lines=500,
                    oldest_backlog_age_s=240.0,
                )
            }
        )
    )
    row = _row(rows, "worker-1")
    assert row.status is Status.WARNING
    assert "worker-1" in row.message and "backlog" in row.message and "500" in row.message
    assert "stopped" not in row.message


def test_a_short_backlog_being_drained_is_not_a_problem():
    rows = _rows(
        _report(
            sources={
                "/spool/worker-1": _source(
                    "worker-1", live=True, idle=False, backlog_lines=5, oldest_backlog_age_s=2.0
                )
            }
        )
    )
    assert _row(rows, "worker-1").status is Status.OK


def test_the_record_refusing_writes_is_named():
    rows = _rows(_report(backend_error="disk I/O error"))
    row = _row(rows, "backend")
    assert row.status is Status.WARNING and "disk I/O error" in row.message


def test_a_backend_that_is_not_ok_is_named():
    rows = _rows(_report(backend={"ok": False, "detail": "cannot open record.duckdb"}))
    row = _row(rows, "backend")
    assert row.status is Status.WARNING and "cannot open record.duckdb" in row.message


def test_quarantined_spool_paths_are_named():
    report = _report()
    report["spool"]["quarantined"] = [["/spool/../evil", "/spool/.quarantine/evil"]]
    report["spool"]["problems"] = ["/spool/odd: not a directory"]
    row = _row(_rows(report), "agent_record_spool")
    assert row.status is Status.WARNING
    assert "/spool/../evil" in row.details and "not a directory" in row.details


# --------------------------------------------------------------------------- #
# The export
# --------------------------------------------------------------------------- #


def test_an_export_one_day_behind_is_normal():
    rows = _rows(_report(export={"lag_days": 1, "last_exported_day": "2026-10-24"}))
    assert _row(rows, "export").status is Status.OK


def test_an_export_more_than_one_day_behind_is_named():
    rows = _rows(_report(export={"lag_days": 3, "last_exported_day": "2026-10-22"}))
    row = _row(rows, "export")
    assert row.status is Status.WARNING
    assert "3" in row.message and "2026-10-22" in row.message
    assert "osprey agent-record export" in row.details


def test_a_failing_export_is_named():
    rows = _rows(_report(export={"last_error": "archive is read-only"}))
    row = _row(rows, "export")
    assert row.status is Status.WARNING and "archive is read-only" in row.message


def test_an_unreadable_archive_is_named():
    rows = _rows(_report(export={"archive_error": "the archive's marks cannot be read: EACCES"}))
    row = _row(rows, "export")
    assert row.status is Status.WARNING and "EACCES" in row.message


def test_an_export_lag_unknown_while_the_backend_is_down_is_not_ok():
    rows = _rows(_report(backend={"ok": False, "detail": "down"}, export={"lag_days": None}))
    assert _row(rows, "export").status is not Status.OK


def test_a_record_with_nothing_exported_yet_is_ok():
    rows = _rows(_report(export={"lag_days": 0, "last_exported_day": None}))
    row = _row(rows, "export")
    assert row.status is Status.OK


# --------------------------------------------------------------------------- #
# No record answers
# --------------------------------------------------------------------------- #


def _unanswered(rows: list[CheckResult]) -> CheckResult:
    assert len(rows) == 1
    (row,) = rows
    assert row.category == CATEGORY
    assert row.status not in (Status.ERROR, Status.WARNING)
    assert HEALTH_COMMAND in row.details
    assert not row.probed
    return row


def test_no_record_configured_says_so_as_information():
    """Hello-world with nothing running: plain ``osprey health`` on the host."""
    row = _unanswered(agent_record(None, environ={})())
    assert "no agent record" in row.message.lower()


def test_a_record_that_does_not_answer_says_so_as_information():
    with reserved_port() as port:  # reserved for the test, and never bound
        env = {URL_ENV: f"http://127.0.0.1:{port}", creds.LOGIN_ENV: "alice:token"}
        row = _unanswered(agent_record(None, environ=env)())
    assert "unreachable" in row.message


def test_an_unavailable_answer_from_the_client_says_so():
    client = _Client(ClientAnswer.failed("agent-record did not answer within 5 s"))
    row = _unanswered(agent_record(None, client=client)())
    assert "did not answer" in row.message


def test_a_report_without_the_expected_shape_is_not_a_crash():
    client = _Client(ClientAnswer(data={"backend": "odd"}, complete=True))
    rows = agent_record(None, client=client)()
    assert rows and all(r.category == CATEGORY for r in rows)


# --------------------------------------------------------------------------- #
# A real record, read as a terminal login
# --------------------------------------------------------------------------- #


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.fixture(scope="module")
def record_config(tmp_path_factory) -> dict:
    config = _config(workers=1)
    config["project_root"] = str(tmp_path_factory.mktemp("project"))
    return config


@pytest.fixture(scope="module")
def record_url(record_config, tmp_path_factory):
    config = record_config
    root = tmp_path_factory.mktemp("service")
    (root / "archive").mkdir(mode=0o700)
    # Every transcript root the deployment renders exists, as its volume would.
    for name in [*ids.transcript_roots(config), "host"]:
        (root / "transcripts" / name).mkdir(parents=True, exist_ok=True)
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    env = {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps(["s3cr3t-token-value-0123456789"])),
        service.SPOOL_ROOT_ENV: str(root / "spool"),
        service.RECORD_DIR_ENV: str(root / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(root / "transcripts"),
        service.ARCHIVE_DIR_ENV: str(root / "archive"),
    }
    with EmbeddedBackend(tmp_path_factory.mktemp("record") / "record.duckdb") as backend:
        app = service.create_app(config=config, backend=backend, env=env)
        with run_app_server(app) as base:
            yield base


def test_the_row_renders_from_a_running_record_as_a_terminal_login(record_config, record_url):
    """A terminal session's environment carries its own login; the row reads with it."""
    terminal = "alice"
    assert ids.identity_class(record_config, terminal) == ids.SOURCE_TERMINAL
    env = {URL_ENV: record_url, creds.LOGIN_ENV: f"{terminal}:{_token(terminal)}"}
    rows = agent_record(None, environ=env)()
    assert rows and all(r.category == CATEGORY for r in rows)
    assert all(r.probed for r in rows)
    assert _row(rows, "backend").status is Status.OK
    assert _row(rows, "export").status is Status.OK
    assert _problems(rows) == [], [(r.name, r.message) for r in _problems(rows)]


def test_a_wrong_terminal_token_is_not_a_healthy_row(record_url):
    env = {URL_ENV: record_url, creds.LOGIN_ENV: "alice:not-the-token"}
    row = _unanswered(agent_record(None, environ=env)())
    assert "401" in row.message
