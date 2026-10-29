"""Tests for ``osprey agent-record health`` and ``osprey agent-record export``.

Both verbs open the record the way every host command does, wait for its
start-up job run, then answer. Three layers are exercised:

* **A real host-local record.** A hello-world-shaped repo with no stack and no
  host session: each verb starts the record itself, answers from it, and
  leaves no process behind. A ClickHouse repo with nothing up is refused with
  the fix.
* **A fake record client** for how each answer is rendered and how each
  failure is reported, without a service.
* **``osprey init --reset``** meeting a running host-local record: a named
  refusal, never a traceback.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import click
import httpx
import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record import writer
from osprey.agent_record.client import URL_ENV, ClientAnswer
from osprey.agent_record.contract import KIND_AUDIT
from osprey.agent_record.service import JOBS_CLOCK_ENV
from osprey.cli import agent_record_cmd
from osprey.cli.agent_record_cmd import agent_record_group
from osprey_connectors_tests._harness._ports import reserved_port

HOST = "builder"


#: The port this test's record would bind, reserved for the test by
#: :func:`_record_port`: the config names it before any record starts.
_RECORD_PORT: list[int] = []


@pytest.fixture(autouse=True)
def _record_port() -> Iterator[None]:
    with reserved_port() as port:
        _RECORD_PORT[:] = [port]
        yield


def _config(root: Path, port: int, backend: str = "embedded") -> dict[str, Any]:
    return {
        "project_name": "arcli",
        "project_root": str(root),
        "build_dir": "./build",
        "deployment": {"host_identity": HOST},
        "agent_record": {"port": port, "backend": backend},
    }


def _make_repo(tmp_path: Path, *, backend: str = "embedded") -> tuple[Path, dict[str, Any]]:
    """A deployment repo: a profile at the root, a rendered build, minted logins."""
    root = (tmp_path / "deployment").resolve()
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")
    config = _config(root, _RECORD_PORT[0], backend)
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    creds.ensure_agent_record_credentials(root, config, reader_keys=False)
    # The archive root every build provisions (`ensure_archive_dir`).
    (root / "var" / "archive").mkdir(parents=True)
    return root, config


@pytest.fixture
def no_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Nothing runs: no record URL in the environment, a Claude dir of the test's own."""
    claude = tmp_path / "claude"
    claude.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.delenv(JOBS_CLOCK_ENV, raising=False)
    yield


def _no_process_left(root: Path, config: dict[str, Any]) -> None:
    state_dir = hr.host_state_dir(root, config)
    try:
        assert hr._running_process(state_dir) is None, "an agent-record process was left"
        port = config["agent_record"]["port"]
        with pytest.raises(httpx.ConnectError):
            httpx.get(f"http://127.0.0.1:{port}/health", trust_env=False, timeout=2)
    finally:
        if state_dir.exists():
            hr.stop_host_process(state_dir, wait_s=10)


# ---------------------------------------------------------------------------
# A real host-local record, no stack up
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_stack")
def test_health_on_a_no_stack_project_answers_from_the_record_and_leaves_no_process(
    tmp_path: Path,
) -> None:
    root, config = _make_repo(tmp_path)

    result = CliRunner().invoke(agent_record_group, ["health", "--repo", str(root)])

    assert result.exit_code == 0, result.output
    assert f"Agent record health in {root}" in result.output
    assert "Backend: " in result.output and ", ok, " in result.output
    assert "Spool: " in result.output
    assert "Export: latest exported day" in result.output
    _no_process_left(root, config)


@pytest.mark.usefixtures("no_stack")
def test_export_on_a_no_stack_project_runs_catch_up_and_leaves_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the jobs' clock a day ahead, today's arrivals are a closed day the
    catch-up exports; the verb names it as the latest exported day."""
    root, config = _make_repo(tmp_path)
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(root / "var" / "spool" / "alice"))
    assert writer.write(KIND_AUDIT, {"surface": "test", "note": "a step"})
    monkeypatch.delenv("OSPREY_SPOOL_DIR")
    today = dt.datetime.now(dt.UTC).date()
    monkeypatch.setenv(JOBS_CLOCK_ENV, (today + dt.timedelta(days=1)).isoformat())

    result = CliRunner().invoke(agent_record_group, ["export", "--repo", str(root)])

    assert result.exit_code == 0, result.output
    assert f"Latest exported day: {today.isoformat()}" in result.output
    assert "Exported " in result.output
    _no_process_left(root, config)


@pytest.mark.usefixtures("no_stack")
@pytest.mark.parametrize("verb", ["health", "export"])
def test_a_clickhouse_project_with_no_stack_is_refused_naming_the_fix(
    tmp_path: Path, verb: str
) -> None:
    root, config = _make_repo(tmp_path, backend="clickhouse")

    result = CliRunner().invoke(agent_record_group, [verb, "--repo", str(root)])

    assert result.exit_code == 1, result.output
    assert "The agent record is not available" in result.output
    assert "osprey up" in result.output
    assert "Traceback" not in result.output
    _no_process_left(root, config)


@pytest.mark.parametrize("verb", ["health", "export"])
def test_a_repo_with_no_build_is_refused(tmp_path: Path, verb: str) -> None:
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")

    result = CliRunner().invoke(agent_record_group, [verb, "--repo", str(root)])

    assert result.exit_code == 1, result.output
    assert "No build found" in result.output
    assert "osprey build" in result.output


# ---------------------------------------------------------------------------
# A fake record client
# ---------------------------------------------------------------------------


def _source(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "identity": "alice",
        "directory": "/r/var/spool/alice",
        "backlog_lines": 0,
        "oldest_backlog_age_s": None,
        "abandoned": [],
        "writer_errors": [],
        "spool_errors": 0,
        "refusals": 0,
        "live": True,
        "stopped": False,
        "idle": False,
    }
    row.update(overrides)
    return row


def _report(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "backend": {
            "ok": True,
            "backend": "embedded",
            "detail": "",
            "entries": 42,
            "oldest_arrival_day": "2026-10-20",
            "newest_arrival_day": "2026-10-26",
        },
        "spool": {
            "backend_error": None,
            "backlog_lines": 0,
            "sources": {"var/spool/alice": _source()},
            "quarantined": [],
            "problems": [],
        },
        "transcripts": {
            "claude/projects": {
                "identity": HOST,
                "error": None,
                "file_errors": {},
                "files": 3,
                "last_seen": "2026-10-26T10:00:00+00:00",
                "live": False,
                "stopped": False,
                "idle": True,
            }
        },
        "export": {"last_exported_day": "2026-10-25", "lag_days": 0, "last_error": None},
        "jobs_clock": None,
    }
    body.update(overrides)
    return body


class _FakeClient:
    url = "http://127.0.0.1:1"

    def __init__(self) -> None:
        self.sources = ClientAnswer(data=_report(), complete=True)
        self.exported = ClientAnswer(data=["2026-10-24", "2026-10-25"], complete=True)
        self.export_timeouts: list[float | None] = []

    def health_sources(self) -> ClientAnswer:
        return self.sources

    def export(self, *, timeout: float | None = None) -> ClientAnswer:
        self.export_timeouts.append(timeout)
        return self.exported


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_FakeClient, Path]]:
    root, _ = _make_repo(tmp_path)
    client = _FakeClient()
    waited: list[str] = []

    @contextmanager
    def _open(_deployment: Any, environ: Any = None) -> Iterator[_FakeClient]:  # noqa: ARG001
        yield client

    monkeypatch.setattr(agent_record_cmd, "_open_record", _open)
    monkeypatch.setattr(agent_record_cmd, "_wait_for_jobs", lambda url, **_kw: waited.append(url))
    client.waited = waited  # type: ignore[attr-defined]
    yield client, root


def _invoke(root: Path, verb: str) -> Any:
    return CliRunner().invoke(agent_record_group, [verb, "--repo", str(root)])


def test_health_prints_every_signal_and_says_no_problems_when_clean(fake) -> None:
    client, root = fake

    result = _invoke(root, "health")

    assert result.exit_code == 0, result.output
    assert client.waited == [client.url], "the start-up job run was not waited on"
    out = result.output
    assert "Backend: embedded, ok, 42 entries, arrival days 2026-10-20 to 2026-10-26" in out
    assert "alice  var/spool/alice: live, backlog 0" in out
    assert f"{HOST}  claude/projects: idle, 3 files" in out
    assert "Export: latest exported day 2026-10-25, closed days not exported 0" in out
    assert "No problems found." in out


def test_health_names_a_stopped_source_and_an_export_lag(fake) -> None:
    client, root = fake
    body = _report()
    body["spool"]["sources"]["var/spool/alice"] = _source(
        live=False,
        stopped=True,
        backlog_lines=12,
        oldest_backlog_age_s=900.0,
        refusals=2,
        spool_errors=1,
        writer_errors=["disk full"],
        abandoned=["seg-1"],
    )
    body["export"] = {"last_exported_day": "2026-10-20", "lag_days": 5, "last_error": "boom"}
    client.sources = ClientAnswer(data=body, complete=True)

    result = _invoke(root, "health")

    assert result.exit_code == 0, result.output
    flowed = " ".join(result.output.split())
    assert "alice var/spool/alice: stopped, backlog 12, oldest 900 s, refusals (24 h) 2" in flowed
    assert "spool source alice has stopped" in flowed
    assert "2 lines refused in the last 24 h" in flowed
    assert "1 lines refused since start-up" in flowed
    assert "1 writer errors" in flowed
    assert "1 abandoned writers" in flowed
    assert "the export is 5 days behind; run `osprey agent-record export`" in flowed
    assert "the last export failed: boom" in flowed
    assert "7 problem(s) found." in flowed


def test_health_names_an_unhealthy_backend(fake) -> None:
    client, root = fake
    client.sources = ClientAnswer(
        data=_report(backend={"ok": False, "backend": "clickhouse", "detail": "refused"}),
        complete=True,
    )

    result = _invoke(root, "health")

    assert "Backend: clickhouse, not healthy: refused" in result.output
    assert "the clickhouse backend is not healthy: refused" in " ".join(result.output.split())


def test_health_fails_when_the_record_does_not_answer(fake) -> None:
    client, root = fake
    client.sources = ClientAnswer.failed("agent-record is unreachable: ConnectError")

    result = _invoke(root, "health")

    assert result.exit_code == 1
    assert "The agent record did not answer" in result.output
    assert "ConnectError" in result.output


def test_export_prints_the_days_it_exported_and_the_latest_day(fake) -> None:
    client, root = fake

    result = _invoke(root, "export")

    assert result.exit_code == 0, result.output
    assert client.waited == [client.url], "the start-up job run was not waited on"
    assert client.export_timeouts == [agent_record_cmd.EXPORT_TIMEOUT_S]
    assert "Exported 2 day(s): 2026-10-24, 2026-10-25" in result.output
    assert "Latest exported day: 2026-10-25" in result.output


def test_export_with_nothing_to_do_says_so(fake) -> None:
    client, root = fake
    client.exported = ClientAnswer(data=[], complete=True)
    client.sources = ClientAnswer(
        data=_report(export={"last_exported_day": None, "lag_days": 0}), complete=True
    )

    result = _invoke(root, "export")

    assert result.exit_code == 0, result.output
    assert "Exported no days" in result.output
    assert "Latest exported day: none yet" in result.output


def test_export_fails_when_the_record_refuses_it(fake) -> None:
    client, root = fake
    client.exported = ClientAnswer.failed("agent-record answered 403: may not run the export")

    result = _invoke(root, "export")

    assert result.exit_code == 1
    assert "The export did not run" in result.output
    assert "403" in result.output


# ---------------------------------------------------------------------------
# Opening the record and waiting for its jobs
# ---------------------------------------------------------------------------


def test_a_set_record_url_is_read_with_the_operator_login_without_a_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_repo(tmp_path)

    def _never(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("a set record URL must not take a host lease")

    monkeypatch.setattr(hr, "ensure_host_record", _never)
    deployment = agent_record_cmd._Deployment(root, config)
    env = {URL_ENV: "http://agent-record:4318"}
    with agent_record_cmd._open_record(deployment, env) as client:
        assert client.url == "http://agent-record:4318"
        assert client.login == "operator"


def test_the_host_lease_is_read_with_the_host_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host login is an exporter; the lease is released on the way out."""
    root, config = _make_repo(tmp_path)
    released: list[bool] = []

    @contextmanager
    def _lease(*_a: Any, **_kw: Any) -> Iterator[hr.HostRecord]:
        yield hr.HostRecord("http://127.0.0.1:9", "host-builder", "tok")
        released.append(True)

    monkeypatch.setattr(hr, "ensure_host_record", _lease)
    deployment = agent_record_cmd._Deployment(root, config)
    with agent_record_cmd._open_record(deployment, {}) as client:
        assert (client.url, client.login) == ("http://127.0.0.1:9", "host-builder")
    assert released == [True]


def test_wait_for_jobs_returns_once_the_start_up_run_finished() -> None:
    answers = iter([(False, "not yet"), (False, "not yet"), (True, "")])
    calls: list[str] = []

    def _probe(url: str) -> tuple[bool, str]:
        calls.append(url)
        return next(answers)

    agent_record_cmd._wait_for_jobs("http://r", wait_s=10, poll_s=0, probe=_probe)

    assert calls == ["http://r"] * 3


def test_wait_for_jobs_refuses_naming_why_when_the_run_never_finishes(capsys) -> None:
    with pytest.raises(click.exceptions.Exit):
        agent_record_cmd._wait_for_jobs(
            "http://r", wait_s=0, poll_s=0, probe=lambda _url: (False, "still running")
        )

    err = capsys.readouterr().err
    assert "The agent record at http://r is not ready" in err
    assert "still running" in err


@pytest.mark.parametrize(
    ("status", "body", "ready", "why"),
    [
        (200, {"status": "ok", "jobs_ready": True}, True, ""),
        (200, {"status": "ok", "ready": True, "jobs_ready": False}, False, "has not finished"),
        (503, {"status": "unconfigured", "reason": "no verifier"}, False, "503: no verifier"),
    ],
)
def test_jobs_state_reads_jobs_ready_from_health(
    monkeypatch: pytest.MonkeyPatch, status: int, body: dict, ready: bool, why: str
) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_a, **_kw: httpx.Response(status, json=body))

    got_ready, got_why = agent_record_cmd._jobs_state("http://r")

    assert got_ready is ready
    assert why in got_why


# ---------------------------------------------------------------------------
# osprey init --reset meeting a running host-local record
# ---------------------------------------------------------------------------


def test_init_reset_refuses_by_name_while_a_host_record_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.cli.main import cli
    from osprey.deployment.reset import HostRecordRunningError, RecordHolder

    error = HostRecordRunningError([RecordHolder(4242, "osprey chat")], None)

    def _raise(repo_root: Any, **_kw: Any) -> Any:  # noqa: ARG001
        raise error

    monkeypatch.setattr("osprey.deployment.reset.reset_for_reinit", _raise)
    # `--reset` is refused up front without a runtime; answered here so the
    # refusal under test is the record's, whatever the host runs.
    monkeypatch.setattr(
        "osprey.deployment.runtime_helper.verify_runtime_is_running",
        lambda config=None: (True, ""),
    )

    result = CliRunner().invoke(
        cli, ["init", str(tmp_path / "demo"), "--preset", "hello-world", "--no-git", "--reset"]
    )

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"the refusal escaped as an exception: {result.exception!r}"
    )
    assert "Traceback" not in result.output
    flowed = " ".join(result.output.split())
    assert HostRecordRunningError.SUMMARY in flowed
    assert "pid 4242 osprey chat" in flowed
    assert "osprey web stop" in flowed
    assert "demo/ was created. Nothing was built and nothing was started." in flowed
