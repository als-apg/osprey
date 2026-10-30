"""CLI group for the agent-record service.

Provides ``osprey agent-record serve``: the entry of the ``agent-record``
service every deployment runs. It takes everything it needs from the rendered
``config.yml`` and the environment the compose file sets (the verifier and the
secret set, the spool, record and transcript locations), and acts on no
deployment repo. On the host the same command runs as the host-local record,
started by :mod:`osprey.agent_record.host` with ``--lease-dir``.

It serves under umask ``0o002``, so the store and its WAL stay writable for
the directory's group: the stack's agent-record, another uid of that group,
takes the same files over.

``osprey agent-record health`` and ``osprey agent-record export`` are the
operator's host verbs. Each opens the record the way every host command does
-- the one ``OSPREY_AGENT_RECORD_URL`` names, else a one-shot lease on the
host-local record, which starts it when nothing answers and stops it again on
the way out -- waits for the record's start-up job run to finish, then
answers: ``health`` with the per-source, backend and export signals of
``GET /health/sources``, ``export`` with the days the catch-up export wrote
now and the latest day the archive holds. A record that cannot be had is
refused with its fix.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import click

from osprey.cli.output import fail, report, warn
from osprey.cli.record_access import Deployment as _Deployment
from osprey.cli.record_access import find_build, operator_login
from osprey.cli.repo_resolver import repo_option

#: How long a verb waits for the record's start-up job run to finish.
JOBS_WAIT_S = 120.0

#: How often the start-up job run is asked after while it is waited on.
JOBS_POLL_S = 0.25

#: How long the catch-up export may run before ``export`` stops waiting.
EXPORT_TIMEOUT_S = 600.0

#: A ``/health`` probe's own bound.
_PROBE_TIMEOUT_S = 5.0


@click.group("agent-record")
def agent_record_group() -> None:
    """The agent record: one store of every agent's transcript, audit and run facts."""


@agent_record_group.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to listen on (the container listens on every interface).",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Port to serve on.  [default: the configured agent_record.port]",
)
@click.option(
    "--lease-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Run as the host-local record: report ready only after the first full ingest "
        "pass, and stop once no lease in this directory names a live process."
    ),
)
def serve(host: str, port: int | None, lease_dir: Path | None) -> None:
    """Serve agent-record: OTLP ingest, the read API and source health.

    Ingests the transcript roots and the spool into the configured backend and
    answers reads for the credentials the deployment minted. It expects the
    environment the deployment renders for it: the verifier and the secret set,
    and the spool, record and transcript locations.

    Example:

    \b
        osprey agent-record serve --host 0.0.0.0
    """
    import os

    import uvicorn

    from osprey.agent_record.service import create_app

    os.umask(0o002)
    app = create_app(lease_dir=lease_dir)
    if port is None:
        port = app.state.settings.record.port
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# The host verbs: health and export
# ---------------------------------------------------------------------------


def _deployment(repo: Path | None) -> _Deployment:
    """The deployment repo and its rendered config; refused when there is no build."""
    from osprey_connectors.config import load_project_config

    build = find_build(repo, "The agent record is the one the rendered config describes.")
    if build is None:
        raise click.exceptions.Exit(1)
    repo_root, config_path = build
    return _Deployment(root=repo_root, config=load_project_config(str(config_path)))


@contextmanager
def _open_record(
    deployment: _Deployment, environ: Mapping[str, str] | None = None
) -> Iterator[Any]:
    """Yield a record client whose login may run the export, holding the record meanwhile.

    With ``OSPREY_AGENT_RECORD_URL`` set, that record is read with the operator
    login. Otherwise a one-shot :func:`~osprey.agent_record.host.ensure_host_record`
    lease is held for the block, read with the host login it hands back: a
    running stack or host session on the loopback port is reused, and when
    nothing runs the host-local process is started and, as the last holder,
    stopped again on the way out.
    """
    import os

    from osprey.agent_record.client import URL_ENV, RecordClient
    from osprey.agent_record.host import HostRecordRefused, ensure_host_record
    from osprey_connectors.workspace import rendered_config_path

    env = os.environ if environ is None else environ
    url = (env.get(URL_ENV) or "").strip()
    if url:
        operator = operator_login(deployment.root)
        if operator is None:
            raise click.exceptions.Exit(1)
        login, token = operator
        yield RecordClient(url, login, token)
        return
    with ExitStack() as stack:
        try:
            record = stack.enter_context(
                ensure_host_record(
                    deployment.root,
                    deployment.config,
                    config_path=rendered_config_path(deployment.root),
                )
            )
        except HostRecordRefused as exc:
            fail("The agent record is not available", str(exc))
            raise click.exceptions.Exit(1) from None
        yield RecordClient(record.url, record.login, record.token)


def _jobs_state(url: str) -> tuple[bool, str]:
    """``(ready, why not)`` of the record's start-up job run, from ``GET /health``."""
    import httpx

    try:
        response = httpx.get(f"{url}/health", timeout=_PROBE_TIMEOUT_S, trust_env=False)
    except httpx.HTTPError as exc:
        return False, f"agent-record is unreachable: {type(exc).__name__}: {exc}"
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return False, f"/health answered {response.status_code} with no JSON object"
    if response.status_code != 200:
        reason = body.get("reason") or body.get("status") or "not ok"
        return False, f"/health answered {response.status_code}: {reason}"
    if body.get("jobs_ready") is True:
        return True, ""
    return False, "its start-up job run has not finished"


def _wait_for_jobs(
    url: str,
    *,
    wait_s: float | None = None,
    poll_s: float | None = None,
    probe: Callable[[str], tuple[bool, str]] = _jobs_state,
) -> None:
    """Return once the record at *url* has finished its start-up job run.

    Refused, naming the last reason, when it has not within *wait_s* seconds.
    """
    wait = JOBS_WAIT_S if wait_s is None else wait_s
    poll = JOBS_POLL_S if poll_s is None else poll_s
    deadline = time.monotonic() + wait
    while True:
        ready, why = probe(url)
        if ready:
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(poll)
    fail(
        f"The agent record at {url} is not ready",
        f"{why} after {wait:g} s.",
        "run the command again; if it keeps failing, see `osprey health`",
    )
    raise click.exceptions.Exit(1)


def _sources_report(client: Any) -> dict[str, Any]:
    """The record's ``/health/sources`` body, or a refusal naming why there is none."""
    answer = client.health_sources()
    if not answer.available or not isinstance(answer.data, dict):
        fail("The agent record did not answer", str(answer.unavailable or "no report"))
        raise click.exceptions.Exit(1)
    return answer.data


def _activity(source: Mapping[str, Any]) -> str:
    if source.get("stopped"):
        return "stopped"
    if source.get("live"):
        return "live"
    if source.get("idle"):
        return "idle"
    return "pending"


def _count(value: Any) -> int:
    if isinstance(value, list | tuple):
        return len(value)
    return value if isinstance(value, int) else 0


def _backend_lines(backend: Mapping[str, Any], problems: list[str]) -> list[str]:
    name = backend.get("backend") or "unknown"
    if not backend.get("ok"):
        detail = backend.get("detail") or "no detail"
        problems.append(f"the {name} backend is not healthy: {detail}")
        return [f"Backend: {name}, not healthy: {detail}"]
    span = ""
    oldest, newest = backend.get("oldest_arrival_day"), backend.get("newest_arrival_day")
    if oldest and newest:
        span = f", arrival days {oldest} to {newest}"
    entries = backend.get("entries")
    held = f"{entries} entries" if isinstance(entries, int) else "entries unknown"
    return [f"Backend: {name}, ok, {held}{span}"]


def _spool_lines(spool: Mapping[str, Any], problems: list[str]) -> list[str]:
    lines = [f"Spool: {_count(spool.get('backlog_lines'))} lines waiting"]
    if spool.get("backend_error"):
        problems.append(f"the spool cannot commit to the record: {spool['backend_error']}")
    sources = spool.get("sources")
    for path, source in sorted((sources or {}).items()):
        if not isinstance(source, dict):
            continue
        who = source.get("identity") or "?"
        backlog = _count(source.get("backlog_lines"))
        state = _activity(source)
        facts = [state, f"backlog {backlog}"]
        age = source.get("oldest_backlog_age_s")
        if backlog and isinstance(age, int | float):
            facts.append(f"oldest {age:.0f} s")
        refusals = source.get("refusals")
        facts.append(f"refusals (24 h) {'unknown' if refusals is None else refusals}")
        spool_errors = _count(source.get("spool_errors"))
        writer_errors = _count(source.get("writer_errors"))
        abandoned = _count(source.get("abandoned"))
        facts += [
            f"spool errors {spool_errors}",
            f"writer errors {writer_errors}",
            f"abandoned {abandoned}",
        ]
        lines.append(f"  {who}  {path}: " + ", ".join(facts))
        if state == "stopped":
            problems.append(f"spool source {who} has stopped: nothing committed from {path}")
        if refusals:
            problems.append(f"spool source {who}: {refusals} lines refused in the last 24 h")
        if spool_errors:
            problems.append(f"spool source {who}: {spool_errors} lines refused since start-up")
        if writer_errors:
            problems.append(f"spool source {who}: {writer_errors} writer errors")
        if abandoned:
            problems.append(f"spool source {who}: {abandoned} abandoned writers")
    problems.extend(f"spool: {problem}" for problem in spool.get("problems") or ())
    quarantined = _count(spool.get("quarantined"))
    if quarantined:
        problems.append(f"spool: {quarantined} segments quarantined")
    return lines


def _transcript_lines(transcripts: Mapping[str, Any], problems: list[str]) -> list[str]:
    if not transcripts:
        return ["Transcripts: no root is read"]
    lines = ["Transcripts:"]
    for name, root in sorted(transcripts.items()):
        if not isinstance(root, dict):
            continue
        who = root.get("identity") or "?"
        state = _activity(root)
        facts = [state, f"{_count(root.get('files'))} files"]
        if root.get("last_seen"):
            facts.append(f"last committed {root['last_seen']}")
        lines.append(f"  {who}  {name}: " + ", ".join(facts))
        if root.get("error"):
            problems.append(f"transcripts {who}: {root['error']}")
        file_errors = len(root.get("file_errors") or {})
        if file_errors:
            problems.append(f"transcripts {who}: {file_errors} files cannot be read")
        if state == "stopped":
            problems.append(f"transcripts {who} have stopped: nothing committed from {name}")
    return lines


def _export_lines(export: Mapping[str, Any], problems: list[str]) -> list[str]:
    from osprey.health.core.agent_record import EXPORT_LAG_ALLOWED_DAYS

    last = export.get("last_exported_day") or "none yet"
    lag = export.get("lag_days")
    behind = "unknown" if lag is None else str(lag)
    lines = [f"Export: latest exported day {last}, closed days not exported {behind}"]
    if isinstance(lag, int) and lag > EXPORT_LAG_ALLOWED_DAYS:
        problems.append(f"the export is {lag} days behind; run `osprey agent-record export`")
    if export.get("last_error"):
        problems.append(f"the last export failed: {export['last_error']}")
    if export.get("archive_error"):
        problems.append(str(export["archive_error"]))
    return lines


@agent_record_group.command("health")
@repo_option
def health(repo: Path | None) -> None:
    """Show the agent record's source, backend and export health.

    Opens the record (starting the host-local one when no stack or host
    session runs it), waits for its start-up job run, and prints every
    spool source and transcript root (live, idle or stopped, backlog,
    refusals and errors), the backend, and how far the export is behind.

    Example:

    \b
        osprey agent-record health
    """
    deployment = _deployment(repo)
    with _open_record(deployment) as client:
        _wait_for_jobs(client.url)
        body = _sources_report(client)
    problems: list[str] = []
    report(f"Agent record health in {deployment.root}")
    lines = _backend_lines(body.get("backend") or {}, problems)
    lines += _spool_lines(body.get("spool") or {}, problems)
    lines += _transcript_lines(body.get("transcripts") or {}, problems)
    lines += _export_lines(body.get("export") or {}, problems)
    if body.get("jobs_clock"):
        lines.append(f"Jobs clock: fixed at {body['jobs_clock']}")
    for line in lines:
        report(line)
    for problem in problems:
        warn(problem)
    report("No problems found." if not problems else f"{len(problems)} problem(s) found.")


@agent_record_group.command("export")
@repo_option
def export(repo: Path | None) -> None:
    """Run the agent record's catch-up export to the archive now.

    Opens the record (starting the host-local one when no stack or host
    session runs it), waits for its start-up job run, exports every closed
    day the archive does not hold yet, and prints the days this call
    exported and the latest day the archive holds.

    Example:

    \b
        osprey agent-record export
    """
    deployment = _deployment(repo)
    with _open_record(deployment) as client:
        _wait_for_jobs(client.url)
        answer = client.export(timeout=EXPORT_TIMEOUT_S)
        if not answer.available:
            fail("The export did not run", str(answer.unavailable))
            raise click.exceptions.Exit(1)
        days = [str(day) for day in answer.data] if isinstance(answer.data, list) else []
        latest = (_sources_report(client).get("export") or {}).get("last_exported_day")
    if days:
        report(f"Exported {len(days)} day(s): {', '.join(days)}")
    else:
        report("Exported no days: every closed day is in the archive already.")
    report(f"Latest exported day: {latest or 'none yet'}")
