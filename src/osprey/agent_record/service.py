"""The agent-record service: OTLP ingest, the read API and source health.

:func:`create_app` builds the one FastAPI app ``osprey agent-record serve``
runs. It hosts:

``POST /v1/logs``
    The OTLP logs receiver (:class:`~osprey.agent_record.ingest.otlp.OtlpReceiver`).
    A batch is fsynced into the spool before it is answered and committed by
    the spool consumer, never written to the backend by the request.
    ``/v1/traces`` and ``/v1/metrics`` answer 404: only logs are taken.

``GET /v1/read/<method>``
    The read API, one route per :class:`~osprey.agent_record.reader.RecordReader`
    method, named exactly as the method (:data:`READ_METHODS`). Arguments are
    query parameters: ``since`` (ISO-8601; not on the timelines), ``after``
    (the ``next`` cursor of a timeline page), ``limit`` (int), ``wait`` (a
    :class:`~osprey.agent_record.contract.WaitMark` as JSON, normally the
    drain ``{"drain": {"session_id": <str> | null}}``) and the method's
    own (``session_id``, ``agent_id``, ``key``, ``run_id``, ``kind``,
    ``surface``, ``conversation``, ``source``, and ``written_by=self``, which
    keeps the audit entries the caller's own source identity wrote). A
    ``Source`` credential may narrow its view with ``narrow_session`` or with
    one or more ``narrow_run``; no credential can widen it. The body is :meth:`ReadAnswer.to_json` (a timeline page's
    ``next`` and ``newer`` included): 200 when the backend answered, 503 with
    ``status: unavailable`` when it did not. An argument that is not valid
    (a malformed ``after`` cursor too), or one the method does not take,
    answers 400.

``GET /health``
    Unauthenticated: 200 ``{"status": "ok", "jobs_ready": <bool>}``, where
    ``jobs_ready`` turns true once the jobs' start-up run has finished
    (whatever its outcome), or 503 ``{"status": "unconfigured"}`` while the
    service env is missing. A host-local service (one given a lease
    directory) adds ``ready``: false until its first full spool pass and
    transcript scan have committed, true after; if that first pass fails it
    answers 503 ``{"status": "error", "ready": false}`` and is never ready.

``GET /health/sources``
    For any authenticated login: per spool source its backlog, the age of
    the oldest uncommitted segment, abandoned writers, writer errors, when it
    last wrote (``last_seen``) and last committed, the recent refused lines,
    ``refusals`` (refused lines stored in the record within
    :data:`REFUSALS_WINDOW`, earlier runs of the service included) and
    ``spool_errors`` (lines refused since this service started, at
    ``spool_errors_since``); per transcript root its last poll, last commit
    (``last_seen``), newest transcript modification and errors; the
    backend's own health; ``export`` with ``last_exported_day``,
    ``lag_days`` (closed arrival days from the oldest through yesterday, by
    the jobs' clock, that the archive holds no mark for; 0 on an empty
    record) and ``last_error`` (the jobs' last export failure); and
    ``jobs_clock``, the jobs' clock override or ``null``. Every source says
    whether it is ``live`` (a spool source: an ``.open`` segment not
    abandoned; a transcript root: a transcript modified within
    :data:`STOPPED_AFTER`), ``stopped`` (live or holding a backlog, with no
    commit from it within :data:`STOPPED_AFTER`) or ``idle`` (not live and
    no backlog). During a backend outage the spools are kept and their
    backlog is shown here.

``POST /jobs/export``
    For the host and operator logins only (:data:`EXPORTERS`; any other is
    refused 403): runs the catch-up export now, on the jobs' clock, and
    answers ``{"status": "ok", "exported": [<ISO day>, ...]}``. A failure is
    kept as the jobs' ``last_error`` and answers 503 ``status: unavailable``
    when the record failed, 500 ``status: error`` otherwise.

**Authentication** is HTTP Basic, ``<login>:<token>``, against the verifier
in :data:`~osprey.agent_record.credentials.VERIFIER_ENV` (one ``<login>:<sha256
hex of the token>`` per line, compared in constant time). The login's grant is
its row of :func:`osprey.agent_record.identities.credential_table`; a login
the verifier knows but the table does not is refused 403. A missing or wrong
credential answers 401. While the verifier or the secret set
(:data:`~osprey.agent_record.credentials.SECRETS_ENV`) is missing every route
but ``/health`` answers 503 and nothing is ingested: without the secret set,
bodies could not be scrubbed.

**The viewer.** Only the ``history`` login may name a viewer, in
:data:`~osprey.agent_record.identities.VIEWER_HEADER`; any other login that
sends the header is refused 403, and ``history`` naming none reads nothing
(403). The history front derives the viewer at sign-in and states it as one
of ``all``, ``own:<person>`` or ``reader:<name>``; a reader's kinds and days
back come from this deployment's ``agent_record.readers``, never from the
header, and a reader the config does not name is refused.

**The drain** is resolved here, never by the caller, from the scope
the login resolves to: ``Own``/``Source`` drain their own identity's spool
directories (every ``dispatch-worker-N`` one in the shared mode) and its
``.otlp/<login>/`` directories, ``All`` and ``Reader`` drain every directory;
with a ``session_id`` the drain also covers that session's transcript files
under every root of those identities. The service snapshots where each ends,
wakes the spool consumer and the tailer, and the reader waits, within its
bound, until both have committed past the snapshot.

**The lifespan** opens the configured backend (``embedded`` here,
``clickhouse`` imported only when configured) unless one was injected, and
runs three tasks: the transcript tailer, the spool consumer (a pass every
:data:`SPOOL_INTERVAL_S` and right after each accepted OTLP batch) and the
jobs (:mod:`osprey.agent_record.scheduler`): export, then retention, once at
start and then daily, in a worker thread so neither start-up nor a read ever
waits on them. Shutdown cancels the tasks, waits for a job thread still
running, and closes the backend the lifespan opened; an injected backend is
the caller's to close.

**The host-local service** is the same app given a lease directory (``osprey
agent-record serve --lease-dir``, which only :mod:`osprey.agent_record.host`
passes). It runs one spool pass and one transcript scan before the three
tasks start and reports ``ready`` once both committed, and it watches the
lease directory: once no lease there names a live process, it takes the
host's ensure lock without waiting, checks again, and only then stops itself
(by default a SIGTERM to its own process, which runs the normal shutdown),
keeping that lock until it exits so no ensure reuses a record about to go.
Shutdown waits for a first pass still running in its thread before the
backend is closed.

**Where things are**, from the environment the deployment renders, each with
a default under the deployment repo root (``project_root`` of the config):

- :data:`SPOOL_ROOT_ENV`: the spool root, default ``<root>/var/spool``.
- :data:`RECORD_DIR_ENV`: the embedded store's directory, default
  ``<root>/var/agent_record``; the store is :data:`EMBEDDED_FILENAME` in it.
- :data:`ARCHIVE_DIR_ENV`: the archive root the record's jobs export into,
  default ``<root>/var/archive``. It is the same tree on the host and, bind
  mounted at the same project-relative path, in the container. A root this
  process cannot write is reported at start, naming the path; start-up goes on.
- :data:`JOBS_CLOCK_ENV`: unset in every deployment; an ISO date here makes
  every job run read that date's UTC midnight as ``now`` (a test hook that
  lets a replay close a day). The service warns at start while it is set,
  and a value that is not an ISO date stops the app from being built.
- :data:`TRANSCRIPTS_DIR_ENV`: the directory holding one transcript root per
  :func:`~osprey.agent_record.identities.transcript_roots` name, each mounted
  at ``<dir>/<root name>``; default :data:`DEFAULT_TRANSCRIPTS_DIR`.
- :data:`HOST_PROJECTS_ENV`: on the host, the host build's own Claude Code
  project directory (``<claude config dir>/projects/<encoded build dir>``).
  It is read as the ``host`` root with the relpath prefix
  ``<encoded build dir>/``, the root and prefix the host session's forwarder
  reads the same files under while the stack owns the record, so their keys
  and state names agree and no sibling project is read. With it set, a root
  missing under the transcripts directory is skipped. Without it there is no
  ``host`` root: the stack's agent-record never reads the host's transcripts,
  which the host session forwards through the spool
  (:func:`osprey.agent_record.host.forward_host_transcripts`).
- :data:`CLICKHOUSE_URL_ENV` plus the account's
  :data:`~osprey.agent_record.credentials.CLICKHOUSE_USER_ENV` and
  :data:`~osprey.agent_record.credentials.CLICKHOUSE_PASSWORD_ENV`, read only
  with ``backend: clickhouse``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from osprey.agent_record import archive, scheduler
from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record.config import (
    BACKEND_CLICKHOUSE,
    BACKEND_EMBEDDED,
    AgentRecordConfig,
    parse_agent_record_block,
)
from osprey.agent_record.contract import (
    EXPORT_PATH,
    HEALTH_SOURCES_PATH,
    REFUSALS_WINDOW,
    REFUSED_STATE_PREFIX,
    SOURCE_HOST,
    SOURCE_OPERATOR,
    STOPPED_AFTER,
    All,
    BackendHealth,
    Narrow,
    Own,
    Reader,
    RecordBackend,
    RecordBackendError,
    Scope,
    Source,
    WaitMark,
    check_identity,
)
from osprey.agent_record.ingest.otlp import (
    LOGS_PATH,
    METRICS_PATH,
    TRACES_PATH,
    OtlpReceiver,
    not_exported,
)
from osprey.agent_record.ingest.spool import SegmentMark, SpoolConsumer, SpoolHealth
from osprey.agent_record.ingest.tailer import POLL_INTERVAL_S, FileMark, TranscriptTailer
from osprey.agent_record.reader import ReadAnswer, RecordReader
from osprey.utils.bearer import credential_bytes

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_DIR_ENV",
    "CLICKHOUSE_URL_ENV",
    "DEFAULT_TRANSCRIPTS_DIR",
    "EMBEDDED_FILENAME",
    "EXPORT_PATH",
    "EXPORTERS",
    "HEALTH_SOURCES_PATH",
    "HOST_PROJECTS_ENV",
    "IngestDrainer",
    "JOBS_CLOCK_ENV",
    "READ_METHODS",
    "RECORD_DIR_ENV",
    "REFUSALS_WINDOW",
    "SPOOL_INTERVAL_S",
    "SPOOL_ROOT_ENV",
    "STOPPED_AFTER",
    "TRANSCRIPTS_DIR_ENV",
    "WRITTEN_BY_SELF",
    "create_app",
    "drain_visible",
    "open_backend",
    "viewer_scope",
]

#: The spool root agent-record consumes (and the OTLP receiver writes under).
SPOOL_ROOT_ENV = "OSPREY_AGENT_RECORD_SPOOL_ROOT"
#: The embedded store's directory.
RECORD_DIR_ENV = "OSPREY_AGENT_RECORD_DIR"
#: The archive root the record's jobs export into.
ARCHIVE_DIR_ENV = "OSPREY_AGENT_RECORD_ARCHIVE_DIR"
#: An ISO date the jobs read as ``now`` (its UTC midnight); a test hook, unset in deployments.
JOBS_CLOCK_ENV = "OSPREY_AGENT_RECORD_JOBS_CLOCK"
#: The directory each transcript root is mounted under, as ``<dir>/<root name>``.
TRANSCRIPTS_DIR_ENV = "OSPREY_AGENT_RECORD_TRANSCRIPTS_DIR"
#: The ClickHouse HTTP endpoint, with ``backend: clickhouse``.
CLICKHOUSE_URL_ENV = "OSPREY_AGENT_RECORD_CLICKHOUSE_URL"
#: On the host, the host build's Claude Code project directory read as the ``host`` root.
HOST_PROJECTS_ENV = "OSPREY_AGENT_RECORD_HOST_PROJECTS"

#: Where the transcript roots are mounted in the agent-record container.
DEFAULT_TRANSCRIPTS_DIR = "/data/transcripts"
#: The embedded store's file in its directory.
EMBEDDED_FILENAME = "record.duckdb"

#: Seconds between two spool passes when no OTLP batch arrives meanwhile.
SPOOL_INTERVAL_S = 2.0
#: Seconds between two looks at the lease directory of a host-local service.
LEASE_POLL_S = 1.0

#: The source classes whose logins may run the export.
EXPORTERS = frozenset({SOURCE_HOST, SOURCE_OPERATOR})

#: The viewer forms the history front states.
VIEWER_ALL = "all"
VIEWER_OWN_PREFIX = "own:"
VIEWER_READER_PREFIX = "reader:"

#: The one ``written_by`` value: the caller's own source identity.
WRITTEN_BY_SELF = "self"

#: Query parameters every read method takes.
_COMMON_PARAMS = frozenset({"wait", "narrow_session", "narrow_run"})


@dataclass(frozen=True)
class _Method:
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


#: Every read route, ``/v1/read/<name>``, by the reader method it calls.
READ_METHODS: Mapping[str, _Method] = {
    "list_sessions": _Method(optional=("since", "limit")),
    "session_users": _Method(optional=("since",)),
    "timeline": _Method(("session_id",), ("after", "limit")),
    "agent_timeline": _Method(("session_id", "agent_id"), ("after", "limit")),
    "full_entry": _Method(("key",)),
    "run": _Method(("run_id",)),
    "run_timeline": _Method(("run_id",)),
    "recent_audit": _Method(optional=("since", "limit", "surface", "written_by")),
    "bridge_history": _Method(optional=("conversation", "since", "limit")),
    "entries": _Method(("kind",), ("since", "limit", "source")),
}


class _Refused(Exception):
    """A request refused with *status* and *reason*."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Settings:
    record: AgentRecordConfig
    root: Path
    spool_root: Path
    record_dir: Path
    archive_root: Path
    transcripts_dir: Path
    #: ``{login: sha256 hex}``, or ``None`` while the service env is missing.
    verifier: Mapping[str, str] | None
    secrets: tuple[str, ...]
    unconfigured: str = ""
    grants: Mapping[str, ids.CredentialGrant] = field(default_factory=dict)
    host_projects: Path | None = None
    #: The jobs' fixed ``now`` from :data:`JOBS_CLOCK_ENV`; ``None`` for the wall clock.
    jobs_clock: datetime | None = None


def _jobs_clock(env: Mapping[str, str]) -> datetime | None:
    """The jobs' clock override from *env*: the named day's UTC midnight.

    Raises:
        ValueError: If :data:`JOBS_CLOCK_ENV` is set to something not an ISO date.
    """
    raw = env.get(JOBS_CLOCK_ENV, "").strip()
    if not raw:
        return None
    try:
        day = date.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"{JOBS_CLOCK_ENV}={raw!r} is not an ISO date (YYYY-MM-DD)") from None
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _settings(config: Mapping[str, Any], env: Mapping[str, str]) -> _Settings:
    from osprey.utils.workspace import ARCHIVE_DIR_RELPATH, RECORD_DIR_RELPATH, SPOOL_DIR_RELPATH
    from osprey_connectors.workspace import resolve_project_root

    record = parse_agent_record_block(config)
    root = resolve_project_root(config)
    verifier: Mapping[str, str] | None = None
    secrets: tuple[str, ...] = ()
    problem = ""
    raw_verifier = env.get(creds.VERIFIER_ENV, "")
    raw_secrets = env.get(creds.SECRETS_ENV, "")
    if not raw_verifier:
        problem = f"{creds.VERIFIER_ENV} is not set"
    elif not raw_secrets:
        problem = f"{creds.SECRETS_ENV} is not set"
    else:
        try:
            verifier = creds.decode_verifier(raw_verifier)
            secrets = tuple(creds.decode_secret_set(raw_secrets))
        except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
            verifier, secrets = None, ()
            problem = f"the service env cannot be decoded: {type(exc).__name__}"
        else:
            if not verifier:
                verifier, problem = None, f"{creds.VERIFIER_ENV} names no login"
    return _Settings(
        record=record,
        root=root,
        spool_root=Path(env.get(SPOOL_ROOT_ENV) or root / SPOOL_DIR_RELPATH),
        record_dir=Path(env.get(RECORD_DIR_ENV) or root / RECORD_DIR_RELPATH),
        archive_root=Path(env.get(ARCHIVE_DIR_ENV) or root / ARCHIVE_DIR_RELPATH),
        transcripts_dir=Path(env.get(TRANSCRIPTS_DIR_ENV) or DEFAULT_TRANSCRIPTS_DIR),
        verifier=verifier,
        secrets=secrets,
        unconfigured=problem,
        grants=ids.credential_table(config),
        host_projects=Path(env[HOST_PROJECTS_ENV]) if env.get(HOST_PROJECTS_ENV) else None,
        jobs_clock=_jobs_clock(env),
    )


def _report_archive_root(root: Path) -> bool:
    """Warn, naming *root*, when this process cannot write the archive root.

    Never raises and never stops start-up: ingest and reads do not need the
    archive, and the jobs that do report their own failure when they run.

    :return: Whether the root is a directory this process can write
    """
    if root.is_dir() and os.access(root, os.W_OK | os.X_OK):
        return True
    state = "is not writable by" if root.is_dir() else "does not exist for"
    logger.warning(
        "the agent-record archive root %s %s this process; the record's days "
        "cannot be exported there until it is (run `osprey build` to provision it)",
        root,
        state,
    )
    return False


def _transcript_roots(
    config: Mapping[str, Any], settings: _Settings
) -> tuple[dict[str, Path], dict[str, str]]:
    """The tailer's roots and their relpath prefixes.

    Each root of :func:`~osprey.agent_record.identities.transcript_roots` is
    ``<transcripts dir>/<name>``. With :data:`HOST_PROJECTS_ENV` set the
    ``host`` root is that one project directory, prefixed with its own name,
    and any other root that does not exist here is left out. Without it there
    is no ``host`` root.
    """
    roots: dict[str, Path] = {}
    prefixes: dict[str, str] = {}
    host_projects = settings.host_projects
    for name in ids.transcript_roots(config):
        if name == ids.HOST_TRANSCRIPT_ROOT:
            if host_projects is not None:
                roots[name] = host_projects
                prefixes[name] = f"{host_projects.name}/"
            continue
        path = settings.transcripts_dir / name
        if host_projects is not None and not path.is_dir():
            continue
        roots[name] = path
    return roots, prefixes


def _stop_self() -> None:
    import signal

    os.kill(os.getpid(), signal.SIGTERM)


async def _watch_leases(lease_dir: Path, stop: Callable[[], None]) -> None:
    """Call *stop* once no lease in *lease_dir* names a live process.

    The decision is taken under the host's ensure lock (:func:`claim_idle_stop`):
    a round whose lock is busy, because an ensure or a release is deciding
    right now, is skipped.
    """
    from osprey.agent_record.host import any_live_lease, claim_idle_stop

    while True:
        try:
            idle = not await asyncio.to_thread(any_live_lease, lease_dir) and (
                await asyncio.to_thread(claim_idle_stop, lease_dir)
            )
        except Exception:  # an unreadable look is not an empty directory
            logger.exception("lease directory %s could not be read", lease_dir)
            idle = False
        if idle:
            logger.info("no live lease in %s; stopping", lease_dir)
            stop()
            return
        await asyncio.sleep(LEASE_POLL_S)


def open_backend(
    record: AgentRecordConfig, record_dir: Path, env: Mapping[str, str]
) -> RecordBackend:
    """Open the backend *record* configures.

    Raises:
        ValueError: If the backend is not one this service runs, or the
            ClickHouse endpoint is not set.
        RecordBackendError: If the backend cannot be opened.
    """
    if record.backend == BACKEND_EMBEDDED:
        from osprey.agent_record.backends.embedded import EmbeddedBackend

        record_dir.mkdir(parents=True, exist_ok=True)
        return EmbeddedBackend(record_dir / EMBEDDED_FILENAME)
    if record.backend == BACKEND_CLICKHOUSE:
        from osprey.agent_record.backends.clickhouse import ClickHouseBackend

        url = env.get(CLICKHOUSE_URL_ENV, "")
        if not url:
            raise ValueError(f"backend clickhouse needs {CLICKHOUSE_URL_ENV}")
        return ClickHouseBackend(
            url,
            user=env.get(creds.CLICKHOUSE_USER_ENV) or creds.CLICKHOUSE_USER,
            password=env.get(creds.CLICKHOUSE_PASSWORD_ENV, ""),
        )
    raise ValueError(f"agent_record.backend {record.backend!r} is not a backend this service runs")


# ---------------------------------------------------------------------------
# Authentication and scope
# ---------------------------------------------------------------------------


def _basic(header: str | None) -> tuple[str, str] | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "basic" or not value:
        return None
    try:
        decoded = base64.b64decode(value.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    login, sep, token = decoded.partition(":")
    if not sep or not login or not token:
        return None
    return login, token


def _authenticate(settings: _Settings, request: Request) -> ids.CredentialGrant:
    """The grant of the request's Basic credential.

    Raises:
        _Refused: 503 while unconfigured, 401 for a missing or wrong
            credential, 403 for a login with no grant in this deployment.
    """
    if settings.verifier is None:
        raise _Refused(503, f"agent-record is not configured: {settings.unconfigured}")
    presented = _basic(request.headers.get("authorization"))
    if presented is None:
        raise _Refused(401, "not authenticated")
    login, token = presented
    expected = settings.verifier.get(login)
    digest = hashlib.sha256(credential_bytes(token)).hexdigest()
    # Compared even for an unknown login, so the answer takes the same time.
    matched = hmac.compare_digest(credential_bytes(digest), credential_bytes(expected or "0" * 64))
    if expected is None or not matched:
        raise _Refused(401, "not authenticated")
    grant = settings.grants.get(login)
    if grant is None:
        raise _Refused(403, f"login {login!r} has no grant in this deployment")
    return grant


def viewer_scope(value: str, grants: Mapping[str, ids.CredentialGrant]) -> Scope:
    """The scope a viewer the history front names reads.

    ``all`` reads :class:`All`; ``own:<person>`` reads :class:`Own` of that
    person; ``reader:<name>`` reads the named reader's configured scope.

    Raises:
        ValueError: If *value* is none of those, or names a reader the
            deployment does not configure.
    """
    if value == VIEWER_ALL:
        return All()
    if value.startswith(VIEWER_OWN_PREFIX):
        return Own(check_identity(value[len(VIEWER_OWN_PREFIX) :]))
    if value.startswith(VIEWER_READER_PREFIX):
        name = value[len(VIEWER_READER_PREFIX) :]
        grant = grants.get(f"{ids.READER_LOGIN_PREFIX}{name}") if name else None
        if grant is None or grant.source_class != ids.SOURCE_READER:
            raise ValueError(f"viewer {value!r} names no configured reader")
        return grant.scope  # type: ignore[return-value]
    raise ValueError(f"viewer {value!r} is not all, own:<person> or reader:<name>")


def _scope(settings: _Settings, grant: ids.CredentialGrant, request: Request) -> Scope:
    """The scope this request reads: the grant's, the viewer's, or a narrowed source.

    Raises:
        _Refused: 403 for a viewer the grant may not name (or must), 400 for
            narrowing a scope that is not a source's or naming both forms.
    """
    viewer = request.headers.get(ids.VIEWER_HEADER)
    params = request.query_params
    narrow_session = params.get("narrow_session")
    narrow_runs = frozenset(params.getlist("narrow_run"))
    if grant.may_assert_viewer:
        if viewer is None:
            raise _Refused(403, f"the history login reads only for a viewer ({ids.VIEWER_HEADER})")
        try:
            scope = viewer_scope(viewer.strip(), settings.grants)
        except ValueError as exc:
            raise _Refused(403, str(exc)) from None
    elif viewer is not None:
        raise _Refused(403, f"login {grant.login!r} may not name a viewer")
    else:
        scope = grant.scope  # type: ignore[assignment]
    if narrow_session is None and not narrow_runs:
        return scope
    if not isinstance(scope, Source):
        raise _Refused(400, "only a source credential narrows its view")
    try:
        narrow = Narrow(session_id=narrow_session, run_ids=narrow_runs)
    except ValueError as exc:
        raise _Refused(400, str(exc)) from None
    return Source(scope.identity, narrow)


# ---------------------------------------------------------------------------
# Read arguments
# ---------------------------------------------------------------------------


def _since(value: str) -> datetime:
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"since {value!r} is not an ISO-8601 time") from None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def _limit(value: str) -> int:
    if not value.isdigit():
        raise ValueError(f"limit {value!r} is not a non-negative integer")
    return int(value)


def _wait(value: str) -> WaitMark:
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError("wait is not JSON") from None
    return WaitMark.from_json(data)


def _read_arguments(method: _Method, request: Request) -> dict[str, Any]:
    """The keyword arguments of one read call, from the query string.

    Raises:
        ValueError: For a parameter the method does not take, a missing
            required one, a repeated one, or a value that does not parse.
    """
    params = request.query_params
    allowed = set(method.required) | set(method.optional) | _COMMON_PARAMS
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
    for name in params:
        if name != "narrow_run" and len(params.getlist(name)) > 1:
            raise ValueError(f"parameter {name!r} is given more than once")
    kwargs: dict[str, Any] = {}
    for name in method.required:
        value = params.get(name)
        if not value:
            raise ValueError(f"parameter {name!r} is required")
        kwargs[name] = value
    for name in method.optional:
        value = params.get(name)
        if value is None:
            continue
        if name == "since":
            kwargs[name] = _since(value)
        elif name == "limit":
            kwargs[name] = _limit(value)
        else:
            kwargs[name] = value
    if params.get("wait") is not None:
        kwargs["wait"] = _wait(params["wait"])
    return kwargs


def _written_by(config: Mapping[str, Any], grant: ids.CredentialGrant, value: str) -> str:
    """The identity ``written_by=self`` stands for: the caller's own source identity.

    Raises:
        _Refused: 400 for a value other than ``self``, or a login that is no
            record source (the history front and the readers).
    """
    if value != WRITTEN_BY_SELF:
        raise _Refused(400, f"written_by {value!r} is not {WRITTEN_BY_SELF!r}")
    if grant.may_assert_viewer:
        raise _Refused(400, f"login {grant.login!r} writes nothing to filter on")
    try:
        return ids.login_source_identity(config, grant.login)
    except ValueError as exc:
        raise _Refused(400, str(exc)) from None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _iso(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _backend_json(health: BackendHealth) -> dict[str, Any]:
    return {
        "ok": health.ok,
        "backend": health.backend,
        "detail": health.detail,
        "entries": health.entries,
        "oldest_arrival_day": _iso(health.oldest_arrival_day),
        "newest_arrival_day": _iso(health.newest_arrival_day),
    }


def _backend_health(backend: RecordBackend) -> BackendHealth:
    try:
        return backend.health()
    except Exception as exc:  # health is a report; a backend that raises is not ok
        return BackendHealth(ok=False, backend=type(backend).__name__, detail=str(exc))


def _activity(
    live: bool, backlog: int, last_commit: datetime | None, started: datetime, now: datetime
) -> dict[str, bool]:
    """``live``, ``stopped`` and ``idle`` of one source.

    A source with nothing committed by this process is measured from *started*,
    so a service that just started does not call a busy source stopped.
    """
    pending = live or backlog > 0
    since = last_commit if last_commit is not None else started
    return {
        "live": live,
        "stopped": pending and now - since > STOPPED_AFTER,
        "idle": not pending,
    }


def _refusals(backend: RecordBackend, identities: set[str], now: datetime) -> dict[str, int | None]:
    """Each identity's refused lines stored within :data:`REFUSALS_WINDOW`; ``None`` if unreadable."""
    counts: dict[str, int | None] = {}
    for identity in sorted(identities):
        try:
            counts[identity] = backend.count_state(
                f"{REFUSED_STATE_PREFIX}:{identity}:", now - REFUSALS_WINDOW
            )
        except Exception:  # health is a report; an unreadable count is unknown, not zero
            counts[identity] = None
    return counts


def _spool_json(
    health: SpoolHealth, refusals: Mapping[str, int | None], started: datetime, now: datetime
) -> dict[str, Any]:
    sources = {}
    for path, source in health.sources.items():
        row = asdict(source)
        row["abandoned"] = list(source.abandoned)
        row["writer_errors"] = list(source.writer_errors)
        row["refused"] = [asdict(r) for r in source.refused]
        row["spool_errors"] = row.pop("refused_total")
        row["refusals"] = refusals.get(source.identity)
        row["last_seen"] = _iso(source.last_seen)
        row["last_commit"] = _iso(source.last_commit)
        row.update(_activity(source.live, source.backlog_lines, source.last_commit, started, now))
        sources[path] = row
    return {
        "backend_error": health.backend_error,
        "backlog_lines": health.backlog_lines,
        "spool_errors_since": _iso(started),
        "sources": sources,
        "quarantined": [list(pair) for pair in health.quarantined],
        "problems": list(health.problems),
    }


def _transcripts_json(
    tailer: TranscriptTailer | None, started: datetime, now: datetime
) -> dict[str, Any]:
    if tailer is None:
        return {}
    rows = {}
    for name, source in tailer.health.items():
        modified = source.last_modified
        live = modified is not None and now - modified <= STOPPED_AFTER
        rows[name] = {
            "identity": source.identity,
            "ok": source.ok,
            "error": source.error,
            "file_errors": dict(source.file_errors),
            "last_polled": _iso(source.last_polled),
            "last_seen": _iso(source.last_seen),
            "last_modified": _iso(modified),
            "files": source.files,
            "appended": source.appended,
            "duplicates": source.duplicates,
            "skipped_retention": source.skipped_retention,
            **_activity(live, 0, source.last_seen, started, now),
        }
    return rows


def _export_json(
    health: BackendHealth, archive_root: Path, today: date, last_error: str | None
) -> dict[str, Any]:
    """The export block: the newest marked day, the closed days not marked, the last failure."""
    report: dict[str, Any] = {
        "last_exported_day": None,
        "lag_days": None,
        "last_error": last_error,
    }
    try:
        done = archive.exported_days(archive_root)
    except OSError as exc:
        report["archive_error"] = f"the archive's marks cannot be read: {exc}"
        return report
    report["last_exported_day"] = _iso(max(done)) if done else None
    if not health.ok:
        return report
    report["lag_days"] = len(archive.unexported_days(health.oldest_arrival_day, today, done))
    return report


def _health_sources(
    runtime: _Runtime,
    archive_root: Path,
    jobs: scheduler.JobsState,
    jobs_now: datetime,
    jobs_clock: datetime | None,
    started: datetime,
) -> dict[str, Any]:
    """The body of ``GET /health/sources``; blocking, run in a worker thread."""
    now = datetime.now(UTC)
    backend_health = _backend_health(runtime.backend)
    spool = runtime.consumer.health()
    identities = {source.identity for source in spool.sources.values()}
    return {
        "backend": _backend_json(backend_health),
        "spool": _spool_json(spool, _refusals(runtime.backend, identities, now), started, now),
        "transcripts": _transcripts_json(runtime.tailer, started, now),
        "export": _export_json(
            backend_health, archive_root, jobs_now.astimezone(UTC).date(), jobs.last_error
        ),
        "jobs_clock": _iso(jobs_clock),
    }


# ---------------------------------------------------------------------------
# The drain
# ---------------------------------------------------------------------------


def drain_visible(
    config: Mapping[str, Any], scope: Scope, also: frozenset[str] = frozenset()
) -> Callable[[str], bool] | None:
    """Which source identities a drain for *scope* covers; ``None``: every one.

    ``Own`` covers the person's identity and ``Source`` its own (mapped, so a
    dispatch worker covers every worker directory in the shared mode); ``All``
    and ``Reader`` see every identity's entries and cover every directory.
    *also* adds identities (the reader's run drains add the identity that
    accepted the run, whose session the scope already sees).
    """
    if isinstance(scope, (All, Reader)):
        return None
    base = scope.person if isinstance(scope, Own) else scope.identity  # type: ignore[union-attr]
    targets = set()
    for identity in {base, *also}:
        try:
            targets.add(ids.spool_source_identity(config, identity))
        except ValueError:
            continue
    return targets.__contains__


@dataclass(frozen=True)
class _DrainProgress:
    consumer: SpoolConsumer
    tailer: TranscriptTailer
    segments: tuple[SegmentMark, ...]
    files: tuple[FileMark, ...]

    def passed(self) -> bool:
        return self.consumer.drained(self.segments) and self.tailer.drained(self.files)


@dataclass(frozen=True)
class IngestDrainer:
    """The :class:`~osprey.agent_record.reader.Drainer` over this service's ingest.

    A call snapshots what the scope's identities have pending in *consumer*'s
    spool root and, with a session, in *tailer*'s roots, then (when anything
    is pending) calls *wake_consumer* and wakes the tailer; both must be safe
    to call from any thread.
    """

    config: Mapping[str, Any]
    consumer: SpoolConsumer
    tailer: TranscriptTailer
    wake_consumer: Callable[[], None]

    def __call__(
        self, scope: Scope, session_id: str | None, also: frozenset[str]
    ) -> _DrainProgress:
        visible = drain_visible(self.config, scope, also)
        segments = self.consumer.snapshot(visible)
        files = self.tailer.snapshot(session_id, visible) if session_id else ()
        progress = _DrainProgress(self.consumer, self.tailer, segments, files)
        if segments or files:
            self.wake_consumer()
            self.tailer.wake()
        return progress


# ---------------------------------------------------------------------------
# The runtime the lifespan owns
# ---------------------------------------------------------------------------


@dataclass
class _Runtime:
    backend: RecordBackend
    reader: RecordReader
    consumer: SpoolConsumer
    tailer: TranscriptTailer
    receiver: OtlpReceiver
    wake: asyncio.Event
    started: datetime = field(default_factory=lambda: datetime.now(UTC))


async def _consume_forever(runtime: _Runtime) -> None:
    while True:
        try:
            await asyncio.to_thread(runtime.consumer.consume)
        except Exception:  # one failed pass never ends the loop
            logger.exception("spool pass failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(runtime.wake.wait(), timeout=SPOOL_INTERVAL_S)
        runtime.wake.clear()


def _refusal(exc: _Refused) -> JSONResponse:
    headers = {"WWW-Authenticate": 'Basic realm="agent-record"'} if exc.status == 401 else None
    return JSONResponse({"status": "refused", "reason": exc.reason}, exc.status, headers=headers)


def _answer(answer: ReadAnswer) -> JSONResponse:
    return JSONResponse(answer.to_json(), status_code=200 if answer.available else 503)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


def create_app(
    *,
    config: Mapping[str, Any] | None = None,
    backend: RecordBackend | None = None,
    env: Mapping[str, str] | None = None,
    lease_dir: Path | None = None,
    stop: Callable[[], None] | None = None,
) -> FastAPI:
    """Create the agent-record FastAPI application.

    Args:
        config: The rendered deployment config; ``load_osprey_config()`` when
            ``None``.
        backend: A backend to run instead of the configured one. It is kept
            open at shutdown: whoever made it closes it.
        env: The environment to read the service settings from; the process
            environment when ``None``.
        lease_dir: Run as the host-local service: report ``ready`` only after
            the first full ingest pass, and stop once no lease in this
            directory names a live process.
        stop: What stopping means under *lease_dir*; a SIGTERM to this
            process when ``None``.

    Returns:
        The app. It is built whatever the service env holds; while the
        verifier or secret set is missing it answers ``/health`` with 503,
        refuses every other route with 503 and ingests nothing.

    Raises:
        BuildProfileError: If the config's ``agent_record`` block is invalid.
        ValueError: If the credential table cannot be built from the config,
            or :data:`JOBS_CLOCK_ENV` is set but not an ISO date.
    """
    if config is None:
        from osprey_connectors.workspace import load_osprey_config

        config = load_osprey_config()
    config = dict(config)
    environ: Mapping[str, str] = os.environ if env is None else env
    settings = _settings(config, environ)
    jobs_clock = (
        scheduler.wall_clock
        if settings.jobs_clock is None
        else scheduler.fixed_clock(settings.jobs_clock)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = None
        app.state.jobs.ready = False
        app.state.jobs.last_error = None
        _report_archive_root(settings.archive_root)
        if settings.jobs_clock is not None:
            logger.warning(
                "%s is set: the record's jobs read %s as now, not the wall clock",
                JOBS_CLOCK_ENV,
                settings.jobs_clock.isoformat(),
            )
        if settings.verifier is None:
            logger.error(
                "agent-record is not configured (%s); ingest is off", settings.unconfigured
            )
            yield
            return
        opened = backend is None
        record = open_backend(settings.record, settings.record_dir, environ) if opened else backend
        assert record is not None
        retention_days = settings.record.retention_days
        roots, prefixes = _transcript_roots(config, settings)
        consumer = SpoolConsumer(
            record, settings.spool_root, config=config, secrets=settings.secrets
        )
        tailer = TranscriptTailer(
            record,
            roots,
            config,
            secret_set=settings.secrets,
            retention_days=retention_days,
            prefixes=prefixes,
        )
        wake = asyncio.Event()
        loop = asyncio.get_running_loop()

        def wake_consumer() -> None:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                loop.call_soon_threadsafe(wake.set)

        runtime = _Runtime(
            backend=record,
            reader=RecordReader(
                record, config, drain=IngestDrainer(config, consumer, tailer, wake_consumer)
            ),
            consumer=consumer,
            tailer=tailer,
            receiver=OtlpReceiver(settings.spool_root, config, retention_days=retention_days),
            wake=wake,
        )
        app.state.runtime = runtime

        tasks: list[asyncio.Task[Any]] = []

        def start_ingest() -> None:
            tasks.append(asyncio.create_task(runtime.tailer.run(interval=POLL_INTERVAL_S)))
            tasks.append(asyncio.create_task(_consume_forever(runtime)))
            tasks.append(
                asyncio.create_task(
                    scheduler.run(
                        record,
                        archive_root=settings.archive_root,
                        retention_days=retention_days,
                        clock=jobs_clock,
                        state=app.state.jobs,
                    )
                )
            )

        def first_pass() -> None:
            runtime.consumer.consume()
            runtime.tailer.poll_once()

        # The first pass runs in an executor thread a task cancel cannot stop;
        # shutdown waits on it before the backend closes under it.
        first: list[asyncio.Future[None]] = []

        async def first_pass_then_ingest() -> None:
            first.append(loop.run_in_executor(None, first_pass))
            try:
                await asyncio.shield(first[0])
            except Exception:  # never ready: whoever waits on it refuses, naming the log
                logger.exception("the first ingest pass failed; agent-record is not ready")
                app.state.first_pass_failed = True
            else:
                app.state.ready = True
            start_ingest()

        if lease_dir is None:
            start_ingest()
        else:
            tasks.append(asyncio.create_task(first_pass_then_ingest()))
            tasks.append(asyncio.create_task(_watch_leases(lease_dir, stop or _stop_self)))
        try:
            yield
        finally:
            # Each task waits for its own job thread on cancel, so the gather
            # returns only once no job still uses the backend.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*first, return_exceptions=True)
            app.state.runtime = None
            app.state.ready = False
            if opened:
                close = getattr(record, "close", None)
                if close is not None:
                    close()

    app = FastAPI(
        title="OSPREY agent-record",
        description="The one agent record: ingest and the read API",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.runtime = None
    app.state.ready = False
    app.state.first_pass_failed = False
    app.state.jobs = scheduler.JobsState()
    app.state.jobs_clock = jobs_clock

    def runtime_of(request: Request) -> _Runtime:
        runtime: _Runtime | None = request.app.state.runtime
        if runtime is None:
            raise _Refused(503, "agent-record is not running")
        return runtime

    @app.get("/health")
    async def health() -> JSONResponse:
        if settings.verifier is None:
            return JSONResponse(
                {"status": "unconfigured", "reason": settings.unconfigured}, status_code=503
            )
        jobs_ready = app.state.jobs.ready
        if lease_dir is not None:
            if app.state.first_pass_failed:
                return JSONResponse({"status": "error", "ready": False}, status_code=503)
            return JSONResponse(
                {"status": "ok", "ready": bool(app.state.ready), "jobs_ready": jobs_ready}
            )
        return JSONResponse({"status": "ok", "jobs_ready": jobs_ready})

    @app.get(HEALTH_SOURCES_PATH)
    async def health_sources(request: Request) -> JSONResponse:
        try:
            _authenticate(settings, request)
            runtime = runtime_of(request)
        except _Refused as exc:
            return _refusal(exc)
        body = await run_in_threadpool(
            _health_sources,
            runtime,
            settings.archive_root,
            app.state.jobs,
            jobs_clock(),
            settings.jobs_clock,
            runtime.started,
        )
        return JSONResponse(body)

    @app.post(EXPORT_PATH)
    async def export(request: Request) -> JSONResponse:
        try:
            grant = _authenticate(settings, request)
            if grant.source_class not in EXPORTERS:
                raise _Refused(403, f"login {grant.login!r} may not run the export")
            runtime = runtime_of(request)
        except _Refused as exc:
            return _refusal(exc)
        jobs: scheduler.JobsState = app.state.jobs
        try:
            days = await run_in_threadpool(
                scheduler.export, runtime.backend, jobs_clock(), settings.archive_root, jobs
            )
        except Exception as exc:
            logger.exception("on-demand export failed")
            if isinstance(exc, RecordBackendError):
                return JSONResponse(
                    {"status": "unavailable", "reason": jobs.last_error}, status_code=503
                )
            return JSONResponse({"status": "error", "reason": jobs.last_error}, status_code=500)
        return JSONResponse({"status": "ok", "exported": [day.isoformat() for day in days]})

    @app.post(LOGS_PATH)
    async def otlp_logs(request: Request) -> Response:
        try:
            grant = _authenticate(settings, request)
            runtime = runtime_of(request)
        except _Refused as exc:
            return _refusal(exc)
        response = await runtime.receiver.receive_logs(request, grant.login)
        if response.status_code == 200:
            runtime.wake.set()
        return response

    for path in (TRACES_PATH, METRICS_PATH):
        app.add_api_route(path, not_exported, methods=["POST"], include_in_schema=False)

    @app.get("/v1/read/{name}")
    async def read(name: str, request: Request) -> JSONResponse:
        method = READ_METHODS.get(name)
        try:
            grant = _authenticate(settings, request)
            if method is None:
                raise _Refused(404, f"no read method {name!r}")
            scope = _scope(settings, grant, request)
            try:
                kwargs = _read_arguments(method, request)
            except ValueError as exc:
                raise _Refused(400, str(exc)) from None
            if "written_by" in kwargs:
                kwargs["written_by"] = _written_by(config, grant, kwargs["written_by"])
            runtime = runtime_of(request)
        except _Refused as exc:
            return _refusal(exc)
        call: Callable[..., ReadAnswer] = getattr(runtime.reader, name)
        try:
            answer = await run_in_threadpool(lambda: call(scope, **kwargs))
        except ValueError as exc:
            return _refusal(_Refused(400, str(exc)))
        except RecordBackendError as exc:  # a backend failure is an unavailable answer
            answer = ReadAnswer.unavailable(str(exc))
        return _answer(answer)

    return app
