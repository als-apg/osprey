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
    query parameters: ``since`` (ISO-8601), ``limit`` (int), ``wait`` (a
    :class:`~osprey.agent_record.contract.WaitMark` as JSON) and the method's
    own (``session_id``, ``agent_id``, ``key``, ``run_id``, ``kind``,
    ``surface``, ``conversation``). A ``Source`` credential may narrow its view
    with ``narrow_session`` or with one or more ``narrow_run``; no credential
    can widen it. The body is :meth:`ReadAnswer.to_json`: 200 when the backend
    answered, 503 with ``status: unavailable`` when it did not. An argument
    that is not valid, or one the method does not take, answers 400.

``GET /health``
    Unauthenticated: 200 ``{"status": "ok"}``, or 503 ``{"status":
    "unconfigured"}`` while the service env is missing.

``GET /health/sources``
    For any authenticated login: per spool source its backlog, the age of
    the oldest uncommitted segment, abandoned writers, writer errors,
    refusals and when it last wrote; per transcript root its last poll, last
    commit and errors; the backend's own health. During a backend outage the
    spools are kept and their backlog is shown here.

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

**The lifespan** opens the configured backend (``embedded`` here,
``clickhouse`` imported only when configured) unless one was injected, and
runs three tasks: the transcript tailer, the spool consumer (a pass every
:data:`SPOOL_INTERVAL_S` and right after each accepted OTLP batch) and the
daily scheduler (:mod:`osprey.agent_record.scheduler`). Shutdown cancels the
tasks and closes the backend the lifespan opened; an injected backend is the
caller's to close.

**Where things are**, from the environment the deployment renders, each with
a default under the deployment repo root (``project_root`` of the config):

- :data:`SPOOL_ROOT_ENV`: the spool root, default ``<root>/var/spool``.
- :data:`RECORD_DIR_ENV`: the embedded store's directory, default
  ``<root>/var/agent_record``; the store is :data:`EMBEDDED_FILENAME` in it.
- :data:`TRANSCRIPTS_DIR_ENV`: the directory holding one transcript root per
  :func:`~osprey.agent_record.identities.transcript_roots` name, each mounted
  at ``<dir>/<root name>``; default :data:`DEFAULT_TRANSCRIPTS_DIR`.
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

from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record import scheduler
from osprey.agent_record.config import (
    BACKEND_CLICKHOUSE,
    BACKEND_EMBEDDED,
    AgentRecordConfig,
    parse_agent_record_block,
)
from osprey.agent_record.contract import (
    All,
    BackendHealth,
    Narrow,
    Own,
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
from osprey.agent_record.ingest.spool import SpoolConsumer, SpoolHealth
from osprey.agent_record.ingest.tailer import POLL_INTERVAL_S, TranscriptTailer
from osprey.agent_record.reader import ReadAnswer, RecordReader
from osprey.utils.bearer import credential_bytes

logger = logging.getLogger(__name__)

__all__ = [
    "CLICKHOUSE_URL_ENV",
    "DEFAULT_TRANSCRIPTS_DIR",
    "EMBEDDED_FILENAME",
    "READ_METHODS",
    "RECORD_DIR_ENV",
    "SPOOL_INTERVAL_S",
    "SPOOL_ROOT_ENV",
    "TRANSCRIPTS_DIR_ENV",
    "create_app",
    "open_backend",
    "viewer_scope",
]

#: The spool root agent-record consumes (and the OTLP receiver writes under).
SPOOL_ROOT_ENV = "OSPREY_AGENT_RECORD_SPOOL_ROOT"
#: The embedded store's directory.
RECORD_DIR_ENV = "OSPREY_AGENT_RECORD_DIR"
#: The directory each transcript root is mounted under, as ``<dir>/<root name>``.
TRANSCRIPTS_DIR_ENV = "OSPREY_AGENT_RECORD_TRANSCRIPTS_DIR"
#: The ClickHouse HTTP endpoint, with ``backend: clickhouse``.
CLICKHOUSE_URL_ENV = "OSPREY_AGENT_RECORD_CLICKHOUSE_URL"

#: Where the transcript roots are mounted in the agent-record container.
DEFAULT_TRANSCRIPTS_DIR = "/data/transcripts"
#: The embedded store's file in its directory.
EMBEDDED_FILENAME = "record.duckdb"

#: Seconds between two spool passes when no OTLP batch arrives meanwhile.
SPOOL_INTERVAL_S = 2.0

#: The viewer forms the history front states.
VIEWER_ALL = "all"
VIEWER_OWN_PREFIX = "own:"
VIEWER_READER_PREFIX = "reader:"

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
    "timeline": _Method(("session_id",), ("since", "limit")),
    "agent_timeline": _Method(("session_id", "agent_id"), ("since", "limit")),
    "full_entry": _Method(("key",)),
    "run": _Method(("run_id",)),
    "run_timeline": _Method(("run_id",)),
    "recent_audit": _Method(optional=("since", "limit", "surface")),
    "bridge_history": _Method(optional=("conversation", "since", "limit")),
    "entries": _Method(("kind",), ("since", "limit")),
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
    transcripts_dir: Path
    #: ``{login: sha256 hex}``, or ``None`` while the service env is missing.
    verifier: Mapping[str, str] | None
    secrets: tuple[str, ...]
    unconfigured: str = ""
    grants: Mapping[str, ids.CredentialGrant] = field(default_factory=dict)


def _settings(config: Mapping[str, Any], env: Mapping[str, str]) -> _Settings:
    from osprey_connectors.workspace import (
        RECORD_DIR_RELPATH,
        SPOOL_DIR_RELPATH,
        resolve_project_root,
    )

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
        transcripts_dir=Path(env.get(TRANSCRIPTS_DIR_ENV) or DEFAULT_TRANSCRIPTS_DIR),
        verifier=verifier,
        secrets=secrets,
        unconfigured=problem,
        grants=ids.credential_table(config),
    )


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


def _spool_json(health: SpoolHealth) -> dict[str, Any]:
    sources = {}
    for path, source in health.sources.items():
        row = asdict(source)
        row["abandoned"] = list(source.abandoned)
        row["writer_errors"] = list(source.writer_errors)
        row["refused"] = [asdict(r) for r in source.refused]
        row["last_seen"] = _iso(source.last_seen)
        sources[path] = row
    return {
        "backend_error": health.backend_error,
        "backlog_lines": health.backlog_lines,
        "sources": sources,
        "quarantined": [list(pair) for pair in health.quarantined],
        "problems": list(health.problems),
    }


def _transcripts_json(tailer: TranscriptTailer | None) -> dict[str, Any]:
    if tailer is None:
        return {}
    return {
        name: {
            "identity": source.identity,
            "ok": source.ok,
            "error": source.error,
            "file_errors": dict(source.file_errors),
            "last_polled": _iso(source.last_polled),
            "last_seen": _iso(source.last_seen),
            "files": source.files,
            "appended": source.appended,
            "duplicates": source.duplicates,
            "skipped_retention": source.skipped_retention,
        }
        for name, source in tailer.health.items()
    }


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
) -> FastAPI:
    """Create the agent-record FastAPI application.

    Args:
        config: The rendered deployment config; ``load_osprey_config()`` when
            ``None``.
        backend: A backend to run instead of the configured one. It is kept
            open at shutdown: whoever made it closes it.
        env: The environment to read the service settings from; the process
            environment when ``None``.

    Returns:
        The app. It is built whatever the service env holds; while the
        verifier or secret set is missing it answers ``/health`` with 503,
        refuses every other route with 503 and ingests nothing.

    Raises:
        BuildProfileError: If the config's ``agent_record`` block is invalid.
        ValueError: If the credential table cannot be built from the config.
    """
    if config is None:
        from osprey_connectors.workspace import load_osprey_config

        config = load_osprey_config()
    config = dict(config)
    environ: Mapping[str, str] = os.environ if env is None else env
    settings = _settings(config, environ)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = None
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
        roots = {name: settings.transcripts_dir / name for name in ids.transcript_roots(config)}
        runtime = _Runtime(
            backend=record,
            reader=RecordReader(record, config),
            consumer=SpoolConsumer(
                record, settings.spool_root, config=config, secrets=settings.secrets
            ),
            tailer=TranscriptTailer(
                record,
                roots,
                config,
                secret_set=settings.secrets,
                retention_days=retention_days,
            ),
            receiver=OtlpReceiver(settings.spool_root, config, retention_days=retention_days),
            wake=asyncio.Event(),
        )
        app.state.runtime = runtime
        tasks = [
            asyncio.create_task(runtime.tailer.run(interval=POLL_INTERVAL_S)),
            asyncio.create_task(_consume_forever(runtime)),
            asyncio.create_task(scheduler.run(record)),
        ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            app.state.runtime = None
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

    def runtime_of(request: Request) -> _Runtime:
        runtime = request.app.state.runtime
        if runtime is None:
            raise _Refused(503, "agent-record is not running")
        return runtime

    @app.get("/health")
    async def health() -> JSONResponse:
        if settings.verifier is None:
            return JSONResponse(
                {"status": "unconfigured", "reason": settings.unconfigured}, status_code=503
            )
        return JSONResponse({"status": "ok"})

    @app.get("/health/sources")
    async def health_sources(request: Request) -> JSONResponse:
        try:
            _authenticate(settings, request)
            runtime = runtime_of(request)
        except _Refused as exc:
            return _refusal(exc)
        backend_health = await run_in_threadpool(_backend_health, runtime.backend)
        return JSONResponse(
            {
                "backend": _backend_json(backend_health),
                "spool": _spool_json(runtime.consumer.health()),
                "transcripts": _transcripts_json(runtime.tailer),
            }
        )

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
