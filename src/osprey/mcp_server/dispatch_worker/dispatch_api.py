"""FastAPI dispatch API for the OSPREY dispatch worker service.

Exposes:
  POST /dispatch                      — enqueue an agent prompt, returns 202 + run_id
  GET  /dispatch/{id}                 — poll a run for completion (body carries artifact descriptors)
  GET  /dispatch/{id}/stream          — SSE stream of run events
  GET  /dispatch/{id}/artifacts       — descriptors of the artifacts the run produced
  GET  /dispatch/{id}/artifacts/{aid} — resolved (renderable) bytes of one artifact the run produced
  DELETE /dispatch/{id}               — cancel an in-flight run
  GET  /health                        — liveness check with run statistics
  GET  /dashboard/runs                — JSON feed consumed by the dispatcher dashboard (token-gated)

Which artifacts belong to a run is decided by the artifact store's write-time
``run_id`` tag (created-by), so all three artifact surfaces — the status-body
list, the list route, and the byte route — share one strict isolation gate and
cannot disagree. See ``osprey.agent_runner.artifact_resolve``.

Every run is also recorded as two ``run`` entries in the worker's own spool
(:mod:`osprey.agent_record.writer`): ``run:<run_id>:accept`` when the run is
accepted, before the agent starts, naming the session the run was minted and
who asked for it; and ``run:<run_id>:terminal`` once, at the run's end,
holding its outcome fields and none of its text. The prompt, the answer and
the thinking are the transcript's facts, read from the run's session.

A run in flight is answered from this process's memory. Every other read of a
run — its status once terminal, the dashboard's run list, a cancel of a run
this process does not hold, the run counts on ``/health`` and the stale-run
sweep — reads the agent record through :class:`~osprey.agent_record.client.RecordClient`
under this worker's own login, so a worker that restarts loses nothing. A
terminal status is reported only from a complete read: while the record is
unavailable or still catching up, a run reads ``pending`` with a ``record``
field saying which, never terminal with a missing answer.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator

from osprey.agent_record import contract as record_contract
from osprey.agent_record import writer as record_writer
from osprey.agent_record.client import DEFAULT_TIMEOUT, ClientAnswer, RecordClient
from osprey.agent_runner import launcher
from osprey.agent_runner.artifact_resolve import (
    _deployed_config,
    deployed_config_path,
    deployed_record_retention_days,
    deployed_render_dir,
    deployed_repo_root,
    describe_run_artifacts,
    describe_run_input_artifacts,
    get_run_store,
    record_lacks_run,
    resolve_single_run_artifact,
)
from osprey.agent_runner.errors import AgentCliUnavailableError
from osprey.agent_runner.tool_names import dispatch_denied_tools
from osprey.mcp_server.dispatch_worker import (
    counters,
    failure_class,
    retention,
    run_stats,
    sdk_runner,
)
from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents
from osprey.mcp_server.dispatch_worker.credentials import WORKER_TOKEN_ENV, worker_credentials
from osprey.mcp_server.dispatch_worker.input_files_policy import (
    InputFilesError,
    sanitize_filename,
    validate_input_files,
)
from osprey.mcp_server.dispatch_worker.prior_answers import (
    PRIOR_ANSWERS_CAPABILITY,
    keep_run_ids,
)
from osprey.utils.bearer import credential_bytes
from osprey.utils.owner_header import owner_from_header
from osprey.utils.tool_rules import matches_denylist

logger = logging.getLogger("osprey.mcp_server.dispatch_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DISPATCH_TIMEOUT_SEC = int(os.environ.get("DISPATCH_TIMEOUT_SEC", "300"))

#: How many agentic turns one dispatched run may take when the request names no
#: ceiling of its own. Authored per facility as ``dispatch.max_turns`` in the
#: build profile and rendered into this container's environment, beside the two
#: clock budgets — a facility whose triggers ask for a multi-step investigation
#: needs a different number from one that dispatches single-shot summaries.
DISPATCH_MAX_TURNS = int(os.environ.get("DISPATCH_MAX_TURNS", "25"))
_QUEUE_TTL_SEC = 60  # discard unconsumed SSE queues after this many seconds post-completion

# Explicit request-body ceiling. A legit /dispatch body carrying an input-files
# batch tops out near ~24 MB (base64 + prompt); 32 MB leaves ~25% headroom while
# bounding the memory a hostile caller can force the worker to buffer. Enforced
# from Content-Length before the body is read (see ``_limit_request_body``).
MAX_REQUEST_BYTES = 32 * 1024 * 1024

# Capabilities this worker advertises on /health. A downstream bridge gates
# feature use on BOTH the dispatcher's and the worker's /health carrying the
# capability, so the list is a plain JSON array on both bodies.
_CAPABILITIES: list[str] = ["input_files", PRIOR_ANSWERS_CAPABILITY]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# Boot nonce: the worker process's start timestamp, captured once at import.
# Stable for the life of the process and different across restarts, so a poller
# can detect a worker restart (and the accompanying reset of the process-lifetime
# failure counters in ``counters``) by watching this value change.
_BOOT_NONCE: float = time.time()

# In-memory run store: run_id -> result dict (includes created_at, completed_at)
_runs: dict[str, dict[str, Any]] = {}

# SSE event queues: run_id -> asyncio.Queue
_queues: dict[str, asyncio.Queue] = {}

# Running background tasks: run_id -> asyncio.Task (used for cancellation)
_tasks: dict[str, asyncio.Task] = {}

# Maximum entries in _runs before old entries are evicted (prevents unbounded growth)
_MAX_RUNS = 1000


#: How a status body names a read of the record that could not settle the run:
#: the record did not answer, or answered without everything the run needs.
RECORD_UNAVAILABLE = "unavailable"
RECORD_INCOMPLETE = "incomplete"

#: Seconds ``/health`` waits on the record for its run counts: a liveness
#: probe must answer whether or not the record does.
_HEALTH_RECORD_TIMEOUT = 5.0

#: The most runs one dashboard answer lists.
_DASHBOARD_RUNS = 50

#: The fields of a run's in-memory record a pending status body repeats.
_PENDING_FIELDS = ("created_at", "prompt", "session_id", "owner")


def record_client(*, timeout: float = DEFAULT_TIMEOUT) -> RecordClient:
    """The worker's agent-record client, resolved per call from the environment.

    The worker's login is a ``Source`` login, so every read sees this worker's
    own runs (every worker's, in the shared mode) and nothing else.
    """
    return RecordClient.from_env(os.environ, timeout=timeout)


#: The outcome fields a run's terminal entry holds, named as the status body
#: names them. None is text: the answer is the transcript's, and the tool-call
#: count is derived from the run's timeline when it is read.
RUN_OUTCOME_FIELDS: tuple[str, ...] = (
    "status",
    "failure_class",
    "error_code",
    "error",
    "cancelled",
    "cost_usd",
    "num_turns",
    "duration_sec",
    "completed_at",
    "artifacts",
    "input_artifacts",
)

#: The outcome fields that are lists, recorded as ``[]`` when the run has none.
_RUN_LIST_FIELDS = frozenset({"artifacts", "input_artifacts"})


def _entry_ts(epoch: Any) -> datetime | None:
    """The entry time for an epoch-seconds record field; ``None`` means now."""
    if isinstance(epoch, int | float) and not isinstance(epoch, bool) and epoch > 0:
        return datetime.fromtimestamp(epoch, UTC)
    return None


def _write_run_accept(run_id: str, record: Mapping[str, Any]) -> None:
    """Record that ``run_id`` was accepted, under which session and for whom.

    Written before the agent starts, so the run's transcript always has an
    accept to be attributed by. The requester is the run's guarded owner, or
    ``None`` for a fire nobody asked for. A write that fails is reported by the
    writer and never fails the run.
    """
    session_id = record.get("session_id")
    record_writer.write(
        record_contract.KIND_RUN,
        {
            record_contract.RUN_PHASE_FIELD: record_contract.RUN_ACCEPT,
            "run_id": run_id,
            record_contract.RUN_SESSION_FIELD: session_id,
            "requester": record.get("owner"),
            "created_at": record.get("created_at"),
        },
        run_id=run_id,
        session_id=session_id,
        ts=_entry_ts(record.get("created_at")),
    )


def _write_run_terminal(run_id: str, record: Mapping[str, Any]) -> None:
    """Record ``run_id``'s outcome: :data:`RUN_OUTCOME_FIELDS` and nothing else.

    Called once per run, by whichever of the run's own task and the stale sweep
    makes its record terminal. A write that fails is reported by the writer and
    never fails the run.
    """
    body: dict[str, Any] = {
        record_contract.RUN_PHASE_FIELD: record_contract.RUN_TERMINAL,
        "run_id": run_id,
    }
    for field in RUN_OUTCOME_FIELDS:
        value = record.get(field)
        if field in _RUN_LIST_FIELDS:
            value = value or []
        elif field == "cancelled":
            value = bool(value)
        body[field] = value
    record_writer.write(
        record_contract.KIND_RUN,
        body,
        run_id=run_id,
        ts=_entry_ts(record.get("completed_at")),
    )


# ---------------------------------------------------------------------------
# Reading runs from the record
# ---------------------------------------------------------------------------

#: Status bodies of runs a complete read found terminal, by run id. A run's
#: terminal facts never change once read whole, so each is read once.
_settled: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _body_of(item: Any) -> Mapping[str, Any]:
    """The body of a record item as the client returns it, ``{}`` when it has none."""
    body = item.get("body") if isinstance(item, Mapping) else None
    return body if isinstance(body, Mapping) else {}


def _terminal_body(run_id: str, facts: Any) -> dict[str, Any] | None:
    """The status body of a run whose record facts hold a terminal outcome.

    Named as every status body names it: the terminal entry's outcome
    fields, the answer of the run's session as ``text_output``, the tool-call
    count of its timeline, and the accept's time, session and requester (as
    ``owner``, present only when someone asked). ``None`` when *facts* hold no
    terminal outcome.
    """
    if not isinstance(facts, Mapping):
        return None
    outcome = _body_of(facts.get("terminal"))
    if outcome.get("status") not in _TERMINAL_STATUSES:
        return None
    accepted = _body_of(facts.get("accept"))
    body: dict[str, Any] = {"run_id": run_id}
    for field in RUN_OUTCOME_FIELDS:
        value = outcome.get(field)
        body[field] = (value or []) if field in _RUN_LIST_FIELDS else value
    answer = facts.get("answer")
    body["text_output"] = answer if isinstance(answer, str) else ""
    body["num_tool_calls"] = facts.get("num_tool_calls")
    body["created_at"] = accepted.get("created_at")
    body["session_id"] = facts.get("session_id")
    requester = accepted.get("requester")
    if isinstance(requester, str) and requester:
        body["owner"] = requester
    return body


def _pending_body(run_id: str, state: str, known: Mapping[str, Any]) -> dict[str, Any]:
    """A ``pending`` status body for a run the record could not settle.

    *state* is :data:`RECORD_UNAVAILABLE` or :data:`RECORD_INCOMPLETE`; *known*
    is what this process holds of the run, if anything.
    """
    body = {field: known[field] for field in _PENDING_FIELDS if field in known}
    body.update({"run_id": run_id, "status": "pending", "record": state})
    return body


def _remember(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    _settled[run_id] = body
    _settled.move_to_end(run_id)
    while len(_settled) > _MAX_RUNS:
        _settled.popitem(last=False)
    return body


def _read_run_status(run_id: str, known: Mapping[str, Any]) -> dict[str, Any] | None:
    """The status body of ``run_id`` from the record, or ``None`` when it holds no such run.

    Terminal only from a complete read. A read the record could not answer,
    or answered incomplete, gives a ``pending`` body saying which; ``None`` is
    answered only when the record served the read and said it has no entry of
    the run at all. Blocking: route handlers call it off the event loop.
    """
    settled = _settled.get(run_id)
    if settled is not None:
        return settled
    answer = record_client().run(run_id)
    if not answer.available:
        return _pending_body(run_id, RECORD_UNAVAILABLE, known)
    if record_lacks_run(answer, run_id):
        return None
    if answer.complete:
        body = _terminal_body(run_id, answer.data)
        if body is not None:
            return _remember(run_id, body)
    return _pending_body(run_id, RECORD_INCOMPLETE, known)


def _record_runs(answer: ClientAnswer) -> dict[str, dict[str, Mapping[str, Any]]]:
    """The run entries of an ``entries("run")`` answer, by run id and phase."""
    runs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for item in answer.data if isinstance(answer.data, list) else []:
        body = _body_of(item)
        run_id = body.get("run_id") or (item.get("run_id") if isinstance(item, Mapping) else None)
        phase = body.get(record_contract.RUN_PHASE_FIELD)
        if (
            isinstance(run_id, str)
            and run_id
            and phase
            in (
                record_contract.RUN_ACCEPT,
                record_contract.RUN_TERMINAL,
            )
        ):
            runs.setdefault(run_id, {}).setdefault(phase, body)
    return runs


def _list_record_runs(
    client: RecordClient | None = None, *, wait: record_contract.Drain | None = None
) -> tuple[ClientAnswer, dict[str, dict[str, Mapping[str, Any]]]]:
    """The run entries the record keeps (``agent_record.retention_days``), by run id.

    Blocking: route handlers call it off the event loop.
    """
    since = datetime.now(UTC) - timedelta(days=deployed_record_retention_days())
    answer = (client or record_client()).entries(record_contract.KIND_RUN, since=since, wait=wait)
    return answer, (_record_runs(answer) if answer.available else {})


#: How every refusal to start opens, whatever its cause.
_REFUSING_TO_START = "Refusing to start the dispatch worker."


def _inject_provider_env_once() -> None:
    """Inject OSPREY provider env vars (auth, base URL, Claude Code alias models) into os.environ.

    Replicates what the OSPREY web server does at startup so the dispatch
    worker's SDK sessions use the same auth and model configuration.
    """
    # The render's config (``<repo>/build/config.yml``), taken from the
    # environment the compose service sets rather than re-derived from the repo
    # root: a flat ``<repo>/config.yml`` names no file in a four-zone
    # deployment, so a read derived that way misses and the worker starts with
    # no provider auth at all — silently, on the warning branch below.
    config_path = deployed_config_path()
    repo_root = deployed_repo_root()
    if not config_path.is_file():
        logger.warning("No config.yml at %s — skipping provider env injection", config_path)
        return

    launch_env: Mapping[str, str] = {}
    try:
        from osprey.agent_runner.provider_env import inject_provider_env, load_provider_spec

        # Read the spec from the render (the directory holding that config.yml)
        # while expanding ``${VAR}`` against the REPO root's ``.env``: secrets
        # are durable state and deliberately do not live in the disposable
        # render, so a custom provider's ``base_url: ${ARGO_PROD_URL}`` resolves
        # from the repo or not at all. (In a container the value normally
        # arrives as process env via the compose ``env_file``, which the overlay
        # includes either way.) The OTLP block comes from the same environment:
        # the record address and the worker's own record login its compose
        # service hands it.
        spec = load_provider_spec(config_path.parent, env_dir=repo_root)
        if spec:
            injected = inject_provider_env(os.environ, spec, project_dir=repo_root)
            logger.info("Provider env injected: %s (provider=%s)", injected, spec.provider)

            # Non-native (OpenAI-protocol) providers need the in-process
            # translation proxy. Deliver the loopback via os.environ (matching
            # the ``osprey chat`` launch path) because build_clean_env copies os.environ
            # into the SDK env; the in-container proxy thread is reachable by the
            # SDK CLI. Start the proxy from spec.upstream_base_url — the OpenAI
            # root *with* /v1 — NOT os.environ["ANTHROPIC_BASE_URL"], which the
            # resolver strips of /v1 for Claude Code; sourcing the upstream from
            # the env var would forward to a /v1-less endpoint.
            if spec.needs_proxy and spec.upstream_base_url:
                from osprey.infrastructure.proxy.lifecycle import start_proxy_for

                port = start_proxy_for(spec, os.environ)
                os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
                logger.info("Translation proxy on :%d (provider=%s)", port, spec.provider)
            launch_env = os.environ
        else:
            logger.warning("No provider configured in config.yml")
    except Exception:
        logger.exception("Failed to inject provider env from config.yml")

    # Managed (enterprise) policy settings outrank the process environment and
    # setting_sources=["project"] alike, so a policy value that differs from the
    # deployment's would silently redirect the worker's agent. The check reads
    # the finished environment, translation-proxy loopback included, and sits
    # outside the try above so the broad except cannot swallow the refusal. A
    # worker with no provider, or one whose injection failed, compares against
    # nothing, so every policy provider key refuses.
    from osprey.agent_runner.provider_env import (
        detect_managed_policy_conflicts,
        format_managed_policy_conflicts,
    )

    policy_conflicts = detect_managed_policy_conflicts(launch_env)
    if policy_conflicts:
        raise RuntimeError(
            f"{_REFUSING_TO_START}\n" + format_managed_policy_conflicts(policy_conflicts)
        )


def _settle_agent_cli() -> str:
    """Settle and name the Claude Code this worker's runs execute.

    The path is the one every run's SDK options carry, resolved here once for
    the life of the process from the render the runs execute in. A worker
    whose project pins a version that is not installed could only refuse every
    job, so it refuses to start instead, with the operator's message.

    Raises:
        AgentCliUnavailableError: When no installed ``claude`` reports the
            pinned version, or its version cannot be read.
        ValueError: When the pin is not one exact version.
    """
    try:
        path = launcher.project_agent_cli_path(deployed_render_dir())
    except AgentCliUnavailableError as exc:
        raise AgentCliUnavailableError(f"{_REFUSING_TO_START}\n{exc}") from None
    except ValueError as exc:
        raise ValueError(f"{_REFUSING_TO_START}\n{exc}") from None
    logger.info(
        "Claude Code CLI: %s (%s)", path, launcher.cli_version_of(path) or "version unknown"
    )
    return path


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Startup/shutdown lifecycle — provider env injection and the periodic sweeps.

    ``.claude/`` and ``data/`` (MCP config, safety hooks, skills) are already in
    the image: ``osprey build`` renders them into the repo's ``build/`` zone on
    the HOST, and the project image copies the whole repo in one step. Nothing
    regenerates them inside the container, at image build or at startup — so the
    worker starts with those artifacts in place. Provider auth/model config and the translation
    proxy (for non-native providers) still have to be established at process
    startup, since they depend on the mounted ``config.yml`` and environment.
    """
    # First, so a worker that could run no job starts nothing else. Off the
    # event loop, because the version probe spawns the CLI.
    await asyncio.to_thread(_settle_agent_cli)
    _inject_provider_env_once()
    # Route every _stamp's per-class increment into the process-lifetime counters
    # before any run can fail, so no early failure goes uncounted.
    counters.install()
    task = asyncio.create_task(_stale_run_cleanup())
    retention_task = _start_retention_sweep()
    yield
    task.cancel()
    if retention_task is not None:
        retention_task.cancel()


app = FastAPI(title="dispatch-worker", version="1.0.0", lifespan=_lifespan)


@app.middleware("http")
async def _limit_request_body(request: Request, call_next):
    """Reject an over-size request body (413) before the route reads it.

    Guards every route from the declared Content-Length; only /dispatch carries a
    large body in practice, but a global guard is simplest and GET routes carry no
    body. A missing/unparseable Content-Length falls through to normal handling
    (our own callers always set it), so this is a cheap bound, not a hard cap on a
    lying/chunked client.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = -1
        if declared > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
    return await call_next(request)


_bearer_scheme = HTTPBearer()


def _denied_tools(job_tools: Iterable[str]) -> frozenset[str]:
    """The denylist a job naming ``job_tools`` runs under.

    The floor plus the render's ``dispatch.denied_app_tools``; with that list
    missing or unreadable, every server ``job_tools`` names outside the
    framework's own is denied whole (see
    :func:`~osprey.agent_runner.tool_names.dispatch_denied_tools`).
    """
    section = _deployed_config().get("dispatch")
    listed = section.get("denied_app_tools") if isinstance(section, dict) else None
    return dispatch_denied_tools(listed, sdk_runner.required_mcp_servers(job_tools))


def _job_tools(request: DispatchRequest) -> list[str]:
    """Every tool a job may reach: its own, its surface's and its subagents'."""
    tools = [*request.allowed_tools, *(request.surface_tools or [])]
    for surface in parse_project_agents(deployed_render_dir()).values():
        tools.extend(surface or ())
    return tools


def _is_denied(tool: str, denied: Iterable[str] | None = None) -> bool:
    """Return True if ``tool`` is on the denylist.

    Entries ending in ``*`` match by prefix (e.g. the ``mcp__plugin_*`` entry
    blocks every plugin server's tools); all other entries match exactly.
    ``denied`` defaults to the list a job naming only ``tool`` runs under.
    """
    return matches_denylist(tool, _denied_tools([tool]) if denied is None else denied)


# ---------------------------------------------------------------------------
# Stale run cleanup
# ---------------------------------------------------------------------------


def _stale_cutoff() -> int:
    """Seconds after its accept a run with no terminal counts as stale."""
    return DISPATCH_TIMEOUT_SEC + 30


def _find_stale_runs(now: float) -> dict[str, Mapping[str, Any]] | None:
    """The runs to end as stale: accepted longer ago than the cutoff, with no terminal.

    Read from the record after draining this worker's spool, so a terminal
    written but not yet ingested is seen, and a run accepted before this
    process started is found. Each run maps to its accept entry's body. In-flight
    runs this process holds count too, accepted or not. ``None`` — sweep
    nothing — when the record is unavailable or could not drain.
    """
    answer, runs = _list_record_runs(wait=record_contract.Drain())
    if not answer.available or not answer.complete:
        logger.warning(
            "Stale-run sweep skipped: the record is %s (%s)",
            RECORD_UNAVAILABLE if not answer.available else RECORD_INCOMPLETE,
            answer.unavailable or answer.reason,
        )
        return None
    cutoff = _stale_cutoff()
    stale: dict[str, Mapping[str, Any]] = {}
    for run_id, phases in runs.items():
        accepted = phases.get(record_contract.RUN_ACCEPT)
        if accepted is None or record_contract.RUN_TERMINAL in phases:
            continue
        created = accepted.get("created_at")
        if isinstance(created, int | float) and created and (now - created) > cutoff:
            stale[run_id] = accepted
    for run_id, run in _runs.items():
        if run_id in stale or run.get("status") != "pending":
            continue
        if record_contract.RUN_TERMINAL in runs.get(run_id, {}):
            continue
        created = run.get("created_at", 0)
        if created and (now - created) > cutoff:
            stale[run_id] = {}
    return stale


def _end_stale_runs(stale: Mapping[str, Mapping[str, Any]], now: float) -> None:
    """Give each stale run its one infrastructure-error terminal and stop its task."""
    cutoff = _stale_cutoff()
    for run_id, accepted in stale.items():
        known = _runs.get(run_id) or {
            "status": "pending",
            "created_at": accepted.get("created_at"),
            "session_id": accepted.get(record_contract.RUN_SESSION_FIELD),
            **({"owner": accepted["requester"]} if accepted.get("requester") else {}),
        }
        logger.warning("Marking stale run %s as error (no terminal after %ds)", run_id, cutoff)
        swept = {
            **known,
            "status": "error",
            "error": f"Timed out after {cutoff}s",
            "completed_at": now,
            "tool_calls": known.get("tool_calls", []),
        }
        # A stale run is a worker-side fault: stamp it infrastructure with the
        # honest tool-call count and record it as THE terminal. Its key is the
        # run's one terminal key, so whichever terminal the record keeps first is
        # the run's only one. When this run's orphaned coroutine later unwinds
        # with CancelledError, that branch sees an already-terminal record and
        # keeps it instead of stamping again — one terminal, one count.
        num_tool_calls = run_stats.get_run_stats(run_id)["num_tool_calls"]
        failure_class._stamp(swept, failure_class.FAILURE_INFRASTRUCTURE, num_tool_calls)
        if run_id in _runs:
            _runs[run_id] = swept
        _write_run_terminal(run_id, swept)
        # Cancel the orphaned task so its Claude CLI subprocess exits; marking
        # the run error alone would leave it running.
        task = _tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()


def _drop_stale_queues(now: float) -> None:
    """Discard the unconsumed SSE queues of runs that ended a while ago."""
    for run_id in list(_queues.keys()):
        run = _runs.get(run_id, {})
        completed_at = run.get("completed_at")
        if completed_at and (now - completed_at) > _QUEUE_TTL_SEC:
            del _queues[run_id]


def _sweep_stale_runs() -> None:
    """One cleanup sweep: end the stale runs the record shows, and drop old SSE queues.

    The blocking whole of one :func:`_stale_run_cleanup` cycle, callable
    directly. While the record is unavailable no run is ended.
    """
    now = time.time()
    stale = _find_stale_runs(now)
    if stale:
        _end_stale_runs(stale, now)
    _drop_stale_queues(now)


async def _stale_run_cleanup() -> None:
    """Periodically sweep stale runs, reading the record off the event loop."""
    while True:
        await asyncio.sleep(60)
        try:
            now = time.time()
            stale = await asyncio.to_thread(_find_stale_runs, now)
            if stale:
                _end_stale_runs(stale, now)
            _drop_stale_queues(now)
        except Exception:
            logger.exception("Stale-run sweep failed — continuing")


def _in_flight_run_ids() -> frozenset[str]:
    """Run ids of currently-pending (in-flight) runs — never swept by retention."""
    return frozenset(rid for rid, run in _runs.items() if run.get("status") == "pending")


def _start_retention_sweep() -> asyncio.Task | None:
    """Start the opt-in artifact retention sweep task, or return ``None`` when disabled.

    Enabled only when ``RETENTION_DAYS`` is a positive integer. The store factory
    mirrors the artifact resolver's root logic (the module singleton is rooted at
    the wrong CWD in the worker process).
    """
    retention_days = retention.retention_days_from_env()
    if retention_days <= 0:
        return None

    from osprey.agent_runner.artifact_resolve import _get_store

    return asyncio.create_task(
        retention.retention_loop(
            _get_store,
            retention_days,
            _in_flight_run_ids,
        )
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> None:
    expected = worker_credentials().worker_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{WORKER_TOKEN_ENV} is not configured",
        )
    # Constant-time comparison to avoid leaking the token via timing, matching
    # the dispatcher's _check_auth / WebhookSource._handle — over bytes, so a
    # bearer carrying anything outside ASCII is refused rather than raised on.
    if not hmac.compare_digest(
        credential_bytes(credentials.credentials), credential_bytes(expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class InputFile(BaseModel):
    """One caller-supplied file carried alongside a dispatch prompt.

    ``ingest`` defaults to ``True`` (the file is meant to be handed to the
    agent). ``filename`` is sanitised to a safe basename on construction — no
    directory or traversal components survive — so every downstream consumer
    sees an already-safe name. Caps and the mime allowlist are enforced at
    request time by :func:`osprey.mcp_server.dispatch_worker.input_files_policy.validate_input_files`,
    not on this model, so an invalid batch rejects the whole request with one
    machine-readable code rather than a field-level 422.
    """

    filename: str
    mime: str
    content_b64: str
    ingest: bool = True

    @field_validator("filename")
    @classmethod
    def _sanitize_filename(cls, value: str) -> str:
        return sanitize_filename(value)


class DispatchRequest(BaseModel):
    """Request body for ``POST /dispatch``.

    ``surface_prompt``/``surface_tools`` are additive: a request omitting them
    (any caller predating this field) is unaffected — both default to ``None``,
    a pure no-op for the dispatched run. Consuming them (system-prompt
    injection, tool narrowing) is left to downstream tasks; this model only
    carries the values across the dispatcher→worker boundary.

    ``input_files`` is likewise additive and defaults to ``None``. When present
    it is validated fail-closed at request time (see :func:`dispatch`); this
    model only carries the batch — ingestion is a downstream task.

    ``max_turns`` defaults to :data:`DISPATCH_MAX_TURNS`, this deployment's own
    ceiling, so a caller that names none gets the facility's number rather than
    a framework literal.

    ``owner`` is the person the dispatcher attributed the fire to, read there
    from the owner header — never a claim the job makes about itself. It is
    additive and defaults to ``None``: a body from a dispatcher naming no owner
    validates and its run is held to the deployment ceiling, as a cron fire is.
    Worker and dispatcher are separately deployed images, so the field must be
    declared here to survive at all — an undeclared key is dropped silently, and
    the run would be judged against nobody's narrowing while the fire was
    attributed to a person. The worker applies the same allowlist the
    dispatcher applied, because the value is written into the run record and
    served to a browser. A refused value becomes ``None`` and the run goes ahead
    owner-less; it is never a 422, since a malformed owner costs the
    attribution and never the work. A dispatcher always sends a value that
    already passed the guard, so the check binds only a caller posting
    ``/dispatch`` directly.

    ``prior_answer_runs`` is additive and defaults to ``None``: the run ids of
    earlier answers the bridge replayed shortened, and the only runs
    ``prior_answer_read`` will read for this run. Malformed ids are dropped
    rather than refused — the list is enrichment, and a dispatch must not fail
    over it.
    """

    prompt: str
    allowed_tools: list[str]
    max_turns: int = DISPATCH_MAX_TURNS
    surface_prompt: str | None = None
    surface_tools: list[str] | None = None
    input_files: list[InputFile] | None = None
    owner: str | None = None
    prior_answer_runs: list[str] | None = None

    @field_validator("owner", mode="before")
    @classmethod
    def _guard_owner(cls, value: Any) -> str | None:
        return owner_from_header(value)

    @field_validator("prior_answer_runs", mode="before")
    @classmethod
    def _keep_valid_prior_answer_runs(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            return None
        return keep_run_ids(value) or None


class DispatchResponse(BaseModel):
    status: str
    run_id: str


# ---------------------------------------------------------------------------
# Input-file ingestion
# ---------------------------------------------------------------------------

# Prompt-assembly seam, keyed by run_id: the per-input-file descriptors that
# ingestion hands to the (downstream) prompt-assembly step. Held in memory only
# for the brief window between ingestion (start of the run task) and prompt
# assembly; the run task's ``finally`` drops any entry a consumer left behind.
_run_input_seam: dict[str, list[dict[str, Any]]] = {}


def take_input_seam(run_id: str) -> list[dict[str, Any]]:
    """Pop and return the input-file seam for ``run_id`` (``[]`` if none).

    The handoff from ingestion to prompt assembly. Each item is
    ``{"filename": str, "mime": str, "entry_id": str | None, "content_b64": str | None}``:
    for an ingested (``ingest=True``) file ``entry_id`` is its artifact-store id
    and ``content_b64`` is ``None`` (fetch the bytes via the byte route); for an
    ``ingest=False`` file ``entry_id`` is ``None`` and ``content_b64`` carries
    the original bytes to inline directly. Pop-once: a second call returns
    ``[]``.
    """
    return _run_input_seam.pop(run_id, [])


def _input_artifact_type(mime: str) -> str:
    """Coarse ``artifact_type`` for an ingested input file, from its mime."""
    if mime.startswith("image/"):
        return "image"
    if mime == "application/json":
        return "json"
    if mime.startswith("text/"):
        return "text"
    return "file"


def ingest_input_files(run_id: str, input_files: list[InputFile]) -> list[dict[str, Any]]:
    """Ingest a validated input-file batch for ``run_id`` before its SDK run.

    ``ingest=True`` files are base64-decoded and written to the artifact store
    tagged with this run's ``run_id`` and ``origin="input"`` — so the byte route
    serves them and the status body's ``input_artifacts`` lists them, while the
    agent's produced-artifact listing excludes them. ``ingest=False`` files are
    never stored; their bytes ride the returned seam straight to prompt
    assembly. Returns the seam (one item per file, in request order) documented
    on :func:`take_input_seam`.

    The batch was validated at request time (mime allowlist, decodable base64,
    size caps), so this never re-rejects — it only decodes and writes. Filenames
    are already sanitised on the model, so the stored name is safe.
    """
    if not input_files:
        return []
    store = get_run_store()
    seam: list[dict[str, Any]] = []
    for f in input_files:
        if f.ingest:
            raw = base64.b64decode(f.content_b64)
            entry = store.save_file(
                file_content=raw,
                filename=f.filename,
                artifact_type=_input_artifact_type(f.mime),
                title=f.filename,
                mime_type=f.mime,
                tool_source="input",
                metadata={"input_filename": f.filename},
                run_id=run_id,
                origin="input",
            )
            seam.append(
                {"filename": f.filename, "mime": f.mime, "entry_id": entry.id, "content_b64": None}
            )
        else:
            seam.append(
                {
                    "filename": f.filename,
                    "mime": f.mime,
                    "entry_id": None,
                    "content_b64": f.content_b64,
                }
            )
    return seam


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


# Statuses that mark a run's record as final. ``pending`` is the only
# non-terminal state; the stale-run sweep and the task's own error branches all
# transition it to one of these.
_TERMINAL_STATUSES = frozenset({"completed", "error"})


def _is_terminal(run: dict[str, Any]) -> bool:
    """True when ``run`` already holds a final (completed/error) record."""
    return run.get("status") in _TERMINAL_STATUSES


def _build_stamped_error(
    run_id: str,
    error: str,
    failure_cls: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a terminal error record for ``run_id`` and stamp its failure class.

    Centralises the two things every ``_run_dispatch_task`` error branch must do
    identically: read the honest tool-call count from the shared stats map
    (while it is still live — before the ``finally`` pops it) and stamp the
    class plus count. ``extra`` overrides defaults (e.g. the timeout branch's
    ``duration_sec``).
    """
    result: dict[str, Any] = {
        "status": "error",
        "text_output": "",
        "tool_calls": [],
        "error": error,
        "duration_sec": 0.0,
        "cost_usd": None,
        "num_turns": None,
        "completed_at": time.time(),
        "artifacts": [],
    }
    if extra:
        result.update(extra)
    num_tool_calls = run_stats.get_run_stats(run_id)["num_tool_calls"]
    failure_class._stamp(result, failure_cls, num_tool_calls)
    return result


def _attributed(record: dict[str, Any], owner: str | None) -> dict[str, Any]:
    """Name the run's owner on *record* and return it.

    A run record carries ``owner`` only when a person was named; an absent key
    means the fire was unattributed.
    """
    if owner is not None:
        record["owner"] = owner
    return record


def _record_terminal(
    run_id: str, accepted: Mapping[str, Any], outcome: dict[str, Any], owner: str | None
) -> dict[str, Any]:
    """Store and record a run's terminal outcome, and return it.

    A terminal record is the run's accepted record with the outcome laid over
    it, so the fields stamped when the run was accepted (``created_at``,
    ``prompt``, ``owner``, ``session_id``) survive every outcome; the outcome's
    keys win. This is the one place ``_run_dispatch_task`` stores and records a
    terminal outcome.
    """
    record = _attributed({**accepted, **outcome}, owner)
    _runs[run_id] = record
    _write_run_terminal(run_id, record)
    return record


async def _run_dispatch_task(run_id: str, request: DispatchRequest) -> None:
    # The record dispatch() wrote, taken before the first await so every terminal
    # record is built on it even if the sweep or the store cap replaces the entry.
    accepted = dict(_runs.get(run_id) or {})
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _queues[run_id] = queue

    logger.info(
        "Dispatch %s accepted: %d tools, max_turns=%d",
        run_id,
        len(request.allowed_tools),
        request.max_turns,
    )
    try:
        # Ingest caller-supplied input files BEFORE the SDK run: ingest=true
        # files are decoded into the artifact store (run_id tag + origin="input")
        # so their entry_ids exist for prompt assembly and the status body, while
        # ingest=false files pass through untouched. The returned seam is stashed
        # for the prompt-assembly step to consume via take_input_seam(); the
        # finally drops it if nothing consumed it.
        _run_input_seam[run_id] = ingest_input_files(run_id, request.input_files or [])

        result = await asyncio.wait_for(
            sdk_runner.run_dispatch(
                prompt=request.prompt,
                allowed_tools=request.allowed_tools,
                max_turns=request.max_turns,
                event_queue=queue,
                denied_tools=_denied_tools(_job_tools(request)),
                run_id=run_id,
                surface_prompt=request.surface_prompt,
                surface_tools=request.surface_tools,
                owner=request.owner,
                prior_answer_runs=request.prior_answer_runs,
                session_id=accepted.get("session_id"),
            ),
            timeout=DISPATCH_TIMEOUT_SEC,
        )
        result["completed_at"] = time.time()
        # Descriptors of the artifacts this run produced (created-by tag), for
        # consumers that republish them (see the /artifacts routes). Render-free:
        # computed once at completion from the store tag and persisted with the
        # run so it survives restart and never disagrees with the live routes.
        result["artifacts"] = describe_run_artifacts(run_id)
        # Caller-supplied inputs ingested for this run, kept separate from the
        # agent's produced ``artifacts`` (both read the same created-by store tag).
        result["input_artifacts"] = describe_run_input_artifacts(run_id)
        result = _record_terminal(run_id, accepted, result, request.owner)
        logger.info("Dispatch %s completed: status=%s", run_id, result.get("status"))
    except TimeoutError:
        logger.error("Dispatch %s timed out after %ds", run_id, DISPATCH_TIMEOUT_SEC)
        err_result = _build_stamped_error(
            run_id,
            f"Timed out after {DISPATCH_TIMEOUT_SEC}s",
            failure_class.FAILURE_INFRASTRUCTURE,
            extra={"duration_sec": DISPATCH_TIMEOUT_SEC},
        )
        _record_terminal(run_id, accepted, err_result, request.owner)
        await queue.put({"type": "error", "message": f"Timed out after {DISPATCH_TIMEOUT_SEC}s"})
    except asyncio.CancelledError:
        existing = _runs.get(run_id)
        if existing is not None and _is_terminal(existing):
            # Race lost to the stale-run sweep: it already wrote and stamped the
            # terminal record (infrastructure) and then cancelled this coroutine.
            # Preserve that record — do not overwrite it with "cancelled by user"
            # and do not stamp again — so the run keeps exactly one terminal
            # record and one counter increment.
            logger.warning(
                "Dispatch %s cancelled after already terminal; preserving swept record (%s)",
                run_id,
                existing.get("error"),
            )
            with suppress(Exception):
                await queue.put({"type": "error", "message": existing.get("error", "cancelled")})
            raise
        # Genuine user cancel of a still-pending run: a run-level terminal state.
        logger.warning("Dispatch %s cancelled by user", run_id)
        err_result = _build_stamped_error(
            run_id,
            "cancelled by user",
            failure_class.FAILURE_RUN,
            extra={"cancelled": True},
        )
        _record_terminal(run_id, accepted, err_result, request.owner)
        # Best-effort SSE notify during cancellation: the run is already
        # recorded as cancelled above, so a closed/full queue here is non-fatal.
        with suppress(Exception):
            await queue.put({"type": "error", "message": "cancelled by user"})
        raise
    except Exception as exc:
        logger.error("Dispatch %s failed: %s", run_id, exc, exc_info=True)
        # Reached only for faults in the worker's own orchestration around the
        # run: run_dispatch handles provider/agent errors internally and returns
        # them already stamped, so anything raised out to this layer is
        # structurally an infrastructure fault (hence the literal class rather
        # than routing through classify_exception).
        err_result = _build_stamped_error(run_id, str(exc), failure_class.FAILURE_INFRASTRUCTURE)
        _record_terminal(run_id, accepted, err_result, request.owner)
        await queue.put({"type": "error", "message": str(exc)})
    finally:
        _tasks.pop(run_id, None)
        run_stats.pop_run_stats(run_id)
        _run_input_seam.pop(run_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/dispatch",
    response_model=DispatchResponse,
    status_code=202,
    dependencies=[Depends(_verify_token)],
)
async def dispatch(request: DispatchRequest) -> DispatchResponse:
    """Enqueue an agent prompt and return immediately with a run_id to poll."""
    # Server-side tool denylist enforcement (supports '*'-suffix wildcards)
    job_denied = _denied_tools(request.allowed_tools)
    denied = [t for t in request.allowed_tools if _is_denied(t, job_denied)]
    if denied:
        logger.warning("Rejected dispatch: denied tools %s", denied)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tools blocked by server denylist: {denied}",
        )

    # Fail-closed input-file validation BEFORE any run is created: mime
    # allowlist, per-file and total decoded-size caps, ingest-file count. A
    # violation rejects the whole request with a machine-readable 400 detail
    # (a bridge maps it to a non-retryable failure class).
    if request.input_files:
        try:
            validate_input_files(request.input_files)
        except InputFilesError as exc:
            logger.warning("Rejected dispatch: input files invalid (%s): %s", exc.detail, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=exc.detail,
            ) from exc

    # Evict oldest entries if store exceeds capacity
    if len(_runs) >= _MAX_RUNS:
        to_remove = list(_runs.keys())[: _MAX_RUNS // 10]
        for key in to_remove:
            del _runs[key]
            _queues.pop(key, None)

    run_id = str(uuid.uuid4())
    # The run's session is minted here, at accept, so the accept entry and the
    # agent's transcript name one session; the task hands it to the SDK.
    accepted = _attributed(
        {
            "status": "pending",
            "created_at": time.time(),
            "prompt": request.prompt,
            "session_id": str(uuid.uuid4()),
        },
        request.owner,
    )
    _runs[run_id] = accepted
    _write_run_accept(run_id, accepted)
    # Use create_task (not BackgroundTasks) so we retain a handle for cancellation.
    _tasks[run_id] = asyncio.create_task(_run_dispatch_task(run_id, request))
    return DispatchResponse(status="accepted", run_id=run_id)


@app.delete("/dispatch/{run_id}", dependencies=[Depends(_verify_token)])
async def cancel_dispatch(run_id: str) -> dict[str, Any]:
    """Cancel an in-flight dispatch.

    Looks up the running asyncio.Task and requests cancellation. The run status
    transitions to `error` with `cancelled: true`. Measured cancel latency is
    ~0.6s end-to-end; a few seconds longer if a tool call is mid-flight.

    A run this process does not hold is looked up in the record: one it has no
    entry of is a 404, and any other cannot be cancelled here, which the answer
    says along with why (finished, no task in this process, or the record could
    not say).
    """
    run = _runs.get(run_id)
    if run is None:
        return await _cancel_unheld(run_id)
    if run.get("status") != "pending":
        return {"run_id": run_id, "cancelled": False, "reason": "already finished"}

    task = _tasks.get(run_id)
    if task is None or task.done():
        return {"run_id": run_id, "cancelled": False, "reason": "no task handle"}

    task.cancel()
    logger.info("Dispatch %s cancellation requested", run_id)
    return {"run_id": run_id, "cancelled": True}


async def _cancel_unheld(run_id: str) -> dict[str, Any]:
    """The cancel answer for a run this process holds no record of."""
    if not record_contract.is_token(run_id):
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    body = await asyncio.to_thread(_read_run_status, run_id, {})
    if body is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    if body.get("status") in _TERMINAL_STATUSES:
        return {"run_id": run_id, "cancelled": False, "reason": "already finished"}
    if body.get("record"):
        return {"run_id": run_id, "cancelled": False, "reason": f"record {body['record']}"}
    return {"run_id": run_id, "cancelled": False, "reason": "no task handle"}


@app.get("/dispatch/{run_id}", dependencies=[Depends(_verify_token)])
async def get_dispatch_result(run_id: str) -> dict[str, Any]:
    """Poll a run by its run_id.

    A run in flight in this process answers from memory. Every other run
    answers from the record: terminal only from a complete read, ``pending``
    with ``record: "unavailable" | "incomplete"`` while the record cannot settle
    it, and 404 only when the record says it holds no entry of the run.
    """
    run = _runs.get(run_id)
    if run is not None and not _is_terminal(run):
        return run
    if not record_contract.is_token(run_id):
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    body = await asyncio.to_thread(_read_run_status, run_id, run or {})
    if body is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    return body


@app.get("/dispatch/{run_id}/artifacts", dependencies=[Depends(_verify_token)])
async def list_dispatch_artifacts(run_id: str) -> list[dict[str, Any]]:
    """List descriptors of the artifacts a run produced (created-by tag).

    Metadata only — never renders a converter, so it is cheap regardless of
    artifact count. Returns ``[]`` for a run with no artifacts AND for an
    unknown run: the strict store-tag filter is itself the existence check (a
    bogus run_id matches nothing), and returning ``[]`` rather than 404 avoids
    a cross-run existence oracle. Use the status body to distinguish known vs
    unknown runs. Item shape is identical to the status body's ``artifacts``.
    """
    return describe_run_artifacts(run_id)


@app.get("/dispatch/{run_id}/artifacts/{artifact_id}", dependencies=[Depends(_verify_token)])
async def get_dispatch_artifact(run_id: str, artifact_id: str) -> FileResponse:
    """Serve the resolved (renderable) bytes of one artifact produced by ``run_id``.

    The artifact must have been *created by* this run (the store's write-time
    ``run_id`` tag): an artifact belonging to a different run, to no run, an
    unknown or traversal-shaped ``artifact_id``, and a missing on-disk file all
    collapse to the same 404 — so a caller holding one run_id can neither read
    nor probe for another run's output. ``artifact_id`` is resolved through the
    store index, never joined into a path. HTML/notebook/etc. artifacts are
    converted to PNG here (only the requested one); images, PDFs and CSV/TSV
    tables pass through.
    """
    ref = await resolve_single_run_artifact(run_id, artifact_id)
    if ref is None:
        # Constant, input-free 404 for every deny reason (unknown run / cross-run /
        # unknown id / missing file), so the response is never an existence oracle.
        raise HTTPException(status_code=404, detail="artifact not found for run")

    return FileResponse(
        ref.abs_path,
        media_type=ref.mime_type or "application/octet-stream",
        filename=ref.filename,
    )


@app.get("/dispatch/{run_id}/stream", dependencies=[Depends(_verify_token)])
async def stream_dispatch(run_id: str, request: Request) -> StreamingResponse:
    """SSE stream of dispatch events for a run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")

    queue = _queues.get(run_id)
    if queue is None:
        raise HTTPException(status_code=410, detail="Stream no longer available")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
        finally:
            _queues.pop(run_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _record_run_counts() -> dict[str, Any]:
    """The run counts ``/health`` reports from the record, with the read's state."""
    answer, runs = _list_record_runs(record_client(timeout=_HEALTH_RECORD_TIMEOUT))
    if not answer.available:
        return {"completed": None, "error": None, "ids": None, "record": RECORD_UNAVAILABLE}
    counts: dict[str, Any] = {"completed": 0, "error": 0, "ids": set(runs)}
    for phases in runs.values():
        status_ = phases.get(record_contract.RUN_TERMINAL, {}).get("status")
        if status_ in counts:
            counts[status_] += 1
    counts["record"] = None if answer.complete else RECORD_INCOMPLETE
    return counts


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check with run statistics — no auth required.

    Pending runs are this process's in-flight runs; the finished counts and
    the total come from the record, within its retention. While the record is
    unavailable those three are ``None`` and ``record`` says so; the check
    itself still answers ``ok``. Carries the worker's control-identity fields;
    see :func:`osprey.deployment.control_identity.health_fields`.
    """
    from osprey.deployment.control_identity import health_fields

    pending = _in_flight_run_ids()
    counts = await asyncio.to_thread(_record_run_counts)
    ids = counts["ids"]
    # Process-lifetime failure counters — monotonic and independent of the
    # record (see counters module docstring).
    lifetime = counters.get_counts()
    body: dict[str, Any] = {
        "status": "ok",
        "pending_runs": len(pending),
        "completed_runs": counts["completed"],
        "error_runs": counts["error"],
        "total_runs": None if ids is None else len(ids | pending),
        "provider_errors": lifetime[failure_class.FAILURE_PROVIDER],
        "infrastructure_errors": lifetime[failure_class.FAILURE_INFRASTRUCTURE],
        "boot_nonce": _BOOT_NONCE,
        "capabilities": _CAPABILITIES,
        **health_fields(),
    }
    if counts["record"] is not None:
        body["record"] = counts["record"]
    return body


# ---------------------------------------------------------------------------
# Dashboard runs API (token-gated, consumed by the dispatcher server-side)
# ---------------------------------------------------------------------------


def _dashboard_row(run_id: str, run: Mapping[str, Any], now: float) -> dict[str, Any]:
    created = run.get("created_at") or 0
    row = {
        "run_id": run_id,
        "status": run.get("status", "unknown"),
        "created_at": created,
        "age_sec": round(now - created, 1) if created else None,
        "duration_sec": run.get("duration_sec"),
        "text_output": run.get("text_output") or "",
        "tool_count": (
            run["num_tool_calls"]
            if isinstance(run.get("num_tool_calls"), int)
            else len(run.get("tool_calls") or [])
        ),
        "error": run.get("error"),
        "num_turns": run.get("num_turns"),
        # The run's session: the OSPREY-forced telemetry session UUID (see
        # sdk_runner) its transcript and the record's timeline are keyed by, so
        # the dashboard can link a run to its history. None when unrecorded.
        "session_id": run.get("session_id"),
        # The person the fire was attributed to, or None for an owner-less
        # fire (the key is always present, like session_id).
        "owner": run.get("owner"),
        "has_stream": run_id in _queues,
    }
    if run.get("record"):
        row["record"] = run["record"]
    return row


@app.get("/dashboard/runs", dependencies=[Depends(_verify_token)])
async def dashboard_runs() -> list[dict[str, Any]]:
    """Recent runs for the dispatcher dashboard — token-gated, most recent first.

    The runs the record keeps (``agent_record.retention_days``) plus the ones
    in flight in this process, newest :data:`_DASHBOARD_RUNS`. A finished run
    reads its status body from the record, so a row is terminal only from a
    complete read; any other row of a run that has ended is ``pending`` with a
    ``record`` field. Carries each run's answer and error, so this is
    bearer-gated like the other worker endpoints; the dispatcher (the only
    caller) holds the token and proxies this feed to its own dashboard. A
    record that cannot be read answers 503.
    """
    answer, runs = await asyncio.to_thread(_list_record_runs)
    if not answer.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"record unavailable: {answer.unavailable}",
        )
    now = time.time()
    candidates: dict[str, dict[str, Any]] = {}
    for run_id, phases in runs.items():
        accepted = phases.get(record_contract.RUN_ACCEPT, {})
        candidates[run_id] = {
            "status": "pending",
            "created_at": accepted.get("created_at"),
            "session_id": accepted.get(record_contract.RUN_SESSION_FIELD),
            "owner": accepted.get("requester"),
            "ended": record_contract.RUN_TERMINAL in phases,
        }
    for run_id, run in _runs.items():
        candidates[run_id] = {**candidates.get(run_id, {}), **run, "ended": _is_terminal(run)}
    newest = sorted(
        candidates.items(), key=lambda item: item[1].get("created_at") or 0, reverse=True
    )[:_DASHBOARD_RUNS]
    ended = [(run_id, run) for run_id, run in newest if run.pop("ended", False)]
    read = await asyncio.gather(
        *(asyncio.to_thread(_read_run_status, run_id, run) for run_id, run in ended)
    )
    settled = {run_id: body for (run_id, _), body in zip(ended, read, strict=True)}
    rows = []
    for run_id, run in newest:
        if run_id in settled:
            body = settled[run_id]
            if body is None:
                continue
            if body.get("record"):
                # Unsettled: nothing this process still holds of the run's
                # outcome is shown beside a status that is not terminal.
                run = {field: run[field] for field in _PENDING_FIELDS if field in run}
            run = {**run, **body}
        rows.append(_dashboard_row(run_id, run, now))
    return rows
