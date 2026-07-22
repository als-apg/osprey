"""FastAPI dispatch API for the OSPREY dispatch worker service.

Exposes:
  POST /dispatch                      — enqueue an agent prompt, returns 202 + run_id
  GET  /dispatch/{id}                 — poll a run for completion (body carries artifact descriptors)
  GET  /dispatch/{id}/stream          — SSE stream of run events
  GET  /dispatch/{id}/artifacts       — descriptors of the artifacts the run produced
  GET  /dispatch/{id}/artifacts/{aid} — resolved (renderable) bytes of one artifact the run produced
  GET  /health                        — liveness check with run statistics
  GET  /dashboard/runs                — JSON feed consumed by the dispatcher dashboard (token-gated)

Which artifacts belong to a run is decided by the artifact store's write-time
``run_id`` tag (created-by), so all three artifact surfaces — the status-body
list, the list route, and the byte route — share one strict isolation gate and
cannot disagree. See ``osprey.agent_runner.artifact_resolve``.
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
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator

from osprey.agent_runner.artifact_resolve import (
    describe_run_artifacts,
    describe_run_input_artifacts,
    get_run_store,
    load_run_record,
    resolve_single_run_artifact,
)
from osprey.mcp_server.dispatch_worker import (
    counters,
    failure_class,
    retention,
    run_stats,
    sdk_runner,
)
from osprey.mcp_server.dispatch_worker.input_files_policy import (
    InputFilesError,
    sanitize_filename,
    validate_input_files,
)
from osprey.utils.tool_rules import matches_denylist

logger = logging.getLogger("osprey.mcp_server.dispatch_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DISPATCH_TIMEOUT_SEC = int(os.environ.get("DISPATCH_TIMEOUT_SEC", "300"))
_QUEUE_TTL_SEC = 60  # discard unconsumed SSE queues after this many seconds post-completion

# Explicit request-body ceiling. A legit /dispatch body carrying an input-files
# batch tops out near ~24 MB (base64 + prompt); 32 MB leaves ~25% headroom while
# bounding the memory a hostile caller can force the worker to buffer. Enforced
# from Content-Length before the body is read (see ``_limit_request_body``).
MAX_REQUEST_BYTES = 32 * 1024 * 1024

# Capabilities this worker advertises on /health. A downstream bridge gates
# feature use on BOTH the dispatcher's and the worker's /health carrying the
# capability, so the list is a plain JSON array on both bodies.
_CAPABILITIES: list[str] = ["input_files"]

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


# Agent-data root for THIS worker. Anchored to OSPREY_PROJECT_DIR rather than
# resolve_shared_data_root(): that helper locates the project via OSPREY_CONFIG
# and falls back to CWD, but the worker is configured with CONFIG_FILE and its
# CWD is the image WORKDIR (/app) — so it would resolve to /app/_agent_data
# while the agent's tools write to <project>/_agent_data.
def _agent_data_dir() -> str:
    return os.path.join(os.environ.get("OSPREY_PROJECT_DIR", "/app/project"), "_agent_data")


# Persistent log directory — lives on the _agent_data volume mount so it
# survives container restarts.
_LOG_DIR = os.path.join(_agent_data_dir(), "dispatch")


def _persist_run(run_id: str, run: dict[str, Any]) -> None:
    """Write a completed run to disk as JSON."""
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        path = os.path.join(_LOG_DIR, f"{run_id}.json")
        with open(path, "w") as f:
            json.dump({"run_id": run_id, **run}, f, indent=2, default=str)
    except Exception:
        logger.exception("Failed to persist run %s", run_id)


def _load_persisted_runs() -> None:
    """Load previously persisted runs from disk into _runs."""
    if not os.path.isdir(_LOG_DIR):
        return
    loaded = 0
    for fname in sorted(os.listdir(_LOG_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        if loaded >= _MAX_RUNS:
            break
        try:
            with open(os.path.join(_LOG_DIR, fname)) as f:
                data = json.load(f)
            run_id = data.pop("run_id", fname.removesuffix(".json"))
            _runs[run_id] = data
            loaded += 1
        except Exception:
            logger.warning("Failed to load %s", fname)
    if loaded:
        logger.info("Loaded %d persisted runs from %s", loaded, _LOG_DIR)


def _inject_provider_env_once() -> None:
    """Inject OSPREY provider env vars (auth, base URL, model tiers) into os.environ.

    Replicates what the OSPREY web server does at startup so the dispatch
    worker's SDK sessions use the same auth and model configuration.
    """
    project_dir = os.environ.get("OSPREY_PROJECT_DIR", "/app/project")
    config_path = os.path.join(project_dir, "config.yml")
    if not os.path.isfile(config_path):
        logger.warning("No config.yml at %s — skipping provider env injection", config_path)
        return

    # Managed (enterprise) policy settings outrank the process environment and
    # setting_sources=["project"] alike, so a policy `env` block setting a
    # provider variable would silently redirect the worker's agent to a backend
    # the project did not configure. Refuse to start — checked before the try
    # below so the broad except cannot swallow the refusal.
    from osprey.build.claude_code_resolver import (
        detect_managed_policy_conflicts,
        format_managed_policy_conflicts,
    )

    policy_conflicts = detect_managed_policy_conflicts()
    if policy_conflicts:
        raise RuntimeError(
            "Refusing to start the dispatch worker.\n"
            + format_managed_policy_conflicts(policy_conflicts)
        )

    try:
        from pathlib import Path

        from osprey.build.claude_code_resolver import inject_provider_env, load_provider_spec
        from osprey.build.claude_code_telemetry import TelemetryConfigError

        # load_provider_spec expands ${VAR} in provider config (e.g. a custom
        # provider's base_url: ${ARGO_PROD_URL}) against the container-mounted
        # /app/project/.env before resolving.
        #
        # Telemetry is an observability add-on; a misconfigured telemetry block
        # must never take down the worker's provider auth. Degrade telemetry
        # (loud ERROR) and re-resolve without it, keeping auth/base-url/model.
        try:
            spec = load_provider_spec(Path(project_dir))
        except TelemetryConfigError:
            logger.error(
                "claude_code.telemetry is misconfigured — continuing WITHOUT "
                "telemetry so provider auth is preserved. Fix the telemetry "
                "block to restore emit.",
                exc_info=True,
            )
            spec = load_provider_spec(Path(project_dir), include_telemetry=False)
        if spec:
            injected = inject_provider_env(os.environ, spec, project_dir=Path(project_dir))
            logger.info("Provider env injected: %s (provider=%s)", injected, spec.provider)

            # Non-native (OpenAI-protocol) providers need the in-process
            # translation proxy. Deliver the loopback via os.environ (matching
            # claude_cmd) because sdk_runner.build_clean_env copies os.environ
            # into the SDK env; the in-container proxy thread is reachable by the
            # SDK CLI. Start the proxy from spec.upstream_base_url — the OpenAI
            # root *with* /v1 — NOT os.environ["ANTHROPIC_BASE_URL"], which the
            # resolver strips of /v1 for Claude Code; sourcing the upstream from
            # the env var would forward to a /v1-less endpoint (issue #312).
            if spec.needs_proxy and spec.upstream_base_url:
                from osprey.infrastructure.proxy.lifecycle import start_proxy

                port = start_proxy(
                    spec.upstream_base_url,
                    os.environ.get(spec.auth_env_var),
                )
                os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
                logger.info("Translation proxy on :%d (provider=%s)", port, spec.provider)
        else:
            logger.warning("No provider configured in config.yml")
    except Exception:
        logger.exception("Failed to inject provider env from config.yml")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — provider env injection and run recovery.

    ``.claude/`` and ``data/`` (MCP config, safety hooks, skills) are baked into
    the project image at build time (``COPY .`` + ``osprey claude regen``), so
    the worker starts with those artifacts already in place — no runtime
    regeneration is needed here. Provider auth/model config and the translation
    proxy (for non-native providers) still have to be established at process
    startup, since they depend on the mounted ``config.yml`` and environment.
    """
    _inject_provider_env_once()
    # Route every _stamp's per-class increment into the process-lifetime counters
    # before any run can fail, so no early failure goes uncounted.
    counters.install()
    _load_persisted_runs()
    task = asyncio.create_task(_stale_run_cleanup())
    retention_task = _start_retention_sweep()
    yield
    task.cancel()
    if retention_task is not None:
        retention_task.cancel()


app = FastAPI(title="dispatch-worker", version="1.0.0", lifespan=_lifespan)


@app.middleware("http")
async def _limit_request_body(request: Request, call_next):  # noqa: ANN001, ANN202
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

# Server-side tool denylist — tools that must NEVER be used by headless dispatch.
# Defense-in-depth: the event dispatcher already restricts tools via triggers.yml,
# but the worker blocks dangerous tools regardless of what the trigger requests.
DENIED_TOOLS: set[str] = {
    "WebFetch",
    "WebSearch",
    "mcp__plugin_playwright_playwright__*",
    # Arbitrary shell access from a headless, unattended run is never warranted —
    # the safety story is the per-trigger allowlist + MCP tools, not a raw shell.
    # ``Bash`` runs commands; ``BashOutput`` reads a background shell's output;
    # ``KillShell`` (the current CLI name; older builds used ``KillBash``) kills
    # one. Deny all three.
    "Bash",
    "BashOutput",
    "KillShell",
    "KillBash",
}


def _is_denied(tool: str) -> bool:
    """Return True if ``tool`` is on the denylist.

    Entries ending in ``*`` match by prefix (e.g. the playwright entry blocks
    every ``mcp__plugin_playwright_playwright__<name>`` tool); all other entries
    match exactly.
    """
    return matches_denylist(tool, DENIED_TOOLS)


# ---------------------------------------------------------------------------
# Stale run cleanup
# ---------------------------------------------------------------------------


def _sweep_stale_runs() -> None:
    """One cleanup sweep: mark stale pending runs as error (cancelling their
    orphaned tasks) and discard old SSE queues.

    Extracted from the periodic loop so it is directly unit-testable without
    driving the ``while True`` / ``sleep`` loop.
    """
    now = time.time()
    stale_cutoff = DISPATCH_TIMEOUT_SEC + 30

    for run_id, run in list(_runs.items()):
        if run.get("status") == "pending":
            created = run.get("created_at", 0)
            if created and (now - created) > stale_cutoff:
                logger.warning(
                    "Marking stale run %s as error (pending > %ds)", run_id, stale_cutoff
                )
                swept = {
                    **run,
                    "status": "error",
                    "error": f"Timed out after {stale_cutoff}s",
                    "completed_at": now,
                    "tool_calls": run.get("tool_calls", []),
                }
                # A stale run is a worker-side fault: stamp it infrastructure with
                # the honest tool-call count and persist it as THE terminal record.
                # When this run's orphaned coroutine later unwinds with
                # CancelledError, that branch sees an already-terminal record and
                # merges instead of stamping again — one terminal record, one count.
                num_tool_calls = run_stats.get_run_stats(run_id)["num_tool_calls"]
                failure_class._stamp(swept, failure_class.FAILURE_INFRASTRUCTURE, num_tool_calls)
                _runs[run_id] = swept
                _persist_run(run_id, swept)
                # Cancel the orphaned task so its Claude CLI subprocess exits;
                # marking the run error alone would leave it running.
                task = _tasks.get(run_id)
                if task is not None and not task.done():
                    task.cancel()

    # Clean up unconsumed SSE queues for completed runs
    for run_id in list(_queues.keys()):
        run = _runs.get(run_id, {})
        completed_at = run.get("completed_at")
        if completed_at and (now - completed_at) > _QUEUE_TTL_SEC:
            del _queues[run_id]


async def _stale_run_cleanup() -> None:
    """Periodically run :func:`_sweep_stale_runs`."""
    while True:
        await asyncio.sleep(60)
        _sweep_stale_runs()


def _in_flight_run_ids() -> frozenset[str]:
    """Run ids of currently-pending (in-flight) runs — never swept by retention."""
    return frozenset(rid for rid, run in _runs.items() if run.get("status") == "pending")


def _start_retention_sweep() -> asyncio.Task | None:
    """Start the opt-in retention sweep task, or return ``None`` when disabled.

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
            _LOG_DIR,
            _get_store,
            retention_days,
            _in_flight_run_ids,
        )
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)) -> None:
    expected = os.environ.get("DISPATCH_WORKER_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DISPATCH_WORKER_TOKEN is not configured",
        )
    # Constant-time comparison to avoid leaking the token via timing, matching
    # the dispatcher's _check_auth / WebhookSource._handle.
    if not hmac.compare_digest(credentials.credentials, expected):
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
    """

    prompt: str
    allowed_tools: list[str]
    max_turns: int = 25
    surface_prompt: str | None = None
    surface_tools: list[str] | None = None
    input_files: list[InputFile] | None = None


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


async def _run_dispatch_task(run_id: str, request: DispatchRequest) -> None:
    queue = asyncio.Queue()
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
                denied_tools=DENIED_TOOLS,
                run_id=run_id,
                surface_prompt=request.surface_prompt,
                surface_tools=request.surface_tools,
            ),
            timeout=DISPATCH_TIMEOUT_SEC,
        )
        result["completed_at"] = time.time()
        result["prompt"] = request.prompt
        # Descriptors of the artifacts this run produced (created-by tag), for
        # consumers that republish them (see the /artifacts routes). Render-free:
        # computed once at completion from the store tag and persisted with the
        # run so it survives restart and never disagrees with the live routes.
        result["artifacts"] = describe_run_artifacts(run_id)
        # Caller-supplied inputs ingested for this run, kept separate from the
        # agent's produced ``artifacts`` (both read the same created-by store tag).
        result["input_artifacts"] = describe_run_input_artifacts(run_id)
        _runs[run_id] = result
        _persist_run(run_id, result)
        logger.info("Dispatch %s completed: status=%s", run_id, result.get("status"))
    except TimeoutError:
        logger.error("Dispatch %s timed out after %ds", run_id, DISPATCH_TIMEOUT_SEC)
        err_result = _build_stamped_error(
            run_id,
            f"Timed out after {DISPATCH_TIMEOUT_SEC}s",
            failure_class.FAILURE_INFRASTRUCTURE,
            extra={"duration_sec": DISPATCH_TIMEOUT_SEC},
        )
        _runs[run_id] = err_result
        _persist_run(run_id, err_result)
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
            try:
                await queue.put({"type": "error", "message": existing.get("error", "cancelled")})
            except Exception:
                pass
            raise
        # Genuine user cancel of a still-pending run: a run-level terminal state.
        logger.warning("Dispatch %s cancelled by user", run_id)
        err_result = _build_stamped_error(
            run_id,
            "cancelled by user",
            failure_class.FAILURE_RUN,
            extra={"cancelled": True},
        )
        _runs[run_id] = err_result
        _persist_run(run_id, err_result)
        try:
            await queue.put({"type": "error", "message": "cancelled by user"})
        except Exception:
            # Best-effort SSE notify during cancellation: the run is already
            # recorded as cancelled above, so a closed/full queue here is non-fatal.
            pass
        raise
    except Exception as exc:
        logger.error("Dispatch %s failed: %s", run_id, exc, exc_info=True)
        # Reached only for faults in the worker's own orchestration around the
        # run: run_dispatch handles provider/agent errors internally and returns
        # them already stamped, so anything raised out to this layer is
        # structurally an infrastructure fault (hence the literal class rather
        # than routing through classify_exception).
        err_result = _build_stamped_error(run_id, str(exc), failure_class.FAILURE_INFRASTRUCTURE)
        _runs[run_id] = err_result
        _persist_run(run_id, err_result)
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
    denied = [t for t in request.allowed_tools if _is_denied(t)]
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
    _runs[run_id] = {
        "status": "pending",
        "created_at": time.time(),
        "prompt": request.prompt,
    }
    # Use create_task (not BackgroundTasks) so we retain a handle for cancellation.
    _tasks[run_id] = asyncio.create_task(_run_dispatch_task(run_id, request))
    return DispatchResponse(status="accepted", run_id=run_id)


@app.delete("/dispatch/{run_id}", dependencies=[Depends(_verify_token)])
async def cancel_dispatch(run_id: str) -> dict[str, Any]:
    """Cancel an in-flight dispatch.

    Looks up the running asyncio.Task and requests cancellation. The run status
    transitions to `error` with `cancelled: true`. Measured cancel latency is
    ~0.6s end-to-end; a few seconds longer if a tool call is mid-flight.
    """
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    if run.get("status") != "pending":
        return {"run_id": run_id, "cancelled": False, "reason": "already finished"}

    task = _tasks.get(run_id)
    if task is None or task.done():
        return {"run_id": run_id, "cancelled": False, "reason": "no task handle"}

    task.cancel()
    logger.info("Dispatch %s cancellation requested", run_id)
    return {"run_id": run_id, "cancelled": True}


@app.get("/dispatch/{run_id}", dependencies=[Depends(_verify_token)])
async def get_dispatch_result(run_id: str) -> dict[str, Any]:
    """Poll a run by its run_id. Returns the stored result dict.

    Falls through to the persisted on-disk record when the run has aged out of
    the in-memory ``_runs`` cap, so the status body (and its ``artifacts``
    descriptors) stays available for exactly as long as the artifact routes,
    which resolve from the same on-disk store.
    """
    result = _runs.get(run_id)
    if result is None:
        result = load_run_record(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    return result


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
    converted to PNG here (only the requested one); images/PDFs pass through.
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


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check with run statistics — no auth required."""
    counts = {"pending": 0, "completed": 0, "error": 0}
    for run in _runs.values():
        s = run.get("status", "")
        if s in counts:
            counts[s] += 1
    # Process-lifetime failure counters — monotonic and independent of the
    # evictable _runs map above (see counters module docstring). The _runs-derived
    # error_runs field is retained unchanged for compatibility.
    lifetime = counters.get_counts()
    return {
        "status": "ok",
        "pending_runs": counts["pending"],
        "completed_runs": counts["completed"],
        "error_runs": counts["error"],
        "total_runs": len(_runs),
        "provider_errors": lifetime[failure_class.FAILURE_PROVIDER],
        "infrastructure_errors": lifetime[failure_class.FAILURE_INFRASTRUCTURE],
        "boot_nonce": _BOOT_NONCE,
        "capabilities": _CAPABILITIES,
    }


# ---------------------------------------------------------------------------
# Dashboard runs API (token-gated, consumed by the dispatcher server-side)
# ---------------------------------------------------------------------------


@app.get("/dashboard/runs", dependencies=[Depends(_verify_token)])
async def dashboard_runs() -> list[dict[str, Any]]:
    """Recent runs for the dispatcher dashboard — token-gated, most recent first.

    Returns full text_output/error per run, so this is bearer-gated like the
    other worker endpoints. The dispatcher (the only caller) holds the token and
    proxies this feed to its own browser-facing dashboard.
    """
    now = time.time()
    runs = []
    for run_id, run in _runs.items():
        created = run.get("created_at", 0)
        runs.append(
            {
                "run_id": run_id,
                "status": run.get("status", "unknown"),
                "created_at": created,
                "age_sec": round(now - created, 1) if created else None,
                "duration_sec": run.get("duration_sec"),
                "text_output": run.get("text_output") or "",
                "tool_count": len(run.get("tool_calls", [])),
                "error": run.get("error"),
                "num_turns": run.get("num_turns"),
                "has_stream": run_id in _queues,
            }
        )
    runs.sort(key=lambda r: r["created_at"] or 0, reverse=True)
    return runs[:50]
