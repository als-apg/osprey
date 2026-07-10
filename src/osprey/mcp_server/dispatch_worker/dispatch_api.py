"""FastAPI dispatch API for the OSPREY dispatch worker service.

Exposes:
  POST /dispatch                      — enqueue an agent prompt, returns 202 + run_id
  GET  /dispatch/{id}                 — poll a run for completion
  GET  /dispatch/{id}/stream          — SSE stream of run events
  GET  /dispatch/{id}/artifacts/{aid} — raw bytes of one artifact the run produced
  GET  /health                        — liveness check with run statistics
  GET  /dashboard/runs                — JSON feed consumed by the dispatcher dashboard (token-gated)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from osprey.mcp_server.dispatch_worker import sdk_runner
from osprey.utils.tool_rules import matches_denylist

logger = logging.getLogger("osprey.mcp_server.dispatch_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DISPATCH_TIMEOUT_SEC = int(os.environ.get("DISPATCH_TIMEOUT_SEC", "300"))
_QUEUE_TTL_SEC = 60  # discard unconsumed SSE queues after this many seconds post-completion

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

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

    try:
        from pathlib import Path

        from osprey.cli.claude_code_resolver import inject_provider_env, load_provider_spec

        # load_provider_spec expands ${VAR} in provider config (e.g. a custom
        # provider's base_url: ${ARGO_PROD_URL}) against the container-mounted
        # /app/project/.env before resolving.
        spec = load_provider_spec(Path(project_dir))
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


def _provision_claude_artifacts_once() -> None:
    """Regenerate Claude Code artifacts (.mcp.json, .claude/, CLAUDE.md) in the project dir.

    The deployed worker mounts only ``config.yml`` into the project dir, so the
    project has no ``.mcp.json`` (MCP servers like ``osprey_workspace``) and no
    ``.claude/`` (safety hooks, settings, skills). Without them the dispatched
    agent runs with built-in tools only — its trigger ``allowed_tools`` that name
    ``mcp__osprey_workspace__*`` silently resolve to nothing, and the facility's
    PreToolUse safety hooks are absent.

    Regenerate from the mounted ``config.yml`` at startup so the container's
    project matches a normally-built one. The staged config has ``python_env_path``
    stripped (see compose_generator), so MCP-server commands resolve to the
    container's own interpreter rather than the build host's. Best-effort: a
    failure here must not stop the worker from serving no-tool triggers.

    Only provisions when ``.mcp.json`` is ABSENT. A normally-built project (the
    subprocess path, or a non-container deploy where the worker runs in the real
    project dir) already ships container-correct artifacts and may carry user
    customizations (e.g. via ``osprey eject``); regenerating would overwrite them.
    The container is the one case where the project dir has only ``config.yml``.
    """
    project_dir = os.environ.get("OSPREY_PROJECT_DIR", "/app/project")
    config_path = os.path.join(project_dir, "config.yml")
    if not os.path.isfile(config_path):
        logger.warning("No config.yml at %s — skipping Claude artifact provisioning", config_path)
        return
    if os.path.isfile(os.path.join(project_dir, ".mcp.json")):
        # Already provisioned (built project / non-container deploy) — don't
        # clobber existing (possibly customized) artifacts.
        return
    try:
        from pathlib import Path

        from osprey.cli.templates.manager import TemplateManager

        result = TemplateManager().regenerate_claude_code(Path(project_dir))
        logger.info(
            "Provisioned Claude Code artifacts in %s: %d file(s) generated",
            project_dir,
            len(result.get("changed", [])),
        )
    except Exception:
        logger.exception(
            "Failed to provision Claude Code artifacts — dispatched agents will "
            "run with built-in tools only (no project MCP servers or safety hooks)"
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — provider env, artifact provisioning, recovery."""
    _inject_provider_env_once()
    _provision_claude_artifacts_once()
    _load_persisted_runs()
    task = asyncio.create_task(_stale_run_cleanup())
    yield
    task.cancel()


app = FastAPI(title="dispatch-worker", version="1.0.0", lifespan=_lifespan)

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
                _runs[run_id] = {
                    **run,
                    "status": "error",
                    "error": f"Timed out after {stale_cutoff}s",
                    "completed_at": now,
                }
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


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _artifact_store():
    """ArtifactStore rooted at this worker's agent-data dir.

    Not session-scoped: the dispatched agent's tools write there (the worker
    process sets no OSPREY_SESSION_ID), and it is the same root the artifact
    gallery reads. Built per call — the index is re-read from disk, so
    artifacts written by the agent's MCP subprocesses are visible here.
    """
    from pathlib import Path

    from osprey.stores.artifact_store import ArtifactStore

    return ArtifactStore(workspace_root=Path(_agent_data_dir()))


def _artifact_ids_for_run(run_id: str) -> list[str]:
    """Ids of the artifacts this run produced, oldest first.

    Strict equality on ``run_id`` — deliberately NOT
    ``list_entries(session_filter=...)``, which also matches entries with an
    EMPTY session id and would therefore attach every untagged legacy artifact
    in the store to every run.
    """
    try:
        entries = _artifact_store().list_entries()
    except Exception:
        logger.warning("Could not read artifact store for run %s", run_id, exc_info=True)
        return []
    return [e.id for e in entries if e.run_id == run_id]


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


class DispatchRequest(BaseModel):
    prompt: str
    allowed_tools: list[str]
    max_turns: int = 25


class DispatchResponse(BaseModel):
    status: str
    run_id: str


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


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
        result = await asyncio.wait_for(
            sdk_runner.run_dispatch(
                prompt=request.prompt,
                allowed_tools=request.allowed_tools,
                max_turns=request.max_turns,
                event_queue=queue,
                denied_tools=DENIED_TOOLS,
                run_id=run_id,
            ),
            timeout=DISPATCH_TIMEOUT_SEC,
        )
        result["completed_at"] = time.time()
        result["prompt"] = request.prompt
        # Artifacts the agent saved during this run, for consumers that
        # republish them (see GET /dispatch/{run_id}/artifacts/{artifact_id}).
        result["artifacts"] = _artifact_ids_for_run(run_id)
        _runs[run_id] = result
        _persist_run(run_id, result)
        logger.info("Dispatch %s completed: status=%s", run_id, result.get("status"))
    except TimeoutError:
        logger.error("Dispatch %s timed out after %ds", run_id, DISPATCH_TIMEOUT_SEC)
        err_result = {
            "status": "error",
            "text_output": "",
            "tool_calls": [],
            "error": f"Timed out after {DISPATCH_TIMEOUT_SEC}s",
            "duration_sec": DISPATCH_TIMEOUT_SEC,
            "cost_usd": None,
            "num_turns": None,
            "completed_at": time.time(),
        }
        _runs[run_id] = err_result
        _persist_run(run_id, err_result)
        await queue.put({"type": "error", "message": f"Timed out after {DISPATCH_TIMEOUT_SEC}s"})
    except asyncio.CancelledError:
        logger.warning("Dispatch %s cancelled by user", run_id)
        err_result = {
            "status": "error",
            "text_output": "",
            "tool_calls": [],
            "error": "cancelled by user",
            "cancelled": True,
            "duration_sec": 0.0,
            "cost_usd": None,
            "num_turns": None,
            "completed_at": time.time(),
        }
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
        err_result = {
            "status": "error",
            "text_output": "",
            "tool_calls": [],
            "error": str(exc),
            "duration_sec": 0.0,
            "cost_usd": None,
            "num_turns": None,
            "completed_at": time.time(),
        }
        _runs[run_id] = err_result
        _persist_run(run_id, err_result)
        await queue.put({"type": "error", "message": str(exc)})
    finally:
        _tasks.pop(run_id, None)


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
    """Poll a run by its run_id. Returns the stored result dict."""
    result = _runs.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")
    return result


@app.get("/dispatch/{run_id}/artifacts/{artifact_id}", dependencies=[Depends(_verify_token)])
async def get_dispatch_artifact(run_id: str, artifact_id: str) -> FileResponse:
    """Serve the raw bytes of one artifact produced by ``run_id``.

    The artifact must be attributed to this run: an artifact belonging to a
    different run (or to no run) is a 404 here, so a caller holding one run_id
    cannot enumerate or read another run's output. ``artifact_id`` is never
    joined into a path directly — it is resolved through the store's index, so
    a traversal-shaped id simply fails to match an entry.
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")

    store = _artifact_store()
    entry = store.get_entry(artifact_id)
    if entry is None or entry.run_id != run_id:
        raise HTTPException(status_code=404, detail=f"artifact {artifact_id!r} not found for run")

    path = store.get_file_path(artifact_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"artifact {artifact_id!r} has no file")

    return FileResponse(
        path,
        media_type=entry.mime_type or "application/octet-stream",
        filename=entry.filename,
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
    return {
        "status": "ok",
        "pending_runs": counts["pending"],
        "completed_runs": counts["completed"],
        "error_runs": counts["error"],
        "total_runs": len(_runs),
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
