"""The ``container`` health probe — deployed-container state and healthcheck.

Checks one deployed service's containers. It builds the runtime's ``ps``
command via :mod:`osprey.deployment.runtime_helper` — passing the probe's config
so the ``container_runtime`` the project configured is the one queried, rather
than whatever auto-detection finds first — runs it off the event loop
with :func:`asyncio.create_subprocess_exec`, and maps the matched containers'
state — and their healthcheck status, when the runtime reports one — to a
:class:`~osprey.health.models.CheckResult`. For a single matched container:

- ``running`` (and not unhealthy) → ``ok``;
- ``running`` but the runtime reports it ``unhealthy`` → ``warning``;
- any other state → ``warning`` carrying the observed state;
- no matching container → ``warning`` "not deployed";
- no container runtime installed or running (:class:`RuntimeError` or
  :class:`FileNotFoundError`) → a single ``skip`` row "no container runtime
  available";
- no config on the context and none loadable from the global config → a
  ``skip`` row "config unavailable", since without the project name this
  deployment's containers cannot be told from another's.

Which containers are graded is decided by
:func:`osprey.deployment.container_ownership.deployment_containers`: a container
labelled for this project is ours, one labelled for another OSPREY or compose
project never is, and an unlabelled one is ours when a name matches the target
by whole name segments. The target's last dotted segment is matched against
each container's names after the ``<project>-`` prefix and against its compose
service label, so ``services.archiver_recorder`` matches
``project-archiver-recorder-1`` and ``postgresql`` matches the
``project-ariel-postgres`` container compose runs for that service.

Every matched container is graded, since one service can run several (numbered
workers, per-lane stacks). With more than one, any container not ``running``
→ ``warning`` naming each such container; all running but some ``unhealthy`` →
``warning`` naming them; otherwise ``ok``. ``details`` lists every matched
container with its state.

Spec keys:
    container: The container/service name to look up (required; alias ``service``).
    name: Result-row name; defaults to ``"container.<short>"``.
    category: Result category; defaults to ``"containers"``.
    timeout_s: Seconds to await the ``ps`` invocation before giving up (default 10).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.container_ownership import container_names, deployment_containers
from osprey.deployment.runtime_helper import get_ps_command
from osprey.health.models import CheckResult, Status
from osprey.utils.config import get_full_configuration

if TYPE_CHECKING:
    from osprey.health.probes import ProbeContext

_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_CATEGORY = "containers"


async def run(spec: Mapping[str, Any], ctx: ProbeContext) -> CheckResult:
    """Check one deployed container's state and healthcheck status.

    Args:
        spec: Parsed check parameters (see the module docstring for keys).
        ctx: Shared per-run handles. Only ``ctx.config`` is read — it selects the
            container runtime to query and names the project whose containers
            are graded; the control-system connector is never needed here.

    Returns:
        A :class:`CheckResult` per the state/healthcheck mapping documented at
        the module level; a ``skip`` row when no container runtime or no config
        is available.
    """
    category = str(spec.get("category", _DEFAULT_CATEGORY))
    target = spec.get("container") or spec.get("service")
    if not target:
        return CheckResult(
            name=str(spec.get("name") or "container"),
            category=category,
            status=Status.ERROR,
            message="container check requires a 'container' (or 'service') name",
        )

    short = str(target).split(".")[-1].lower()
    name = str(spec.get("name") or f"container.{short}")
    timeout_s = float(spec.get("timeout_s", _DEFAULT_TIMEOUT_S))

    start = time.perf_counter()
    config = _probe_config(ctx)

    try:
        ps_cmd = get_ps_command(config, all_containers=True)
    except (RuntimeError, FileNotFoundError) as exc:
        return CheckResult(
            name=name,
            category=category,
            status=Status.SKIP,
            message="no container runtime available",
            details=str(exc),
        )

    if config is None:
        return CheckResult(
            name=name,
            category=category,
            status=Status.SKIP,
            message="config unavailable",
            details="cannot tell this deployment's containers from another's",
        )

    try:
        stdout, returncode = await _run_ps(ps_cmd, timeout_s)
    except (FileNotFoundError, RuntimeError) as exc:
        # The runtime binary disappeared between detection and exec.
        return CheckResult(
            name=name,
            category=category,
            status=Status.SKIP,
            message="no container runtime available",
            details=str(exc),
        )
    except TimeoutError:
        return CheckResult(
            name=name,
            category=category,
            status=Status.WARNING,
            message=f"{target}: container query timed out after {timeout_s:g}s",
            latency_ms=_elapsed_ms(start),
        )

    latency_ms = _elapsed_ms(start)

    if returncode != 0:
        return CheckResult(
            name=name,
            category=category,
            status=Status.WARNING,
            message=f"{target}: container query failed (exit {returncode})",
            latency_ms=latency_ms,
        )

    matching = deployment_containers(
        _parse_ps_json(stdout), project_name=resolve_project_name(config), services=(short,)
    ).for_service(short)
    if not matching:
        return CheckResult(
            name=name,
            category=category,
            status=Status.WARNING,
            message=f"{target}: not deployed",
            value="not found",
            latency_ms=latency_ms,
        )

    if len(matching) > 1:
        return _grade_several(matching, target, name, category, latency_ms)

    container = matching[0]
    state = str(container.get("State", "unknown"))
    health = _extract_health(container)

    if state == "running":
        if health == "unhealthy":
            return CheckResult(
                name=name,
                category=category,
                status=Status.WARNING,
                message=f"{target}: running but unhealthy",
                value=state,
                latency_ms=latency_ms,
            )
        detail = f" ({health})" if health else ""
        return CheckResult(
            name=name,
            category=category,
            status=Status.OK,
            message=f"{target}: running{detail}",
            value=state,
            latency_ms=latency_ms,
        )

    return CheckResult(
        name=name,
        category=category,
        status=Status.WARNING,
        message=f"{target}: {state}",
        value=state,
        latency_ms=latency_ms,
    )


def _probe_config(ctx: ProbeContext) -> Mapping[str, Any] | None:
    """The config this probe reads, or ``None`` when none is available.

    ``ctx.config`` is authoritative when present; otherwise the global
    configuration is loaded, and any failure to load it yields ``None``.
    """
    if ctx.config is not None:
        return ctx.config
    try:
        return get_full_configuration()
    except Exception:  # config unavailability degrades to a skip row
        return None


def _grade_several(
    matching: list[Mapping[str, Any]],
    target: object,
    name: str,
    category: str,
    latency_ms: float,
) -> CheckResult:
    """Grade a service that runs more than one container; the worst state wins.

    Args:
        matching: The service's containers, in reporting order.
        target: The service name as the spec gave it, for the message.
        name: Result-row name.
        category: Result category.
        latency_ms: Duration of the ``ps`` query.

    Returns:
        ``warning`` naming each container that is not running, else ``warning``
        naming each unhealthy one, else ``ok``; ``details`` lists every container.
    """
    rows = [
        (
            (container_names(c) or ["unknown"])[0],
            str(c.get("State", "unknown")),
            _extract_health(c),
        )
        for c in matching
    ]
    count = len(rows)
    details = "; ".join(
        f"{row_name}: {state}" + (f" ({health})" if health else "")
        for row_name, state, health in rows
    )
    stopped = [(row_name, state) for row_name, state, _ in rows if state != "running"]
    if stopped:
        listed = ", ".join(f"{row_name}: {state}" for row_name, state in stopped)
        return CheckResult(
            name=name,
            category=category,
            status=Status.WARNING,
            message=f"{target}: {len(stopped)} of {count} containers not running ({listed})",
            value=stopped[0][1],
            details=details,
            latency_ms=latency_ms,
        )
    unhealthy = [row_name for row_name, _, health in rows if health == "unhealthy"]
    if unhealthy:
        return CheckResult(
            name=name,
            category=category,
            status=Status.WARNING,
            message=(
                f"{target}: {len(unhealthy)} of {count} containers running but unhealthy "
                f"({', '.join(unhealthy)})"
            ),
            value="running",
            details=details,
            latency_ms=latency_ms,
        )
    return CheckResult(
        name=name,
        category=category,
        status=Status.OK,
        message=f"{target}: {count} containers running",
        value="running",
        details=details,
        latency_ms=latency_ms,
    )


async def _run_ps(ps_cmd: list[str], timeout_s: float) -> tuple[str, int]:
    """Run the ``ps`` command off the loop, returning ``(stdout, returncode)``.

    Raises:
        TimeoutError: If the command does not complete within ``timeout_s``; the
            child process is killed and reaped before the error propagates.
    """
    proc = await asyncio.create_subprocess_exec(
        *ps_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # already exited
        await proc.wait()
        raise
    return stdout_b.decode(errors="replace"), proc.returncode or 0


def _elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since ``start`` (a ``perf_counter`` reading)."""
    return (time.perf_counter() - start) * 1000.0


def _parse_ps_json(stdout: str) -> list[dict[str, Any]]:
    """Parse ``ps --format json`` output (Podman array or Docker NDJSON).

    Podman emits a single JSON array; Docker emits one JSON object per line.
    Unparseable lines are skipped rather than aborting the whole probe.
    """
    text = stdout.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if data is not None:
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict)]
        return [data] if isinstance(data, dict) else []

    containers: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            containers.append(obj)
    return containers


def _extract_health(container: Mapping[str, Any]) -> str:
    """Return the container's healthcheck status, or ``""`` if none is reported.

    Prefers an explicit ``Health`` field, falling back to parsing the human
    ``Status`` string (e.g. ``"Up 2 hours (healthy)"``). ``"unhealthy"`` is
    tested before ``"healthy"`` because the latter is its substring.
    """
    health = container.get("Health")
    if isinstance(health, str) and health.strip():
        return health.strip().lower()
    status_str = str(container.get("Status", "")).lower()
    for token in ("unhealthy", "healthy", "starting"):
        if token in status_str:
            return token
    return ""
