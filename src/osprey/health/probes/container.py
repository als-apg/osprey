"""The ``container`` health probe — deployed-container state and healthcheck.

Checks a single deployed container by name. It builds the runtime's ``ps``
command via :mod:`osprey.deployment.runtime_helper`, runs it off the event loop
with :func:`asyncio.create_subprocess_exec`, and maps the matched container's
state — and its healthcheck status, when the runtime reports one — to a
:class:`~osprey.health.models.CheckResult`:

- ``running`` (and not unhealthy) → ``ok``;
- ``running`` but the runtime reports it ``unhealthy`` → ``warning``;
- any other state → ``warning`` carrying the observed state;
- no matching container → ``warning`` "not deployed";
- no container runtime installed or running (:class:`RuntimeError` or
  :class:`FileNotFoundError`) → a single ``skip`` row "no container runtime
  available".

Container matching uses a fuzzy short-name rule: the target's last dotted
segment, lower-cased, is matched
against each container's ``Names`` with underscore/hyphen variants, so
``osprey.jupyter`` matches a container named ``project-osprey_jupyter-1``.

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

from osprey.deployment.runtime_helper import get_ps_command
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.probes import ProbeContext

_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_CATEGORY = "containers"


async def run(spec: Mapping[str, Any], ctx: ProbeContext) -> CheckResult:
    """Check one deployed container's state and healthcheck status.

    Args:
        spec: Parsed check parameters (see the module docstring for keys).
        ctx: Shared per-run handles; unused — the container probe never needs
            the control-system connector.

    Returns:
        A :class:`CheckResult` per the state/healthcheck mapping documented at
        the module level; a ``skip`` row when no container runtime is available.
    """
    del ctx  # container probe does not use the control-system connector

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

    try:
        ps_cmd = get_ps_command(all_containers=True)
    except (RuntimeError, FileNotFoundError) as exc:
        return CheckResult(
            name=name,
            category=category,
            status=Status.SKIP,
            message="no container runtime available",
            details=str(exc),
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

    matching = _match_containers(_parse_ps_json(stdout), short)
    if not matching:
        return CheckResult(
            name=name,
            category=category,
            status=Status.WARNING,
            message=f"{target}: not deployed",
            value="not found",
            latency_ms=latency_ms,
        )

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


def _match_containers(containers: list[dict[str, Any]], short: str) -> list[dict[str, Any]]:
    """Return containers whose ``Names`` fuzzily match ``short``.

    Matches the short name and its underscore/hyphen variants against each
    container's ``Names`` field (a list or a string), lower-cased.
    """
    variants = {short, short.replace("_", "-"), short.replace("-", "_")}
    matching: list[dict[str, Any]] = []
    for container in containers:
        names = container.get("Names", [])
        if isinstance(names, list):
            names_str = " ".join(str(n) for n in names).lower()
        else:
            names_str = str(names).lower()
        if any(variant in names_str for variant in variants):
            matching.append(container)
    return matching


def _extract_health(container: dict[str, Any]) -> str:
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
