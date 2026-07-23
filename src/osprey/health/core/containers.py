"""Core ``containers`` health category.

Reports container-runtime availability and per-service container state by
composing the declarative :mod:`~osprey.health.probes.container` probe. The
category emits:

* ``<runtime>_available`` — ``ok`` with the ``docker``/``podman`` ``--version``
  banner, or ``error`` if that command fails (allowlisted to ``error``);
* one ``container_<service>`` row per entry in ``deployed_services``, produced by
  the container probe (``running`` → ``ok``; other state or missing → ``warning``).

Two whole-category shortcuts replace per-service rows:

* **No container runtime** (``get_runtime_command`` raises ``RuntimeError`` or
  ``FileNotFoundError``) → a single ``skip`` row ``container_runtime`` "no
  container runtime available" (exit 0, no per-service rows).
* **Degraded mode** (``config is None`` — the project's ``config.yml`` did not
  load) → the runtime-availability row still runs, and the per-service portion
  collapses to a single ``skip`` row ``container_services`` "config unavailable",
  since the service list is unknown.

A broken runtime (``--version`` fails) yields only the ``error``
``<runtime>_available`` row; the container probe is not run.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from osprey.deployment.runtime_helper import get_runtime_command
from osprey.health.models import CheckResult, Status
from osprey.health.probes import ProbeContext
from osprey.health.probes.container import run as container_probe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.probes import ProbeCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "containers"

_VERSION_TIMEOUT_S = 5.0


def containers(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    probe: ProbeCallable | None = None,
) -> CategoryCallable:
    """Build the ``containers`` category callable.

    Args:
        config: Parsed config mapping, or ``None`` when config is unavailable
            (degraded mode). Read for ``deployed_services``.
        context: Health runtime; forwarded to the container probe, which ignores
            it (no control-system connector is needed).
        probe: Container probe callable to compose, for dependency injection in
            tests; defaults to :func:`osprey.health.probes.container.run`.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    run_probe: ProbeCallable = probe or container_probe
    ctx = ProbeContext(runtime=cast("HealthRuntime", context))

    async def _run() -> list[CheckResult]:
        try:
            runtime_cmd = get_runtime_command()
        except (RuntimeError, FileNotFoundError) as exc:
            return [
                CheckResult(
                    "container_runtime",
                    CATEGORY,
                    Status.SKIP,
                    "no container runtime available",
                    details=str(exc),
                )
            ]

        runtime = runtime_cmd[0]
        available = await _check_runtime_available(runtime)
        rows = [available]
        if available.status is Status.ERROR:
            return rows  # broken runtime — do not probe individual services

        if config is None:
            rows.append(
                CheckResult(
                    "container_services",
                    CATEGORY,
                    Status.SKIP,
                    "config unavailable",
                    details="deployed_services unknown; config.yml did not load.",
                )
            )
            return rows

        deployed = config.get("deployed_services", []) or []
        if deployed:
            specs = [
                {"container": svc, "name": f"container_{svc}", "category": CATEGORY}
                for svc in deployed
            ]
            probe_rows = await asyncio.gather(*(run_probe(spec, ctx) for spec in specs))
            rows.extend(probe_rows)
        return rows

    return _run


async def _check_runtime_available(runtime: str) -> CheckResult:
    """Report the runtime's ``--version`` banner as ``<runtime>_available``.

    ``ok`` carries the version string; a non-zero exit, missing binary, or
    timeout yields an ``error`` row (the sole allowlisted error in this category).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            runtime,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, RuntimeError):
        return CheckResult(
            f"{runtime}_available", CATEGORY, Status.ERROR, f"{runtime.capitalize()} command failed"
        )

    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=_VERSION_TIMEOUT_S)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return CheckResult(
            f"{runtime}_available",
            CATEGORY,
            Status.ERROR,
            f"{runtime.capitalize()} --version timed out",
        )

    if (proc.returncode or 0) == 0:
        version = stdout_b.decode(errors="replace").strip()
        return CheckResult(f"{runtime}_available", CATEGORY, Status.OK, version)
    return CheckResult(
        f"{runtime}_available", CATEGORY, Status.ERROR, f"{runtime.capitalize()} command failed"
    )
