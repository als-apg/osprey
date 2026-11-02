"""Core ``simulator`` health category.

One row per served physics model on each simulator the deployment serves from
a process of its own -- the ``va`` target when its virtual accelerator runs in
a container, and the ``standin`` target -- **only when the render carries a
simulator view** that names a physics model. A deployment with no view, or one
whose simulator runs only in process, contributes no rows: an in-process
simulator lives in each agent process, and ``osprey sim status`` reports it.

Each row reads the model's status channel, ``<code>:SIM:<model>:STATUS``,
through the target's own connector, the same read ``osprey sim status
--target`` makes. Each target is read in a child process: categories run
concurrently and a process has one Channel Access context, so connecting to a
target here would repoint the connector the rest of the suite reads through.

* ``ok`` -- the model serves;
* ``ok`` -- the model names no deck and wires no channel, so it is not built,
  by design; the message says so, so it is never mistaken for a served model;
* ``error`` -- the model failed; the message names it and its error. Its
  readbacks serve NaN while every other model and the texture serve on, which
  is why the container's own healthcheck stays green and only this row says
  so;
* ``warning`` -- the target's status channels could not be read (the server
  is down or unreachable); the containers category reports the server itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "simulator"

#: Seconds one target's child may take to start and answer for all its models.
_READ_TIMEOUT_S = 20.0


def simulator(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    cwd: Path | None = None,
) -> CategoryCallable:
    """Build the ``simulator`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
            Read for the ``control_system`` section.
        context: Health runtime. Unused -- each target is read through a
            connector built for it, as ``osprey sim status --target`` builds one.
        cwd: The deployment's repo root, under whose render the simulator
            view is found.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}
    repo_root = Path(cwd) if cwd is not None else Path.cwd()

    async def _run() -> list[CheckResult]:
        from osprey_connectors.simulation.view import SimulatorView, ViewSchemaError

        section = cfg.get("control_system") or {}
        targets = _served_targets(section)
        if not targets:
            return []
        try:
            view = SimulatorView.find(repo_root)
        except (OSError, ValueError, ViewSchemaError):
            view = None
        if view is None:
            return []
        models = [model.name for model in view.physics_models()]
        if not models:
            return []
        addresses = dict(view.status_addresses())
        from osprey_connectors.workspace import rendered_config_path

        config_path = rendered_config_path(repo_root)
        probed = await asyncio.gather(
            *(_probe(section, target, models, addresses, config_path) for target in targets)
        )
        return [row for rows in probed for row in rows]

    return _run


def _served_targets(section: Any) -> list[str]:
    """The simulator targets this deployment serves from a process of its own."""
    from osprey_connectors.types import (
        TARGET_STANDIN,
        TARGET_VA,
        TRANSPORT_IN_PROCESS,
        configured_targets,
        connector_transport,
        resolve_target,
    )

    try:
        configured = configured_targets(section)
    except ValueError:
        return []
    served: list[str] = []
    for target in configured:
        if target not in (TARGET_VA, TARGET_STANDIN):
            continue
        try:
            transport = connector_transport(section, resolve_target(section, target))
        except ValueError:
            continue
        if transport != TRANSPORT_IN_PROCESS:
            served.append(target)
    return served


async def _probe(
    section: Any, target: str, models: list[str], addresses: Mapping[str, str], config_path: Path
) -> list[CheckResult]:
    """Read every model's status on one target and turn each into a row."""
    from osprey.simulation.status import read_in_child
    from osprey_connectors.simulation import MODEL_NO_DECK, MODEL_OK, model_state

    start = perf_counter()
    try:
        statuses = await read_in_child(
            section, target, models, addresses, config_path=config_path, timeout_s=_READ_TIMEOUT_S
        )
    except Exception as exc:  # any read failure is one advisory row for the target
        return [
            CheckResult(
                f"{CATEGORY}_{target}",
                CATEGORY,
                Status.WARNING,
                f"{target}: the model status channels could not be read",
                details=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
                probed=True,
            )
        ]
    latency_ms = (perf_counter() - start) * 1000.0
    rows: list[CheckResult] = []
    for model in models:
        status = statuses[model]
        state = model_state(status)
        name = f"{CATEGORY}_{target}_{model}"
        if state == MODEL_OK:
            rows.append(
                CheckResult(
                    name,
                    CATEGORY,
                    Status.OK,
                    f"{model} on {target}: ok",
                    value=status,
                    latency_ms=latency_ms,
                    probed=True,
                )
            )
        elif state == MODEL_NO_DECK:
            rows.append(
                CheckResult(
                    name,
                    CATEGORY,
                    Status.OK,
                    f"{model} on {target}: no deck, not built by design",
                    value=status,
                    latency_ms=latency_ms,
                    probed=True,
                    details="The model names no deck and wires no channel; "
                    "its channels are served by the texture.",
                )
            )
        else:
            rows.append(
                CheckResult(
                    name,
                    CATEGORY,
                    Status.ERROR,
                    f"{model} on {target}: failed",
                    value="failed",
                    latency_ms=latency_ms,
                    probed=True,
                    details=(
                        f"{status}\nIts readbacks serve NaN; every other model and the "
                        "texture serve on. 'osprey sim status --target "
                        f"{target}' names the model log."
                    ),
                )
            )
    return rows
