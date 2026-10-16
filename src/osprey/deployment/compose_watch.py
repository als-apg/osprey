"""Watch a ``compose up`` while it runs, and explain it when it never finishes.

``compose up`` can wait forever. podman-compose implements
``depends_on: {condition: service_healthy}`` by retrying ``podman wait
--condition=healthy`` once a second with no limit, so a dependency that was
never created (a full disk) or never becomes healthy (a service that crashes at
startup) holds the whole deploy, and the reason is only in the spool file.
Docker Compose fails instead, but a start can still take long enough that
silence reads as a hang.

:func:`watched_compose_up` puts three things around that invocation:

* every :data:`WATCH_INTERVAL_S` it prints which of the project's containers
  are not up yet, and the last line compose wrote;
* it stops after the time limit (:func:`compose_up_timeout`) and attaches the
  diagnosis to the failure: what was still pending, and the log tail of every
  container that exited or is unhealthy;
* on Ctrl-C it prints the same diagnosis before the interrupt propagates.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.cli.phase_reporter import current_reporter, format_elapsed, report_note
from osprey.deployment.errors import CapturedProcessTimeout
from osprey.deployment.runtime_helper import container_log_tail
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")

#: The config key and environment variable that set :func:`compose_up_timeout`.
COMPOSE_UP_TIMEOUT_KEY = "compose_up_timeout"
COMPOSE_UP_TIMEOUT_ENV = "OSPREY_COMPOSE_UP_TIMEOUT"

#: How long one ``compose up`` may run before it is stopped. Generous on
#: purpose: it is a hang guard, not a performance budget, and an ``up`` that
#: pulls images over a slow link legitimately takes many minutes. The
#: still-waiting notes make a stall visible long before it is reached.
DEFAULT_COMPOSE_UP_TIMEOUT_S = 1800.0

#: Seconds between still-waiting notes.
WATCH_INTERVAL_S = 30.0

#: How much of a stuck container's log the diagnosis carries.
STALL_LOG_TAIL_LINES = 20

#: How much of compose's last line a note carries.
_LAST_LINE_LIMIT = 160

#: How much of the spool's end is read to find its last line.
_SPOOL_TAIL_BYTES = 8192


def positive_seconds(
    config: Mapping[str, Any] | None, *, key: str, env_var: str, default: float
) -> float:
    """A positive number of seconds from ``env_var``, else config ``key``, else ``default``.

    The variable wins over the key, so one shell can lengthen a limit a config
    pins. A value that is not a positive number is refused with the setting
    that holds it: a typo here would otherwise become a limit nobody chose.

    :raises ValueError: The variable or the key holds something other than a
        positive number.
    """
    raw: Any = os.environ.get(env_var)
    source = env_var
    if raw is None or not str(raw).strip():
        raw = (config or {}).get(key)
        source = key
    if raw is None:
        return default
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = 0.0
    if not seconds > 0 or isinstance(raw, bool):
        raise ValueError(f"{source} must be a positive number of seconds, not {raw!r}")
    return seconds


def compose_up_timeout(config: Mapping[str, Any] | None) -> float:
    """Seconds one ``compose up`` may run: ``OSPREY_COMPOSE_UP_TIMEOUT``, the config's
    ``compose_up_timeout``, or :data:`DEFAULT_COMPOSE_UP_TIMEOUT_S`.
    """
    return positive_seconds(
        config,
        key=COMPOSE_UP_TIMEOUT_KEY,
        env_var=COMPOSE_UP_TIMEOUT_ENV,
        default=DEFAULT_COMPOSE_UP_TIMEOUT_S,
    )


@dataclass(frozen=True)
class ContainerState:
    """One of the project's containers as the runtime lists it.

    Attributes:
        name: The container's name.
        status: The runtime's own status text (``Up 3 minutes (unhealthy)``).
    """

    name: str
    status: str

    @property
    def state(self) -> str:
        """The status in one word or two: ``created``, ``starting``, ``exited (1)``, ...

        Empty for a container that is up and, if it has a healthcheck, healthy:
        those are not waited on.
        """
        text = self.status.strip()
        lowered = text.lower()
        if lowered.startswith("up"):
            if "unhealthy" in lowered:
                return "unhealthy"
            if "starting" in lowered:
                return "starting"
            return ""
        if lowered.startswith("exited"):
            code = text.split("(", 1)[1].split(")", 1)[0] if "(" in text else ""
            return f"exited ({code})" if code else "exited"
        return lowered.split(" ", 1)[0] if lowered else "unknown"

    @property
    def stuck(self) -> bool:
        """Whether this container cannot get better by waiting: exited, dead or unhealthy."""
        return self.state.startswith(("exited", "dead", "unhealthy"))


def project_containers(runtime: str, project: str) -> list[ContainerState] | None:
    """Every container of compose project ``project``, or ``None`` if the runtime did not say.

    Listed by the ``com.docker.compose.project`` label, which Docker Compose and
    podman-compose both set, through the runtime itself rather than through
    compose: the compose process is the thing being waited on.
    """
    try:
        result = subprocess.run(
            [
                runtime,
                "ps",
                "-a",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.Names}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    containers = []
    for line in result.stdout.splitlines():
        name, _, status = line.partition("\t")
        if name.strip():
            containers.append(ContainerState(name.strip(), status.strip()))
    return containers


def last_spool_line(spool_path: Path | None) -> str | None:
    """The last non-empty line compose wrote to ``spool_path``, clipped; ``None`` if none."""
    if spool_path is None:
        return None
    try:
        with open(spool_path, "rb") as spool:
            spool.seek(0, os.SEEK_END)
            spool.seek(max(0, spool.tell() - _SPOOL_TAIL_BYTES))
            tail = spool.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if line.strip():
            line = line.strip()
            if len(line) > _LAST_LINE_LIMIT:
                line = line[: _LAST_LINE_LIMIT - 1] + "…"
            return line
    return None


def _pending_clause(containers: Sequence[ContainerState]) -> str:
    pending = [c for c in containers if c.state]
    if not pending:
        return "every container is up"
    return ", ".join(f"{c.name} ({c.state})" for c in pending)


class _UpWatcher:
    """The still-waiting note, printed between slices of a running ``up``."""

    def __init__(self, runtime: str, project: str) -> None:
        self.runtime = runtime
        self.project = project

    def __call__(self, elapsed: float, spool_path: Path | None) -> None:
        containers = project_containers(self.runtime, self.project)
        lines = [f"still waiting ({format_elapsed(elapsed)})"]
        if containers is not None:
            lines[0] += f": {_pending_clause(containers)}"
        last = last_spool_line(spool_path)
        if last:
            lines.append(f"compose: {last}")
        report_note("\n".join(lines))


def stall_report(runtime: str, project: str, spool_path: Path | None) -> str:
    """What a ``compose up`` that did not finish was waiting on, for the failure report.

    The pending containers, compose's last line, and the last
    :data:`STALL_LOG_TAIL_LINES` log lines of every container that exited or
    is unhealthy, which is where the reason a dependency never came up is.
    Total: what cannot be read is left out, never raised.
    """
    blocks: list[str] = []
    containers = project_containers(runtime, project)
    if containers is not None:
        blocks.append(f"Waiting on: {_pending_clause(containers)}.")
    last = last_spool_line(spool_path)
    if last:
        blocks.append(f"compose's last line: {last}")
    for container in containers or []:
        if not container.stuck:
            continue
        tail = container_log_tail(runtime, container.name, lines=STALL_LOG_TAIL_LINES)
        if tail is None:
            blocks.append(f"{container.name} is {container.state}; its log could not be read.")
            continue
        blocks.append(
            f"{container.name} is {container.state}. Its last {STALL_LOG_TAIL_LINES} log lines:\n"
            + textwrap.indent(tail, "  ")
        )
    return "\n".join(blocks)


@contextmanager
def watched_compose_up(
    cmd: Sequence[str],
    *,
    config: Mapping[str, Any] | None,
    env: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    """Bound and watch the detached ``compose up`` run inside the block.

    Yields the keyword arguments that bound the run, for the call site to hand
    to its own ``run_captured``::

        with watched_compose_up(cmd, config=config, env=run_env) as bounds:
            run_captured(cmd, env=run_env, spool_name="compose-up", repo_root=root, **bounds)

    The call stays the call site's own, through the ``run_captured`` name bound
    in its module, so a test that patches that name still sees every
    invocation. What this adds is around it: a time limit stopped run leaves
    with :func:`stall_report` as its ``detail``, and an interrupted one prints
    the same report before the interrupt propagates.

    :param cmd: The full ``<runtime> compose ... up ... -d`` argv. Its first
        element is the runtime the watch lists containers with.
    :param config: The deploy config, for :func:`compose_up_timeout`.
    :param env: The compose environment; its ``COMPOSE_PROJECT_NAME`` names the
        project whose containers are watched.
    :raises ValueError: The time limit setting is not a positive number.
    """
    timeout = compose_up_timeout(config)
    runtime = cmd[0]
    project = env.get("COMPOSE_PROJECT_NAME", "")
    try:
        yield {
            "timeout": timeout,
            "watch": _UpWatcher(runtime, project),
            "watch_interval": WATCH_INTERVAL_S,
        }
    except CapturedProcessTimeout as exc:
        report = stall_report(runtime, project, exc.spool_path)
        limit = (
            f"The limit is {format_elapsed(timeout)}; raise it with `{COMPOSE_UP_TIMEOUT_KEY}` "
            f"in the config or {COMPOSE_UP_TIMEOUT_ENV} if this start is just slow."
        )
        exc.detail = f"{report}\n{limit}" if report else limit
        raise
    except KeyboardInterrupt:
        phase = current_reporter().current_phase
        try:
            report = stall_report(runtime, project, phase.spool if phase else None)
        except Exception as exc:  # the interrupt is the news; this is a courtesy
            logger.debug("Could not describe the interrupted compose up: %s", exc)
        else:
            if report:
                report_note(report)
        raise
