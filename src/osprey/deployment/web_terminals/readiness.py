"""Wait for the web-terminal stack to answer before ``osprey up`` reports success.

``compose up -d`` returns once the containers are created and started, which
for a web terminal is twenty seconds or more before its app listens: the
startup checks the Claude Code artifacts, launches the companion servers and
validates bundles first. nginx is up long before that, so a browser opened on
the strength of a finished ``up`` meets a raw 502. This is the readiness gate
that closes that gap, in the same place the store waits close it for the
databases.

Readiness is each service's OWN healthcheck, run on demand with ``exec``
rather than read from the runtime's health status. The status only refreshes on
the healthcheck's interval, so it lags the real moment by up to that interval;
and rootless podman runs healthchecks from systemd timers, which a host without
a user session never fires, so a status that stays ``starting`` forever would
fail a stack that is serving. Running the check itself asks the same question
the template defines, inside the container's own network, at once.
"""

from __future__ import annotations

import subprocess
import textwrap
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from osprey.cli.phase_reporter import report_step
from osprey.deployment.compose_watch import positive_seconds
from osprey.deployment.errors import DeploymentError
from osprey.deployment.runtime_helper import UNHEALTHY_LOG_TAIL_LINES, container_log_tail
from osprey.utils.logger import get_logger

logger = get_logger("deployment.lifecycle")

#: The config key and environment variable that set :func:`web_terminal_ready_timeout`.
READY_TIMEOUT_KEY = "web_terminal_ready_timeout"
READY_TIMEOUT_ENV = "OSPREY_WEB_TERMINAL_READY_TIMEOUT"

#: How long ``up`` waits for the web stack to answer. A terminal's startup takes
#: tens of seconds on a quiet host and minutes on a loaded one; past this, a
#: terminal that still does not answer is not starting slowly but failing.
DEFAULT_READY_TIMEOUT_S = 300.0

#: Seconds between rounds of checks.
_POLL_S = 1.0

#: How long one check may take: the template's own healthcheck timeout is 5s.
_CHECK_TIMEOUT_S = 10.0

#: Container states that waiting cannot fix. ``restarting`` is not among them:
#: a restart policy may still bring the container up inside the window.
_FINAL_STATES = frozenset({"exited", "dead"})


def web_terminal_ready_timeout(config: Mapping[str, Any] | None) -> float:
    """Seconds ``up`` waits for the web stack: ``OSPREY_WEB_TERMINAL_READY_TIMEOUT``,
    the config's ``web_terminal_ready_timeout``, or :data:`DEFAULT_READY_TIMEOUT_S`.
    """
    return positive_seconds(
        config, key=READY_TIMEOUT_KEY, env_var=READY_TIMEOUT_ENV, default=DEFAULT_READY_TIMEOUT_S
    )


@dataclass(frozen=True)
class ReadinessTarget:
    """One container of the web stack and the check that says it is serving.

    Attributes:
        label: How the operator knows it: ``terminal alice``, ``nginx``.
        container: The container's name.
        check: The healthcheck's shell command, as the template spells it.
    """

    label: str
    container: str
    check: str


class WebTerminalsNotReady(DeploymentError):
    """Part of the web stack did not answer within the readiness window.

    Nothing is torn down: the containers that are ready keep serving, and the
    message names them beside the ones that are not. ``detail`` holds the log
    tail of each one that is not, which is where the reason is.
    """

    def __init__(
        self, not_ready: Sequence[str], ready: Sequence[str], reason: str, detail: str | None
    ) -> None:
        self.not_ready = list(not_ready)
        self.ready = list(ready)
        self.detail = detail
        serving = f" ({', '.join(ready)} {'is' if len(ready) == 1 else 'are'} up and serving)"
        super().__init__(
            f"{', '.join(not_ready)} {reason}{serving if ready else ''}. "
            f"To wait longer, set `{READY_TIMEOUT_KEY}` in the config or {READY_TIMEOUT_ENV}."
        )


def _shell_command(test: Any) -> str | None:
    """The healthcheck ``test`` as one shell command, or ``None`` when it has none."""
    if isinstance(test, str):
        return test or None
    if not isinstance(test, list) or not test:
        return None
    kind, *args = test
    if kind == "CMD-SHELL" and args:
        return str(args[0])
    if kind == "CMD" and args:
        return " ".join(str(arg) for arg in args)
    return None


def readiness_targets(compose_file: Path, project: str) -> list[ReadinessTarget]:
    """Every service of the rendered web stack that declares a healthcheck.

    The rendered document is the one source of truth for which containers the
    stack has and how each says it is serving, so nothing here reconstructs a
    service name. An unreadable document yields no targets: the gate then
    waits on nothing rather than failing a deploy over its own bookkeeping.
    """
    try:
        document = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Readiness wait skipped: could not read %s: %s", compose_file, exc)
        return []
    targets = []
    terminal_prefix = f"{project}-web-"
    for key, service in (document.get("services") or {}).items():
        service = service or {}
        container = service.get("container_name")
        check = _shell_command((service.get("healthcheck") or {}).get("test"))
        if not container or not check:
            continue
        if container.startswith(terminal_prefix):
            label = f"terminal {container[len(terminal_prefix) :]}"
        elif container.startswith(f"{project}-"):
            label = container[len(project) + 1 :]
        else:
            label = str(key)
        targets.append(ReadinessTarget(label, container, check))
    return targets


def _container_state(runtime: str, container: str, env: Mapping[str, str]) -> str | None:
    """The container's state (``running``, ``exited``, ...), or ``None`` if unknown."""
    try:
        result = subprocess.run(
            [runtime, "inspect", "--format", "{{.State.Status}}", container],
            capture_output=True,
            text=True,
            check=False,
            timeout=_CHECK_TIMEOUT_S,
            env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return str(result.stdout or "").strip().lower() or None


def _check_passes(runtime: str, target: ReadinessTarget, env: Mapping[str, str]) -> bool:
    try:
        result = subprocess.run(
            [runtime, "exec", target.container, "sh", "-c", target.check],
            capture_output=True,
            check=False,
            timeout=_CHECK_TIMEOUT_S,
            env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _log_tails(runtime: str, targets: Sequence[ReadinessTarget]) -> str | None:
    blocks = []
    for target in targets:
        tail = container_log_tail(runtime, target.container)
        if tail is None:
            blocks.append(f"{target.label}: its log could not be read; `osprey logs` shows it.")
            continue
        blocks.append(
            f"{target.label}, last {UNHEALTHY_LOG_TAIL_LINES} log lines:\n"
            + textwrap.indent(tail, "  ")
        )
    return "\n".join(blocks) or None


def wait_until_ready(
    runtime: str,
    targets: Sequence[ReadinessTarget],
    env: Mapping[str, str],
    timeout: float,
) -> None:
    """Block until every target's check passes, reporting each as it does.

    Fails as soon as a target's container has stopped for good, without waiting
    out the window, and otherwise once the window ends.

    :param runtime: The container runtime binary.
    :param targets: What to wait on (:func:`readiness_targets`).
    :param env: Environment for the runtime calls.
    :param timeout: Seconds to wait in all.
    :raises WebTerminalsNotReady: A target stopped, or the window ended first.
    """
    pending = list(targets)
    ready: list[str] = []
    deadline = time.monotonic() + timeout
    while pending:
        for target in list(pending):
            state = _container_state(runtime, target.container, env)
            if state in _FINAL_STATES:
                raise WebTerminalsNotReady(
                    [target.label],
                    ready,
                    f"stopped before it answered (container {state})",
                    _log_tails(runtime, [target]),
                )
            if state == "running" and _check_passes(runtime, target, env):
                pending.remove(target)
                ready.append(target.label)
                report_step(f"{target.label} ready")
        if not pending:
            return
        if time.monotonic() >= deadline:
            labels = [t.label for t in pending]
            raise WebTerminalsNotReady(
                labels,
                ready,
                f"did not answer within {timeout:g}s",
                _log_tails(runtime, pending),
            )
        time.sleep(_POLL_S)
