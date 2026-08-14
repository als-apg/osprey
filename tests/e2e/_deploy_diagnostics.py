"""Failure diagnostics shared by the deploy-backed e2e suites.

When a deployed service dies at boot, compose reports only that a dependency
"is unhealthy". The reason lives in that container's own log -- and every
deploy fixture here tears the stack down in a ``finally``, which removes the
container before anyone can read it. Diagnosing such a failure afterwards then
means inferring a cause from the build log instead of reading one off the
service that actually failed.

So the logs are captured at the moment of failure, by the fixture, and folded
into the failure message.
"""

from __future__ import annotations

import subprocess

#: Lines of container log to include per stopped container. Enough to carry a
#: Python traceback and the startup lines above it, short enough that several
#: dead services do not bury the deploy output they accompany.
LOG_TAIL_LINES = 80


def dead_container_logs(project_prefix: str) -> str:
    """Return logs from every container of ``project_prefix`` that is not running.

    Best-effort by construction: this only ever runs on a path that is already
    failing, so it must never raise and mask the real error. Every failure to
    collect is reported inline instead.
    """
    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={project_prefix}-",
                "--format",
                "{{.Names}}\t{{.State}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(could not list containers: {exc})"

    sections: list[str] = []
    for line in listed.stdout.splitlines():
        name, _, state = line.partition("\t")
        if not name or state == "running":
            continue
        try:
            logs = subprocess.run(
                ["docker", "logs", "--tail", str(LOG_TAIL_LINES), name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            sections.append(f"--- {name} (state={state}) --- (logs unavailable: {exc})")
            continue
        sections.append(f"--- {name} (state={state}) ---\n{logs.stdout}\n{logs.stderr}")

    if not sections:
        return "(no stopped containers found for this deployment)"
    return "\n".join(sections)


def container_logs(*names: str) -> str:
    """Return the log tail of each NAMED container, running or not.

    The companion to :func:`dead_container_logs`, for the failures where every
    container is up and the fault is in what one of them DID -- a service that
    logged a warning and carried on. ``dead_container_logs`` skips those by
    design; this reads them by name.

    Best-effort in the same way: only ever called on an already-failing path,
    so it reports collection failures inline instead of raising over the real
    error.
    """
    sections: list[str] = []
    for name in names:
        try:
            logs = subprocess.run(
                ["docker", "logs", "--tail", str(LOG_TAIL_LINES), name],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            sections.append(f"--- {name} --- (logs unavailable: {exc})")
            continue
        sections.append(f"--- {name} ---\n{logs.stdout}\n{logs.stderr}")
    return "\n".join(sections)


def queue_stack_logs(project_prefix: str) -> str:
    """Log tails from the two containers that own the RE worker environment.

    The evidence a "the worker environment never opened" failure needs and
    that a torn-down stack cannot be asked for afterwards: the bridge logs its
    startup open at WARNING when it fails (``app.py``'s
    ``_open_environment_at_startup``), and the manager's own side of the same
    story is in the queueserver's log.
    """
    return container_logs(
        f"{project_prefix}-bluesky-bridge", f"{project_prefix}-bluesky-queueserver"
    )
