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
