"""Wait for a ``docker run`` sidecar to answer HTTP, failing with its logs.

The live web-terminal proofs start their containers with plain ``docker run``,
which returns long before the server inside is listening. Both wait the same
way: poll a URL until it answers with an expected status, and on timeout put
the tail of each container's log in the failure, since the log, not the
deadline, is the diagnosis.
"""

from __future__ import annotations

import subprocess

import httpx

from tests._live import wait_until


def docker_logs(name: str) -> str:
    """The last 40 lines of container *name*'s log, headed by its name."""
    result = subprocess.run(
        ["docker", "logs", "--tail", "40", name], capture_output=True, text=True, timeout=60
    )
    return f"--- {name} ---\n{result.stdout}\n{result.stderr}"


def wait_for_http(
    url: str,
    *,
    containers: list[str],
    what: str,
    timeout: float,
    ok: tuple[int, ...] = (200,),
) -> None:
    """Poll *url* until it answers with an *ok* status, else fail with the logs."""
    last = "no attempt made"

    def answered() -> bool:
        nonlocal last
        try:
            response = httpx.get(url, timeout=5)
        except httpx.HTTPError as exc:  # not up yet
            last = repr(exc)
            return False
        last = f"HTTP {response.status_code}"
        return response.status_code in ok

    try:
        wait_until(answered, timeout=timeout, interval=0.5, what=what)
    except AssertionError:
        logs = "\n".join(docker_logs(name) for name in containers)
        raise AssertionError(f"{what} never became ready ({last})\n{logs}") from None
