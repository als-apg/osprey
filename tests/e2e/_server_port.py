"""The port a server child bound, read from the startup line it logs.

A server the test starts binds port 0 itself and names the port it got in its
startup line: uvicorn's ``Uvicorn running on http://<host>:<port>`` for the
dispatcher and the dispatch worker, ``osprey web``'s own line for the web
terminal. :func:`bound_port` reads that line back from the log file the child
writes to. A log file rather than a pipe: a pipe nobody drains after the startup
line fills with the access log and blocks the server mid-test.

The wait is :func:`tests._container_support.wait_until_ready`, with the child
standing in for a container: a child that exits ends the wait at once with its
exit code and the tail of its log, and a log that is still growing counts as
progress.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests._container_support import wait_until_ready

#: How uvicorn's startup line opens, before ``http://<host>:<port>``.
UVICORN_STARTUP = "Uvicorn running on"


class _ChildSubject:
    """A child process and its log, seen through the container interface the wait reads."""

    def __init__(self, child: subprocess.Popen, log_path: Path) -> None:
        self.child = child
        self.log_path = log_path
        self.status: str | None = None
        self.attrs: dict = {}

    def get_wrapped_container(self) -> _ChildSubject:
        return self

    def reload(self) -> None:
        code = self.child.poll()
        self.status = "running" if code is None else "exited"
        self.attrs = {"State": {"ExitCode": code}}

    def logs(self, tail: int | None = None) -> bytes:
        try:
            data = self.log_path.read_bytes()
        except FileNotFoundError:
            return b""
        if tail is None:
            return data
        return b"".join(data.splitlines(keepends=True)[-tail:])


def port_in(text: str, prefix: str) -> int | None:
    """The port of the URL after *prefix* on the first line that names one, else ``None``.

    Only a line that ends in a newline counts: the last line of a log the child
    is still writing may be cut short in its port.
    """
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        _, found, rest = line.partition(prefix)
        if found and rest.split():
            authority = rest.split()[0].split("://", 1)[-1].split("/", 1)[0]
            # The port follows the last colon whatever the host, an IPv6 literal
            # included.
            return int(authority.rsplit(":", 1)[1])
    return None


def bound_port(
    log_path: Path, prefix: str, *, child: subprocess.Popen, timeout: float = 60.0
) -> int:
    """Wait for *child* to log its startup line in *log_path*; return the port it names.

    Args:
        log_path: The file the child's stdout and stderr are written to.
        prefix: The words before the URL in the startup line, e.g.
            :data:`UVICORN_STARTUP`.
        child: The server process, watched so that one that exits first is
            reported at once rather than after the whole wait.
        timeout: Seconds the child may log nothing new before the wait fails.

    Raises:
        tests._container_support.ContainerExitedError: The child exited first.
        AssertionError: The line did not appear.
    """
    found: list[int] = []

    def _probe() -> None:
        port = port_in(log_path.read_text(encoding="utf-8", errors="replace"), prefix)
        if port is None:
            raise LookupError(f"no {prefix!r} line in {log_path} yet")
        found.append(port)

    wait_until_ready(
        _probe,
        f"startup line {prefix!r} of pid {child.pid}",
        timeout=timeout,
        interval=0.1,
        container=_ChildSubject(child, log_path),
    )
    return found[0]
