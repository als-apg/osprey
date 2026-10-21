"""Reserve a loopback port that a test-started server binds later.

Picking a port by binding port 0 and releasing it leaves a gap before the
server binds it, and the port is inside the kernel's ephemeral range: the kernel
hands ports in that range to every ``connect()`` and ``bind(0)`` of every
process on the host, so any socket can take it in the gap. Ports below the
ephemeral range are handed only to explicit binds. ``reserved_port()`` picks
one there and holds an exclusive lock file named after it for the whole
``with`` block, which serializes explicit binds among test processes: no
concurrent test process, in this run or another run on the same host, is
handed the same port while the lock is held.
"""

from __future__ import annotations

import errno
import fcntl
import os
import random
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Lowest port a reservation hands out, clear of the well-known and registered
#: ports fixed services listen on.
PORT_FLOOR = 20000

_LINUX_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
_MACOS_RANGE_KEY = "net.inet.ip.portrange.first"


def ephemeral_range_start() -> int:
    """The first port of the kernel's ephemeral range.

    Raises:
        RuntimeError: The range cannot be read on this platform.
    """
    try:
        if sys.platform.startswith("linux"):
            return int(_LINUX_RANGE.read_text().split()[0])
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", _MACOS_RANGE_KEY],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            return int(out.strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot read the ephemeral port range: {exc!r}") from exc
    raise RuntimeError(f"cannot read the ephemeral port range on {sys.platform}")


def lock_path(port: int) -> Path:
    """The lock file whose holder owns *port* among test processes."""
    return Path(tempfile.gettempdir()) / "osprey-test-ports" / f"{port}.lock"


def _candidates(first: int) -> list[int]:
    """Every port in ``[PORT_FLOOR, first)``, from a random start so concurrent
    processes do not all try the same port first."""
    ports = list(range(PORT_FLOOR, first))
    if not ports:
        return ports
    start = random.randrange(len(ports))
    return ports[start:] + ports[:start]


def _lock(port: int) -> int | None:
    """An open descriptor holding *port*'s lock, or ``None`` when another holds it."""
    path = lock_path(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return None
        raise
    return fd


def _bindable(port: int) -> bool:
    """Whether nothing holds *port* on 127.0.0.1 for TCP or for UDP."""
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        with socket.socket(socket.AF_INET, kind) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True


@contextmanager
def reserved_port() -> Iterator[int]:
    """Yield one loopback port, below the ephemeral range, held for the block.

    The port's lock is held until the block ends, and at reservation time the
    port binds on 127.0.0.1 for both TCP and UDP; the probe sockets are closed
    before the port is yielded, so the caller's server can bind it.

    Raises:
        RuntimeError: Every candidate below the ephemeral range was taken, or
            the range cannot be read.
    """
    candidates = _candidates(ephemeral_range_start())
    for port in candidates:
        fd = _lock(port)
        if fd is None:
            continue
        try:
            if not _bindable(port):
                continue
            yield port
            return
        finally:
            os.close(fd)
    raise RuntimeError(
        f"no loopback port free below the ephemeral range: {len(candidates)} candidate(s) tried"
    )
