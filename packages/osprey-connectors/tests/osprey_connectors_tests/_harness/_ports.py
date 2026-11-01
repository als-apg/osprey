"""Loopback ports for tests, named by what the number is.

A port a test needs is one of three things. A server the test controls binds
port 0 itself and reports the number. A port that is free and held, which
nothing answers on or a server binds later, is a :func:`reserved_port`. A port
that must be occupied, for a test of what a bind conflict does, is a
:func:`bound_socket`.

Picking a port by binding port 0 and releasing it leaves a gap before the
server binds it, and the port is inside the kernel's ephemeral range: the kernel
hands ports in that range to every ``connect()`` and ``bind(0)`` of every
process on the host, so any socket can take it in the gap. Ports below the
ephemeral range are handed only to explicit binds. ``reserved_port()`` picks
one there and holds an exclusive lock file named after it for the whole
``with`` block, which serializes explicit binds among test processes: no
concurrent test process, in this run or another run on the same host, is
handed the same port while the lock is held. The lock files live in one
directory per host, whatever ``TMPDIR`` each process sees.
"""

from __future__ import annotations

import atexit
import errno
import fcntl
import os
import random
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

#: Lowest port a reservation hands out, clear of the well-known and registered
#: ports fixed services listen on.
PORT_FLOOR = 20000

#: The ``deployment.port_base`` blocks the e2e lanes book. A deployment derives a
#: port from every slot of its block and refuses a pinned port that lands on one,
#: so a reservation inside a block can fail that lane's build. A lane that books a
#: new block at or above the floor adds it here.
BOOKED_BLOCKS = (range(20000, 23100), range(29000, 31000))

#: The one directory every test process on the host locks ports in, shared by
#: every user of the host.
LOCK_DIR = Path("/tmp/osprey-test-ports")

#: The reservations :func:`hold_ports` keeps for the life of the process.
_HELD = ExitStack()
atexit.register(_HELD.close)

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
    return LOCK_DIR / f"{port}.lock"


def _candidates(first: int) -> list[int]:
    """Every port in ``[PORT_FLOOR, first)`` outside :data:`BOOKED_BLOCKS`, from a
    random start so concurrent processes do not all try the same port first."""
    ports = [
        port
        for port in range(PORT_FLOOR, first)
        if not any(port in block for block in BOOKED_BLOCKS)
    ]
    if not ports:
        return ports
    start = random.randrange(len(ports))
    return ports[start:] + ports[:start]


def _lock(port: int) -> int | None:
    """An open descriptor holding *port*'s lock, or ``None`` when another holds it."""
    LOCK_DIR.mkdir(exist_ok=True)
    with suppress(PermissionError):
        LOCK_DIR.chmod(0o1777)
    fd = os.open(lock_path(port), os.O_RDONLY | os.O_CREAT, 0o644)
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
    before the port is yielded, so the caller's server can bind it. Until the
    caller binds it, no socket holds the port, so a connect to it on 127.0.0.1
    is refused at once.

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


@contextmanager
def reserved_ports(count: int) -> Iterator[list[int]]:
    """Yield *count* distinct loopback ports, each a :func:`reserved_port`,
    all held until the block ends."""
    with ExitStack() as stack:
        yield [stack.enter_context(reserved_port()) for _ in range(count)]


@contextmanager
def port_reserver() -> Iterator[Callable[[], int]]:
    """Yield a call that reserves one more :func:`reserved_port` each time.

    Each call returns a port distinct from every port the call returned before,
    and every reservation is held until the block ends.
    """
    with ExitStack() as held:
        yield lambda: held.enter_context(reserved_port())


def hold_ports(count: int) -> list[int]:
    """*count* distinct :func:`reserved_ports`, held until the process exits."""
    return _HELD.enter_context(reserved_ports(count))


@contextmanager
def bound_socket(host: str = "127.0.0.1") -> Iterator[socket.socket]:
    """Yield a TCP socket bound to ``(host, 0)``, closed when the block ends.

    The socket is opened ``AF_INET6`` when *host* contains ``:`` and ``AF_INET``
    otherwise. While the block is open no other bind can take its port; the
    socket is never put in ``listen()`` here, and a caller that needs a
    listener calls ``listen()`` on it.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        yield sock


def require_ipv6_loopback() -> None:
    """Skip the calling test unless this host can bind ``::1``.

    Called from inside a test, never from a ``skipif`` condition: a condition
    is evaluated at import, and the probe opens a socket.
    """
    import pytest

    if not socket.has_ipv6:
        pytest.skip("host has no IPv6 support")
    try:
        with bound_socket("::1"):
            pass
    except OSError as exc:
        pytest.skip(f"host has no usable IPv6 loopback ({exc})")
