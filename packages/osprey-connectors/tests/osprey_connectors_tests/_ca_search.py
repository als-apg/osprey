"""A raw Channel Access name search, for tests that check what a soft IOC serves.

:func:`ca_raw_search` opens one TCP connection to a CA server, sends a
``CA_PROTO_VERSION`` followed by one ``CA_PROTO_SEARCH`` per name, and reads
every reply frame until the deadline. It is standard library only on purpose:
the soft-IOC fixture tests use it to tell that an IOC serves (or no longer
serves) a name on its port without loading a CA client library into the test
process. The connector never uses it.

The searches carry ``DONT_REPLY``, so a server answers only the names it serves;
a name nobody serves is never answered and therefore never reported.
"""

from __future__ import annotations

import socket
import struct
import time
from collections.abc import Iterable

__all__ = ["CA_PROTO_SEARCH", "CA_PROTO_VERSION", "ca_raw_search"]

#: Channel Access command codes and constants (``caProto.h``).
CA_PROTO_VERSION = 0
CA_PROTO_SEARCH = 6
DONT_REPLY = 5
CA_MINOR_PROTOCOL_REVISION = 13

#: The 16-byte CA header: command, payload size, data type, count, parameter 1, 2.
_HEADER = struct.Struct(">HHHHII")
#: Extended header tail, present when ``postsize == 0xFFFF`` (large payloads).
_EXTENDED = struct.Struct(">II")


def _search_frame(name: bytes, search_id: int) -> bytes:
    """One ``CA_PROTO_SEARCH`` request: the name NUL-terminated and padded to 8 bytes."""
    padded = name + b"\0" * (8 - len(name) % 8)
    return (
        _HEADER.pack(
            CA_PROTO_SEARCH,
            len(padded),
            DONT_REPLY,
            CA_MINOR_PROTOCOL_REVISION,
            search_id,
            search_id,
        )
        + padded
    )


def ca_raw_search(host: str, port: int, names: Iterable[str], timeout: float = 0.5) -> set[str]:
    """Return the subset of ``names`` the CA server at ``host:port`` answers over TCP.

    ``timeout`` bounds the whole call, connect included. Replies that arrive before
    the deadline are all counted; the call returns early once every name has been
    answered. A refused, reset or timed-out connection returns an empty set, so a
    caller never mistakes an unreachable server for one that answered.
    """
    wanted = list(dict.fromkeys(names))
    if not wanted:
        return set()
    deadline = time.monotonic() + timeout
    by_id = dict(enumerate(wanted, start=1))
    request = _HEADER.pack(CA_PROTO_VERSION, 0, 0, CA_MINOR_PROTOCOL_REVISION, 0, 0)
    request += b"".join(
        _search_frame(name.encode(), search_id) for search_id, name in by_id.items()
    )

    found: set[str] = set()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(request)
            buffer = b""
            while len(found) < len(by_id):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                sock.settimeout(remaining)
                try:
                    chunk = sock.recv(65536)
                except TimeoutError:
                    break
                if not chunk:
                    break  # server closed cleanly: what it answered stands
                buffer += chunk
                buffer = _consume_frames(buffer, by_id, found)
    except OSError:
        # Refused, reset, unreachable, or the connect itself timed out.
        return set()
    return found


def _consume_frames(buffer: bytes, by_id: dict[int, str], found: set[str]) -> bytes:
    """Parse every complete frame in ``buffer``; return the unparsed remainder."""
    while len(buffer) >= _HEADER.size:
        command, postsize, _dtype, count, _p1, p2 = _HEADER.unpack_from(buffer)
        offset = _HEADER.size
        if postsize == 0xFFFF and count == 0:
            if len(buffer) < offset + _EXTENDED.size:
                return buffer
            postsize, _count = _EXTENDED.unpack_from(buffer, offset)
            offset += _EXTENDED.size
        if len(buffer) < offset + postsize:
            return buffer
        if command == CA_PROTO_SEARCH and p2 in by_id:
            found.add(by_id[p2])
        buffer = buffer[offset + postsize :]
    return buffer
