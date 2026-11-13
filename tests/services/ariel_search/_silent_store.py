"""A store that takes the connection and never answers.

The silence is real: a loopback listener whose backlog completes the TCP
handshake while nothing ever reads or writes, so the client's start-up packet
waits for a reply that does not come. The wait is real too: a test passes a
short deadline through the keyword its entry point passes, and nothing patches
a clock, a sleep or the pool.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

from osprey_connectors_tests._harness._ports import reserved_port

#: A deadline short enough for a unit test and long enough to be measured.
SHORT_DEADLINE_S = 0.4

#: What a silent open may take past its deadline: closing the pool it began.
MARGIN_S = 2.0


@contextmanager
def silent_store() -> Iterator[int]:
    """Yield the port of a listener that accepts connections and stays quiet."""
    with reserved_port() as port, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen(16)
        yield port


def silent_uri(port: int, database: str = "ariel") -> str:
    """The DSN of the silent store on *port*, with a login and password to keep out of words."""
    return f"postgresql://ariel:owner-secret@127.0.0.1:{port}/{database}"
