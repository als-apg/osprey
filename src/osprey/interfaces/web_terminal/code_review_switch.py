"""Code review's switch, served to this terminal's sessions over a private UNIX socket.

Code review's consumers run in the agent's processes (:mod:`osprey.services.code_review.switch`),
so they cannot ask :func:`~osprey.interfaces.web_terminal.safety_switches.feature_is_on`.
The terminal listens on a socket in a fresh directory only its account can enter, publishes the
path as :data:`~osprey.services.code_review.switch.SWITCH_SOCKET_ENV` for every session it spawns
afterwards, and answers each connection with one JSON object -- ``on``, ``off_by``, ``off_at`` --
then closes it. A connection carries no request and the socket moves nothing.

The listening process is the terminal itself, so the pid a consumer reads off the connection
(``SO_PEERCRED``, ``LOCAL_PEERPID``) is the terminal's, which is how the consumer tells it apart
from a socket the agent bound in its place.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import socket
import tempfile
from dataclasses import dataclass
from typing import Any

from osprey.interfaces.web_terminal.safety_switches import CODE_REVIEW_SWITCH, switches_of
from osprey.services.code_review.switch import SWITCH_SOCKET_ENV

logger = logging.getLogger(__name__)

#: The longest socket path every platform binds: ``sun_path`` holds 104 bytes on macOS.
_MAX_PATH = 103

#: Seconds one answer may take to leave; it is a few dozen bytes.
_ANSWER_TIMEOUT_S = 2.0


def switch_state(app: Any) -> dict[str, Any]:
    """Whether code review runs in *app* now, and who turned it off when not.

    No registry, or no code review switch in it, is on.
    """
    switches = switches_of(app)
    row = switches.row(CODE_REVIEW_SWITCH) if switches is not None else None
    if row is None:
        return {"on": True, "off_by": None, "off_at": None}
    return {"on": row["on"], "off_by": row["off_by"], "off_at": row["off_at"]}


@dataclass
class SwitchListener:
    """The running listener: its socket path and the server answering on it."""

    path: str
    server: asyncio.AbstractServer

    async def close(self) -> None:
        """Stop answering, remove the socket and its directory, and unpublish the path."""
        self.server.close()
        with contextlib.suppress(Exception):
            await self.server.wait_closed()
        shutil.rmtree(os.path.dirname(self.path), ignore_errors=True)
        if os.environ.get(SWITCH_SOCKET_ENV) == self.path:
            del os.environ[SWITCH_SOCKET_ENV]


def _private_socket_path() -> str:
    """A socket path in a new directory only this account can enter (``mkdtemp`` makes it 0700)."""
    for parent in (None, "/tmp"):
        directory = tempfile.mkdtemp(prefix="osprey-cr-", dir=parent)
        path = os.path.join(directory, "s")
        if len(os.fsencode(path)) <= _MAX_PATH:
            return path
        os.rmdir(directory)
    raise OSError("no temporary directory gives a socket path short enough to bind")


async def serve_switch(app: Any) -> SwitchListener:
    """Start answering code review's switch for *app* and publish where.

    Raises:
        OSError: When the socket cannot be made; nothing is published then.
    """
    path = _private_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        os.chmod(path, 0o600)
        sock.setblocking(False)

        async def answer(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                writer.write(json.dumps(switch_state(app)).encode())
                await asyncio.wait_for(writer.drain(), _ANSWER_TIMEOUT_S)
            except Exception as exc:
                logger.debug("code review switch answer not delivered: %s", exc)
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(writer.wait_closed(), _ANSWER_TIMEOUT_S)

        server = await asyncio.start_unix_server(answer, sock=sock)
    except BaseException:
        sock.close()
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        raise
    os.environ[SWITCH_SOCKET_ENV] = path
    return SwitchListener(path=path, server=server)
