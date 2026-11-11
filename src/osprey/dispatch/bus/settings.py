"""The environment that switches each end onto the bus, read in one place.

The dispatcher serves the bus when :data:`BIND_ENV` is set; a worker joins it
when :data:`CONNECT_ENV` is set. Either end without its variable keeps to HTTP,
which is how a deployment migrates one switch at a time.

Key material is named by path. With :data:`SECRET_KEY_ENV` unset the end runs
unauthenticated, which the broker and the link allow on a local endpoint only.
With it set, the other key variables of that end are required.

Nothing here imports pyzmq until a variable says the bus is in use, so an end
that stays on HTTP never loads it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.dispatch.bus.keys import ClientKeys, ServerKeys

#: Dispatcher: the endpoint to bind, e.g. ``tcp://0.0.0.0:10050``.
BIND_ENV = "DISPATCH_BUS_BIND"
#: Dispatcher: the directory of worker certificates (``<worker>.key``) it accepts.
WORKER_KEYS_ENV = "DISPATCH_BUS_WORKER_KEYS"
#: Dispatcher: seconds a dispatch waits for a worker with room.
WORKER_WAIT_ENV = "DISPATCH_BUS_WORKER_WAIT_SEC"
#: Worker: the dispatcher's endpoint, e.g. ``tcp://event-dispatcher:10050``.
CONNECT_ENV = "DISPATCH_BUS_CONNECT"
#: Worker: its name on the bus, the stem of its certificate.
WORKER_NAME_ENV = "DISPATCH_BUS_WORKER_NAME"
#: Worker: the dispatcher's public certificate (``server.key``).
SERVER_KEY_ENV = "DISPATCH_BUS_SERVER_KEY"
#: Either end: its own secret certificate (``<name>.key_secret``).
SECRET_KEY_ENV = "DISPATCH_BUS_SECRET_KEY"  # noqa: S105 -- names an environment variable, holds no secret


@dataclass(frozen=True)
class ServerSettings:
    bind: str
    keys: ServerKeys | None
    worker_wait_sec: float


@dataclass(frozen=True)
class WorkerSettings:
    connect: str
    name: str
    keys: ClientKeys | None


def _required(environ: Mapping[str, str], name: str, because: str) -> str:
    from osprey.dispatch.bus.keys import BusKeyError

    value = (environ.get(name) or "").strip()
    if not value:
        raise BusKeyError(f"{name} is required when {because} is set")
    return value


def server_settings(environ: Mapping[str, str] = os.environ) -> ServerSettings | None:
    """The dispatcher's bus settings, or ``None`` when it does not serve the bus.

    Raises:
        BusKeyError: The key variables are incomplete or name unreadable keys.
        ValueError: The wait is not a number.
    """
    bind = (environ.get(BIND_ENV) or "").strip()
    if not bind:
        return None
    keys = None
    secret = (environ.get(SECRET_KEY_ENV) or "").strip()
    if secret:
        from osprey.dispatch.bus.keys import load_server_keys

        workers = _required(environ, WORKER_KEYS_ENV, SECRET_KEY_ENV)
        keys = load_server_keys(Path(secret), Path(workers))
    wait = float(environ.get(WORKER_WAIT_ENV) or 30.0)
    return ServerSettings(bind, keys, wait)


def worker_settings(environ: Mapping[str, str] = os.environ) -> WorkerSettings | None:
    """A worker's bus settings, or ``None`` when it serves HTTP instead.

    Raises:
        BusKeyError: The name is missing, or the key variables are incomplete or
            name unreadable keys.
    """
    connect = (environ.get(CONNECT_ENV) or "").strip()
    if not connect:
        return None
    name = _required(environ, WORKER_NAME_ENV, CONNECT_ENV)
    keys = None
    secret = (environ.get(SECRET_KEY_ENV) or "").strip()
    if secret:
        from osprey.dispatch.bus.keys import load_client_keys

        server = _required(environ, SERVER_KEY_ENV, SECRET_KEY_ENV)
        keys = load_client_keys(Path(secret), Path(server))
    return WorkerSettings(connect, name, keys)
