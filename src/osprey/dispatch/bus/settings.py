"""The environment that switches each end onto the bus, read in one place.

The dispatcher serves the bus when :data:`BIND_ENV` is set; a worker joins it
when :data:`CONNECT_ENV` is set. Either end without its variable keeps to HTTP,
which is how a deployment migrates one switch at a time.

Keys are carried as hex values (see :mod:`~osprey.dispatch.bus.keys`). With
:data:`SECRET_KEY_ENV` unset an end runs unauthenticated, which the broker and
the link allow on a local endpoint only. With it set, the other key variable of
that end is required. A deployment hands each container its own keys in an
env file that container alone reads (:mod:`osprey.deployment.dispatch_bus_keys`).

A worker takes its secret **out** of the environment as it reads it: the agent
runs it starts inherit the worker's environment, and a run holding a worker's
key could join the bus as that worker.

Nothing here imports pyzmq until a variable says the bus is in use, so an end
that stays on HTTP never loads it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.dispatch.bus.keys import ClientKeys, ServerKeys

#: Dispatcher: the endpoint to bind, e.g. ``tcp://0.0.0.0:10050``.
BIND_ENV = "DISPATCH_BUS_BIND"
#: Dispatcher: the workers it accepts, ``name=<public hex>,name=<public hex>``.
WORKER_KEYS_ENV = "DISPATCH_BUS_WORKER_KEYS"
#: Dispatcher: seconds a dispatch waits for a worker with room.
WORKER_WAIT_ENV = "DISPATCH_BUS_WORKER_WAIT_SEC"
#: Worker: the dispatcher's endpoint, e.g. ``tcp://event-dispatcher:10050``.
CONNECT_ENV = "DISPATCH_BUS_CONNECT"
#: Worker: its name on the bus, the one the dispatcher lists its key under.
WORKER_NAME_ENV = "DISPATCH_BUS_WORKER_NAME"
#: Worker: the dispatcher's public key, hex.
SERVER_KEY_ENV = "DISPATCH_BUS_SERVER_KEY"
#: Either end: its own secret key, hex.
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


def _value(environ: Mapping[str, str], name: str) -> str:
    return (environ.get(name) or "").strip()


def _required(environ: Mapping[str, str], name: str, because: str) -> str:
    from osprey.dispatch.bus.keys import BusKeyError

    value = _value(environ, name)
    if not value:
        raise BusKeyError(f"{name} is required when {because} is set")
    return value


def server_settings(environ: Mapping[str, str] = os.environ) -> ServerSettings | None:
    """The dispatcher's bus settings, or ``None`` when it does not serve the bus.

    Raises:
        BusKeyError: The key variables are incomplete or malformed.
        ValueError: The wait is not a number.
    """
    bind = _value(environ, BIND_ENV)
    if not bind:
        return None
    keys = None
    secret = _value(environ, SECRET_KEY_ENV)
    if secret:
        from osprey.dispatch.bus.keys import parse_worker_keys, server_keys

        workers = _required(environ, WORKER_KEYS_ENV, SECRET_KEY_ENV)
        keys = server_keys(secret, parse_worker_keys(workers))
    wait = float(environ.get(WORKER_WAIT_ENV) or 30.0)
    return ServerSettings(bind, keys, wait)


def worker_settings(environ: MutableMapping[str, str] = os.environ) -> WorkerSettings | None:
    """A worker's bus settings, or ``None`` when it serves HTTP instead.

    Pops :data:`SECRET_KEY_ENV` out of ``environ`` once read, so nothing this
    process starts later inherits it.

    Raises:
        BusKeyError: The name is missing, or the key variables are incomplete or
            malformed.
    """
    connect = _value(environ, CONNECT_ENV)
    if not connect:
        return None
    name = _required(environ, WORKER_NAME_ENV, CONNECT_ENV)
    secret = (environ.pop(SECRET_KEY_ENV, "") or "").strip()
    keys = None
    if secret:
        from osprey.dispatch.bus.keys import client_keys

        server = _required(environ, SERVER_KEY_ENV, SECRET_KEY_ENV)
        keys = client_keys(secret, server)
    return WorkerSettings(connect, name, keys)
