"""CURVE key material for the dispatch worker bus, and the authenticator that names peers.

The dispatcher holds a server keypair and the public key of every worker it
accepts, each under the worker's name. A worker holds its own keypair and the
dispatcher's public key.

Keys travel as **hex** (64 characters for 32 bytes) rather than ZeroMQ's own
z85: z85's alphabet carries ``$``, ``#`` and ``=``, which an env file's parser
may interpolate, cut at or split on, while hex survives every reader unquoted.
A public key is always derived from its secret, so a pair cannot disagree.

A worker's name is the one the dispatcher lists its key under, and the
dispatcher learns it from the key a connection authenticated with, never from
anything the worker says about itself. So a worker cannot claim to be another.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import zmq
from zmq.auth.asyncio import AsyncioAuthenticator
from zmq.utils import z85

_KEY_BYTES = 32


class BusKeyError(ValueError):
    """Key material the bus cannot start with."""


@dataclass(frozen=True)
class ServerKeys:
    """The dispatcher's keypair and the public keys of the workers it accepts.

    Attributes:
        public: The server's public key, z85, as pyzmq takes it.
        secret: The server's secret key, z85.
        workers: Worker name by the worker's public key (z85).
    """

    public: bytes
    secret: bytes
    workers: dict[bytes, str]


@dataclass(frozen=True)
class ClientKeys:
    """A worker's keypair and the public key of the dispatcher it trusts (all z85)."""

    public: bytes
    secret: bytes
    server_public: bytes


def _z85(hex_key: str, what: str) -> bytes:
    try:
        raw = bytes.fromhex(hex_key.strip())
    except ValueError:
        raise BusKeyError(f"{what} is not hex") from None
    if len(raw) != _KEY_BYTES:
        raise BusKeyError(f"{what} is {len(raw)} bytes, not {_KEY_BYTES}")
    return bytes(z85.encode(raw))


def _hex(z85_key: bytes) -> str:
    return bytes(z85.decode(z85_key)).hex()


def new_secret() -> str:
    """A fresh secret key, hex."""
    _, secret = zmq.curve_keypair()
    return _hex(secret)


def public_of(secret_hex: str) -> str:
    """The public key that pairs with ``secret_hex``, hex.

    Raises:
        BusKeyError: ``secret_hex`` is not a 32-byte hex key.
    """
    return _hex(zmq.curve_public(_z85(secret_hex, "the secret key")))


def server_keys(secret_hex: str, workers: Mapping[str, str]) -> ServerKeys:
    """The dispatcher's key material, from its secret and each worker's public key.

    Raises:
        BusKeyError: A key is malformed, a key is listed twice, or no worker is
            listed. A bus that accepts no key is a bus nobody can join, and the
            alternative — accepting any key — is never taken.
    """
    if not workers:
        raise BusKeyError("no worker key is listed")
    secret = _z85(secret_hex, "the dispatcher's secret key")
    by_key: dict[bytes, str] = {}
    for name, public_hex in workers.items():
        public = _z85(public_hex, f"the public key of {name!r}")
        if public in by_key:
            raise BusKeyError(f"{name!r} and {by_key[public]!r} list the same key")
        by_key[public] = name
    return ServerKeys(zmq.curve_public(secret), secret, by_key)


def client_keys(secret_hex: str, server_public_hex: str) -> ClientKeys:
    """A worker's key material, from its secret and the dispatcher's public key."""
    secret = _z85(secret_hex, "the worker's secret key")
    server_public = _z85(server_public_hex, "the dispatcher's public key")
    return ClientKeys(zmq.curve_public(secret), secret, server_public)


def parse_worker_keys(text: str) -> dict[str, str]:
    """Read ``name=hex,name=hex`` into ``{name: hex}``.

    Raises:
        BusKeyError: An entry is not ``name=key``, or a name repeats.
    """
    workers: dict[str, str] = {}
    for entry in (part.strip() for part in text.split(",")):
        if not entry:
            continue
        name, sep, key = entry.partition("=")
        if not sep or not name.strip() or not key.strip():
            raise BusKeyError(f"worker key entry {entry!r} is not name=key")
        if name.strip() in workers:
            raise BusKeyError(f"worker {name.strip()!r} is listed twice")
        workers[name.strip()] = key.strip()
    return workers


def format_worker_keys(workers: Mapping[str, str]) -> str:
    """The inverse of :func:`parse_worker_keys`, in name order."""
    return ",".join(f"{name}={workers[name]}" for name in sorted(workers))


class NamedCurveAuthenticator(AsyncioAuthenticator):
    """A ZAP authenticator that accepts the listed worker keys and names each.

    The name it returns becomes the connection's ``User-Id``, which the broker
    reads off every message it receives.
    """

    def __init__(self, context: zmq.Context, keys: ServerKeys) -> None:
        super().__init__(context)
        self._names = keys.workers

    def start(self) -> None:
        super().start()
        self.configure_curve_callback(domain="*", credentials_provider=self)

    def callback(self, _domain: str, z85_key: bytes) -> bool:
        """Accept exactly the listed keys (pyzmq hands the key in z85)."""
        return z85_key in self._names

    def curve_user_id(self, client_public_key: bytes) -> str:
        return self._names.get(z85.encode(client_public_key), "")
