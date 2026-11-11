"""CURVE key material for the dispatch worker bus, and the authenticator that names peers.

The dispatcher holds a server keypair and a directory of the workers' public
keys, one ``<worker-name>.key`` certificate per worker. A worker holds its own
keypair and the server's public key. Certificates are the files
:func:`zmq.auth.create_certificates` writes: ``<name>.key`` (public) and
``<name>.key_secret`` (both halves).

A worker's name is the stem of its certificate, and the dispatcher learns it
from the key a connection authenticated with, never from anything the worker
says about itself. So a worker cannot claim to be another one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import zmq
import zmq.auth
from zmq.auth.asyncio import AsyncioAuthenticator
from zmq.utils import z85


class BusKeyError(ValueError):
    """Key material the bus cannot start with."""


@dataclass(frozen=True)
class ServerKeys:
    """The dispatcher's keypair and the public keys of the workers it accepts.

    Attributes:
        public: The server's public key, z85.
        secret: The server's secret key, z85.
        workers: Worker name by the worker's public key (z85).
        workers_dir: The directory the worker certificates were read from.
    """

    public: bytes
    secret: bytes
    workers: dict[bytes, str]
    workers_dir: Path


@dataclass(frozen=True)
class ClientKeys:
    """A worker's keypair and the public key of the dispatcher it trusts."""

    public: bytes
    secret: bytes
    server_public: bytes


def _load_pair(secret_file: Path) -> tuple[bytes, bytes]:
    try:
        public, secret = zmq.auth.load_certificate(secret_file)
    except (OSError, ValueError) as exc:
        raise BusKeyError(f"cannot read the bus key {secret_file}: {exc}") from None
    if secret is None:
        raise BusKeyError(f"{secret_file} holds no secret key")
    return public, secret


def load_server_keys(secret_file: Path, workers_dir: Path) -> ServerKeys:
    """Read the dispatcher's keypair and the worker certificates it accepts.

    Raises:
        BusKeyError: A file cannot be read, or the directory names no worker. A
            bus that accepted no key would be a bus nobody can join, and the
            alternative — accepting any key — is never taken.
    """
    public, secret = _load_pair(secret_file)
    workers: dict[bytes, str] = {}
    for cert in sorted(workers_dir.glob("*.key")):
        try:
            worker_public, _ = zmq.auth.load_certificate(cert)
        except (OSError, ValueError) as exc:
            raise BusKeyError(f"cannot read the worker key {cert}: {exc}") from None
        workers[worker_public] = cert.stem
    if not workers:
        raise BusKeyError(f"{workers_dir} holds no worker certificate (*.key)")
    return ServerKeys(public, secret, workers, workers_dir)


def load_client_keys(secret_file: Path, server_public_file: Path) -> ClientKeys:
    """Read a worker's keypair and the dispatcher's public key."""
    public, secret = _load_pair(secret_file)
    try:
        server_public, _ = zmq.auth.load_certificate(server_public_file)
    except (OSError, ValueError) as exc:
        raise BusKeyError(f"cannot read the server key {server_public_file}: {exc}") from None
    return ClientKeys(public, secret, server_public)


def generate(directory: Path, server: str, workers: Iterable[str]) -> None:
    """Write a server certificate and one certificate per worker into ``directory``.

    The layout is the one :func:`load_server_keys` and :func:`load_client_keys`
    read. Used by the tests, and the seam a build step generates keys through.
    """
    directory.mkdir(parents=True, exist_ok=True)
    zmq.auth.create_certificates(directory, server)
    workers_dir = directory / "workers"
    workers_dir.mkdir(exist_ok=True)
    for name in workers:
        public_file, _ = zmq.auth.create_certificates(directory, name)
        Path(public_file).replace(workers_dir / Path(public_file).name)


class NamedCurveAuthenticator(AsyncioAuthenticator):
    """A ZAP authenticator that accepts the listed worker keys and names each by its file.

    The name it returns becomes the connection's ``User-Id``, which the broker
    reads off every message it receives.
    """

    def __init__(self, context: zmq.Context, keys: ServerKeys) -> None:
        super().__init__(context)
        self._names = keys.workers
        self._workers_dir = keys.workers_dir

    def start(self) -> None:
        super().start()
        self.configure_curve(domain="*", location=self._workers_dir)

    def curve_user_id(self, client_public_key: bytes) -> str:
        return self._names.get(z85.encode(client_public_key), "")
