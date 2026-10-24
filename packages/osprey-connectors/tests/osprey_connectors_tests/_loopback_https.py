"""A loopback HTTPS server on a throwaway CA, for login and trust tests.

The CA and the server certificate are minted per run, the servers listen on
``127.0.0.1`` port 0, and every request is recorded as ``(path, Authorization)``.
Each server answers from a per-path table of :class:`Reply` entries; an
unlisted path gets 404.
"""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import json
import ssl
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


@dataclass
class Reply:
    """One canned answer: a status, a JSON body and an optional redirect target."""

    status: int = 200
    body: Any = None
    location: str | None = None


@dataclass
class LoopbackServer:
    """A running server: its base url, its answer table and what it received."""

    url: str
    routes: dict[str, Reply] = field(default_factory=dict)
    seen: list[tuple[str, str | None]] = field(default_factory=list)

    def authorizations(self) -> list[str | None]:
        return [auth for _, auth in self.seen]


@dataclass
class LoopbackPair:
    """An HTTPS server, its plain-HTTP twin and the CA file that signs the HTTPS one."""

    https: LoopbackServer
    http: LoopbackServer
    ca_pem: Path


def _mint(tmp_path: Path) -> tuple[Path, Path, Path]:
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "loopback test CA")])
    ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(ca_ski, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = tmp_path / "ca.pem"
    ca_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_pem = tmp_path / "server.pem"
    cert_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_pem = tmp_path / "server.key"
    key_pem.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_pem, cert_pem, key_pem


def _handler(server: LoopbackServer) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            server.seen.append((path, self.headers.get("Authorization")))
            reply = server.routes.get(path, Reply(status=404))
            body = json.dumps(reply.body).encode() if reply.body is not None else b""
            self.send_response(reply.status)
            if reply.location:
                self.send_header("Location", reply.location)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    return Handler


@contextmanager
def _serving(
    server: LoopbackServer, wrap: Callable[[Any], None] | None, scheme: str
) -> Iterator[LoopbackServer]:
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _handler(server))
    if wrap is not None:
        wrap(httpd)
    server.url = f"{scheme}://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@contextmanager
def loopback_pair(tmp_path: Path) -> Iterator[LoopbackPair]:
    """Serve an HTTPS server on a fresh CA and a plain-HTTP twin, both on loopback."""
    ca_pem, cert_pem, key_pem = _mint(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_pem, key_pem)

    def wrap(httpd: Any) -> None:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    with (
        _serving(LoopbackServer(url=""), wrap, "https") as secure,
        _serving(LoopbackServer(url=""), None, "http") as plain,
    ):
        yield LoopbackPair(https=secure, http=plain, ca_pem=ca_pem)
