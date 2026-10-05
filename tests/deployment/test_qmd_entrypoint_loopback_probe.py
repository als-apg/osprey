"""The qmd entrypoint's wait for its own daemon, run for real against a proxy env.

``start_daemon`` probes the daemon's ``/health`` on both loopback families and
hands the forwarder whichever one answered. Those are requests from the
container to its own process, so they must never be routed to a proxy: a site
container commonly carries ``http_proxy`` with a ``no_proxy`` that lists
``localhost,127.0.0.1`` but not ``::1``, and curl then sends the ``[::1]``
probe to the proxy. With a daemon bound to ``::1`` every probe fails and the
sidecar dies at its health timeout with a healthy daemon behind it.

The test sources the shipped entrypoint (minus its final ``main`` call) under
``sh``, puts a stub ``qmd`` on PATH so the daemon launch is a no-op, stands up a
real ``/health`` listener on one loopback family, points the proxy variables at
a port nothing listens on, and runs ``start_daemon`` with real curl.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/entrypoint.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("sh") is None,
    reason="needs curl and sh, as the sidecar image has",
)


class _Health(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


def _serve_health(family: socket.AddressFamily, host: str) -> ThreadingHTTPServer:
    server_cls = type("_Server", (ThreadingHTTPServer,), {"address_family": family})
    try:
        server = server_cls((host, 0), _Health)
    except OSError as exc:
        pytest.skip(f"no {host} loopback on this host: {exc}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _dead_port() -> int:
    """A loopback port nothing listens on: a proxy there refuses every request."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_start_daemon(tmp_path: Path, internal_port: int, no_proxy: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "qmd"
    stub.write_text("#!/bin/sh\nexec sleep 30 >/dev/null 2>&1\n")
    stub.chmod(0o755)

    lines = ENTRYPOINT.read_text().splitlines()
    assert lines[-1] == 'main "$@"', "the entrypoint no longer ends with its main call"
    library = tmp_path / "entrypoint-lib.sh"
    library.write_text("\n".join(lines[:-1]) + "\n")

    proxy = f"http://127.0.0.1:{_dead_port()}"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "OSPREY_QMD_PORT": "1",
        "OSPREY_QMD_INTERNAL_PORT": str(internal_port),
        "OSPREY_QMD_HEALTH_TIMEOUT": "2",
        "http_proxy": proxy,
        "HTTP_PROXY": proxy,
        "all_proxy": proxy,
        "ALL_PROXY": proxy,
        "no_proxy": no_proxy,
        "NO_PROXY": no_proxy,
    }
    script = f'. "{library}"; start_daemon; kill "$QMD_PID"; echo "TARGET=$DAEMON_TARGET"'
    return subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize(
    ("family", "host", "target", "no_proxy"),
    [
        # The field shape: no_proxy names the IPv4 loopback but not ::1.
        pytest.param(socket.AF_INET6, "::1", "[::1]", "localhost,127.0.0.1", id="ipv6-daemon"),
        # No no_proxy at all: the IPv4 probe must not depend on it either.
        pytest.param(socket.AF_INET, "127.0.0.1", "127.0.0.1", "", id="ipv4-daemon"),
    ],
)
def test_the_health_probe_reaches_the_daemon_whatever_the_proxy_env_says(
    tmp_path, family, host, target, no_proxy
):
    server = _serve_health(family, host)
    try:
        result = _run_start_daemon(tmp_path, server.server_address[1], no_proxy)
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TARGET={target}" in result.stdout
