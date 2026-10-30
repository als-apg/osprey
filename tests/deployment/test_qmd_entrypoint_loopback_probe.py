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

import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.deployment._qmd_entrypoint import run_entrypoint

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


def _run_start_daemon(tmp_path: Path, internal_port: int, no_proxy: str, proxy_port: int):
    """Start the daemon with every proxy pointed at *proxy_port*, where nothing listens."""
    proxy = f"http://127.0.0.1:{proxy_port}"
    env = {
        "OSPREY_QMD_INTERNAL_PORT": str(internal_port),
        "OSPREY_QMD_HEALTH_TIMEOUT": "2",
        "http_proxy": proxy,
        "HTTP_PROXY": proxy,
        "all_proxy": proxy,
        "ALL_PROXY": proxy,
        "no_proxy": no_proxy,
        "NO_PROXY": no_proxy,
    }
    # The stub daemon is stopped through the entrypoint's own shutdown helper, which waits for it, so the shell has no child left to reap when it writes the result line.
    script = 'start_daemon; stop_child "$QMD_PID"; echo "TARGET=$DAEMON_TARGET"'
    stub = "#!/bin/sh\nexec sleep 30 >/dev/null 2>&1\n"
    return run_entrypoint(tmp_path, stub=stub, script=script, env=env)


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
    tmp_path, family, host, target, no_proxy, reserve_port
):
    server = _serve_health(family, host)
    try:
        result = _run_start_daemon(tmp_path, server.server_address[1], no_proxy, reserve_port())
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TARGET={target}" in result.stdout
