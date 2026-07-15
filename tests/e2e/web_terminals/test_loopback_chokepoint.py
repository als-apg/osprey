"""e2e proof of the C3 loopback chokepoint over a REAL socket.

Criterion **C3** requires that in the multi-user deployment, every per-user
``osprey web`` app binds ONLY loopback (127.0.0.1) so the nginx reverse proxy
in front of it is the sole off-host path (see
``src/osprey/templates/modules/web_terminals/nginx.conf.j2`` and
``docker-compose.web.yml.j2``, which sets ``OSPREY_TERMINAL_BIND_HOST`` on
every per-user container). Two existing tests each cover only HALF of that
guarantee, and neither one touches a real listening socket:

* ``tests/deployment/web_terminals/test_render.py::
  test_per_user_services_bind_loopback_not_0_0_0_0`` asserts only that the
  rendered compose *artifact text* doesn't contain the string ``"0.0.0.0"``
  with ``OSPREY_TERMINAL_BIND_HOST`` set. It cannot catch a stale/hostile
  image ``CMD`` that passes ``--host 0.0.0.0`` at container-launch time —
  the artifact was never wrong to begin with, so the string-match passes
  regardless of what actually binds at runtime.
* ``tests/cli/test_web_bind.py`` asserts a **stubbed** ``run_web`` receives
  ``host="127.0.0.1"``. It patches out the launch entirely
  (``monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", ...)``),
  so no socket is ever bound and a regression in the real bind call (e.g. a
  typo, or a code path that bypasses ``resolve_bind_host()`` entirely) would
  sail through green.

Neither test is wrong — they're both cheap, deterministic unit tests, and
that's the right shape for most of this logic. But together they leave a real
two-party gap: the CONTRACT ("declared env wins") is tested, and the
ARTIFACT ("compose file doesn't say 0.0.0.0") is tested, but nothing proves
those two facts compose into an actually-closed chokepoint on a live process.

This test closes that gap: it launches a REAL ``osprey web`` subprocess with
the multi-user env declared (``OSPREY_TERMINAL_BIND_HOST=127.0.0.1``) AND a
deliberately hostile ``--host 0.0.0.0`` — exactly the "stale image CMD" attack
this whole mechanism defends against — then proves via real TCP connects that
the server is reachable on loopback but NOT on the machine's routable (LAN)
interface. If ``resolve_bind_host()`` regressed (or some new code path started
honoring ``--host`` again over the declared env), the LAN connect would
SUCCEED and this test would catch it; the two tests above would not.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osprey.cli.web_cmd import DECLARED_BIND_ENV, resolve_bind_host

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_smoke]

_READY_TIMEOUT = 15.0
_OFFHOST_CONNECT_TIMEOUT = 2.0
_TEARDOWN_TIMEOUT = 5.0


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it.

    Mirrors the identical idiom in ``tests/cli/test_web_bind.py::_free_port``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _routable_ipv4_addresses() -> list[str]:
    """Best-effort, dependency-free discovery of this host's non-loopback IPv4s.

    Two independent, zero-real-traffic probes:

    1. The UDP-connect trick: ``connect()`` on a ``SOCK_DGRAM`` socket only
       makes the kernel pick a route/interface for that destination — no
       packet is actually sent — and ``getsockname()`` then reports the local
       address that route would use.
    2. ``getaddrinfo(gethostname(), None)``, filtered to non-loopback IPv4,
       which catches hostname-resolvable addresses the routing trick might
       miss (e.g. hosts with no default route but a resolvable LAN name).

    Either probe failing (offline sandbox, no default route, unresolvable
    hostname) is expected on some CI hosts and is not itself a test failure —
    the caller skips when the combined result is empty.
    """
    addrs: set[str] = set()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.connect(("8.8.8.8", 80))
            ip = udp.getsockname()[0]
        if ip and not ip.startswith("127."):
            addrs.add(ip)
    except OSError:
        pass

    try:
        for family, _kind, _proto, _canon, sockaddr in socket.getaddrinfo(
            socket.gethostname(), None
        ):
            if family != socket.AF_INET:
                continue
            ip = sockaddr[0]
            if ip and not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass

    return sorted(addrs)


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """Poll until *host*:*port* accepts a connection, or *proc* exits early.

    Mirrors ``osprey.cli.web_cmd._wait_for_server``: also checking
    ``proc.poll()`` on every iteration means a server that crashes on startup
    (bad flag, port collision, import error) is detected immediately instead
    of just burning the full timeout before failing.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def test_declared_loopback_survives_hostile_host_flag_over_real_socket(tmp_path: Path) -> None:
    """The C3 chokepoint, end to end, against a real listening socket.

    Setup mirrors the multi-user compose exactly: the deployment DECLARES
    ``OSPREY_TERMINAL_BIND_HOST=127.0.0.1`` via env, while the launch is given
    the hostile ``--host 0.0.0.0`` a stale/compromised image ``CMD`` might
    pass. ``resolve_bind_host()`` is the single source of truth for which one
    wins (see ``src/osprey/cli/web_cmd.py``); the assertion target below is
    computed FROM that resolver, not hardcoded, so this test tracks the
    resolver's actual contract rather than a guess about it.
    """
    expected_bind_host = resolve_bind_host("0.0.0.0", None, {DECLARED_BIND_ENV: "127.0.0.1"})
    assert expected_bind_host == "127.0.0.1", (
        "resolve_bind_host() contract changed: a declared "
        f"{DECLARED_BIND_ENV} no longer wins over --host 0.0.0.0. The rest of "
        "this test assumes loopback is authoritative, so there is nothing "
        "meaningful left to probe if that assumption doesn't hold."
    )

    free_port = _free_port()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    log_path = tmp_path / "server.log"

    # sys.executable, NEVER bare "python": in this shared worktree, bare
    # `python` resolves to the MAIN repo's checkout, not this worktree's venv.
    cmd = [
        sys.executable,
        "-m",
        "osprey.cli.main",
        "web",
        "--host",
        "0.0.0.0",  # the hostile flag; DECLARED_BIND_ENV below must win anyway
        "--port",
        str(free_port),
        "--shell",
        "true",  # avoid spawning `claude`; the PTY just runs /usr/bin/true
        "--skip-preflight",  # avoid companion-port/provider pre-flight probes
    ]

    env = dict(os.environ)
    env[DECLARED_BIND_ENV] = "127.0.0.1"
    # Neutralize the browser-open side effect (run_web -> _open_browser_when_ready
    # -> webbrowser.open()). The BROWSER env var is read by the webbrowser
    # module's standard-browser registration and, when set, is registered
    # `preferred=True` ahead of the platform default — so it wins over macOS's
    # own osascript-based browser opener. /usr/bin/true silently no-ops.
    env["BROWSER"] = "/usr/bin/true"

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_dir),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

    try:
        ready = _wait_for_port("127.0.0.1", free_port, proc, timeout=_READY_TIMEOUT)
        if not ready:
            exit_code = proc.poll()
            log_contents = log_path.read_text(encoding="utf-8", errors="replace")
            pytest.fail(
                f"Server never became reachable on 127.0.0.1:{free_port} within "
                f"{_READY_TIMEOUT}s (proc exit code: {exit_code}).\n"
                f"--- server log ---\n{log_contents}"
            )

        # -- 1. Loopback assertion: the server is up and loopback-reachable. --
        with socket.create_connection(("127.0.0.1", free_port), timeout=2):
            pass  # connection succeeding is the assertion

        # -- 2. Guarded off-host probe: the discriminating assertion. --------
        lan_ips = _routable_ipv4_addresses()
        if not lan_ips:
            pytest.skip("no non-loopback interface available for off-host probe")

        lan_ip = lan_ips[0]
        with pytest.raises(OSError):
            with socket.create_connection((lan_ip, free_port), timeout=_OFFHOST_CONNECT_TIMEOUT):
                pass  # pragma: no cover - reaching here IS the regression
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=_TEARDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_TEARDOWN_TIMEOUT)
