"""In-container probe: the account name an EPICS client puts on the wire.

Run as root inside a project container, fed to the image's python on stdin
(``docker exec -i -u 0 <cid> python - <mode> --user <user>``). The script is
the SERVER half of the proof and spawns the CLIENT half itself, under
``gosu <user>``, so the name is the one the dropped process resolves through
``getpwuid`` — the same lookup every real client in the container makes.

Two modes, each printing exactly one marker-prefixed JSON line on stdout:

``ca``
    A stdlib TCP listener stands in for a Channel Access name server. The
    client is pyepics' libca with ``EPICS_CA_NAME_SERVERS`` pointing at the
    listener, ``EPICS_CA_AUTO_ADDR_LIST=NO`` and an empty
    ``EPICS_CA_ADDR_LIST``: libca opens a TCP circuit to the name server and
    sends VERSION (cmd 0), CLIENT_NAME (cmd 20) and HOST_NAME (cmd 21) before
    any reply, so no UDP and no real IOC are involved. The listener parses
    16-byte big-endian headers (``cmd, postsize, dtype, count: u16``;
    ``p1, p2: u32``; the extended form when ``postsize == 0xFFFF``) and returns
    as soon as it has the CLIENT_NAME payload, a NUL-terminated user name::

        __CA_CLIENT_NAME__{"client_name": "<name>", "commands": [0, 20]}

``pva``
    A p4p ``SharedPV`` server on loopback records ``op.account()`` for one put
    made by a p4p client that reaches it through ``EPICS_PVA_NAME_SERVERS``::

        __PVA_ACCOUNT__{"account": "<name>", "peer": "<addr>"}

Both clients start with ``PYEPICS_LIBCA`` removed from the environment, so
no operator override is in play. The CA client then resolves its libca the way
the production connector does (``_configure_pyepics_libca``, which picks
``epicscorelibs``' per-architecture build — pyepics' own bundled ``clibs`` are
x86_64-only and fail to load on an arm64 host), so the name on the wire comes
from the same library and the same ``getpwuid`` lookup a card uses. Each
client force-exits once its work is sent, skipping the libca teardown that
asserts in ``EPICS_CA_NAME_SERVERS``-only mode (see ``_va_host_ca_op.py``).

Failure — no connection, no CLIENT_NAME, no put within the timeout — prints
one ``__PROBE_ERROR__`` line carrying the client's output, and exits 1. This
file is underscore-prefixed so pytest never collects it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time

CA_MARKER = "__CA_CLIENT_NAME__"
PVA_MARKER = "__PVA_ACCOUNT__"
ERROR_MARKER = "__PROBE_ERROR__"

CA_CLIENT_NAME = 20
_HEADER = struct.Struct(">HHHHII")
_EXTENDED = struct.Struct(">II")
_PV_NAME = "OSPREY:E2E:ACCOUNT-PROBE"

_CA_CLIENT = """
import os
from osprey.connectors.control_system.epics_connector import _configure_pyepics_libca

_configure_pyepics_libca()
import epics.ca as ca

ca.initialize_libca()
ca.create_channel({pv!r}, connect=False, auto_cb=False)
for _ in range(200):
    ca.pend_event(0.05)
    ca.flush_io()
os._exit(0)
"""

_PVA_CLIENT = """
import os
from p4p.client.thread import Context

ctxt = Context(
    "pva",
    conf={{
        "EPICS_PVA_NAME_SERVERS": {server!r},
        "EPICS_PVA_AUTO_ADDR_LIST": "NO",
        "EPICS_PVA_ADDR_LIST": "",
    }},
    useenv=False,
)
ctxt.put({pv!r}, 1.0, timeout=5.0)
print("put-done", flush=True)
os._exit(0)
"""


def _client_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYEPICS_LIBCA"}
    env.update(extra)
    return env


def _spawn(user: str, source: str, env: dict[str, str]) -> subprocess.Popen:
    gosu = shutil.which("gosu")
    if gosu is None:
        raise SystemExit(f"{ERROR_MARKER}" + json.dumps({"error": "no gosu on PATH"}))
    return subprocess.Popen(
        [gosu, user, sys.executable, "-c", source],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )


def _reap(client: subprocess.Popen) -> str:
    """Stop the client if it is still running; return whatever it printed."""
    if client.poll() is None:
        client.kill()
    try:
        out, _ = client.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        return "<client did not exit>"
    return (out or "")[-2000:]


def _fail(message: str, client_output: str) -> None:
    print(ERROR_MARKER + json.dumps({"error": message, "client_output": client_output}))
    sys.stdout.flush()
    os._exit(1)


def _recv_exact(conn: socket.socket, size: int, deadline: float) -> bytes:
    buf = b""
    while len(buf) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out mid-message")
        conn.settimeout(remaining)
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError(f"circuit closed after {len(buf)} of {size} bytes")
        buf += chunk
    return buf


def _read_client_name(conn: socket.socket, deadline: float) -> tuple[str, list[int]]:
    """Parse CA messages off *conn* until CLIENT_NAME; return (name, commands seen)."""
    commands: list[int] = []
    while True:
        cmd, postsize, _dtype, _count, _p1, _p2 = _HEADER.unpack(
            _recv_exact(conn, _HEADER.size, deadline)
        )
        if postsize == 0xFFFF:
            postsize, _ = _EXTENDED.unpack(_recv_exact(conn, _EXTENDED.size, deadline))
        payload = _recv_exact(conn, postsize, deadline) if postsize else b""
        commands.append(cmd)
        if cmd == CA_CLIENT_NAME:
            return payload.split(b"\0", 1)[0].decode("utf-8", "replace"), commands


def probe_ca(user: str, timeout: float) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    client = _spawn(
        user,
        _CA_CLIENT.format(pv=_PV_NAME),
        _client_env(
            {
                "EPICS_CA_NAME_SERVERS": f"127.0.0.1:{port}",
                "EPICS_CA_AUTO_ADDR_LIST": "NO",
                "EPICS_CA_ADDR_LIST": "",
            }
        ),
    )
    deadline = time.monotonic() + timeout
    listener.settimeout(timeout)
    try:
        conn, _ = listener.accept()
    except (TimeoutError, OSError) as exc:
        _fail(f"no CA circuit reached the listener: {exc}", _reap(client))
        return
    try:
        name, commands = _read_client_name(conn, deadline)
    except (TimeoutError, ConnectionError, OSError, struct.error) as exc:
        _fail(f"no CLIENT_NAME on the circuit: {exc}", _reap(client))
        return
    finally:
        conn.close()
        listener.close()
    _reap(client)
    print(CA_MARKER + json.dumps({"client_name": name, "commands": commands}))
    sys.stdout.flush()
    os._exit(0)


def probe_pva(user: str, timeout: float) -> None:
    from p4p.nt import NTScalar
    from p4p.server import Server
    from p4p.server.thread import SharedPV

    seen: list[dict[str, str]] = []
    done = threading.Event()
    pv = SharedPV(nt=NTScalar("d"), initial=0.0)

    @pv.put
    def _on_put(pv_, op):
        seen.append({"account": op.account(), "peer": str(op.peer())})
        pv_.post(op.value())
        op.done()
        done.set()

    server = Server(
        providers=[{_PV_NAME: pv}],
        conf={
            "EPICS_PVAS_INTF_ADDR_LIST": "127.0.0.1",
            "EPICS_PVAS_SERVER_PORT": "0",
            "EPICS_PVAS_BROADCAST_PORT": "0",
            "EPICS_PVAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_PVAS_BEACON_ADDR_LIST": "",
        },
        useenv=False,
    )
    port = server.conf()["EPICS_PVAS_SERVER_PORT"]
    client = _spawn(
        user,
        _PVA_CLIENT.format(server=f"127.0.0.1:{port}", pv=_PV_NAME),
        _client_env({}),
    )
    if not done.wait(timeout) or not seen:
        _fail("no PVA put reached the SharedPV", _reap(client))
        return
    _reap(client)
    print(PVA_MARKER + json.dumps(seen[0]))
    sys.stdout.flush()
    os._exit(0)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="_ca_client_name_sniffer.py")
    parser.add_argument("mode", choices=("ca", "pva"))
    parser.add_argument("--user", required=True, help="the name gosu drops the client to")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.mode == "ca":
        probe_ca(args.user, args.timeout)
    else:
        probe_pva(args.user, args.timeout)


if __name__ == "__main__":
    main(sys.argv[1:])
