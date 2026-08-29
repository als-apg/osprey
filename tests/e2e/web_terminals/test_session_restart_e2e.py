"""End-to-end proof that a browser session survives the server that issued it.

A Web Terminal session used to live only in the serving process's memory, so
every restart of that process — a config change, a container roll, an operator
running ``osprey web stop && osprey web`` — silently logged out everyone holding
a valid cookie. The session store closes that: the digest of each session id is
persisted under the deployment's agent-data directory, and a starting process
reads it back before it serves anything.

``tests/interfaces/test_auth_middleware.py`` proves the *in-process* half — a
second :class:`WebCredentials` built over the same store admits a cookie the
first one issued. That half cannot see the seam this test is about, because it
never crosses a process boundary: it constructs both holders itself, in one
interpreter, with the environment it chose. The real restart is two separate
launcher invocations minutes apart, and everything that has to line up between
them is settled by *code under test* rather than by the test — where the store
directory is (published by ``osprey web`` as
``OSPREY_TERMINAL_SESSION_STORE_DIR``, derived from ``agent_data.base_dir``
anchored on the repo), what the store file is called (named for the settled
``OSPREY_WEB_PORT``), and what the cookie is called (``session_cookie_name()``,
also named for that port). A drift in any one of those three would leave the in-
process test green and every real operator logged out on restart anyway.

So this test drives the actual command. Against one deployment repo, on one
port, it asserts:

1. **The cookie is persistent, not a window-lifetime one.** The ``?token=``
   login URL the detached launcher prints is exchanged (``GET`` → ``303`` +
   ``Set-Cookie``) and the cookie carries ``Max-Age`` — without it the browser
   drops the session on close and the store underneath has nothing to survive
   *for*.
2. **The store holds digests, never ids.** The file the first server wrote is
   ``{"v": 1, "sessions": {<64 hex>: <epoch>}}``, and the cookie value this test
   is holding does not appear anywhere in it. A leaked store must be a list of
   deadlines, not a list of credentials.
3. **The session survives the process.** ``osprey web stop`` is run, the port is
   observed to close and the server PID to exit, a *second* ``osprey web
   --detach`` is started on the same port and repo — a genuinely new process
   with a newly minted operator secret — and the SAME cookie is admitted ``200``
   on a gated route. A request without it is refused ``401`` from that same
   process, so the ``200`` is the cookie's doing and not an open perimeter.
4. **Logout is a revocation, not a courtesy.** ``POST /api/terminal/logout``
   with that cookie reports ``sessions_revoked >= 1``, and the next request
   carrying it is refused ``401`` — the server dropped the digest rather than
   merely asking the browser to forget the value.

The boot shape is ``test_terminal_auth_e2e.py``'s (whose helpers this module
imports rather than copies): a minimal deployment repo, ``sys.executable``
(never bare ``python`` — in this shared worktree it resolves to the MAIN
checkout), ``--shell true`` so the PTY never spawns an agent, ``--skip-preflight``
to skip provider probes, and ``BROWSER=/usr/bin/true`` to neutralise the
browser-open side effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from http.cookies import Morsel, SimpleCookie
from pathlib import Path

import httpx
import pytest

from osprey.interfaces.web_auth import DEFAULT_SESSION_LIFETIME
from tests.e2e.web_terminals.test_terminal_auth_e2e import (
    _free_port,
    _operator_secret_from_url,
    _read_login_url,
)

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_smoke]

_HOST = "127.0.0.1"
#: How long the ``--detach`` launcher itself may take to return. It is not a
#: server-readiness budget — the launcher has its own, and waits for the child
#: to answer before it prints the login URL — but a ceiling on the whole
#: parent-imports-plus-child-imports-plus-app-startup chain.
_LAUNCH_TIMEOUT = 90.0
_READY_TIMEOUT = 30.0
#: Deliberately never spent under ``--detach``: ``_read_login_url`` is handed a
#: launcher this module has already reaped, so it reads the log once and stops
#: on the exited process. That is the right answer — no further output is
#: coming — and the constant is kept at the precedent's value so a reuse against
#: a foreground launch, where the budget is real, needs no new number.
_LOGIN_URL_TIMEOUT = 20.0
_STOP_TIMEOUT = 30.0
_HTTP_TIMEOUT = 15.0
_TEARDOWN_TIMEOUT = 5.0
_POLL_INTERVAL = 0.2

#: Where ``osprey web`` publishes the session store, spelled out rather than
#: computed so that a change to the location has to be made here too. The
#: directory is ``<agent_data.base_dir>/web_terminal`` anchored on the repo, and
#: an unconfigured ``agent_data`` takes ``DEFAULT_AGENT_DATA_BASE_DIR``
#: (``var/agent_data``, in ``osprey_connectors.workspace``). The file is named
#: for the settled port, so two terminals on one host never share a store.
_STORE_RELDIR = Path("var") / "agent_data" / "web_terminal"

#: ``cli/web_cmd.py::PID_FILE``. Read before ``osprey web stop`` runs, because
#: stopping removes it — and it is the only handle on a child this test never
#: spawned directly, so teardown needs the value in hand.
_PID_RELPATH = Path("var") / "osprey-web.pid"

#: ``cli/web_cmd.py::LOG_FILE`` — the detached child's own stdout. Read for
#: failure messages only, and only while it exists (``stop`` removes it too).
_CHILD_LOG_RELPATH = Path("var") / "osprey-web.log"

_HEX = re.compile(r"\A[0-9a-f]{64}\Z")


def _launcher_env() -> dict[str, str]:
    """The environment both launches and the stop run share.

    Single-user shape, exactly as ``test_terminal_auth_e2e.py`` builds it: no
    declared secret (so the launcher mints and announces one), no declared bind
    host (which would make it demand a supplied secret), and no inherited
    ``OSPREY_CONFIG``/``CONFIG_FILE`` pointing the run at another deployment —
    ``cwd`` decides. The session-store and session-lifetime carriers are
    deliberately NOT set here: publishing them is the launcher's job, and this
    test's whole subject is that it does it the same way twice.
    """
    env = dict(os.environ)
    for name in (
        "OSPREY_TERMINAL_SECRET",
        "OSPREY_TERMINAL_BIND_HOST",
        # The bind host's companion declaration, and the one inherited variable
        # that could redirect the launch away from the port this test reserved:
        # `resolve_web_port` treats it as AUTHORITATIVE over `--port`, so an
        # ambient value would start a detached server on a port a real
        # deployment may be using.
        "OSPREY_TERMINAL_WEB_PORT",
        "OSPREY_CONFIG",
        "CONFIG_FILE",
    ):
        env.pop(name, None)
    env["BROWSER"] = "/usr/bin/true"  # registered preferred, silently no-ops
    return env


def _port_is_open(host: str, port: int) -> bool:
    """One connect attempt against *host*:*port*."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _wait_for_open_port(host: str, port: int, timeout: float) -> bool:
    """Poll until *host*:*port* accepts a connection.

    The precedent's ``_wait_for_port`` is deliberately not reused for this:
    it treats the launcher process exiting as a failed start, which is right for
    a foreground server (the launcher *is* the server) and wrong for a detached
    one, where the launcher exits by design the moment its child answers.

    Its early-exit short-circuit is replaced by two things rather than one. A
    launcher that refuses outright — a bad flag, a repo it cannot resolve — is
    caught in :func:`_launch_detached` by its exit code. A *child* that dies
    after being spawned is not: ``_start_detached`` reports that through
    ``output.fail``, which by its own contract neither raises nor sets an exit
    code, so the launcher still exits 0. That case is caught here, by this poll
    expiring — and :func:`_launch_detached` tails both the launcher log and the
    child's own ``var/osprey-web.log`` when it does, so the diagnosis is in the
    failure message even though it costs the full timeout to reach.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_is_open(host, port):
            return True
        time.sleep(_POLL_INTERVAL)
    return False


def _wait_for_closed_port(host: str, port: int, timeout: float) -> bool:
    """Poll until nothing accepts connections on *host*:*port* any more."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_is_open(host, port):
            return True
        time.sleep(_POLL_INTERVAL)
    return False


def _wait_for_pid_exit(pid: int, timeout: float) -> bool:
    """Poll until *pid* is gone. Signal 0 is an existence check, not a signal."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False  # alive, owned by someone else
        time.sleep(_POLL_INTERVAL)
    return False


def _read_pid(repo: Path) -> int | None:
    """The detached server's PID, or ``None`` when there is no readable file."""
    try:
        return int((repo / _PID_RELPATH).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _diagnostics(*logs: Path) -> str:
    """Tail every named log that exists, for an assertion message."""
    parts = []
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts.append(f"--- {log.name} ---\n{text[-4000:]}")
    return "\n".join(parts) or "(no logs)"


def _launch_detached(repo: Path, port: int, log_path: Path) -> str:
    """Start ``osprey web --detach`` on *port* and return its login URL.

    The launcher's stdout is captured to *log_path* because that — not the
    child's log file — is where the ``Open: …?token=…`` line is printed: the
    detached child's stdout is redirected into ``var/osprey-web.log``, and the
    parent prints the one line carrying the secret precisely so the token never
    reaches a file on disk that outlives the launch.
    """
    cmd = [
        sys.executable,  # never bare "python": it resolves to the MAIN checkout here
        "-m",
        "osprey.cli.main",
        "web",
        "--detach",
        "--host",
        _HOST,
        "--port",
        str(port),
        "--shell",
        "true",  # never spawn an agent; the PTY just runs /usr/bin/true
        "--skip-preflight",  # skip provider / companion-port pre-flight probes
    ]
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=_launcher_env(),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    try:
        returncode = proc.wait(timeout=_LAUNCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=_TEARDOWN_TIMEOUT)
        pytest.fail(
            f"`osprey web --detach --port {port}` did not return within "
            f"{_LAUNCH_TIMEOUT}s.\n{_diagnostics(log_path, repo / _CHILD_LOG_RELPATH)}"
        )
    assert returncode == 0, (
        f"`osprey web --detach --port {port}` exited {returncode}.\n"
        f"{_diagnostics(log_path, repo / _CHILD_LOG_RELPATH)}"
    )
    if not _wait_for_open_port(_HOST, port, _READY_TIMEOUT):
        pytest.fail(
            f"the detached server never answered on {_HOST}:{port} within "
            f"{_READY_TIMEOUT}s.\n{_diagnostics(log_path, repo / _CHILD_LOG_RELPATH)}"
        )
    return _read_login_url(log_path, proc, timeout=_LOGIN_URL_TIMEOUT)


def _stop(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run ``osprey web stop`` against *repo* the way an operator would."""
    return subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "web", "stop"],
        cwd=str(repo),
        env=_launcher_env(),
        capture_output=True,
        text=True,
        timeout=_STOP_TIMEOUT,
    )


def _session_cookie(exchange: httpx.Response, cookie_name: str) -> tuple[str, Morsel[str]]:
    """Return ``(raw Set-Cookie header, parsed morsel)`` for *cookie_name*.

    The raw header comes back alongside the morsel because the attribute
    assertions below are about what the *server sent*, and a failure message
    that quotes the header the browser would have seen is the one worth having.
    """
    raw = None
    for header_name, header_value in exchange.headers.multi_items():
        if header_name.lower() == "set-cookie" and header_value.startswith(f"{cookie_name}="):
            raw = header_value
            break
    assert raw is not None, (
        f"token exchange set no {cookie_name!r} cookie; headers: {exchange.headers!r}"
    )
    jar: SimpleCookie = SimpleCookie()
    jar.load(raw)
    return raw, jar[cookie_name]


def test_browser_session_survives_a_server_restart(tmp_path: Path) -> None:
    """Stop and restart a real ``osprey web`` and prove the cookie still works.

    Asserts, across two launcher invocations against one repo and one port, the
    four properties enumerated in the module docstring.
    """
    port = _free_port()

    # `osprey web` serves a deployment repo's RENDER: it walks up to the nearest
    # `profile.yml` and refuses when the repo has no `build/config.yml` to serve,
    # so the run directory must be a real (if minimal) repo. It is also what
    # anchors the session store, which is the point here — the store must land
    # under THIS repo and be found again by the second launch.
    repo = tmp_path / "session-restart-e2e"
    (repo / "build").mkdir(parents=True)
    (repo / "profile.yml").write_text("name: session-restart-e2e\n", encoding="utf-8")
    (repo / "build" / "config.yml").write_text(
        "system:\n"
        "  name: session-restart-e2e\n"
        "control_system:\n"
        "  type: mock\n"
        "  writes_enabled: false\n",
        encoding="utf-8",
    )

    base_url = f"http://{_HOST}:{port}"
    # `session_cookie_name()` names the cookie for the settled OSPREY_WEB_PORT,
    # and `SessionStore` names its file for the same value. Both are spelled out
    # here so a drift between them shows up as this test failing rather than as
    # an operator being logged out.
    cookie_name = f"osprey_terminal_session_{port}"
    store_path = repo / _STORE_RELDIR / f"sessions-{port}.json"
    first_log = tmp_path / "launch-1.log"
    second_log = tmp_path / "launch-2.log"

    # Every server this test starts, remembered as it starts. The PID file is
    # NOT a sufficient handle for teardown: `osprey web stop` unlinks it
    # immediately after sending SIGTERM, without waiting for the process to
    # die, so on every failure path between the explicit stop below and the
    # second launch writing a new file there is no file to read — and the
    # assertion that fires there is precisely the one that fires when the first
    # server has NOT exited. Kept in a list so teardown can reach it anyway.
    pids: list[int] = []

    try:
        # -- First server: exchange the login URL for a session cookie. ------
        first_url = _launch_detached(repo, port, first_log)
        first_pid = _read_pid(repo)
        assert first_pid is not None, (
            f"the detached launcher wrote no readable PID file at {repo / _PID_RELPATH}.\n"
            f"{_diagnostics(first_log, repo / _CHILD_LOG_RELPATH)}"
        )
        pids.append(first_pid)

        with httpx.Client(follow_redirects=False, timeout=_HTTP_TIMEOUT) as client:
            exchange = client.get(first_url)
        assert exchange.status_code == 303, (
            f"token exchange did not 303 (got {exchange.status_code})"
        )
        raw_cookie, morsel = _session_cookie(exchange, cookie_name)
        cookie_value = morsel.value

        # -- 1. Persistent, not window-lifetime. -----------------------------
        #
        # The parsed number, not the presence of the attribute: `Max-Age=0` and
        # `Max-Age=-1` are delete-this-cookie-now directives — the exact
        # opposite of the property — and both would satisfy a substring check.
        # Nothing downstream backstops this one assertion either, because
        # `Max-Age` is browser-side only: the store contents, the restart
        # admission and the logout revocation below are all identical whatever
        # the value. The tmp repo configures no lifetime, so the launcher
        # publishes the default and the exact number is knowable here.
        max_age = morsel["max-age"]
        assert max_age != "", (
            "the session cookie carries no Max-Age, so the browser drops it when the "
            f"window closes and the store has nothing to survive for: {raw_cookie!r}"
        )
        assert int(max_age) == DEFAULT_SESSION_LIFETIME, (
            f"the session cookie's Max-Age is {max_age!r}, not the configured lifetime "
            f"({DEFAULT_SESSION_LIFETIME}s): {raw_cookie!r}"
        )

        # -- 2. The store holds digests, never ids. --------------------------
        assert store_path.is_file(), (
            f"no session store at {store_path} after a successful exchange — the "
            "launcher did not publish the store directory, or named a different one.\n"
            f"{_diagnostics(first_log, repo / _CHILD_LOG_RELPATH)}"
        )
        store_text = store_path.read_text(encoding="utf-8")
        stored = json.loads(store_text)
        assert stored.get("v") == 1, f"unexpected store version in {store_path}: {stored!r}"
        digests = stored.get("sessions")
        assert isinstance(digests, dict) and digests, f"the store recorded no sessions: {stored!r}"
        for key in digests:
            assert _HEX.match(key), (
                f"session store key {key!r} is not a sha256 digest — the store is "
                "holding something other than digests"
            )
        # The positive half: digests, and specifically the right one. Without
        # this the shape check alone would pass on a store holding an unrelated
        # 64-hex key. `_session_key` is a plain unsalted sha256 of the id, so
        # the digest of the cookie in hand is computable here.
        assert hashlib.sha256(cookie_value.encode("utf-8")).hexdigest() in digests, (
            "the store holds no digest of the session id this browser was handed, so "
            "the file is well-formed but not about this session"
        )
        assert cookie_value not in store_text, (
            f"the raw session id the browser was handed appears verbatim in {store_path} — "
            "a leaked store would be a list of live credentials, not of deadlines"
        )

        # -- 3. The session survives the process. ----------------------------
        stopped = _stop(repo)
        assert stopped.returncode == 0, (
            f"`osprey web stop` exited {stopped.returncode}: {stopped.stdout}{stopped.stderr}"
        )
        assert _wait_for_closed_port(_HOST, port, _STOP_TIMEOUT), (
            f"something was still answering on {_HOST}:{port} {_STOP_TIMEOUT}s after "
            "`osprey web stop`"
        )
        assert _wait_for_pid_exit(first_pid, _STOP_TIMEOUT), (
            f"the first server (PID {first_pid}) had not exited {_STOP_TIMEOUT}s after "
            "`osprey web stop`"
        )

        # A genuinely new process, with a newly minted operator secret it never
        # shares with the old one. Only the store on disk connects the two.
        second_url = _launch_detached(repo, port, second_log)
        second_pid = _read_pid(repo)
        if second_pid is not None:
            pids.append(second_pid)
        assert _operator_secret_from_url(second_url) != _operator_secret_from_url(first_url), (
            "the restarted server announced the same operator secret, so this would "
            "not be a restart the session had to survive"
        )

        with httpx.Client(follow_redirects=False, timeout=_HTTP_TIMEOUT) as client:
            revived = client.get(f"{base_url}/api/config", cookies={cookie_name: cookie_value})
            assert revived.status_code == 200, (
                "the session cookie issued by the FIRST server was not admitted by the "
                f"restarted one (got {revived.status_code}): {revived.text[:200]}\n"
                f"{_diagnostics(second_log, repo / _CHILD_LOG_RELPATH)}"
            )

            # The control: the same route, same process, no cookie. Without this
            # the 200 above would also be explained by an ungated perimeter.
            anonymous = client.get(f"{base_url}/api/config")
            assert anonymous.status_code == 401, (
                "a credential-less GET /api/config was not refused 401 (got "
                f"{anonymous.status_code}) — the restarted server is not gating at all, "
                "so the admission above proves nothing"
            )

            # -- 4. Logout is a revocation, not a courtesy. ------------------
            logout = client.post(
                f"{base_url}/api/terminal/logout",
                cookies={cookie_name: cookie_value},
                headers={"Origin": base_url},  # mutations check Origin
            )
            assert logout.status_code == 200, (
                f"POST /api/terminal/logout was not admitted 200 (got {logout.status_code}): "
                f"{logout.text[:200]}"
            )
            revoked = logout.json().get("sessions_revoked")
            assert isinstance(revoked, int) and revoked >= 1, (
                "logout reported no revoked session, so the restored session was never "
                f"in the map it revokes from: {logout.text[:200]}"
            )

            after_logout = client.get(f"{base_url}/api/config", cookies={cookie_name: cookie_value})
            assert after_logout.status_code == 401, (
                "the revoked session cookie was still admitted (got "
                f"{after_logout.status_code}) — logout cleared the browser's copy but "
                "not the server's"
            )
    finally:
        # Always stop every server this test started, whatever failed: a leaked
        # detached process holds the port and outlives the whole pytest run.
        # The PID file is read too — it costs nothing and covers a server
        # started but not yet recorded — but the recorded PIDs are what makes
        # this reliable, for the reason given where `pids` is declared.
        from_file = _read_pid(repo)
        try:
            _stop(repo)
        except (OSError, subprocess.SubprocessError):
            pass
        for pid in dict.fromkeys([*pids, *([from_file] if from_file is not None else [])]):
            if _wait_for_pid_exit(pid, _TEARDOWN_TIMEOUT):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                # ProcessLookupError (it exited in the meantime) is the ordinary
                # case; PermissionError would mean it is not ours to kill.
                pass
