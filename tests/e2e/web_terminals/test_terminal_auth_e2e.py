"""End-to-end proof that the agent→operator escalation chain is broken.

Until the web-auth gate existed, every OSPREY web surface answered any caller on
loopback with no credential at all. Because the agent's own Python-executor
sandbox runs as a host subprocess on that same loopback, code the agent authored
could reach the very server that spawned it and ``PATCH /api/config`` — rewriting
the ``writes_enabled`` kill-switch and every other safety-relevant field — from
*outside* the tool layer whose in-tool re-checks are the actual safety authority.
That is the chain this phase breaks (PROPOSAL "Break the Chain", requirements
Functional 1–3).

Unit and integration tests cover the pieces (``tests/interfaces/test_web_auth.py``,
``test_auth_middleware.py``, ``test_executor_env_scrub.py``), but each is a half
of the guarantee: the middleware is tested against a Starlette ``TestClient``,
and the sandbox scrub is tested by inspecting the env dict a mocked
``create_subprocess_exec`` was handed. Neither proves the two facts *compose*
into a closed chain on a live process reached over a real socket. This test
closes that gap end to end. It boots a real single-user ``osprey web`` subprocess
(no ``TestClient``, no mocks) and, against it, asserts the five properties that
together are the broken chain:

1. **The core break.** Agent code, run through the *real* Python-executor
   sandbox (:func:`osprey.mcp_server.python_executor.executor.execute_code`,
   whose subprocess env is built by ``scrub_sensitive_env``), attempts
   ``PATCH /api/config`` and is refused ``401``. The launcher's *real* operator
   secret is injected into the executor's parent environment precisely so that a
   scrub regression that leaked it would let the write through and turn this
   assertion green-to-red.
2. **Panel coordination still works.** The same sandbox process reaches
   ``POST /api/panel-focus`` with the panel token and is admitted ``200``. Note
   that ``OSPREY_PANEL_TOKEN`` is *also* scrubbed from the sandbox env (it joins
   ``OSPREY_TERMINAL_SECRET`` in ``sensitive_env.SENSITIVE_ENV_EXACT``; the
   sandbox asserts both are absent). The token is therefore supplied to the
   sandbox code out-of-band — standing in for the trusted in-process companion
   (``mcp_server/http.py``) that legitimately holds it — because the property
   under test is that the *route* honours the two-tier split, granting the weak
   credential panel coordination while still refusing it ``/api/config`` (which
   assertion 1 already showed returns ``401`` from this same process).
3. **The operator can still configure.** The ``Open: …?token=…`` login URL the
   launcher prints is exchanged (``GET`` → ``303`` + ``Set-Cookie``) for a
   session cookie, and a cookie-bearing ``PATCH /api/config`` with a matching
   ``Origin`` succeeds ``200``.
4. **CSRF defence.** The same cookie-bearing ``PATCH`` with a *foreign* ``Origin``
   is refused ``403``.
5. **WebSocket refusal is pre-accept and real.** A real
   :func:`websockets.sync.client.connect` to ``/ws/terminal`` with no credential
   raises :class:`websockets.InvalidStatus` carrying HTTP ``401`` — the handshake
   is refused before the socket opens, proven against a real websockets client
   rather than a ``TestClient``.

The subprocess boot/teardown mirrors ``test_loopback_chokepoint.py``: a minimal
deployment repo, ``sys.executable`` (never bare ``python`` — in this shared
worktree bare ``python`` resolves to the MAIN checkout), ``--shell true`` so the
PTY never spawns ``claude``, ``--skip-preflight`` to skip provider probes, and
``BROWSER=/usr/bin/true`` to neutralise the browser-open side effect.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import httpx
import pytest
import websockets
from websockets.sync.client import connect as ws_connect

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_smoke]

_READY_TIMEOUT = 20.0
_LOGIN_URL_TIMEOUT = 20.0
_SANDBOX_TIMEOUT = 150.0
_HTTP_TIMEOUT = 15.0
_WS_TIMEOUT = 10.0
_TEARDOWN_TIMEOUT = 5.0

#: A known panel token pinned into the server via ``OSPREY_PANEL_TOKEN`` so the
#: sandbox can present it and this test can assert panel coordination. ``osprey
#: web`` reads this carrier once (``web_auth._populate`` pops it) instead of
#: minting one, which is what makes the value predictable here.
_PANEL_TOKEN = "e2e-panel-token-vXnQ2p7Lr9"

#: Matches the launcher's ``Open: http://<host>:<port>/?token=<secret>`` line.
_OPEN_URL_RE = re.compile(r"Open:\s*(http://\S+\?token=\S+)")


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it.

    Mirrors ``test_loopback_chokepoint.py::_free_port`` and
    ``tests/cli/test_web_bind.py::_free_port``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout: float) -> bool:
    """Poll until *host*:*port* accepts a connection, or *proc* exits early.

    Checking ``proc.poll()`` each iteration means a server that dies on startup
    (bad flag, port collision, import error) is caught immediately rather than
    after burning the whole timeout.
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


def _read_login_url(log_path: Path, proc: subprocess.Popen, timeout: float) -> str:
    """Return the ``?token=`` login URL the launcher printed, or fail.

    Polls the captured stdout because the URL is printed once, at startup, by
    ``mint_and_announce`` (via ``cli/web_cmd.py``). ``proc.poll()`` short-circuits
    a launcher that has already exited.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = _OPEN_URL_RE.search(text)
        if match:
            return match.group(1)
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    raise AssertionError(
        "launcher never printed an 'Open: …?token=…' login URL within "
        f"{timeout}s (proc exit code: {proc.poll()}).\n--- server log ---\n"
        f"{log_path.read_text(encoding='utf-8', errors='replace')}"
    )


def _operator_secret_from_url(login_url: str) -> str:
    """Extract the operator secret carried as ``?token=`` in the login URL."""
    query = urllib.parse.urlparse(login_url).query
    values = urllib.parse.parse_qs(query).get("token")
    assert values, f"login URL carries no token: {login_url!r}"
    return values[0]


# ---------------------------------------------------------------------------
# The real Python-executor sandbox driver.
#
# ``execute_code`` is async and reads config/workspace state from its process
# environment, so it is driven in a dedicated subprocess (the same shape
# ``tests/e2e/test_execution_environment_parity.py`` uses) rather than in the
# pytest process. That subprocess stands in for the MCP-server process that
# legitimately holds both credentials: the launcher's real operator secret and
# the pinned panel token are placed in ITS environment, and ``scrub_sensitive_env``
# (inside ``execute_code``) is what must strip them before the grandchild sandbox
# runs. The sandbox then proves, over a real socket, that neither credential
# survived — and that the route tiers behave regardless.
# ---------------------------------------------------------------------------

# Runs INSIDE the executor sandbox (a scrubbed grandchild subprocess). Uses only
# the standard library — the sandbox interpreter is whatever the project venv
# resolves to, so nothing beyond stdlib is guaranteed. ``__BASE__`` and
# ``__PANEL_TOKEN__`` are substituted by the driver; the panel token is a baked
# literal precisely because the env-borne one has been scrubbed away.
_SANDBOX_CODE = r"""
import os
import json
import urllib.request
import urllib.error

BASE = "__BASE__"
PANEL_TOKEN = "__PANEL_TOKEN__"


def _status(method, path, headers, body):
    req = urllib.request.Request(BASE + path, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


# These MUST have been scrubbed from the sandbox environment; a non-empty value
# is a scrub regression that this test exists to catch.
inherited_secret = os.environ.get("OSPREY_TERMINAL_SECRET", "")
inherited_panel = os.environ.get("OSPREY_PANEL_TOKEN", "")

# (1) The core break: attempt to rewrite safety config using whatever operator
# secret the sandbox inherited (scrubbed -> empty) -> refused 401.
config_headers = {"Content-Type": "application/json"}
if inherited_secret:
    config_headers["X-Osprey-Terminal-Secret"] = inherited_secret
config_status = _status(
    "PATCH",
    "/api/config",
    config_headers,
    b'{"updates": {"control_system.writes_enabled": true}}',
)

# (2) Panel coordination with the panel token -> admitted 200.
panel_status = _status(
    "POST",
    "/api/panel-focus",
    {"Content-Type": "application/json", "Authorization": "Bearer " + PANEL_TOKEN},
    b'{"panel": "artifacts"}',
)

print(
    "SANDBOX_RESULT "
    + json.dumps(
        {
            "inherited_secret_present": bool(inherited_secret),
            "inherited_panel_present": bool(inherited_panel),
            "config_status": config_status,
            "panel_status": panel_status,
        }
    )
)
"""

# Runs in the executor's PARENT subprocess: imports the real adapter and awaits
# it, so the whole ``scrub_sensitive_env`` + ``ExecutionWrapper`` + subprocess
# spawn path is exercised, not re-implemented.
_DRIVER_CODE = r"""
import os
import sys
import json
import asyncio

from osprey.mcp_server.python_executor.executor import execute_code

code = open(os.environ["E2E_SANDBOX_CODE_FILE"], encoding="utf-8").read()
result = asyncio.run(execute_code(code, "readonly", "e2e auth chain-break probe"))
print(
    "DRIVER_RESULT "
    + json.dumps(
        {
            "success": result.success,
            "stdout": result.stdout,
            "stderr": (result.stderr or "")[-2000:],
            "error": result.error_message,
        }
    )
)
"""


def _run_executor_sandbox(tmp_path: Path, base_url: str, operator_secret: str) -> dict[str, object]:
    """Run the two sandbox probes through the real executor and return the result.

    Builds a minimal executor deployment (mock control system, subprocess
    execution) and drives ``execute_code`` in a subprocess whose environment
    carries BOTH real credentials, so the scrub inside ``execute_code`` is the
    thing being tested. Returns the parsed ``SANDBOX_RESULT`` dict.
    """
    exec_project = tmp_path / "executor-project"
    exec_project.mkdir()
    (exec_project / "config.yml").write_text(
        "control_system:\n"
        "  type: mock\n"
        "  limits_checking:\n"
        "    enabled: false\n"
        "execution:\n"
        "  execution_method: subprocess\n"
        "python_executor:\n"
        "  execution_timeout_seconds: 120\n",
        encoding="utf-8",
    )

    sandbox_code = _SANDBOX_CODE.replace("__BASE__", base_url).replace(
        "__PANEL_TOKEN__", _PANEL_TOKEN
    )
    sandbox_file = tmp_path / "sandbox_code.py"
    sandbox_file.write_text(sandbox_code, encoding="utf-8")
    driver_file = tmp_path / "executor_driver.py"
    driver_file.write_text(_DRIVER_CODE, encoding="utf-8")

    env = dict(os.environ)
    env["OSPREY_CONFIG"] = str(exec_project / "config.yml")
    env.pop("CONFIG_FILE", None)
    env.pop("OSPREY_TERMINAL_BIND_HOST", None)
    # The credentials the scrub must strip before the grandchild sandbox runs.
    # The operator secret is the launcher's REAL one, so a scrub regression that
    # leaked it would make the sandbox's config PATCH succeed and fail this test.
    env["OSPREY_TERMINAL_SECRET"] = operator_secret
    env["OSPREY_PANEL_TOKEN"] = _PANEL_TOKEN
    env["E2E_SANDBOX_CODE_FILE"] = str(sandbox_file)

    completed = subprocess.run(
        [sys.executable, str(driver_file)],
        cwd=str(exec_project),
        env=env,
        capture_output=True,
        text=True,
        timeout=_SANDBOX_TIMEOUT,
    )

    driver_line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith("DRIVER_RESULT ")),
        None,
    )
    if driver_line is None:
        pytest.fail(
            "executor driver produced no DRIVER_RESULT line.\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    driver_result = json.loads(driver_line[len("DRIVER_RESULT ") :])
    if not driver_result["success"]:
        pytest.fail(
            "executor sandbox run failed:\n"
            f"error: {driver_result['error']}\n"
            f"stderr: {driver_result['stderr']}"
        )

    sandbox_line = next(
        (ln for ln in driver_result["stdout"].splitlines() if ln.startswith("SANDBOX_RESULT ")),
        None,
    )
    if sandbox_line is None:
        pytest.fail(
            "sandbox code produced no SANDBOX_RESULT line.\n"
            f"--- sandbox stdout ---\n{driver_result['stdout']}\n"
            f"--- sandbox stderr ---\n{driver_result['stderr']}"
        )
    return json.loads(sandbox_line[len("SANDBOX_RESULT ") :])


def test_escalation_chain_is_broken_end_to_end(tmp_path: Path) -> None:
    """Boot a real single-user ``osprey web`` and prove the chain is broken.

    Asserts, against one live process reached over real sockets, all five
    properties enumerated in the module docstring.
    """
    port = _free_port()

    # `osprey web` serves a deployment repo's RENDER: it walks up to the nearest
    # `profile.yml` and refuses when the repo has no `build/config.yml` to serve,
    # so the run directory must be a real (if minimal) repo — same as
    # test_loopback_chokepoint.py. The served config carries a patchable
    # safety field so the operator PATCH below has something real to write.
    repo = tmp_path / "terminal-auth-e2e"
    (repo / "build").mkdir(parents=True)
    (repo / "profile.yml").write_text("name: terminal-auth-e2e\n", encoding="utf-8")
    (repo / "build" / "config.yml").write_text(
        "system:\n"
        "  name: terminal-auth-e2e\n"
        "control_system:\n"
        "  type: mock\n"
        "  writes_enabled: false\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "server.log"

    cmd = [
        sys.executable,  # never bare "python": it resolves to the MAIN checkout here
        "-m",
        "osprey.cli.main",
        "web",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--shell",
        "true",  # never spawn `claude`; the PTY just runs /usr/bin/true
        "--skip-preflight",  # skip provider / companion-port pre-flight probes
    ]

    env = dict(os.environ)
    # Pin the panel token so the sandbox and this test can present a known value.
    env["OSPREY_PANEL_TOKEN"] = _PANEL_TOKEN
    # Single-user shape: no declared secret (the launcher mints and announces
    # one) and no declared bind host (which would make it demand a supplied
    # secret). A shell-exported OSPREY_CONFIG/CONFIG_FILE would point the run at
    # another deployment, so both are dropped and `cwd=repo` decides.
    env.pop("OSPREY_TERMINAL_SECRET", None)
    env.pop("OSPREY_TERMINAL_BIND_HOST", None)
    env.pop("OSPREY_CONFIG", None)
    env.pop("CONFIG_FILE", None)
    # Neutralize the browser-open side effect (webbrowser.open); /usr/bin/true
    # is registered preferred and silently no-ops.
    env["BROWSER"] = "/usr/bin/true"

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )

    try:
        if not _wait_for_port("127.0.0.1", port, proc, timeout=_READY_TIMEOUT):
            pytest.fail(
                f"server never became reachable on 127.0.0.1:{port} within "
                f"{_READY_TIMEOUT}s (proc exit code: {proc.poll()}).\n"
                f"--- server log ---\n{log_path.read_text(encoding='utf-8', errors='replace')}"
            )

        base_url = f"http://127.0.0.1:{port}"
        login_url = _read_login_url(log_path, proc, timeout=_LOGIN_URL_TIMEOUT)
        operator_secret = _operator_secret_from_url(login_url)

        # -- 1 & 2. The real Python-executor sandbox: config write refused, panel
        #    coordination admitted, both credentials proven scrubbed. --------
        sandbox = _run_executor_sandbox(tmp_path, base_url, operator_secret)
        assert sandbox["inherited_secret_present"] is False, (
            "OSPREY_TERMINAL_SECRET survived the executor scrub into the sandbox — "
            "the escalation chain is OPEN"
        )
        assert sandbox["inherited_panel_present"] is False, (
            "OSPREY_PANEL_TOKEN survived the executor scrub into the sandbox"
        )
        assert sandbox["config_status"] == 401, (
            "sandbox PATCH /api/config was not refused 401 (got "
            f"{sandbox['config_status']}) — agent-spawned code can rewrite safety config"
        )
        assert sandbox["panel_status"] == 200, (
            "sandbox POST /api/panel-focus with the panel token was not admitted 200 "
            f"(got {sandbox['panel_status']}) — panel coordination is broken"
        )

        # -- 3. The operator exchanges the login URL for a session cookie and
        #    configures with it (matching Origin). ---------------------------
        with httpx.Client(follow_redirects=False, timeout=_HTTP_TIMEOUT) as client:
            exchange = client.get(login_url)
            assert exchange.status_code == 303, (
                f"token exchange did not 303 (got {exchange.status_code})"
            )
            assert "set-cookie" in {k.lower() for k in exchange.headers}, (
                "token exchange returned no Set-Cookie"
            )

            # `hooks.debug`, not `control_system.writes_enabled`: the latter is
            # a protected key now, refused for every caller whatever credential
            # it carries. Written here it would refuse the admitted operator in
            # step 3 and refuse the forged Origin in step 4 for the same reason,
            # leaving the CSRF assertion passing without ever exercising the
            # Origin check. `hooks.debug` is the Hook Debug switch the Web
            # Terminal ships, reached by exactly this call, and exempt from the
            # protected set — so a refusal below can only come from the credential.
            configured = client.patch(
                f"{base_url}/api/config",
                json={"updates": {"hooks.debug": True}},
                headers={"Origin": base_url},
            )
            assert configured.status_code == 200, (
                "cookie-bearing PATCH /api/config with a matching Origin was not "
                f"admitted 200 (got {configured.status_code}): {configured.text[:200]}"
            )

            # -- 3b. Authentication is not authorisation. The same admitted
            #    operator may not rewrite the safety surface itself. ----------
            protected = client.patch(
                f"{base_url}/api/config",
                json={"updates": {"control_system.writes_enabled": True}},
                headers={"Origin": base_url},
            )
            assert protected.status_code == 403, (
                "a fully authenticated operator with a matching Origin was allowed "
                f"to PATCH a protected key (got {protected.status_code}): "
                f"{protected.text[:200]}"
            )

            # -- 4. CSRF defence: the same cookie with a foreign Origin is refused. --
            forged = client.patch(
                f"{base_url}/api/config",
                json={"updates": {"hooks.debug": False}},
                headers={"Origin": "http://evil.example.com"},
            )
            assert forged.status_code == 403, (
                "cookie-bearing PATCH with a foreign Origin was not refused 403 "
                f"(got {forged.status_code}) — CSRF defence is open"
            )

        # -- 5. WebSocket handshake with no credential is refused pre-accept, 401,
        #    proven against a real websockets client. ------------------------
        with pytest.raises(websockets.InvalidStatus) as excinfo:
            with ws_connect(f"ws://127.0.0.1:{port}/ws/terminal", open_timeout=_WS_TIMEOUT):
                pass  # pragma: no cover - a successful open IS the regression
        assert excinfo.value.response.status_code == 401, (
            "credential-less /ws/terminal handshake was not refused 401 (got "
            f"{excinfo.value.response.status_code})"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=_TEARDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_TEARDOWN_TIMEOUT)
