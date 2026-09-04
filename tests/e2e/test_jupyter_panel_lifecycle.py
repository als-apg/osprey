"""Real-container lifecycle of the JUPYTER panel: a kernel that follows the terminal.

The unit tests pin the kernel launcher's stamps against a synthetic binding,
and the proxy integration test pins the panel's HTTP and websocket legs against
a sidecar started by hand. Neither can answer the question this file exists
for: inside a deployed persona container, with a real terminal session, a real
controls MCP server and a real control system behind the cells, does a
notebook kernel read, refuse, follow and survive the way the panel promises?

THE DEPLOYMENT THIS BUILDS
--------------------------
One ``osprey up --dev`` of a ``control-assistant`` render, trimmed to one
roster user (``alice``) on one persona that keeps the preset's write gate armed
and inherits the preset's panel set — which is what selects the JUPYTER
panel. The control system is the virtual accelerator, baselined as the ``va``
target, with the live stand-in deployed beside it as the ``standin`` target: a
switch needs two targets, and the mock connector cannot be one — its baseline
is ``live``, which a mock deployment cannot resolve, so a kernel bound to a
mock session would always run fail-closed and never carry a target stamp.

Everything is reached at the per-user web port directly, carrying the operator
secret as the header nginx injects in the container shape (``token`` posture
clears a client-supplied copy at nginx, so the direct port is the one place a
test can present it). The notebook panel lives behind the terminal's own panel
proxy (``/panel/jupyter/...``), so every request to the sidecar passes through
the proxy that re-issues the sidecar's credential.

WHAT IS ASSERTED, IN ORDER
--------------------------
The order is load-bearing: the first kernel must start before any terminal
session exists, and the restart must come last.

1. The starter notebook is listed on first open, and only the ``osprey``
   kernelspec exists.
2. A kernel started with NO chat session reads a channel, and refuses a write
   with the open-a-session line and one audit record.
3. Attaching a terminal binds the session, and the controls server publishes
   the session's target.
4. A kernel started while the session's writes were off reads, and refuses a
   write with the executor's text plus the turn-writes-on line and exactly one
   audit record on the ``notebook_kernel`` surface.
5. After a chip target switch the same kernel's next write raises
   ``ControlTargetChangedError`` with the restart line; a restarted kernel
   follows the new target.
6. ``NotebookEdit`` under ``notebooks/`` is allowed by the rendered settings and
   the guard hook and badges the panel; outside it is denied.
7. The sidecar's runtime directory is not under the agent-data root.
8. Notebooks survive ``osprey down && osprey up``; kernels do not.

CONTAINER-OPS SAFETY: every runtime-mutating call below names an EXACT resource
this test created — the ``<prefix>-*`` containers, this project's volumes and
the ``:local`` image tags — or is the project-scoped ``compose down`` the deploy
lifecycle itself uses. Nothing here ever runs a prune, an ``-a``/``--all``
sweep, or a wildcard removal. Set ``E2E_REUSE_IMAGES`` to keep the built images
between runs.

CI: the ``dockerbuild`` marker keeps this module out of the shared ``e2e-tests``
lane, which ``--ignore``s every marked file. It runs in its own job,
``jupyter-panel-e2e``, whose result the merge gate reads. It is excluded from
the model matrix: nothing here contacts a model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import nbformat
import pytest
import yaml
from websockets.sync.client import connect as ws_connect

from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from osprey.port_layout import PORT_BASE_CONFIG_KEY, default_port
from osprey.utils.dotenv import parse_dotenv_file
from tests.e2e._orm_stack import VA_CA_PORT
from tests.e2e._volumes import remove_project_volumes

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.dockerbuild]

RUNTIME = "docker"

#: The repo DIRECTORY name — the compose project name and the ``com.osprey.project``
#: label on every image this deploy builds.
PROJECT_NAME = "osprey-e2e-jnb"
PREFIX = "jnb"
PRESET = "control-assistant"
USER = "alice"
PERSONA = "operator"
#: ``osprey build`` renders a persona delta to ``build/<repo>-<persona>``, and
#: the catalog must name that same project or a start refuses the render.
PERSONA_PROJECT = f"{PROJECT_NAME}-{PERSONA}"
PERSONA_IMAGE_TAG = f"{PERSONA_PROJECT}:local"
#: Service images ``osprey up`` builds per project, exact tags for the teardown.
SERVICE_IMAGE_TAGS = (f"{PROJECT_NAME}-va:local", f"{PROJECT_NAME}-qmd:local")

#: This module's own thousand-port block; every host port follows it.
PORT_BASE = 24000
WEB_PORT = default_port("web", base=PORT_BASE)
BASE_URL = f"http://127.0.0.1:{WEB_PORT}"
WS_URL = f"ws://127.0.0.1:{WEB_PORT}"
#: The origin nginx publishes and every mutating request must name.
ORIGIN = f"http://127.0.0.1:{default_port('nginx', base=PORT_BASE)}"
PANEL = "/panel/jupyter"
KERNELSPEC = "osprey"
STARTER_NOTEBOOK = "getting-started.ipynb"
SURVIVOR_NOTEBOOK = "e2e-survives.ipynb"

#: A readback the virtual accelerator always serves.
CHANNEL = "SR:DIAG:DCCT:01:CURRENT:RB"

#: The three hint lines the kernel prints before a refusal's traceback.
HINT_NO_SESSION = (
    "No chat session was open when this kernel started. Open one, then restart the kernel."
)
HINT_WRITES_OFF = (
    "This kernel started with writes off. Turn writes on from the chip, then restart the kernel."
)
HINT_TARGET_CHANGED = "The session's control target changed. Restart the kernel to follow it."

#: The audit surface a cell's refusals file under, and the ledger it lands in.
SURFACE = "notebook_kernel"
LEDGER = Path("var") / "audit" / USER / f"{SURFACE}.jsonl"

DEPLOY_UP_TIMEOUT_SEC = 2400
VERB_TIMEOUT_SEC = 180
RENDER_TIMEOUT_SEC = 600
READY_TIMEOUT_SEC = 300.0
#: How long a kernel gets to answer one cell. The first cell imports the
#: framework runtime, which is the slow one.
CELL_TIMEOUT_SEC = 240.0
#: How long the controls server inside a fresh ``claude`` gets to publish.
SESSION_TARGET_TIMEOUT_SEC = 240.0
SWITCH_TIMEOUT_SEC = 120.0

_ENV_APPEND = "ANTHROPIC_API_KEY=fake-llm-key-value\nZO_INGEST_SA_TOKEN=fake-telemetry-token\n"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Runtime and CLI helpers
# ---------------------------------------------------------------------------


def _runtime_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([RUNTIME, *args], capture_output=True, text=True, timeout=timeout)


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run_osprey(
    osprey_bin: Path, args: list[str], cwd: Path, timeout: int = VERB_TIMEOUT_SEC
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(osprey_bin), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": "", "CONTAINER_RUNTIME": RUNTIME},
    )


def _web_container() -> str:
    return f"{PREFIX}-web-{USER}"


def _logs(name: str) -> str:
    result = _runtime_cli("logs", "--tail", "80", name, timeout=20)
    return f"--- {name} stdout ---\n{result.stdout}\n--- {name} stderr ---\n{result.stderr}"


def _exec(*argv: str, user: str = "1000", cwd: str | None = None, stdin: str | None = None):
    """Run one command inside the terminal container, as the runtime uid."""
    args = [RUNTIME, "exec", "-i", "--user", user]
    if cwd:
        args += ["-w", cwd]
    args += [_web_container(), *argv]
    return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=60)


def _exec_text(*argv: str) -> str:
    result = _exec(*argv)
    assert result.returncode == 0, _fmt(f"docker exec {' '.join(argv)}", result)
    return result.stdout


def _process_env(pid: int) -> dict[str, str]:
    """The environment of *pid* inside the container, read off ``/proc``."""
    result = subprocess.run(
        [RUNTIME, "exec", "--user", "1000", _web_container(), "cat", f"/proc/{pid}/environ"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"could not read /proc/{pid}/environ: {result.stderr!r}"
    return dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)


def _state_records(agent_data_root: str) -> list[dict[str, Any]]:
    """Every control-target state record under the container's agent-data root."""
    listing = _exec("sh", "-c", f"cat {agent_data_root}/control_target/target_state_*.json")
    if listing.returncode != 0:
        return []
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    text = listing.stdout.strip()
    while text:
        record, end = decoder.raw_decode(text)
        records.append(record)
        text = text[end:].strip()
    return records


# ---------------------------------------------------------------------------
# The repo
# ---------------------------------------------------------------------------


def _override_text() -> str:
    """The ``--override`` overlay for the deployment.

    The virtual accelerator is the baseline and the stand-in the second target;
    the preset's archive stays (a simulated machine may not be paired with an
    invented history) but shrunk to seconds of seeding. The bluesky stack, the
    dispatcher and telemetry are dropped: nothing here reaches them, and every
    container they add is a minute of build. ``modules.web_terminals`` cannot
    carry the roster from here — list values union with the preset's — so the
    roster and the persona catalog are rewritten in :func:`_shape_repo`.
    """
    return yaml.safe_dump(
        {
            "config": {
                "container_runtime": RUNTIME,
                "facility.name": "E2E Notebook Panel Fixture",
                "facility.prefix": PREFIX,
                "facility.timezone": "UTC",
                "deploy.fqdn": "127.0.0.1",
                PORT_BASE_CONFIG_KEY: PORT_BASE,
                "control_system.type": "virtual_accelerator",
                "claude_code.telemetry.enabled": False,
                "claude_code.servers.bluesky.enabled": False,
                "claude_code.servers.health.enabled": False,
                "modules.web_terminals": {
                    "enabled": True,
                    "image_source": "local",
                    "default_persona": PERSONA,
                    "auth": {"method": "token"},
                },
            },
            "channel_finder_mode": "hierarchical",
            "dispatch": None,
            "bluesky": None,
            "bluesky_web": None,
            "va_archiver": {"retention_days": 2, "hot_span_hours": 2},
        },
        sort_keys=False,
    )


def _shape_repo(repo: Path) -> None:
    """One roster user on one persona delta, after ``osprey init`` materialized the preset's.

    ``osprey init`` renders one delta per preset persona and unions the roster
    override onto the preset's own; the profile is rewritten here to the shape
    this lane deploys. The delta keeps the write gate armed the way the preset's
    read-write persona does, and declares no ``extends`` — a file in
    ``personas/`` inherits from the profile beside it.
    """
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    terminals = profile["config"]["modules.web_terminals"]
    terminals["users"] = [{"name": USER, "index": 0, "persona": PERSONA}]
    terminals["personas"] = {
        PERSONA: {
            "project": PERSONA_PROJECT,
            "project_path": f"build/{PERSONA_PROJECT}",
            "build_profile": f"personas/{PERSONA}.yml",
        }
    }
    terminals["default_persona"] = PERSONA
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    personas = repo / "personas"
    for delta in personas.glob("*.yml"):
        delta.unlink()
    (personas / f"{PERSONA}.yml").write_text(
        f"name: {PROJECT_NAME} ({PERSONA})\n"
        "deploy_services: false\n"
        "config:\n"
        "  control_system.writes_enabled: true\n"
        "  web.ui_mode: expert\n"
        "  modules.web_terminals.enabled: false\n",
        encoding="utf-8",
    )


def _make_repo(tmp_path: Path, osprey_bin: Path) -> Path:
    repo = tmp_path / PROJECT_NAME
    override_path = tmp_path / "override.yml"
    override_path.write_text(_override_text(), encoding="utf-8")

    init = _run_osprey(
        osprey_bin,
        [
            "init",
            str(repo),
            "--preset",
            PRESET,
            "--no-git",
            "--override",
            str(override_path),
            "--set",
            f"virtual_accelerator.port={VA_CA_PORT}",
        ],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init (notebook panel)", init)
    _shape_repo(repo)

    build = _run_osprey(
        osprey_bin,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        tmp_path,
        timeout=RENDER_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt("osprey build (notebook panel)", build)

    rendered = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    resolved_base = (rendered.get("deployment") or {}).get("port_base")
    assert resolved_base == PORT_BASE, (
        f"{PROJECT_NAME} resolved {PORT_BASE_CONFIG_KEY}={resolved_base!r}, not {PORT_BASE}"
    )

    # `osprey build` pointed .env at the channel manifest; append, never replace.
    env_path = repo / ".env"
    with env_path.open("a", encoding="utf-8") as handle:
        handle.write(_ENV_APPEND)
    os.chmod(env_path, 0o600)
    return repo


def _compose_project() -> str | None:
    """The compose project stamped on the terminal container, if it exists."""
    result = _runtime_cli(
        "inspect",
        "--type",
        "container",
        "-f",
        '{{index .Config.Labels "com.docker.compose.project"}}',
        _web_container(),
        timeout=15,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _teardown(project: str | None) -> None:
    """Exact-named sweep; failures swallowed (a safety net, never an assertion)."""
    _runtime_cli("rm", "-f", _web_container())
    _runtime_cli("rm", "-f", f"{PREFIX}-nginx")
    for project_name in {project, PROJECT_NAME} - {None}:
        _runtime_cli("compose", "-p", str(project_name), "down", timeout=120)
        remove_project_volumes(str(project_name), runtime=RUNTIME)
    if not os.environ.get("E2E_REUSE_IMAGES"):
        for tag in (PERSONA_IMAGE_TAG, *SERVICE_IMAGE_TAGS):
            _runtime_cli("rmi", "-f", tag, timeout=120)


# ---------------------------------------------------------------------------
# HTTP against the terminal
# ---------------------------------------------------------------------------


@dataclass
class Terminal:
    """One deployed terminal, reached at its web port with the operator secret."""

    repo: Path
    osprey_bin: Path
    secret: str
    client: httpx.Client
    #: State the ordered tests hand each other.
    session_id: str | None = None
    pty_pid: int | None = None
    kernel_id: str | None = None
    notebook_session_id: str | None = None
    first_target: str | None = None
    agent_data_root: str | None = None
    starter_cells: list[Any] = field(default_factory=list)

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Osprey-Terminal-Secret": self.secret}

    @property
    def mutating_headers(self) -> dict[str, str]:
        return {**self.headers, "Origin": ORIGIN}

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.client.get(f"{BASE_URL}{path}", headers=self.headers, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.client.post(f"{BASE_URL}{path}", headers=self.mutating_headers, **kwargs)

    def put(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.client.put(f"{BASE_URL}{path}", headers=self.mutating_headers, **kwargs)

    def delete(self, path: str) -> httpx.Response:
        return self.client.delete(f"{BASE_URL}{path}", headers=self.mutating_headers)

    def posture(self) -> dict[str, Any]:
        response = self.get("/api/terminal/posture", params={"session_id": self.session_id})
        assert response.status_code == 200, response.text
        return response.json()

    def ledger(self) -> list[dict[str, Any]]:
        path = self.repo / LEDGER
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def start_notebook_session(self, notebook: str) -> dict[str, Any]:
        response = self.post(
            f"{PANEL}/api/sessions",
            json={
                "path": notebook,
                "name": notebook,
                "type": "notebook",
                "kernel": {"name": KERNELSPEC},
            },
        )
        assert response.status_code == 201, response.text
        return response.json()

    def kernel_channels(self, kernel_id: str, session_id: str):
        return ws_connect(
            f"{WS_URL}{PANEL}/api/kernels/{kernel_id}/channels?session_id={session_id}",
            additional_headers=self.headers,
            max_size=None,
            open_timeout=60,
        )


def _wait_for_ready(terminal: Terminal, timeout: float) -> None:
    """Poll until the terminal answers and reports the notebook sidecar available."""
    deadline = time.monotonic() + timeout
    last = "(no attempt yet)"
    while time.monotonic() < deadline:
        try:
            response = terminal.get("/api/jupyter-server")
            if response.status_code == 200 and response.json().get("available"):
                return
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(3.0)
    raise AssertionError(
        f"the notebook panel never became available after {timeout:.0f}s (last: {last})\n"
        f"{_logs(_web_container())}"
    )


# ---------------------------------------------------------------------------
# Talking to a kernel over the proxied channels socket
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    stdout: str
    stderr: str
    error_name: str | None
    error_value: str
    traceback: str


def _message(msg_type: str, content: dict[str, Any], session_id: str) -> dict[str, Any]:
    return {
        "header": {
            "msg_id": uuid.uuid4().hex,
            "session": session_id,
            "username": "osprey",
            "date": datetime.now(UTC).isoformat(),
            "msg_type": msg_type,
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": content,
        "channel": "shell",
        "buffers": [],
    }


def _run_cell(socket: Any, code: str, session_id: str) -> CellResult:
    """Execute *code* and collect what it wrote, up to the ``idle`` status."""
    request = _message(
        "execute_request",
        {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        session_id,
    )
    socket.send(json.dumps(request))
    result = CellResult("", "", None, "", "")
    deadline = time.monotonic() + CELL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        raw = socket.recv(timeout=max(1.0, deadline - time.monotonic()))
        message = json.loads(raw)
        if (message.get("parent_header") or {}).get("msg_id") != request["header"]["msg_id"]:
            continue
        msg_type = message["header"]["msg_type"]
        content = message.get("content") or {}
        if msg_type == "stream":
            text = str(content.get("text", ""))
            if content.get("name") == "stdout":
                result.stdout += text
            else:
                result.stderr += text
        elif msg_type == "error":
            result.error_name = content.get("ename")
            result.error_value = _ANSI.sub("", str(content.get("evalue", "")))
            result.traceback = _ANSI.sub("", "\n".join(content.get("traceback", [])))
        elif msg_type == "status" and content.get("execution_state") == "idle":
            return result
    raise AssertionError(f"the cell did not finish within {CELL_TIMEOUT_SEC:.0f} s")


def _run_ok(socket: Any, code: str, session_id: str) -> str:
    result = _run_cell(socket, code, session_id)
    assert result.error_name is None, (
        f"cell raised {result.error_name}: {result.error_value}\n{result.traceback}\n"
        f"stderr: {result.stderr}"
    )
    return result.stdout


#: A cell that prints the stamps a kernel carries, as JSON.
STAMPS_CELL = (
    "import os, json\n"
    "print(json.dumps({k: v for k, v in os.environ.items() if k.startswith('OSPREY_CONTROL') "
    "or k in ('OSPREY_LAUNCH_POSTURE', 'OSPREY_POSTURE_SESSION', 'OSPREY_AGENT_DATA_ROOT') "
    "or k == 'JUPYTER_TOKEN'}))\n"
)
READ_CELL = f"from osprey.runtime import read_channel\nprint(repr(read_channel({CHANNEL!r})))\n"
WRITE_CELL = f"from osprey.runtime import write_channel\nwrite_channel({CHANNEL!r}, 1.0)\n"


def _stamps(socket: Any, session_id: str) -> dict[str, str]:
    return json.loads(_run_ok(socket, STAMPS_CELL, session_id))


def _read_value(socket: Any, session_id: str) -> float:
    printed = _run_ok(socket, READ_CELL, session_id).strip()
    try:
        return float(printed)
    except ValueError:
        raise AssertionError(
            f"read_channel({CHANNEL!r}) printed {printed!r}, not a number"
        ) from None


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def terminal(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Terminal]:
    """One ``osprey up --dev`` for the whole module; torn down whatever happens."""
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    tmp_path = tmp_path_factory.mktemp("notebook-panel")
    osprey_bin = _find_osprey_console_script()
    repo = _make_repo(tmp_path, osprey_bin)

    client = httpx.Client(timeout=60)
    try:
        _teardown(_compose_project())
        up = _run_osprey(osprey_bin, ["up", "--dev"], repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt("osprey up --dev (notebook panel)", up)
        secret = (parse_dotenv_file(repo / ".env").get(terminal_secret_var(USER)) or "").strip()
        assert secret, f"{terminal_secret_var(USER)} was not provisioned into the deploy .env"
        deployed = Terminal(repo=repo, osprey_bin=osprey_bin, secret=secret, client=client)
        _wait_for_ready(deployed, READY_TIMEOUT_SEC)
        yield deployed
    finally:
        project = _compose_project()
        _run_osprey(osprey_bin, ["down"], repo)
        _teardown(project)
        client.close()


# ---------------------------------------------------------------------------
# 1. First open
# ---------------------------------------------------------------------------


def test_the_starter_notebook_is_present_on_first_open(terminal: Terminal) -> None:
    contents = terminal.get(f"{PANEL}/api/contents")
    assert contents.status_code == 200, contents.text
    names = [entry["name"] for entry in contents.json()["content"]]
    assert names == [STARTER_NOTEBOOK], names

    kernelspecs = terminal.get(f"{PANEL}/api/kernelspecs")
    assert kernelspecs.status_code == 200, kernelspecs.text
    specs = kernelspecs.json()["kernelspecs"]
    assert sorted(specs) == [KERNELSPEC]
    terminal.agent_data_root = specs[KERNELSPEC]["spec"]["env"]["OSPREY_AGENT_DATA_ROOT"]

    starter = terminal.get(f"{PANEL}/api/contents/{STARTER_NOTEBOOK}", params={"content": "1"})
    assert starter.status_code == 200, starter.text
    terminal.starter_cells = starter.json()["content"]["cells"]
    assert any(
        cell["cell_type"] == "code" and "osprey.runtime" in cell["source"]
        for cell in terminal.starter_cells
    ), terminal.starter_cells


# ---------------------------------------------------------------------------
# 2. No chat session
# ---------------------------------------------------------------------------


def test_a_kernel_with_no_chat_session_reads_and_refuses_writes(terminal: Terminal) -> None:
    """Started before any terminal attached: no binding, fail-closed pin, reads route to the baseline."""
    assert terminal.get(f"{PANEL}/api/kernels").json() == []
    session = terminal.start_notebook_session(STARTER_NOTEBOOK)
    kernel_id = session["kernel"]["id"]
    channel_session = uuid.uuid4().hex
    records_before = len(terminal.ledger())
    try:
        with terminal.kernel_channels(kernel_id, channel_session) as socket:
            stamps = _stamps(socket, channel_session)
            assert "JUPYTER_TOKEN" not in stamps
            assert stamps.get("OSPREY_LAUNCH_POSTURE") == "*=sandbox", stamps
            assert "OSPREY_CONTROL_TARGET" not in stamps, stamps

            assert _read_value(socket, channel_session) == pytest.approx(500.0, abs=50.0)

            refused = _run_cell(socket, WRITE_CELL, channel_session)
            assert refused.error_name == "ChannelWriteBlockedError", refused
            assert f"Write to '{CHANNEL}' blocked" in refused.error_value, refused.error_value
            assert HINT_NO_SESSION in refused.stdout, refused.stdout
    finally:
        terminal.delete(f"{PANEL}/api/sessions/{session['id']}")

    new_records = terminal.ledger()[records_before:]
    assert len(new_records) == 1, new_records
    assert new_records[0]["surface"] == SURFACE
    assert new_records[0]["decision"] == "refused"
    assert new_records[0]["reason"] == "channel_write_blocked"


# ---------------------------------------------------------------------------
# 3. A terminal session binds the kernel launcher
# ---------------------------------------------------------------------------


def test_attaching_a_terminal_binds_the_session(terminal: Terminal) -> None:
    """The attach writes the binding; the controls server inside the session publishes its target."""
    with ws_connect(
        f"{WS_URL}/ws/terminal?mode=new",
        additional_headers=terminal.headers,
        max_size=None,
        open_timeout=60,
    ) as socket:
        deadline = time.monotonic() + 60
        while terminal.session_id is None and time.monotonic() < deadline:
            raw = socket.recv(timeout=max(1.0, deadline - time.monotonic()))
            if isinstance(raw, str) and raw.startswith("{"):
                frame = json.loads(raw)
                if frame.get("type") == "session_info":
                    terminal.session_id = frame["session_id"]
    assert terminal.session_id, "the terminal never sent session_info"

    assert terminal.agent_data_root
    binding = json.loads(
        _exec_text("cat", f"{terminal.agent_data_root}/jupyter/session-binding.json")
    )
    assert binding["session_id"] == terminal.session_id, binding
    assert binding["agent_data_root"] == terminal.agent_data_root, binding
    assert isinstance(binding["pty_pid"], int) and binding["pty_pid"] > 0, binding
    terminal.pty_pid = binding["pty_pid"]

    # The published record is the evidence: ``session_target`` on the posture
    # route answers the deployment baseline until the controls server inside
    # the fresh ``claude`` has written its state file, so a target read off the
    # route says nothing about whether a switch could be asked for yet.
    deadline = time.monotonic() + SESSION_TARGET_TIMEOUT_SEC
    record: dict[str, Any] | None = None
    while record is None and time.monotonic() < deadline:
        for entry in _state_records(terminal.agent_data_root):
            if entry.get("owner_ppid") == terminal.pty_pid:
                record = entry
        if record is None:
            time.sleep(2.0)
    assert record is not None, (
        f"no controls server published a state record for pid {terminal.pty_pid} within "
        f"{SESSION_TARGET_TIMEOUT_SEC:.0f}s\n{_logs(_web_container())}"
    )
    terminal.first_target = record["target"]
    assert terminal.first_target in ("va", "standin"), record
    assert terminal.posture()["session_target"] == terminal.first_target


# ---------------------------------------------------------------------------
# 4. Writes off at launch
# ---------------------------------------------------------------------------


def test_a_sandbox_pinned_kernel_refuses_writes_with_the_turn_writes_on_line(
    terminal: Terminal,
) -> None:
    assert terminal.session_id and terminal.first_target
    target = terminal.first_target
    narrowed = terminal.post(
        "/api/terminal/posture",
        json={"session_id": terminal.session_id, "target": target, "posture": "sandbox"},
    )
    assert narrowed.status_code == 200, narrowed.text
    assert narrowed.json()["entry"].get(target) == "sandbox", narrowed.text

    session = terminal.start_notebook_session(STARTER_NOTEBOOK)
    terminal.kernel_id = session["kernel"]["id"]
    terminal.notebook_session_id = session["id"]
    channel_session = uuid.uuid4().hex
    records_before = len(terminal.ledger())
    with terminal.kernel_channels(terminal.kernel_id, channel_session) as socket:
        stamps = _stamps(socket, channel_session)
        assert stamps.get("OSPREY_POSTURE_SESSION") == terminal.session_id, stamps
        assert stamps.get("OSPREY_CONTROL_TARGET") == target, stamps
        assert stamps.get("OSPREY_LAUNCH_POSTURE") == f"{target}=sandbox", stamps

        assert _read_value(socket, channel_session) == pytest.approx(500.0, abs=50.0)

        refused = _run_cell(socket, WRITE_CELL, channel_session)
        assert refused.error_name == "ChannelWriteBlockedError", refused
        assert (
            f"Write to '{CHANNEL}' blocked: this run launched while writes were off for "
            f"'{target}' in this session" in refused.error_value
        ), refused.error_value
        assert HINT_WRITES_OFF in refused.stdout, refused.stdout

    new_records = terminal.ledger()[records_before:]
    assert len(new_records) == 1, new_records
    record = new_records[0]
    assert record["surface"] == SURFACE
    assert record["decision"] == "refused"
    assert record["reason"] == "channel_write_blocked"
    assert record["session"] == terminal.session_id
    assert record["subject"] == "notebook_cell"
    assert record["detail"] == f"channel={CHANNEL}"


# ---------------------------------------------------------------------------
# 5. A chip target switch
# ---------------------------------------------------------------------------


def test_a_target_switch_is_refused_until_the_kernel_restarts(terminal: Terminal) -> None:
    assert terminal.session_id and terminal.first_target and terminal.kernel_id
    other = "standin" if terminal.first_target == "va" else "va"
    switched = terminal.post(
        "/api/terminal/target", json={"session_id": terminal.session_id, "target": other}
    )
    assert switched.status_code == 202, switched.text

    deadline = time.monotonic() + SWITCH_TIMEOUT_SEC
    posture = terminal.posture()
    while posture.get("session_target") != other and time.monotonic() < deadline:
        time.sleep(1.0)
        posture = terminal.posture()
    assert posture.get("session_target") == other, (
        f"the session never moved to {other!r}: {posture.get('last_switch')}"
    )

    records_before = len(terminal.ledger())
    channel_session = uuid.uuid4().hex
    try:
        with terminal.kernel_channels(terminal.kernel_id, channel_session) as socket:
            refused = _run_cell(socket, WRITE_CELL, channel_session)
            assert refused.error_name == "ControlTargetChangedError", refused
            assert "Refusing to write: this execution was started against control target" in (
                refused.error_value
            ), refused.error_value
            assert HINT_TARGET_CHANGED in refused.stdout, refused.stdout

        new_records = terminal.ledger()[records_before:]
        assert len(new_records) == 1, new_records
        assert new_records[0]["reason"] == "control_target_changed"
        assert new_records[0]["surface"] == SURFACE

        restarted = terminal.post(f"{PANEL}/api/kernels/{terminal.kernel_id}/restart")
        assert restarted.status_code == 200, restarted.text
        channel_session = uuid.uuid4().hex
        with terminal.kernel_channels(terminal.kernel_id, channel_session) as socket:
            followed = _run_ok(
                socket, 'import os\nprint(os.environ["OSPREY_CONTROL_TARGET"])\n', channel_session
            )
            assert followed.strip() == other, followed
    finally:
        terminal.delete(f"{PANEL}/api/sessions/{terminal.notebook_session_id}")


# ---------------------------------------------------------------------------
# 6. NotebookEdit: settings, guard, badge
# ---------------------------------------------------------------------------


def _run_hook(
    script: str, payload: dict[str, Any], env: dict[str, str]
) -> subprocess.CompletedProcess:
    """Run one rendered hook inside the container with a synthetic hook input."""
    args = [RUNTIME, "exec", "-i", "--user", "1000"]
    for name, value in env.items():
        args += ["-e", f"{name}={value}"]
    args += [_web_container(), "python3", script]
    return subprocess.run(
        args, input=json.dumps(payload), capture_output=True, text=True, timeout=60
    )


def test_notebook_edit_is_allowed_under_notebooks_and_badges_the_panel(terminal: Terminal) -> None:
    """The rendered allow rules, the guard's verdicts, and the badge the update hook posts.

    The hooks are driven the way ``tests/hooks`` drives them — the script on
    stdin with a synthetic tool call — but inside the container, under the
    environment of the ``claude`` process the terminal launched, which is what
    hands the update hook the panel token and web port it posts with.
    """
    assert terminal.agent_data_root and terminal.pty_pid
    settings = json.loads(
        (terminal.repo / "build" / PERSONA_PROJECT / ".claude" / "settings.json").read_text(
            encoding="utf-8"
        )
    )
    allow = settings["permissions"]["allow"]
    assert "NotebookEdit(var/agent_data/notebooks/**)" in allow, allow
    assert "NotebookEdit(var/agent_data/artifacts/**)" in allow, allow

    agent_env = _process_env(terminal.pty_pid)
    config_path = agent_env["OSPREY_CONFIG"]
    hooks_dir = str(Path(config_path).parent / ".claude" / "hooks")
    project_dir = str(Path(config_path).parent)
    hook_env = {
        name: agent_env[name]
        for name in ("OSPREY_PANEL_TOKEN", "OSPREY_WEB_PORT", "OSPREY_CONFIG")
        if name in agent_env
    }
    assert hook_env.get("OSPREY_PANEL_TOKEN"), "the agent carries no panel token"

    inside = f"{terminal.agent_data_root}/notebooks/x.ipynb"
    outside = f"{terminal.agent_data_root}/foo.ipynb"

    def pre_tool_use(path: str) -> dict[str, Any]:
        return {
            "hook_event_name": "PreToolUse",
            "tool_name": "NotebookEdit",
            "tool_input": {"notebook_path": path, "cell_id": "0", "new_source": "print(1)"},
            "cwd": project_dir,
        }

    allowed = _run_hook(f"{hooks_dir}/osprey_memory_guard.py", pre_tool_use(inside), hook_env)
    assert allowed.returncode == 0, _fmt("memory guard (inside)", allowed)
    assert json.loads(allowed.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"

    denied = _run_hook(f"{hooks_dir}/osprey_memory_guard.py", pre_tool_use(outside), hook_env)
    assert denied.returncode == 0, _fmt("memory guard (outside)", denied)
    verdict = json.loads(denied.stdout)["hookSpecificOutput"]
    assert verdict["permissionDecision"] == "deny", verdict
    assert "NOTEBOOK EDIT DENIED" in verdict["permissionDecisionReason"], verdict

    badged = _run_hook(
        f"{hooks_dir}/osprey_notebook_update.py",
        {**pre_tool_use(inside), "hook_event_name": "PostToolUse", "tool_response": {}},
        hook_env,
    )
    assert badged.returncode == 0, _fmt("notebook update", badged)
    assert badged.stdout == "", badged.stdout

    # The strip renders a NotebookEdit panel frame as "agent edited <detail>"
    # (static/js/activity-format.js); the frame is what the history endpoint holds.
    recent = terminal.get("/api/agent-activity/recent")
    assert recent.status_code == 200, recent.text
    frames = [event for event in recent.json()["events"] if event["tool"] == "NotebookEdit"]
    assert frames, recent.text
    assert frames[0]["target"] == {"kind": "panel", "panel": "jupyter", "detail": "x.ipynb"}


# ---------------------------------------------------------------------------
# 7. The runtime directory
# ---------------------------------------------------------------------------


def test_the_runtime_dir_is_not_under_the_agent_data_root(terminal: Terminal) -> None:
    assert terminal.agent_data_root
    pids = _exec_text("pgrep", "-f", "jupyter_sidecar_main").split()
    assert len(pids) == 1, pids
    sidecar_env = _process_env(int(pids[0]))
    runtime_dir = sidecar_env["JUPYTER_RUNTIME_DIR"]
    root = terminal.agent_data_root.rstrip("/") + "/"
    assert not runtime_dir.startswith(root), runtime_dir
    assert sidecar_env["JUPYTER_DATA_DIR"].startswith(root), sidecar_env["JUPYTER_DATA_DIR"]
    listing = _exec_text("ls", runtime_dir).split()
    assert f"jpserver-{pids[0]}.json" in listing, listing


# ---------------------------------------------------------------------------
# 8. down && up
# ---------------------------------------------------------------------------


def test_notebooks_survive_a_restart_and_kernels_do_not(terminal: Terminal) -> None:
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('survives')")])
    saved = terminal.put(
        f"{PANEL}/api/contents/{SURVIVOR_NOTEBOOK}",
        json={"type": "notebook", "format": "json", "content": notebook},
    )
    assert saved.status_code == 201, saved.text
    terminal.start_notebook_session(SURVIVOR_NOTEBOOK)
    assert len(terminal.get(f"{PANEL}/api/kernels").json()) >= 1

    down = _run_osprey(terminal.osprey_bin, ["down"], terminal.repo)
    assert down.returncode == 0, _fmt("osprey down", down)
    up = _run_osprey(
        terminal.osprey_bin, ["up", "--dev"], terminal.repo, timeout=DEPLOY_UP_TIMEOUT_SEC
    )
    assert up.returncode == 0, _fmt("osprey up --dev (second start)", up)
    _wait_for_ready(terminal, READY_TIMEOUT_SEC)

    names = sorted(
        entry["name"] for entry in terminal.get(f"{PANEL}/api/contents").json()["content"]
    )
    assert names == sorted([STARTER_NOTEBOOK, SURVIVOR_NOTEBOOK]), names
    assert terminal.get(f"{PANEL}/api/kernels").json() == []
    assert terminal.get(f"{PANEL}/api/sessions").json() == []

    survivor = terminal.get(f"{PANEL}/api/contents/{SURVIVOR_NOTEBOOK}", params={"content": "1"})
    assert survivor.json()["content"]["cells"][0]["source"] == "print('survives')"
    # The seed is written into an EMPTY tree only; a restart never rewrites it.
    starter = terminal.get(f"{PANEL}/api/contents/{STARTER_NOTEBOOK}", params={"content": "1"})
    assert starter.json()["content"]["cells"] == terminal.starter_cells
