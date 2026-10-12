"""A built deployment whose measurements reach a served machine the way a session's do.

Three live lanes drive ``pyaml_measure`` against a container: the guarded-run
cases, check B and the chain. Each needs the same pieces, kept here once:

* **The deployment.** A real ``osprey init`` and ``osprey build`` of a repo
  whose only control target is the virtual accelerator on loopback, armed for
  writes, whose default set -- the built-in ``still`` scenario -- is
  activated in its state directory as ``osprey up`` activates it. The connector table is
  stated whole, so no block of the preset's that names another machine
  survives into the render: every target the render configures resolves to the
  simulator (:func:`only_simulated_targets`).
* **The container.** One per deployment, over the build's own data root and
  the deployment's state directory, published on the port the build was given.
* **A session's environment.** The rendered config, an agent-data root and a
  posture session, set the way a session child carries them; the hook, the
  tool and the sandbox the tool spawns all resolve from them.
* **The approval hook.** The build's own rendered ``osprey_approval.py`` run as
  the harness runs it, on a PreToolUse payload; an ``ask`` that is approved
  hands the tool the input the hook rewrote, under the same tool-use id, which
  is how the executor finds the stamp the hook left.

Every Channel Access read here runs in a subprocess that leaves through
``os._exit``, for the reasons the directory conftest gives; this process
never becomes a Channel Access client. The executor sandbox is a subprocess
too, so the tool can be called in this process.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.va.e2e import conftest as e2e_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The source trees a sandbox imports, stated rather than inherited.
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The image every container here serves from.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", e2e_conftest.IMAGE)

#: A local run on Apple Silicon is emulated.
BOOT_TIMEOUT_S = 240.0

#: Bound on one Channel Access subprocess.
CA_CHILD_TIMEOUT_S = 120.0

#: The sandbox deadline every measurement runs under: a facility-scale orbit
#: response on an emulated container reads every monitor at every step.
EXECUTION_TIMEOUT_S = 900

#: The prefix of the one line a Channel Access child reports on.
REPORT_MARK = "REPORT "

#: The guarded-run directory of the deployment's one target.
VA_TARGET = "va"

#: The approval hook the build renders, relative to the render root.
APPROVAL_HOOK = Path(".claude") / "hooks" / "osprey_approval.py"

#: The scenario every deployment here starts in.
STILL = "still"


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


#: Where a build's port is drawn from: below every common ephemeral range
#: (Linux from 32768, macOS from 49152), so an outgoing connection the host
#: opens while the build runs does not take the port the build named.
PORT_RANGE = (20000, 30000)

#: How many ports a deployment tries before giving up on publishing one.
PUBLISH_ATTEMPTS = 3


def reserve_port() -> int:
    """A host port free at reservation time, for the build to name before the boot."""
    import random

    for _ in range(200):
        port = random.randrange(*PORT_RANGE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {PORT_RANGE}")


def isolation_settings(port: int, probe: str, *, approval: bool) -> list[str]:
    """The ``osprey set`` arguments of a deployment whose one machine is the simulator.

    The connector table is stated whole, so every block the preset carried is
    replaced by one ``virtual_accelerator`` block on ``127.0.0.1:<port>``
    armed for writes, and the stand-in is switched off. The approval block is
    stated, never left to a default: ``approval`` false is a deployment that
    asks nobody, true one whose every guarded tool asks.
    """
    gateway = {"address": "127.0.0.1", "port": port, "use_name_server": True}
    connector = {
        "virtual_accelerator": {
            "gateways": {"read_only": gateway, "write_access": gateway},
            "probe_channel": probe,
            "writes_enabled": True,
            "timeout_s": 30.0,
        }
    }
    return [
        "connector=virtual_accelerator",
        "config.control_system.writes_enabled=true",
        f"config.control_system.connector={json.dumps(connector)}",
        f"virtual_accelerator.port={port}",
        "virtual_accelerator.live_standin=null",
        f"config.simulation.default_scenarios=[{STILL}]",
        f"config.approval.enabled={'true' if approval else 'false'}",
        "config.approval.default_policy=always",
        "config.approval.tools.pyaml_measure=always",
        "config.approval.tools.execute=always",
        f"config.python_executor.execution_timeout_seconds={EXECUTION_TIMEOUT_S}",
    ]


def osprey(*args: str) -> str:
    """Run one ``osprey`` verb in this process and insist it succeeded; its output."""
    from click.testing import CliRunner

    from osprey.cli.main import cli

    result = CliRunner().invoke(cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, f"osprey {' '.join(args)} failed:\n{result.output}"
    return result.output


def set_measurement(repo: Path, model: str, **changes: Any) -> None:
    """Edit one model's measurement file in the deployment, as an operator does."""
    path = repo / "data" / "facility" / "measurement" / f"{model}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    document.update(changes)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def build(repo: Path, settings: Sequence[str]) -> None:
    """Apply ``settings`` to ``repo`` and build it."""
    osprey("set", "--repo", str(repo), *settings)
    osprey("build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")


@dataclass(frozen=True)
class Deployment:
    """One built repo and the port its machine is served on.

    Attributes:
        repo: The deployment repo.
        port: The host port the container publishes and the build names.
    """

    repo: Path
    port: int

    @property
    def build_dir(self) -> Path:
        """The render root: the directory the agent runs in."""
        return self.repo / "build"

    @property
    def config_path(self) -> Path:
        """The rendered ``config.yml``."""
        return self.build_dir / "config.yml"

    @property
    def data_root(self) -> Path:
        """The render's data root, which the container mounts."""
        return self.build_dir / "data"

    @property
    def config(self) -> dict[str, Any]:
        """The rendered config, parsed."""
        document: dict[str, Any] = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        return document

    def journal(self, target: str = VA_TARGET) -> Path:
        """The guarded-run journal of ``target``: deployment-wide, under the repo's state zone."""
        from osprey.runtime.guarded_run import GUARDED_RUN_DIR, JOURNAL_FILE_NAME

        return self.repo / "var" / GUARDED_RUN_DIR / target / JOURNAL_FILE_NAME

    def view(self) -> Any:
        """The simulator view the container serves."""
        from osprey_connectors.simulation.view import VIEW_RELPATH, SimulatorView

        return SimulatorView.open(self.data_root / VIEW_RELPATH.name)

    def pyaml_view(self, model: str) -> Path:
        """The pyAML configuration the build wrote for ``model``."""
        return self.data_root / "pyaml" / model / "configuration.yaml"

    @property
    def state_dir(self) -> Path:
        """The simulation state directory: the active set the container serves."""
        from osprey_connectors.workspace import resolve_simulation_state_dir

        return resolve_simulation_state_dir(self.config, self.repo)

    def active_scenarios(self) -> list[str]:
        """The scenarios the container serves, as the state directory records them."""
        from osprey_connectors.simulation.state import read_active_state

        names, _anchor = read_active_state(self.state_dir)
        return [str(name) for name in names]

    def activate_default_scenarios(self) -> None:
        """Activate the profile's default set, as ``osprey up`` does on a first start."""
        from osprey.simulation.apply import activate_default_scenarios

        activate_default_scenarios(self.config, self.repo)
        assert self.active_scenarios(), "the build activated no scenario"
        readable_by_anyone(self.state_dir)


def readable_by_anyone(root: Path) -> None:
    """Open ``root``'s tree to the container's user, which is not this process's."""
    for directory, _subdirs, files in os.walk(root):
        Path(directory).chmod(0o755)
        for name in files:
            (Path(directory) / name).chmod(0o644)


def only_simulated_targets(config: Mapping[str, Any]) -> list[str]:
    """Every target the render configures that is not the simulator; empty when none is.

    A write reaches hardware only through a target that resolves to a real
    control system, so a render with none of them records zero hardware writes
    by construction.
    """
    from osprey_connectors.types import configured_targets, resolve_target

    section = config.get("control_system") or {}
    found: list[str] = []
    for target in configured_targets(section):
        kind = resolve_target(section, target)
        if kind != "virtual_accelerator":
            found.append(f"{target} -> {kind}")
    return found


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


def _docker(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def require_image() -> None:
    """Fail loudly unless the image is present."""
    inspected = _docker("image", "inspect", IMAGE, "--format", "{{.Id}}", timeout=60)
    if inspected.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} is not present. Build it with "
            "scripts/va/build_and_boot_check.sh, or name another with OSPREY_VA_E2E_IMAGE."
        )


class PortTaken(RuntimeError):
    """The port a build named was taken before its container could publish it."""


@contextmanager
def deployed(repo: Path, settings: Any, *, prefix: str, probe: str) -> Iterator[Deployment]:
    """Build ``repo`` on a fresh port and serve it; the deployment, while it is served.

    ``settings`` maps a port to the ``osprey set`` arguments of the build. A
    port taken between the build and the boot is replaced by another and the
    repo rebuilt, since the port is part of the render.
    """
    for _ in range(PUBLISH_ATTEMPTS):
        port = reserve_port()
        build(repo, settings(port))
        deployment = Deployment(repo=repo, port=port)
        deployment.activate_default_scenarios()
        try:
            with serving(deployment, prefix=prefix, probe=probe):
                yield deployment
            return
        except PortTaken:
            continue
    raise RuntimeError(f"no port could be published in {PUBLISH_ATTEMPTS} attempts")


@contextmanager
def serving(deployment: Deployment, *, prefix: str, probe: str) -> Iterator[None]:
    """Boot one container over ``deployment``'s data root on its port; wait until it serves.

    The published port and the server's own are one number, which also names
    the container, so a concurrent run cannot remove it.
    """
    require_image()
    port = deployment.port
    name = f"{prefix}-{port}-{uuid.uuid4().hex[:8]}"
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        *e2e_conftest.DEMO_NAMESPACE_RUN_ARGS,
        *e2e_conftest.data_root_run_args(deployment.data_root),
        # The active set is a separate mount, read-only: the state is the
        # deployment's, written on the host.
        "-v",
        f"{deployment.state_dir}:/state/simulation:ro",
        "-e",
        "VA_STATE_DIR=/state/simulation",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        IMAGE,
    )
    if started.returncode != 0:
        _docker("rm", "-f", name, timeout=60)
        if "address already in use" in started.stderr:
            raise PortTaken(port)
        raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        last = "no read completed"
        while time.monotonic() < deadline:
            try:
                if ca_read(port, [probe], timeout=2.0)[probe] is not None:
                    break
                last = f"{probe} read back as None"
            except (subprocess.TimeoutExpired, AssertionError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0)
        else:
            logs = _docker("logs", "--tail", "60", name, timeout=60)
            raise AssertionError(
                f"{name} never served {probe} within {BOOT_TIMEOUT_S}s ({last}).\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        yield
    finally:
        _docker("rm", "-f", name, timeout=60)


def ca_read(port: int, addresses: Sequence[str], *, timeout: float = 10.0) -> dict[str, Any]:
    """Each address's value on the container serving ``port``, read out of process.

    An address that does not answer reads ``None``; a waveform reads as a list.
    """
    code = (
        "import json, os, sys, epics\n"
        f"names = {list(addresses)!r}\n"
        f"values = epics.caget_many(names, connection_timeout={timeout}, timeout={timeout})\n"
        "def plain(v):\n"
        "    if v is None: return None\n"
        "    try: return float(v)\n"
        "    except TypeError: return [float(x) for x in v]\n"
        "out = {n: plain(v) for n, v in zip(names, values)}\n"
        f"sys.stdout.write('\\n' + {REPORT_MARK!r} + json.dumps(out) + '\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
        environment.pop(stale, None)
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=CA_CHILD_TIMEOUT_S,
        env=environment,
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith(REPORT_MARK)]
    assert lines, f"the Channel Access child reported nothing:\n{run.stdout}\n{run.stderr}"
    report: dict[str, Any] = json.loads(lines[-1][len(REPORT_MARK) :])
    return report


# ---------------------------------------------------------------------------
# A session's environment
# ---------------------------------------------------------------------------


@contextmanager
def acting_in(
    deployment: Deployment,
    agent_data: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Run the block as a session of ``deployment`` whose agent-data root is ``agent_data``.

    Two sessions with different roots are two containers of one deployment:
    they share the repo's guarded-run directory and nothing else. Every stamp
    a session child must not inherit from this process is removed.
    """
    from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
    from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
    from osprey_connectors import posture_store
    from osprey_connectors.workspace import reset_config_cache

    agent_data.mkdir(parents=True, exist_ok=True)
    values = {
        "OSPREY_CONFIG": str(deployment.config_path),
        # The sandbox wrapper initialises its registry from this name.
        "CONFIG_FILE": str(deployment.config_path),
        posture_store.AGENT_DATA_ROOT_ENV_VAR: str(agent_data),
        POSTURE_SESSION_ENV_VAR: str(uuid.uuid5(uuid.NAMESPACE_URL, str(agent_data))),
        "PYTHONPATH": os.pathsep.join(REPO_PATHS),
        **dict(environment or {}),
    }
    removed = (
        "OSPREY_EXECUTION_MODE",
        "OSPREY_DISPATCH_RUN",
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        ENV_CONTROL_TARGET,
        ENV_CONTROL_TARGET_GENERATION,
    )
    with pytest.MonkeyPatch.context() as patch:
        for name in removed:
            if name not in values:
                patch.delenv(name, raising=False)
        for name, value in values.items():
            patch.setenv(name, value)
        reset_config_cache()
        posture_store.invalidate_cache()
        try:
            yield
        finally:
            reset_config_cache()
            posture_store.invalidate_cache()


# ---------------------------------------------------------------------------
# The approval hook and the tool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Asked:
    """What the approval hook answered for one call.

    Attributes:
        decision: ``ask``, ``deny``, ``allow`` or ``None`` when it decided nothing.
        reason: The prompt text, or the refusal.
        updated_input: The input an approved call runs with.
        tool_use_id: The call's id, under which the hook left its stamp.
    """

    decision: str | None
    reason: str
    updated_input: dict[str, Any]
    tool_use_id: str


def ask(
    deployment: Deployment,
    tool: str,
    tool_input: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> Asked:
    """Run the build's approval hook on one PreToolUse call of ``tool``.

    Called inside :func:`acting_in`, whose environment the hook inherits; the
    harness runs the hook with the render as its project directory.
    """
    tool_use_id = f"toolu_{uuid.uuid4().hex[:20]}"
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": f"mcp__python__{tool}",
        "tool_input": dict(tool_input),
        "tool_use_id": tool_use_id,
        "cwd": str(deployment.build_dir),
    }
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(deployment.build_dir), **(environment or {})}
    done = subprocess.run(
        [sys.executable, str(deployment.build_dir / APPROVAL_HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(deployment.build_dir),
        timeout=120,
    )
    assert done.returncode == 0, f"the approval hook failed:\n{done.stdout}\n{done.stderr}"
    output: dict[str, Any] = {}
    for line in reversed(done.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            output = json.loads(line)
            break
    specific = output.get("hookSpecificOutput") or {}
    return Asked(
        decision=specific.get("permissionDecision"),
        reason=str(specific.get("permissionDecisionReason") or ""),
        updated_input=dict(specific.get("updatedInput") or tool_input),
        tool_use_id=tool_use_id,
    )


@dataclass(frozen=True)
class Answer:
    """What a tool call returned or raised.

    Attributes:
        result: The parsed response of a call that returned, else ``None``.
        error: The parsed error envelope of a call that raised, else ``None``.
    """

    result: dict[str, Any] | None
    error: dict[str, Any] | None

    @property
    def text(self) -> str:
        """Everything the answer says, for a refusal's words."""
        return json.dumps(self.result if self.error is None else self.error)


async def _answer(call: Any) -> Answer:
    from fastmcp.exceptions import ToolError

    try:
        returned = await call
    except ToolError as exc:
        try:
            envelope = json.loads(str(exc))
        except ValueError:
            envelope = {"error_message": str(exc)}
        return Answer(result=None, error=envelope)
    if isinstance(returned, str):
        try:
            return Answer(result=json.loads(returned), error=None)
        except ValueError:
            return Answer(result={"text": returned}, error=None)
    return Answer(result={"text": str(returned)}, error=None)


def _tool(name: str) -> Any:
    if name == "pyaml_measure":
        from osprey.mcp_server.python_executor.tools.pyaml_measure import pyaml_measure

        return pyaml_measure
    from osprey.mcp_server.python_executor.tools import python_execute
    from tests.mcp_server.conftest import get_tool_fn

    return get_tool_fn(python_execute.execute)


async def call(tool: str, arguments: Mapping[str, Any], tool_use_id: str | None = None) -> Answer:
    """Call ``tool`` as the server does, under ``tool_use_id``; what it answered.

    An approved call is made with the hook's ``updated_input`` and the
    ``tool_use_id`` the hook stamped; a call no hook saw has none.
    """
    from osprey.audit.call import call_scope

    with call_scope(tool_use_id, None):
        return await _answer(_tool(tool)(**dict(arguments)))


async def approved(
    deployment: Deployment, tool: str, tool_input: Mapping[str, Any]
) -> tuple[Asked, Answer]:
    """Ask the hook, insist it asked, and run the call it approved."""
    asked = ask(deployment, tool, tool_input)
    assert asked.decision == "ask", f"the hook answered {asked.decision}: {asked.reason}"
    return asked, await call(tool, asked.updated_input, asked.tool_use_id)


# ---------------------------------------------------------------------------
# A run killed after its first write
# ---------------------------------------------------------------------------

#: How often the journal is read while a run is waited on.
JOURNAL_POLL_S = 0.02

#: How long after the journal records its first setpoint the run is killed.
#: The write follows the record at once and the measurement then waits its
#: settle time (:data:`SETTLE_S`) before the next write, so the kill lands
#: after the first write and before the second.
KILL_AFTER_RECORD_S = 0.5

#: The settle time the deployments here give every measurement step, in
#: seconds: long enough that a kill timed from the first record lands inside
#: it.
SETTLE_S = 2.0

#: Bound on a run reaching its first write.
FIRST_WRITE_TIMEOUT_S = 300.0


def pending(deployment: Deployment, target: str = VA_TARGET) -> Any:
    """The pending journal of ``target``, or ``None`` when it holds no record."""
    from osprey.runtime.journal import parse_pending_journal

    path = deployment.journal(target)
    if not path.is_file():
        return None
    return parse_pending_journal(path.read_bytes(), path)


async def killed_after_first_write(
    deployment: Deployment, tool: str, arguments: Mapping[str, Any], tool_use_id: str | None
) -> tuple[Any, Answer]:
    """Start a call, kill its sandbox once its first setpoint is written; what it left.

    The journal's header names the process the guarded run runs in, and its
    first record is written immediately before the first write; the kill comes
    :data:`KILL_AFTER_RECORD_S` later, inside the step's settle time.

    Returns:
        The pending journal the killed run left, and what the call answered.
    """
    from osprey.audit.call import call_scope

    assert pending(deployment) is None, "a journal is pending before the run starts"
    with call_scope(tool_use_id, None):
        task = asyncio.ensure_future(_answer(_tool(tool)(**dict(arguments))))
    deadline = time.monotonic() + FIRST_WRITE_TIMEOUT_S
    left = None
    while time.monotonic() < deadline and not task.done():
        left = pending(deployment)
        if left is not None and left.pid is not None:
            break
        await asyncio.sleep(JOURNAL_POLL_S)
    assert not task.done(), f"the run ended before its first write: {(await task).text}"
    assert left is not None, f"no setpoint was journaled within {FIRST_WRITE_TIMEOUT_S}s"
    await asyncio.sleep(KILL_AFTER_RECORD_S)
    os.kill(int(left.pid), signal.SIGKILL)
    answer = await task
    journal = pending(deployment)
    assert journal is not None, "the killed run left no journal"
    return journal, answer
