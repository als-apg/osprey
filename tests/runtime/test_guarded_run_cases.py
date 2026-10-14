"""The run lock across processes, control targets and the executor's sandbox.

One lock per control target serves every process of a deployment, so these
cases run more than one process against one deployment repo: a run killed in
another container, a run on another target, and the wrapped scripts the
executor launches for ``execute``. A killed run leaves the machine where it
stopped and frees its target. The machine is a JSON file of setpoints that
every process reads and writes through ``osprey.runtime``; no control system is
involved. A child announces a phase by writing one byte to a pipe it inherits,
and the test waits on that pipe.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import GUARDED_RUN_DIR, LOCK_FILE_NAME, OspreyRunBusy, lock
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey_connectors.workspace import reset_config_cache
from tests._guarded_run import lock_path, reset_run_state, write_config

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

#: How long a child gets to reach a phase or to finish; a hang guard, never a
#: statement about speed.
CHILD_TIMEOUT_S = 120.0

#: The setpoints of the machine before any run.
HOME = {"Q": 3.0, "S": 5.0, "T": 8.0}

#: Where a child finds the machine and the pipe it announces a phase on.
ENV_VALUES = "GUARDED_RUN_CASES_VALUES"
ENV_READY_FD = "GUARDED_RUN_CASES_READY_FD"

#: Installed as ``sitecustomize`` in every child: ``osprey.runtime`` reads and
#: writes the machine file before any other code of the child runs.
_SITECUSTOMIZE = textwrap.dedent(
    f"""
    import json, os

    import osprey.runtime

    _path = os.environ[{ENV_VALUES!r}]

    def _load():
        with open(_path) as handle:
            return json.load(handle)

    def _read_channels(addresses, **_kwargs):
        values = _load()
        return [values[a] for a in addresses]

    def _write_channel(address, value, **_kwargs):
        values = _load()
        values[address] = value
        with open(_path, "w") as handle:
            json.dump(values, handle)

    osprey.runtime.read_channels = _read_channels
    osprey.runtime.write_channel = _write_channel
    osprey.runtime.channel_limits = lambda address: None
    """
)

#: The last statements of a child that stays inside its run until it is killed.
_ANNOUNCE_AND_WAIT = textwrap.dedent(
    f"""
    os.write(int(os.environ[{ENV_READY_FD!r}]), b"1")
    while True:
        signal.pause()
    """
)

#: A run that displaces ``Q`` under the target's lock and then stays inside it.
_DISPLACE_AND_WAIT = textwrap.dedent(
    """
    import os, signal

    import osprey.runtime
    from osprey.runtime.guarded_run import lock

    with lock(None):
        osprey.runtime.write_channel("Q", 9.0)
    """
) + textwrap.indent(_ANNOUNCE_AND_WAIT, "    ")

#: User code of a readwrite ``execute``: write ``Q`` under the run's lock, stay inside.
_WRITE_UNDER_THE_LOCK_AND_WAIT = (
    textwrap.dedent(
        """
    import os, signal

    import osprey.runtime

    osprey.runtime.write_channel("Q", 9.0)
    """
    )
    + _ANNOUNCE_AND_WAIT
)

#: User code that takes the run on this process's own target and names its directory.
_TAKE_THE_RUN = textwrap.dedent(
    """
    from osprey.runtime.guarded_run import lock

    with lock(None) as directory:
        print(directory, flush=True)
    """
)


class _Machine:
    """The machine file as this process's ``osprey.runtime`` reads and writes it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writes: list[tuple[str, Any]] = []
        path.write_text(json.dumps(HOME), encoding="utf-8")

    @property
    def values(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def set(self, address: str, value: Any) -> None:
        """Move a setpoint as something outside any run would."""
        self.path.write_text(json.dumps({**self.values, address: value}), encoding="utf-8")

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        values = self.values
        return [values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.writes.append((address, value))
        self.set(address, value)

    def channel_limits(self, _address: str) -> None:
        return None


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A host deployment repo on a live baseline, entered and stamped ``live``.

    The approval hook is wired and every tool's call is put to a human. Nothing
    is held by this process, and the config caches are empty going in and
    coming out.
    """
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root)
    site = root / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    reset_run_state(monkeypatch)
    yield root
    reset_config_cache()


@pytest.fixture
def machine(repo: Path, monkeypatch: pytest.MonkeyPatch) -> _Machine:
    """The machine at :data:`HOME`, served to this process's ``osprey.runtime``."""
    fake = _Machine(repo / "values.json")
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


def _child_env(repo: Path, **extra: str) -> dict[str, str]:
    """A child's environment: on the machine file, unstamped unless *extra* stamps it."""
    env = os.environ.copy()
    for name in (
        "OSPREY_CONFIG",
        "CONFIG_FILE",
        "OSPREY_EXECUTION_MODE",
        "OSPREY_AGENT_DATA_ROOT",
        "OSPREY_TERMINAL_USER",
        ENV_CONTROL_TARGET,
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo / "site"), str(_SRC_ROOT), env.get("PYTHONPATH")])
    )
    env[ENV_VALUES] = str(repo / "values.json")
    env.update(extra)
    return env


class _Held:
    """A child that announces one phase and then stays where it is until killed."""

    def __init__(self, argv: list[str], repo: Path, env: dict[str, str]) -> None:
        self._read, write = os.pipe()
        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=repo,
                env={**env, ENV_READY_FD: str(write)},
                pass_fds=(write,),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        finally:
            os.close(write)

    def wait_ready(self) -> None:
        """Return once the child announced its phase; fail if it ended or hung first."""
        readable, _, _ = select.select([self._read], [], [], CHILD_TIMEOUT_S)
        if readable and os.read(self._read, 1) == b"1":
            return
        self.proc.kill()
        out, err = self.proc.communicate()
        pytest.fail(
            f"the child never announced its phase (exit {self.proc.returncode}):\n{out}{err}"
        )

    def kill(self) -> None:
        """SIGKILL the child and reap it."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.communicate()
        os.close(self._read)


class _Execute:
    """One ``execute`` call's wrapped script, in an execution folder of the repo."""

    def __init__(self, repo: Path, mode: str, code: str) -> None:
        self.repo = repo
        self.mode = mode
        self.folder = repo / "exec"
        self.folder.mkdir(exist_ok=True)
        self.script = self.folder / "wrapped_script.py"
        self.script.write_text(
            ExecutionWrapper(execution_mode=mode).create_wrapper(code, self.folder),
            encoding="utf-8",
        )

    def env(self, **extra: str) -> dict[str, str]:
        stamp = {"OSPREY_EXECUTION_MODE": self.mode, ENV_CONTROL_TARGET: "live"}
        return _child_env(self.repo, **stamp, **extra)

    def hold(self, **extra: str) -> _Held:
        """Start the script, whose code announces a phase and stays there."""
        return _Held([sys.executable, str(self.script)], self.repo, self.env(**extra))


def test_another_containers_killed_run_is_busy_while_it_lives_and_free_after_it_dies(
    repo: Path, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes with an agent-data root each share the target's lock."""
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(repo / "agent_data_b"))
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "bob")
    other = _Held(
        [sys.executable, "-c", _DISPLACE_AND_WAIT],
        repo,
        _child_env(
            repo,
            OSPREY_AGENT_DATA_ROOT=str(repo / "agent_data_a"),
            OSPREY_TERMINAL_USER="alice",
            **{ENV_CONTROL_TARGET: "live"},
        ),
    )
    try:
        other.wait_ready()
        with pytest.raises(OspreyRunBusy) as busy, lock(None):
            pytest.fail("a run never starts while another container's run holds the target")
    finally:
        other.kill()

    assert busy.value.pid == other.proc.pid
    assert lock_path().parent == repo.resolve() / "var" / GUARDED_RUN_DIR / "live"

    with lock(None):
        pass

    assert machine.writes == []
    assert machine.values["Q"] == 9.0, "the killed run left its setpoint where it stopped"


def test_a_killed_readwrite_execute_frees_the_target(repo: Path, machine: _Machine) -> None:
    """``execute`` holds the target's lock for its writes; killed, it frees it."""
    held = _Execute(repo, "readwrite", _WRITE_UNDER_THE_LOCK_AND_WAIT).hold()
    try:
        held.wait_ready()
        with pytest.raises(OspreyRunBusy), lock(None):
            pytest.fail("the execute holds the target's lock while it writes")
    finally:
        held.kill()

    with lock(None) as directory:
        pass

    assert sorted(path.name for path in directory.iterdir()) == [LOCK_FILE_NAME]
    assert machine.writes == []
    assert machine.values["Q"] == 9.0, "a write the killed code made stays as it was written"


def test_a_held_live_run_does_not_block_a_run_on_va(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstamped run on a live baseline holds ``live``; ``va`` has a lock of its own."""
    monkeypatch.delenv(ENV_CONTROL_TARGET)

    with lock(None) as held:
        other = subprocess.run(
            [sys.executable, "-c", _TAKE_THE_RUN],
            cwd=repo,
            env=_child_env(repo, **{ENV_CONTROL_TARGET: "va"}),
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
            check=False,
        )
        holder = json.loads((held / LOCK_FILE_NAME).read_text(encoding="utf-8"))

    state = repo.resolve() / "var" / GUARDED_RUN_DIR
    assert held == state / "live"
    assert holder["pid"] == os.getpid()
    assert other.returncode == 0, other.stderr
    assert Path(other.stdout.strip()) == state / "va"
    assert (state / "va" / LOCK_FILE_NAME).exists()
