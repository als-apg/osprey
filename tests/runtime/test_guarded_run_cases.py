"""The guarded run across processes, control targets and the executor's sandbox.

One lock and one journal per control target serve every process of a
deployment, so these cases run more than one process against one deployment
repo: a run killed in another container, a process rewriting the journal after
a call was approved, a run on another target, and the wrapped scripts the
executor launches for ``execute``. The machine is a JSON file of setpoints that
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
from osprey.errors import ChannelWriteBlockedError
from osprey.mcp_server.python_executor import executor
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import (
    APPROVED_NO_JOURNAL,
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
    GUARDED_RUN_DIR,
    LOCK_FILE_NAME,
    RESTORE_REPORT_TAG,
    OspreyJournalChanged,
    OspreyRunBusy,
    lock,
)
from osprey.runtime.journal import OspreyRestoreIncomplete, read_pending_journal
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey_connectors.workspace import reset_config_cache
from tests.runtime._guarded_run_fixtures import (
    GENERATION,
    approve,
    digest,
    journal_path,
    plant,
    reset_run_state,
    write_config,
)

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

#: A journaled run that displaces ``Q`` and then stays inside the run.
_DISPLACE_AND_WAIT = textwrap.dedent(
    """
    import os, signal

    import osprey.runtime
    from osprey.runtime.guarded_run import journaled_run
    from osprey.runtime.journal import guarded_write

    with journaled_run(None):
        guarded_write(["Q"], lambda: osprey.runtime.write_channel("Q", 9.0), lambda exc: exc)
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

#: User code of an ``execute`` that tries the run and records what it met.
_TRY_THE_RUN = textwrap.dedent(
    f"""
    import json, os

    from osprey.runtime.guarded_run import lock

    refusal = None
    try:
        with lock(None):
            pass
    except Exception as exc:
        refusal = f"{{type(exc).__name__}}: {{exc}}"
    with open(os.environ[{ENV_VALUES!r}]) as handle:
        results = {{"seen": json.load(handle), "refusal": refusal}}
    """
)


class _Machine:
    """The machine file as this process's ``osprey.runtime`` reads and writes it.

    With ``writes_enabled`` off every write is refused the way a deployment
    whose writes are disabled refuses it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writes: list[tuple[str, Any]] = []
        self.writes_enabled = True
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
        if not self.writes_enabled:
            raise ChannelWriteBlockedError(
                address, "WRITES_DISABLED", f"Write to '{address}' blocked: writes are disabled."
            )
        self.writes.append((address, value))
        self.set(address, value)

    def channel_limits(self, _address: str) -> None:
        return None


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A host deployment repo on a live baseline, entered and stamped ``live``.

    The approval hook is wired and every tool's call is put to a human. No call
    is bound to this process, and the config caches are empty going in and
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
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, str(GENERATION))
    for name in (
        "OSPREY_EXECUTION_MODE",
        "OSPREY_EXECUTION_DEADLINE",
        ENV_APPROVED_JOURNAL_SHA256,
        ENV_APPROVED_TARGET,
    ):
        monkeypatch.delenv(name, raising=False)
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


def _report_lines(*streams: str) -> list[str]:
    """Every line of *streams* that starts with the restore report's tag."""
    return [
        line
        for text in streams
        for line in text.splitlines()
        if line.startswith(RESTORE_REPORT_TAG)
    ]


def _child_env(repo: Path, **extra: str) -> dict[str, str]:
    """A child's environment: on the machine file, unstamped unless *extra* stamps it."""
    env = os.environ.copy()
    for name in (
        "OSPREY_CONFIG",
        "CONFIG_FILE",
        "OSPREY_EXECUTION_MODE",
        "OSPREY_EXECUTION_DEADLINE",
        "OSPREY_AGENT_DATA_ROOT",
        "OSPREY_TERMINAL_USER",
        ENV_CONTROL_TARGET,
        ENV_CONTROL_TARGET_GENERATION,
        ENV_APPROVED_JOURNAL_SHA256,
        ENV_APPROVED_TARGET,
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
            ExecutionWrapper(execution_mode=mode, tool="execute").create_wrapper(code, self.folder),
            encoding="utf-8",
        )

    def env(self, **extra: str) -> dict[str, str]:
        stamp = {
            "OSPREY_EXECUTION_MODE": self.mode,
            ENV_CONTROL_TARGET: "live",
            ENV_CONTROL_TARGET_GENERATION: str(GENERATION),
        }
        return _child_env(self.repo, **stamp, **extra)

    def run(self, **extra: str) -> dict[str, Any]:
        """Run the script to its end; return its recorded metadata, with any results."""
        subprocess.run(
            [sys.executable, str(self.script)],
            cwd=self.repo,
            env=self.env(**extra),
            capture_output=True,
            timeout=CHILD_TIMEOUT_S,
            check=False,
        )
        metadata = json.loads((self.folder / "execution_metadata.json").read_text(encoding="utf-8"))
        results = self.folder / "results.json"
        if results.exists():
            metadata["_results"] = json.loads(results.read_text(encoding="utf-8"))
        return metadata

    def hold(self, **extra: str) -> _Held:
        """Start the script, whose code announces a phase and stays there."""
        return _Held([sys.executable, str(self.script)], self.repo, self.env(**extra))


def test_another_containers_killed_run_is_busy_while_it_lives_and_restored_after(
    repo: Path,
    machine: _Machine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two processes with an agent-data root each share the target's lock and journal."""
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(repo / "agent_data_b"))
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "bob")
    other = _Held(
        [sys.executable, "-c", _DISPLACE_AND_WAIT],
        repo,
        _child_env(
            repo,
            OSPREY_AGENT_DATA_ROOT=str(repo / "agent_data_a"),
            OSPREY_TERMINAL_USER="alice",
            **{ENV_CONTROL_TARGET: "live", ENV_CONTROL_TARGET_GENERATION: str(GENERATION - 1)},
        ),
    )
    try:
        other.wait_ready()
        with pytest.raises(OspreyRunBusy) as busy, lock(None):
            pytest.fail("a run never starts while another container's run holds the target")
    finally:
        other.kill()

    assert busy.value.pid == other.proc.pid
    assert machine.values["Q"] == 9.0, "the killed run left its setpoint displaced"
    path = journal_path()
    assert path.parent == repo.resolve() / "var" / GUARDED_RUN_DIR / "live"
    pending = read_pending_journal(path)
    assert pending is not None
    assert (pending.identity, pending.pid) == ("alice", other.proc.pid)
    assert (pending.target, pending.generation) == ("live", GENERATION - 1)
    assert pending.values == {"Q": 3.0}

    approve(monkeypatch, digest(path), enter_lock=True)

    assert machine.writes == [("Q", 3.0)]
    assert machine.values == HOME
    assert path.read_bytes() == b""
    (line,) = _report_lines(capsys.readouterr().err)
    assert json.loads(line[len(RESTORE_REPORT_TAG) :])["restored"] == ["Q"]


def test_a_journal_and_every_stamp_rewritten_after_approval_refuses_the_run(
    repo: Path, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest rides in the approved call; nothing another process rewrites moves it."""
    path = plant({"Q": 0.0})
    stamps = [
        repo / root / "control_target" / identity / "control_context.json"
        for root, identity in (("agent_data_a", "alice"), ("agent_data_b", "bob"))
    ]
    for stamp in stamps:
        stamp.parent.mkdir(parents=True)
        stamp.write_text(
            json.dumps({"schema": 1, "target": "live", "generation": GENERATION}), encoding="utf-8"
        )
    approved = digest(path)

    rewrite = textwrap.dedent(
        """
        import json, sys

        journal, generation, stamps = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
        header = {"target": "live", "generation": generation, "identity": "mallory", "pid": 1}
        lines = [{"header": header}, {"address": "Q", "value": 7.0}, {"address": "S", "value": 1.0}]
        with open(journal, "w") as handle:
            handle.write("".join(json.dumps(line) + "\\n" for line in lines))
        for stamp in stamps:
            with open(stamp, "w") as handle:
                json.dump({"schema": 1, "target": "live", "generation": generation}, handle)
        """
    )
    subprocess.run(
        [sys.executable, "-c", rewrite, str(path), str(GENERATION), *map(str, stamps)],
        check=True,
        timeout=CHILD_TIMEOUT_S,
    )
    rewritten = path.read_bytes()
    assert digest(path) != approved

    with pytest.raises(OspreyJournalChanged):
        approve(monkeypatch, approved, enter_lock=True)

    assert machine.writes == []
    assert machine.values == HOME
    assert path.read_bytes() == rewritten


def test_writes_disabled_keep_the_displaced_entries_and_rearmed_writes_restore_them(
    machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = plant({"Q": 0.0, "S": 5.0, "T": 7.0})
    machine.writes_enabled = False

    with pytest.raises(OspreyRestoreIncomplete) as caught:
        approve(monkeypatch, digest(path), enter_lock=True)

    assert [(address, left) for address, _reason, left in caught.value.entries] == [
        ("Q", 3.0),
        ("T", 8.0),
    ]
    assert all("WRITES_DISABLED" in reason for _address, reason, _left in caught.value.entries)
    pending = read_pending_journal(path)
    assert pending is not None
    assert pending.values == {"Q": 0.0, "T": 7.0}, "the entry already back is gone, the rest stay"
    assert machine.values == HOME

    machine.writes_enabled = True
    approve(monkeypatch, digest(path), enter_lock=True)

    assert machine.writes == [("Q", 0.0), ("T", 7.0)]
    assert machine.values == {"Q": 0.0, "S": 5.0, "T": 7.0}
    assert path.read_bytes() == b""


def test_a_readonly_execute_replays_nothing_from_a_pending_journal(
    repo: Path, machine: _Machine
) -> None:
    path = plant({"Q": 0.0})
    before = path.read_bytes()

    metadata = _Execute(repo, "readonly", _TRY_THE_RUN).run(
        **{ENV_APPROVED_JOURNAL_SHA256: digest(path), ENV_APPROVED_TARGET: "live"}
    )

    assert metadata["success"] is True, metadata.get("traceback")
    assert metadata["_results"]["seen"] == HOME
    assert metadata["_results"]["refusal"].startswith("OspreyWriteRefused")
    assert "readonly execution mode" in metadata["_results"]["refusal"]
    assert machine.values == HOME
    assert path.read_bytes() == before
    assert _report_lines(metadata["stdout"], metadata["stderr"]) == []


def test_a_killed_readwrite_execute_leaves_no_journal(
    repo: Path, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execute`` holds the target's lock for its writes and journals none of them."""
    held = _Execute(repo, "readwrite", _WRITE_UNDER_THE_LOCK_AND_WAIT).hold(
        **{ENV_APPROVED_JOURNAL_SHA256: APPROVED_NO_JOURNAL, ENV_APPROVED_TARGET: "live"}
    )
    try:
        held.wait_ready()
        with pytest.raises(OspreyRunBusy), lock(None):
            pytest.fail("the execute holds the target's lock while it writes")
    finally:
        held.kill()

    path = journal_path()
    assert not path.exists() or path.read_bytes() == b""
    assert machine.values["Q"] == 9.0, "a write the killed code made stays as it was written"

    approve(monkeypatch, APPROVED_NO_JOURNAL, enter_lock=True)

    assert machine.writes == []
    assert machine.values["Q"] == 9.0


def test_a_held_live_run_does_not_block_a_run_on_va(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstamped run on a live baseline holds ``live``; ``va`` has a lock of its own."""
    monkeypatch.delenv(ENV_CONTROL_TARGET)
    monkeypatch.delenv(ENV_CONTROL_TARGET_GENERATION)

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


def test_an_execute_replay_prints_one_restore_line_filed_as_complete(
    repo: Path, machine: _Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay before user code reports once, however often the code takes the run."""
    machine.set("Q", 9.0)
    path = plant({"Q": 3.0})
    env = {ENV_APPROVED_JOURNAL_SHA256: digest(path), ENV_APPROVED_TARGET: "live"}
    execute = _Execute(repo, "readwrite", _TRY_THE_RUN)

    metadata = execute.run(**env)

    assert metadata["success"] is True, metadata.get("traceback")
    assert metadata["_results"]["seen"] == HOME
    assert metadata["_results"]["refusal"].startswith("OspreyRunBusy"), "the run holds the lock"
    assert path.read_bytes() == b""
    assert len(_report_lines(metadata["stdout"], metadata["stderr"])) == 1

    monkeypatch.setattr(executor, "_resolve_project_root", lambda: repo)
    ledger = tmp_path / "audit" / "executor.jsonl"
    filed = executor._file_run_record(
        execute.folder, "live", metadata, abnormal=False, ledger=ledger
    )
    (report,) = filed["sandbox"]
    assert report["restored"] == ["Q"] and not report["refused"] and not report["failed"]
    (record,) = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert record["reason"] == executor.REASON_RESTORE_COMPLETE
    assert record["detail"].startswith("source=sandbox ")
    assert "source=observed_on_disk" not in ledger.read_text(encoding="utf-8")
