"""The guarded run's lock: one run per control target, and a dead run's journal restored.

``lock(target)`` takes the target's lock without blocking, records its holder
in the lock file and restores what a killed run's journal left before the run
starts. ``journaled_run(target)`` adds the durable journal, and is the only
context where ``guarded_write`` writes. Channels are a dict checked by a real
``LimitsValidator``; no control system is involved.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime
import osprey.runtime.guarded_run as guarded_run
from osprey.errors import ChannelLimitsViolationError
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import (
    JOURNAL_FILE_NAME,
    LOCK_FILE_NAME,
    GuardedRunDirError,
    OspreyRunBusy,
    guarded_run_dir,
    journaled_run,
    lock,
)
from osprey.runtime.journal import (
    OspreyRestoreIncomplete,
    OspreyStaleJournal,
    OspreyWriteRefused,
    guarded_write,
    pop_journal,
    push_journal,
    read_pending_journal,
)
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.types import LIMITS_MODE_OPTIONAL

#: The generation this process is stamped with.
GENERATION = 4

#: Seconds a child process may take to import the runtime and try the lock.
CHILD_TIMEOUT_S = 120.0


class _Channels:
    """Channels whose every write is first checked by a real ``LimitsValidator``."""

    def __init__(self, values: dict[str, Any], max_step: dict[str, float]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, Any]] = []
        self.validator = LimitsValidator(
            {a: ChannelLimitsConfig(a, max_step=s, writable=True) for a, s in max_step.items()},
            {"mode": LIMITS_MODE_OPTIONAL},
        )

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        return [self.values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.validator.validate(address, value, read_current=self.values.__getitem__)
        self.writes.append((address, value))
        self.values[address] = value

    def channel_limits(self, address: str) -> ChannelLimitsConfig | None:
        return self.validator.limits.get(address)


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    (root / "build" / "config.yml").write_text(
        yaml.safe_dump({"control_system": {"type": "epics", "connector": {"epics": {}}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, str(GENERATION))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    return root


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> _Channels:
    """``Q`` with a ``max_step`` of 1.0 and ``S`` without a limits record."""
    fake = _Channels({"Q": 3.0, "S": 5.0}, {"Q": 1.0})
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


def _journal_path() -> Path:
    return guarded_run_dir("live") / JOURNAL_FILE_NAME


def _plant(values: dict[str, Any], *, target: str = "live", generation: int = GENERATION) -> Path:
    """A journal a killed run on ``target`` left, holding ``values``."""
    header = {
        "target": target,
        "generation": generation,
        "identity": "alice",
        "pid": 4242,
        "started": "2026-01-01T00:00:00Z",
    }
    lines = [{"header": header}] + [{"address": a, "value": v} for a, v in values.items()]
    path = _journal_path()
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _holder(directory: Path) -> dict[str, Any] | None:
    text = (directory / LOCK_FILE_NAME).read_text(encoding="utf-8")
    return json.loads(text) if text else None


def test_osprey_runtime_reexports_the_directory_and_its_names() -> None:
    assert osprey.runtime.guarded_run_dir is guarded_run.guarded_run_dir
    assert osprey.runtime.GUARDED_RUN_DIR is guarded_run.GUARDED_RUN_DIR
    assert osprey.runtime.LOCK_FILE_NAME is guarded_run.LOCK_FILE_NAME
    assert osprey.runtime.JOURNAL_FILE_NAME is guarded_run.JOURNAL_FILE_NAME
    assert {"guarded_run_dir", "GUARDED_RUN_DIR", "LOCK_FILE_NAME", "JOURNAL_FILE_NAME"} <= set(
        osprey.runtime.__all__
    )


@pytest.mark.usefixtures("channels")
def test_the_lock_records_its_holder_and_releases() -> None:
    with lock("live") as directory:
        assert directory == guarded_run_dir("live")
        holder = _holder(directory)
        assert holder is not None and holder["pid"] == os.getpid()
        assert isinstance(holder["started"], str)

    assert _holder(directory) is None, "a released lock records no holder"
    with lock("live"):
        pass


@pytest.mark.usefixtures("channels")
def test_a_created_lock_is_group_writable_under_umask_022() -> None:
    previous = os.umask(0o022)
    try:
        with lock("live") as directory:
            pass
    finally:
        os.umask(previous)

    assert (directory / LOCK_FILE_NAME).stat().st_mode & 0o060 == 0o060


@pytest.mark.usefixtures("channels")
def test_a_second_lock_is_busy() -> None:
    with lock("live") as directory:
        with pytest.raises(OspreyRunBusy) as caught:
            with lock(None):
                pytest.fail("a busy run never starts")

    assert caught.value.root == str(directory)
    assert caught.value.pid == os.getpid()
    assert f"pid {os.getpid()}" in str(caught.value)


def test_another_process_is_busy_while_the_lock_is_held(repo: Path) -> None:
    child = textwrap.dedent(
        """
        import sys
        from osprey.runtime.guarded_run import OspreyRunBusy, lock

        try:
            with lock("live"):
                pass
        except OspreyRunBusy as exc:
            print(exc.pid, flush=True)
            sys.exit(3)
        """
    )
    env = {**os.environ}

    with lock("live"):
        held = subprocess.run(
            [sys.executable, "-c", child],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    free = subprocess.run(
        [sys.executable, "-c", child],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )

    assert held.returncode == 3, held.stderr
    assert held.stdout.strip() == str(os.getpid())
    assert free.returncode == 0, free.stderr


def test_a_readonly_run_raises_before_the_lock(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with pytest.raises(OspreyWriteRefused, match="readonly execution mode"):
        with lock("live"):
            pytest.fail("a readonly run never takes the lock")

    assert not (repo / "var").exists()


def test_no_guarded_run_directory_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(bare)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    with pytest.raises(GuardedRunDirError, match="guarded runs need var/guarded_run"):
        with lock("live"):
            pytest.fail("a run with no directory never starts")


def test_guarded_write_under_the_lock_alone_is_refused(channels: _Channels) -> None:
    with lock("live"):
        with pytest.raises(OspreyWriteRefused):
            guarded_write(["Q"], lambda: channels.write_channel("Q", 2.5), lambda exc: exc)
        level = push_journal()
        try:
            with pytest.raises(OspreyWriteRefused):
                guarded_write(["Q"], lambda: channels.write_channel("Q", 2.5), lambda exc: exc)
        finally:
            pop_journal(level)

    assert channels.writes == []
    assert channels.values["Q"] == 3.0


def test_guarded_write_under_a_journaled_run_journals_then_writes(
    channels: _Channels,
) -> None:
    on_disk_before_write: list[dict[str, Any]] = []

    def write() -> None:
        pending = read_pending_journal(_journal_path())
        assert pending is not None
        on_disk_before_write.append(pending.values)
        channels.write_channel("Q", 2.5)

    with journaled_run("live") as journal:
        guarded_write(["Q"], write, lambda exc: exc)
        pending = read_pending_journal(_journal_path())

    assert on_disk_before_write == [{"Q": 3.0}], "the record is on disk before the write"
    assert journal.values == {"Q": 3.0}
    assert channels.writes == [("Q", 2.5)]
    assert pending is not None
    assert (pending.target, pending.generation, pending.pid) == ("live", GENERATION, os.getpid())
    assert _journal_path().read_bytes() == b"", "a run that ends clears its journal"


def test_a_journaled_run_inside_the_lock_runs_under_it(channels: _Channels) -> None:
    with lock("live"):
        with journaled_run(None) as journal:
            guarded_write(["Q"], lambda: channels.write_channel("Q", 2.5), lambda exc: exc)
        with pytest.raises(OspreyWriteRefused):
            guarded_write(["Q"], lambda: channels.write_channel("Q", 2.0), lambda exc: exc)

    assert journal.values == {"Q": 3.0}
    assert channels.writes == [("Q", 2.5)]


def test_a_dead_run_restores_in_max_step_writes_before_the_run(
    channels: _Channels, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _plant({"Q": 0.0, "S": 5.0})
    seen: list[list[tuple[str, Any]]] = []

    with lock("live"):
        seen.append(list(channels.writes))

    assert seen == [[("Q", 2.0), ("Q", 1.0), ("Q", 0.0)]], (
        "three limits-accepted writes for an entry 3 x max_step away, none for one already back"
    )
    assert path.read_bytes() == b""
    assert "restored 1 addresses from a dead run (pid 4242)" in capsys.readouterr().out


def test_an_incomplete_restore_keeps_what_stayed_displaced_and_does_not_start(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0, "S": 4.0})

    def refuse_q(address: str, value: Any, **kwargs: Any) -> None:
        if address == "Q":
            raise ChannelLimitsViolationError(address, value, "MAX_VALUE", "outside the band")
        channels.write_channel(address, value, **kwargs)

    monkeypatch.setattr(osprey.runtime, "write_channel", refuse_q)

    with pytest.raises(OspreyRestoreIncomplete) as caught:
        with lock("live"):
            pytest.fail("the run does not start on an incomplete restore")

    assert caught.value.entries == (("Q", "outside the band", 3.0),)
    assert "Q: outside the band (left at 3.0)" in str(caught.value)
    pending = read_pending_journal(path)
    assert pending is not None
    assert pending.values == {"Q": 0.0}, "the journal holds exactly the refused entry"
    assert channels.writes == [("S", 4.0)]


def test_a_journal_for_another_generation_is_stale(channels: _Channels) -> None:
    path = _plant({"Q": 0.0}, generation=GENERATION - 1)
    before = path.read_bytes()

    with pytest.raises(OspreyStaleJournal, match="after checking the listed setpoints"):
        with lock("live"):
            pytest.fail("a stale journal stops the run")

    assert channels.writes == []
    assert path.read_bytes() == before


def test_a_header_only_journal_is_cleared(channels: _Channels) -> None:
    path = _journal_path()
    path.write_text(json.dumps({"header": {"target": "live"}}) + "\n", encoding="utf-8")

    with journaled_run("live"):
        lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1, "one header, never a second one behind a dead run's"
    assert json.loads(lines[0])["header"]["pid"] == os.getpid()
    assert channels.writes == []
