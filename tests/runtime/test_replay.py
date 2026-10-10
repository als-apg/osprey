"""A pending journal is replayed only as the approved call allows.

The approval prompt of a guarded tool lists the pending journal and hands the
call its sha256 and the target; the sandbox binds both before user code runs.
A journal is restored only when its bytes still hash to the approved digest,
and a run refuses before any write when it is on another target than the one
approved. A call that carries no digest refuses when its tool's calls are put
to a human, and otherwise restores a journal of its own target and generation.
After a restore the journal holds exactly what stayed displaced. Channels are a
dict checked by a real ``LimitsValidator``; no control system is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime
import osprey.runtime.guarded_run as guarded_run
from osprey.errors import ChannelLimitsViolationError
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import (
    APPROVED_NO_JOURNAL,
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
    JOURNAL_FILE_NAME,
    RESTORE_REPORT_TAG,
    OspreyJournalChanged,
    approval_asks,
    guarded_run_dir,
    journaled_run,
    lock,
)
from osprey.runtime.journal import (
    OspreyRestoreIncomplete,
    OspreyStaleJournal,
    OspreyWriteRefused,
    guarded_write,
    read_pending_journal,
)
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.types import LIMITS_MODE_OPTIONAL
from osprey_connectors.workspace import reset_config_cache

#: The generation this process is stamped with.
GENERATION = 4

#: An ``approval`` block whose hook is wired and puts every tool's call to a human.
_ASKING = {"enabled": True, "default_policy": "always", "hook_wired": True}


class _Channels:
    """Channels whose every write is first checked by a real ``LimitsValidator``."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, Any]] = []
        self.validator = LimitsValidator(
            {"Q": ChannelLimitsConfig("Q", max_step=10.0, writable=True)},
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


def _write_config(root: Path, approval: dict[str, Any] | None = None) -> None:
    config: dict[str, Any] = {"control_system": {"type": "epics", "connector": {"epics": {}}}}
    if approval is not None:
        config["approval"] = approval
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline, entered and stamped ``live``.

    The approval hook is wired and every tool's call is put to a human; a test
    that wants otherwise rewrites the config.
    """
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    _write_config(root, _ASKING)
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, str(GENERATION))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(ENV_APPROVED_JOURNAL_SHA256, raising=False)
    monkeypatch.delenv(ENV_APPROVED_TARGET, raising=False)
    monkeypatch.setattr(guarded_run, "_APPROVED", None)
    return root


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> _Channels:
    """``Q`` with a ``max_step`` of 10.0 and ``S`` without a limits record."""
    fake = _Channels({"Q": 3.0, "S": 5.0})
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


def _journal_path() -> Path:
    return guarded_run_dir("live") / JOURNAL_FILE_NAME


def _plant(
    values: dict[str, Any],
    *,
    generation: int = GENERATION,
    identity: str = "alice",
) -> Path:
    """A journal a killed run on ``live`` left, holding ``values``."""
    header = {
        "target": "live",
        "generation": generation,
        "identity": identity,
        "pid": 4242,
        "started": "2026-01-01T00:00:00Z",
    }
    lines = [{"header": header}] + [{"address": a, "value": v} for a, v in values.items()]
    path = _journal_path()
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approve(
    monkeypatch: pytest.MonkeyPatch,
    digest: str | None,
    target: str | None,
    tool: str = "execute",
) -> None:
    """Hand the approved fields to the sandbox and bind them, as the wrapper does."""
    if digest is not None:
        monkeypatch.setenv(ENV_APPROVED_JOURNAL_SHA256, digest)
    if target is not None:
        monkeypatch.setenv(ENV_APPROVED_TARGET, target)
    guarded_run._open_approved_call(tool)


def _reports(text: str) -> list[dict[str, Any]]:
    prefix = RESTORE_REPORT_TAG + " "
    return [
        json.loads(line[len(prefix) :]) for line in text.splitlines() if line.startswith(prefix)
    ]


def test_the_restore_report_tag_is_the_guarded_runs() -> None:
    assert RESTORE_REPORT_TAG == "OSPREY_GUARDED_RUN_RESTORE"


def test_an_approved_digest_restores_another_identitys_run_before_user_code(
    channels: _Channels,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _plant({"Q": 0.0, "S": 5.0}, generation=GENERATION - 1, identity="bob")

    _approve(monkeypatch, _digest(path), "live")

    assert channels.writes == [("Q", 0.0)], "restored at bind time, before any user code"
    assert path.read_bytes() == b""
    (report,) = _reports(capsys.readouterr().err)
    assert report["restored"] == ["Q"] and report["unchanged"] == ["S"]
    assert report["aborted"] is True


def test_the_approved_fields_leave_the_environment_once_bound(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approve(monkeypatch, APPROVED_NO_JOURNAL, "live")

    assert ENV_APPROVED_JOURNAL_SHA256 not in os.environ
    assert ENV_APPROVED_TARGET not in os.environ
    with pytest.raises(RuntimeError, match="already bound"):
        guarded_run._open_approved_call("execute")
    assert channels.writes == []


def test_a_digest_mismatch_writes_nothing_and_leaves_the_journal(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0})
    approved = _digest(path)
    path.write_text(path.read_text(encoding="utf-8") + '{"address": "S", "value": 1.0}\n')
    before = path.read_bytes()

    with pytest.raises(OspreyJournalChanged):
        _approve(monkeypatch, approved, "live")

    assert channels.writes == []
    assert path.read_bytes() == before


def test_a_pending_journal_under_digest_none_writes_nothing(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approve(monkeypatch, APPROVED_NO_JOURNAL, "live")
    path = _plant({"Q": 0.0})
    before = path.read_bytes()

    with pytest.raises(OspreyJournalChanged), lock(None):
        pytest.fail("the run does not start on a journal its prompt did not list")

    assert channels.writes == []
    assert path.read_bytes() == before


def test_another_target_than_the_approved_one_refuses_before_any_write(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0})
    before = path.read_bytes()

    with pytest.raises(OspreyWriteRefused, match="approved for target 'va'"):
        _approve(monkeypatch, _digest(path), "va")

    assert channels.writes == []
    assert path.read_bytes() == before


def test_a_call_without_the_approved_fields_refuses_when_its_tool_asks(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0})
    before = path.read_bytes()
    _approve(monkeypatch, None, None)

    with pytest.raises(OspreyWriteRefused, match="carries none"), lock(None):
        pytest.fail("a missing field is a refusal, never a skipped comparison")

    assert channels.writes == []
    assert path.read_bytes() == before


def test_a_call_carrying_only_the_digest_refuses_when_its_tool_asks(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0})

    with pytest.raises(OspreyWriteRefused, match="carries none"):
        _approve(monkeypatch, _digest(path), None)

    assert channels.writes == []


@pytest.mark.parametrize(
    "approval",
    [
        pytest.param({**_ASKING, "tools": {"execute": "skip"}}, id="skip"),
        pytest.param({**_ASKING, "enabled": False}, id="approval-disabled"),
        pytest.param({**_ASKING, "hook_wired": False}, id="no-hook"),
        pytest.param({"enabled": True, "default_policy": "always"}, id="hook-unstated"),
    ],
)
def test_a_call_whose_tool_never_asks_restores_its_own_generation(
    repo: Path,
    channels: _Channels,
    monkeypatch: pytest.MonkeyPatch,
    approval: dict[str, Any],
) -> None:
    _write_config(repo, approval)
    path = _plant({"Q": 0.0})
    _approve(monkeypatch, None, None)

    with lock(None):
        seen = list(channels.writes)

    assert seen == [("Q", 0.0)]
    assert path.read_bytes() == b""


def test_a_stale_journal_names_the_approval_remedy_when_the_tool_asks(
    channels: _Channels,
) -> None:
    path = _plant({"Q": 0.0}, generation=GENERATION - 1)
    before = path.read_bytes()

    with pytest.raises(OspreyStaleJournal) as caught, lock(None):
        pytest.fail("a stale journal stops the run")

    assert str(caught.value).endswith(
        "call any guarded tool under approval: its prompt lists and restores these setpoints."
    )
    assert channels.writes == []
    assert path.read_bytes() == before


def test_a_stale_journal_names_the_remove_remedy_with_approval_disabled(
    repo: Path, channels: _Channels
) -> None:
    _write_config(repo, {**_ASKING, "enabled": False})
    path = _plant({"Q": 0.0}, generation=GENERATION - 1)

    with pytest.raises(OspreyStaleJournal) as caught, lock(None):
        pytest.fail("a stale journal stops the run")

    assert str(caught.value).endswith(f"remove {path} after checking the listed setpoints.")
    assert channels.writes == []


def test_a_stale_journal_names_the_remove_remedy_without_the_approval_hook(
    repo: Path, channels: _Channels
) -> None:
    _write_config(repo, {**_ASKING, "hook_wired": False})
    path = _plant({"Q": 0.0}, generation=GENERATION - 1)

    with pytest.raises(OspreyStaleJournal) as caught, lock(None):
        pytest.fail("a stale journal stops the run")

    assert str(caught.value).endswith(f"remove {path} after checking the listed setpoints.")
    assert channels.writes == []


def test_approval_asks_reads_the_policy_and_the_wired_hook(repo: Path) -> None:
    assert approval_asks("execute") is True
    _write_config(repo, {**_ASKING, "tools": {"execute": "skip"}})
    assert approval_asks("execute") is False
    assert approval_asks("execute_file") is True
    assert approval_asks(None) is True, "any guarded tool that asks"
    _write_config(repo, {**_ASKING, "hook_wired": False})
    assert approval_asks("execute_file") is False
    assert approval_asks(None) is False


def test_approval_asks_never_reads_the_rendered_settings(repo: Path) -> None:
    """A hook rule in ``.claude/settings.json`` says nothing; the config does."""
    claude = repo / "build" / ".claude"
    claude.mkdir()
    rule = {"hooks": [{"type": "command", "command": "python3 .claude/hooks/osprey_approval.py"}]}
    (claude / "settings.json").write_text(
        json.dumps({"hooks": {"PreToolUse": [rule]}}), encoding="utf-8"
    )
    _write_config(repo, {**_ASKING, "hook_wired": False})
    assert approval_asks("execute") is False


def test_an_incomplete_replay_keeps_exactly_the_entries_left_displaced(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0, "S": 4.0, "T": 7.0})
    header = path.read_bytes().split(b"\n", 1)[0]
    approved = _digest(path)
    channels.values["T"] = 8.0

    def refuse_q(address: str, value: Any, **kwargs: Any) -> None:
        if address == "Q":
            raise ChannelLimitsViolationError(address, value, "MAX_VALUE", "outside the band")
        if address == "T":
            raise RuntimeError("no confirmation")
        channels.write_channel(address, value, **kwargs)

    monkeypatch.setattr(osprey.runtime, "write_channel", refuse_q)

    with pytest.raises(OspreyRestoreIncomplete) as caught:
        _approve(monkeypatch, approved, "live")

    assert caught.value.entries == (
        ("Q", "outside the band", 3.0),
        ("T", "no confirmation", None),
    )
    assert path.read_bytes().split(b"\n", 1)[0] == header, "the header is kept byte for byte"
    pending = read_pending_journal(path)
    assert pending is not None
    assert pending.values == {"Q": 0.0, "T": 7.0}
    assert channels.writes == [("S", 4.0)]


def test_a_rewrite_never_follows_a_link_planted_in_the_journal_directory(
    tmp_path: Path, channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewrite goes through a file it created itself, never one that was waiting."""
    path = _plant({"Q": 0.0, "S": 4.0})
    victim = tmp_path / "victim"
    victim.write_text("untouched", encoding="utf-8")
    planted = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    planted.symlink_to(victim)

    def refuse_q(address: str, value: Any, **kwargs: Any) -> None:
        if address == "Q":
            raise ChannelLimitsViolationError(address, value, "MAX_VALUE", "outside the band")
        channels.write_channel(address, value, **kwargs)

    monkeypatch.setattr(osprey.runtime, "write_channel", refuse_q)
    with pytest.raises(OspreyRestoreIncomplete):
        _approve(monkeypatch, _digest(path), "live")

    assert victim.read_text(encoding="utf-8") == "untouched"
    assert planted.is_symlink()
    assert not path.is_symlink()
    pending = read_pending_journal(path)
    assert pending is not None and pending.values == {"Q": 0.0}
    leftovers = sorted(p.name for p in path.parent.iterdir() if p.name.endswith(".tmp"))
    assert leftovers == [planted.name]


def test_a_rewritten_journal_restores_the_rest_on_the_next_approved_run(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _plant({"Q": 0.0, "S": 4.0})
    original = channels.write_channel

    def refuse_q(address: str, value: Any, **kwargs: Any) -> None:
        if address == "Q":
            raise ChannelLimitsViolationError(address, value, "MAX_VALUE", "outside the band")
        original(address, value, **kwargs)

    monkeypatch.setattr(osprey.runtime, "write_channel", refuse_q)
    with pytest.raises(OspreyRestoreIncomplete):
        _approve(monkeypatch, _digest(path), "live")

    monkeypatch.setattr(osprey.runtime, "write_channel", original)
    monkeypatch.setattr(guarded_run, "_APPROVED", None)
    _approve(monkeypatch, _digest(path), "live")

    assert channels.values == {"Q": 0.0, "S": 4.0}
    assert path.read_bytes() == b""


def test_an_exception_escaping_a_journaled_run_restores_and_clears(
    channels: _Channels, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _journal_path()

    with pytest.raises(ValueError, match="mid-run") as caught, journaled_run("live"):
        guarded_write(["Q"], lambda: osprey.runtime.write_channel("Q", 9.0), lambda e: e)
        raise ValueError("mid-run")

    assert channels.values["Q"] == 3.0
    assert path.read_bytes() == b""
    (report,) = _reports(capsys.readouterr().err)
    assert report["restored"] == ["Q"] and report["aborted"] is True
    assert json.loads(caught.value.restore_report.to_json()) == report


def test_an_exception_escaping_a_journaled_run_keeps_what_stayed_displaced(
    channels: _Channels, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _journal_path()
    original = channels.write_channel

    with pytest.raises(KeyboardInterrupt), journaled_run("live"):
        guarded_write(["Q"], lambda: osprey.runtime.write_channel("Q", 9.0), lambda e: e)
        guarded_write(["S"], lambda: osprey.runtime.write_channel("S", 6.0), lambda e: e)

        def refuse_q(address: str, value: Any, **kwargs: Any) -> None:
            if address == "Q":
                raise ChannelLimitsViolationError(address, value, "MAX_VALUE", "outside the band")
            original(address, value, **kwargs)

        monkeypatch.setattr(osprey.runtime, "write_channel", refuse_q)
        raise KeyboardInterrupt

    assert channels.values == {"Q": 9.0, "S": 5.0}
    pending = read_pending_journal(path)
    assert pending is not None
    assert pending.values == {"Q": 3.0}


def test_a_clean_journaled_run_clears_and_prints_no_report(
    channels: _Channels, capsys: pytest.CaptureFixture[str]
) -> None:
    with journaled_run("live"):
        guarded_write(["Q"], lambda: osprey.runtime.write_channel("Q", 9.0), lambda e: e)

    assert channels.values["Q"] == 9.0
    assert _journal_path().read_bytes() == b""
    assert _reports(capsys.readouterr().err) == []
