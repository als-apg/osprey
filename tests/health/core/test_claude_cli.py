"""Tests for the core ``claude_cli`` and ``claude_cli_pinned`` categories.

Branch coverage patches the module's ``_run_version_command`` helper; a few
tests patch ``asyncio.create_subprocess_exec`` directly to exercise the real
subprocess helper (success, timeout-kill, missing executable).
"""

from __future__ import annotations

import asyncio

import pytest

from osprey.health.core import claude_cli as mod
from osprey.health.core.claude_cli import claude_cli, claude_cli_pinned
from osprey.health.models import Status


def _stub_run(monkeypatch, *, returncode=0, stdout="", stderr="", raises=None):
    """Patch ``_run_version_command`` to return canned output or raise."""

    async def _fake(_argv, _timeout_s):
        if raises is not None:
            raise raises
        return (returncode, stdout, stderr)

    monkeypatch.setattr(mod, "_run_version_command", _fake)


# --------------------------------------------------------------------------- #
# claude_cli (poll)
# --------------------------------------------------------------------------- #


class TestClaudeCliPoll:
    async def test_detected_version_is_ok(self, monkeypatch):
        _stub_run(monkeypatch, returncode=0, stdout="1.2.3 (Claude Code)")
        results = await claude_cli(None)()
        assert len(results) == 1
        row = results[0]
        assert row.name == "claude_cli_version"
        assert row.category == "claude_cli"
        assert row.status == Status.OK
        assert "1.2.3" in row.message

    async def test_missing_binary_is_warning(self, monkeypatch):
        _stub_run(monkeypatch, raises=FileNotFoundError())
        results = await claude_cli({})()
        assert results[0].status == Status.WARNING
        assert "not found in PATH" in results[0].message

    async def test_timeout_is_warning(self, monkeypatch):
        _stub_run(monkeypatch, raises=TimeoutError())
        results = await claude_cli({})()
        assert results[0].status == Status.WARNING
        assert "timed out" in results[0].message

    async def test_unparseable_output_is_warning(self, monkeypatch):
        _stub_run(monkeypatch, returncode=0, stdout="no version here", stderr="err")
        results = await claude_cli({})()
        assert results[0].status == Status.WARNING
        assert "Could not parse" in results[0].message

    async def test_nonzero_returncode_is_warning(self, monkeypatch):
        # rc != 0 => detected forced to None => unparseable warning branch.
        _stub_run(monkeypatch, returncode=1, stdout="1.2.3")
        results = await claude_cli({})()
        assert results[0].status == Status.WARNING


# --------------------------------------------------------------------------- #
# claude_cli_pinned (on_demand)
# --------------------------------------------------------------------------- #


class TestClaudeCliPinned:
    """The row proves the executable a headless run would be pointed at.

    The check asks the launcher for that executable instead of spelling a
    command of its own, so it cannot pass for a path no run takes. Nothing here
    runs a real CLI: the ``PATH`` lookup and the version probe are replaced.
    """

    CONFIG = {"claude_code": {"cli_version": "1.2.3"}}

    @pytest.fixture(autouse=True)
    def no_command_of_its_own(self, monkeypatch):
        _stub_run(monkeypatch, raises=AssertionError("the check spells no command of its own"))

    @staticmethod
    def _install(monkeypatch, version, path="/usr/local/bin/claude"):
        from osprey.agent_runner import launcher

        monkeypatch.setattr("shutil.which", lambda *_a, **_k: path)

        def probe(_exe):
            if isinstance(version, BaseException):
                raise version
            return version

        monkeypatch.setattr(launcher, "_probe_cli_version", probe)

    async def test_no_pin_is_single_skip_row(self):
        for config in (None, {}, {"claude_code": {}}, {"claude_code": None}):
            results = await claude_cli_pinned(config)()
            assert len(results) == 1
            row = results[0]
            assert row.name == "claude_cli_pinned"
            assert row.category == "claude_cli_pinned"
            assert row.status == Status.SKIP
            assert row.message == "no cli_version pin configured"

    async def test_an_installed_claude_that_reports_the_pin_is_ok(self, monkeypatch):
        self._install(monkeypatch, "1.2.3")
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert row.name == "claude_cli_pinned"
        assert row.status == Status.OK
        assert "1.2.3" in row.message
        assert "/usr/local/bin/claude" in row.message

    async def test_the_check_runs_the_executable_a_headless_run_is_pointed_at(self, monkeypatch):
        from osprey.agent_runner import launcher

        asked = []
        monkeypatch.setattr(
            launcher, "agent_cli_path", lambda cc: asked.append(dict(cc)) or "/opt/claude"
        )
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert asked == [self.CONFIG["claude_code"]]
        assert row.status == Status.OK
        assert "/opt/claude" in row.message

    async def test_another_installed_version_is_an_error_with_both_ways_out(self, monkeypatch):
        self._install(monkeypatch, "9.9.9")
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert row.status == Status.ERROR
        assert "1.2.3" in row.message
        assert "reports 9.9.9" in row.details
        assert "npm install -g @anthropic-ai/claude-code@1.2.3" in row.details
        assert "change claude_code.cli_version" in row.details

    async def test_no_installed_claude_is_an_error(self, monkeypatch):
        self._install(monkeypatch, None, path=None)
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert row.status == Status.ERROR
        assert "no `claude` was found on PATH" in row.details
        assert "npm install -g @anthropic-ai/claude-code@1.2.3" in row.details

    async def test_a_claude_that_reports_no_version_is_an_error(self, monkeypatch):
        self._install(monkeypatch, None)
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert row.status == Status.ERROR
        assert "did not report a version" in row.details

    async def test_a_version_probe_that_times_out_is_an_error(self, monkeypatch):
        import subprocess

        self._install(monkeypatch, subprocess.TimeoutExpired(["claude", "--version"], 15))
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert row.status == Status.ERROR
        assert "did not answer" in row.details

    async def test_the_check_never_downloads_anything(self, monkeypatch):
        from osprey.agent_runner import launcher

        self._install(monkeypatch, None, path=None)

        def refuse(argv, *_a, **_k):
            raise AssertionError(f"the check ran {argv}")

        monkeypatch.setattr(launcher.subprocess, "run", refuse)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
        (row,) = await claude_cli_pinned(self.CONFIG)()
        assert row.status == Status.ERROR

    async def test_the_answer_is_fresh_on_every_run_of_the_check(self, monkeypatch):
        self._install(monkeypatch, "9.9.9")
        (before,) = await claude_cli_pinned(self.CONFIG)()
        self._install(monkeypatch, "1.2.3")
        (after,) = await claude_cli_pinned(self.CONFIG)()
        assert (before.status, after.status) == (Status.ERROR, Status.OK)


# --------------------------------------------------------------------------- #
# _run_version_command against a fake asyncio subprocess
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, hang=False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(10)
        return (self._stdout, self._stderr)

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


class TestRunVersionCommand:
    async def test_success_decodes_output(self, monkeypatch):
        proc = _FakeProc(stdout=b"1.2.3\n", stderr=b"", returncode=0)

        async def _fake_exec(*argv, **kwargs):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        rc, out, err = await mod._run_version_command(["claude", "--version"], 5.0)
        assert rc == 0
        assert out == "1.2.3\n"
        assert err == ""

    async def test_timeout_kills_process(self, monkeypatch):
        proc = _FakeProc(hang=True)

        async def _fake_exec(*argv, **kwargs):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await mod._run_version_command(["claude", "--version"], 0.01)
        assert proc.killed is True

    async def test_missing_executable_propagates(self, monkeypatch):
        async def _fake_exec(*argv, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        with pytest.raises(FileNotFoundError):
            await mod._run_version_command(["nope"], 5.0)
