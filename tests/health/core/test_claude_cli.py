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

    async def _fake(argv, timeout_s):
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
    async def test_no_pin_is_single_skip_row(self, monkeypatch):
        # Should never invoke the subprocess when unpinned.
        _stub_run(monkeypatch, raises=AssertionError("must not run npx"))
        for config in (None, {}, {"claude_code": {}}, {"claude_code": None}):
            results = await claude_cli_pinned(config)()
            assert len(results) == 1
            row = results[0]
            assert row.name == "claude_cli_pinned"
            assert row.category == "claude_cli_pinned"
            assert row.status == Status.SKIP
            assert row.message == "no cli_version pin configured"

    async def test_pin_match_is_ok(self, monkeypatch):
        _stub_run(monkeypatch, returncode=0, stdout="1.2.3 (Claude Code)")
        config = {"claude_code": {"cli_version": "1.2.3"}}
        results = await claude_cli_pinned(config)()
        assert results[0].name == "claude_cli_pinned"
        assert results[0].status == Status.OK
        assert "1.2.3" in results[0].message

    async def test_pin_mismatch_is_warning(self, monkeypatch):
        _stub_run(monkeypatch, returncode=0, stdout="9.9.9")
        config = {"claude_code": {"cli_version": "1.2.3"}}
        results = await claude_cli_pinned(config)()
        assert results[0].status == Status.WARNING
        assert "1.2.3" in results[0].message
        assert "9.9.9" in results[0].message

    async def test_pin_unparseable_reports_unknown_warning(self, monkeypatch):
        _stub_run(monkeypatch, returncode=0, stdout="garbage")
        config = {"claude_code": {"cli_version": "1.2.3"}}
        results = await claude_cli_pinned(config)()
        assert results[0].status == Status.WARNING
        assert "unknown" in results[0].message

    async def test_missing_npx_is_error(self, monkeypatch):
        _stub_run(monkeypatch, raises=FileNotFoundError())
        config = {"claude_code": {"cli_version": "1.2.3"}}
        results = await claude_cli_pinned(config)()
        assert results[0].status == Status.ERROR
        assert "npx not found" in results[0].message

    async def test_npx_timeout_is_error(self, monkeypatch):
        _stub_run(monkeypatch, raises=TimeoutError())
        config = {"claude_code": {"cli_version": "1.2.3"}}
        results = await claude_cli_pinned(config)()
        assert results[0].status == Status.ERROR
        assert "timed out" in results[0].message

    async def test_npx_nonzero_returncode_is_error(self, monkeypatch):
        _stub_run(monkeypatch, returncode=1, stdout="", stderr="boom")
        config = {"claude_code": {"cli_version": "1.2.3"}}
        results = await claude_cli_pinned(config)()
        assert results[0].status == Status.ERROR
        assert "npx failed" in results[0].message
        assert "boom" in results[0].details


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
