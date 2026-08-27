"""Tests for the core ``systemd_unit`` category.

Branch coverage patches the module's ``_run_systemctl`` helper and its
``shutil.which``/``Path.home`` seams; a few tests patch
``asyncio.create_subprocess_exec`` directly to exercise the real subprocess
helper (success, timeout-kill, missing executable).

Every seam is patched, so the suite passes on hosts with no systemd at all
(macOS, containers) as well as on Linux.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from osprey.health.core import systemd_unit as mod
from osprey.health.core.systemd_unit import systemd_unit
from osprey.health.models import Status

_UNIT = "osprey.service"


def _stub_run(monkeypatch, *, returncode=0, stdout="", stderr="", raises=None):
    """Patch ``_run_systemctl`` to return canned output or raise."""

    async def _fake(argv, timeout_s):
        if raises is not None:
            raise raises
        return (returncode, stdout, stderr)

    monkeypatch.setattr(mod, "_run_systemctl", _fake)


def _have_systemctl(monkeypatch, present=True):
    """Patch ``shutil.which`` as seen by the module."""
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/systemctl" if present else None)


def _repo_with_unit(tmp_path: Path) -> Path:
    """Create a deployment repo root holding a scaffolded unit file."""
    (tmp_path / _UNIT).write_text("[Unit]\n")
    return tmp_path


def _install_dir(monkeypatch, tmp_path: Path, *, installed: bool) -> Path:
    """Point ``XDG_CONFIG_HOME`` at ``tmp_path`` and optionally install the unit."""
    xdg = tmp_path / "xdg"
    unit_dir = xdg / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    if installed:
        (unit_dir / _UNIT).write_text("[Unit]\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    return unit_dir


# --------------------------------------------------------------------------- #
# state table
# --------------------------------------------------------------------------- #


class TestStateTable:
    async def test_no_repo_unit_is_skip(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=AssertionError("must not run systemctl"))
        _have_systemctl(monkeypatch)
        results = await systemd_unit(None, cwd=tmp_path)()
        assert len(results) == 1
        row = results[0]
        assert row.name == "systemd_unit"
        assert row.category == "systemd_unit"
        assert row.status == Status.SKIP
        assert "no scaffolded" in row.message

    async def test_no_systemctl_is_skip(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=AssertionError("must not run systemctl"))
        _have_systemctl(monkeypatch, present=False)
        results = await systemd_unit({}, cwd=_repo_with_unit(tmp_path))()
        assert len(results) == 1
        assert results[0].status == Status.SKIP
        assert "systemctl not found" in results[0].message

    async def test_scaffolded_but_not_installed_is_warning(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=AssertionError("must not run systemctl"))
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=False)
        results = await systemd_unit({}, cwd=repo)()
        assert len(results) == 1
        row = results[0]
        assert row.status == Status.WARNING
        assert "not installed" in row.message
        assert "daemon-reload" in row.details

    async def test_installed_but_not_found_is_error(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, returncode=0, stdout="not-found\n")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert len(results) == 1
        row = results[0]
        assert row.status == Status.ERROR
        assert "not-found" in row.message
        assert "systemctl --user daemon-reload" in row.details
        assert "NFS/autofs" in row.details

    @pytest.mark.parametrize("load_state", ["loaded", "masked", "error", "bad-setting"])
    async def test_installed_and_visible_is_ok(self, monkeypatch, tmp_path, load_state):
        _stub_run(monkeypatch, returncode=0, stdout=f"{load_state}\n")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert len(results) == 1
        assert results[0].status == Status.OK
        assert results[0].value == load_state

    async def test_bus_unreachable_is_skip(self, monkeypatch, tmp_path):
        _stub_run(
            monkeypatch,
            returncode=1,
            stderr="Failed to connect to bus: No such file or directory",
        )
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.SKIP
        assert "no reachable systemd" in results[0].message
        assert "Failed to connect to bus" in results[0].details

    async def test_bus_unreachable_without_stderr_still_skips(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, returncode=1, stderr="")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.SKIP
        assert results[0].details

    async def test_query_timeout_is_skip(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=TimeoutError())
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.SKIP
        assert "timed out" in results[0].message

    async def test_systemctl_vanishing_between_which_and_exec_is_skip(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=FileNotFoundError())
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.SKIP
        assert "systemctl not found" in results[0].message

    async def test_empty_load_state_is_ok_unknown(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, returncode=0, stdout="   \n")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.OK
        assert results[0].value == "unknown"


# --------------------------------------------------------------------------- #
# install-directory resolution
# --------------------------------------------------------------------------- #


class TestInstallDirResolution:
    async def test_xdg_config_home_directory_reports_ok(self, monkeypatch, tmp_path):
        """The regression a hardcoded ``~/.config`` would cause.

        With ``XDG_CONFIG_HOME`` set and the unit installed underneath it, the
        row must be OK — not the WARNING a ``~/.config``-only lookup would give.
        """
        _stub_run(monkeypatch, returncode=0, stdout="loaded\n")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        # A home that holds nothing, to prove the XDG path is what was read.
        empty_home = tmp_path / "home"
        (empty_home / ".config" / "systemd" / "user").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.OK

    async def test_blank_xdg_config_home_falls_back_to_home(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, returncode=0, stdout="loaded\n")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", "   ")
        home = tmp_path / "home"
        unit_dir = home / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True)
        (unit_dir / _UNIT).write_text("[Unit]\n")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.OK

    async def test_unset_xdg_config_home_uses_home(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=AssertionError("must not run systemctl"))
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        home = tmp_path / "home"
        (home / ".config" / "systemd" / "user").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.WARNING

    async def test_unresolvable_home_is_skip(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=AssertionError("must not run systemctl"))
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        def _boom(cls):
            raise RuntimeError("Could not determine home directory")

        monkeypatch.setattr(Path, "home", classmethod(_boom))
        results = await systemd_unit({}, cwd=repo)()
        assert len(results) == 1
        assert results[0].status == Status.SKIP
        assert "cannot resolve" in results[0].message
        assert "RuntimeError" in results[0].details

    async def test_oserror_resolving_install_dir_is_skip(self, monkeypatch, tmp_path):
        _stub_run(monkeypatch, raises=AssertionError("must not run systemctl"))
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)

        def _boom() -> Path:
            raise OSError("ENAMETOOLONG")

        monkeypatch.setattr(mod, "_user_unit_dir", _boom)
        results = await systemd_unit({}, cwd=repo)()
        assert results[0].status == Status.SKIP
        assert "OSError" in results[0].details


# --------------------------------------------------------------------------- #
# cwd defaulting
# --------------------------------------------------------------------------- #


class TestCwdDefault:
    async def test_cwd_defaults_at_call_time(self, monkeypatch, tmp_path):
        """No ``cwd=`` => ``Path.cwd()`` resolved when the callable runs."""
        _stub_run(monkeypatch, returncode=0, stdout="loaded\n")
        _have_systemctl(monkeypatch)
        repo = _repo_with_unit(tmp_path)
        _install_dir(monkeypatch, tmp_path, installed=True)
        callable_ = systemd_unit({})
        monkeypatch.chdir(repo)
        results = await callable_()
        assert results[0].status == Status.OK


# --------------------------------------------------------------------------- #
# _run_systemctl against a fake asyncio subprocess
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, hang=False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.reaped = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(10)
        return (self._stdout, self._stderr)

    def kill(self):
        self.killed = True

    async def wait(self):
        self.reaped = True
        return self.returncode


class TestRunSystemctl:
    async def test_success_decodes_output(self, monkeypatch):
        proc = _FakeProc(stdout=b"loaded\n", stderr=b"", returncode=0)

        async def _fake_exec(*argv, **kwargs):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        rc, out, err = await mod._run_systemctl(["systemctl", "--user", "show"], 5.0)
        assert rc == 0
        assert out == "loaded\n"
        assert err == ""

    async def test_timeout_kills_and_reaps_process(self, monkeypatch):
        proc = _FakeProc(hang=True)

        async def _fake_exec(*argv, **kwargs):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await mod._run_systemctl(["systemctl", "--user", "show"], 0.01)
        assert proc.killed is True
        assert proc.reaped is True

    async def test_missing_executable_propagates(self, monkeypatch):
        async def _fake_exec(*argv, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
        with pytest.raises(FileNotFoundError):
            await mod._run_systemctl(["nope"], 5.0)
