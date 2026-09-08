"""Tests for shell command resolution utilities.

These tests verify that ``user_bin_dirs()`` and ``resolve_shell_command()``
correctly find executables in well-known user-local directories when they
are missing from the default PATH.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.utils.shell_resolver import resolve_shell_command, user_bin_dirs


class TestUserBinDirs:
    """Test user_bin_dirs() filtering logic."""

    def test_returns_only_existing_dirs(self, tmp_path):
        """Dirs that don't exist on disk are excluded."""
        existing = tmp_path / "bin"
        existing.mkdir()
        missing = tmp_path / "nope"

        candidates = [existing, missing]
        with patch("osprey.utils.shell_resolver._user_bin_candidates", lambda: candidates):
            with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
                result = user_bin_dirs()

        assert str(existing) in result
        assert str(missing) not in result

    def test_excludes_dirs_already_on_path(self, tmp_path):
        """Dirs already on PATH are excluded."""
        d = tmp_path / "bin"
        d.mkdir()

        candidates = [d]
        with patch("osprey.utils.shell_resolver._user_bin_candidates", lambda: candidates):
            with patch.dict(os.environ, {"PATH": str(d)}, clear=False):
                result = user_bin_dirs()

        assert result == []

    def test_returns_dirs_not_on_path(self, tmp_path):
        """Dirs not on PATH that exist are returned."""
        d = tmp_path / "bin"
        d.mkdir()

        candidates = [d]
        with patch("osprey.utils.shell_resolver._user_bin_candidates", lambda: candidates):
            with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
                result = user_bin_dirs()

        assert result == [str(d)]


class TestResolveShellCommand:
    """Test resolve_shell_command() lookup logic."""

    def test_found_on_current_path(self):
        """Commands already on PATH resolve normally."""
        # 'sh' is universally available
        result = resolve_shell_command("sh")
        assert os.path.isabs(result)
        assert os.path.isfile(result)

    def test_found_in_user_bin_dir(self, tmp_path):
        """Commands in user-local dirs are found when not on PATH."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_cmd = bin_dir / "my-shell"
        fake_cmd.touch()
        fake_cmd.chmod(fake_cmd.stat().st_mode | stat.S_IEXEC)

        candidates = [bin_dir]
        with patch("osprey.utils.shell_resolver._user_bin_candidates", lambda: candidates):
            with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
                result = resolve_shell_command("my-shell")

        assert result == str(fake_cmd)

    def test_absolute_path_exists(self, tmp_path):
        """Absolute paths that exist and are executable pass through."""
        fake_cmd = tmp_path / "my-shell"
        fake_cmd.touch()
        fake_cmd.chmod(fake_cmd.stat().st_mode | stat.S_IEXEC)

        result = resolve_shell_command(str(fake_cmd))
        assert result == str(fake_cmd)

    def test_absolute_path_missing_raises(self, tmp_path):
        """Absolute paths that don't exist raise FileNotFoundError."""
        missing = str(tmp_path / "does-not-exist")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_shell_command(missing)

    def test_not_found_anywhere_raises(self, tmp_path):
        """Commands not on PATH or in user dirs raise FileNotFoundError."""
        candidates = [tmp_path / "empty"]
        with patch("osprey.utils.shell_resolver._user_bin_candidates", lambda: candidates):
            with pytest.raises(FileNotFoundError, match="not found on PATH"):
                resolve_shell_command("this-command-definitely-does-not-exist-anywhere")

    def test_error_message_includes_config_hint(self, tmp_path):
        """The error message mentions config.yml as an escape hatch."""
        candidates = [tmp_path / "empty"]
        with patch("osprey.utils.shell_resolver._user_bin_candidates", lambda: candidates):
            with pytest.raises(FileNotFoundError, match="web_terminal.shell"):
                resolve_shell_command("nonexistent-cmd")


class TestAnAccountWithNoHome:
    """A uid with no passwd entry and no ``HOME`` degrades; it does not raise.

    A random-uid cluster policy gives a container neither. ``Path.home()`` then
    raises ``RuntimeError``, and this module used to make that call twice at
    import — so a module that merely imports this one (``agent_runner.clean_env``
    does, at module scope) failed as an ImportError chain rather than running
    with a shorter PATH, which is all this list is for.
    """

    def _no_home(self, monkeypatch):
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("USERPROFILE", raising=False)
        monkeypatch.setattr(
            "osprey.utils.shell_resolver.Path.home",
            staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no home"))),
        )

    def test_user_bin_dirs_still_answers(self, monkeypatch):
        self._no_home(monkeypatch)
        monkeypatch.setenv("PATH", "/usr/bin")

        assert isinstance(user_bin_dirs(), list)

    def test_the_home_relative_entries_are_skipped(self, monkeypatch):
        from osprey.utils import shell_resolver

        self._no_home(monkeypatch)

        candidates = shell_resolver._user_bin_candidates()

        assert candidates == [Path("/usr/local/bin")]

    def test_a_command_on_path_still_resolves(self, monkeypatch, tmp_path):
        self._no_home(monkeypatch)
        binary = tmp_path / "tool"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", str(tmp_path))

        assert resolve_shell_command("tool") == str(binary)

    def test_the_not_found_message_still_renders(self, monkeypatch):
        self._no_home(monkeypatch)
        monkeypatch.setenv("PATH", "/nonexistent")

        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            resolve_shell_command("definitely-not-a-real-command")
