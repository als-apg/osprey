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
        with patch("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", candidates):
            with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
                result = user_bin_dirs()

        assert str(existing) in result
        assert str(missing) not in result

    def test_excludes_dirs_already_on_path(self, tmp_path):
        """Dirs already on PATH are excluded."""
        d = tmp_path / "bin"
        d.mkdir()

        candidates = [d]
        with patch("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", candidates):
            with patch.dict(os.environ, {"PATH": str(d)}, clear=False):
                result = user_bin_dirs()

        assert result == []

    def test_returns_dirs_not_on_path(self, tmp_path):
        """Dirs not on PATH that exist are returned."""
        d = tmp_path / "bin"
        d.mkdir()

        candidates = [d]
        with patch("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", candidates):
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
        with patch("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", candidates):
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
        with patch("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", candidates):
            with pytest.raises(FileNotFoundError, match="not found on PATH"):
                resolve_shell_command("this-command-definitely-does-not-exist-anywhere")

    def test_error_message_includes_config_hint(self, tmp_path):
        """The error message mentions config.yml as an escape hatch."""
        candidates = [tmp_path / "empty"]
        with patch("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", candidates):
            with pytest.raises(FileNotFoundError, match="web_terminal.shell"):
                resolve_shell_command("nonexistent-cmd")
