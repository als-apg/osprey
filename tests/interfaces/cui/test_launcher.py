"""Tests for CUI subprocess launcher."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.cui.launcher import CUIProcessLauncher, ensure_cui_server


class TestResolveCommand:
    def test_binary_in_path(self):
        with patch("shutil.which", side_effect=lambda x: "/usr/local/bin/cui-server" if x == "cui-server" else None):
            cmd = CUIProcessLauncher._resolve_command("127.0.0.1", 3001)
        assert cmd == ["/usr/local/bin/cui-server", "--port", "3001", "--host", "127.0.0.1", "--skip-auth-token"]

    def test_falls_back_to_npx(self):
        def which_side_effect(name):
            if name == "cui-server":
                return None
            if name == "npx":
                return "/usr/local/bin/npx"
            return None

        with patch("shutil.which", side_effect=which_side_effect):
            cmd = CUIProcessLauncher._resolve_command("127.0.0.1", 3001)
        assert cmd[0] == "/usr/local/bin/npx"
        assert cmd[1] == "cui-server"
        assert "--port" in cmd
        assert "3001" in cmd

    def test_raises_when_nothing_found(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="cui-server not found"):
                CUIProcessLauncher._resolve_command("127.0.0.1", 3001)


class TestIsRunning:
    def test_healthy(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert CUIProcessLauncher._is_running("127.0.0.1", 3001) is True

    def test_not_healthy(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError):
            assert CUIProcessLauncher._is_running("127.0.0.1", 3001) is False


class TestEnsureRunning:
    def test_skips_when_already_launched(self):
        launcher = CUIProcessLauncher()
        launcher._launched = True
        # Should be a no-op — no Popen called
        with patch("subprocess.Popen") as mock_popen:
            launcher.ensure_running("127.0.0.1", 3001)
        mock_popen.assert_not_called()

    def test_skips_when_already_running(self):
        launcher = CUIProcessLauncher()
        with (
            patch.object(CUIProcessLauncher, "_is_running", return_value=True),
            patch("subprocess.Popen") as mock_popen,
        ):
            launcher.ensure_running("127.0.0.1", 3001)
        mock_popen.assert_not_called()
        assert launcher._launched is True

    def test_launches_when_not_running(self):
        launcher = CUIProcessLauncher()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        # First _is_running call (pre-check) returns False, second (readiness) returns True
        with (
            patch.object(CUIProcessLauncher, "_is_running", side_effect=[False, True]),
            patch.object(CUIProcessLauncher, "_resolve_command", return_value=["cui-server", "--port", "3001"]),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            launcher.ensure_running("127.0.0.1", 3001, cwd="/tmp")
        mock_popen.assert_called_once()
        assert launcher._launched is True


class TestEnsureCuiServer:
    def test_skips_when_disabled(self):
        with (
            patch("osprey.interfaces.cui.launcher._cui_auto_launch", return_value=False),
            patch("osprey.interfaces.cui.launcher._cui_launcher") as mock_launcher,
        ):
            ensure_cui_server()
        mock_launcher.ensure_running.assert_not_called()


class TestStop:
    def test_terminates_process(self):
        launcher = CUIProcessLauncher()
        mock_proc = MagicMock()
        launcher._process = mock_proc
        launcher._launched = True

        launcher.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert launcher._process is None
        assert launcher._launched is False

    def test_kills_on_timeout(self):
        launcher = CUIProcessLauncher()
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="cui-server", timeout=5), None]
        launcher._process = mock_proc
        launcher._launched = True

        launcher.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert launcher._process is None

    def test_stop_noop_when_no_process(self):
        launcher = CUIProcessLauncher()
        launcher.stop()  # Should not raise
