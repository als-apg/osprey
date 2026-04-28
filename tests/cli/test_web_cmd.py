"""Tests for osprey web CLI command (detach/stop and backward compat)."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.web_cmd import (
    LOG_FILE,
    PID_FILE,
    _read_pid,
    _wait_for_server,
    _write_pid,
    web,
)


@pytest.fixture
def runner():
    return CliRunner()


# -- help / backward compat ------------------------------------------------


def test_web_help_shows_detach_and_stop(runner: CliRunner):
    result = runner.invoke(web, ["--help"])
    assert result.exit_code == 0
    assert "--detach" in result.output
    assert "stop" in result.output


# -- _read_pid --------------------------------------------------------------


def test_read_pid_missing(tmp_path: Path):
    assert _read_pid(tmp_path) is None


def test_read_pid_valid(tmp_path: Path):
    pid = os.getpid()  # current process — guaranteed alive
    (tmp_path / PID_FILE).write_text(str(pid))
    assert _read_pid(tmp_path) == pid


def test_read_pid_stale(tmp_path: Path):
    (tmp_path / PID_FILE).write_text("999999999")
    with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
        result = _read_pid(tmp_path)
    assert result is None
    assert not (tmp_path / PID_FILE).exists()


def test_read_pid_corrupt(tmp_path: Path):
    (tmp_path / PID_FILE).write_text("not-a-number")
    assert _read_pid(tmp_path) is None
    assert not (tmp_path / PID_FILE).exists()


# -- _write_pid -------------------------------------------------------------


def test_write_pid(tmp_path: Path):
    _write_pid(tmp_path, 42)
    assert (tmp_path / PID_FILE).read_text() == "42"


# -- _wait_for_server -------------------------------------------------------


def test_wait_for_server_success():
    proc = MagicMock()
    proc.poll.return_value = None

    with patch("osprey.cli.web_cmd.socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock()
        assert _wait_for_server("127.0.0.1", 8087, proc, timeout=2.0) is True


def test_wait_for_server_timeout():
    proc = MagicMock()
    proc.poll.return_value = None

    with patch("osprey.cli.web_cmd.socket.create_connection", side_effect=OSError):
        assert _wait_for_server("127.0.0.1", 8087, proc, timeout=0.5) is False


def test_wait_for_server_early_crash():
    proc = MagicMock()
    proc.poll.return_value = 1  # process already exited

    assert _wait_for_server("127.0.0.1", 8087, proc, timeout=5.0) is False


# -- detach -----------------------------------------------------------------


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_spawns_subprocess(
    mock_config, mock_popen, mock_wait, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    mock_popen.assert_called_once()
    call_kwargs = mock_popen.call_args
    assert call_kwargs.kwargs["start_new_session"] is True


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_writes_pid_file(mock_config, mock_popen, mock_wait, mock_resolve, tmp_path, runner):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(web, ["--detach"])
        pid_content = (Path(td) / PID_FILE).read_text()

    assert result.exit_code == 0
    assert pid_content == "12345"


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._read_pid", return_value=99999)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_idempotent_when_running(mock_config, mock_read_pid, mock_resolve, runner, tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    assert "already running" in result.output


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_cleans_stale_pid(
    mock_config, mock_popen, mock_wait, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 55555
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # Write a stale PID file
        (Path(td) / PID_FILE).write_text("999999999")

        # _read_pid will call os.kill which raises ProcessLookupError for stale
        with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
            result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    # Should have started a new server
    mock_popen.assert_called_once()


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_shows_url_and_pid(
    mock_config, mock_popen, mock_wait, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["--detach"])

    assert "PID 12345" in result.output
    assert "http://127.0.0.1:8087" in result.output
    assert "osprey web stop" in result.output


# -- stop -------------------------------------------------------------------


def test_stop_kills_process(tmp_path, runner):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        (Path(td) / PID_FILE).write_text("12345")
        (Path(td) / LOG_FILE).write_text("some log")

        with patch("osprey.cli.web_cmd.os.kill") as mock_kill:
            result = runner.invoke(web, ["stop"])

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert result.exit_code == 0
        assert "Stopped" in result.output
        assert not (Path(td) / PID_FILE).exists()
        assert not (Path(td) / LOG_FILE).exists()


def test_stop_no_pid_file(tmp_path, runner):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["stop"])

    assert result.exit_code == 0
    assert "No running" in result.output


def test_stop_stale_pid(tmp_path, runner):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        (Path(td) / PID_FILE).write_text("999999999")

        with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
            result = runner.invoke(web, ["stop"])

        assert result.exit_code == 0
        assert "not found" in result.output
        assert not (Path(td) / PID_FILE).exists()
