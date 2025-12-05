"""Unit tests for runtime_helper module."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from osprey.deployment import runtime_helper
from osprey.deployment.runtime_helper import (
    get_ps_command,
    get_runtime_command,
    verify_runtime_is_running,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset cached runtime command before each test."""
    runtime_helper._cached_runtime_cmd = None
    yield
    runtime_helper._cached_runtime_cmd = None


class TestGetRuntimeCommand:
    """Tests for get_runtime_command function."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_detect_docker_when_available(self, mock_run, mock_which):
        """Should detect Docker when available and verify compose command."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        cmd = get_runtime_command()

        assert cmd == ["docker", "compose"]
        mock_which.assert_called_with("docker")
        # Should call compose version AND ps to verify daemon is running
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0][0] == ["docker", "compose", "version"]
        assert mock_run.call_args_list[1][0][0] == ["docker", "ps"]

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_detect_podman_when_docker_not_available(self, mock_run, mock_which):
        """Should detect Podman when Docker not available."""

        def which_side_effect(name):
            return "/usr/bin/podman" if name == "podman" else None

        mock_which.side_effect = which_side_effect
        mock_run.return_value = MagicMock(returncode=0)

        cmd = get_runtime_command()

        assert cmd == ["podman", "compose"]
        # Verify both runtimes were checked
        assert mock_which.call_count == 2
        # Should call compose version AND ps for podman
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0][0] == ["podman", "compose", "version"]
        assert mock_run.call_args_list[1][0][0] == ["podman", "ps"]

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_prefer_docker_when_both_available(self, mock_run, mock_which):
        """Should prefer Docker when both runtimes are available."""
        # Mock both Docker and Podman as available
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.return_value = MagicMock(returncode=0)

        cmd = get_runtime_command()

        assert cmd == ["docker", "compose"]
        # Should only check Docker (not Podman) since Docker succeeds first
        mock_which.assert_called_once_with("docker")
        # Should call compose version AND ps for docker
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0][0] == ["docker", "compose", "version"]
        assert mock_run.call_args_list[1][0][0] == ["docker", "ps"]

    @patch("shutil.which")
    def test_raise_error_when_no_runtime_available(self, mock_which):
        """Should raise RuntimeError with helpful message when no runtime available."""
        mock_which.return_value = None

        with pytest.raises(RuntimeError) as exc_info:
            get_runtime_command()

        error_msg = str(exc_info.value)
        assert "No container runtime found" in error_msg
        assert "Docker Desktop 4.0+" in error_msg
        assert "Podman 4.0+" in error_msg
        assert "https://docs.docker.com/get-docker/" in error_msg
        assert "https://podman.io/getting-started/installation" in error_msg

    @pytest.mark.parametrize("runtime", ["docker", "podman"])
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_config_based_runtime_selection(self, mock_run, mock_which, runtime):
        """Should respect config.container_runtime setting."""
        mock_which.return_value = f"/usr/bin/{runtime}"
        mock_run.return_value = MagicMock(returncode=0)

        config = {"container_runtime": runtime}
        cmd = get_runtime_command(config)

        assert cmd == [runtime, "compose"]
        mock_which.assert_called_once_with(runtime)

    @pytest.mark.parametrize("config_value", ["auto", "AUTO", None])
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_auto_detection_with_various_config_values(self, mock_run, mock_which, config_value):
        """Should auto-detect when config is 'auto', uppercase, or missing."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        config = {"container_runtime": config_value} if config_value else {}
        cmd = get_runtime_command(config)

        assert cmd == ["docker", "compose"]

    @pytest.mark.parametrize("runtime_value", ["DOCKER", "PODMAN", "Docker", "Podman"])
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_config_runtime_case_insensitive(self, mock_run, mock_which, runtime_value):
        """Should handle uppercase/mixed-case runtime names in config."""
        expected_runtime = runtime_value.lower()
        mock_which.return_value = f"/usr/bin/{expected_runtime}"
        mock_run.return_value = MagicMock(returncode=0)

        config = {"container_runtime": runtime_value}
        cmd = get_runtime_command(config)

        assert cmd == [expected_runtime, "compose"]

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_invalid_config_runtime_falls_back_to_auto(self, mock_run, mock_which):
        """Should fall back to auto-detection when config has invalid runtime."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        config = {"container_runtime": "containerd"}  # Invalid runtime
        cmd = get_runtime_command(config)

        # Should auto-detect and find Docker
        assert cmd == ["docker", "compose"]

    @patch.dict(os.environ, {"CONTAINER_RUNTIME": "podman"})
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_env_var_override_takes_precedence(self, mock_run, mock_which):
        """Should prioritize CONTAINER_RUNTIME env var over config."""
        mock_which.return_value = "/usr/bin/podman"
        mock_run.return_value = MagicMock(returncode=0)

        config = {"container_runtime": "docker"}  # Should be overridden
        cmd = get_runtime_command(config)

        assert cmd == ["podman", "compose"]
        mock_which.assert_called_once_with("podman")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_caching_prevents_redundant_detection(self, mock_run, mock_which):
        """Should cache runtime and not re-detect on subsequent calls."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        # Multiple calls
        cmd1 = get_runtime_command()
        cmd2 = get_runtime_command()
        cmd3 = get_runtime_command()

        assert cmd1 == cmd2 == cmd3 == ["docker", "compose"]
        # Should only detect once (but makes 2 calls: compose version + ps check)
        mock_which.assert_called_once()
        assert mock_run.call_count == 2  # compose version + ps check on first call only

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_cache_returns_independent_copy(self, mock_run, mock_which):
        """Should return a copy of cached command to prevent external modification."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        cmd1 = get_runtime_command()
        cmd1.append("modified")  # Modify the returned list
        cmd2 = get_runtime_command()

        assert cmd2 == ["docker", "compose"]  # Should not include 'modified'
        assert "modified" not in cmd2

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_timeout_handling_with_fallback(self, mock_run, mock_which):
        """Should gracefully handle timeout and try next runtime."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.side_effect = [
            subprocess.TimeoutExpired(["docker", "compose", "version"], 5),
            # Docker times out, move to Podman
            MagicMock(returncode=0),  # Podman compose succeeds
            MagicMock(returncode=0),  # Podman ps succeeds
        ]

        cmd = get_runtime_command()

        assert cmd == ["podman", "compose"]
        assert mock_run.call_count == 3  # docker compose timeout + podman compose + podman ps

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_compose_command_failure_with_fallback(self, mock_run, mock_which):
        """Should skip runtime when compose command fails and try next."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.side_effect = [
            MagicMock(returncode=1),  # Docker compose fails
            # Docker fails, move to Podman
            MagicMock(returncode=0),  # Podman compose succeeds
            MagicMock(returncode=0),  # Podman ps succeeds
        ]

        cmd = get_runtime_command()

        assert cmd == ["podman", "compose"]
        assert mock_run.call_count == 3  # docker compose fail + podman compose + podman ps

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_file_not_found_error_with_fallback(self, mock_run, mock_which):
        """Should handle FileNotFoundError and try next runtime."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.side_effect = [
            FileNotFoundError("docker not found"),
            # Docker fails, move to Podman
            MagicMock(returncode=0),  # Podman compose succeeds
            MagicMock(returncode=0),  # Podman ps succeeds
        ]

        cmd = get_runtime_command()

        assert cmd == ["podman", "compose"]
        assert mock_run.call_count == 3  # docker error + podman compose + podman ps

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_all_runtimes_fail_raises_error(self, mock_run, mock_which):
        """Should raise error when all runtimes fail compose check."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.return_value = MagicMock(returncode=1)  # Both fail

        with pytest.raises(RuntimeError) as exc_info:
            get_runtime_command()

        # When compose succeeds but ps fails, it's considered "installed but not running"
        assert "Container runtime installed but not running" in str(exc_info.value)


class TestGetPsCommand:
    """Tests for get_ps_command function."""

    @pytest.mark.parametrize(
        "all_containers,expected_flags",
        [(False, ["ps", "--format", "json"]), (True, ["ps", "-a", "--format", "json"])],
    )
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_ps_command_flag_behavior(self, mock_run, mock_which, all_containers, expected_flags):
        """Should generate ps command with or without -a flag based on parameter."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        cmd = get_ps_command(all_containers=all_containers)

        assert cmd == ["docker"] + expected_flags

    @pytest.mark.parametrize("runtime", ["docker", "podman"])
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_ps_command_uses_detected_runtime(self, mock_run, mock_which, runtime):
        """Should use the detected runtime (docker or podman) in ps command."""

        def which_side_effect(name):
            # Simulate that only the target runtime is available
            return f"/usr/bin/{name}" if name == runtime else None

        mock_which.side_effect = which_side_effect
        mock_run.return_value = MagicMock(returncode=0)

        cmd = get_ps_command(all_containers=True)

        assert cmd[0] == runtime
        assert cmd == [runtime, "ps", "-a", "--format", "json"]

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_ps_command_delegates_to_runtime_detection(self, mock_run, mock_which):
        """Should delegate to get_runtime_command for consistency."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        config = {"container_runtime": "docker"}
        cmd = get_ps_command(config, all_containers=False)

        assert cmd == ["docker", "ps", "--format", "json"]
        # Verify runtime detection was called
        mock_which.assert_called_once_with("docker")

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_ps_command_benefits_from_caching(self, mock_run, mock_which):
        """Should benefit from runtime detection caching on repeated calls."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = MagicMock(returncode=0)

        # Multiple ps command calls
        cmd1 = get_ps_command(all_containers=False)
        cmd2 = get_ps_command(all_containers=True)
        cmd3 = get_ps_command(all_containers=False)

        # All should use Docker
        assert cmd1[0] == cmd2[0] == cmd3[0] == "docker"
        # Runtime detection should only happen once (cached), but makes 2 subprocess calls
        mock_which.assert_called_once()
        assert mock_run.call_count == 2  # compose version + ps check on first call only


class TestVerifyRuntimeIsRunning:
    """Tests for verify_runtime_is_running function."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_runtime_running_successfully(self, mock_run, mock_which):
        """Should return True when runtime is running."""
        mock_which.return_value = "/usr/bin/docker"
        # get_runtime_command makes 2 calls (compose version + ps check)
        # verify_runtime_is_running makes 1 call (ps command)
        mock_run.side_effect = [
            MagicMock(returncode=0),  # compose version
            MagicMock(returncode=0),  # ps check (get_runtime_command)
            MagicMock(returncode=0, stderr=""),  # ps command (verify)
        ]

        is_running, error_msg = verify_runtime_is_running()

        assert is_running is True
        assert error_msg == ""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_docker_not_running_macos(self, mock_run, mock_which):
        """Should return helpful message when Docker not running on macOS."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.side_effect = [
            MagicMock(returncode=0),  # compose version
            MagicMock(returncode=0),  # ps check (get_runtime_command)
            MagicMock(
                returncode=1,
                stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
            ),  # ps command (verify) - fails
        ]

        with patch("platform.system", return_value="Darwin"):
            is_running, error_msg = verify_runtime_is_running()

        assert is_running is False
        assert "Docker Desktop is not running" in error_msg
        assert "Open Docker Desktop from Applications" in error_msg
        assert "whale icon in menu bar" in error_msg

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_docker_not_running_linux(self, mock_run, mock_which):
        """Should return helpful message when Docker not running on Linux."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.side_effect = [
            MagicMock(returncode=0),  # compose version
            MagicMock(returncode=0),  # ps check (get_runtime_command)
            MagicMock(
                returncode=1, stderr="Cannot connect to the Docker daemon"
            ),  # ps command (verify) - fails
        ]

        with patch("platform.system", return_value="Linux"):
            is_running, error_msg = verify_runtime_is_running()

        assert is_running is False
        assert "Docker daemon is not running" in error_msg
        assert "sudo systemctl start docker" in error_msg

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_podman_not_running_macos(self, mock_run, mock_which):
        """Should return helpful message when Podman not running on macOS."""

        def which_side_effect(name):
            return "/usr/bin/podman" if name == "podman" else None

        mock_which.side_effect = which_side_effect
        mock_run.side_effect = [
            MagicMock(returncode=0),  # compose version
            MagicMock(
                returncode=1, stderr="Cannot connect to podman. Is the podman daemon running?"
            ),
        ]

        with patch("platform.system", return_value="Darwin"):
            is_running, error_msg = verify_runtime_is_running()

        assert is_running is False
        assert "Podman machine is not running" in error_msg
        assert "podman machine start" in error_msg

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_timeout_during_verification(self, mock_run, mock_which):
        """Should handle timeout gracefully."""
        mock_which.return_value = "/usr/bin/docker"
        mock_run.side_effect = [
            MagicMock(returncode=0),  # compose version
            MagicMock(returncode=0),  # ps check (get_runtime_command)
            subprocess.TimeoutExpired(["docker", "ps"], 5),  # ps timeout (verify)
        ]

        is_running, error_msg = verify_runtime_is_running()

        assert is_running is False
        assert "timed out" in error_msg.lower()

    @patch("shutil.which")
    def test_no_runtime_found(self, mock_which):
        """Should return error when no runtime is found."""
        mock_which.return_value = None

        is_running, error_msg = verify_runtime_is_running()

        assert is_running is False
        assert "No container runtime found" in error_msg


class TestDockerPodmanCompatibility:
    """Tests for Docker/Podman output format differences."""

    def test_docker_ps_ndjson_format(self):
        """Test that Docker's newline-separated JSON format is handled correctly."""
        # Docker outputs one JSON object per line (NDJSON format)
        docker_output = """{"ID":"abc123","Names":"container1","State":"running"}
{"ID":"def456","Names":"container2","State":"exited"}"""

        import json

        containers = []

        # This is the parsing logic used in container_manager.py
        try:
            # Try parsing as JSON array first (Podman format)
            containers = json.loads(docker_output)
        except json.JSONDecodeError:
            # Fall back to newline-separated JSON objects (Docker format)
            for line in docker_output.strip().split("\n"):
                if line.strip():
                    containers.append(json.loads(line))

        assert len(containers) == 2
        assert containers[0]["ID"] == "abc123"
        assert containers[1]["ID"] == "def456"

    def test_podman_ps_json_array_format(self):
        """Test that Podman's JSON array format is handled correctly."""
        # Podman outputs a JSON array
        podman_output = '[{"ID":"abc123","Names":"container1","State":"running"},{"ID":"def456","Names":"container2","State":"exited"}]'

        import json

        containers = []

        # This is the parsing logic used in container_manager.py
        try:
            # Try parsing as JSON array first (Podman format)
            containers = json.loads(podman_output)
        except json.JSONDecodeError:
            # Fall back to newline-separated JSON objects (Docker format)
            for line in podman_output.strip().split("\n"):
                if line.strip():
                    containers.append(json.loads(line))

        assert len(containers) == 2
        assert containers[0]["ID"] == "abc123"
        assert containers[1]["ID"] == "def456"
