"""Simple container runtime detection for Docker and Podman.

Detects available container runtime and provides command helpers.
Requires modern runtimes: Docker Desktop 4.0+ or Podman 4.0+.

Examples:
    Basic usage::

        from osprey.deployment.runtime_helper import get_runtime_command

        cmd = get_runtime_command(config)
        # Returns: ['docker', 'compose'] or ['podman', 'compose']
"""

import os
import shutil
import subprocess

# Module-level cache for runtime command
_cached_runtime_cmd: list[str] | None = None


def get_runtime_command(config: dict | None = None) -> list[str]:
    """Get container compose command.

    Checks CONTAINER_RUNTIME env var, then config.container_runtime,
    auto-detects if 'auto' or not set. Result is cached after first detection.

    Args:
        config: Optional configuration dictionary

    Returns:
        Command list: ['docker', 'compose'] or ['podman', 'compose']

    Raises:
        RuntimeError: If no container runtime with compose support found
    """
    global _cached_runtime_cmd

    if _cached_runtime_cmd is not None:
        return _cached_runtime_cmd.copy()

    # Determine which runtime to try (priority: env var > config > auto)
    config_runtime = None
    if config:
        config_runtime = config.get("container_runtime", "auto")

    # Environment variable override
    env_runtime = os.getenv("CONTAINER_RUNTIME")
    if env_runtime:
        config_runtime = env_runtime

    # Build list of runtimes to try
    if config_runtime and config_runtime.lower() in ["docker", "podman"]:
        runtimes_to_try = [config_runtime.lower()]
    else:
        # Auto-detect: Docker first, then Podman
        runtimes_to_try = ["docker", "podman"]

    # Try each runtime
    for runtime in runtimes_to_try:
        if not shutil.which(runtime):
            continue

        try:
            # Check if compose subcommand works
            result = subprocess.run([runtime, "compose", "version"], capture_output=True, timeout=5)

            if result.returncode != 0:
                continue

            # Also verify daemon is actually running
            # This prevents selecting Docker when only Docker CLI is installed but daemon isn't running
            ps_result = subprocess.run([runtime, "ps"], capture_output=True, timeout=5)

            if ps_result.returncode == 0:
                _cached_runtime_cmd = [runtime, "compose"]
                return _cached_runtime_cmd.copy()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # No runtime found - check if any are installed but not running
    docker_installed = shutil.which("docker") is not None
    podman_installed = shutil.which("podman") is not None

    if docker_installed or podman_installed:
        # Runtimes are installed but not running
        error_parts = ["Container runtime installed but not running:\n"]
        if docker_installed:
            error_parts.append("\n" + _get_docker_not_running_message())
        if podman_installed:
            error_parts.append("\n" + _get_podman_not_running_message())
        raise RuntimeError("".join(error_parts))
    else:
        # No runtime installed at all
        raise RuntimeError(
            "No container runtime found. Install Docker Desktop 4.0+ or Podman 4.0+\n"
            "Docker: https://docs.docker.com/get-docker/\n"
            "Podman: https://podman.io/getting-started/installation"
        )


def verify_runtime_is_running(config: dict | None = None) -> tuple[bool, str]:
    """Verify that the detected container runtime is actually running.

    Args:
        config: Optional configuration dictionary

    Returns:
        Tuple of (is_running: bool, error_message: str)
        If running: (True, "")
        If not running: (False, helpful error message)
    """
    try:
        cmd = get_runtime_command(config)
        runtime = cmd[0]  # 'docker' or 'podman'

        # Try a simple ps command to verify daemon is accessible
        result = subprocess.run([runtime, "ps"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            return True, ""

        # Check for common error patterns
        stderr = result.stderr.lower()

        if "cannot connect to the docker daemon" in stderr or "docker daemon" in stderr:
            return False, _get_docker_not_running_message()

        if "cannot connect to podman" in stderr or "connection refused" in stderr:
            return False, _get_podman_not_running_message()

        # Generic error
        return False, f"{runtime.capitalize()} is installed but not responding:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        runtime = "container runtime"
        try:
            cmd = get_runtime_command(config)
            runtime = cmd[0]
        except Exception:
            pass
        return False, f"{runtime.capitalize()} command timed out. The service may not be running."
    except RuntimeError as e:
        # No runtime found at all
        return False, str(e)
    except Exception as e:
        return False, f"Error checking container runtime: {e}"


def _get_docker_not_running_message() -> str:
    """Get platform-specific message for Docker not running."""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        return (
            "Docker Desktop is not running.\n\n"
            "To fix this:\n"
            "1. Open Docker Desktop from Applications\n"
            "2. Wait for Docker to start (whale icon in menu bar should be steady)\n"
            "3. Try your command again\n\n"
            "If Docker Desktop is not installed:\n"
            "https://docs.docker.com/desktop/install/mac-install/"
        )
    elif system == "Windows":
        return (
            "Docker Desktop is not running.\n\n"
            "To fix this:\n"
            "1. Start Docker Desktop from the Start menu\n"
            "2. Wait for Docker to start (system tray icon should be running)\n"
            "3. Try your command again\n\n"
            "If Docker Desktop is not installed:\n"
            "https://docs.docker.com/desktop/install/windows-install/"
        )
    else:  # Linux
        return (
            "Docker daemon is not running.\n\n"
            "To fix this:\n"
            "1. Start Docker: sudo systemctl start docker\n"
            "2. Enable on boot: sudo systemctl enable docker\n"
            "3. Check status: sudo systemctl status docker\n\n"
            "If permission issues, add user to docker group:\n"
            "sudo usermod -aG docker $USER\n"
            "(then log out and back in)"
        )


def _get_podman_not_running_message() -> str:
    """Get platform-specific message for Podman not running."""
    import platform

    system = platform.system()

    if system in ["Darwin", "Windows"]:  # macOS or Windows
        return (
            "Podman machine is not running.\n\n"
            "To fix this:\n"
            "1. Start Podman: podman machine start\n"
            "2. Check status: podman machine list\n\n"
            "If no machine exists:\n"
            "1. Initialize: podman machine init\n"
            "2. Start: podman machine start"
        )
    else:  # Linux
        return (
            "Podman service is not responding.\n\n"
            "To fix this:\n"
            "1. Check status: systemctl --user status podman.socket\n"
            "2. Start if needed: systemctl --user start podman.socket\n\n"
            "For rootful Podman:\n"
            "sudo systemctl start podman.socket"
        )


def get_ps_command(config: dict | None = None, all_containers: bool = False) -> list[str]:
    """Get container ps command.

    Args:
        config: Optional configuration dictionary
        all_containers: Include stopped containers (-a flag)

    Returns:
        Command list for ps operation with JSON format
    """
    cmd = get_runtime_command(config)
    runtime = cmd[0]  # 'docker' or 'podman'

    ps_cmd = [runtime, "ps"]
    if all_containers:
        ps_cmd.append("-a")
    ps_cmd.extend(["--format", "json"])

    return ps_cmd
