"""Unit tests for runtime detection in ``deployment.runtime_helper``.

Covers the parts ``test_runtime_helper.py`` leaves untested: ``get_runtime_command``
(env/config priority, compose+daemon probing, caching, error paths),
``verify_runtime_is_running``, ``get_ps_command``, and the platform-specific
"not running" help messages.

The module caches the detected runtime in ``_cached_runtime_cmd``. Every test
runs under the ``reset_runtime_cache`` fixture so the serial unit lane never
leaks a cached decision (or the host's real docker/podman) between tests.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import runtime_helper
from osprey.deployment.runtime_helper import (
    get_ps_command,
    get_runtime_command,
    verify_runtime_is_running,
)


@pytest.fixture(autouse=True)
def reset_runtime_cache(monkeypatch):
    """Clear the module-level runtime cache and env override around each test."""
    monkeypatch.setattr(runtime_helper, "_cached_runtime_cmd", None)
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    yield
    runtime_helper._cached_runtime_cmd = None


def _make_run(exit_map):
    """Build a fake subprocess.run keyed on the (runtime, subcommand) invoked.

    ``exit_map`` maps a tuple like ("docker", "compose") or ("docker", "ps")
    to a return code. Unlisted invocations return code 1 (failure).
    """

    def _fake_run(cmd, **kwargs):
        key = (cmd[0], cmd[1])
        rc = exit_map.get(key, 1)
        return subprocess.CompletedProcess(cmd, rc, stdout=b"", stderr=b"")

    return _fake_run


class TestGetRuntimeCommandDetection:
    def test_docker_selected_when_compose_and_daemon_ok(self, monkeypatch):
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _make_run({("docker", "compose"): 0, ("docker", "ps"): 0}),
        )

        assert get_runtime_command() == ["docker", "compose"]

    def test_falls_through_to_podman_when_docker_daemon_down(self, monkeypatch):
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
        # Docker compose works but its daemon (ps) is down; podman is healthy.
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _make_run(
                {
                    ("docker", "compose"): 0,
                    ("docker", "ps"): 1,
                    ("podman", "compose"): 0,
                    ("podman", "ps"): 0,
                }
            ),
        )

        assert get_runtime_command() == ["podman", "compose"]

    def test_config_runtime_pins_choice(self, monkeypatch):
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _make_run({("podman", "compose"): 0, ("podman", "ps"): 0}),
        )

        assert get_runtime_command({"container_runtime": "podman"}) == ["podman", "compose"]

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_RUNTIME", "podman")
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _make_run({("podman", "compose"): 0, ("podman", "ps"): 0}),
        )

        # Config says docker, env says podman → env wins.
        assert get_runtime_command({"container_runtime": "docker"}) == ["podman", "compose"]

    def test_result_is_cached_across_calls(self, monkeypatch):
        calls = {"n": 0}

        def _run(cmd, **kwargs):
            calls["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(runtime_helper.subprocess, "run", _run)

        first = get_runtime_command()
        after_first = calls["n"]
        second = get_runtime_command()

        assert first == second
        # Second call served from cache — no further subprocess probing.
        assert calls["n"] == after_first

    def test_cached_result_is_a_copy(self, monkeypatch):
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _make_run({("docker", "compose"): 0, ("docker", "ps"): 0}),
        )

        cmd = get_runtime_command()
        cmd.append("mutated")
        # Mutating the returned list must not corrupt the cache.
        assert get_runtime_command() == ["docker", "compose"]

    def test_no_runtime_installed_raises_with_install_hint(self, monkeypatch):
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: None)

        with pytest.raises(RuntimeError, match="No container runtime found"):
            get_runtime_command()

    def test_installed_but_not_running_raises_with_start_hint(self, monkeypatch):
        # docker is on PATH but its compose probe fails → treated as not running.
        monkeypatch.setattr(
            runtime_helper.shutil,
            "which",
            lambda name: "/usr/bin/docker" if name == "docker" else None,
        )
        monkeypatch.setattr(runtime_helper.subprocess, "run", _make_run({("docker", "compose"): 1}))

        with pytest.raises(RuntimeError, match="installed but not running"):
            get_runtime_command()

    def test_timeout_during_probe_is_swallowed_then_reported_missing(self, monkeypatch):
        monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")

        def _run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(runtime_helper.subprocess, "run", _run)

        # Both runtimes time out during probing → falls through to the
        # installed-but-not-running branch (both are on PATH).
        with pytest.raises(RuntimeError, match="installed but not running"):
            get_runtime_command()


class TestVerifyRuntimeIsRunning:
    def test_running_returns_true_empty_message(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
        )
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        )

        assert verify_runtime_is_running() == (True, "")

    def test_docker_daemon_down_returns_helpful_message(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
        )
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="Cannot connect to the Docker daemon"
            ),
        )

        ok, msg = verify_runtime_is_running()
        assert ok is False
        assert "Docker" in msg

    def test_podman_connection_refused_returns_podman_message(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["podman", "compose"]
        )
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="connection refused"
            ),
        )

        ok, msg = verify_runtime_is_running()
        assert ok is False
        assert "Podman" in msg

    def test_generic_nonzero_returns_raw_stderr(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
        )
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="something weird happened"
            ),
        )

        ok, msg = verify_runtime_is_running()
        assert ok is False
        assert "something weird happened" in msg

    def test_timeout_returns_timeout_message(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
        )

        def _run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(runtime_helper.subprocess, "run", _run)

        ok, msg = verify_runtime_is_running()
        assert ok is False
        assert "timed out" in msg

    def test_no_runtime_returns_runtime_error_text(self, monkeypatch):
        def _raise(config=None):
            raise RuntimeError("No container runtime found")

        monkeypatch.setattr(runtime_helper, "get_runtime_command", _raise)

        ok, msg = verify_runtime_is_running()
        assert ok is False
        assert "No container runtime found" in msg


class TestGetPsCommand:
    def test_ps_command_json_format(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
        )

        assert get_ps_command() == ["docker", "ps", "--format", "json"]

    def test_ps_command_all_containers_flag(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["podman", "compose"]
        )

        assert get_ps_command(all_containers=True) == [
            "podman",
            "ps",
            "-a",
            "--format",
            "json",
        ]


class TestNotRunningMessages:
    def test_docker_message_is_platform_specific(self, monkeypatch):
        import platform

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert "Docker Desktop" in runtime_helper._get_docker_not_running_message()

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert "systemctl" in runtime_helper._get_docker_not_running_message()

    def test_podman_message_is_platform_specific(self, monkeypatch):
        import platform

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert "podman machine" in runtime_helper._get_podman_not_running_message()

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert "podman.socket" in runtime_helper._get_podman_not_running_message()
