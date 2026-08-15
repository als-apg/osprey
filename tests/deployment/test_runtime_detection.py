"""Unit tests for runtime detection in ``deployment.runtime_helper``.

Covers the parts ``test_runtime_helper.py`` leaves untested: ``get_runtime_command``
(env/config priority, compose+daemon probing, caching, error paths),
``verify_runtime_is_running``, ``get_ps_command``, and the platform-specific
"not running" help messages.

The module memoizes the detected runtime per process. The suite-wide
``reset_runtime_cache`` fixture in ``conftest.py`` clears that memo around every
test, so the serial unit lane never leaks a cached decision (or the host's real
docker/podman) between tests.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import runtime_helper
from osprey.deployment.runtime_helper import (
    PODMAN_COMPOSE_MIN_VERSION,
    ComposeProvider,
    UnsupportedComposeProviderError,
    detect_compose_provider,
    get_ps_command,
    get_runtime_command,
    verify_runtime_is_running,
)


@pytest.fixture(autouse=True)
def clear_container_runtime_env(monkeypatch):
    """Drop any host ``CONTAINER_RUNTIME`` override so detection is deterministic."""
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)


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


#: Real banners, as the providers print them. ``podman compose`` announces the
#: external provider it is about to hand off to on stderr and the provider's own
#: banner lands on stdout, which is why the probe reads both streams.
_PODMAN_DISPATCH_NOTICE = (
    '>>>> Executing external compose provider "/usr/bin/podman-compose". '
    "Please refer to the documentation for details. <<<<"
)
_DOCKER_DISPATCH_NOTICE = (
    '>>>> Executing external compose provider "/usr/libexec/docker/cli-plugins/docker-compose". '
    "Please refer to the documentation for details. <<<<"
)


def _fake_version_probe(stdout="", stderr="", returncode=0, calls=None):
    """Fake ``subprocess.run`` returning a canned ``compose version`` banner."""

    def _run(cmd, **kwargs):
        if calls is not None:
            calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    return _run


class TestDetectComposeProvider:
    def test_podman_compose_banner_detected_with_version(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(
                stdout="podman-compose version 1.0.6\npodman version 4.9.4-rhel\n",
                stderr=_PODMAN_DISPATCH_NOTICE,
            ),
        )

        info = detect_compose_provider(["podman", "compose"])

        assert info.provider is ComposeProvider.PODMAN_COMPOSE
        assert info.version == (1, 0, 6)
        assert info.version_text == "1.0.6"

    def test_docker_compose_v2_banner_detected(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="Docker Compose version v2.24.5\n"),
        )

        info = detect_compose_provider(["docker", "compose"])

        assert info.provider is ComposeProvider.DOCKER_V2
        assert info.version == (2, 24, 5)

    def test_podman_delegating_to_docker_reports_docker_v2(self, monkeypatch):
        # The binary is podman, but Docker Compose v2 is what parses the argv —
        # the shape must follow the provider, not cmd[0].
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(
                stdout="Docker Compose version v2.29.1\n", stderr=_DOCKER_DISPATCH_NOTICE
            ),
        )

        info = detect_compose_provider(["podman", "compose"])

        assert info.provider is ComposeProvider.DOCKER_V2
        assert info.version_text == "2.29.1"

    def test_podman_compose_below_floor_fails_closed(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="podman-compose version 1.0.3\n"),
        )

        with pytest.raises(UnsupportedComposeProviderError) as excinfo:
            detect_compose_provider(["podman", "compose"])

        message = str(excinfo.value)
        # Names both the version found and the floor it fell short of.
        assert "1.0.3" in message
        assert ".".join(str(part) for part in PODMAN_COMPOSE_MIN_VERSION) in message

    def test_podman_compose_at_floor_is_accepted(self, monkeypatch):
        floor = ".".join(str(part) for part in PODMAN_COMPOSE_MIN_VERSION)
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout=f"podman-compose version {floor}\n"),
        )

        info = detect_compose_provider(["podman", "compose"])

        assert info.provider is ComposeProvider.PODMAN_COMPOSE
        assert info.version == PODMAN_COMPOSE_MIN_VERSION

    def test_unrecognized_banner_fails_closed_citing_supported_providers(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="some-other-compose 0.4.2\n"),
        )

        with pytest.raises(UnsupportedComposeProviderError) as excinfo:
            detect_compose_provider(["podman", "compose"])

        message = str(excinfo.value)
        assert "Docker Compose v2" in message
        assert "podman-compose" in message
        assert "compatibility statement" in message
        # The banner is echoed so an operator can see what answered.
        assert "some-other-compose 0.4.2" in message

    def test_compose_v1_banner_is_not_read_as_v2(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="docker-compose version 1.29.2, build 5becea4c\n"),
        )

        with pytest.raises(UnsupportedComposeProviderError):
            detect_compose_provider(["docker", "compose"])

    def test_podman_compose_without_parseable_version_fails_closed(self, monkeypatch):
        # The dispatcher named podman-compose but no banner followed, so the
        # floor cannot be checked.
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stderr=_PODMAN_DISPATCH_NOTICE),
        )

        with pytest.raises(UnsupportedComposeProviderError, match="no parseable version"):
            detect_compose_provider(["podman", "compose"])

    def test_probe_runs_version_against_the_given_base(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="Docker Compose version v2.24.5\n", calls=calls),
        )

        detect_compose_provider(["docker", "compose"])

        assert calls == [["docker", "compose", "version"]]

    def test_base_defaults_to_detected_runtime_command(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper, "get_runtime_command", lambda config=None: ["podman", "compose"]
        )
        calls: list[list[str]] = []
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="podman-compose version 1.5.0\n", calls=calls),
        )

        info = detect_compose_provider()

        assert info.provider is ComposeProvider.PODMAN_COMPOSE
        assert calls == [["podman", "compose", "version"]]

    def test_result_is_memoized_per_base(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="podman-compose version 1.0.6\n", calls=calls),
        )

        first = detect_compose_provider(["podman", "compose"])
        second = detect_compose_provider(["podman", "compose"])

        assert first == second
        assert len(calls) == 1

        # A different base is a different question, so it probes again.
        detect_compose_provider(["docker", "compose"])
        assert len(calls) == 2

    def test_reset_runtime_cache_clears_the_provider_memo(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="podman-compose version 1.0.6\n", calls=calls),
        )

        detect_compose_provider(["podman", "compose"])
        runtime_helper.reset_runtime_cache()
        detect_compose_provider(["podman", "compose"])

        assert len(calls) == 2

    def test_failed_probe_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(
            runtime_helper.subprocess, "run", _fake_version_probe(stdout="mystery-compose 9\n")
        )
        with pytest.raises(UnsupportedComposeProviderError):
            detect_compose_provider(["podman", "compose"])

        # Host fixed, same process: the refusal must not have been pinned.
        monkeypatch.setattr(
            runtime_helper.subprocess,
            "run",
            _fake_version_probe(stdout="podman-compose version 1.5.0\n"),
        )
        assert detect_compose_provider(["podman", "compose"]).version == (1, 5, 0)

    def test_timeout_fails_closed(self, monkeypatch):
        def _run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 15)

        monkeypatch.setattr(runtime_helper.subprocess, "run", _run)

        with pytest.raises(UnsupportedComposeProviderError, match="timed out"):
            detect_compose_provider(["podman", "compose"])

    def test_missing_binary_fails_closed(self, monkeypatch):
        def _run(cmd, **kwargs):
            raise FileNotFoundError("podman")

        monkeypatch.setattr(runtime_helper.subprocess, "run", _run)

        with pytest.raises(UnsupportedComposeProviderError, match="could not run"):
            detect_compose_provider(["podman", "compose"])

    def test_refusal_is_a_runtime_error(self):
        assert issubclass(UnsupportedComposeProviderError, RuntimeError)


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
