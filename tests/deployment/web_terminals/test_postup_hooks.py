"""Unit tests for the advisory host-side hooks around the web-terminal deploy.

Covers ``osprey.deployment.web_terminals.postup_hooks`` in isolation: the
rootless-podman linger step, the post-up host-reachability probe, and the
advisory ``verify.sh`` smoke check.
"""

from __future__ import annotations

import pytest

from osprey.deployment.web_terminals import postup_hooks

# ---------------------------------------------------------------------------
# enable_linger -- rootless-podman persistence via loginctl
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_enable_linger_skips_on_docker_runtime(monkeypatch):
    """Docker has no per-user systemd session -- linger never applies there."""
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(
        postup_hooks.shutil,
        "which",
        lambda name: pytest.fail("loginctl should not be probed for docker"),
    )
    calls = []
    monkeypatch.setattr(postup_hooks.subprocess, "run", lambda *a, **k: calls.append(a))

    postup_hooks.enable_linger({}, {})

    assert calls == []


def test_enable_linger_skips_when_loginctl_absent(monkeypatch):
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(postup_hooks.subprocess, "run", lambda *a, **k: calls.append(a))

    postup_hooks.enable_linger({}, {})

    assert calls == []


def test_enable_linger_noop_when_already_enabled(monkeypatch):
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd == ["loginctl", "show-user", "deployuser", "--property=Linger"]
        return _FakeCompletedProcess(returncode=0, stdout="Linger=yes\n")

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    postup_hooks.enable_linger({}, {})

    # Only the status check ran -- enable-linger is never invoked once we
    # already know it's on, so a no-op deploy stays quiet.
    assert len(calls) == 1


def test_enable_linger_enables_when_not_yet_enabled(monkeypatch):
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    postup_hooks.enable_linger({}, {})

    assert calls == [
        ["loginctl", "show-user", "deployuser", "--property=Linger"],
        ["loginctl", "enable-linger", "deployuser"],
    ]


def test_enable_linger_enable_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")

    def _fake_run(cmd, **kwargs):
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="Permission denied")

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    postup_hooks.enable_linger({}, {})  # must not raise


def test_enable_linger_status_check_error_still_attempts_enable(monkeypatch):
    """A broken status check (loginctl show-user itself errors) must not
    prevent the enable attempt -- only a confirmed already-enabled state
    short-circuits it."""
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            raise OSError("boom")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    postup_hooks.enable_linger({}, {})  # must not raise

    assert calls[-1] == ["loginctl", "enable-linger", "deployuser"]


def test_enable_linger_enable_call_error_does_not_raise(monkeypatch):
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")

    def _fake_run(cmd, **kwargs):
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        raise OSError("no systemd")

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    postup_hooks.enable_linger({}, {})  # must not raise


def test_enable_linger_getuser_keyerror_does_not_raise(monkeypatch):
    """getpass.getuser() falls back to pwd.getpwuid(os.getuid()) when
    USER/LOGNAME/LNAME/USERNAME are all unset, which raises KeyError (<=3.12)
    or OSError (3.13+) for a uid with no passwd entry -- e.g. an LDAP/NSS
    user under a stripped-env systemd/cron context. That must be caught
    here, not propagate through the post-up hook and abort the deploy after
    `up -d` already succeeded."""
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")

    def _raise_keyerror():
        raise KeyError("getpwuid(): uid not found: 1234")

    monkeypatch.setattr(postup_hooks.getpass, "getuser", _raise_keyerror)
    calls = []
    monkeypatch.setattr(postup_hooks.subprocess, "run", lambda *a, **k: calls.append(a))

    postup_hooks.enable_linger({}, {})  # must not raise

    assert calls == []  # no loginctl call was ever attempted


def test_enable_linger_status_check_timeout_still_attempts_enable(monkeypatch):
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(postup_hooks.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            raise postup_hooks.subprocess.TimeoutExpired(cmd=cmd, timeout=10)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    postup_hooks.enable_linger({}, {})  # must not raise

    assert calls[-1] == ["loginctl", "enable-linger", "deployuser"]


# ---------------------------------------------------------------------------
# warn_if_web_stack_unreachable -- advisory post-up host-reachability probe
# (the Docker Desktop network_mode:host trap: healthy stack, unreachable host).
# ---------------------------------------------------------------------------

_PROBE_CONFIG = {"modules": {"web_terminals": {"enabled": True, "nginx_port": 9080}}}


def test_web_stack_reachable_no_warning(monkeypatch, caplog):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", lambda url, timeout: _Resp())

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "not reachable" not in caplog.text


def test_web_stack_unreachable_warns_with_docker_desktop_hint(monkeypatch, caplog):
    def _refuse(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(postup_hooks.sys, "platform", "darwin")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "http://127.0.0.1:9080/" in caplog.text
    assert "Enable host networking" in caplog.text


def test_web_stack_unreachable_on_linux_warns_without_desktop_hint(monkeypatch, caplog):
    def _refuse(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(postup_hooks.sys, "platform", "linux")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "not reachable" in caplog.text
    assert "Enable host networking" not in caplog.text


# ---------------------------------------------------------------------------
# Docker Desktop stale-forwarder self-heal.
#
# Docker Desktop's host-network forwarder registers a VM-side listen socket by
# WATCHING for the listen event. A container that came back up via
# `restart: unless-stopped` while Docker Desktop was still starting (an update,
# a reboot, an Apply & Restart) opens its socket unobserved and stays invisible
# to the host forever -- and because its compose definition never changed,
# every later `up -d` leaves it `Running` and reconciles nothing, so the deploy
# can never fix itself. Bouncing the container re-opens the socket while the
# forwarder is watching. That is the common case whenever this probe fails on
# Docker Desktop, so the restart is tried BEFORE blaming the opt-in setting.
# ---------------------------------------------------------------------------


def _refusing_urlopen(succeed_after: int, calls: list[int]):
    """urlopen stub that refuses the first ``succeed_after`` calls, then answers."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _urlopen(url, timeout):
        calls.append(1)
        if len(calls) <= succeed_after:
            raise OSError("connection refused")
        return _Resp()

    return _urlopen


def test_docker_desktop_unreachable_self_heals_via_restart(monkeypatch, caplog):
    """A stale forwarder registration is repaired by a restart, with no warning."""
    probes: list[int] = []
    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _refusing_urlopen(2, probes))
    monkeypatch.setattr(postup_hooks.sys, "platform", "darwin")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])

    ran: list[list[str]] = []
    monkeypatch.setattr(
        postup_hooks.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd) or _FakeCompletedProcess(0),
    )

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(
            _PROBE_CONFIG,
            attempts=2,
            delay=0,
            web_cmd=["docker", "compose", "-f", "docker-compose.web.yml"],
            run_env={},
        )

    assert ran == [["docker", "compose", "-f", "docker-compose.web.yml", "restart"]]
    assert "not reachable" not in caplog.text
    assert "Enable host networking" not in caplog.text


def test_docker_desktop_warns_only_after_restart_fails_to_help(monkeypatch, caplog):
    """When the bounce does not help, the setting really is the likely cause."""
    probes: list[int] = []
    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _refusing_urlopen(999, probes))
    monkeypatch.setattr(postup_hooks.sys, "platform", "darwin")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])

    ran: list[list[str]] = []
    monkeypatch.setattr(
        postup_hooks.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd) or _FakeCompletedProcess(0),
    )

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(
            _PROBE_CONFIG, attempts=2, delay=0, web_cmd=["docker", "compose"], run_env={}
        )

    assert ran == [["docker", "compose", "restart"]]
    assert "not reachable" in caplog.text
    assert "Enable host networking" in caplog.text


def test_self_heal_never_fires_on_linux(monkeypatch, caplog):
    """Linux host networking is real -- an unreachable port means something else."""
    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _refusing_urlopen(999, []))
    monkeypatch.setattr(postup_hooks.sys, "platform", "linux")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])

    ran: list[list[str]] = []
    monkeypatch.setattr(
        postup_hooks.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd) or _FakeCompletedProcess(0),
    )

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(
            _PROBE_CONFIG, attempts=2, delay=0, web_cmd=["docker", "compose"], run_env={}
        )

    assert ran == []
    assert "not reachable" in caplog.text


def test_self_heal_skipped_when_caller_supplies_no_compose_cmd(monkeypatch, caplog):
    """Lifecycle callers that never had a web_cmd keep the old warn-only behaviour."""
    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _refusing_urlopen(999, []))
    monkeypatch.setattr(postup_hooks.sys, "platform", "darwin")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])

    ran: list[list[str]] = []
    monkeypatch.setattr(
        postup_hooks.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd) or _FakeCompletedProcess(0),
    )

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert ran == []
    assert "Enable host networking" in caplog.text


def test_web_stack_http_error_counts_as_reachable(monkeypatch, caplog):
    def _http_error(url, timeout):
        raise postup_hooks.urllib.error.HTTPError(url, 502, "Bad Gateway", None, None)

    monkeypatch.setattr(postup_hooks.urllib.request, "urlopen", _http_error)

    with caplog.at_level("WARNING"):
        postup_hooks.warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "not reachable" not in caplog.text


# ---------------------------------------------------------------------------
# run_verify_script -- advisory post-up smoke check in isolation
# ---------------------------------------------------------------------------


def test_run_verify_script_skips_silently_when_absent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(postup_hooks.subprocess, "run", lambda *a, **k: calls.append(a))

    postup_hooks.run_verify_script(str(tmp_path), {})

    assert calls == []


def test_run_verify_script_runs_via_bash_with_cwd_and_env(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    verify_path = scripts_dir / "verify.sh"
    verify_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(postup_hooks.subprocess, "run", _fake_run)

    run_env = {"COMPOSE_PROJECT_NAME": "test"}
    postup_hooks.run_verify_script(str(tmp_path), run_env)

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == ["bash", str(verify_path)]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"] == run_env


def test_run_verify_script_nonzero_exit_does_not_raise(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")

    monkeypatch.setattr(
        postup_hooks.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="boom"),
    )

    postup_hooks.run_verify_script(str(tmp_path), {})  # must not raise


def test_run_verify_script_oserror_does_not_raise(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    def _raise(*a, **k):
        raise OSError("no bash")

    monkeypatch.setattr(postup_hooks.subprocess, "run", _raise)

    postup_hooks.run_verify_script(str(tmp_path), {})  # must not raise
