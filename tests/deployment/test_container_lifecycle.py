"""Unit tests for container lifecycle argv construction.

These stub out the compose-file preparation, runtime checks, and the actual
subprocess invocation, then assert on the argv that ``deploy_up`` would run —
the cheapest way to lock in flag behavior (notably ``--build`` under ``--dev``)
without a container runtime.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators and capture the compose argv.

    Runs in ``detached=True`` mode so the call lands on ``subprocess.run`` (which
    we capture) rather than ``os.execvpe`` (which would replace the process).
    """
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd
        captured["env"] = env

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


def test_deploy_up_dev_mode_ups_no_build(captured_argv, tmp_path):
    """--dev builds in a separate step, so the final `up` carries --no-build,
    never --build in the same invocation (see the Defect A split tests for the
    full build-then-up assertion)."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=True)
    assert "up" in captured_argv["cmd"]
    assert "--no-build" in captured_argv["cmd"]
    assert "--build" not in captured_argv["cmd"]


def test_deploy_up_non_dev_omits_build(captured_argv, tmp_path):
    """Non-dev leaves a plain `up` (neither --build nor --no-build) so compose's
    implicit build-on-up still covers a build-only service with no upstream tag."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)
    assert "--build" not in captured_argv["cmd"]
    assert "--no-build" not in captured_argv["cmd"]
    assert "up" in captured_argv["cmd"]


# ---------------------------------------------------------------------------
# Dispatch token auto-generation (fail-closed auth)
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_token_env(monkeypatch):
    """Ensure the dispatch token vars are unset in the process env."""
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    p = tmp_path / ".env"
    return parse_dotenv_file(p) if p.is_file() else {}


def test_deploy_up_generates_tokens_when_unset(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("EVENT_DISPATCHER_TOKEN") and env["EVENT_DISPATCHER_TOKEN"] != "dev-token"
    assert env.get("DISPATCH_WORKER_TOKEN") and env["DISPATCH_WORKER_TOKEN"] != "dev-token"
    # token_urlsafe(32) → ~43 url-safe chars
    assert len(env["EVENT_DISPATCHER_TOKEN"]) >= 40


def test_token_generation_is_idempotent(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["EVENT_DISPATCHER_TOKEN"] == second["EVENT_DISPATCHER_TOKEN"]
    assert first["DISPATCH_WORKER_TOKEN"] == second["DISPATCH_WORKER_TOKEN"]
    # No duplicate keys appended on the second run.
    text = (tmp_path / ".env").read_text()
    assert text.count("EVENT_DISPATCHER_TOKEN=") == 1
    assert text.count("DISPATCH_WORKER_TOKEN=") == 1


def test_existing_env_token_is_preserved(captured_argv, _clean_token_env, tmp_path):
    (tmp_path / ".env").write_text("EVENT_DISPATCHER_TOKEN=my-real-token\n", encoding="utf-8")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["EVENT_DISPATCHER_TOKEN"] == "my-real-token"  # untouched
    assert env.get("DISPATCH_WORKER_TOKEN")  # the missing one was generated


def test_process_env_token_not_written_to_dotenv(captured_argv, monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "from-shell")
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    # A token resolvable from the process env is not duplicated into .env.
    assert "EVENT_DISPATCHER_TOKEN" not in env
    assert env.get("DISPATCH_WORKER_TOKEN")


def test_non_dispatch_deploy_generates_no_tokens(monkeypatch, _clean_token_env, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["mock"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert not (tmp_path / ".env").exists()


def test_expose_refuses_empty_token(captured_argv, monkeypatch, tmp_path):
    # A token explicitly set empty must not be auto-overwritten, and --expose must
    # refuse rather than bind a fail-open server to 0.0.0.0.
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "")
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="refusing to --expose"):
        container_lifecycle.deploy_up(
            str(tmp_path / "config.yml"), detached=True, expose_network=True
        )


# ---------------------------------------------------------------------------
# Web-terminal reconcile (osprey deploy up, modules.web_terminals.enabled)
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_web_runs(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a web-terminals-enabled deploy.

    Captures every ``subprocess.run`` invocation (argv + env) in order, so
    tests can inspect both the ``pull`` and the ``up -d`` calls the web
    reconcile path issues. ``write_web_terminal_artifacts`` is stubbed out —
    its own rendering is covered by ``tests/deployment/web_terminals/``, not
    here — but still records that it was called with the config.

    Defaults to registry mode (no ``image_source`` key), so a
    ``.env.production`` marker is pre-written to ``tmp_path``:
    ``ensure_env_production``'s registry-mode branch only exists-checks (see
    its own tests), so without this every test using this fixture would hit
    its "not found" RuntimeError before ever reaching a compose call.
    """
    calls: list[dict] = []
    written: list = []

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {
                "deployed_services": [],
                "modules": {"web_terminals": {"enabled": True}},
            },
            [],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_write_artifacts(config, dest_dir="."):
        written.append(config)
        return []

    monkeypatch.setattr(container_lifecycle, "write_web_terminal_artifacts", _fake_write_artifacts)

    def _fake_run(cmd, env=None, check=False):
        calls.append({"cmd": list(cmd), "env": env})

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return {"calls": calls, "written": written}


def test_web_only_deploy_does_not_early_return(captured_web_runs, tmp_path):
    """Empty deployed_services + web_terminals.enabled must still reconcile."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert captured_web_runs["written"], "write_web_terminal_artifacts was never called"
    assert captured_web_runs["calls"], "no compose commands were run"


def test_web_deploy_writes_artifacts_and_includes_web_compose_file(captured_web_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    up_calls = [c for c in captured_web_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 1
    up_cmd = up_calls[0]["cmd"]
    assert "-f" in up_cmd
    assert "docker-compose.web.yml" in up_cmd


def test_web_deploy_always_runs_detached(captured_web_runs, tmp_path):
    """Even with detached=False, the web path never execvpe's — it always
    lands on subprocess.run with -d, since compose up (non-detached) would
    replace the process and the post-up hook could never run."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    up_calls = [c for c in captured_web_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 1
    assert "-d" in up_calls[0]["cmd"]


def test_web_deploy_pins_compose_project_name(captured_web_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    for call in captured_web_runs["calls"]:
        assert call["env"] is not None
        assert "COMPOSE_PROJECT_NAME" in call["env"]


def test_web_deploy_idempotent_pull_then_up_no_force_recreate(captured_web_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in captured_web_runs["calls"]]
    assert any("pull" in cmd for cmd in cmds)
    assert any("up" in cmd and "-d" in cmd for cmd in cmds)
    for cmd in cmds:
        assert "--force-recreate" not in cmd


def test_web_deploy_no_wildcard_or_prune_flags(captured_web_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    for call in captured_web_runs["calls"]:
        cmd = call["cmd"]
        assert "prune" not in cmd
        assert "-a" not in cmd
        assert not any(arg == "--all" for arg in cmd)


def test_services_only_deploy_is_unchanged(captured_argv, tmp_path):
    """No web_terminals.enabled -> the pre-existing services path is untouched:
    still a single subprocess.run/up invocation, no -f docker-compose.web.yml,
    no pull step."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert "docker-compose.web.yml" not in captured_argv["cmd"]
    assert "pull" not in captured_argv["cmd"]
    assert "up" in captured_argv["cmd"]


# ---------------------------------------------------------------------------
# _web_terminals_enabled — null-stanza defense (review fix #1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"modules": {}},
        {"modules": None},
        {"modules": {"web_terminals": None}},
        {"modules": {"web_terminals": {"enabled": False}}},
    ],
)
def test_web_terminals_enabled_treats_null_stanza_as_disabled(config):
    """A present-but-null `modules` or `modules.web_terminals` stanza (e.g. a
    bare `web_terminals:` key in YAML, which parses to None) must read as
    disabled, not raise — matching lint's own _as_dict coercion."""
    assert container_lifecycle._web_terminals_enabled(config) is False


def test_web_terminals_enabled_true_when_set():
    config = {"modules": {"web_terminals": {"enabled": True}}}
    assert container_lifecycle._web_terminals_enabled(config) is True


def test_deploy_up_does_not_crash_on_null_web_terminals_stanza(monkeypatch, tmp_path):
    """A null modules.web_terminals stanza + empty deployed_services must hit
    the ordinary "nothing to deploy" early return, not crash."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"deployed_services": [], "modules": {"web_terminals": None}},
            [],
        ),
    )
    # No collaborator should even be reached past the early return.
    monkeypatch.setattr(
        container_lifecycle,
        "verify_runtime_is_running",
        lambda config: pytest.fail("should have early-returned"),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


# ---------------------------------------------------------------------------
# Web reconcile dev_mode --build (review fix #2)
#
# --build only ever belongs on the BACKEND SERVICES invocation (it rebuilds a
# stale cached image tag like osprey-dispatch:local); the web stack's images
# (nginx:*-alpine, <registry>/web-terminal:latest) have no `build:` block, so
# it never gets --build regardless of dev_mode. A web-terminals-ONLY deploy
# (no services invocation at all) therefore never emits --build anywhere.
# ---------------------------------------------------------------------------


def test_web_only_deploy_dev_mode_never_adds_build(captured_web_runs, tmp_path):
    """No backend services -> no services invocation -> nowhere for --build to
    land, even under --dev."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=True)

    up_calls = [c for c in captured_web_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 1
    assert "--build" not in up_calls[0]["cmd"]
    assert "-d" in up_calls[0]["cmd"]


def test_web_only_deploy_non_dev_mode_omits_build(captured_web_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=False)

    up_calls = [c for c in captured_web_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 1
    assert "--build" not in up_calls[0]["cmd"]


# ---------------------------------------------------------------------------
# Combined services + web_terminals deploy (review fix #3)
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_combined_runs(monkeypatch, tmp_path):
    """Both a backend service (event_dispatcher) and web_terminals enabled.

    Tracks subprocess.run calls plus how many times _build_project_image and
    _ensure_service_tokens fire, to prove the combined deploy does one
    detached reconcile rather than double-running the shared prelude.
    """
    calls: list[dict] = []
    build_calls: list[dict] = []
    token_calls: list[dict] = []

    monkeypatch.chdir(tmp_path)
    # Registry mode (default) -- pre-write .env.production so
    # ensure_env_production's exists-check passes (see captured_web_runs).
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {
                "deployed_services": ["event_dispatcher"],
                "modules": {"web_terminals": {"enabled": True}},
            },
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "write_web_terminal_artifacts", lambda config, dest_dir=".": []
    )

    def _fake_build(config, dev_mode, env):
        build_calls.append({"config": config, "dev_mode": dev_mode})

    monkeypatch.setattr(container_lifecycle, "_build_project_image", _fake_build)

    def _fake_tokens(config, expose_network, env_path=None):
        token_calls.append({"config": config})

    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", _fake_tokens)

    def _fake_run(cmd, env=None, check=False):
        calls.append({"cmd": list(cmd), "env": env})

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return {"calls": calls, "build_calls": build_calls, "token_calls": token_calls}


def test_combined_services_and_web_deploy_two_detached_up_calls(captured_combined_runs, tmp_path):
    """Real-daemon regression guard (compose path-resolution bug):

    Compose resolves every relative path in EVERY merged `-f` file against
    the directory of the FIRST `-f` file. compose_files (build/services/...)
    and docker-compose.web.yml (project root) are written to resolve against
    two DIFFERENT directories, so they must never be merged into one `-f ...
    -f docker-compose.web.yml` argv -- a real `osprey deploy up` with both
    enabled failed immediately with "env file .../build/services/
    .env.production not found" until this was split into two invocations.
    """
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    up_calls = [c for c in captured_combined_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 2

    services_up = [c["cmd"] for c in up_calls if "docker-compose.yml" in c["cmd"]]
    web_up = [c["cmd"] for c in up_calls if "docker-compose.web.yml" in c["cmd"]]
    assert len(services_up) == 1
    assert len(web_up) == 1

    # The services and web compose files must never appear together in one
    # argv -- that merge is exactly what broke path resolution.
    assert "docker-compose.web.yml" not in services_up[0]
    assert "docker-compose.yml" not in web_up[0]

    for cmd in (services_up[0], web_up[0]):
        assert "-d" in cmd


def test_combined_services_up_gets_dev_build_web_up_never_does(captured_combined_runs, tmp_path):
    """Under --dev the services stack builds in its OWN step then `up --no-build`
    (never `up --build` in one call, per Defect A); the web stack never builds."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=True)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"]]
    up_calls = [c for c in cmds if "up" in c]
    services_up = next(c for c in up_calls if "docker-compose.yml" in c)
    web_up = next(c for c in up_calls if "docker-compose.web.yml" in c)

    # A standalone services `build` ran (services compose file, no `up`).
    services_build = [c for c in cmds if c[-1] == "build" and "docker-compose.yml" in c]
    assert len(services_build) == 1

    # No `up --build` anywhere; the services `up` is explicitly --no-build.
    assert not any("up" in c and "--build" in c for c in cmds)
    assert "--no-build" in services_up
    assert "--build" not in web_up
    assert "--no-build" not in web_up


def test_combined_services_and_web_deploy_build_and_tokens_called_once(
    captured_combined_runs, tmp_path
):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(captured_combined_runs["build_calls"]) == 1
    assert len(captured_combined_runs["token_calls"]) == 1


def test_web_only_deploy_never_runs_a_services_up(captured_web_runs, tmp_path):
    """No deployed_services -> no services `up` invocation at all (compose
    `up` on the network-only top-level file with zero services fails
    outright with "no service selected")."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    up_calls = [c["cmd"] for c in captured_web_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 1
    assert "docker-compose.web.yml" in up_calls[0]


def test_combined_deploy_services_up_never_pulls(captured_combined_runs, tmp_path):
    """Only the web stack's images are always registry-hosted; a deployed
    service may declare only a `build:` block with no published upstream tag,
    and `compose pull` hard-fails on that -- so the services invocation must
    never carry a `pull` step (matching the plain non-web path, which never
    pulls either)."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    pull_calls = [c["cmd"] for c in captured_combined_runs["calls"] if "pull" in c["cmd"]]
    assert len(pull_calls) == 1
    assert "docker-compose.web.yml" in pull_calls[0]
    assert "docker-compose.yml" not in pull_calls[0]


# ---------------------------------------------------------------------------
# _enable_linger (task 2.3) -- rootless-podman persistence via loginctl
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_enable_linger_skips_on_docker_runtime(monkeypatch):
    """Docker has no per-user systemd session -- linger never applies there."""
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle.shutil,
        "which",
        lambda name: pytest.fail("loginctl should not be probed for docker"),
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: calls.append(a))

    container_lifecycle._enable_linger({}, {})

    assert calls == []


def test_enable_linger_skips_when_loginctl_absent(monkeypatch):
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: calls.append(a))

    container_lifecycle._enable_linger({}, {})

    assert calls == []


def test_enable_linger_noop_when_already_enabled(monkeypatch):
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(container_lifecycle.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd == ["loginctl", "show-user", "deployuser", "--property=Linger"]
        return _FakeCompletedProcess(returncode=0, stdout="Linger=yes\n")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle._enable_linger({}, {})

    # Only the status check ran -- enable-linger is never invoked once we
    # already know it's on, so a no-op deploy stays quiet.
    assert len(calls) == 1


def test_enable_linger_enables_when_not_yet_enabled(monkeypatch):
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(container_lifecycle.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle._enable_linger({}, {})

    assert calls == [
        ["loginctl", "show-user", "deployuser", "--property=Linger"],
        ["loginctl", "enable-linger", "deployuser"],
    ]


def test_enable_linger_enable_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(container_lifecycle.getpass, "getuser", lambda: "deployuser")

    def _fake_run(cmd, **kwargs):
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="Permission denied")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle._enable_linger({}, {})  # must not raise


def test_enable_linger_status_check_error_still_attempts_enable(monkeypatch):
    """A broken status check (loginctl show-user itself errors) must not
    prevent the enable attempt -- only a confirmed already-enabled state
    short-circuits it."""
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(container_lifecycle.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            raise OSError("boom")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle._enable_linger({}, {})  # must not raise

    assert calls[-1] == ["loginctl", "enable-linger", "deployuser"]


def test_enable_linger_enable_call_error_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(container_lifecycle.getpass, "getuser", lambda: "deployuser")

    def _fake_run(cmd, **kwargs):
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        raise OSError("no systemd")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle._enable_linger({}, {})  # must not raise


def test_enable_linger_getuser_keyerror_does_not_raise(monkeypatch):
    """getpass.getuser() falls back to pwd.getpwuid(os.getuid()) when
    USER/LOGNAME/LNAME/USERNAME are all unset, which raises KeyError (<=3.12)
    or OSError (3.13+) for a uid with no passwd entry -- e.g. an LDAP/NSS
    user under a stripped-env systemd/cron context. That must be caught
    here, not propagate through the post-up hook and abort the deploy after
    `up -d` already succeeded."""
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")

    def _raise_keyerror():
        raise KeyError("getpwuid(): uid not found: 1234")

    monkeypatch.setattr(container_lifecycle.getpass, "getuser", _raise_keyerror)
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: calls.append(a))

    container_lifecycle._enable_linger({}, {})  # must not raise

    assert calls == []  # no loginctl call was ever attempted


def test_enable_linger_status_check_timeout_still_attempts_enable(monkeypatch):
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(container_lifecycle.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            raise container_lifecycle.subprocess.TimeoutExpired(cmd=cmd, timeout=10)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle._enable_linger({}, {})  # must not raise

    assert calls[-1] == ["loginctl", "enable-linger", "deployuser"]


def test_web_deploy_docker_runtime_linger_adds_no_subprocess_calls(captured_web_runs, tmp_path):
    """captured_web_runs defaults to docker -- the post-up hook's linger step
    must add zero subprocess calls beyond the ordinary pull + up."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(captured_web_runs["calls"]) == 2


# ---------------------------------------------------------------------------
# ensure_env_production (task 3.1) -- module-conditional CI-subset generator
# for local-mode web-terminal deploys.
# ---------------------------------------------------------------------------


def _write_dotenv(path, values: dict) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


# A facility config with every relevant module enabled, plus every excluded
# secret's config-declared name present too -- this is the fixture the
# security spec (the exclusion list) gets unit-tested against.
_FULL_CONFIG = {
    "facility": {"name": "Test Facility", "prefix": "test", "timezone": "America/Los_Angeles"},
    "llm": {"provider": "cborg", "api_key_env_var": "CBORG_API_KEY"},
    "ci": {"provider": "gitlab", "token_env_var": "TEST_CI_TOKEN"},
    "registry": {
        "url": "registry.example.org/test",
        "token_env_var": "TEST_REGISTRY_TOKEN",
        "external_projects": [
            {
                "name": "beam-viewer",
                "url": "registry.example.org/beam-viewer",
                "image": "beam-viewer:latest",
                "token_env_var": "BEAM_VIEWER_DEPLOY_TOKEN",
            }
        ],
    },
    "modules": {
        "web_terminals": {"enabled": True, "image_source": "local"},
        "olog": {
            "enabled": True,
            "username_env_var": "OLOG_USERNAME",
            "password_env_var": "OLOG_PASSWORD",
        },
        "wiki_search": {"enabled": True, "token_env_var": "CONFLUENCE_ACCESS_TOKEN"},
        "event_dispatcher": {
            "enabled": True,
            "token_env_var": "EVENT_DISPATCHER_TOKEN",
            "sidecar_token_env_var": "DISPATCH_SIDECAR_TOKEN",
        },
        "ariel": {
            "enabled": True,
            "dsn": "postgresql://ariel:ariel@ariel-postgres:5432/ariel",
        },
    },
}

# Every secret .env.production must NEVER contain, keyed by the config path
# that names it -- the exclusion list is the security spec for this task.
_EXCLUDED_ENV = {
    "TEST_CI_TOKEN": "ci-secret",
    "TEST_REGISTRY_TOKEN": "registry-secret",
    "BEAM_VIEWER_DEPLOY_TOKEN": "external-project-secret",
    "DISPATCH_SIDECAR_TOKEN": "sidecar-secret",
}

_INCLUDED_ENV = {
    "CBORG_API_KEY": "llm-secret",
    "OLOG_USERNAME": "olog-user",
    "OLOG_PASSWORD": "olog-pass",
    "CONFLUENCE_ACCESS_TOKEN": "wiki-secret",
    "EVENT_DISPATCHER_TOKEN": "dispatcher-secret",
}


def test_env_production_present_returned_as_is(tmp_path):
    marker = "# operator-authored, do not touch\nFOO=bar\n"
    (tmp_path / ".env.production").write_text(marker, encoding="utf-8")

    result = container_lifecycle.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result == tmp_path / ".env.production"
    assert result.read_text(encoding="utf-8") == marker


def test_env_production_present_in_registry_mode_returned_as_is(tmp_path):
    marker = "FOO=bar\n"
    (tmp_path / ".env.production").write_text(marker, encoding="utf-8")
    config = {**_FULL_CONFIG, "modules": {**_FULL_CONFIG["modules"], "web_terminals": {}}}

    result = container_lifecycle.ensure_env_production(config, tmp_path)

    assert result.read_text(encoding="utf-8") == marker


def test_env_production_neither_present_raises_actionably(tmp_path):
    with pytest.raises(RuntimeError, match=r"\.env\.production.*\.env"):
        container_lifecycle.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert not (tmp_path / ".env.production").exists()


def test_env_production_registry_mode_never_generates_even_with_env_present(tmp_path):
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})
    config = {**_FULL_CONFIG, "modules": {**_FULL_CONFIG["modules"], "web_terminals": {}}}

    with pytest.raises(RuntimeError, match="Registry-mode"):
        container_lifecycle.ensure_env_production(config, tmp_path)

    assert not (tmp_path / ".env.production").exists()


def test_env_production_local_mode_generates_from_env(tmp_path):
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = container_lifecycle.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result == tmp_path / ".env.production"
    generated = container_lifecycle.parse_dotenv_file(result)

    # Included: llm key, module-gated olog/wiki/dispatcher, ARIEL_DSN, TZ.
    assert generated["CBORG_API_KEY"] == "llm-secret"
    assert generated["OLOG_USERNAME"] == "olog-user"
    assert generated["OLOG_PASSWORD"] == "olog-pass"
    assert generated["CONFLUENCE_ACCESS_TOKEN"] == "wiki-secret"
    assert generated["EVENT_DISPATCHER_TOKEN"] == "dispatcher-secret"
    assert generated["ARIEL_DSN"] == "postgresql://ariel:ariel@ariel-postgres:5432/ariel"
    assert generated["TZ"] == "America/Los_Angeles"


def test_env_production_never_includes_excluded_secrets(tmp_path):
    """The security spec: registry token, sidecar token, and external-project
    tokens must never appear in the generated file -- neither their key nor
    their value, even though the source .env contains all of them."""
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = container_lifecycle.ensure_env_production(_FULL_CONFIG, tmp_path)

    generated = container_lifecycle.parse_dotenv_file(result)
    raw_text = result.read_text(encoding="utf-8")

    for excluded_key, excluded_value in _EXCLUDED_ENV.items():
        assert excluded_key not in generated
        assert excluded_key not in raw_text
        assert excluded_value not in raw_text

    # And the CI/registry token vars named in config are also never copied,
    # confirming the omission is by construction, not incidental.
    assert "TEST_CI_TOKEN" not in generated
    assert "TEST_REGISTRY_TOKEN" not in generated


def test_env_production_generated_file_is_mode_0600(tmp_path):
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)

    result = container_lifecycle.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert (result.stat().st_mode & 0o777) == 0o600


def test_env_production_created_with_restrictive_mode_atomically(monkeypatch, tmp_path):
    """Regression guard for the write-then-chmod umask race: the file must be
    opened with mode 0600 from the very first os.open call (O_CREAT with an
    explicit restrictive mode), never created at the process umask (e.g.
    0644) and tightened only after every secret has already been written."""
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)

    captured: dict = {}
    real_open = os.open

    def _spy_open(path, flags, mode=0o777):
        if str(path).endswith(".env.production"):
            captured["flags"] = flags
            captured["mode"] = mode
        return real_open(path, flags, mode)

    monkeypatch.setattr(container_lifecycle.os, "open", _spy_open)

    container_lifecycle.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert captured, "os.open was never called for .env.production"
    assert captured["mode"] == 0o600
    assert captured["flags"] & os.O_CREAT


def test_env_production_module_disabled_omits_its_vars(tmp_path):
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)
    config = {
        "facility": {"timezone": "UTC"},
        "llm": {"api_key_env_var": "CBORG_API_KEY"},
        "modules": {
            "web_terminals": {"image_source": "local"},
            "olog": {"enabled": False, "username_env_var": "OLOG_USERNAME"},
            "wiki_search": {"enabled": False, "token_env_var": "CONFLUENCE_ACCESS_TOKEN"},
            "event_dispatcher": {"enabled": False, "token_env_var": "EVENT_DISPATCHER_TOKEN"},
            "ariel": {"enabled": False, "dsn": "postgresql://ariel:ariel@ariel-postgres/ariel"},
        },
    }

    result = container_lifecycle.ensure_env_production(config, tmp_path)
    generated = container_lifecycle.parse_dotenv_file(result)

    assert generated == {"CBORG_API_KEY": "llm-secret", "TZ": "UTC"}


def test_env_production_missing_var_in_env_is_skipped_not_fabricated(tmp_path):
    # .env exists but doesn't set the olog vars -- never fabricated.
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": "llm-secret"})
    config = {
        "facility": {},
        "llm": {"api_key_env_var": "CBORG_API_KEY"},
        "modules": {
            "web_terminals": {"image_source": "local"},
            "olog": {
                "enabled": True,
                "username_env_var": "OLOG_USERNAME",
                "password_env_var": "OLOG_PASSWORD",
            },
        },
    }

    result = container_lifecycle.ensure_env_production(config, tmp_path)
    generated = container_lifecycle.parse_dotenv_file(result)

    assert generated == {"CBORG_API_KEY": "llm-secret", "TZ": "UTC"}


def test_env_production_local_mode_defaults_when_image_source_absent_is_registry(tmp_path):
    """No modules.web_terminals.image_source at all -> defaults to registry
    (fail-closed), so an absent .env.production still raises rather than
    silently generating from a stray .env."""
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)
    config = {"facility": {}, "llm": {}, "modules": {"web_terminals": {}}}

    with pytest.raises(RuntimeError, match="Registry-mode"):
        container_lifecycle.ensure_env_production(config, tmp_path)


# ---------------------------------------------------------------------------
# build_persona_images (Task 3.2) -- local-mode per-persona image builder
# ---------------------------------------------------------------------------


def _make_persona_project(tmp_path, name, cli_version=None):
    """Create a minimal persona project dir with a Dockerfile + config.yml."""
    project_dir = tmp_path / name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    if cli_version is not None:
        (project_dir / "config.yml").write_text(
            f"claude_code:\n  cli_version: {cli_version!r}\n", encoding="utf-8"
        )
    else:
        (project_dir / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    return str(project_dir)


@pytest.fixture
def _no_dev_wheel_staging(monkeypatch):
    """Stub out the dev-wheel staging collaborator (its own coverage lives with
    _build_project_image's tests) so build_persona_images tests never touch a
    real wheel build."""
    monkeypatch.setattr(
        container_lifecycle, "_copy_local_framework_for_override", lambda project_root: None
    )


def test_build_persona_images_noop_in_registry_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: calls.append(a))
    config = {"modules": {"web_terminals": {"image_source": "registry"}}}

    container_lifecycle.build_persona_images(config, [{"persona": "ops"}], False, {})

    assert calls == []


def test_build_persona_images_local_without_catalog_raises(tmp_path):
    config = {"modules": {"web_terminals": {"image_source": "local"}}}

    with pytest.raises(ValueError, match="requires both"):
        container_lifecycle.build_persona_images(config, [], False, {})


def test_build_persona_images_local_without_default_persona_raises(tmp_path):
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "personas": {"ops": {"project": "ops-app", "project_path": str(tmp_path)}},
            }
        }
    }

    with pytest.raises(ValueError, match="requires both"):
        container_lifecycle.build_persona_images(config, [], False, {})


def test_build_persona_images_builds_each_referenced_persona_once(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    sci_path = _make_persona_project(tmp_path, "sci-app")

    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {"project": "ops-app", "project_path": ops_path},
                    "sci": {"project": "sci-app", "project_path": sci_path},
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "persona": "ops", "project": "ops-app"},
        {"name": "bob", "persona": "ops", "project": "ops-app"},  # shares ops -- must not rebuild
        {"name": "carol", "persona": "sci", "project": "sci-app"},
    ]

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle.build_persona_images(config, resolved_users, False, {})

    assert len(calls) == 2  # one build per DISTINCT persona, not per user

    ops_cmd = next(c for c in calls if "ops-app-ops:local" in c)
    sci_cmd = next(c for c in calls if "sci-app-sci:local" in c)

    assert ops_cmd[0] == "docker"
    assert "-f" in ops_cmd
    assert os.path.join(ops_path, "Dockerfile") == ops_cmd[ops_cmd.index("-f") + 1]
    assert ops_path == ops_cmd[-1]  # context is project_path
    assert "--label" in ops_cmd
    assert "com.osprey.project=myfacility" in ops_cmd

    assert "com.osprey.project=myfacility" in sci_cmd
    assert sci_path == sci_cmd[-1]


def test_build_persona_images_never_builds_zero_migration_entries(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """An entry with persona=None (no persona system in effect) is skipped --
    it never contributes a build unit, even in local mode."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "legacy", "persona": None, "project": "myfacility-assistant"}]

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle.build_persona_images(config, resolved_users, False, {})

    assert calls == []


def test_build_persona_images_includes_cli_version_from_persona_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version="2.1.99")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "CLAUDE_CLI_VERSION=2.1.99" in cmd


def test_build_persona_images_omits_cli_version_when_unset_in_persona_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """The persona's own config.yml has no claude_code.cli_version -- the
    build-arg must be omitted entirely (never falls back to the framework
    default the facility/dispatch-worker path uses)."""
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any(str(arg).startswith("CLAUDE_CLI_VERSION=") for arg in cmd)
    # The facility config's own claude_code.cli_version (if any) must never
    # leak into a persona build either -- there is none set here, but the
    # generic OSPREY_PIP_SPEC build-arg is still present.
    assert any(str(arg).startswith("OSPREY_PIP_SPEC=") for arg in cmd)


def test_build_persona_images_never_reads_facility_cli_version(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """A claude_code.cli_version set on the FACILITY config must never leak
    into a persona build -- only the persona's own project_path/config.yml is
    consulted."""
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "claude_code": {"cli_version": "9.9.9"},  # facility-level pin -- must be ignored
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any("9.9.9" in str(arg) for arg in cmd)


def test_build_persona_images_dev_mode_stages_and_cleans_wheel(monkeypatch, tmp_path):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")

    monkeypatch.setattr(container_lifecycle, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: None)

    container_lifecycle.build_persona_images(config, resolved_users, True, {})

    # Staged wheel must be cleaned up after the build so it can't poison a
    # later non-dev build's wheel-drop branch.
    assert list(Path(ops_path).glob("*.whl")) == []


def test_build_persona_images_no_referenced_personas_runs_no_build(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Local mode + catalog + default_persona configured, but resolved_users
    references no catalog entry (e.g. empty roster) -- no-op, no crash."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }

    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle.build_persona_images(config, [], False, {})

    assert calls == []


# ---------------------------------------------------------------------------
# _auto_render_missing_personas (Task 3.1) -- render a referenced persona's
# project on demand when its project_path directory is absent, BEFORE
# build_persona_images builds its image. Renders network-free (--skip-deps),
# never overwrites a complete (user-owned) render, and hard-errors on a
# partial render or a missing build_profile.
# ---------------------------------------------------------------------------


def _auto_render_config(tmp_path, **persona_overrides):
    """A local-mode config whose single persona 'ops' renders to <tmp_path>/ops-app.

    Defaults to a usable build_profile so the render path is exercised; pass
    ``build_profile=None`` to drop it.
    """
    persona = {
        "project": "ops-app",
        "project_path": str(tmp_path / "ops-app"),
        "build_profile": "control-assistant",
    }
    persona.update(persona_overrides)
    return {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": persona},
            }
        }
    }


_AUTO_RENDER_USERS = [{"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"}]


def test_auto_render_renders_when_project_path_missing(monkeypatch, tmp_path):
    """No directory at project_path -> exactly one `osprey build` render, argv
    verbatim: <project> --preset <build_profile> -o <parent(project_path)>
    --skip-deps (rendered into the parent so it lands AT project_path). The CLI
    is re-entered via the RUNNING interpreter (`python -m osprey`), never a
    bare `osprey` that PATH could resolve to a different install."""
    config = _auto_render_config(tmp_path)  # <tmp_path>/ops-app does not exist
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == [
        [
            sys.executable,
            "-m",
            "osprey",
            "build",
            "ops-app",
            "--preset",
            "control-assistant",
            "-o",
            str(tmp_path),
            "--skip-deps",
        ]
    ]


def test_auto_render_partial_render_raises(monkeypatch, tmp_path):
    """project_path exists but is missing its Dockerfile -> a partial render;
    raise (naming the dir) rather than silently rebuild over it."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    # Dockerfile deliberately absent -> partial render.
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    with pytest.raises(ValueError, match="partial render") as excinfo:
        container_lifecycle._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert str(project_path) in str(excinfo.value)
    assert "Dockerfile" in str(excinfo.value)
    assert calls == []  # never rendered over the partial tree


def test_auto_render_complete_render_is_noop(monkeypatch, tmp_path):
    """project_path exists with both config.yml and Dockerfile -> user-owned
    complete render; never overwrite it, run no `osprey build`."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    (project_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == []


def test_auto_render_missing_build_profile_raises(monkeypatch, tmp_path):
    """project_path absent (a render IS needed) but the catalog entry has no
    build_profile -> raise, since there's nothing to render from."""
    config = _auto_render_config(tmp_path, build_profile=None)
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    with pytest.raises(ValueError, match="build_profile"):
        container_lifecycle._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == []  # nothing rendered


def test_auto_render_renders_each_distinct_persona_once(monkeypatch, tmp_path):
    """Two users sharing a persona collapse to one render; a second, distinct
    persona renders separately -- one `osprey build` per DISTINCT persona."""
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {
                        "project": "ops-app",
                        "project_path": str(tmp_path / "ops-app"),
                        "build_profile": "control-assistant",
                    },
                    "sci": {
                        "project": "sci-app",
                        "project_path": str(tmp_path / "sci-app"),
                        "build_profile": "physicist",
                    },
                },
            }
        }
    }
    resolved_users = [
        {"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"},
        {"name": "bob", "index": 1, "persona": "ops", "project": "ops-app"},  # shares ops
        {"name": "carol", "index": 2, "persona": "sci", "project": "sci-app"},
    ]
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    container_lifecycle._auto_render_missing_personas(config, resolved_users, {})

    assert len(calls) == 2
    assert any("ops-app" in c and "control-assistant" in c for c in calls)
    assert any("sci-app" in c and "physicist" in c for c in calls)


def test_web_deploy_calls_enable_linger_in_post_up_hook(monkeypatch, tmp_path):
    """The post-up hook wires _enable_linger(config, run_env) -- the same
    COMPOSE_PROJECT_NAME-pinned env the compose calls around it use."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"deployed_services": [], "modules": {"web_terminals": {"enabled": True}}},
            [],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "write_web_terminal_artifacts", lambda config, dest_dir=".": []
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    linger_calls = []
    monkeypatch.setattr(
        container_lifecycle,
        "_enable_linger",
        lambda config, run_env: linger_calls.append(run_env),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(linger_calls) == 1
    assert "COMPOSE_PROJECT_NAME" in linger_calls[0]


# ---------------------------------------------------------------------------
# _deploy_up_web_terminals mode wiring (Task 3.3) -- image_source local vs.
# registry branching: ensure_env_production always runs first, build_persona_images
# is wired via resolve_personas(strict=True) in local mode only, and local mode
# never emits a `pull` argv (the pull-guard: `compose pull` hard-fails on a
# local-only tag).
# ---------------------------------------------------------------------------


def _web_terminals_config(image_source: str, **web_terminals_overrides) -> dict:
    """A minimal web-terminals-only facility config for the given image_source.

    Always carries a persona catalog + default_persona (satisfies local
    mode's own configuration requirement, matching build_persona_images'
    ValueError guard) even for registry-mode tests, so the same helper
    covers both branches without a second config shape.
    """
    web_terminals = {
        "enabled": True,
        "image_source": image_source,
        "default_persona": "ops",
        "personas": {
            "ops": {
                "project": "ops-app",
                "project_path": "/nonexistent/ops-app",
                "build_profile": "control-assistant",
            }
        },
    }
    web_terminals.update(web_terminals_overrides)
    return {
        "deployed_services": [],
        "facility": {"prefix": "test"},
        "registry": {"url": "registry.example.org/test"},
        "modules": {"web_terminals": web_terminals},
    }


@pytest.fixture
def _mode_wiring_collab(monkeypatch, tmp_path):
    """Collaborator stubs shared by the mode-wiring tests: chdir,
    verify_runtime_is_running, get_runtime_command, write_web_terminal_artifacts,
    and a captured subprocess.run that returns a 0-exit CompletedProcess stand-in
    (needed because _run_verify_script inspects .returncode on every call it
    makes, not just compose's). Deliberately does NOT pre-write .env.production
    or .env -- each test supplies exactly what its mode needs to exercise
    ensure_env_production's own branches.
    """
    calls: list[dict] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "write_web_terminal_artifacts", lambda config, dest_dir=".": []
    )
    # Auto-render is a separate concern with its own dedicated tests below;
    # keep it inert here so the mode-wiring tests exercise only the local/
    # registry step ordering, never a real `osprey build` subprocess.
    monkeypatch.setattr(
        container_lifecycle,
        "_auto_render_missing_personas",
        lambda config, resolved_users, env: None,
    )

    def _fake_run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "env": kwargs.get("env")})
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return calls


def test_local_mode_never_emits_a_pull_argv(monkeypatch, tmp_path, _mode_wiring_collab):
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")
    config = _web_terminals_config("local")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in _mode_wiring_collab]
    assert not any("pull" in cmd for cmd in cmds)
    assert any("up" in cmd and "-d" in cmd for cmd in cmds)
    # ensure_env_production generated .env.production from .env since neither
    # was present -- local mode's own branch, exercised end-to-end here.
    assert (tmp_path / ".env.production").is_file()


def test_registry_mode_still_pulls(monkeypatch, tmp_path, _mode_wiring_collab):
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in _mode_wiring_collab]
    assert any("pull" in cmd for cmd in cmds)
    assert any("up" in cmd and "-d" in cmd for cmd in cmds)


def test_registry_mode_raises_before_any_compose_call_when_env_production_missing(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """Neither .env.production nor .env present -- ensure_env_production raises
    its registry-mode "not found" error before compose ever runs."""
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    with pytest.raises(RuntimeError, match="not found"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert _mode_wiring_collab == []  # no compose subprocess ever ran


def test_local_mode_unresolvable_persona_raises_before_any_compose_call(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """A user referencing a persona absent from the catalog must raise via
    resolve_personas(strict=True) before build_persona_images or any compose
    call -- surfacing actionably instead of an opaque unbuilt-tag failure at
    `compose up` (the reviewer integration note from task 3.2)."""
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")
    config = _web_terminals_config(
        "local", users=[{"name": "alice", "index": 0, "persona": "no-such-persona"}]
    )
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    with pytest.raises(ValueError, match="no-such-persona"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert _mode_wiring_collab == []  # no compose subprocess ever ran


def test_local_mode_calls_ensure_env_production_then_auto_render_then_build_then_compose(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """The local-mode preflight order is load-bearing: ensure_env_production,
    THEN auto-render any missing persona project, THEN build its image, THEN
    compose. A spy on _auto_render_missing_personas (overriding the fixture's
    inert stub) proves the wiring line actually runs it -- and runs it BEFORE
    build_persona_images, which needs the rendered context to exist."""
    order: list[str] = []
    config = _web_terminals_config("local")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))
    monkeypatch.setattr(
        container_lifecycle,
        "ensure_env_production",
        lambda cfg, root: order.append("ensure_env_production"),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_auto_render_missing_personas",
        lambda cfg, resolved_users, env: order.append("auto_render"),
    )

    def _fake_build(cfg, resolved_users, dev_mode, env):
        order.append("build_persona_images")

    monkeypatch.setattr(container_lifecycle, "build_persona_images", _fake_build)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda *a, **k: order.append("compose") or _FakeCompletedProcess(returncode=0),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    # Exactly one compose call (web `up -d`; no deployed_services, no pull in
    # local mode) after all three preflight steps, in this order.
    assert order == ["ensure_env_production", "auto_render", "build_persona_images", "compose"]


def test_local_mode_passes_resolve_personas_output_to_build_persona_images(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """build_persona_images must receive resolve_personas(strict=True)'s own
    output, not some other user-list shape -- confirmed by asserting the
    resolved persona/project fields it actually threads through."""
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")
    config = _web_terminals_config(
        "local",
        users=[{"name": "alice", "index": 0}],  # no explicit persona -> falls back to "ops"
    )
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))
    # This test's roster is non-empty (unlike the other mode-wiring tests), so
    # seed_user_containers would actually attempt to seed -- stub it out, its
    # own coverage lives in tests/deployment/web_terminals/.
    monkeypatch.setattr(container_lifecycle, "seed_user_containers", lambda cfg, env=None: None)

    captured_users = []

    def _fake_build(cfg, resolved_users, dev_mode, env):
        captured_users.extend(resolved_users)

    monkeypatch.setattr(container_lifecycle, "build_persona_images", _fake_build)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(captured_users) == 1
    assert captured_users[0]["name"] == "alice"
    assert captured_users[0]["persona"] == "ops"
    assert captured_users[0]["project"] == "ops-app"


def test_registry_mode_never_calls_build_persona_images(monkeypatch, tmp_path, _mode_wiring_collab):
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    build_calls = []
    monkeypatch.setattr(
        container_lifecycle,
        "build_persona_images",
        lambda *a, **k: build_calls.append(a),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert build_calls == []


def test_registry_mode_never_calls_auto_render(monkeypatch, tmp_path, _mode_wiring_collab):
    """Auto-render is a local-mode-only step (registry mode pulls prebuilt
    images) -- it must never run on the registry path, mirroring the
    build_persona_images guard. A recording spy overrides the fixture's inert
    stub so a stray call would be caught, not swallowed."""
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    render_calls = []
    monkeypatch.setattr(
        container_lifecycle,
        "_auto_render_missing_personas",
        lambda *a, **k: render_calls.append(a),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert render_calls == []


def test_registry_mode_calls_ensure_env_production_before_pull_before_up(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    order: list[str] = []
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))
    monkeypatch.setattr(
        container_lifecycle,
        "ensure_env_production",
        lambda cfg, root: order.append("ensure_env_production"),
    )

    def _fake_run(cmd, **kwargs):
        order.append("pull" if "pull" in cmd else "up")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert order == ["ensure_env_production", "pull", "up"]


def test_post_up_hook_order_is_linger_then_seed_then_verify(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    order: list[str] = []
    monkeypatch.setattr(
        container_lifecycle, "_enable_linger", lambda cfg, run_env: order.append("linger")
    )
    monkeypatch.setattr(
        container_lifecycle,
        "seed_user_containers",
        lambda cfg, env=None: order.append("seed"),
    )
    monkeypatch.setattr(
        container_lifecycle, "_run_verify_script", lambda root, run_env: order.append("verify")
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert order == ["linger", "seed", "verify"]


def test_deploy_up_runs_verify_script_when_present_ignoring_exit_code(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """A nonzero verify.sh exit must not propagate out of `osprey deploy up` --
    advisory only, per the script's own convention and _run_verify_script's
    contract."""
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    verify_path = scripts_dir / "verify.sh"
    verify_path.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")

    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    def _fake_run(cmd, **kwargs):
        _mode_wiring_collab.append({"cmd": list(cmd), "env": kwargs.get("env")})
        is_verify_call = cmd[:1] == ["bash"]
        return _FakeCompletedProcess(returncode=1 if is_verify_call else 0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    # Must not raise, despite verify.sh exiting 1.
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    verify_calls = [c for c in _mode_wiring_collab if c["cmd"][:1] == ["bash"]]
    assert len(verify_calls) == 1
    assert verify_calls[0]["cmd"] == ["bash", str(verify_path)]


def test_deploy_up_skips_verify_script_silently_when_absent(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert not any(c["cmd"][:1] == ["bash"] for c in _mode_wiring_collab)


# ---------------------------------------------------------------------------
# _run_verify_script (Task 3.3) -- advisory post-up smoke check in isolation
# ---------------------------------------------------------------------------


def test_run_verify_script_skips_silently_when_absent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: calls.append(a))

    container_lifecycle._run_verify_script(str(tmp_path), {})

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

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    run_env = {"COMPOSE_PROJECT_NAME": "test"}
    container_lifecycle._run_verify_script(str(tmp_path), run_env)

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
        container_lifecycle.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="boom"),
    )

    container_lifecycle._run_verify_script(str(tmp_path), {})  # must not raise


def test_run_verify_script_oserror_does_not_raise(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    def _raise(*a, **k):
        raise OSError("no bash")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _raise)

    container_lifecycle._run_verify_script(str(tmp_path), {})  # must not raise


# ---------------------------------------------------------------------------
# Shared-disk preflight (task 3.7) -- ports retired deploy.sh step 2b: abort
# before any compose invocation if modules.shared_disk.host_path is
# configured but missing on this host.
# ---------------------------------------------------------------------------


def test_shared_disk_preflight_module_absent_is_noop():
    container_lifecycle._check_shared_disk_preflight({})  # must not raise


@pytest.mark.parametrize(
    "config",
    [
        {"modules": {"shared_disk": {"enabled": False, "host_path": "/does/not/exist"}}},
        {"modules": {"shared_disk": None}},
        {"modules": None},
    ],
)
def test_shared_disk_preflight_disabled_or_null_is_noop(config):
    container_lifecycle._check_shared_disk_preflight(config)  # must not raise


def test_shared_disk_preflight_enabled_without_host_path_is_noop():
    """enabled=True but no host_path configured -- nothing to check."""
    config = {"modules": {"shared_disk": {"enabled": True}}}
    container_lifecycle._check_shared_disk_preflight(config)  # must not raise


def test_shared_disk_preflight_existing_dir_passes(tmp_path):
    config = {"modules": {"shared_disk": {"enabled": True, "host_path": str(tmp_path)}}}
    container_lifecycle._check_shared_disk_preflight(config)  # must not raise


def test_shared_disk_preflight_missing_path_raises_actionably(tmp_path):
    missing = tmp_path / "no-such-mount"
    config = {"modules": {"shared_disk": {"enabled": True, "host_path": str(missing)}}}

    with pytest.raises(RuntimeError, match="does not exist on this server"):
        container_lifecycle._check_shared_disk_preflight(config)


def test_shared_disk_preflight_path_is_a_file_not_dir_raises(tmp_path):
    """A host_path that exists but is a file (not a directory) is also invalid --
    a bind mount needs a directory, matching the retired shell check's `[[ ! -d ]]`."""
    path_is_file = tmp_path / "not-a-directory"
    path_is_file.write_text("", encoding="utf-8")
    config = {"modules": {"shared_disk": {"enabled": True, "host_path": str(path_is_file)}}}

    with pytest.raises(RuntimeError, match="does not exist on this server"):
        container_lifecycle._check_shared_disk_preflight(config)


def test_deploy_up_raises_before_any_compose_call_when_shared_disk_missing(
    captured_argv, monkeypatch, tmp_path
):
    """Wired into deploy_up: a missing shared_disk host_path aborts before the
    plain services path reaches subprocess.run."""
    missing = tmp_path / "no-such-mount"
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {
                "deployed_services": ["event_dispatcher"],
                "modules": {"shared_disk": {"enabled": True, "host_path": str(missing)}},
            },
            ["docker-compose.yml"],
        ),
    )

    with pytest.raises(RuntimeError, match="does not exist on this server"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert "cmd" not in captured_argv


def test_web_deploy_raises_before_any_compose_call_when_shared_disk_missing(
    captured_web_runs, monkeypatch, tmp_path
):
    """Wired into deploy_up: a missing shared_disk host_path aborts before the
    web-terminals path (which also reaches compose via _deploy_up_web_terminals)."""
    missing = tmp_path / "no-such-mount"
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {
                "deployed_services": [],
                "modules": {
                    "web_terminals": {"enabled": True},
                    "shared_disk": {"enabled": True, "host_path": str(missing)},
                },
            },
            [],
        ),
    )

    with pytest.raises(RuntimeError, match="does not exist on this server"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)


# ---------------------------------------------------------------------------
# COMPOSE_PROJECT_NAME pinning on the plain (non-web) deploy paths
#
# The web path already pins it (see test_web_deploy_pins_compose_project_name).
# These lock in that deploy_up's plain branch, deploy_down, deploy_restart, and
# rebuild_deployment route their runtime env through runtime_env() too. Without
# the pin, compose derives the project from the first -f file's directory (the
# shared "services" project), so one deploy's up/down adopts and destroys a
# sibling deploy's containers and volumes.
# ---------------------------------------------------------------------------


def test_deploy_up_plain_pins_compose_project_name(captured_argv, tmp_path):
    """deploy_up's plain (non-web) branch runs compose under a pinned project."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert captured_argv["env"] is not None
    # captured_argv's config carries no project_name/project_root -> the
    # resolve_project_name fallback, but crucially it is PINNED, not inherited.
    assert captured_argv["env"]["COMPOSE_PROJECT_NAME"] == "unnamed-project"


def _mock_down_config(monkeypatch, project_name):
    """Wire deploy_down's config load to a fixed, normalized config dict."""
    monkeypatch.setattr(
        container_lifecycle,
        "ConfigBuilder",
        lambda p: types.SimpleNamespace(raw_config={"project_name": project_name}),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "normalize_facility_config",
        lambda raw: {"project_name": project_name, "deployed_services": ["event_dispatcher"]},
    )
    monkeypatch.setattr(
        "osprey.deployment.compose_generator.find_existing_compose_files",
        lambda *a, **k: ["docker-compose.yml"],
    )
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )


def test_deploy_down_pins_compose_project_name(monkeypatch, tmp_path):
    """deploy_down must target the same project it brought up — pinned, and via
    execvpe (env-carrying), not the bare-env execvp."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch, "myproj")
    captured: dict = {}
    # Guard BOTH exec variants: the fix flips execvp -> execvpe, and an
    # unpatched real execvp would replace the test process.
    monkeypatch.setattr(
        container_lifecycle.os, "execvp", lambda file, args: captured.update(args=args, env=None)
    )
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvpe",
        lambda file, args, env: captured.update(file=file, args=args, env=env),
    )
    container_lifecycle.deploy_down(str(tmp_path / "config.yml"))
    assert captured["env"]["COMPOSE_PROJECT_NAME"] == "myproj"
    assert "down" in captured["args"]


def test_deploy_restart_pins_compose_project_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"project_name": "myproj", "deployed_services": ["event_dispatcher"]},
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    captured: dict = {}
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda cmd, env=None, **k: captured.update(cmd=cmd, env=env),
    )
    container_lifecycle.deploy_restart(str(tmp_path / "config.yml"))
    assert captured["env"]["COMPOSE_PROJECT_NAME"] == "myproj"
    assert "restart" in captured["cmd"]


def test_rebuild_deployment_pins_compose_project_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"project_name": "myproj", "deployed_services": ["event_dispatcher"]},
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "clean_deployment", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    # The pre-up `compose build` split (Defect A) lands on subprocess.run; swallow it.
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)
    captured: dict = {}
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvpe",
        lambda file, args, env: captured.update(file=file, args=args, env=env),
    )
    container_lifecycle.rebuild_deployment(str(tmp_path / "config.yml"))
    assert captured["env"]["COMPOSE_PROJECT_NAME"] == "myproj"
    assert "up" in captured["args"]


def test_clean_deployment_pins_compose_project_name(monkeypatch, tmp_path):
    """compose_generator.clean_deployment's down/rmi invocations must also be
    pinned — an unpinned `down --volumes` would target the shared project."""
    monkeypatch.chdir(tmp_path)
    from osprey.deployment import compose_generator

    monkeypatch.setattr(
        compose_generator, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    envs: list = []
    monkeypatch.setattr(
        compose_generator.subprocess, "run", lambda cmd, env=None, **k: envs.append(env)
    )
    compose_generator.clean_deployment(["docker-compose.yml"], {"project_name": "myproj"})
    assert envs, "clean_deployment ran no compose commands"
    for env in envs:
        assert env is not None and env["COMPOSE_PROJECT_NAME"] == "myproj"


# ---------------------------------------------------------------------------
# Defect A: build/create split — never `up --build` in one invocation
#
# Under Docker's containerd image store, `compose up --build` can build a
# local-only tag and then fail container-create with "No such image" in the same
# call. Wherever a build is intended, run `compose build` first, then
# `up --no-build`. The non-dev services path is deliberately left on a plain
# `up` (no --no-build) so compose's implicit build-on-up still covers a
# build-only service with no published upstream tag.
# ---------------------------------------------------------------------------


def test_deploy_up_dev_mode_splits_build_from_up(monkeypatch, tmp_path):
    """--dev must not produce a single `up --build`; it must be `build` then
    `up --no-build`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    runs: list = []
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda cmd, env=None, **k: runs.append(cmd)
    )
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=True)

    joined = [" ".join(c) for c in runs]
    # No single invocation combines `up` with `--build`.
    assert not any("up" in c and "--build" in c for c in runs)
    # A standalone `build` ran, and a subsequent `up --no-build`.
    assert any(c[-1] == "build" for c in runs), joined
    assert any("up" in c and "--no-build" in c for c in runs), joined


def test_rebuild_deployment_splits_build_from_up(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "clean_deployment", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    runs: list = []
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda cmd, env=None, **k: runs.append(cmd)
    )
    execd: dict = {}
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvpe",
        lambda file, args, env: execd.update(args=args),
    )
    container_lifecycle.rebuild_deployment(str(tmp_path / "config.yml"))
    # `build` ran as its own subprocess; the exec'd `up` carries --no-build.
    assert any(c[-1] == "build" for c in runs), [" ".join(c) for c in runs]
    assert "up" in execd["args"] and "--no-build" in execd["args"]
    assert "--build" not in execd["args"]


def test_web_services_dev_mode_splits_build_from_up(monkeypatch, tmp_path):
    """The web path's backend-services stack: --dev builds then ups --no-build,
    never `up --build` in one call. Needs a non-empty deployed_services (the
    services block is guarded on it), which captured_web_runs lacks."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {
                "deployed_services": ["event_dispatcher"],
                "modules": {"web_terminals": {"enabled": True}},
            },
            ["build/services/docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "write_web_terminal_artifacts", lambda *a, **k: [])
    monkeypatch.setattr(container_lifecycle, "_enable_linger", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "seed_user_containers", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_run_verify_script", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    runs: list = []
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda cmd, env=None, **k: runs.append(list(cmd))
    )
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=True)

    # The services-stack invocations (the ones carrying the services compose file).
    svc = [c for c in runs if "build/services/docker-compose.yml" in c]
    assert not any("up" in c and "--build" in c for c in svc)
    assert any(c[-1] == "build" for c in svc), [" ".join(c) for c in svc]
    assert any("up" in c and "--no-build" in c for c in svc), [" ".join(c) for c in svc]
