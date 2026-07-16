"""Unit tests for container lifecycle argv construction.

These stub out the compose-file preparation, runtime checks, and the actual
subprocess invocation, then assert on the argv that ``deploy_up`` would run —
the cheapest way to lock in flag behavior (notably ``--build`` under ``--dev``)
without a container runtime.
"""

from __future__ import annotations

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

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


def test_deploy_up_dev_mode_adds_build(captured_argv, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=True)
    assert "--build" in captured_argv["cmd"]
    assert "up" in captured_argv["cmd"]


def test_deploy_up_non_dev_omits_build(captured_argv, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)
    assert "--build" not in captured_argv["cmd"]
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
    """
    calls: list[dict] = []
    written: list = []

    monkeypatch.chdir(tmp_path)
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
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=True)

    up_calls = [c for c in captured_combined_runs["calls"] if "up" in c["cmd"]]
    services_up = next(c["cmd"] for c in up_calls if "docker-compose.yml" in c["cmd"])
    web_up = next(c["cmd"] for c in up_calls if "docker-compose.web.yml" in c["cmd"])

    assert "--build" in services_up
    assert "--build" not in web_up


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


def test_web_deploy_calls_enable_linger_in_post_up_hook(monkeypatch, tmp_path):
    """The post-up hook wires _enable_linger(config, run_env) -- the same
    COMPOSE_PROJECT_NAME-pinned env the compose calls around it use."""
    monkeypatch.chdir(tmp_path)
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
