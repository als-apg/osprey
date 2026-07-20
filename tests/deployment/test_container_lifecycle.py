"""Unit tests for container lifecycle argv construction.

These stub out the compose-file preparation, runtime checks, and the actual
subprocess invocation, then assert on the argv that ``deploy_up`` would run —
the cheapest way to lock in flag behavior (notably ``--build`` under ``--dev``)
without a container runtime.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.web_terminals import provision


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
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])

    def _fake_write_artifacts(config, dest_dir="."):
        written.append(config)
        return []

    monkeypatch.setattr(provision, "write_web_terminal_artifacts", _fake_write_artifacts)

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
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])

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


def test_web_deploy_docker_runtime_linger_adds_no_subprocess_calls(captured_web_runs, tmp_path):
    """captured_web_runs defaults to docker -- the post-up hook's linger step
    must add zero subprocess calls beyond the ordinary rm preflight + pull + up."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(captured_web_runs["calls"]) == 3


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
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    linger_calls = []
    monkeypatch.setattr(
        provision,
        "_enable_linger",
        lambda config, run_env: linger_calls.append(run_env),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(linger_calls) == 1
    assert "COMPOSE_PROJECT_NAME" in linger_calls[0]


# ---------------------------------------------------------------------------
# deploy_up_web_terminals mode wiring (Task 3.3) -- image_source local vs.
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
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    # Auto-render is a separate concern with its own dedicated tests below;
    # keep it inert here so the mode-wiring tests exercise only the local/
    # registry step ordering, never a real `osprey build` subprocess.
    monkeypatch.setattr(
        provision,
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


def test_local_mode_calls_auto_render_then_ensure_env_production_then_build_then_compose(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """The local-mode preflight order is load-bearing: auto-render any missing
    persona project FIRST, then ensure_env_production, then build the image,
    then compose. ensure_env_production's claude_code credential sweep reads
    each rendered persona's config.yml, so on a first deploy it must run
    after auto-render (and still before any compose call). A spy on
    _auto_render_missing_personas (overriding the fixture's inert stub)
    proves the wiring line actually runs it -- and runs it BEFORE
    build_persona_images, which needs the rendered context to exist."""
    order: list[str] = []
    config = _web_terminals_config("local")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))
    monkeypatch.setattr(
        provision,
        "ensure_env_production",
        lambda cfg, root: order.append("ensure_env_production"),
    )
    monkeypatch.setattr(
        provision,
        "_auto_render_missing_personas",
        lambda cfg, resolved_users, env: order.append("auto_render"),
    )

    def _fake_build(cfg, resolved_users, dev_mode, env):
        order.append("build_persona_images")

    monkeypatch.setattr(provision, "build_persona_images", _fake_build)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda *a, **k: order.append("compose") or _FakeCompletedProcess(returncode=0),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    # Exactly two compose calls (the stale-container `rm -f` preflight, then
    # the web `up -d`; no deployed_services, no pull in local mode) after all
    # three preflight steps, in this order.
    assert order == [
        "auto_render",
        "ensure_env_production",
        "build_persona_images",
        "compose",
        "compose",
    ]


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
    monkeypatch.setattr(provision, "seed_user_containers", lambda cfg, env=None: None)

    captured_users = []

    def _fake_build(cfg, resolved_users, dev_mode, env):
        captured_users.extend(resolved_users)

    monkeypatch.setattr(provision, "build_persona_images", _fake_build)

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
        provision,
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
        provision,
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
        provision,
        "ensure_env_production",
        lambda cfg, root: order.append("ensure_env_production"),
    )

    def _fake_run(cmd, **kwargs):
        if "rm" in cmd:
            order.append("rm")
        else:
            order.append("pull" if "pull" in cmd else "up")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert order == ["ensure_env_production", "rm", "pull", "up"]


def test_post_up_hook_order_is_linger_then_seed_then_verify(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    (tmp_path / ".env.production").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    order: list[str] = []
    monkeypatch.setattr(provision, "_enable_linger", lambda cfg, run_env: order.append("linger"))
    monkeypatch.setattr(
        provision,
        "seed_user_containers",
        lambda cfg, env=None: order.append("seed"),
    )
    monkeypatch.setattr(
        provision, "_run_verify_script", lambda root, run_env: order.append("verify")
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
    web-terminals path (which also reaches compose via deploy_up_web_terminals)."""
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


def _capture_exec_and_web_down(monkeypatch):
    """Record the services execvpe and any deploy_down_web_terminals call."""
    captured: dict = {"web_down_order": None, "exec_order": None}
    order = iter(range(100))
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvpe",
        lambda file, args, env: captured.update(args=args, env=env, exec_order=next(order)),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "deploy_down_web_terminals",
        lambda config, env, env_file_args: captured.update(
            web_down_config=config, web_down_order=next(order)
        ),
    )
    return captured


def test_deploy_down_tears_down_web_stack_before_services(monkeypatch, tmp_path):
    """With modules.web_terminals.enabled, deploy_down must run the web
    stack's own `compose down` (deploy_up_web_terminals' mirror) BEFORE the
    services execvpe replaces the process — the services `-f` list can never
    carry docker-compose.web.yml (root-relative paths), so skipping this
    leaves the fixed-name web/nginx containers running after every
    `osprey deploy down`."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch, "myproj")
    monkeypatch.setattr(
        container_lifecycle,
        "normalize_facility_config",
        lambda raw: {
            "project_name": "myproj",
            "deployed_services": ["event_dispatcher"],
            "modules": {"web_terminals": {"enabled": True}},
        },
    )
    captured = _capture_exec_and_web_down(monkeypatch)

    container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert captured["web_down_order"] is not None, "web stack was never torn down"
    assert captured["web_down_order"] < captured["exec_order"], (
        "web-stack down must run before the process-replacing services down"
    )
    assert captured["web_down_config"]["project_name"] == "myproj"


def test_deploy_down_skips_web_stack_when_module_disabled(monkeypatch, tmp_path):
    """No modules.web_terminals.enabled → the plain services-only down, no
    web-stack invocation."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch, "myproj")
    captured = _capture_exec_and_web_down(monkeypatch)

    container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert captured["web_down_order"] is None
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
    # The stale-container `rm -f` preflight lands on subprocess.run; swallow it.
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


def test_rebuild_deployment_dev_mode_splits_build_from_up(monkeypatch, tmp_path):
    """rebuild delegates its up phase to deploy_up, so --dev inherits the same
    build/up split (Defect A): standalone `build`, then exec'd `up --no-build`."""
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
    container_lifecycle.rebuild_deployment(str(tmp_path / "config.yml"), dev_mode=True)
    # `build` ran as its own subprocess; the exec'd `up` carries --no-build.
    assert any(c[-1] == "build" for c in runs), [" ".join(c) for c in runs]
    assert "up" in execd["args"] and "--no-build" in execd["args"]
    assert "--build" not in execd["args"]


def test_rebuild_deployment_cleans_before_delegating_to_deploy_up(monkeypatch, tmp_path):
    """rebuild = clean, then the real deploy_up (single definition of every
    up-path behavior); clean must run first."""
    monkeypatch.chdir(tmp_path)
    order: list[str] = []
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "clean_deployment", lambda *a, **k: order.append("clean")
    )
    monkeypatch.setattr(
        container_lifecycle,
        "deploy_up",
        lambda *a, **k: order.append("deploy_up"),
    )
    container_lifecycle.rebuild_deployment(str(tmp_path / "config.yml"))
    assert order == ["clean", "deploy_up"]


def test_rebuild_deployment_reconciles_web_terminals_stack(monkeypatch, tmp_path):
    """A web-terminals project's rebuild must reach the web reconcile — the
    pre-delegation rebuild ran only the plain services path, so nginx and the
    persona containers never came back up after clean."""
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
    monkeypatch.setattr(container_lifecycle, "clean_deployment", lambda *a, **k: None)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    calls: list = []

    def _fake_run(cmd, env=None, **k):
        calls.append(list(cmd))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    container_lifecycle.rebuild_deployment(str(tmp_path / "config.yml"))
    assert any("docker-compose.web.yml" in c and "up" in c for c in calls)


# ---------------------------------------------------------------------------
# Stale-container preflight — self-healing `deploy up`
#
# An aborted deploy leaves containers wedged in created/exited state, and
# Docker Desktop reserves published host ports at container CREATE time — so
# the next `up` collides with its own ghost ("address already in use" with
# nothing listening on the port). Every up path first runs a service-scoped
# `rm -f` (removes only non-running containers; running containers and
# volumes untouched; exit-0 no-op on a clean stack). The plain path's `up`
# additionally carries --remove-orphans; the web path must NOT — its two
# invocations share one COMPOSE_PROJECT_NAME, so orphan-removal in either
# would destroy the other stack's containers as "orphans".
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_plain_runs(monkeypatch, tmp_path):
    """Plain (non-web) deploy_up with every subprocess.run argv captured in order."""
    calls: list[list[str]] = []
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
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda cmd, env=None, **k: calls.append(list(cmd))
    )
    return calls


def test_deploy_up_runs_stale_container_preflight_before_up(captured_plain_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    rm_idx = next(i for i, c in enumerate(captured_plain_runs) if c[-2:] == ["rm", "-f"])
    up_idx = next(i for i, c in enumerate(captured_plain_runs) if "up" in c)
    assert rm_idx < up_idx
    # Scoped to this deploy's own compose files — and it is `rm`, never a
    # `down` (which would stop running containers).
    rm_cmd = captured_plain_runs[rm_idx]
    assert "docker-compose.yml" in rm_cmd
    assert "down" not in rm_cmd


def test_deploy_up_preflight_never_stops_or_removes_volumes(captured_plain_runs, tmp_path):
    """`rm -f` must stay surgical: no -s/--stop (would touch running
    containers) and no -v/--volumes (destroying state is clean/rebuild's job)."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    rm_cmd = next(c for c in captured_plain_runs if c[-2:] == ["rm", "-f"])
    for forbidden in ("-s", "--stop", "-v", "--volumes"):
        assert forbidden not in rm_cmd


def test_deploy_up_plain_up_carries_remove_orphans(captured_plain_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    up_cmd = next(c for c in captured_plain_runs if "up" in c)
    assert "--remove-orphans" in up_cmd


def test_web_deploy_preflights_rm_and_never_remove_orphans(captured_web_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in captured_web_runs["calls"]]
    rm_idx = next(i for i, c in enumerate(cmds) if c[-2:] == ["rm", "-f"])
    up_idx = next(i for i, c in enumerate(cmds) if "up" in c)
    assert rm_idx < up_idx
    for cmd in cmds:
        assert "--remove-orphans" not in cmd


def test_combined_deploy_each_stack_gets_its_own_rm_preflight(captured_combined_runs, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"]]
    rm_calls = [c for c in cmds if c[-2:] == ["rm", "-f"]]
    assert len(rm_calls) == 2
    assert any("docker-compose.yml" in c for c in rm_calls)
    assert any("docker-compose.web.yml" in c for c in rm_calls)
    # The two stacks' files are never merged into one rm argv, and the
    # shared-project path never orphan-removes.
    for c in rm_calls:
        assert not ("docker-compose.yml" in c and "docker-compose.web.yml" in c)
    for c in cmds:
        assert "--remove-orphans" not in c


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
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda *a, **k: [])
    monkeypatch.setattr(provision, "_enable_linger", lambda *a, **k: None)
    monkeypatch.setattr(provision, "seed_user_containers", lambda *a, **k: None)
    monkeypatch.setattr(provision, "_run_verify_script", lambda *a, **k: None)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
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


# ---------------------------------------------------------------------------
# _project_image_build_cmd -- com.osprey.project label + OSPREY_DEV build-arg
# (task 2.5). The label lets a later `nuke` verify a tag belongs to this
# deployment before removing it (matching the persona build path); OSPREY_DEV=1
# is added iff --dev, mirroring the persona dev path.
# ---------------------------------------------------------------------------


def test_project_image_build_cmd_carries_project_label():
    cmd = container_lifecycle._project_image_build_cmd(
        {"project_name": "myfacility"}, "docker", "/proj"
    )
    assert "--label" in cmd
    assert "com.osprey.project=myfacility" in cmd
    # The label value tracks resolve_project_name (normalized), same as the tag.
    assert f"{cmd[cmd.index('-t') + 1]}" == "myfacility:local"
    assert cmd[-1] == "/proj"  # context stays last


def test_project_image_build_cmd_non_dev_omits_osprey_dev_build_arg():
    cmd = container_lifecycle._project_image_build_cmd(
        {"project_name": "myfacility"}, "docker", "/proj", dev_mode=False
    )
    assert "OSPREY_DEV=1" not in cmd
    assert not any(str(a) == "OSPREY_DEV=1" for a in cmd)


def test_project_image_build_cmd_dev_adds_osprey_dev_build_arg():
    cmd = container_lifecycle._project_image_build_cmd(
        {"project_name": "myfacility"}, "docker", "/proj", dev_mode=True
    )
    assert "OSPREY_DEV=1" in cmd
    # Properly paired behind a --build-arg flag, with the context still last.
    assert cmd[cmd.index("OSPREY_DEV=1") - 1] == "--build-arg"
    assert cmd[-1] == "/proj"


# ---------------------------------------------------------------------------
# _build_project_image -- OSPREY_DEV is keyed on ACTUAL wheel-staging success
# (fail-closed). A --dev build whose wheel build/staging failed must NOT pass
# the pin-relaxing OSPREY_DEV=1 arg: with an unreleased pin that arg would
# silently install the latest published release instead of the local code the
# flag promises. The image is still built -- just with fail-loud pin semantics.
# ---------------------------------------------------------------------------


def _project_image_dev_build_cmds(monkeypatch, tmp_path, staging_result):
    """Run _build_project_image under --dev with a stubbed staging outcome;
    return the captured build argv list."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_copy_local_framework_for_override",
        lambda project_root: staging_result,
    )
    calls = []
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    config = {"project_name": "myfacility", "deployed_services": ["dispatch_worker"]}
    container_lifecycle._build_project_image(config, dev_mode=True, env={})
    return calls


def test_build_project_image_dev_passes_osprey_dev_when_wheel_staged(monkeypatch, tmp_path):
    (cmd,) = _project_image_dev_build_cmds(monkeypatch, tmp_path, staging_result=True)
    assert "OSPREY_DEV=1" in cmd
    assert cmd[cmd.index("OSPREY_DEV=1") - 1] == "--build-arg"


def test_build_project_image_dev_omits_osprey_dev_when_staging_fails(monkeypatch, tmp_path):
    (cmd,) = _project_image_dev_build_cmds(monkeypatch, tmp_path, staging_result=False)
    assert "OSPREY_DEV=1" not in cmd


def _fake_wheel_and_manifest_stage(project_root):
    """Staging stub that drops BOTH dev artifacts, like the real helper does."""
    Path(project_root, "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
    Path(project_root, "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
    return True


def test_build_project_image_dev_cleans_staged_wheel_and_manifest(monkeypatch, tmp_path):
    """The finally-cleanup must remove BOTH staged artifacts — wheel AND
    osprey-local-requirements.txt — so neither poisons a later non-dev build."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        container_lifecycle, "_copy_local_framework_for_override", _fake_wheel_and_manifest_stage
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda cmd, **k: None)
    config = {"project_name": "myfacility", "deployed_services": ["dispatch_worker"]}

    container_lifecycle._build_project_image(config, dev_mode=True, env={})

    assert list(tmp_path.glob("*.whl")) == []
    assert not (tmp_path / "osprey-local-requirements.txt").exists()


def test_build_project_image_dev_cleans_staged_artifacts_on_build_failure(monkeypatch, tmp_path):
    """Cleanup runs in a finally: a failing image build must still remove the
    staged wheel + manifest."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        container_lifecycle, "_copy_local_framework_for_override", _fake_wheel_and_manifest_stage
    )

    def _failing_build(cmd, **k):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _failing_build)
    config = {"project_name": "myfacility", "deployed_services": ["dispatch_worker"]}

    with pytest.raises(subprocess.CalledProcessError):
        container_lifecycle._build_project_image(config, dev_mode=True, env={})

    assert list(tmp_path.glob("*.whl")) == []
    assert not (tmp_path / "osprey-local-requirements.txt").exists()


# ---------------------------------------------------------------------------
# _warn_unignored_build_dir -- --dev context-bloat guard (task 3.3). Warns when
# the rendered project's build/ dir would be swept into the --dev build context
# because .dockerignore doesn't exclude it; silent otherwise.
# ---------------------------------------------------------------------------


@pytest.fixture
def _captured_warnings(monkeypatch):
    warnings: list = []
    monkeypatch.setattr(
        container_lifecycle.logger, "warning", lambda *a, **k: warnings.append((a, k))
    )
    return warnings


def test_warn_unignored_build_dir_fires_when_build_present_and_unignored(
    _captured_warnings, tmp_path
):
    """build/ exists and no .dockerignore (missing == not-matching) -> warn once."""
    (tmp_path / "build").mkdir()
    container_lifecycle._warn_unignored_build_dir(str(tmp_path))
    assert len(_captured_warnings) == 1


def test_warn_unignored_build_dir_fires_when_dockerignore_lacks_build(_captured_warnings, tmp_path):
    """build/ exists, .dockerignore present but lists other paths -> still warn."""
    (tmp_path / "build").mkdir()
    (tmp_path / ".dockerignore").write_text("*.whl\n.venv/\n", encoding="utf-8")
    container_lifecycle._warn_unignored_build_dir(str(tmp_path))
    assert len(_captured_warnings) == 1


def test_warn_unignored_build_dir_silent_when_build_absent(_captured_warnings, tmp_path):
    container_lifecycle._warn_unignored_build_dir(str(tmp_path))
    assert _captured_warnings == []


@pytest.mark.parametrize("ignore_line", ["build/", "build"])
def test_warn_unignored_build_dir_silent_when_dockerignore_excludes_build(
    _captured_warnings, tmp_path, ignore_line
):
    """A matching build/ (or build) line in .dockerignore silences the warning."""
    (tmp_path / "build").mkdir()
    (tmp_path / ".dockerignore").write_text(f"*.whl\n{ignore_line}\n", encoding="utf-8")
    container_lifecycle._warn_unignored_build_dir(str(tmp_path))
    assert _captured_warnings == []
