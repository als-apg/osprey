"""Unit tests for container lifecycle argv construction.

These stub out the compose-file preparation, runtime checks, and the actual
subprocess invocation, then assert on the argv that ``deploy_up`` would run —
the cheapest way to lock in flag behavior (notably ``--build`` under ``--dev``)
without a container runtime.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.errors import (
    ArchiverClientMissingError,
    DeploymentPreconditionError,
    UnmetPreconditionsError,
)
from osprey.deployment.web_terminals import postup_hooks, provision


def _fake_popen(record):
    """A ``subprocess.Popen`` stand-in for the watched capture path.

    The dev-mode ``compose build`` runs with an ``on_line`` watcher, which
    routes it through ``Popen`` rather than ``subprocess.run`` — so a test that
    fakes only ``run`` would let the build escape to a real child. The stand-in
    records the argv through ``record(cmd, env)``, yields no output, and exits
    0, keeping the argv assertions blind to which of the two paths ran a call.
    """

    class FakePopen:
        def __init__(self, cmd, env=None, **kwargs):
            record(list(cmd), env)
            self.stdout = iter(())

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self):
            return 0

    return FakePopen


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators and capture the compose argv.

    Runs in ``detached=True`` mode so the call lands on ``subprocess.run`` (which
    we capture) rather than ``os.execvpe`` (which would replace the process).
    """
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    # An operator shell that exported the prebuilt-images switch would delete
    # the dev-mode build from every deploy driven here (see the switch tests).
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: captured.update(cmd=cmd, env=env)),
    )
    return captured


def _addresses(cmd: list[str], compose_filename: str) -> bool:
    """True when this argv's ``-f`` list names *compose_filename*.

    The pinned invocation contract spells every ``-f`` as a repo-anchored
    absolute path, so a plain ``filename in cmd`` membership does not hold.
    Matched on the trailing path segment, which keeps ``docker-compose.yml``
    from matching ``docker-compose.web.yml``.
    """
    return any(arg == compose_filename or arg.endswith("/" + compose_filename) for arg in cmd)


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


def test_tokens_are_minted_into_the_repo_root_not_the_cwd(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """The mint follows the config's repo, not wherever the command was typed.

    Regression guard: the provisioners' ``env_path`` defaults to a cwd-relative
    ``.env``, and this path left it unset while resolving the repo root a few
    lines above. Run from any other directory and real secrets landed in a stray
    file the stack never reads — the containers then start with their
    fail-closed tokens unset, which looks secure and reports nothing.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    container_lifecycle.deploy_up(str(repo / "config.yml"), detached=True)

    assert _parse_env(repo).get("EVENT_DISPATCHER_TOKEN")
    assert not (elsewhere / ".env").exists()


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
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda *a, **k: _FakeCompletedProcess()
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert not (tmp_path / ".env").exists()


def test_an_exposed_deploy_refuses_an_empty_token(captured_argv, monkeypatch, tmp_path):
    # A token explicitly set empty must not be auto-overwritten, and a deployment
    # reachable off-host must refuse rather than bind a fail-open server to it.
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "")
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="reachable off-host with an empty token"):
        container_lifecycle.deploy_up(
            str(tmp_path / "config.yml"), detached=True, expose_network=True
        )


# ---------------------------------------------------------------------------
# Web-terminal reconcile (osprey up, modules.web_terminals.enabled)
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_web_runs(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a web-terminals-enabled deploy.

    Captures every ``subprocess.run`` invocation (argv + env) in order, so
    tests can inspect both the ``pull`` and the ``up -d`` calls the web
    reconcile path issues. ``write_web_terminal_artifacts`` is stubbed out —
    its own rendering is covered by ``tests/deployment/web_terminals/``, not
    here — but still records that it was called with the config.

    The post-up host-reachability probe answers ``True``, i.e. this deploy's
    landing page is reachable — the ordinary outcome, and the only one these
    tests are about. Left real it would open a socket to the deployment's real
    gateway port five times over ten seconds and then, finding nothing, ask
    Docker Desktop about host networking and bounce the stack, putting three
    host-inspection commands into ``calls`` that no test here is asserting
    about. Its unreachable and self-heal branches are covered directly in
    ``tests/deployment/web_terminals/test_postup_hooks.py``.

    Defaults to registry mode (no ``image_source`` key), so a
    ``.env.users`` marker is pre-written to ``tmp_path``:
    ``ensure_env_production``'s registry-mode branch only exists-checks (see
    its own tests), so without this every test using this fixture would hit
    its "not found" RuntimeError before ever reaching a compose call.
    """
    calls: list[dict] = []
    written: list = []

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(postup_hooks, "_host_port_answers", lambda url, attempts, delay: True)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_write_artifacts(config, dest_dir="."):
        written.append(config)
        return []

    monkeypatch.setattr(provision, "write_web_terminal_artifacts", _fake_write_artifacts)

    def _fake_run(cmd, env=None, check=False, **kwargs):
        calls.append({"cmd": list(cmd), "env": env})
        return _FakeCompletedProcess(returncode=0)

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
    assert _addresses(up_cmd, "docker-compose.web.yml")


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

    assert not _addresses(captured_argv["cmd"], "docker-compose.web.yml")
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
    # Same hygiene as captured_argv: the switch tests set this deliberately, and
    # an exported one would otherwise remove the build these tests count.
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)
    # Registry mode (default) -- pre-write .env.users so
    # ensure_env_production's exists-check passes (see captured_web_runs).
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(postup_hooks, "_host_port_answers", lambda url, attempts, delay: True)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])

    def _fake_build(config, dev_mode, env, build_context=None):
        build_calls.append({"config": config, "dev_mode": dev_mode})

    monkeypatch.setattr(container_lifecycle, "_build_project_image", _fake_build)

    def _fake_tokens(config, expose_network, env_path=None):
        token_calls.append({"config": config})

    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", _fake_tokens)

    def _fake_run(cmd, env=None, check=False, **kwargs):
        calls.append({"cmd": list(cmd), "env": env})
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: calls.append({"cmd": cmd, "env": env})),
    )
    return {"calls": calls, "build_calls": build_calls, "token_calls": token_calls}


def test_combined_services_and_web_deploy_two_detached_up_calls(captured_combined_runs, tmp_path):
    """Real-daemon regression guard (compose path-resolution bug):

    Compose resolves every relative path in EVERY merged `-f` file against
    the directory of the FIRST `-f` file. compose_files (build/services/...)
    and docker-compose.web.yml (project root) are written to resolve against
    two DIFFERENT directories, so they must never be merged into one `-f ...
    -f docker-compose.web.yml` argv -- a real `osprey up` with both
    enabled failed immediately with "env file .../build/services/
    .env.users not found" until this was split into two invocations.
    """
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    up_calls = [c for c in captured_combined_runs["calls"] if "up" in c["cmd"]]
    assert len(up_calls) == 2

    services_up = [c["cmd"] for c in up_calls if _addresses(c["cmd"], "docker-compose.yml")]
    web_up = [c["cmd"] for c in up_calls if _addresses(c["cmd"], "docker-compose.web.yml")]
    assert len(services_up) == 1
    assert len(web_up) == 1

    # The services and web compose files must never appear together in one
    # argv -- that merge is exactly what broke path resolution.
    assert not _addresses(services_up[0], "docker-compose.web.yml")
    assert not _addresses(web_up[0], "docker-compose.yml")

    for cmd in (services_up[0], web_up[0]):
        assert "-d" in cmd


def test_combined_services_up_gets_dev_build_web_up_never_does(captured_combined_runs, tmp_path):
    """Under --dev the services stack builds in its OWN step then `up --no-build`
    (never `up --build` in one call, per Defect A); the web stack never builds."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=True)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"]]
    up_calls = [c for c in cmds if "up" in c]
    services_up = next(c for c in up_calls if _addresses(c, "docker-compose.yml"))
    web_up = next(c for c in up_calls if _addresses(c, "docker-compose.web.yml"))

    # A standalone services `build` ran (services compose file, no `up`).
    services_build = [c for c in cmds if c[-1] == "build" and _addresses(c, "docker-compose.yml")]
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
    assert _addresses(up_calls[0], "docker-compose.web.yml")


def test_combined_deploy_services_up_never_pulls(captured_combined_runs, tmp_path):
    """Only the web stack's images are always registry-hosted; a deployed
    service may declare only a `build:` block with no published upstream tag,
    and `compose pull` hard-fails on that -- so the services invocation must
    never carry a `pull` step (matching the plain non-web path, which never
    pulls either)."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    pull_calls = [c["cmd"] for c in captured_combined_runs["calls"] if "pull" in c["cmd"]]
    assert len(pull_calls) == 1
    assert _addresses(pull_calls[0], "docker-compose.web.yml")
    assert not _addresses(pull_calls[0], "docker-compose.yml")


# ---------------------------------------------------------------------------
# enable_linger (task 2.3) -- rootless-podman persistence via loginctl
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_web_deploy_docker_runtime_linger_adds_no_subprocess_calls(captured_web_runs, tmp_path):
    """captured_web_runs defaults to docker -- the post-up hook's linger step
    must add zero subprocess calls beyond the ordinary rm preflight + pull + up
    + nginx config reload."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert len(captured_web_runs["calls"]) == 4


def test_web_deploy_callsenable_linger_in_post_up_hook(monkeypatch, tmp_path):
    """The post-up hook wires enable_linger(config, run_env) -- the same
    COMPOSE_PROJECT_NAME-pinned env the compose calls around it use."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(postup_hooks, "_host_port_answers", lambda url, attempts, delay: True)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["podman", "compose"]
    )
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0),
    )

    linger_calls = []
    monkeypatch.setattr(
        provision,
        "enable_linger",
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
    (needed because run_verify_script inspects .returncode on every call it
    makes, not just compose's). Deliberately does NOT pre-write .env.users
    or .env -- each test supplies exactly what its mode needs to exercise
    ensure_env_production's own branches.
    """
    calls: list[dict] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(postup_hooks, "_host_port_answers", lambda url, attempts, delay: True)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    # The persona-render check is a separate concern with its own dedicated
    # tests below; keep it inert here so the mode-wiring tests exercise only the
    # local/registry step ordering rather than any repo's rendered state.
    monkeypatch.setattr(
        provision,
        "verify_persona_renders",
        lambda config, resolved_users, repo_root=None: None,
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
    # ensure_env_production generated .env.users from .env since neither
    # was present -- local mode's own branch, exercised end-to-end here.
    assert (tmp_path / ".env.users").is_file()


def test_registry_mode_still_pulls(monkeypatch, tmp_path, _mode_wiring_collab):
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in _mode_wiring_collab]
    assert any("pull" in cmd for cmd in cmds)
    assert any("up" in cmd and "-d" in cmd for cmd in cmds)


def test_up_hot_reloads_nginx_after_web_stack_up(monkeypatch, tmp_path, _mode_wiring_collab):
    """`up -d` never restarts a running nginx whose bind-mounted config CONTENT
    changed (the container definition is unchanged), so the post-up hook must
    issue a `compose exec nginx nginx -s reload` — after the web stack's up."""
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in _mode_wiring_collab]
    reload_idx = next(
        (i for i, cmd in enumerate(cmds) if cmd[-4:] == ["nginx", "nginx", "-s", "reload"]), None
    )
    assert reload_idx is not None, f"no nginx reload argv emitted: {cmds}"
    assert "exec" in cmds[reload_idx] and "-T" in cmds[reload_idx]
    up_idx = next(i for i, cmd in enumerate(cmds) if "up" in cmd and "-d" in cmd)
    assert reload_idx > up_idx


def test_nginx_reload_failure_is_advisory(monkeypatch, tmp_path, _mode_wiring_collab):
    """A failing nginx reload (e.g. container still starting) warns but never
    fails a deploy that did reconcile."""
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    def _fake_run(cmd, **kwargs):
        rc = 1 if "exec" in cmd else 0
        return _FakeCompletedProcess(returncode=rc)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)  # must not raise


def test_registry_mode_raises_before_any_compose_call_when_env_production_missing(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """Neither .env.users nor .env present -- ensure_env_production raises
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
    `compose up` (the reviewer integration note from task 3.2).

    The collect-all pass now asks this question one step ahead of the
    preflight that used to raise it, so the refusal arrives in the aggregate
    frame. The persona is still named, which is the property that matters."""
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")
    config = _web_terminals_config(
        "local", users=[{"name": "alice", "index": 0, "persona": "no-such-persona"}]
    )
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    with pytest.raises(UnmetPreconditionsError, match="no-such-persona"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert _mode_wiring_collab == []  # no compose subprocess ever ran


def test_local_mode_verifies_renders_then_ensure_env_production_then_build_then_compose(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """The local-mode preflight order is load-bearing: check every persona has
    the render `osprey build` wrote FIRST, then ensure_env_production, then
    build the image, then compose. ensure_env_production's claude_code
    credential sweep reads each rendered persona's config.yml, so a deploy
    missing one has to be refused before that sweep runs (and long before any
    compose call). A spy on verify_persona_renders (overriding the fixture's
    inert stub) proves the wiring line actually runs it -- and runs it BEFORE
    build_persona_images, which needs the rendered context to exist.

    The collect-all pass asks the same render question once more, ahead of the
    preflight, which is why the spy sees a third call. It only reads, so the
    extra call changes nothing about the deployment -- and the preflight below
    it still runs unchanged, which is the property the rest of this order
    pins."""
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
        "verify_persona_renders",
        lambda cfg, resolved_users, repo_root=None: order.append("verify_persona_renders"),
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

    # The collect-all pass reads the renders first (and nothing else -- it
    # probes .env.users generation rather than calling ensure_env_production).
    # The fail-fast deploy_up preflight then runs the persona check +
    # ensure_env_production once BEFORE the image-build stage, and
    # deploy_up_web_terminals re-runs the same (idempotent) pair; then exactly
    # three compose calls (the stale-container `rm -f` preflight, the web
    # `up -d`, then the advisory nginx config reload; no deployed_services, no
    # pull in local mode).
    assert order == [
        "verify_persona_renders",
        "verify_persona_renders",
        "ensure_env_production",
        "verify_persona_renders",
        "ensure_env_production",
        "build_persona_images",
        "compose",
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
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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


def test_registry_mode_never_verifies_persona_renders(monkeypatch, tmp_path, _mode_wiring_collab):
    """The persona-render check is a local-mode-only step (registry mode pulls
    prebuilt images and needs no render at all) -- it must never run on the
    registry path, mirroring the build_persona_images guard. A recording spy
    overrides the fixture's inert stub so a stray call would be caught, not
    swallowed."""
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    render_calls = []
    monkeypatch.setattr(
        provision,
        "verify_persona_renders",
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
        elif "exec" in cmd:
            order.append("nginx_reload")
        else:
            order.append("pull" if "pull" in cmd else "up")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    # Twice before any compose call: once from deploy_up's fail-fast preflight
    # (pre-image-build), once from deploy_up_web_terminals' own idempotent run.
    assert order == [
        "ensure_env_production",
        "ensure_env_production",
        "rm",
        "pull",
        "up",
        "nginx_reload",
    ]


def test_post_up_hook_order_is_linger_then_seed_then_verify(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
    config = _web_terminals_config("registry")
    monkeypatch.setattr(container_lifecycle, "prepare_compose_files", lambda *a, **k: (config, []))

    order: list[str] = []
    monkeypatch.setattr(provision, "enable_linger", lambda cfg, run_env: order.append("linger"))
    monkeypatch.setattr(
        provision,
        "seed_user_containers",
        lambda cfg, env=None: order.append("seed"),
    )
    monkeypatch.setattr(
        provision, "run_verify_script", lambda root, run_env: order.append("verify")
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert order == ["linger", "seed", "verify"]


def test_deploy_up_runs_verify_script_when_present_ignoring_exit_code(
    monkeypatch, tmp_path, _mode_wiring_collab
):
    """A nonzero verify.sh exit must not propagate out of `osprey up` --
    advisory only, per the script's own convention and run_verify_script's
    contract."""
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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


def _mock_down_config(monkeypatch, project_name, extra=None):
    """Wire deploy_down's config load to a fixed config dict.

    ``extra`` is merged over the base dict, so a caller can add the config keys
    its own branch turns on (e.g. ``modules.web_terminals``).
    """
    raw_config = {"project_name": project_name, "deployed_services": ["event_dispatcher"]}
    raw_config.update(extra or {})
    monkeypatch.setattr(
        container_lifecycle,
        "load_project_config",
        lambda p, **kwargs: raw_config,
    )
    monkeypatch.setattr(
        "osprey.deployment.compose_generator.find_existing_compose_files",
        lambda *a, **k: ["docker-compose.yml"],
    )
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )


# deploy_down's own compose invocation — the pin it carries and the web-stack
# ordering around it — is covered against the captured-run seam in
# tests/deployment/test_down_conversion.py, which owns the conversion. What is
# only asserted here is the *negative* branch: a project with the module off
# must not touch the web stack at all.


def _capture_services_down_and_web_down(monkeypatch):
    """Record the services compose ``down`` and any deploy_down_web_terminals call.

    ``down`` no longer ``execvpe``-replaces the process; it runs through
    ``run_captured`` and exits on the child's code, so the seam to patch is
    ``container_lifecycle.run_captured`` and the caller must expect SystemExit.
    """
    captured: dict = {"web_down_order": None, "down_order": None}
    order = iter(range(100))

    def _fake_run_captured(cmd, **kwargs):
        captured.update(args=list(cmd), env=kwargs.get("env"), down_order=next(order))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle, "run_captured", _fake_run_captured)
    monkeypatch.setattr(
        container_lifecycle,
        "deploy_down_web_terminals",
        lambda config, env, env_file_args, **kwargs: captured.update(
            web_down_config=config, web_down_order=next(order)
        ),
    )
    return captured


def test_deploy_down_skips_web_stack_when_module_disabled(monkeypatch, tmp_path):
    """No modules.web_terminals.enabled → the plain services-only down, no
    web-stack invocation."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch, "myproj")
    captured = _capture_services_down_and_web_down(monkeypatch)

    with pytest.raises(SystemExit):
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
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda *a, **k: _FakeCompletedProcess()
    )
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
        compose_generator,
        "run_captured",
        lambda cmd, env=None, **k: envs.append(env) or _FakeCompletedProcess(),
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
        container_lifecycle.subprocess,
        "run",
        lambda cmd, env=None, **k: runs.append(cmd) or _FakeCompletedProcess(),
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: runs.append(cmd)),
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
        container_lifecycle.subprocess,
        "run",
        lambda cmd, env=None, **k: runs.append(cmd) or _FakeCompletedProcess(),
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: runs.append(cmd)),
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
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    calls: list = []

    def _fake_run(cmd, env=None, **k):
        calls.append(list(cmd))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    container_lifecycle.rebuild_deployment(str(tmp_path / "config.yml"))
    assert any(_addresses(c, "docker-compose.web.yml") and "up" in c for c in calls)


# ---------------------------------------------------------------------------
# Stale-container preflight — self-healing `osprey up`
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
        container_lifecycle.subprocess,
        "run",
        lambda cmd, env=None, **k: calls.append(list(cmd)) or _FakeCompletedProcess(),
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
    assert _addresses(rm_cmd, "docker-compose.yml")
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
    assert any(_addresses(c, "docker-compose.yml") for c in rm_calls)
    assert any(_addresses(c, "docker-compose.web.yml") for c in rm_calls)
    # The two stacks' files are never merged into one rm argv, and the
    # shared-project path never orphan-removes.
    for c in rm_calls:
        assert not (_addresses(c, "docker-compose.yml") and _addresses(c, "docker-compose.web.yml"))
    for c in cmds:
        assert "--remove-orphans" not in c


def test_web_services_dev_mode_splits_build_from_up(monkeypatch, tmp_path):
    """The web path's backend-services stack: --dev builds then ups --no-build,
    never `up --build` in one call. Needs a non-empty deployed_services (the
    services block is guarded on it), which captured_web_runs lacks."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.users").write_text("", encoding="utf-8")
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
    monkeypatch.setattr(provision, "enable_linger", lambda *a, **k: None)
    monkeypatch.setattr(provision, "seed_user_containers", lambda *a, **k: None)
    monkeypatch.setattr(provision, "run_verify_script", lambda *a, **k: None)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    runs: list = []

    def _fake_run(cmd, env=None, **k):
        runs.append(list(cmd))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: runs.append(cmd)),
    )
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=True)

    # The services-stack invocations (the ones carrying the services compose file).
    svc = [c for c in runs if _addresses(c, "docker-compose.yml")]
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
# `--progress plain` on the project image build, docker only. The live build
# view is parsed from BuildKit's plain stream; relying on `auto` degrading to
# plain under the capture pipe would make that parse depend on undocumented
# behaviour. Podman is excluded for the same reason `with_plain_progress`
# excludes it: `podman build` has no such flag and would fail the deploy.
# ---------------------------------------------------------------------------


def test_project_image_build_cmd_pins_plain_progress_on_docker():
    cmd = container_lifecycle._project_image_build_cmd(
        {"project_name": "myfacility"}, "docker", "/proj"
    )
    assert cmd[cmd.index("--progress") + 1] == "plain"
    assert cmd[-1] == "/proj"  # the flag lands ahead of the context, not after


def test_project_image_build_cmd_omits_plain_progress_on_podman():
    cmd = container_lifecycle._project_image_build_cmd(
        {"project_name": "myfacility"}, "podman", "/proj"
    )
    assert "--progress" not in cmd
    assert cmd[-1] == "/proj"


# ---------------------------------------------------------------------------
# The project image build is watched, labeled with its own tag. A single-image
# build's BuildKit headers name no service (`#10 [ 2/13]`), so an unlabeled
# model parses the whole build into nothing -- silently, with no error. The
# assertion is therefore behavioural: feed the watcher a nameless header and
# require the row to come back under the image tag.
# ---------------------------------------------------------------------------


def test_build_project_image_watches_the_build_under_its_image_tag(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", lambda config: ["docker"])
    calls = []
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **kwargs: calls.append(kwargs) or _FakeCompletedProcess(),
    )
    config = {"project_name": "myfacility", "deployed_services": ["dispatch_worker"]}

    container_lifecycle._build_project_image(config, dev_mode=False, env={})

    (kwargs,) = calls
    watcher = kwargs["on_line"]
    assert watcher is not None, "the project image build runs unwatched"
    watcher("#10 [ 2/13] RUN pip install --no-cache-dir osprey-framework")
    (row,) = watcher.model.snapshot()
    assert row.service == "myfacility:local"
    assert row.step == "2/13"


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
    # Stubbed at `run_captured`, not at `subprocess.run`: the build is watched
    # (`on_line=`), and a watched capture reads the child through a pipe rather
    # than writing it straight to the spool — so a `subprocess.run` stub would
    # be walked straight past and this test would launch a real `docker build`.
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **k: calls.append(cmd) or _FakeCompletedProcess(),
    )
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
    monkeypatch.setattr(
        container_lifecycle, "run_captured", lambda cmd, **k: _FakeCompletedProcess()
    )
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

    monkeypatch.setattr(container_lifecycle, "run_captured", _failing_build)
    config = {"project_name": "myfacility", "deployed_services": ["dispatch_worker"]}

    with pytest.raises(subprocess.CalledProcessError):
        container_lifecycle._build_project_image(config, dev_mode=True, env={})

    assert list(tmp_path.glob("*.whl")) == []
    assert not (tmp_path / "osprey-local-requirements.txt").exists()


# ---------------------------------------------------------------------------
# Staleness advisory + endpoint summary wiring
#
# Both features are advisory modules that only matter if deploy_up actually
# invokes them — an unwired check reproduces the silent-stale-deploy failure
# they exist to prevent, so the wiring itself is under test.
# ---------------------------------------------------------------------------


@pytest.fixture
def _wiring_calls(monkeypatch, tmp_path):
    """Stub deploy_up collaborators; record staleness/summary invocations."""
    calls: dict = {}

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
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda *a, **k: _FakeCompletedProcess()
    )
    monkeypatch.setattr(
        container_lifecycle,
        "warn_if_project_stale",
        lambda project_dir: calls.setdefault("stale", []).append(project_dir),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "log_endpoint_summary",
        lambda config, compose_files: calls.setdefault("summary", []).append(compose_files),
    )
    return calls


def test_deploy_up_runs_staleness_check(_wiring_calls, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _wiring_calls["stale"] == [tmp_path.resolve()]


def test_deploy_up_prints_endpoint_summary_detached(_wiring_calls, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _wiring_calls["summary"] == [["docker-compose.yml"]]


def test_deploy_up_prints_endpoint_summary_on_web_path(_wiring_calls, monkeypatch, tmp_path):
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"modules": {"web_terminals": {"enabled": True}}},
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "deploy_up_web_terminals", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "preflight_web_terminals", lambda *a, **k: None)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _wiring_calls["summary"] == [["docker-compose.yml"]]


def test_deploy_up_summarizes_even_when_nothing_deploys(_wiring_calls, monkeypatch, tmp_path):
    """The maximally-stale shape: no services, no web tier. The early return
    must still emit the summary so 'web terminal (not configured)' is seen."""
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": []}, ["docker-compose.yml"]),
    )
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _wiring_calls["stale"] == [tmp_path.resolve()]
    assert _wiring_calls["summary"] == [["docker-compose.yml"]]


# ---------------------------------------------------------------------------
# deploy_up ordering: the web-terminal preflight (persona render check +
# .env.users credential gate) must run BEFORE the expensive project-image
# build -- a missing provider secret aborts in seconds, not after minutes of
# docker build.
# ---------------------------------------------------------------------------


def test_deploy_up_runs_web_terminal_preflight_before_image_build(monkeypatch, tmp_path):
    order: list[str] = []

    web_config = {
        "deployed_services": [],
        "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (web_config, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    # The collect-all pass sits ahead of the preflight and probes the same
    # config; recorded rather than answered, so this test stays about ordering
    # and is not also asserting on the (empty) roster it declares.
    monkeypatch.setattr(
        container_lifecycle,
        "_collect_unmet_preconditions",
        lambda *a, **k: order.append("collect"),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "preflight_web_terminals",
        lambda *a, **k: order.append("preflight"),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda *a, **k: order.append("build_image"),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "deploy_up_web_terminals",
        lambda *a, **k: order.append("web_up"),
    )
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert order == ["collect", "preflight", "build_image", "web_up"]


def test_deploy_up_no_web_terminals_skips_preflight(monkeypatch, tmp_path):
    order: list[str] = []

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
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle,
        "preflight_web_terminals",
        lambda *a, **k: order.append("preflight"),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_build_project_image",
        lambda *a, **k: order.append("build_image"),
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda *a, **k: _FakeCompletedProcess()
    )
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert "preflight" not in order
    assert "build_image" in order


class TestResolvePipSpec:
    """The ``OSPREY_PIP_SPEC`` build arg must never name a nonexistent release.

    A hand-maintained version literal used to let this emit
    ``osprey-framework==<unreleased>``, which fails at pip resolve inside the
    image build with no indication of why.
    """

    def test_operator_override_wins(self, monkeypatch):
        monkeypatch.setenv("OSPREY_PIP_SPEC", "git+https://example.invalid/osprey@abc123")
        assert (
            container_lifecycle._resolve_pip_spec() == "git+https://example.invalid/osprey@abc123"
        )

    def test_release_pins_to_the_release(self, monkeypatch):
        monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
        monkeypatch.setattr("osprey.version.is_release", lambda: True)
        monkeypatch.setattr("osprey.version.get_release_version", lambda: "2026.6.2")

        assert container_lifecycle._resolve_pip_spec() == "osprey-framework==2026.6.2"

    def test_development_build_refuses_and_names_the_way_out(self, monkeypatch):
        from osprey.deployment.errors import UnreleasedVersionPinError

        monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
        monkeypatch.setattr("osprey.version.is_release", lambda: False)
        monkeypatch.setattr("osprey.version.get_release_version", lambda: "2026.6.2")
        monkeypatch.setattr(
            "osprey.version.get_running_version", lambda: "2026.6.2.post783+g83fda5e60"
        )

        with pytest.raises(UnreleasedVersionPinError) as excinfo:
            container_lifecycle._resolve_pip_spec()

        assert "783 commits past v2026.6.2" in excinfo.value.reason
        assert "--dev" in excinfo.value.remedy

    def test_override_still_wins_from_a_development_build(self, monkeypatch):
        """The escape hatch must not be gated behind being a release."""
        monkeypatch.setenv("OSPREY_PIP_SPEC", "osprey-framework==2026.6.2")
        monkeypatch.setattr("osprey.version.is_release", lambda: False)

        assert container_lifecycle._resolve_pip_spec() == "osprey-framework==2026.6.2"

    def test_dev_mode_does_not_refuse(self, monkeypatch):
        """``--dev`` is used *from* a development checkout — it must not refuse.

        The staged wheel is what the Dockerfile installs, so this spec is inert.
        Gating it here would block the workflow the refusal message recommends.
        """
        monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
        monkeypatch.setattr("osprey.version.is_release", lambda: False)
        monkeypatch.setattr("osprey.version.get_release_version", lambda: "2026.6.2")

        assert container_lifecycle._resolve_pip_spec(dev_mode=True) == "osprey-framework==2026.6.2"

    def test_dev_run_whose_wheel_staging_failed_still_refuses(self, monkeypatch):
        """Callers pass the *effective* dev mode, which is False when staging failed.

        That build installs from PyPI after all, so it must not quietly ship
        released code in place of the checkout `--dev` was asked to test.
        """
        from osprey.deployment.errors import UnreleasedVersionPinError

        monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
        monkeypatch.setattr("osprey.version.is_release", lambda: False)
        monkeypatch.setattr("osprey.version.get_release_version", lambda: "2026.6.2")
        monkeypatch.setattr(
            "osprey.version.get_running_version", lambda: "2026.6.2.post783+g83fda5e60"
        )

        with pytest.raises(UnreleasedVersionPinError):
            container_lifecycle._resolve_pip_spec(dev_mode=False)


class TestResolvePrebuiltImages:
    """Some deployment hosts cannot build images at all.

    No build tooling, no reachable registry, images side-loaded from a tarball
    instead. There a dev deploy's ``compose build`` is not slow but impossible,
    and the only thing that can run is an ``up`` against tags already present.
    """

    def test_the_default_is_to_build(self, monkeypatch):
        monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)
        assert container_lifecycle._resolve_prebuilt_images({}) is False

    @pytest.mark.parametrize("value", sorted(container_lifecycle._TRUTHY))
    def test_every_on_spelling_is_accepted(self, monkeypatch, value):
        """Operators reach for whichever word the rest of the framework took.

        A spelling that parsed as "off" would start a build on a host that
        cannot run one, and the failure would name compose rather than the
        variable that was misread.
        """
        monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", value)
        assert container_lifecycle._resolve_prebuilt_images({}) is True

    @pytest.mark.parametrize("value", sorted(container_lifecycle._FALSY))
    def test_every_off_spelling_overrides_a_config_that_says_prebuilt(self, monkeypatch, value):
        """The env var wins in both directions, not just when it says "on".

        A one-way override would leave a config-pinned host unable to prove its
        build works again without an edit someone has to remember to revert.
        """
        monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", value)
        assert container_lifecycle._resolve_prebuilt_images({"prebuilt_images": True}) is False

    @pytest.mark.parametrize("value", [" true ", "TRUE", "\tYes\n", "On"])
    def test_case_and_surrounding_whitespace_do_not_change_the_answer(self, monkeypatch, value):
        """A value pasted out of a shell script keeps its padding."""
        monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", value)
        assert container_lifecycle._resolve_prebuilt_images({}) is True

    def test_the_config_key_turns_the_switch_on(self, monkeypatch):
        monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)
        assert container_lifecycle._resolve_prebuilt_images({"prebuilt_images": True}) is True

    def test_the_config_key_can_also_spell_out_the_default(self, monkeypatch):
        monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)
        assert container_lifecycle._resolve_prebuilt_images({"prebuilt_images": False}) is False

    def test_env_on_overrides_a_config_that_says_build(self, monkeypatch):
        monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "yes")
        assert container_lifecycle._resolve_prebuilt_images({"prebuilt_images": False}) is True

    @pytest.mark.parametrize("value", ["", "   ", "maybe", "2", "yes please"])
    def test_an_unreadable_value_defers_to_the_config(self, monkeypatch, value):
        """An empty or unrecognized export must not silently mean "off".

        Reading it as a decision would let a stray ``OSPREY_PREBUILT_IMAGES=``
        in an env file quietly re-enable building on a host whose config
        already said it cannot build.
        """
        monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", value)
        assert container_lifecycle._resolve_prebuilt_images({"prebuilt_images": True}) is True
        assert container_lifecycle._resolve_prebuilt_images({}) is False


# ---------------------------------------------------------------------------
# The prebuilt-images switch at the two dev-mode build sites
# ---------------------------------------------------------------------------


def _dev_deploy_cmds(
    monkeypatch, tmp_path, config_extra: dict | None = None, dev_mode: bool = True
) -> list[list[str]]:
    """Every argv a deploy runs down the plain (no web) path, ``--dev`` by default."""
    cmds: list[list[str]] = []
    config = {"deployed_services": ["event_dispatcher"], **(config_extra or {})}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    # The project image is built by its own helper, outside the compose build
    # this switch gates; leaving it live would add argv that says nothing about
    # which branch ran.
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)

    def _fake_run(cmd, env=None, check=False, **kwargs):
        cmds.append(list(cmd))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "Popen",
        _fake_popen(lambda cmd, env: cmds.append(list(cmd))),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=dev_mode)
    return cmds


def _compose_builds(cmds: list[list[str]]) -> list[list[str]]:
    return [cmd for cmd in cmds if cmd[-1] == "build" and _addresses(cmd, "docker-compose.yml")]


def test_a_dev_deploy_builds_the_service_images_by_default(monkeypatch, tmp_path):
    """The baseline the switch has to leave alone."""
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path)

    assert len(_compose_builds(cmds)) == 1


def test_the_switch_removes_the_services_build_from_a_dev_deploy(monkeypatch, tmp_path):
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path)

    assert _compose_builds(cmds) == []


def test_the_config_key_removes_it_too(monkeypatch, tmp_path):
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path, {"prebuilt_images": True})

    assert _compose_builds(cmds) == []


def test_env_off_restores_the_build_a_prebuilt_config_would_have_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "false")

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path, {"prebuilt_images": True})

    assert len(_compose_builds(cmds)) == 1


def test_the_up_still_carries_no_build_when_the_build_was_skipped(monkeypatch, tmp_path):
    """``up --no-build`` is what makes the skip safe on a host that cannot build.

    Without it compose would build on ``up`` instead, moving the failure rather
    than removing it.
    """
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path)

    up = next(cmd for cmd in cmds if "up" in cmd)
    assert "--no-build" in up
    assert "--build" not in up


def test_the_web_terminals_path_skips_its_services_build_too(
    captured_combined_runs, monkeypatch, tmp_path
):
    """A web-terminals deploy builds the backend services on its own path.

    That second build site is the one a web-terminals host actually hits, so a
    switch wired only into the plain path would look correct in tests and still
    fail the deployment it was written for.
    """
    steps: list[str] = []
    monkeypatch.setattr(provision, "_report_step", steps.append)
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=True)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"]]
    assert _compose_builds(cmds) == []
    assert "skipped image build (prebuilt images)" in steps
    assert "built service images" not in steps


def test_the_web_terminals_path_builds_when_the_switch_is_off(
    captured_combined_runs, monkeypatch, tmp_path
):
    steps: list[str] = []
    monkeypatch.setattr(provision, "_report_step", steps.append)
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "off")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False, dev_mode=True)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"]]
    assert len(_compose_builds(cmds)) == 1
    assert "built service images" in steps
    assert "skipped image build (prebuilt images)" not in steps


def test_a_non_dev_up_omits_no_build_unless_the_images_are_prebuilt(monkeypatch, tmp_path):
    """The baseline the non-dev half of the switch has to leave alone.

    A non-dev deploy has no build step of its own, and a build-only service
    with no published upstream tag reaches its image solely through compose's
    implicit build-on-up. Suppressing that unconditionally would break every
    ordinary deploy.
    """
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path, dev_mode=False)

    up = next(cmd for cmd in cmds if "up" in cmd)
    assert "--no-build" not in up


def test_the_switch_adds_no_build_to_a_non_dev_up(monkeypatch, tmp_path):
    """A pull-only mirror deploy must fail on a missing image, not build one.

    Non-dev keeps no build step to skip, so compose's implicit build-on-up is
    the whole of what the switch has to suppress here: without ``--no-build``
    a tag the mirror never delivered would be answered by a locally-tagged
    impostor built from the template's ``build:`` block.
    """
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path, dev_mode=False)

    up = next(cmd for cmd in cmds if "up" in cmd)
    assert "--no-build" in up
    assert "--build" not in up


def test_the_config_key_adds_no_build_to_a_non_dev_up_too(monkeypatch, tmp_path):
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path, {"prebuilt_images": True}, dev_mode=False)

    up = next(cmd for cmd in cmds if "up" in cmd)
    assert "--no-build" in up


def test_env_off_restores_the_implicit_build_a_prebuilt_config_would_have_suppressed(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "false")

    cmds = _dev_deploy_cmds(monkeypatch, tmp_path, {"prebuilt_images": True}, dev_mode=False)

    up = next(cmd for cmd in cmds if "up" in cmd)
    assert "--no-build" not in up


def test_the_web_terminals_services_up_gets_no_build_in_non_dev_too(
    captured_combined_runs, monkeypatch, tmp_path
):
    """The second implicit-build site, wired for the same reason as the first.

    A web-terminals host reaches its backend services through this invocation
    and never through the plain path, so a switch wired only there would look
    correct in tests and still build an impostor on the deployment it was
    written for. The web-terminal stack's own ``up`` is untouched: its images
    are registry-hosted and it has no ``build:`` block to suppress.
    """
    steps: list[str] = []
    monkeypatch.setattr(provision, "_report_step", steps.append)
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"] if "up" in c["cmd"]]
    services_up = next(c for c in cmds if _addresses(c, "docker-compose.yml"))
    web_up = next(c for c in cmds if _addresses(c, "docker-compose.web.yml"))
    assert "--no-build" in services_up
    assert "--no-build" not in web_up
    # Non-dev has no build step to report skipping — that line stays dev-only.
    assert "skipped image build (prebuilt images)" not in steps


def test_the_web_terminals_services_up_omits_no_build_when_prebuilt_is_off(
    captured_combined_runs, monkeypatch, tmp_path
):
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "off")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    cmds = [c["cmd"] for c in captured_combined_runs["calls"] if "up" in c["cmd"]]
    services_up = next(c for c in cmds if _addresses(c, "docker-compose.yml"))
    assert "--no-build" not in services_up


# ---------------------------------------------------------------------------
# The prebuilt-images switch at the PROJECT image build
# ---------------------------------------------------------------------------
# The dispatch worker's image has no compose `build:` block, so it is built by
# `_build_project_image` — a `<runtime> build` of this repo, outside every
# compose invocation the switch was originally wired into. A prebuilt host runs
# that build with no build tooling; worse, under an `images.registry` the tag it
# builds is the registry-qualified one, so a host that PULLED the worker image
# would have it overwritten by a local rebuild wearing the same name.

_WORKER_CONFIG = {"project_name": "myfacility", "deployed_services": ["dispatch_worker"]}
_MIRRORED_WORKER_CONFIG = {
    "project_name": "myfacility",
    "deployed_services": ["dispatch_worker"],
    "images": {"registry": "registry.example.org/physics", "tag": "2026.6.2"},
}


def _project_image_build_calls(monkeypatch, tmp_path, config, *, dev_mode=False):
    """Every argv ``_build_project_image`` runs, plus the steps it reported."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "get_runtime_command", lambda config: ["docker"])
    cmds: list[list[str]] = []
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **kwargs: cmds.append(list(cmd)) or _FakeCompletedProcess(),
    )
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)
    monkeypatch.setattr(container_lifecycle, "_report_group", lambda name: None)

    container_lifecycle._build_project_image(config, dev_mode=dev_mode, env={})
    return cmds, steps


def test_the_project_image_is_built_when_the_switch_is_off(monkeypatch, tmp_path):
    """The baseline the switch has to leave alone."""
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)

    cmds, _steps = _project_image_build_calls(monkeypatch, tmp_path, _WORKER_CONFIG)

    assert len(cmds) == 1 and cmds[0][:2] == ["docker", "build"]


def test_the_switch_removes_the_project_image_build(monkeypatch, tmp_path):
    """A host that cannot build must not be handed a `docker build` anyway."""
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    cmds, steps = _project_image_build_calls(monkeypatch, tmp_path, _WORKER_CONFIG)

    assert cmds == []
    assert "skipped image build (prebuilt images)" in steps


def test_the_config_key_removes_the_project_image_build_too(monkeypatch, tmp_path):
    """The switch is a property of the deployment as well as of the shell."""
    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)

    cmds, steps = _project_image_build_calls(
        monkeypatch, tmp_path, {**_WORKER_CONFIG, "prebuilt_images": True}
    )

    assert cmds == []
    assert "skipped image build (prebuilt images)" in steps


def test_a_pulled_registry_image_is_never_rebuilt_over(monkeypatch, tmp_path):
    """The failure the switch prevents on a host that CAN build.

    With an ``images.registry`` axis the project image's tag is the registry
    reference the deploy pulled. Building it locally does not merely waste
    minutes — it replaces the published image with a locally assembled one
    carrying the same name, and nothing downstream can tell the two apart.
    """
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "yes")

    cmds, _steps = _project_image_build_calls(monkeypatch, tmp_path, _MIRRORED_WORKER_CONFIG)

    assert cmds == []


def test_env_off_restores_the_project_image_build_a_prebuilt_config_would_have_skipped(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "false")

    cmds, _steps = _project_image_build_calls(
        monkeypatch, tmp_path, {**_WORKER_CONFIG, "prebuilt_images": True}
    )

    assert len(cmds) == 1


def test_a_dev_run_on_a_prebuilt_host_builds_no_project_image_either(monkeypatch, tmp_path):
    """``--dev`` is the mode that stages a wheel and builds — and the one a
    build-tool-less host cannot run at all."""
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    cmds, _steps = _project_image_build_calls(monkeypatch, tmp_path, _WORKER_CONFIG, dev_mode=True)

    assert cmds == []


def test_the_skip_line_is_only_reported_where_a_build_was_actually_taken_away(
    monkeypatch, tmp_path
):
    """A deployment without the worker had no project image build to skip."""
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    cmds, steps = _project_image_build_calls(
        monkeypatch, tmp_path, {"project_name": "myfacility", "deployed_services": ["mongodb"]}
    )

    assert cmds == []
    assert steps == []


def test_the_switch_answers_the_build_target_question_too(monkeypatch):
    """`_project_image_build_target` is what the preflight probe reads.

    It reports the unreleased-pin refusal as a DEFINITE one — a fact about a
    build that is going to happen. On a prebuilt host no build happens, so a
    refusal there would send the operator to fix something that never runs.
    """
    monkeypatch.delenv("OSPREY_PIP_SPEC", raising=False)
    monkeypatch.setattr("osprey.version.is_release", lambda: False)
    monkeypatch.setattr("osprey.version.get_release_version", lambda: "2026.6.2")
    monkeypatch.setattr("osprey.version.get_running_version", lambda: "2026.6.2.post783+g83fda5e60")

    monkeypatch.delenv("OSPREY_PREBUILT_IMAGES", raising=False)
    assert container_lifecycle._project_image_build_target(_WORKER_CONFIG, {}) is not None
    assert container_lifecycle._unreleased_pin_problem(_WORKER_CONFIG, {}, dev_mode=False)

    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")
    assert container_lifecycle._project_image_build_target(_WORKER_CONFIG, {}) is None
    assert container_lifecycle._unreleased_pin_problem(_WORKER_CONFIG, {}, dev_mode=False) is None


# ---------------------------------------------------------------------------
# Staged archiver bring-up
# ---------------------------------------------------------------------------


class _FakeAdmin:
    """``client.admin``, answering ``ping`` only after ``fail_times`` refusals."""

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.pings = 0

    def command(self, name):
        self.pings += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            from pymongo.errors import ServerSelectionTimeoutError

            raise ServerSelectionTimeoutError("store not up yet")
        return {"ok": 1}


class _FakeCollection:
    """Just enough of a pymongo collection for the staged bring-up's own calls."""

    def __init__(self, fail_pings: int = 0):
        self.admin = _FakeAdmin(fail_pings)
        self.dropped = False
        # collection.database.client.admin is the path the health poll walks.
        self.database = type("_Db", (), {"client": type("_Client", (), {"admin": self.admin})()})()

    def drop(self):
        self.dropped = True


ARCHIVER_CONFIG = {
    "deployed_services": ["mongodb", "archiver_recorder", "virtual_accelerator"],
    "services": {"mongodb": {"compression": "zstd", "port_host": 27017}},
    "va_archiver": {
        "retention_days": 1,
        "hot_span_hours": 1,
        "hot_cadence_sec": 10,
        "tail_cadence_sec": 60,
    },
    "archiver": {
        "mongodb_archiver": {
            "host": "localhost",
            "port": 27017,
            "name": "osprey_archiver",
            "collection": "pv_history",
            "auth": "admin",
            "username": "osprey",
            "password_env": "MONGO_ROOT_PASSWORD",
            "timeout": 5,
        }
    },
}


@pytest.fixture
def staged_archiver(monkeypatch, tmp_path):
    """Drive ``deploy_up`` for an archiver project with every store call faked.

    Records the compose argv in order (so the *sequence* of staged store, quiesce
    and full bring-up is assertable) plus what the seeder was asked to do.
    """
    from osprey.simulation import apply as apply_mod
    from osprey.simulation import archiver_seed

    state: dict = {
        "cmds": [],
        "collection": _FakeCollection(),
        "seeded": [],
        "reapplied": [],
        "fingerprint_state": archiver_seed.SeedState.ABSENT,
        "differences": (),
        "returncode": 0,
    }

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=s3cret\n")

    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (dict(ARCHIVER_CONFIG), ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda *a, **k: None)

    def _fake_run(cmd, env=None, check=False, **kwargs):
        state["cmds"].append(list(cmd))
        # A real CompletedProcess, because the quiesce checks its returncode.
        # `returncode` models the *quiesce* specifically: every other compose
        # call runs under check=True, so a blanket non-zero would abort the
        # deploy before the behaviour under test is reached.
        rc = state["returncode"] if "stop" in cmd else 0
        return subprocess.CompletedProcess(list(cmd), rc)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)

    # The seed inputs are a manifest read plus a machine-model load; both are
    # exercised by their own module's tests, and neither belongs in an argv test.
    monkeypatch.setattr(
        container_lifecycle,
        "_archiver_seed_inputs",
        lambda config, project_dir: ([{"address": "SR:BPM1:X"}], None, {}, None, None),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_reapply_active_scenarios",
        lambda config, project_dir, engine: state["reapplied"].append(project_dir),
    )

    @contextmanager
    def _fake_collection(store):
        state["store"] = store
        yield state["collection"]

    monkeypatch.setattr(apply_mod, "archiver_collection", _fake_collection)
    monkeypatch.setattr(
        archiver_seed,
        "compare_fingerprint",
        lambda collection, fingerprint: archiver_seed.FingerprintComparison(
            state["fingerprint_state"], state["differences"]
        ),
    )

    def _fake_seed_base(collection, channels, knobs, **kwargs):
        state["seeded"].append({"channels": list(channels), "knobs": knobs, "kwargs": kwargs})
        # The staged step reports on what it wrote, so hand back a real report.
        return archiver_seed.SeedReport(documents=10, channels=len(channels))

    monkeypatch.setattr(archiver_seed, "seed_base", _fake_seed_base)
    return state


def _verbs(cmds):
    """The compose verb (plus service argument) of each recorded invocation.

    Strips the runtime command and the leading option pairs — the invocation
    contract's ``--project-directory``, plus ``-f``, ``--env-file`` and
    ``--progress`` — which are the same on every invocation and only obscure
    the ordering under test.
    """
    verbs = []
    for cmd in cmds:
        rest = list(cmd)
        while rest and rest[0] in ("docker", "compose", "podman"):
            rest.pop(0)
        while rest and rest[0] in ("-f", "--env-file", "--progress", "--project-directory"):
            del rest[:2]
        verbs.append(" ".join(rest))
    return verbs


def test_staged_store_comes_up_before_the_full_bring_up(staged_archiver, tmp_path):
    """The store is started alone first: the recorder must never race the seeder
    into creating the collection with the wrong indexes."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    verbs = _verbs(staged_archiver["cmds"])
    staged = verbs.index("up -d mongodb")
    full = verbs.index("up --remove-orphans -d")
    assert staged < full


def test_reseed_quiesces_the_recorder_before_dropping_the_collection(staged_archiver, tmp_path):
    """The recorder writes into the collection being dropped, so it is stopped
    first. The following `up` restores it — nothing here has to undo the stop."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    verbs = _verbs(staged_archiver["cmds"])
    assert verbs.index("stop archiver-recorder") < verbs.index("up --remove-orphans -d")
    assert staged_archiver["collection"].dropped
    assert len(staged_archiver["seeded"]) == 1


def test_absent_fingerprint_seeds_after_a_volume_wipe(staged_archiver, tmp_path):
    """`clean`/`rebuild` remove the volume, leaving no manifest — the same path a
    first deploy takes, which is what makes a wiped store re-seed."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert len(staged_archiver["seeded"]) == 1
    assert staged_archiver["reapplied"] == [tmp_path]


def test_matching_fingerprint_skips_the_seed(staged_archiver, tmp_path):
    """Unchanged knobs mean the stored archive already describes this profile:
    nothing is dropped, written, or re-applied, and the recorder keeps running."""
    from osprey.simulation.archiver_seed import SeedState

    staged_archiver["fingerprint_state"] = SeedState.MATCH
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    verbs = _verbs(staged_archiver["cmds"])
    assert "up -d mongodb" in verbs
    assert "stop archiver-recorder" not in verbs
    assert staged_archiver["seeded"] == []
    assert staged_archiver["reapplied"] == []
    assert not staged_archiver["collection"].dropped


def test_mismatched_fingerprint_rebuilds_and_reapplies(staged_archiver, tmp_path):
    """Changed knobs make the stored coverage wrong, so the base is rebuilt and
    the active scenario set re-applied onto it — otherwise the deployment would
    claim a fault whose history it had just erased."""
    from osprey.simulation.archiver_seed import SeedState

    staged_archiver["fingerprint_state"] = SeedState.MISMATCH
    staged_archiver["differences"] = (("retention_days", 30, 1),)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert staged_archiver["collection"].dropped
    assert len(staged_archiver["seeded"]) == 1
    assert staged_archiver["reapplied"] == [tmp_path]


@pytest.mark.parametrize("state", ["ABSENT", "MISMATCH"])
def test_drop_and_rebuild_always_implies_a_quiesced_recorder(staged_archiver, tmp_path, state):
    """The invariant, stated once for every path that reaches it: a rebuild of
    the base implies a quiesced recorder.

    Not "mismatch quiesces and absent does not" — that conditional would leave
    the absent path letting a writer run into a collection being dropped
    underneath it. Stopping a service that was never started is a no-op, so the
    unconditional rule is both simpler and strictly safer.
    """
    from osprey.simulation.archiver_seed import SeedState

    staged_archiver["fingerprint_state"] = getattr(SeedState, state)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    verbs = _verbs(staged_archiver["cmds"])
    assert "stop archiver-recorder" in verbs
    assert staged_archiver["collection"].dropped


@pytest.mark.parametrize(
    ("state", "expect_seed"),
    [("MISMATCH", False), ("ABSENT", True)],
)
def test_keep_archiver_base_suppresses_only_the_mismatch_rebuild(
    staged_archiver, tmp_path, state, expect_seed
):
    """The flag keeps an EXISTING base; it does not mean "never seed".

    On a mismatch it preserves recorded history at the cost of honesty about the
    new knobs. On an absent manifest — a first deploy, or a wiped volume — there
    is no base to keep, so the seed must still happen; otherwise the flag would
    silently leave the deployment with no history at all. This is also why
    `rebuild` needs no flag of its own: it always lands on the absent path.
    """
    from osprey.simulation.archiver_seed import SeedState

    staged_archiver["fingerprint_state"] = getattr(SeedState, state)
    staged_archiver["differences"] = (("retention_days", 30, 1),)
    container_lifecycle.deploy_up(
        str(tmp_path / "config.yml"), detached=True, keep_archiver_base=True
    )

    assert bool(staged_archiver["seeded"]) is expect_seed
    assert staged_archiver["collection"].dropped is expect_seed
    assert ("stop archiver-recorder" in _verbs(staged_archiver["cmds"])) is expect_seed


def test_seeder_authenticates_with_the_project_dotenv_password(staged_archiver, tmp_path):
    """Read from the project's own `.env` by name — never from whatever the
    ambient environment happens to export for another deployment's store."""
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert staged_archiver["store"]["password"] == "s3cret"


def test_deploy_without_the_store_service_stages_nothing(captured_argv, monkeypatch, tmp_path):
    """A project that reads a store someone else runs must never have its history
    seeded by a local deploy."""
    staged: list = []
    monkeypatch.setattr(
        container_lifecycle, "_stage_archiver_store", lambda *a, **k: staged.append(a)
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert staged == []


def test_pymongo_preflight_aborts_before_any_container_work(monkeypatch, tmp_path):
    """Naming the install in seconds beats an ImportError after a minutes-long
    image build, so the check sits beside the token mint, not at the seeder."""
    import sys

    ran: list = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (dict(ARCHIVER_CONFIG), ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "_build_project_image", lambda *a, **k: ran.append("build")
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess, "run", lambda *a, **k: ran.append("compose")
    )
    # A None entry in sys.modules is what an incomplete environment looks like to
    # `import pymongo` — the import machinery raises ImportError without touching
    # the installed package.
    monkeypatch.setitem(sys.modules, "pymongo", None)

    with pytest.raises(ArchiverClientMissingError) as exc_info:
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert ran == []
    assert "pip install --upgrade osprey-framework" in exc_info.value.remedy


def test_pymongo_preflight_refusal_names_the_interpreter(monkeypatch):
    """The refusal has to say WHICH environment is missing pymongo.

    A `dependencies:` entry in the build profile installs into the project's
    `build/.venv`, its recorded pyproject.toml and its image — never into the
    interpreter running the CLI, which is where the seeder runs. An operator who
    has declared pymongo and still sees this refusal has already tried the
    obvious lever, so the reason names `sys.executable` and says that entry is
    not the one, rather than repeating "pymongo is not installed" at them.
    """
    import sys

    monkeypatch.setitem(sys.modules, "pymongo", None)

    with pytest.raises(ArchiverClientMissingError) as exc_info:
        container_lifecycle._preflight_archiver_pymongo(dict(ARCHIVER_CONFIG))

    reason = exc_info.value.reason
    assert sys.executable in reason
    assert "build/.venv" in reason
    assert "dependencies:" in reason


def test_pymongo_preflight_refusal_is_a_precondition_not_a_bug():
    """It must leave `up` through the precondition handler, not the catch-all.

    `deploy_cmd.up_verb` renders `DeploymentPreconditionError` as a refusal with
    a remedy and renders everything else as "Deployment failed". A bare
    RuntimeError here would be indistinguishable from a genuine bug — which is
    exactly what DeploymentPreconditionError's own docstring forbids.
    """
    assert issubclass(ArchiverClientMissingError, DeploymentPreconditionError)


def test_pymongo_preflight_is_silent_without_the_store_service():
    """No archiver in the deploy, no dependency to demand."""
    container_lifecycle._preflight_archiver_pymongo({"deployed_services": ["event_dispatcher"]})


def test_health_poll_returns_once_the_store_answers(monkeypatch):
    """A store on a fresh volume creates its admin user before it accepts
    connections, so early refusals are expected rather than fatal."""
    monkeypatch.setattr(container_lifecycle.time, "sleep", lambda seconds: None)
    collection = _FakeCollection(fail_pings=3)

    container_lifecycle._wait_for_archiver_store(
        collection, time.monotonic() + container_lifecycle._ARCHIVER_HEALTH_TIMEOUT_S
    )
    assert collection.admin.pings == 4


def test_health_poll_is_bounded(monkeypatch):
    """An unreachable store fails the deploy with the store named, rather than
    leaving `osprey up` polling forever."""
    monkeypatch.setattr(container_lifecycle.time, "sleep", lambda seconds: None)
    collection = _FakeCollection(fail_pings=10_000)

    with pytest.raises(RuntimeError, match="did not become reachable"):
        container_lifecycle._wait_for_archiver_store(collection, time.monotonic() - 1)


def test_missing_store_password_aborts_with_the_variable_named(
    staged_archiver, monkeypatch, tmp_path
):
    """Without the credential the store is created with, the seeder cannot open
    the store it is staging — and the fix is a named variable."""
    (tmp_path / ".env").write_text("")
    monkeypatch.delenv("MONGO_ROOT_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="MONGO_ROOT_PASSWORD"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


def test_reapply_anchors_on_the_persisted_t0_not_a_fresh_one(monkeypatch, tmp_path):
    """The re-apply must land on the anchor the running world is already on.

    Minting a fresh T0 would slide the live VA's events, the seeded logbook and
    the archive's event windows to an instant nobody asked for, as a side effect
    of a deploy that was only supposed to rebuild the store. Reading the anchor
    is `osprey.simulation.apply.persisted_scenario_anchor`'s job (and is tested
    there); what this pins is that the deploy passes what it read straight
    through to `apply_scenarios` as `now`.
    """
    from datetime import UTC, datetime

    from osprey.simulation import apply as apply_mod

    anchor = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    forwarded: dict = {}

    def _record(project_dir, names, **kwargs):
        forwarded.update(names=names, **kwargs)
        return apply_mod.ApplyResult(active=tuple(names), logbook_seeded=0, purged=False)

    monkeypatch.setattr(apply_mod, "persisted_scenario_anchor", lambda config, project_dir: anchor)
    monkeypatch.setattr(apply_mod, "apply_scenarios", _record)
    engine = type("_Engine", (), {"active_scenarios": lambda self: ("nominal", "rf-thermal")})()

    container_lifecycle._reapply_active_scenarios({}, tmp_path, engine)

    assert forwarded["now"] == anchor
    # The logbook is a knob change's business only if the narrative changed, and
    # it did not — purging ARIEL here would destroy history nobody asked to touch.
    assert forwarded["seed_logbook"] is False


def _manifest_channel(address: str) -> dict:
    """One manifest entry carrying the full per-channel schema the loader demands."""
    return {
        "address": address,
        "ring": "SR",
        "system": "diagnostics",
        "family": "BPM",
        "device": "1",
        "field": "X",
        "subfield": "",
        "partition": "static-noisy",
        "record_type": "ai",
        "noise": 0.01,
    }


def test_seed_inputs_read_the_manifest_the_project_env_names(tmp_path):
    """The seeded channel set is the manifest the VA and the recorder read, found
    the way they find it — so seeded history covers exactly what the live half
    serves rather than another facility's namespace.

    The manifest lives in the RENDER's data dir (``build/data/simulation``):
    the build generates it only there, and that is the directory the containers
    mount as ``/data/simulation`` — the source ``data/`` zone never holds it."""
    simulation_dir = tmp_path / "build" / "data" / "simulation"
    simulation_dir.mkdir(parents=True)
    (simulation_dir / "channel_manifest.json").write_text(
        json.dumps({"channels": [_manifest_channel("SR:BPM1:X")]})
    )
    (tmp_path / ".env").write_text("VA_CHANNELS_FILE=channel_manifest.json\n")

    channels, engine, boot_values, _, _ = container_lifecycle._archiver_seed_inputs({}, tmp_path)

    assert [c["address"] for c in channels] == ["SR:BPM1:X"]
    # No machine model in this project: every channel is procedural, which is a
    # valid configuration rather than a fault.
    assert engine is None
    assert boot_values == {}


def test_reapply_without_a_machine_model_is_a_no_op(tmp_path):
    """A store-only project has no scenarios to restore onto the rebuilt base."""
    container_lifecycle._reapply_active_scenarios({}, tmp_path, None)


def test_exported_password_is_used_when_the_dotenv_has_none(staged_archiver, monkeypatch, tmp_path):
    """`osprey up` is the process that hands compose its environment.

    When the password is exported rather than written to `.env`, the exported
    value is what the store container is created with — so it is also what the
    seeder must authenticate with. (`sim apply` deliberately does NOT do this: it
    runs from anywhere and must not pick up a foreign deployment's credential.)
    """
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("MONGO_ROOT_PASSWORD", "exported-not-written")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert staged_archiver["store"]["password"] == "exported-not-written"


def test_store_without_a_connection_block_seeds_nothing(
    staged_archiver, monkeypatch, tmp_path, caplog
):
    """A project deploying the store but declaring no connection block cannot be
    seeded. It says so and leaves the store to the normal bring-up rather than
    guessing a host — an empty archive read honestly beats a wrong one."""
    config = {key: value for key, value in ARCHIVER_CONFIG.items() if key != "archiver"}
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )

    with caplog.at_level(logging.WARNING):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert "archiver.mongodb_archiver" in caplog.text
    verbs = _verbs(staged_archiver["cmds"])
    assert "up -d mongodb" not in verbs
    assert staged_archiver["seeded"] == []
    assert not staged_archiver["collection"].dropped


def test_authentication_is_retried_while_a_fresh_volume_initializes(monkeypatch):
    """A FRESH volume refuses authentication before it has created its root user.

    mongod accepts connections before it applies MONGO_INITDB_ROOT_PASSWORD, so a
    probe landing in that window is refused by a store that is about to be
    perfectly healthy. Treating that as terminal aborts the FIRST deploy of every
    new project — intermittently, depending on which side of the race the probe
    lands, which is the worst way for it to fail.
    """
    from pymongo.errors import OperationFailure

    monkeypatch.setattr(container_lifecycle.time, "sleep", lambda seconds: None)
    collection = _FakeCollection()
    refusals = [3]

    def _initializing(name):
        collection.admin.pings += 1
        if refusals[0] > 0:
            refusals[0] -= 1
            raise OperationFailure("Authentication failed.", code=18)
        return {"ok": 1}

    collection.admin.command = _initializing

    container_lifecycle._wait_for_archiver_store(
        collection,
        time.monotonic() + container_lifecycle._ARCHIVER_HEALTH_TIMEOUT_S,
        "osprey@localhost:27017",
    )
    assert collection.admin.pings == 4


def test_authentication_failure_past_the_grace_window_fails_with_the_cause(monkeypatch):
    """After the store has had time to initialize, the SAME error means the
    stale-volume shape instead: a rotated password against a volume that keeps
    the credentials it was created with. That will never succeed, so it fails
    with the cause named rather than consuming the full reachability budget."""
    from pymongo.errors import OperationFailure

    monkeypatch.setattr(container_lifecycle.time, "sleep", lambda seconds: None)
    # Collapse the grace window so the refusal is read as final immediately.
    monkeypatch.setattr(container_lifecycle, "_ARCHIVER_AUTH_GRACE_S", 0.0)
    collection = _FakeCollection()

    def _refuse(name):
        collection.admin.pings += 1
        raise OperationFailure("Authentication failed.", code=18)

    collection.admin.command = _refuse

    with pytest.raises(RuntimeError, match="rejected the credentials"):
        container_lifecycle._wait_for_archiver_store(
            collection,
            time.monotonic() + container_lifecycle._ARCHIVER_HEALTH_TIMEOUT_S,
            "osprey@localhost:27017",
        )
    # Terminal on the first refusal once the grace window has passed — it will
    # not start working, and the full budget would only delay the diagnosis.
    assert collection.admin.pings == 1


def test_a_non_auth_operation_failure_is_not_swallowed(monkeypatch):
    """Only AuthenticationFailed gets the grace/stale-volume treatment; any other
    server-side command failure propagates as itself rather than being retimed
    into a credentials message that would misdirect the reader."""
    from pymongo.errors import OperationFailure

    monkeypatch.setattr(container_lifecycle.time, "sleep", lambda seconds: None)
    collection = _FakeCollection()

    def _fail(name):
        raise OperationFailure("not authorized on admin", code=13)

    collection.admin.command = _fail

    with pytest.raises(OperationFailure, match="not authorized"):
        container_lifecycle._wait_for_archiver_store(
            collection, time.monotonic() + 5, "osprey@localhost:27017"
        )


def test_a_failed_reapply_names_the_command_that_fixes_it(monkeypatch, tmp_path):
    """The manifest is already written by this point, so the next deploy reads
    MATCH and skips both the reseed and this step — leaving a clean base under a
    faulted machine forever. The error has to hand over the one command that
    repairs it, because retrying the deploy will not."""
    from osprey.simulation import apply as apply_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("store went away")

    monkeypatch.setattr(apply_mod, "apply_scenarios", _boom)
    engine = type("_Engine", (), {"active_scenarios": lambda self: ("nominal", "rf-thermal")})()

    with pytest.raises(RuntimeError, match=r"osprey sim apply rf-thermal") as caught:
        container_lifecycle._reapply_active_scenarios({}, tmp_path, engine)

    assert "shows a clean machine" in str(caught.value)
    # The cause is chained rather than replaced — the original failure is what a
    # maintainer needs, the recovery is what the operator needs.
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_a_recorder_that_will_not_stop_is_reported(staged_archiver, tmp_path, caplog):
    """A recorder still running writes into the collection being dropped. The
    rebuild is still right, but the breach of the quiesce invariant must be
    visible rather than swallowed by a return code nobody reads."""
    staged_archiver["returncode"] = 1

    with caplog.at_level(logging.WARNING):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert "archiver-recorder" in caplog.text
    # Reported, not fatal: the seed still ran.
    assert len(staged_archiver["seeded"]) == 1


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("snappy", "snappy"), (None, "zstd")],
)
def test_compression_comes_from_the_service_block_not_the_knobs(
    staged_archiver, monkeypatch, tmp_path, configured, expected
):
    """The block compressor is a property of the SERVER, not of the archive's
    shape, so it lives at `services.mongodb.compression` and deliberately not in
    SeedKnobs. The seeder is handed it from that path, and it reaches the
    fingerprint from there — which is what makes changing the compressor a
    reported reseed rather than a store that is half one codec and half another.
    """
    config = dict(ARCHIVER_CONFIG)
    config["services"] = {"mongodb": {"port_host": 27017}}
    if configured is not None:
        config["services"]["mongodb"]["compression"] = configured
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert staged_archiver["seeded"][0]["kwargs"]["compression"] == expected


def test_the_auth_grace_sits_between_the_healthcheck_and_the_reachability_budget():
    """The 15/45/180 ordering documented at ``_ARCHIVER_AUTH_GRACE_S`` IS the design.

    Each constant is pinned somewhere already, but only in isolation: nothing
    compared them, so moving the template's ``start_period`` — the one the
    comment says "this has to move with it" — inverted the relationship
    silently. Below the start_period, a fresh volume's normal root-user
    creation is reported as a wrong password and the first deploy of every new
    project aborts. Above the reachability budget, the grace never expires and
    a genuinely stale volume burns the full budget for a diagnosis that was
    available in seconds.
    """
    import re

    import osprey

    template = (
        Path(osprey.__file__).parent
        / "templates"
        / "services"
        / "mongodb"
        / "docker-compose.yml.j2"
    ).read_text(encoding="utf-8")

    match = re.search(r"start_period:\s*(\d+)s", template)
    assert match, "no healthcheck start_period found in the mongodb template"
    start_period_s = float(match.group(1))

    assert start_period_s < container_lifecycle._ARCHIVER_AUTH_GRACE_S, (
        f"the auth grace ({container_lifecycle._ARCHIVER_AUTH_GRACE_S}s) must outlast the "
        f"store's own healthcheck start_period ({start_period_s}s), or a fresh volume's "
        "root-user creation is reported as a wrong password"
    )
    assert (
        container_lifecycle._ARCHIVER_AUTH_GRACE_S < container_lifecycle._ARCHIVER_HEALTH_TIMEOUT_S
    ), (
        f"the auth grace ({container_lifecycle._ARCHIVER_AUTH_GRACE_S}s) must expire inside the "
        f"reachability budget ({container_lifecycle._ARCHIVER_HEALTH_TIMEOUT_S}s), or a stale "
        "volume burns the whole budget before saying so"
    )


# ---------------------------------------------------------------------------
# The shared half of the env chain, at the store preflight's read site
# ---------------------------------------------------------------------------
# The deployment's environment is a two-file chain at the repo root —
# ``.env.shared`` (committed defaults) below ``.env`` (host-local secrets).
# The preflight below never opens the first of those files: it calls
# ``parse_dotenv_file(env_path)``, which names the LOCAL one, and reads
# ``os.environ`` for everything else. So a value living only in the shared half
# reaches it by one indirect route — the CLI entry point loads the whole chain
# over ``os.environ`` before any deploy code runs.
#
# The site gets both cells: the value in the shared file alone, and the same
# value once that entry-point load has happened. The pair is the measurement.
# What it records is pinned as observed, including where the observation is that
# the shared half is not seen at all.

#: Obviously-fake stand-ins throughout this section. None is a credential.
SHARED_HALF_CREDENTIAL = "shared-half-fixture-credential"
VOLUME_BORN_WITH = "the-credential-the-volume-was-born-with"
CHAIN_PROJECT = "demo-deployment"


def _write_shared_half(repo: Path, text: str) -> Path:
    """Lay down the chain's committed-defaults file beside the local one."""
    from osprey.utils.dotenv import ENV_SHARED_FILENAME

    path = repo / ENV_SHARED_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def _load_the_entry_point_chain(monkeypatch, repo: Path) -> None:
    """Run the load ``osprey <verb>`` performs before it reaches any deploy code.

    ``osprey.cli.main`` calls this first thing, cwd-rooted, with override
    semantics, so by the time a provisioner runs ``os.environ`` carries the
    merged chain. Reproduced verbatim rather than simulated with ``setenv``,
    because what is being measured IS that this step is what makes the shared
    half visible downstream.
    """
    import osprey.utils.config as config

    monkeypatch.setattr(config, "_dotenv_shell_overrides", {})
    monkeypatch.chdir(repo)
    config.load_project_dotenv()


class _StoreRuntime:
    """Answers the two argv shapes the stale-volume preflight builds.

    Argv-shaped rather than a mock of the probe: the subject is which questions
    the deploy asks the runtime, so a stand-in that accepted anything would pass
    while the real command was wrong.
    """

    def __init__(self, volumes: list[str], container_env: dict[str, dict[str, str]]):
        self.volumes = volumes
        self.container_env = container_env

    def __call__(self, cmd, **kwargs):
        argv = list(cmd)[1:]  # drop the runtime binary
        if argv[:2] == ["volume", "ls"]:
            return subprocess.CompletedProcess(list(cmd), 0, stdout="\n".join(self.volumes))
        if argv[:2] == ["container", "inspect"]:
            env = self.container_env.get(argv[2])
            if env is None:
                return subprocess.CompletedProcess(list(cmd), 1, stdout="")
            lines = "\n".join(f"{k}={v}" for k, v in env.items())
            return subprocess.CompletedProcess(list(cmd), 0, stdout=lines)
        return subprocess.CompletedProcess(list(cmd), 0, stdout="")


@pytest.fixture
def store_preflight(monkeypatch, tmp_path):
    """One store, one surviving volume whose container holds a different value.

    The mismatch is the setup, not the assertion: whether the preflight *sees*
    it is what each test below records.
    """
    monkeypatch.delenv("MONGO_ROOT_PASSWORD", raising=False)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        _StoreRuntime(
            volumes=[f"{CHAIN_PROJECT}_archiver_mongodb_data"],
            container_env={
                f"{CHAIN_PROJECT}-archiver-mongodb": {
                    "MONGO_INITDB_ROOT_PASSWORD": VOLUME_BORN_WITH
                }
            },
        ),
    )

    def _run(repo: Path):
        config = {"deployed_services": ["mongodb"], "project_name": CHAIN_PROJECT}
        container_lifecycle._preflight_stale_store_volumes(config, set(), repo / ".env")

    return _run


def test_a_shared_only_store_credential_that_the_volume_rejects_is_not_caught(
    store_preflight, tmp_path
):
    """The preflight's disagreement rule cannot see a shared-half credential.

    ``.env.shared`` names a credential the surviving volume was never
    initialized with, which is exactly the state this check exists to refuse.
    It does not refuse it: the effective value it computes is empty (it parsed
    the local ``.env``, which does not exist), and an empty value never
    disagrees with anything. The deploy proceeds and the store's own
    authentication failure arrives minutes later instead.

    Pinned as it behaves. A change that makes this raise is a change in which
    file the preflight reads — the signal, not a break.
    """
    _write_shared_half(tmp_path, f"MONGO_ROOT_PASSWORD={SHARED_HALF_CREDENTIAL}\n")

    store_preflight(tmp_path)  # no refusal


def test_the_same_shared_only_credential_is_refused_after_the_entry_point_chain_load(
    store_preflight, monkeypatch, tmp_path
):
    """The route by which the shared half reaches the preflight at all.

    Same files, same runtime answers, with the one step a real ``osprey up``
    takes first. The chain load puts the shared value in ``os.environ``, the
    effective value stops being empty, and the disagreement with the container
    is found before anything starts.
    """
    _write_shared_half(tmp_path, f"MONGO_ROOT_PASSWORD={SHARED_HALF_CREDENTIAL}\n")

    _load_the_entry_point_chain(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        store_preflight(tmp_path)
    message = str(excinfo.value)
    assert "archiver_mongodb_data" in message
    # The report names the store and the file, never either credential.
    assert SHARED_HALF_CREDENTIAL not in message
    assert VOLUME_BORN_WITH not in message


# ---------------------------------------------------------------------------
# The Python prediction of the worker's image vs. the rendered compose default
#
# Two derivations of one fact. `_worker_image_target` answers, before compose
# has been written, which image the worker will run — and that answer is what
# decides whether `osprey up` builds the project image at all. The rendered
# document decides what the worker ACTUALLY runs. A disagreement between them
# is silent in both directions: a build for a tag nothing references, or a
# reference to a tag nothing built.
#
# So the two are asserted equal here rather than by reading the code, across
# both image axes (unset and set) and each worker-config shape that reaches a
# different layer of the fallback chain. The value is also pinned literally in
# each case: parity alone would still hold if BOTH sides ignored the axes.
# ---------------------------------------------------------------------------

#: The axis variables, cleared before every parity case — an exported
#: ``OSPREY_IMAGE_TAG`` in the developer's shell otherwise reaches both sides.
_IMAGE_AXIS_VARS = ("OSPREY_IMAGE_REGISTRY", "OSPREY_IMAGE_TAG")

#: A complete image reference a profile pinned itself. No axis may edit it:
#: naming an image outright names a whole coordinate, registry included.
_PINNED_WORKER_IMAGE = "ghcr.io/vendor/prebuilt-worker:4.2"


def _parity_config(worker_cfg: dict | None) -> dict:
    """A deploy config whose only variable is the ``dispatch_worker`` block."""
    return {
        "project_name": "parity-proj",
        "project_root": "/r/parity-proj",
        "services": {"dispatch_worker": worker_cfg},
        "deployed_services": ["dispatch_worker"],
        "system": {"timezone": "UTC"},
    }


def _rendered_worker_image(config: dict) -> str:
    """The image the worker's compose service renders, minus the compose layer.

    Rendered through an Environment rooted at the packaged templates directory
    because the template imports the shared network-axis macros, and an import
    needs a loader. The ``${OSPREY_WORKER_IMAGE:-...}`` wrapper is stripped: it
    is the compose-time layer, which ``_worker_image_target`` reads from its
    ``env`` argument rather than from the document.
    """
    from importlib import resources

    import yaml
    from jinja2 import Environment, FileSystemLoader

    from osprey.deployment.compose_generator import _inject_project_metadata

    templates_root = resources.files("osprey").joinpath("templates")
    jinja_env = Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)
    template = jinja_env.get_template("services/dispatch_worker/docker-compose.yml.j2")
    rendered = yaml.safe_load(template.render(**_inject_project_metadata(config)))

    image = rendered["services"]["dispatch-worker-1"]["image"]
    prefix = "${OSPREY_WORKER_IMAGE:-"
    assert image.startswith(prefix) and image.endswith("}"), image
    return image[len(prefix) : -1]


_AXES_UNSET: dict[str, str] = {}
_AXES_SET = {
    "OSPREY_IMAGE_REGISTRY": "registry.example.org/beam",
    "OSPREY_IMAGE_TAG": "2026.08.19",
}

#: Each axis setting with the project image it must produce.
_AXIS_CASES = [
    pytest.param(_AXES_UNSET, "parity-proj:local", id="axes-unset"),
    pytest.param(_AXES_SET, "registry.example.org/beam/parity-proj:2026.08.19", id="axes-set"),
]

#: The same settings alone, for the outcomes no axis may change.
_AXIS_ENV_CASES = [
    pytest.param(_AXES_UNSET, id="axes-unset"),
    pytest.param(_AXES_SET, id="axes-set"),
]


def _set_image_axes(monkeypatch, axes: dict[str, str]) -> None:
    """Put the process env in exactly the axis state a case describes."""
    for var in _IMAGE_AXIS_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in axes.items():
        monkeypatch.setenv(var, value)


_WORKER_CFG_CASES = [
    # A `dispatch_worker:` mapping — _inject_project_metadata's setdefault
    # supplies the image, so the template's own default never fires.
    pytest.param({}, False, id="worker-block-present"),
    # A null `dispatch_worker:` key (legal YAML). The setdefault's isinstance
    # guard skips it, exposing the template-level `default(osprey_images.worker)`.
    pytest.param(None, False, id="null-worker-key"),
    # A profile that pinned its own image: layer 2 wins on both sides.
    pytest.param({"image": _PINNED_WORKER_IMAGE}, True, id="per-service-override"),
]


@pytest.mark.parametrize("axes, axis_image", _AXIS_CASES)
@pytest.mark.parametrize("worker_cfg, pinned", _WORKER_CFG_CASES)
def test_worker_image_target_parity_with_rendered_worker_default(
    monkeypatch, axes, axis_image, worker_cfg, pinned
):
    """``_worker_image_target`` predicts exactly what the worker service renders."""
    _set_image_axes(monkeypatch, axes)

    config = _parity_config(worker_cfg)
    expected = _PINNED_WORKER_IMAGE if pinned else axis_image

    predicted = container_lifecycle._worker_image_target(config, env={})
    assert predicted == _rendered_worker_image(config), (
        "the Python prediction and the rendered compose default must be the "
        "same image — they are one fact derived twice"
    )
    assert predicted == expected


@pytest.mark.parametrize("axes, axis_image", _AXIS_CASES)
def test_project_image_build_target_parity_with_rendered_worker_default(
    monkeypatch, axes, axis_image
):
    """What ``osprey up`` builds is the tag the worker service references.

    The build target is the same prediction seen from the other end: with
    nothing pinned, the image the host builds and tags has to be the one the
    rendered document names, or the worker starts against a tag that exists
    nowhere. Asserted for the build command's own ``-t`` too — the target is
    only a claim until the argv carries it.
    """
    _set_image_axes(monkeypatch, axes)

    config = _parity_config({})

    build_target = container_lifecycle._project_image_build_target(config, env={})
    assert build_target == _rendered_worker_image(config)
    assert build_target == axis_image

    cmd = container_lifecycle._project_image_build_cmd(config, "docker", "/proj")
    assert cmd[cmd.index("-t") + 1] == build_target


@pytest.mark.parametrize("axes", _AXIS_ENV_CASES)
def test_a_pinned_worker_image_builds_nothing_under_either_axis(monkeypatch, axes):
    """The build gate reads the prediction, so a pin skips the build, axes or not.

    A profile that pinned its worker image wants a prebuilt one; the axes must
    not turn that into a build of an image the worker will never run.
    """
    _set_image_axes(monkeypatch, axes)

    pinned = _parity_config({"image": _PINNED_WORKER_IMAGE})
    assert container_lifecycle._project_image_build_target(pinned, env={}) is None

    override = _parity_config({})
    assert (
        container_lifecycle._project_image_build_target(
            override, env={"OSPREY_WORKER_IMAGE": _PINNED_WORKER_IMAGE}
        )
        is None
    )
