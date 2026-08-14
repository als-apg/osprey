"""``deploy_down`` runs compose through the capture helper, not ``execvpe``.

The legacy ``down`` used to hand itself to ``os.execvpe``, which made its
output impossible to capture: the compose child inherited the terminal and this
process was gone. It now runs through :func:`run_captured` and exits on the
child's own status, so the spool file and the phase reporter both survive the
call — while the reason ``execvpe`` (and not ``execvp``) was chosen in the first
place, the ``COMPOSE_PROJECT_NAME`` pin, still has to reach compose.

The attached ``up``/``restart`` paths are deliberately still ``execvpe``: there
the operator WANTS compose's live output, so those are guarded here against an
over-broad conversion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle


def _mock_down_config(monkeypatch, project_name="myproj", extra=None):
    """Wire deploy_down's config load and compose-file discovery to fixtures."""
    raw_config = {"project_name": project_name, "deployed_services": ["event_dispatcher"]}
    raw_config.update(extra or {})
    monkeypatch.setattr(container_lifecycle, "load_project_config", lambda p, **kwargs: raw_config)
    monkeypatch.setattr(
        "osprey.deployment.compose_generator.find_existing_compose_files",
        lambda *a, **k: ["docker-compose.yml"],
    )
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )


def _forbid_exec(monkeypatch):
    """Make any ``os.exec*`` from the code under test a loud failure.

    Both variants, and both loudly: an unpatched real ``execvp`` would replace
    the test process rather than fail it.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("the down path must not exec away")

    monkeypatch.setattr(container_lifecycle.os, "execvp", _boom)
    monkeypatch.setattr(container_lifecycle.os, "execvpe", _boom)


def _capture_run(monkeypatch, returncode=0):
    """Patch run_captured in container_lifecycle's namespace; record its call."""
    calls: list[dict] = []

    def _fake(cmd, **kwargs):
        calls.append({"cmd": list(cmd), **kwargs})
        return subprocess.CompletedProcess(list(cmd), returncode)

    monkeypatch.setattr(container_lifecycle, "run_captured", _fake)
    return calls


def test_down_runs_compose_through_the_capture_helper(monkeypatch, tmp_path):
    """The compose ``down`` is a captured run, spooled under its own name."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch)
    _forbid_exec(monkeypatch)
    calls = _capture_run(monkeypatch)

    with pytest.raises(SystemExit):
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert len(calls) == 1, calls
    call = calls[0]
    assert "down" in call["cmd"]
    assert call["spool_name"] == "compose-down"
    assert call["repo_root"] is not None


def test_down_carries_the_pinned_compose_project_name(monkeypatch, tmp_path):
    """SC8: the pin ``execvpe`` existed to deliver must reach the captured run.

    Unpinned, ``down`` either misses this deploy's containers or tears down the
    shared "services" project that every deploy on the host collapses into.
    """
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch, "myproj")
    _forbid_exec(monkeypatch)
    calls = _capture_run(monkeypatch)

    with pytest.raises(SystemExit):
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    env = calls[0]["env"]
    assert env is not None, "the captured run inherited the parent env, losing the pin"
    assert env["COMPOSE_PROJECT_NAME"] == "myproj"
    # The rest of the environment still comes through — the pin is an overlay on
    # os.environ, not a replacement for it (compose reads PATH, DOCKER_HOST, ...).
    assert "PATH" in env


@pytest.mark.parametrize("returncode", [0, 1, 137])
def test_down_exits_with_the_compose_child_s_own_code(monkeypatch, tmp_path, returncode):
    """``execvpe`` gave the caller compose's exit status verbatim; so does this.

    ``check=False`` plus an explicit exit is what preserves that: ``check=True``
    would turn every non-zero compose exit into one uniform failure code.
    """
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch)
    _forbid_exec(monkeypatch)
    calls = _capture_run(monkeypatch, returncode=returncode)

    with pytest.raises(SystemExit) as excinfo:
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert excinfo.value.code == returncode
    assert calls[0]["check"] is False, "a raising run would never reach the exit"


def test_a_signal_killed_compose_exits_the_way_a_shell_spells_it(monkeypatch, tmp_path):
    """SIGTERM: -15 from Popen, 143 from the exec'd child a shell would see.

    Passed through raw, sys.exit(-15) masks to 241 — a status no signal maps to.
    """
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch)
    _forbid_exec(monkeypatch)
    _capture_run(monkeypatch, returncode=-15)

    with pytest.raises(SystemExit) as excinfo:
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert excinfo.value.code == 143


def test_a_failed_down_reports_no_step(monkeypatch, tmp_path):
    """Steps name work that happened: a non-zero ``down`` stopped nothing."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch)
    _forbid_exec(monkeypatch)
    _capture_run(monkeypatch, returncode=1)
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)

    with pytest.raises(SystemExit):
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert steps == []


def test_a_successful_down_reports_its_step(monkeypatch, tmp_path):
    """The verb's phase gets one line for the containers it stopped."""
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch)
    _forbid_exec(monkeypatch)
    _capture_run(monkeypatch)
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)

    with pytest.raises(SystemExit):
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert steps == ["containers stopped"]


def test_the_web_stack_still_comes_down_before_the_services(monkeypatch, tmp_path):
    """Ordering survives the conversion.

    The services ``down`` still ends this process (now by exiting rather than by
    exec), and its ``-f`` list can never carry ``docker-compose.web.yml``, so a
    web ``down`` that ran afterwards would never run at all.
    """
    monkeypatch.chdir(tmp_path)
    _mock_down_config(monkeypatch, extra={"modules": {"web_terminals": {"enabled": True}}})
    _forbid_exec(monkeypatch)
    order: list[str] = []

    monkeypatch.setattr(
        container_lifecycle,
        "deploy_down_web_terminals",
        lambda config, env, env_file_args: order.append("web"),
    )

    def _fake(cmd, **kwargs):
        order.append("services")
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(container_lifecycle, "run_captured", _fake)

    with pytest.raises(SystemExit):
        container_lifecycle.deploy_down(str(tmp_path / "config.yml"))

    assert order == ["web", "services"]


# ---------------------------------------------------------------------------
# down_deployment — the LIVE path (`osprey down` -> down_verb -> here).
#
# deploy_down above is the legacy library entry point with no caller left in
# src/, so converting it alone would have left the verb an operator actually
# types still streaming compose to the terminal.
# ---------------------------------------------------------------------------


def _mock_live_down(monkeypatch, tmp_path):
    """Wire down_deployment's as-built reads to fixtures under *tmp_path*."""
    config_path = tmp_path / "build" / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project_name: myrepo\n")

    monkeypatch.setattr(container_lifecycle, "as_built_config_path", lambda root: config_path)
    monkeypatch.setattr(
        container_lifecycle, "load_project_config", lambda p, **k: {"project_name": "myrepo"}
    )
    monkeypatch.setattr(
        container_lifecycle, "as_built_compose_files", lambda config, root: ["docker-compose.yml"]
    )
    monkeypatch.setattr(container_lifecycle, "_web_terminals_enabled", lambda config: False)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )


def test_the_live_down_captures_compose_under_the_pinned_project(monkeypatch, tmp_path):
    """The verb's own compose invocation is spooled, not streamed — with the pin."""
    _mock_live_down(monkeypatch, tmp_path)
    calls = _capture_run(monkeypatch)

    container_lifecycle.down_deployment(tmp_path)

    assert len(calls) == 1, calls
    call = calls[0]
    assert "down" in call["cmd"]
    assert call["spool_name"] == "compose-down"
    assert Path(call["repo_root"]) == tmp_path
    assert call["check"] is False
    assert call["env"]["COMPOSE_PROJECT_NAME"] == "myrepo"


def test_the_live_down_reports_its_step(monkeypatch, tmp_path):
    """The step hangs off the path the verb actually drives, not the legacy one."""
    _mock_live_down(monkeypatch, tmp_path)
    _capture_run(monkeypatch)
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)

    container_lifecycle.down_deployment(tmp_path)

    assert steps == ["containers stopped"]


def test_a_failed_live_down_still_says_the_containers_may_be_running(monkeypatch, tmp_path):
    """The consequence a bare CapturedProcessError would not state.

    The spool is still written and noted on the reporter, so the failure line
    replays it; this path adds what a non-zero ``down`` MEANS.
    """
    _mock_live_down(monkeypatch, tmp_path)
    _capture_run(monkeypatch, returncode=1)
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)

    with pytest.raises(RuntimeError, match="may still be running"):
        container_lifecycle.down_deployment(tmp_path)

    assert steps == [], "a failed down stopped nothing to report"


def test_attached_up_still_execvpes(monkeypatch, tmp_path):
    """Non-regression: the conversion is scoped to ``down``.

    A non-detached ``up`` hands the terminal to compose on purpose — the
    operator asked to watch it — so that call site must still exec.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **k: subprocess.CompletedProcess(list(cmd), 0),
    )
    execd: dict = {}
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvpe",
        lambda file, args, env: execd.update(args=list(args), env=env),
    )
    # execvp too: a call that lost its env would otherwise replace the test
    # process instead of failing the assertion below.
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvp",
        lambda file, args: pytest.fail("attached up must exec WITH its env"),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert "up" in execd["args"], execd
    assert execd["env"]["COMPOSE_PROJECT_NAME"]
