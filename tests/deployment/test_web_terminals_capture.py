"""The web-terminal deploy path spools its children instead of streaming them.

`deploy_up` returns through `deploy_up_web_terminals` for every
`modules.web_terminals` deployment, which makes this the path an operator of
the control-assistant preset actually watches. Every inherited-stdio child on
it — the persona and auth-sidecar image builds, both compose stacks' `rm`,
`build`, `pull`, `up` and the post-`up` recreate/restart — now runs through
`run_captured`, and each reports itself as one phase sub-step instead of
thousands of BuildKit lines.

The headline assertion is `test_default_view_hides_buildkit_lines_...`: a real
child emitting BuildKit-shaped output, through the real reporter and the real
capture helper, must leave no `#N [stage]` line on the terminal while the spool
file holds all of them.

Every "nothing reached the terminal" assertion here uses `capfd`, never
`capsys`. An uncaptured child writes to file descriptor 1 directly, which
Python-level capture cannot see at all — a `capsys` version of these tests
passes even with the capture seam removed, which is exactly the regression they
exist to catch.
"""

import re
import sys

import pytest

from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment.subprocess_capture import SPOOL_DIR, CapturedProcess
from osprey.deployment.web_terminals import persona_images, postup_hooks, provision

# A BuildKit progress line: `#5 [stage-0 2/7] RUN ...`. The whole point of the
# capture seam is that this shape never reaches the operator's terminal.
BUILDKIT_LINE = re.compile(r"#\d+ \[")

BUILDKIT_OUTPUT = [
    "#1 [internal] load build definition from Dockerfile",
    "#5 [stage-0 2/7] RUN pip install --no-cache-dir osprey-framework",
    "#9 exporting to image",
]


class RecordingReporter(PhaseReporter):
    """A reporter that keeps its lines instead of printing them."""

    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        self.lines.append(text)

    @property
    def steps(self) -> list[str]:
        """Just the sub-step lines, stripped of their `  · ` prefix and lap."""
        return [line[4:].split(" (")[0] for line in self.lines if line.startswith("  · ")]


@pytest.fixture
def reporter():
    """A recording reporter with one phase open, uninstalled afterwards."""
    rep = RecordingReporter()
    previous = install_reporter(rep)
    rep.phase("Starting the stack")
    try:
        yield rep
    finally:
        install_reporter(previous)


@pytest.fixture
def terminal_reporter(monkeypatch):
    """The real reporter, printing to the captured stdout, one phase open."""
    rep = PhaseReporter(color=False)
    previous = install_reporter(rep)
    rep.phase("Starting the stack")
    try:
        yield rep
    finally:
        install_reporter(previous)


class RunRecorder:
    """Stands in for `run_captured`, recording every call's argv and kwargs.

    Returns the real return type, so a caller that reads `spool_path` off the
    result (the smoke-check hook does) sees `None` — this stand-in spooled
    nothing — rather than raising `AttributeError`.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self, cmd, *, env=None, cwd=None, spool_name, repo_root=None, check=True, on_line=None
    ):
        self.calls.append(
            {
                "cmd": list(cmd),
                "env": env,
                "cwd": cwd,
                "spool_name": spool_name,
                "repo_root": repo_root,
                "check": check,
                "on_line": on_line,
            }
        )
        return CapturedProcess(list(cmd), 0, spool_path=None)

    def by_spool(self, spool_name: str) -> dict:
        matches = [call for call in self.calls if call["spool_name"] == spool_name]
        assert matches, f"no captured run named {spool_name!r} (got {self.spool_names})"
        assert len(matches) == 1, f"{spool_name!r} ran {len(matches)} times"
        return matches[0]

    @property
    def spool_names(self) -> list[str]:
        return [call["spool_name"] for call in self.calls]


def _echo_cmd(lines: list[str]) -> list[str]:
    """A real child printing `lines` — a stand-in for a BuildKit-noisy build."""
    body = "\n".join(f"print({line!r})" for line in lines)
    return [sys.executable, "-c", body]


def _spool_files(repo_root):
    return sorted((repo_root / SPOOL_DIR).glob("*.log"))


# --------------------------------------------------------------------------
# persona_images.build_persona_images — the headline scenario's slow step
# --------------------------------------------------------------------------


def _persona_config() -> dict:
    return {
        "project_name": "demo",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {}, "physics": {}},
            }
        },
    }


def _stub_persona_builds(monkeypatch, tmp_path, units, build_cmd=None):
    """Neuter everything around the persona build except the run itself."""
    monkeypatch.chdir(tmp_path)  # resolve_repo_root -> tmp_path
    monkeypatch.setattr(persona_images, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(persona_images, "_referenced_personas", lambda config, users: units)
    monkeypatch.setattr(
        persona_images,
        "_persona_image_build_cmd",
        lambda *args: list(build_cmd) if build_cmd else ["docker", "build", "."],
    )


def _unit(project: str, persona: str, tmp_path) -> dict:
    return {"project": project, "persona": persona, "project_path": str(tmp_path / persona)}


def test_persona_build_captures_per_image_with_its_own_spool(monkeypatch, tmp_path, reporter):
    """Each persona image is its own captured run, named for the image it
    builds, so a failed build names a spool holding only that build's output."""
    units = [_unit("demo", "ops", tmp_path), _unit("demo", "physics", tmp_path)]
    _stub_persona_builds(monkeypatch, tmp_path, units)
    recorder = RunRecorder()
    monkeypatch.setattr(persona_images, "run_captured", recorder)

    persona_images.build_persona_images(_persona_config(), [], False, {"HOME": "/x"})

    assert recorder.spool_names == ["build-persona-demo-ops", "build-persona-demo-physics"]
    for call in recorder.calls:
        assert call["cmd"] == ["docker", "build", "."]
        assert call["env"] == {"HOME": "/x"}
        assert call["repo_root"] == tmp_path
        assert call["check"] is True


def test_persona_build_reports_one_step_per_image(monkeypatch, tmp_path, reporter):
    """One sub-step per persona image, naming that image: the only progress an
    operator sees while a run of multi-minute builds goes by."""
    units = [_unit("demo", "ops", tmp_path), _unit("demo", "physics", tmp_path)]
    _stub_persona_builds(monkeypatch, tmp_path, units)
    monkeypatch.setattr(persona_images, "run_captured", RunRecorder())

    persona_images.build_persona_images(_persona_config(), [], False, {})

    assert reporter.steps == ["persona image demo-ops:local", "persona image demo-physics:local"]


def test_default_view_hides_buildkit_lines_but_the_spool_keeps_them(
    monkeypatch, tmp_path, capfd, terminal_reporter
):
    """SC2, end to end on the headline path: a real child emitting BuildKit
    progress, the real reporter, the real capture helper. The operator's
    terminal shows the step line and nothing of the build; the spool file
    exists and holds every line."""
    units = [_unit("demo", "ops", tmp_path)]
    _stub_persona_builds(monkeypatch, tmp_path, units, build_cmd=_echo_cmd(BUILDKIT_OUTPUT))

    persona_images.build_persona_images(_persona_config(), [], False, None)

    out = capfd.readouterr().out
    assert not BUILDKIT_LINE.search(out), f"raw build output reached the terminal:\n{out}"
    for line in BUILDKIT_OUTPUT:
        assert line not in out
    assert "· persona image demo-ops:local" in out

    spools = _spool_files(tmp_path)
    assert len(spools) == 1, f"expected one spool file, got {[p.name for p in spools]}"
    assert spools[0].name.startswith("build-persona-demo-ops-")
    spooled = spools[0].read_text()
    for line in BUILDKIT_OUTPUT:
        assert line in spooled


# --------------------------------------------------------------------------
# provision.build_auth_sidecar_image
# --------------------------------------------------------------------------


def test_auth_sidecar_build_is_captured_and_reported(monkeypatch, tmp_path, reporter):
    """The sidecar build is the other local-mode image build; same treatment,
    and its own spool name so the two never share a file."""
    monkeypatch.chdir(tmp_path)
    context = tmp_path / "build" / "auth"
    context.mkdir(parents=True)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        provision, "_materialize_auth_build_context", lambda repo_root, dev_mode: context
    )
    recorder = RunRecorder()
    monkeypatch.setattr(provision, "run_captured", recorder)

    config = {
        "project_name": "demo",
        "modules": {
            "web_terminals": {"image_source": "local", "auth": {"method": "password"}},
        },
    }
    provision.build_auth_sidecar_image(config, False, {"HOME": "/x"})

    call = recorder.by_spool("build-auth-sidecar")
    assert call["cmd"][:2] == ["docker", "build"]
    assert call["cmd"][-1] == str(context)
    assert call["env"] == {"HOME": "/x"}
    assert call["repo_root"] == tmp_path
    assert call["check"] is True
    assert reporter.steps == [f"auth sidecar image {provision.auth_sidecar_local_tag(config)}"]


# --------------------------------------------------------------------------
# provision._force_recreate_services — post-up recreate and the auth rotation
# --------------------------------------------------------------------------


def test_force_recreate_is_captured_with_the_callers_repo_root(monkeypatch, tmp_path, reporter):
    """The recreate is a compose `up` like any other: captured, and anchored to
    the repo root the caller already resolved rather than the process cwd."""
    recorder = RunRecorder()
    monkeypatch.setattr(provision, "run_captured", recorder)

    provision._force_recreate_services(
        ["docker", "compose", "-f", "web.yml"],
        {"COMPOSE_PROJECT_NAME": "demo"},
        ["osprey-auth", "nginx"],
        repo_root=tmp_path,
    )

    call = recorder.by_spool("compose-force-recreate")
    assert call["cmd"] == [
        "docker",
        "compose",
        "-f",
        "web.yml",
        "up",
        "-d",
        "--force-recreate",
        "osprey-auth",
        "nginx",
    ]
    assert call["repo_root"] == tmp_path
    assert call["check"] is True
    assert reporter.steps == ["recreated osprey-auth, nginx"]


def test_image_drift_reconcile_forwards_the_deploys_repo_root(monkeypatch, tmp_path, reporter):
    """The post-`up` reconcile is handed the root `deploy_up_web_terminals`
    already resolved, so the recreate spools beside the `up` that preceded it
    rather than wherever the process happens to be."""
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "docker-compose.web.yml").write_text(
        "services:\n  nginx:\n    image: nginx:local\n    container_name: demo-nginx\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman"])
    monkeypatch.setattr(provision, "get_image_id", lambda *a, **kw: "sha256:new")
    monkeypatch.setattr(provision, "get_container_image_id", lambda *a, **kw: "sha256:old")
    recorded: dict = {}
    monkeypatch.setattr(
        provision,
        "_force_recreate_services",
        lambda web_cmd, run_env, services, **kwargs: recorded.update(
            services=services, repo_root=kwargs.get("repo_root")
        ),
    )

    provision._reconcile_web_stack_recreates(
        {"project_name": "demo"}, ["podman", "compose"], {}, repo_root=tmp_path
    )

    assert recorded == {"services": ["nginx"], "repo_root": tmp_path}


# --------------------------------------------------------------------------
# provision.deploy_up_web_terminals — both compose stacks
# --------------------------------------------------------------------------


def _stub_web_stack(monkeypatch, tmp_path):
    """Neuter every collaborator of deploy_up_web_terminals except the runs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(provision, "runtime_env", lambda config, env, **kw: dict(env))
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest=".": [])
    monkeypatch.setattr(provision, "ensure_env_production", lambda config, root: None)
    monkeypatch.setattr(provision, "resolve_personas", lambda *a, **kw: [])
    monkeypatch.setattr(provision, "verify_persona_renders", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "build_persona_images", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "build_auth_sidecar_image", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "_reconcile_web_stack_recreates", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "reload_nginx_config", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "enable_linger", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "seed_user_containers", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "run_verify_script", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "warn_if_web_stack_unreachable", lambda *a, **kw: None)
    recorder = RunRecorder()
    monkeypatch.setattr(provision, "run_captured", recorder)
    return recorder


def _web_config(image_source: str = "registry") -> dict:
    return {
        "project_name": "demo",
        "deployed_services": ["ariel"],
        "modules": {
            "web_terminals": {"image_source": image_source, "auth": {"method": "none"}},
        },
    }


def test_dev_mode_up_captures_every_compose_invocation(monkeypatch, tmp_path, reporter):
    """A dev-mode registry deploy runs both stacks: services rm/build/up, then
    the web stack's rm/pull/up. Every one is captured, in that order, each
    under its own spool name and anchored to the repo root."""
    recorder = _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(_web_config(), ["docker-compose.yml"], True, {}, [])

    assert recorder.spool_names == [
        "compose-services-rm",
        "build-services",
        "compose-services-up",
        "compose-web-rm",
        "compose-web-pull",
        "compose-web-up",
    ]
    assert all(call["repo_root"] == tmp_path for call in recorder.calls)
    assert recorder.by_spool("compose-services-rm")["cmd"][-2:] == ["rm", "-f"]
    assert recorder.by_spool("build-services")["cmd"][-1] == "build"
    assert recorder.by_spool("compose-services-up")["cmd"][-3:] == ["up", "--no-build", "-d"]
    assert recorder.by_spool("compose-web-rm")["cmd"][-2:] == ["rm", "-f"]
    assert recorder.by_spool("compose-web-pull")["cmd"][-1] == "pull"
    assert recorder.by_spool("compose-web-up")["cmd"][-2:] == ["up", "-d"]


def test_stale_container_preflights_stay_non_fatal(monkeypatch, tmp_path, reporter):
    """Both `rm -f` preflights no-op on a clean stack and must never abort the
    deploy on a non-zero exit — check=False, exactly as before the conversion.
    Every other invocation stays fail-loud."""
    recorder = _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(_web_config(), ["docker-compose.yml"], False, {}, [])

    checks = {call["spool_name"]: call["check"] for call in recorder.calls}
    assert checks == {
        "compose-services-rm": False,
        "compose-services-up": True,
        "compose-web-rm": False,
        "compose-web-pull": True,
        "compose-web-up": True,
    }


def test_up_reports_a_step_for_each_stack(monkeypatch, tmp_path, reporter):
    """The default view of a deploy: a handful of sub-steps, no compose output."""
    _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(_web_config(), ["docker-compose.yml"], False, {}, [])

    assert reporter.steps == [
        "cleared stale service containers",
        "backend services started",
        "cleared stale web-terminal containers",
        "pulled web-terminal images",
        "web-terminal stack started",
    ]


def test_the_services_build_streams_per_image_progress(monkeypatch, tmp_path, reporter):
    """The web path's `build` gets the same per-image steps as the plain path.

    It is the identical half-hour silence — same compose build, reached through
    `deploy_up_web_terminals` instead of `_start_stack` — so the two sites must
    not drift apart on whether an operator can see it progressing.
    """
    recorder = _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(_web_config(), ["docker-compose.yml"], True, {}, [])

    assert callable(recorder.by_spool("build-services")["on_line"])
    for other in ("compose-services-rm", "compose-services-up", "compose-web-up"):
        assert recorder.by_spool(other)["on_line"] is None


def test_local_mode_up_still_never_pulls(monkeypatch, tmp_path, reporter):
    """The local-only tags have no upstream, so local mode issues no pull —
    the conversion must not smuggle one back in."""
    recorder = _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(_web_config("local"), ["docker-compose.yml"], False, {}, [])

    assert "compose-web-pull" not in recorder.spool_names


# --------------------------------------------------------------------------
# postup_hooks — the advisory verify.sh smoke check
# --------------------------------------------------------------------------


def _write_verify_script(project_root, body: str = "echo ok") -> None:
    scripts = project_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "verify.sh").write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")


def test_verify_script_is_captured_from_the_project_root(monkeypatch, tmp_path, reporter):
    """The smoke check spools like every other child, but must still RUN from
    the project root — its own `./scripts/...` assumptions depend on it, so
    `cwd` and `repo_root` are passed separately."""
    _write_verify_script(tmp_path)
    recorder = RunRecorder()
    monkeypatch.setattr(postup_hooks, "run_captured", recorder)

    postup_hooks.run_verify_script(str(tmp_path), {"COMPOSE_PROJECT_NAME": "demo"})

    call = recorder.by_spool("verify-script")
    assert call["cmd"] == ["bash", str(tmp_path / "scripts" / "verify.sh")]
    assert call["cwd"] == str(tmp_path)
    assert call["repo_root"] == str(tmp_path)
    # Advisory: the script's own convention is to always exit 0, and a
    # site-customized copy that does not must still never fail the deploy.
    assert call["check"] is False
    assert reporter.steps == ["smoke check verify.sh — exit 0"]


def test_verify_script_output_never_reaches_the_terminal(
    monkeypatch, tmp_path, capfd, terminal_reporter
):
    """Real script, real capture: a chatty health report belongs in the spool,
    with only the step line on the operator's terminal."""
    _write_verify_script(tmp_path, body="\n".join(f"echo '{line}'" for line in BUILDKIT_OUTPUT))

    postup_hooks.run_verify_script(str(tmp_path), {})

    out = capfd.readouterr().out
    assert not BUILDKIT_LINE.search(out)
    assert "· smoke check verify.sh — exit 0" in out
    spooled = _spool_files(tmp_path)[0].read_text()
    for line in BUILDKIT_OUTPUT:
        assert line in spooled


def test_failing_verify_script_names_its_spool_and_does_not_raise(
    monkeypatch, tmp_path, caplog, terminal_reporter
):
    """A non-zero exit stays advisory — and now that nothing streamed, the
    warning has to name the file holding the report."""
    _write_verify_script(tmp_path, body="echo 'probe failed'\nexit 3")

    with caplog.at_level("WARNING"):
        postup_hooks.run_verify_script(str(tmp_path), {})

    spool = _spool_files(tmp_path)[0]
    assert "exited 3" in caplog.text
    assert str(spool) in caplog.text


def test_failing_verify_script_names_its_spool_with_no_phase_open(monkeypatch, tmp_path, caplog):
    """The hook has callers outside the lifecycle verbs (no reporter phase, so
    nothing recorded a spool path for them). The path comes off the completed
    process, so those callers get it too rather than being pointed at output
    that is not on their terminal."""
    _write_verify_script(tmp_path, body="echo 'probe failed'\nexit 3")

    with caplog.at_level("WARNING"):
        postup_hooks.run_verify_script(str(tmp_path), {})

    assert str(_spool_files(tmp_path)[0]) in caplog.text
    assert "Review the output above" not in caplog.text


def test_missing_verify_script_runs_nothing(monkeypatch, tmp_path, reporter):
    """A profile that carries no verify.sh deploys exactly as before."""
    recorder = RunRecorder()
    monkeypatch.setattr(postup_hooks, "run_captured", recorder)

    postup_hooks.run_verify_script(str(tmp_path), {})

    assert recorder.calls == []
    assert reporter.steps == []


# --------------------------------------------------------------------------
# postup_hooks — the Docker Desktop host-port self-heal restart
# --------------------------------------------------------------------------


def test_host_port_self_heal_restart_is_captured(monkeypatch, tmp_path, reporter):
    """The self-heal restart's compose chatter would bury the warning it sits
    under, so it spools; advisory as before (check=False), and it reports the
    bounce as a step."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(postup_hooks.sys, "platform", "darwin")
    monkeypatch.setattr(postup_hooks, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(postup_hooks, "_host_port_answers", lambda url, attempts, delay: False)
    recorder = RunRecorder()
    monkeypatch.setattr(postup_hooks, "run_captured", recorder)

    postup_hooks.warn_if_web_stack_unreachable(
        {"modules": {"web_terminals": {"nginx_port": 8080}}},
        attempts=1,
        delay=0,
        web_cmd=["docker", "compose", "-f", "web.yml"],
        run_env={"COMPOSE_PROJECT_NAME": "demo"},
    )

    call = recorder.by_spool("compose-web-restart")
    assert call["cmd"] == ["docker", "compose", "-f", "web.yml", "restart"]
    assert call["repo_root"] == tmp_path
    assert call["check"] is False
    assert reporter.steps == ["bounced the web stack for host-port re-registration"]


def test_host_port_probe_that_answers_runs_nothing(monkeypatch, tmp_path, reporter):
    """A reachable port is the common case: no restart, no spool, no step."""
    monkeypatch.setattr(postup_hooks, "_host_port_answers", lambda url, attempts, delay: True)
    recorder = RunRecorder()
    monkeypatch.setattr(postup_hooks, "run_captured", recorder)

    postup_hooks.warn_if_web_stack_unreachable(
        {"modules": {"web_terminals": {"nginx_port": 8080}}},
        web_cmd=["docker", "compose"],
        run_env={},
    )

    assert recorder.calls == []
    assert reporter.steps == []
