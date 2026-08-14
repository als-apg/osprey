"""The container lifecycle's subprocesses spool their output instead of the terminal.

``osprey up`` used to hand its terminal to whatever it started: an image build's
thousands of layer lines, compose's per-container churn, the archiver store's
boot chatter. The phase lines a verb prints are only readable if none of that
lands beside them, so every child process on this path now runs through
:func:`osprey.deployment.subprocess_capture.run_captured` — output to
``<repo>/var/logs/<spool_name>-<timestamp>.log``, and one ``· step`` line
reported in its place.

What these tests pin, site by site:

* the child runs through ``run_captured``, never a bare ``subprocess.run`` — the
  regression this feature exists to prevent is a site quietly reverting to
  inherited stdio, which no output assertion would catch;
* each site names its own ``spool_name``, so a failed deploy's spool directory
  says which step produced which file;
* ``repo_root`` is passed at every site: the helper's fallback is
  :func:`Path.cwd`, so an omission spools into whatever directory the operator
  happened to run from rather than the deployment repo;
* the step line is reported *after* the work, because
  :meth:`~osprey.cli.phase_reporter.Phase.step` times the lap since the previous
  step and a line printed first would credit the cost to the next unit;
* the best-effort sites (``compose rm``, the recorder quiesce) keep
  ``check=False`` — capturing must not turn a tolerated non-zero exit into a
  failed deploy.

No container runtime is touched: ``run_captured`` itself is stubbed, and the
reporter is a real :class:`~osprey.cli.phase_reporter.PhaseReporter` whose
``emit`` records lines rather than printing them.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment import container_lifecycle
from osprey.simulation import archiver_seed as seed_mod

#: The ``(1.2s)`` / ``(2m03s)`` suffix ``Phase.step`` appends to a slow lap.
_DURATION = re.compile(r" \((?:\d+\.\d+s|\d+m\d{2}s)\)$")


class RecordingReporter(PhaseReporter):
    """A real reporter whose lines land in a list instead of on stdout."""

    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        self.lines.append(text)

    @property
    def steps(self) -> list[str]:
        """Just the sub-step names, with the bullet and any duration stripped."""
        return [
            _DURATION.sub("", line.strip()[2:]) for line in self.lines if line.startswith("  ·")
        ]


@pytest.fixture
def reporter():
    """An installed reporter with one open phase, as a verb would have."""
    recorder = RecordingReporter()
    previous = install_reporter(recorder)
    recorder.phase("test phase")
    try:
        yield recorder
    finally:
        install_reporter(previous)


class CaptureRecorder:
    """Stand-in for ``run_captured`` that records how each site called it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, cmd, *, env=None, spool_name, repo_root=None, check=True):
        self.calls.append(
            {
                "cmd": list(cmd),
                "env": env,
                "spool_name": spool_name,
                "repo_root": repo_root,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(list(cmd), 0)

    @property
    def spool_names(self) -> list[str]:
        return [call["spool_name"] for call in self.calls]

    def call(self, spool_name: str) -> dict:
        """The one call made under ``spool_name`` (fails if it never happened)."""
        matches = [call for call in self.calls if call["spool_name"] == spool_name]
        assert len(matches) == 1, f"expected one {spool_name!r} call, got {self.spool_names}"
        return matches[0]


@pytest.fixture
def captured(monkeypatch):
    """Route every converted site's child process into a recorder."""
    recorder = CaptureRecorder()
    monkeypatch.setattr(container_lifecycle, "run_captured", recorder)
    return recorder


@pytest.fixture
def no_bare_subprocess(monkeypatch):
    """Fail any site that still runs a child with inherited stdio.

    The point of the conversion is that nothing on this path writes to the
    terminal behind the reporter's back. A site that regressed to
    ``subprocess.run`` would still pass every assertion about the *other* sites,
    so the absence is pinned directly.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError(f"a lifecycle site ran a bare subprocess: {args!r}")

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _forbidden)


# ---------------------------------------------------------------------------
# The dispatch worker's project image
# ---------------------------------------------------------------------------


def _image_config() -> dict:
    return {"project_name": "proj", "deployed_services": ["dispatch_worker"]}


@pytest.fixture
def image_build_stubs(monkeypatch, tmp_path):
    """Everything ``_build_project_image`` reaches for except the build itself."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(container_lifecycle, "resolve_project_name", lambda config: "proj")
    monkeypatch.setattr(container_lifecycle, "resolve_repo_root", lambda config, *a: repo)
    monkeypatch.setattr(
        container_lifecycle, "_worker_image_target", lambda config, env: "proj:local"
    )
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_project_image_build_cmd",
        lambda config, runtime, root, dev: ["docker", "build", "-t", "proj:local", str(root)],
    )
    return repo


def test_the_project_image_build_spools_its_output(
    captured, reporter, image_build_stubs, no_bare_subprocess
):
    """An image build is the loudest thing on the deploy path — and the slowest.

    Layer-by-layer output is exactly what an operator does not need until
    something fails, at which point they need all of it. Spooling gives both:
    one line while it works, the whole log named on the failure path.
    """
    container_lifecycle._build_project_image(_image_config(), False, {}, None)

    call = captured.call("build-project-image")
    assert call["cmd"][:2] == ["docker", "build"]
    assert call["repo_root"] == image_build_stubs
    assert call["check"] is True
    assert reporter.steps == ["project image proj:local"]


def test_the_image_step_is_reported_after_the_build(reporter, image_build_stubs, monkeypatch):
    """The step line names the build's own cost, so it follows the build.

    ``Phase.step`` reports the lap since the previous step. A line emitted
    before the work would time the *setup* and leave the minutes-long build
    charged to whatever step came next.
    """

    def _marking(cmd, *, env=None, spool_name, repo_root=None, check=True):
        reporter.lines.append("<<ran>>")
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(container_lifecycle, "run_captured", _marking)
    reporter.lines.clear()

    container_lifecycle._build_project_image(_image_config(), False, {}, None)

    marks = [line for line in reporter.lines if line == "<<ran>>" or line.startswith("  ·")]
    assert marks[0] == "<<ran>>", "the step line was reported before the build it times"
    assert marks[1].startswith("  · project image proj:local")


def test_the_image_build_is_skipped_without_a_dispatch_worker(captured, reporter):
    """No worker, no image, no step line claiming one was built."""
    container_lifecycle._build_project_image(
        {"project_name": "proj", "deployed_services": ["postgresql"]}, False, {}, None
    )

    assert captured.calls == []
    assert reporter.steps == []


# ---------------------------------------------------------------------------
# The staged archiver store, and the recorder it quiesces
# ---------------------------------------------------------------------------


@pytest.fixture
def archiver_stubs(monkeypatch):
    """Reduce ``_stage_archiver_store`` to its two subprocess calls.

    The seeder's own machinery — the store connection, the fingerprint
    comparison, the base rewrite — is covered elsewhere. What matters here is
    that the two compose invocations it makes are captured.
    """
    monkeypatch.setattr(
        container_lifecycle,
        "_archiver_store_connection",
        lambda config, project_dir: {
            "password": "pw",
            "password_env": "MONGO_PW",
            "username": "root",
            "host": "localhost",
            "port": 27017,
        },
    )
    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda root=None: [])
    monkeypatch.setattr(
        container_lifecycle, "_archiver_seed_inputs", lambda config, project_dir: ([], None, {})
    )
    monkeypatch.setattr(container_lifecycle, "_wait_for_archiver_store", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_reapply_active_scenarios", lambda *a, **k: None)

    knobs = SimpleNamespace(
        retention_days=7, hot_span_hours=6, hot_cadence_sec=1, tail_cadence_sec=60
    )
    monkeypatch.setattr(
        seed_mod, "SeedKnobs", SimpleNamespace(from_config=staticmethod(lambda config: knobs))
    )
    monkeypatch.setattr(seed_mod, "seed_fingerprint", lambda *a, **k: "fp")
    monkeypatch.setattr(
        seed_mod,
        "compare_fingerprint",
        lambda collection, fingerprint: SimpleNamespace(state=seed_mod.SeedState.ABSENT),
    )
    monkeypatch.setattr(
        seed_mod, "seed_base", lambda *a, **k: SimpleNamespace(describe=lambda: "seeded")
    )

    class _Collection:
        def drop(self) -> None:
            pass

    class _Connection:
        def __enter__(self):
            return _Collection()

        def __exit__(self, *exc):
            return False

    from osprey.simulation import apply as apply_mod

    monkeypatch.setattr(apply_mod, "archiver_collection", lambda store: _Connection())


def _stage(config: dict, project_dir: Path) -> None:
    container_lifecycle._stage_archiver_store(config, ["compose.yml"], {}, project_dir)


def test_the_archiver_store_bring_up_spools_its_output(
    captured, reporter, archiver_stubs, no_bare_subprocess, tmp_path
):
    """The store is started on its own, ahead of the stack, and boots noisily."""
    _stage({"deployed_services": ["archiver_store"]}, tmp_path)

    call = captured.call("archiver-store-up")
    assert call["cmd"] == ["docker", "compose", "up", "-d", "mongodb"]
    assert call["repo_root"] == tmp_path
    assert "archiver store started" in reporter.steps


def test_the_recorder_quiesce_stays_best_effort(
    captured, reporter, archiver_stubs, no_bare_subprocess, tmp_path
):
    """A recorder that will not stop warns; it does not abort the reseed.

    Capturing must not change that. ``check=False`` keeps the non-zero exit a
    return code the caller inspects rather than a ``CapturedProcessError`` that
    takes down a deploy the old code carried on through.
    """
    _stage(
        {"deployed_services": ["archiver_store", "archiver_recorder"]},
        tmp_path,
    )

    call = captured.call("archiver-recorder-stop")
    assert call["cmd"] == ["docker", "compose", "stop", "archiver-recorder"]
    assert call["repo_root"] == tmp_path
    assert call["check"] is False
    assert "archiver recorder quiesced" in reporter.steps


def test_a_recorder_that_will_not_stop_still_warns(
    captured, reporter, archiver_stubs, monkeypatch, tmp_path, caplog
):
    """The quiesce invariant is still reported when the stop exits non-zero."""

    def _failing(cmd, *, env=None, spool_name, repo_root=None, check=True):
        code = 1 if spool_name == "archiver-recorder-stop" else 0
        return subprocess.CompletedProcess(list(cmd), code)

    monkeypatch.setattr(container_lifecycle, "run_captured", _failing)

    with caplog.at_level(logging.WARNING):
        _stage({"deployed_services": ["archiver_store", "archiver_recorder"]}, tmp_path)

    assert "Could not stop `archiver-recorder`" in caplog.text


def test_the_archiver_compose_echoes_are_debug_only(
    captured, reporter, archiver_stubs, tmp_path, caplog
):
    """The argv echoes are for debugging, not for the operator's terminal.

    These two were the last ``INFO``-level ``Running command:`` lines on the
    deploy path — the reason a quiet ``up`` printed raw compose argv beside its
    phase lines.
    """
    with caplog.at_level(logging.INFO):
        _stage({"deployed_services": ["archiver_store", "archiver_recorder"]}, tmp_path)
    assert "Running command" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _stage({"deployed_services": ["archiver_store", "archiver_recorder"]}, tmp_path)
    assert caplog.text.count("Running command") == 2


# ---------------------------------------------------------------------------
# The stack's own compose invocations
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    """A repo root with the ``.env`` and compose file a start sequence reads."""
    root = tmp_path / "repo"
    services = root / "build" / "services"
    services.mkdir(parents=True)
    (root / ".env").write_text("", encoding="utf-8")
    (services / "docker-compose.0.yml").write_text("services: {}\n", encoding="utf-8")
    return root


@pytest.fixture
def start_stack_stubs(monkeypatch):
    """The host-touching preflights ``_start_stack`` runs before its compose calls."""
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_preflight_host_ports", lambda config, files: None)
    # Asks the runtime which of this project's data volumes exist. Stubbed for
    # the same reason as the two above: left live it reads the developer's own
    # daemon, so a stray `proj_*` volume on one machine would fail tests that
    # are about spooling, and CI — with no such volume — could never reproduce
    # it. Its own behaviour is pinned in test_stale_store_volume_preflight.py.
    monkeypatch.setattr(container_lifecycle, "_preflight_stale_store_volumes", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda config, files: None)
    monkeypatch.setattr(
        container_lifecycle, "_build_project_image", lambda config, dev, env, ctx=None: None
    )


def _start(repo: Path, *, dev_mode: bool = False) -> None:
    container_lifecycle._start_stack(
        {"project_name": "proj", "deployed_services": ["postgresql"]},
        ["build/services/docker-compose.0.yml"],
        repo,
        detached=True,
        dev_mode=dev_mode,
        env_path=repo / ".env",
    )


def test_the_stack_compose_calls_all_spool_their_output(
    captured, reporter, start_stack_stubs, no_bare_subprocess, tmp_path
):
    """``rm``, ``build`` and ``up`` — the three loudest calls in a dev deploy.

    Ordering is part of the contract: the self-heal ``rm`` clears wedged
    containers before the build, and ``up --no-build`` runs on the images the
    build just produced.
    """
    repo = _repo(tmp_path)

    _start(repo, dev_mode=True)

    assert captured.spool_names == ["compose-rm", "compose-build", "compose-up"]
    for name in captured.spool_names:
        assert captured.call(name)["repo_root"] == repo, f"{name} spooled outside the repo"
    assert reporter.steps == [
        "cleared stopped containers",
        "service images built",
        "containers started",
    ]


def test_the_self_heal_removal_stays_best_effort(
    captured, reporter, start_stack_stubs, no_bare_subprocess, tmp_path
):
    """``compose rm`` is advisory: if it fails, ``up`` surfaces the real error.

    Raising here would turn a self-heal that was always allowed to fail into a
    deploy-stopping error, which is exactly the behaviour change a capture
    conversion must not smuggle in.
    """
    _start(_repo(tmp_path))

    assert captured.call("compose-rm")["check"] is False
    assert captured.call("compose-up")["check"] is True


def test_a_non_dev_deploy_does_not_build(captured, reporter, start_stack_stubs, tmp_path):
    """Compose's implicit build-on-up covers it; a separate build step would not."""
    _start(_repo(tmp_path))

    assert captured.spool_names == ["compose-rm", "compose-up"]
    assert "service images built" not in reporter.steps


def test_the_stack_compose_echoes_are_debug_only(
    captured, reporter, start_stack_stubs, tmp_path, caplog
):
    """No raw compose argv on an operator's terminal."""
    with caplog.at_level(logging.INFO):
        _start(_repo(tmp_path), dev_mode=True)

    assert "Running command" not in caplog.text


# ---------------------------------------------------------------------------
# Called outside a verb
# ---------------------------------------------------------------------------


def test_steps_are_silent_without_an_open_phase(captured, start_stack_stubs, tmp_path):
    """These helpers are also library calls, and tests, and ``up_as_built``.

    Only a lifecycle verb installs a reporter and opens a phase. Every other
    caller reaches the same code with the default ``NullReporter``, so a step
    line must be a no-op rather than an attribute error on ``None``.
    """
    _start(_repo(tmp_path), dev_mode=True)

    assert captured.spool_names == ["compose-rm", "compose-build", "compose-up"]
