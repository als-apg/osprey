"""The lifecycle verbs' wiring to the phase reporter.

What is asserted here is the wiring, not the rendering: which reporter each verb
installs, that a chained verb reports into the outer verb's reporter instead of
installing a second one, and the phase titles each verb opens in order. The line
shapes themselves belong to ``test_phase_reporter.py``.

Phase lines are read off a recording reporter rather than out of ``result.output``
on purpose. Reporter output goes through the Rich console singleton, which wraps
to the terminal width a CliRunner happens to present, so an assertion on captured
stdout is an assertion about wrapping.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.cli.phase_reporter import (
    PhaseReporter,
    current_reporter,
    install_reporter,
    is_verbose,
)


class RecordingReporter(PhaseReporter):
    """A real reporter that keeps its lines instead of printing them."""

    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        self.lines.append(text)


def titles(reporter: RecordingReporter) -> list[str]:
    """The phase-opening lines, in the order the verb opened them."""
    return [line for line in reporter.lines if line.startswith("→ ")]


def closed(reporter: RecordingReporter) -> list[str]:
    """The phase-closing lines (✓/✗/⚠), stripped of their indent, in order."""
    return [line.strip() for line in reporter.lines if line.lstrip()[:1] in {"✓", "✗", "⚠"}]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _container_runtime_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer ``init``'s runtime preflight without consulting the host.

    ``--reset`` and ``--up`` are refused up front when no container runtime
    answers, which is the right behaviour and the wrong dependency for a unit
    test: left unstubbed these tests pass on a developer's machine with Docker
    Desktop open and fail on a CI runner that has no daemon, which is a
    property of the runner rather than of the verb.

    Patched on ``runtime_helper`` itself because that is where the preflight
    reads it from, at call time. The other callers of this function bind it at
    import into their own modules, so this reaches exactly the one check these
    tests need to get past and none of theirs.
    ``TestContainerRuntimePreflight`` patches it back down, in the test body,
    where it wins over this.
    """
    monkeypatch.setattr(
        "osprey.deployment.runtime_helper.verify_runtime_is_running",
        lambda config=None: (True, ""),
    )


@pytest.fixture
def recorder():
    """Pre-install a recording reporter, as an outer verb would have.

    Verbs invoked with this in place find a reporter already installed and
    report into it — the same path ``init --up`` puts ``build`` and ``up`` on.
    """
    recording = RecordingReporter()
    previous = install_reporter(recording)
    try:
        yield recording
    finally:
        install_reporter(previous)


@pytest.fixture
def restore_reporter():
    """Put the module handle back, for tests that let a verb install its own."""
    previous = current_reporter()
    try:
        yield
    finally:
        install_reporter(previous)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A directory standing in for a built deployment repo."""
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text("{}\n")
    return tmp_path


@pytest.fixture
def seen() -> list:
    """Where the stubs record what reporter was installed when they ran."""
    return []


@pytest.fixture
def deploy_stubs(monkeypatch, repo: Path, seen: list):
    """Stub everything `up`/`restart`/`down` reach outside the CLI layer."""
    from osprey.cli import deploy_cmd, repo_resolver
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(
        repo_resolver, "find_repo_root", lambda start=None: Path(start) if start else repo
    )
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "ensure_repo_env", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "load_project_config", lambda *a, **k: {})
    monkeypatch.setattr(
        container_lifecycle, "as_built_config_path", lambda root: repo / "build" / "config.yml"
    )

    def record(verb: str):
        def stub(*args, **kwargs):
            seen.append((verb, type(current_reporter()).__name__, is_verbose()))

        return stub

    monkeypatch.setattr(container_lifecycle, "up_as_built", record("up"))
    monkeypatch.setattr(container_lifecycle, "restart_deployment", record("restart"))
    monkeypatch.setattr(container_lifecycle, "down_deployment", record("down"))
    return seen


# ---------------------------------------------------------------------------
# Which reporter gets installed
# ---------------------------------------------------------------------------


class TestInstalledReporter:
    """A verb installs one reporter, and `--verbose` decides which."""

    @pytest.mark.parametrize("verb", ["up", "restart", "down"])
    def test_a_start_or_stop_verb_installs_a_real_reporter(
        self, runner, deploy_stubs, restore_reporter, verb
    ):
        result = runner.invoke(cli, [verb])

        assert result.exit_code == 0, result.output
        assert deploy_stubs == [(verb, "PhaseReporter", False)]

    @pytest.mark.parametrize("verb", ["up", "restart", "down"])
    def test_verbose_installs_a_null_reporter_that_says_it_is_verbose(
        self, runner, deploy_stubs, restore_reporter, verb
    ):
        """A bare NullReporter would silence the phases AND leave `is_verbose()`
        False, which is what the capture helper reads to decide between
        streaming a subprocess and spooling it. Under `--verbose` the operator
        asked for the stream, so the flag has to travel with the reporter."""
        result = runner.invoke(cli, ["--verbose", verb])

        assert result.exit_code == 0, result.output
        assert deploy_stubs == [(verb, "NullReporter", True)]

    def test_the_default_handle_is_restored_afterwards(
        self, runner, deploy_stubs, restore_reporter
    ):
        """The install is scoped to the verb: a process that goes on to do
        something else must not keep printing into a finished run's reporter."""
        before = current_reporter()

        result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0, result.output
        assert current_reporter() is before

    def test_an_already_installed_reporter_is_not_clobbered(self, runner, deploy_stubs, recorder):
        """The rule the `init --up` chain rests on: an inner verb reports into
        the reporter it finds rather than installing a second one."""
        result = runner.invoke(cli, ["up"])

        assert result.exit_code == 0, result.output
        assert deploy_stubs == [("up", "RecordingReporter", False)]
        assert current_reporter() is recorder


# ---------------------------------------------------------------------------
# The phases each verb opens
# ---------------------------------------------------------------------------


class TestDeployVerbPhases:
    def test_up_reports_a_preflight_and_then_the_start(self, runner, deploy_stubs, recorder, repo):
        result = runner.invoke(cli, ["up", "-d"])

        assert result.exit_code == 0, result.output
        assert titles(recorder) == ["→ Preflight", f"→ Starting {repo.name}"]
        assert closed(recorder)[0].startswith("✓ Preflight")
        assert closed(recorder)[0].endswith("— build/ and .env are in place")
        assert closed(recorder)[1].startswith(f"✓ Starting {repo.name}")

    def test_restart_reports_the_stop_and_start_as_one_phase(
        self, runner, deploy_stubs, recorder, repo
    ):
        """`restart` recreates the containers in one call. A ✓ on a separate
        stop phase would read as "it is down now" while the start is still to
        come."""
        result = runner.invoke(cli, ["restart", "-d"])

        assert result.exit_code == 0, result.output
        assert titles(recorder) == ["→ Preflight", f"→ Restarting {repo.name}"]

    def test_down_reports_one_stopping_phase(self, runner, deploy_stubs, recorder, repo):
        result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0, result.output
        assert titles(recorder) == [f"→ Stopping {repo.name}"]

    def test_an_attached_start_renders_its_phases_up_to_the_exec_point(
        self, runner, deploy_stubs, recorder, monkeypatch, repo
    ):
        """Attached (`up` without `-d`) hands the terminal to compose with
        os.execvpe, so this process is gone before the start phase can close.
        What is guaranteed is what this asserts: the phases before the exec are
        rendered, and the start phase is open when the exec happens."""
        from osprey.deployment import container_lifecycle

        open_phase: list[str | None] = []

        def stub(*args, **kwargs):
            phase = current_reporter().current_phase
            open_phase.append(phase.title if phase else None)

        monkeypatch.setattr(container_lifecycle, "up_as_built", stub)

        result = runner.invoke(cli, ["up"])

        assert result.exit_code == 0, result.output
        assert open_phase == [f"Starting {repo.name}"]
        assert closed(recorder)[0].startswith("✓ Preflight")

    def test_a_failed_start_closes_its_phase_before_the_error(
        self, runner, deploy_stubs, recorder, monkeypatch, repo
    ):
        """The ✗ line has to come from the phase, not from the handler: the
        handler's message is about the deployment, the phase line is about which
        step of the run stopped."""
        from osprey.deployment import container_lifecycle

        def boom(*args, **kwargs):
            raise RuntimeError("compose said no")

        monkeypatch.setattr(container_lifecycle, "up_as_built", boom)

        result = runner.invoke(cli, ["up", "-d"])

        assert result.exit_code != 0
        assert closed(recorder)[-1].startswith(f"✗ Starting {repo.name}")

    def test_preflight_fails_its_phase_when_there_is_no_build(
        self, runner, deploy_stubs, recorder, monkeypatch, tmp_path
    ):
        from osprey.deployment import container_lifecycle

        monkeypatch.setattr(
            container_lifecycle,
            "as_built_config_path",
            lambda root: tmp_path / "gone" / "config.yml",
        )

        result = runner.invoke(cli, ["up", "-d"])

        assert result.exit_code != 0
        assert titles(recorder) == ["→ Preflight"]
        assert closed(recorder) == [closed(recorder)[0]]
        assert closed(recorder)[0].startswith("✗ Preflight")


class TestBuildPhases:
    """`build` reports the venv and the six render passes.

    Everything under the verb is stubbed: what is asserted is which phases the
    render opens and in what order, and no real build is run for it.
    """

    @pytest.fixture
    def build_stubs(self, monkeypatch, tmp_path: Path, seen):
        from osprey.cli import build_cmd, build_profile, build_profile_deploy

        class _Lifecycle:
            pre_build = None
            post_build = None
            validate = None

        class _Profile:
            name = "probe"
            data_bundle = "control-assistant"
            provider = "anthropic"
            requires_osprey_version = None
            dependencies: list[str] = []
            lifecycle = _Lifecycle()
            config: dict = {}
            deploy = None
            # A deploying profile: _build_repo reads the flag to decide whether
            # the render just written becomes the host config every attached
            # render after it is projected from.
            deploy_services = True
            # No live stand-in: _build_repo reads the block after the swap to
            # decide whether to run the stand-in's lattice gate.
            virtual_accelerator = None

            def resolved_tier(self) -> int:
                return 1

        class _Resolved:
            profile = _Profile()
            excluded_artifacts: list[str] = []

        (tmp_path / "profile.yml").write_text("name: probe\n")
        monkeypatch.setattr(build_cmd, "find_repo_root", lambda repo=None: tmp_path)
        monkeypatch.setattr(build_profile, "resolve_build_document", lambda *a, **k: _Resolved())
        monkeypatch.setattr(build_profile_deploy, "deploy_aware_config_errors", lambda *a, **k: [])
        monkeypatch.setattr(
            build_profile_deploy, "deploy_aware_config_warnings", lambda *a, **k: []
        )
        monkeypatch.setattr(build_cmd, "TemplateManager", lambda *a, **k: object())
        monkeypatch.setattr(build_cmd, "_create_project_venv", lambda *a, **k: [])

        def _render(*a, injected_out=None, **k):
            """Stand in for a render, recording one injected service.

            Returns a directory with a ``config.yml`` because the real render
            does: `_build_repo` reads the deployment's rendered config back as
            the host every later attached render is projected from.
            """
            if injected_out is not None:
                injected_out.append("event dispatch")
            render_dir = tmp_path / "build-stage"
            render_dir.mkdir(exist_ok=True)
            (render_dir / "config.yml").write_text("{}\n", encoding="utf-8")
            return render_dir

        monkeypatch.setattr(build_cmd, "_render_project", _render)
        monkeypatch.setattr(build_cmd, "_render_compose_files", lambda *a, **k: {})
        monkeypatch.setattr(
            build_cmd, "_render_persona_projects", lambda *a, **k: [tmp_path / "readonly"]
        )
        monkeypatch.setattr(
            build_cmd, "_render_container_projects", lambda *a, **k: [tmp_path / "image"]
        )
        monkeypatch.setattr(build_cmd, "_backup_outgoing_claude_artifacts", lambda *a, **k: None)
        monkeypatch.setattr(build_cmd, "_prune_runtime_state_from_stage", lambda *a, **k: None)
        monkeypatch.setattr(build_cmd, "_swap_in_render", lambda *a, **k: None)
        monkeypatch.setattr(build_cmd, "_wire_build_derived_env", lambda *a, **k: None)
        monkeypatch.setattr(build_cmd, "_warn_if_deployment_running", lambda *a, **k: None)
        return seen

    def test_build_reports_the_venv_and_the_render(self, runner, build_stubs, recorder):
        """Six render passes, one phase. Reported one by one they would read as
        six builds of six different things."""
        result = runner.invoke(cli, ["build"])

        assert result.exit_code == 0, result.output
        assert titles(recorder) == [
            "→ Preparing the project environment",
            "→ Rendering the configuration",
        ]
        steps = [line.split(" (")[0].strip() for line in recorder.lines if "·" in line]
        assert steps == [
            "· project files, agent artifacts and services",
            # Named once, from the deployment's render — the other five passes
            # inject the same set and say nothing about it.
            "· services injected: event dispatch",
            "· compose files",
            "· 1 persona render(s)",
            "· 1 image build context(s)",
        ]

    def test_skip_deps_opens_no_environment_phase(self, runner, build_stubs, recorder):
        """--skip-deps is CI's "there is no venv to make". A phase announced for
        work that was skipped is the kind of small lie this feature removes."""
        result = runner.invoke(cli, ["build", "--skip-deps"])

        assert result.exit_code == 0, result.output
        assert titles(recorder) == ["→ Rendering the configuration"]

    @pytest.fixture
    def installed(self, monkeypatch):
        """What `build` had installed by the time the render would have run."""
        from osprey.cli import build_cmd

        recorded: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            build_cmd,
            "_build_repo",
            lambda *a, **k: recorded.append((type(current_reporter()).__name__, is_verbose())),
        )
        return recorded

    def test_build_installs_its_own_reporter(
        self, runner, build_stubs, restore_reporter, installed
    ):
        result = runner.invoke(cli, ["build"])

        assert result.exit_code == 0, result.output
        assert installed == [("PhaseReporter", False)]

    def test_verbose_silences_the_build_phases(
        self, runner, build_stubs, restore_reporter, installed
    ):
        """The render is the biggest captured-subprocess consumer on the build
        path, so `is_verbose()` being True here is what makes `--verbose` mean
        "stream it" rather than just "print no phases"."""
        result = runner.invoke(cli, ["--verbose", "build"])

        assert result.exit_code == 0, result.output
        assert installed == [("NullReporter", True)]


class TestResetPhases:
    """`reset` gets a reporter, but keeps reporting on its own `emit` callback."""

    @pytest.fixture
    def reset_stub(self, monkeypatch, repo: Path, seen):
        from osprey.cli import reset_cmd
        from osprey.deployment import reset as reset_mod

        # reset_cmd binds find_repo_root at import, so the name to replace is
        # the one in reset_cmd, not the one in repo_resolver.
        monkeypatch.setattr(reset_cmd, "find_repo_root", lambda start=None: repo)
        monkeypatch.setattr(reset_mod, "reset_deployment", lambda *a, **k: _record(seen))
        return seen

    def test_reset_installs_a_reporter_and_opens_no_phase(self, runner, reset_stub, recorder):
        result = runner.invoke(cli, ["reset", "--dry-run", "-y"])

        assert result.exit_code == 0, result.output
        assert reset_stub == [("reset", "RecordingReporter", False)]
        assert recorder.lines == []

    def test_verbose_reaches_reset_too(self, runner, reset_stub, restore_reporter):
        """Reset's teardown reaches the same captured-subprocess helpers the
        start verbs do, and `--verbose` has to mean the same thing there."""
        result = runner.invoke(cli, ["--verbose", "reset", "--dry-run", "-y"])

        assert result.exit_code == 0, result.output
        assert reset_stub == [("reset", "NullReporter", True)]


def _record(seen):
    """Record the installed reporter, then end the reset as a dry run."""
    from osprey.deployment.reset import ResetOutcome

    seen.append(("reset", type(current_reporter()).__name__, is_verbose()))
    return ResetOutcome.DRY_RUN


# ---------------------------------------------------------------------------
# init, and the one reporter its --up chain shares
# ---------------------------------------------------------------------------


@pytest.fixture
def init_stubs(monkeypatch, seen):
    """Stub what `init` reaches outside the CLI layer, and the chained verbs.

    `find_repo_root` is deliberately NOT stubbed: `init` asks it whether the
    target would nest inside an existing deployment, and the chained verbs ask
    it where the deployment they were handed is. Both answers have to come from
    the same directory the run actually creates, so the materialize stub writes
    the `profile.yml` that makes the target a repo — the one thing the real
    helper does that the rest of the chain depends on.
    """
    from osprey.cli import build_cmd, deploy_cmd, init_cmd, profile_cmd
    from osprey.deployment import container_lifecycle

    class _Materialized:
        profile_name = "probe"
        deploy_declared = False

    def materialize(target, *args, **kwargs):
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "profile.yml").write_text("name: probe\n")
        return _Materialized()

    def rendered_config(root: Path) -> Path:
        path = Path(root) / "build" / "config.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return path

    monkeypatch.setattr(profile_cmd, "_materialize_profile_directory", materialize)
    monkeypatch.setattr(init_cmd, "_emit_ci", lambda target, declared: [])
    monkeypatch.setattr(init_cmd, "_report", lambda *a, **k: None)

    # The chain: both verbs are looked up on the root group and invoked for
    # real, so what is stubbed is the work under them, not the wiring.
    monkeypatch.setattr(build_cmd, "_build_repo", lambda *a, **k: _seen(seen, "build"))
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "ensure_repo_env", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "load_project_config", lambda *a, **k: {})
    monkeypatch.setattr(container_lifecycle, "as_built_config_path", rendered_config)
    monkeypatch.setattr(container_lifecycle, "up_as_built", lambda *a, **k: _seen(seen, "up"))
    return seen


def _seen(seen, verb: str) -> None:
    seen.append((verb, type(current_reporter()).__name__, id(current_reporter())))


class TestInitPhases:
    def test_init_reports_one_creation_phase(self, runner, init_stubs, recorder, tmp_path):
        target = tmp_path / "probe"

        result = runner.invoke(
            cli, ["init", str(target), "--preset", "control-assistant", "--no-git"]
        )

        assert result.exit_code == 0, result.output
        assert titles(recorder) == ["→ Creating probe"]
        assert "  · settings and data from preset control-assistant" in [
            line.split(" (")[0] for line in recorder.lines
        ]
        assert closed(recorder)[-1].startswith("✓ Creating probe")

    def test_a_refused_init_opens_no_phase(self, runner, init_stubs, recorder, tmp_path):
        """A run that is turned away prints its reason and nothing else — a
        `→ Creating x` above it would announce work that never started."""
        result = runner.invoke(cli, ["init", str(tmp_path / "probe")])

        assert result.exit_code != 0
        assert recorder.lines == []

    def test_the_up_chain_reports_into_one_reporter(self, runner, init_stubs, recorder, tmp_path):
        """`init --up` is three commands and one run. The chained verbs find
        this reporter through the module handle — nothing is threaded through
        their signatures — so the operator reads one sequence of phases."""
        target = tmp_path / "probe"

        result = runner.invoke(
            cli,
            ["init", str(target), "--preset", "control-assistant", "--no-git", "--up", "-d"],
        )

        assert result.exit_code == 0, result.output
        assert [verb for verb, _, _ in init_stubs] == ["build", "up"]
        assert {name for _, name, _ in init_stubs} == {"RecordingReporter"}
        assert len({handle for _, _, handle in init_stubs}) == 1
        assert titles(recorder) == ["→ Creating probe", "→ Preflight", "→ Starting probe"]

    def test_the_chain_shares_the_reporter_init_installed(
        self, runner, init_stubs, restore_reporter, tmp_path
    ):
        """Same property with nothing pre-installed: the reporter the chained
        verbs report into is the one `init` installed, not one of their own."""
        target = tmp_path / "probe"

        result = runner.invoke(
            cli,
            ["init", str(target), "--preset", "control-assistant", "--no-git", "--up", "-d"],
        )

        assert result.exit_code == 0, result.output
        assert {name for _, name, _ in init_stubs} == {"PhaseReporter"}
        assert len({handle for _, _, handle in init_stubs}) == 1


# ---------------------------------------------------------------------------
# What `--verbose` puts back
# ---------------------------------------------------------------------------


class _RecordSink(logging.Handler):
    """Every record that survived the level filter, as its formatted message."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def debug_probe(monkeypatch, deploy_stubs, restore_reporter):
    """One DEBUG record emitted from inside a running verb, and what survived.

    The demoted call sites all still log — at DEBUG, which
    ``test_build_output_quiet.py`` pins. What decides whether an operator ever
    SEES them again is the level `--verbose` hands ``configure_logging``, and
    that is a different link in the chain from the reporter the verb installs.
    Emitting from inside the verb reads the level the group callback actually
    applied, rather than the one the flag was supposed to imply.
    """
    from osprey.deployment import container_lifecycle

    sink = _RecordSink()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(sink)

    monkeypatch.setattr(
        container_lifecycle,
        "down_deployment",
        lambda *a, **k: logging.getLogger("build").debug("a demoted call site speaking"),
    )
    try:
        yield sink
    finally:
        root.removeHandler(sink)
        root.setLevel(previous_level)


def test_the_default_view_still_filters_the_demoted_records_out(runner, debug_probe):
    """Without the flag the demotion is a real one: the record never emits."""
    result = runner.invoke(cli, ["down"])

    assert result.exit_code == 0, result.output
    assert debug_probe.messages == []


def test_verbose_lets_the_demoted_records_through(runner, debug_probe):
    """SC4: `--verbose` is the escape hatch for everything SC1 took off INFO."""
    result = runner.invoke(cli, ["--verbose", "down"])

    assert result.exit_code == 0, result.output
    assert "a demoted call site speaking" in debug_probe.messages


# ---------------------------------------------------------------------------
# The lazy-import budget
# ---------------------------------------------------------------------------


def test_a_fresh_interpreter_neither_imports_nor_installs_a_reporter():
    """Two invariants that only a fresh process can be asked about.

    First: `osprey --help` must not pay for the reporter, which is why the
    verbs import `phase_reporter` inside their command bodies — a top-level
    import anywhere on the CLI entry path would put it back on the budget.

    Second: the module default is a NullReporter that does NOT claim
    verbosity. That is the one shape no verb ever installs, which is what makes
    "somebody already installed one" decidable at all.

    In-process both questions are unanswerable: this test module has imported
    the reporter, and every test above installs one.
    """
    probe = (
        "import sys; import osprey.cli.main;"
        "print('osprey.cli.phase_reporter' in sys.modules);"
        "from osprey.cli.phase_reporter import NullReporter, current_reporter;"
        "d = current_reporter();"
        "print(isinstance(d, NullReporter), d.verbose)"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "True", "False"]
