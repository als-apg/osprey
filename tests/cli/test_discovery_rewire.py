"""One discovery rule, applied to ``web``, ``web stop``, ``query`` and ``sim``.

These four were the last commands deciding for themselves where a deployment
is. ``web`` had a three-way contract (``--project``, then an ambient
``OSPREY_CONFIG``, then ``./config.yml``) with different chdir semantics on
each branch; ``web stop`` had a *fourth*, so an operator could stop a server
they had not started. ``query`` had a fifth (``--project``, then
``OSPREY_PROJECT``, then cwd), and ``sim`` simply required the working
directory to be the project root. What is pinned here is that all of them now
walk up to the nearest ``profile.yml`` and nothing else does the deciding.

Three properties recur, and each is checked per command rather than once:

- **Any subdirectory works.** Every command is driven from a directory nested
  inside the exemplar repo, never from its root, so a command that still read
  cwd as the project root would fail rather than pass by coincidence.
- **The render is what is acted on.** ``build/config.yml`` is the config; the
  repo root is what durable paths anchor at. The two are different directories
  now and a command that conflates them is caught by the paths it produces.
- **The environment does not discover.** An ambient ``OSPREY_CONFIG`` naming a
  *different* deployment must not change what these commands serve. It stays
  a publication these commands make for their children, which is the FR-10
  carve-out.

Nothing here starts a server, an agent, or a database: the handoffs (``run_web``,
``run_query``, ``subprocess.Popen``) are recorded instead.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.query_cmd import query
from osprey.cli.sim import sim_group
from osprey.cli.web_cmd import LOG_FILE, PID_FILE, web
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess

#: A render that is also simulation-backed. The model path is repo-relative on
#: purpose: resolving it is exactly the question of which directory the sim
#: commands anchor at, and the exemplar ships the model at that path.
SIM_CONFIG = """\
claude_code:
  provider: anthropic
control_system:
  type: mock
  connector:
    mock:
      simulation_file: data/simulation/machine.json
"""


def nested(repo: Path) -> Path:
    """A directory deep inside *repo* to run commands from.

    Part of the source zone rather than an invented one: an operator editing
    ``data/knowledge/`` and then typing ``osprey web`` is the case the walk-up
    rule exists for.
    """
    deep = repo / "data" / "knowledge"
    deep.mkdir(parents=True, exist_ok=True)
    return deep


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Undo what an in-process launch does to the interpreter's globals.

    These commands chdir into the render and publish ``OSPREY_CONFIG`` — both
    deliberate, both process-global, and both able to reach every test that
    runs after this one in the same worker.
    """
    cwd = Path.cwd()
    environ = dict(os.environ)
    yield
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def _no_ambient_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an environment that names no deployment.

    A developer's own shell often exports these. Clearing them keeps the
    "environment does not discover" assertions honest: the tests that *set*
    ``OSPREY_CONFIG`` set it themselves, and the rest must not inherit one.
    """
    for key in ("OSPREY_CONFIG", "OSPREY_PROJECT", "OSPREY_WEB_PORT"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# osprey web — the launch
# ---------------------------------------------------------------------------


@dataclass
class WebLaunch:
    """One recorded handoff to the web-terminal server."""

    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def config_path(self) -> Path:
        return Path(self.kwargs["config_path"])

    @property
    def project_dir(self) -> Path:
        return Path(self.kwargs["project_dir"])


@pytest.fixture
def web_launch(monkeypatch: pytest.MonkeyPatch) -> WebLaunch:
    """Record ``run_web``'s arguments instead of binding a port.

    Patched on the module ``web_cmd`` imports it *from*, because the import is
    function-local — there is no name in ``web_cmd`` to replace.
    """
    launch = WebLaunch()

    def _record(**kwargs: Any) -> None:
        launch.kwargs = kwargs

    import osprey.interfaces.web_terminal as web_terminal

    monkeypatch.setattr(web_terminal, "run_web", _record)
    return launch


def run_web_cmd(runner: CliRunner, args: list[str], cwd: Path):
    """Invoke ``osprey web`` from *cwd* with pre-flight and the port probe out of the way.

    Pre-flight is skipped because its probes (companion ports, provider auth)
    are not what these tests are about, and the socket bind is stubbed so a
    busy 8087 on the developer's machine cannot silently move the port.
    """
    os.chdir(cwd)
    with patch("socket.socket"):
        return runner.invoke(web, [*args, "--skip-preflight"], catch_exceptions=False)


class TestWebResolvesTheRepo:
    """``osprey web`` serves the deployment enclosing the caller."""

    def test_serves_the_render_from_a_nested_subdirectory(self, runner, lifecycle_repo, web_launch):
        build = stub_build(lifecycle_repo)

        result = run_web_cmd(runner, [], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert web_launch.config_path == build / "config.yml"
        assert web_launch.project_dir == build

    def test_repo_flag_naming_a_subdirectory_resolves_the_repo(
        self, runner, lifecycle_repo, tmp_path, web_launch
    ):
        """``--repo`` moves the walk's starting point; it does not skip the walk."""
        build = stub_build(lifecycle_repo)
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        result = run_web_cmd(runner, ["--repo", str(nested(lifecycle_repo))], outside)

        assert result.exit_code == 0, result.output
        assert web_launch.project_dir == build

    def test_outside_any_repo_is_refused(self, runner, tmp_path, web_launch):
        outside = tmp_path / "not-a-repo"
        outside.mkdir()

        result = run_web_cmd(runner, [], outside)

        assert result.exit_code == 1
        assert "No OSPREY deployment repo found" in result.output
        assert web_launch.kwargs == {}

    def test_repo_without_a_build_is_refused(self, runner, lifecycle_repo, web_launch):
        result = run_web_cmd(runner, [], nested(lifecycle_repo))

        assert result.exit_code == 1
        assert "no build found" in result.output
        assert "osprey build" in result.output
        assert web_launch.kwargs == {}

    def test_ambient_osprey_config_does_not_choose_the_deployment(
        self, runner, lifecycle_repo, lifecycle_repo_factory, tmp_path, monkeypatch, web_launch
    ):
        """The env var is a publication for children, never a discovery input."""
        other = lifecycle_repo_factory(tmp_path / "other-deployment")
        other_build = stub_build(other)
        build = stub_build(lifecycle_repo)
        monkeypatch.setenv("OSPREY_CONFIG", str(other_build / "config.yml"))

        result = run_web_cmd(runner, [], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert web_launch.project_dir == build
        # ...and the stale export was replaced, not merely ignored, so the PTY
        # shells and their MCP servers resolve what this process resolved.
        assert os.environ["OSPREY_CONFIG"] == str(build / "config.yml")

    def test_project_flag_is_gone(self, runner, lifecycle_repo):
        os.chdir(lifecycle_repo)
        result = runner.invoke(web, ["--project", str(lifecycle_repo)])

        assert result.exit_code == 2
        assert "no such option" in result.output.lower()


# ---------------------------------------------------------------------------
# osprey web --detach / osprey web stop — one rule, both ends
# ---------------------------------------------------------------------------


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch):
    """Record the detached child's argv without starting a process.

    Scoped through ``patch_subprocess(..., popen=...)``, never by setting
    ``Popen`` on the stdlib module — after ``import subprocess``,
    ``web_cmd.subprocess`` *is* ``sys.modules["subprocess"]``, so patching
    through it is global and would swallow the ``git init`` the exemplar fixture
    runs (and anything else spawning in this worker).
    """
    calls: list[list[str]] = []

    class _Proc:
        pid = 4321

        def poll(self):
            return None

    def _popen(cmd, **_kwargs):
        calls.append(list(cmd))
        return _Proc()

    import osprey.cli.web_cmd as web_cmd

    monkeypatch.setattr(web_cmd, "_wait_for_server", lambda *a, **k: True)
    with patch_subprocess("osprey.cli.web_cmd", popen=_popen):
        yield calls


class TestDetachedServerHandle:
    """The PID/log pair belongs to the repo, and ``stop`` looks where ``start`` wrote."""

    def test_pid_and_log_land_in_the_state_zone(self, runner, lifecycle_repo, spawned):
        build = stub_build(lifecycle_repo)

        result = run_web_cmd(runner, ["--detach"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert (lifecycle_repo / "var" / "osprey-web.pid").read_text() == "4321"
        assert (lifecycle_repo / "var" / "osprey-web.log").is_file()
        # Not the render: every build wipes it, and a rebuild underneath a
        # running server would take the only handle on it too.
        assert not (build / PID_FILE).exists()
        assert not (build / LOG_FILE).exists()

    def test_a_running_server_leaves_the_repo_git_clean(
        self, runner, lifecycle_repo_factory, tmp_path, spawned
    ):
        """SC-1's clean `git status` has to survive a detached server.

        This is why the pair is in ``var/`` rather than at the repo root: the
        shipped ``.gitignore`` anchors ``/var/``, so the files are covered by a
        rule the repo already has. At the root they would need a new one, and
        an operator who started a server would find their deployment dirty.
        """
        repo = lifecycle_repo_factory(tmp_path / "git-backed", git=True)
        stub_build(repo)

        run_web_cmd(runner, ["--detach"], nested(repo))

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert status.stdout == ""

    def test_child_argv_names_the_repo(self, runner, lifecycle_repo, spawned):
        stub_build(lifecycle_repo)

        run_web_cmd(runner, ["--detach"], nested(lifecycle_repo))

        argv = spawned[0]
        assert "--project" not in argv
        assert argv[argv.index("--repo") + 1] == str(lifecycle_repo)

    def test_stop_from_a_nested_subdirectory_finds_the_running_server(self, runner, lifecycle_repo):
        stub_build(lifecycle_repo)
        (lifecycle_repo / PID_FILE).write_text("4321")
        (lifecycle_repo / LOG_FILE).write_text("")

        os.chdir(nested(lifecycle_repo))
        with patch("os.kill") as kill:
            result = runner.invoke(web, ["stop"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        kill.assert_called_once()
        assert kill.call_args[0][0] == 4321
        assert not (lifecycle_repo / PID_FILE).exists()

    def test_stop_takes_repo_before_the_subcommand_too(self, runner, lifecycle_repo, tmp_path):
        """``osprey web --repo X stop`` and ``osprey web stop --repo X`` agree."""
        stub_build(lifecycle_repo)
        (lifecycle_repo / PID_FILE).write_text("4321")
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        os.chdir(outside)
        with patch("os.kill") as kill:
            result = runner.invoke(
                web, ["--repo", str(lifecycle_repo), "stop"], catch_exceptions=False
            )

        assert result.exit_code == 0, result.output
        assert kill.call_args[0][0] == 4321

    def test_stop_outside_any_repo_is_refused(self, runner, tmp_path):
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        os.chdir(outside)

        result = runner.invoke(web, ["stop"], catch_exceptions=False)

        assert result.exit_code == 1
        assert "No OSPREY deployment repo found" in result.output

    def test_stop_project_flag_is_gone(self, runner, lifecycle_repo):
        os.chdir(lifecycle_repo)
        result = runner.invoke(web, ["stop", "--project", str(lifecycle_repo)])

        assert result.exit_code == 2
        assert "no such option" in result.output.lower()


# ---------------------------------------------------------------------------
# osprey query — same rule, plus chat's drift warning
# ---------------------------------------------------------------------------


@dataclass
class QueryCall:
    """One recorded handoff to the agent runner."""

    project_dir: Path | None = None


@pytest.fixture
def query_call(monkeypatch: pytest.MonkeyPatch) -> QueryCall:
    """Record ``run_query``'s project directory and return an empty result."""
    from osprey.agent_runner import SDKWorkflowResult

    call = QueryCall()

    async def _record(project_dir, prompt, **_kwargs):
        call.project_dir = Path(project_dir)
        return SDKWorkflowResult()

    import osprey.cli.query_cmd as query_cmd

    monkeypatch.setattr(query_cmd, "run_query", _record)
    monkeypatch.setattr(query_cmd, "read_only_disallowed_tools", lambda _d: [])
    monkeypatch.setattr(query_cmd, "expected_mcp_servers", lambda _d: [])
    monkeypatch.setattr(query_cmd, "evaluate_verdict", lambda _r, _e: 0)
    return call


def run_query_cmd(runner: CliRunner, args: list[str], cwd: Path):
    os.chdir(cwd)
    return runner.invoke(query, args, catch_exceptions=False)


class TestQueryResolvesTheRepo:
    def test_asks_the_render_from_a_nested_subdirectory(self, runner, lifecycle_repo, query_call):
        build = stub_build(lifecycle_repo)

        result = run_query_cmd(runner, ["what is the beam current?"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert query_call.project_dir == build

    def test_repo_flag_resolves_another_deployment(
        self, runner, lifecycle_repo, tmp_path, query_call
    ):
        build = stub_build(lifecycle_repo)
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        result = run_query_cmd(
            runner, ["--repo", str(lifecycle_repo), "list vacuum sections"], outside
        )

        assert result.exit_code == 0, result.output
        assert query_call.project_dir == build

    def test_no_build_is_a_usage_error(self, runner, lifecycle_repo, query_call):
        """Exit 2, not 1: nothing to query is a usage problem, not a failed verdict."""
        result = run_query_cmd(runner, ["anything"], nested(lifecycle_repo))

        assert result.exit_code == 2
        assert "no build found" in result.output
        assert "osprey build" in result.output
        assert query_call.project_dir is None

    def test_outside_any_repo_is_refused(self, runner, tmp_path, query_call):
        outside = tmp_path / "not-a-repo"
        outside.mkdir()

        result = run_query_cmd(runner, ["anything"], outside)

        assert result.exit_code == 1
        assert "No OSPREY deployment repo found" in result.output
        assert query_call.project_dir is None

    def test_project_flag_is_gone(self, runner, lifecycle_repo):
        os.chdir(lifecycle_repo)
        result = runner.invoke(query, ["-p", str(lifecycle_repo), "anything"])

        assert result.exit_code == 2
        assert "no such option" in result.output.lower()


class TestQueryWarnsOnDrift:
    """The warning chat gives, given here too — and never in place of an answer."""

    def test_profile_drift_warns_and_still_answers(self, runner, lifecycle_repo, query_call):
        build = stub_build(lifecycle_repo, stamped_hash="a-build-the-profile-moved-on-from")

        result = run_query_cmd(runner, ["anything"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output
        assert query_call.project_dir == build

    def test_unverifiable_build_warns_and_still_answers(self, runner, lifecycle_repo, query_call):
        stub_build(lifecycle_repo, stamped_hash=None)

        result = run_query_cmd(runner, ["anything"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output

    def test_version_skew_warns_independently(self, runner, lifecycle_repo, query_call):
        """A build can be a faithful render and still predate the framework."""
        stub_build(lifecycle_repo, version="2000.1.0")

        result = run_query_cmd(runner, ["anything"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output

    def test_a_clean_build_says_nothing(self, runner, lifecycle_repo, query_call):
        stub_build(lifecycle_repo)

        result = run_query_cmd(runner, ["anything"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert "WARNING" not in result.output


# ---------------------------------------------------------------------------
# osprey sim — the render holds the model, the repo root holds the state
# ---------------------------------------------------------------------------


def run_sim(runner: CliRunner, args: list[str], cwd: Path):
    os.chdir(cwd)
    return runner.invoke(sim_group, args, catch_exceptions=False)


class TestSimResolvesTheRepo:
    def test_list_works_from_a_nested_subdirectory(self, runner, lifecycle_repo):
        stub_build(lifecycle_repo, config=SIM_CONFIG)

        result = run_sim(runner, ["list"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert "nominal" in result.output
        assert "vacuum-burst" in result.output

    def test_status_works_from_a_nested_subdirectory(self, runner, lifecycle_repo):
        stub_build(lifecycle_repo, config=SIM_CONFIG)

        result = run_sim(runner, ["status"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert "Active scenarios" in result.output

    def test_repo_flag_resolves_another_deployment(self, runner, lifecycle_repo, tmp_path):
        stub_build(lifecycle_repo, config=SIM_CONFIG)
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        result = run_sim(runner, ["list", "--repo", str(lifecycle_repo)], outside)

        assert result.exit_code == 0, result.output
        assert "nominal" in result.output

    def test_no_build_is_refused(self, runner, lifecycle_repo):
        result = run_sim(runner, ["list"], nested(lifecycle_repo))

        assert result.exit_code == 1
        assert "no build found" in result.output

    def test_apply_writes_state_under_var_not_into_the_render(self, runner, lifecycle_repo):
        """The scenario an operator activates has to survive the next build.

        It also has to be the file the mock connectors read, and they anchor on
        the rendered config's ``project_root`` — the repo root. State written
        inside ``build/`` would be both wiped by a rebuild and invisible to the
        running control system.
        """
        build = stub_build(lifecycle_repo, config=SIM_CONFIG)

        result = run_sim(runner, ["apply", "vacuum-burst", "--no-seed"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        state = lifecycle_repo / "var" / "agent_data" / "simulation" / "active_scenarios"
        assert "vacuum-burst" in state.read_text()
        assert not (build / "var").exists()

    def test_apply_renders_physics_into_the_repo_root_dotenv(self, runner, lifecycle_repo):
        """``.env`` is the SECRETS zone at the repo root, not a file in the render."""
        build = stub_build(lifecycle_repo, config=SIM_CONFIG)

        result = run_sim(runner, ["apply", "nominal", "--no-seed"], nested(lifecycle_repo))

        assert result.exit_code == 0, result.output
        assert not (build / ".env").exists()
