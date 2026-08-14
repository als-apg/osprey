"""The closing summary card, and which verbs print one.

Three things are asserted here and kept apart on purpose: what the card SAYS
(rendered from a repo and a state), that it says it through the phase reporter,
and WHICH verb prints one — a detached start, never an attached one, and exactly
one for a whole ``init --up`` chain rather than one per verb in it.

Card lines are read off a recording reporter or off the renderer directly,
never out of ``result.output``: reporter output goes through the Rich console
singleton, which wraps to whatever width a CliRunner presents, so an assertion
on captured stdout is an assertion about wrapping — and the card carries a
filesystem path, which is exactly what wrapping breaks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli import summary_card
from osprey.cli.main import cli
from osprey.cli.phase_reporter import (
    NullReporter,
    PhaseReporter,
    current_reporter,
    install_reporter,
)
from osprey.cli.summary_card import format_summary_card, owns_summary_card, print_summary_card


class RecordingReporter(PhaseReporter):
    """A real reporter that keeps its lines instead of printing them."""

    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        self.lines.append(text)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def recorder():
    """Install a recording reporter for the duration of one test."""
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
    """A deployment repo whose build declares one web-terminal endpoint."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text(
        "project_name: demo\n"
        "deployed_services: []\n"
        "modules:\n"
        "  web_terminals:\n"
        "    enabled: true\n"
        "    nginx_port: 8080\n"
    )
    return tmp_path


@pytest.fixture
def cards(monkeypatch) -> list[tuple[Path, str]]:
    """Record every card a verb asks for, without rendering any.

    The verbs import the function inside their own body, so patching the
    module attribute catches the call however the verb was reached.
    """
    printed: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        summary_card, "print_summary_card", lambda root, state: printed.append((Path(root), state))
    )
    return printed


# ---------------------------------------------------------------------------
# What the card says
# ---------------------------------------------------------------------------


class TestCardContent:
    def test_the_card_leads_with_the_deployment_name_and_its_state(self, repo: Path) -> None:
        """The repo's directory name IS the deployment's name — the same one the
        compose project and the container labels carry."""
        assert format_summary_card(repo, "running")[0] == f"{repo.name} — running"

    def test_a_running_card_shows_the_urls_the_render_declares(self, repo: Path) -> None:
        """Read out of the repo's own build/, so the card cannot name an endpoint
        that this deployment does not publish."""
        card = "\n".join(format_summary_card(repo, "running"))

        assert "web terminal" in card
        assert "http://127.0.0.1:8080" in card

    def test_only_addresses_that_can_be_opened_reach_the_card(
        self, repo: Path, monkeypatch
    ) -> None:
        """A card is the "what now" surface: the addresses on it are the ones
        somebody can act on. Bare host:port services, and the explicit
        "not configured" facts, stay on `osprey status`, which lists them all."""
        from osprey.deployment import deploy_summary

        monkeypatch.setattr(
            deploy_summary,
            "as_built_endpoint_entries",
            lambda root: [
                ("event-dispatcher", "http://127.0.0.1:8001"),
                ("postgres", "127.0.0.1:5432"),
                ("web terminal", "(not configured in this project)"),
            ],
        )

        card = "\n".join(format_summary_card(repo, "running"))

        assert "http://127.0.0.1:8001" in card
        assert "postgres" not in card
        assert "not configured" not in card

    @pytest.mark.parametrize("state", ["created", "built", "stopped"])
    def test_a_deployment_that_is_not_running_advertises_no_endpoints(
        self, repo: Path, state: str
    ) -> None:
        """Nothing answers on them, and a URL printed under `down` reads as an
        invitation to open something that is not there."""
        assert "http://" not in "\n".join(format_summary_card(repo, state))

    def test_the_card_says_where_the_command_output_went(self, repo: Path) -> None:
        """The spool directory is the one thing a quiet run cannot be read back
        from anywhere else."""
        card = "\n".join(format_summary_card(repo, "running"))

        assert str(repo / "var" / "logs") in card

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("created", "osprey build"),
            ("built", "osprey up -d"),
            ("running", "osprey down"),
            ("stopped", "osprey up -d"),
        ],
    )
    def test_every_state_ends_with_the_command_that_follows_it(
        self, repo: Path, state: str, expected: str
    ) -> None:
        assert expected in format_summary_card(repo, state)[-1]

    def test_the_endpoints_come_from_the_endpoint_summary_module(
        self, repo: Path, monkeypatch
    ) -> None:
        """One derivation of "what answers where", not two. The card asks
        deploy_summary for this repo's entries rather than reading ports out of
        the compose files a second way — two readers would eventually disagree
        about the same deployment."""
        from osprey.deployment import deploy_summary

        asked: list[Path] = []

        def record(root):
            asked.append(Path(root))
            return []

        monkeypatch.setattr(deploy_summary, "as_built_endpoint_entries", record)

        format_summary_card(repo, "running")

        assert asked == [repo]


# ---------------------------------------------------------------------------
# How it is printed
# ---------------------------------------------------------------------------


class TestCardPrinting:
    def test_the_card_prints_through_the_installed_reporter(
        self, repo: Path, recorder: RecordingReporter
    ) -> None:
        """Same path as the phase lines, so the card is plain, unwrapped, and
        colored only when stdout is a terminal."""
        print_summary_card(repo, "running")

        assert recorder.lines[0] == ""
        assert recorder.lines[1] == f"{repo.name} — running"
        assert any("http://127.0.0.1:8080" in line for line in recorder.lines)

    def test_a_card_that_cannot_be_rendered_does_not_fail_the_verb(
        self, repo: Path, recorder: RecordingReporter, monkeypatch
    ) -> None:
        """The work succeeded. An unreadable render is not a reason to end a
        successful deploy with an error."""
        from osprey.deployment import deploy_summary

        def boom(root):
            raise RuntimeError("build/ moved under us")

        monkeypatch.setattr(deploy_summary, "as_built_endpoint_entries", boom)

        print_summary_card(repo, "running")

        assert recorder.lines == []

    def test_a_repo_with_nothing_rendered_still_gets_a_card(self, tmp_path: Path) -> None:
        """`osprey init` prints one before anything has been built."""
        card = format_summary_card(tmp_path, "created")

        assert card[0] == f"{tmp_path.name} — created"


class TestCardOwnership:
    """Who prints the card is decided by the same rule as who owns the reporter."""

    def test_a_verb_that_finds_the_quiet_default_owns_the_card(self, restore_reporter) -> None:
        install_reporter(NullReporter())

        assert owns_summary_card() is True

    def test_a_verb_chained_under_another_one_does_not(self, recorder) -> None:
        assert owns_summary_card() is False

    def test_a_verb_chained_under_a_verbose_run_does_not_either(self, restore_reporter) -> None:
        """`--verbose` installs a NullReporter that is not the quiet default:
        the flag travels with it, and the outer verb still owns the run."""
        install_reporter(NullReporter(verbose=True))

        assert owns_summary_card() is False


# ---------------------------------------------------------------------------
# Which verbs print one
# ---------------------------------------------------------------------------


@pytest.fixture
def deploy_stubs(monkeypatch, repo: Path):
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
    monkeypatch.setattr(container_lifecycle, "up_as_built", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "restart_deployment", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "down_deployment", lambda *a, **k: None)


class TestVerbsThatPrintOne:
    @pytest.mark.parametrize("verb", ["up", "restart"])
    def test_a_detached_start_ends_with_a_running_card(
        self, runner, deploy_stubs, restore_reporter, cards, repo: Path, verb: str
    ) -> None:
        result = runner.invoke(cli, [verb, "-d"])

        assert result.exit_code == 0, result.output
        assert cards == [(repo, "running")]

    @pytest.mark.parametrize("verb", ["up", "restart"])
    def test_an_attached_start_prints_no_card(
        self, runner, deploy_stubs, restore_reporter, cards, verb: str
    ) -> None:
        """Attached, compose owns the terminal from the exec point on: `up` never
        comes back to print a card, and a `restart` that ends in the log stream
        has nothing to append it to."""
        result = runner.invoke(cli, [verb])

        assert result.exit_code == 0, result.output
        assert cards == []

    def test_down_ends_with_a_stopped_card(
        self, runner, deploy_stubs, restore_reporter, cards, repo: Path
    ) -> None:
        result = runner.invoke(cli, ["down"])

        assert result.exit_code == 0, result.output
        assert cards == [(repo, "stopped")]

    def test_a_failed_start_prints_no_card(
        self, runner, deploy_stubs, restore_reporter, cards, monkeypatch
    ) -> None:
        """The card states what is now true. Nothing is running, so nothing on
        it would be."""
        from osprey.deployment import container_lifecycle

        def boom(*args, **kwargs):
            raise RuntimeError("compose said no")

        monkeypatch.setattr(container_lifecycle, "up_as_built", boom)

        result = runner.invoke(cli, ["up", "-d"])

        assert result.exit_code != 0
        assert cards == []

    def test_a_start_chained_under_another_verb_leaves_the_card_to_it(
        self, runner, deploy_stubs, recorder, cards
    ) -> None:
        """The mechanism `init --up` rests on: a verb that finds a reporter
        already installed reports into it and prints no card of its own."""
        result = runner.invoke(cli, ["up", "-d"])

        assert result.exit_code == 0, result.output
        assert cards == []


# ---------------------------------------------------------------------------
# The chain, and the one verb that gets a line instead
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_stubs(monkeypatch):
    """Stub the render and the start, leaving the chain itself real."""
    from osprey.cli import build_cmd, deploy_cmd
    from osprey.deployment import container_lifecycle

    monkeypatch.setattr(build_cmd, "_build_repo", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "_preflight", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "up_as_built", lambda *a, **k: None)


class TestTheInitChain:
    def test_a_fresh_repo_ends_with_a_created_card(
        self, runner, restore_reporter, cards, tmp_path: Path
    ) -> None:
        target = tmp_path / "demo"

        result = runner.invoke(
            cli, ["init", str(target), "--preset", "hello-world"], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert cards == [(target, "created")]

    def test_init_up_prints_exactly_one_card_for_the_whole_chain(
        self, runner, restore_reporter, chain_stubs, cards, tmp_path: Path
    ) -> None:
        """`init --up -d` runs three verbs — create, build, start — as one run.
        Three cards for one run is the duplication this replaces, so the outer
        verb prints the only one, and it describes the end state."""
        target = tmp_path / "chained"

        result = runner.invoke(
            cli,
            ["init", str(target), "--preset", "hello-world", "--up", "-d"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert cards == [(target, "running")]

    def test_build_on_its_own_ends_with_a_built_card(
        self, runner, restore_reporter, chain_stubs, cards, repo: Path, monkeypatch
    ) -> None:
        from osprey.cli import build_cmd

        monkeypatch.setattr(build_cmd, "find_repo_root", lambda start=None: repo)

        result = runner.invoke(cli, ["build"])

        assert result.exit_code == 0, result.output
        assert cards == [(repo, "built")]


class TestResetGetsALineNotACard:
    def test_reset_says_it_is_finished_through_its_own_emit_callback(
        self, recorder: RecordingReporter, monkeypatch, tmp_path: Path
    ) -> None:
        """`reset` prints a plan, asks, and destroys; its ending belongs to that
        transcript, in the same voice and through the same callback, not on a
        card describing a deployment that no longer exists."""
        from osprey.deployment import reset as reset_module

        class FakePlan:
            project = "demo"
            repo_root = tmp_path
            removes_anything = True

            def render(self):
                return ["Reset plan for demo:"]

            def confirmation_summary(self):
                return "1 container"

        class FakeProbe:
            failures: list[tuple[str, str]] = []

        monkeypatch.setattr(reset_module, "plan_reset", lambda *a, **k: FakePlan())
        monkeypatch.setattr(reset_module, "execute_reset", lambda *a, **k: None)
        monkeypatch.setattr(reset_module, "confirm_destroy", lambda *a, **k: True)

        lines: list[str] = []
        reset_module.reset_deployment(
            tmp_path, assume_yes=True, emit=lines.append, probe=FakeProbe()
        )

        assert lines[-1] == "Reset complete. `osprey build && osprey up -d` starts demo again."
        assert lines.count("Reset complete. `osprey build && osprey up -d` starts demo again.") == 1
        assert recorder.lines == []
