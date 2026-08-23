"""The closing summary card, and which verbs print one.

Three things are asserted here and kept apart on purpose: what the card SAYS
(rendered from a repo and a state), that it says it through the CLI's renderer
and therefore under every reporter, and WHICH verb prints one — a detached
start, never an attached one, and exactly one for a whole ``init --up`` chain
rather than one per verb in it.

The card is echo class: it is the verb's own closing output, not decoration on
its progress, so it goes out through :mod:`osprey.cli.output` and survives the
``NullReporter`` that global ``--verbose`` installs. That is a change in what
``osprey -v up`` prints, and it is pinned here rather than left to be noticed.

Card lines are read off a recording console or off the plain-text renderer
directly, never out of ``result.output``: the renderer prints through whichever
console the installed reporter owns, and a CliRunner presents one of its own
width — so an assertion on captured stdout is an assertion about wrapping, and
the card carries a filesystem path, which is exactly what wrapping breaks.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from osprey.cli import output, phase_reporter, summary_card
from osprey.cli.main import cli
from osprey.cli.phase_reporter import (
    NullReporter,
    PhaseReporter,
    current_reporter,
    install_reporter,
)
from osprey.cli.styles import osprey_theme
from osprey.cli.summary_card import format_summary_card, owns_summary_card, print_summary_card


def recording_console(*, terminal: bool = False) -> tuple[Console, io.StringIO]:
    """A themed console writing into a buffer instead of at a terminal.

    The theme is not optional: the card's heading and labels are semantic
    tokens, and a bare ``Console()`` raises ``MissingStyle`` on the first one.
    Explicit color settings keep the terminal rendering independent of
    ``NO_COLOR`` and ``TERM``; the default non-terminal rendering stays plain
    for every line assertion here.
    """
    buffer = io.StringIO()
    return (
        Console(
            file=buffer,
            theme=osprey_theme,
            force_terminal=terminal,
            color_system="standard" if terminal else None,
            no_color=not terminal,
            width=200,
        ),
        buffer,
    )


class RecordingReporter(PhaseReporter):
    """A real reporter whose console is a buffer, not the terminal.

    Overrides ``out()``, not ``emit()``: ``out()`` is the seam the renderer
    resolves per call, so a fake that intercepted ``emit`` would see nothing the
    card prints any more.
    """

    def __init__(self, console: Console) -> None:
        super().__init__(color=False)
        self._console = console

    def out(self) -> Console:
        return self._console


class Recorded:
    """The lines written to a recording console so far."""

    def __init__(self, buffer: io.StringIO) -> None:
        self._buffer = buffer

    @property
    def lines(self) -> list[str]:
        return self._buffer.getvalue().splitlines()


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
    """Install a recording reporter for the duration of one test."""
    console, buffer = recording_console()
    previous = install_reporter(RecordingReporter(console))
    try:
        yield Recorded(buffer)
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


@pytest.fixture(autouse=True)
def empty_ledger():
    """Start and leave every test with no collected rows.

    The ledger is process-scoped and the card flushes it before printing, so a
    row some earlier test left behind would print above the card and shift
    every line index asserted here.
    """
    output.clear_ledger()
    yield
    output.clear_ledger()


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


@pytest.fixture
def repo_with_token_logins(repo: Path) -> Path:
    """A multi-user deployment with NO login wall — the default posture.

    With `auth.method` at `none` nginx runs no login flow and injects no
    operator secret, so each terminal's own token->cookie gate is the only one
    and the only way through it is that user's `?token=` URL.
    """
    (repo / "build" / "config.yml").write_text(
        "project_name: demo\n"
        "deployed_services: []\n"
        "modules:\n"
        "  web_terminals:\n"
        "    enabled: true\n"
        "    nginx_port: 8080\n"
        "    users:\n"
        "      - name: alice\n"
        "        index: 0\n"
        "      - name: bob\n"
        "        index: 1\n"
    )
    return repo


@pytest.fixture
def repo_with_fqdn(repo: Path) -> Path:
    """The same repo, declaring the host browsers actually reach it on.

    Every real multi-user deployment has this: the render refuses to produce one
    with a roster and no origin. The bare `repo` fixture is the degenerate case
    (module on, roster empty), which is why the two are separate.
    """
    config = repo / "build" / "config.yml"
    config.write_text(config.read_text() + "deploy:\n  host: dls-deploy\n  fqdn: dls.example.org\n")
    return repo


@pytest.fixture
def repo_with_logins(repo: Path) -> Path:
    """The same repo, with a password-authenticated roster whose two logins the
    profile declared and nobody has since edited."""
    build = repo / "build" / "config.yml"
    build.write_text(
        "project_name: demo\n"
        "deployed_services: []\n"
        "modules:\n"
        "  web_terminals:\n"
        "    enabled: true\n"
        "    nginx_port: 8080\n"
        "    auth:\n"
        "      method: password\n"
        "    users:\n"
        "      - name: alice\n"
        "        index: 0\n"
        "      - name: bob\n"
        "        index: 1\n"
    )
    (repo / "profile.yml").write_text(
        "env:\n  defaults:\n    OSPREY_AUTH_PW_ALICE: alice\n    OSPREY_AUTH_PW_BOB: bob\n"
    )
    (repo / ".env").write_text("OSPREY_AUTH_PW_ALICE=alice\nOSPREY_AUTH_PW_BOB=bob\n")
    return repo


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

    def test_a_build_that_rendered_scenarios_offers_the_demo_seed(self, repo: Path) -> None:
        """The seed hint has no line of its own any more, so the one next-step
        surface has to carry it -- and it is a next step: an operator who just
        built a project with demo scenarios can put data behind them."""
        (repo / "build" / "data" / "simulation" / "scenarios").mkdir(parents=True)

        card = "\n".join(format_summary_card(repo, "built"))

        assert "osprey sim apply nominal" in card

    def test_a_build_with_no_scenarios_is_not_told_to_apply_one(self, repo: Path) -> None:
        """A command with nothing to apply is worse advice than no advice."""
        assert "osprey sim apply" not in "\n".join(format_summary_card(repo, "built"))

    @pytest.mark.parametrize("state", ["created", "running", "stopped"])
    def test_the_seed_hint_belongs_to_the_built_card_alone(self, repo: Path, state: str) -> None:
        """`built` is the state the deleted hint was printed in. A running
        deployment's next step is reading it, not reseeding it."""
        (repo / "build" / "data" / "simulation" / "scenarios").mkdir(parents=True)

        assert "osprey sim apply" not in "\n".join(format_summary_card(repo, state))

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
    def test_the_card_prints_through_the_renderer(self, repo: Path, recorder: Recorded) -> None:
        """Out through `osprey.cli.output`, on the console the installed
        reporter owns — so the card lands above a mounted live region rather
        than inside it, unwrapped, and colored only on a terminal."""
        print_summary_card(repo, "running")

        assert recorder.lines[0] == ""
        assert recorder.lines[1] == f"{repo.name} — running"
        assert any("http://127.0.0.1:8080" in line for line in recorder.lines)

    def test_the_printed_card_is_the_one_the_text_renderer_renders(
        self, repo: Path, recorder: Recorded
    ) -> None:
        """The rows are laid out by `output.section` and by
        `format_summary_card` separately — one for the screen, one for a caller
        that wants the card as text. Pinned against each other so the two
        layouts cannot drift into two different cards."""
        print_summary_card(repo, "running")

        card = format_summary_card(repo, "running")
        assert recorder.lines[: len(card) + 1] == [""] + card

    def test_a_verbose_run_still_ends_with_the_card(
        self, repo: Path, restore_reporter, monkeypatch
    ) -> None:
        """The card is echo class: it says what the run left behind, which is
        the verb's own output rather than decoration on its progress. Under
        `--verbose` the phase record is swallowed and the card is not."""
        console, buffer = recording_console()
        monkeypatch.setattr(phase_reporter, "console", console)
        install_reporter(NullReporter(verbose=True))

        print_summary_card(repo, "running")

        card = format_summary_card(repo, "running")
        assert Recorded(buffer).lines[: len(card) + 1] == [""] + card

    def test_the_heading_is_styled_on_a_terminal(
        self, repo: Path, restore_reporter, monkeypatch
    ) -> None:
        """The card keeps its bold heading through the renderer. Asserted as
        "carries styling" rather than as an exact escape sequence: which bold
        the theme resolves to is the theme's business."""
        console, buffer = recording_console(terminal=True)
        install_reporter(RecordingReporter(console))

        print_summary_card(repo, "running")

        heading = next(line for line in buffer.getvalue().splitlines() if repo.name in line)
        assert "\x1b[" in heading

    def test_a_card_that_cannot_be_rendered_does_not_fail_the_verb(
        self, repo: Path, recorder: Recorded, monkeypatch
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

    def test_a_verbose_start_asks_for_a_card_too(
        self, runner, deploy_stubs, restore_reporter, cards, repo: Path
    ) -> None:
        """`--verbose` changes which reporter the run installs, not who owns the
        run: the outermost verb still finds the quiet default and still asks for
        the card. Printing it is `print_summary_card`'s half, pinned above."""
        result = runner.invoke(cli, ["-v", "up", "-d"])

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
        self, recorder: Recorded, monkeypatch, tmp_path: Path
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


# ---------------------------------------------------------------------------
# The line under the card
# ---------------------------------------------------------------------------


class TestClosingCallToAction:
    """A card is an inventory; this is the instruction under it.

    Six equally-prominent addresses leave the reader to work out which one is
    the front door, so the closing line names it and says what to sign in with.
    """

    def test_the_closing_line_names_the_landing_page(self, repo: Path, recorder: Recorded) -> None:
        print_summary_card(repo, "running")

        assert "Everything is running. Open http://127.0.0.1:8080 to start." in recorder.lines, (
            recorder.lines
        )

    def test_a_loopback_only_address_says_so(self, repo: Path, recorder: Recorded) -> None:
        """This repo declares no `deploy.fqdn` and no `external_origin`, so the
        only address there is to print is the deploy host's own loopback — where
        the landing page answers and no terminal will accept a write, because
        each checks a mutating request's Origin against a value this deployment
        never declared. Said out loud rather than left to be met as a 403 inside
        a container.
        """
        print_summary_card(repo, "running")

        printed = "\n".join(recorder.lines)
        assert "serves the landing page only" in printed
        assert "external_origin" in printed

    def test_the_closing_line_names_the_origin_the_terminals_check_against(
        self, repo_with_fqdn: Path, recorder: Recorded
    ) -> None:
        """`deploy.fqdn` is set, so that IS the address browsers reach — and the
        one every terminal compares a mutating request's Origin against. Sending
        an operator to 127.0.0.1 instead would produce a deployment whose pages
        all load and whose every write is refused."""
        print_summary_card(repo_with_fqdn, "running")

        printed = "\n".join(recorder.lines)
        assert "Everything is running. Open http://dls.example.org:8080 to start." in printed
        assert "127.0.0.1:8080 to start" not in printed
        assert "serves the landing page only" not in printed

    def test_an_explicit_external_origin_wins_over_the_derived_one(
        self, repo_with_fqdn: Path, recorder: Recorded
    ) -> None:
        """A load balancer terminating TLS in front of this nginx is what the
        browser actually reaches, so its address is the origin — and the address
        to open. Nothing derivable from `deploy.fqdn` can name it."""
        config = repo_with_fqdn / "build" / "config.yml"
        config.write_text(
            config.read_text().replace(
                "    nginx_port: 8080\n",
                "    nginx_port: 8080\n    external_origin: https://terminals.example.org\n",
            )
        )

        print_summary_card(repo_with_fqdn, "running")

        printed = "\n".join(recorder.lines)
        assert "Everything is running. Open https://terminals.example.org to start." in printed
        assert "dls.example.org" not in printed

    def test_the_closing_line_comes_after_the_card(self, repo: Path, recorder: Recorded) -> None:
        """Order is the point: what was written, then where it answers, then
        what to do about it."""
        print_summary_card(repo, "running")

        card_line = recorder.lines.index(f"{repo.name} — running")
        closing = next(i for i, line in enumerate(recorder.lines) if line.startswith("Everything"))
        assert card_line < closing

    def test_the_ledger_comes_before_the_card(self, repo: Path, recorder: Recorded) -> None:
        """The receipt belongs with the run that wrote it, so the card and the
        line under it close the run together."""
        from osprey.cli import output

        class _Logger:
            def key_info(self, message: str) -> None: ...

        output.report_fact(_Logger(), "minted a token", wrote=(".env", "A_TOKEN"))
        try:
            print_summary_card(repo, "running")
        finally:
            output.clear_ledger()

        assert recorder.lines.index("This deploy wrote") < recorder.lines.index(
            f"{repo.name} — running"
        )

    def test_seeded_logins_are_shown_with_the_landing_page(
        self, repo_with_logins: Path, recorder: Recorded
    ) -> None:
        """`osprey init` wrote these from the profile, so nobody was ever told
        them."""
        print_summary_card(repo_with_logins, "running")

        assert any("alice / alice · bob / bob" in line for line in recorder.lines)

    def test_a_password_the_operator_changed_is_not_shown(
        self, repo_with_logins: Path, recorder: Recorded
    ) -> None:
        (repo_with_logins / ".env").write_text(
            "OSPREY_AUTH_PW_ALICE=chosen-by-the-operator\nOSPREY_AUTH_PW_BOB=bob\n"
        )

        print_summary_card(repo_with_logins, "running")

        printed = "\n".join(recorder.lines)
        assert "chosen-by-the-operator" not in printed
        assert "bob / bob" in printed

    def test_no_closing_line_for_a_deployment_that_is_not_running(
        self, repo: Path, recorder: Recorded
    ) -> None:
        print_summary_card(repo, "built")

        assert not any(line.startswith("Everything is running") for line in recorder.lines)

    def test_no_closing_line_without_a_landing_page(
        self, tmp_path: Path, recorder: Recorded
    ) -> None:
        """A backend-only project has no front door to send anyone to; the
        card's `next` row is already the right answer for it."""
        build = tmp_path / "build"
        build.mkdir()
        (build / "config.yml").write_text(
            "project_name: demo\ndeployed_services: []\nmodules:\n  web_terminals:\n"
            "    enabled: false\n"
        )

        print_summary_card(tmp_path, "running")

        assert not any(line.startswith("Everything is running") for line in recorder.lines)


class TestTokenLoginClosingLine:
    """A terminal with no login page needs the verb that opens it.

    "Open the landing page" is the whole instruction for a deployment behind a
    login wall. Without one, the landing page routes to a terminal that answers
    nothing — and the only way to a first session is that user's own `?token=`
    URL, which nothing else in the multi-user shape hands out.
    """

    def test_an_auth_off_roster_is_told_the_verb(
        self, repo_with_token_logins: Path, recorder: Recorded
    ) -> None:
        print_summary_card(repo_with_token_logins, "running")

        printed = "\n".join(recorder.lines)
        assert "osprey users login-url alice" in printed
        assert "alice, bob" in printed

    def test_the_urls_themselves_are_never_printed(
        self, repo_with_token_logins: Path, recorder: Recorded
    ) -> None:
        """Each carries a live operator secret, and this output is read over
        shoulders, pasted into tickets and captured whole by CI logs."""
        (repo_with_token_logins / ".env").write_text(
            "OSPREY_TERMINAL_SECRET_ALICE=alice-operator-secret\n"
        )

        print_summary_card(repo_with_token_logins, "running")

        printed = "\n".join(recorder.lines)
        assert "alice-operator-secret" not in printed
        assert "?token=" not in printed

    def test_a_walled_roster_gets_no_such_line(
        self, repo_with_logins: Path, recorder: Recorded
    ) -> None:
        """Every user there logs in through the sidecar, so the login page is
        the front door and naming a second one would be noise."""
        print_summary_card(repo_with_logins, "running")

        assert "login-url" not in "\n".join(recorder.lines)

    def test_a_login_exempt_entry_is_named_even_with_auth_on(
        self, repo_with_logins: Path, recorder: Recorded
    ) -> None:
        """`login: false` puts one entry outside the wall, and that terminal is
        reached exactly the way an auth-off one is."""
        config = repo_with_logins / "build" / "config.yml"
        config.write_text(
            config.read_text() + "      - name: kiosk\n        index: 2\n        login: false\n"
        )

        print_summary_card(repo_with_logins, "running")

        printed = "\n".join(recorder.lines)
        assert "osprey users login-url kiosk" in printed
        # ...and only that entry: alice and bob still have a login page.
        assert "alice" not in printed.split("no login page")[1]
