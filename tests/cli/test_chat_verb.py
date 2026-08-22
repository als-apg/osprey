"""The ``osprey chat`` verb: what it launches, where, and what it refuses.

Every test here stops at the handoff boundary. The real command ends in
``subprocess.run([...agent CLI...])`` and never returns; the fixture below
replaces that call with a recorder, so what gets asserted is the three things
the handoff carries — the argv, the working directory the agent CLI will treat
as its project root, and the environment it inherits — without an agent process
ever starting.

The deployment under test is the exemplar repo with a *stubbed* build: the
fixture writes ``build/config.yml`` and a manifest whose stamped fingerprint the
test chooses. That is what makes drift a one-line parameter here — a real
``osprey build`` is a different task's subject, and chat cannot tell the
difference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from osprey.cli.chat_cmd import chat
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess

#: The richest provider env block — a base URL, a tier model per tier, and a
#: secret variable that is not ANTHROPIC_API_KEY. The injection tests need all
#: three to have anything to assert. The default (anthropic) rendered config is
#: :data:`tests.cli._lifecycle_build.STUB_CONFIG`.
CBORG_CONFIG = "claude_code:\n  provider: cborg\n"


@dataclass
class Launch:
    """One recorded handoff to the agent CLI."""

    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Undo what an in-process launch does to the interpreter's globals.

    ``chat`` chdirs into ``build/`` and overlays the deployment's ``.env`` onto
    ``os.environ`` — both deliberate, both process-global, and both able to
    reach every test that runs after this one in the same worker.
    """
    cwd = Path.cwd()
    environ = dict(os.environ)
    yield
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def _no_managed_policy(monkeypatch: pytest.MonkeyPatch):
    """Pin the managed-policy scan to "no conflicts".

    It reads OS-standard policy files, so on a machine that has one, every
    launch test would refuse for a reason none of them is about.
    """
    monkeypatch.setattr(
        "osprey.build.claude_code_resolver.detect_managed_policy_conflicts",
        lambda paths=None: {},
    )


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch):
    """Record handoffs instead of starting an agent. Yields the recorded list."""
    recorded: list[Launch] = []

    def fake_run(argv, *args, **kwargs):
        recorded.append(Launch(argv=list(argv), cwd=Path.cwd(), env=dict(os.environ)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])
    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        yield recorded


class TestChatSurface:
    """The flags, and the registration that makes them reachable."""

    def test_help_documents_every_flag(self, runner):
        result = runner.invoke(chat, ["--help"])

        assert result.exit_code == 0
        for flag in ("--repo", "--resume", "--print", "--effort", "--no-pin"):
            assert flag in result.output

    def test_registered_on_the_top_level_cli(self):
        """``osprey chat`` resolves through the lazy command registry."""
        import click

        from osprey.cli.main import cli

        loaded = cli.get_command(click.Context(cli), "chat")

        assert loaded is not None
        assert loaded.name == "chat"
        assert "chat" in cli.list_commands(click.Context(cli))


class TestNoBuild:
    """The one thing chat refuses: having nothing rendered to talk to."""

    def test_refuses_when_build_is_absent(self, runner, launches, lifecycle_repo):
        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 1
        assert "no build found" in result.output
        assert "osprey build" in result.output
        assert launches == []

    def test_refuses_when_the_build_has_no_rendered_config(self, runner, launches, lifecycle_repo):
        """A manifest with no config.yml beside it is a build in name only."""
        stub_build(lifecycle_repo, with_config=False)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 1
        assert "no build found" in result.output
        assert launches == []

    def test_refuses_outside_a_deployment_repo(self, runner, launches, tmp_path):
        elsewhere = tmp_path / "not-a-repo"
        elsewhere.mkdir()

        result = runner.invoke(chat, ["--repo", str(elsewhere)])

        assert result.exit_code == 1
        assert "profile.yml" in result.output
        assert launches == []


class TestUnreadableConfig:
    """A render whose config will not parse is refused, not crashed on."""

    def test_corrupt_config_refuses_without_a_traceback(self, runner, launches, lifecycle_repo):
        stub_build(lifecycle_repo, config="claude_code:\n  provider: [unclosed\n")

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 1
        assert "config.yml" in result.output
        assert "osprey build" in result.output
        assert "Traceback" not in result.output
        assert launches == []


class TestLaunch:
    """What the agent CLI is handed once there is a build."""

    def test_launches_in_the_build_directory(self, runner, launches, lifecycle_repo):
        build = stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert len(launches) == 1
        # build/ IS the rendered project, and the agent CLI reads its working
        # directory as the project root.
        assert launches[0].cwd == build.resolve()
        assert launches[0].argv == ["claude", "--setting-sources", "project"]

    def test_exit_code_is_the_agent_process_exit_code(self, runner, monkeypatch, lifecycle_repo):
        stub_build(lifecycle_repo)
        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])

        with patch_subprocess("osprey.cli.chat_cmd", return_value=SimpleNamespace(returncode=3)):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 3

    def test_resolves_the_repo_by_walking_up_from_the_working_directory(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """No --repo: chat acts on the repo enclosing wherever the operator is."""
        build = stub_build(lifecycle_repo)
        monkeypatch.chdir(lifecycle_repo / "data")

        result = runner.invoke(chat, [])

        assert result.exit_code == 0
        assert launches[0].cwd == build.resolve()

    def test_does_not_re_render_the_build(self, runner, launches, lifecycle_repo, monkeypatch):
        """`osprey build` owns rendering — chat launches what is already there."""
        from osprey.cli.templates.manager import TemplateManager

        def explode(*args, **kwargs):
            raise AssertionError("chat must not regenerate agent artifacts")

        monkeypatch.setattr(TemplateManager, "regenerate_claude_code", explode)
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert len(launches) == 1


class TestFlags:
    """Flags reach the agent CLI unchanged; the pin comes from the build."""

    def test_resume_print_and_effort_are_forwarded(self, runner, launches, lifecycle_repo):
        stub_build(lifecycle_repo)

        result = runner.invoke(
            chat,
            ["--repo", str(lifecycle_repo), "--resume", "abc123", "--print", "--effort", "high"],
        )

        assert result.exit_code == 0
        argv = launches[0].argv
        assert argv[:3] == ["claude", "--setting-sources", "project"]
        assert argv[3:] == ["--resume", "abc123", "--print", "--effort", "high"]

    def test_prompt_is_passed_as_the_opening_message(self, runner, launches, lifecycle_repo):
        stub_build(lifecycle_repo)

        result = runner.invoke(
            chat, ["--repo", str(lifecycle_repo), "--print", "what is the beam current?"]
        )

        assert result.exit_code == 0
        assert launches[0].argv[-2:] == ["--print", "what is the beam current?"]

    def test_an_unquoted_prompt_reaches_the_agent_as_one_message(
        self, runner, launches, lifecycle_repo
    ):
        """`osprey chat --print what is the beam current` — no quotes, one message.

        The agent CLI reads a single trailing argument as the opening message,
        so forwarding the shell's word split would send it only "what".
        """
        stub_build(lifecycle_repo)

        result = runner.invoke(
            chat,
            ["--repo", str(lifecycle_repo), "--print", "what", "is", "the", "beam", "current"],
        )

        assert result.exit_code == 0
        assert launches[0].argv[-2:] == ["--print", "what is the beam current"]

    def test_a_quoted_prompt_produces_the_same_single_argument(
        self, runner, launches, lifecycle_repo
    ):
        """Quoted or not, the agent CLI is handed one identical positional."""
        stub_build(lifecycle_repo)

        result = runner.invoke(
            chat, ["--repo", str(lifecycle_repo), "--print", "what is the beam current"]
        )

        assert result.exit_code == 0
        assert launches[0].argv[-2:] == ["--print", "what is the beam current"]

    def test_effort_defaults_to_the_builds_configured_value(self, runner, launches, lifecycle_repo):
        stub_build(
            lifecycle_repo,
            config="claude_code:\n  provider: anthropic\n  effort: medium\n",
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].argv[-2:] == ["--effort", "medium"]

    def test_cli_version_pin_is_honored(self, runner, launches, lifecycle_repo):
        stub_build(
            lifecycle_repo,
            config='claude_code:\n  provider: anthropic\n  cli_version: "2.1.146"\n',
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].argv[:3] == ["npx", "-y", "@anthropic-ai/claude-code@2.1.146"]

    def test_no_pin_drops_the_pin_but_keeps_provider_isolation(
        self, runner, launches, lifecycle_repo
    ):
        stub_build(
            lifecycle_repo,
            config='claude_code:\n  provider: anthropic\n  cli_version: "2.1.146"\n',
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo), "--no-pin"])

        assert result.exit_code == 0
        assert launches[0].argv == ["claude", "--setting-sources", "project"]


#: The opening of the DRIFT verdict's own message, which ``chat`` prints
#: verbatim. Held in one place because two of the tests below assert its
#: ABSENCE: a fragment that drifted out of the product would make those pass
#: against a string that exists nowhere, which is a test that can no longer
#: fail. Kept short so the warning's rich-wrapping cannot split it across lines.
DRIFT_HEADLINE = "profile.yml or a file it points at"


class TestDriftWarning:
    """Drift is reported and then launched through — chat never refuses on it."""

    def test_clean_build_says_nothing_about_drift(self, runner, launches, lifecycle_repo):
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert DRIFT_HEADLINE not in result.output
        assert "cannot verify" not in result.output
        assert "rendered by osprey" not in result.output

    def test_edited_profile_warns_and_still_launches(self, runner, launches, lifecycle_repo):
        stub_build(lifecycle_repo)
        profile = lifecycle_repo / "profile.yml"
        profile.write_text(
            profile.read_text(encoding="utf-8").replace("model: haiku", "model: sonnet"),
            encoding="utf-8",
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert DRIFT_HEADLINE in result.output
        assert len(launches) == 1

    def test_unverifiable_build_warns_and_still_launches(self, runner, launches, lifecycle_repo):
        """No stamped fingerprint: a start verb refuses here, chat does not."""
        stub_build(lifecycle_repo, stamped_hash=None)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert "cannot verify profile ↔ build consistency" in result.output
        assert len(launches) == 1

    def test_version_skew_warns_independently_of_the_verdict(
        self, runner, launches, lifecycle_repo
    ):
        """A build can match the profile exactly and still predate the framework."""
        stub_build(lifecycle_repo, version="2000.1.0")

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert "rendered by osprey 2000.1.0" in result.output
        assert DRIFT_HEADLINE not in result.output
        assert len(launches) == 1


class TestProviderEnvironment:
    """The launch environment: the repo's .env, then the provider's own block."""

    def test_repo_env_is_overlaid_in_full(self, runner, launches, lifecycle_repo, monkeypatch):
        """Every key, not a provider subset — .mcp.json expands ${VAR} from here."""
        monkeypatch.delenv("EPICS_CA_ADDR_LIST", raising=False)
        stub_build(lifecycle_repo)
        (lifecycle_repo / ".env").write_text(
            "ANTHROPIC_API_KEY=sk-ant-from-dotenv\nEPICS_CA_ADDR_LIST=10.0.0.1 10.0.0.2\n",
            encoding="utf-8",
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        # The SECRETS zone is the repo root, not the disposable build output.
        assert launches[0].env["EPICS_CA_ADDR_LIST"] == "10.0.0.1 10.0.0.2"
        assert launches[0].env["ANTHROPIC_API_KEY"] == "sk-ant-from-dotenv"

    def test_repo_env_beats_a_stale_shell_export(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stale-shell-value")
        stub_build(lifecycle_repo)
        (lifecycle_repo / ".env").write_text(
            "ANTHROPIC_API_KEY=sk-ant-from-dotenv\n", encoding="utf-8"
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].env["ANTHROPIC_API_KEY"] == "sk-ant-from-dotenv"

    def test_dotenv_cannot_select_a_backend_when_no_provider_is_configured(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """Backend isolation is not the provider path's privilege.

        A build with no ``claude_code`` block resolves no spec, so the injection
        that normally scrubs the managed vars never runs. The overlay must still
        not let a ``.env`` key choose which backend the agent talks to, while
        the keys ``.mcp.json`` expands from go through untouched.
        """
        monkeypatch.delenv("EPICS_CA_ADDR_LIST", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        stub_build(lifecycle_repo, config="control_system:\n  type: mock\n")
        (lifecycle_repo / ".env").write_text(
            "ANTHROPIC_BASE_URL=https://elsewhere.example.org\n"
            "EPICS_CA_ADDR_LIST=10.0.0.1 10.0.0.2\n",
            encoding="utf-8",
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert "ANTHROPIC_BASE_URL" not in launches[0].env
        assert launches[0].env["EPICS_CA_ADDR_LIST"] == "10.0.0.1 10.0.0.2"

    def test_a_shell_export_still_authenticates_a_provider_less_build(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """What the overlay never touched is not the overlay's to take away.

        A deployment that configures no provider authenticates from the
        operator's shell, so the managed vars are reverted by provenance rather
        than cleared — clearing them would trade a silent redirect for a silent
        auth failure.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-shell")
        stub_build(lifecycle_repo, config="control_system:\n  type: mock\n")
        (lifecycle_repo / ".env").write_text("EPICS_CA_ADDR_LIST=10.0.0.1\n", encoding="utf-8")

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].env["ANTHROPIC_API_KEY"] == "sk-ant-from-the-shell"

    def test_a_dotenv_managed_var_reverts_to_the_shell_value(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """Both provenances at once: the overlay's value goes, the shell's stays."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://shell.example.org")
        stub_build(lifecycle_repo, config="control_system:\n  type: mock\n")
        (lifecycle_repo / ".env").write_text(
            "ANTHROPIC_BASE_URL=https://dotenv.example.org\n", encoding="utf-8"
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].env["ANTHROPIC_BASE_URL"] == "https://shell.example.org"

    def test_missing_auth_secret_is_a_warning_not_a_refusal(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert "ANTHROPIC_API_KEY" in result.output
        assert len(launches) == 1

    def test_the_auth_token_is_set_from_the_providers_secret_variable(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """The agent authenticates with a token derived from the deployment's own
        provider secret, never with whatever the shell happened to export."""
        stub_build(lifecycle_repo, config=CBORG_CONFIG)
        monkeypatch.setenv("CBORG_API_KEY", "test-secret-123")

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].env["ANTHROPIC_AUTH_TOKEN"] == "test-secret-123"

    def test_a_stale_managed_var_is_scrubbed_and_re_injected_from_the_spec(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """A managed variable exported in the shell must not survive into the
        agent: it is scrubbed and replaced by the resolved provider's own value,
        or dropped when that provider declares none."""
        stub_build(lifecycle_repo, config=CBORG_CONFIG)
        monkeypatch.setenv("CBORG_API_KEY", "test-secret-123")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://stale-shell-value.example.com")
        monkeypatch.setenv("ANTHROPIC_MODEL", "stale-model")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-scrubbed")

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        env = launches[0].env
        assert env["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"
        assert "ANTHROPIC_MODEL" in env
        # Not in cborg's env block, so the scrub is the last word on it.
        assert "ANTHROPIC_API_KEY" not in env

    def test_every_tier_model_variable_reaches_the_agent(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """The whole env block, not just the auth half — a missing tier variable
        would let the agent fall back to a model this deployment never chose."""
        stub_build(lifecycle_repo, config=CBORG_CONFIG)
        monkeypatch.setenv("CBORG_API_KEY", "test-secret-123")

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        env = launches[0].env
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" in env
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in env
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" in env

    def test_managed_policy_conflict_refuses_to_launch(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """Policy env outranks provider isolation, so the wrong backend is possible."""
        stub_build(lifecycle_repo)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.detect_managed_policy_conflicts",
            lambda paths=None: {
                "ANTHROPIC_BASE_URL": ("https://elsewhere.example.org", "/etc/policy.json")
            },
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 1
        assert "Refusing to launch" in result.output
        assert launches == []


def _telemetry_config(password: str, *, user: str = "ingest@example.com") -> str:
    """A rendered config whose telemetry block names ``password`` as its secret.

    Shaped like the block every bundled preset ships: an OpenObserve backend
    with no explicit endpoint (it derives one) and a user that already has a
    value, so the password is the only credential a test is varying.
    """
    return (
        "claude_code:\n"
        "  provider: anthropic\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: openobserve\n"
        "    openobserve:\n"
        f"      user: {user}\n"
        f"      password: {password}\n"
        "      org: default\n"
    )


class TestTelemetryCredentialNotIssuedYet:
    """Chat on a deployment that has never run ``osprey up``.

    The shipped telemetry block names ``${ZO_INGEST_SA_TOKEN}`` with no
    fallback, and that token does not exist until a deploy provisions it into
    the repo's ``.env``. Resolving the provider resolves telemetry too, so
    without a carve-out the very first ``osprey chat`` on a fresh scaffold
    cannot start at all — for a value the operator has no way to supply.

    The carve-out is exactly as wide as the store-issued registry and no wider:
    every other unresolved credential is an ordinary missing secret and keeps
    refusing.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_store_credentials(self, monkeypatch: pytest.MonkeyPatch):
        """A developer machine that happens to export the token would hide the bug."""
        monkeypatch.delenv("ZO_INGEST_SA_TOKEN", raising=False)
        monkeypatch.delenv("OPERATOR_OTLP_SECRET", raising=False)

    def test_the_token_under_test_is_really_store_issued(self):
        """Guards every assertion below from passing for the wrong reason."""
        from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS

        assert "ZO_INGEST_SA_TOKEN" in _STORE_ISSUED_VARS
        assert "OPERATOR_OTLP_SECRET" not in _STORE_ISSUED_VARS

    def test_an_unissued_store_credential_starts_the_session_without_telemetry(
        self, runner, launches, lifecycle_repo
    ):
        stub_build(lifecycle_repo, config=_telemetry_config("${ZO_INGEST_SA_TOKEN}"))

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert len(launches) == 1
        # Degraded, not deferred: an exporter without its auth header would post
        # to an auth-gated store and drop every span, so the whole block goes.
        assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in launches[0].env
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in launches[0].env

    def test_the_operator_is_told_which_verb_issues_it(self, runner, launches, lifecycle_repo):
        """Silence would read as "this deployment has no telemetry configured"."""
        stub_build(lifecycle_repo, config=_telemetry_config("${ZO_INGEST_SA_TOKEN}"))

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert "ZO_INGEST_SA_TOKEN" in result.output
        assert "osprey up" in result.output
        assert "Traceback" not in result.output

    def test_an_operator_supplied_credential_still_refuses(self, runner, launches, lifecycle_repo):
        """The carve-out reads the registry, not "unresolved" — this one is a
        real missing secret and starting without it would hide a broken pipeline."""
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        stub_build(lifecycle_repo, config=_telemetry_config("${OPERATOR_OTLP_SECRET}"))

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code != 0
        assert isinstance(result.exception, ObservabilityCredentialError)
        assert launches == []

    def test_a_mixed_refusal_is_not_deferred(self, runner, launches, lifecycle_repo):
        """One name in the set that the operator does have to supply and the whole
        set stands refused — a session told "nothing to do here" about half of it
        leaves the other half to be discovered as a store that never authenticates."""
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        stub_build(
            lifecycle_repo,
            config=_telemetry_config("${OPERATOR_OTLP_SECRET}${ZO_INGEST_SA_TOKEN}"),
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert isinstance(result.exception, ObservabilityCredentialError)
        # Both names reach the decision — the refusal is the mixed verdict, not
        # an artifact of only one of them being visible.
        assert result.exception.unresolved_vars == (
            "OPERATOR_OTLP_SECRET",
            "ZO_INGEST_SA_TOKEN",
        )
        assert launches == []

    def test_a_blank_credential_still_refuses(self, runner, launches, lifecycle_repo):
        """No variable is named at all, so there is nothing a deploy could issue."""
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        stub_build(lifecycle_repo, config=_telemetry_config('""'))

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert isinstance(result.exception, ObservabilityCredentialError)
        assert launches == []

    def test_a_resolvable_token_keeps_telemetry_on(
        self, runner, launches, lifecycle_repo, monkeypatch
    ):
        """The deferral is about absence only — once `osprey up` has written the
        token, the same config resolves and the session exports normally."""
        stub_build(lifecycle_repo, config=_telemetry_config("${ZO_INGEST_SA_TOKEN}"))
        (lifecycle_repo / ".env").write_text(
            "ZO_INGEST_SA_TOKEN=issued-by-the-store\n", encoding="utf-8"
        )

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert launches[0].env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
        assert "Authorization=Basic " in launches[0].env["OTEL_EXPORTER_OTLP_HEADERS"]
        assert "Telemetry is off" not in result.output
