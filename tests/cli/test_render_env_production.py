"""Unit tests for ``osprey deploy render-env-production``.

The verb is a thin wrapper: it loads the deploy config the way the deploy path
loads it, parses the ``.env`` it is handed, and hands both to
:func:`osprey.deployment.web_terminals.env_production._build_env_production_subset`
— which is the specification of what a web terminal may see. These tests pin
the wrapper, not the subset rules (those live in
tests/deployment/web_terminals/test_env_production.py): that the right file is
read, that nothing else leaks in, and that the rendered bytes reach stdout or
``--output`` intact.
"""

from __future__ import annotations

import logging
import stat
import textwrap

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_cmd import deploy

# A config with every credential-bearing module enabled, so one render
# exercises the whole subset.
CONFIG_YML = textwrap.dedent(
    """
    facility:
      name: Test Facility
      prefix: test
      timezone: America/Los_Angeles
    llm:
      provider: cborg
      api_key_env_var: CBORG_API_KEY
    ci:
      provider: gitlab
      token_env_var: TEST_CI_TOKEN
    registry:
      url: registry.example.org/test
      token_env_var: TEST_REGISTRY_TOKEN
    modules:
      web_terminals:
        enabled: true
        image_source: local
      olog:
        enabled: true
        username_env_var: OLOG_USERNAME
        password_env_var: OLOG_PASSWORD
      wiki_search:
        enabled: true
        token_env_var: CONFLUENCE_ACCESS_TOKEN
      event_dispatcher:
        enabled: true
        token_env_var: EVENT_DISPATCHER_TOKEN
      ariel:
        enabled: true
        dsn: postgresql://ariel:ariel@ariel-postgres:5432/ariel
    """
)

#: Credentials the agent inside a web terminal presents to systems outside the
#: deploy — the only kind that earns a place in ``.env.production``.
INCLUDED_ENV = {
    "CBORG_API_KEY": "llm-secret",
    "OLOG_USERNAME": "olog-user",
    "OLOG_PASSWORD": "olog-pass",
    "CONFLUENCE_ACCESS_TOKEN": "wiki-secret",
}

#: Build-time credentials and OSPREY's own service-to-service tokens. Present
#: in the ``.env`` on purpose: the render must never carry one across.
EXCLUDED_ENV = {
    "TEST_CI_TOKEN": "ci-secret",
    "TEST_REGISTRY_TOKEN": "registry-secret",
    "EVENT_DISPATCHER_TOKEN": "dispatcher-secret",
    "DISPATCH_WORKER_TOKEN": "worker-secret",
    "BLUESKY_LAUNCH_TOKEN": "bluesky-secret",
}


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


def _write_dotenv(path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def _parse_rendered(text: str) -> dict[str, str]:
    """Parse rendered ``.env.production`` output back into a dict."""
    return dict(
        line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#")
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory holding config.yml and a .env with every secret.

    The command chdirs into the project it resolves; monkeypatch.chdir here
    guarantees the original cwd is restored at teardown.
    """
    project_dir = tmp_path / "facility-project"
    project_dir.mkdir()
    (project_dir / "config.yml").write_text(CONFIG_YML, encoding="utf-8")
    _write_dotenv(project_dir / ".env", {**INCLUDED_ENV, **EXCLUDED_ENV})
    monkeypatch.chdir(project_dir)
    return project_dir


class TestRenderedSubset:
    """What the verb renders, given a project and its .env."""

    def test_renders_expected_subset_to_stdout(self, cli_runner, project):
        """Every external-facing credential, and nothing else."""
        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 0
        assert _parse_rendered(result.stdout) == {
            **INCLUDED_ENV,
            "ARIEL_DSN": "postgresql://ariel:ariel@ariel-postgres:5432/ariel",
            "TZ": "America/Los_Angeles",
        }

    def test_stdout_carries_the_file_and_nothing_else(self, cli_runner, project):
        """`... > .env.production` must produce a usable file, so no banner."""
        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 0
        for line in result.stdout.splitlines():
            assert "=" in line, f"non-assignment line on stdout: {line!r}"

    def test_build_and_service_secrets_are_never_rendered(self, cli_runner, project):
        """The exclusion list is a security spec, not a default."""
        result = cli_runner.invoke(deploy, ["render-env-production"])

        rendered = _parse_rendered(result.stdout)
        for var in EXCLUDED_ENV:
            assert var not in rendered
        for value in EXCLUDED_ENV.values():
            assert value not in result.stdout

    @pytest.mark.parametrize(
        ("module", "expected_vars"),
        [
            ("olog", ("OLOG_USERNAME", "OLOG_PASSWORD")),
            ("wiki_search", ("CONFLUENCE_ACCESS_TOKEN",)),
        ],
    )
    def test_module_credentials_follow_the_enabled_flag(
        self, cli_runner, project, module, expected_vars
    ):
        """A disabled module's credential stays out of the rendered file."""
        enabled = _parse_rendered(cli_runner.invoke(deploy, ["render-env-production"]).stdout)
        for var in expected_vars:
            assert enabled[var] == INCLUDED_ENV[var]

        disabled_config = CONFIG_YML.replace(
            f"  {module}:\n    enabled: true",
            f"  {module}:\n    enabled: false",
        )
        assert disabled_config != CONFIG_YML, f"config fixture no longer enables {module}"
        (project / "config.yml").write_text(disabled_config, encoding="utf-8")
        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 0
        disabled = _parse_rendered(result.stdout)
        for var in expected_vars:
            assert var not in disabled

    def test_timezone_defaults_to_utc(self, cli_runner, project):
        """TZ is always rendered, from facility.timezone or its documented default."""
        (project / "config.yml").write_text(
            CONFIG_YML.replace("  timezone: America/Los_Angeles\n", ""), encoding="utf-8"
        )
        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 0
        assert _parse_rendered(result.stdout)["TZ"] == "UTC"

    def test_long_values_are_not_wrapped(self, cli_runner, project):
        """Rendered lines are file bytes, not console output — never re-wrapped."""
        long_secret = "sk-" + "x" * 200
        _write_dotenv(project / ".env", {**INCLUDED_ENV, "CBORG_API_KEY": long_secret})

        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 0
        assert f"CBORG_API_KEY={long_secret}" in result.stdout.splitlines()


class TestInputSelection:
    """Which files the verb reads — and which it refuses to read."""

    def test_ambient_environment_is_not_a_source(self, cli_runner, project, monkeypatch):
        """Only the handed .env counts: the CLI bulk-loads .env into os.environ
        at startup, so a var that lives solely in the environment must not be
        picked up, or the render would depend on the caller's shell."""
        _write_dotenv(project / ".env", {"CBORG_API_KEY": "from-dotenv"})
        monkeypatch.setenv("CONFLUENCE_ACCESS_TOKEN", "from-shell-only")
        monkeypatch.setenv("OLOG_USERNAME", "from-shell-only")

        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 0
        rendered = _parse_rendered(result.stdout)
        assert rendered["CBORG_API_KEY"] == "from-dotenv"
        assert "CONFLUENCE_ACCESS_TOKEN" not in rendered
        assert "OLOG_USERNAME" not in rendered

    def test_env_file_flag_reads_the_file_it_is_handed(self, cli_runner, project, tmp_path):
        """--env-file overrides the project's own .env."""
        other = tmp_path / "profile.env"
        _write_dotenv(other, {**INCLUDED_ENV, "CBORG_API_KEY": "from-profile-env"})

        result = cli_runner.invoke(deploy, ["render-env-production", "--env-file", str(other)])

        assert result.exit_code == 0
        assert _parse_rendered(result.stdout)["CBORG_API_KEY"] == "from-profile-env"

    def test_relative_env_file_resolves_against_the_invocation_directory(
        self, cli_runner, tmp_path, monkeypatch
    ):
        """A relative --env-file means what the operator typed, not what the
        project directory happens to contain after the command chdirs."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "config.yml").write_text(CONFIG_YML, encoding="utf-8")
        _write_dotenv(project_dir / ".env", {"CBORG_API_KEY": "project-env"})

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        _write_dotenv(elsewhere / "shared.env", {"CBORG_API_KEY": "typed-relative"})
        monkeypatch.chdir(elsewhere)

        result = cli_runner.invoke(
            deploy,
            ["render-env-production", "--project", str(project_dir), "--env-file", "shared.env"],
        )

        assert result.exit_code == 0
        assert _parse_rendered(result.stdout)["CBORG_API_KEY"] == "typed-relative"


class TestOutputFile:
    """--output writes the file the deploy would have generated."""

    def test_output_writes_file_and_keeps_stdout_clean(self, cli_runner, project, caplog):
        caplog.set_level(logging.INFO)
        target = project / ".env.production"

        result = cli_runner.invoke(deploy, ["render-env-production", "--output", str(target)])

        assert result.exit_code == 0
        assert result.stdout == ""
        assert _parse_rendered(target.read_text(encoding="utf-8")) == {
            **INCLUDED_ENV,
            "ARIEL_DSN": "postgresql://ariel:ariel@ariel-postgres:5432/ariel",
            "TZ": "America/Los_Angeles",
        }
        assert str(target) in caplog.text

    def test_output_file_is_created_private(self, cli_runner, project):
        """The file holds live credentials: 0600, like every generated secrets file."""
        target = project / ".env.production"

        cli_runner.invoke(deploy, ["render-env-production", "--output", str(target)])

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_output_overwrites_an_existing_file(self, cli_runner, project):
        """An explicit --output is an instruction, so a re-render replaces it."""
        target = project / ".env.production"
        target.write_text("STALE=leftover\n", encoding="utf-8")

        result = cli_runner.invoke(deploy, ["render-env-production", "--output", str(target)])

        assert result.exit_code == 0
        assert "STALE" not in target.read_text(encoding="utf-8")


class TestMissingInput:
    """Exit codes when there is nothing to render from."""

    def test_missing_env_file_exits_nonzero(self, cli_runner, project, caplog):
        (project / ".env").unlink()

        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 1
        assert ".env" in caplog.text

    def test_missing_named_env_file_names_the_path(self, cli_runner, project, caplog):
        result = cli_runner.invoke(
            deploy, ["render-env-production", "--env-file", str(project / "nope.env")]
        )

        assert result.exit_code == 1
        assert "nope.env" in caplog.text

    def test_missing_config_exits_nonzero(self, cli_runner, project):
        (project / "config.yml").unlink()

        result = cli_runner.invoke(deploy, ["render-env-production"])

        assert result.exit_code == 1
