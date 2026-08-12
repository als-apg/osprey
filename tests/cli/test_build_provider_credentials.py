"""Tests for the build-time provider credential summary.

``osprey build`` used to report provider API keys only by their *absence*: the
generic ``${VAR}`` resolver logged one INFO line per unresolved placeholder and
said nothing at all about the keys it did resolve. That made the build output
the complement of the useful signal — and never asserted the one fact that
matters, namely whether the *selected* provider's key is available.

These tests pin the replacement: an explicit summary that names the selected
provider, states where its key came from, and groups the rest.
"""

from __future__ import annotations

import logging

import pytest

from osprey.cli.build_environment import (
    CredentialStatus,
    detect_provider_credentials,
    report_provider_credentials,
)


@pytest.fixture
def repo(tmp_path):
    """An empty deployment repo — the directory whose ``.env`` holds the keys."""
    p = tmp_path / "my-control-assistant"
    p.mkdir()
    return p


@pytest.fixture
def project(repo):
    """The repo's build zone, which is what a build hands the detector."""
    p = repo / "build"
    p.mkdir()
    return p


@pytest.fixture
def profile_dir(repo):
    """The repo root, under the parameter name the detector still uses."""
    return repo


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Drop every provider key from the process env.

    ``osprey.utils.config`` eagerly loads a cwd ``.env`` into ``os.environ`` at
    import time, so a developer's real keys would otherwise leak into these
    assertions.
    """
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    for var in PROVIDER_API_KEYS.values():
        if var is not None:
            monkeypatch.delenv(var, raising=False)


def _status_for(statuses: list[CredentialStatus], provider: str) -> CredentialStatus:
    return next(s for s in statuses if s.provider == provider)


class TestDetectProviderCredentials:
    """Detection covers every source the built deployment can authenticate from."""

    def test_key_in_project_env_is_found(self, project, profile_dir):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        cborg = _status_for(statuses, "cborg")
        assert cborg.found is True
        assert cborg.var == "CBORG_API_KEY"
        assert cborg.source == "build/.env"

    def test_key_in_shell_environment_is_found(self, project, profile_dir, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        anthropic = _status_for(statuses, "anthropic")
        assert anthropic.found is True
        assert anthropic.source == "shell environment"

    def test_key_in_profile_env_is_found(self, project, profile_dir):
        """The repo's ``.env`` is the deployment's one secret store."""
        (profile_dir / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        cborg = _status_for(statuses, "cborg")
        assert cborg.found is True
        assert cborg.source == "repo .env"

    def test_project_env_wins_over_profile_env(self, project, profile_dir):
        (profile_dir / ".env").write_text("CBORG_API_KEY=from-profile\n", encoding="utf-8")
        (project / ".env").write_text("CBORG_API_KEY=from-project\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        assert _status_for(statuses, "cborg").source == "build/.env"

    def test_profile_env_wins_over_the_shell(self, project, profile_dir, monkeypatch):
        monkeypatch.setenv("CBORG_API_KEY", "from-shell")
        (profile_dir / ".env").write_text("CBORG_API_KEY=from-profile\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        assert _status_for(statuses, "cborg").source == "repo .env"

    def test_working_directory_dotenv_is_not_a_source(self, project, tmp_path, monkeypatch):
        """An ambient ``.env`` beside the build's cwd is not a credential source.

        It is host state the deployment never carries, so reporting it as
        found described the machine rather than the deployment.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("CBORG_API_KEY=from-cwd\n", encoding="utf-8")

        statuses = detect_provider_credentials(project)

        assert _status_for(statuses, "cborg").found is False

    def test_no_profile_directory_is_accepted(self, project):
        """A build with no resolvable profile still reports the other sources."""
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        statuses = detect_provider_credentials(project)

        assert _status_for(statuses, "cborg").source == "build/.env"

    def test_missing_key_is_reported_not_found(self, project, profile_dir):
        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        openai = _status_for(statuses, "openai")
        assert openai.found is False
        assert openai.source is None

    def test_empty_value_counts_as_missing(self, project, profile_dir):
        """``.env.example`` ships bare ``VAR=`` lines; an empty key is not a key."""
        (project / ".env").write_text("CBORG_API_KEY=\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        assert _status_for(statuses, "cborg").found is False

    def test_keyless_providers_are_excluded(self, project, profile_dir):
        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        names = {s.provider for s in statuses}
        assert "ollama" not in names
        assert "vllm" not in names
        assert "ds4" not in names

    def test_covers_every_keyed_provider_in_the_registry(self, project, profile_dir):
        from osprey.models.provider_registry import PROVIDER_API_KEYS

        statuses = detect_provider_credentials(project, profile_dir=profile_dir)

        expected = {p for p, v in PROVIDER_API_KEYS.items() if v is not None}
        assert {s.provider for s in statuses} == expected


class TestReportProviderCredentials:
    """The summary leads with the selected provider and prints found keys."""

    def test_found_selected_provider_is_reported_with_source(self, project, profile_dir, caplog):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "cborg", profile_dir=profile_dir)

        text = caplog.text
        assert "cborg" in text
        assert "CBORG_API_KEY" in text
        assert "build/.env" in text

    def test_secret_value_is_never_logged(self, project, profile_dir, caplog):
        (project / ".env").write_text("CBORG_API_KEY=super-secret-value\n", encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            report_provider_credentials(project, "cborg", profile_dir=profile_dir)

        assert "super-secret-value" not in caplog.text

    def test_profile_secret_value_is_never_logged(self, project, profile_dir, caplog):
        (profile_dir / ".env").write_text("CBORG_API_KEY=profile-secret\n", encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            report_provider_credentials(project, "cborg", profile_dir=profile_dir)

        assert "profile-secret" not in caplog.text
        assert "repo .env" in caplog.text

    def test_other_found_keys_are_listed(self, project, profile_dir, monkeypatch, caplog):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")
        monkeypatch.setenv("ALS_APG_API_KEY", "als-key")

        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "cborg", profile_dir=profile_dir)

        assert "ALS_APG_API_KEY" in caplog.text

    def test_unset_keys_are_still_listed(self, project, profile_dir, caplog):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "cborg", profile_dir=profile_dir)

        assert "OPENAI_API_KEY" in caplog.text
        assert "ANTHROPIC_API_KEY" in caplog.text

    def test_missing_selected_provider_key_warns(self, project, profile_dir, caplog):
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "anthropic", profile_dir=profile_dir)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a missing selected-provider key must warn, not whisper"
        assert "ANTHROPIC_API_KEY" in caplog.text

    def test_missing_selected_provider_names_the_profile_env_path(
        self, project, profile_dir, caplog
    ):
        """The remedy must point at the profile's ``.env``, which owns the secret.

        Pointing at the render's directory would name the one place the key is
        guaranteed not to survive: the render is ``build/``, wiped and re-made
        whole by every build, and it holds no ``.env`` for a key to be written
        into at all. This is the only message an operator sees when a key is
        missing, so naming the wrong file loses their secret silently.
        """
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "anthropic", profile_dir=profile_dir)

        assert str(profile_dir / ".env") in caplog.text
        assert str(project / ".env") not in caplog.text

    def test_missing_selected_provider_key_does_not_abort(self, project, profile_dir):
        """A missing key is a warning: the project is still worth building."""
        statuses = report_provider_credentials(project, "anthropic", profile_dir=profile_dir)

        assert _status_for(statuses, "anthropic").found is False

    def test_keyless_selected_provider_reports_no_key_required(self, project, profile_dir, caplog):
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "ollama", profile_dir=profile_dir)

        assert "ollama" in caplog.text
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, "a keyless provider must not warn about a missing key"

    def test_unknown_provider_does_not_crash_the_build(self, project, profile_dir, caplog):
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "not-a-real-provider", profile_dir=profile_dir)

        assert "not-a-real-provider" in caplog.text


class TestResolverNoLongerSpamsBuildOutput:
    """The generic ``${VAR}`` resolver reports misses at DEBUG, not INFO."""

    def test_unresolved_placeholder_is_not_logged_at_info(self, caplog, monkeypatch):
        from osprey.utils.config import resolve_env_vars

        monkeypatch.delenv("SOME_UNSET_VAR", raising=False)

        with caplog.at_level(logging.INFO, logger="CONFIG"):
            result = resolve_env_vars("${SOME_UNSET_VAR}")

        assert result == "${SOME_UNSET_VAR}", "placeholder must still survive verbatim"
        assert "SOME_UNSET_VAR" not in caplog.text

    def test_unresolved_placeholder_is_still_available_at_debug(self, caplog, monkeypatch):
        from osprey.utils.config import resolve_env_vars

        monkeypatch.delenv("SOME_UNSET_VAR", raising=False)

        with caplog.at_level(logging.DEBUG, logger="CONFIG"):
            resolve_env_vars("${SOME_UNSET_VAR}")

        assert "SOME_UNSET_VAR" in caplog.text
