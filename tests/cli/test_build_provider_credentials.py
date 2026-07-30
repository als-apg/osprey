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
def project(tmp_path):
    """An empty built-project directory."""
    p = tmp_path / "my-control-assistant"
    p.mkdir()
    return p


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
    """Detection covers every source the built project can authenticate from."""

    def test_key_in_project_env_is_found(self, project, tmp_path):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, cwd=tmp_path)

        cborg = _status_for(statuses, "cborg")
        assert cborg.found is True
        assert cborg.var == "CBORG_API_KEY"
        assert cborg.source == "project .env"

    def test_key_in_shell_environment_is_found(self, project, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")

        statuses = detect_provider_credentials(project, cwd=tmp_path)

        anthropic = _status_for(statuses, "anthropic")
        assert anthropic.found is True
        assert anthropic.source == "shell environment"

    def test_key_in_cwd_dotenv_is_found(self, project, tmp_path):
        """The build's own cwd ``.env`` — the case that confused the user."""
        (tmp_path / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, cwd=tmp_path)

        cborg = _status_for(statuses, "cborg")
        assert cborg.found is True
        assert str(tmp_path) in cborg.source

    def test_project_env_wins_over_cwd_dotenv(self, project, tmp_path):
        (tmp_path / ".env").write_text("CBORG_API_KEY=from-cwd\n", encoding="utf-8")
        (project / ".env").write_text("CBORG_API_KEY=from-project\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, cwd=tmp_path)

        assert _status_for(statuses, "cborg").source == "project .env"

    def test_missing_key_is_reported_not_found(self, project, tmp_path):
        statuses = detect_provider_credentials(project, cwd=tmp_path)

        openai = _status_for(statuses, "openai")
        assert openai.found is False
        assert openai.source is None

    def test_empty_value_counts_as_missing(self, project, tmp_path):
        """``.env.template`` ships ``VAR=`` lines; an empty key is not a key."""
        (project / ".env").write_text("CBORG_API_KEY=\n", encoding="utf-8")

        statuses = detect_provider_credentials(project, cwd=tmp_path)

        assert _status_for(statuses, "cborg").found is False

    def test_keyless_providers_are_excluded(self, project, tmp_path):
        statuses = detect_provider_credentials(project, cwd=tmp_path)

        names = {s.provider for s in statuses}
        assert "ollama" not in names
        assert "vllm" not in names
        assert "ds4" not in names

    def test_covers_every_keyed_provider_in_the_registry(self, project, tmp_path):
        from osprey.models.provider_registry import PROVIDER_API_KEYS

        statuses = detect_provider_credentials(project, cwd=tmp_path)

        expected = {p for p, v in PROVIDER_API_KEYS.items() if v is not None}
        assert {s.provider for s in statuses} == expected


class TestReportProviderCredentials:
    """The summary leads with the selected provider and prints found keys."""

    def test_found_selected_provider_is_reported_with_source(self, project, tmp_path, caplog):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "cborg", cwd=tmp_path)

        text = caplog.text
        assert "cborg" in text
        assert "CBORG_API_KEY" in text
        assert "project .env" in text

    def test_secret_value_is_never_logged(self, project, tmp_path, caplog):
        (project / ".env").write_text("CBORG_API_KEY=super-secret-value\n", encoding="utf-8")

        with caplog.at_level(logging.DEBUG):
            report_provider_credentials(project, "cborg", cwd=tmp_path)

        assert "super-secret-value" not in caplog.text

    def test_other_found_keys_are_listed(self, project, tmp_path, monkeypatch, caplog):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")
        monkeypatch.setenv("ALS_APG_API_KEY", "als-key")

        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "cborg", cwd=tmp_path)

        assert "ALS_APG_API_KEY" in caplog.text

    def test_unset_keys_are_still_listed(self, project, tmp_path, caplog):
        (project / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "cborg", cwd=tmp_path)

        assert "OPENAI_API_KEY" in caplog.text
        assert "ANTHROPIC_API_KEY" in caplog.text

    def test_missing_selected_provider_key_warns(self, project, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "anthropic", cwd=tmp_path)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a missing selected-provider key must warn, not whisper"
        assert "ANTHROPIC_API_KEY" in caplog.text

    def test_missing_selected_provider_names_the_project_env_path(self, project, tmp_path, caplog):
        """The hint must point at the built project, not the build cwd."""
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "anthropic", cwd=tmp_path)

        assert str(project / ".env") in caplog.text

    def test_missing_selected_provider_key_does_not_abort(self, project, tmp_path):
        """A missing key is a warning: the project is still worth building."""
        statuses = report_provider_credentials(project, "anthropic", cwd=tmp_path)

        assert _status_for(statuses, "anthropic").found is False

    def test_keyless_selected_provider_reports_no_key_required(self, project, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "ollama", cwd=tmp_path)

        assert "ollama" in caplog.text
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, "a keyless provider must not warn about a missing key"

    def test_unknown_provider_does_not_crash_the_build(self, project, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            report_provider_credentials(project, "not-a-real-provider", cwd=tmp_path)

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
