"""Unit tests for reading a benchmarked project's secrets and providers."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.channel_finder.benchmarks.project_env import (
    expand_api_providers,
    project_dotenv,
    project_env,
)


class TestProjectDotenv:
    def test_no_env_file_is_an_empty_mapping(self, tmp_path: Path):
        assert project_dotenv(tmp_path) == {}

    def test_reads_the_project_env_file(self, tmp_path: Path):
        (tmp_path / ".env").write_text("GW_TOKEN=tok\nGW_URL=https://gw.example.org/v1\n")

        assert project_dotenv(tmp_path) == {
            "GW_TOKEN": "tok",
            "GW_URL": "https://gw.example.org/v1",
        }


class TestProjectEnv:
    def test_os_environ_wins_over_the_project_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / ".env").write_text("GW_TOKEN=from-file\nGW_ONLY_IN_FILE=file\n")
        monkeypatch.setenv("GW_TOKEN", "from-shell")

        env = project_env(tmp_path)

        assert env["GW_TOKEN"] == "from-shell"
        assert env["GW_ONLY_IN_FILE"] == "file"


class TestExpandApiProviders:
    def test_references_expand_against_the_overlay(self):
        config = {"api": {"providers": {"gw": {"base_url": "${GW_URL}", "api_key": "${GW_TOKEN}"}}}}

        providers = expand_api_providers(
            config, {"GW_URL": "https://gw.example.org/v1", "GW_TOKEN": "tok"}
        )

        assert providers == {"gw": {"base_url": "https://gw.example.org/v1", "api_key": "tok"}}

    def test_an_unset_reference_is_kept_verbatim(self):
        config = {"api": {"providers": {"gw": {"api_key": "${UNSET_GW_TOKEN}"}}}}

        providers = expand_api_providers(config, {})

        assert providers["gw"]["api_key"] == "${UNSET_GW_TOKEN}"

    @pytest.mark.parametrize(
        "config",
        [{}, {"api": None}, {"api": {}}, {"api": {"providers": None}}],
        ids=["no-api", "api-null", "api-empty", "providers-null"],
    )
    def test_a_config_with_no_providers_is_empty(self, config):
        assert expand_api_providers(config, {"X": "y"}) == {}
