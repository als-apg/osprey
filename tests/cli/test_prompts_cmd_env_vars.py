"""Tests that _load_config resolves environment variables in config.yml."""

from __future__ import annotations

import yaml


class TestLoadConfigResolvesEnvVars:
    """_load_config should return config with env vars resolved."""

    def test_load_config_resolves_env_vars(self, tmp_path, monkeypatch):
        """Config values like ${MY_VAR:-default} are resolved by _load_config."""
        from osprey.cli.prompts_cmd import _load_config

        # Write a config.yml with bash-style env var syntax
        config_path = tmp_path / "config.yml"
        config_path.write_text(
            yaml.dump(
                {
                    "facility": "${TEST_FACILITY:-default_facility}",
                    "nested": {"tz": "${TEST_TZ:-UTC}"},
                }
            ),
            encoding="utf-8",
        )

        # Set one var, leave the other to use its default
        monkeypatch.setenv("TEST_FACILITY", "ALS")

        result = _load_config(tmp_path)

        # TEST_FACILITY was set -> resolved to "ALS"
        assert result["facility"] == "ALS"
        # TEST_TZ was not set -> resolved to default "UTC"
        assert result["nested"]["tz"] == "UTC"
