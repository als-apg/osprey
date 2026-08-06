"""`api.providers.<name>.base_url` overrides the built-in provider URL.

Before this, `ANTHROPIC_BASE_URL` for a built-in provider (anthropic / cborg /
als-apg) came solely from the hardcoded `CLAUDE_CODE_PROVIDERS` table, so a
facility that pointed a built-in provider at its own gateway got a green
`osprey health` probe against an endpoint the agent never used. The config
value now wins, on the same precedence rule as the `models` tier map, with the
trailing-`/v1` strip unchanged.
"""

from __future__ import annotations

import pytest

from osprey.build.claude_code_resolver import CLAUDE_CODE_PROVIDERS, ClaudeCodeModelResolver

FACILITY_GATEWAY = "https://llm.facility.example.org"


def _tier_map() -> dict[str, str]:
    return {"haiku": "fast", "sonnet": "balanced", "opus": "capable"}


class TestBuiltInProviderOverride:
    """A base_url under api.providers reaches ANTHROPIC_BASE_URL."""

    @pytest.mark.parametrize("provider", sorted(CLAUDE_CODE_PROVIDERS))
    def test_custom_base_url_reaches_env_block(self, provider):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": provider},
            api_providers={provider: {"base_url": FACILITY_GATEWAY}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    @pytest.mark.parametrize("provider", sorted(CLAUDE_CODE_PROVIDERS))
    def test_override_does_not_mutate_the_builtin_table(self, provider):
        """The built-in dict is module-level shared state — resolve() must not write to it."""
        before = CLAUDE_CODE_PROVIDERS[provider]["base_url"]
        ClaudeCodeModelResolver.resolve(
            {"provider": provider},
            api_providers={provider: {"base_url": FACILITY_GATEWAY}},
        )
        assert CLAUDE_CODE_PROVIDERS[provider]["base_url"] == before

    def test_builtin_url_survives_when_config_names_no_base_url(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            api_providers={"als-apg": {"api_key": "k"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.gianlucamartino.com"

    def test_anthropic_gains_a_base_url_only_when_config_gives_one(self):
        """The built-in anthropic entry has no URL; config is the only source."""
        bare = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert "ANTHROPIC_BASE_URL" not in bare.env_block

        fronted = ClaudeCodeModelResolver.resolve(
            {"provider": "anthropic"},
            api_providers={"anthropic": {"base_url": FACILITY_GATEWAY}},
        )
        assert fronted.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY


class TestV1StripSurvivesTheOverride:
    """Claude Code appends /v1/messages, so the env var must never end in /v1."""

    def test_overridden_url_is_stripped_of_v1(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": f"{FACILITY_GATEWAY}/v1"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_overridden_url_is_stripped_of_trailing_slash_and_v1(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": f"{FACILITY_GATEWAY}/v1/"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_shipped_template_urls_are_unchanged_by_the_override(self):
        """cborg/als-apg templates ship the built-in URL + /v1 — same result as before."""
        for provider, shipped, expected in (
            ("cborg", "https://api.cborg.lbl.gov/v1", "https://api.cborg.lbl.gov"),
            (
                "als-apg",
                "https://llm.gianlucamartino.com/v1",
                "https://llm.gianlucamartino.com",
            ),
        ):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": provider}, api_providers={provider: {"base_url": shipped}}
            )
            assert spec.env_block["ANTHROPIC_BASE_URL"] == expected


class TestProxyUpstreamFollowsTheOverride:
    """upstream_base_url stays the single source the launch paths start the proxy from."""

    def test_builtin_providers_are_native_so_no_upstream_is_set(self):
        """Built-ins skip the translation proxy — overriding the URL must not change that."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": f"{FACILITY_GATEWAY}/v1"}},
        )
        assert spec.needs_proxy is False
        assert spec.upstream_base_url is None

    def test_custom_proxy_upstream_keeps_v1_from_the_same_resolved_url(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "my-gateway"},
            api_providers={
                "my-gateway": {"base_url": f"{FACILITY_GATEWAY}/v1", "models": _tier_map()}
            },
        )
        assert spec.needs_proxy is True
        assert spec.upstream_base_url == f"{FACILITY_GATEWAY}/v1"
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY


class TestEndToEndThroughLoadProviderSpec:
    """The on-disk path: config.yml override reaches the spec, ${VAR} included."""

    def test_config_yml_override_reaches_env_block(self, tmp_path):
        from osprey.build.claude_code_resolver import load_provider_spec

        (tmp_path / "config.yml").write_text(
            "api:\n"
            "  providers:\n"
            "    cborg:\n"
            f"      base_url: {FACILITY_GATEWAY}/v1\n"
            "claude_code:\n"
            "  provider: cborg\n"
        )
        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_env_placeholder_in_builtin_override_is_expanded(self, tmp_path, monkeypatch):
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.delenv("FACILITY_GATEWAY_URL", raising=False)
        (tmp_path / "config.yml").write_text(
            "api:\n"
            "  providers:\n"
            "    cborg:\n"
            "      base_url: ${FACILITY_GATEWAY_URL}\n"
            "claude_code:\n"
            "  provider: cborg\n"
        )
        (tmp_path / ".env").write_text(f"FACILITY_GATEWAY_URL={FACILITY_GATEWAY}/v1\n")

        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY


class TestEnvVarBreakGlassOverride:
    """A supplied ``base_url_env_var`` value beats config and the built-in URL.

    Config is often baked into container images at build time; the env var is
    the runtime lever that redirects an already-deployed system at a fallback
    gateway without a rebuild. Only providers that declare ``base_url_env_var``
    in CLAUDE_CODE_PROVIDERS participate, and the value must be handed in via
    ``environ`` — see :class:`TestResolveNeverReadsAmbientEnviron` for why
    ``resolve()`` refuses to read the process environment itself.
    """

    def test_supplied_value_wins_over_builtin_url(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            environ={"ALS_APG_BASE_URL": f"{FACILITY_GATEWAY}/v1"},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_supplied_value_wins_over_config_base_url(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            api_providers={"als-apg": {"base_url": "https://baked.example.org/v1"}},
            environ={"ALS_APG_BASE_URL": f"{FACILITY_GATEWAY}/v1"},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_absent_value_preserves_existing_behavior(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"}, environ={})
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.gianlucamartino.com"

    def test_empty_value_is_ignored(self):
        """An empty export must not blank the URL — fall through to the default."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"}, environ={"ALS_APG_BASE_URL": ""}
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.gianlucamartino.com"

    def test_provider_without_the_declaration_ignores_the_value(self):
        """The override is opt-in per provider, so it cannot leak across them."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            environ={"ALS_APG_BASE_URL": f"{FACILITY_GATEWAY}/v1"},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"


class TestResolveNeverReadsAmbientEnviron:
    """``resolve()`` is pure: the process environment must not reach it.

    This method also renders ``settings.json`` at *build* time
    (``cli/templates/claude_code.py``, ``cli/templates/manager.py``). If it read
    ``os.environ``, whatever gateway the build machine happened to export would
    be baked into the shipped artifact — a CI builder would ship its own test
    endpoint to production. Overrides therefore arrive only via ``environ``,
    which is what ``load_provider_spec`` supplies on the runtime paths.
    """

    def test_process_env_does_not_reach_the_env_block(self, monkeypatch):
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://build-machine.example.org/v1")
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.gianlucamartino.com"

    def test_process_env_does_not_override_a_config_base_url(self, monkeypatch):
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://build-machine.example.org/v1")
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            api_providers={"als-apg": {"base_url": f"{FACILITY_GATEWAY}/v1"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_supplied_environ_beats_the_process_env(self, monkeypatch):
        """The caller's mapping is authoritative, not a merge with os.environ."""
        monkeypatch.setenv("ALS_APG_BASE_URL", "https://build-machine.example.org/v1")
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            environ={"ALS_APG_BASE_URL": f"{FACILITY_GATEWAY}/v1"},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY


class TestOverrideReachesTheRuntimePath:
    """``load_provider_spec`` is where the override becomes live.

    It supplies the ``os.environ`` + project-``.env`` overlay, so a deployment
    redirects itself by exporting the variable (or writing it into ``.env``)
    and restarting the process — the config baked into its image is untouched.
    """

    BAKED_CONFIG = (
        "api:\n"
        "  providers:\n"
        "    als-apg:\n"
        "      base_url: https://baked.example.org/v1\n"
        "claude_code:\n"
        "  provider: als-apg\n"
    )

    def test_process_env_overrides_the_baked_config(self, tmp_path, monkeypatch):
        """The production mechanism: the value arrives in the container's env."""
        from osprey.build.claude_code_resolver import load_provider_spec

        (tmp_path / "config.yml").write_text(self.BAKED_CONFIG)
        monkeypatch.setenv("ALS_APG_BASE_URL", f"{FACILITY_GATEWAY}/v1")

        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_project_dotenv_overrides_the_baked_config(self, tmp_path, monkeypatch):
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
        (tmp_path / "config.yml").write_text(self.BAKED_CONFIG)
        (tmp_path / ".env").write_text(f"ALS_APG_BASE_URL={FACILITY_GATEWAY}/v1\n")

        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_baked_config_stands_when_nothing_overrides_it(self, tmp_path, monkeypatch):
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
        (tmp_path / "config.yml").write_text(self.BAKED_CONFIG)

        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://baked.example.org"
