"""L3 — resolver/config integration for a custom `cborg-oss` Claude Code provider.

Validates the configuration path that makes CBORG's self-hosted models reachable
from Claude Code (Route B): a custom provider pointing at CBORG's OpenAI endpoint,
routed through OSPREY's translation proxy.

Pure Python, no network.

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

from osprey.build.claude_code_resolver import (
    ClaudeCodeModelResolver,
    inject_provider_env,
)

# A user wanting cborg-coder in Claude Code adds this to config.yml:
CBORG_OSS_API_PROVIDERS = {
    "cborg-oss": {
        "api_key": "${CBORG_API_KEY}",
        "base_url": "https://api.cborg.lbl.gov/v1",  # OpenAI route, WITH /v1
        "models": {
            "haiku": "cborg-coder-fast",
            "sonnet": "cborg-coder",
            "opus": "cborg-deepthought",
        },
    }
}
CBORG_OSS_CC_CONFIG = {"provider": "cborg-oss", "default_model": "sonnet"}


def test_custom_provider_resolves_with_proxy():
    spec = ClaudeCodeModelResolver.resolve(CBORG_OSS_CC_CONFIG, CBORG_OSS_API_PROVIDERS)
    assert spec is not None
    assert spec.provider == "cborg-oss"
    # Self-hosted model IDs from api.providers win:
    assert spec.tier_to_model["sonnet"] == "cborg-coder"
    assert spec.tier_to_model["haiku"] == "cborg-coder-fast"
    assert spec.tier_to_model["opus"] == "cborg-deepthought"
    # Default tier model is the selected ANTHROPIC_MODEL:
    assert spec.env_block["ANTHROPIC_MODEL"] == "cborg-coder"
    # Crucially: a custom (non-native) provider needs the translation proxy:
    assert spec.needs_proxy is True
    # Upstream keeps /v1 (proxy appends /chat/completions); the Claude-Code-facing
    # var is stripped of /v1 since Claude Code appends /v1/messages (issue #312).
    assert spec.upstream_base_url == "https://api.cborg.lbl.gov/v1"
    assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"


def test_builtin_cborg_is_anthropic_native_no_proxy():
    """Contrast: the built-in `cborg` provider talks Anthropic natively (Claude tiers)."""
    spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"}, {})
    assert spec is not None
    assert spec.needs_proxy is False
    assert spec.upstream_base_url is None
    # Built-in cborg pins Claude models, not self-hosted ones:
    assert spec.tier_to_model["sonnet"] == "claude-sonnet-4-6"


def test_auth_secret_env_is_derived_from_provider_name():
    """FOOTGUN: the proxy key comes from a name-derived env var, NOT api_key in config.

    For provider `cborg-oss`, the resolver expects CBORG_OSS_API_KEY — it does NOT
    read api.providers['cborg-oss'].api_key (that feeds OSPREY's direct LLM path).
    Users must export CBORG_OSS_API_KEY (e.g. =$CBORG_API_KEY).
    """
    spec = ClaudeCodeModelResolver.resolve(CBORG_OSS_CC_CONFIG, CBORG_OSS_API_PROVIDERS)
    assert spec.auth_secret_env == "CBORG_OSS_API_KEY"
    assert spec.auth_env_var == "ANTHROPIC_AUTH_TOKEN"


def test_inject_provider_env_wires_auth_and_scrubs_managed_vars():
    spec = ClaudeCodeModelResolver.resolve(CBORG_OSS_CC_CONFIG, CBORG_OSS_API_PROVIDERS)
    environ = {
        "CBORG_OSS_API_KEY": "secret-123",
        # Stale managed var that must be scrubbed before injection:
        "ANTHROPIC_API_KEY": "stale-direct-key",
        "ANTHROPIC_MODEL": "stale-model",
    }
    injected = inject_provider_env(environ, spec)

    # Auth aliased into the var Claude Code reads for proxy providers:
    assert environ["ANTHROPIC_AUTH_TOKEN"] == "secret-123"
    # Stale direct key scrubbed (proxy provider must not present a direct key):
    assert "ANTHROPIC_API_KEY" not in environ
    # Project's chosen model is authoritative over the stale shell value:
    assert environ["ANTHROPIC_MODEL"] == "cborg-coder"
    # Claude-Code-facing var is stripped of the OpenAI /v1 (issue #312).
    assert environ["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"
    assert "ANTHROPIC_BASE_URL" in injected


def test_per_agent_tier_overrides_resolve_to_self_hosted_models():
    cc = {
        "provider": "cborg-oss",
        "default_model": "sonnet",
        "agent_models": {"channel-finder": "haiku", "logbook-deep-research": "opus"},
    }
    spec = ClaudeCodeModelResolver.resolve(cc, CBORG_OSS_API_PROVIDERS)
    assert spec.agent_model("channel-finder") == "cborg-coder-fast"  # haiku tier
    assert spec.agent_model("logbook-deep-research") == "cborg-deepthought"  # opus tier
    assert spec.agent_model("data-visualizer") == "cborg-coder"  # default sonnet tier
