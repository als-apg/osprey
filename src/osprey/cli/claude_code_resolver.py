"""Model provider resolution for Claude Code agent deployments.

Maps canonical model tiers (haiku/sonnet/opus) to provider-specific model IDs
and generates the env block for settings.json. Pure Python, no I/O.

Design: model IDs are owned by the provider and live in ``api.providers``
in config.yml.  ``CLAUDE_CODE_PROVIDERS`` defines only the auth pattern,
base URL, and default tier — never model IDs.  The resolver reads model IDs
from ``api.providers[name].models`` at runtime, falling back to built-in
defaults only for backward-compatibility with configs that pre-date this
design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CLAUDE_CODE_PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "auth_env_var": "ANTHROPIC_API_KEY",  # Claude Code env var that receives the key
        "auth_secret_env": "ANTHROPIC_API_KEY",  # Shell env var holding the actual secret
        "base_url": None,  # No base URL for direct Anthropic
        "default_model_tier": "sonnet",
        # Fallback model IDs (used when api.providers.anthropic.models is absent)
        "models": {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-5-20250929",
            "opus": "claude-opus-4-6",
        },
    },
    "cborg": {
        "auth_env_var": "ANTHROPIC_AUTH_TOKEN",  # Bearer auth for proxy
        "auth_secret_env": "CBORG_API_KEY",  # Shell env var holding the secret
        "base_url": "https://api.cborg.lbl.gov",  # Well-known URL (no /v1)
        "default_model_tier": "haiku",
        # Fallback model IDs (used when api.providers.cborg.models is absent)
        "models": {
            "haiku": "anthropic/claude-haiku",
            "sonnet": "anthropic/claude-sonnet",
            "opus": "anthropic/claude-opus",
        },
    },
    "als-apg": {
        "auth_env_var": "ANTHROPIC_AUTH_TOKEN",  # Bearer auth for proxy
        "auth_secret_env": "ALS_APG_API_KEY",  # Shell env var holding the secret
        "base_url": "https://llm.gianlucamartino.com",  # ALS-APG AWS proxy (no /v1)
        "default_model_tier": "haiku",
        # Fallback model IDs (used when api.providers.als-apg.models is absent)
        "models": {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-6",
        },
    },
}

AGENT_DEFAULT_TIERS: dict[str, str] = {
    "channel-finder": "haiku",
    "logbook-search": "sonnet",
    "logbook-deep-research": "opus",
    "literature-search": "sonnet",
    "data-visualizer": "sonnet",
    "wiki-search": "sonnet",
    "matlab-search": "sonnet",
    "direct-channel-finder": "sonnet",
}

from osprey.models.tiers import VALID_TIERS

# Env vars that settings.json controls — scrubbed from shell before launch
# so runtime-injected provider vars are authoritative.
MANAGED_ENV_VARS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
    }
)


def inject_provider_env(
    environ: dict[str, str],
    spec: ClaudeCodeModelSpec,
    project_dir: Path | None = None,
) -> list[str]:
    """Scrub managed vars, inject provider env block and auth into environ.

    Mutates environ in-place. Returns list of injected var names for logging.

    Args:
        environ: Environment dict to mutate (typically os.environ).
        spec: Resolved provider specification.
        project_dir: Project directory containing .env file. If provided,
            loads .env values into environ before reading the auth secret,
            so project-level API keys take precedence over stale shell exports.
    """
    # Load project .env so API keys configured there override shell env.
    # This is critical: users update .env but may have stale shell exports.
    if project_dir is not None:
        env_file = Path(project_dir) / ".env"
        if env_file.is_file():
            try:
                from dotenv import dotenv_values

                for key, value in dotenv_values(env_file).items():
                    if value is not None:
                        environ[key] = value
            except ImportError:
                pass

    # Read auth secret BEFORE scrubbing — auth_secret_env may be in MANAGED_ENV_VARS
    # (e.g. ANTHROPIC_API_KEY for the anthropic provider)
    secret = None
    if spec.auth_secret_env:
        secret = environ.get(spec.auth_secret_env)

    # Scrub all managed vars
    for var in MANAGED_ENV_VARS:
        environ.pop(var, None)

    # Inject provider env block
    for key, value in spec.env_block.items():
        environ[key] = value

    # Inject auth
    if spec.auth_secret_env and secret:
        environ[spec.auth_env_var] = secret

    return sorted(spec.env_block.keys())


@dataclass(frozen=True)
class ClaudeCodeModelSpec:
    """Resolved model provider configuration for Claude Code.

    Attributes:
        provider: Provider name (e.g. "cborg", "anthropic").
        env_block: Key-value pairs to inject into settings.json ``env``.
            Contains only literal values (no ``${VAR}`` references, since
            Claude Code's env block does not expand them).
        tier_to_model: Maps canonical tiers to concrete model IDs.
        agent_overrides: Per-agent tier overrides from config.
        default_model_tier: Default model tier for settings.json ``model`` key.
        shell_exports: Shell export lines the user must add to their profile
            (e.g. ``export ANTHROPIC_AUTH_TOKEN="$CBORG_API_KEY"``).
    """

    provider: str
    env_block: dict[str, str] = field(default_factory=dict)
    tier_to_model: dict[str, str] = field(default_factory=dict)
    agent_overrides: dict[str, str] = field(default_factory=dict)
    default_model_tier: str = "sonnet"
    shell_exports: tuple[str, ...] = ()
    auth_env_var: str = ""
    auth_secret_env: str = ""

    def agent_tier(self, name: str) -> str:
        """Resolve the tier alias for a named agent (for Claude Code model: frontmatter).

        Resolution order:
        1. Per-agent override from config
        2. Default tier from AGENT_DEFAULT_TIERS
        3. Falls back to "sonnet" for unknown agents
        """
        if name in self.agent_overrides:
            return self.agent_overrides[name]
        elif name in AGENT_DEFAULT_TIERS:
            return AGENT_DEFAULT_TIERS[name]
        return "sonnet"

    def agent_model(self, name: str) -> str:
        """Resolve the concrete model ID for a named agent."""
        tier = self.agent_tier(name)
        return self.tier_to_model.get(tier, tier)

    def detect_env_conflicts(self, environ: dict[str, str]) -> dict[str, tuple[str, str]]:
        """Return {var: (shell_value, settings_value)} for vars where shell != settings.json."""
        conflicts = {}
        for var, settings_val in self.env_block.items():
            if var in environ and environ[var] != settings_val:
                conflicts[var] = (environ[var], settings_val)
        return conflicts


class ClaudeCodeModelResolver:
    """Resolves Claude Code model configuration from project config."""

    @staticmethod
    def resolve(
        claude_code_config: dict,
        api_providers: dict | None = None,
    ) -> ClaudeCodeModelSpec | None:
        """Build a ``ClaudeCodeModelSpec`` from config.

        Model ID resolution order (highest to lowest priority):
        1. ``claude_code.models`` per-tier overrides in config.yml
        2. ``api.providers[name].models`` — the provider's own model IDs
        3. Built-in ``models`` in CLAUDE_CODE_PROVIDERS (backward compat)

        This means providers own their model naming: set models under
        ``api.providers`` and the framework picks them up automatically.
        No model IDs need to be hardcoded in Python.

        Args:
            claude_code_config: The ``claude_code`` section of config.yml.
            api_providers: The ``api.providers`` section (optional).

        Returns:
            Resolved spec, or ``None`` when no provider is configured.

        Raises:
            ValueError: If the provider name is not in CLAUDE_CODE_PROVIDERS
                and not in api_providers.
        """
        provider_name = claude_code_config.get("provider")
        if not provider_name:
            return None

        api_providers = api_providers or {}

        if provider_name not in CLAUDE_CODE_PROVIDERS:
            # Custom proxy: must be defined in api.providers
            if provider_name not in api_providers:
                supported = ", ".join(sorted(CLAUDE_CODE_PROVIDERS))
                raise ValueError(
                    f"Unknown Claude Code provider '{provider_name}'. "
                    f"Built-in providers: {supported}. "
                    f"To use a custom provider, add it to api.providers in config.yml."
                )
            provider_entry = api_providers[provider_name]
            provider_def = {
                "auth_env_var": "ANTHROPIC_AUTH_TOKEN",
                "auth_secret_env": f"{provider_name.upper().replace('-', '_')}_API_KEY",
                "base_url": provider_entry.get("base_url"),
                "default_model_tier": "opus",
                "models": {},
            }
        else:
            provider_def = CLAUDE_CODE_PROVIDERS[provider_name]

        # ── Build tier → model mapping ───────────────────────────
        # Priority: built-in fallback < api.providers models < claude_code.models

        # Start with built-in fallbacks (backward compat for old configs)
        tier_to_model = dict(provider_def.get("models", {}))

        # Override with models defined in api.providers[name].models
        # This is the authoritative source: providers own their model naming.
        provider_api_models = api_providers.get(provider_name, {}).get("models", {})
        for tier, model_id in provider_api_models.items():
            if tier in VALID_TIERS:
                tier_to_model[tier] = model_id

        # Apply explicit per-tier overrides from claude_code.models
        model_overrides = claude_code_config.get("models", {})
        for tier, model_id in (model_overrides or {}).items():
            if tier in VALID_TIERS:
                tier_to_model[tier] = model_id

        # Ensure all three tiers are present — use Anthropic direct IDs as
        # last resort so env block generation never crashes.
        _last_resort = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-5-20250929",
            "opus": "claude-opus-4-6",
        }
        for tier, fallback_id in _last_resort.items():
            tier_to_model.setdefault(tier, fallback_id)

        # ── Build env block (literals only — no ${VAR} refs) ────
        # Claude Code's settings.json env block does NOT expand
        # shell variable references; values are treated as literals.
        env_block: dict[str, str] = {}

        # Base URL: use provider literal (no /v1); skip for direct Anthropic
        if provider_def.get("base_url"):
            env_block["ANTHROPIC_BASE_URL"] = provider_def["base_url"]

        # Tier model env vars (all providers)
        env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = tier_to_model["haiku"]
        env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] = tier_to_model["sonnet"]
        env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] = tier_to_model["opus"]

        # ── Shell exports (auth key — must be set in user's profile) ──
        auth_env_var = provider_def["auth_env_var"]
        auth_secret_env = provider_def["auth_secret_env"]
        if auth_env_var == auth_secret_env:
            # Direct provider (e.g. anthropic): just needs the var set
            shell_exports = (f'export {auth_env_var}="<your-api-key>"',)
        else:
            # Proxy provider (e.g. cborg): alias one var to another
            shell_exports = (f'export {auth_env_var}="${auth_secret_env}"',)

        # ── Default model tier ───────────────────────────────────
        default_tier = claude_code_config.get("default_model", provider_def["default_model_tier"])
        if default_tier not in VALID_TIERS:
            default_tier = provider_def["default_model_tier"]

        # ANTHROPIC_MODEL: override any shell-level value so the
        # project's chosen tier is authoritative.
        env_block["ANTHROPIC_MODEL"] = tier_to_model[default_tier]

        # ── Agent overrides ──────────────────────────────────────
        agent_overrides = dict(claude_code_config.get("agent_models", {}) or {})

        return ClaudeCodeModelSpec(
            provider=provider_name,
            env_block=env_block,
            tier_to_model=tier_to_model,
            agent_overrides=agent_overrides,
            default_model_tier=default_tier,
            shell_exports=tuple(shell_exports),
            auth_env_var=auth_env_var,
            auth_secret_env=auth_secret_env,
        )

    @staticmethod
    def validate_provider(name: str, api_providers: dict | None = None) -> bool:
        """Check whether a provider name is supported.

        Returns True for built-in providers and for any name that appears in
        api_providers (custom proxies).
        """
        if name in CLAUDE_CODE_PROVIDERS:
            return True
        return api_providers is not None and name in api_providers
