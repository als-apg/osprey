"""Model provider resolution for Claude Code agent deployments.

Maps canonical model tiers (haiku/sonnet/opus) to provider-specific model IDs
and generates the env block for settings.json. ``ClaudeCodeModelResolver`` does
no file or network I/O; the ``load_provider_spec`` and ``inject_provider_env``
helpers in this module do read ``config.yml`` / ``.env`` from disk.

Design: model IDs are owned by the provider and live in ``api.providers``
in config.yml.  ``CLAUDE_CODE_PROVIDERS`` defines only the auth pattern,
base URL, and default tier — never model IDs.  The resolver reads model IDs
from ``api.providers[name].models`` at runtime, falling back to built-in
defaults only for backward-compatibility with configs that pre-date this
design.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from osprey.build.claude_code_telemetry import (
    TELEMETRY_ENV_VARS,
    _build_telemetry_env,
    _openobserve_host_override,
    _running_in_container,
)
from osprey.models.tiers import VALID_TIERS

logger = logging.getLogger("osprey.build.claude_code_resolver")

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
        # Fallback model IDs (used when api.providers.cborg.models is absent).
        # Pinned to specific versions so Claude Code can pattern-match the model
        # and send the correct thinking/effort schema (e.g. adaptive for 4.7).
        # Unversioned aliases like "anthropic/claude-opus" break capability
        # detection and cause 400s on Vertex-backed Opus 4.7.
        "models": {
            "haiku": "claude-haiku-4-5",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-7",
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


def provider_auth_secret_env(provider_name: str, api_providers: dict | None = None) -> str | None:
    """Name of the shell env var holding ``provider_name``'s auth secret.

    The single source of the secret-var naming rule, shared by
    :meth:`ClaudeCodeModelResolver.resolve` (which injects the secret at
    launch) and the web-terminal ``.env.production`` generator (which must
    ship the same var into per-user containers): built-in providers declare
    ``auth_secret_env`` in :data:`CLAUDE_CODE_PROVIDERS`; a custom proxy
    defined under ``api.providers`` derives ``<NAME>_API_KEY``. Returns
    ``None`` for a provider known to neither — the caller decides whether
    that's an error (:meth:`~ClaudeCodeModelResolver.resolve` raises) or a
    skip (the generator leaves unknown providers to the resolver's own
    validation).
    """
    if provider_name in CLAUDE_CODE_PROVIDERS:
        return CLAUDE_CODE_PROVIDERS[provider_name]["auth_secret_env"]
    if api_providers and provider_name in api_providers:
        return f"{provider_name.upper().replace('-', '_')}_API_KEY"
    return None


AGENT_DEFAULT_TIERS: dict[str, str] = {
    "channel-finder": "haiku",
    "logbook-search": "sonnet",
    "logbook-deep-research": "opus",
    "data-visualizer": "sonnet",
    "pyat-specialist": "sonnet",
}

# Single source of truth for the Claude Code tier→model env vars.
#
# MANAGED_ENV_VARS (scrub, below), resolve() (inject), and
# _apply_e2e_overrides() (e2e-force) all derive the model-tier env-var names
# from this one map, so adding a tier is a one-line change here that cannot
# desync those sites — the drift class behind #350 (a fifth model var reached
# env_block but not the e2e force-tuple).
TIER_MODEL_ENV_VARS: dict[str, str] = {
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
}

# Invariant: the tier map covers exactly the canonical tiers. A drift (a tier
# added to models/tiers.py but not mirrored here, or vice versa) is a module-
# load error, not a silently partial env block.
assert set(TIER_MODEL_ENV_VARS) == VALID_TIERS, (
    "TIER_MODEL_ENV_VARS keys must equal VALID_TIERS "
    f"({sorted(TIER_MODEL_ENV_VARS)} != {sorted(VALID_TIERS)})"
)

# Env vars that settings.json controls — scrubbed from shell before launch
# so runtime-injected provider vars are authoritative.
#
# The rule: scrub every ANTHROPIC_* / CLAUDE_CODE_* var that selects a *backend*
# or a *model*. A stale one of these reroutes the agent away from the configured
# provider without any error — the worst failure mode for a framework that talks
# to control systems. Shared cloud-SDK vars (AWS_REGION, GCLOUD_PROJECT,
# CLOUD_ML_REGION) are deliberately left alone: they belong to other tooling in
# the operator's shell, and only reach Claude Code when a CLAUDE_CODE_USE_* flag
# is set — which is scrubbed here. ANTHROPIC_CUSTOM_HEADERS is likewise left
# alone: headers cannot redirect the endpoint, so a stale value fails loudly,
# and it is the only way to supply corporate-proxy headers.
MANAGED_ENV_VARS = frozenset(
    {
        # Auth + endpoint
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        # Model selectors. The per-tier ANTHROPIC_DEFAULT_*_MODEL names derive
        # from the single TIER_MODEL_ENV_VARS source above, so the scrub set
        # cannot drift from what resolve() injects. (ANTHROPIC_SMALL_FAST_MODEL
        # is deprecated upstream but still honored; CLAUDE_CODE_SUBAGENT_MODEL
        # overrides AGENT_DEFAULT_TIERS.)
        "ANTHROPIC_MODEL",
        *TIER_MODEL_ENV_VARS.values(),
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        # Backend selectors — no OSPREY provider sets these; Bedrock and friends
        # are reached through a proxy base_url, never Claude Code's native backend.
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_MANTLE",
        # Backend endpoint / auth overrides — inert once the flags above are
        # scrubbed, cleared anyway so the agent environment carries no stale
        # backend configuration at all.
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH",
        "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
    }
)


def _managed_policy_settings_paths() -> list[Path]:
    """Return the Claude Code managed-policy settings files for this OS.

    Managed (enterprise) policy settings are the one scope that outranks
    everything OSPREY can reach — the process environment, the project settings
    file, and the ``--setting-sources`` restriction alike. The main file plus
    any fragments in the ``managed-settings.d`` drop-in directory are returned in
    load order (docs: https://code.claude.com/docs/en/settings).
    """
    if sys.platform == "darwin":
        root = Path("/Library/Application Support/ClaudeCode")
    elif sys.platform == "win32":
        root = Path(r"C:\Program Files\ClaudeCode")
    else:
        root = Path("/etc/claude-code")
    paths = [root / "managed-settings.json"]
    dropin = root / "managed-settings.d"
    if dropin.is_dir():
        # Claude Code ignores dropin fragments whose name starts with a dot;
        # skip them too so OSPREY never refuses on a file Claude never applies.
        paths.extend(sorted(p for p in dropin.glob("*.json") if not p.name.startswith(".")))
    return paths


def detect_managed_policy_conflicts(
    paths: list[Path] | None = None,
) -> dict[str, tuple[str, str]]:
    """Return managed-policy ``env`` entries that shadow OSPREY-managed vars.

    A managed-policy ``env`` block outranks OSPREY's runtime-injected provider
    configuration and the ``--setting-sources project`` restriction, so any key
    it sets that OSPREY also manages silently redirects the agent — the wrong
    failure mode for a framework driving control systems. Callers refuse to
    launch on a non-empty result rather than start against a provider the
    project did not configure.

    Args:
        paths: Override the managed-policy files to scan (for testing).
            Defaults to the OS-standard locations.

    Returns:
        ``{var: (policy_value, source_file)}`` for each :data:`MANAGED_ENV_VARS`
        key found in a managed-policy ``env`` block. Missing or unreadable files
        are skipped; a later fragment overriding an earlier one keeps the last
        source, matching Claude Code's own merge order.
    """
    if paths is None:
        paths = _managed_policy_settings_paths()
    conflicts: dict[str, tuple[str, str]] = {}
    for path in paths:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        env = data.get("env")
        if not isinstance(env, dict):
            continue
        for var, value in env.items():
            if var in MANAGED_ENV_VARS:
                conflicts[var] = (str(value), str(path))
    return conflicts


def format_managed_policy_conflicts(conflicts: dict[str, tuple[str, str]]) -> str:
    """Render a launch-refusal message for managed-policy conflicts.

    Shared by every launch path (CLI, Web Terminal, dispatch worker) so the
    refusal reads identically regardless of where it fires.
    """
    lines = [
        "Managed-policy settings override OSPREY-managed provider variables:",
    ]
    for var, (value, source) in sorted(conflicts.items()):
        lines.append(f"    {var} = {value}  ({source})")
    lines.append(
        "Managed policy outranks the project's provider configuration. Remove "
        "these keys from the policy file or reconcile them with config.yml "
        "before launching."
    )
    return "\n".join(lines)


def _load_dotenv(project_dir: Path) -> dict[str, str]:
    """Load a project ``.env`` into a plain dict — the shared raw loader.

    Returns the non-``None`` entries of ``dotenv_values(project_dir/.env)``, or
    ``{}`` when the file is absent or ``python-dotenv`` is not importable. Pure:
    it never touches ``os.environ``, expands no ``${VAR}`` refs, and applies no
    secret or precedence logic — callers own the overlay, expansion, and auth
    handling. Deduplicates the identical load formerly inlined in
    :func:`inject_provider_env`, ``provider_env_for_project``, and
    :func:`load_provider_spec`.
    """
    env_file = Path(project_dir) / ".env"
    if not env_file.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def _env_lookup(project_dir: Path) -> dict[str, str]:
    """Return ``os.environ`` overlaid with the project ``.env`` (``.env`` wins).

    Never mutates global ``os.environ``. Shared by :func:`load_provider_spec`
    and ``osprey.agent_runner.primitives.provider_env_for_project`` — both need
    this same merged view for ``${VAR}``/secret lookups, previously built via
    an identical inline dict-merge in each.
    """
    return {**os.environ, **_load_dotenv(project_dir)}


_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")

# WHATWG URL preprocessing: leading/trailing C0 controls (U+0000–U+001F) and
# space are stripped before parsing. Mirrored here so the predicate matches
# the runtime's parser instead of flagging copy-paste whitespace artifacts.
_C0_AND_SPACE = "".join(map(chr, range(0x21)))


def _warn_on_invalid_proxy_env(environ: dict[str, str]) -> None:
    """Warn (never rewrite) when a proxy env var cannot be parsed as a URL.

    Claude Code's runtime rejects any ``*_PROXY`` value its WHATWG URL parser
    cannot handle (``Invalid proxy URL``) and refuses to start: no scheme, no
    host, a non-numeric port, whitespace, or a comma-joined list all crash it.
    It accepts any WHATWG-parseable URL, including non-http schemes like
    ``socks5://``, so the predicate here must be scheme-agnostic — a bare
    ``startswith("http")`` check would false-positive on a working SOCKS proxy
    at every launch. ``urlsplit`` alone is more lenient than the runtime (it
    permits empty hosts and whitespace, and defers port validation until
    ``.port`` is accessed), so the check forces ``.port`` and rejects
    remaining whitespace in the scheme/authority. It is also *stricter* than a
    WHATWG parser in two ways, mirrored/accepted here: WHATWG preprocessing
    strips leading/trailing space/C0 controls and removes tab/CR/LF before
    parsing (mirrored — such values are validated after the same cleanup, and
    spaces in the path are fine since WHATWG percent-encodes them), and WHATWG
    normalizes slash-elided special-scheme forms like ``http:proxy.corp:3128``
    to ``http://proxy.corp:3128/`` (a known, accepted over-warn: ``urlsplit``
    yields no host there, so this rare typo shape is flagged even though the
    runtime would start — the warning is advisory). On the CLI path the
    runtime's own error is visible, but on the web-terminal and dispatch-worker
    paths a startup crash is opaque — this warning is the only surface that
    names the cause there. The value is deliberately left in place: blanking it
    would break other consumers of the proxy vars (httpx, requests, DuckDB) in
    a quieter way and hide the misconfiguration instead of reporting it.
    """
    for var in _PROXY_ENV_VARS:
        value = environ.get(var, "")
        if not value:
            continue
        # Validate what a WHATWG parser would see, not the raw value: strip
        # leading/trailing space/C0 controls, remove tab/CR/LF anywhere.
        cleaned = value.strip(_C0_AND_SPACE)
        cleaned = cleaned.replace("\t", "").replace("\n", "").replace("\r", "")
        try:
            parts = urlsplit(cleaned)
            _ = parts.port  # raises ValueError on a non-numeric/malformed port
            # Whitespace in the path is percent-encoded by WHATWG, so only
            # scheme/authority whitespace is fatal to the runtime.
            valid = bool(parts.scheme and parts.hostname) and not any(
                c.isspace() for c in parts.scheme + parts.netloc
            )
        except ValueError:
            valid = False
        if not valid:
            logger.warning(
                "%s=%r is not a valid proxy URL — Claude Code will refuse to "
                "start ('Invalid proxy URL'). Fix or unset it in your shell "
                "environment or the project .env.",
                var,
                value,
            )


def inject_provider_env(
    environ: dict[str, str],
    spec: ClaudeCodeModelSpec,
    project_dir: Path | None = None,
) -> list[str]:
    """Scrub managed vars, then overlay the project ``.env``, provider env
    block, and auth into ``environ``.

    Mutates environ in-place. Returns list of injected var names for logging.

    The ``.env`` step copies **every** project ``.env`` key into ``environ``,
    not just API keys: on the host launch paths the ``claude`` CLI expands
    ``.mcp.json`` ``${VAR}`` references (``EPICS_CA_ADDR_LIST``,
    ``PHOEBUS_BRIDGE_URL``, ``BLUESKY_*``) from ``os.environ``, so this is a
    full host-propagation contract — narrowing it would silently break control-
    system MCP addressing. ``.env`` wins over a stale shell export. An
    unparseable ``HTTP_PROXY`` (from ``.env`` or the shell) is carried straight
    through but reported via :func:`_warn_on_invalid_proxy_env` — #352.

    Args:
        environ: Environment dict to mutate (typically os.environ).
        spec: Resolved provider specification.
        project_dir: Project directory containing .env file. If provided, the
            full ``.env`` is copied into ``environ`` (see above) before the auth
            secret is read, so project-level values take precedence over stale
            shell exports.
    """
    # Overlay the full project .env onto environ (host-propagation contract —
    # see docstring). .env wins over stale shell exports.
    if project_dir is not None:
        for key, value in _load_dotenv(project_dir).items():
            environ[key] = value

    # After the overlay, so it sees the effective value regardless of whether
    # it came from .env or a shell export. This is the chokepoint shared by all
    # launch paths (CLI, web terminal, dispatch worker).
    _warn_on_invalid_proxy_env(environ)

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

    # Inject auth. Set the CLI's auth var, and — for proxy providers, where the
    # names differ (e.g. ANTHROPIC_AUTH_TOKEN vs CBORG_API_KEY) — re-assert the
    # raw auth_secret_env too, so the in-context channel-finder MCP subprocess
    # can expand config.yml's ${SECRET} from it. Mirrors provider_env_for_project.
    if spec.auth_secret_env and secret:
        environ[spec.auth_env_var] = secret
        if spec.auth_secret_env != spec.auth_env_var:
            environ[spec.auth_secret_env] = secret

    return sorted(spec.env_block.keys())


def load_provider_spec(
    project_dir: Path,
    *,
    provider: str | None = None,
    include_telemetry: bool = True,
) -> ClaudeCodeModelSpec | None:
    """Read ``config.yml``, expand ``${VAR}`` placeholders, and resolve the spec.

    This is the single chokepoint that resolves environment-variable
    placeholders (e.g. a custom provider's ``base_url: ${ARGO_PROD_URL}``)
    before handing the config to the pure :class:`ClaudeCodeModelResolver`.
    Expansion uses an ``os.environ`` + project ``.env`` overlay (``.env``
    wins, mirroring :func:`inject_provider_env`) and never mutates global
    ``os.environ`` — preserving SDK env-isolation and benchmark
    cross-provider-sweep safety.

    Use this anywhere a ``${VAR}`` in a custom provider's ``base_url`` is
    consumed at *runtime* (the CLI chat/status paths, the SDK runner, the
    web-terminal lifespan, the dispatch worker). Callsites that stay on the
    pure :class:`ClaudeCodeModelResolver` do so on purpose: the template-render
    paths (``templates/claude_code.py``, ``templates/manager.py``) want the
    literal ``${VAR}`` written into ``settings.json`` for deferred runtime
    expansion, and the model-id-only readers (``benchmarks/sdk.py``,
    ``channel_finder_in_context/server_context.py``) consume only
    ``tier_to_model`` wire ids, which never contain ``${VAR}``. See
    ``benchmarks/backends/react_backend.py`` for the one genuine deferral.

    Args:
        project_dir: Path to an initialized OSPREY project (contains config.yml).
        provider: When given, overrides ``claude_code.provider`` in the loaded
            config before resolving — used by cross-provider model sweeps.

    Returns:
        Resolved :class:`ClaudeCodeModelSpec` with ``${VAR}`` expanded in both
        ``env_block['ANTHROPIC_BASE_URL']`` and ``upstream_base_url``, or
        ``None`` when no provider is configured.
    """
    import yaml

    from osprey.utils.config import resolve_env_vars

    project_dir = Path(project_dir)
    raw = yaml.safe_load((project_dir / "config.yml").read_text()) or {}

    # Build an os.environ + .env overlay (.env wins) WITHOUT mutating os.environ.
    lookup: dict[str, str] = _env_lookup(project_dir)

    cfg = resolve_env_vars(raw, environ=lookup)
    cc_config = cfg.get("claude_code", {})
    if provider is not None:
        cc_config = {**cc_config, "provider": provider}
    return ClaudeCodeModelResolver.resolve(
        cc_config,
        cfg.get("api", {}).get("providers", {}),
        include_telemetry=include_telemetry,
    )


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
    needs_proxy: bool = False
    upstream_base_url: str | None = None

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
        """Return {var: (shell_value, settings_value)} for vars where shell != settings.json.

        Telemetry vars (:data:`TELEMETRY_ENV_VARS`) are exempt: they configure
        observability, not the provider backend, so a pre-existing operator
        ``OTEL_*`` / ``CLAUDE_CODE_ENABLE_TELEMETRY`` export is a legitimate
        override — not a conflict that should hard-refuse Web Terminal startup.
        """
        conflicts = {}
        for var, settings_val in self.env_block.items():
            if var in TELEMETRY_ENV_VARS:
                continue
            if var in environ and environ[var] != settings_val:
                conflicts[var] = (environ[var], settings_val)
        return conflicts


class ClaudeCodeModelResolver:
    """Resolves Claude Code model configuration from project config."""

    @staticmethod
    def resolve(
        claude_code_config: dict,
        api_providers: dict | None = None,
        *,
        include_telemetry: bool = True,
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
                "auth_secret_env": provider_auth_secret_env(provider_name, api_providers),
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

        # Ensure all three tiers are present — use Anthropic direct IDs (the
        # built-in "anthropic" provider's own models, not a re-typed copy) as
        # last resort so env block generation never crashes.
        for tier, fallback_id in CLAUDE_CODE_PROVIDERS["anthropic"]["models"].items():
            tier_to_model.setdefault(tier, fallback_id)

        # ── Build env block (literals only — no ${VAR} refs) ────
        # Claude Code's settings.json env block does NOT expand
        # shell variable references; values are treated as literals.
        env_block: dict[str, str] = {}

        # Base URL for Claude Code. Claude Code always appends "/v1/messages",
        # so ANTHROPIC_BASE_URL must be the bare origin — never ending in /v1.
        # OpenAI-compatible endpoints carry a trailing /v1 by convention (it is
        # the OpenAI API root); strip it here so an anthropic-native provider
        # configured with such a URL doesn't resolve to "…/v1/v1/messages"
        # (issue #312). The proxy upstream keeps the original /v1 (see
        # upstream_base_url below); for proxy providers this value is overwritten
        # with the loopback URL at launch. Skipped for direct Anthropic (no base_url).
        if provider_def.get("base_url"):
            env_block["ANTHROPIC_BASE_URL"] = (
                provider_def["base_url"].rstrip("/").removesuffix("/v1")
            )

        # Tier model env vars (all providers) — derived from the single
        # TIER_MODEL_ENV_VARS declaration so this key set can never drift from
        # the e2e-force and scrub-agreement paths (#357). tier_to_model always
        # carries all three tiers (the last-resort setdefault loop above), and the
        # map's insertion order is haiku→sonnet→opus, so both the key set and
        # the insertion order are byte-identical to the prior literal block.
        for tier, env_var in TIER_MODEL_ENV_VARS.items():
            env_block[env_var] = tier_to_model[tier]

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

        # ── Proxy detection ───────────────────────────────────────
        from osprey.infrastructure.proxy.lifecycle import is_proxy_needed

        _needs_proxy = is_proxy_needed(provider_name, api_providers)
        # The proxy forwards to upstream + "/chat/completions", so the upstream
        # must keep its /v1. Use the raw configured base_url, NOT the /v1-stripped
        # ANTHROPIC_BASE_URL above. Every launch path starts the proxy from this
        # field (never from the env var) — see runner.py / dispatch_api.py.
        _upstream_url = provider_def.get("base_url") if _needs_proxy else None

        # ── Telemetry (absent block == disabled → helper returns {}) ──
        # Container context is the ONE place fs/env is consulted; the helper
        # itself stays pure. Telemetry keys are deliberately excluded from
        # MANAGED_ENV_VARS (they are not backend/model selectors).
        # Telemetry is an observability concern, not a provider/model selector.
        # Callers that only read tier_to_model (model-id readers) or that must
        # not let a telemetry misconfig abort provider resolution pass
        # include_telemetry=False; a raised TelemetryConfigError then cannot
        # poison the rest of the spec.
        if include_telemetry:
            telemetry_cfg = claude_code_config.get("telemetry")
            env_block.update(
                _build_telemetry_env(
                    telemetry_cfg,
                    in_container=_running_in_container(),
                    openobserve_host=_openobserve_host_override(),
                )
            )

        return ClaudeCodeModelSpec(
            provider=provider_name,
            env_block=env_block,
            tier_to_model=tier_to_model,
            agent_overrides=agent_overrides,
            default_model_tier=default_tier,
            shell_exports=tuple(shell_exports),
            auth_env_var=auth_env_var,
            auth_secret_env=auth_secret_env,
            needs_proxy=_needs_proxy,
            upstream_base_url=_upstream_url,
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
