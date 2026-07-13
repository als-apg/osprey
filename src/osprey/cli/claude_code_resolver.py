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

import base64
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from osprey.models.tiers import VALID_TIERS

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

AGENT_DEFAULT_TIERS: dict[str, str] = {
    "channel-finder": "haiku",
    "logbook-search": "sonnet",
    "logbook-deep-research": "opus",
    "data-visualizer": "sonnet",
}

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
        # Model selectors (ANTHROPIC_SMALL_FAST_MODEL is deprecated upstream
        # but still honored; CLAUDE_CODE_SUBAGENT_MODEL overrides AGENT_DEFAULT_TIERS)
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
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

# OTEL / Claude Code telemetry vars the resolver may inject into the env block.
#
# These are deliberately NOT in MANAGED_ENV_VARS: they configure observability,
# not the provider backend or model, so a stale shell export of one of them
# cannot silently reroute the agent. They are enumerated here so
# ``detect_env_conflicts`` can exempt them — a pre-existing operator OTEL export
# is a legitimate override, not a launch-blocking conflict.
TELEMETRY_ENV_VARS: frozenset[str] = frozenset(
    {
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "OTEL_METRICS_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_RESOURCE_ATTRIBUTES",
        "OTEL_LOG_USER_PROMPTS",
        "OTEL_LOG_ASSISTANT_RESPONSES",
        "OTEL_LOG_TOOL_DETAILS",
        "OTEL_LOG_RAW_API_BODIES",
    }
)

# Content-capture gates: OTEL env var → config key that suppresses it.
# Each defaults ON (emitted as "1") and is dropped only when its config key is
# explicitly ``false``. ``OTEL_LOG_TOOL_CONTENT`` is intentionally absent — it
# requires tracing, which is out of scope for this metrics/logs pipeline.
_TELEMETRY_CONTENT_GATES: dict[str, str] = {
    "OTEL_LOG_USER_PROMPTS": "log_user_prompts",
    "OTEL_LOG_ASSISTANT_RESPONSES": "log_assistant_responses",
    "OTEL_LOG_TOOL_DETAILS": "log_tool_details",
    "OTEL_LOG_RAW_API_BODIES": "log_raw_api_bodies",
}


def _running_in_container() -> bool:
    """Best-effort detection of whether this process runs inside a container.

    Consulted once per :meth:`ClaudeCodeModelResolver.resolve` call to pick the
    OpenObserve default host (``openobserve`` service name vs ``localhost``).
    This is the single place container/filesystem/env state is read; the pure
    :func:`_build_telemetry_env` helper never touches ``os``.
    """
    return os.path.exists("/.dockerenv") or bool(os.environ.get("OSPREY_IN_CONTAINER"))


def _parse_header_map(value: dict | str) -> dict[str, str]:
    """Parse OTLP headers config into a ``{key: value}`` dict.

    Accepts either a mapping (used directly) or a comma-separated ``k=v`` string
    (as OTLP expects on the wire). Only the first ``=`` in each pair is treated
    as the separator, so header values may themselves contain ``=``.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    parsed: dict[str, str] = {}
    for pair in str(value).split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, _sep, val = pair.partition("=")
        parsed[key.strip()] = val.strip()
    return parsed


def _render_kv_map(value: dict | str) -> str:
    """Render a resource-attributes value as a comma-joined ``k=v`` string.

    A pre-formatted string is returned unchanged; a mapping is rendered as
    comma-joined ``key=value`` pairs.
    """
    if isinstance(value, dict):
        return ",".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _resolve_telemetry_endpoint(telemetry_cfg: dict, *, in_container: bool) -> str:
    """Resolve the OTLP exporter endpoint, failing loud on misconfiguration.

    Priority: an explicit ``endpoint`` wins verbatim; otherwise an
    ``openobserve`` backend derives the org endpoint from the container context.
    Any other enabled-but-unaddressed config is a misconfiguration and raises.

    Raises:
        ValueError: If no endpoint can be resolved, or if the resolved endpoint
            still carries a literal ``${VAR}`` — an unresolved placeholder whose
            referenced env var was unset. An unresolved var keeps the literal
            ``${VAR}`` (it does not become empty), so shipping it would point the
            exporter at a nonsense URL; refuse instead.
    """
    endpoint = telemetry_cfg.get("endpoint")
    if not endpoint:
        if telemetry_cfg.get("backend") == "openobserve":
            host = "openobserve" if in_container else "localhost"
            org = telemetry_cfg.get("openobserve", {}).get("org", "default")
            endpoint = f"http://{host}:5080/api/{org}"
        else:
            raise ValueError(
                "claude_code.telemetry is enabled but no 'endpoint' is set and "
                "'backend' is not 'openobserve'; cannot resolve an OTLP endpoint. "
                "Set claude_code.telemetry.endpoint or backend: openobserve."
            )
    endpoint = str(endpoint)
    if "${" in endpoint:
        raise ValueError(
            f"OTLP endpoint still contains an unresolved '${{VAR}}': {endpoint!r}. "
            "The referenced environment variable is unset — refusing to ship a "
            "literal placeholder in the exporter URL."
        )
    return endpoint


def _openobserve_auth_header(telemetry_cfg: dict) -> tuple[str, str]:
    """Compute the HTTP Basic auth header for an OpenObserve backend.

    Both credentials arrive already ``${VAR}``-expanded in the config (the
    config loader expands ``claude_code.telemetry`` before the resolver runs),
    so this reads them straight from the config — never ``os.environ``. The
    base64 header value cannot be expressed as ``${VAR}``; it is computed here.

    Returns:
        ``("Authorization", "Basic <base64(user:password)>")``.

    Raises:
        ValueError: If either credential is missing or blank — an
            unauthenticated exporter to an auth-gated store would silently drop
            every span, so refuse to emit one — or if either credential still
            carries a literal ``${VAR}`` (an unresolved placeholder whose env
            var is unset).
    """
    oo = telemetry_cfg.get("openobserve") or {}
    user = oo.get("user")
    password = oo.get("password")
    if not user or not password:
        raise ValueError(
            "claude_code.telemetry.backend is 'openobserve' but "
            "openobserve.user / openobserve.password are missing or blank; "
            "refusing to emit an unauthenticated OTLP exporter to an "
            "auth-gated store."
        )
    # Fail loud on an unresolved ${VAR} — same contract as the endpoint check.
    # The config loader leaves the literal ``${VAR}`` when the referenced env
    # var is unset; base64-encoding that literal would silently 401 at runtime
    # instead of failing clearly here at resolve() time.
    for field_name, cred in (("openobserve.user", user), ("openobserve.password", password)):
        if "${" in str(cred):
            raise ValueError(
                f"{field_name} still contains an unresolved '${{VAR}}': {cred!r}. "
                "The referenced environment variable is unset — refusing to "
                "encode a literal placeholder into the OpenObserve auth header."
            )
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return "Authorization", f"Basic {token}"


def _build_telemetry_env(
    telemetry_cfg: dict | None, *, in_container: bool = False
) -> dict[str, str]:
    """Build the OTEL/telemetry env block from the ``claude_code.telemetry`` config.

    Pure and side-effect-free: it reads no environment and does no I/O — the
    ``${VAR}``-expansion and container detection happen upstream. Returns an
    empty dict when telemetry is absent or disabled.

    Args:
        telemetry_cfg: The ``claude_code.telemetry`` config section (already
            ``${VAR}``-expanded), or ``None``.
        in_container: Whether the agent runs in a container — selects the
            OpenObserve default host (``openobserve`` vs ``localhost``).

    Returns:
        A ``{VAR: "value"}`` dict; every value is a string (never bool). Keys
        are a subset of :data:`TELEMETRY_ENV_VARS`.

    Raises:
        ValueError: On an unresolvable endpoint, a leaked ``${VAR}`` in the
            endpoint, or an ``openobserve`` backend with missing/blank creds.
    """
    if not telemetry_cfg or not telemetry_cfg.get("enabled"):
        return {}

    env: dict[str, str] = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": str(telemetry_cfg.get("protocol", "http/protobuf")),
        "OTEL_EXPORTER_OTLP_ENDPOINT": _resolve_telemetry_endpoint(
            telemetry_cfg, in_container=in_container
        ),
    }

    # Headers: config headers first, then the computed OpenObserve Basic auth,
    # which wins on key collision. OTLP wire format is comma-separated k=v.
    headers: dict[str, str] = {}
    configured = telemetry_cfg.get("headers")
    if configured:
        headers.update(_parse_header_map(configured))
    if telemetry_cfg.get("backend") == "openobserve":
        key, value = _openobserve_auth_header(telemetry_cfg)
        headers[key] = value
    if headers:
        env["OTEL_EXPORTER_OTLP_HEADERS"] = ",".join(f"{k}={v}" for k, v in headers.items())

    resource_attrs = telemetry_cfg.get("resource_attributes")
    if resource_attrs:
        env["OTEL_RESOURCE_ATTRIBUTES"] = _render_kv_map(resource_attrs)

    # Content-capture gates default ON; each is dropped only on an explicit false.
    for env_var, cfg_key in _TELEMETRY_CONTENT_GATES.items():
        if telemetry_cfg.get(cfg_key) is not False:
            env[env_var] = "1"

    return env


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


def load_provider_spec(
    project_dir: Path,
    *,
    provider: str | None = None,
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
    lookup: dict[str, str] = dict(os.environ)
    env_file = project_dir / ".env"
    if env_file.is_file():
        try:
            from dotenv import dotenv_values

            for key, value in dotenv_values(env_file).items():
                if value is not None:
                    lookup[key] = value
        except ImportError:
            pass

    cfg = resolve_env_vars(raw, environ=lookup)
    cc_config = cfg.get("claude_code", {})
    if provider is not None:
        cc_config = {**cc_config, "provider": provider}
    return ClaudeCodeModelResolver.resolve(
        cc_config,
        cfg.get("api", {}).get("providers", {}),
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
        telemetry_cfg = claude_code_config.get("telemetry")
        env_block.update(_build_telemetry_env(telemetry_cfg, in_container=_running_in_container()))

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
