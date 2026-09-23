"""Model provider resolution for Claude Code agent deployments.

Resolves the deployment's main model, each agent's model and Claude Code's own
three alias models from the ids the provider serves, and builds the env block
injected into the agent's process environment at launch.
``ClaudeCodeModelResolver`` does no file or network I/O; the
``load_provider_spec`` and ``inject_provider_env`` helpers in this module do
read ``config.yml`` / ``.env`` from disk.

Design: model ids and endpoints are owned by the provider and live in
``api.providers`` in config.yml — each entry lists the ids its gateway serves
and names its ``default_model``. ``CLAUDE_CODE_PROVIDERS`` defines the auth
pattern and fallback base URLs for configs that name none — config always wins
over the built-in table.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from osprey.build.claude_code_telemetry import (
    TELEMETRY_ENV_VARS,
    _build_telemetry_env,
    _openobserve_host_override,
    _running_in_container,
    resolve_openobserve_port,
)
from osprey.models.display import CLAUDE_CODE_ALIASES, claude_code_alias_candidates
from osprey.models.spend_attribution import apply_attribution_env, gateway_for
from osprey.utils.dotenv import chain_files
from osprey_connectors import yaml_loader
from osprey_connectors.config import is_unresolved_placeholder

logger = logging.getLogger("osprey.build.claude_code_resolver")

CLAUDE_CODE_PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "auth_env_var": "ANTHROPIC_API_KEY",  # Claude Code env var that receives the key
        "auth_secret_env": "ANTHROPIC_API_KEY",  # Shell env var holding the actual secret
        "base_url": None,  # No base URL for direct Anthropic
    },
    "cborg": {
        "auth_env_var": "ANTHROPIC_AUTH_TOKEN",  # Bearer auth for proxy
        "auth_secret_env": "CBORG_API_KEY",  # Shell env var holding the secret
        "base_url": "https://api.cborg.lbl.gov",  # Well-known URL (no /v1)
    },
    "als-apg": {
        "auth_env_var": "ANTHROPIC_AUTH_TOKEN",  # Bearer auth for proxy
        "auth_secret_env": "ALS_APG_API_KEY",  # Shell env var holding the secret
        # The gateway's own address, so a deployment that names none still
        # reaches it. Spelled without /v1 because this value becomes
        # ANTHROPIC_BASE_URL and Claude Code appends /v1/messages itself.
        "base_url": "https://llm.als.lbl.gov",
        # A site whose gateway is elsewhere names it in
        # api.providers.als-apg.base_url, or in this variable, which beats both
        # the config and the address above — so an already-deployed system can
        # be pointed at a fallback gateway without a rebuild (mirrors the
        # provider adapter's base_url_env_var).
        "base_url_env_var": "ALS_APG_BASE_URL",
    },
}

#: The main model of direct Anthropic when neither ``api.providers.anthropic``
#: nor the packaged catalog names one.
_DIRECT_ANTHROPIC_DEFAULT = "claude-sonnet-5"


class ProviderEndpointError(ValueError):
    """A provider that fronts a gateway was given no endpoint to call.

    Its own error type, not a bare ``ValueError``, because one caller has to
    tell it apart: the web pre-flight skips the ordinary config faults its
    provider read can raise (an unparseable file, an unknown provider name —
    each already diagnosed elsewhere) and would otherwise swallow this one too,
    leaving the operator with a silent launch and a server that exits during
    startup. Subclasses ``ValueError`` so every caller that catches the broad
    type keeps catching this.
    """


def provider_base_url_env(provider_name: str) -> str | None:
    """Name of the env var a built-in provider reads its gateway endpoint from.

    The endpoint counterpart of :func:`provider_auth_secret_env`, and the same
    indirection: the table records WHERE a deployment keeps its gateway URL,
    never the URL. ``None`` for a provider that declares no such variable, and
    for one the built-in table has never heard of — a custom proxy's endpoint
    is named by its own ``api.providers`` entry instead.
    """
    return (CLAUDE_CODE_PROVIDERS.get(provider_name) or {}).get("base_url_env_var")


def provider_requires_base_url(provider_name: str) -> bool:
    """Whether ``provider_name`` fronts a gateway that ships no default endpoint.

    True means a launch that resolves no URL is refused
    (:class:`ProviderEndpointError`) rather than falling back to a host, so
    whatever names the endpoint has to be delivered to the process that calls
    the gateway.
    """
    return bool((CLAUDE_CODE_PROVIDERS.get(provider_name) or {}).get("requires_base_url"))


def provider_auth_secret_env(provider_name: str, api_providers: dict | None = None) -> str | None:
    """Name of the shell env var holding ``provider_name``'s auth secret.

    The single source of the secret-var naming rule, shared by
    :meth:`ClaudeCodeModelResolver.resolve` (which injects the secret at
    launch) and the web-terminal ``.env.users`` generator (which must
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


# The three env vars Claude Code reads for its own alias names — the model it
# runs when it, or an agent's ``model:`` frontmatter, asks for haiku, sonnet or
# opus, and the model behind its own background calls. The alias words and the
# var names are Claude Code's contract; OSPREY fills them and names its own
# models by id.
#
# MANAGED_ENV_VARS (scrub, below), resolve() (inject), and
# _apply_e2e_overrides() (e2e-force) all derive the alias env-var names from
# this one map, so the three sites cannot desync.
TIER_MODEL_ENV_VARS: dict[str, str] = {
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
}

# Invariant: one env var per Claude Code alias name, in the same order. A drift
# is a module-load error, not a silently partial env block.
assert tuple(TIER_MODEL_ENV_VARS) == CLAUDE_CODE_ALIASES, (
    "TIER_MODEL_ENV_VARS keys must be Claude Code's alias names "
    f"({list(TIER_MODEL_ENV_VARS)} != {list(CLAUDE_CODE_ALIASES)})"
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
        # Model selectors. The per-alias ANTHROPIC_DEFAULT_*_MODEL names derive
        # from the single TIER_MODEL_ENV_VARS source above, so the scrub set
        # cannot drift from what resolve() injects. (ANTHROPIC_SMALL_FAST_MODEL
        # is deprecated upstream but still honored; CLAUDE_CODE_SUBAGENT_MODEL
        # overrides every agent's model: frontmatter.)
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
    """Load a project's env chain into one plain dict — the shared raw loader.

    Reads the chain files that exist under ``project_dir`` in ascending
    precedence (``.env.shared`` then ``.env``) and lays each over the last, so
    a key both files set arrives with the host-local value. Returns the
    non-``None`` entries, or ``{}`` when no chain file is present or
    ``python-dotenv`` is not importable. Pure: it never touches ``os.environ``,
    applies no secret logic, and leaves the overlay, ``${VAR}`` expansion, and
    auth handling to its callers. The single load shared by
    :func:`inject_provider_env`, ``provider_env_for_project``, and
    :func:`load_provider_spec`, so all three see the same chain the same way.

    Each file is read with ``dotenv_values`` rather than the plain parser
    because these values reach a launched agent's environment, and the CLI's
    own loaders read the same chain with python-dotenv — one parser across
    both halves keeps a quoted or multi-line secret meaning the same thing
    wherever it is read.
    """
    paths = chain_files(Path(project_dir))
    if not paths:
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    merged: dict[str, str] = {}
    for path in paths:
        merged.update(
            {key: value for key, value in dotenv_values(path).items() if value is not None}
        )
    return merged


def _env_lookup(project_dir: Path) -> dict[str, str]:
    """Return ``os.environ`` overlaid with the project's env chain (chain wins).

    Never mutates global ``os.environ``. Shared by :func:`load_provider_spec`
    and ``osprey.agent_runner.primitives.provider_env_for_project`` — both need
    this same merged view for ``${VAR}``/secret lookups, and must agree on the
    precedence.
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
    """Scrub managed vars, then overlay the project's env chain, provider env
    block, and auth into ``environ``.

    Mutates environ in-place. Returns list of injected var names for logging.

    The env step copies **every** key of the project's env chain into
    ``environ``, not just API keys: on the host launch paths the ``claude`` CLI
    expands ``.mcp.json`` ``${VAR}`` references (``EPICS_CA_ADDR_LIST``,
    ``PHOEBUS_BRIDGE_URL``, ``BLUESKY_*``) from ``os.environ``, so this is a
    full host-propagation contract — narrowing it would silently break control-
    system MCP addressing. The chain is merged local-over-shared before the
    overlay, and the result wins over a stale shell export. An unparseable
    ``HTTP_PROXY`` (from the chain or the shell) is carried straight through
    but reported via :func:`_warn_on_invalid_proxy_env` — #352.

    Args:
        environ: Environment dict to mutate (typically os.environ).
        spec: Resolved provider specification.
        project_dir: Project directory holding the env chain. If provided, the
            full merged chain is copied into ``environ`` (see above) before the
            auth secret is read, so project-level values take precedence over
            stale shell exports.
    """
    # Overlay the full project env chain onto environ (host-propagation
    # contract — see docstring). The chain wins over stale shell exports.
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

    # On a LiteLLM-fronted provider, stamp the acting identity onto every
    # request the agent makes (ANTHROPIC_CUSTOM_HEADERS), merged into any
    # corporate-proxy headers the operator already carries there. Resolved
    # here, at launch, because the identity is a property of the running
    # container (OSPREY_TERMINAL_USER), not of the render.
    apply_attribution_env(environ, spec.gateway)

    return sorted(spec.env_block.keys())


def _without_unresolved_base_urls(api_providers: dict) -> dict:
    """Drop every ``base_url`` that is still an unexported ``${VAR}``.

    This is the *runtime* path, where a placeholder is not a value: the config
    resolver keeps ``${VAR}`` verbatim when the variable is unset, and a config
    may well spell its gateway endpoint as a reference
    (``base_url: ${SITE_GATEWAY_URL}``). Left in place the literal would be
    exported as ``ANTHROPIC_BASE_URL``, i.e. handed to the agent as a hostname.
    Blanked, it falls through the same precedence chain a missing key does —
    to the built-in URL, or to the refusal that names the variable.

    Only a launch blanks them. The template-render callers stay on the pure
    :class:`ClaudeCodeModelResolver` precisely because they *want* the literal
    ``${VAR}`` written into ``settings.json`` for expansion at launch, and the
    build's reachability check reads a render the same way — see
    ``defer_unresolved_base_url`` on :func:`load_provider_spec`.

    Args:
        api_providers: The ``api.providers`` mapping, already env-resolved.

    Returns:
        The same mapping with unresolved ``base_url`` values set to ``None``.
    """
    return {
        name: (
            {**entry, "base_url": None}
            if isinstance(entry, dict) and is_unresolved_placeholder(entry.get("base_url"))
            else entry
        )
        for name, entry in (api_providers or {}).items()
    }


def load_provider_spec(
    project_dir: Path,
    *,
    env_dir: Path | None = None,
    provider: str | None = None,
    include_telemetry: bool = True,
    defer_unresolved_telemetry_creds: bool = False,
    defer_unresolved_base_url: bool = False,
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
    model ids, which never contain ``${VAR}``.
    ``benchmarks/backends/react_backend.py`` keeps its own resolver — its
    contract is a synthetic config plus litellm ``api_base``, not a spec — but
    expands and refuses the same way this does.

    Args:
        project_dir: Directory holding the ``config.yml`` to resolve.
        env_dir: Directory holding the deployment's ``.env``, when it is not the
            one holding the config. A deployment repo keeps secrets at its root
            and the rendered config under ``build/``, so a caller reading the
            render has to name the repo root here or a ``base_url:
            ${ARGO_PROD_URL}`` resolves to the literal placeholder. Defaults to
            ``project_dir`` — the flat layout, where the two coincide.
        provider: When given, overrides ``claude_code.provider`` in the loaded
            config before resolving — used by cross-provider model sweeps.
        defer_unresolved_base_url: Keep a provider ``base_url`` that is still an
            unexported ``${VAR}`` instead of reading it as "no endpoint". The
            build's reachability check sets it: a render is an artifact that can
            start on another host — a container image is built here and given
            its gateway there — so a deferred reference is that render's
            contract with its runtime, not a missing value. Every launch path
            leaves it False, so the process about to call the gateway is the one
            that refuses, by name.

    Returns:
        Resolved :class:`ClaudeCodeModelSpec` with ``${VAR}`` expanded in both
        ``env_block['ANTHROPIC_BASE_URL']`` and ``upstream_base_url``, or
        ``None`` when no provider is configured.
    """

    from osprey.utils.config import resolve_env_vars

    project_dir = Path(project_dir)
    raw = yaml_loader.safe_load((project_dir / "config.yml").read_text()) or {}

    # Build an os.environ + .env overlay (.env wins) WITHOUT mutating os.environ.
    lookup: dict[str, str] = _env_lookup(Path(env_dir) if env_dir is not None else project_dir)

    cfg = resolve_env_vars(raw, environ=lookup)
    cc_config = cfg.get("claude_code", {})
    if provider is not None:
        cc_config = {**cc_config, "provider": provider}
    api_providers = cfg.get("api", {}).get("providers", {})
    if not defer_unresolved_base_url:
        api_providers = _without_unresolved_base_urls(api_providers)
    return ClaudeCodeModelResolver.resolve(
        cc_config,
        api_providers,
        include_telemetry=include_telemetry,
        defer_unresolved_telemetry_creds=defer_unresolved_telemetry_creds,
        environ=lookup,
        # A runtime launch: the deploy environment's port declaration wins,
        # else the port this deployment publishes the store on. Resolved only
        # when telemetry is being built: it is a telemetry input, and a
        # malformed port must not poison provider resolution for a caller
        # that asked for none (the dispatch worker's degrade-and-retry relies
        # on include_telemetry=False never raising a telemetry fault).
        openobserve_port=resolve_openobserve_port(cfg) if include_telemetry else None,
    )


@dataclass(frozen=True)
class ClaudeCodeModelSpec:
    """Resolved model provider configuration for Claude Code.

    Attributes:
        provider: Provider name (e.g. "cborg", "anthropic").
        default_model_id: The deployment's main model — ``claude_code.default_model``
            when set, else the provider entry's ``default_model``. It is what
            ``ANTHROPIC_MODEL`` carries and what every agent without a model of
            its own runs.
        env_block: Key-value pairs injected into the agent's process
            environment at launch. Contains only literal values (no ``${VAR}``
            references).
        alias_models: Claude Code's three alias names (haiku, sonnet, opus) →
            the model id each resolves to. Always carries all three.
        alias_origin: Per alias, where its model came from:
            ``"claude_code.aliases"``, ``"catalog"`` (the entry's
            ``claude_code_aliases``), ``"derived"`` (the newest served id of
            that family) or ``"main model"``.
        agent_models: Agent name → model id, from ``claude_code.agent_models``.
        served_models: The ids the provider entry lists as served.
        shell_exports: Shell export lines the user must add to their profile
            (e.g. ``export ANTHROPIC_AUTH_TOKEN="$CBORG_API_KEY"``).
        gateway: ``"litellm"`` when a LiteLLM proxy fronts the provider, else
            ``None``; the built-ins ``als-apg`` and ``cborg`` are, and a custom
            provider declares it with ``gateway: litellm``.
    """

    provider: str
    default_model_id: str
    env_block: dict[str, str] = field(default_factory=dict)
    alias_models: dict[str, str] = field(default_factory=dict)
    alias_origin: dict[str, str] = field(default_factory=dict)
    agent_models: dict[str, str] = field(default_factory=dict)
    served_models: list[str] = field(default_factory=list)
    shell_exports: tuple[str, ...] = ()
    auth_env_var: str = ""
    auth_secret_env: str = ""
    needs_proxy: bool = False
    upstream_base_url: str | None = None
    #: The gateway kind fronting the provider (``"litellm"``), or ``None`` for a
    #: direct vendor. Decides whether the launch paths stamp the acting identity
    #: onto the agent's requests — see :mod:`osprey.models.spend_attribution`.
    gateway: str | None = None

    def agent_model(self, name: str) -> str:
        """The model id a named agent runs (its ``model:`` frontmatter).

        ``claude_code.agent_models.<name>`` when set, else the main model.
        """
        return self.agent_models.get(name, self.default_model_id)

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


def _warn_dropped_alias_keys(source: str, aliases: Mapping[str, Any]) -> None:
    """Warn when an alias map carries keys that are not Claude Code alias names.

    The keys are still dropped — only haiku/sonnet/opus reach the env block —
    but dropping them silently turns a typo like ``sonet:`` into an alias that
    quietly falls back to another model. Name the dropped keys instead.
    """
    dropped = [key for key in aliases if key not in TIER_MODEL_ENV_VARS]
    if dropped:
        logger.warning(
            "%s: ignoring key(s) %s — Claude Code's alias names are %s.",
            source,
            ", ".join(str(key) for key in dropped),
            ", ".join(TIER_MODEL_ENV_VARS),
        )


def _served_models(
    provider_name: str, api_providers: Mapping[str, Any]
) -> tuple[list[str], str | None, dict[str, str]]:
    """The ids a provider serves, its default model, and its catalog alias map.

    Read from the provider's ``api.providers`` entry. A built-in provider with
    no entry there (direct Anthropic configured by name alone) reads its
    packaged catalog entry instead, and a built-in the packaged catalog does not
    carry reads its own :data:`CLAUDE_CODE_PROVIDERS` row.
    """
    entry = api_providers.get(provider_name) or {}
    if "models" not in entry and "default_model" not in entry:
        if provider_name in CLAUDE_CODE_PROVIDERS:
            from osprey.profiles.providers import load_provider_catalog

            entry = (
                load_provider_catalog(None).entries.get(provider_name)
                or (CLAUDE_CODE_PROVIDERS[provider_name])
            )
    models = entry.get("models") or []
    if not isinstance(models, list):
        raise ValueError(
            f"api.providers.{provider_name}.models must be a list of the model ids "
            f"the gateway serves, got {type(models).__name__}. `osprey profile expand "
            f"--providers` refreshes a copied catalog to the packaged entries."
        )
    default = entry.get("default_model")
    if default is None and provider_name == "anthropic":
        default = _DIRECT_ANTHROPIC_DEFAULT
    aliases = entry.get("claude_code_aliases") or {}
    return [str(m) for m in models], default, dict(aliases)


def _checked_model_id(key: str, value: Any, provider_name: str, served: list[str]) -> str:
    """A configured model id, refused when it is a bare alias word.

    ``haiku``/``sonnet``/``opus`` are Claude Code's alias names, never a model
    id, so a value spelled as one is refused with the ids the provider serves.
    An id the served list does not carry is trusted: refusing it would keep
    every model the list does not name yet — a newly released id, a
    gateway-only alias — unusable until the catalog caught up, so a misspelt id
    fails at the gateway (a 404 naming it), not here.
    """
    model_id = str(value)
    if model_id in TIER_MODEL_ENV_VARS:
        served_text = ", ".join(served) if served else "no listed models"
        raise ValueError(
            f"`{key}: {model_id}` is not a model id. Provider '{provider_name}' "
            f"serves: {served_text}."
        )
    if served and model_id not in served:
        logger.info(
            "%s: %r is not in the served list of provider %r — trusting the gateway.",
            key,
            model_id,
            provider_name,
        )
    return model_id


def _resolve_aliases(
    provider_name: str,
    served: list[str],
    catalog_aliases: Mapping[str, Any],
    configured_aliases: Mapping[str, Any],
    main_model: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Claude Code's three alias models and where each came from.

    Derived from the served list by family name (newest version wins); a
    catalog entry's ``claude_code_aliases`` beats derivation; a deployment's
    ``claude_code.aliases`` beats both. An alias nothing resolves runs the main
    model, and one warning names every such substitution.
    """
    models: dict[str, str] = {}
    origin: dict[str, str] = {}
    for alias, model_id in claude_code_alias_candidates(served).items():
        models[alias], origin[alias] = model_id, "derived"
    _warn_dropped_alias_keys(f"api.providers.{provider_name}.claude_code_aliases", catalog_aliases)
    for alias in TIER_MODEL_ENV_VARS:
        if alias in catalog_aliases:
            models[alias], origin[alias] = str(catalog_aliases[alias]), "catalog"
    _warn_dropped_alias_keys("claude_code.aliases", configured_aliases)
    for alias in TIER_MODEL_ENV_VARS:
        if alias in configured_aliases:
            models[alias] = _checked_model_id(
                f"claude_code.aliases.{alias}", configured_aliases[alias], provider_name, served
            )
            origin[alias] = "claude_code.aliases"
    missing = [alias for alias in TIER_MODEL_ENV_VARS if alias not in models]
    if missing:
        what = (
            "no Claude models"
            if len(missing) == len(TIER_MODEL_ENV_VARS)
            else ("no model of " + ("that family" if len(missing) == 1 else "those families"))
        )
        logger.warning(
            "Claude Code's %s alias%s → %s: '%s' serves %s; Claude Code's own "
            "background calls will use the main model. Set claude_code.aliases.<name> "
            "to choose.",
            ", ".join(missing),
            "es" if len(missing) > 1 else "",
            main_model,
            provider_name,
            what,
        )
        for alias in missing:
            models[alias], origin[alias] = main_model, "main model"
    ordered = {alias: models[alias] for alias in TIER_MODEL_ENV_VARS}
    return ordered, {alias: origin[alias] for alias in TIER_MODEL_ENV_VARS}


class ClaudeCodeModelResolver:
    """Resolves Claude Code model configuration from project config."""

    @staticmethod
    def resolve(
        claude_code_config: dict,
        api_providers: dict | None = None,
        *,
        include_telemetry: bool = True,
        defer_unresolved_telemetry_creds: bool = False,
        environ: Mapping[str, str] | None = None,
        openobserve_port: int | None = None,
    ) -> ClaudeCodeModelSpec | None:
        """Build a ``ClaudeCodeModelSpec`` from config.

        Models are named by the id the gateway serves. The provider's
        ``api.providers`` entry lists those ids (``models``) and names its
        ``default_model``; the main model is ``claude_code.default_model`` when
        set, else that default. Each agent runs ``claude_code.agent_models.<name>``
        or the main model. Claude Code's own alias names (haiku, sonnet, opus)
        are filled from the served list — see :func:`_resolve_aliases`. A
        configured value spelled as a bare alias word is refused; an id the
        served list does not carry is trusted (:func:`_checked_model_id`).

        ``base_url`` follows the same rule: ``api.providers[name].base_url``
        overrides the built-in URL, so a facility can front a built-in provider
        with its own gateway and have the agent use the endpoint ``osprey
        health`` probes. Above both sits the break-glass env override: a
        built-in provider that declares ``base_url_env_var`` lets a set
        (non-empty) value beat config and the built-in URL, so an
        already-deployed system whose config is baked into an image can be
        redirected at a fallback gateway without a rebuild. That value is read
        only from an explicitly supplied ``environ`` — this method never
        consults ``os.environ``, because it also renders ``settings.json`` at
        build time, where an ambient read would bake the builder's endpoint
        into the artifact. The trailing ``/v1`` is stripped for
        ``ANTHROPIC_BASE_URL`` either way (see below). A built-in that declares
        ``requires_base_url`` — it fronts a gateway each site hosts itself, so
        there is no endpoint to default to — is refused outright when no source
        names one, rather than falling through to Claude Code's native backend
        with the gateway's bearer token.

        A provider that lists no models and names no default model, with no
        ``claude_code.default_model`` either, is refused.

        Args:
            claude_code_config: The ``claude_code`` section of config.yml.
            api_providers: The ``api.providers`` section (optional).
            environ: Mapping the ``base_url_env_var`` override is read from.
                Omitted means "no override" — never ``os.environ``, so
                build-time rendering is reproducible on any machine.
                :func:`load_provider_spec` passes its ``os.environ`` +
                project-``.env`` overlay, which is what enables the override on
                every runtime path.
            openobserve_port: The port the telemetry store is reached on from
                where this spec will run — ``services.openobserve.port`` for a
                build-time render, the deploy environment's declaration or
                that same key for a runtime launch
                (:func:`~osprey.build.claude_code_telemetry.resolve_openobserve_port`).
                Threaded in rather than read here, for the same reason as
                ``environ``: this method also renders ``settings.json`` at
                build time. Omitted, the derived endpoint falls back to the
                store's listen port.

        Returns:
            Resolved spec, or ``None`` when no provider is configured.

        Raises:
            ValueError: If the provider name is not in CLAUDE_CODE_PROVIDERS
                and not in api_providers, if a provider that declares
                ``requires_base_url`` resolves no endpoint, if no main model can
                be named, or if a configured model is a bare alias word.
        """
        provider_name = claude_code_config.get("provider")
        if not provider_name:
            return None

        api_providers = api_providers or {}

        if provider_name not in CLAUDE_CODE_PROVIDERS:
            # Custom proxy: must be defined in api.providers
            if provider_name not in api_providers:
                # The accepted set is the UNION of the built-ins and whatever
                # api.providers declares, so the message names the union — an
                # error that lists only the built-ins reads as "this framework
                # supports three providers" and sends an operator off to add a
                # proxy that is often already in their own config.yml (#725).
                builtin = sorted(CLAUDE_CODE_PROVIDERS)
                configured = sorted(set(api_providers) - set(CLAUDE_CODE_PROVIDERS))
                available = sorted(set(builtin) | set(api_providers))
                close = difflib.get_close_matches(provider_name, available, n=1)
                hint = f" Did you mean '{close[0]}'?" if close else ""
                raise ValueError(
                    f"Unknown Claude Code provider '{provider_name}'.{hint} "
                    f"Available providers: {', '.join(available)} "
                    f"(built-in: {', '.join(builtin)}; "
                    f"from api.providers in config.yml: {', '.join(configured) or 'none'}). "
                    f"To add another, declare it under `config:` api.providers "
                    f"in profile.yml and run `osprey build` (config.yml is generated "
                    f"from profile.yml)."
                )
            provider_def: dict[str, Any] = {
                "auth_env_var": "ANTHROPIC_AUTH_TOKEN",
                "auth_secret_env": provider_auth_secret_env(provider_name, api_providers),
            }
        else:
            provider_def = CLAUDE_CODE_PROVIDERS[provider_name]

        # ── Base URL ─────────────────────────────────────────────
        # A base_url under api.providers wins over the built-in
        # CLAUDE_CODE_PROVIDERS entry. A facility that
        # fronts a built-in provider with its own gateway therefore points the
        # agent at the endpoint `osprey health` already probes, instead of
        # having the built-in URL silently win. The built-in URL is the
        # fallback for configs that name none; a custom provider has no
        # built-in entry, so its api.providers value is the only source.
        # Above all of that sits the break-glass env override (providers that
        # declare base_url_env_var): config is often baked into a container
        # image, and a set env var must be able to redirect the deployment at
        # a fallback gateway without a rebuild. Empty means unset.
        # No ambient os.environ read: the override is honored only when a
        # caller hands in a lookup, i.e. from load_provider_spec's runtime
        # overlay. resolve() also renders settings.json at *build* time
        # (templates/claude_code.py, templates/manager.py), where reading the
        # builder's environment would bake whatever gateway that machine
        # happened to export into the shipped artifact.
        env_lookup: Mapping[str, str] = environ if environ is not None else {}
        env_var = provider_def.get("base_url_env_var")
        base_url = (
            (env_lookup.get(env_var) if env_var else None)
            or api_providers.get(provider_name, {}).get("base_url")
            or provider_def.get("base_url")
        )
        # A provider that has no built-in endpoint must be told one. Left
        # unresolved, `base_url` is simply absent from the env block below and
        # Claude Code talks to its native backend — so a gateway's bearer token
        # would be presented to api.anthropic.com, which reads as an auth error
        # nowhere near its cause.
        if provider_def.get("requires_base_url") and not base_url:
            sources = f"api.providers.{provider_name}.base_url in config.yml"
            if env_var:
                sources = f"{env_var}, or {sources}"
            raise ProviderEndpointError(
                f"Provider '{provider_name}' has no base_url. It fronts models "
                f"through a gateway that has no default endpoint, so the URL has "
                f"to be named: set {sources}."
            )

        # ── Models ───────────────────────────────────────────────
        served, catalog_default, catalog_aliases = _served_models(provider_name, api_providers)
        configured_default = claude_code_config.get("default_model")
        if configured_default:
            main_model = _checked_model_id(
                "claude_code.default_model", configured_default, provider_name, served
            )
        elif catalog_default:
            main_model = str(catalog_default)
        else:
            raise ValueError(
                f"Provider '{provider_name}' lists no models and names no default_model; "
                f"add `models:` (a list of the ids it serves) and `default_model:` under "
                f"api.providers.{provider_name} — under `config:` in profile.yml for "
                f"profile-built projects, or in the providers.yml beside it — then run "
                f"`osprey build`."
            )
        alias_models, alias_origin = _resolve_aliases(
            provider_name,
            served,
            catalog_aliases,
            claude_code_config.get("aliases") or {},
            main_model,
        )
        agent_models = {
            str(name): _checked_model_id(
                f"claude_code.agent_models.{name}", model_id, provider_name, served
            )
            for name, model_id in (claude_code_config.get("agent_models") or {}).items()
        }

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
        # with the loopback URL at launch. Skipped when neither config nor the
        # built-in table names a URL (direct Anthropic with no api.providers entry).
        if base_url:
            env_block["ANTHROPIC_BASE_URL"] = base_url.rstrip("/").removesuffix("/v1")

        # Claude Code's alias env vars (all providers) — derived from the single
        # TIER_MODEL_ENV_VARS declaration so this key set can never drift from
        # the e2e-force and scrub-agreement paths. alias_models always carries
        # all three names, in TIER_MODEL_ENV_VARS order.
        for alias, env_var in TIER_MODEL_ENV_VARS.items():
            env_block[env_var] = alias_models[alias]

        # ── Shell exports (auth key — must be set in user's profile) ──
        auth_env_var = provider_def["auth_env_var"]
        auth_secret_env = provider_def["auth_secret_env"]
        if auth_env_var == auth_secret_env:
            # Direct provider (e.g. anthropic): just needs the var set
            shell_exports = (f'export {auth_env_var}="<your-api-key>"',)
        else:
            # Proxy provider (e.g. cborg): alias one var to another
            shell_exports = (f'export {auth_env_var}="${auth_secret_env}"',)

        # ANTHROPIC_MODEL: override any shell-level value so the project's
        # main model is authoritative.
        env_block["ANTHROPIC_MODEL"] = main_model

        # ── Proxy detection ───────────────────────────────────────
        from osprey.infrastructure.proxy.lifecycle import is_proxy_needed

        _needs_proxy = is_proxy_needed(provider_name, api_providers)
        # The proxy forwards to upstream + "/chat/completions", so the upstream
        # must keep its /v1. Use the raw resolved base_url, NOT the /v1-stripped
        # ANTHROPIC_BASE_URL above. Every launch path starts the proxy from this
        # field (never from the env var) — see runner.py / dispatch_api.py.
        _upstream_url = base_url if _needs_proxy else None

        # ── Telemetry (absent block == disabled → helper returns {}) ──
        # Container context is the ONE place fs/env is consulted; the helper
        # itself stays pure. Telemetry keys are deliberately excluded from
        # MANAGED_ENV_VARS (they are not backend/model selectors).
        # Telemetry is an observability concern, not a provider/model selector.
        # Callers that only read model ids (model-id readers) or that must
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
                    openobserve_port=openobserve_port,
                    defer_unresolved_creds=defer_unresolved_telemetry_creds,
                )
            )

        return ClaudeCodeModelSpec(
            provider=provider_name,
            default_model_id=main_model,
            env_block=env_block,
            alias_models=alias_models,
            alias_origin=alias_origin,
            agent_models=agent_models,
            served_models=served,
            shell_exports=tuple(shell_exports),
            auth_env_var=auth_env_var,
            auth_secret_env=auth_secret_env,
            needs_proxy=_needs_proxy,
            upstream_base_url=_upstream_url,
            gateway=gateway_for(provider_name, api_providers),
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
